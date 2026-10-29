"""lr-9a: AtomicAddF32 on the pointer-atomic forms evidence/num-atomic-add does not run.

The evidence's pointer kernel adds a contiguous, aligned block, which Triton packs into
`atom.global.gpu.<sem>.add.v4.f32`. AtomicAddF32 also claims the scalar `atom….add.f32`
(scattered or unaligned addresses, a scalar pointer), every `sem` and every `scope`. This runs:
- a strided tensor atomic (every other cell, so no vector form), with sem in
  {acq_rel, relaxed, acquire, release} and scope in {gpu, cta, sys};
- a scalar atomic (one cell per program);
- the old value returned by the atomic is used (stored), and unused;
on subnormal, random, cancelling and near-overflow pairs, and checks the fact
(|n - (o+v)| <= 2^-23 |o+v| + 2^-124, n finite when |o+v| <= maxVal), as the evidence does.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I atomic_scalar.py
"""
import re
import torch
import triton
import triton.language as tl

torch.manual_seed(929)
DEV = "cuda"


@triton.jit
def strided(c_ptr, v_ptr, r_ptr, n, B: tl.constexpr, SEM: tl.constexpr, SCOPE: tl.constexpr, USE: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    v = tl.load(v_ptr + offs, mask=m)
    old = tl.atomic_add(c_ptr + 2 * offs, v, mask=m, sem=SEM, scope=SCOPE)
    if USE:
        tl.store(r_ptr + offs, old, mask=m)


@triton.jit
def scalar(c_ptr, v_ptr):
    pid = tl.program_id(0)
    v = tl.load(v_ptr + pid)
    tl.atomic_add(c_ptr + 2 * pid, v)


def pairs(kind, n):
    if kind == "random":
        o = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=DEV).to(torch.int32).view(torch.float32)
        v = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=DEV).to(torch.int32).view(torch.float32)
    elif kind == "subnormal":
        o = torch.randint(-(1 << 24), 1 << 24, (n,), device=DEV).to(torch.int32).view(torch.float32)
        v = torch.randint(-(1 << 24), 1 << 24, (n,), device=DEV).to(torch.int32).view(torch.float32)
    elif kind == "near-normal":   # both just below 2^-126: the sum is near 2^-125 and is flushed
        o = (torch.randint((1 << 23) - (1 << 12), 1 << 23, (n,), device=DEV)).to(torch.int32).view(torch.float32)
        v = (torch.randint((1 << 23) - (1 << 12), 1 << 23, (n,), device=DEV)).to(torch.int32).view(torch.float32)
    elif kind == "cancel":
        o = torch.randn(n, device=DEV) * torch.exp2(torch.randint(-140, 120, (n,), device=DEV).float())
        v = -o * (1 + torch.randint(-8, 9, (n,), device=DEV).float() * 2.0 ** -23)
    else:  # edge
        o = (torch.rand(n, device=DEV) + 0.9) * 2.0 ** 127
        v = (torch.rand(n, device=DEV) - 0.5) * 2.0 ** 126
    return o.contiguous(), v.contiguous()


def check(o, v, nn):
    od, vd, nd = o.double(), v.double(), nn.double()
    s = od + vd
    ok = torch.isfinite(od) & torch.isfinite(vd) & (s.abs() <= 3.4028234663852886e38)
    bound = 2.0 ** -23 * s.abs() + 2.0 ** -124
    err = (nd - s).abs() + 2.0 ** -52 * s.abs()
    fin = torch.isfinite(nd)
    viol = int((ok & fin & (err > bound)).sum()) + int((ok & ~fin).sum())
    ratio = torch.where(ok & fin, err / bound, torch.zeros_like(err)).max().item()
    return viol, ratio, int(ok.sum())


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| form | sem, scope, old used | inputs | pairs checked | outside the fact | largest error / bound | atomic instructions |")
    print("|---|---|---|---|---|---|---|")
    n = 1 << 22
    B = 1024
    total = 0
    kinds = ["random", "subnormal", "near-normal", "cancel", "edge"]
    combos = [("acq_rel", "gpu", False), ("relaxed", "gpu", False), ("acquire", "gpu", False), ("release", "gpu", False),
              ("acq_rel", "cta", False), ("acq_rel", "sys", False), ("relaxed", "gpu", True), ("acq_rel", "gpu", True)]
    for sem, scope, use in combos:
        for kind in kinds:
            o, v = pairs(kind, n)
            cells = torch.zeros(2 * n, device=DEV)
            cells[0::2] = o
            r = torch.empty(n, device=DEV)
            h = strided[(n // B,)](cells, v, r, n, B=B, SEM=sem, SCOPE=scope, USE=use)
            vi, ra, k = check(o, v, cells[0::2])
            if use:
                vi += int((r.view(torch.int32) != o.view(torch.int32)).sum())  # old value returned bit for bit
            total += vi
            ops = sorted(set(re.findall(r"\b(atom\.[a-z0-9._]+|red\.[a-z0-9._]+)", h.asm["ptx"])))
            print(f"| strided tensor | {sem}, {scope}, {use} | {kind} | {k} | {vi} | {ra:.4f} | {', '.join(ops)} |")
    m = 1 << 20
    for kind in kinds:
        o, v = pairs(kind, m)
        cells = torch.zeros(2 * m, device=DEV)
        cells[0::2] = o
        h = scalar[(m,)](cells, v)
        vi, ra, k = check(o, v, cells[0::2])
        total += vi
        ops = sorted(set(re.findall(r"\b(atom\.[a-z0-9._]+|red\.[a-z0-9._]+)", h.asm["ptx"])))
        print(f"| scalar | acq_rel, gpu, False | {kind} | {k} | {vi} | {ra:.4f} | {', '.join(ops)} |")
    print(f"\nTotal outside the fact (and old values not returned bit for bit): {total}")


if __name__ == "__main__":
    main()
