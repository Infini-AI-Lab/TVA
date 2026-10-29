"""TVA library evidence (library issue L34): float32 `atomic_add` on sm_90, both instructions.

Run from the TVA root:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-atomic-add/run.py
Output: evidence/num-atomic-add/out.txt (README.md says what it shows).

`P.atomicAdd .binary32` is the new value of a float32 `tl.atomic_add` (pointer atomics,
`atom.add.f32` / `red.add.f32`) and of a descriptor `desc.atomic_add` (the TMA reduction
`cp.reduce.async.bulk.tensor ... .add.f32`). The fact `AtomicAddF32` (docs/rules/num.md) says:
for finite old value o and operand v with |o + v| <= maxVal, the new value is finite and within
2^-23 |o + v| + 2^-124 of o + v. The script also counts the pairs outside the tight bound
2^-24 |o + v| + 2^-125 (rounding to nearest, subnormals flushed). Each lane adds its own v to its own cell (distinct addresses),
from bit patterns chosen adversarially (random patterns, subnormals, near-cancellations, sums at
the edge of the range); o + v is computed in float64 and its rounding error, at most
2^-53 |o + v|, is added to the error. It also records the
scalar `tl.atomic_add(p, 0.0)` (relaxed), which Triton lowers to a plain load.
"""
import re
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

torch.manual_seed(20261028)
DEV = "cuda"


@triton.jit
def ptr_add(c_ptr, v_ptr, n, B: tl.constexpr, SEM: tl.constexpr):
    offs = tl.program_id(0) * B + tl.arange(0, B)
    m = offs < n
    v = tl.load(v_ptr + offs, mask=m)
    tl.atomic_add(c_ptr + offs, v, mask=m, sem=SEM)


@triton.jit
def tma_add(c_desc, v_ptr, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    v = tl.load(v_ptr + offs)
    c_desc.atomic_add([pid * B], v)


@triton.jit
def zero_add(c_ptr):
    tl.atomic_add(c_ptr, 0.0, sem="relaxed")


def pairs(kind, n):
    if kind == "random":
        o = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=DEV).to(torch.int32).view(torch.float32)
        v = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=DEV).to(torch.int32).view(torch.float32)
    elif kind == "subnormal":
        o = (torch.randint(-(1 << 24), 1 << 24, (n,), device=DEV).to(torch.int32)).view(torch.float32)
        v = (torch.randint(-(1 << 24), 1 << 24, (n,), device=DEV).to(torch.int32)).view(torch.float32)
    elif kind == "cancel":
        o = torch.randn(n, device=DEV) * torch.exp2(torch.randint(-140, 120, (n,), device=DEV).float())
        v = -o * (1 + torch.randint(-8, 9, (n,), device=DEV).float() * 2.0 ** -23)
    elif kind == "scale":
        o = torch.randn(n, device=DEV) * torch.exp2(torch.randint(-149, 127, (n,), device=DEV).float())
        v = torch.randn(n, device=DEV) * torch.exp2(torch.randint(-149, 127, (n,), device=DEV).float())
    elif kind == "edge":
        o = (torch.rand(n, device=DEV) + 0.9) * 2.0 ** 127
        v = (torch.rand(n, device=DEV) - 0.5) * 2.0 ** 126
    return o.contiguous(), v.contiguous()


def check(o, v, nnew):
    od, vd, nd = o.double(), v.double(), nnew.double()
    s = od + vd                                   # within 2^-53 |o + v| of the exact sum
    ok = torch.isfinite(od) & torch.isfinite(vd) & (s.abs() <= 3.4028234663852886e38)
    bound = 2.0 ** -23 * s.abs() + 2.0 ** -124   # AtomicAddF32
    tight = 2.0 ** -24 * s.abs() + 2.0 ** -125
    err = (nd - s).abs() + 2.0 ** -52 * s.abs()  # the reference's rounding added
    fin = torch.isfinite(nd)
    viol = int((ok & fin & (err > bound)).sum()) + int((ok & ~fin).sum())
    vt = int((ok & fin & (err > tight)).sum())
    ratio = torch.where(ok & fin, err / bound, torch.zeros_like(err)).max().item()
    rt = torch.where(ok & fin, err / tight, torch.zeros_like(err)).max().item()
    # where o, v and the sum are normal and the float64 sum is exact, is the result the float32
    # rounding to nearest of o + v?
    normal = ok & (od.abs() >= 2.0 ** -126) & (vd.abs() >= 2.0 ** -126) & (s.abs() >= 2.0 ** -126)
    exact = (s - od == vd) & (s - vd == od)
    rn = bool(torch.where(normal & exact, s.float().double() == nd, torch.ones_like(ok)).all().item())
    return viol, ratio, int(ok.sum()), rn, vt, rt


def main():
    print(f"Triton {triton.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}\n")
    print("| instruction | inputs | pairs checked | outside the bound | largest error / bound | outside the tight bound | largest error / tight bound | normal o, v, o + v: rounded to nearest |")
    print("|---|---|---|---|---|---|---|---|")
    n = 1 << 24
    B = 1024
    total = 0
    hs = {}
    for kind in ["random", "subnormal", "cancel", "scale", "edge"]:
        for sem in ["acq_rel", "relaxed"]:
            o, v = pairs(kind, n)
            c = o.clone()
            hs[sem] = ptr_add[(n // B,)](c, v, n, B=B, SEM=sem)
            vi, r, k, rn, vt, rt = check(o, v, c)
            total += vi
            print(f"| pointer `tl.atomic_add`, sem={sem} | {kind} | {k} | {vi} | {r:.4f} | {vt} | {rt:.4f} | {rn} |")
        o, v = pairs(kind, n)
        c = o.clone()
        desc = TensorDescriptor.from_tensor(c, [B])
        h2 = tma_add[(n // B,)](desc, v, B=B)
        vi, r, k, rn, vt, rt = check(o, v, c)
        total += vi
        print(f"| descriptor `atomic_add` (TMA) | {kind} | {k} | {vi} | {r:.4f} | {vt} | {rt:.4f} | {rn} |")
    print(f"\nTotal pairs outside the bound: {total}")
    # a few special cases on both paths: are subnormal operands and results flushed?
    cases = [(2.0 ** -149, 0.0), (0.0, 2.0 ** -140), (2.0 ** -126, -(2.0 ** -127)),
             (2.0 ** -126, 2.0 ** -149), (-0.0, 0.0), (-0.0, -0.0), (1.0, -1.0), (3.0e38, 3.0e38)]
    co = torch.zeros(B, device=DEV); cv = torch.zeros(B, device=DEV)
    for j, (a, b) in enumerate(cases):
        co[j], cv[j] = a, b
    c1 = co.clone(); ptr_add[(1,)](c1, cv, B, B=B, SEM="acq_rel")
    c2 = co.clone(); tma_add[(1,)](TensorDescriptor.from_tensor(c2, [B]), cv, B=B)
    print("\n| old | operand | pointer: new | descriptor: new |")
    print("|---|---|---|---|")
    for j, (a, b) in enumerate(cases):
        print(f"| {a!r} | {b!r} | {c1[j].item()!r} | {c2[j].item()!r} |")
    print()
    for name, hh in [("pointer, sem=acq_rel", hs["acq_rel"]), ("pointer, sem=relaxed", hs["relaxed"]),
                     ("descriptor", h2)]:
        ops = sorted(set(re.findall(r"\b(atom\.[a-z0-9._]+|red\.[a-z0-9._]+|cp\.reduce\.[a-z0-9._:]+)", hh.asm["ptx"])))
        print(f"- {name}: " + ", ".join(f"`{x}`" for x in ops))
    c = torch.tensor([-0.0], device=DEV)
    hz = zero_add[(1,)](c)
    ops = sorted(set(re.findall(r"\b(atom\.[a-z0-9.]+|red\.[a-z0-9.]+|ld\.[a-z0-9.]+)", hz.asm["ptx"])))
    print(f"- scalar `tl.atomic_add(p, 0.0, sem=\"relaxed\")` on -0.0: memory after {c.item()!r} "
          f"(sign bit {int(c.view(torch.int32).item()) & 0x80000000 != 0}); instructions: " + ", ".join(f"`{x}`" for x in ops))


if __name__ == "__main__":
    main()
