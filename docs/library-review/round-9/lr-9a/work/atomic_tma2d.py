"""lr-9a: AtomicAddF32 on the descriptor forms evidence/num-atomic-add does not run: a 2-D host
descriptor (`TensorDescriptor`, `cp.reduce.async.bulk.tensor.2d`) and a descriptor built in the
kernel (`tl.make_tensor_descriptor`), on random, subnormal, cancelling and near-overflow pairs.
Each element is one P.atomicAdd step (distinct cells); the fact: |n - (o+v)| <= 2^-23 |o+v| +
2^-124, n finite when |o+v| <= maxVal.

Run from this folder: CUDA_VISIBLE_DEVICES=0 python -I atomic_tma2d.py
"""
import re
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

torch.manual_seed(931)
DEV = "cuda"
R, C = 64, 64


@triton.jit
def host2d(c_desc, v_ptr, NC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    pid_r = tl.program_id(0); pid_c = tl.program_id(1)
    r = pid_r * BR + tl.arange(0, BR)[:, None]
    c = pid_c * BC + tl.arange(0, BC)[None, :]
    v = tl.load(v_ptr + r * NC + c)
    c_desc.atomic_add([pid_r * BR, pid_c * BC], v)


@triton.jit
def dev2d(c_ptr, v_ptr, NR: tl.constexpr, NC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    desc = tl.make_tensor_descriptor(c_ptr, shape=[NR, NC], strides=[NC, 1], block_shape=[BR, BC])
    pid_r = tl.program_id(0); pid_c = tl.program_id(1)
    r = pid_r * BR + tl.arange(0, BR)[:, None]
    c = pid_c * BC + tl.arange(0, BC)[None, :]
    v = tl.load(v_ptr + r * NC + c)
    desc.atomic_add([pid_r * BR, pid_c * BC], v)


def pairs(kind, n):
    if kind == "random":
        o = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=DEV).to(torch.int32).view(torch.float32)
        v = torch.randint(-2 ** 31, 2 ** 31, (n,), dtype=torch.int64, device=DEV).to(torch.int32).view(torch.float32)
    elif kind == "subnormal":
        o = torch.randint(-(1 << 24), 1 << 24, (n,), device=DEV).to(torch.int32).view(torch.float32)
        v = torch.randint(-(1 << 24), 1 << 24, (n,), device=DEV).to(torch.int32).view(torch.float32)
    elif kind == "cancel":
        o = torch.randn(n, device=DEV) * torch.exp2(torch.randint(-140, 120, (n,), device=DEV).float())
        v = -o * (1 + torch.randint(-8, 9, (n,), device=DEV).float() * 2.0 ** -23)
    else:
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
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=DEV))
    NR, NC = 2048, 2048
    n = NR * NC
    print("| descriptor | inputs | pairs checked | outside the fact | largest error / bound | instruction |")
    print("|---|---|---|---|---|---|")
    total = 0
    for kind in ["random", "subnormal", "cancel", "edge"]:
        for which in ["host 2-D", "in-kernel 2-D"]:
            vs = 0; rm = 0.0; ks = 0
            for rep in range(4):
                o, v = pairs(kind, n)
                cells = o.clone().view(NR, NC)
                if which == "host 2-D":
                    d = TensorDescriptor.from_tensor(cells, [R, C])
                    h = host2d[(NR // R, NC // C)](d, v, NC=NC, BR=R, BC=C)
                else:
                    h = dev2d[(NR // R, NC // C)](cells, v, NR=NR, NC=NC, BR=R, BC=C)
                vi, ra, k = check(o, v, cells.reshape(-1))
                total += vi; vs += vi; rm = max(rm, ra); ks += k
            ops = sorted(set(re.findall(r"\b(cp\.reduce\.[a-z0-9._:]+|atom\.[a-z0-9._]+|red\.[a-z0-9._]+)", h.asm["ptx"])))
            print(f"| {which} | {kind} | {ks} | {vs} | {rm:.4f} | {', '.join(ops)} |")
    print(f"\nTotal outside the fact: {total}")


if __name__ == "__main__":
    main()
