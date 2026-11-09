"""lr-35a (round 35, L164): does an atom run for a NaN operand of a float32 atomic_max/atomic_min?
Old values 1.0, -1.0, 1.0, -1.0; operands +NaN, +NaN, -NaN, -NaN; prints memory after the call.
Run from the TVA root: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file>
"""
import torch, triton, triton.language as tl


@triton.jit
def k(P, V, R, MAX: tl.constexpr, N: tl.constexpr):
    o = tl.arange(0, N)
    v = tl.load(V + o)
    if MAX:
        r = tl.atomic_max(P + o, v)
    else:
        r = tl.atomic_min(P + o, v)
    tl.store(R + o, r)


def t(xs):
    return torch.tensor([x - (1 << 32) if x >= 1 << 31 else x for x in xs], dtype=torch.int32, device="cuda").view(torch.float32)


olds = [0x3f800000, 0xbf800000, 0x3f800000, 0xbf800000]
vs = [0x7fc00000, 0x7fc00000, 0xffc00000, 0xffc00000]
for MAX in (True, False):
    P, V = t(olds), t(vs)
    R = torch.empty_like(P)
    k[(1,)](P, V, R, MAX=MAX, N=4)
    torch.cuda.synchronize()
    print("atomic_max" if MAX else "atomic_min", "old", [hex(x) for x in olds], "operand", [hex(x) for x in vs],
          "memory after", [hex(x & 0xffffffff) for x in P.view(torch.int32).tolist()],
          "returned", [hex(x & 0xffffffff) for x in R.view(torch.int32).tolist()])
