"""lr-4a, L26: Triton's combine pass (`CombineBroadcastMulReducePattern`, Combine.cpp:112-198) checks
that the reduce's combine region is one float `addf` (and the return), but not that the `addf`
adds the region's two arguments. So a `tl.reduce` of a broadcast product whose combine function
returns `a + a` (not the sum of its two arguments, so not `[sum-bcast-mul]`'s computation) may be
rewritten into `tl.dot` too.

float32, M = N = 16 (rewritten) and 8 (not), K = 4: x[i, k] = k + 1, y = 1, so the products along k
are 1, 2, 3, 4 and their sum is 10. A reduction whose combine is `a + a` (or `b + b`) doubles one
side at each step of its tree and never gives 10; the rewritten `tl.dot` gives 10.

Run from the TVA root:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I docs/library-review/round-4/lr-4a/work/bcast_reduce_aa.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def comb_aa(a, b):
    return a + a


@triton.jit
def comb_bb(a, b):
    return b + b


@triton.jit
def comb_ab(a, b):
    return a + b


@triton.jit
def k(x_ptr, y_ptr, o_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, WHICH: tl.constexpr):
    rm = tl.arange(0, M)
    rk = tl.arange(0, K)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :])
    y = tl.load(y_ptr + rk[:, None] * N + rn[None, :])
    p = x[:, :, None] * y[None, :, :]
    if WHICH == 0:
        r = tl.reduce(p, 1, comb_aa)
    elif WHICH == 1:
        r = tl.reduce(p, 1, comb_bb)
    else:
        r = tl.reduce(p, 1, comb_ab)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], r)


def main():
    print("triton", triton.__version__, "torch", torch.__version__, torch.cuda.get_device_name(0))
    for M, N in [(16, 16), (8, 8)]:
        K = 4
        x = (torch.arange(K, device="cuda", dtype=torch.float32) + 1).repeat(M, 1).contiguous()
        y = torch.ones(K, N, device="cuda")
        for which, name in [(0, "a + a"), (1, "b + b"), (2, "a + b")]:
            o = torch.empty(M, N, device="cuda")
            h = k[(1,)](x, y, o, M=M, K=K, N=N, WHICH=which)
            torch.cuda.synchronize()
            dot = "tt.dot" in h.asm["ttir"]
            vals = sorted(set(o.flatten().tolist()))
            print(f"M={M} N={N} K={K}, tl.reduce(x[:, :, None] * y[None, :, :], 1, f) with f returning {name}: "
                  f"tt.dot in TTIR: {dot}; output values {vals} (sum of products: 10.0)")


if __name__ == "__main__":
    main()
