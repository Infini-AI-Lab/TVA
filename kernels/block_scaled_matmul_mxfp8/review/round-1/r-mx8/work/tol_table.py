"""r-mx8, round 1: the README's table ("Its size on the host's data"), recomputed on the CPU.

Each sample is one output element: a row of `a` and a row of `b` drawn as the host draws them
(`MXFP4Tensor(...).random()` values, scales `MXScaleTensor(torch.rand + 1e-8)`, one scale byte per
group of 32 along K), the products p, and the claim's `tol p (K / 128)` by its recursion
(Defs.lean `tol`, rho = 130 * 2^-22, alpha = 130 * 2^-126), in float64.
The whole bound adds the float16 rounding: tol + (|S| + tol) * 2^-11 + 2^-25.
"""
import sys

import torch
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

RHO = 130 * 2.0 ** -22
ALPHA = 130 * 2.0 ** -126


def tol(p):
    t = 0.0
    T = p.numel() // 128
    part = torch.cumsum(p, 0)
    for s in range(T):
        ps = float(part[128 * s - 1]) if s > 0 else 0.0
        chunk = float(p[128 * s:128 * (s + 1)].abs().sum())
        t = t + RHO * (abs(ps) + t + chunk) + ALPHA
    return t


def row(K):
    x = MXFP4Tensor(size=(1, K), device="cpu").random().to(torch.float32).double()[0]
    sc = MXScaleTensor(torch.rand(K // 32) + 1e-8).to(torch.float32).double()
    return x * sc.repeat_interleave(32)


def main(nsamp, seed):
    torch.manual_seed(seed)
    print(f"samples per K: {nsamp}, seed {seed}")
    print("K | median |S| | median sum|p| | median tol | max tol/sum|p| | median tol/|S| | median bound/|S|")
    for K in (512, 8192, 16384):
        S, A, Tl, r1, r2, r3 = [], [], [], [], [], []
        for _ in range(nsamp):
            p = row(K) * row(K)
            s = float(p.sum())
            a = float(p.abs().sum())
            t = tol(p)
            bound = t + (abs(s) + t) * 2.0 ** -11 + 2.0 ** -25
            S.append(abs(s)); A.append(a); Tl.append(t)
            r1.append(t / a); r2.append(t / abs(s) if s != 0 else float("inf")); r3.append(bound / abs(s) if s != 0 else float("inf"))
        med = lambda v: float(torch.tensor(v).median())
        print(f"{K} | {med(S):.1f} | {med(A):.0f} | {med(Tl):.4f} | {max(r1):.2e} | {100 * med(r2):.3f}% | {100 * med(r3):.3f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
