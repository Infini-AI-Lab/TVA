"""Reviewer r-bwd2: the relative size of dVTol (Spec.lean) against sum_m P*|dO|, per pair and
accumulated, at the test's and the benchmark's sizes, with sum_c |K*Q| at its expected value for
the host's inputs (q, k normal with std sigma, K = fp16(k * sm_scale / ln 2)). Run: python3 -I tol.py"""
import math
from fractions import Fraction as F

u32 = 2.0**-24; u16 = 2.0**-11
rel = lambda K: (K + 2) * 2.0**-22
rho = float(F(34, 2**22) + 3 * F(1, 2**24) * (1 + F(34, 2**22)))

def steps(N, causal_worst):
    return 8 + (N - 128) // 32 if causal_worst else N // 32

for name, sm, sigma, N in [("test", 0.5, 0.5, 4096), ("bench", 1.3, 1.0, 16384)]:
    for HD in (64, 128, 256):
        kscale = sm / math.log(2)
        aQK = HD * (2 / math.pi) * sigma * (sigma * kscale)   # E sum_c |K Q|
        eX = rel(HD) * aQK + 3 * u32 * (2 * aQK)             # dot error + the subtraction (|s|,|L| ~ aQK)
        pBr_rel = (1 + 2**-22) * 2**eX                       # pBr / 2^x (2^-126 term negligible)
        p16err_rel = pBr_rel - 1 + u16 * pBr_rel
        T = steps(N, True)
        acc_rel = ((1 + rho)**T - 1) * (1 + u16) * pBr_rel
        print(f"{name:5s} N={N:5d} HD={HD:3d} T={T:3d} aQK~{aQK:7.1f}  per-pair {100*p16err_rel:.3f}%  "
              f"accumulation {100*acc_rel:.3f}%  total {100*(p16err_rel+acc_rel):.3f}% of sum P|dO|")
