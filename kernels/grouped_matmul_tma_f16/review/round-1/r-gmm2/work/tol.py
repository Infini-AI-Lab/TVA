"""Reviewer r-gmm2: the tolerance of Post at the covered K, and where it stops rejecting a
bfloat16 rounding. Run with python3 -I from anywhere; exact rationals, then floats."""
from fractions import Fraction as F

rel = F(66, 2**22)                      # relDot16 64
rho = rel + F(3, 2**24) * (1 + rel)     # DotLoop.stepRel relDot16 64
u16 = F(1, 2**11)                       # Format.binary16.u
bf = F(1, 2**8) / (1 + F(1, 2**8))      # largest relative error of a bfloat16 rounding


def tol_rel(K):
    T = (K + 63) // 64                  # steps K
    return (1 + rho) ** T - 1


print("rho =", float(rho))
for K in (64, 128, 1024, 8192):
    r = tol_rel(K)
    print(f"K={K}: (1+rho)^T-1 = {float(r):.6g}, with the last float16 rounding "
          f"{float(r + (1 + r) * u16):.6g}")
print("bfloat16 largest relative error =", float(bf))

T = 1
while (1 + rho) ** T - 1 < bf:
    T += 1
print(f"tol alone reaches it at T={T}, K >= {64 * (T - 1) + 1}")
T = 1
while ((1 + rho) ** T - 1) * (1 + u16) + u16 < bf:
    T += 1
print(f"Post's bound on c (tol + (|S|+tol)*2^-11) reaches it at T={T}, K >= {64 * (T - 1) + 1}")
