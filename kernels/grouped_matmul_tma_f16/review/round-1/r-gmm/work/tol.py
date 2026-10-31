"""r-gmm, round 1: the tolerance of `Post` at the covered sizes, in exact rationals.

rho = relDot16 64 + 3*2^-24*(1 + relDot16 64), alpha = absDot16 64*(1 + 3*2^-24) + 4*2^-150
(Basic.lean, rho_eq and alpha_eq); tol K A = ((1 + rho)^T - 1)*A + T*(1 + rho)^T*alpha with
T = cdiv(K, 64) (Defs.lean, tol and steps). Post bounds the stored float16 c by
tol + (|S| + tol)*2^-11 + 2^-25 when there is room (OutSpec).
"""
from fractions import Fraction as F

rel = F(66, 2**22)
ab = F(66, 2**126)
rho = rel + F(3, 2**24) * (1 + rel)
alpha = ab * (1 + F(3, 2**24)) + F(4, 2**150)
u16 = F(1, 2**11)
bf16_worst = F(2**-8) / (1 + F(2**-8))

print("rho =", float(rho), " alpha =", float(alpha))


def steps(K):
    return (K + 63) // 64


def rel_tol(K):
    return (1 + rho) ** steps(K) - 1


for K in [64, 128, 1024, 8192]:
    r = rel_tol(K)
    tot = r + (1 + r) * u16
    print(f"K={K:5d} T={steps(K):3d} (1+rho)^T-1 = {float(r):.4e}  with the float16 rounding: {float(tot):.4e}")

print("bfloat16's largest relative rounding error:", float(bf16_worst))

# The smallest K at which the accumulation part alone reaches the bfloat16 error (the README's
# "about K = 15745"), and at which Post's whole relative bound on c (accumulation plus the last
# float16 rounding) does.
K1 = next(64 * T for T in range(1, 10**4) if (1 + rho) ** T - 1 >= bf16_worst)
K2 = next(64 * T for T in range(1, 10**4)
          if ((1 + rho) ** T - 1) + (1 + ((1 + rho) ** T - 1)) * u16 >= bf16_worst)
print("accumulation part alone reaches the bfloat16 error from K about", K1)
print("Post's whole bound on c reaches it from K about", K2)

# The README's witness at K = 2^13: rows of ones against a row of 8160 ones and 32 twos, so
# S = A = 8224 = 2^13 (1 + 2^-8), exact in float32 and in float16; its bfloat16 rounding is 8192.
K = 2**13
S = F(8224)
tol = rel_tol(K) * S + steps(K) * (1 + rho) ** steps(K) * alpha
bound = tol + (S + tol) * u16 + F(1, 2**25)
print("K=8192 witness: Post allows |c - S| <=", float(bound), "; a bfloat16 rounding is off by", 8224 - 8192)
