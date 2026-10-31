"""Reviewer r-bwdpre2: the tolerance `tol D B` of Model.lean, evaluated exactly at the covered
head dimensions, and two wrong kernels it should reject (run with python3 -I)."""
from fractions import Fraction as Q

u = Q(1, 2**24)
eta = Q(1, 2**150)


def err_bound(delta, e, d, n, a):
    # Tree.errBound delta e d N A = ((1 + delta)^d - 1) * A + e * N * (1 + delta)^d
    g = (1 + delta) ** d
    return (g - 1) * a + e * n * g


def tol(D, B):
    return err_bound(3 * u, 4 * eta, D - 1, D - 1, (1 + u) * B + D * eta) + (u * B + D * eta)


for D in (16, 32, 64, 128, 256):
    rel = tol(D, Q(1)) - tol(D, Q(0))  # the coefficient of B
    ab = tol(D, Q(0))
    print(f"D={D:4d}  tol/B = {float(rel):.4e} = {float(rel / u):.2f} u   absolute = {float(ab / eta):.2f} eta")

# tol_le: tol D B <= 2^-14 B + 2^-138 for D <= 256
print("tol_le slack at D=256, B=1:", float(Q(1, 2**14) + Q(1, 2**138) - tol(256, Q(1))))

# Wrong kernel 1: the result rounded to float16 (README's example, HEAD_DIM = 64).
D = 64
exact = Q(64) + Q(1, 2**10)
B = exact
f16_result = Q(64)  # 64 + 2^-10 rounds to 64 in float16 (spacing 2^-4 at 64)
print("f16 rounding error", float(abs(f16_result - exact)), "tol", float(tol(D, B)),
      "rejected:", abs(f16_result - exact) > tol(D, B))

# Wrong kernel 2: one column dropped (k = HEAD_DIM - 1), all products 1.
for D in (16, 256):
    print(f"dropped column D={D}: error 1, tol {float(tol(D, Q(D))):.3e}, rejected: {1 > tol(D, Q(D))}")

# Wrong kernel 3: accumulation in bfloat16/float16 of the products: at D = 256, all products
# 1 + 2^-9 (o = 1 + 2^-9, do = 1, both float16); float16 sum stalls once spacing exceeds 2^-9.
D = 256
p = Q(1) + Q(1, 2**9)
exact = D * p
print(f"D=256 exact {float(exact)}, tol {float(tol(D, exact)):.3e}")
