"""Reviewer r-softmax, round 1: the tolerance of `Post` against half-precision outputs.

Run with `python3 -I` from anywhere. Exact arithmetic (`fractions`), with the claim's own constants:
`relTol B = (B + 4)·2^-22`, `absTol = 2^-22`, and the error pieces of `Softmax/Err.lean`
(`sum_rel`: the sum within `(13B + 20)·2^-26` relatively; `final_bound`'s intermediate `hD`;
`DivFull`: `2^-22` relative, `2^-148` absolute).
"""
from fractions import Fraction as F
import math

u22 = F(1, 2**22)


def rel_tol(B):
    return (B + 4) * u22


ABS_TOL = u22

print("1. Unit roundoffs against relTol")
for B in [2**10, 2**11, 2**12, 2**13, 2**14]:
    print(f"   B = {B:5d}: relTol = {float(rel_tol(B)):.7f}; > 2^-11 (float16): {rel_tol(B) > F(1, 2**11)};"
          f" > 2^-8 (bfloat16): {rel_tol(B) > F(1, 2**8)}")

print("2. This kernel with its output rounded to float16 before the store, worst case by the claim's lemmas")
for B in [2**10, 2**11, 2**12, 2**13, 2**14]:
    d = F(13 * B + 20, 2**26)                      # sum_rel
    hD_rel, hD_abs = (u22 + d) * F(300, 299), F(1, 2**23) * F(300, 299)   # final_bound, hD
    c_rel = u22 + (1 + u22) * hD_rel               # with DivFull
    c_abs = (1 + u22) * hD_abs + F(1, 2**148)
    uh = F(1, 2**11)                               # one float16 rounding: 2^-11·|c| + 2^-25
    tot_rel = uh * (1 + c_rel) + c_rel
    tot_abs = uh * c_abs + c_abs + F(1, 2**25)
    ok = tot_rel <= rel_tol(B) and tot_abs <= ABS_TOL
    print(f"   B = {B:5d}: worst-case bound within Post (so Post holds on every covered instance): {ok}"
          f" (relative {float(tot_rel):.6f} vs {float(rel_tol(B)):.6f})")

print("3. absTol against the outputs of a 12672-column row")
N = 12672
typ = 1 / (N * math.exp(0.5))
print(f"   randn row: typical softmax {typ:.3e}; absTol / typical = {2**-22 / typ:.4f};"
      f" float16 rounding of a typical output <= {2**-11 * typ:.3e} < absTol: {2**-11 * typ < 2**-22}")
small = 1 / (math.exp(20) + N - 1)
print(f"   row [20, 0, ..., 0]: small outputs {small:.3e}; absTol / small = {2**-22 / small:.1f}")

print("4. The source's test (torch.allclose: rtol 1e-5, atol 1e-8) against Post at its BLOCK_SIZE 1024")
print(f"   relTol(1024) / 1e-5 = {float(rel_tol(1024)) / 1e-5:.1f}; absTol / 1e-8 = {2**-22 / 1e-8:.1f}")
