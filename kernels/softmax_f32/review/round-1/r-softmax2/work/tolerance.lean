import Softmax

/-! r-softmax2, round 1: the kernel's own `relTol` and `absTol` against a rounding of the exact
softmax to a lower precision, at the block sizes the host's benchmark uses.

The largest relative error of rounding to nearest into a format with `p` bits of precision (normal
range) is `2^-p / (1 + 2^-p)`: `bfloat16` has `p = 8`, `float16` `p = 11`. -/

open Kernels.Softmax

/-- The largest relative rounding error of a `p`-bit format (normal range). -/
def maxRel (p : Nat) : ℚ := (1 / 2 ^ p) / (1 + 1 / 2 ^ p)

-- `relTol 16384` (the host's 35 largest benchmark shapes) exceeds bfloat16's largest relative
-- rounding error: a kernel returning the exact softmax rounded to bfloat16 meets `Post` there.
example : maxRel 8 < relTol 16384 := by norm_num [maxRel, relTol]
-- At `8192` it does not.
example : relTol 8192 < maxRel 8 := by norm_num [maxRel, relTol]
-- `relTol 2048` (and above) exceeds float16's: the exact softmax rounded to float16 meets it.
example : maxRel 11 < relTol 2048 := by norm_num [maxRel, relTol]
example : relTol 1024 < maxRel 11 := by norm_num [maxRel, relTol]

-- The relative coefficient `final_bound`'s own steps give, before its last step rounds it up to
-- `(B + 4)·2^-22`: `2^-22 + (1 + 2^-22)·(2^-22 + δ)·(300/299)` with `sum_rel`'s
-- `δ = (13·B + 20)·2^-26` (`hD`, `hy'`). It stays below bfloat16's error at `B = 2^14` (about
-- `3.19·10^-3`), and below float16's at `2048`.
def proofRel (B : Nat) : ℚ :=
  1 / 2 ^ 22 + (1 + 1 / 2 ^ 22) * (1 / 2 ^ 22 + (13 * B + 20) / 2 ^ 26) * (300 / 299)
example : proofRel 16384 < maxRel 8 := by norm_num [proofRel, maxRel]
example : proofRel 16384 < 319 / 10 ^ 5 := by norm_num [proofRel]
example : proofRel 2048 < maxRel 11 := by norm_num [proofRel, maxRel]

-- The library's bound for a sum along any tree of `B` leaves is `(1 + 2^-24)^(B-1) - 1`, at most
-- `(B-1)·2^-24 / (1 - (B-1)·2^-24)` (`DotLoop.pow_mul_le`, as `pow_u_facts` uses it). At
-- `B = 2^14` that is below `relTol 16384 / 3.99`, and below a third of bfloat16's error.
example : ((16383 : ℚ) / 2 ^ 24) / (1 - 16383 / 2 ^ 24) < relTol 16384 / (399 / 100) := by
  norm_num [relTol]
example : ((16383 : ℚ) / 2 ^ 24) / (1 - 16383 / 2 ^ 24) < maxRel 8 / 3 := by norm_num [maxRel]

-- `absTol` against the outputs it lets go to zero: every exact softmax value at most
-- `absTol / (1 - relTol 16384)` may be `+0`; `e^-16 < 1.13·10^-7` is below it.
example : (113 / 10 ^ 9 : ℚ) < absTol / (1 - relTol 16384) := by norm_num [absTol, relTol]
-- `absTol` is more than 23 times the source test's `atol` (`torch.allclose`, `1e-8`).
example : (23 / 10 ^ 8 : ℚ) < absTol := by norm_num [absTol]
