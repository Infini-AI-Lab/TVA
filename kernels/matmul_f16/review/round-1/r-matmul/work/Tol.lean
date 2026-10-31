import Matmul

/-! r-matmul, round 1: the tolerance of `Post` against one `bfloat16` rounding (relative error up to
`2^-8`), at the largest covered `K = 2^16` and at `K = 2^13`. -/

open Kernels.Matmul TVA

theorem relDot16_64' : Rules.Num.relDot16 64 = 66 / 2 ^ 22 := by
  unfold Rules.Num.relDot16; norm_num

theorem absDot16_64_nonneg : 0 ≤ Rules.Num.absDot16 64 := by
  unfold Rules.Num.absDot16; positivity

/-- At `K = 2^16` (1024 steps) the tolerance exceeds `2^-8·A`: a result rounded to `bfloat16`
instead of `float16`, from an accumulation well inside the library's bound, can meet `Post`. -/
theorem tol_big (A : ℚ) (hA : 0 < A) : A / 2 ^ 8 < tol (2 ^ 16) A := by
  unfold tol steps DotLoop.bound
  rw [show ((2 : ℕ) ^ 16 + 63) / 64 = 1024 by norm_num, relDot16_64']
  have hb : 1 + ((1024 : ℕ) : ℚ) * (66 / 2 ^ 22) ≤ (1 + 66 / 2 ^ 22 : ℚ) ^ 1024 :=
    one_add_mul_le_pow (by norm_num) 1024
  have hα := absDot16_64_nonneg
  have hc : 1 + ((1024 : ℕ) : ℚ) * (66 / 2 ^ 22) = 1 + 66 / 2 ^ 12 := by norm_num
  rw [hc] at hb
  generalize (1 + 66 / 2 ^ 22 : ℚ) ^ 1024 = x at hb ⊢
  have hx0 : 0 ≤ x := by linarith
  have h1 : (0 : ℚ) ≤ ((1024 : ℕ) : ℚ) * x * Rules.Num.absDot16 64 := by positivity
  have h2 : A / 2 ^ 8 < (x - 1) * A := by
    have : (1 : ℚ) / 2 ^ 8 < x - 1 := by linarith
    nlinarith
  push_cast at h1 ⊢
  linarith

/-- At `K = 2^13` (128 steps) the relative part of the tolerance is below `2^-8`. -/
theorem tol_mid : (1 + Rules.Num.relDot16 64) ^ steps (2 ^ 13) - 1 < 1 / 2 ^ 8 := by
  rw [relDot16_64', show steps (2 ^ 13) = 128 by unfold steps; norm_num]
  norm_num
