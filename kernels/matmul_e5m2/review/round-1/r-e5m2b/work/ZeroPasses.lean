import MatmulE5m2

/-! From `K = 2817` on, the exact value `0` meets the accuracy bound of `OutSpec` for every finite
row and column: `|0 - S| ≤ tol K A`. So a kernel that stores `+0` (the `float16` rounding of the
finite `float32` `+0`) in every element of `c`, and touches nothing else, meets `Post` at every
covered `K` in `[2817, 4096]`. -/

namespace Kernels.MatmulE5m2.Review

open TVA Kernels.MatmulE5m2

theorem zero_meets_tol {K : Nat} (hK : 2816 < K) (va vb : Nat → F8E5) :
    |(0 : ℚ) - dotSum va vb K| ≤ tol K (dotAbs va vb K) := by
  have hr := tol_rel_gt_one hK
  have hS : |dotSum va vb K| ≤ dotAbs va vb K := Finset.abs_sum_le_sum_abs _ _
  have hA : 0 ≤ dotAbs va vb K := Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  have hρ : (0 : ℚ) ≤ Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have hα : (0 : ℚ) ≤ Rules.Num.absDot8 64 := by rw [absDot8_64]; norm_num
  have habs : 0 ≤ (steps K : ℚ) * (1 + Rules.Num.relDot8 64) ^ steps K * Rules.Num.absDot8 64 :=
    mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg (by linarith) _)) hα
  have hm := mul_le_mul_of_nonneg_right hr.le hA
  rw [one_mul] at hm
  unfold tol DotLoop.bound
  rw [zero_sub, abs_neg]
  linarith

/-- The same at the benchmark's largest `K`. -/
example (va vb : Nat → F8E5) : |(0 : ℚ) - dotSum va vb 4096| ≤ tol 4096 (dotAbs va vb 4096) :=
  zero_meets_tol (by norm_num) va vb

/-- And a result half the exact sum meets it from `K = 1665` on (`(1 + ρ)^27 − 1 ≥ 1/2`). -/
theorem half_meets_tol {K : Nat} (hK : 1664 < K) (va vb : Nat → F8E5) :
    |dotSum va vb K / 2 - dotSum va vb K| ≤ tol K (dotAbs va vb K) := by
  have hT : 27 ≤ steps K := by unfold steps; omega
  have h1 : (1 : ℚ) ≤ 1 + Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have h2 := pow_le_pow_right₀ h1 hT
  have h3 : 3 / 2 ≤ (1 + Rules.Num.relDot8 64) ^ 27 := by rw [relDot8_64]; norm_num
  have hS : |dotSum va vb K| ≤ dotAbs va vb K := Finset.abs_sum_le_sum_abs _ _
  have hA : 0 ≤ dotAbs va vb K := Finset.sum_nonneg (fun _ _ => abs_nonneg _)
  have hα : (0 : ℚ) ≤ Rules.Num.absDot8 64 := by rw [absDot8_64]; norm_num
  have habs : 0 ≤ (steps K : ℚ) * (1 + Rules.Num.relDot8 64) ^ steps K * Rules.Num.absDot8 64 :=
    mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg (by linarith) _)) hα
  have hm := mul_le_mul_of_nonneg_right (show (1 : ℚ) / 2 ≤ (1 + Rules.Num.relDot8 64) ^ steps K - 1 by
    linarith) hA
  have e : dotSum va vb K / 2 - dotSum va vb K = -(dotSum va vb K / 2) := by ring
  unfold tol DotLoop.bound
  rw [e, abs_neg, abs_div, abs_two]
  linarith

end Kernels.MatmulE5m2.Review
