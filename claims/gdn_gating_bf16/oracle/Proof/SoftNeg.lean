import Claim
import TVA.Proof

/-!
# Softplus for a very negative input

For `x0 ≤ -20` and a computed value `l2` of size at most `2^-140`, `|l2 - spRef x0|` is within the softplus tolerance.
-/

namespace Claims.GdnGating

open TVA

/-- For a very negative exact input, a softplus value near zero is within the softplus tolerance. -/
theorem softplus_neg {x0 l2 : ℝ} (hx0 : x0 ≤ -20) (hl2 : |l2| ≤ (2 : ℝ) ^ (-140 : ℤ)) :
    |l2 - spRef x0| ≤
      52 * (2 : ℝ) ^ (-23 : ℤ) * spRef x0 + (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-19 : ℤ)) := by
  have hx20 : x0 ≤ 20 := by linarith
  have hsp : spRef x0 = Real.log (1 + Real.exp x0) := by
    simp only [spRef]
    exact ite_eq_left hx20
  have hS_nonneg : 0 ≤ spRef x0 := by
    rw [hsp]
    exact Real.log_nonneg (by linarith [Real.exp_pos x0])
  have hS_le : spRef x0 ≤ Real.exp x0 := by
    rw [hsp]
    have hpos : 0 < 1 + Real.exp x0 := by linarith [Real.exp_pos x0]
    have h := Real.log_le_sub_one_of_pos hpos
    linarith
  have hexp_le : Real.exp x0 ≤ (2 : ℝ) ^ (-24 : ℤ) := by
    have hx : Real.exp x0 ≤ Real.exp (-20) := Real.exp_le_exp.mpr hx0
    have hlog : (24 : ℝ) * Real.log 2 ≤ 20 := by nlinarith [Real.log_two_lt_d9]
    have h24 : (2 : ℝ) ^ 24 ≤ Real.exp 20 := by
      have h := Real.exp_le_exp.mpr hlog
      rw [show (24 : ℝ) = ((24 : ℕ) : ℝ) by norm_num, Real.exp_nat_mul,
        Real.exp_log (by norm_num : (0 : ℝ) < 2)] at h
      exact h
    have hexp20 : Real.exp (-20) ≤ (2 : ℝ) ^ (-24 : ℤ) := by
      calc Real.exp (-20) = (Real.exp 20)⁻¹ := Real.exp_neg 20
        _ ≤ ((2 : ℝ) ^ 24)⁻¹ := (inv_le_inv₀ (Real.exp_pos 20) (by positivity)).mpr h24
        _ = (2 : ℝ) ^ (-24 : ℤ) := by
          rw [← zpow_natCast (2 : ℝ) 24, ← zpow_neg]
          norm_num
    exact hx.trans hexp20
  have hnum : (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (-19 : ℤ) := by
    calc (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-43 : ℤ) :=
          zpow_le_zpow_right₀ (by norm_num) (by norm_num)
      _ = (2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (-19 : ℤ) := by
          rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
          norm_num
  have hc : (0 : ℝ) ≤ 52 * (2 : ℝ) ^ (-23 : ℤ) * spRef x0 := by positivity
  have hS2 : spRef x0 ≤ (2 : ℝ) ^ (-24 : ℤ) := hS_le.trans hexp_le
  calc |l2 - spRef x0| ≤ |l2| + |spRef x0| := abs_sub l2 (spRef x0)
    _ = |l2| + spRef x0 := by rw [abs_of_nonneg hS_nonneg]
    _ ≤ (2 : ℝ) ^ (-140 : ℤ) + (2 : ℝ) ^ (-24 : ℤ) := by linarith
    _ ≤ (2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (-19 : ℤ) + (2 : ℝ) ^ (-24 : ℤ) := by linarith
    _ = (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-19 : ℤ)) := by ring
    _ ≤ 52 * (2 : ℝ) ^ (-23 : ℤ) * spRef x0 +
        (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-19 : ℤ)) := by linarith

end Claims.GdnGating
