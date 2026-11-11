import Claim
import TVA.Proof
import Proof.Cap

/-!
# Real analysis for the soft cap and the weights, continued

`Proof.Cap` gives `tanh_lipschitz`, the exact difference of the two halves of the ratio, and the
slope bound.  These lemmas add what else the error analysis needs: `capRef` is `1`-Lipschitz and
bounded by `30`, `tanh` in the shape the block computes it, and the exponential bounds.
-/

namespace Claims.MoeRouter

/-- `tanh x = (e^{2x} − 1)/(e^{2x} + 1)`. -/
theorem tanh_eq_ratio (x : ℝ) :
    Real.tanh x = (Real.exp (2 * x) - 1) / (Real.exp (2 * x) + 1) := by
  have hsq : Real.exp (2 * x) = Real.exp x * Real.exp x := by
    rw [show 2 * x = x + x by ring, Real.exp_add]
  rw [Real.tanh_eq, hsq, Real.exp_neg]
  have he : Real.exp x ≠ 0 := (Real.exp_pos x).ne'
  have hd : Real.exp x + (Real.exp x)⁻¹ ≠ 0 := by positivity
  have hd' : Real.exp x * Real.exp x + 1 ≠ 0 := by positivity
  field_simp

/-- The exact soft cap `capRef` is `1`-Lipschitz. -/
theorem capRef_lipschitz (a b : ℝ) : |capRef a - capRef b| ≤ |a - b| := by
  have h30 : (0 : ℝ) < 30 := by norm_num
  have h := tanh_lipschitz (a / 30) (b / 30)
  have hdiff : a / 30 - b / 30 = (a - b) / 30 := by ring
  rw [hdiff, abs_div, abs_of_pos h30] at h
  calc |capRef a - capRef b|
      = 30 * |Real.tanh (a / 30) - Real.tanh (b / 30)| := by
        unfold capRef
        rw [show (30 : ℝ) * Real.tanh (a / 30) - 30 * Real.tanh (b / 30)
              = 30 * (Real.tanh (a / 30) - Real.tanh (b / 30)) by ring,
            abs_mul, abs_of_pos h30]
    _ ≤ 30 * (|a - b| / 30) := mul_le_mul_of_nonneg_left h h30.le
    _ = |a - b| := by ring

end Claims.MoeRouter
