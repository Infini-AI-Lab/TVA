import Proof.Rows

/-!
# Constants and exponential estimates

The unit roundoff `uR = 2^-24` and the small powers of two the row bound uses, and elementary
bounds on `Real.exp` (differences, small arguments, large negative arguments).
-/
namespace Claims.Softmax

open TVA

/-! ## Numeric constants, in `ℝ` -/

/-- The unit roundoff of `float32`, `2^-24`. -/
noncomputable def uR : ℝ := 2 ^ (-24 : ℤ)
/-- The smallest subnormal of `float32`, `2^-150` (half of it is the absolute rounding error). -/
noncomputable def etaR : ℝ := 2 ^ (-150 : ℤ)

/-- A numeric fact about the constant. -/
lemma uR_pos : 0 < uR := by unfold uR; norm_num
/-- A numeric fact about the constant. -/
lemma etaR_pos : 0 < etaR := by unfold etaR; norm_num
/-- A numeric fact about the constant. -/
lemma uR_le : uR ≤ 1 / 4 := by unfold uR; norm_num
/-- A numeric fact about the constant. -/
lemma etaR_le_uR : etaR ≤ uR := by unfold uR etaR; norm_num
/-- A numeric fact about powers of two. -/
lemma e22 : (2 : ℝ) ^ (-22 : ℤ) = 4 * uR := by unfold uR; norm_num
/-- A numeric fact about powers of two. -/
lemma e23 : (2 : ℝ) ^ (-23 : ℤ) = 2 * uR := by unfold uR; norm_num
/-- A numeric fact about powers of two. -/
lemma e21 : (2 : ℝ) ^ (-21 : ℤ) = 8 * uR := by unfold uR; norm_num
/-- A numeric fact about powers of two. -/
lemma e20 : (2 : ℝ) ^ (-20 : ℤ) = 16 * uR := by unfold uR; norm_num
/-- A numeric fact about powers of two. -/
lemma e17 : (2 : ℝ) ^ (-17 : ℤ) = 128 * uR := by unfold uR; norm_num
/-- A numeric fact about powers of two. -/
lemma e18 : (2 : ℝ) ^ (-18 : ℤ) = 64 * uR := by unfold uR; norm_num
/-- A numeric fact about powers of two. -/
lemma e24 : (2 : ℝ) ^ (-24 : ℤ) = uR := rfl
/-- A numeric fact about powers of two. -/
lemma e33_le : (2 : ℝ) ^ (-33 : ℤ) ≤ uR := by unfold uR; norm_num
/-- A numeric fact about powers of two. -/
lemma e139_le : (2 : ℝ) ^ (-139 : ℤ) ≤ (2 : ℝ) ^ (-133 : ℤ) := by norm_num
/-- A numeric fact about powers of two. -/
lemma e140_le : (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-133 : ℤ) := by norm_num
/-- A numeric fact about powers of two. -/
lemma e148_le : (2 : ℝ) ^ (-148 : ℤ) ≤ (2 : ℝ) ^ (-133 : ℤ) := by norm_num
/-- A numeric fact about powers of two. -/
lemma e150_le_133 : (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-133 : ℤ) := by norm_num
/-- A numeric fact about powers of two. -/
lemma e140_pos : (0:ℝ) < (2 : ℝ) ^ (-140 : ℤ) := by norm_num
/-- A numeric fact about powers of two. -/
lemma e133_pos : (0:ℝ) < (2 : ℝ) ^ (-133 : ℤ) := by norm_num

/-! ## Elementary exponential facts -/

/-- `e^t − e^s ≤ (t − s)·e^t` for `s ≤ t`. -/
lemma exp_sub_le {s t : ℝ} (h : s ≤ t) :
    Real.exp t - Real.exp s ≤ (t - s) * Real.exp t := by
  have h6 : Real.exp (s - t) = Real.exp (-(t - s)) := by ring_nf
  have h7 : Real.exp s = Real.exp t * Real.exp (s - t) := by
    rw [← Real.exp_add]; ring_nf
  have h8 : (0:ℝ) < Real.exp t := Real.exp_pos t
  have h5 : 1 - (t - s) ≤ Real.exp (-(t - s)) := Real.one_sub_le_exp_neg (t - s)
  have hx : (0:ℝ) ≤ t - s := by linarith
  calc Real.exp t - Real.exp s = Real.exp t * (1 - Real.exp (s - t)) := by rw [h7]; ring
    _ ≤ Real.exp t * (t - s) := by
        apply mul_le_mul_of_nonneg_left _ h8.le
        rw [h6]; linarith
    _ = (t - s) * Real.exp t := by ring

/-- `|e^s − e^t| ≤ (t − s)·e^t` for `s ≤ t`. -/
lemma abs_exp_sub_le {s t : ℝ} (h : s ≤ t) :
    |Real.exp s - Real.exp t| ≤ (t - s) * Real.exp t := by
  have h1 : Real.exp s ≤ Real.exp t := Real.exp_le_exp.mpr h
  have h2 : (0:ℝ) ≤ (t - s) * Real.exp t := by
    have := Real.exp_pos t; nlinarith
  rw [abs_of_nonpos (by linarith), neg_sub]
  exact exp_sub_le h

/-- `|e^s − e^t| ≤ (s − t)·e^s` for `t ≤ s`. -/
lemma abs_exp_sub_le' {s t : ℝ} (h : t ≤ s) :
    |Real.exp s - Real.exp t| ≤ (s - t) * Real.exp s := by
  rw [abs_sub_comm]
  exact abs_exp_sub_le h

/-- For `0 ≤ x ≤ 1/2`, `e^x ≤ 1 + 2x`. -/
lemma exp_le_one_add_two_mul {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1/2) :
    Real.exp x ≤ 1 + 2 * x := by
  have h2 : 1 - x ≤ Real.exp (-x) := Real.one_sub_le_exp_neg x
  have h3 : 0 < 1 - x := by linarith
  have h4 : Real.exp x = (Real.exp (-x))⁻¹ := by
    rw [← Real.exp_neg]; ring_nf
  rw [h4]
  have h5 : (Real.exp (-x))⁻¹ ≤ (1 - x)⁻¹ := by
    rw [inv_le_inv₀ (Real.exp_pos (-x)) h3]
    linarith
  have h6 : (1 - x)⁻¹ ≤ 1 + 2 * x := by
    rw [inv_le_iff_one_le_mul₀ h3]
    nlinarith
  linarith

/-- For `32 ≤ z`, `(3z+7)·e^{-z} ≤ 103·e^{-32}`. -/
lemma three_mul_exp_le {z : ℝ} (hz : 32 ≤ z) :
    (3 * z + 7) * Real.exp (-z) ≤ 103 * Real.exp (-32) := by
  have h1 : 1 + 3 * (z - 32) / 103 ≤ Real.exp (3 * (z - 32) / 103) := by
    have := Real.add_one_le_exp (3 * (z - 32) / 103); linarith
  have h2 : Real.exp (3 * (z - 32) / 103) ≤ Real.exp (z - 32) := by
    apply Real.exp_le_exp.mpr; nlinarith
  have key : 103 + 3 * (z - 32) ≤ 103 * Real.exp (z - 32) := by nlinarith
  have h3 : 3 * z + 7 ≤ 103 * Real.exp (z - 32) := by linarith
  have h5 : Real.exp (-z) = Real.exp (-32) * (Real.exp (z - 32))⁻¹ := by
    rw [show (-z : ℝ) = -32 + (-(z-32)) by ring, Real.exp_add,
      show Real.exp (-(z-32)) = (Real.exp (z - 32))⁻¹ by
        rw [show (-(z-32) : ℝ) = -(z-32) by ring, Real.exp_neg]]
  have hexp30 : (0:ℝ) < Real.exp (-32) := Real.exp_pos _
  have hexpz : (0:ℝ) < Real.exp (z - 32) := Real.exp_pos _
  have hmul : (3*z+7) * Real.exp (-z) = Real.exp (-32) * ((3*z+7) / Real.exp (z-32)) := by
    rw [h5]; field_simp
  rw [hmul]
  have hdiv : (3*z+7) / Real.exp (z-32) ≤ 103 := by
    rw [div_le_iff₀ hexpz]; linarith
  calc Real.exp (-32) * ((3*z+7) / Real.exp (z-32)) ≤ Real.exp (-32) * 103 :=
        mul_le_mul_of_nonneg_left hdiv hexp30.le
    _ = 103 * Real.exp (-32) := by ring

/-- For `x ≥ 133`, `e^{-x} ≤ 2^-133`. -/
lemma exp_neg_le_small {x : ℝ} (hx : 133 ≤ x) : Real.exp (-x) ≤ (2:ℝ)^(-133:ℤ) := by
  have h1 : Real.exp (133:ℝ) ≤ Real.exp x := Real.exp_le_exp.mpr hx
  have h2 : Real.exp (133:ℝ) = (Real.exp 1) ^ 133 := by
    rw [← Real.exp_nat_mul]; norm_num
  have h3 : (2:ℝ) ≤ Real.exp 1 := le_of_lt Real.exp_one_gt_two
  have h4 : (2:ℝ)^133 ≤ (Real.exp 1)^133 := pow_le_pow_left₀ (by norm_num) h3 133
  have h5 : (2:ℝ)^133 ≤ Real.exp x := by rw [h2] at h1; linarith
  have h6 : Real.exp (-x) = (Real.exp x)⁻¹ := by rw [← Real.exp_neg]
  have h9 : (2:ℝ)^(-133:ℤ) = ((2:ℝ)^133)⁻¹ := by norm_num
  rw [h6, h9]
  exact (inv_le_inv₀ (Real.exp_pos x) (by positivity)).mpr h5

end Claims.Softmax
