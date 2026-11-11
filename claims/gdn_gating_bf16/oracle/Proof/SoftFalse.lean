import Claim
import TVA.Proof

/-!
# The softplus, in the reals: the `x` branch

`t1 = fl(1*xf) > 20`, so the model keeps `xf` itself (`tl.where` takes the second operand).
-/

namespace Claims.GdnGating

open TVA

/-- `spRef x` is at least `x`. -/
lemma le_spRef (x : ℝ) : x ≤ spRef x := by
  rcases le_or_gt x 20 with h | h
  · have hsp : spRef x = Real.log (1 + Real.exp x) := by simp [spRef, h]
    rw [hsp]
    have hlog : Real.log (Real.exp x) ≤ Real.log (1 + Real.exp x) := by
      apply Real.log_le_log (Real.exp_pos x)
      linarith [Real.exp_pos x]
    calc x = Real.log (Real.exp x) := (Real.log_exp x).symm
      _ ≤ Real.log (1 + Real.exp x) := hlog
  · have hsp : spRef x = x := by simp [spRef, not_le.mpr h]
    rw [hsp]

/-- A numerical bound: `e^{-18} ≤ 2^{-14}`. -/
lemma exp_neg18_le : Real.exp (-(18 : ℝ)) ≤ (2 : ℝ) ^ (-14 : ℤ) := by
  have h2 : (2 : ℝ) ≤ Real.exp 1 := le_of_lt Real.exp_one_gt_two
  have h14 : (2 : ℝ) ^ 14 ≤ Real.exp 14 := by
    have h1 : (2 : ℝ) ^ 14 ≤ (Real.exp 1) ^ 14 := pow_le_pow_left₀ (by norm_num) h2 14
    have h2' : (Real.exp 1) ^ 14 = Real.exp 14 := by
      rw [← Real.exp_nat_mul]
      norm_num
    linarith
  have hle18 : Real.exp 14 ≤ Real.exp 18 := by
    apply Real.exp_le_exp.mpr
    norm_num
  have hp : (0 : ℝ) < Real.exp 18 := Real.exp_pos 18
  have h2p : (0 : ℝ) < (2 : ℝ) ^ 14 := by norm_num
  have h3 : (Real.exp 18)⁻¹ ≤ ((2 : ℝ) ^ 14)⁻¹ :=
    (inv_le_inv₀ hp h2p).mpr (le_trans h14 hle18)
  calc Real.exp (-(18 : ℝ)) = (Real.exp 18)⁻¹ := Real.exp_neg 18
    _ ≤ ((2 : ℝ) ^ 14)⁻¹ := h3
    _ = (2 : ℝ) ^ (-14 : ℤ) := by norm_num

/-- For `x ≥ 18`, `spRef x` exceeds `x` by at most `2^-14` (it is `x` above `20`, and
`log (1 + e^x) = x + log (1 + e^{-x}) ≤ x + e^{-x}` below). -/
lemma spRef_sub_le {x : ℝ} (h : 18 ≤ x) : spRef x - x ≤ (2 : ℝ) ^ (-14 : ℤ) := by
  rcases le_or_gt x 20 with hx | hx
  · have hsp : spRef x = Real.log (1 + Real.exp x) := by simp [spRef, hx]
    rw [hsp]
    have hmul : Real.exp x * Real.exp (-x) = 1 := by
      rw [← Real.exp_add]
      simp
    have hfac : (1 : ℝ) + Real.exp x = Real.exp x * (1 + Real.exp (-x)) := by
      rw [mul_add, mul_one, hmul]
      ring
    have hpos : (0 : ℝ) < 1 + Real.exp (-x) := by positivity
    have hexppos : (0 : ℝ) < Real.exp x := Real.exp_pos x
    have hlog : Real.log (1 + Real.exp x) = x + Real.log (1 + Real.exp (-x)) := by
      rw [hfac, Real.log_mul (ne_of_gt hexppos) (ne_of_gt hpos), Real.log_exp]
    have hle : Real.log (1 + Real.exp (-x)) ≤ Real.exp (-x) := by
      have := Real.log_le_sub_one_of_pos hpos
      linarith
    have hexp_le : Real.exp (-x) ≤ (2 : ℝ) ^ (-14 : ℤ) := by
      have hmono : Real.exp (-x) ≤ Real.exp (-(18 : ℝ)) := by
        apply Real.exp_le_exp.mpr
        linarith
      linarith [exp_neg18_le]
    linarith
  · have hsp : spRef x = x := by simp [spRef, not_le.mpr hx]
    rw [hsp]
    norm_num

/-- **The `x` branch** (`t1 = fl(1*xf) > 20`): the model keeps `xf` itself, and `spRef x0 = x0`
(up to `e^-20`) because `t1 > 20` forces `x0 > 20 - 2^-20`. -/
theorem softplus_false {x0 xf t1 : ℝ}
    (hxf : |xf - x0| ≤ 2 ^ (-24 : ℤ) * |x0| + 2 ^ (-150 : ℤ))
    (ht1 : |t1 - xf| ≤ 2 ^ (-24 : ℤ) * |xf| + 2 ^ (-150 : ℤ))
    (hgt : 20 < t1) :
    |xf - spRef x0| ≤ 52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) := by
  have hu : (2 : ℝ) ^ (-24 : ℤ) = ((2 : ℝ) ^ 24)⁻¹ := by norm_num
  have hU : (2 : ℝ) ^ (-23 : ℤ) = ((2 : ℝ) ^ 23)⁻¹ := by norm_num
  have he : (2 : ℝ) ^ (-150 : ℤ) = ((2 : ℝ) ^ 150)⁻¹ := by norm_num
  have h19 : (2 : ℝ) ^ (-19 : ℤ) = ((2 : ℝ) ^ 19)⁻¹ := by norm_num
  have h14 : (2 : ℝ) ^ (-14 : ℤ) = ((2 : ℝ) ^ 14)⁻¹ := by norm_num
  rw [hu, he] at hxf ht1
  rw [hu, hU, h19] at ⊢
  have hu_pos : (0 : ℝ) < ((2 : ℝ) ^ 24)⁻¹ := by positivity
  have h1u : (0 : ℝ) < 1 - ((2 : ℝ) ^ 24)⁻¹ := by norm_num
  have he1 : ((2 : ℝ) ^ 150)⁻¹ < 1 := by norm_num
  have hu100 : ((2 : ℝ) ^ 24)⁻¹ ≤ 1 / 100 := by norm_num
  have he100 : ((2 : ℝ) ^ 150)⁻¹ ≤ 1 / 100 := by norm_num
  -- Step 1: `t1 > 20` forces `xf ≥ 19`.
  have ht1_le : t1 - xf ≤ ((2 : ℝ) ^ 24)⁻¹ * |xf| + ((2 : ℝ) ^ 150)⁻¹ := (abs_le.mp ht1).2
  have hstep : 20 - ((2 : ℝ) ^ 24)⁻¹ * |xf| - ((2 : ℝ) ^ 150)⁻¹ < xf := by linarith
  have hxf_nonneg : 0 ≤ xf := by
    by_contra h
    have h : xf < 0 := not_le.mp h
    have habs : |xf| = -xf := abs_of_neg h
    rw [habs] at hstep
    have hneg : xf * (1 - ((2 : ℝ) ^ 24)⁻¹) < 0 := mul_neg_of_neg_of_pos h h1u
    nlinarith
  have hxf_abs : |xf| = xf := abs_of_nonneg hxf_nonneg
  have ht1_le' : t1 - xf ≤ ((2 : ℝ) ^ 24)⁻¹ * xf + ((2 : ℝ) ^ 150)⁻¹ := by
    rw [hxf_abs] at ht1_le
    exact ht1_le
  have huxf : ((2 : ℝ) ^ 24)⁻¹ * xf ≤ (1 / 100) * xf :=
    mul_le_mul_of_nonneg_right hu100 hxf_nonneg
  have hxf19 : 19 ≤ xf := by linarith
  -- Step 2: `|xf - x0| ≤ 2^-24 |x0| + 2^-150` and `xf ≥ 19` force `x0 ≥ 18`.
  have hxf_le0 : xf - x0 ≤ ((2 : ℝ) ^ 24)⁻¹ * |x0| + ((2 : ℝ) ^ 150)⁻¹ :=
    (abs_le.mp hxf).2
  have hx0_nonneg : 0 ≤ x0 := by
    by_contra h
    have h : x0 < 0 := not_le.mp h
    have habs : |x0| = -x0 := abs_of_neg h
    rw [habs] at hxf_le0
    have hneg : x0 * (1 - ((2 : ℝ) ^ 24)⁻¹) < 0 := mul_neg_of_neg_of_pos h h1u
    nlinarith
  have hx0_abs : |x0| = x0 := abs_of_nonneg hx0_nonneg
  rw [hx0_abs] at hxf
  have hxf_le' : |xf - x0| ≤ ((2 : ℝ) ^ 24)⁻¹ * x0 + ((2 : ℝ) ^ 150)⁻¹ := hxf
  have hxf_sub : xf - x0 ≤ ((2 : ℝ) ^ 24)⁻¹ * x0 + ((2 : ℝ) ^ 150)⁻¹ :=
    le_trans (le_abs_self _) hxf_le'
  have hux0 : ((2 : ℝ) ^ 24)⁻¹ * x0 ≤ (1 / 100) * x0 :=
    mul_le_mul_of_nonneg_right hu100 hx0_nonneg
  have hx0_18 : 18 ≤ x0 := by linarith
  -- Step 3: `spRef x0` is above `x0` and within `2^-14` of it.
  have hsp_ge : x0 ≤ spRef x0 := le_spRef x0
  have hsp18 : 18 ≤ spRef x0 := le_trans hx0_18 hsp_ge
  have hsp_le : spRef x0 - x0 ≤ ((2 : ℝ) ^ 14)⁻¹ := by
    have h := spRef_sub_le hx0_18
    rw [h14] at h
    exact h
  have htri : |xf - spRef x0| ≤ |xf - x0| + (spRef x0 - x0) := by
    have h := abs_sub_le xf x0 (spRef x0)
    rw [abs_sub_comm x0 (spRef x0),
      abs_of_nonneg (by linarith : (0 : ℝ) ≤ spRef x0 - x0)] at h
    exact h
  have hstep1 : |xf - spRef x0| ≤
      ((2 : ℝ) ^ 24)⁻¹ * spRef x0 + ((2 : ℝ) ^ 150)⁻¹ + ((2 : ℝ) ^ 14)⁻¹ := by
    have hux : ((2 : ℝ) ^ 24)⁻¹ * x0 ≤ ((2 : ℝ) ^ 24)⁻¹ * spRef x0 :=
      mul_le_mul_of_nonneg_left hsp_ge (le_of_lt hu_pos)
    linarith
  -- Step 4: the numerical constant comparison.
  have hc : (0 : ℝ) ≤ 52 * ((2 : ℝ) ^ 23)⁻¹ - ((2 : ℝ) ^ 24)⁻¹ := by norm_num
  have hkey : 927 * ((2 : ℝ) ^ 23)⁻¹ ≤
      (52 * ((2 : ℝ) ^ 23)⁻¹ - ((2 : ℝ) ^ 24)⁻¹) * spRef x0 := by
    have h := mul_le_mul_of_nonneg_left hsp18 hc
    have hnum : (52 * ((2 : ℝ) ^ 23)⁻¹ - ((2 : ℝ) ^ 24)⁻¹) * 18 =
        927 * ((2 : ℝ) ^ 23)⁻¹ := by norm_num
    linarith
  have hnum2 : ((2 : ℝ) ^ 150)⁻¹ + ((2 : ℝ) ^ 14)⁻¹ ≤ 927 * ((2 : ℝ) ^ 23)⁻¹ := by
    norm_num
  have h1 : ((2 : ℝ) ^ 24)⁻¹ * spRef x0 + ((2 : ℝ) ^ 150)⁻¹ + ((2 : ℝ) ^ 14)⁻¹ ≤
      52 * ((2 : ℝ) ^ 23)⁻¹ * spRef x0 := by
    have h2 : 52 * ((2 : ℝ) ^ 23)⁻¹ * spRef x0 =
        ((2 : ℝ) ^ 24)⁻¹ * spRef x0 +
          (52 * ((2 : ℝ) ^ 23)⁻¹ - ((2 : ℝ) ^ 24)⁻¹) * spRef x0 := by
      ring
    linarith
  have hfin : ((2 : ℝ) ^ 24)⁻¹ * spRef x0 + ((2 : ℝ) ^ 150)⁻¹ + ((2 : ℝ) ^ 14)⁻¹ ≤
      52 * ((2 : ℝ) ^ 23)⁻¹ * spRef x0 +
        ((2 : ℝ) ^ 24)⁻¹ * (1 + ((2 : ℝ) ^ 19)⁻¹) := by
    have hsplit : ((2 : ℝ) ^ 24)⁻¹ * (1 + ((2 : ℝ) ^ 19)⁻¹) =
        ((2 : ℝ) ^ 24)⁻¹ + ((2 : ℝ) ^ 24)⁻¹ * ((2 : ℝ) ^ 19)⁻¹ := by ring
    have hu19 : (0 : ℝ) ≤ ((2 : ℝ) ^ 24)⁻¹ * ((2 : ℝ) ^ 19)⁻¹ := by positivity
    rw [hsplit]
    linarith
  linarith

end Claims.GdnGating
