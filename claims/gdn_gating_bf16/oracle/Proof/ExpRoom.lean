import Claim
import TVA.Proof

/-!
# Room for `tl.exp`

`ExpRoom t` holds for every `t` between `-10^40` and `21`: the exponential's error bound at `|t|` fits `float32`. It is used for the softplus's exponential, whose argument is at most about `20`.
-/

namespace Claims.GdnGating

open TVA

/-- `ExpRoom` holds well below `float32`'s exponential ceiling. -/
theorem expRoom_of_bounds {t : ℝ} (hlo : -((10 : ℝ) ^ 40) ≤ t) (hhi : t ≤ 21) : ExpRoom t := by
  unfold ExpRoom
  have hmax : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (10:ℝ)^40)) * 1 + 2 ^ (-140 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal]
    norm_num
  by_cases hle : t ≤ 0
  · have habs : |t| ≤ (10:ℝ)^40 := by
      rw [abs_of_nonpos hle]
      linarith
    have hexp : Real.exp t ≤ 1 := Real.exp_le_one_iff.mpr hle
    have hc : (0:ℝ) ≤ 2 ^ (-23 : ℤ) := by positivity
    have hmul := mul_le_mul_of_nonneg_left habs hc
    have hA : 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|) ≤
        1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (10:ℝ)^40) := by linarith
    have hA0 : (0:ℝ) ≤ 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (10:ℝ)^40) := by positivity
    have hprod : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|)) * Real.exp t ≤
        (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (10:ℝ)^40)) * 1 :=
      mul_le_mul hA hexp (Real.exp_nonneg t) hA0
    calc (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|)) * Real.exp t + 2 ^ (-140 : ℤ)
        ≤ (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (10:ℝ)^40)) * 1 + 2 ^ (-140 : ℤ) :=
          add_le_add hprod (le_refl _)
      _ ≤ (Format.binary32.maxVal : ℝ) := hmax
  · have hpos : 0 < t := lt_of_not_ge hle
    have habs : |t| = t := abs_of_pos hpos
    rw [habs]
    have hexp21 : Real.exp 21 ≤ (2:ℝ)^31 := by
      have h : (21:ℝ) ≤ 31 * Real.log 2 := by nlinarith [Real.log_two_gt_d9]
      have h2 : Real.exp (31 * Real.log 2) = (2:ℝ)^31 := by
        rw [show (31:ℝ) = ((31:ℕ):ℝ) by norm_num, Real.exp_nat_mul,
          Real.exp_log (by norm_num)]
      have h3 := Real.exp_le_exp.mpr h
      rwa [h2] at h3
    have hexp : Real.exp t ≤ (2:ℝ)^31 := le_trans (Real.exp_le_exp.mpr hhi) hexp21
    have hc : (0:ℝ) ≤ 2 ^ (-23 : ℤ) := by positivity
    have hmul := mul_le_mul_of_nonneg_left hhi hc
    have hA : 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * t) ≤
        1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 21) := by linarith
    have hA0 : (0:ℝ) ≤ 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 21) := by positivity
    have hprod : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * t)) * Real.exp t ≤
        (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 21)) * (2:ℝ)^31 :=
      mul_le_mul hA hexp (Real.exp_nonneg t) hA0
    have hnum : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 21)) * (2:ℝ)^31 + 2 ^ (-140 : ℤ) ≤
        (Format.binary32.maxVal : ℝ) := by
      rw [Format.binary32_maxVal]
      norm_num
    calc (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * t)) * Real.exp t + 2 ^ (-140 : ℤ)
        ≤ (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 21)) * (2:ℝ)^31 + 2 ^ (-140 : ℤ) :=
          add_le_add hprod (le_refl _)
      _ ≤ (Format.binary32.maxVal : ℝ) := hnum

/-- `ExpRoom` forces its argument below 89. -/
theorem expRoom_le {t : ℝ} (h : ExpRoom t) : t ≤ 89 := by
  by_contra hc
  push Not at hc
  unfold ExpRoom at h
  have hlog : (128:ℝ) * Real.log 2 < 89 := by nlinarith [Real.log_two_lt_d9]
  have hexp2 : Real.exp (128 * Real.log 2) = (2:ℝ)^128 := by
    rw [show (128:ℝ) = ((128:ℕ):ℝ) by norm_num, Real.exp_nat_mul,
      Real.exp_log (by norm_num)]
  have he1 : (2:ℝ)^128 < Real.exp 89 := by
    have h3 := Real.exp_lt_exp.mpr hlog
    rwa [hexp2] at h3
  have hmax : (Format.binary32.maxVal : ℝ) < Real.exp 89 := by
    have hm : (Format.binary32.maxVal : ℝ) < (2:ℝ)^128 := by
      rw [Format.binary32_maxVal]
      norm_num
    linarith
  have hf : (1:ℝ) ≤ 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|) := by
    have h1 : (0:ℝ) ≤ 2 ^ (-22 : ℤ) := by positivity
    have h2 : (0:ℝ) ≤ 2 ^ (-23 : ℤ) * |t| := by positivity
    linarith
  have he : Real.exp 89 < Real.exp t := Real.exp_lt_exp.mpr hc
  have hge : Real.exp t ≤ (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|)) * Real.exp t := by
    calc Real.exp t = 1 * Real.exp t := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right hf (Real.exp_nonneg t)
  have hchain : Real.exp t ≤ (Format.binary32.maxVal : ℝ) := by
    calc Real.exp t ≤ (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|)) * Real.exp t := hge
      _ ≤ (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |t|)) * Real.exp t + 2 ^ (-140 : ℤ) := by
          have h0 : (0:ℝ) ≤ 2 ^ (-140 : ℤ) := by positivity
          linarith
      _ ≤ (Format.binary32.maxVal : ℝ) := h
  linarith

end Claims.GdnGating
