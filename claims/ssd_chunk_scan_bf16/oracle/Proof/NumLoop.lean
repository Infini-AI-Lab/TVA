import Proof.Bounds
import Proof.Zero

/-!
# The loop: numeric facts

One weight of the causal sum (lines 398–414) and the accumulation loop (line 420): the invariant
`LoopInv`, its start, one trip and its end.
-/

namespace Claims.SsdChunkScan

open TVA

/-- Auxiliary fact: maxVal32 eq. -/
private lemma maxVal32_eq : (Format.binary32.maxVal : ℝ) = ((2:ℝ)^24 - 1) * (2:ℝ)^104 := by
  rw [Format.binary32_maxVal]; push_cast; norm_num

/-- Auxiliary fact: maxVal32 le pow128. -/
private lemma maxVal32_le_pow128 : (Format.binary32.maxVal : ℝ) ≤ (2:ℝ)^128 := by
  rw [maxVal32_eq]
  have h : ((2:ℝ)^24 - 1) * (2:ℝ)^104 ≤ (2:ℝ)^24 * (2:ℝ)^104 :=
    mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  have h2 : (2:ℝ)^24 * (2:ℝ)^104 = (2:ℝ)^128 := by
    rw [show (128:ℕ) = 24 + 104 by norm_num, pow_add]
  linarith

/-- Auxiliary fact: pow109 le maxVal32. -/
private lemma pow109_le_maxVal32 : (2:ℝ)^109 ≤ (Format.binary32.maxVal : ℝ) := by
  rw [maxVal32_eq]
  have h32 : (32:ℝ) ≤ (2:ℝ)^24 - 1 := by norm_num
  have h := mul_le_mul_of_nonneg_right h32 (show (0:ℝ) ≤ (2:ℝ)^104 by positivity)
  have h2 : (32:ℝ) * (2:ℝ)^104 = (2:ℝ)^109 := by
    rw [show (109:ℕ) = 5 + 104 by norm_num, pow_add]; norm_num
  linarith

/-- Auxiliary fact: pow126 le maxVal32. -/
private lemma pow126_le_maxVal32 : (2:ℝ)^126 ≤ (Format.binary32.maxVal : ℝ) := by
  rw [maxVal32_eq]
  have h32 : (2:ℝ)^22 ≤ (2:ℝ)^24 - 1 := by norm_num
  have h := mul_le_mul_of_nonneg_right h32 (show (0:ℝ) ≤ (2:ℝ)^104 by positivity)
  have h2 : (2:ℝ)^22 * (2:ℝ)^104 = (2:ℝ)^126 := by
    rw [show (126:ℕ) = 22 + 104 by norm_num, pow_add]
  linarith

/-- Auxiliary fact: maxValbf eq. -/
private lemma maxValbf_eq : (Format.bfloat16.maxVal : ℝ) = ((2:ℝ)^8 - 1) * (2:ℝ)^120 := by
  rw [Format.bfloat16_maxVal]; push_cast; norm_num

/-- Auxiliary fact: three pow126 le maxValbf. -/
private lemma three_pow126_le_maxValbf : 3 * (2:ℝ)^126 ≤ (Format.bfloat16.maxVal : ℝ) := by
  rw [maxValbf_eq]
  have h : (3:ℝ) * (2:ℝ)^126 = (192:ℝ) * (2:ℝ)^120 := by
    have h1 : (2:ℝ)^126 = (2:ℝ)^6 * (2:ℝ)^120 := by
      rw [show (126:ℕ) = 6 + 120 by norm_num, pow_add]
    rw [h1]; ring
  rw [h]
  have h2 : (192:ℝ) ≤ (2:ℝ)^8 - 1 := by norm_num
  exact mul_le_mul_of_nonneg_right h2 (by positivity)

/-- Auxiliary fact: exp65 le pow104. -/
private lemma exp65_le_pow104 : Real.exp 65 ≤ (2:ℝ)^104 := by
  have h : Real.exp 65 = (Real.exp 1)^65 := by
    rw [show (65:ℝ) = (65:ℕ) * 1 by norm_num, Real.exp_nat_mul]
  rw [h]
  have h1 : (Real.exp 1)^65 ≤ (3:ℝ)^65 :=
    pow_le_pow_left₀ (Real.exp_pos 1).le Real.exp_one_lt_three.le 65
  have h2 : (3:ℝ)^65 ≤ (2:ℝ)^104 := by norm_num
  linarith

/-- Auxiliary fact: exp sub le. -/
private lemma exp_sub_le {D t e : ℝ} (_he0 : 0 ≤ e) (ht : |t - D| ≤ e) :
    |Real.exp t - Real.exp D| ≤ Real.exp (D + e) - Real.exp D := by
  have ht1 : t ≤ D + e := by have := (abs_le.mp ht).2; linarith
  have ht2 : D - e ≤ t := by have := (abs_le.mp ht).1; linarith
  have hA : (0:ℝ) < Real.exp D := Real.exp_pos D
  have hx : (0:ℝ) < Real.exp e := Real.exp_pos e
  have hkey : 2 ≤ Real.exp e + (Real.exp e)⁻¹ := by
    have hsq : 0 ≤ (Real.exp e - 1)^2 := sq_nonneg _
    have hmul : (Real.exp e) * (Real.exp e)⁻¹ = 1 := mul_inv_cancel₀ (ne_of_gt hx)
    nlinarith [hsq, hx, hmul]
  have h3 : Real.exp D - Real.exp (D - e) ≤ Real.exp (D + e) - Real.exp D := by
    have e1 : Real.exp (D - e) = Real.exp D / Real.exp e := by rw [Real.exp_sub]
    have e2 : Real.exp (D + e) = Real.exp D * Real.exp e := Real.exp_add ..
    rw [e1, e2, div_eq_mul_inv]
    have h2 : Real.exp D * 2 ≤ Real.exp D * (Real.exp e + (Real.exp e)⁻¹) :=
      mul_le_mul_of_nonneg_left hkey hA.le
    nlinarith [h2]
  rw [abs_le]
  constructor
  · have h2 : Real.exp (D - e) ≤ Real.exp t := Real.exp_le_exp.mpr ht2
    linarith
  · have h2 : Real.exp t ≤ Real.exp (D + e) := Real.exp_le_exp.mpr ht1
    linarith

/-- Auxiliary fact: exp room. -/
private lemma exp_room {D t : ℝ} (hD : D ≤ 64)
    (hDm : |D| ≤ (Format.binary32.maxVal : ℝ))
    (ht : |t - D| ≤ |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)) :
    (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) *
      (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
      * |Real.exp t| + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
  set e : ℝ := |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) with he
  have he0 : 0 ≤ e := by rw [he]; positivity
  have htle : t ≤ D + e := by have := (abs_le.mp ht).2; linarith
  rw [abs_of_pos (Real.exp_pos t)]
  by_cases hcase : |D| ≤ 1024
  · have he1 : e ≤ 1 := by
      rw [he]
      have h1 : |D| * (2:ℝ)^(-24:ℤ) ≤ 1024 * (2:ℝ)^(-24:ℤ) :=
        mul_le_mul_of_nonneg_right hcase (by positivity)
      have h2 : (1024:ℝ) * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) ≤ 1 := by norm_num
      linarith
    have ht65 : t ≤ 65 := by linarith
    have hE1 : Real.exp t ≤ (2:ℝ)^104 :=
      le_trans (Real.exp_le_exp.mpr ht65) exp65_le_pow104
    have hK1 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ 1 := by
      have h1 : |D| + e ≤ (1025:ℝ) := by linarith
      have h2 : (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^(-23:ℤ) * (1025:ℝ) :=
        mul_le_mul_of_nonneg_left h1 (by positivity)
      have h3 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (1025:ℝ) ≤ 1 := by norm_num
      linarith
    have h2 : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) ≤ 2 := by linarith
    have hmul : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp t
        ≤ 2 * (2:ℝ)^104 :=
      mul_le_mul h2 hE1 (Real.exp_pos t).le (by norm_num)
    have hfin : 2 * (2:ℝ)^104 + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
      have h2 : (2:ℝ)^105 + (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^109 := by norm_num
      have h3 : (2:ℝ) * (2:ℝ)^104 = (2:ℝ)^105 := by
        rw [show (105:ℕ) = 1 + 104 by norm_num, pow_add]; norm_num
      linarith [pow109_le_maxVal32]
    linarith
  · have h1024 : (1024:ℝ) < |D| := lt_of_not_ge hcase
    have hDneg : D < 0 := by
      rcases lt_or_ge D 0 with h | h
      · exact h
      · rw [abs_of_nonneg h] at h1024; linarith
    have hDle : D ≤ -1024 := by
      rw [abs_of_neg hDneg] at h1024; linarith
    have hDabs : |D| = -D := abs_of_neg hDneg
    have hee : e = (-D) * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by rw [he, hDabs]
    have hDp : D + e < 0 := by
      rw [hee]
      have h1 : D + (-D) * (2:ℝ)^(-24:ℤ) = D * (1 - (2:ℝ)^(-24:ℤ)) := by ring
      have h2 : D * (1 - (2:ℝ)^(-24:ℤ)) ≤ (-1024) * (1 - (2:ℝ)^(-24:ℤ)) :=
        mul_le_mul_of_nonneg_right hDle (by norm_num)
      have h3 : (-1024:ℝ) * (1 - (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ) < 0 := by norm_num
      linarith
    have ht0 : t < 0 := by linarith
    have hexpt1 : Real.exp t ≤ 1 := by
      have h := Real.exp_le_exp.mpr ht0.le
      simpa using h
    have heM : e ≤ (Format.binary32.maxVal : ℝ) := by
      rw [he]
      have h1 : |D| * (2:ℝ)^(-24:ℤ) ≤ (2:ℝ)^128 * (2:ℝ)^(-24:ℤ) :=
        mul_le_mul_of_nonneg_right (le_trans hDm maxVal32_le_pow128) (by positivity)
      have h2 : (2:ℝ)^128 * (2:ℝ)^(-24:ℤ) = (2:ℝ)^104 := by norm_num
      have h3 : (2:ℝ)^104 + (2:ℝ)^(-150:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
        rw [maxVal32_eq]; norm_num
      linarith
    have hK107 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^107 := by
      have hsum : |D| + e ≤ 2 * (2:ℝ)^128 := by
        have h := maxVal32_le_pow128
        linarith
      have h2 : (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^(-23:ℤ) * (2 * (2:ℝ)^128) :=
        mul_le_mul_of_nonneg_left hsum (by positivity)
      have h3 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (2 * (2:ℝ)^128) ≤ (2:ℝ)^107 := by
        norm_num
      linarith
    have h1K : 1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) ≤ (2:ℝ)^108 := by
      have h : (2:ℝ)^107 + 1 ≤ (2:ℝ)^108 := by norm_num
      linarith
    have hmul : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp t
        ≤ (2:ℝ)^108 * 1 :=
      mul_le_mul h1K hexpt1 (Real.exp_pos t).le (by positivity)
    have hfin : (2:ℝ)^108 + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
      have h2 : (2:ℝ)^108 + (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^109 := by norm_num
      linarith [pow109_le_maxVal32]
    linarith

/-! ### The three rounding steps of the chain, abstractly -/


private lemma exp_combined {r t D e : ℝ} (he0 : 0 ≤ e) (ht : |t - D| ≤ e)
    (hr : |r - Real.exp t| ≤
      ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * |Real.exp t| + (2:ℝ)^(-140:ℤ)) :
    |r - Real.exp D| ≤
      (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp (D + e)
        - Real.exp D + (2:ℝ)^(-140:ℤ) := by
  have h2 := exp_sub_le he0 ht
  have htle : t ≤ D + e := by have h := (abs_le.mp ht).2; linarith
  have hexpt : Real.exp t ≤ Real.exp (D + e) := Real.exp_le_exp.mpr htle
  have hK0 : (0:ℝ) ≤ (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) := by positivity
  have h3 : ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * |Real.exp t|
      ≤ ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * Real.exp (D + e) :=
    mul_le_mul_of_nonneg_left (by rwa [abs_of_pos (Real.exp_pos t)]) hK0
  have htri : |r - Real.exp D| ≤ |r - Real.exp t| + |Real.exp t - Real.exp D| := by
    have h := abs_add_le (r - Real.exp t) (Real.exp t - Real.exp D)
    rw [show (r - Real.exp t) + (Real.exp t - Real.exp D) = r - Real.exp D by ring] at h
    exact h
  have hid : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp (D + e)
      - Real.exp D + (2:ℝ)^(-140:ℤ)
      = ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * Real.exp (D + e)
        + (Real.exp (D + e) - Real.exp D) + (2:ℝ)^(-140:ℤ) := by ring
  linarith [htri, hr, h3, h2]


/-- The three roundings of the weight chain, on reals. -/
theorem NL_wchain_alg {a ee E δ q1 q2 w eH eE : ℝ}
    (h1 : |q1 - a * ee| ≤ |a * ee| * uF + etaF)
    (h2 : |q2 - q1 * δ| ≤ |q1 * δ| * uF + etaF)
    (h3 : |w - q2| ≤ |q2| * uB + etaB)
    (hee : |ee| ≤ eH) (hE : |ee - E| ≤ eE) :
    |w - a * E * δ| ≤ |a| * |δ| * eE + (uB + 3 * uF) * (|a| * eH * |δ|) +
        (2 ^ (-133 : ℤ) + 2 ^ (-148 : ℤ) * |δ|) ∧
      |w| ≤ (1 + uB + 3 * uF) * (|a| * eH * |δ|) + (2 ^ (-133 : ℤ) + 2 ^ (-148 : ℤ) * |δ|) := by
  have hu : uB = 1 / 256 := by norm_num [uB]
  have hf : uF = 1 / 16777216 := by norm_num [uF]
  have hn : etaF = 1 / 2 ^ 150 := by norm_num [etaF]
  have hb : etaB = 1 / 2 ^ 134 := by norm_num [etaB]
  have h133 : (2 : ℝ) ^ (-133 : ℤ) = 1 / 2 ^ 133 := by norm_num
  have h148 : (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 := by norm_num
  simp only [hu, hf, hn, hb] at h1 h2 h3
  simp only [hu, hf, h133, h148]
  have hδ := abs_nonneg δ
  have ha := abs_nonneg a
  have hM : |a * ee| * |δ| ≤ |a| * eH * |δ| := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hee ha) hδ
  have hq1 : |q1| ≤ |a * ee| * (1 + 1 / 16777216) + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub q1 (a * ee)
    linarith
  have hr : |q1 * δ| ≤ (|a * ee| * (1 + 1 / 16777216) + 1 / 2 ^ 150) * |δ| := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_right hq1 hδ
  have hq2 : |q2| ≤ |q1 * δ| * (1 + 1 / 16777216) + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub q2 (q1 * δ)
    linarith
  have h4 : |(q1 - a * ee) * δ| ≤ (|a * ee| * (1 / 16777216) + 1 / 2 ^ 150) * |δ| := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_right h1 hδ
  have h5 : |a * δ * (ee - E)| ≤ |a| * |δ| * eE := by
    rw [abs_mul, abs_mul]
    exact mul_le_mul_of_nonneg_left hE (mul_nonneg ha hδ)
  have hid : w - a * E * δ = (w - q2) + (q2 - q1 * δ) + (q1 - a * ee) * δ + a * δ * (ee - E) := by
    ring
  have htri : |w - a * E * δ| ≤ |w - q2| + |q2 - q1 * δ| + |(q1 - a * ee) * δ| + |a * δ * (ee - E)| := by
    rw [hid]
    calc _ ≤ |(w - q2) + (q2 - q1 * δ) + (q1 - a * ee) * δ| + |a * δ * (ee - E)| := abs_add_le _ _
      _ ≤ |(w - q2) + (q2 - q1 * δ)| + |(q1 - a * ee) * δ| + |a * δ * (ee - E)| := by
          gcongr; exact abs_add_le _ _
      _ ≤ _ := by gcongr; exact abs_add_le _ _
  have hw : |w| ≤ |q2| + |w - q2| := by
    have := abs_add_le q2 (w - q2); rw [show q2 + (w - q2) = w by ring] at this; exact this
  have hprod : (|a * ee| * (1 + 1 / 16777216) + 1 / 2 ^ 150) * |δ| =
      |a * ee| * |δ| * (1 + 1 / 16777216) + 1 / 2 ^ 150 * |δ| := by ring
  have hprod2 : (|a * ee| * (1 / 16777216) + 1 / 2 ^ 150) * |δ| =
      |a * ee| * |δ| * (1 / 16777216) + 1 / 2 ^ 150 * |δ| := by ring
  rw [hprod] at hr
  rw [hprod2] at h4
  simp only [abs_mul] at hr hq1 hq2 h2 htri h4 h5 hM
  have hq2' : |q2| ≤ (|a| * |ee| * |δ|) * ((1 + 1 / 16777216) * (1 + 1 / 16777216)) + (1 / 2 ^ 150 * |δ| * (1 + 1 / 16777216) + 1 / 2 ^ 150) := by
    nlinarith
  have hM0 : 0 ≤ |a| * |ee| * |δ| := by positivity
  have hA : |w - q2| ≤ (|a| * |ee| * |δ|) * ((1/256) * ((1 + 1 / 16777216) * (1 + 1 / 16777216))) + (1 / 256 * (1 / 2 ^ 150 * |δ| * (1 + 1 / 16777216) + 1 / 2 ^ 150) + 1 / 2 ^ 134) := by
    linarith
  have hB : |q2 - q1 * δ| ≤ (|a| * |ee| * |δ|) * (1 / 16777216 * (1 + 1 / 16777216)) + (1 / 16777216 * (1 / 2 ^ 150 * |δ|) + 1 / 2 ^ 150) := by
    linarith
  have hL : |w - q2| + |q2 - q1 * δ| + |q1 - a * ee| * |δ| ≤ (|a| * |ee| * |δ|) * (1/256 + 3 * (1 / 16777216)) + (1 / 2 ^ 133 + 1 / 2 ^ 148 * |δ|) := by
    linarith
  constructor
  · linarith
  · linarith

/-! ### The exponential of the decay exponent -/

private lemma NL_u32 : ((Format.binary32.u : ℚ) : ℝ) = uF := by
  rw [Format.binary32_u]; unfold uF; push_cast; norm_num

/-- Auxiliary fact: NL eta32. -/
private lemma NL_eta32 : ((Format.binary32.eta : ℚ) : ℝ) = etaF := by
  rw [Format.binary32_eta]; unfold etaF; push_cast; norm_num

/-- Auxiliary fact: NL ubf. -/
private lemma NL_ubf : ((Format.bfloat16.u : ℚ) : ℝ) = uB := by
  rw [Format.bfloat16_u]; unfold uB; push_cast; norm_num

/-- Auxiliary fact: NL etabf. -/
private lemma NL_etabf : ((Format.bfloat16.eta : ℚ) : ℝ) = etaB := by
  rw [Format.bfloat16_eta]; unfold etaB; push_cast; norm_num

/-- The error constants are nonnegative reals. -/
private lemma NL_consts_nonneg : 0 ≤ uF ∧ 0 ≤ etaF ∧ 0 ≤ uB ∧ 0 ≤ etaB := by
  unfold uF etaF uB etaB; refine ⟨?_, ?_, ?_, ?_⟩ <;> positivity

/-- `errE` is nonnegative at a nonnegative error radius. -/
theorem NL_errE_nonneg {x δ : ℝ} (hδ : 0 ≤ δ) : 0 ≤ errE x δ := by
  unfold errE expRel
  have h1 : Real.exp x ≤ Real.exp (x + δ) := Real.exp_le_exp.mpr (by linarith)
  have h2 : (0 : ℝ) ≤ 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|x| + δ) := by positivity
  have h3 : Real.exp (x + δ) ≤ (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (|x| + δ))) * Real.exp (x + δ) := by
    nlinarith [Real.exp_pos (x + δ)]
  have h4 : (0 : ℝ) ≤ 2 ^ (-140 : ℤ) := by positivity
  linarith

/-- The computed decay exponent's error radius `dErrS` is nonnegative. -/
theorem NL_dErrS_nonneg (m₀ : Mem) (d : Inst) (t h s : Nat) : 0 ≤ dErrS m₀ d t h s := by
  unfold dErrS uF etaF; positivity

/-- `eHat` is positive at a nonnegative error radius. -/
theorem NL_eHat_nonneg {x δ : ℝ} (hδ : 0 ≤ δ) : 0 ≤ eHat x δ := by
  unfold eHat
  have := NL_errE_nonneg (x := x) hδ
  have := Real.exp_pos x
  linarith

/-- **`tl.exp` of the subtraction** `a − b` of two finite non-positive `float32` values whose
difference is at most `64`: the exponential is finite, within `errE` of `exp(a − b)` and at most
`eHat`. -/
private theorem NL_exp_step {P : Platform} (hexp : Device.ExpF32 P) {a b xt ex : F32} {D : ℝ}
    (hDeq : D = (a.toRat : ℝ) - b.toRat) (ha : a.isFinite = true) (hb : b.isFinite = true) (ha0 : a.toRat ≤ 0) (hb0 : b.toRat ≤ 0)
    (hD : D ≤ 64)
    (hsub : FloatV.NaNEq (FloatV.sub a b) xt) (hex : P.math .exp .binary32 xt ex) :
    ex.isFinite = true ∧ |(ex.toRat : ℝ) - Real.exp D| ≤ errE D (uF * |D| + etaF) ∧
      |(ex.toRat : ℝ)| ≤ eHat D (uF * |D| + etaF) := by
  subst hDeq
  set D : ℝ := (a.toRat : ℝ) - b.toRat with hDdef
  have hamax := FloatV.abs_toRat_le_maxVal Format.binary32_regular ha
  have hbmax := FloatV.abs_toRat_le_maxVal Format.binary32_regular hb
  have ha' : (-(a.toRat : ℝ)) ≤ (Format.binary32.maxVal : ℝ) := by
    have h := abs_of_nonpos ha0
    have : |(a.toRat : ℝ)| ≤ (Format.binary32.maxVal : ℝ) := by exact_mod_cast hamax
    rw [← Rat.cast_abs, h] at this ; push_cast at this; linarith
  have hb' : (-(b.toRat : ℝ)) ≤ (Format.binary32.maxVal : ℝ) := by
    have h := abs_of_nonpos hb0
    have : |(b.toRat : ℝ)| ≤ (Format.binary32.maxVal : ℝ) := by exact_mod_cast hbmax
    rw [← Rat.cast_abs, h] at this ; push_cast at this; linarith
  have ha0r : (a.toRat : ℝ) ≤ 0 := by exact_mod_cast ha0
  have hb0r : (b.toRat : ℝ) ≤ 0 := by exact_mod_cast hb0
  have hDm : |D| ≤ (Format.binary32.maxVal : ℝ) := by
    rw [abs_le]; constructor <;> linarith
  have hsuberr := FloatV.sub_err_real Format.binary32_regular ha hb
    (by simpa only [FloatV.toReal] using hDm)
  have hsubnan : (FloatV.sub a b).isNaN = false := (FloatV.isFinite_iff.mp hsuberr.1).1
  have hxteq : xt = FloatV.sub a b := FloatV.NaNEq.eq hsub hsubnan
  have hxtfin : xt.isFinite = true := by rw [hxteq]; exact hsuberr.1
  have hdErr0 : 0 ≤ uF * |D| + etaF := by unfold uF etaF; positivity
  have hxtD : |(xt.toRat : ℝ) - D| ≤ |D| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have h := hsuberr.2
    rw [NL_u32, NL_eta32] at h
    simp only [FloatV.toReal] at h
    rw [hxteq, hDdef]
    unfold uF etaF at h
    simpa using h
  obtain ⟨ref, href, hprop⟩ := hexp (|D| + (uF * |D| + etaF)) (by positivity)
  have href' : ref = Real.exp := by
    have h1 : MathFn.ref .exp = some Real.exp := rfl
    rw [h1] at href
    cases href
    rfl
  rw [href'] at hprop
  have hmem : xt.toReal ∈ Set.Icc (-(|D| + (uF * |D| + etaF))) (|D| + (uF * |D| + etaF)) := by
    have h2 := abs_le.mp hxtD
    have h3 : -|D| ≤ D := neg_abs_le _
    have h4 : D ≤ |D| := le_abs_self _
    have e1 : (2 : ℝ) ^ (-24 : ℤ) = uF := rfl
    have e2 : (2 : ℝ) ^ (-150 : ℤ) = etaF := rfl
    rw [e1, e2] at h2
    simp only [FloatV.toReal, Set.mem_Icc]
    constructor <;> nlinarith [h2.1, h2.2]
  have hroom := exp_room (t := xt.toReal) hD hDm (by simpa [FloatV.toReal] using hxtD)
  have e : uF * |D| + etaF = |D| * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    unfold uF etaF; ring
  have hres := hprop xt ex hex hxtfin hmem (by simp [MathFn.natDom]) (by rw [e]; exact hroom)
  have hexfin := hres.1
  have herr : |(ex.toRat : ℝ) - Real.exp D| ≤ errE D (uF * |D| + etaF) := by
    have h := exp_combined (D := D) (e := uF * |D| + etaF) hdErr0
      (t := xt.toReal) (r := ex.toReal) (by rw [e]; simpa [FloatV.toReal] using hxtD)
      (by rw [e] at *; simpa [FloatV.toReal] using hres.2)
    rw [e] at h ⊢
    simpa only [errE, expRel, FloatV.toReal] using h
  refine ⟨hexfin, herr, ?_⟩
  have h1 := abs_add_le (Real.exp D) ((ex.toRat : ℝ) - Real.exp D)
  rw [show Real.exp D + ((ex.toRat : ℝ) - Real.exp D) = ex.toRat by ring,
    abs_of_pos (Real.exp_pos D)] at h1
  unfold eHat
  linarith


/-! ### The weight chain -/

/-- A `float32` product with a zero factor is a finite zero. -/
private theorem NL_mul_zero {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (h0 : x.toRat = 0) : (FloatV.mul x y).isFinite = true ∧ (FloatV.mul x y).toRat = 0 := by
  unfold FloatV.mul
  rw [Format.mul_of_finite .rn hx hy]
  have h : Format.binary32.toRat x.pat = 0 := h0
  rw [h, zero_mul]
  obtain ⟨hlt, hfin, hval⟩ := Format.roundRat_zero_spec Format.binary32_regular (rnd := .rn)
    (sat := false) (zs := (Format.binary32.sign x.pat != Format.binary32.sign y.pat))
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hfin, hval⟩

/-- The `bfloat16` rounding of a `float32` zero is a finite zero. -/
private theorem NL_cvt_zero {x : F32} (hx : x.isFinite = true) (h0 : x.toRat = 0) :
    (FloatV.cvtWith .rn x : BF16).isFinite = true ∧ (FloatV.cvtWith .rn x : BF16).toRat = 0 := by
  unfold FloatV.cvtWith
  rw [Format.cvt_of_finite Format.bfloat16 .rn false hx]
  have h : Format.binary32.toRat x.pat = 0 := h0
  rw [h]
  obtain ⟨hlt, hfin, hval⟩ := Format.roundRat_zero_spec Format.bfloat16_regular (rnd := .rn)
    (sat := false) (zs := Format.binary32.sign x.pat)
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hfin, hval⟩

/-- Auxiliary fact: pow121 le maxVal32. -/
private lemma pow121_le_maxVal32 : (2 : ℝ) ^ 121 ≤ (Format.binary32.maxVal : ℝ) :=
  le_trans (by norm_num) pow126_le_maxVal32

/-- Auxiliary fact: pow120 le maxVal32. -/
private lemma pow120_le_maxVal32 : (2 : ℝ) ^ 120 ≤ (Format.binary32.maxVal : ℝ) :=
  le_trans (by norm_num) pow126_le_maxVal32

/-- Auxiliary fact: pow122 le maxValbf. -/
private lemma pow122_le_maxValbf : (2 : ℝ) ^ 122 ≤ (Format.bfloat16.maxVal : ℝ) :=
  le_trans (by norm_num) three_pow126_le_maxValbf

/-- The room of the three roundings: a bound on the products. -/
private theorem NL_room {cb ex q1 dt eH : ℝ} (hee : |ex| ≤ eH) (h1 : |cb| * eH ≤ 2 ^ 120)
    (h2 : |cb| * eH * |dt| ≤ 2 ^ 120) (hdt : |dt| ≤ 2 ^ 128)
    (hq1 : |q1 - cb * ex| ≤ |cb * ex| * uF + etaF) :
    |cb * ex| ≤ 2 ^ 120 ∧ |q1 * dt| ≤ 2 ^ 121 := by
  have hu : uF = 1 / 16777216 := by norm_num [uF]
  have hn : etaF = 1 / 2 ^ 150 := by norm_num [etaF]
  rw [hu, hn] at hq1
  have hcb := abs_nonneg cb
  have hdt0 := abs_nonneg dt
  have hp : |cb * ex| ≤ |cb| * eH := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hee hcb
  have hp1 : |cb * ex| ≤ 2 ^ 120 := hp.trans h1
  refine ⟨hp1, ?_⟩
  have hq : |q1| ≤ |cb * ex| * (1 + 1 / 16777216) + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub q1 (cb * ex)
    linarith
  have hr : |q1 * dt| ≤ (|cb * ex| * (1 + 1 / 16777216) + 1 / 2 ^ 150) * |dt| := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_right hq hdt0
  have hpd : |cb * ex| * |dt| ≤ 2 ^ 120 := by
    calc |cb * ex| * |dt| ≤ |cb| * eH * |dt| := mul_le_mul_of_nonneg_right hp hdt0
      _ ≤ 2 ^ 120 := h2
  have hz : (1 / 2 ^ 150 : ℝ) * |dt| ≤ 1 / 2 ^ 150 * 2 ^ 128 :=
    mul_le_mul_of_nonneg_left hdt (by positivity)
  have hexp : (|cb * ex| * (1 + 1 / 16777216) + 1 / 2 ^ 150) * |dt| =
      |cb * ex| * |dt| * (1 + 1 / 16777216) + 1 / 2 ^ 150 * |dt| := by ring
  rw [hexp] at hr
  have : (2 : ℝ) ^ 120 * (1 + 1 / 16777216) + 1 / 2 ^ 150 * 2 ^ 128 ≤ 2 ^ 121 := by norm_num
  nlinarith

/-- The three roundings `cb · ex`, `· dt`, `.to(bfloat16)` of a weight: all finite, and each within
its `float32`/`bfloat16` rounding error of the exact value. -/
private theorem NL_chain {cb ex dt q1 q2 : F32} {w16 : BF16} (hcb : cb.isFinite = true)
    (hex : ex.isFinite = true) (hdt : dt.isFinite = true) {eH : ℝ} (hee : |(ex.toRat : ℝ)| ≤ eH)
    (h1 : |(cb.toRat : ℝ)| * eH ≤ 2 ^ 120) (h2 : |(cb.toRat : ℝ)| * eH * |(dt.toRat : ℝ)| ≤ 2 ^ 120)
    (hq1 : FloatV.NaNEq (FloatV.mul cb ex) q1) (hq2 : FloatV.NaNEq (FloatV.mul q1 dt) q2)
    (hw : FloatV.NaNEq (FloatV.cvtWith .rn q2) w16) :
    q1.isFinite = true ∧ q2.isFinite = true ∧ w16.isFinite = true ∧
      |(q1.toRat : ℝ) - cb.toRat * ex.toRat| ≤ |(cb.toRat : ℝ) * ex.toRat| * uF + etaF ∧
      |(q2.toRat : ℝ) - q1.toRat * dt.toRat| ≤ |(q1.toRat : ℝ) * dt.toRat| * uF + etaF ∧
      |(w16.toRat : ℝ) - q2.toRat| ≤ |(q2.toRat : ℝ)| * uB + etaB := by
  have hdtm : |(dt.toRat : ℝ)| ≤ 2 ^ 128 := by
    have := FloatV.abs_toRat_le_maxVal Format.binary32_regular hdt
    have h : |(dt.toRat : ℝ)| ≤ (Format.binary32.maxVal : ℝ) := by exact_mod_cast this
    exact h.trans maxVal32_le_pow128
  -- the first product
  have hm1 : |(cb.toRat : ℝ) * ex.toRat| ≤ (2 : ℝ) ^ 120 := by
    have hp : |(cb.toRat : ℝ) * ex.toRat| ≤ |(cb.toRat : ℝ)| * eH := by
      rw [abs_mul]; exact mul_le_mul_of_nonneg_left hee (abs_nonneg _)
    exact hp.trans h1
  have e1 := FloatV.mul_err_real Format.binary32_regular hcb hex
    (by simp only [FloatV.toReal]; exact hm1.trans (pow120_le_maxVal32))
  have hq1eq : q1 = FloatV.mul cb ex :=
    FloatV.NaNEq.eq hq1 (FloatV.isFinite_iff.mp e1.1).1
  have hq1fin : q1.isFinite = true := by rw [hq1eq]; exact e1.1
  have hq1err : |(q1.toRat : ℝ) - cb.toRat * ex.toRat| ≤ |(cb.toRat : ℝ) * ex.toRat| * uF + etaF := by
    have h := e1.2
    rw [NL_u32, NL_eta32] at h
    simpa only [FloatV.toReal, hq1eq] using h
  obtain ⟨-, hroom2⟩ := NL_room hee h1 h2 hdtm hq1err
  -- the second product
  have e2 := FloatV.mul_err_real Format.binary32_regular hq1fin hdt
    (by simp only [FloatV.toReal]; exact hroom2.trans (pow121_le_maxVal32))
  have hq2eq : q2 = FloatV.mul q1 dt :=
    FloatV.NaNEq.eq hq2 (FloatV.isFinite_iff.mp e2.1).1
  have hq2fin : q2.isFinite = true := by rw [hq2eq]; exact e2.1
  have hq2err : |(q2.toRat : ℝ) - q1.toRat * dt.toRat| ≤ |(q1.toRat : ℝ) * dt.toRat| * uF + etaF := by
    have h := e2.2
    rw [NL_u32, NL_eta32] at h
    simpa only [FloatV.toReal, hq2eq] using h
  -- the conversion to bfloat16
  have hq2m : |(q2.toRat : ℝ)| ≤ (Format.bfloat16.maxVal : ℝ) := by
    have hu : uF = 1 / 16777216 := by norm_num [uF]
    have hn : etaF = 1 / 2 ^ 150 := by norm_num [etaF]
    rw [hu, hn] at hq2err
    have := abs_sub_abs_le_abs_sub (q2.toRat : ℝ) (q1.toRat * dt.toRat)
    rw [abs_mul] at hroom2 hq2err this
    have hh : |(q2.toRat : ℝ)| ≤ 2 ^ 122 := by
      have : (2 : ℝ) ^ 121 * (1 + 1 / 16777216) + 1 / 2 ^ 150 ≤ 2 ^ 122 := by norm_num
      nlinarith [abs_nonneg ((q1.toRat : ℝ) * dt.toRat)]
    exact hh.trans pow122_le_maxValbf
  have e3 := FloatV.cvt_err_real (f := .binary32) (g := .bfloat16) Format.bfloat16_regular hq2fin
    (by simpa only [FloatV.toReal] using hq2m)
  have hw16eq : w16 = (FloatV.cvt q2 : BF16) :=
    FloatV.NaNEq.eq hw (FloatV.isFinite_iff.mp e3.1).1
  refine ⟨hq1fin, hq2fin, by rw [hw16eq]; exact e3.1, hq1err, hq2err, ?_⟩
  have h := e3.2
  rw [NL_ubf, NL_etabf] at h
  simpa only [FloatV.toReal, hw16eq] using h

/-! ## One weight (lines 398–414) -/

/-- What the loop's weight lane of token `s` (row `t`'s causal sum) satisfies: finite; for a token
of the logical chunk within `ewS` of `wTS` and of magnitude at most `wHat`; for any other token
(after `t`, or before the logical chunk and so against a zero `cb`) exactly `0`. -/
def TokW (m₀ : Mem) (d : Inst) (t h s : Nat) (w : BF16) : Prop :=
  w.isFinite = true ∧
    (s ∈ toks d t → |(w.toRat : ℝ) - wTS m₀ d t h s| ≤ ewS m₀ d t h s ∧
      |(w.toRat : ℝ)| ≤ wHat m₀ d t h s) ∧
    (s ∉ toks d t → w.toRat = 0)

/-- Token `t` lies in a chunk below `nc`. -/
private theorem NL_chk_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) : chk d t < nc d := by
  have hcs := cs_pos hd
  unfold chk nc
  rw [Nat.div_lt_iff_lt_mul hcs]
  have h1 := Nat.div_add_mod (d.T + d.cs - 1) d.cs
  have h2 := Nat.mod_lt (d.T + d.cs - 1) hcs
  rw [Nat.mul_comm] at h1
  generalize (d.T + d.cs - 1) / d.cs * d.cs = q at *
  omega

/-- A token `s ≤ t` sits at an offset below `cs` in `t`'s chunk. -/
private theorem NL_idx_lt {d : Inst} (hd : d.ok) {t s : Nat} (hst : s ≤ t) :
    s - chk d t * d.cs < d.cs := by
  have hcs := cs_pos hd
  have h : t % d.cs + t / d.cs * d.cs = t := by rw [Nat.mul_comm]; exact Nat.mod_add_div t d.cs
  have h2 := Nat.mod_lt t hcs
  unfold chk
  generalize t / d.cs * d.cs = q at *
  omega

/-- **The weight chain.** Token `s` is in the same physical chunk as `t`, `chk*cs ≤ s`, and
`s ≤ t` or `s` is past the chunk; the lane's `cb` (`cbL`), `dA_cs_m` (`dAm`), `dA_cs_k` (`dAk`)
and `dt_k` (`dtk`) are loaded or `+0` (masked), and the chain
`cb·exp(dAm − dAk)·dt`, `where(s ≤ t, ·, 0)`, `.to(bfloat16)` gives `w`. -/
theorem weight_chain {P : Platform} (hexp : Device.ExpF32 P) {m₀ : Mem} {d : Inst} (hd : d.ok)
    (hV : Vals m₀ d) (hcb0 : CbZero d m₀) {t h p s : Nat} (ht : t < d.T) (hh : h < d.H)
    (hp : p < d.P) (hs1 : chk d t * d.cs ≤ s)
    {cbL dAm dAk dtk xt ex q1 q2 q3 : F32} {w16 : BF16}
    (hkt : s ≤ t → s ∈ kToks d t)
    (hcbL : s ≤ t →
      m₀.read .f32 (d.cb.ptr (cbIdx d (chk d t) (grp d h) (t % d.cs) (s - chk d t * d.cs) : Int)).addr = some cbL)
    (hdAm : m₀.read .f32 (d.dA.ptr (dtIdx d h (chk d t) (t % d.cs) : Int)).addr = some dAm)
    (hdAk : s ≤ t →
      m₀.read .f32 (d.dA.ptr (dtIdx d h (chk d t) (s - chk d t * d.cs) : Int)).addr = some dAk)
    (hdtk : s ≤ t →
      m₀.read .f32 (d.dt.ptr (dtIdx d h (chk d t) (s - chk d t * d.cs) : Int)).addr = some dtk)
    (hsub : FloatV.NaNEq (FloatV.sub dAm dAk) xt) (hex : P.math .exp .binary32 xt ex)
    (hq1 : FloatV.NaNEq (FloatV.mul cbL ex) q1) (hq2 : FloatV.NaNEq (FloatV.mul q1 dtk) q2)
    (hq3 : q3 = if s ≤ t then q2 else zf32) (hw : FloatV.NaNEq (FloatV.cvtWith .rn q3) w16) :
    TokW m₀ d t h s w16 := by
  by_cases hst : s ≤ t
  · have hk := hkt hst
    have hidx := NL_idx_lt hd (t := t) (s := s) hst
    have hcs := cs_pos hd
    have hc := NL_chk_lt hd ht
    have hmod : t % d.cs < d.cs := Nat.mod_lt _ hcs
    obtain ⟨-, -, -, -, -, hdtV, hdAV, hcbV, hlast⟩ := hV
    obtain ⟨vm, hvm, hvmf, hvm0⟩ := hdAV h (chk d t) (t % d.cs) hh hc hmod
    obtain ⟨vk, hvk, hvkf, hvk0⟩ := hdAV h (chk d t) (s - chk d t * d.cs) hh hc hidx
    obtain ⟨vd, hvd, hvdf⟩ := hdtV h (chk d t) (s - chk d t * d.cs) hh hc hidx
    have hAm : dAm = vm := Option.some.inj (hdAm.symm.trans hvm)
    have hAk : dAk = vk := Option.some.inj ((hdAk hst).symm.trans hvk)
    have hDt : dtk = vd := Option.some.inj ((hdtk hst).symm.trans hvd)
    subst hAm hAk hDt
    have hvA : ∀ i (v : F32), m₀.read .f32 (d.dA.ptr (dtIdx d h (chk d t) i : Int)).addr = some v →
        dAv m₀ d h (chk d t) i = (v.toRat : ℝ) := by
      intro i v hv; rw [dAv, valF, hv]
    have hDeq : dS m₀ d t h s = (dAm.toRat : ℝ) - dAk.toRat := by
      rw [dS, hvA _ _ hdAm, hvA _ _ (hdAk hst)]
    have hdtT : dtS m₀ d t h s = (dtk.toRat : ℝ) := by rw [dtS, dtv, valF, hdtk hst]
    have hq3' : q3 = q2 := by rw [hq3]; simp [hst]
    rw [hq3'] at hw
    obtain ⟨hD64, hcbe, hwm⟩ := (hlast t h p ht hh hp).2.1 s hk
    have hE := NL_exp_step hexp hDeq hvmf hvkf hvm0 hvk0 (hDeq ▸ hD64) hsub hex
    have hcbT : ∀ v : F32, m₀.read .f32 (d.cb.ptr (cbIdx d (chk d t) (grp d h) (t % d.cs) (s - chk d t * d.cs) : Int)).addr = some v →
        cbTS m₀ d t h s = (v.toRat : ℝ) := by
      intro v hv; rw [cbTS, cbv, valF, hv]
    by_cases hs : s ∈ toks d t
    · obtain ⟨vc, hvc, hvcf⟩ := hcbV t (grp d h) s ht (grp_lt hd hh) hs
      have hcbc : cbL = vc := Option.some.inj ((hcbL hst).symm.trans hvc)
      subst hcbc
      have hcT := hcbT cbL (hcbL hst)
      obtain ⟨hf1, hf2, hf3, he1, he2, he3⟩ := NL_chain hvcf hE.1 hvdf hE.2.2
        (by rw [← hcT]; exact hcbe) (by rw [← hcT, ← hdtT]; exact hwm) hq1 hq2 hw
      obtain ⟨hA1, hA2⟩ := NL_wchain_alg he1 he2 he3 hE.2.2 hE.2.1
      refine ⟨hf3, fun _ => ?_, fun h' => absurd hs h'⟩
      have hw0 : wTS m₀ d t h s = (cbL.toRat : ℝ) * Real.exp (dS m₀ d t h s) * (dtk.toRat : ℝ) := by
        rw [wTS, hcT, hdtT]
      constructor
      · rw [hw0]
        unfold ewS wMag wAbs
        rw [hcT, hdtT]
        exact hA1
      · unfold wHat wMag wAbs
        rw [hcT, hdtT]
        exact hA2
    · -- a token before the logical chunk (no initial states): `cb` is a zero there
      have hinit : d.hasInit = false := by
        by_contra hne
        have hT : d.hasInit = true := by simpa using hne
        exact hs (by simpa [kToks, hT] using hk)
      have hlt : s < lTok d t := by
        by_contra hge
        exact hs (by unfold toks; rw [Finset.mem_Ico]; omega)
      obtain ⟨vc, hvc, hvcf, hvc0⟩ := hcb0 hinit t s (grp d h) ht hs1 hst
        (reqOf_ne_of_lt_lTok hd ht hs1 hlt) (grp_lt hd hh)
      have hcbc : cbL = vc := Option.some.inj ((hcbL hst).symm.trans hvc)
      subst hcbc
      obtain ⟨m1f, m10⟩ := NL_mul_zero hvcf hE.1 hvc0
      have hq1e : q1 = FloatV.mul cbL ex := FloatV.NaNEq.eq hq1 (FloatV.isFinite_iff.mp m1f).1
      rw [← hq1e] at m1f m10
      obtain ⟨m2f, m20⟩ := NL_mul_zero m1f hvdf m10
      have hq2e : q2 = FloatV.mul q1 dtk := FloatV.NaNEq.eq hq2 (FloatV.isFinite_iff.mp m2f).1
      rw [← hq2e] at m2f m20
      obtain ⟨cf, c0⟩ := NL_cvt_zero m2f m20
      have hwe : w16 = FloatV.cvtWith .rn q2 := FloatV.NaNEq.eq hw (FloatV.isFinite_iff.mp cf).1
      rw [← hwe] at cf c0
      exact ⟨cf, fun h' => absurd h' hs, fun _ => c0⟩
  · -- past the token: `where` gives `+0`
    have hq3' : q3 = zf32 := by rw [hq3]; simp [hst]
    have hz : (FloatV.cvtWith .rn q3 : BF16) = FloatV.ofPat 0 := by
      rw [hq3', zf32_eq]; exact cvt_f32_bf16_zero
    have hw16 : w16 = FloatV.ofPat 0 :=
      (FloatV.NaNEq.eq hw (by rw [hz]; exact BF16_zero_notNaN)).trans hz
    have hns : s ∉ toks d t := by
      unfold toks; rw [Finset.mem_Ico]; omega
    refine ⟨by rw [hw16]; exact BF16_zero_finite, fun h' => absurd h' hns, fun _ => ?_⟩
    rw [hw16]; exact BF16_zero_toRat

/-! ## The loop (lines 397–420) -/

/-- The tokens of `toks t` whose rows are below `o + 32·n`: those the first `n` trips of the loop
(rows `o + 32·j + l`) have seen, in the physical chunk `c`. -/
def sPart (d : Inst) (c o n t : Nat) : Finset Nat :=
  (toks d t).filter (fun s => s < c * d.cs + o + BK * n)

/-- One `acc += tl.dot` trip's relative and absolute error constants. -/
noncomputable def rhoS : ℚ := DotLoop.stepRel Device.relDot16 BK
/-- Auxiliary fact: alpS. -/
noncomputable def alpS : ℚ := DotLoop.stepAbs Device.absDot16 BK

/-- The lifted magnitude of the first `n` trips' weights times `x`. -/
noncomputable def lWp (m₀ : Mem) (d : Inst) (c o t h p n : Nat) : ℝ :=
  ∑ s ∈ sPart d c o n t, (wHat m₀ d t h s + mB) * (|xv m₀ d s h p| + mB)

/-- The magnitude of the first `n` trips' weights times `x`. -/
noncomputable def mWp (m₀ : Mem) (d : Inst) (c o t h p n : Nat) : ℝ :=
  ∑ s ∈ sPart d c o n t, wHat m₀ d t h s * |xv m₀ d s h p|

/-- The weights' error in the first `n` trips. -/
noncomputable def errWp (m₀ : Mem) (d : Inst) (c o t h p n : Nat) : ℝ :=
  ∑ s ∈ sPart d c o n t, ewS m₀ d t h s * |xv m₀ d s h p|

/-- The exact output's state term plus the first `n` trips' causal sum. -/
noncomputable def refP (m₀ : Mem) (d : Inst) (c o t h p n : Nat) : ℝ :=
  Real.exp (dSt m₀ d t h) * sig m₀ d t h p + ∑ s ∈ sPart d c o n t, wTS m₀ d t h s * xv m₀ d s h p

/-- `DotLoop.bound` over the reals. -/
noncomputable def bndR (n : Nat) (B : ℝ) : ℝ :=
  ((1 + (rhoS : ℝ)) ^ n - 1) * B + n * (1 + (rhoS : ℝ)) ^ n * (alpS : ℝ)

/-- **The loop invariant** of one accumulator lane (token `t`, head `h`, channel `p`) after `n`
trips: finite; within the dot loop's error `bndR n (aSig + lWp)` of a real `T` (the computed
`acc₀` plus the computed weights times `x`), which is within `errSig` and the weights' error of the
exact partial sum `refP` and of magnitude at most `aSig + mWp`. -/
def LoopInv (m₀ : Mem) (d : Inst) (c o t h p n : Nat) (acc : F32) : Prop :=
  acc.isFinite = true ∧ ∃ T : ℝ,
    |(acc.toRat : ℝ) - T| ≤ bndR n (aSig m₀ d t h p + lWp m₀ d c o t h p n) ∧
    |T - refP m₀ d c o t h p n| ≤ errSig m₀ d t h p + errWp m₀ d c o t h p n ∧
    |T| ≤ aSig m₀ d t h p + mWp m₀ d c o t h p n

/-- Before the first trip nothing is seen. -/
private theorem NL_sPart_zero {d : Inst} {c o t : Nat} (hl : c * d.cs + o ≤ lTok d t) :
    sPart d c o 0 t = ∅ := by
  unfold sPart
  rw [Finset.filter_eq_empty_iff]
  intro s hs
  have := (Finset.mem_Ico.mp hs).1
  omega

/-- **Before the loop**: the state term gives the invariant at `n = 0`. -/
theorem loop_init {m₀ : Mem} {d : Inst} {c o t h p : Nat} (hl : c * d.cs + o ≤ lTok d t)
    {acc : F32} (hfin : acc.isFinite = true)
    (h1 : |(acc.toRat : ℝ) - Real.exp (dSt m₀ d t h) * sig m₀ d t h p| ≤ errSig m₀ d t h p)
    (h2 : |(acc.toRat : ℝ)| ≤ aSig m₀ d t h p) : LoopInv m₀ d c o t h p 0 acc := by
  refine ⟨hfin, acc.toRat, ?_, ?_, ?_⟩
  · simp [bndR]
  · simpa [refP, errWp, NL_sPart_zero hl] using h1
  · simpa [mWp, NL_sPart_zero hl] using h2

/-! ### The loop's recursion over the reals -/

private theorem NL_rho_nonneg : 0 ≤ rhoS := by
  unfold rhoS DotLoop.stepRel Device.relDot16
  norm_num [Format.binary32_u, BK]

/-- Auxiliary fact: NL alp nonneg. -/
private theorem NL_alp_nonneg : 0 ≤ alpS := by
  unfold alpS DotLoop.stepAbs Device.absDot16
  norm_num [Format.binary32_u, Format.binary32_eta, BK]

/-- The trip's relative constant is below `2^-16`. -/
private theorem NL_rho_le : rhoS ≤ 2 ^ (-16 : ℤ) := by
  unfold rhoS DotLoop.stepRel Device.relDot16
  norm_num [Format.binary32_u, BK]

/-- The trip's relative constant is within `2^-21` of the dot's. -/
private theorem NL_rho_le_rel : (rhoS : ℝ) ≤ relD BK + 2 ^ (-21 : ℤ) := by
  have h : rhoS ≤ Device.relDot16 BK + 2 ^ (-21 : ℤ) := by
    unfold rhoS DotLoop.stepRel Device.relDot16
    norm_num [Format.binary32_u, BK]
  have h' := (Rat.cast_le (K := ℝ)).mpr h
  push_cast at h'
  exact h'

/-- The trip's absolute constant is within `2^-140` of the dot's. -/
private theorem NL_alp_le_abs : (alpS : ℝ) ≤ absD BK + 2 ^ (-140 : ℤ) := by
  have h : alpS ≤ Device.absDot16 BK + 2 ^ (-140 : ℤ) := by
    unfold alpS DotLoop.stepAbs Device.absDot16
    norm_num [Format.binary32_u, Format.binary32_eta, BK]
  have h' := (Rat.cast_le (K := ℝ)).mpr h
  push_cast at h'
  exact h'

/-- The trip's absolute constant is tiny. -/
private theorem NL_alp_le : alpS ≤ 2 ^ (-119 : ℤ) := by
  unfold alpS DotLoop.stepAbs Device.absDot16
  norm_num [Format.binary32_u, Format.binary32_eta, BK]

/-- Auxiliary fact: NL bndR nonneg. -/
private theorem NL_bndR_nonneg (n : Nat) {B : ℝ} (hB : 0 ≤ B) : 0 ≤ bndR n B := by
  have hr : (0 : ℝ) ≤ rhoS := by exact_mod_cast NL_rho_nonneg
  have ha : (0 : ℝ) ≤ alpS := by exact_mod_cast NL_alp_nonneg
  unfold bndR
  have : 1 ≤ (1 + (rhoS : ℝ)) ^ n := one_le_pow₀ (by linarith)
  have : 0 ≤ ((1 + (rhoS : ℝ)) ^ n - 1) * B := mul_nonneg (by linarith) hB
  positivity

/-- Auxiliary fact: NL bndR mono. -/
private theorem NL_bndR_mono (n : Nat) {B B' : ℝ} (h : B ≤ B') : bndR n B ≤ bndR n B' := by
  have hr : (0 : ℝ) ≤ rhoS := by exact_mod_cast NL_rho_nonneg
  unfold bndR
  have : 1 ≤ (1 + (rhoS : ℝ)) ^ n := one_le_pow₀ (by linarith)
  have := mul_le_mul_of_nonneg_left h (by linarith : 0 ≤ (1 + (rhoS : ℝ)) ^ n - 1)
  linarith

/-- **The closed form** of the loop's error: `bndR n B ≤ 2n(ρB + α)` for at most `512` trips. -/
private theorem NL_bndR_le {n : Nat} (hn : n ≤ 512) {B : ℝ} (hB : 0 ≤ B) :
    bndR n B ≤ 2 * n * ((rhoS : ℝ) * B + alpS) := by
  have hr : (0 : ℝ) ≤ rhoS := by exact_mod_cast NL_rho_nonneg
  have ha : (0 : ℝ) ≤ alpS := by exact_mod_cast NL_alp_nonneg
  have hnr : (n : ℚ) * rhoS ≤ 1 / 2 := by
    have h1 : (n : ℚ) ≤ 512 := by exact_mod_cast hn
    have h2 := NL_rho_le
    calc (n : ℚ) * rhoS ≤ 512 * 2 ^ (-16 : ℤ) :=
          mul_le_mul h1 h2 NL_rho_nonneg (by norm_num)
      _ ≤ 1 / 2 := by norm_num
  have hp : ((1 + (rhoS : ℝ)) ^ n) ≤ 1 + 2 * n * rhoS := by
    have := DotLoop.pow_le NL_rho_nonneg n hnr
    exact_mod_cast this
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hnrR : (n : ℝ) * rhoS ≤ 1 / 2 := by
    have h' := (Rat.cast_le (K := ℝ)).mpr hnr
    push_cast at h'
    exact h'
  unfold bndR
  have e1 : ((1 + (rhoS : ℝ)) ^ n - 1) * B ≤ 2 * n * rhoS * B :=
    mul_le_mul_of_nonneg_right (by linarith) hB
  have e2 : (n : ℝ) * (1 + (rhoS : ℝ)) ^ n * alpS ≤ n * 2 * alpS := by
    have : (1 + (rhoS : ℝ)) ^ n ≤ 2 := by nlinarith
    exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left this hn0) ha
  nlinarith

/-- The error is small compared to the magnitude: `bndR n B ≤ B/64 + 1` for at most `512` trips. -/
private theorem NL_bndR_small {n : Nat} (hn : n ≤ 512) {B : ℝ} (hB : 0 ≤ B) :
    bndR n B ≤ B / 64 + 1 := by
  have h := NL_bndR_le hn hB
  have hr : (rhoS : ℝ) ≤ 2 ^ (-16 : ℤ) := by
    have h' := (Rat.cast_le (K := ℝ)).mpr NL_rho_le
    push_cast at h'
    exact h'
  have ha : (alpS : ℝ) ≤ 2 ^ (-119 : ℤ) := by
    have h' := (Rat.cast_le (K := ℝ)).mpr NL_alp_le
    push_cast at h'
    exact h'
  have hr0 : (0 : ℝ) ≤ rhoS := by exact_mod_cast NL_rho_nonneg
  have hn1 : (n : ℝ) ≤ 512 := by exact_mod_cast hn
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have e16 : (2 : ℝ) ^ (-16 : ℤ) = 1 / 65536 := by norm_num
  have e119 : (2 : ℝ) ^ (-119 : ℤ) ≤ 1 / 2048 := by norm_num
  rw [e16] at hr
  have h1 : 2 * (n : ℝ) * (rhoS * B) ≤ 2 * 512 * (1 / 65536 * B) := by gcongr
  have h2 : 2 * (n : ℝ) * alpS ≤ 2 * 512 * (1 / 2048) := by
    have : (0 : ℝ) ≤ alpS := by exact_mod_cast NL_alp_nonneg
    calc 2 * (n : ℝ) * alpS ≤ 2 * 512 * alpS := by gcongr
      _ ≤ 2 * 512 * (1 / 2048) := by gcongr; exact ha.trans e119
  have h5 : 2 * (n : ℝ) * ((rhoS : ℝ) * B + alpS) = 2 * n * (rhoS * B) + 2 * n * alpS := by ring
  linarith

/-- **One trip of the recursion over the reals** (`DotLoop.accum_step` without rationals). -/
private theorem NL_accum_step {ρ α : ℝ} (hρ : 0 ≤ ρ) (hα : 0 ≤ α) {t : ℕ} {a a' St S Bt A : ℝ}
    (hSt : |St| ≤ Bt) (hSA : |S| ≤ A) (he : |a - St| ≤ ((1 + ρ) ^ t - 1) * Bt + t * (1 + ρ) ^ t * α)
    (hs : |a' - (a + S)| ≤ ρ * (|a| + A) + α) :
    |a' - (St + S)| ≤ ((1 + ρ) ^ (t + 1) - 1) * (Bt + A) + (t + 1 : ℕ) * (1 + ρ) ^ (t + 1) * α := by
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hSA
  have hacc : |a| ≤ Bt + (((1 + ρ) ^ t - 1) * Bt + t * (1 + ρ) ^ t * α) := by
    have := abs_sub_abs_le_abs_sub a St
    linarith
  have htri : |a' - (St + S)| ≤ |a' - (a + S)| + |a - St| := by
    have e2 : a' - (St + S) = (a' - (a + S)) + (a - St) := by ring
    rw [e2]; exact abs_add_le _ _
  set P := (1 + ρ) ^ t with hP
  have hP1 : 1 ≤ P := one_le_pow₀ (by linarith)
  rw [pow_succ, ← hP]
  push_cast
  have hBt : 0 ≤ Bt := (abs_nonneg _).trans hSt
  have hm1 : ρ * |a| ≤ ρ * (Bt + ((P - 1) * Bt + t * P * α)) := mul_le_mul_of_nonneg_left hacc hρ
  nlinarith [mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℝ) ≤ P - 1)) hA0,
    mul_nonneg (by linarith : (0 : ℝ) ≤ P - 1) hA0, mul_nonneg hρ hA0, mul_nonneg hρ hα,
    mul_nonneg (mul_nonneg hρ (by linarith : (0 : ℝ) ≤ P)) hα, mul_nonneg (Nat.cast_nonneg t) hα,
    mul_nonneg (mul_nonneg (Nat.cast_nonneg t) (by linarith : (0 : ℝ) ≤ P)) hα]

/-! ### Sums over the seen tokens -/

/-- A sum over `sPart` is a sum over the rows below `c·cs + o + 32·n`, zero off `toks`. -/
private theorem NL_sum_sPart (d : Inst) (c o n t : Nat) (g : Nat → ℝ) :
    ∑ s ∈ sPart d c o n t, g s =
      ∑ s ∈ Finset.range (c * d.cs + o + BK * n), if s ∈ toks d t then g s else 0 := by
  unfold sPart
  rw [Finset.sum_ite_mem]
  congr 1
  ext s
  simp only [Finset.mem_filter, Finset.mem_inter, Finset.mem_range]
  exact and_comm

/-- The tokens seen after trip `n` are those after trip `n − 1` and the trip's lanes in `toks`. -/
private theorem NL_sum_sPart_succ (d : Inst) (c o n t : Nat) (g : Nat → ℝ) :
    ∑ s ∈ sPart d c o (n + 1) t, g s = ∑ s ∈ sPart d c o n t, g s +
      ∑ l ∈ Finset.range BK, if c * d.cs + o + BK * n + l ∈ toks d t
        then g (c * d.cs + o + BK * n + l) else 0 := by
  rw [NL_sum_sPart, NL_sum_sPart, show c * d.cs + o + BK * (n + 1) = (c * d.cs + o + BK * n) + BK by
    ring, Finset.sum_range_add]

/-- Once the trips cover the token's row, every token of `toks` has been seen. -/
private theorem NL_sPart_full {d : Inst} {c o nt t : Nat} (hcov : t < c * d.cs + o + BK * nt) :
    sPart d c o nt t = toks d t := by
  unfold sPart
  rw [Finset.filter_true_of_mem]
  intro s hs
  have := (Finset.mem_Ico.mp hs).2
  omega

/-- The seen tokens are tokens of `toks`. -/
private theorem NL_sPart_subset (d : Inst) (c o n t : Nat) : sPart d c o n t ⊆ toks d t :=
  Finset.filter_subset _ _

/-- Auxiliary fact: NL wHat nonneg. -/
private theorem NL_wHat_nonneg (m₀ : Mem) (d : Inst) (t h s : Nat) : 0 ≤ wHat m₀ d t h s := by
  obtain ⟨hu, -, -, -⟩ := NL_consts_nonneg
  have hw : 0 ≤ wMag m₀ d t h s :=
    mul_nonneg (mul_nonneg (abs_nonneg _) (NL_eHat_nonneg (NL_dErrS_nonneg m₀ d t h s)))
      (abs_nonneg _)
  have hu3 : 0 ≤ uB := by unfold uB; positivity
  have hab : 0 ≤ wAbs m₀ d t h s := by unfold wAbs; positivity
  unfold wHat
  have : 0 ≤ (1 + uB + 3 * uF) * wMag m₀ d t h s := mul_nonneg (by linarith) hw
  linarith

/-- The magnitudes of the seen tokens' weights times `x` are below the lifted ones. -/
private theorem NL_mWp_le_lWp (m₀ : Mem) (d : Inst) (c o t h p n : Nat) :
    mWp m₀ d c o t h p n ≤ lWp m₀ d c o t h p n := by
  unfold mWp lWp
  refine Finset.sum_le_sum fun s _ => ?_
  have h1 := NL_wHat_nonneg m₀ d t h s
  have h2 := abs_nonneg (xv m₀ d s h p)
  have h3 : (0 : ℝ) ≤ mB := by unfold mB; positivity
  nlinarith [mul_nonneg h1 h3, mul_nonneg h2 h3]

/-- The lifted magnitude of the seen tokens is at most that of all of `toks`. -/
private theorem NL_lWp_le_lW (m₀ : Mem) (d : Inst) (c o t h p n : Nat) :
    lWp m₀ d c o t h p n ≤ lW m₀ d t h p := by
  unfold lWp lW
  refine Finset.sum_le_sum_of_subset_of_nonneg (NL_sPart_subset d c o n t) fun s _ _ => ?_
  have h3 : (0 : ℝ) ≤ mB := by unfold mB; positivity
  exact mul_nonneg (by linarith [NL_wHat_nonneg m₀ d t h s]) (by linarith [abs_nonneg (xv m₀ d s h p)])

/-! ### One lane of a trip -/

/-- A lane's term of the trip: the weight at `s` with the `x` value `x`, at its own token. -/
private def NL_ind (d : Inst) (t base : Nat) (g : Nat → ℝ) (l : Nat) : ℝ :=
  if base + l ∈ toks d t then g (base + l) else 0

/-- **One lane of a trip.** A lane whose token `s` is in `toks` contributes its weight's product
with `x`, which is within `ewS·|x|` of the exact term and at most `wHat·|x|`; any other lane has
the weight `0`. The lifted magnitudes obey the same pattern. -/
private theorem NL_lane (m₀ : Mem) (d : Inst) (t h p s : Nat) {w x : BF16}
    (hw : TokW m₀ d t h s w) (hx : s ∈ toks d t → (x.toRat : ℝ) = xv m₀ d s h p) :
    |(w.toRat : ℝ) * x.toRat| ≤ (if s ∈ toks d t then wHat m₀ d t h s * |xv m₀ d s h p| else 0) ∧
    |(w.toRat : ℝ) * x.toRat - (if s ∈ toks d t then wTS m₀ d t h s * xv m₀ d s h p else 0)| ≤
      (if s ∈ toks d t then ewS m₀ d t h s * |xv m₀ d s h p| else 0) ∧
    ((Device.liftMag .bf16 w.toRat : ℚ) : ℝ) * ((Device.liftMag .bf16 x.toRat : ℚ) : ℝ) ≤
      (if s ∈ toks d t then (wHat m₀ d t h s + mB) * (|xv m₀ d s h p| + mB) else 0) := by
  by_cases hs : s ∈ toks d t
  · obtain ⟨-, hin, -⟩ := hw
    obtain ⟨herr, habs⟩ := hin hs
    have hxe := hx hs
    simp only [hs, ite_true]
    have hmB : (0 : ℝ) ≤ mB := by unfold mB; positivity
    have hwH := NL_wHat_nonneg m₀ d t h s
    refine ⟨?_, ?_, ?_⟩
    · rw [hxe, abs_mul]
      exact mul_le_mul_of_nonneg_right habs (abs_nonneg _)
    · rw [hxe, show (w.toRat : ℝ) * xv m₀ d s h p - wTS m₀ d t h s * xv m₀ d s h p =
        ((w.toRat : ℝ) - wTS m₀ d t h s) * xv m₀ d s h p by ring, abs_mul]
      exact mul_le_mul_of_nonneg_right herr (abs_nonneg _)
    · have hl : ∀ q : ℚ, ((Device.liftMag .bf16 q : ℚ) : ℝ) ≤ |(q : ℝ)| + mB := by
        intro q
        have h := (Rat.cast_le (K := ℝ)).mpr (Device.liftMag_le .bf16 q)
        rw [Device.liftFloor_bf16] at h
        push_cast at h
        exact h
      have hn : ∀ q : ℚ, (0 : ℝ) ≤ ((Device.liftMag .bf16 q : ℚ) : ℝ) := fun q => by
        exact_mod_cast Device.liftMag_nonneg .bf16 q
      refine mul_le_mul ?_ ?_ (hn _) (by linarith [abs_nonneg ((w.toRat : ℝ))])
      · have := hl w.toRat; linarith
      · have := hl x.toRat; rw [hxe] at this; linarith
  · have h0 := hw.2.2 hs
    simp [hs, h0]

/-! ### One trip -/

/-- The trip's lanes extend the seen tokens: the sum over `sPart (n+1)` is that over `sPart n` plus
the lanes' terms. -/
private theorem NL_sum_succ (d : Inst) (c o n t : Nat) (g : Nat → ℝ) :
    ∑ s ∈ sPart d c o (n + 1) t, g s = ∑ s ∈ sPart d c o n t, g s +
      ∑ l : Fin BK, NL_ind d t (c * d.cs + o + BK * n) g l := by
  rw [NL_sum_sPart_succ, Fin.sum_univ_eq_sum_range (fun l => NL_ind d t (c * d.cs + o + BK * n) g l) BK]
  rfl

/-- Auxiliary fact: NL ind mono. -/
private theorem NL_ind_mono {d : Inst} {t base : Nat} {g g' : Nat → ℝ} (h : ∀ s, g s ≤ g' s) (l : Nat) :
    NL_ind d t base g l ≤ NL_ind d t base g' l := by
  unfold NL_ind
  split_ifs
  · exact h _
  · exact le_rfl

/-- Auxiliary fact: NL ind nonneg. -/
private theorem NL_ind_nonneg {d : Inst} {t base : Nat} {g : Nat → ℝ} (h : ∀ s, 0 ≤ g s) (l : Nat) :
    0 ≤ NL_ind d t base g l := by
  unfold NL_ind
  split_ifs
  · exact h _
  · exact le_rfl

/-- The room `DotLift.dotAdd_step` asks for, from `|acc| + lifted ≤ 2^121`. -/
private theorem NL_room_loop {vc A : ℚ} (hA : 0 ≤ A) (h : |vc| + A ≤ 2 ^ 121) :
    2 * ((1 + Device.relDot16 BK) * (|vc| + A) + Device.absDot16 BK + 2 * Format.binary32.eta) ≤
      Format.binary32.maxVal := by
  have hr : Device.relDot16 BK ≤ 1 := by unfold Device.relDot16 BK; norm_num
  have hr0 : 0 ≤ Device.relDot16 BK := by unfold Device.relDot16; positivity
  have hs0 : 0 ≤ |vc| + A := by positivity
  have h1 : (1 + Device.relDot16 BK) * (|vc| + A) ≤ 2 * 2 ^ 121 :=
    mul_le_mul (by linarith) h hs0 (by norm_num)
  have h2 : Device.absDot16 BK + 2 * Format.binary32.eta ≤ 1 := by
    unfold Device.absDot16 BK; rw [Format.binary32_eta]; norm_num
  rw [Format.binary32_maxVal]
  have h3 : (2 : ℚ) * (2 * 2 ^ 121 + 1) ≤ ((2 : ℚ) ^ 24 - 1) * 2 ^ 104 := by norm_num
  linarith

/-- **One trip** of the loop at accumulator lane `(ii, jj)` (token `t`, channel `p`): trip `n`
covers rows `o + 32·n + l`, `l < 32`, of the physical chunk `c`; `a` holds its `bfloat16` weights
(lane `(ii, l)`: `TokW` of token `c·cs + o + 32·n + l`), `x` its `x` rows (read below the limit
`L`, else `+0`), and `cc → r` is `acc += tl.dot(a, x)`. -/
theorem loop_step {P : Platform} (hdot : Device.DotBF16 P) {m₀ : Mem} {d : Inst} (hd : d.ok)
    (hV : Vals m₀ d) {c o L t h p n : Nat} (ht : t < d.T) (hh : h < d.H) (hp : p < d.P)
    (_hct : chk d t = c) (_hl : c * d.cs + o ≤ lTok d t) (hL : t < c * d.cs + L)
    (_hLc : L ≤ d.cs) (hcL : c * d.cs + L ≤ d.T) (hn : n < nIt d)
    {a : Tensor [BM, BK] BF16} {x : Tensor [BK, BN] BF16} {cc z r : Tensor [BM, BN] F32}
    {ii : Fin BM} {jj : Fin BN}
    (hr : P.dot cfgBF16 a x cc r ∨ ∃ dd, P.dot cfgBF16 a x z dd ∧
      ∀ ix, AccAdd true .f32 (dd.get ix) (cc.get ix) (r.get ix))
    (hz : DType.val? .f32 (z.get2 ii jj) = some 0)
    (hw : ∀ l : Fin BK, TokW m₀ d t h (c * d.cs + o + BK * n + l) (a.get2 ii l))
    (hx : ∀ l : Fin BK,
      (o + BK * n + l < L →
        m₀.read .bf16 (d.x.ptr (xIdx d (c * d.cs + (o + BK * n + l)) h p : Int)).addr =
          some (x.get2 l jj)) ∧
      (¬ o + BK * n + l < L → x.get2 l jj = FloatV.ofPat 0))
    (hinv : LoopInv m₀ d c o t h p n (cc.get2 ii jj)) :
    LoopInv m₀ d c o t h p (n + 1) (r.get2 ii jj) := by
  obtain ⟨hcfin, T, hT1, hT2, hT3⟩ := hinv
  obtain ⟨-, hxV, -, -, -, -, -, -, hlast⟩ := hV
  have hroomV : aSig m₀ d t h p + lW m₀ d t h p ≤ 2 ^ 120 := (hlast t h p ht hh hp).2.2.2.2.1
  have hnI : n ≤ 512 := by
    have := hd.cs_le
    have h' : n < nIt d := hn
    unfold nIt BK at h'; omega
  have hBK : BK = 32 := rfl
  -- the `x` lanes
  have hxl : ∀ l : Fin BK, (x.get2 l jj).isFinite = true ∧
      (o + BK * n + l < L →
        ((x.get2 l jj).toRat : ℝ) = xv m₀ d (c * d.cs + o + BK * n + l) h p) ∧
      (¬ o + BK * n + l < L → (x.get2 l jj).toRat = 0) := by
    intro l
    by_cases hlt : o + BK * n + l < L
    · have hread := (hx l).1 hlt
      have hTlt : c * d.cs + (o + BK * n + l) < d.T := by omega
      obtain ⟨v, hv, hvf⟩ := hxV _ h p hTlt hh hp
      have hv' : m₀.read .bf16 (d.x.ptr (xIdx d (c * d.cs + (o + BK * n + l)) h p : Int)).addr =
          some v := hv
      have hvx : x.get2 l jj = v := Option.some.inj (hread.symm.trans hv')
      have e : c * d.cs + (o + BK * n + (l : Nat)) = c * d.cs + o + BK * n + (l : Nat) := by ring
      refine ⟨by rw [hvx]; exact hvf, fun _ => ?_, fun h' => absurd hlt h'⟩
      rw [e] at hv
      rw [hvx]
      simp [xv, valB, hv]
    · rw [(hx l).2 hlt]
      exact ⟨FloatV.isFinite_zero_bf16, fun h' => absurd h' hlt, fun _ => FloatV.toRat_ofPat_zero⟩
  have hxs : ∀ l : Fin BK, c * d.cs + o + BK * n + l ∈ toks d t →
      ((x.get2 l jj).toRat : ℝ) = xv m₀ d (c * d.cs + o + BK * n + l) h p := by
    intro l hs
    have := (Finset.mem_Ico.mp hs).2
    exact (hxl l).2.1 (by omega)
  have hlane := fun l : Fin BK => NL_lane m₀ d t h p (c * d.cs + o + BK * n + l) (hw l) (hxs l)
  -- the trip's sums
  set base := c * d.cs + o + BK * n with hbase
  set S : ℝ := ∑ l : Fin BK, ((a.get2 ii l).toRat : ℝ) * (x.get2 l jj).toRat with hS
  set A4 : ℝ := ∑ l : Fin BK, NL_ind d t base
    (fun s => (wHat m₀ d t h s + mB) * (|xv m₀ d s h p| + mB)) l with hA4
  have hg14 : ∀ s, wHat m₀ d t h s * |xv m₀ d s h p| ≤
      (wHat m₀ d t h s + mB) * (|xv m₀ d s h p| + mB) := by
    intro s
    have h1 := NL_wHat_nonneg m₀ d t h s
    have h2 := abs_nonneg (xv m₀ d s h p)
    have h3 : (0 : ℝ) ≤ mB := by unfold mB; positivity
    nlinarith [mul_nonneg h1 h3, mul_nonneg h2 h3]
  have hg4 : ∀ s, 0 ≤ (wHat m₀ d t h s + mB) * (|xv m₀ d s h p| + mB) := by
    intro s
    have h3 : (0 : ℝ) ≤ mB := by unfold mB; positivity
    exact mul_nonneg (by linarith [NL_wHat_nonneg m₀ d t h s]) (by linarith [abs_nonneg (xv m₀ d s h p)])
  have hA40 : 0 ≤ A4 := Finset.sum_nonneg fun l _ => NL_ind_nonneg hg4 l
  have hS1 : |S| ≤ ∑ l : Fin BK, NL_ind d t base (fun s => wHat m₀ d t h s * |xv m₀ d s h p|) l :=
    (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun l _ => (hlane l).1)
  have hSA : |S| ≤ A4 := hS1.trans (Finset.sum_le_sum fun l _ => NL_ind_mono hg14 l)
  have hS2 : |S - ∑ l : Fin BK, NL_ind d t base (fun s => wTS m₀ d t h s * xv m₀ d s h p) l| ≤
      ∑ l : Fin BK, NL_ind d t base (fun s => ewS m₀ d t h s * |xv m₀ d s h p|) l := by
    rw [hS, ← Finset.sum_sub_distrib]
    exact (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun l _ => (hlane l).2.1)
  -- the dot
  have hva : ∀ k : Fin BK, DType.val? .bf16 (a.get2 ii k) = some (a.get2 ii k).toRat :=
    fun k => DType.val?_bf16.mpr ⟨(hw k).1, rfl⟩
  have hvb : ∀ k : Fin BK, DType.val? .bf16 (x.get2 k jj) = some (x.get2 k jj).toRat :=
    fun k => DType.val?_bf16.mpr ⟨(hxl k).1, rfl⟩
  have hAdot : ∑ k : Fin BK, ((Device.liftMag .bf16 (a.get2 ii k).toRat : ℚ) : ℝ) *
      ((Device.liftMag .bf16 (x.get2 k jj).toRat : ℚ) : ℝ) ≤ A4 :=
    Finset.sum_le_sum fun l _ => (hlane l).2.2
  have hWp := NL_sum_succ d c o n t (fun s => (wHat m₀ d t h s + mB) * (|xv m₀ d s h p| + mB))
  have hlWp : lWp m₀ d c o t h p (n + 1) = lWp m₀ d c o t h p n + A4 := hWp
  have hBt : |T| ≤ aSig m₀ d t h p + lWp m₀ d c o t h p n :=
    hT3.trans (by linarith [NL_mWp_le_lWp m₀ d c o t h p n])
  have hBt0 : 0 ≤ aSig m₀ d t h p + lWp m₀ d c o t h p n := (abs_nonneg _).trans hBt
  have hcap : aSig m₀ d t h p + lWp m₀ d c o t h p n + A4 ≤ 2 ^ 120 := by
    have := NL_lWp_le_lW m₀ d c o t h p (n + 1)
    linarith
  have hc0 : |((cc.get2 ii jj).toRat : ℝ)| ≤
      (aSig m₀ d t h p + lWp m₀ d c o t h p n) / 64 + 1 +
        (aSig m₀ d t h p + lWp m₀ d c o t h p n) := by
    have h1 := NL_bndR_small hnI hBt0
    have h2 := NL_bndR_mono n (le_refl (aSig m₀ d t h p + lWp m₀ d c o t h p n))
    have h3 := abs_sub_abs_le_abs_sub ((cc.get2 ii jj).toRat : ℝ) T
    linarith
  have hroomR : |((cc.get2 ii jj).toRat : ℝ)| + A4 ≤ 2 ^ 121 := by linarith
  have hAq0 : (0 : ℚ) ≤ ∑ k : Fin BK, Device.liftMag .bf16 (a.get2 ii k).toRat *
      Device.liftMag .bf16 (x.get2 k jj).toRat :=
    Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hroomQ : |(cc.get2 ii jj).toRat| + ∑ k : Fin BK, Device.liftMag .bf16 (a.get2 ii k).toRat *
      Device.liftMag .bf16 (x.get2 k jj).toRat ≤ 2 ^ 121 := by
    have : ((|(cc.get2 ii jj).toRat| + ∑ k : Fin BK, Device.liftMag .bf16 (a.get2 ii k).toRat *
        Device.liftMag .bf16 (x.get2 k jj).toRat : ℚ) : ℝ) ≤ ((2 ^ 121 : ℚ) : ℝ) := by
      push_cast
      linarith
    exact_mod_cast this
  have hrel : 0 ≤ Device.relDot16 BK := by unfold Device.relDot16; positivity
  have habs : 0 ≤ Device.absDot16 BK := by unfold Device.absDot16; positivity
  obtain ⟨hrfin, hrerr⟩ := DotLift.dotAdd_step (hdot .tf32) hrel habs hr ii jj hz hva hvb hcfin
    (NL_room_loop hAq0 hroomQ)
  have hrR : |((r.get2 ii jj).toRat : ℝ) - (((cc.get2 ii jj).toRat : ℝ) + S)| ≤
      (rhoS : ℝ) * (|((cc.get2 ii jj).toRat : ℝ)| + A4) + alpS := by
    have h1 := (Rat.cast_le (K := ℝ)).mpr hrerr
    push_cast at h1
    have hr0 : (0 : ℝ) ≤ rhoS := by exact_mod_cast NL_rho_nonneg
    have h2 := mul_le_mul_of_nonneg_left hAdot hr0
    have h3 : (rhoS : ℝ) * (|((cc.get2 ii jj).toRat : ℝ)| + ∑ k : Fin BK,
        ((Device.liftMag .bf16 (a.get2 ii k).toRat : ℚ) : ℝ) *
          ((Device.liftMag .bf16 (x.get2 k jj).toRat : ℚ) : ℝ)) ≤
        (rhoS : ℝ) * (|((cc.get2 ii jj).toRat : ℝ)| + A4) :=
      mul_le_mul_of_nonneg_left (by linarith) hr0
    have h4 : (((DotLoop.stepRel Device.relDot16 BK : ℚ)) : ℝ) = (rhoS : ℝ) := rfl
    have h5 : (((DotLoop.stepAbs Device.absDot16 BK : ℚ)) : ℝ) = (alpS : ℝ) := rfl
    rw [h4, h5] at h1
    linarith
  have hρ : (0 : ℝ) ≤ rhoS := by exact_mod_cast NL_rho_nonneg
  have hα : (0 : ℝ) ≤ alpS := by exact_mod_cast NL_alp_nonneg
  have hstep := NL_accum_step hρ hα hBt hSA hT1 hrR
  refine ⟨hrfin, T + S, ?_, ?_, ?_⟩
  · have : aSig m₀ d t h p + lWp m₀ d c o t h p (n + 1) =
        aSig m₀ d t h p + lWp m₀ d c o t h p n + A4 := by rw [hlWp]; ring
    rw [this]
    exact hstep
  · have e1 : refP m₀ d c o t h p (n + 1) = refP m₀ d c o t h p n +
        ∑ l : Fin BK, NL_ind d t base (fun s => wTS m₀ d t h s * xv m₀ d s h p) l := by
      unfold refP
      rw [NL_sum_succ d c o n t (fun s => wTS m₀ d t h s * xv m₀ d s h p)]
      ring
    have e2 : errWp m₀ d c o t h p (n + 1) = errWp m₀ d c o t h p n +
        ∑ l : Fin BK, NL_ind d t base (fun s => ewS m₀ d t h s * |xv m₀ d s h p|) l := by
      unfold errWp
      rw [NL_sum_succ d c o n t (fun s => ewS m₀ d t h s * |xv m₀ d s h p|)]
    rw [e1, e2]
    have := abs_add_le (T - refP m₀ d c o t h p n)
      (S - ∑ l : Fin BK, NL_ind d t base (fun s => wTS m₀ d t h s * xv m₀ d s h p) l)
    have e3 : T + S - (refP m₀ d c o t h p n +
        ∑ l : Fin BK, NL_ind d t base (fun s => wTS m₀ d t h s * xv m₀ d s h p) l) =
        (T - refP m₀ d c o t h p n) +
          (S - ∑ l : Fin BK, NL_ind d t base (fun s => wTS m₀ d t h s * xv m₀ d s h p) l) := by
      ring
    rw [e3]
    linarith
  · have e1 : mWp m₀ d c o t h p (n + 1) = mWp m₀ d c o t h p n +
        ∑ l : Fin BK, NL_ind d t base (fun s => wHat m₀ d t h s * |xv m₀ d s h p|) l := by
      unfold mWp
      rw [NL_sum_succ d c o n t (fun s => wHat m₀ d t h s * |xv m₀ d s h p|)]
    rw [e1]
    have := abs_add_le T S
    linarith

/-- **After the loop**: with `nt ≤ nIt` trips covering the token's row, the invariant is the
output of the loop: `acc` is within `errSig + errW + errDot` of the exact `exp(dSt)·Σ C·S +
Σ_{s ∈ toks} wTS·x`, and of magnitude at most `aAcc`. -/
theorem loop_final {m₀ : Mem} {d : Inst} (hd : d.ok) (_hV : Vals m₀ d) {c o t h p nt : Nat}
    (_ht : t < d.T) (_hh : h < d.H) (_hp : p < d.P) (_hl : c * d.cs + o ≤ lTok d t)
    (hnt : nt ≤ nIt d) (hcov : t < c * d.cs + o + BK * nt) {acc : F32}
    (hinv : LoopInv m₀ d c o t h p nt acc) :
    acc.isFinite = true ∧
      |(acc.toRat : ℝ) - (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
        ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p)| ≤
        errSig m₀ d t h p + errW m₀ d t h p + errDot m₀ d t h p ∧
      |(acc.toRat : ℝ)| ≤ aAcc m₀ d t h p := by
  obtain ⟨hfin, T, h1, h2, h3⟩ := hinv
  have hS := NL_sPart_full (d := d) (c := c) (o := o) hcov
  have eL : lWp m₀ d c o t h p nt = lW m₀ d t h p := by unfold lWp lW; rw [hS]
  have eM : mWp m₀ d c o t h p nt = mW m₀ d t h p := by unfold mWp mW; rw [hS]
  have eE : errWp m₀ d c o t h p nt = errW m₀ d t h p := by unfold errWp errW; rw [hS]
  have eR : refP m₀ d c o t h p nt = Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
      ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p := by unfold refP; rw [hS]
  rw [eL] at h1
  rw [eE, eR] at h2
  rw [eM] at h3
  have hmW := NL_mWp_le_lWp m₀ d c o t h p nt
  rw [eL, eM] at hmW
  have hB : 0 ≤ aSig m₀ d t h p + lW m₀ d t h p := by
    have := abs_nonneg T; linarith
  have hnI : nIt d ≤ 512 := by
    have := hd.cs_le
    unfold nIt BK; omega
  have hbd : bndR nt (aSig m₀ d t h p + lW m₀ d t h p) ≤ errDot m₀ d t h p := by
    refine (NL_bndR_le (hnt.trans hnI) hB).trans ?_
    unfold errDot
    have hr0 : (0 : ℝ) ≤ rhoS := by exact_mod_cast NL_rho_nonneg
    have ha0 : (0 : ℝ) ≤ alpS := by exact_mod_cast NL_alp_nonneg
    have hX : (rhoS : ℝ) * (aSig m₀ d t h p + lW m₀ d t h p) + alpS ≤
        (relD BK + 2 ^ (-21 : ℤ)) * (aSig m₀ d t h p + lW m₀ d t h p) + absD BK +
          2 ^ (-140 : ℤ) := by
      have := mul_le_mul_of_nonneg_right NL_rho_le_rel hB
      have := NL_alp_le_abs
      linarith
    have hn : (nt : ℝ) ≤ nIt d := by exact_mod_cast hnt
    have hn0 : (0 : ℝ) ≤ nt := Nat.cast_nonneg nt
    have hX0 : 0 ≤ (rhoS : ℝ) * (aSig m₀ d t h p + lW m₀ d t h p) + alpS := by positivity
    have := mul_le_mul (mul_le_mul_of_nonneg_left hn (by norm_num : (0:ℝ) ≤ 2)) hX hX0
      (by positivity)
    linarith
  have hbd0 := NL_bndR_nonneg nt hB
  refine ⟨hfin, ?_, ?_⟩
  · have := abs_sub_le (acc.toRat : ℝ) T (Real.exp (dSt m₀ d t h) * sig m₀ d t h p +
      ∑ s ∈ toks d t, wTS m₀ d t h s * xv m₀ d s h p)
    linarith
  · unfold aAcc
    have := abs_sub_abs_le_abs_sub (acc.toRat : ℝ) T
    linarith

end Claims.SsdChunkScan
