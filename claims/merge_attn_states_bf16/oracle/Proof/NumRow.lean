import Claim
import TVA.Proof
import TVA
import Proof.NumFacts
import Proof.NumWide
import Proof.NumMerge

/-!
# The row stored by the merge branch

The two scales, the two products, their sum and the conversion to `bfloat16`, turned into the
claim's reference output.
-/

set_option maxHeartbeats 1600000

namespace Claims.MergeAttnStates

open TVA

/-- The weight of one side is nonnegative. -/
theorem side_nonneg' (x : F32) (m : ℝ) : 0 ≤ side x m := by
  unfold side
  split
  · norm_num
  · exact Real.exp_nonneg _

/-- The first weight is at most one. -/
theorem side_le_one_left' (p s : F32) : side p (refMax p s) ≤ 1 := by
  by_cases hp : p = negInfC
  · rw [hp, side_negInf]; norm_num
  · rw [side_of_ne hp]
    by_cases hs : s = negInfC
    · rw [hs, refMax_negInf_right hp, Real.exp_le_one_iff]; linarith
    · rw [refMax_eq_max hp hs, Real.exp_le_one_iff]
      linarith [le_max_left p.toReal s.toReal]

/-- The second weight is at most one. -/
theorem side_le_one_right' (p s : F32) : side s (refMax p s) ≤ 1 := by
  by_cases hs : s = negInfC
  · rw [hs, side_negInf]; norm_num
  · rw [side_of_ne hs]
    by_cases hp : p = negInfC
    · rw [hp, refMax_negInf_left, Real.exp_le_one_iff]; linarith
    · rw [refMax_eq_max hp hs, Real.exp_le_one_iff]
      linarith [le_max_right p.toReal s.toReal]

/-- The two weights sum to at least one. -/
theorem one_le_side_sum' {p s : F32} (hne : ¬ (p = negInfC ∧ s = negInfC)) :
    1 ≤ side p (refMax p s) + side s (refMax p s) := by
  by_cases hp : p = negInfC
  · have hs : s ≠ negInfC := fun h => hne ⟨hp, h⟩
    rw [hp, side_negInf, refMax_negInf_left, side_of_ne hs]
    have h0 : s.toReal - s.toReal = 0 := by ring
    rw [h0, Real.exp_zero]; norm_num
  · by_cases hs : s = negInfC
    · rw [hs, side_negInf, refMax_negInf_right hp, side_of_ne hp]
      have h0 : p.toReal - p.toReal = 0 := by ring
      rw [h0, Real.exp_zero]; norm_num
    · rw [refMax_eq_max hp hs, side_of_ne hp, side_of_ne hs]
      rcases le_total p.toReal s.toReal with h | h
      · have h0 : s.toReal - max p.toReal s.toReal = 0 := by rw [max_eq_right h]; ring
        rw [h0, Real.exp_zero]
        linarith [Real.exp_nonneg (p.toReal - max p.toReal s.toReal)]
      · have h0 : p.toReal - max p.toReal s.toReal = 0 := by rw [max_eq_left h]; ring
        rw [h0, Real.exp_zero]
        linarith [Real.exp_nonneg (s.toReal - max p.toReal s.toReal)]

/-- The reference output is a convex combination of the two values. -/
theorem abs_refOut_le_max' {p s : F32} (hne : ¬ (p = negInfC ∧ s = negInfC)) (a b : ℝ) :
    |refOut p s a b| ≤ max |a| |b| := by
  set SP := side p (refMax p s) with hSP
  set SS := side s (refMax p s) with hSS
  set W := refSum p s with hW
  have hWsum : W = SP + SS := by rw [hW, refSum, ← hSP, ← hSS]
  have hWge : 1 ≤ W := by
    have h := one_le_side_sum' (p := p) (s := s) hne
    rw [← hSP, ← hSS, ← hWsum] at h; exact h
  have hWpos : 0 < W := by linarith
  have hWne : W ≠ 0 := ne_of_gt hWpos
  have h1 : 0 ≤ SP / W := div_nonneg (side_nonneg' _ _) hWpos.le
  have h2 : 0 ≤ SS / W := div_nonneg (side_nonneg' _ _) hWpos.le
  have h3 : SP / W + SS / W = 1 := by
    rw [← add_div, ← hWsum]
    exact div_self hWne
  have e : refOut p s a b = (SP / W) * a + (SS / W) * b := by
    rw [refOut, ← hSP, ← hSS, ← hW]
    field_simp
  rw [e]
  exact abs_le_max_of_weights h1 h2 h3

/-- A rational absolute bound, cast to the reals. -/
theorem cast_abs_le' {a c : ℚ} (h : |a| ≤ c) : |(a : ℝ)| ≤ (c : ℝ) := by
  have h' : ((|a| : ℚ) : ℝ) ≤ (c : ℝ) := Rat.cast_le.mpr h
  rwa [Rat.cast_abs] at h'

/-- A real ratio bound, cast down to the rationals. -/
theorem rat_div_abs_le' {a b c : ℚ} (h : |(a : ℝ) / (b : ℝ)| ≤ (c : ℝ)) : |a / b| ≤ c := by
  have h' : ((|a / b| : ℚ) : ℝ) ≤ (c : ℝ) := by
    rw [Rat.cast_abs, Rat.cast_div]
    exact h
  exact Rat.cast_le.mp h'

/-- The conversion of a finite `bfloat16` value into `float32` is exact. -/
theorem cvt_bf16_f32 {pv : BF16} {p32 : F32} (hpv : pv.isFinite = true)
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn pv) p32) :
    p32 = FloatV.cvtWith .rn pv ∧ p32.isFinite = true ∧ p32.toRat = pv.toRat := by
  obtain ⟨hfin, hrat⟩ := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
    Format.bfloat16_regular Format.binary32_regular Format.wider_bfloat16_binary32 .rn hpv
  have hnan : (FloatV.cvtWith .rn pv : F32).isNaN = false := (FloatV.isFinite_iff.mp hfin).1
  have hy : p32 = FloatV.cvtWith .rn pv := hc.eq hnan
  exact ⟨hy, by rw [hy]; exact hfin, by rw [hy]; exact hrat⟩


/-- A quotient of a bounded numerator by a positive denominator near one. -/
theorem abs_div_le_close' {a b : ℝ} (hb : 1 - 2 ^ (-18 : ℤ) ≤ b)
    (ha : |a| ≤ 1 + 2 ^ (-16 : ℤ)) :
    |a / b| ≤ 1 + 2 ^ (-15 : ℤ) := by
  have hpos : (0 : ℝ) < b := by
    have h : (0 : ℝ) < 1 - 2 ^ (-18 : ℤ) := by norm_num
    linarith
  rw [abs_div, abs_of_pos hpos, div_le_iff₀ hpos]
  have hm := mul_le_mul_of_nonneg_left hb (by positivity : (0 : ℝ) ≤ 1 + 2 ^ (-15 : ℤ))
  have hnum : (1 : ℝ) + 2 ^ (-16 : ℤ) ≤ (1 + 2 ^ (-15 : ℤ)) * (1 - 2 ^ (-18 : ℤ)) := by
    norm_num
  linarith [ha, hm, hnum]

/-- The `float32` quotient of the two exponentials is below `2^126`. -/
theorem div_ratio_q' {pe o : F32} (ho : 1 - 2 ^ (-18 : ℤ) ≤ o.toReal)
    (hpe : |pe.toReal| ≤ 1 + 2 ^ (-16 : ℤ)) :
    |pe.toRat / o.toRat| ≤ (2 : ℚ) ^ 126 := by
  have hpos : (0 : ℝ) < o.toReal := by
    have h : (0 : ℝ) < 1 - 2 ^ (-18 : ℤ) := by norm_num
    linarith
  have hratio : |pe.toReal / o.toReal| ≤ 1 + 2 ^ (-15 : ℤ) := abs_div_le_close' ho hpe
  apply rat_div_abs_le' (a := pe.toRat) (b := o.toRat) (c := (2 : ℚ) ^ 126)
  show |pe.toReal / o.toReal| ≤ (((2 : ℚ) ^ 126 : ℚ) : ℝ)
  push_cast
  have h2 : (1 : ℝ) + 2 ^ (-15 : ℤ) ≤ (2 : ℝ) ^ 126 := by norm_num
  linarith

/-- The scale is computed within `2^-16 + 2^-20`. -/
theorem scale_err' {pe o SP W ps : ℝ}
    (ho0 : 1 - 2 ^ (-18 : ℤ) ≤ o)
    (hpe : |pe - SP| ≤ 2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ))
    (hSP0 : 0 ≤ SP) (hSP1 : SP ≤ 1)
    (hW1 : 1 ≤ W) (hW2 : W ≤ 2)
    (hWo : |W - o| ≤ 2 ^ (-18 : ℤ))
    (hps : |ps - pe / o| ≤ 2 ^ (-22 : ℤ) * (1 + 2 ^ (-15 : ℤ)) + 2 ^ (-148 : ℤ)) :
    |ps - SP / W| ≤ 2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ) := by
  have hhalf : (1 : ℝ) / 2 ≤ o := by
    have hx : (2 : ℝ) ^ (-18 : ℤ) = 1 / 262144 := by norm_num
    rw [hx] at ho0
    linarith
  have ho_pos : 0 < o := by linarith
  have hWpos : 0 < W := by linarith
  have hpe1 : |pe| ≤ 1 + 2 ^ (-16 : ℤ) := by
    obtain ⟨hlo, hhi⟩ := abs_le.mp hpe
    have hnum : (2 : ℝ) ^ (-20 : ℤ) + 2 ^ (-21 : ℤ) ≤ (2 : ℝ) ^ (-16 : ℤ) := by norm_num
    rw [abs_le]
    constructor <;> linarith
  have hratio : |pe / o| ≤ 2 := by
    rw [abs_div, abs_of_pos ho_pos, div_le_iff₀ ho_pos]
    have hnum : (2 : ℝ) ^ (-16 : ℤ) + 2 * 2 ^ (-18 : ℤ) ≤ 1 := by norm_num
    nlinarith [hpe1, ho0, hnum]
  have hstep1 : |ps - pe / o| ≤ 2 ^ (-20 : ℤ) := by
    have h1 : 2 ^ (-22 : ℤ) * |pe / o| ≤ 2 ^ (-22 : ℤ) * 2 :=
      mul_le_mul_of_nonneg_left hratio (by positivity)
    have h2 : (2 : ℝ) ^ (-22 : ℤ) * (1 + 2 ^ (-15 : ℤ)) + 2 ^ (-148 : ℤ) ≤
        (2 : ℝ) ^ (-20 : ℤ) := by
      norm_num
    linarith [hps, h1]
  have hstep2 : |pe / o - SP / W| ≤ 2 ^ (-16 : ℤ) := by
    have hnum : |pe * W - SP * o| ≤ 2 ^ (-17 : ℤ) := by
      have e : pe * W - SP * o = (pe - SP) * W + SP * (W - o) := by ring
      rw [e]
      have hW2' : |W| ≤ 2 := by rw [abs_of_pos hWpos]; exact hW2
      have h1 : |(pe - SP) * W| ≤ (2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ)) * 2 := by
        rw [abs_mul]
        exact mul_le_mul hpe hW2' (abs_nonneg _) (by positivity)
      have hSP1' : |SP| ≤ 1 := by rw [abs_of_nonneg hSP0]; exact hSP1
      have h2 : |SP * (W - o)| ≤ 1 * 2 ^ (-18 : ℤ) := by
        rw [abs_mul]
        exact mul_le_mul hSP1' hWo (abs_nonneg _) (by norm_num)
      have h3 : ((2 : ℝ) ^ (-20 : ℤ) + 2 ^ (-21 : ℤ)) * 2 + 1 * 2 ^ (-18 : ℤ) ≤
          (2 : ℝ) ^ (-17 : ℤ) := by
        norm_num
      calc |(pe - SP) * W + SP * (W - o)| ≤ |(pe - SP) * W| + |SP * (W - o)| := abs_add_le _ _
        _ ≤ (2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ)) * 2 + 1 * 2 ^ (-18 : ℤ) := by linarith
        _ ≤ 2 ^ (-17 : ℤ) := h3
    have hden : 0 < o * W := mul_pos ho_pos hWpos
    have hden2 : (1 : ℝ) / 2 ≤ o * W := by nlinarith [hhalf, hW1]
    have e : pe / o - SP / W = (pe * W - SP * o) / (o * W) := by
      field_simp
    rw [e, abs_div, abs_of_pos hden, div_le_iff₀ hden]
    have hm := mul_le_mul_of_nonneg_left hden2 (by positivity : (0 : ℝ) ≤ 2 ^ (-16 : ℤ))
    have hc : (2 : ℝ) ^ (-16 : ℤ) * (1 / 2) = (2 : ℝ) ^ (-17 : ℤ) := by norm_num
    rw [hc] at hm
    linarith [hnum, hm]
  have e : ps - SP / W = (ps - pe / o) + (pe / o - SP / W) := by ring
  rw [e]
  calc |(ps - pe / o) + (pe / o - SP / W)| ≤ |ps - pe / o| + |pe / o - SP / W| := abs_add_le _ _
    _ ≤ 2 ^ (-20 : ℤ) + 2 ^ (-16 : ℤ) := by linarith
    _ = 2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ) := by ring

/-- The division of one exponential by their sum: finite, at most `1 + 2^-11`, and within
`2^-16 + 2^-20` of the reference scale. -/
theorem scale_bound {P : Platform} (hD : Device.DivFull P)
    {e o ps : F32} {A W : ℝ}
    (hefin : e.isFinite = true) (hofin : o.isFinite = true)
    (ho0 : 1 - 2 ^ (-18 : ℤ) ≤ o.toReal)
    (heA : |e.toReal - A| ≤ 2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ))
    (_he1 : e.toReal ≤ 1 + 2 ^ (-16 : ℤ))
    (hA0 : 0 ≤ A) (hA1 : A ≤ 1)
    (hW1 : 1 ≤ W) (hW2 : W ≤ 2)
    (hWo : |W - o.toReal| ≤ 2 ^ (-18 : ℤ))
    (hd : P.div .binary32 e o ps) :
    ps.isFinite = true ∧ |ps.toReal| ≤ 1 + 2 ^ (-11 : ℤ) ∧
      |ps.toReal - A / W| ≤ 2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ) := by
  have hAabs : |A| ≤ 1 := by rw [abs_of_nonneg hA0]; exact hA1
  have heabs : |e.toReal| ≤ 1 + 2 ^ (-16 : ℤ) := by
    have h3 := abs_add_le (e.toReal - A) A
    have e : (e.toReal - A) + A = e.toReal := by ring
    rw [e] at h3
    have h2 : (2 : ℝ) ^ (-20 : ℤ) + 2 ^ (-21 : ℤ) + 1 ≤ 1 + 2 ^ (-16 : ℤ) := by norm_num
    exact h3.trans (by linarith [heA, hAabs, h2])
  have ho_pos : (0 : ℝ) < o.toReal := by
    have h : (0 : ℝ) < 1 - 2 ^ (-18 : ℤ) := by norm_num
    linarith
  have hoRat : o.toRat ≠ 0 := by
    intro h
    have hz : o.toReal = 0 := by show (o.toRat : ℝ) = 0; rw [h]; norm_num
    linarith
  have hratioQ : |e.toRat / o.toRat| ≤ (2 : ℚ) ^ 126 := div_ratio_q' ho0 heabs
  obtain ⟨hpsfin, hpsq⟩ := hD e o ps hd hefin hofin hoRat hratioQ
  have hpsReal : |ps.toReal - e.toReal / o.toReal| ≤
      2 ^ (-22 : ℤ) * |e.toReal / o.toReal| + 2 ^ (-148 : ℤ) := by
    have hc : ((|ps.toRat - e.toRat / o.toRat| : ℚ) : ℝ) ≤
        (((2 : ℚ) ^ (-22 : ℤ) * |e.toRat / o.toRat| + (2 : ℚ) ^ (-148 : ℤ) : ℚ) : ℝ) :=
      Rat.cast_le.mpr hpsq
    push_cast at hc
    simpa [FloatV.toReal] using hc
  have hratio : |e.toReal / o.toReal| ≤ 1 + 2 ^ (-15 : ℤ) := abs_div_le_close' ho0 heabs
  have hps' : |ps.toReal - e.toReal / o.toReal| ≤
      2 ^ (-22 : ℤ) * (1 + 2 ^ (-15 : ℤ)) + 2 ^ (-148 : ℤ) := by
    have h := mul_le_mul_of_nonneg_left hratio (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ))
    linarith
  have habs_le : |ps.toReal| ≤ 1 + 2 ^ (-11 : ℤ) := by
    have hstep : |ps.toReal - e.toReal / o.toReal| ≤ 2 ^ (-21 : ℤ) := by
      have hnum : (2 : ℝ) ^ (-22 : ℤ) * (1 + 2 ^ (-15 : ℤ)) + 2 ^ (-148 : ℤ) ≤
          2 ^ (-21 : ℤ) := by norm_num
      linarith
    have h3 := abs_add_le (ps.toReal - e.toReal / o.toReal) (e.toReal / o.toReal)
    have e : (ps.toReal - e.toReal / o.toReal) + e.toReal / o.toReal = ps.toReal := by ring
    rw [e] at h3
    have hnum : (2 : ℝ) ^ (-21 : ℤ) + (1 + 2 ^ (-15 : ℤ)) ≤ 1 + 2 ^ (-11 : ℤ) := by norm_num
    exact h3.trans ((add_le_add hstep hratio).trans hnum)
  exact ⟨hpsfin, habs_le,
    scale_err' (pe := e.toReal) (o := o.toReal) (SP := A) (W := W) (ps := ps.toReal)
      ho0 heA hA0 hA1 hW1 hW2 hWo hps'⟩

/-- A product of a `bfloat16`-bounded value and a scale is in `float32` range. -/
theorem mul_prod_le' {p32 ps : F32}
    (h1 : |p32.toRat| ≤ Format.bfloat16.maxVal) (h2 : |ps.toRat| ≤ 1 + 2 ^ (-11 : ℤ)) :
    |p32.toRat * ps.toRat| ≤ Format.binary32.maxVal := by
  have hmax0 : (0 : ℚ) ≤ Format.bfloat16.maxVal := by rw [Format.bfloat16_maxVal]; positivity
  have hle : |p32.toRat * ps.toRat| ≤ Format.bfloat16.maxVal * (1 + 2 ^ (-11 : ℤ)) := by
    rw [abs_mul]
    exact mul_le_mul h1 h2 (abs_nonneg _) hmax0
  refine hle.trans ?_
  rw [Format.bfloat16_maxVal, Format.binary32_maxVal]
  norm_num


/-- The product of a converted row and a scale: finite, and within one rounding of the exact
product, bounded by the row's magnitude. -/
theorem prod_bound {p32 ps prodp : F32} {v MX : ℝ}
    (hp32R : p32.toReal = v) (hMX0 : 0 ≤ MX)
    (hp32le : |p32.toReal| ≤ MX) (hps_le : |ps.toReal| ≤ 1 + 2 ^ (-11 : ℤ))
    (hp32fin : p32.isFinite = true) (hpsfin : ps.isFinite = true)
    (hp32maxq : |p32.toRat| ≤ Format.bfloat16.maxVal)
    (hm : FloatV.NaNEq (FloatV.mul p32 ps) prodp) :
    prodp.isFinite = true ∧
      |prodp.toReal - v * ps.toReal| ≤ 2 ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ) ∧
      |prodp.toReal| ≤ (1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ) := by
  have hpsRat : |ps.toRat| ≤ 1 + 2 ^ (-11 : ℤ) := by
    have h : ((|ps.toRat| : ℚ) : ℝ) ≤ (((1 : ℚ) + (2 : ℚ) ^ (-11 : ℤ) : ℚ) : ℝ) := by
      push_cast
      simpa [FloatV.toReal] using hps_le
    exact Rat.cast_le.mp h
  have hmulmax : |p32.toRat * ps.toRat| ≤ Format.binary32.maxVal := mul_prod_le' hp32maxq hpsRat
  obtain ⟨hfin, herr⟩ :=
    FloatV.mul_err (f := Format.binary32) Format.binary32_regular hp32fin hpsfin hmulmax
  have heq : prodp = FloatV.mul p32 ps := hm.eq (FloatV.isFinite_iff.mp hfin).1
  have hReal : |prodp.toReal - p32.toReal * ps.toReal| ≤
      2 ^ (-24 : ℤ) * |p32.toReal * ps.toReal| + 2 ^ (-150 : ℤ) := by
    rw [heq]
    have hc : ((|(FloatV.mul p32 ps).toRat - p32.toRat * ps.toRat| : ℚ) : ℝ) ≤
        ((|p32.toRat * ps.toRat| * Format.binary32.u + Format.binary32.eta : ℚ) : ℝ) :=
      Rat.cast_le.mpr herr
    rw [Format.binary32_u, Format.binary32_eta] at hc
    push_cast at hc
    simpa [FloatV.toReal, abs_mul, mul_comm, mul_left_comm, mul_assoc] using hc
  have hmulabs : |p32.toReal * ps.toReal| ≤ MX * (1 + 2 ^ (-11 : ℤ)) := by
    rw [abs_mul]
    exact mul_le_mul hp32le hps_le (abs_nonneg _) hMX0
  have hmain : |prodp.toReal - v * ps.toReal| ≤ 2 ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ) := by
    rw [← hp32R]
    have hcoef : (2 : ℝ) ^ (-24 : ℤ) * (MX * (1 + 2 ^ (-11 : ℤ))) ≤ (2 : ℝ) ^ (-22 : ℤ) * MX := by
      have hc : (2 : ℝ) ^ (-24 : ℤ) * (1 + 2 ^ (-11 : ℤ)) ≤ (2 : ℝ) ^ (-22 : ℤ) := by norm_num
      have he : (2 : ℝ) ^ (-24 : ℤ) * (MX * (1 + 2 ^ (-11 : ℤ))) =
          ((2 : ℝ) ^ (-24 : ℤ) * (1 + 2 ^ (-11 : ℤ))) * MX := by ring
      rw [he]
      exact mul_le_mul_of_nonneg_right hc hMX0
    have hb := mul_le_mul_of_nonneg_left hmulabs (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-24 : ℤ))
    linarith [hReal, hb, hcoef]
  refine ⟨?_, hmain, ?_⟩
  · rw [heq]; exact hfin
  · have e : prodp.toReal = (prodp.toReal - v * ps.toReal) + v * ps.toReal := by ring
    have hmulabs' : |v * ps.toReal| ≤ MX * (1 + 2 ^ (-11 : ℤ)) := by
      rw [← hp32R]; exact hmulabs
    calc |prodp.toReal| = |(prodp.toReal - v * ps.toReal) + v * ps.toReal| :=
          congrArg (fun z : ℝ => |z|) e
      _ ≤ |prodp.toReal - v * ps.toReal| + |v * ps.toReal| := abs_add_le _ _
      _ ≤ (2 ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ)) + MX * (1 + 2 ^ (-11 : ℤ)) :=
          add_le_add hmain hmulabs'
      _ ≤ (1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ) := by
          have hc : (2 : ℝ) ^ (-22 : ℤ) + (1 + 2 ^ (-11 : ℤ)) ≤ 1 + 2 ^ (-10 : ℤ) := by norm_num
          have he : (2 : ℝ) ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ) + MX * (1 + 2 ^ (-11 : ℤ)) =
              ((2 : ℝ) ^ (-22 : ℤ) + (1 + 2 ^ (-11 : ℤ))) * MX + 2 ^ (-150 : ℤ) := by ring
          rw [he]
          have := mul_le_mul_of_nonneg_right hc hMX0
          linarith

/-- A self-referential bound: if `c` is within one rounding of `o` and `o` within `d` of `r`,
then `c` is close to `r`, with the factor `(1-u)` absorbed. -/
theorem merge_row_final {c o r MX u eta d : ℝ}
    (hu0 : 0 ≤ u) (hu : u ≤ 1 / 2) (heta : 0 ≤ eta) (hd : 0 ≤ d)
    (_hMX0 : 0 ≤ MX) (hr : |r| ≤ MX)
    (h1 : |c - o| ≤ |c| * u + eta) (h2 : |o - r| ≤ d) :
    |c - r| ≤ u * |r| + 2 * u ^ 2 * MX + (1 + 2 * u) * (eta + d) := by
  have htri : |c - r| ≤ |c| * u + eta + d := by
    have e : c - r = (c - o) + (o - r) := by ring
    calc |c - r| = |(c - o) + (o - r)| := congrArg (fun z : ℝ => |z|) e
      _ ≤ |c - o| + |o - r| := abs_add_le _ _
      _ ≤ (|c| * u + eta) + d := add_le_add h1 h2
  have hcle : |c| ≤ |r| + |c - r| := by
    have e : c = r + (c - r) := by ring
    calc |c| = |r + (c - r)| := congrArg (fun z : ℝ => |z|) e
      _ ≤ |r| + |c - r| := abs_add_le _ _
  have hcu : |c| * u ≤ u * |r| + |c - r| * u := by
    have h := mul_le_mul_of_nonneg_right hcle hu0
    linarith
  have hE2 : |c - r| * (1 - u) ≤ u * |r| + eta + d := by linarith [htri, hcu]
  have hpos : 0 < 1 - u := by linarith
  have hS : 0 ≤ u * |r| + eta + d := by
    have h1' : 0 ≤ u * |r| := mul_nonneg hu0 (abs_nonneg r)
    linarith
  have hfac : 1 / (1 - u) ≤ 1 + 2 * u := by
    rw [div_le_iff₀ hpos]
    nlinarith [mul_nonneg hu0 (show (0 : ℝ) ≤ 1 - 2 * u by linarith)]
  have hstep : |c - r| ≤ (u * |r| + eta + d) / (1 - u) := by
    rw [le_div_iff₀ hpos]
    linarith [hE2]
  calc |c - r| ≤ (u * |r| + eta + d) / (1 - u) := hstep
    _ = (u * |r| + eta + d) * (1 / (1 - u)) := by ring
    _ ≤ (u * |r| + eta + d) * (1 + 2 * u) := mul_le_mul_of_nonneg_left hfac hS
    _ = u * |r| + 2 * u ^ 2 * |r| + (1 + 2 * u) * (eta + d) := by ring
    _ ≤ u * |r| + 2 * u ^ 2 * MX + (1 + 2 * u) * (eta + d) := by
        have h2u : 2 * u ^ 2 * |r| ≤ 2 * u ^ 2 * MX :=
          mul_le_mul_of_nonneg_left hr (by positivity)
        linarith

/-- A fused addition of two finite products of bounded size is within `2^-21` of the exact
sum (in units of the row magnitude). -/
theorem add_bound {prodp prods out32 : F32} {MX : ℝ}
    (hMX0 : 0 ≤ MX) (hMX : MX ≤ (255 : ℝ) * 2 ^ (120 : ℤ))
    (hf1 : prodp.isFinite = true) (hf2 : prods.isFinite = true)
    (ha1 : |prodp.toReal| ≤ (1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ))
    (ha2 : |prods.toReal| ≤ (1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ))
    (hs : |prodp.toReal + prods.toReal| ≤ (1 + 2 ^ (-14 : ℤ)) * MX + 2 ^ (-148 : ℤ))
    (ha : AddRel true prodp prods out32) :
    out32.isFinite = true ∧
      |out32.toReal - (prodp.toReal + prods.toReal)| ≤
        2 ^ (-21 : ℤ) * MX + 2 ^ (-147 : ℤ) := by
  have hMreal : |prodp.toReal + prods.toReal| +
      (2 : ℝ) ^ (-24 : ℤ) * (|prodp.toReal| + |prods.toReal|) +
      2 * (2 : ℝ) ^ (-150 : ℤ) ≤ ((2 : ℝ) ^ 24 - 1) * (2 : ℝ) ^ (104 : ℤ) := by
    have hsum2 : |prodp.toReal| + |prods.toReal| ≤
        2 * ((1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ)) := by linarith [ha1, ha2]
    have hm : 2 ^ (-24 : ℤ) * (|prodp.toReal| + |prods.toReal|) ≤
        2 ^ (-24 : ℤ) * (2 * ((1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ))) :=
      mul_le_mul_of_nonneg_left hsum2 (by positivity)
    have hexp : |prodp.toReal + prods.toReal| +
        2 ^ (-24 : ℤ) * (|prodp.toReal| + |prods.toReal|) + 2 * 2 ^ (-150 : ℤ) ≤
        ((1 + 2 ^ (-14 : ℤ)) * MX + 2 ^ (-148 : ℤ))
          + 2 ^ (-24 : ℤ) * (2 * ((1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ)))
          + 2 * 2 ^ (-150 : ℤ) := by linarith [hs, hm]
    have hcoef0 : (0 : ℝ) ≤ (1 + 2 ^ (-14 : ℤ)) +
        2 ^ (-24 : ℤ) * (2 * (1 + 2 ^ (-10 : ℤ))) := by positivity
    have h1b : ((1 + 2 ^ (-14 : ℤ)) + 2 ^ (-24 : ℤ) * (2 * (1 + 2 ^ (-10 : ℤ)))) * MX ≤
        ((1 + 2 ^ (-14 : ℤ)) + 2 ^ (-24 : ℤ) * (2 * (1 + 2 ^ (-10 : ℤ)))) *
          (255 * 2 ^ (120 : ℤ)) :=
      mul_le_mul_of_nonneg_left hMX hcoef0
    have h2b : ((1 + 2 ^ (-14 : ℤ)) + 2 ^ (-24 : ℤ) * (2 * (1 + 2 ^ (-10 : ℤ)))) *
          (255 * 2 ^ (120 : ℤ))
        + (2 ^ (-148 : ℤ) + 2 ^ (-24 : ℤ) * (2 * 2 ^ (-150 : ℤ)) + 2 * 2 ^ (-150 : ℤ))
        ≤ ((2 : ℝ) ^ 24 - 1) * 2 ^ (104 : ℤ) := by norm_num
    have hring : ((1 + 2 ^ (-14 : ℤ)) * MX + 2 ^ (-148 : ℤ))
          + 2 ^ (-24 : ℤ) * (2 * ((1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ)))
          + 2 * 2 ^ (-150 : ℤ)
        = ((1 + 2 ^ (-14 : ℤ)) + 2 ^ (-24 : ℤ) * (2 * (1 + 2 ^ (-10 : ℤ)))) * MX
          + (2 ^ (-148 : ℤ) + 2 ^ (-24 : ℤ) * (2 * 2 ^ (-150 : ℤ)) + 2 * 2 ^ (-150 : ℤ)) := by
      ring
    linarith [hexp, hring, h1b, h2b]
  have hM : |prodp.toRat + prods.toRat| +
      Format.binary32.u * (|prodp.toRat| + |prods.toRat|) +
      2 * Format.binary32.eta ≤ Format.binary32.maxVal := by
    apply Rat.cast_le (K := ℝ).mp
    rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
    push_cast
    simpa [FloatV.toReal] using hMreal
  obtain ⟨hfin, herr⟩ := addRel_err_w (f := Format.binary32) Format.binary32_regular
    (by decide : Format.binary32.specials ≠ .finite)
    (by rw [Format.binary32_u]; norm_num : Format.binary32.u ≤ 1 / 16) ha hf1 hf2 hM
  refine ⟨hfin, ?_⟩
  have hcast : ((|out32.toRat - (prodp.toRat + prods.toRat)| : ℚ) : ℝ) ≤
      ((3 * Format.binary32.u * (|prodp.toRat| + |prods.toRat|) +
        4 * Format.binary32.eta : ℚ) : ℝ) :=
    Rat.cast_le.mpr herr
  rw [Format.binary32_u, Format.binary32_eta] at hcast
  push_cast at hcast
  have hbase : |out32.toReal - (prodp.toReal + prods.toReal)| ≤
      3 * 2 ^ (-24 : ℤ) * (|prodp.toReal| + |prods.toReal|) + 4 * 2 ^ (-150 : ℤ) := by
    simpa [FloatV.toReal] using hcast
  have hsum2 : |prodp.toReal| + |prods.toReal| ≤
      2 * ((1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ)) := by linarith [ha1, ha2]
  have hmul : 3 * 2 ^ (-24 : ℤ) * (|prodp.toReal| + |prods.toReal|) ≤
      3 * 2 ^ (-24 : ℤ) * (2 * ((1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ))) :=
    mul_le_mul_of_nonneg_left hsum2 (by positivity)
  have hcoef : 3 * (2 : ℝ) ^ (-24 : ℤ) * (2 * (1 + (2 : ℝ) ^ (-10 : ℤ))) ≤
      (2 : ℝ) ^ (-21 : ℤ) := by norm_num
  have hcoefM := mul_le_mul_of_nonneg_right hcoef hMX0
  have hconst : 3 * (2 : ℝ) ^ (-24 : ℤ) * (2 * (2 : ℝ) ^ (-150 : ℤ)) +
      4 * (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-147 : ℤ) := by norm_num
  have hring : 3 * 2 ^ (-24 : ℤ) * (2 * ((1 + 2 ^ (-10 : ℤ)) * MX + 2 ^ (-150 : ℤ)))
      = (3 * 2 ^ (-24 : ℤ) * (2 * (1 + 2 ^ (-10 : ℤ)))) * MX
        + 3 * 2 ^ (-24 : ℤ) * (2 * 2 ^ (-150 : ℤ)) := by ring
  linarith [hbase, hmul, hring, hcoefM, hconst]

/-- **The row stored by the merge branch.** -/
theorem merge_row_num {P : Platform} (hD : Device.DivFull P)
    {p s M u v pe se o ps ss p32 s32 prodp prods out32 : F32} {pv sv w : BF16}
    (hm : LseMid p s M u v pe se o) (hne : ¬ (p = negInfC ∧ s = negInfC))
    (hpv : pv.isFinite = true) (hsv : sv.isFinite = true)
    (hd1 : P.div .binary32 pe o ps) (hd2 : P.div .binary32 se o ss)
    (hc1 : FloatV.NaNEq (FloatV.cvtWith .rn pv) p32)
    (hc2 : FloatV.NaNEq (FloatV.cvtWith .rn sv) s32)
    (hm1 : FloatV.NaNEq (FloatV.mul p32 ps) prodp)
    (hm2 : FloatV.NaNEq (FloatV.mul s32 ss) prods)
    (ha : AddRel true prodp prods out32)
    (hw : FloatV.NaNEq (FloatV.cvtWith .rn out32) w) :
    w.isFinite = true ∧
      |w.toReal - refOut p s pv.toReal sv.toReal| ≤
        (bf16Round : ℝ) * |refOut p s pv.toReal sv.toReal| +
          (outTol : ℝ) * max |pv.toReal| |sv.toReal| + (absTol : ℝ) := by
  obtain ⟨hp32eq, hp32fin, hp32rat⟩ := cvt_bf16_f32 hpv hc1
  obtain ⟨hs32eq, hs32fin, hs32rat⟩ := cvt_bf16_f32 hsv hc2
  have hp32R : p32.toReal = pv.toReal := by
    show (p32.toRat : ℝ) = (pv.toRat : ℝ)
    rw [hp32rat]
  have hs32R : s32.toReal = sv.toReal := by
    show (s32.toRat : ℝ) = (sv.toRat : ℝ)
    rw [hs32rat]
  set SP := side p (refMax p s) with hSPdef
  set SS := side s (refMax p s) with hSSdef
  set W := refSum p s with hWdef
  have hSP0 : 0 ≤ SP := side_nonneg' p _
  have hSS0 : 0 ≤ SS := side_nonneg' s _
  have hSP1 : SP ≤ 1 := side_le_one_left' p s
  have hSS1 : SS ≤ 1 := side_le_one_right' p s
  have hWsum : W = SP + SS := by rw [hWdef, refSum, ← hSPdef, ← hSSdef]
  have hW1 : 1 ≤ W := by
    have h := one_le_side_sum' (p := p) (s := s) hne
    rw [← hSPdef, ← hSSdef] at h
    rw [hWsum]
    exact h
  have hW2 : W ≤ 2 := by rw [hWsum]; linarith
  have hWo : |W - o.toReal| ≤ 2 ^ (-18 : ℤ) := by
    rw [hWdef]
    exact (abs_sub_comm _ _).trans_le hm.hpo
  have hpeSP : |pe.toReal - SP| ≤ 2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ) := by
    simpa [hSPdef] using hm.hpe
  have hseSS : |se.toReal - SS| ≤ 2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ) := by
    simpa [hSSdef] using hm.hse
  obtain ⟨hpsfin, hpsabs, hpserr⟩ :=
    scale_bound hD hm.hpefin hm.hofin hm.ho0 hpeSP hm.hpe1 hSP0 hSP1 hW1 hW2 hWo hd1
  obtain ⟨hssfin, hssabs, hsserr⟩ :=
    scale_bound hD hm.hsefin hm.hofin hm.ho0 hseSS hm.hse1 hSS0 hSS1 hW1 hW2 hWo hd2
  set MX := max |pv.toReal| |sv.toReal| with hMXdef
  have hMX0 : 0 ≤ MX := by
    rw [hMXdef]
    exact le_trans (abs_nonneg _) (le_max_left _ _)
  have hpvMX : |pv.toReal| ≤ MX := by rw [hMXdef]; exact le_max_left _ _
  have hsvMX : |sv.toReal| ≤ MX := by rw [hMXdef]; exact le_max_right _ _
  have hp32le : |p32.toReal| ≤ MX := by rw [hp32R]; exact hpvMX
  have hs32le : |s32.toReal| ≤ MX := by rw [hs32R]; exact hsvMX
  have hpvmax : |pv.toRat| ≤ Format.bfloat16.maxVal :=
    FloatV.abs_toRat_le_maxVal (f := Format.bfloat16) Format.bfloat16_regular hpv
  have hsvmax : |sv.toRat| ≤ Format.bfloat16.maxVal :=
    FloatV.abs_toRat_le_maxVal (f := Format.bfloat16) Format.bfloat16_regular hsv
  have hp32maxq : |p32.toRat| ≤ Format.bfloat16.maxVal := by rw [hp32rat]; exact hpvmax
  have hs32maxq : |s32.toRat| ≤ Format.bfloat16.maxVal := by rw [hs32rat]; exact hsvmax
  have hbfmax : ((Format.bfloat16.maxVal : ℚ) : ℝ) ≤ (255 : ℝ) * 2 ^ (120 : ℤ) := by
    rw [Format.bfloat16_maxVal]
    push_cast
    norm_num
  have hMXle : MX ≤ (255 : ℝ) * 2 ^ (120 : ℤ) := by
    rw [hMXdef]
    apply max_le
    · show |(pv.toRat : ℝ)| ≤ (255 : ℝ) * 2 ^ (120 : ℤ)
      exact (cast_abs_le' hpvmax).trans hbfmax
    · show |(sv.toRat : ℝ)| ≤ (255 : ℝ) * 2 ^ (120 : ℤ)
      exact (cast_abs_le' hsvmax).trans hbfmax
  obtain ⟨hprodpfin, hprodperr, hprodple⟩ :=
    prod_bound (p32 := p32) (ps := ps) (prodp := prodp) (v := pv.toReal) (MX := MX)
      hp32R hMX0 hp32le hpsabs hp32fin hpsfin hp32maxq hm1
  obtain ⟨hprodsfin, hprodserr, hprodsle⟩ :=
    prod_bound (p32 := s32) (ps := ss) (v := sv.toReal) (MX := MX)
      hs32R hMX0 hs32le hssabs hs32fin hssfin hs32maxq hm2
  have hrefOut : refOut p s pv.toReal sv.toReal =
      pv.toReal * (SP / W) + sv.toReal * (SS / W) := by
    rw [refOut, ← hSPdef, ← hSSdef, ← hWdef]
    field_simp
  have hro' : |refOut p s pv.toReal sv.toReal| ≤ MX := by
    rw [hMXdef]
    exact abs_refOut_le_max' hne _ _
  have hb1 : |pv.toReal * ps.toReal - pv.toReal * (SP / W)| ≤
      MX * (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) := by
    have e : pv.toReal * ps.toReal - pv.toReal * (SP / W) =
        pv.toReal * (ps.toReal - SP / W) := by ring
    rw [e, abs_mul]
    exact mul_le_mul hpvMX hpserr (abs_nonneg _) hMX0
  have hb2 : |sv.toReal * ss.toReal - sv.toReal * (SS / W)| ≤
      MX * (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) := by
    have e : sv.toReal * ss.toReal - sv.toReal * (SS / W) =
        sv.toReal * (ss.toReal - SS / W) := by ring
    rw [e, abs_mul]
    exact mul_le_mul hsvMX hsserr (abs_nonneg _) hMX0
  have hsumErr : |(prodp.toReal + prods.toReal) - refOut p s pv.toReal sv.toReal| ≤
      2 ^ (-14 : ℤ) * MX + 2 ^ (-148 : ℤ) := by
    rw [hrefOut]
    have e : (prodp.toReal + prods.toReal) - (pv.toReal * (SP / W) + sv.toReal * (SS / W)) =
        (prodp.toReal - pv.toReal * (SP / W)) +
          (prods.toReal - sv.toReal * (SS / W)) := by ring
    rw [e]
    have hd1 : |prodp.toReal - pv.toReal * (SP / W)| ≤
        (2 ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ)) +
          MX * (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) := by
      have e2 : prodp.toReal - pv.toReal * (SP / W) =
          (prodp.toReal - pv.toReal * ps.toReal) +
            (pv.toReal * ps.toReal - pv.toReal * (SP / W)) := by ring
      rw [e2]
      exact (abs_add_le _ _).trans (add_le_add hprodperr hb1)
    have hd2 : |prods.toReal - sv.toReal * (SS / W)| ≤
        (2 ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ)) +
          MX * (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) := by
      have e2 : prods.toReal - sv.toReal * (SS / W) =
          (prods.toReal - sv.toReal * ss.toReal) +
            (sv.toReal * ss.toReal - sv.toReal * (SS / W)) := by ring
      rw [e2]
      exact (abs_add_le _ _).trans (add_le_add hprodserr hb2)
    have h4 := abs_add_le (prodp.toReal - pv.toReal * (SP / W))
      (prods.toReal - sv.toReal * (SS / W))
    have hcoef : (2 : ℝ) ^ (-22 : ℤ) + 2 ^ (-22 : ℤ) + (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) +
        (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) ≤ (2 : ℝ) ^ (-14 : ℤ) := by norm_num
    have hcoefM := mul_le_mul_of_nonneg_right hcoef hMX0
    have hconst : (2 : ℝ) ^ (-150 : ℤ) + 2 ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-148 : ℤ) := by norm_num
    have hcomb : (2 ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ)) +
          MX * (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) +
          ((2 ^ (-22 : ℤ) * MX + 2 ^ (-150 : ℤ)) +
            MX * (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)))
        = (2 ^ (-22 : ℤ) + 2 ^ (-22 : ℤ) + (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ)) +
            (2 ^ (-16 : ℤ) + 2 ^ (-20 : ℤ))) * MX +
          (2 ^ (-150 : ℤ) + 2 ^ (-150 : ℤ)) := by ring
    linarith [h4, hd1, hd2, hcoefM, hconst, hcomb]
  have hsumAbs : |prodp.toReal + prods.toReal| ≤
      (1 + 2 ^ (-14 : ℤ)) * MX + 2 ^ (-148 : ℤ) := by
    have e : prodp.toReal + prods.toReal =
        ((prodp.toReal + prods.toReal) - refOut p s pv.toReal sv.toReal) +
          refOut p s pv.toReal sv.toReal := by ring
    calc |prodp.toReal + prods.toReal| =
          |((prodp.toReal + prods.toReal) - refOut p s pv.toReal sv.toReal) +
            refOut p s pv.toReal sv.toReal| := congrArg (fun z : ℝ => |z|) e
      _ ≤ |(prodp.toReal + prods.toReal) - refOut p s pv.toReal sv.toReal| +
            |refOut p s pv.toReal sv.toReal| := abs_add_le _ _
      _ ≤ (2 ^ (-14 : ℤ) * MX + 2 ^ (-148 : ℤ)) + MX := add_le_add hsumErr hro'
      _ = (1 + 2 ^ (-14 : ℤ)) * MX + 2 ^ (-148 : ℤ) := by ring
  obtain ⟨houtfin, houterr⟩ :=
    add_bound hMX0 hMXle hprodpfin hprodsfin hprodple hprodsle hsumAbs ha
  have hDout : |out32.toReal - refOut p s pv.toReal sv.toReal| ≤
      (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ) := by
    have e : out32.toReal - refOut p s pv.toReal sv.toReal =
        (out32.toReal - (prodp.toReal + prods.toReal)) +
          ((prodp.toReal + prods.toReal) - refOut p s pv.toReal sv.toReal) := by ring
    rw [e]
    have h4 := abs_add_le (out32.toReal - (prodp.toReal + prods.toReal))
      ((prodp.toReal + prods.toReal) - refOut p s pv.toReal sv.toReal)
    have hconst : (2 : ℝ) ^ (-147 : ℤ) + 2 ^ (-148 : ℤ) ≤ (2 : ℝ) ^ (-146 : ℤ) := by norm_num
    have hcomb : (2 ^ (-21 : ℤ) * MX + 2 ^ (-147 : ℤ)) +
          (2 ^ (-14 : ℤ) * MX + 2 ^ (-148 : ℤ))
        = (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX +
          (2 ^ (-147 : ℤ) + 2 ^ (-148 : ℤ)) := by ring
    linarith [h4, houterr, hsumErr, hcomb, hconst]
  have houtAbs : |out32.toReal| ≤ (1 + 2 ^ (-13 : ℤ)) * MX + 2 ^ (-146 : ℤ) := by
    have e : out32.toReal =
        (out32.toReal - refOut p s pv.toReal sv.toReal) +
          refOut p s pv.toReal sv.toReal := by ring
    calc |out32.toReal| =
          |(out32.toReal - refOut p s pv.toReal sv.toReal) +
            refOut p s pv.toReal sv.toReal| := congrArg (fun z : ℝ => |z|) e
      _ ≤ |out32.toReal - refOut p s pv.toReal sv.toReal| +
            |refOut p s pv.toReal sv.toReal| := abs_add_le _ _
      _ ≤ ((2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ)) + MX :=
          add_le_add hDout hro'
      _ ≤ (1 + 2 ^ (-13 : ℤ)) * MX + 2 ^ (-146 : ℤ) := by
          have hring : ((2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ)) + MX =
              (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ) + 1) * MX + 2 ^ (-146 : ℤ) := by ring
          have hc : (2 : ℝ) ^ (-14 : ℤ) + 2 ^ (-21 : ℤ) + 1 ≤ 1 + (2 : ℝ) ^ (-13 : ℤ) := by norm_num
          have hm := mul_le_mul_of_nonneg_right hc hMX0
          linarith [hring, hm]
  have hqreal : |out32.toReal| < ((255 : ℝ) + 1 / 2) * 2 ^ (120 : ℤ) := by
    have h1 : (1 + 2 ^ (-13 : ℤ)) * MX ≤
        (1 + 2 ^ (-13 : ℤ)) * ((255 : ℝ) * 2 ^ (120 : ℤ)) :=
      mul_le_mul_of_nonneg_left hMXle (by positivity)
    have h2 : (1 + 2 ^ (-13 : ℤ)) * ((255 : ℝ) * 2 ^ (120 : ℤ)) + 2 ^ (-146 : ℤ) <
        ((255 : ℝ) + 1 / 2) * 2 ^ (120 : ℤ) := by norm_num
    calc |out32.toReal| ≤ (1 + 2 ^ (-13 : ℤ)) * MX + 2 ^ (-146 : ℤ) := houtAbs
      _ ≤ (1 + 2 ^ (-13 : ℤ)) * ((255 : ℝ) * 2 ^ (120 : ℤ)) + 2 ^ (-146 : ℤ) := by
          linarith [h1]
      _ < ((255 : ℝ) + 1 / 2) * 2 ^ (120 : ℤ) := h2
  have hq : |out32.toRat| < (255 + 1 / 2) * 2 ^ (120 : ℤ) := by
    have hcast : ((|out32.toRat| : ℚ) : ℝ) <
        (((255 + 1 / 2 : ℚ) * 2 ^ (120 : ℤ) : ℚ) : ℝ) := by
      have hR : ((((255 + 1 / 2 : ℚ) * 2 ^ (120 : ℤ) : ℚ)) : ℝ) =
          ((255 : ℝ) + 1 / 2) * 2 ^ (120 : ℤ) := by norm_num
      rw [hR, Rat.cast_abs]
      simpa [FloatV.toReal] using hqreal
    exact Rat.cast_lt.mp hcast
  obtain ⟨hcvtfin, hcvterr⟩ := cvtWith_bf16_rn (x := out32) houtfin hq
  have hweq : w = (FloatV.cvtWith .rn out32 : BF16) :=
    hw.eq (FloatV.isFinite_iff.mp hcvtfin).1
  have hcvterrR : |w.toReal - out32.toReal| ≤ |w.toReal| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ) := by
    have hcast : ((|(FloatV.cvtWith .rn out32 : BF16).toRat - out32.toRat| : ℚ) : ℝ) ≤
        ((|(FloatV.cvtWith .rn out32 : BF16).toRat| * 2 ^ (-8 : ℤ) +
          2 ^ (-134 : ℤ) : ℚ) : ℝ) :=
      Rat.cast_le.mpr hcvterr
    push_cast at hcast
    rw [hweq]
    simpa [FloatV.toReal] using hcast
  refine ⟨?_, ?_⟩
  · rw [hweq]
    exact hcvtfin
  · have hbf : (bf16Round : ℝ) = 2 ^ (-8 : ℤ) := by norm_num [bf16Round]
    have hout : (outTol : ℝ) = 2 ^ (-13 : ℤ) := by norm_num [outTol]
    have habs : (absTol : ℝ) = 2 ^ (-133 : ℤ) := by norm_num [absTol]
    have hd0 : (0 : ℝ) ≤ (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ) := by
      have hc : (0 : ℝ) ≤ 2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ) := by positivity
      have h1 : 0 ≤ (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX := mul_nonneg hc hMX0
      have h2 : (0 : ℝ) ≤ 2 ^ (-146 : ℤ) := by positivity
      linarith
    have hmain := merge_row_final (c := w.toReal) (o := out32.toReal)
      (r := refOut p s pv.toReal sv.toReal) (MX := MX)
      (u := 2 ^ (-8 : ℤ)) (eta := 2 ^ (-134 : ℤ))
      (d := (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ))
      (by norm_num) (by norm_num) (by norm_num) hd0 hMX0 hro' hcvterrR hDout
    rw [hbf, hout, habs]
    have hu2 : 2 * ((2 : ℝ) ^ (-8 : ℤ)) ^ 2 = 2 ^ (-15 : ℤ) := by norm_num
    rw [hu2] at hmain
    have hEt : 0 ≤ 2 ^ (-134 : ℤ) +
        ((2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ)) := by
      have hc : (0 : ℝ) ≤ 2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ) := by positivity
      have h1 : 0 ≤ (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX := mul_nonneg hc hMX0
      have h2 : (0 : ℝ) ≤ 2 ^ (-134 : ℤ) := by positivity
      have h3 : (0 : ℝ) ≤ 2 ^ (-146 : ℤ) := by positivity
      linarith
    have hu1 : 1 + 2 * (2 : ℝ) ^ (-8 : ℤ) ≤ 1 + 2 ^ (-7 : ℤ) := by norm_num
    have hstep1 : (1 + 2 * 2 ^ (-8 : ℤ)) *
          (2 ^ (-134 : ℤ) + ((2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ))) ≤
        (1 + 2 ^ (-7 : ℤ)) *
          (2 ^ (-134 : ℤ) + ((2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ))) :=
      mul_le_mul_of_nonneg_right hu1 hEt
    have hexp : (1 + 2 ^ (-7 : ℤ)) *
          (2 ^ (-134 : ℤ) + ((2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) * MX + 2 ^ (-146 : ℤ)))
        = ((1 + 2 ^ (-7 : ℤ)) * (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ))) * MX +
          (1 + 2 ^ (-7 : ℤ)) * (2 ^ (-146 : ℤ) + 2 ^ (-134 : ℤ)) := by ring
    have hcoef : (2 : ℝ) ^ (-15 : ℤ) + (1 + 2 ^ (-7 : ℤ)) * (2 ^ (-14 : ℤ) + 2 ^ (-21 : ℤ)) ≤
        2 ^ (-13 : ℤ) := by norm_num
    have hcoefM := mul_le_mul_of_nonneg_right hcoef hMX0
    have hconst : (1 + (2 : ℝ) ^ (-7 : ℤ)) * (2 ^ (-146 : ℤ) + 2 ^ (-134 : ℤ)) ≤
        2 ^ (-133 : ℤ) := by norm_num
    linarith [hmain, hstep1, hexp, hcoefM, hconst]

end Claims.MergeAttnStates
