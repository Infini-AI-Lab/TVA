import TVA.Proof
import Proof.NumRstd

/-!
# `layer_norm_gated_fwd_bf16`: the constants and the `rstd` chain

The `float32` constant `1` of `1 / sqrt(…)` and the `bfloat16` `+0` of `other=0.0`; the device's
`float32` copy of a host `eps ≥ 2^-125` stays at least `2^-125` (`hostF32_ge`); and the `rstd`
chain `fl(1 / sqrt(fl(s1 + eps)))` within `2^-12 − 2^-16` of `(√(A + eps))⁻¹` when the variance
`s1` is within `2^-13·A + (3/4)·2^-137` of `A` (`rstd_err_wide`), with the range of `s2` and the
room of `tl.sqrt` (`s2_range`, `sqrt_room`, `q_low`).
-/

set_option maxHeartbeats 2000000

namespace Claims.LayerNormGatedFwd

open TVA
open TVA TVA.Format

/-! ### The literal numerator `1.0` -/

/-- The `float64` pattern `0x3FF0000000000000` is `1`. -/
theorem b64_toRat_one : Format.binary64.toRat 0x3FF0000000000000 = 1 := by
  rw [Format.toRat_eq, Format.magVal_eq]
  simp only [Format.sign, Format.mag, Format.signBit, Format.bias, Format.emin, Format.binary64]
  norm_num

/-- The `float32` pattern `0x3F800000` is `1`. -/
theorem b32_toRat_one : Format.binary32.toRat 0x3F800000 = 1 := by
  rw [Format.toRat_eq, Format.magVal_eq]
  simp only [Format.sign, Format.mag, Format.signBit, Format.bias, Format.emin, Format.binary32]
  norm_num

/-- The numerator `1` of `1 / sqrt(…)`, as a `float32` constant, is exactly `1`. -/
theorem litF32_one_toRat : (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) : F32).toRat = 1 := by
  have hlt64 : 0x3FF0000000000000 < 2 ^ Format.binary64.width := by
    norm_num [Format.width, Format.binary64]
  have hz : (FloatV.ofPat 0x3FF0000000000000 : F64).isZero = false := by decide
  have hnan : (FloatV.ofPat 0x3FF0000000000000 : F64).isNaN = false := by decide
  have hfin64 : (FloatV.ofPat 0x3FF0000000000000 : F64).isFinite = true := by decide
  have hR : Format.binary64.toRat (FloatV.ofPat 0x3FF0000000000000 : F64).pat = 1 := by
    rw [FloatV.pat_ofPat hlt64]; exact b64_toRat_one
  have hsign : Format.binary64.sign (FloatV.ofPat 0x3FF0000000000000 : F64).pat = false := by
    rw [FloatV.pat_ofPat hlt64]; decide
  rw [FloatV.litF32, ite_eq_right_of_eq_false _ _ (eq_false (by simp [hz])), hostF32_eq_cvt hnan]
  unfold FloatV.cvt
  rw [Format.cvt_of_finite Format.binary32 .rn false hfin64, hsign, hR]
  obtain ⟨hlt, -, hval⟩ :=
    Format.roundRat_spec (f := Format.binary32) Format.binary32_regular .rn false false
      (by norm_num : (1 : ℚ) ≠ 0) (by rw [abs_one]; rw [Format.binary32_maxVal]; norm_num)
  rw [FloatV.toRat_ofPat hlt, hval]
  have hqexp : Format.binary32.qexp 1 = -23 := by
    rw [Format.qexp_eq (by norm_num : (0 : ℚ) < 1), Int.log_one_right]
    norm_num [Format.binary32, Format.emin, Format.bias]
  have hK : (1 : ℚ) * (2 : ℚ) ^ (-(-23 : ℤ)) = ((2 ^ 23 : ℕ) : ℚ) := by norm_num
  have hsig := Format.roundSig_exact (rnd := .rn) (a := (1 : ℚ)) (Q := -23) (K := 2 ^ 23) hK
  have habs : |(1 : ℚ)| = 1 := abs_one
  rw [habs, hqexp, ite_eq_right_of_eq_false _ _ (eq_false (by norm_num : ¬ (1 : ℚ) < 0)), one_mul]
  exact hsig

/-- The `float32` constant `1` is finite. -/
theorem litF32_one_finite :
    (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) : F32).isFinite = true := by
  have hlt64 : 0x3FF0000000000000 < 2 ^ Format.binary64.width := by
    norm_num [Format.width, Format.binary64]
  have hz : (FloatV.ofPat 0x3FF0000000000000 : F64).isZero = false := by decide
  have hnan : (FloatV.ofPat 0x3FF0000000000000 : F64).isNaN = false := by decide
  have hfin64 : (FloatV.ofPat 0x3FF0000000000000 : F64).isFinite = true := by decide
  have hR : Format.binary64.toRat (FloatV.ofPat 0x3FF0000000000000 : F64).pat = 1 := by
    rw [FloatV.pat_ofPat hlt64]; exact b64_toRat_one
  have hsign : Format.binary64.sign (FloatV.ofPat 0x3FF0000000000000 : F64).pat = false := by
    rw [FloatV.pat_ofPat hlt64]; decide
  rw [FloatV.litF32, ite_eq_right_of_eq_false _ _ (eq_false (by simp [hz])), hostF32_eq_cvt hnan]
  unfold FloatV.cvt
  rw [Format.cvt_of_finite Format.binary32 .rn false hfin64, hsign, hR]
  obtain ⟨hlt, hfin, -⟩ :=
    Format.roundRat_spec (f := Format.binary32) Format.binary32_regular .rn false false
      (by norm_num : (1 : ℚ) ≠ 0) (by rw [abs_one]; rw [Format.binary32_maxVal]; norm_num)
  rw [FloatV.isFinite_ofPat hlt]
  exact hfin

/-! ### The masked-off lane's literal `+0` -/

/-- The `other = 0.0` of the masked load is `+0` in `bfloat16`. -/
theorem litBF16_zero : FloatV.litBF16 (FloatV.ofPat 0) = (FloatV.ofPat 0 : BF16) := by
  have hz : (FloatV.ofPat (0 : ℕ) : F64).isZero = true := by decide
  rw [FloatV.litBF16, ite_eq_left_of_eq_true _ _ (eq_true hz)]

/-- The `bfloat16` constant `0` is finite. -/
theorem litBF16_zero_finite : (FloatV.litBF16 (FloatV.ofPat 0) : BF16).isFinite = true := by
  rw [litBF16_zero, FloatV.isFinite_ofPat (by norm_num [Format.width, Format.bfloat16])]
  decide

/-- The `bfloat16` constant `0` has real value `0`. -/
theorem litBF16_zero_toReal : (FloatV.litBF16 (FloatV.ofPat 0) : BF16).toReal = 0 := by
  rw [litBF16_zero]
  simp only [FloatV.toReal]
  have hz : (FloatV.ofPat (0 : ℕ) : BF16).isZero = true := by decide
  rw [FloatV.toRat_of_isZero hz]
  norm_num

/-- The `bfloat16` constant `0` has rational value `0`. -/
theorem litBF16_zero_toRat : (FloatV.litBF16 (FloatV.ofPat 0) : BF16).toRat = 0 := by
  have h : (((FloatV.litBF16 (FloatV.ofPat 0) : BF16).toRat : ℚ) : ℝ) = 0 := by
    simpa only [FloatV.toReal] using litBF16_zero_toReal
  exact_mod_cast h

/-- The `bfloat16` constant `0` is not NaN. -/
theorem litBF16_zero_isNaN : (FloatV.litBF16 (FloatV.ofPat 0) : BF16).isNaN = false := by
  rw [litBF16_zero, FloatV.isNaN_ofPat (by norm_num [Format.width, Format.bfloat16])]
  decide

/-! ### The value `2^-125` survives the round trip through `f32`, for small arguments -/

/-- Rounding a rational in `[2^-125, maxVal]` to `float32` stays at least `2^-125`. -/
theorem roundRat_rn_ge (sat zs : Bool) {q : ℚ} (hq0 : 0 < q)
    (hlo : (2 : ℚ) ^ (-125 : ℤ) ≤ q) (hhi : q ≤ Format.binary32.maxVal) :
    (2 : ℚ) ^ (-125 : ℤ) ≤ Format.binary32.toRat (Format.binary32.roundRat .rn sat zs q) := by
  have hqne : q ≠ 0 := ne_of_gt hq0
  obtain ⟨hlt, hfin, hval⟩ :=
    Format.roundRat_spec (f := Format.binary32) Format.binary32_regular .rn sat zs hqne
      (by rw [abs_of_pos hq0]; exact hhi)
  have hsign : ¬ (q < 0) := not_lt.mpr (le_of_lt hq0)
  rw [hval]
  simp only [hsign, ite_false, one_mul, abs_of_pos hq0]
  have hlog : (-125 : ℤ) ≤ Int.log 2 q := Format.le_log_of_le hq0 hlo
  have hemin : Format.binary32.emin ≤ Int.log 2 q := by
    rw [Format.binary32_emin]; linarith
  have hsig : (2 ^ Format.binary32.m : ℕ) ≤ Format.roundSig .rn q (Format.binary32.qexp q) :=
    Format.roundSig_ge .rn hq0 hemin
  have hsig' : (2 : ℚ) ^ (23 : ℤ) ≤ ((Format.roundSig .rn q (Format.binary32.qexp q) : ℕ) : ℚ) := by
    have h2 : (2 : ℚ) ^ (23 : ℤ) = ((2 ^ Format.binary32.m : ℕ) : ℚ) := by
      norm_num [Format.binary32]
    rw [h2]
    exact_mod_cast hsig
  have hQ : (-148 : ℤ) ≤ Format.binary32.qexp q := by
    rw [Format.qexp_eq hq0]
    have hm : (Format.binary32.m : ℤ) = 23 := by decide
    have h1 : Int.log 2 q ≤ max (Int.log 2 q) Format.binary32.emin := le_max_left _ _
    omega
  have hpow : (2 : ℚ) ^ (-148 : ℤ) ≤ (2 : ℚ) ^ (Format.binary32.qexp q) :=
    zpow_le_zpow_right₀ (by norm_num) hQ
  have hmul := mul_le_mul hsig' hpow (by positivity) (by positivity)
  calc (2 : ℚ) ^ (-125 : ℤ) = (2 : ℚ) ^ (23 : ℤ) * (2 : ℚ) ^ (-148 : ℤ) := by
        rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
        norm_num
    _ ≤ _ := hmul

/-- The device's `float32` value of a host `eps ≥ 2^-125` is at least `2^-125`. -/
theorem hostF32_ge {c : F64} (hfin : c.isFinite = true)
    (hlo : (2 : ℚ) ^ (-125 : ℤ) ≤ c.toRat) (hhi : c.toRat ≤ (2 : ℚ) ^ (126 : ℤ)) :
    (2 : ℝ) ^ (-125 : ℤ) ≤ (FloatV.hostF32 c).toReal := by
  have hnan : c.isNaN = false := Format.not_isNaN_of_isFinite hfin
  rw [hostF32_eq_cvt hnan]
  have hq0 : 0 < c.toRat := lt_of_lt_of_le (by norm_num) hlo
  have hhi' : c.toRat ≤ Format.binary32.maxVal := by
    refine hhi.trans ?_
    rw [Format.binary32_maxVal]; norm_num
  have hlt : Format.binary32.roundRat .rn false (Format.binary64.sign c.pat)
      (Format.binary64.toRat c.pat) < 2 ^ Format.binary32.width :=
    (Format.roundRat_spec (f := Format.binary32) Format.binary32_regular .rn false _
      (ne_of_gt hq0) (by rw [abs_of_pos hq0]; exact hhi')).1
  have hcvt : (FloatV.cvt c : F32).toRat
      = Format.binary32.toRat (Format.binary32.roundRat .rn false (Format.binary64.sign c.pat)
          (Format.binary64.toRat c.pat)) := by
    unfold FloatV.cvt
    rw [Format.cvt_of_finite Format.binary32 .rn false hfin]
    exact FloatV.toRat_ofPat hlt
  simp only [FloatV.toReal]
  rw [hcvt]
  have hge := roundRat_rn_ge false (Format.binary64.sign c.pat) hq0 hlo hhi'
  have hgeR := Rat.cast_le (K := ℝ).mpr hge
  push_cast at hgeR
  exact hgeR

/-! ### `rstd_err` with room up to `2^127` (the `f32` copy of `eps` may exceed `2^126`) -/

/-- `NumRstd.rstd_err` with the weaker upper bound `A + eps ≤ 2^127` (the room the kernel's `float32` sums have). -/
theorem rstd_err_wide {A e s1 s2 q r : ℝ}
    (he0 : 0 ≤ e) (hA0 : 0 ≤ A) (he : (2 : ℝ) ^ (-125 : ℤ) ≤ e)
    (hVhi : A + e ≤ (2 : ℝ) ^ (127 : ℤ))
    (h1 : |s1 - A| ≤ (2 : ℝ) ^ (-13 : ℤ) * A + (3/4) * (2 : ℝ) ^ (-137 : ℤ))
    (h2 : |s2 - (s1 + e)| ≤ 3 * (2 : ℝ) ^ (-24 : ℤ) * (|s1| + e) + 4 * (2 : ℝ) ^ (-150 : ℤ))
    (hq0 : 0 < q)
    (h3 : |q - Real.sqrt s2| ≤ (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 + (2 : ℝ) ^ (-149 : ℤ))
    (h4 : |r - 1 / q| ≤ (2 : ℝ) ^ (-22 : ℤ) * |1 / q| + (2 : ℝ) ^ (-148 : ℤ)) :
    |r - (Real.sqrt (A + e))⁻¹|
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * (Real.sqrt (A + e))⁻¹ := by
  obtain ⟨V, hVdef⟩ : ∃ V : ℝ, V = A + e := ⟨_, rfl⟩
  obtain ⟨u, hudef⟩ : ∃ u : ℝ, u = (2 : ℝ) ^ (-24 : ℤ) := ⟨_, rfl⟩
  obtain ⟨u3, hu3def⟩ : ∃ u3 : ℝ, u3 = (2 : ℝ) ^ (-22 : ℤ) := ⟨_, rfl⟩
  obtain ⟨c, hcdef⟩ : ∃ c : ℝ, c = (3/4) * (2 : ℝ) ^ (-137 : ℤ) := ⟨_, rfl⟩
  have hu3 : 0 < u3 := by rw [hu3def]; norm_num
  have hu34 : u3 ≤ 1 / 4 := by rw [hu3def]; norm_num
  obtain ⟨κ, hκdef⟩ : ∃ κ : ℝ, κ = (2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) :=
    ⟨_, rfl⟩
  have hu : 0 < u := by rw [hudef]; norm_num
  have hu4 : u ≤ 1 / 4 := by rw [hudef]; norm_num
  have hcpos : 0 < c := by rw [hcdef]; norm_num
  have hκ0 : 0 ≤ κ := by rw [hκdef]; norm_num
  have hκ1 : κ ≤ 1 / 2 := by rw [hκdef]; norm_num
  have hVpos : 0 < V := by
    have h0 : (0 : ℝ) < (2 : ℝ) ^ (-125 : ℤ) := by norm_num
    rw [hVdef]; linarith
  have hVlow : (2 : ℝ) ^ (-125 : ℤ) ≤ V := by rw [hVdef]; linarith
  have hVhigh : V ≤ (2 : ℝ) ^ (127 : ℤ) := by rw [hVdef]; exact hVhi
  have hAleV : A ≤ V := by rw [hVdef]; linarith
  have heV : e ≤ V := by rw [hVdef]; linarith
  have hs1 : |s1| ≤ A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + c := by
    have hle := (abs_le.1 h1).2
    have habs : |s1| ≤ A + |s1 - A| := by
      have h := abs_add_le A (s1 - A)
      rw [abs_of_nonneg hA0] at h
      have heq : A + (s1 - A) = s1 := by ring
      rwa [heq] at h
    rw [hcdef]
    nlinarith [habs, hle]
  have hs2 : |s2 - V| ≤ A * ((2 : ℝ) ^ (-13 : ℤ) + 3 * u * (1 + (2 : ℝ) ^ (-13 : ℤ)))
      + 3 * u * e + c * (1 + 3 * u) + 4 * (2 : ℝ) ^ (-150 : ℤ) := by
    have hVA : (s1 + e) - V = s1 - A := by rw [hVdef]; ring
    have htri : |s2 - V| ≤ |s2 - (s1 + e)| + |(s1 + e) - V| := abs_sub_le s2 (s1 + e) V
    rw [hVA] at htri
    have h1' : |s1 - A| ≤ (2 : ℝ) ^ (-13 : ℤ) * A + c := by rw [hcdef]; exact h1
    have h2' : |s2 - (s1 + e)| ≤ 3 * u * (|s1| + e) + 4 * (2 : ℝ) ^ (-150 : ℤ) := by
      rw [hudef]; exact h2
    have hbase : |s2 - V| ≤ 3 * u * (|s1| + e) + 4 * (2 : ℝ) ^ (-150 : ℤ)
        + ((2 : ℝ) ^ (-13 : ℤ) * A + c) := by linarith [htri, h2', h1']
    have hmul : 3 * u * |s1| ≤ 3 * u * (A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + c) :=
      mul_le_mul_of_nonneg_left hs1 (by positivity)
    nlinarith [hbase, hmul]
  have hδ : |s2 - V| ≤ κ * V := by
    have hAcoef : (2 : ℝ) ^ (-13 : ℤ) + 3 * u * (1 + (2 : ℝ) ^ (-13 : ℤ))
        ≤ (2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ) := by rw [hudef]; norm_num
    have h12 : A * ((2 : ℝ) ^ (-13 : ℤ) + 3 * u * (1 + (2 : ℝ) ^ (-13 : ℤ)))
        ≤ A * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ)) :=
      mul_le_mul_of_nonneg_left hAcoef hA0
    have hAV : A * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ))
        ≤ V * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ)) :=
      mul_le_mul_of_nonneg_right hAleV (by positivity)
    have h3u : 3 * u * e ≤ (2 : ℝ) ^ (-22 : ℤ) * V := by
      have hcoef : 3 * u ≤ (2 : ℝ) ^ (-22 : ℤ) := by rw [hudef]; norm_num
      have h1 : 3 * u * e ≤ (2 : ℝ) ^ (-22 : ℤ) * e := by nlinarith [hcoef, he0]
      have h2 : (2 : ℝ) ^ (-22 : ℤ) * e ≤ (2 : ℝ) ^ (-22 : ℤ) * V :=
        mul_le_mul_of_nonneg_left heV (by norm_num)
      linarith
    have hc1 : c * (1 + 3 * u) + 4 * (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-12 : ℤ) * V := by
      have h1 : c * (1 + 3 * u) + 4 * (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-137 : ℤ) := by
        rw [hcdef, hudef]; norm_num
      have h2 : (2 : ℝ) ^ (-137 : ℤ) = (2 : ℝ) ^ (-12 : ℤ) * (2 : ℝ) ^ (-125 : ℤ) := by norm_num
      have h3 : (2 : ℝ) ^ (-12 : ℤ) * (2 : ℝ) ^ (-125 : ℤ) ≤ (2 : ℝ) ^ (-12 : ℤ) * V :=
        mul_le_mul_of_nonneg_left hVlow (by norm_num)
      linarith
    have hsum : A * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ))
        + (2 : ℝ) ^ (-22 : ℤ) * V + (2 : ℝ) ^ (-12 : ℤ) * V ≤ κ * V := by
      rw [hκdef]
      have h4 : (2 : ℝ) ^ (-22 : ℤ) * V + (2 : ℝ) ^ (-22 : ℤ) * V = (2 : ℝ) ^ (-21 : ℤ) * V := by
        have h5 : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-22 : ℤ) = (2 : ℝ) ^ (-21 : ℤ) := by norm_num
        nlinarith
      linarith [hAV, h4]
    linarith [hs2, h12, hAV, h3u, hc1, hsum]
  have hs2low : V * (1 - κ) ≤ s2 := by
    have h := (abs_le.1 hδ).1; linarith
  have hs2pos : 0 < s2 := by
    have hκV : V * κ ≤ V * (1 / 2) := mul_le_mul_of_nonneg_left hκ1 hVpos.le
    nlinarith [hs2low, hκV, hVpos]
  have hVsqrtpos : 0 < Real.sqrt V := Real.sqrt_pos.2 hVpos
  have hVsqrt_low : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt V := by
    have h1 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 ≤ V := by
      have h2 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = (2 : ℝ) ^ (-126 : ℤ) := by norm_num
      have h3 : (2 : ℝ) ^ (-126 : ℤ) ≤ (2 : ℝ) ^ (-125 : ℤ) := by norm_num
      rw [h2]; linarith
    exact (Real.le_sqrt (by norm_num) hVpos.le).2 h1
  have hVsqrt_high : Real.sqrt V ≤ (2 : ℝ) ^ (64 : ℤ) := by
    have h1 : V ≤ ((2 : ℝ) ^ (64 : ℤ)) ^ 2 := by
      have h2 : ((2 : ℝ) ^ (64 : ℤ)) ^ 2 = (2 : ℝ) ^ (128 : ℤ) := by norm_num
      rw [h2]
      have h3 : (2 : ℝ) ^ (127 : ℤ) ≤ (2 : ℝ) ^ (128 : ℤ) := by norm_num
      linarith
    exact Real.sqrt_le_iff.2 ⟨by norm_num, h1⟩
  have hs2sqrt_low : Real.sqrt V * (1 - κ) ≤ Real.sqrt s2 := by
    have h1 : (Real.sqrt V * (1 - κ)) ^ 2 ≤ s2 := by
      have h2 : (Real.sqrt V * (1 - κ)) ^ 2 = V * (1 - κ) ^ 2 := by
        rw [mul_pow, Real.sq_sqrt hVpos.le]
      rw [h2]
      have hfac : (1 - κ) ^ 2 ≤ 1 - κ := by nlinarith
      have hmul : V * (1 - κ) ^ 2 ≤ V * (1 - κ) := mul_le_mul_of_nonneg_left hfac hVpos.le
      linarith [hmul, hs2low]
    have h0 : 0 ≤ Real.sqrt V * (1 - κ) := by
      have : 0 < 1 - κ := by linarith
      positivity
    exact (Real.le_sqrt h0 hs2pos.le).2 h1
  have hinv_sqrt : |(Real.sqrt s2)⁻¹ - (Real.sqrt V)⁻¹|
      ≤ κ / ((2 - κ) * (1 - κ)) * (Real.sqrt V)⁻¹ :=
    inv_sqrt_abs_sub_le hs2pos.le hVpos hκ0 hκ1 hδ
  have hs2sqrt_ge : (2 : ℝ) ^ (-64 : ℤ) ≤ Real.sqrt s2 := by
    have hA : (2 : ℝ) ^ (-64 : ℤ) = (2 : ℝ) ^ (-63 : ℤ) * (1 / 2) := by norm_num
    have hB : (2 : ℝ) ^ (-63 : ℤ) * (1 / 2) ≤ Real.sqrt V * (1 / 2) :=
      mul_le_mul_of_nonneg_right hVsqrt_low (by norm_num)
    have hC : Real.sqrt V * (1 / 2) ≤ Real.sqrt V * (1 - κ) :=
      mul_le_mul_of_nonneg_left (by linarith : (1 : ℝ) / 2 ≤ 1 - κ) hVsqrtpos.le
    exact le_trans (hA.trans_le hB) (le_trans hC hs2sqrt_low)
  have hb : (2 : ℝ) ^ (-149 : ℤ) ≤ u3 * Real.sqrt s2 := by
    have h1 : u3 * (2 : ℝ) ^ (-64 : ℤ) ≤ u3 * Real.sqrt s2 :=
      mul_le_mul_of_nonneg_left hs2sqrt_ge hu3.le
    have h2 : (2 : ℝ) ^ (-149 : ℤ) ≤ u3 * (2 : ℝ) ^ (-64 : ℤ) := by rw [hu3def]; norm_num
    linarith
  have h3' : |q - Real.sqrt s2| ≤ u3 * Real.sqrt s2 + (2 : ℝ) ^ (-149 : ℤ) := by
    rw [hu3def]; exact h3
  have hqinv : |1 / q - (Real.sqrt s2)⁻¹| ≤ 4 * u3 * (Real.sqrt s2)⁻¹ :=
    inv_q_abs_sub_le hs2pos.le hq0 hu3 hu34 (by norm_num) hb h3'
  have hs2sqrtinv_le : (Real.sqrt s2)⁻¹ ≤ 2 * (Real.sqrt V)⁻¹ := by
    have h2 : Real.sqrt V * (1 / 2) ≤ Real.sqrt s2 := by nlinarith [hs2sqrt_low, hκ1]
    have hpos : (0 : ℝ) < Real.sqrt V * (1 / 2) := by positivity
    have h := one_div_le_one_div_of_le hpos h2
    rw [one_div, one_div] at h
    have hR : (Real.sqrt V * (1 / 2))⁻¹ = 2 * (Real.sqrt V)⁻¹ := by
      rw [mul_inv, mul_comm]; norm_num
    rwa [hR] at h
  have hqinv' : |1 / q - (Real.sqrt s2)⁻¹| ≤ 8 * u3 * (Real.sqrt V)⁻¹ := by
    have h1 : 4 * u3 * (Real.sqrt s2)⁻¹ ≤ 4 * u3 * (2 * (Real.sqrt V)⁻¹) :=
      mul_le_mul_of_nonneg_left hs2sqrtinv_le (by positivity)
    nlinarith [hqinv, h1]
  have hq_low : Real.sqrt s2 * (1 - 2 * u3) ≤ q := by
    have h1 := (abs_le.1 h3').1
    nlinarith [h1, hb]
  have hqinv_abs : |1 / q| ≤ 4 * (Real.sqrt V)⁻¹ := by
    have hqge : Real.sqrt s2 * (1 / 2) ≤ q := by nlinarith [hq_low, hu34]
    have hpos : (0 : ℝ) < Real.sqrt s2 * (1 / 2) := by
      have : 0 < Real.sqrt s2 := Real.sqrt_pos.2 hs2pos
      positivity
    have h1 : (1 : ℝ) / q ≤ 1 / (Real.sqrt s2 * (1 / 2)) := one_div_le_one_div_of_le hpos hqge
    have h2 : (1 : ℝ) / (Real.sqrt s2 * (1 / 2)) = 2 * (Real.sqrt s2)⁻¹ := by
      rw [one_div, mul_inv, mul_comm]; norm_num
    have h3 : 2 * (Real.sqrt s2)⁻¹ ≤ 4 * (Real.sqrt V)⁻¹ := by nlinarith [hs2sqrtinv_le]
    rw [abs_of_pos (by positivity : (0 : ℝ) < 1 / q)]
    linarith [h1, h2, h3]
  have hr : |r - 1 / q| ≤ 4 * u3 * (Real.sqrt V)⁻¹ + (2 : ℝ) ^ (-22 : ℤ) * (Real.sqrt V)⁻¹ := by
    have h4' : |r - 1 / q| ≤ u3 * |1 / q| + (2 : ℝ) ^ (-148 : ℤ) := by rw [hu3def]; exact h4
    have h1 : u3 * |1 / q| ≤ u3 * (4 * (Real.sqrt V)⁻¹) :=
      mul_le_mul_of_nonneg_left hqinv_abs hu3.le
    have h2 : (2 : ℝ) ^ (-148 : ℤ) ≤ (2 : ℝ) ^ (-22 : ℤ) * (Real.sqrt V)⁻¹ := by
      have h3 : (2 : ℝ) ^ (-148 : ℤ) * Real.sqrt V ≤ (2 : ℝ) ^ (-22 : ℤ) := by
        have h4 : (2 : ℝ) ^ (-148 : ℤ) * Real.sqrt V ≤ (2 : ℝ) ^ (-148 : ℤ) * (2 : ℝ) ^ (64 : ℤ) :=
          mul_le_mul_of_nonneg_left hVsqrt_high (by norm_num)
        have h5 : (2 : ℝ) ^ (-148 : ℤ) * (2 : ℝ) ^ (64 : ℤ) ≤ (2 : ℝ) ^ (-22 : ℤ) := by norm_num
        linarith
      have hdiv := (le_div_iff₀ hVsqrtpos).2 h3
      rwa [div_eq_mul_inv] at hdiv
    nlinarith [h4', h1, h2]
  have htri : |r - (Real.sqrt V)⁻¹|
      ≤ |r - 1 / q| + |1 / q - (Real.sqrt s2)⁻¹| + |(Real.sqrt s2)⁻¹ - (Real.sqrt V)⁻¹| := by
    have h1 : |r - (Real.sqrt V)⁻¹| ≤ |r - 1 / q| + |1 / q - (Real.sqrt V)⁻¹| :=
      abs_sub_le r (1 / q) ((Real.sqrt V)⁻¹)
    have h2 : |1 / q - (Real.sqrt V)⁻¹|
        ≤ |1 / q - (Real.sqrt s2)⁻¹| + |(Real.sqrt s2)⁻¹ - (Real.sqrt V)⁻¹| :=
      abs_sub_le (1 / q) ((Real.sqrt s2)⁻¹) ((Real.sqrt V)⁻¹)
    linarith
  have hnum : κ / ((2 - κ) * (1 - κ)) + 4 * u3 + (2 : ℝ) ^ (-22 : ℤ) + 8 * u3
      ≤ (2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) := by
    rw [hκdef, hu3def]; norm_num
  have hsum : |r - (Real.sqrt V)⁻¹|
      ≤ (κ / ((2 - κ) * (1 - κ)) + 4 * u3 + (2 : ℝ) ^ (-22 : ℤ) + 8 * u3) * (Real.sqrt V)⁻¹ := by
    nlinarith [htri, hr, hqinv', hinv_sqrt]
  have hlast : (κ / ((2 - κ) * (1 - κ)) + 4 * u3 + (2 : ℝ) ^ (-22 : ℤ) + 8 * u3) * (Real.sqrt V)⁻¹
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * (Real.sqrt V)⁻¹ :=
    mul_le_mul_of_nonneg_right hnum (by positivity)
  have hres : |r - (Real.sqrt V)⁻¹|
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * (Real.sqrt V)⁻¹ := le_trans hsum hlast
  rw [hVdef] at hres
  exact hres

/-- The row's two sums: `s1` (the row sum) and `s2 = fl(s1 + eps)` (the row sum plus the
regulariser), with the `float32` facts about `eps`. -/
theorem cast_abs_le {q c : ℚ} (h : |q| ≤ c) : |(q : ℝ)| ≤ (c : ℝ) := by
  have h' := Rat.cast_le (K := ℝ).mpr h
  rwa [Rat.cast_abs] at h'

/-- An inequality between casts of rationals gives the inequality of the rationals. -/
theorem rat_le_of_cast {a b : ℚ} (h : (a : ℝ) ≤ (b : ℝ)) : a ≤ b :=
  (Rat.cast_le (K := ℝ)).mp h

/-- `s2 = fl(s1 + eps32)` is in `float32`'s normal range: `2^-126 ≤ s2 ≤ 2^127`. -/
theorem s2_range {A e s1 e32 s2 : ℝ} (hA0 : 0 ≤ A) (he0 : 0 ≤ e)
    (he : (2 : ℝ) ^ (-125 : ℤ) ≤ e) (hAe : A + e ≤ (2 : ℝ) ^ (126 : ℤ))
    (he32pos : 0 ≤ e32)
    (h1 : |s1 - A| ≤ (2 : ℝ) ^ (-13 : ℤ) * A + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
    (he32 : |e32 - e| ≤ (2 : ℝ) ^ (-24 : ℤ) * e + (2 : ℝ) ^ (-150 : ℤ))
    (h2 : |s2 - (s1 + e32)|
      ≤ 3 * (2 : ℝ) ^ (-24 : ℤ) * (|s1| + e32) + 4 * (2 : ℝ) ^ (-150 : ℤ)) :
    (2 : ℝ) ^ (-126 : ℤ) ≤ s2 ∧ s2 ≤ (2 : ℝ) ^ (127 : ℤ) := by
  obtain ⟨sa, hsa⟩ : ∃ sa : ℝ, sa = |s1| := ⟨_, rfl⟩
  have hsa0 : 0 ≤ sa := by rw [hsa]; exact abs_nonneg s1
  have h1l := (abs_le.1 h1).1
  have h1r := (abs_le.1 h1).2
  have he32l := (abs_le.1 he32).1
  have he32r := (abs_le.1 he32).2
  have h2sa := h2
  rw [← hsa] at h2sa
  clear h1 he32 h2
  have h2l := (abs_le.1 h2sa).1
  have h2r := (abs_le.1 h2sa).2
  clear h2sa
  have hs1_ub : sa ≤ A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
    rw [hsa, abs_le]
    constructor <;> linarith
  clear hsa
  have hs1_lb : A * (1 - (2 : ℝ) ^ (-13 : ℤ)) - (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) ≤ s1 := by
    linarith
  have he32_ub : e32 ≤ e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ) := by linarith
  have he32_lb : e * (1 - (2 : ℝ) ^ (-24 : ℤ)) - (2 : ℝ) ^ (-150 : ℤ) ≤ e32 := by linarith
  have hS_ub : sa + e32 ≤ A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ)
      + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)) :=
    add_le_add hs1_ub he32_ub
  have h3u : (0 : ℝ) ≤ 3 * (2 : ℝ) ^ (-24 : ℤ) := by norm_num
  have hmul := mul_le_mul_of_nonneg_left hS_ub h3u
  have hlow : (s1 + e32)
      - (3 * (2 : ℝ) ^ (-24 : ℤ) * (sa + e32) + 4 * (2 : ℝ) ^ (-150 : ℤ)) ≤ s2 := by
    linarith
  have hup : s2 ≤ (s1 + e32)
      + (3 * (2 : ℝ) ^ (-24 : ℤ) * (sa + e32) + 4 * (2 : ℝ) ^ (-150 : ℤ)) := by
    linarith
  have hα : (0 : ℝ) ≤ 1 - (2 : ℝ) ^ (-13 : ℤ)
      - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ)) := by norm_num [zpow_neg]
  have hβ : (0 : ℝ) ≤ 1 - (2 : ℝ) ^ (-24 : ℤ)
      - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-24 : ℤ)) := by norm_num [zpow_neg]
  have hAterm : (0 : ℝ) ≤ A * ((1 - (2 : ℝ) ^ (-13 : ℤ))
      - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ))) :=
    mul_nonneg hA0 hα
  have hEterm : (2 : ℝ) ^ (-125 : ℤ) * ((1 - (2 : ℝ) ^ (-24 : ℤ))
      - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-24 : ℤ)))
      ≤ e * ((1 - (2 : ℝ) ^ (-24 : ℤ))
      - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-24 : ℤ))) :=
    mul_le_mul_of_nonneg_right he hβ
  constructor
  · -- lower bound on `s2`
    have hS := add_le_add hs1_lb he32_lb
    have h2' : (A * (1 - (2 : ℝ) ^ (-13 : ℤ)) - (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (e * (1 - (2 : ℝ) ^ (-24 : ℤ)) - (2 : ℝ) ^ (-150 : ℤ))
          - (3 * (2 : ℝ) ^ (-24 : ℤ)
              * (A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ)
                + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)))
            + 4 * (2 : ℝ) ^ (-150 : ℤ))
        ≤ (s1 + e32)
          - (3 * (2 : ℝ) ^ (-24 : ℤ) * (sa + e32) + 4 * (2 : ℝ) ^ (-150 : ℤ)) := by
      linarith [hS, hmul]
    have heq : (A * (1 - (2 : ℝ) ^ (-13 : ℤ)) - (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (e * (1 - (2 : ℝ) ^ (-24 : ℤ)) - (2 : ℝ) ^ (-150 : ℤ))
          - (3 * (2 : ℝ) ^ (-24 : ℤ)
              * (A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ)
                + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)))
            + 4 * (2 : ℝ) ^ (-150 : ℤ))
        = A * ((1 - (2 : ℝ) ^ (-13 : ℤ))
            - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ)))
          + (e * ((1 - (2 : ℝ) ^ (-24 : ℤ))
            - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-24 : ℤ)))
            - ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
              + (2 : ℝ) ^ (-150 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
              + 4 * (2 : ℝ) ^ (-150 : ℤ))) := by
      ring
    rw [heq] at h2'
    have hstep : (2 : ℝ) ^ (-125 : ℤ) * ((1 - (2 : ℝ) ^ (-24 : ℤ))
          - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-24 : ℤ)))
        - ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
          + (2 : ℝ) ^ (-150 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
          + 4 * (2 : ℝ) ^ (-150 : ℤ))
        ≤ A * ((1 - (2 : ℝ) ^ (-13 : ℤ))
          - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ)))
        + (e * ((1 - (2 : ℝ) ^ (-24 : ℤ))
          - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-24 : ℤ)))
          - ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
            + (2 : ℝ) ^ (-150 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
            + 4 * (2 : ℝ) ^ (-150 : ℤ))) := by
      have h1' := add_le_add hAterm (sub_le_sub_right hEterm
        ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
          + (2 : ℝ) ^ (-150 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
          + 4 * (2 : ℝ) ^ (-150 : ℤ)))
      simpa using h1'
    have hnum : (2 : ℝ) ^ (-126 : ℤ)
        ≤ (2 : ℝ) ^ (-125 : ℤ) * ((1 - (2 : ℝ) ^ (-24 : ℤ))
            - 3 * (2 : ℝ) ^ (-24 : ℤ) * (1 + (2 : ℝ) ^ (-24 : ℤ)))
          - ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
            + (2 : ℝ) ^ (-150 : ℤ) * (1 + 3 * (2 : ℝ) ^ (-24 : ℤ))
            + 4 * (2 : ℝ) ^ (-150 : ℤ)) := by
      norm_num [zpow_neg]
    exact le_trans (le_trans hnum hstep) (le_trans h2' hlow)
  · -- upper bound on `s2`
    have hs1le : s1 ≤ A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
      linarith
    have he32le : e32 ≤ e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ) := he32_ub
    have hsum_ub : s1 + e32 ≤ (A * (1 + (2 : ℝ) ^ (-13 : ℤ))
          + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
        + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)) :=
      add_le_add hs1le he32le
    have hC_le : (A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ))
        ≤ (2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ))
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := by
      have hu24u13 : (2 : ℝ) ^ (-24 : ℤ) ≤ (2 : ℝ) ^ (-13 : ℤ) := by norm_num
      have heu : e * (1 + (2 : ℝ) ^ (-24 : ℤ)) ≤ e * (1 + (2 : ℝ) ^ (-13 : ℤ)) := by
        have h1 : e * (2 : ℝ) ^ (-24 : ℤ) ≤ e * (2 : ℝ) ^ (-13 : ℤ) :=
          mul_le_mul_of_nonneg_left hu24u13 he0
        nlinarith [h1]
      have hcoefK : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-13 : ℤ) := by norm_num
      have hkey : (A + e) * (1 + (2 : ℝ) ^ (-13 : ℤ))
          ≤ (2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ)) :=
        mul_le_mul_of_nonneg_right hAe hcoefK
      have hAe13 : A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + e * (1 + (2 : ℝ) ^ (-24 : ℤ))
          ≤ (2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ)) := by
        have h1 : A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + e * (1 + (2 : ℝ) ^ (-24 : ℤ))
            ≤ A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + e * (1 + (2 : ℝ) ^ (-13 : ℤ)) :=
          add_le_add (le_refl _) heu
        have h2 : A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + e * (1 + (2 : ℝ) ^ (-13 : ℤ))
            = (A + e) * (1 + (2 : ℝ) ^ (-13 : ℤ)) := by ring
        rw [h2] at h1
        exact le_trans h1 hkey
      have hgroup : (A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
            + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ))
          = (A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + e * (1 + (2 : ℝ) ^ (-24 : ℤ)))
            + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := by ring
      rw [hgroup]
      exact add_le_add hAe13 (le_refl _)
    have hmul2 : 3 * (2 : ℝ) ^ (-24 : ℤ)
          * ((A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
            + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)))
        ≤ 3 * (2 : ℝ) ^ (-24 : ℤ)
          * ((2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ))
            + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ))) :=
      mul_le_mul_of_nonneg_left hC_le h3u
    have hnum : (2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ))
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ))
        + (3 * (2 : ℝ) ^ (-24 : ℤ)
            * ((2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ))
              + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)))
          + 4 * (2 : ℝ) ^ (-150 : ℤ))
        ≤ (2 : ℝ) ^ (127 : ℤ) := by
      norm_num [zpow_neg]
    have hstep1 : s2 ≤ ((A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)))
        + (3 * (2 : ℝ) ^ (-24 : ℤ)
            * ((A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
              + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)))
          + 4 * (2 : ℝ) ^ (-150 : ℤ)) :=
      le_trans hup (add_le_add hsum_ub (add_le_add hmul (le_refl _)))
    have hstep2 : ((A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)))
        + (3 * (2 : ℝ) ^ (-24 : ℤ)
            * ((A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
              + (e * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ)))
          + 4 * (2 : ℝ) ^ (-150 : ℤ))
        ≤ ((2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ))
            + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)))
          + (3 * (2 : ℝ) ^ (-24 : ℤ)
              * ((2 : ℝ) ^ (126 : ℤ) * (1 + (2 : ℝ) ^ (-13 : ℤ))
                + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)))
            + 4 * (2 : ℝ) ^ (-150 : ℤ)) :=
      add_le_add hC_le (add_le_add hmul2 (le_refl _))
    exact le_trans (le_trans hstep1 hstep2) hnum

/-- The `float32` room condition for `tl.sqrt` on an argument at most `2^127`. -/
theorem sqrt_room {s : ℝ} (_hs0 : 0 ≤ s) (hs : s ≤ (2 : ℝ) ^ (127 : ℤ)) :
    (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.sqrt s| + (2 : ℝ) ^ (-149 : ℤ)
      ≤ (Format.binary32.maxVal : ℝ) := by
  have hsq : |Real.sqrt s| ≤ (2 : ℝ) ^ (64 : ℤ) := by
    rw [abs_of_nonneg (Real.sqrt_nonneg s)]
    have h1 : s ≤ ((2 : ℝ) ^ (64 : ℤ)) ^ 2 := by
      have h2 : ((2 : ℝ) ^ (64 : ℤ)) ^ 2 = (2 : ℝ) ^ (128 : ℤ) := by norm_num
      have h3 : (2 : ℝ) ^ (127 : ℤ) ≤ (2 : ℝ) ^ (128 : ℤ) := by norm_num
      linarith
    calc Real.sqrt s ≤ Real.sqrt (((2 : ℝ) ^ (64 : ℤ)) ^ 2) := Real.sqrt_le_sqrt h1
      _ = (2 : ℝ) ^ (64 : ℤ) := Real.sqrt_sq (by norm_num)
  have hmain : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.sqrt s| ≤ (2 : ℝ) ^ (65 : ℤ) := by
    have h1 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.sqrt s|
        ≤ (1 + (2 : ℝ) ^ (-22 : ℤ)) * (2 : ℝ) ^ (64 : ℤ) :=
      mul_le_mul_of_nonneg_left hsq (by norm_num)
    have h2 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * (2 : ℝ) ^ (64 : ℤ) ≤ (2 : ℝ) ^ (65 : ℤ) := by norm_num
    linarith
  have h3 : (2 : ℝ) ^ (65 : ℤ) + (2 : ℝ) ^ (-149 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  linarith

/-- `q = fl(√s2)` is normal, with `1/q ≤ 2^65`. -/
theorem q_low {s2 q : ℝ} (hs2 : (2 : ℝ) ^ (-126 : ℤ) ≤ s2)
    (h3 : |q - Real.sqrt s2| ≤ (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 + (2 : ℝ) ^ (-149 : ℤ)) :
    (2 : ℝ) ^ (-65 : ℤ) ≤ q ∧ (1 : ℝ) / q ≤ (2 : ℝ) ^ (65 : ℤ) := by
  have hs2pos : (0 : ℝ) < s2 := lt_of_lt_of_le (by norm_num) hs2
  have hsqrt_low : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt s2 := by
    have h1 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 ≤ s2 := by
      have h2 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = (2 : ℝ) ^ (-126 : ℤ) := by norm_num
      rw [h2]; exact hs2
    exact (Real.le_sqrt (by norm_num) hs2pos.le).2 h1
  have hq_low : (2 : ℝ) ^ (-65 : ℤ) ≤ q := by
    have h := (abs_le.1 h3).1
    have hu3 : (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 ≤ (1 / 2) * Real.sqrt s2 := by
      have hle : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 / 2 := by norm_num
      exact mul_le_mul_of_nonneg_right hle (Real.sqrt_nonneg s2)
    have heta : (2 : ℝ) ^ (-149 : ℤ) ≤ (2 : ℝ) ^ (-65 : ℤ) := by norm_num
    have hhalf : (1 / 2 : ℝ) * (2 : ℝ) ^ (-63 : ℤ) - (2 : ℝ) ^ (-65 : ℤ)
        = (2 : ℝ) ^ (-65 : ℤ) := by norm_num
    have hstep1 : (1 / 2 : ℝ) * (2 : ℝ) ^ (-63 : ℤ) ≤ (1 / 2 : ℝ) * Real.sqrt s2 :=
      mul_le_mul_of_nonneg_left hsqrt_low (by norm_num)
    have hstep2 : (1 / 2 : ℝ) * Real.sqrt s2
        ≤ Real.sqrt s2 - (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 := by linarith [hu3]
    have hstep3 : (1 / 2 : ℝ) * (2 : ℝ) ^ (-63 : ℤ)
        ≤ Real.sqrt s2 - (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 := le_trans hstep1 hstep2
    have hstep4 : (1 / 2 : ℝ) * (2 : ℝ) ^ (-63 : ℤ) - (2 : ℝ) ^ (-149 : ℤ)
        ≤ (Real.sqrt s2 - (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2) - (2 : ℝ) ^ (-149 : ℤ) :=
      sub_le_sub_right hstep3 _
    have hstep5 : (2 : ℝ) ^ (-65 : ℤ) ≤ (1 / 2 : ℝ) * (2 : ℝ) ^ (-63 : ℤ) - (2 : ℝ) ^ (-149 : ℤ) := by
      linarith [heta, hhalf]
    have htarget : (2 : ℝ) ^ (-65 : ℤ)
        ≤ Real.sqrt s2 - ((2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 + (2 : ℝ) ^ (-149 : ℤ)) := by
      have hmain : (2 : ℝ) ^ (-65 : ℤ)
          ≤ (Real.sqrt s2 - (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2) - (2 : ℝ) ^ (-149 : ℤ) :=
        le_trans hstep5 hstep4
      have hA' : Real.sqrt s2 - ((2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 + (2 : ℝ) ^ (-149 : ℤ))
          = (Real.sqrt s2 - (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2) - (2 : ℝ) ^ (-149 : ℤ) := by
        ring
      rw [hA']
      exact hmain
    linarith [h, htarget]
  refine ⟨hq_low, ?_⟩
  have hqpos : 0 < q := lt_of_lt_of_le (by norm_num) hq_low
  have h1 : (1 : ℝ) / q ≤ 1 / (2 : ℝ) ^ (-65 : ℤ) :=
    one_div_le_one_div_of_le (by norm_num) hq_low
  have h2 : (1 : ℝ) / (2 : ℝ) ^ (-65 : ℤ) = (2 : ℝ) ^ (65 : ℤ) := by norm_num
  linarith

end Claims.LayerNormGatedFwd
