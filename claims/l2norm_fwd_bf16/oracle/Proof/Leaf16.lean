import Proof.Defs
import Proof.RowSum
import Proof.NumRstd
import Proof.NumY

/-!
# `l2norm_fwd_bf16`: the result of one program, `y` of `bfloat16`

After the walk of the core (`bodyCore`) and the two stores, the program has written its tile of `y`
(as `bfloat16`) and its `BT` elements of `rstd`. This is the leaf: from the values the walk
computed (`b_x16`, `b_x`, `b_xx`, `s1`, `s2`, `sq`, `b_rstd`, `b_y`, `b_y16`) and the facts the
`Safe` rules gave about them, prove `Qn d init p m` for the final memory `m`.

The proof has three parts:
1. per lane of the tile, the `f32` square `b_xx` is within `u·a² + η` of the exact `a²` of the
   row's `bfloat16` input `a` (`b_x` is the exact widening of `b_x16`, so `b_x.toReal = a`),
   and the masked-off lanes are `+0`, whose square is `0`;
2. per row, `rstd_err` (`NumRstd`) with `sum_err_bd` for the sum and the platform facts for the
   square root and the division;
3. per lane, `y_err_bf16` (`NumY`) for the stored `bfloat16`.
-/

set_option maxHeartbeats 2000000

namespace Claims.L2NormFwd

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

/-- The `float32` room condition for `x * rstd`. -/
theorem mul_abs_le {A e x r : ℝ} (hAe : 0 < A + e) (he0 : 0 ≤ e) (hx2 : x ^ 2 ≤ A)
    (hr : |r - (Real.sqrt (A + e))⁻¹| ≤ (2 : ℝ) ^ (-12 : ℤ) * (Real.sqrt (A + e))⁻¹) :
    |x * r| ≤ (Format.binary32.maxVal : ℝ) := by
  have hA0 : 0 ≤ A := le_trans (sq_nonneg x) hx2
  have hsqrtpos : 0 < Real.sqrt (A + e) := Real.sqrt_pos.2 hAe
  have hinvpos : 0 < (Real.sqrt (A + e))⁻¹ := inv_pos.2 hsqrtpos
  have hsqrta : |x| ≤ Real.sqrt A := Real.abs_le_sqrt hx2
  have hAle : Real.sqrt A ≤ Real.sqrt (A + e) := Real.sqrt_le_sqrt (by linarith)
  have hxle : |x| ≤ Real.sqrt (A + e) := le_trans hsqrta hAle
  have hrle : |r| ≤ (1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹ := by
    have h := (abs_le.1 hr).2
    have h2 : |(Real.sqrt (A + e))⁻¹| = (Real.sqrt (A + e))⁻¹ := abs_of_pos hinvpos
    rw [← h2] at h
    have h3 : |r| ≤ |(Real.sqrt (A + e))⁻¹| + (2 : ℝ) ^ (-12 : ℤ) * (Real.sqrt (A + e))⁻¹ := by
      have h4 : |r| ≤ |(Real.sqrt (A + e))⁻¹| + |r - (Real.sqrt (A + e))⁻¹| := by
        have h5 := abs_add_le ((Real.sqrt (A + e))⁻¹) (r - (Real.sqrt (A + e))⁻¹)
        have h6 : (Real.sqrt (A + e))⁻¹ + (r - (Real.sqrt (A + e))⁻¹) = r := by ring
        rw [h6] at h5
        exact h5
      linarith [hr]
    rw [h2] at h3
    linarith
  have hstep : |x * r| ≤ Real.sqrt (A + e) * ((1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹) := by
    simpa only [abs_mul] using mul_le_mul hxle hrle (abs_nonneg r) (by positivity)
  have hcancel : Real.sqrt (A + e) * (Real.sqrt (A + e))⁻¹ = 1 :=
    mul_inv_cancel₀ (ne_of_gt hsqrtpos)
  have hval : Real.sqrt (A + e) * ((1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹)
      = 1 + (2 : ℝ) ^ (-12 : ℤ) := by
    calc Real.sqrt (A + e) * ((1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹)
        = (Real.sqrt (A + e) * (Real.sqrt (A + e))⁻¹) * (1 + (2 : ℝ) ^ (-12 : ℤ)) := by ring
      _ = 1 + (2 : ℝ) ^ (-12 : ℤ) := by rw [hcancel, one_mul]
  rw [hval] at hstep
  refine hstep.trans ?_
  rw [Format.binary32_maxVal]
  norm_num

/-- The `float32` room condition is not merely below `maxVal`: `|x * r| ≤ 1 + 2^-12`, so the
product is far inside `bfloat16`'s range. -/
theorem mul_small {A e x r : ℝ} (hAe : 0 < A + e) (he0 : 0 ≤ e) (hx2 : x ^ 2 ≤ A)
    (hr : |r - (Real.sqrt (A + e))⁻¹| ≤ (2 : ℝ) ^ (-12 : ℤ) * (Real.sqrt (A + e))⁻¹) :
    |x * r| ≤ 1 + (2 : ℝ) ^ (-12 : ℤ) := by
  have hA0 : 0 ≤ A := le_trans (sq_nonneg x) hx2
  have hsqrtpos : 0 < Real.sqrt (A + e) := Real.sqrt_pos.2 hAe
  have hinvpos : 0 < (Real.sqrt (A + e))⁻¹ := inv_pos.2 hsqrtpos
  have hsqrta : |x| ≤ Real.sqrt A := Real.abs_le_sqrt hx2
  have hAle : Real.sqrt A ≤ Real.sqrt (A + e) := Real.sqrt_le_sqrt (by linarith)
  have hxle : |x| ≤ Real.sqrt (A + e) := le_trans hsqrta hAle
  have hrle : |r| ≤ (1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹ := by
    have h := (abs_le.1 hr).2
    have h2 : |(Real.sqrt (A + e))⁻¹| = (Real.sqrt (A + e))⁻¹ := abs_of_pos hinvpos
    rw [← h2] at h
    have h3 : |r| ≤ |(Real.sqrt (A + e))⁻¹| + (2 : ℝ) ^ (-12 : ℤ) * (Real.sqrt (A + e))⁻¹ := by
      have h4 : |r| ≤ |(Real.sqrt (A + e))⁻¹| + |r - (Real.sqrt (A + e))⁻¹| := by
        have h5 := abs_add_le ((Real.sqrt (A + e))⁻¹) (r - (Real.sqrt (A + e))⁻¹)
        have h6 : (Real.sqrt (A + e))⁻¹ + (r - (Real.sqrt (A + e))⁻¹) = r := by ring
        rw [h6] at h5
        exact h5
      linarith [hr]
    rw [h2] at h3
    linarith
  have hstep : |x * r| ≤ Real.sqrt (A + e) * ((1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹) := by
    simpa only [abs_mul] using mul_le_mul hxle hrle (abs_nonneg r) (by positivity)
  have hcancel : Real.sqrt (A + e) * (Real.sqrt (A + e))⁻¹ = 1 :=
    mul_inv_cancel₀ (ne_of_gt hsqrtpos)
  have hval : Real.sqrt (A + e) * ((1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹)
      = 1 + (2 : ℝ) ^ (-12 : ℤ) := by
    calc Real.sqrt (A + e) * ((1 + (2 : ℝ) ^ (-12 : ℤ)) * (Real.sqrt (A + e))⁻¹)
        = (Real.sqrt (A + e) * (Real.sqrt (A + e))⁻¹) * (1 + (2 : ℝ) ^ (-12 : ℤ)) := by ring
      _ = 1 + (2 : ℝ) ^ (-12 : ℤ) := by rw [hcancel, one_mul]
  rw [hval] at hstep
  exact hstep

/-- The row sums of the tile: `s1` is within `2^-13·A + (3/4)·2^-137` of the exact `A`, and `s2` is the finite `float32` sum with `eps` that `rstd_err_wide` needs. -/
theorem qn16_sum (d : Inst) (hd : d.ok) {P : Platform}
    {bxx : Nat → F32} {a : Nat → BF16} {i0 : Fin d.BT}
    {s1 s2 : Tensor [d.BT] F32}
    (hbxx_fin : ∀ j, j < d.BD → (bxx j).isFinite = true)
    (hbxx_err : ∀ j, j < d.BD →
      |(bxx j).toReal - (a j).toReal ^ 2|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ))
    (htree : ∃ t : Tree, t.Covers d.BD ∧
      Tree.Eval (AddRel ((launch d).env P).fusion) bxx t (s1.get1 i0))
    (haddc : AddRel ((launch d).env P).fusion (s1.get1 i0)
      ((bc (FloatV.hostF32 d.eps)).get1 i0) (s2.get1 i0))
    (hrowsum : rowSum a d.BD = rowSum a d.D)
    (hAe : rowSum a d.D + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ))
    (he : (2 : ℝ) ^ (-125 : ℤ) ≤ d.eps.toReal) :
    (s1.get1 i0).isFinite = true ∧
    |(s1.get1 i0).toReal - rowSum a d.D|
      ≤ (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) ∧
    (s2.get1 i0).isFinite = true ∧
    |(s2.get1 i0).toReal - ((s1.get1 i0).toReal + (FloatV.hostF32 d.eps).toReal)|
      ≤ 3 * (2 : ℝ) ^ (-24 : ℤ) * (|(s1.get1 i0).toReal| + (FloatV.hostF32 d.eps).toReal)
        + 4 * (2 : ℝ) ^ (-150 : ℤ) ∧
    (0 : ℝ) ≤ (FloatV.hostF32 d.eps).toReal ∧
    (2 : ℝ) ^ (-125 : ℤ) ≤ (FloatV.hostF32 d.eps).toReal ∧
    |(FloatV.hostF32 d.eps).toReal - d.eps.toReal|
      ≤ (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ) ∧
    (0 : ℝ) ≤ d.eps.toReal := by
  have hAeBD : rowSum a d.BD + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by
    rw [hrowsum]; exact hAe
  obtain ⟨hs1fin, hs1err⟩ := sum_err_bd hd ((launch d).env P).fusion
    hbxx_fin hbxx_err hAeBD he htree
  have hs1 : |(s1.get1 i0).toReal - rowSum a d.D|
      ≤ (2 : ℝ) ^ (-13 : ℤ) * rowSum a d.D + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
    have h := hs1err
    rw [hrowsum] at h
    exact h
  have hehi : d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hd.eps_hi
    rw [Rat.cast_zpow, Rat.cast_ofNat] at h
    simpa only [FloatV.toReal] using h
  have he0 : (0 : ℝ) ≤ d.eps.toReal := le_trans (by norm_num) he
  have hepos : (0 : ℚ) < d.eps.toRat := lt_of_lt_of_le (by norm_num) hd.eps_lo
  obtain ⟨heps32_fin, heps32_err⟩ := hostF32_err hd.eps_fin hepos.le hd.eps_hi
  have heps32_errR : |(FloatV.hostF32 d.eps).toReal - d.eps.toReal|
      ≤ (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr heps32_err
    simp only [FloatV.toReal]
    push_cast at h ⊢
    simpa only [Rat.cast_sub, Rat.cast_abs] using h
  have he32_lo : (2 : ℝ) ^ (-125 : ℤ) ≤ (FloatV.hostF32 d.eps).toReal :=
    hostF32_ge hd.eps_fin hd.eps_lo hd.eps_hi
  have he32_nn : (0 : ℝ) ≤ (FloatV.hostF32 d.eps).toReal := le_trans (by norm_num) he32_lo
  have heps32_abs : |(FloatV.hostF32 d.eps).toReal| = (FloatV.hostF32 d.eps).toReal :=
    abs_of_nonneg he32_nn
  have hVhi : rowSum a d.D + (FloatV.hostF32 d.eps).toReal ≤ (2 : ℝ) ^ (127 : ℤ) := by
    have hup : (FloatV.hostF32 d.eps).toReal
        ≤ d.eps.toReal + ((2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ)) := by
      have hb := (abs_le.1 heps32_errR).2
      linarith
    have h1e : (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (126 : ℤ) :=
      mul_le_mul_of_nonneg_left hehi (by norm_num)
    have hkey : (2 : ℝ) ^ (126 : ℤ)
        + ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (126 : ℤ)
          + (2 : ℝ) ^ (-150 : ℤ)) ≤ (2 : ℝ) ^ (127 : ℤ) := by
      norm_num [zpow_neg]
    linarith [hAe, hup, hkey, h1e]
  have hA0 : (0 : ℝ) ≤ rowSum a d.D := Finset.sum_nonneg (fun j _ => sq_nonneg _)
  have hs1abs : |(s1.get1 i0).toReal|
      ≤ rowSum a d.D + rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
    have h2 : |(s1.get1 i0).toReal|
        ≤ |(s1.get1 i0).toReal - rowSum a d.D| + |rowSum a d.D| := by
      have h := abs_add_le ((s1.get1 i0).toReal - rowSum a d.D) (rowSum a d.D)
      rwa [show (s1.get1 i0).toReal - rowSum a d.D + rowSum a d.D = (s1.get1 i0).toReal
        from by ring] at h
    have hAa : |rowSum a d.D| = rowSum a d.D := abs_of_nonneg hA0
    nlinarith [hs1, h2, hAa]
  have he32_absR : |(FloatV.hostF32 d.eps).toReal|
      ≤ d.eps.toReal + d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have h2 : |(FloatV.hostF32 d.eps).toReal|
        ≤ |(FloatV.hostF32 d.eps).toReal - d.eps.toReal| + |d.eps.toReal| := by
      have h := abs_add_le ((FloatV.hostF32 d.eps).toReal - d.eps.toReal) d.eps.toReal
      rwa [show (FloatV.hostF32 d.eps).toReal - d.eps.toReal + d.eps.toReal
        = (FloatV.hostF32 d.eps).toReal from by ring] at h
    have hea : |d.eps.toReal| = d.eps.toReal := abs_of_nonneg he0
    nlinarith [heps32_errR, h2, hea]
  have hsum_absR : |(s1.get1 i0).toReal| + |(FloatV.hostF32 d.eps).toReal|
      ≤ (2 : ℝ) ^ (126 : ℤ) + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-13 : ℤ)
        + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-24 : ℤ)
        + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
    have hA126 : rowSum a d.D ≤ (2 : ℝ) ^ (126 : ℤ) := by linarith [hAe, he0]
    have heps126 : d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by linarith [hAe, hA0]
    have hAu : rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ)
        ≤ (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-13 : ℤ) :=
      mul_le_mul_of_nonneg_right hA126 (by norm_num)
    have heu : d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ)
        ≤ (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right heps126 (by norm_num)
    have h1 : |(s1.get1 i0).toReal| + |(FloatV.hostF32 d.eps).toReal|
        ≤ (rowSum a d.D + rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ)
            + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (d.eps.toReal + d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ)
            + (2 : ℝ) ^ (-150 : ℤ)) :=
      add_le_add hs1abs he32_absR
    have h2 : (rowSum a d.D + rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ)
            + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (d.eps.toReal + d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ)
            + (2 : ℝ) ^ (-150 : ℤ))
        = (rowSum a d.D + d.eps.toReal)
          + (rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ) + d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ))
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := by ring
    have h4 : (rowSum a d.D + d.eps.toReal)
          + (rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ) + d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ))
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ))
        ≤ (2 : ℝ) ^ (126 : ℤ)
          + ((2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-13 : ℤ)
            + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-24 : ℤ))
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) :=
      add_le_add (add_le_add hAe (add_le_add hAu heu)) le_rfl
    calc |(s1.get1 i0).toReal| + |(FloatV.hostF32 d.eps).toReal|
        ≤ (rowSum a d.D + rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ)
            + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ))
          + (d.eps.toReal + d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ)
            + (2 : ℝ) ^ (-150 : ℤ)) := h1
      _ = (rowSum a d.D + d.eps.toReal)
          + (rowSum a d.D * (2 : ℝ) ^ (-13 : ℤ) + d.eps.toReal * (2 : ℝ) ^ (-24 : ℤ))
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := h2
      _ ≤ (2 : ℝ) ^ (126 : ℤ)
          + ((2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-13 : ℤ)
            + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-24 : ℤ))
          + ((3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) := h4
      _ = (2 : ℝ) ^ (126 : ℤ) + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-13 : ℤ)
        + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-24 : ℤ)
        + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by ring
  have hsum_absq : |(s1.get1 i0).toRat| + |(FloatV.hostF32 d.eps).toRat|
      ≤ (2 : ℚ) ^ (126 : ℤ) + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-13 : ℤ)
        + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-24 : ℤ)
        + (3 / 4) * (2 : ℚ) ^ (-137 : ℤ) + (2 : ℚ) ^ (-150 : ℤ) := by
    have h2 : ((|(s1.get1 i0).toRat| + |(FloatV.hostF32 d.eps).toRat| : ℚ) : ℝ)
        ≤ (((2 : ℚ) ^ (126 : ℤ) + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-13 : ℤ)
          + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-24 : ℤ)
          + (3 / 4) * (2 : ℚ) ^ (-137 : ℤ) + (2 : ℚ) ^ (-150 : ℤ) : ℚ) : ℝ) := by
      have e : (((2 : ℚ) ^ (126 : ℤ) + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-13 : ℤ)
          + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-24 : ℤ)
          + (3 / 4) * (2 : ℚ) ^ (-137 : ℤ) + (2 : ℚ) ^ (-150 : ℤ) : ℚ) : ℝ)
          = (2 : ℝ) ^ (126 : ℤ) + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-13 : ℤ)
            + (2 : ℝ) ^ (126 : ℤ) * (2 : ℝ) ^ (-24 : ℤ)
            + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) + (2 : ℝ) ^ (-150 : ℤ) := by
        push_cast; norm_num
      rw [e]
      simpa only [Rat.cast_add, Rat.cast_abs, FloatV.toReal] using hsum_absR
    exact Rat.cast_le (K := ℝ).mp h2
  have hMq : |(s1.get1 i0).toRat| + |(FloatV.hostF32 d.eps).toRat|
      ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have hkey : (2 : ℚ) ^ (126 : ℤ) + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-13 : ℤ)
        + (2 : ℚ) ^ (126 : ℤ) * (2 : ℚ) ^ (-24 : ℤ)
        + (3 / 4) * (2 : ℚ) ^ (-137 : ℤ) + (2 : ℚ) ^ (-150 : ℤ)
        ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
      rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num [zpow_neg]
    linarith [hsum_absq, hkey]
  obtain ⟨hs2fin, hs2err⟩ := FloatV.AddRel.err (f := Format.binary32) Format.binary32_regular
    (by rw [Format.Std.ieee Format.binary32_std]; decide) (by rw [Format.binary32_u]; norm_num [zpow_neg]) haddc hs1fin
    heps32_fin hMq
  have hs2R : |(s2.get1 i0).toReal
        - ((s1.get1 i0).toReal + (FloatV.hostF32 d.eps).toReal)|
      ≤ 3 * (2 : ℝ) ^ (-24 : ℤ)
          * (|(s1.get1 i0).toReal| + (FloatV.hostF32 d.eps).toReal)
        + 4 * (2 : ℝ) ^ (-150 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hs2err
    have hbc : (bc (FloatV.hostF32 d.eps)).get1 i0 = FloatV.hostF32 d.eps := rfl
    rw [hbc] at h
    have hu : ((Format.binary32.u : ℚ) : ℝ) = (2 : ℝ) ^ (-24 : ℤ) := by
      rw [Format.binary32_u]; norm_num
    have heta : ((Format.binary32.eta : ℚ) : ℝ) = (2 : ℝ) ^ (-150 : ℤ) := by
      rw [Format.binary32_eta]; norm_num
    push_cast at h
    rw [hu, heta] at h
    simp only [FloatV.toReal] at h heps32_abs ⊢
    simpa only [heps32_abs] using h
  exact ⟨hs1fin, hs1, hs2fin, hs2R, he32_nn, he32_lo, heps32_errR, he0⟩

/-- **One program, `bfloat16` `y`**: from the values the walk computed and the facts the `Safe` rules gave, the final memory satisfies `Qn`. -/
theorem qn16 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {yb : Buf .bf16}
    (hy : d.y = .inl yb) (p : Nat) (m : Mem)
    (P : Platform) (hsqrt_acc : Device.SqrtF32 P) (hdiv_full : Device.DivFull P)
    (b_x16 : Tensor [d.BT, d.BD] BF16) (b_x b_xx : Tensor [d.BT, d.BD] F32)
    (s1 s2 sq b_rstd : Tensor [d.BT] F32) (b_y : Tensor [d.BT, d.BD] F32)
    (b_y16 : Tensor [d.BT, d.BD] BF16)
    (hx16 : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      init.read .bf16 (d.x.ptr (gx d p j)).addr = some (b_x16.get j))
    (hz : ∀ j : Idx [d.BT, d.BD], enx d p j = false →
      b_x16.get j = FloatV.litBF16 (FloatV.ofPat 0))
    (hcvt : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.cvtWith .rn (b_x16.get j)) (b_x.get j))
    (hmul : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.mul (b_x.get j) (b_x.get j)) (b_xx.get j))
    (htree : ∀ r : Fin d.BT, ∃ t : Tree, t.Covers d.BD ∧
      t.Eval (AddRel ((launch d).env P).fusion)
        (fun j => if h : j < d.BD then b_xx.get2 r ⟨j, h⟩ else default) (s1.get1 r))
    (haddc : ∀ j : Idx [d.BT],
      AddRel ((launch d).env P).fusion (s1.get j) ((bc (FloatV.hostF32 d.eps)).get j) (s2.get j))
    (hsqrt : ∀ j : Idx [d.BT], P.math .sqrt .binary32 (s2.get j) (sq.get j))
    (hdiv : ∀ j : Idx [d.BT],
      P.div .binary32 ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000))).get j) (sq.get j)
        (b_rstd.get j))
    (hmul2 : ∀ j : Idx [d.BT, d.BD],
      FloatV.NaNEq (FloatV.mul (b_x.get j) ((b_rstd.col d.BD).get j)) (b_y.get j))
    (hcvt16 : ∀ j : Idx [d.BT, d.BD], FloatV.NaNEq (FloatV.cvtWith .rn (b_y.get j)) (b_y16.get j))
    (hyread : ∀ j : Idx [d.BT, d.BD], enx d p j = true →
      m.read .bf16 (yb.ptr (gx d p j)).addr = some (b_y16.get j))
    (hrread : ∀ j : Idx [d.BT], enT d p j = true →
      m.read .f32 (d.rstd.ptr (gT d p j)).addr = some (b_rstd.get j)) :
    Qn d init p m := by
  intro t ht
  have htT : t < d.T := ht.1
  have hlo : p * d.BT ≤ t := ht.2.1
  have hhi : t < p * d.BT + d.BT := ht.2.2
  have hsub : t - p * d.BT < d.BT := by omega
  let i0 : Fin d.BT := ⟨t - p * d.BT, hsub⟩
  let a : Nat → BF16 := fun j =>
    if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else FloatV.litBF16 (FloatV.ofPat 0)
  have htm : t ∈ tileRow d p := ⟨htT, hlo, hhi⟩
  have hi0 : (i0 : Fin d.BT).val = t - p * d.BT := rfl
  -- ## the row's inputs
  have hreadA : ∀ j, j < d.D → init.read .bf16 (d.x.ptr (xIdx d t j)).addr = some (a j) := by
    intro j hj
    have hav : a j = b_x16.get2 i0 ⟨j, col_lt_bd hd hj⟩ := by
      show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
        FloatV.litBF16 (FloatV.ofPat 0)) = _
      rw [dite_eq_left_of_eq_true (eq_true hj)]
    rw [hav]
    have h := hx16 (i0, (⟨j, col_lt_bd hd hj⟩, ())) (lane_enx htm hj)
    rwa [lane_gx htm] at h
  have hAe := hpre.2.2.2.2 t htT a hreadA
  have hAeZ : rowSum a d.D + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by simpa using hAe
  have hA0 : (0 : ℝ) ≤ rowSum a d.D := Finset.sum_nonneg (fun j _ => sq_nonneg _)
  have he : (2 : ℝ) ^ (-125 : ℤ) ≤ d.eps.toReal := by
    have h := Rat.cast_le (K := ℝ).mpr hd.eps_lo
    push_cast at h
    simpa only [FloatV.toReal] using h
  -- ## the per-column facts
  have hcol : ∀ j (hj : j < d.BD),
      (b_x16.get2 i0 ⟨j, hj⟩).isFinite = true ∧
      (b_x.get2 i0 ⟨j, hj⟩).isFinite = true ∧
      (b_x.get2 i0 ⟨j, hj⟩).toReal = (a j).toReal ∧
      |(b_x.get2 i0 ⟨j, hj⟩).toRat| ≤ (2 : ℚ) ^ (63 : ℤ) ∧
      (b_xx.get2 i0 ⟨j, hj⟩).isFinite = true ∧
      |(b_xx.get2 i0 ⟨j, hj⟩).toReal - (a j).toReal ^ 2|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
    intro j hj
    by_cases hjd : j < d.D
    · have hreadj := hx16 (i0, (⟨j, hj⟩, ())) (lane_enx htm hjd)
      rw [lane_gx htm] at hreadj
      obtain ⟨a', ha', hfin'⟩ := hpre.2.2.2.1 t htT j hjd
      have hb16eq : b_x16.get2 i0 ⟨j, hj⟩ = a' := by
        have hs : some (b_x16.get (i0, (⟨j, hj⟩, ()))) = some a' := by rw [← hreadj]; exact ha'
        exact Option.some.inj hs
      have hfin16 : (b_x16.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hb16eq]; exact hfin'
      have hcvtw := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
        Format.bfloat16_regular Format.binary32_regular wider_bfloat16_binary32 .rn hfin16
      obtain ⟨hcvtfin, hcvts⟩ := hcvtw
      have hbxeq : b_x.get2 i0 ⟨j, hj⟩ =
          (FloatV.cvtWith .rn (b_x16.get2 i0 ⟨j, hj⟩) : F32) :=
        (hcvt (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hcvtfin).1
      have hbxfin : (b_x.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hbxeq]; exact hcvtfin
      have hbxs : (b_x.get2 i0 ⟨j, hj⟩).toRat = (b_x16.get2 i0 ⟨j, hj⟩).toRat := by
        rw [hbxeq, hcvts]
      have haj : a j = b_x16.get2 i0 ⟨j, hj⟩ := by
        show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
          FloatV.litBF16 (FloatV.ofPat 0)) = _
        rw [dite_eq_left_of_eq_true (eq_true hjd)]
      have hbxR : (b_x.get2 i0 ⟨j, hj⟩).toReal = (a j).toReal := by
        simp only [FloatV.toReal]
        rw [hbxs, ← haj]
      have hbxRu : ((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ) = ((a j).toRat : ℝ) := by
        simpa only [FloatV.toReal] using hbxR
      have hsq_le : (a j).toReal ^ 2 ≤ (2 : ℝ) ^ (126 : ℤ) := by
        have hle : (a j).toReal ^ 2 ≤ rowSum a d.D :=
          Finset.single_le_sum (f := fun k => (a k).toReal ^ 2) (fun k _ => sq_nonneg _)
            (Finset.mem_range.mpr hjd)
        have he0' : (0 : ℝ) ≤ d.eps.toReal := le_trans (by norm_num [zpow_neg]) he
        linarith
      have hq63 : |(b_x.get2 i0 ⟨j, hj⟩).toRat| ≤ (2 : ℚ) ^ (63 : ℤ) := by
        refine (Rat.cast_le (K := ℝ)).mp ?_
        push_cast
        have h2 : (2 : ℝ) ^ (63 : ℤ) * (2 : ℝ) ^ (63 : ℤ) = (2 : ℝ) ^ (126 : ℤ) := by norm_num
        have hsqq : |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| *
            |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| = (a j).toReal ^ 2 := by
          rw [← abs_mul, abs_of_nonneg (mul_self_nonneg (((b_x.get2 i0 ⟨j, hj⟩).toRat : ℚ) : ℝ)),
            hbxRu]
          simp only [FloatV.toReal, pow_two]
        have h3 : |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| *
            |((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)|
            ≤ (2 : ℝ) ^ (63 : ℤ) * (2 : ℝ) ^ (63 : ℤ) := by
          rw [hsqq, h2]; exact hsq_le
        have hc : (0 : ℝ) ≤ (2 : ℝ) ^ (63 : ℤ) := by norm_num
        nlinarith [h3, abs_nonneg ((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ), hc,
          sq_nonneg (|((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ)| - (2 : ℝ) ^ (63 : ℤ))]
      have hmaxq : |(b_x.get2 i0 ⟨j, hj⟩).toRat * (b_x.get2 i0 ⟨j, hj⟩).toRat|
          ≤ Format.binary32.maxVal := by
        have hqq : (b_x.get2 i0 ⟨j, hj⟩).toRat * (b_x.get2 i0 ⟨j, hj⟩).toRat
            = |(b_x.get2 i0 ⟨j, hj⟩).toRat| * |(b_x.get2 i0 ⟨j, hj⟩).toRat| := by
          rw [← abs_mul, abs_of_nonneg (mul_self_nonneg _)]
        rw [abs_of_nonneg (mul_self_nonneg _), hqq, Format.binary32_maxVal]
        have h := mul_le_mul hq63 hq63 (abs_nonneg _) (by norm_num : (0:ℚ) ≤ (2:ℚ)^(63:ℤ))
        refine h.trans ?_
        norm_num [zpow_neg]
      obtain ⟨hxxfin, hxxerr⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
        hbxfin hbxfin hmaxq
      have hxxeq : b_xx.get2 i0 ⟨j, hj⟩ =
          FloatV.mul (b_x.get2 i0 ⟨j, hj⟩) (b_x.get2 i0 ⟨j, hj⟩) :=
        (hmul (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hxxfin).1
      have hxxfin' : (b_xx.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hxxeq]; exact hxxfin
      have hxxerrR : |(b_xx.get2 i0 ⟨j, hj⟩).toReal - (a j).toReal ^ 2|
          ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
        have h := Rat.cast_le (K := ℝ).mpr hxxerr
        rw [Format.binary32_u, Format.binary32_eta] at h
        push_cast at h
        simp only [FloatV.toReal] at h ⊢
        rw [← hxxeq, hbxRu] at h
        simpa [pow_two, mul_comm, mul_left_comm, mul_assoc] using h
      exact ⟨hfin16, hbxfin, hbxR, hq63, hxxfin', hxxerrR⟩
    · have henx : enx d p (i0, (⟨j, hj⟩, ())) = false := by
        have hrow : p * d.BT + (i0 : Fin d.BT).val < d.T := by rw [hi0]; omega
        simp only [enx]
        rw [decide_eq_true_eq.mpr hrow, decide_eq_false_iff_not.mpr hjd, Bool.and_false]
      have hz16 : b_x16.get2 i0 ⟨j, hj⟩ = FloatV.litBF16 (FloatV.ofPat 0) := hz _ henx
      have hfin16 : (b_x16.get2 i0 ⟨j, hj⟩).isFinite = true := by
        rw [hz16]; exact litBF16_zero_finite
      have haj : a j = FloatV.litBF16 (FloatV.ofPat 0) := by
        show (if h : j < d.D then b_x16.get2 i0 ⟨j, col_lt_bd hd h⟩ else
          FloatV.litBF16 (FloatV.ofPat 0)) = _
        rw [dite_eq_right_of_eq_false (eq_false hjd)]
      have hzeroR : (a j).toReal = 0 := by rw [haj]; exact litBF16_zero_toReal
      have hzeroRu : ((a j).toRat : ℝ) = 0 := by simpa only [FloatV.toReal] using hzeroR
      have hcvtw := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
        Format.bfloat16_regular Format.binary32_regular wider_bfloat16_binary32 .rn hfin16
      obtain ⟨hcvtfin, hcvts⟩ := hcvtw
      have hbxeq : b_x.get2 i0 ⟨j, hj⟩ =
          (FloatV.cvtWith .rn (b_x16.get2 i0 ⟨j, hj⟩) : F32) :=
        (hcvt (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hcvtfin).1
      have hbxfin : (b_x.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hbxeq]; exact hcvtfin
      have hbxzero : (b_x.get2 i0 ⟨j, hj⟩).toRat = 0 := by
        rw [hbxeq, hcvts, hz16]; exact litBF16_zero_toRat
      have hbxR : (b_x.get2 i0 ⟨j, hj⟩).toReal = (a j).toReal := by
        simp only [FloatV.toReal]
        rw [hbxzero, hzeroRu]
        norm_num
      have hq63 : |(b_x.get2 i0 ⟨j, hj⟩).toRat| ≤ (2 : ℚ) ^ (63 : ℤ) := by
        rw [hbxzero, abs_zero]; norm_num
      have hmaxq : |(b_x.get2 i0 ⟨j, hj⟩).toRat * (b_x.get2 i0 ⟨j, hj⟩).toRat|
          ≤ Format.binary32.maxVal := by
        rw [hbxzero, mul_zero, abs_zero, Format.binary32_maxVal]; norm_num
      obtain ⟨hxxfin, hxxerr⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
        hbxfin hbxfin hmaxq
      have hxxeq : b_xx.get2 i0 ⟨j, hj⟩ =
          FloatV.mul (b_x.get2 i0 ⟨j, hj⟩) (b_x.get2 i0 ⟨j, hj⟩) :=
        (hmul (i0, (⟨j, hj⟩, ()))).eq (FloatV.isFinite_iff.mp hxxfin).1
      have hxxfin' : (b_xx.get2 i0 ⟨j, hj⟩).isFinite = true := by rw [hxxeq]; exact hxxfin
      have hxxerrR : |(b_xx.get2 i0 ⟨j, hj⟩).toReal - (a j).toReal ^ 2|
          ≤ (2 : ℝ) ^ (-24 : ℤ) * (a j).toReal ^ 2 + (2 : ℝ) ^ (-150 : ℤ) := by
        have h := Rat.cast_le (K := ℝ).mpr hxxerr
        rw [Format.binary32_u, Format.binary32_eta] at h
        push_cast at h
        have hbxz : ((b_x.get2 i0 ⟨j, hj⟩).toRat : ℝ) = 0 := by rw [hbxzero]; norm_num
        simp only [FloatV.toReal, hbxz, hzeroRu] at h ⊢
        rw [← hxxeq] at h
        simpa [mul_zero, mul_comm, mul_left_comm, mul_assoc, pow_two] using h
      exact ⟨hfin16, hbxfin, hbxR, hq63, hxxfin', hxxerrR⟩
  -- ## the row sum
  have hsum_eq : rowSum a d.BD = rowSum a d.D := by
    unfold rowSum
    refine (Finset.sum_subset (Finset.range_subset_range.mpr hd.d_bd) ?_).symm
    intro x _ hx
    have hxd : ¬ x < d.D := by
      intro h'; exact hx (Finset.mem_range.mpr h')
    have hax : a x = FloatV.litBF16 (FloatV.ofPat 0) := by
      show (if h : x < d.D then b_x16.get2 i0 ⟨x, col_lt_bd hd h⟩ else
        FloatV.litBF16 (FloatV.ofPat 0)) = _
      rw [dite_eq_right_of_eq_false (eq_false hxd)]
    rw [hax, litBF16_zero_toReal]; norm_num
  obtain ⟨hs1fin, hs1, hs2fin, hs2R, he32_nn, he32_lo, heps32_errR, he0⟩ :=
    qn16_sum d hd (P := P)
      (bxx := fun j => if h : j < d.BD then b_xx.get2 i0 ⟨j, h⟩ else default)
      (a := a) (i0 := i0) (s1 := s1) (s2 := s2)
      (fun j hj => by rw [dite_eq_left_of_eq_true (eq_true hj)]; exact (hcol j hj).2.2.2.2.1)
      (fun j hj => by rw [dite_eq_left_of_eq_true (eq_true hj)]; exact (hcol j hj).2.2.2.2.2)
      (htree i0) (haddc (i0, ())) hsum_eq hAeZ he
  -- ## the row's `rstd` and `y`
  have hehi : d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) := by linarith [hAeZ, hA0]
  have hVhi : rowSum a d.D + (FloatV.hostF32 d.eps).toReal ≤ (2 : ℝ) ^ (127 : ℤ) := by
    have hup : (FloatV.hostF32 d.eps).toReal
        ≤ d.eps.toReal + ((2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ)) := by
      have hb := (abs_le.1 heps32_errR).2
      linarith
    have h1e : (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal
        ≤ (2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (126 : ℤ) :=
      mul_le_mul_of_nonneg_left hehi (by norm_num)
    have hkey : (2 : ℝ) ^ (126 : ℤ)
        + ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (126 : ℤ)
          + (2 : ℝ) ^ (-150 : ℤ)) ≤ (2 : ℝ) ^ (127 : ℤ) := by
      norm_num [zpow_neg]
    linarith [hAeZ, hup, hkey, h1e]
  have hrstd_tol : (rstdTol : ℝ) = (2 : ℝ) ^ (-12 : ℤ) := by rw [rstdTol]; norm_num
  have hAe_pos : (0 : ℝ) < rowSum a d.D + d.eps.toReal := by
    have h0 : (0 : ℝ) < (2 : ℝ) ^ (-125 : ℤ) := by norm_num
    linarith [hA0, he]
  have hR0 : 0 < rstdRef a d.D d.eps.toReal := by
    simp only [rstdRef]
    exact inv_pos.2 (Real.sqrt_pos.2 hAe_pos)
  obtain ⟨hs2lo, hs2hi⟩ := s2_range (A := rowSum a d.D) (e := d.eps.toReal)
    (s1 := (s1.get1 i0).toReal) (e32 := (FloatV.hostF32 d.eps).toReal)
    (s2 := (s2.get1 i0).toReal) hA0 he0 he hAeZ he32_nn hs1 heps32_errR hs2R
  obtain ⟨sref, hsref, hacc⟩ := hsqrt_acc
  have hsref' : sref = Real.sqrt := by
    have h : MathFn.ref MathFn.sqrt = some Real.sqrt := rfl
    rw [h] at hsref
    exact (Option.some.inj hsref).symm
  have hs2dom : (s2.get1 i0).toReal ∈ Device.sqrtDom := Or.inr hs2lo
  have hs2nat : (s2.get1 i0).toReal ∈ MathFn.natDom MathFn.sqrt := by
    have h : MathFn.natDom MathFn.sqrt = Set.Ici (0 : ℝ) := rfl
    rw [h]
    exact le_trans (by norm_num) hs2lo
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |sref (s2.get1 i0).toReal|
      + (2 : ℝ) ^ (-149 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [hsref']
    exact sqrt_room (le_trans (by norm_num) hs2lo) hs2hi
  obtain ⟨hsqfin, hsqerr0⟩ :=
    hacc (s2.get1 i0) (sq.get1 i0) (hsqrt (i0, ())) hs2fin hs2dom hs2nat hroom
  have hsqerr : |(sq.get1 i0).toReal - Real.sqrt (s2.get1 i0).toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt (s2.get1 i0).toReal + (2 : ℝ) ^ (-149 : ℤ) := by
    have h := hsqerr0
    rw [hsref'] at h
    simpa only [abs_of_nonneg (Real.sqrt_nonneg (s2.get1 i0).toReal)] using h
  obtain ⟨hqlow, hqinv⟩ := q_low hs2lo hsqerr
  have hqposR : (0 : ℝ) < (sq.get1 i0).toReal := lt_of_lt_of_le (by norm_num) hqlow
  have hqpos : (0 : ℚ) < (sq.get1 i0).toRat := by
    refine (Rat.cast_lt (K := ℝ)).mp ?_
    simpa only [FloatV.toReal, Rat.cast_zero] using hqposR
  have hqne : (sq.get1 i0).toRat ≠ 0 := ne_of_gt hqpos
  have honeq : ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
      Tensor [d.BT] F32).get (i0, ())).toRat = 1 := by
    have h : (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
        Tensor [d.BT] F32).get (i0, ()) = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) := rfl
    rw [h]
    exact litF32_one_toRat
  have honefin : ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
      Tensor [d.BT] F32).get (i0, ())).isFinite = true := by
    have h : (bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
        Tensor [d.BT] F32).get (i0, ()) = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) := rfl
    rw [h]
    exact litF32_one_finite
  have hqbound : |((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) :
      Tensor [d.BT] F32).get (i0, ())).toRat / (sq.get1 i0).toRat| ≤ 2 ^ 126 := by
    rw [honeq]
    refine rat_le_of_cast ?_
    rw [Rat.cast_abs]
    have hc1 : (((1 : ℚ) / (sq.get1 i0).toRat : ℚ) : ℝ)
        = (1 : ℝ) / (sq.get1 i0).toReal := by
      push_cast
      rfl
    rw [hc1]
    rw [abs_of_pos (one_div_pos.mpr hqposR)]
    have h2 : (2 : ℝ) ^ (65 : ℤ) ≤ (((2 : ℚ) ^ 126 : ℚ) : ℝ) := by
      push_cast
      norm_num
    linarith [hqinv, h2]
  obtain ⟨hbrfin, hbrerr⟩ := hdiv_full
    ((bc (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) : Tensor [d.BT] F32).get (i0, ()))
    (sq.get1 i0) (b_rstd.get1 i0) (hdiv (i0, ())) honefin hsqfin hqne hqbound
  have hbrerr' : |(b_rstd.get1 i0).toRat - 1 / (sq.get1 i0).toRat|
      ≤ (2 : ℚ) ^ (-22 : ℤ) * |1 / (sq.get1 i0).toRat| + (2 : ℚ) ^ (-148 : ℤ) := by
    simpa only [honeq] using hbrerr
  have hbrerrR : |(b_rstd.get1 i0).toReal - (1 : ℝ) / (sq.get1 i0).toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * |(1 : ℝ) / (sq.get1 i0).toReal| + (2 : ℝ) ^ (-148 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hbrerr'
    push_cast at h
    simpa only [FloatV.toReal] using h
  have hwide := rstd_err_wide (A := rowSum a d.D) (e := (FloatV.hostF32 d.eps).toReal)
    (s1 := (s1.get1 i0).toReal) (s2 := (s2.get1 i0).toReal) (q := (sq.get1 i0).toReal)
    (r := (b_rstd.get1 i0).toReal) he32_nn hA0 he32_lo hVhi hs1 hs2R
    (lt_of_lt_of_le (by norm_num) hqlow) hsqerr hbrerrR
  have hfinal := eps_transfer (A := rowSum a d.D) (e := d.eps.toReal)
    (e32 := (FloatV.hostF32 d.eps).toReal) (r := (b_rstd.get1 i0).toReal)
    hA0 he heps32_errR hwide
  refine ⟨a, hreadA, ?_, ?_⟩
  · have hread := hrread (i0, ()) (lane_enT htm (j := 0) (h1 := hsub) (_h2 := bd_pos hd))
    rw [lane_gT htm (j := 0) (_h2 := bd_pos hd)] at hread
    refine ⟨b_rstd.get1 i0, ?_, hbrfin, ?_⟩
    · show m.read .f32 (d.rstd.ptr t).addr = some (b_rstd.get (i0, ()))
      exact hread
    · rw [hrstd_tol]
      exact hfinal
  · intro j hj
    have hjbd : j < d.BD := col_lt_bd hd hj
    obtain ⟨hx16fin, hbxfin, hbxR, hq63, hxxfin, hxxerr⟩ := hcol j hjbd
    have hyv : (b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())) = b_rstd.get1 i0 := by
      rw [Tensor.get_col]
      rfl
    have hbrfin' : ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).isFinite = true := by
      rw [hyv]
      exact hbrfin
    have hx2 : (a j).toReal ^ 2 ≤ rowSum a d.D :=
      Finset.single_le_sum (f := fun k => (a k).toReal ^ 2) (fun k _ => sq_nonneg _)
        (Finset.mem_range.mpr hj)
    have hxr_small : |(a j).toReal * (b_rstd.get1 i0).toReal| ≤ 2 := by
      have h := mul_small hAe_pos he0 hx2 hfinal
      have h2 : (1 + (2 : ℝ) ^ (-12 : ℤ)) ≤ 2 := by norm_num
      linarith
    have hmaxy : |(b_x.get2 i0 ⟨j, hjbd⟩).toRat * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toRat|
        ≤ Format.binary32.maxVal := by
      refine rat_le_of_cast ?_
      rw [Rat.cast_abs]
      have hcast : (((b_x.get2 i0 ⟨j, hjbd⟩).toRat
          * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toRat : ℚ) : ℝ)
          = (b_x.get2 i0 ⟨j, hjbd⟩).toReal
            * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toReal := by
        push_cast
        rfl
      rw [hcast, hyv, hbxR]
      exact mul_abs_le hAe_pos he0 hx2 hfinal
    obtain ⟨hmulfin, hmulq⟩ := FloatV.mul_err (f := Format.binary32) Format.binary32_regular
      hbxfin hbrfin' hmaxy
    have hbyeq : b_y.get2 i0 ⟨j, hjbd⟩ =
        FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩) ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))) :=
      (hmul2 (i0, (⟨j, hjbd⟩, ()))).eq (FloatV.isFinite_iff.mp hmulfin).1
    have hyfin : (b_y.get2 i0 ⟨j, hjbd⟩).isFinite = true := by
      rw [hbyeq]
      exact hmulfin
    have hmulqR : |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
        ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal
        - (b_x.get2 i0 ⟨j, hjbd⟩).toReal
          * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toReal|
        ≤ (2 : ℝ) ^ (-24 : ℤ)
            * |(b_x.get2 i0 ⟨j, hjbd⟩).toReal
              * ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ()))).toReal|
          + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := Rat.cast_le (K := ℝ).mpr hmulq
      rw [Format.binary32_u, Format.binary32_eta] at h
      push_cast at h
      simpa only [FloatV.toReal, mul_comm, mul_left_comm, mul_assoc] using h
    have hy_err : |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
        ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal
        - (a j).toReal * (b_rstd.get1 i0).toReal|
        ≤ (2 : ℝ) ^ (-24 : ℤ) * |(a j).toReal * (b_rstd.get1 i0).toReal|
          + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := hmulqR
      simp only [hyv, hbxR] at h
      exact h
    have hysmall : |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
        ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal| ≤ 3 := by
      have h1 : |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
            ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal|
          ≤ |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
              ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal
              - (a j).toReal * (b_rstd.get1 i0).toReal|
            + |(a j).toReal * (b_rstd.get1 i0).toReal| := by
        have h := abs_add_le ((FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
            ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal
            - (a j).toReal * (b_rstd.get1 i0).toReal)
          ((a j).toReal * (b_rstd.get1 i0).toReal)
        have e : ((FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
              ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal
              - (a j).toReal * (b_rstd.get1 i0).toReal)
            + (a j).toReal * (b_rstd.get1 i0).toReal
            = (FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
              ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal := by ring
        rwa [e] at h
      have hu : (2 : ℝ) ^ (-24 : ℤ) * |(a j).toReal * (b_rstd.get1 i0).toReal|
          ≤ (2 : ℝ) ^ (-24 : ℤ) * 2 :=
        mul_le_mul_of_nonneg_left hxr_small (by norm_num)
      have h3 : ((2 : ℝ) ^ (-24 : ℤ) * |(a j).toReal * (b_rstd.get1 i0).toReal|
            + (2 : ℝ) ^ (-150 : ℤ)) + |(a j).toReal * (b_rstd.get1 i0).toReal|
          ≤ ((2 : ℝ) ^ (-24 : ℤ) * 2 + (2 : ℝ) ^ (-150 : ℤ)) + 2 :=
        add_le_add (add_le_add hu le_rfl) hxr_small
      have hconst : ((2 : ℝ) ^ (-24 : ℤ) * 2 + (2 : ℝ) ^ (-150 : ℤ)) + 2 ≤ 3 := by
        norm_num
      exact le_trans h1 (le_trans (add_le_add hy_err le_rfl) (le_trans h3 hconst))
    have hmaxcvt : |(b_y.get2 i0 ⟨j, hjbd⟩).toRat| ≤ Format.bfloat16.maxVal := by
      refine rat_le_of_cast ?_
      rw [Rat.cast_abs]
      have hcast : (((b_y.get2 i0 ⟨j, hjbd⟩).toRat : ℚ) : ℝ) = (b_y.get2 i0 ⟨j, hjbd⟩).toReal := rfl
      rw [hcast, hbyeq]
      refine le_trans hysmall ?_
      rw [Format.bfloat16_maxVal]
      push_cast
      norm_num
    obtain ⟨hcvtfin, hcvterr⟩ := FloatV.cvt_err (g := Format.bfloat16)
      Format.bfloat16_regular hyfin hmaxcvt
    have hcvt16eq : b_y16.get2 i0 ⟨j, hjbd⟩ = (FloatV.cvt (b_y.get2 i0 ⟨j, hjbd⟩) : BF16) := by
      have h := (hcvt16 (i0, (⟨j, hjbd⟩, ()))).eq (FloatV.isFinite_iff.mp hcvtfin).1
      rwa [FloatV.cvtWith_rn] at h
    have hc : |(b_y16.get2 i0 ⟨j, hjbd⟩).toReal
        - (FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
          ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal|
        ≤ (2 : ℝ) ^ (-8 : ℤ)
            * |(FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
              ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal|
          + (2 : ℝ) ^ (-134 : ℤ) := by
      rw [hcvt16eq, hbyeq]
      have h := Rat.cast_le (K := ℝ).mpr hcvterr
      rw [Format.bfloat16_u, Format.bfloat16_eta] at h
      push_cast at h
      rw [hbyeq] at h
      simpa only [FloatV.toReal, mul_comm, mul_left_comm, mul_assoc] using h
    have hyfinal : |(b_y16.get2 i0 ⟨j, hjbd⟩).toReal
        - (a j).toReal * rstdRef a d.D d.eps.toReal|
        ≤ (2 : ℝ) ^ (-7 : ℤ) * |(a j).toReal * rstdRef a d.D d.eps.toReal|
          + (2 : ℝ) ^ (-132 : ℤ) :=
      y_err_bf16 (x := (a j).toReal) (R := rstdRef a d.D d.eps.toReal)
        (r := (b_rstd.get1 i0).toReal)
        (y := (FloatV.mul (b_x.get2 i0 ⟨j, hjbd⟩)
          ((b_rstd.col d.BD).get (i0, (⟨j, hjbd⟩, ())))).toReal)
        (c := (b_y16.get2 i0 ⟨j, hjbd⟩).toReal) hR0 hfinal hy_err hc
    have hyfin16 : (b_y16.get2 i0 ⟨j, hjbd⟩).isFinite = true := by
      rw [hcvt16eq]
      exact hcvtfin
    have hreadj := hyread (i0, (⟨j, hjbd⟩, ())) (lane_enx htm hj)
    rw [lane_gx htm] at hreadj
    unfold YOk
    rw [hy]
    refine ⟨b_y16.get2 i0 ⟨j, hjbd⟩, ?_, hyfin16, ?_⟩
    · show m.read .bf16 (yb.ptr (xIdx d t j)).addr = some (b_y16.get (i0, (⟨j, hjbd⟩, ())))
      exact hreadj
    · have htol : (yTol : ℝ) = (2 : ℝ) ^ (-7 : ℤ) := by rw [yTol]; norm_num
      have hfloor : (yFloor : ℝ) = (2 : ℝ) ^ (-132 : ℤ) := by rw [yFloor]; norm_num
      have hscale : (yTol : ℝ) * (|(a j).toReal| * rstdRef a d.D d.eps.toReal)
          = (yTol : ℝ) * |(a j).toReal * rstdRef a d.D d.eps.toReal| := by
        rw [abs_mul, abs_of_pos hR0]
      rw [hscale, htol, hfloor]
      exact hyfinal
end Claims.L2NormFwd
