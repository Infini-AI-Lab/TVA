import Claim
import TVA.Proof

/-!
# The `float32` constants of the model, as real numbers

The model builds its `float32` constants from `float64` literals through `c32`.  These lemmas
record each constant's value, whether it is finite, infinite or NaN, and the error of each
against the exact real number it stands for.
-/

namespace Claims.MoeRouter

open TVA
open TVA

/-! ## Zero -/

/-! ## The `bfloat16` zero the `x` load pads with -/

/-! ## c30 -/

/-- The constant `30.0` is a finite `float32`. -/
theorem c30_isFinite : (c32 0x403E000000000000 : F32).isFinite = true := by
  have hz : (FloatV.ofPat 0x403E000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x403E000000000000 : F64).isNaN = false := by decide
  have hfin : (FloatV.ofPat 0x403E000000000000 : F64).isFinite = true := by decide
  have hval : (FloatV.ofPat 0x403E000000000000 : F64).toRat = 30 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hc : (c32 0x403E000000000000 : F32) = FloatV.cvt (FloatV.ofPat 0x403E000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hmax : |(FloatV.ofPat 0x403E000000000000 : F64).toRat| ≤ Format.binary32.maxVal := by
    rw [hval, Format.binary32_maxVal]; norm_num
  obtain ⟨hcf, herr⟩ := FloatV.cvt_err (f := Format.binary64)
    (g := Format.binary32) (by decide) hfin hmax
  rw [hc]
  exact hcf

/-! ## c2 -/

/-- The constant `2.0` is a finite `float32`. -/
theorem c2_isFinite : (c32 0x4000000000000000 : F32).isFinite = true := by
  have hz : (FloatV.ofPat 0x4000000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x4000000000000000 : F64).isNaN = false := by decide
  have hfin : (FloatV.ofPat 0x4000000000000000 : F64).isFinite = true := by decide
  have hval : (FloatV.ofPat 0x4000000000000000 : F64).toRat = 2 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hc : (c32 0x4000000000000000 : F32) = FloatV.cvt (FloatV.ofPat 0x4000000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hmax : |(FloatV.ofPat 0x4000000000000000 : F64).toRat| ≤ Format.binary32.maxVal := by
    rw [hval, Format.binary32_maxVal]; norm_num
  obtain ⟨hcf, herr⟩ := FloatV.cvt_err (f := Format.binary64)
    (g := Format.binary32) (by decide) hfin hmax
  rw [hc]
  exact hcf

/-! ## c1 -/

/-- The constant `1.0` is a finite `float32`. -/
theorem c1_isFinite : (c32 0x3FF0000000000000 : F32).isFinite = true := by
  have hz : (FloatV.ofPat 0x3FF0000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x3FF0000000000000 : F64).isNaN = false := by decide
  have hfin : (FloatV.ofPat 0x3FF0000000000000 : F64).isFinite = true := by decide
  have hval : (FloatV.ofPat 0x3FF0000000000000 : F64).toRat = 1 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hc : (c32 0x3FF0000000000000 : F32) = FloatV.cvt (FloatV.ofPat 0x3FF0000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hmax : |(FloatV.ofPat 0x3FF0000000000000 : F64).toRat| ≤ Format.binary32.maxVal := by
    rw [hval, Format.binary32_maxVal]; norm_num
  obtain ⟨hcf, herr⟩ := FloatV.cvt_err (f := Format.binary64)
    (g := Format.binary32) (by decide) hfin hmax
  rw [hc]
  exact hcf

/-! ## Minus infinity, the `tl.where` fill of the second argmax -/

/-- The constant `-inf` is not a NaN. -/
theorem cninf_isNaN : (c32 0xFFF0000000000000 : F32).isNaN = false := by
  have hc : (c32 0xFFF0000000000000 : F32) = FloatV.ofPat 0xFF800000 := by
    have hz : (FloatV.ofPat 0xFFF0000000000000 : F64).isZero = false := by decide
    have hn : (FloatV.ofPat 0xFFF0000000000000 : F64).isNaN = false := by decide
    have hn2 : Format.binary64.isNaN (FloatV.ofPat 0xFFF0000000000000 : F64).pat = false := by decide
    have hi : Format.binary64.isInf (FloatV.ofPat 0xFFF0000000000000 : F64).pat = true := by decide
    have hs : Format.binary64.sign (FloatV.ofPat 0xFFF0000000000000 : F64).pat = true := by decide
    unfold c32 FloatV.litF32 FloatV.hostF32 FloatV.cvt Format.cvt
    simp only [hz, hn, hn2, hi, hs, Bool.false_eq_true, ite_false, ite_true]
    rw [show Format.binary32.infPat true = 0xFF800000 from by decide]
  rw [hc]
  decide

/-- Minus infinity is below every finite value in the order `tl.argmaxAll` uses. -/
theorem cninf_sortKey_lt {x : F32} (hx : x.isFinite = true) :
    FloatV.sortKey (c32 0xFFF0000000000000) < FloatV.sortKey x := by
  have hc : (c32 0xFFF0000000000000 : F32) = FloatV.ofPat 0xFF800000 := by
    have hz : (FloatV.ofPat 0xFFF0000000000000 : F64).isZero = false := by decide
    have hn : (FloatV.ofPat 0xFFF0000000000000 : F64).isNaN = false := by decide
    have hn2 : Format.binary64.isNaN (FloatV.ofPat 0xFFF0000000000000 : F64).pat = false := by decide
    have hi : Format.binary64.isInf (FloatV.ofPat 0xFFF0000000000000 : F64).pat = true := by decide
    have hs : Format.binary64.sign (FloatV.ofPat 0xFFF0000000000000 : F64).pat = true := by decide
    unfold c32 FloatV.litF32 FloatV.hostF32 FloatV.cvt Format.cvt
    simp only [hz, hn, hn2, hi, hs, Bool.false_eq_true, ite_false, ite_true]
    rw [show Format.binary32.infPat true = 0xFF800000 from by decide]
  rw [hc]
  unfold FloatV.sortKey
  rw [FloatV.pat_ofPat (by decide : 0xFF800000 < 2 ^ Format.binary32.width)]
  rw [← Format.toRat_lt_iff_orderKey (f := Format.binary32) (0xFF800000) x.pat]
  have hin : Format.binary32.isInf 0xFF800000 = true := by decide
  have hs : Format.binary32.sign 0xFF800000 = true := by decide
  rw [Format.toRat_of_isInf (f := Format.binary32) (by decide) hin, hs]
  simp only [ite_true, show ((Format.binary32.bias : ℤ) + 1) = 128 from by decide]
  have habs := Format.abs_toRat_le_maxVal (f := Format.binary32) (by decide) hx
  rw [Format.binary32_maxVal] at habs
  rw [abs_le] at habs
  have hbig : ((2 ^ 24 - 1 : ℚ) * 2 ^ 104) < (2:ℚ)^(128:ℤ) := by norm_num
  linarith [habs.1]

/-! ## Negation -/


/-! ## The exact values of `30.0`, `2.0` and `1.0` -/

/-! ## The exact values of the constants -/

/-- The constant `30.0` has the value `30`. -/
theorem c30_val : (c32 0x403E000000000000 : F32).toReal = 30 := by
  have hz : (FloatV.ofPat 0x403E000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x403E000000000000 : F64).isNaN = false := by decide
  have hc : (c32 0x403E000000000000 : F32)
      = FloatV.cvt (FloatV.ofPat 0x403E000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hK : (30:ℚ) * (2:ℚ)^(-(-19:ℤ)) = (15728640:ℕ) := by norm_num
  have hq0 : (30:ℚ) ≠ 0 := by norm_num
  have hq : |(30:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_nonneg (by norm_num : (0:ℚ) ≤ 30), Format.binary32_maxVal]
    norm_num
  have hround : Format.binary32.toRat
      (Format.binary32.roundRat RoundMode.rn false false (30:ℚ)) = 30 := by
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec
      (f := Format.binary32) (by decide) RoundMode.rn false false hq0 hq
    rw [hval, abs_of_nonneg (by norm_num : (0:ℚ) ≤ 30)]
    have hqe : Format.binary32.qexp (30:ℚ) = -19 := by
      have he : Format.binary32.emin = -126 := by decide
      have hm : (Format.binary32.m : ℤ) = 23 := rfl
      have hlog : Int.log 2 (30:ℚ) = 4 := by
        have h1 : (2:ℚ)^(4:ℤ) ≤ 30 := by norm_num
        have h2 : (30:ℚ) < (2:ℚ)^(5:ℤ) := by norm_num
        have h3 : (4:ℤ) ≤ Int.log 2 (30:ℚ) := Format.le_log_of_le (by norm_num) h1
        have h4 : Int.log 2 (30:ℚ) < 5 := Format.log_lt_of_lt (by norm_num) h2
        omega
      rw [Format.qexp_eq (by norm_num), hlog, he, hm]
      rw [max_eq_left (by norm_num : (-126:ℤ) ≤ 4)]
      norm_num
    rw [hqe]
    rw [Format.roundSig_exact RoundMode.rn hK]
    norm_num
  have hval64 : (FloatV.ofPat 0x403E000000000000 : F64).toRat = 30 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hsign : (FloatV.ofPat 0x403E000000000000 : F64).sign = false := by decide
  have hfin : (FloatV.ofPat 0x403E000000000000 : F64).isFinite = true := by decide
  have hfin' : Format.binary64.isFinite
      (FloatV.ofPat 0x403E000000000000 : F64).pat = true := hfin
  have hsign' : Format.binary64.sign
      (FloatV.ofPat 0x403E000000000000 : F64).pat = false := hsign
  have hval' : Format.binary64.toRat
      (FloatV.ofPat 0x403E000000000000 : F64).pat = 30 := hval64
  have hcvt : (FloatV.cvt (FloatV.ofPat 0x403E000000000000 : F64) : F32)
      = FloatV.ofPat (Format.binary32.roundRat RoundMode.rn false false (30:ℚ)) := by
    unfold FloatV.cvt
    rw [Format.cvt_of_finite (f := Format.binary64)
      (g := Format.binary32) RoundMode.rn false hfin', hsign', hval']
  rw [hc, hcvt, FloatV.toReal]
  rw [FloatV.toRat_ofPat (f := Format.binary32) (by
    exact (Format.roundRat_spec (f := Format.binary32) (by decide)
      RoundMode.rn false false hq0 hq).1)]
  rw [hround]
  norm_num

/-- The constant `2.0` has the value `2`. -/
theorem c2_val : (c32 0x4000000000000000 : F32).toReal = 2 := by
  have hz : (FloatV.ofPat 0x4000000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x4000000000000000 : F64).isNaN = false := by decide
  have hc : (c32 0x4000000000000000 : F32)
      = FloatV.cvt (FloatV.ofPat 0x4000000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hK : (2:ℚ) * (2:ℚ)^(-(-22:ℤ)) = (8388608:ℕ) := by norm_num
  have hq0 : (2:ℚ) ≠ 0 := by norm_num
  have hq : |(2:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_nonneg (by norm_num : (0:ℚ) ≤ 2), Format.binary32_maxVal]
    norm_num
  have hround : Format.binary32.toRat
      (Format.binary32.roundRat RoundMode.rn false false (2:ℚ)) = 2 := by
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec
      (f := Format.binary32) (by decide) RoundMode.rn false false hq0 hq
    rw [hval, abs_of_nonneg (by norm_num : (0:ℚ) ≤ 2)]
    have hqe : Format.binary32.qexp (2:ℚ) = -22 := by
      have he : Format.binary32.emin = -126 := by decide
      have hm : (Format.binary32.m : ℤ) = 23 := rfl
      have hlog : Int.log 2 (2:ℚ) = 1 := by
        have h1 : (2:ℚ)^(1:ℤ) ≤ 2 := by norm_num
        have h2 : (2:ℚ) < (2:ℚ)^(2:ℤ) := by norm_num
        have h3 : (1:ℤ) ≤ Int.log 2 (2:ℚ) := Format.le_log_of_le (by norm_num) h1
        have h4 : Int.log 2 (2:ℚ) < 2 := Format.log_lt_of_lt (by norm_num) h2
        omega
      rw [Format.qexp_eq (by norm_num), hlog, he, hm]
      rw [max_eq_left (by norm_num : (-126:ℤ) ≤ 1)]
      norm_num
    rw [hqe]
    rw [Format.roundSig_exact RoundMode.rn hK]
    norm_num
  have hval64 : (FloatV.ofPat 0x4000000000000000 : F64).toRat = 2 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hsign : (FloatV.ofPat 0x4000000000000000 : F64).sign = false := by decide
  have hfin : (FloatV.ofPat 0x4000000000000000 : F64).isFinite = true := by decide
  have hfin' : Format.binary64.isFinite
      (FloatV.ofPat 0x4000000000000000 : F64).pat = true := hfin
  have hsign' : Format.binary64.sign
      (FloatV.ofPat 0x4000000000000000 : F64).pat = false := hsign
  have hval' : Format.binary64.toRat
      (FloatV.ofPat 0x4000000000000000 : F64).pat = 2 := hval64
  have hcvt : (FloatV.cvt (FloatV.ofPat 0x4000000000000000 : F64) : F32)
      = FloatV.ofPat (Format.binary32.roundRat RoundMode.rn false false (2:ℚ)) := by
    unfold FloatV.cvt
    rw [Format.cvt_of_finite (f := Format.binary64)
      (g := Format.binary32) RoundMode.rn false hfin', hsign', hval']
  rw [hc, hcvt, FloatV.toReal]
  rw [FloatV.toRat_ofPat (f := Format.binary32) (by
    exact (Format.roundRat_spec (f := Format.binary32) (by decide)
      RoundMode.rn false false hq0 hq).1)]
  rw [hround]
  norm_num

/-- The constant `1.0` has the value `1`. -/
theorem c1_val : (c32 0x3FF0000000000000 : F32).toReal = 1 := by
  have hz : (FloatV.ofPat 0x3FF0000000000000 : F64).isZero = false := by decide
  have hn : (FloatV.ofPat 0x3FF0000000000000 : F64).isNaN = false := by decide
  have hc : (c32 0x3FF0000000000000 : F32)
      = FloatV.cvt (FloatV.ofPat 0x3FF0000000000000 : F64) := by
    unfold c32 FloatV.litF32 FloatV.hostF32
    simp only [hz, hn, Bool.false_eq_true, ite_false]
  have hK : (1:ℚ) * (2:ℚ)^(-(-23:ℤ)) = (8388608:ℕ) := by norm_num
  have hq0 : (1:ℚ) ≠ 0 := by norm_num
  have hq : |(1:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_nonneg (by norm_num : (0:ℚ) ≤ 1), Format.binary32_maxVal]
    norm_num
  have hround : Format.binary32.toRat
      (Format.binary32.roundRat RoundMode.rn false false (1:ℚ)) = 1 := by
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec
      (f := Format.binary32) (by decide) RoundMode.rn false false hq0 hq
    rw [hval, abs_of_nonneg (by norm_num : (0:ℚ) ≤ 1)]
    have hqe : Format.binary32.qexp (1:ℚ) = -23 := by
      have he : Format.binary32.emin = -126 := by decide
      have hm : (Format.binary32.m : ℤ) = 23 := rfl
      have hlog : Int.log 2 (1:ℚ) = 0 := by
        have h1 : (2:ℚ)^(0:ℤ) ≤ 1 := by norm_num
        have h2 : (1:ℚ) < (2:ℚ)^(1:ℤ) := by norm_num
        have h3 : (0:ℤ) ≤ Int.log 2 (1:ℚ) := Format.le_log_of_le (by norm_num) h1
        have h4 : Int.log 2 (1:ℚ) < 1 := Format.log_lt_of_lt (by norm_num) h2
        omega
      rw [Format.qexp_eq (by norm_num), hlog, he, hm]
      rw [max_eq_left (by norm_num : (-126:ℤ) ≤ 0)]
      norm_num
    rw [hqe]
    rw [Format.roundSig_exact RoundMode.rn hK]
    norm_num
  have hval64 : (FloatV.ofPat 0x3FF0000000000000 : F64).toRat = 1 := by
    rw [FloatV.toRat_ofPat (by norm_num [Format.binary64, Format.width])]
    simp only [Format.toRat, Format.mag, Format.magVal,
      Format.pow2_eq, Format.binary64, Format.sign,
      Format.signBit, Format.bias, Format.emin]
    norm_num
  have hsign : (FloatV.ofPat 0x3FF0000000000000 : F64).sign = false := by decide
  have hfin : (FloatV.ofPat 0x3FF0000000000000 : F64).isFinite = true := by decide
  have hfin' : Format.binary64.isFinite
      (FloatV.ofPat 0x3FF0000000000000 : F64).pat = true := hfin
  have hsign' : Format.binary64.sign
      (FloatV.ofPat 0x3FF0000000000000 : F64).pat = false := hsign
  have hval' : Format.binary64.toRat
      (FloatV.ofPat 0x3FF0000000000000 : F64).pat = 1 := hval64
  have hcvt : (FloatV.cvt (FloatV.ofPat 0x3FF0000000000000 : F64) : F32)
      = FloatV.ofPat (Format.binary32.roundRat RoundMode.rn false false (1:ℚ)) := by
    unfold FloatV.cvt
    rw [Format.cvt_of_finite (f := Format.binary64)
      (g := Format.binary32) RoundMode.rn false hfin', hsign', hval']
  rw [hc, hcvt, FloatV.toReal]
  rw [FloatV.toRat_ofPat (f := Format.binary32) (by
    exact (Format.roundRat_spec (f := Format.binary32) (by decide)
      RoundMode.rn false false hq0 hq).1)]
  rw [hround]
  norm_num

end Claims.MoeRouter
