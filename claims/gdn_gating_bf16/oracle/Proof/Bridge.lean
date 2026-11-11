import Claim
import TVA.Proof

/-!
# The `float32` constants `oneF32` and `twentyF32`

The constants Triton builds for the Python numbers `1.0` and `20.0` are finite `float32` values of value `1` and `20` (they are exact, since both numbers are representable).
-/

namespace Claims.GdnGating

open TVA

/-- The `float64` pattern `0x3FF0000000000000` has value `1`. -/
private theorem binary64_toRat_one : Format.binary64.toRat 0x3FF0000000000000 = 1 := by
  have h : Format.binary64 = ⟨11, 52, Format.Specials.ieee⟩ := rfl
  rw [h]
  simp only [Format.toRat, Format.magVal, Format.mag, Format.sign, Format.signBit,
    Format.emin, Format.bias, Format.pow2_eq]
  norm_num

/-- The `float64` pattern `0x4034000000000000` has value `20`. -/
private theorem binary64_toRat_twenty : Format.binary64.toRat 0x4034000000000000 = 20 := by
  have h : Format.binary64 = ⟨11, 52, Format.Specials.ieee⟩ := rfl
  rw [h]
  simp only [Format.toRat, Format.magVal, Format.mag, Format.sign, Format.signBit,
    Format.emin, Format.bias, Format.pow2_eq]
  norm_num

/-- `1` rounds to itself in `float32`. -/
private theorem binary32_round_one : Format.binary32.toRat
    (Format.binary32.roundRat RoundMode.rn false false 1) = 1 := by
  have hreg : Format.binary32.Regular := by decide
  have hq1 : |(1:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_one]; exact Format.one_le_maxVal hreg
  obtain ⟨-, -, hv⟩ := Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
    (by norm_num) hq1
  rw [abs_one] at hv
  rw [hv]
  have hqexp : Format.binary32.qexp (1:ℚ) = -23 := by
    rw [Format.qexp_eq (by norm_num : (0:ℚ) < 1)]
    have hb32 : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
    rw [hb32]
    norm_num [Format.emin, Format.bias]
  rw [hqexp]
  have hsig := Format.roundSig_exact (rnd := RoundMode.rn) (Q := -23) (K := 2^23) (a := (1:ℚ))
    (by norm_num)
  rw [hsig]
  norm_num

/-- `20` rounds to itself in `float32`. -/
private theorem binary32_round_twenty : Format.binary32.toRat
    (Format.binary32.roundRat RoundMode.rn false false 20) = 20 := by
  have hreg : Format.binary32.Regular := by decide
  have hq20 : |(20:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos (by norm_num : (0:ℚ) < 20)]
    have hb32 : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
    rw [hb32]
    norm_num [Format.maxVal, Format.topSig, Format.topExp, Format.bias, Format.pow2_eq]
  obtain ⟨-, -, hv⟩ := Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
    (by norm_num) hq20
  rw [abs_of_pos (by norm_num : (0:ℚ) < 20)] at hv
  rw [hv]
  have hqexp : Format.binary32.qexp (20:ℚ) = -19 := by
    rw [Format.qexp_eq (by norm_num : (0:ℚ) < 20)]
    have hb32 : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
    rw [hb32]
    norm_num [Format.emin, Format.bias]
  rw [hqexp]
  have hsig := Format.roundSig_exact (rnd := RoundMode.rn) (Q := -19) (K := 20 * 2^19) (a := (20:ℚ))
    (by norm_num)
  rw [hsig]
  norm_num

/-- The `float32` constant `oneF32` has value `1`. -/
theorem oneF32_toRat : oneF32.toRat = 1 := by
  have hz : (FloatV.ofPat (f := Format.binary64) 0x3FF0000000000000).isZero = false := by decide
  have hn : (FloatV.ofPat (f := Format.binary64) 0x3FF0000000000000).isNaN = false := by decide
  have hfin : (FloatV.ofPat (f := Format.binary64) 0x3FF0000000000000).isFinite = true := by decide
  have hw64 : 0x3FF0000000000000 < 2 ^ Format.binary64.width := by decide
  have hfin64 : Format.binary64.isFinite 0x3FF0000000000000 = true := by
    rw [← FloatV.isFinite_ofPat hw64]; exact hfin
  have hsign : Format.binary64.sign 0x3FF0000000000000 = false := by decide
  have hreg : Format.binary32.Regular := by decide
  have hq1 : |(1:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_one]; exact Format.one_le_maxVal hreg
  have hcvt : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x3FF0000000000000 =
      Format.binary32.roundRat RoundMode.rn false (Format.binary64.sign 0x3FF0000000000000)
        (Format.binary64.toRat 0x3FF0000000000000) :=
    Format.cvt_of_finite (f := Format.binary64) (g := Format.binary32) RoundMode.rn false hfin64
  have hbound : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x3FF0000000000000
      < 2 ^ Format.binary32.width := by
    rw [hcvt, hsign, binary64_toRat_one]
    exact (Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
      (by norm_num) hq1).1
  unfold oneF32 FloatV.litF32 FloatV.hostF32 FloatV.cvt
  rw [hz, hn]
  simp only [Bool.false_eq_true, ite_false]
  rw [FloatV.pat_ofPat hw64, FloatV.toRat_ofPat hbound, hcvt, hsign, binary64_toRat_one,
    binary32_round_one]

/-- `oneF32` is finite. -/
theorem oneF32_isFinite : oneF32.isFinite = true := by
  have hz : (FloatV.ofPat (f := Format.binary64) 0x3FF0000000000000).isZero = false := by decide
  have hn : (FloatV.ofPat (f := Format.binary64) 0x3FF0000000000000).isNaN = false := by decide
  have hfin : (FloatV.ofPat (f := Format.binary64) 0x3FF0000000000000).isFinite = true := by decide
  have hw64 : 0x3FF0000000000000 < 2 ^ Format.binary64.width := by decide
  have hfin64 : Format.binary64.isFinite 0x3FF0000000000000 = true := by
    rw [← FloatV.isFinite_ofPat hw64]; exact hfin
  have hsign : Format.binary64.sign 0x3FF0000000000000 = false := by decide
  have hreg : Format.binary32.Regular := by decide
  have hq1 : |(1:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_one]; exact Format.one_le_maxVal hreg
  have hcvt : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x3FF0000000000000 =
      Format.binary32.roundRat RoundMode.rn false (Format.binary64.sign 0x3FF0000000000000)
        (Format.binary64.toRat 0x3FF0000000000000) :=
    Format.cvt_of_finite (f := Format.binary64) (g := Format.binary32) RoundMode.rn false hfin64
  have hbound : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x3FF0000000000000
      < 2 ^ Format.binary32.width := by
    rw [hcvt, hsign, binary64_toRat_one]
    exact (Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
      (by norm_num) hq1).1
  have hfinval : Format.binary32.isFinite
      (Format.binary32.roundRat RoundMode.rn false false 1) = true := by
    exact (Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
      (by norm_num) hq1).2.1
  unfold oneF32 FloatV.litF32 FloatV.hostF32 FloatV.cvt
  rw [hz, hn]
  simp only [Bool.false_eq_true, ite_false]
  rw [FloatV.pat_ofPat hw64, FloatV.isFinite_ofPat hbound, hcvt, hsign, binary64_toRat_one,
    hfinval]

/-- The `float32` constant `twentyF32` has value `20`. -/
theorem twentyF32_toRat : twentyF32.toRat = 20 := by
  have hz : (FloatV.ofPat (f := Format.binary64) 0x4034000000000000).isZero = false := by decide
  have hn : (FloatV.ofPat (f := Format.binary64) 0x4034000000000000).isNaN = false := by decide
  have hfin : (FloatV.ofPat (f := Format.binary64) 0x4034000000000000).isFinite = true := by decide
  have hw64 : 0x4034000000000000 < 2 ^ Format.binary64.width := by decide
  have hfin64 : Format.binary64.isFinite 0x4034000000000000 = true := by
    rw [← FloatV.isFinite_ofPat hw64]; exact hfin
  have hsign : Format.binary64.sign 0x4034000000000000 = false := by decide
  have hreg : Format.binary32.Regular := by decide
  have hq20 : |(20:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos (by norm_num : (0:ℚ) < 20)]
    have hb32 : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
    rw [hb32]
    norm_num [Format.maxVal, Format.topSig, Format.topExp, Format.bias, Format.pow2_eq]
  have hcvt : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x4034000000000000 =
      Format.binary32.roundRat RoundMode.rn false (Format.binary64.sign 0x4034000000000000)
        (Format.binary64.toRat 0x4034000000000000) :=
    Format.cvt_of_finite (f := Format.binary64) (g := Format.binary32) RoundMode.rn false hfin64
  have hbound : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x4034000000000000
      < 2 ^ Format.binary32.width := by
    rw [hcvt, hsign, binary64_toRat_twenty]
    exact (Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
      (by norm_num) hq20).1
  unfold twentyF32 FloatV.litF32 FloatV.hostF32 FloatV.cvt
  rw [hz, hn]
  simp only [Bool.false_eq_true, ite_false]
  rw [FloatV.pat_ofPat hw64, FloatV.toRat_ofPat hbound, hcvt, hsign, binary64_toRat_twenty,
    binary32_round_twenty]

/-- `twentyF32` is finite. -/
theorem twentyF32_isFinite : twentyF32.isFinite = true := by
  have hz : (FloatV.ofPat (f := Format.binary64) 0x4034000000000000).isZero = false := by decide
  have hn : (FloatV.ofPat (f := Format.binary64) 0x4034000000000000).isNaN = false := by decide
  have hfin : (FloatV.ofPat (f := Format.binary64) 0x4034000000000000).isFinite = true := by decide
  have hw64 : 0x4034000000000000 < 2 ^ Format.binary64.width := by decide
  have hfin64 : Format.binary64.isFinite 0x4034000000000000 = true := by
    rw [← FloatV.isFinite_ofPat hw64]; exact hfin
  have hsign : Format.binary64.sign 0x4034000000000000 = false := by decide
  have hreg : Format.binary32.Regular := by decide
  have hq20 : |(20:ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos (by norm_num : (0:ℚ) < 20)]
    have hb32 : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl
    rw [hb32]
    norm_num [Format.maxVal, Format.topSig, Format.topExp, Format.bias, Format.pow2_eq]
  have hcvt : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x4034000000000000 =
      Format.binary32.roundRat RoundMode.rn false (Format.binary64.sign 0x4034000000000000)
        (Format.binary64.toRat 0x4034000000000000) :=
    Format.cvt_of_finite (f := Format.binary64) (g := Format.binary32) RoundMode.rn false hfin64
  have hbound : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false 0x4034000000000000
      < 2 ^ Format.binary32.width := by
    rw [hcvt, hsign, binary64_toRat_twenty]
    exact (Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
      (by norm_num) hq20).1
  have hfinval : Format.binary32.isFinite
      (Format.binary32.roundRat RoundMode.rn false false 20) = true := by
    exact (Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false false
      (by norm_num) hq20).2.1
  unfold twentyF32 FloatV.litF32 FloatV.hostF32 FloatV.cvt
  rw [hz, hn]
  simp only [Bool.false_eq_true, ite_false]
  rw [FloatV.pat_ofPat hw64, FloatV.isFinite_ofPat hbound, hcvt, hsign, binary64_toRat_twenty,
    hfinval]

end Claims.GdnGating
