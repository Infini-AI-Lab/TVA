import Claim
import TVA.Proof

/-!
# `float32` operations on finite operands give no NaN

Rounding never produces a NaN pattern, so the sum and the product of two finite `float32` values are not NaN: they are finite or an infinity.
-/

namespace Claims.GdnGating

open TVA

/-! ## Helpers about `binary32` patterns -/

/-- The magnitude of an infinity of `binary32` is below its sign bit. -/
theorem binary32_infMag_lt_signBit : Format.binary32.infMag < Format.binary32.signBit := by
  show (2 ^ 8 - 1) * 2 ^ 23 < 2 ^ (8 + 23)
  rw [pow_add]
  exact Nat.mul_lt_mul_of_pos_right (by norm_num) (by norm_num)

/-- `+0` is not a NaN. -/
theorem binary32_isNaN_zero : Format.binary32.isNaN 0 = false := by decide

/-- The magnitude of an infinity is not a NaN. -/
theorem binary32_isNaN_infMag : Format.binary32.isNaN Format.binary32.infMag = false := by
  decide

/-- A positive value's rounding is not a NaN. -/
theorem binary32_roundMag_isNaN_false {a : ℚ} (ha : 0 < a) (rnd : RoundMode) (sat : Bool) :
    Format.binary32.isNaN (Format.binary32.roundMag rnd sat a) = false := by
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := Format.binary32) rnd ha
  unfold Format.roundMag Format.encode
  rw [Format.pow2_eq]
  by_cases hmax : (Format.roundSig rnd a (Format.binary32.qexp a) : ℚ) *
      (2 : ℚ) ^ (Format.binary32.qexp a) ≤ Format.binary32.maxVal
  · rw [ite_eq_left hmax]
    have hle := Format.pack_le_maxMag (f := Format.binary32) Format.binary32_regular hN hQ hsub hmax
    have hfin := Format.isFinite_of_mag_le (f := Format.binary32) Format.binary32_regular
      (n := Format.binary32.pack (Format.roundSig rnd a (Format.binary32.qexp a))
        (Format.binary32.qexp a))
      (le_trans (Format.mag_le (f := Format.binary32) _) hle)
    exact ((Format.isFinite_iff _).mp hfin).1
  · rw [ite_eq_right hmax]
    by_cases hsat : (sat || rnd == RoundMode.rz) = true
    · rw [ite_eq_left hsat]
      have hfin := Format.isFinite_of_mag_le (f := Format.binary32) Format.binary32_regular
        (n := Format.binary32.maxMag) (Format.mag_le (f := Format.binary32) _)
      exact ((Format.isFinite_iff _).mp hfin).1
    · rw [ite_eq_right hsat]
      exact binary32_isNaN_infMag

/-- A sign with a magnitude below the sign bit is `setSign`. -/
theorem binary32_add_sign (q : ℚ) (M : ℕ) (hM : M < Format.binary32.signBit) :
    (if q < 0 then Format.binary32.signBit else 0) + M =
      Format.binary32.setSign (decide (q < 0)) M := by
  rw [Format.setSign_of_lt (f := Format.binary32) (s := decide (q < 0)) (M := M) hM]
  by_cases h : q < 0 <;> simp [h]

/-! ## The five theorems -/

theorem roundMag_lt_signBit {rnd : RoundMode} {sat : Bool} {a : ℚ} (ha : 0 < a) :
    Format.binary32.roundMag rnd sat a < Format.binary32.signBit := by
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := Format.binary32) rnd ha
  unfold Format.roundMag Format.encode
  rw [Format.pow2_eq]
  by_cases hmax : (Format.roundSig rnd a (Format.binary32.qexp a) : ℚ) *
      (2 : ℚ) ^ (Format.binary32.qexp a) ≤ Format.binary32.maxVal
  · rw [ite_eq_left hmax]
    exact lt_of_le_of_lt
      (Format.pack_le_maxMag (f := Format.binary32) Format.binary32_regular hN hQ hsub hmax)
      (Format.maxMag_lt Format.binary32_regular)
  · rw [ite_eq_right hmax]
    by_cases hsat : (sat || rnd == RoundMode.rz) = true
    · rw [ite_eq_left hsat]
      exact Format.maxMag_lt Format.binary32_regular
    · rw [ite_eq_right hsat]
      exact binary32_infMag_lt_signBit

/-- A `float32` `roundRat` pattern fits the width. -/
theorem roundRat_lt_width (rnd : RoundMode) (sat zs : Bool) (q : ℚ) :
    Format.binary32.roundRat rnd sat zs q < 2 ^ Format.binary32.width := by
  by_cases hq : q = 0
  · subst hq
    rw [Format.roundRat_zero]
    exact Format.setSign_lt (f := Format.binary32) zs 0
  · rw [Format.roundRat_of_ne (f := Format.binary32) hq]
    have ha : 0 < |q| := abs_pos.mpr hq
    have hlt := roundMag_lt_signBit (rnd := rnd) (sat := sat) ha
    have hle : (if q < 0 then Format.binary32.signBit else 0) ≤ Format.binary32.signBit := by
      split <;> omega
    have h2 := Format.two_signBit (f := Format.binary32)
    rw [← h2]
    omega

/-- **`float32` rounding never gives a NaN.** -/
theorem roundRat_isNaN_false (rnd : RoundMode) (sat zs : Bool) (q : ℚ) :
    Format.binary32.isNaN (Format.binary32.roundRat rnd sat zs q) = false := by
  by_cases hq : q = 0
  · subst hq
    rw [Format.roundRat_zero, Format.isNaN_setSign]
    exact binary32_isNaN_zero
  · rw [Format.roundRat_of_ne (f := Format.binary32) hq]
    have ha : 0 < |q| := abs_pos.mpr hq
    have hlt := roundMag_lt_signBit (rnd := rnd) (sat := sat) ha
    rw [binary32_add_sign q _ hlt, Format.isNaN_setSign]
    exact binary32_roundMag_isNaN_false ha rnd sat

/-- The sum of two finite `float32` values is not a NaN. -/
theorem add_isNaN_false {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    (FloatV.add x y).isNaN = false := by
  unfold FloatV.add
  rw [Format.add_of_finite (f := Format.binary32) RoundMode.rn hx hy]
  have hlt := roundRat_lt_width RoundMode.rn false
    (Format.binary32.sign x.pat && Format.binary32.sign y.pat)
    (Format.binary32.toRat x.pat + Format.binary32.toRat y.pat)
  rw [FloatV.isNaN_ofPat hlt]
  exact roundRat_isNaN_false RoundMode.rn false _ _

/-- The product of two finite `float32` values is not a NaN. -/
theorem mul_isNaN_false {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    (FloatV.mul x y).isNaN = false := by
  unfold FloatV.mul
  rw [Format.mul_of_finite (f := Format.binary32) RoundMode.rn hx hy]
  have hlt := roundRat_lt_width RoundMode.rn false
    (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
    (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat)
  rw [FloatV.isNaN_ofPat hlt]
  exact roundRat_isNaN_false RoundMode.rn false _ _

end Claims.GdnGating
