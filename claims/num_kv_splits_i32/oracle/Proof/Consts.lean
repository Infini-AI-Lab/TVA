import Claim
import TVA.Proof

/-!
# Exact `float32` values: the constants `64.0` and `1.0`, and small integers

The constants Triton builds for the Python floats `64.0` and `1.0` are finite `float32` values of
value `64` and `1`; an integer of magnitude below `2^24` converts to `float32` exactly
(`itof_exact`), so `tl.cast(max_seq_len, tl.float32)` and the `float32` core count are exact.
-/

namespace Claims.NumKvSplits

open TVA

/-- A Python float whose `float64` pattern `n` has a nonzero value `q` that `float32` holds
exactly gives the `float32` constant `q`. -/
theorem litF32_of {n : ℕ} {q : ℚ} (hz : (FloatV.ofPat (f := Format.binary64) n).isZero = false)
    (hn : (FloatV.ofPat (f := Format.binary64) n).isNaN = false) (hw : n < 2 ^ Format.binary64.width)
    (hfin64 : Format.binary64.isFinite n = true) (hsign : Format.binary64.sign n = false)
    (hq : Format.binary64.toRat n = q) (hq0 : q ≠ 0) (hqmax : |q| ≤ Format.binary32.maxVal)
    (hround : Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false false q) = q) :
    (FloatV.litF32 (FloatV.ofPat n)).isFinite = true ∧
      (FloatV.litF32 (FloatV.ofPat n)).toRat = q := by
  have hreg : Format.binary32.Regular := by decide
  have hcvt : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false n =
      Format.binary32.roundRat RoundMode.rn false (Format.binary64.sign n)
        (Format.binary64.toRat n) :=
    Format.cvt_of_finite (f := Format.binary64) (g := Format.binary32) RoundMode.rn false hfin64
  obtain ⟨hlt, hfin, -⟩ := Format.roundRat_spec (f := Format.binary32) hreg RoundMode.rn false
    false hq0 hqmax
  have hbound : Format.cvt Format.binary64 Format.binary32 RoundMode.rn false n
      < 2 ^ Format.binary32.width := by
    rw [hcvt, hsign, hq]; exact hlt
  unfold FloatV.litF32 FloatV.hostF32 FloatV.cvt
  rw [hz, hn]
  simp only [Bool.false_eq_true, ite_false]
  rw [FloatV.pat_ofPat hw, FloatV.isFinite_ofPat hbound, FloatV.toRat_ofPat hbound, hcvt, hsign,
    hq]
  exact ⟨hfin, hround⟩

/-- `binary32` written out. -/
theorem binary32_eq : Format.binary32 = ⟨8, 23, Format.Specials.ieee⟩ := rfl

/-- `2^24` is below `float32`'s largest value. -/
theorem two24_le_maxVal32 : (2 : ℚ) ^ 24 ≤ Format.binary32.maxVal := by
  rw [binary32_eq]
  norm_num [Format.maxVal, Format.topSig, Format.topExp, Format.bias, Format.pow2_eq]

/-- An integer of magnitude at most `2^24` is a `float32` value: it rounds to itself. -/
theorem round32_int {z : ℤ} (hz0 : z ≠ 0) (hz : |z| ≤ 2 ^ 24) :
    Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false false (z : ℚ)) = true ∧
      Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false false (z : ℚ)) = z := by
  have hf := Format.binary32_regular
  have hq0 : (z : ℚ) ≠ 0 := by exact_mod_cast hz0
  have hzq : |(z : ℚ)| ≤ 2 ^ 24 := by exact_mod_cast hz
  have hq : |(z : ℚ)| ≤ Format.binary32.maxVal := hzq.trans two24_le_maxVal32
  obtain ⟨-, h2, h3⟩ := Format.roundRat_spec hf .rn false false hq0 hq
  refine ⟨h2, ?_⟩
  rw [h3]
  have ha : 0 < |(z : ℚ)| := abs_pos.mpr hq0
  have hlog : Int.log 2 |(z : ℚ)| ≤ 24 := by
    have : |(z : ℚ)| < (2 : ℚ) ^ (25 : ℤ) := by
      have : (2 : ℚ) ^ (25 : ℤ) = 2 ^ 25 := by norm_num
      rw [this]; linarith
    have := Format.log_lt_of_lt ha this
    omega
  have hQ : Format.binary32.qexp |(z : ℚ)| ≤ 1 := by
    rw [Format.qexp_eq ha]
    have : Format.binary32.emin = -126 := by decide
    have hm : (Format.binary32.m : ℤ) = 23 := rfl
    rw [this, hm]; omega
  set Q := Format.binary32.qexp |(z : ℚ)| with hQdef
  -- at `Q = 1` the magnitude is `2^24` exactly, an even multiple of `2`
  by_cases hQ1 : Q = 1
  · have hz24 : |(z : ℚ)| = 2 ^ 24 := by
      have hge : (2 : ℚ) ^ (24 : ℤ) ≤ |(z : ℚ)| := by
        have hlog' : 24 ≤ Int.log 2 |(z : ℚ)| := by
          rw [hQdef, Format.qexp_eq ha] at hQ1
          have : Format.binary32.emin = -126 := by decide
          have hm : (Format.binary32.m : ℤ) = 23 := rfl
          rw [this, hm] at hQ1; omega
        calc (2 : ℚ) ^ (24 : ℤ) ≤ (2 : ℚ) ^ Int.log 2 |(z : ℚ)| :=
              zpow_le_zpow_right₀ (by norm_num) hlog'
          _ ≤ |(z : ℚ)| := Format.two_zpow_log_le ha
      have : (2 : ℚ) ^ (24 : ℤ) = 2 ^ 24 := by norm_num
      rw [this] at hge; linarith
    have hK : |(z : ℚ)| * (2 : ℚ) ^ (-Q) = ((2 ^ 23 : ℕ) : ℚ) := by
      rw [hQ1, hz24]; norm_num
    rw [Format.roundSig_exact .rn hK]
    by_cases hneg : (z : ℚ) < 0
    · rw [ite_eq_left hneg, abs_of_neg hneg]; ring
    · rw [ite_eq_right hneg, abs_of_nonneg (not_lt.mp hneg), one_mul]
  · have hQ0 : Q ≤ 0 := by omega
    have hK : |(z : ℚ)| * (2 : ℚ) ^ (-Q) = ((z.natAbs * 2 ^ (-Q).toNat : ℕ) : ℚ) := by
      push_cast
      rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
      congr 1
      rw [Nat.cast_natAbs, Int.cast_abs]
    rw [Format.roundSig_exact .rn hK]
    by_cases hneg : (z : ℚ) < 0
    · rw [ite_eq_left hneg, abs_of_neg hneg]; ring
    · rw [ite_eq_right hneg, abs_of_nonneg (not_lt.mp hneg), one_mul]

/-- **An integer of magnitude at most `2^24` converts to `float32` exactly**, by any of the
alternatives `tl.ItofRel` allows. -/
theorem itof_exact {z : ℤ} (hz : |z| ≤ 2 ^ 24) {r : F32} (h : tl.ItofRel .binary32 z r) :
    r.isFinite = true ∧ r.toRat = z := by
  have hz53 : |z| < 2 ^ 53 := lt_of_le_of_lt hz (by norm_num)
  rw [tl.itofRel_binary32 hz53] at h
  subst h
  by_cases hz0 : z = 0
  · subst hz0
    simp only [Int.cast_zero]
    rw [FloatV.ofRat_zero]
    exact ⟨by decide, by simp⟩
  obtain ⟨hfin, hval⟩ := round32_int hz0 hz
  have hlt := (Format.roundRat_spec (f := Format.binary32) Format.binary32_regular .rn false false
    (by exact_mod_cast hz0) ((show |(z : ℚ)| ≤ 2 ^ 24 by exact_mod_cast hz).trans
      two24_le_maxVal32)).1
  unfold FloatV.ofRat
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨hfin, hval⟩

/-- The constant `c64` (`64.0`) is the finite value `64`. -/
theorem c64_val : c64.isFinite = true ∧ c64.toRat = 64 := by
  have h := round32_int (z := 64) (by norm_num) (by norm_num)
  push_cast at h
  refine litF32_of (by decide) (by decide) (by decide) (by decide) (by decide) ?_ (by norm_num)
    (by
      have := two24_le_maxVal32
      rw [abs_of_pos (by norm_num : (0 : ℚ) < 64)]; linarith) h.2
  rw [show Format.binary64 = ⟨11, 52, Format.Specials.ieee⟩ from rfl]
  simp only [Format.toRat, Format.magVal, Format.mag, Format.sign, Format.signBit,
    Format.emin, Format.bias, Format.pow2_eq]
  norm_num

/-- The constant `c1` (`1.0`) is the finite value `1`. -/
theorem c1_val : c1.isFinite = true ∧ c1.toRat = 1 := by
  have h := round32_int (z := 1) (by norm_num) (by norm_num)
  push_cast at h
  refine litF32_of (by decide) (by decide) (by decide) (by decide) (by decide) ?_ (by norm_num)
    (by
      have := two24_le_maxVal32
      rw [abs_of_pos (by norm_num : (0 : ℚ) < 1)]; linarith) h.2
  rw [show Format.binary64 = ⟨11, 52, Format.Specials.ieee⟩ from rfl]
  simp only [Format.toRat, Format.magVal, Format.mag, Format.sign, Format.signBit,
    Format.emin, Format.bias, Format.pow2_eq]
  norm_num

end Claims.NumKvSplits
