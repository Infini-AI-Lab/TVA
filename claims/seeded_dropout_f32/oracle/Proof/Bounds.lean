import Proof.Consts

/-!
# Bounds on the random numbers and on the divisor

The product `uRef` is a finite `float32` that is not a NaN, and the divisor `dRef = 1 - p` is
finite, and nonzero when the kernel's `p` is below `1`.
-/

namespace Claims.SeededDropout

open TVA
open TVA.Format

/-- `2^31` is below `float32`'s largest value. -/
theorem two_pow_31_le_maxVal : (2 : ℚ) ^ 31 ≤ Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]
  norm_num

/-- `Philox.scale32` is at most `1` in magnitude. -/
theorem scale32_toRat_abs_le_one : |(Philox.scale32 : F32).toRat| ≤ 1 := by
  rw [scale32_toRat, abs_of_pos (by norm_num)]
  norm_num

/-- The `float32` nearest to a non-negative integer below `2^31` is finite. -/
theorem ofRat_isFinite {z : Nat} (hz : z < 2 ^ 31) :
    (FloatV.ofRat (z : ℚ) : F32).isFinite = true := by
  refine (FloatV.ofRat_err Format.binary32_regular ?_).1
  rw [abs_of_nonneg (Nat.cast_nonneg z)]
  have hzq : (z : ℚ) < (2 : ℚ) ^ 31 := by
    have h : ((z : ℕ) : ℚ) < ((2 : ℕ) ^ 31 : ℚ) := by exact_mod_cast hz
    push_cast at h
    exact h
  exact le_trans hzq.le two_pow_31_le_maxVal

/-- Every random number of the claim is finite and not a NaN. -/
theorem uRef_isFinite (seed : Int) (j : Nat) :
    (uRef seed j).isNaN = false ∧ (uRef seed j).isFinite = true := by
  have hA : (FloatV.ofRat (rint seed j : ℚ) : F32).isFinite = true :=
    ofRat_isFinite (rint_lt seed j)
  have hmax : |(FloatV.ofRat (rint seed j : ℚ) : F32).toRat * (Philox.scale32 : F32).toRat|
      ≤ Format.binary32.maxVal := by
    rw [abs_mul]
    refine le_trans (mul_le_mul
      (FloatV.abs_toRat_le_maxVal Format.binary32_regular hA)
      scale32_toRat_abs_le_one (abs_nonneg _) ?_) (mul_one _).le
    exact le_trans zero_le_one (Format.one_le_maxVal Format.binary32_regular)
  have hfin : (uRef seed j).isFinite = true :=
    (FloatV.mul_err Format.binary32_regular hA scale32_isFinite hmax).1
  exact ⟨(FloatV.isFinite_iff.mp hfin).1, hfin⟩

/-- The divisor of a covered case is a finite `float32`. -/
theorem dRef_isFinite {d : Inst} (hd : d.ok) : (dRef d).isFinite = true := by
  have hmax : |(one32 : F32).toRat - (p32 d).toRat| ≤ Format.binary32.maxVal := by
    rw [one32_toRat]
    have h1 : |(1 : ℚ) - (p32 d).toRat| ≤ 1 := by
      rw [abs_le]
      constructor <;> linarith [hd.p_ge, hd.p_le]
    exact le_trans h1 (Format.one_le_maxVal Format.binary32_regular)
  exact (FloatV.sub_err Format.binary32_regular one32_isFinite hd.p_fin hmax).1

/-- When the kernel's `p` is below `1`, the divisor is not zero: no `float32` in `[0, 1)`
is within rounding distance of `0` for `1 - p`, since `1 - p ≥ 2^-24`. -/
theorem dRef_toRat_ne_zero {d : Inst} (hd : d.ok) (hp : (p32 d).toRat < 1) :
    (dRef d).toRat ≠ 0 := by
  intro h0
  have hmax : |(one32 : F32).toRat - (p32 d).toRat| ≤ Format.binary32.maxVal := by
    rw [one32_toRat]
    have h1 : |(1 : ℚ) - (p32 d).toRat| ≤ 1 := by
      rw [abs_le]
      constructor <;> linarith [hd.p_ge, hd.p_le]
    exact le_trans h1 (Format.one_le_maxVal Format.binary32_regular)
  have h1 : |(1 : ℚ) - (p32 d).toRat| = 1 - (p32 d).toRat := by
    rw [abs_of_nonneg]
    linarith [hd.p_le]
  have herr := (FloatV.sub_err Format.binary32_regular one32_isFinite hd.p_fin hmax).2
  rw [dRef_eq] at h0
  rw [one32_toRat, h0, Format.binary32_u, Format.binary32_eta] at herr
  rw [zero_sub, abs_neg, h1] at herr
  have hp24 : (2 : ℚ) ^ (-24 : ℤ) ≤ 1 - (p32 d).toRat := by
    linarith [toRat_le_of_lt_one hp]
  have hstep : (1 - (p32 d).toRat) * (1 - 2 ^ (-24 : ℤ)) ≤ 2 ^ (-150 : ℤ) := by
    nlinarith
  have hfac : (0 : ℚ) ≤ 1 - 2 ^ (-24 : ℤ) := by norm_num
  have hmul := mul_le_mul_of_nonneg_right hp24 hfac
  have hcontra : (2 : ℚ) ^ (-24 : ℤ) * (1 - 2 ^ (-24 : ℤ)) ≤ 2 ^ (-150 : ℤ) :=
    le_trans hmul hstep
  norm_num at hcontra

end Claims.SeededDropout
