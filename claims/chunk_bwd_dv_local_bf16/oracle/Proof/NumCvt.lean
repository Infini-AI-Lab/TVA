import Claim
import TVA.Proof
import Proof.Util

/-!
# Rounding a `float32` to `bfloat16`

`tl.cvt` of a finite `float32` of magnitude at most `2^127` is finite and within one rounding
(`2^-8` relative, `2^-134` absolute) of it.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- **The rounding of a `float32` to `bfloat16`** in the reals. -/
theorem cvt_bf16_err {P : F32} (hP : P.isFinite = true) (hmax : |P.toReal| ≤ 2 ^ 127) :
    (FloatV.cvt P : BF16).isFinite = true ∧
      |(FloatV.cvt P : BF16).toReal - P.toReal| ≤ 1 / 2 ^ 8 * |P.toReal| + 1 / 2 ^ 133 := by
  have hmaxQ : |P.toRat| ≤ Format.bfloat16.maxVal := by
    have h : ((|P.toRat| : ℚ) : ℝ) ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
      rw [Format.bfloat16_maxVal_real]
      push_cast
      unfold FloatV.toReal at hmax
      refine hmax.trans ?_
      norm_num
    exact_mod_cast h
  obtain ⟨hfin, herr⟩ := FloatV.cvt_err (g := Format.bfloat16) Format.bfloat16_regular hP hmaxQ
  refine ⟨hfin, ?_⟩
  have h := (by exact_mod_cast herr :
    |((FloatV.cvt P : BF16).toRat : ℝ) - (P.toRat : ℝ)| ≤
      |(P.toRat : ℝ)| * ((Format.bfloat16.u : ℚ) : ℝ) + ((Format.bfloat16.eta : ℚ) : ℝ))
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h
  have e1 : (2 : ℝ) ^ (-8 : ℤ) = 1 / 2 ^ 8 := by norm_num
  have e2 : (2 : ℝ) ^ (-134 : ℤ) ≤ 1 / 2 ^ 133 := by norm_num
  rw [e1] at h
  unfold FloatV.toReal
  linarith

end Claims.ChunkBwdDvLocal
