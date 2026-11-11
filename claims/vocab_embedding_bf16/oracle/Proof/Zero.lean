import Claim
import TVA.Proof

/-!
# The `bfloat16` `+0` the kernel uses as the load's masked-off value

`other = tl.convert .f32 .bf16 (tl.full [BLOCK_H] (litF32 (ofPat 0)))`: every element is the
`float32` `+0` rounded to `bfloat16`, which is `+0`.
-/

namespace Claims.VocabEmbedding

open TVA

/-- The `float64` `+0`'s pattern is zero's, so its `isZero` is `true`. -/
private theorem f64_ofPat_zero_isZero : (FloatV.ofPat 0 : F64).isZero = true := by
  rw [FloatV.isZero_ofPat (f := Format.binary64) (by norm_num)]
  exact Format.isZero_zero

/-- Rounding the `float32` `+0` to `bfloat16` gives the `bfloat16` `+0`. -/
theorem cvt_zero_f32_bf16 :
    (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32) : BF16) = FloatV.ofPat 0 := by
  rw [FloatV.litF32, ite_eq_left f64_ofPat_zero_isZero]
  show FloatV.ofPat (Format.cvt Format.binary32 Format.bfloat16 .rn false
    ((FloatV.ofPat 0 : F32).pat)) = FloatV.ofPat 0
  rw [FloatV.pat_ofPat (f := Format.binary32) (by norm_num)]
  rw [Format.cvt_of_finite (f := Format.binary32) (g := Format.bfloat16) .rn false (x := 0)
    (Format.isFinite_zero Format.binary32_regular)]
  have hs : Format.binary32.sign 0 = false := by simp [Format.sign]
  rw [hs, Format.toRat_of_isZero (Format.isZero_zero (f := Format.binary32)), Format.roundRat_zero]
  simp [Format.setSign, Format.mag]

/-- Any conversion of the `float32` `+0` to `bfloat16` is `+0`. -/
theorem convRel_bf16_zero {P : Platform} {r : BF16}
    (h : ConvRel P .f32 .bf16 (FloatV.litF32 (FloatV.ofPat 0) : F32) r) :
    r = FloatV.ofPat 0 := by
  have hnf : (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32) : BF16).isNaN = false := by
    rw [cvt_zero_f32_bf16]
    rw [FloatV.isNaN_ofPat (f := Format.bfloat16) (by norm_num)]
    decide
  have h' := (FloatRel.iff_of_std (P := P) (f := .binary32) (g := .bfloat16) (by decide)
    (by decide) (by decide)).mp h
  rw [FloatV.NaNEq.eq h' hnf, cvt_zero_f32_bf16]

end Claims.VocabEmbedding
