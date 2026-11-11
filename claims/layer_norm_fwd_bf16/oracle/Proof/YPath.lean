import Proof.RowVals

/-! Two conversion facts for the output path: converting a `bfloat16` value to `float32`
preserves its real value, and a `NaNEq` out of such a conversion determines the result. -/

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section


/-- **Conversion preserves the real value**: the `float32` image of a finite `bfloat16` value
has the same real value. -/
theorem cvt_toReal {x : BF16} (hx : x.isFinite = true) :
    (FloatV.cvt x : F32).toReal = x.toReal := by
  unfold FloatV.toReal
  exact_mod_cast (cvt_bf16_f32_spec hx).2

/-- **A conversion is finite**: the `float32` image of a finite `bfloat16` value is finite. -/
theorem cvt_isFinite {x : BF16} (hx : x.isFinite = true) :
    (FloatV.cvt x : F32).isFinite = true :=
  (cvt_bf16_f32_spec hx).1

/-- **A `NaNEq` out of a conversion**: a `float32` value that is `NaNEq` to the conversion of a
finite `bfloat16` value is that conversion. -/
theorem naneq_cvt_eq {x : BF16} {y : F32} (h : FloatV.NaNEq (FloatV.cvtWith .rn x) y)
    (hx : x.isFinite = true) : y = FloatV.cvt x := by
  have h1 : (FloatV.cvtWith .rn x : F32).isFinite = true := by
    rw [FloatV.cvtWith_rn]
    exact cvt_isFinite hx
  have h2 := naneq_eq_of_isFinite h h1
  rwa [FloatV.cvtWith_rn] at h2

/-- **A zero converts to a zero**: the conversion of a finite `bfloat16` value with real value
zero has real value zero. -/
theorem cvt_zero_toReal {x : BF16} (hx : x.isFinite = true) (h : x.toReal = 0) :
    (FloatV.cvt x : F32).toReal = 0 := by
  rw [cvt_toReal hx]
  exact h

/-- **A conversion of a zero is a zero**: as `cvt_zero_toReal`, through a `NaNEq`. -/
theorem naneq_cvt_zero {x : BF16} {y : F32} (hc : FloatV.NaNEq (FloatV.cvtWith .rn x) y)
    (hx : x.isFinite = true) (h : x.toReal = 0) : y.toReal = 0 := by
  rw [naneq_cvt_eq hc hx]
  exact cvt_zero_toReal hx h

end

end Claims.LayerNormFwd
