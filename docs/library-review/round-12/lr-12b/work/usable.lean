import TVA
/-! lr-12b: can a kernel settle the float facts it needs about constants by evaluation? -/
open TVA TVA.Math.Float

-- `m_i = tl.zeros(...) - float("inf")` (tutorial 06): `+0 − (+∞) = −∞` in `float32`
example : FloatV.sub (FloatV.ofPat 0 : F32) (FloatV.ofPat 0x7f800000) = FloatV.ofPat 0xff800000 := by decide
-- the masked load `other=-float('inf')` into `float32`: `litF32 (−∞) = −∞`
example : FloatV.litF32 (FloatV.ofPat 0xfff0000000000000) = Rules.Num.negInfF32 := by decide
-- the two names of `float32` `−∞`
example : (FloatV.negInf : F32) = Rules.Num.negInfF32 := by decide
-- a finite rounding: `1 + 2^-24` rounds to `1` in `float32`
example : FloatV.add (FloatV.ofPat 0x3f800000 : F32) (FloatV.ofPat 0x33800000) = FloatV.ofPat 0x3f800000 := by decide +kernel
-- `tl.exp`'s `log2 e` constant: `float32(1.44269504)` (a decimal literal through `float64`)
example : (FloatV.litF32 (FloatV.ofPat 0x3FF71547652B82FE) : F32) = FloatV.ofPat 0x3fb8aa3b := by decide +kernel

-- the same two by `decide +kernel`, and the axioms they use
theorem round1 : FloatV.add (FloatV.ofPat 0x3f800000 : F32) (FloatV.ofPat 0x33800000) = FloatV.ofPat 0x3f800000 := by decide +kernel
theorem log2e : (FloatV.litF32 (FloatV.ofPat 0x3FF71547652B82FE) : F32) = FloatV.ofPat 0x3fb8aa3b := by decide +kernel
#print axioms round1
#print axioms log2e
