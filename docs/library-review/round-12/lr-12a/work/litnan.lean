import TVA.Lang.Scalar
import TVA.Rules.Num.Sm90
import TVA.Lang.Random

/-! lr-12a: the library's constants for a Python NaN (`float('nan')`, the double `0x7ff8000000000000`,
and its negation), against the device's patterns in `nan_probe.out` (`tl.full(…, nan, dtype)`:
`f32 7fc00000`, `f16 7e00`, `bf16 7fc0`, `f64 7ff8000000000000`). Also the patterns of a few
constants the facts use. -/
open TVA TVA.Math.Float

def pyNaN : F64 := FloatV.ofPat 0x7ff8000000000000
def pyNegNaN : F64 := FloatV.ofPat 0xfff8000000000000

def hex (n : Nat) : String := String.ofList (Nat.toDigits 16 n)

#eval s!"litF64 nan = {hex (FloatV.litF64 pyNaN).pat}, litF32 nan = {hex (FloatV.litF32 pyNaN).pat}, litF16 nan = {hex (FloatV.litF16 pyNaN).pat}, litBF16 nan = {hex (FloatV.litBF16 pyNaN).pat}"
#eval s!"litF64 -nan = {hex (FloatV.litF64 pyNegNaN).pat}, litF32 -nan = {hex (FloatV.litF32 pyNegNaN).pat}, litF16 -nan = {hex (FloatV.litF16 pyNegNaN).pat}, litBF16 -nan = {hex (FloatV.litBF16 pyNegNaN).pat}"

-- the constants of the facts
#eval (Rules.Num.negInfF32.isInf, Rules.Num.negInfF32.sign, Rules.Num.zeroF32.isZero, Rules.Num.zeroF32.sign)
#eval (Format.e4m3.maxMag, Format.e5m2.maxMag, Format.e4m3.nanMag, Format.e5m2.nanMag, Format.e2m1.maxMag)
#eval (Format.e4m3.isNaN 0x7f, Format.e4m3.isNaN 0x7e, Format.e4m3.isNaN 0xff, Format.e4m3.isInf 0x78,
       Format.e5m2.isInf 0x7c, Format.e5m2.isNaN 0x7d, Format.e2m1.isFinite 0xf)
#eval (Format.e2m1.toRat 1, Format.e2m1.toRat 7, Format.e2m1.toRat 0xf, Format.e4m3.toRat 1, Format.e4m3.toRat 0x7e,
       Format.e5m2.toRat 1, Format.e5m2.toRat 0x7b)

-- `tl.rand`'s scale (`4.6566127342e-10` as a float32 is 0x2fffffff by numpy)
#eval hex TVA.Philox.scale32.pat
