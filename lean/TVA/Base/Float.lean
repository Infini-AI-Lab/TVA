import TVA.Base.Float.Format
import TVA.Base.Float.Round
import TVA.Base.Float.Arith
import TVA.Base.Float.Value
import TVA.Base.Float.Literal

/-!
# Floating point

TVA's model of IEEE 754 binary floating point and of the OCP 8-bit and 4-bit formats, exact on
rationals. Read in this order:

* `TVA.Base.Float.Format`: `Format` (`binary16`, `bfloat16`, `binary32`, `binary64`, `e5m2`,
  `e4m3`, `e2m1`), how a bit pattern encodes a value (`toRat`, `isNaN`, `isInf`, …), and the
  constants `maxVal`, `u`, `eta`.
* `TVA.Base.Float.Round`: `RoundMode` and `roundRat`, the rounding of an exact value.
* `TVA.Base.Float.Arith`: the IEEE operations on patterns (`add`, `mul`, `fma`, `div`, `sqrt`,
  `maxNum`, `cmp`, `cvt`, …).
* `TVA.Base.Float.Value`: `FloatV f`, a value of a float dtype, with the operations rounding to
  nearest even, `ofRat`, `cvt`, `cvtWith`, `cvtSat`, `toRat`, `toReal`, `NaNEq`; `F16`, `BF16`,
  `F32`, `F64`, `F8E5`, `F8E4`.
* `TVA.Base.Float.Literal`: the constants Triton builds for Python numbers (`litF32`, …) and the
  host's conversion `hostF32`.
-/
