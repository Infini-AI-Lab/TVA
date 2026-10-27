import TVA.Math.Float.Format.Binary32
import TVA.Math.Float.Format.Convert

/-!
# Bridges from the existing float definitions to `TVA.Math.Float.Format`

The generic formats of `TVA.Math.Float.Format` agree with the definitions the library already
had, so that the conformance harness and later numeric work can use either:

* **binary32** (`TVA.Math.Float.Format.Binary32`): for every `x : F32.Bits`, the fields,
  classification and value (`F32.expField_binary32`, `F32.isNaN_binary32`, `F32.isFinite_binary32`,
  `F32.toRat_binary32`, ...), the rounding (`F32.roundMag_binary32`, `F32.roundRat_binary32`,
  `F32.withSign_binary32`) and the operations (`F32.add_binary32`, `sub`, `mul`, `fma`, `rcp`,
  `div`, `fmax`, `fmin`) are those of `Format.binary32` on `x.toNat`, with mode `rn` and the
  canonical NaN `0x7fffffff`;
* **IEEE field decoding** (same module): `f32IsNaN`, `bf16IsNaN`, `f16IsNaN` and `f32Compare`,
  `bf16Compare`, `f16Compare` are the `isNaN` and `compare` of `binary32`, `bfloat16` and
  `binary16`;
* **conversions** (`TVA.Math.Float.Format.Convert`): `bf16ToF32Bits`, `f16ToF32Bits`,
  `f32ToBf16RNE` and `f32ToF16RNE` are `Format.cvt` between those formats (`*_cvt`), for every
  natural number.

The generic rounding facts behind them are in `TVA.Math.Float.Format.Round` (`roundMag_of_scale`,
`roundDiv_congr`, `roundDiv_rn_two_pow`). Proof modules (Mathlib).
-/
