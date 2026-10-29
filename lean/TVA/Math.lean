import TVA.Math.Defs
import TVA.Math.LittleEndian
import TVA.Math.Encoding
import TVA.Math.Rounding
import TVA.Math.Arith

/-!
# TVA's float layer

Exact binary floating point, written for TVA from IEEE 754-2019 and the OCP 8-bit and 4-bit
formats, on rationals:

* `TVA.Math.Defs`: every definition, in core Lean only, so that `evidence/float-layer` compiles
  the very same constants to native code and compares them with the device.
* `TVA.Math.LittleEndian` (namespace `TVA.Math.Bytes`): bytes and the little-endian encoding of
  an element in memory.
* `TVA.Math.Encoding`: formats (`Format`: `binary16`, `bfloat16`, `binary32`, `binary64`,
  `e5m2`, `e4m3`, `e2m1`), what a pattern means (`toRat`, `isNaN`, `isInf`, `isFinite`, …), the
  constants `maxVal`, `u`, `eta`, and the order of values (`toRat_lt_iff_orderKey`).
* `TVA.Math.Rounding`: `roundRat` (to nearest even or toward zero, with or without saturation)
  and the rounding error bounds (`roundRat_rn_err`, `roundRat_rn_err_normal`,
  `roundRat_rn_err_of_finite`, `roundRat_rz_err`); `qlog2_spec` (`qlog2 = ⌊log₂⌋`).
* `TVA.Math.Arith`: the IEEE operations on patterns (`add`, `sub`, `mul`, `fma`, `div`, `sqrt`,
  `neg`, `absPat`, `maxNum`, `minNum`, `maxNaN`, `minNaN`, `cmp`, `cvt`) and exact widening
  conversions (`Wider`, `cvt_of_wider`).

`TVA.Lang.Scalar` wraps the patterns as `FloatV f`, and `TVA.Lemmas.FloatErr` lifts the bounds to
it. The agreement of these definitions with Triton 3.8.0 on the H200 NVL is TVA's evidence
`evidence/float-layer`.
-/
