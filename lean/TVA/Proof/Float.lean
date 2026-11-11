import TVA.Proof.Float.Encoding
import TVA.Proof.Float.Constants
import TVA.Proof.Float.Quantum
import TVA.Proof.Float.Round
import TVA.Proof.Float.Ops
import TVA.Proof.Float.Sqrt
import TVA.Proof.Float.Value
import TVA.Proof.Float.Error
import TVA.Proof.Float.Max
import TVA.Proof.Float.Facts
import TVA.Proof.Float.Widen
import TVA.Proof.Float.Tree
import TVA.Proof.Float.Fused
import TVA.Proof.Float.Sum
import TVA.Proof.Float.SumBcast

/-!
# The float part of the proof kit

Everything here is proved from the definitions of `TVA.Base.Float` and the `tl.*` relations of
`TVA.Triton`; nothing is trusted. The idea of every float bound in the library:

* **One rounding.** An IEEE operation computes its exact result `q` and rounds it once. To
  nearest, the result is within `|q|·u + η` of `q`, where `u = 2^-(m+1)` (half an ulp of `1`,
  `2^-24` in `float32`) is the *relative* error and `η = 2^(emin-m-1)` (half the smallest
  subnormal, `2^-150` in `float32`) the *absolute* error, which matters only below the normal
  range. The result is finite when `|q| ≤ maxVal`.
* **Up to NaN payload.** A `tl.*` float operation returns the IEEE result up to NaN payload
  (`NaNEq`); a finite IEEE result is returned exactly.
* **Trees.** A reduction adds along a tree Triton chooses; a bound per node, relative and
  absolute, gives a bound for every tree of a given depth (`Tree.errBound`).

The files, in the order they build on each other:

| file | contents |
|---|---|
| `Encoding` | what a pattern means: sign, magnitude, `magVal`, `toRat`, order keys, `maxVal` |
| `Constants` | `u`, `η`, `maxVal` of each format (`ℚ`, `ℝ`); infinities and NaN; standard formats |
| `Quantum` | the steps of `roundRat`: quantum, significand, packing |
| `Round` | the error of one rounding (`roundRat_rn_err`, …), saturation |
| `Ops` | the pattern operations on finite operands, negation, comparison, exact widening |
| `Sqrt` | the rounded square root |
| `Value` | `FloatV` and its pattern, `toReal`, `NaNEq` |
| `Error` | `add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err`, `sqrtRn_err`, `cvt_err`, … |
| `Max` | `tl.maximum`, `tl.max` reductions, maxima of approximations, `−∞` |
| `Facts` | zeros, bounds of a value, finiteness from magnitude, `DType.val?` |
| `Widen` | exact widening conversions, `x.to(…)` (`FloatRel`, `ConvRel`) |
| `Tree` | the error of a reduction tree (`errBound`, `eval_err_abs`, `eval_err`) |
| `Fused` | fusable additions (`AddRel.err`, `mulAdd_err`) |
| `Sum` | float reductions (`sum_err*`, `sumc_err*`) |
| `SumBcast` | rounded `float32` products summed along a tree |
-/
