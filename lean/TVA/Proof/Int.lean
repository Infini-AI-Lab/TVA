import TVA.Proof.Int.Basic
import TVA.Proof.Int.Arith
import TVA.Proof.Int.Cast
import TVA.Proof.Int.Lanes

/-!
# The integer kit

Triton's index arithmetic is `tl.int32` (sometimes `tl.int64`) and wraps modulo `2^w`; `//` and
`%` truncate and are undefined for a zero divisor (`TVA.Base.Int`). A proof shows that no step
wraps and then computes with Lean's `Int` and `Nat`. This kit is what it needs.

## Conventions

* **`toInt` is the value.** `a.toInt : Int` is the number an `IntV` stands for (two's complement
  when signed). Facts are stated about it: two integers of one type are equal when their values
  are (`IntV.ext_toInt`), and `==`, `<`, `<=`, `tl.maximum`, `tl.minimum` compare values
  (`IntV.eq_eq`, `lt_eq`, `le_eq`, `toInt_max`, `toInt_min`).
* **`SFits w x`** means that the integer `x` fits `w` signed bits, `-2^(w-1) ≤ x < 2^(w-1)`.
  A signed operation whose exact result `SFits` does not wrap (`IntV.toInt_add`, `toInt_sub`,
  `toInt_mul`). `IntV.sfits_iff` writes it out; `sfits_of_nonneg`, `sfits_natCast` and
  `SFits.mono` establish it.
* **Tracked values.** Lemmas take the operands' values as hypotheses and give the result's:
  `a.toInt = x → b.toInt = y → x + y < 2^(w-1) → (a + b).toInt = x + y` (`IntV.add_nat`). A proof
  carries the value of each intermediate from step to step and discharges the bounds with
  `omega`. The `_nat` forms take natural values and one bound, the plain forms integer values and
  an `SFits`; to use a plain form on an operand's own value, pass `rfl`.
* **Any width.** Each fact is stated once for every width `w`, and needs no `0 < w` unless it is
  false without it. The `I32.*` and `I64.*` forms (`IntV.I32.add_nat`, `IntV.I32.toInt_ofInt`, …)
  restate the common ones with the bounds written out (`2^31`, `2^63`).
* **`simp`** rewrites only constants and lanes: the value of a numeral or `IntV.ofNat n` in range
  (`toInt_ofNat`, `toInt_ofNat_lit`, `toInt_zero`, `toInt_neg_one`) and the lane of an elementwise
  operation (`Tensor.get1_add`, …). Everything else is applied by name.

## Files

* `TVA.Proof.Int.Basic`: `SFits`, constants, equality and comparisons.
* `TVA.Proof.Int.Arith`: `+ - *` without wrapping, `//` and `%` of nonnegative values.
* `TVA.Proof.Int.Cast`: `IntV.cast` and the integer conversions `tl.convert` makes
  (`ConvRel.iff_cast_<d>_<d'>`).
* `TVA.Proof.Int.Lanes`: lanes of integer vectors, `tl.arange`, the lanes of `tl.idiv`, `tl.irem`.
-/
