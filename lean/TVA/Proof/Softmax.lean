import TVA.Proof.Softmax.Exp
import TVA.Proof.Softmax.Exact
import TVA.Proof.Softmax.Error

/-!
# Online softmax

An attention kernel computes, for each query row, the softmax-weighted sum of the values without
ever holding a whole score row: it walks the keys block by block, keeping a running maximum `m`, a
running sum `l` of `exp (s − m)` and an accumulator `acc` of `exp (s − m)·v`, and rescales `l` and
`acc` by `exp (m_old − m_new)` whenever the maximum grows. At the end it divides `acc` by `l`.

Why this is right: subtracting a reference from every score multiplies numerator and denominator
of the softmax ratio by the same factor, so the ratio does not depend on the reference; the
rescaling keeps `l` and `acc` as the full sums at the current reference. With rounding, the same
recurrence holds up to per-trip errors, and because a running maximum never decreases the
rescaling never amplifies them.

This is mathematics over `ℝ` (Mathlib only). A kernel connects it to its float values with the
float error lemmas and the device facts (`tl.math.exp2`, `tl.dot`, `tl.sum`).

* `TVA.Proof.Softmax.Exp`: bounds on the exponential of a rounded argument; `exp2` as `exp`.
* `TVA.Proof.Softmax.Exact`: `softmax`, `wavg`, the closed forms `lsum`, `asum`, the exact
  invariant (`l_eq`, `acc_eq`, `ratio_eq`) and keys in blocks (`wavg_blocks`).
* `TVA.Proof.Softmax.Error`: the running error bound (`accum`, `l_err`, `acc_err`), one trip from
  its parts (`step_err`, `l_trip`) and the output's error (`err_final`, `err_final_mono`).

Everything is in the namespace `TVA.Softmax`.
-/
