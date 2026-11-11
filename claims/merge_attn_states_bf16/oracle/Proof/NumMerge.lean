import Claim
import TVA.Proof
import TVA

/-!
# The numerical content of the merge branch

The merge branch computes
`exp(p − M)`, `exp(s − M)`, their sum `o`, `log o + M` for the lse, and the weighted average of the
two rows for the output. These lemmas (§`NumMid`, `NumOut`, `NumRow`) turn the platform's accuracy
assumptions into the claim's reference values.
-/

namespace Claims.MergeAttnStates

open TVA

/-- **What the middle of the merge branch establishes**: the maximum is the reference maximum, the
two exponentials are the reference weights, and their sum `o` is the reference sum. -/
structure LseMid (p s M u v pe se o : F32) : Prop where
  /-- `M` is not a NaN. -/
  hMn : M.isNaN = false
  /-- `M` is finite or `−∞`. -/
  hMf : M.isFinite = true ∨ M = negInfC
  /-- `M` is finite (the two values are not both `−∞`). -/
  hMfin : M.isFinite = true
  /-- `M` is the reference maximum of the two corrected values. -/
  hM : M.toReal = refMax p s
  /-- The first exponential is finite. -/
  hpefin : pe.isFinite = true
  /-- The first exponential is at most one, up to the rounding. -/
  hpe1 : pe.toReal ≤ 1 + 2 ^ (-16 : ℤ)
  /-- The second exponential is finite. -/
  hsefin : se.isFinite = true
  /-- The second exponential is at most one, up to the rounding. -/
  hse1 : se.toReal ≤ 1 + 2 ^ (-16 : ℤ)
  /-- The first exponential is the reference weight. -/
  hpe : |pe.toReal - side p (refMax p s)| ≤ 2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ)
  /-- The second exponential is the reference weight. -/
  hse : |se.toReal - side s (refMax p s)| ≤ 2 ^ (-20 : ℤ) + 2 ^ (-21 : ℤ)
  /-- The sum of the weights is positive. -/
  ho0 : 1 - 2 ^ (-18 : ℤ) ≤ o.toReal
  /-- The sum of the weights is at most two, up to the rounding. -/
  ho1 : o.toReal ≤ 2 + 2 ^ (-16 : ℤ)
  /-- The sum of the weights is finite. -/
  hofin : o.isFinite = true
  /-- The sum of the weights is the reference sum. -/
  hpo : |o.toReal - refSum p s| ≤ 2 ^ (-18 : ℤ)

end Claims.MergeAttnStates
