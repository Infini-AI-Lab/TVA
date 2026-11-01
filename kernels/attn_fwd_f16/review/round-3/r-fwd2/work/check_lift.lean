import AttnFwd

/-! Reviewer r-fwd2, round 3: print the changed definitions with full names, the assumption
structure and the claim's axioms; check the lift's value on a subnormal and a normal `float16`
value, and the round-25 constants. -/

set_option pp.fullNames true

#print Kernels.AttnFwd.scoreMag
#print Kernels.AttnFwd.HeadOk
#print Kernels.AttnFwd.absA
#print Kernels.AttnFwd.omega0
#print Kernels.AttnFwd.Assumptions
#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.liftMag
#print axioms Kernels.AttnFwd.main

open TVA.Rules.Num in
/-- A subnormal `float16` value (`2^-24`) is lifted to `2^-14`. -/
example : liftMag .f16 ((1 : ℚ) / 2 ^ 24) = 1 / 2 ^ 14 := by
  unfold liftMag minNormal8; norm_num

open TVA.Rules.Num in
/-- A normal one (`-3/2`) keeps its magnitude. -/
example : liftMag .f16 (-3 / 2 : ℚ) = 3 / 2 := by
  unfold liftMag minNormal8; norm_num

/-- The per-block budget: `relDot16 64 · 5·2^-9 = 330·2^-31` (about `1.29·2^-23`) fits in `absA`. -/
example : TVA.Rules.Num.relDot16 64 * (5 / 2 ^ 9) = 330 / 2 ^ 31 ∧
    (330 : ℚ) / 2 ^ 31 < Kernels.AttnFwd.absA := by
  unfold TVA.Rules.Num.relDot16 Kernels.AttnFwd.absA; norm_num
