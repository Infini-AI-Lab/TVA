import AttnFwd

/-! Reviewer r-fwd2, round 4: print the changed spec definitions with full names, the kernel's
real-number counterparts and the claim's axioms; check the docstring figures of `absA` and `kapA`. -/

set_option pp.fullNames true

#print Kernels.AttnFwd.absA
#print Kernels.AttnFwd.kapA
#print Kernels.AttnFwd.absQ0
#print Kernels.AttnFwd.tolF
#print Kernels.AttnFwd.kapAr
#print Kernels.AttnFwd.absAr
#print Kernels.AttnFwd.AInv
#print axioms Kernels.AttnFwd.main

/-- `absA`: `relDot16 64` times the block's rest of the lift, at most `2^-22 + 2^-32`, is about
`0.52·2^-37` and below `absA`. -/
example : TVA.Rules.Num.relDot16 64 * (1 / 2 ^ 22 + 1 / 2 ^ 32) = 66 / 2 ^ 44 + 66 / 2 ^ 54 ∧
    (66 : ℚ) / 2 ^ 44 + 66 / 2 ^ 54 < 53 / 100 * Kernels.AttnFwd.absA ∧
    51 / 100 * Kernels.AttnFwd.absA < (66 : ℚ) / 2 ^ 44 := by
  unfold TVA.Rules.Num.relDot16 Kernels.AttnFwd.absA; norm_num

/-- `kapA`: `66·2^-36·8/7 ≤ 2^-29`. -/
example : (66 : ℚ) / 2 ^ 36 * (8 / 7) ≤ Kernels.AttnFwd.kapA := by
  unfold Kernels.AttnFwd.kapA; norm_num
