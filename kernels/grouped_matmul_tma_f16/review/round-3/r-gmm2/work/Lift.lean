import GroupedMatmulTma

/-! Reviewer r-gmm2, round 3: the restated `DotF16` and the new tolerance's magnitude. -/

set_option pp.fullNames true

#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.minNormal8
#print Kernels.GroupedMatmulTma.dotLift
#print Kernels.GroupedMatmulTma.OutSpec
#check @Kernels.GroupedMatmulTma.Assumptions.mk
#print axioms Kernels.GroupedMatmulTma.main
#print axioms Kernels.GroupedMatmulTma.consistent
#print axioms Kernels.GroupedMatmulTma.satisfiable

/-! `liftMag .f16` on sample values: zero, a normal value, the smallest normal, the smallest and
the largest subnormal. -/
#eval TVA.Rules.Num.liftMag .f16 0
#eval TVA.Rules.Num.liftMag .f16 (3 / 4)
#eval TVA.Rules.Num.liftMag .f16 (-(2 ^ (-14 : ℤ)))
#eval TVA.Rules.Num.liftMag .f16 (2 ^ (-24 : ℤ))
#eval TVA.Rules.Num.liftMag .f16 (1023 * 2 ^ (-24 : ℤ))
