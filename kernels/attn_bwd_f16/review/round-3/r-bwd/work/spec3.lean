import AttnBwd

/-! r-bwd, round 3: the restated fact the claim cites, the new tolerance terms, and why
`pBr`, `pE`, `p16Err`, `dsErr` hash differently although their text did not change. -/

set_option pp.fullNames true
set_option pp.proofs true

#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.DotLiftAcc
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.minNormal8
#print Kernels.AttnBwd.lm
#print Kernels.AttnBwd.lift16
#print Kernels.AttnBwd.p16L
#print Kernels.AttnBwd.ds16L
#print Kernels.AttnBwd.aQK
#print Kernels.AttnBwd.aVD
#print Kernels.AttnBwd.dVTol
#print Kernels.AttnBwd.dKAcc
#print Kernels.AttnBwd.dQAcc
#print Kernels.AttnBwd.pBr
#print Kernels.AttnBwd.pE
#print Kernels.AttnBwd.p16Err
#print Kernels.AttnBwd.dsErr
#print Kernels.AttnBwd.lift16._proof_1
#print Kernels.AttnBwd.Assumptions
#check @Kernels.AttnBwd.main
#print axioms Kernels.AttnBwd.main
#print axioms Kernels.AttnBwd.consistent
#print axioms Kernels.AttnBwd.satisfiable

/-- `lm` of a `float16` subnormal is `2^-14`, of a normal value its magnitude, of zero `0`:
spot checks on rationals. -/
example : TVA.Rules.Num.liftMag .f16 (2 ^ (-24 : ℤ)) = 2 ^ (-14 : ℤ) := by
  unfold TVA.Rules.Num.liftMag TVA.Rules.Num.minNormal8; norm_num
example : TVA.Rules.Num.liftMag .f16 (-3/4) = 3/4 := by
  unfold TVA.Rules.Num.liftMag TVA.Rules.Num.minNormal8; norm_num
example : TVA.Rules.Num.liftMag .f16 0 = 0 := by
  unfold TVA.Rules.Num.liftMag; norm_num
