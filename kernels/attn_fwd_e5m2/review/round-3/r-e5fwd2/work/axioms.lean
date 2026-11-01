import AttnFwdE5m2

/-! Reviewer r-e5fwd2, re-review after library round 24: the axioms of the claim, the assumptions'
fields, the new specification's definitions as Lean elaborates them, and the library fact `dot` cites. -/

set_option pp.fullNames true

#print axioms Kernels.AttnFwdE5m2.main
#print axioms Kernels.AttnFwdE5m2.consistent
#print axioms Kernels.AttnFwdE5m2.satisfiable
#print Kernels.AttnFwdE5m2.Assumptions
#print Kernels.AttnFwdE5m2.HeadOk
#print Kernels.AttnFwdE5m2.scoreLift
#print Kernels.AttnFwdE5m2.scoreErr
#print Kernels.AttnFwdE5m2.scoreE
#print Kernels.AttnFwdE5m2.tolO
#print Kernels.AttnFwdE5m2.accNext
#print Kernels.AttnFwdE5m2.errNext
#print Kernels.AttnFwdE5m2.blkLW
#print Kernels.AttnFwdE5m2.iotaV
#print Kernels.AttnFwdE5m2.magCp
#print Kernels.AttnFwdE5m2.magD
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Dot8Chunk.errB
#print TVA.Dot8Chunk.relLift8
#check @Kernels.AttnFwdE5m2.cstep_real
