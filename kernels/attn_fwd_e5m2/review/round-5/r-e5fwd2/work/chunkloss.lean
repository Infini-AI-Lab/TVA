import AttnFwdE5m2

/-! Reviewer r-e5fwd2, re-review of the claim on `Num.DotE5m2ChunkLoss`: the axioms, the `dot` field
and the fact it cites, and the new specification's definitions as Lean elaborates them. -/

set_option pp.fullNames true

#print axioms Kernels.AttnFwdE5m2.main
#print axioms Kernels.AttnFwdE5m2.consistent
#print axioms Kernels.AttnFwdE5m2.satisfiable
#check @Kernels.AttnFwdE5m2.Assumptions.dot
#print TVA.Rules.Num.DotE5m2ChunkLoss
#print TVA.Rules.Num.DotF8ChunkLoss
#print TVA.Rules.Num.DotChunkLoss
#print Kernels.AttnFwdE5m2.scoreErr
#print Kernels.AttnFwdE5m2.stepC
#print Kernels.AttnFwdE5m2.stC
#print Kernels.AttnFwdE5m2.tolOW
#print Kernels.AttnFwdE5m2.tolO
#print Kernels.AttnFwdE5m2.errStep
#print Kernels.AttnFwdE5m2.accStep
#print Kernels.AttnFwdE5m2.blkV
#print Kernels.AttnFwdE5m2.blkAbsV
#print Kernels.AttnFwdE5m2.lS8
#print Kernels.AttnFwdE5m2.lC8
#print Kernels.AttnFwdE5m2.lP8
#print TVA.Dot8Chunk.errM
