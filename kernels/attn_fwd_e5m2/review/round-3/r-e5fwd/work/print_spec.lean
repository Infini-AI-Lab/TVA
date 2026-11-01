import AttnFwdE5m2

/-! Reviewer r-e5fwd, re-proof on the chunk facts: print the new specification constants, `Post`,
`HeadOk` and the `dot` assumption with full names, to check that every operation is a `TVA.*`,
Mathlib or kernel definition whose meaning was read; and the axioms of the three theorems. -/

set_option pp.fullNames true

#print Kernels.AttnFwdE5m2.scoreLift
#print Kernels.AttnFwdE5m2.scoreErr
#print Kernels.AttnFwdE5m2.HeadOk
#print Kernels.AttnFwdE5m2.scoreE
#print Kernels.AttnFwdE5m2.blkLW
#print Kernels.AttnFwdE5m2.accNext
#print Kernels.AttnFwdE5m2.errNext
#print Kernels.AttnFwdE5m2.tolO
#print Kernels.AttnFwdE5m2.iotaV
#print Kernels.AttnFwdE5m2.magCp
#print Kernels.AttnFwdE5m2.Post
#print Kernels.AttnFwdE5m2.Assumptions
#print TVA.Dot8Chunk.relLift8
#print TVA.Dot8Chunk.chunkLift8
#print axioms Kernels.AttnFwdE5m2.main
#print axioms Kernels.AttnFwdE5m2.consistent
#print axioms Kernels.AttnFwdE5m2.satisfiable
