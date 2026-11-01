import MatmulE5m2

/-! Reviewer r-e5m2, re-review after library round 24: the new specification's names and the
`dot` assumption, printed with full names, and the statements of the rejection lemmas. -/

set_option pp.fullNames true

#print Kernels.MatmulE5m2.tol
#print Kernels.MatmulE5m2.errAt
#print Kernels.MatmulE5m2.liftV
#print Kernels.MatmulE5m2.chunkMax
#print Kernels.MatmulE5m2.partSum
#print Kernels.MatmulE5m2.tolA
#print Kernels.MatmulE5m2.Assumptions
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.minNormal8
#check @Kernels.MatmulE5m2.tol_le_tolA
#check @Kernels.MatmulE5m2.tol_spike
#check @Kernels.MatmulE5m2.zero_rejected
#check @Kernels.MatmulE5m2.drop_rejected
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable
#print axioms Kernels.MatmulE5m2.zero_rejected
#print axioms Kernels.MatmulE5m2.drop_rejected
