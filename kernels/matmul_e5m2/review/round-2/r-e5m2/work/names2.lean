import MatmulE5m2

/-! Reviewer r-e5m2, round 2: what the new specification's names mean, printed with full names,
and the statements of the rejection lemmas. -/

set_option pp.fullNames true

#print Kernels.MatmulE5m2.tol
#print Kernels.MatmulE5m2.errAt
#print Kernels.MatmulE5m2.partSum
#print Kernels.MatmulE5m2.stepMag
#print Kernels.MatmulE5m2.prodV
#print Kernels.MatmulE5m2.OutSpec
#print Kernels.MatmulE5m2.Post
#check @Kernels.MatmulE5m2.tol_le_tolA
#check @Kernels.MatmulE5m2.zero_rejected
#check @Kernels.MatmulE5m2.drop_rejected
#check @Kernels.MatmulE5m2.tol_spike
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable
#print axioms Kernels.MatmulE5m2.zero_rejected
#print axioms Kernels.MatmulE5m2.drop_rejected
