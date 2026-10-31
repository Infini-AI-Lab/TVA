import MatmulE5m2

/-! Reviewer r-e5m2: what the model's names mean, printed with full names. -/

set_option pp.fullNames true
set_option pp.proofs false

open Kernels.MatmulE5m2

#print Kernels.MatmulE5m2.matmul_kernel
#print Kernels.MatmulE5m2.model
#print Kernels.MatmulE5m2.launch
#print Kernels.MatmulE5m2.Assumptions
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable

#print TVA.Rules.Num.DotE5m2
#print TVA.Rules.Num.CvtF32E5m2
#print TVA.Rules.Num.relDot8
#print TVA.Rules.Num.absDot8
#print TVA.Rules.Num.chunks8
#print TVA.DotLoop.bound
#print TVA.tl.load
#print TVA.tl.convert
#print TVA.tl.dot
#print TVA.tl.cvt
#print TVA.Launch.grid1
