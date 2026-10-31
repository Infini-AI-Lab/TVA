import MatmulE5m2

/-! What the model's names mean, printed with full names; the cited rules; the axioms. -/

set_option pp.fullNames true
set_option pp.proofs true

#print Kernels.MatmulE5m2.matmul_kernel
#print Kernels.MatmulE5m2.model
#print Kernels.MatmulE5m2.launch
#print Kernels.MatmulE5m2.Assumptions
#print Kernels.MatmulE5m2.tol
#print Kernels.MatmulE5m2.steps
#print TVA.Launch.grid1
#print TVA.tl.cvt
#print TVA.tl.convert
#print TVA.tl.full
#print TVA.tl.dot
#print TVA.tl.load
#print TVA.tl.store
#print TVA.tl.assume
#print TVA.tl.forRange
#print TVA.tl.idiv
#print TVA.tl.irem
#print TVA.tl.iminimum
#print TVA.tl.program_id
#print TVA.tl.zeros
#print TVA.FloatV.litF32
#print TVA.Rules.Num.DotE5m2
#print TVA.Rules.Num.relDot8
#print TVA.Rules.Num.absDot8
#print TVA.Rules.Num.chunks8
#print TVA.Rules.Num.CvtF32E5m2
#print TVA.Rules.Num.CvtTotal
#print TVA.Rules.Num.DotTotal
#print TVA.DotLoop.bound
#print TVA.DotCfg
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable
