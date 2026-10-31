import Matmul

/-! r-matmul, round 1: what the names of the model and the claim mean, printed with full names. -/

set_option pp.fullNames true
set_option pp.proofs true

#print Kernels.Matmul.matmul_kernel
#print Kernels.Matmul.model
#print Kernels.Matmul.launch
#print Kernels.Matmul.Inst.ok
#print Kernels.Matmul.Pre
#print Kernels.Matmul.OutSpec
#print Kernels.Matmul.Post
#print Kernels.Matmul.Spec
#print Kernels.Matmul.tol
#print Kernels.Matmul.steps
#print Kernels.Matmul.dotSum
#print Kernels.Matmul.dotAbs
#print Kernels.Matmul.Assumptions
#print TVA.Launch.grid1
#print TVA.DotLoop.bound
#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.DotTotal
#print TVA.Tensor.col
#print TVA.Tensor.row
#print TVA.Tensor.ltS
#print TVA.bc
#print TVA.tl.load
#print TVA.tl.convert
#print TVA.tl.full
#print TVA.tl.cvt
#print TVA.FloatV.litF32
#print TVA.Claim.Meets
#print axioms Kernels.Matmul.main
#print axioms Kernels.Matmul.consistent
#print axioms Kernels.Matmul.satisfiable
