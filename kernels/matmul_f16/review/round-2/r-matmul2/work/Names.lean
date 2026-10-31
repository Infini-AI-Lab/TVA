import Matmul

/-! Reviewer r-matmul2: print the model, the launch and the claim's definitions with full names,
to check that every operation is the library's (`TVA.*`) or a kernel helper read in full. -/

set_option pp.fullNames true
set_option pp.notation false
set_option pp.coercions true
set_option pp.explicit false

#print Kernels.Matmul.matmul_kernel
#print Kernels.Matmul.model
#print Kernels.Matmul.launch
#print Kernels.Matmul.Pre
#print Kernels.Matmul.OutSpec
#print Kernels.Matmul.Post
#print Kernels.Matmul.tol
#print Kernels.Matmul.Assumptions
#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.DotAcc
#print TVA.Rules.Num.DotTotal
#print TVA.Launch.grid1
#print axioms Kernels.Matmul.main
#print axioms Kernels.Matmul.consistent
#print axioms Kernels.Matmul.satisfiable
