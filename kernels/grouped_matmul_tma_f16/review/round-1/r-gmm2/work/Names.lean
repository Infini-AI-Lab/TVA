import GroupedMatmulTma

/-! Reviewer r-gmm2: what the model's names mean, printed with full names. -/

set_option pp.fullNames true
set_option pp.proofs false

#print Kernels.GroupedMatmulTma.grouped_matmul_tma_kernel
#print Kernels.GroupedMatmulTma.model
#print Kernels.GroupedMatmulTma.launch
#print TVA.DotCfg
#print TVA.Launch.grid1
#print TVA.instHAddPtrIntV
#print TVA.tl.load1
#print TVA.tl.convert
#print TVA.tl.cvt
#print TVA.tl.zeros
#print TVA.tl.program_id
#print TVA.tl.intToPtr
#print axioms Kernels.GroupedMatmulTma.main
#print axioms Kernels.GroupedMatmulTma.consistent
#print axioms Kernels.GroupedMatmulTma.satisfiable
#check @Kernels.GroupedMatmulTma.Assumptions.mk
