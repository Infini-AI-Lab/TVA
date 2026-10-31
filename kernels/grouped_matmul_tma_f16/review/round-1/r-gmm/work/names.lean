import GroupedMatmulTma

/-! r-gmm, round 1: what the names of the model, the launch and the claim mean. -/

set_option pp.fullNames true
set_option pp.proofs true

open TVA

#print Kernels.GroupedMatmulTma.grouped_matmul_tma_kernel
#print Kernels.GroupedMatmulTma.model
#print Kernels.GroupedMatmulTma.launch
#print TVA.Launch.grid1
#print TVA.tl.dot_add
#print TVA.tl.cvt
#print TVA.tl.make_tensor_descriptor
#print TVA.tl.desc_load
#print TVA.tl.desc_store
#print TVA.tl.whileLoop
#print TVA.tl.forRange
#print TVA.tl.load1
#print TVA.tl.intToPtr
#print TVA.IntV.le
#print TVA.IntV.lt
#print Kernels.GroupedMatmulTma.Assumptions
#print Kernels.GroupedMatmulTma.Post
#print Kernels.GroupedMatmulTma.OutSpec
#print axioms Kernels.GroupedMatmulTma.main
#print axioms Kernels.GroupedMatmulTma.consistent
#print axioms Kernels.GroupedMatmulTma.satisfiable
