import BlockScaledMxfp8

/-! Reviewer r-mx8b: what the model's names mean, printed with full names. -/

set_option pp.fullNames true
set_option pp.proofs false

open Kernels.BlockScaledMxfp8

#print Kernels.BlockScaledMxfp8.block_scaled_matmul_kernel
#print Kernels.BlockScaledMxfp8.model
#print Kernels.BlockScaledMxfp8.launch
#print TVA.Launch.grid1
#print Kernels.BlockScaledMxfp8.cfgMx
#print Kernels.BlockScaledMxfp8.aScaleDesc
#print Kernels.BlockScaledMxfp8.bScaleDesc
#print Kernels.BlockScaledMxfp8.Assumptions
#print Kernels.BlockScaledMxfp8.Post
#print Kernels.BlockScaledMxfp8.OutSpec
#print Kernels.BlockScaledMxfp8.sIdx
#print Kernels.BlockScaledMxfp8.InDom
#print axioms Kernels.BlockScaledMxfp8.main
#print axioms Kernels.BlockScaledMxfp8.consistent
#print axioms Kernels.BlockScaledMxfp8.satisfiable

-- the scalar arithmetic the model uses on `I32`
example (a b : TVA.I32) : a * b = a * b := rfl
#check (inferInstance : HMul TVA.I32 TVA.I32 TVA.I32)
#check (inferInstance : HAdd TVA.I32 TVA.I32 TVA.I32)
#synth HMul TVA.I32 TVA.I32 TVA.I32
#synth HAdd TVA.I32 TVA.I32 TVA.I32
