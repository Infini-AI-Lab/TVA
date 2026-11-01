import BlockScaledMxfp8

/-! r-mx8, round 1: print the model, the launch, the assumptions and the axioms with full names. -/

set_option pp.fullNames true
set_option pp.proofs false

open Kernels.BlockScaledMxfp8

#print Kernels.BlockScaledMxfp8.block_scaled_matmul_kernel
#print Kernels.BlockScaledMxfp8.model
#print Kernels.BlockScaledMxfp8.launch
#print Kernels.BlockScaledMxfp8.cfgMx
#print Kernels.BlockScaledMxfp8.Assumptions
#print Kernels.BlockScaledMxfp8.aScaleDesc
#print Kernels.BlockScaledMxfp8.bScaleDesc
#print TVA.Launch.grid1
#print TVA.tl.cvt
#print axioms Kernels.BlockScaledMxfp8.main
#print axioms Kernels.BlockScaledMxfp8.consistent
#print axioms Kernels.BlockScaledMxfp8.satisfiable

-- The `dot_total` configuration is `cfgMx`.
example : cfgMx.lhs = TVA.MXFmt.e4m3 ∧ cfgMx.rhs = TVA.MXFmt.e4m3 ∧
    cfgMx.lhsScale = TVA.ScaleFmt.e8m0 ∧ cfgMx.rhsScale = TVA.ScaleFmt.e8m0 ∧ cfgMx.group = 32 ∧
    cfgMx.lhsKPack = true ∧ cfgMx.rhsKPack = true ∧ cfgMx.fastMath = false :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

-- The step constant is the library's `relDot16 128`, `130 · 2^-22`.
example : rho = 130 / 2 ^ 22 := rho_eq

#print Kernels.BlockScaledMxfp8.OutSpec
#print Kernels.BlockScaledMxfp8.Post
#print Kernels.BlockScaledMxfp8.Pre
#print Kernels.BlockScaledMxfp8.InDom
#print Kernels.BlockScaledMxfp8.tol
#print Kernels.BlockScaledMxfp8.Inst.ok
