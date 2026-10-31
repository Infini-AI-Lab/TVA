import Matmul

/-! r-matmul, round 2: the model and the claim after the prover's revision, printed with full
names, and the new lemma `tol_rel_lt`. -/

set_option pp.fullNames true

#print Kernels.Matmul.matmul_kernel
#print Kernels.Matmul.model
#print Kernels.Matmul.launch
#print Kernels.Matmul.Inst.ok
#print Kernels.Matmul.Post
#print Kernels.Matmul.OutSpec
#print Kernels.Matmul.tol
#print Kernels.Matmul.Assumptions
#check @Kernels.Matmul.tol_rel_lt
#print axioms Kernels.Matmul.tol_rel_lt
#print axioms Kernels.Matmul.main
#print axioms Kernels.Matmul.consistent
#print axioms Kernels.Matmul.satisfiable

open Kernels.Matmul TVA in
/-- At the new largest `K = 2^13` the relative part of `tol` is below the largest relative error of
a rounding to `bfloat16`, `2^-8 / (1 + 2^-8)`, with room for one `float16` half-ulp (`2^-11`). -/
example : (1 + Rules.Num.relDot16 64) ^ steps (2 ^ 13) - 1 + 1 / 2 ^ 11 < (1 / 2 ^ 8) / (1 + 1 / 2 ^ 8) := by
  have h := tol_rel_lt (K := 2 ^ 13) le_rfl
  have : (21 : ℚ) / 10000 + 1 / 2 ^ 11 < (1 / 2 ^ 8) / (1 + 1 / 2 ^ 8) := by norm_num
  linarith
