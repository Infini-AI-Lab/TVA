import Softmax

/-! Reviewer r-softmax, round 1: what the model's names mean, the axioms, and the tolerance at
the largest covered block size. -/

open TVA

set_option pp.fullNames true in
#print Kernels.Softmax.softmax_kernel

set_option pp.fullNames true in
#print Kernels.Softmax.model

set_option pp.fullNames true in
#print Kernels.Softmax.launch

set_option pp.fullNames true in
#print Kernels.Softmax.Post

set_option pp.fullNames true in
#print Kernels.Softmax.Assumptions

#print axioms Kernels.Softmax.main
#print axioms Kernels.Softmax.consistent
#print axioms Kernels.Softmax.satisfiable

/-- At `BLOCK_SIZE = 2^14` the relative tolerance exceeds `bfloat16`'s unit roundoff `2^-8`. -/
example : (2 : ℚ) ^ (-8 : ℤ) < Kernels.Softmax.relTol (2 ^ 14) := by
  norm_num [Kernels.Softmax.relTol]

/-- At `BLOCK_SIZE = 2^11` it exceeds `float16`'s unit roundoff `2^-11`. -/
example : (2 : ℚ) ^ (-11 : ℤ) < Kernels.Softmax.relTol (2 ^ 11) := by
  norm_num [Kernels.Softmax.relTol]

/-- At the test's `BLOCK_SIZE = 1024` the relative tolerance is above the test's `rtol = 1e-5`
(`torch.allclose`'s default) by more than 24 times. -/
example : 24 * (1 / 100000 : ℚ) < Kernels.Softmax.relTol (2 ^ 10) := by
  norm_num [Kernels.Softmax.relTol]

/-- `absTol` is `2^-8 · 2^-14`: a `bfloat16` rounding of any `σ ≤ 2^-14` is within it. -/
example : (2 : ℚ) ^ (-8 : ℤ) * 2 ^ (-14 : ℤ) ≤ Kernels.Softmax.absTol := by
  norm_num [Kernels.Softmax.absTol]
