import Matmul

/-! Reviewer r-matmul2: the model with every instance argument shown, to list the constants it
uses. -/

set_option pp.fullNames true
set_option pp.explicit true
set_option pp.deepTerms true
set_option pp.maxSteps 1000000
set_option pp.proofs true

#print Kernels.Matmul.matmul_kernel
