import GroupedGemm

set_option pp.fullNames true
set_option pp.proofs false

open TVAExamples.GroupedGemm in
#print kstep
open TVAExamples.GroupedGemm in
#print tile
open TVAExamples.GroupedGemm in
#print problem
open TVAExamples.GroupedGemm in
#print grouped_matmul_kernel
#print axioms TVAExamples.GroupedGemm.main
#print axioms TVAExamples.GroupedGemm.consistent
#print TVA.DotLoop.stepRel
#print TVA.DotLoop.stepAbs
#print TVA.DotLoop.bound
#print TVA.Tensor.bcastCols
#print TVA.Tensor.bcastRows
#print TVA.Tensor.instHMulIntV
#print TVA.Tensor.instHMulIntV_1
#print TVA.Tensor.instHAddIntV
#print TVA.IntV.div?
#print TVA.IntV.rem?
