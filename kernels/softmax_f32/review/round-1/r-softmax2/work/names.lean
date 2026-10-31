import Softmax

/-! r-softmax2, round 1: what every name of the model, the conditions and the specification
resolves to, with full names. -/

set_option pp.fullNames true
set_option pp.proofs true

#print Kernels.Softmax.softmax_kernel
#print Kernels.Softmax.model
#print Kernels.Softmax.launch
#print Kernels.Softmax.Inst.ok
#print Kernels.Softmax.Pre
#print Kernels.Softmax.InputOk
#print Kernels.Softmax.Post
#print Kernels.Softmax.softmaxRef
#print Kernels.Softmax.expIn
#print Kernels.Softmax.relTol
#print Kernels.Softmax.absTol
#print Kernels.Softmax.Spec
#print Kernels.Softmax.Assumptions
#print Kernels.Softmax.inIdx
#print Kernels.Softmax.outIdx
#print Kernels.Softmax.inSet
#print Kernels.Softmax.outSet
#print axioms Kernels.Softmax.main
#print axioms Kernels.Softmax.consistent
#print axioms Kernels.Softmax.satisfiable

/-! The operators of lines 92, 96, 98, 107, 108, without notation, with the instances shown. -/
set_option pp.notation false in
set_option pp.instances true in
set_option pp.explicit true in
#print Kernels.Softmax.softmax_kernel

/-! The auxiliary proof the model's elaboration made. -/
#print Kernels.Softmax.softmax_kernel._proof_1
