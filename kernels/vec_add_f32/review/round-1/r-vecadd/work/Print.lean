import VecAdd

/-! Reviewer r-vecadd, round 1: print the claim's definitions with full names, and the axioms. -/

set_option pp.fullNames true

#print Kernels.VecAdd.add_kernel
#print Kernels.VecAdd.model
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.Inst.ok
#print Kernels.VecAdd.Pre
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Spec
#print Kernels.VecAdd.Assumptions
#print TVA.Rules.Compiler.Faithful
#print TVA.Buf.elems
#print TVA.Tensor.instHAddIntV
#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
