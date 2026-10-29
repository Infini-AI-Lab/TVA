import VecAdd

/-! r-vecadd2, round 2: what every name of the model means, printed with full names,
and the axioms of the three theorems. -/

set_option pp.fullNames true
set_option pp.notation false
set_option pp.explicit false

#print Kernels.VecAdd.add_kernel
#print Kernels.VecAdd.model
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.Inst
#print Kernels.VecAdd.Inst.ok
#print Kernels.VecAdd.Pre
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Spec
#print Kernels.VecAdd.Assumptions

#check @Kernels.VecAdd.main
#check @Kernels.VecAdd.consistent
#check @Kernels.VecAdd.satisfiable

#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
#print axioms Kernels.VecAdd.satisfiable
