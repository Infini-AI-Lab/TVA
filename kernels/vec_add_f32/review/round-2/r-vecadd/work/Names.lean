import VecAdd

/-! r-vecadd, round 2 (second filing): what the names of the model mean, printed with full names,
and the axioms of the three theorems. -/

set_option pp.fullNames true

#print Kernels.VecAdd.add_kernel
#print Kernels.VecAdd.model
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.Inst.ok
#print Kernels.VecAdd.Pre
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Spec
#print Kernels.VecAdd.Assumptions

-- the instances behind `/`, `*`, `+` and `-` in `launch`, and `*`, `+` in the kernel body
set_option pp.explicit true in
#print Kernels.VecAdd.launch

set_option pp.explicit true in
#print Kernels.VecAdd.add_kernel

#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
#print axioms Kernels.VecAdd.satisfiable
