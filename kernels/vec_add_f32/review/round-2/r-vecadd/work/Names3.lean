import VecAdd

/-! r-vecadd, round 2, third filing (gate 990243026f1289fb): what every name of the claim means,
printed with full names and explicit instances, and the axioms of the three theorems. -/

set_option pp.fullNames true

#print Kernels.VecAdd.add_kernel
#print Kernels.VecAdd.model
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.Inst
#print Kernels.VecAdd.Inst.ok
#print Kernels.VecAdd.Pre
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Spec
#print Kernels.VecAdd.Assumptions

-- the instances behind `*`, `+`, `/`, `-` in the body and the grid formula
set_option pp.explicit true in
#print Kernels.VecAdd.add_kernel

set_option pp.explicit true in
#print Kernels.VecAdd.launch

set_option pp.explicit true in
#print Kernels.VecAdd.Post

-- the library operations the model names, as the library defines them
#print TVA.tl.program_id
#print TVA.tl.arange
#print TVA.tl.load
#print TVA.tl.store
#print TVA.tl.add
#print TVA.tl.addrs
#print TVA.Tensor.ltS
#print TVA.Tensor.ptrAdd
#print TVA.Launch.grid1
#print TVA.Rules.Compiler.Faithful

-- the claim's statement
#check @Kernels.VecAdd.main
#check @Kernels.VecAdd.consistent
#check @Kernels.VecAdd.satisfiable

#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
#print axioms Kernels.VecAdd.satisfiable
