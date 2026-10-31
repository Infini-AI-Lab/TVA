import VecAdd

/-! r-vecadd, round 2, fifth filing (gate 5f65eb1f7a961315): what every name of the claim means,
printed with full names and explicit instances, the library operations the model names, the
structure `Assumptions` and its one field, the two dtypes whose docstrings moved (L118), and the
axioms of the three theorems. -/

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
#check @Kernels.VecAdd.Assumptions.faithful

-- the instances behind `*`, `+`, `/`, `-` in the body, the grid formula and `Post`
set_option pp.explicit true in
#print Kernels.VecAdd.add_kernel

set_option pp.explicit true in
#print Kernels.VecAdd.launch

set_option pp.explicit true in
#print Kernels.VecAdd.model

set_option pp.explicit true in
#print Kernels.VecAdd.Post

set_option pp.explicit true in
#print Kernels.VecAdd.Inst.ok

-- the library operations the model names, as the library defines them
#print TVA.tl.program_id
#print TVA.tl.arange
#print TVA.Tensor.arange
#print TVA.tl.load
#print TVA.tl.store
#print TVA.tl.add
#print TVA.tl.upToNaN
#print TVA.tl.addrs
#print TVA.Tensor.ltS
#print TVA.IntV.lt
#print TVA.IntV.ofNat
#print TVA.Tensor.ptrAdd
#print TVA.Ptr.offset
#print TVA.Launch.grid1
#print TVA.Launch.ctx
#print TVA.Kernel
#print TVA.Rules.Compiler.Faithful

-- L118: the 32-bit index type the model uses, and the 8-bit one the docstring sat on before
#print TVA.I32
#print TVA.I8

-- the claim's statement
#check @Kernels.VecAdd.main
#check @Kernels.VecAdd.consistent
#check @Kernels.VecAdd.satisfiable

#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
#print axioms Kernels.VecAdd.satisfiable
