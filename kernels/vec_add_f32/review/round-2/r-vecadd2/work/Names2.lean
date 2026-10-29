import VecAdd

/-! r-vecadd2, round 2, at gate 990243026f1289fb: what every name of the claim means, printed
with full names and explicit instances, and the axioms of the three theorems. The kernel's own
constants are unchanged since gate 401e1fa92db421d7 (the snapshots' cones differ only in library
constants and files); this re-prints them against the library as it is now. -/

set_option pp.fullNames true
set_option pp.notation false

section Plain
#print Kernels.VecAdd.add_kernel
#print Kernels.VecAdd.model
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.Inst
#print Kernels.VecAdd.Inst.ok
#print Kernels.VecAdd.Pre
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Spec
#print Kernels.VecAdd.Assumptions
end Plain

section Explicit
set_option pp.explicit true
#print Kernels.VecAdd.add_kernel
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Assumptions
end Explicit

-- The library definitions the model and the specification reach, as they are now.
#print TVA.Launch.grid1
#print TVA.Launch.Valid
#print TVA.Rules.Compiler.Faithful
#print TVA.Claim.Meets
#print TVA.FloatV.NaNEq
#print TVA.FloatV.add
#print TVA.Buf.Valid
#print TVA.Buf.elems
#print TVA.Mem.read
#print TVA.tl.load
#print TVA.tl.store
#print TVA.tl.add
#print TVA.tl.addrs
#print TVA.Tensor.ltS
#print TVA.tl.arange
#print TVA.tl.program_id

#check @Kernels.VecAdd.main
#check @Kernels.VecAdd.consistent
#check @Kernels.VecAdd.satisfiable

#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
#print axioms Kernels.VecAdd.satisfiable
