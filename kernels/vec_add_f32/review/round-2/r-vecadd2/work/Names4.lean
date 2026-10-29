import VecAdd

/-! r-vecadd2, round 2, at gate a2e8d45aaf58a6d2 (filing 3): what every name of the claim means,
printed with full names and explicit instances, and the axioms of the three theorems. Judged in
full against the kernel's Lean as it is now, after the prover's wording fixes and library round 22
(L116). -/

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
#print Kernels.VecAdd.Inst.ok
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Assumptions
end Explicit

-- The library definitions the model, the launch and the specification reach, as they are now.
#print TVA.Launch.grid1
#print TVA.Launch.Valid
#print TVA.Launch.ctx
#print TVA.Rules.Compiler.Faithful
#print TVA.Claim.Meets
#print TVA.FloatV.NaNEq
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
#print TVA.Tensor.ptrAdd

#check @Kernels.VecAdd.main
#check @Kernels.VecAdd.consistent
#check @Kernels.VecAdd.satisfiable

#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
#print axioms Kernels.VecAdd.satisfiable
