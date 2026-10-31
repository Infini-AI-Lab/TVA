import VecAdd

/-! r-vecadd2, round 2, at gate 5f65eb1f7a961315 (filing 4): what every name of the claim means,
printed with full names and explicit instances, the library definitions it reaches as they are
after library round 23 (L117, L118), and the axioms of the three theorems. -/

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
#print Kernels.VecAdd.model
#print Kernels.VecAdd.launch
#print Kernels.VecAdd.Inst.ok
#print Kernels.VecAdd.Pre
#print Kernels.VecAdd.Post
#print Kernels.VecAdd.Assumptions
end Explicit

-- L118: the `int32` docstring is now on `I32`, and `I8` has its own; the definitions are as before.
#print TVA.I8
#print TVA.I32

-- The library definitions the model, the launch and the specification reach, as they are now.
#print TVA.Launch.grid1
#print TVA.Launch.Valid
#print TVA.Launch.ctx
#print TVA.Launch.size
#print TVA.Rules.Compiler.Faithful
#print TVA.Claim.Meets
#print TVA.FloatV.NaNEq
#print TVA.Buf.Valid
#print TVA.Buf.elems
#print TVA.Buf.ptr
#print TVA.Mem.read
#print TVA.tl.load
#print TVA.tl.store
#print TVA.tl.add
#print TVA.tl.upToNaN
#print TVA.tl.addrs
#print TVA.Tensor.ltS
#print TVA.tl.arange
#print TVA.Tensor.arange
#print TVA.tl.program_id
#print TVA.Tensor.ptrAdd
#print TVA.Ptr.offset
#print TVA.IntV.ofNat
#print TVA.IntV.lt
#print TVA.Tensor.instHAddIntV
#print TVA.instHAddPtrTensorIntV
#print TVA.IntV.instMul

#check @Kernels.VecAdd.main
#check @Kernels.VecAdd.consistent
#check @Kernels.VecAdd.satisfiable

#print axioms Kernels.VecAdd.main
#print axioms Kernels.VecAdd.consistent
#print axioms Kernels.VecAdd.satisfiable
