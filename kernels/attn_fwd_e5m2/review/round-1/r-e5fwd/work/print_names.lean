import AttnFwdE5m2

/-! Reviewer r-e5fwd: print the model, the launch, the conditions, the specification and the
assumptions with full names, to check that every operation is a `TVA.*` definition or a kernel
helper whose definition was read; and the axioms of the three theorems. -/

set_option pp.fullNames true
set_option pp.proofs true

#print Kernels.AttnFwdE5m2.maybe_make_tensor_desc
#print Kernels.AttnFwdE5m2.cfg8
#print Kernels.AttnFwdE5m2.attn_fwd_inner
#print Kernels.AttnFwdE5m2.attn_fwd
#print Kernels.AttnFwdE5m2.descArg
#print Kernels.AttnFwdE5m2.descArgV
#print Kernels.AttnFwdE5m2.model
#print Kernels.AttnFwdE5m2.launch
#print Kernels.AttnFwdE5m2.Inst.ok
#print Kernels.AttnFwdE5m2.Pre
#print Kernels.AttnFwdE5m2.Post
#print Kernels.AttnFwdE5m2.HeadOk
#print Kernels.AttnFwdE5m2.Assumptions
#print Kernels.AttnFwdE5m2.Spec
#print axioms Kernels.AttnFwdE5m2.main
#print axioms Kernels.AttnFwdE5m2.consistent
#print axioms Kernels.AttnFwdE5m2.satisfiable
