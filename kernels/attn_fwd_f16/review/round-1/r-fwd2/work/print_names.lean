import AttnFwd

/-! Reviewer r-fwd2: print the model, the launch and the conditions with full names, and the
axioms of the claim. -/

set_option pp.fullNames true
set_option pp.proofs true

#print Kernels.AttnFwd.maybe_make_tensor_desc
#print Kernels.AttnFwd.cfg16
#print Kernels.AttnFwd.attn_fwd_inner
#print Kernels.AttnFwd.attn_fwd
#print Kernels.AttnFwd.model
#print Kernels.AttnFwd.launch
#print Kernels.AttnFwd.descArg
#print Kernels.AttnFwd.below
#print Kernels.AttnFwd.Inst.ok
#print Kernels.AttnFwd.Pre
#print Kernels.AttnFwd.Post
#print Kernels.AttnFwd.Assumptions
#print axioms Kernels.AttnFwd.main
#print axioms Kernels.AttnFwd.consistent
#print axioms Kernels.AttnFwd.satisfiable
