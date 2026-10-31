import AttnFwd

/-! Reviewer r-fwd: print the model, launch, conditions and specification with full names, to
check that every operation is `TVA.*` or a kernel helper read in full. -/

set_option pp.fullNames true
set_option pp.proofs true

#print Kernels.AttnFwd.attn_fwd_inner
#print Kernels.AttnFwd.attn_fwd
#print Kernels.AttnFwd.maybe_make_tensor_desc
#print Kernels.AttnFwd.model
#print Kernels.AttnFwd.launch
#print Kernels.AttnFwd.descArg
#print Kernels.AttnFwd.cfg16
#print Kernels.AttnFwd.Assumptions
#print TVA.DotCfg
#print TVA.Launch
#print axioms Kernels.AttnFwd.main
