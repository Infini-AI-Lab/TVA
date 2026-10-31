import AttnBwd

/-! Reviewer r-bwd2: print the model, the launch, the conditions, the specification and the
assumptions with full names, to check that every operation is a `TVA.*` definition or a kernel
definition read in full. Also the axioms of `main`. -/

set_option pp.fullNames true
set_option pp.proofs false

#print Kernels.AttnBwd.dotCfg
#print Kernels.AttnBwd.attn_bwd_dkdv
#print Kernels.AttnBwd.attn_bwd_dq
#print Kernels.AttnBwd.attn_bwd
#print Kernels.AttnBwd.model
#print Kernels.AttnBwd.launch
#print Kernels.AttnBwd.blocks
#print Kernels.AttnBwd.Inst.ok
#print Kernels.AttnBwd.Pre
#print Kernels.AttnBwd.Post
#print Kernels.AttnBwd.OutSpec
#print Kernels.AttnBwd.HeadV.Ok
#print Kernels.AttnBwd.Assumptions
#print axioms Kernels.AttnBwd.main
#print axioms Kernels.AttnBwd.consistent
#print axioms Kernels.AttnBwd.satisfiable
