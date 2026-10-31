import AttnBwdPre

/-! Print the model, the launch, the conditions and the specification with full names, and the
axioms of the claim. -/

set_option pp.fullNames true
set_option pp.notation false
set_option pp.explicit false

#print Kernels.AttnBwdPre._attn_bwd_preprocess
#print Kernels.AttnBwdPre.model
#print Kernels.AttnBwdPre.launch
#print Kernels.AttnBwdPre.Pre
#print Kernels.AttnBwdPre.Post
#print Kernels.AttnBwdPre.OutSpec
#print Kernels.AttnBwdPre.Assumptions
#print Kernels.AttnBwdPre.Spec
#print axioms Kernels.AttnBwdPre.main
#print axioms Kernels.AttnBwdPre.consistent
#print axioms Kernels.AttnBwdPre.satisfiable
