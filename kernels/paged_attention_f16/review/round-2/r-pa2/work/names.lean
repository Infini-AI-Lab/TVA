import PagedAttention

/-! What every name in the model, the launch and the claim's statements means: printed with full
names, so that each operation shows as `TVA.*` or as a kernel definition read in full. -/

set_option pp.fullNames true
set_option pp.notation false
set_option pp.proofs false
set_option pp.deepTerms true
set_option pp.maxSteps 10000000

#print Kernels.PagedAttention.paged_attention_kernel
#print Kernels.PagedAttention.model
#print Kernels.PagedAttention.launch
#print Kernels.PagedAttention.Post
#print Kernels.PagedAttention.SeqOk
#print Kernels.PagedAttention.Pre
#print Kernels.PagedAttention.Assumptions
#print axioms Kernels.PagedAttention.main
#print axioms Kernels.PagedAttention.consistent
#print axioms Kernels.PagedAttention.satisfiable
