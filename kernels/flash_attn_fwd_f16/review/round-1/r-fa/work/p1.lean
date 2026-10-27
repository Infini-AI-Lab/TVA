import FlashAttn
set_option pp.fullNames true
open TVAExamples.FlashAttn
#print TVAExamples.FlashAttn.inner
#print TVAExamples.FlashAttn.attn_fwd
#print TVA.Softmax.posInf32
#print TVA.Softmax.negInf32
#print TVAExamples.FlashAttn.zero32
#print axioms TVAExamples.FlashAttn.main
#print axioms TVAExamples.FlashAttn.consistent
