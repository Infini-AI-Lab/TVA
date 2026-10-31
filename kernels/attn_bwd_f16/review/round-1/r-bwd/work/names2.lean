import AttnBwd

/-! r-bwd, round 1: the whole body of `attn_bwd`, with full names and no elision. -/

set_option pp.fullNames true
set_option pp.notation false
set_option pp.deepTerms true
set_option pp.proofs true
set_option pp.maxSteps 1000000

#print Kernels.AttnBwd.attn_bwd
#print TVA.IntV.cast
#print TVA.Tensor.ptrAdd
#print TVA.Tensor.ptrAddT
#print TVA.bc
#print TVA.FloatV.litF32
