import AttnFwdE5m2

/-! Reviewer r-e5fwd2: the instances behind the model's operators, and more library names. -/

set_option pp.fullNames true
set_option pp.proofs false

section
set_option pp.explicit true
#check fun (M : TVA.Ptr .f32) (a b : TVA.I32) (o : TVA.Tensor [128] TVA.I32) => M + a * b + o
#check fun (s : TVA.I32) (o : TVA.Tensor [128, 64] TVA.I32) => s + o
#check fun (a b c : TVA.I32) => a * b * c + a
end

#print TVA.IntV.le
#print TVA.TensorDesc.elemAt
#print TVA.Ptr.offset
#print TVA.tl.addrs
#print TVA.Buf.ptr
#print TVA.tl.RangeOk
#print TVA.tl.tripCount
#print TVA.tl.forN
#print TVA.tl.exp2
#print TVA.tl.log2
#print TVA.tl.div
#print TVA.tl.sub
#print TVA.tl.add
#print TVA.tl.mul
#print TVA.tl.maximum
#print TVA.tl.addc
#print TVA.FloatV.litF32
#print TVA.Tensor.scalar
#print TVA.Tensor.item
#print TVA.tl.zeros
#print TVA.tl.arange
#print TVA.Tensor.trans
#print TVA.DotCfg
#print TVA.Padding.Holds
#print TVA.tl.sum1With
#print TVA.Ctx
