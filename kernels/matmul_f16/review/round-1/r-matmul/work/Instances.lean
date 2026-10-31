import Matmul

/-! r-matmul, round 1: the instances behind the model's operators, and the `tl.dot` configuration's
defaults. -/

set_option pp.fullNames true
set_option pp.notation false
set_option pp.structureInstances false

#print Kernels.Matmul.matmul_kernel
#print TVA.DotCfg
#print TVA.tl.dot
#print TVA.Tensor.arange
#print TVA.tl.arange
#print TVA.tl.irem
#print TVA.tl.idiv
#print TVA.tl.iminimum
#print TVA.tl.store
#print TVA.tl.addrs
#print TVA.tl.forRange
#print TVA.tl.zeros
#print TVA.tl.program_id
#print TVA.Tensor.ptrAdd
#print TVA.Tensor.ptrAddT
#print TVA.Ptr.offset
#print TVA.Buf.ptr
#print TVA.IntV.lt
#print TVA.IntV.le
#print TVA.Tensor.and
#print TVA.Launch.size
#print TVA.Launch.ctx
