import TVA
open TVA

#print axioms TVA.Platform.sm90ref.dotI8
#print axioms TVA.Tensor.get_broadcastTo
#print axioms TVA.tl.bitcast_self
#print axioms TVA.DType.ofBits_toBits
#check @TVA.Rules.Num.DotI8
#print TVA.DType.bitwidth
#check @TVA.tl.bitcast
-- DotI8 at a concrete platform does not hold vacuously: the room is the int32 bound
example : TVA.DType.bound .i32 = 2 ^ 31 - 1 := rfl
-- broadcastTo default proof for literal shapes
example (x : TVA.Tensor [4, 1] Nat) : TVA.Tensor [4, 8] Nat := x.broadcastTo [4, 8]
