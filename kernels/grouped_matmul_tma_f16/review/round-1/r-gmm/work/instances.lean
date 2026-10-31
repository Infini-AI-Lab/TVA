import GroupedMatmulTma

/-! r-gmm, round 1: which instances the model's operators `+`, `-`, `*` and `&&` resolve to. -/

set_option pp.fullNames true

open TVA

#synth HAdd (Ptr .i32) I32 (Ptr .i32)
#synth HAdd (Ptr .i64) I32 (Ptr .i64)
#synth HAdd I32 I32 I32
#synth HSub I32 I32 I32
#synth HMul I32 I32 I32

set_option pp.explicit true in
example (p : Ptr .i32) (g : I32) : p + g * (IntV.ofInt 3 : I32) = p + g * (IntV.ofInt 3 : I32) := by
  show (p + g * (IntV.ofInt 3 : I32)) = _
  rfl

-- the meaning of the instances
#print TVA.Ptr.offset
#print TVA.IntV.toInt
