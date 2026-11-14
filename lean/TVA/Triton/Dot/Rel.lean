import TVA.Triton.Arith.Rel

/-!
# The relations of dot products

The addition of an accumulator dtype (`AccAdd`) and the product of a float dtype (`tl.MulRel`),
which a `tl.dot` folded with an addition and a sum of broadcast products allow
(`TVA.Triton.Dot` gives the operations). They come before the catalog of sites (`TVA.Site`).
-/
namespace TVA

/-- The addition of an accumulator dtype: `AddRel` for the standard float formats, wrapping `+`
for integers, nothing for others. -/
def AccAdd (fusion : Bool) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .f16, x, y, r => AddRel fusion x y r | .bf16, x, y, r => AddRel fusion x y r
  | .f32, x, y, r => AddRel fusion x y r | .f64, x, y, r => AddRel fusion x y r
  | .i8, x, y, r => r = x + y | .i16, x, y, r => r = x + y | .i32, x, y, r => r = x + y
  | .i64, x, y, r => r = x + y
  | .u8, x, y, r => r = x + y | .u16, x, y, r => r = x + y | .u32, x, y, r => r = x + y
  | .u64, x, y, r => r = x + y
  | _, _, _, _ => False

namespace tl

/-- The product of a standard float dtype: the IEEE product up to NaN payload (nothing for other
dtypes). -/
def MulRel : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .f16, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | .bf16, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | .f32, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | .f64, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | _, _, _, _ => False

end tl

end TVA
