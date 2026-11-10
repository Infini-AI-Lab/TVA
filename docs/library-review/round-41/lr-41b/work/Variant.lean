/- lr-41b, round 41: the IntW docstring example of safe_convert_iff (V1, V3) against the form that works (V2). V1 and V3 fail to elaborate. -/
import TVA
namespace TVA
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
-- V1: the docstring's form, at shape [n]
example {n : Nat} (x : Tensor [n] I32) (k : Tensor [n] I64 → Prog α) :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff (fun i r => ConvRel.iff_cast_i32_i64) (y := x.castInt)
-- V2: the author's form, at a general shape
example {s : Shape} (x : Tensor s I32) (k : Tensor s I64 → Prog α) :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff fun _ _ => ConvRel.iff_cast_i32_i64
-- V3: named argument first
example {s : Shape} (x : Tensor s I32) (k : Tensor s I64 → Prog α) :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff (y := x.castInt) (fun i r => ConvRel.iff_cast_i32_i64)
end TVA
