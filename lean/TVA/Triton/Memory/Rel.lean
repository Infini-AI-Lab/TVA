import TVA.Core

/-!
# Padding and the updates of atomics

What a lane outside the bounds of a block-pointer or descriptor load holds (`Padding`,
`Padding.Holds`), and the new value an atomic may write (`tl.addUpd`, `tl.maxUpd`, `tl.bitUpd`).
`TVA.Triton.Memory` gives the loads and the atomics; these come before the catalog of sites
(`TVA.Site`).
-/
namespace TVA

/-- `padding_option` of a block-pointer or descriptor load: what a lane outside the bounds holds.
`undef` is an unspecified value (block pointers' default `""`), `zero` the zero bit pattern, `nan`
a NaN (float dtypes only; Triton rejects it for integers). -/
inductive Padding where
  /-- An unspecified value. -/
  | undef
  /-- The zero bit pattern. -/
  | zero
  /-- A NaN. -/
  | nan
  deriving DecidableEq, Repr

/-- A lane outside the bounds holds what the padding says. -/
def Padding.Holds {d : DType} : Padding → d.Val → Prop
  | .undef, _ => True
  | .zero, x => x = d.ofBits 0
  | .nan, x => d.IsNaN x

instance (d : DType) (x : d.Val) : Decidable (d.IsNaN x) := by
  cases d <;> simp only [DType.IsNaN] <;> infer_instance

namespace tl

/-- The new value `n` of an `atomic_add` of `v` to a cell holding `o`: integers wrap; floats as the
platform's memory system rounds (`Platform.atomicAdd`). -/
def addUpd (e : Env) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = o + v | .i16, v, o, n => n = o + v | .i32, v, o, n => n = o + v
  | .i64, v, o, n => n = o + v
  | .u8, v, o, n => n = o + v | .u16, v, o, n => n = o + v | .u32, v, o, n => n = o + v
  | .u64, v, o, n => n = o + v
  | .f16, v, o, n => e.P.atomicAdd _ o v n | .bf16, v, o, n => e.P.atomicAdd _ o v n
  | .f32, v, o, n => e.P.atomicAdd _ o v n | .f64, v, o, n => e.P.atomicAdd _ o v n
  | _, _, _, _ => False

/-- The new value `n` of a float `atomic_max` (`isMax`) or `atomic_min` of `v` on a cell holding
`o`: one of the two, and, when neither is NaN, not below (above) either. Which zero wins between
`-0.0` and `+0.0`, and which value with a NaN operand, are left open (Triton implements them with
integer atomics on the bit pattern, `semantic.py:1292-1373`). -/
def fBound {f : Format} (isMax : Bool) (o v n : FloatV f) : Prop :=
  (n = o ∨ n = v) ∧ (o.isNaN = false → v.isNaN = false →
    if isMax then o.le n = true ∧ v.le n = true else n.le o = true ∧ n.le v = true)

/-- The new value of `atomic_max` (`isMax`) or `atomic_min`: integers compare as their signedness
says, floats as `fBound`. -/
def maxUpd (isMax : Bool) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = if isMax then o.max v else o.min v
  | .i16, v, o, n => n = if isMax then o.max v else o.min v
  | .i32, v, o, n => n = if isMax then o.max v else o.min v
  | .i64, v, o, n => n = if isMax then o.max v else o.min v
  | .u8, v, o, n => n = if isMax then o.max v else o.min v
  | .u16, v, o, n => n = if isMax then o.max v else o.min v
  | .u32, v, o, n => n = if isMax then o.max v else o.min v
  | .u64, v, o, n => n = if isMax then o.max v else o.min v
  | .f16, v, o, n => fBound isMax o v n | .bf16, v, o, n => fBound isMax o v n
  | .f32, v, o, n => fBound isMax o v n | .f64, v, o, n => fBound isMax o v n
  | _, _, _, _ => False

/-- The new value of a bitwise atomic (`op` on the bit vectors), on integers only. -/
def bitUpd (op : ∀ {sg : Bool} {w : Nat}, IntV sg w → IntV sg w → IntV sg w) :
    (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = op o v | .i16, v, o, n => n = op o v | .i32, v, o, n => n = op o v
  | .i64, v, o, n => n = op o v
  | .u8, v, o, n => n = op o v | .u16, v, o, n => n = op o v | .u32, v, o, n => n = op o v
  | .u64, v, o, n => n = op o v
  | _, _, _, _ => False

instance {f : Format} (isMax : Bool) (o v n : FloatV f) : Decidable (fBound isMax o v n) := by
  unfold fBound; cases isMax <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;> infer_instance

end tl

end TVA
