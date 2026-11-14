import TVA.Triton.Memory.Rel
import TVA.Triton.Arith.Rel

/-!
# The reductions of descriptor atomics

The reductions of `desc.atomic_*` (`RedOp`), the dtypes Triton allows for each, and the new value
each may write (`RedOp.Upd`). `TVA.Triton.Desc` gives the operations; these come before the
catalog of sites (`TVA.Site`).
-/
namespace TVA

/-- The reductions of `desc.atomic_*`. -/
inductive RedOp where
  | add | min | max | and | or | xor
  deriving DecidableEq, Repr

namespace RedOp

/-- The dtypes Triton allows for each reduction (`semantic.py:1092-1141`). -/
def Allowed : RedOp → DType → Prop
  | .add, d => d ∈ [DType.u32, .i32, .u64, .f32, .f16, .bf16]
  | .min, d | .max, d => d ∈ [DType.u32, .i32, .u64, .i64, .f16, .bf16]
  | .and, d | .or, d | .xor, d => d ∈ [DType.u32, .i32, .u64, .i64]

instance (op : RedOp) (d : DType) : Decidable (op.Allowed d) := by
  cases op <;> unfold Allowed <;> infer_instance

/-- The new value of an element under a float `min`/`max` reduction: as `tl.maximum`, or a NaN
when an operand is NaN (the hardware may propagate it). -/
def fMinMax {f : Format} (isMax : Bool) (v o n : FloatV f) : Prop :=
  FloatV.MaxRel isMax o v n ∨ ((o.isNaN = true ∨ v.isNaN = true) ∧ n.isNaN = true)

/-- `Upd e op d v o n`: the new value `n` of an element holding `o` reduced with `v`. -/
def Upd (e : Env) : RedOp → (d : DType) → d.Val → d.Val → d.Val → Prop
  | .add, d => tl.addUpd e d
  | .min, .f16 => fMinMax false | .min, .bf16 => fMinMax false
  | .max, .f16 => fMinMax true | .max, .bf16 => fMinMax true
  | .min, d => tl.maxUpd false d
  | .max, d => tl.maxUpd true d
  | .and, d => tl.bitUpd IntV.and d
  | .or, d => tl.bitUpd IntV.or d
  | .xor, d => tl.bitUpd IntV.xor d

end RedOp

end TVA
