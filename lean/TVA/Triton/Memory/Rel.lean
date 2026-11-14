import TVA.Core

/-!
# Padding

What a lane outside the bounds of a block-pointer or descriptor load holds (`Padding`,
`Padding.Holds`). `TVA.Triton.Memory` gives the loads; this comes before the catalog of sites
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

end TVA
