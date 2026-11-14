import TVA.Core

/-!
# The relations of index reductions and histograms

Triton's comparisons on an element type (`CmpKey`), the combine of `tl.argmax`/`tl.argmin`
(`tl.argCombine`), what an index reduction may return (`tl.ArgRes`), and the lanes a histogram's
mask enables (`tl.enabledBy`). `TVA.Triton.Reduce` gives the operations; these come before the
catalog of sites (`TVA.Site`).
-/
namespace TVA

/-- The comparisons Triton's index reductions use on an element type: `<` and `==` (signed or
unsigned integer comparison; the ordered float comparison, false with a NaN), and which elements
are NaN (none for integers). -/
class CmpKey (α : Type) where
  /-- Triton's `<`. -/
  lt : α → α → Bool
  /-- Triton's `==`. -/
  eq : α → α → Bool
  /-- The element is NaN. -/
  nan : α → Bool

instance {sg : Bool} {w : Nat} : CmpKey (IntV sg w) := ⟨IntV.lt, IntV.eq, fun _ => false⟩
instance {f : Format} : CmpKey (FloatV f) := ⟨FloatV.lt, FloatV.eq, FloatV.isNaN⟩

namespace tl

variable {α : Type}

/-- Triton's `_argmax_combine` (`isMax`) and `_argmin_combine` on two `(value, index)` pairs. `a`
wins when its value is strictly greater (`isMax`; smaller otherwise) than `b`'s, or, with
`tieLeft` (`tie_break_left=True`, the default), when the values are `==` and its index is smaller;
otherwise `b` wins. -/
def argCombine [CmpKey α] (isMax tieLeft : Bool) (a b : α × Nat) : α × Nat :=
  let tie := tieLeft && CmpKey.eq a.1 b.1 && decide (a.2 < b.2)
  let win := (if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) || tie
  if win then a else b

/-- What an index reduction over the elements `g 0, …, g (n-1)` may return: the pair that
`argCombine` gives along some tree whose leaf `j` is `(g j, j)`; or, when some element is NaN, a
value that is one of the elements (up to NaN payload) with an index in range, not necessarily
from the same position (the device's `float16` and `bfloat16` results do mix them). -/
def ArgRes [CmpKey α] (isMax tieLeft : Bool) (n : Nat) (g : Nat → α) (r : α × I32) : Prop :=
  (∃ p : α × Nat, ∃ t : Tree, t.Covers n ∧
    t.Eval (fun a b c => c = argCombine isMax tieLeft a b) (fun j => (g j, j)) p ∧
    r = (p.1, IntV.ofNat p.2)) ∨
  ((∃ j, j < n ∧ CmpKey.nan (g j) = true) ∧
    (∃ j, j < n ∧ (r.1 = g j ∨ (CmpKey.nan r.1 = true ∧ CmpKey.nan (g j) = true))) ∧
    ∃ j, j < n ∧ r.2 = IntV.ofNat j)

/-- Lane `j` is enabled by an optional mask. -/
def enabledBy {n : Nat} (mask : Option (Tensor [n] Bool)) (j : Fin n) : Bool :=
  match mask with
  | none => true
  | some mk => mk.get1 j

end tl

end TVA
