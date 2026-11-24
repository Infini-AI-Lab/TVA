import TVA.Core

/-!
# The relations of scans

Which tree a scan may evaluate for an output position (`ScanLeaves`), and a scan's combine
(`Combine`): the relation the model reads `combine_fn` as, with a function that gives one of its
results. `TVA.Triton.Scan` gives the operations; these come before the catalog of sites
(`TVA.Site`), whose `Site.scan` they describe.
-/
namespace TVA

/-- The leaves a scan combines for position `j` of an axis of size `n`: `0, …, j` in order;
reversed, `n-1, …, j` in that order. -/
def ScanLeaves (reverse : Bool) (n j : Nat) (t : Tree) : Prop :=
  if reverse then t.leaves = (List.range' j (n - j)).reverse
  else t.leaves = List.range (j + 1)

instance (reverse : Bool) (n j : Nat) (t : Tree) : Decidable (ScanLeaves reverse n j t) := by
  unfold ScanLeaves; infer_instance

/-- A scan's combine: `rel a b c` says `c` is a possible result of `combine_fn(a, b)` (the
model's reading of the source's function), and `fn a b` is always one of them (`fn_rel`). Only
`rel` is the meaning; `fn` is a witness, so a combine always has a result and an executor can
compute one. -/
structure Combine (α : Type) where
  /-- The results `combine_fn(a, b)` may give. -/
  rel : α → α → α → Prop
  /-- One of them. -/
  fn : α → α → α
  /-- `fn a b` is a result `rel` allows. -/
  fn_rel : ∀ a b, rel a b (fn a b)

/-- A deterministic combine: `combine_fn(a, b)` is `g a b` (integer and bitwise operations,
integer `max`, tuples of them). -/
def Combine.ofFn {α : Type} (g : α → α → α) : Combine α :=
  ⟨fun a b c => c = g a b, g, fun _ _ => rfl⟩

end TVA
