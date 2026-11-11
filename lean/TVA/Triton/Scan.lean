import TVA.Triton.Arith

/-!
# Scans and sorting: `tl.associative_scan`, `tl.cumsum`, `tl.sort`

**Scans** (`core.py:3141`, `standard.py:334-365`). Triton requires the combine function to be
associative and evaluates it along a tree of its choosing: sequentially within a thread, then
across lanes and warps. Every combination is `combine_fn(acc, cur)` with `acc` the earlier segment
(an empty accumulator gives `cur` itself; there is no identity element). So output position `j` is
the evaluation of some binary tree whose leaves are `0, …, j` in order (`ScanLeaves`).
`reverse=True` is lowered as `flip(scan(flip(x)))`, so its leaves are `n-1, …, j` in that order:
the left operand covers the higher indices. The combine function is a relation, the model's
reading of the source's `combine_fn` (`fun a b c => c = a + b` for an integer addition, `AddRel`
for a float one, `FloatV.MaxRel` for `tl.maximum`).

Triton's promotions are written in the model: `tl.cumsum` converts `bfloat16` to `float32` and
integers narrower than 32 bits to 32 bits (keeping signedness); with `dtype=d` it converts
`bfloat16` to `float32`, then the input to `d`, and scans in `d` (`standard.py:270-281`).
`tl.cumprod` converts `bfloat16` to `float32`.

**Sorting** (`tl.sort`, `standard.py:374-466`): a bitonic network along the last dimension. Each
compare-and-swap keeps an element unless `x > y` says to take the partner's, so with an unordered
pair (a NaN) both positions may end up with the same element, and with `-0`/`+0` either zero may
be copied. Each row without NaN comes out as a permutation of itself up to `==`, sorted; a row with
a NaN may come out as anything.
-/
namespace TVA

/-- The leaves a scan combines for position `j` of an axis of size `n`: `0, …, j` in order;
reversed, `n-1, …, j` in that order. -/
def ScanLeaves (reverse : Bool) (n j : Nat) (t : Tree) : Prop :=
  if reverse then t.leaves = (List.range' j (n - j)).reverse
  else t.leaves = List.range (j + 1)

/-- How `tl.sort` compares elements: `lt` is Triton's `<`; `Ordered` elements are those `lt`
orders totally (not NaN); `Same` is equality for `lt` (`==`: `-0 == +0` for floats). -/
class SortKey (α : Type) where
  /-- Triton's `<`. -/
  lt : α → α → Bool
  /-- The element is ordered (not NaN). -/
  Ordered : α → Prop
  /-- Triton's `==`. -/
  Same : α → α → Prop

instance {sg : Bool} {w : Nat} : SortKey (IntV sg w) := ⟨IntV.lt, fun _ => True, Eq⟩
instance {f : Format} : SortKey (FloatV f) :=
  ⟨FloatV.lt, fun x => x.isNaN = false, fun x y => FloatV.eq x y = true⟩

/-- Row `i` of `r` along axis `a` (of size `n`) is sorted: ascending, no later element is below an
earlier one; descending, the reverse. -/
def SortedAlong {s : Shape} {α : Type} [SortKey α] (descending : Bool) (a n : Nat) (r : Tensor s α)
    (i : Idx s) : Prop :=
  ∀ t₁ t₂, t₁ < t₂ → t₂ < n →
    SortKey.lt (r.get (i.set a (if descending then t₁ else t₂)))
      (r.get (i.set a (if descending then t₂ else t₁))) = false

/-- Row `i` of `r` along axis `a` is a permutation of `x`'s, up to `Same`. -/
def PermAlong {s : Shape} {α : Type} [SortKey α] (a n : Nat) (x r : Tensor s α) (i : Idx s) : Prop :=
  ∃ σ : Nat → Nat, (∀ t, t < n → σ t < n) ∧ (∀ t₁ t₂, t₁ < n → t₂ < n → σ t₁ = σ t₂ → t₁ = t₂) ∧
    ∀ t, t < n → SortKey.Same (r.get (i.set a t)) (x.get (i.set a (σ t)))

namespace tl

variable {s : Shape} {α : Type}

/-! ## Scans -/

/-- `tl.associative_scan(x, axis, combine_fn, reverse)`, with the combine function given as the
relation `op a b c` (`c` is a possible result of `combine_fn(a, b)`). -/
def associative_scan (x : Tensor s α) (axis : Nat) (op : α → α → α → Prop)
    (reverse : Bool := false) : Prog (Tensor s α) :=
  .pick (fun r : Tensor s α => ∀ i, ∃ t : Tree,
    ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
    t.Eval op (fun j => x.get (i.set axis j)) (r.get i)) .ret

section Float
variable {f : Format}

/-- A float cumulative sum whose additions are `AddRel fusion`. -/
def cumsumWith (fusion : Bool) (x : Tensor s (FloatV f)) (axis : Nat := 0)
    (reverse : Bool := false) : Prog (Tensor s (FloatV f)) :=
  associative_scan x axis (AddRel fusion) reverse

/-- `tl.cumsum(x, axis, reverse)` of a float tensor that is not a product: IEEE additions. -/
def cumsum (x : Tensor s (FloatV f)) (axis : Nat := 0) (reverse : Bool := false) :
    Prog (Tensor s (FloatV f)) :=
  cumsumWith false x axis reverse

/-- `tl.cumsum` of a product: the additions may be fused (as `tl.sumc`). -/
def cumsumc (x : Tensor s (FloatV f)) (axis : Nat := 0) (reverse : Bool := false) :
    Prog (Tensor s (FloatV f)) :=
  .env fun e => cumsumWith e.fusion x axis reverse

/-- `tl.cumprod(x, axis, reverse)` of a float tensor: IEEE products. -/
def cumprod (x : Tensor s (FloatV f)) (axis : Nat := 0) (reverse : Bool := false) :
    Prog (Tensor s (FloatV f)) :=
  associative_scan x axis (fun a b c => FloatV.NaNEq (FloatV.mul a b) c) reverse

end Float

/-- `tl.cumsum` of an integer tensor (wrapping). -/
def icumsum {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) (axis : Nat := 0)
    (reverse : Bool := false) : Prog (Tensor s (IntV sg w)) :=
  associative_scan x axis (fun a b c => c = a + b) reverse

/-- `tl.cumprod` of an integer tensor (wrapping). -/
def icumprod {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) (axis : Nat := 0)
    (reverse : Bool := false) : Prog (Tensor s (IntV sg w)) :=
  associative_scan x axis (fun a b c => c = a * b) reverse

/-! ## Sorting -/

/-- `tl.sort(x, descending=…)` along the last dimension: each row without NaN comes out as a
permutation of itself (up to `==`), sorted; a row with a NaN is unconstrained. -/
def sort [SortKey α] (x : Tensor s α) (descending : Bool := false) : Prog (Tensor s α) :=
  .pick (fun r : Tensor s α => ∀ i,
    (∀ t, t < Shape.dim s (s.length - 1) → SortKey.Ordered (x.get (i.set (s.length - 1) t))) →
      PermAlong (s.length - 1) (Shape.dim s (s.length - 1)) x r i ∧
      SortedAlong descending (s.length - 1) (Shape.dim s (s.length - 1)) r i) .ret

end tl

end TVA
