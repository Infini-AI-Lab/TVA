import TVA.Lang.Ops
import Mathlib.Data.Finset.Card

/-!
# Scans, sorting and histograms

**Scans** (`tl.associative_scan`, `tl.cumsum`, `tl.cumprod`; `core.py:3141`,
`standard.py:334-365`). Triton requires the combine function to be associative and evaluates
it along a tree of its choosing: sequentially within a thread, then across lanes and warps
(`ScanOpToLLVM.cpp`). Every combination is `combine_fn(acc, cur)` with `acc` the earlier
segment (`ReduceScanCommon.h:41-63`; an empty accumulator gives `cur` itself, no identity
element). So output position `j` is the evaluation of some binary tree whose leaves are
`0, …, j` in order (`ScanLeaves`). `reverse=True` is lowered as `flip(scan(flip(x)))`
(`ScanOpToLLVM.cpp:491-501, 559-562`), so its leaves are `n-1, …, j` in that order: the
left operand covers the higher indices. The combine function is given as a relation, the
model's reading of the source's `combine_fn` (`fun a b c => c = a + b` for an integer add,
`AddRel` for a float add, …). It is up to the model to state it, and the reviewer checks it.

`tl.cumsum` first converts `bfloat16` to `float32` and integers narrower than 32 bits to 32
bits; with `dtype=d` it converts `bfloat16` to `float32` and then the input to `d`, with no other
widening, and scans in `d` (`standard.py:270-281, 334-343`; `evidence/front-end-subset` S1-S3;
library issue L87). `tl.cumprod` converts `bfloat16` to `float32`. The model writes these
conversions.

**Sorting** (`tl.sort`, `standard.py:374-466`): a bitonic network along the last dimension.
Each compare-and-swap keeps an element unless the comparison `x > y` says to take the
partner's. With an unordered pair (a NaN), both positions may end up with the same element, and
with `-0`/`+0` either zero may be copied. So the library promises, for each row without NaN, a
permutation up to `==` (`SortKey.Same`) that is sorted. A row with a NaN may come out as
anything.

**Histogram** (`tl.histogram`, `semantic.py:1764-1774`): counts of the values `0 … bins-1`
among the enabled elements of a 1-D integer tensor. The documentation does not say what a value
outside `[0, bins)` does, so it is undefined behaviour here.
-/
namespace TVA

/-- The leaves a scan combines for position `j` of an axis of size `n`: `0, …, j` in order;
reversed, `n-1, …, j` in that order. -/
def ScanLeaves (reverse : Bool) (n j : Nat) (t : Tree) : Prop :=
  if reverse then t.leaves = (List.range' j (n - j)).reverse
  else t.leaves = List.range (j + 1)

/-- How `tl.sort` compares elements: `lt` is Triton's `<`. `Ordered` elements are those `lt`
orders totally (not NaN), and `Same` is equality for `lt` (`==`: `-0 == +0` for floats). -/
class SortKey (α : Type) where
  lt : α → α → Bool
  Ordered : α → Prop
  Same : α → α → Prop

instance {sg : Bool} {w : Nat} : SortKey (IntV sg w) := ⟨IntV.lt, fun _ => True, Eq⟩
instance {f : TVA.Math.Float.Format} : SortKey (FloatV f) :=
  ⟨FloatV.lt, fun x => x.isNaN = false, fun x y => FloatV.eq x y = true⟩

/-- Row `i` of `r` along axis `a` (of size `n`) is sorted: ascending, no later element is below
an earlier one; descending, the reverse. -/
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
def associative_scan (x : Tensor s α) (axis : Nat) (op : α → α → α → Prop) (reverse : Bool := false) :
    Prog (Tensor s α) :=
  .pick (fun r : Tensor s α => ∀ i, ∃ t : Tree, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
    t.Eval op (fun j => x.get (i.set axis j)) (r.get i)) Solve.none .ret

section Float
variable {f : TVA.Math.Float.Format}

/-- A float cumulative sum whose additions are `AddRel fusion`. -/
def cumsumWith (fusion : Bool) (x : Tensor s (FloatV f)) (axis : Nat := 0) (reverse : Bool := false) :
    Prog (Tensor s (FloatV f)) :=
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
def icumsum {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) (axis : Nat := 0) (reverse : Bool := false) :
    Prog (Tensor s (IntV sg w)) :=
  associative_scan x axis (fun a b c => c = a + b) reverse

/-- `tl.cumprod` of an integer tensor (wrapping). -/
def icumprod {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) (axis : Nat := 0) (reverse : Bool := false) :
    Prog (Tensor s (IntV sg w)) :=
  associative_scan x axis (fun a b c => c = a * b) reverse

/-! ## Sorting -/

/-- `tl.sort(x, descending=…)` along the last dimension: each row without NaN comes out as a
permutation of itself (up to `==`), sorted. -/
def sort [SortKey α] (x : Tensor s α) (descending : Bool := false) : Prog (Tensor s α) :=
  .pick (fun r : Tensor s α => ∀ i,
    (∀ t, t < Shape.dim s (s.length - 1) → SortKey.Ordered (x.get (i.set (s.length - 1) t))) →
      PermAlong (s.length - 1) (Shape.dim s (s.length - 1)) x r i ∧
      SortedAlong descending (s.length - 1) (Shape.dim s (s.length - 1)) r i) Solve.none .ret

/-! ## Histograms -/

/-- Lane `j` is enabled by an optional mask. -/
def enabledBy {n : Nat} (mask : Option (Tensor [n] Bool)) (j : Fin n) : Bool :=
  match mask with
  | none => true
  | some mk => mk.get1 j

/-- `tl.histogram(x, bins, mask)`: element `b` counts the enabled elements equal to `b`. -/
def histogram {n : Nat} {sg : Bool} {w : Nat} (x : Tensor [n] (IntV sg w)) (bins : Nat)
    (mask : Option (Tensor [n] Bool) := none) : Prog (Tensor [bins] I32) :=
  .pick (fun r : Tensor [bins] I32 =>
    (∀ j, enabledBy mask j = true → 0 ≤ (x.get1 j).toInt ∧ (x.get1 j).toInt < bins) ∧
    ∀ b : Fin bins, r.get1 b =
      IntV.ofNat (Finset.univ.filter fun j : Fin n => enabledBy mask j = true ∧ (x.get1 j).toInt = b).card) (Solve.only (Tensor.of1 fun b => IntV.ofNat (Finset.univ.filter fun j : Fin n => enabledBy mask j = true ∧ (x.get1 j).toInt = b).card) fun _ hr => ⟨hr.1, fun _ => rfl⟩) .ret

end tl

end TVA
