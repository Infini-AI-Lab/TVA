import TVA.Triton.Arith

/-!
# Reductions: `tl.sum`, `tl.max`, `tl.argmax`, `tl.histogram`

**Float reductions** combine their elements along a tree Triton chooses (`Tree`, `Tree.Eval`): the
tree covers the axis's `n` positions in some order, and its depth is at most
`tl.sumBound P n = max (P.sumDepth n) ⌈log₂ n⌉` on the launch's platform (the device facts state
sm_90's depth). As for `tl.add`, the model says whether a product may feed the additions: `tl.sum`
(`sum1`, `sum0`) for inputs that are not products, every node an IEEE addition; `tl.sumc`
(`sum1c`, `sum0c`) for a sum of products (`tl.sum(a * b)`), every node `AddRel` with the launch's
fusion flag. `tl.max`/`tl.min` combine with `FloatV.MaxRel` along some tree.

**Integer reductions** do not depend on the order: they are the folds of `TVA.Tensor`
(`Tensor.foldAll`, `foldRows`, `foldCols`).

**Triton's promotions** are written in the model: `tl.max`/`tl.min` (without `return_indices`)
widen floats narrower than 32 bits to `float32` (`standard.py:178-190`); `tl.sum` widens no float;
the index reductions (`return_indices`, `tl.argmax`, `tl.argmin`) widen only `bfloat16`, to
`float32`. A reduction keeps the source's axis; with `keep_dims=True` the reduced axis stays, of
size 1 (the model reshapes).
-/
namespace TVA

/-! ## Solving a reduction

The solvers of the reductions' `pick`s (`Solve`) ask the oracle for a tree, fall back to the
sequential tree, and evaluate the chosen tree with a function that each node's relation allows
(the IEEE sum, `FloatV.max`, `argCombine`). -/

namespace Tree

/-- Evaluate a tree with the function `op` at each node. -/
def evalWith {α : Type} (op : α → α → α) (x : Nat → α) : Tree → α
  | leaf i => x i
  | node l r => op (evalWith op x l) (evalWith op x r)

/-- A function that each node's relation allows gives a result the tree can produce. -/
theorem eval_evalWith {α : Type} {rel : α → α → α → Prop} {op : α → α → α}
    (h : ∀ a b, rel a b (op a b)) (x : Nat → α) : (t : Tree) → t.Eval rel x (t.evalWith op x)
  | leaf i => .leaf i
  | node l r => .node (eval_evalWith h x l) (eval_evalWith h x r) (h _ _)

instance (n : Nat) (t : Tree) : Decidable (t.Covers n) := by unfold Covers; infer_instance

/-- A result along the oracle's tree for `n` leaves, or the sequential tree, if it satisfies
`C`; no answer otherwise. -/
def solve {α : Type} (n : Nat) (C : Tree → Prop) [DecidablePred C] {rel : α → α → α → Prop}
    (op : α → α → α) (h : ∀ a b, rel a b (op a b)) (x : Nat → α) (H : Hints) :
    Option (Ans (fun r => ∃ t : Tree, t.Covers n ∧ C t ∧ t.Eval rel x r)) :=
  let try1 (t : Tree) : Option (Ans (fun r => ∃ t : Tree, t.Covers n ∧ C t ∧ t.Eval rel x r)) :=
    if hc : t.Covers n ∧ C t then some (.ok _ ⟨t, hc.1, hc.2, eval_evalWith h x t⟩) else none
  match H (.tree n) with
  | some t => (try1 t).orElse fun _ => try1 (seq n)
  | none => try1 (seq n)

/-- `solve` without a condition on the tree. -/
def solve' {α : Type} (n : Nat) {rel : α → α → α → Prop} (op : α → α → α)
    (h : ∀ a b, rel a b (op a b)) (x : Nat → α) (H : Hints) :
    Option (Ans (fun r => ∃ t : Tree, t.Covers n ∧ t.Eval rel x r)) :=
  (solve n (fun _ => True) op h x H).map fun
    | .ok r hr => .ok r (by obtain ⟨t, h1, _, h2⟩ := hr; exact ⟨t, h1, h2⟩)
    | .empty hn => .empty fun r ⟨t, h1, h2⟩ => hn r ⟨t, h1, trivial, h2⟩
    | .assume r => .assume r

end Tree

/-- `FloatV.max` (`isMax`) or `FloatV.min`: a result `MaxRel` allows. -/
def FloatV.maxOf {f : Format} (isMax : Bool) (a b : FloatV f) : FloatV f :=
  if isMax then FloatV.max a b else FloatV.min a b

theorem FloatV.maxRel_maxOf {f : Format} (isMax : Bool) (a b : FloatV f) :
    FloatV.MaxRel isMax a b (FloatV.maxOf isMax a b) := by
  cases isMax <;> exact Or.inl (Or.inl rfl)

namespace tl

variable {s : Shape}

/-! ## Float reductions -/

section Float
variable {f : Format}

/-- The largest depth a float reduction's tree over `n` positions may have on `P`. -/
def sumBound (P : Platform) (n : Nat) : Nat := max (P.sumDepth n) (Nat.clog 2 n)

/-- A whole rank-1 reduction along some tree, each node an `AddRel fusion` addition. -/
def sumWith (fusion : Bool) {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) :=
  .env fun e => .pick (fun r => ∃ t : Tree, t.Covers n ∧ t.depth ≤ sumBound e.P n ∧
    t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r)
    ⟨Tree.solve n _ FloatV.add (AddRel.ieee fusion) _⟩ .ret

/-- Row reductions of a rank-2 tensor (`axis=1`), each along its own tree. -/
def sum1With (fusion : Bool) {m n : Nat} (x : Tensor [m, n] (FloatV f)) :
    Prog (Tensor [m] (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor [m] (FloatV f) => ∀ i, ∃ t : Tree, t.Covers n ∧
    t.depth ≤ sumBound e.P n ∧
    t.Eval (AddRel fusion) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.get1 i))
    (.each1 fun H _ => Tree.solve n _ FloatV.add (AddRel.ieee fusion) _ H) .ret

/-- Column reductions of a rank-2 tensor (`axis=0`), each along its own tree. -/
def sum0With (fusion : Bool) {m n : Nat} (x : Tensor [m, n] (FloatV f)) :
    Prog (Tensor [n] (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor [n] (FloatV f) => ∀ j, ∃ t : Tree, t.Covers m ∧
    t.depth ≤ sumBound e.P m ∧
    t.Eval (AddRel fusion) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.get1 j))
    (.each1 fun H _ => Tree.solve m _ FloatV.add (AddRel.ieee fusion) _ H) .ret

/-- `tl.sum(x)` of a rank-1 float tensor that is not a product: IEEE additions along a tree, in the
input's own format. -/
def sum {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) := sumWith false x
/-- `tl.sum(x, axis=1)` of a rank-2 float tensor, IEEE additions. -/
def sum1 {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  sum1With false x
/-- `tl.sum(x, axis=0)` of a rank-2 float tensor, IEEE additions. -/
def sum0 {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  sum0With false x

/-- `tl.sum(x)` of a product: the additions may be fused. -/
def sumc {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) := .env fun e => sumWith e.fusion x
/-- `tl.sum(x, axis=1)` of a product. -/
def sum1c {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  .env fun e => sum1With e.fusion x
/-- `tl.sum(x, axis=0)` of a product. -/
def sum0c {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  .env fun e => sum0With e.fusion x

/-- `tl.max(x, axis=1)` (`isMax`) or `tl.min(x, axis=1)` of a rank-2 float tensor: each row
combined with `MaxRel` along a tree. -/
def max1 {m n : Nat} (isMax : Bool) (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  .pick (fun r : Tensor [m] (FloatV f) => ∀ i, ∃ t : Tree, t.Covers n ∧
    t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default)
      (r.get1 i))
    (.each1 fun H _ => Tree.solve' n (FloatV.maxOf isMax) (FloatV.maxRel_maxOf isMax) _ H) .ret

/-- `tl.max(x)` (`isMax`) or `tl.min(x)` of a rank-1 float tensor: the whole tensor combined with
`MaxRel` along a tree. -/
def maxAll {n : Nat} (isMax : Bool) (x : Tensor [n] (FloatV f)) : Prog (FloatV f) :=
  .pick (fun r => ∃ t : Tree, t.Covers n ∧
    t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r)
    ⟨Tree.solve' n (FloatV.maxOf isMax) (FloatV.maxRel_maxOf isMax) _⟩ .ret

/-- `tl.max(x, axis=0)` (`isMax`) or `tl.min(x, axis=0)` of a rank-2 float tensor: each column
combined with `MaxRel` along a tree. -/
def max0 {m n : Nat} (isMax : Bool) (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  .pick (fun r : Tensor [n] (FloatV f) => ∀ j, ∃ t : Tree, t.Covers m ∧
    t.Eval (FloatV.MaxRel isMax) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default)
      (r.get1 j))
    (.each1 fun H _ => Tree.solve' m (FloatV.maxOf isMax) (FloatV.maxRel_maxOf isMax) _ H) .ret

end Float

end tl

/-! ## Index reductions

`tl.max(x, axis, return_indices=True)`, `tl.min(…, return_indices=True)`, `tl.argmax` and
`tl.argmin` reduce `(value, index)` pairs along an axis with Triton's `_argmax_combine` and
`_argmin_combine` (`standard.py:146-155, 199-208`), along a tree Triton chooses. The index is an
`int32` `tl.arange` along the axis (`core.py:3093-3107`). -/

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

variable {s : Shape} {α : Type}

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

/-- `ArgRes` along the oracle's tree (or the sequential one), for the solvers (`Solve`). -/
def argAns [CmpKey α] (isMax tieLeft : Bool) (n : Nat) (g : Nat → α) (H : Hints) :
    Option (Ans (ArgRes isMax tieLeft n g)) :=
  (Tree.solve' n (rel := fun a b c => c = argCombine isMax tieLeft a b) (argCombine isMax tieLeft)
    (fun _ _ => rfl) (fun j => (g j, j)) H).bind fun
    | .ok p hp => some (.ok (p.1, IntV.ofNat p.2) (Or.inl (by
        obtain ⟨t, h1, h2⟩ := hp
        exact ⟨p, t, h1, h2, rfl⟩)))
    | .empty _ => none
    | .assume p => some (.assume (p.1, IntV.ofNat p.2))

/-- `tl.max(x, 0, return_indices=True, return_indices_tie_break_left=tieLeft)` (`isMax`) or
`tl.min(…)` of a rank-1 tensor: a pair `ArgRes` allows. `tl.argmax(x, 0, tie_break_left=tieLeft)`
(`tl.argmin`) is its index. -/
def argmaxAll [CmpKey α] [Inhabited α] {n : Nat} (isMax tieLeft : Bool) (x : Tensor [n] α) :
    Prog (α × I32) :=
  .pick (fun r : α × I32 => ArgRes isMax tieLeft n
    (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r) ⟨argAns isMax tieLeft n _⟩ .ret

/-- `tl.max(x, 1, return_indices=True)` (`isMax`) or `tl.min(x, 1, …)` of a rank-2 tensor: for each
row, a pair `ArgRes` allows; `tl.argmax(x, 1)` is the index tensor. -/
def argmax1 [CmpKey α] [Inhabited α] {m n : Nat} (isMax tieLeft : Bool) (x : Tensor [m, n] α) :
    Prog (Tensor [m] α × Tensor [m] I32) :=
  .pick (fun r : Tensor [m] α × Tensor [m] I32 => ∀ i, ArgRes isMax tieLeft n
    (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.1.get1 i, r.2.get1 i))
    ((Solve.each1 fun H i => argAns isMax tieLeft n
      (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) H).conv
      (fun t => (t.map Prod.fst, t.map Prod.snd)) fun _ h i => h i) .ret

/-- `tl.max(x, 0, return_indices=True)` (`isMax`) or `tl.min(x, 0, …)` of a rank-2 tensor: for each
column, a pair `ArgRes` allows; `tl.argmax(x, 0)` is the index tensor. -/
def argmax0 [CmpKey α] [Inhabited α] {m n : Nat} (isMax tieLeft : Bool) (x : Tensor [m, n] α) :
    Prog (Tensor [n] α × Tensor [n] I32) :=
  .pick (fun r : Tensor [n] α × Tensor [n] I32 => ∀ j, ArgRes isMax tieLeft m
    (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.1.get1 j, r.2.get1 j))
    ((Solve.each1 fun H j => argAns isMax tieLeft m
      (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) H).conv
      (fun t => (t.map Prod.fst, t.map Prod.snd)) fun _ h j => h j) .ret

/-! ## Histograms -/

/-- Lane `j` is enabled by an optional mask. -/
def enabledBy {n : Nat} (mask : Option (Tensor [n] Bool)) (j : Fin n) : Bool :=
  match mask with
  | none => true
  | some mk => mk.get1 j

/-- `tl.histogram(x, bins, mask)` (`semantic.py:1764-1774`): bin `b` counts the enabled elements
equal to `b`, an `int32`. An enabled element outside `[0, bins)` is undefined behaviour (the
documentation does not say what it does). -/
def histogram {n : Nat} {sg : Bool} {w : Nat} (x : Tensor [n] (IntV sg w)) (bins : Nat)
    (mask : Option (Tensor [n] Bool) := none) : Prog (Tensor [bins] I32) :=
  .pick (fun r : Tensor [bins] I32 =>
    (∀ j, enabledBy mask j = true → 0 ≤ (x.get1 j).toInt ∧ (x.get1 j).toInt < bins) ∧
    ∀ b : Fin bins, r.get1 b =
      IntV.ofNat (Finset.univ.filter fun j : Fin n =>
        enabledBy mask j = true ∧ (x.get1 j).toInt = b).card)
    ⟨fun _ => if h : ∀ j, enabledBy mask j = true → 0 ≤ (x.get1 j).toInt ∧ (x.get1 j).toInt < bins then
      some (.ok (Tensor.of1 fun b => IntV.ofNat (Finset.univ.filter fun j : Fin n =>
        enabledBy mask j = true ∧ (x.get1 j).toInt = b).card) ⟨h, fun _ => rfl⟩)
      else some (.empty fun _ hr => h hr.1)⟩ .ret

end tl

end TVA
