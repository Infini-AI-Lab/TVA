import TVA.Tensor.Basic

/-!
# Reduction trees

Triton reduces in an order of its own choosing: within a thread, then across the lanes of a warp,
then across warps. Integer reductions do not depend on that order (`Tensor.foldAll`), but
floating-point ones do, since each combination rounds. A float reduction over `n` positions is
therefore described by a binary tree whose leaves are the positions `0, …, n-1`, each once, in
some order (`Tree.Covers`), evaluated with a *relation* for each combination (`Tree.Eval`): a
rounded add, possibly fused, may itself have several results.

How deep the device's trees may be is a platform fact (`Platform.sumDepth`). A scan keeps the
order of its leaves; its trees are constrained in `TVA.Triton.Reduce`.

Used by the float reductions and scans of `TVA.Triton.Reduce`, and by the error bounds of the
proof kit.
-/
namespace TVA

/-- A binary combination tree with leaves numbered by `Nat` (the positions combined). -/
inductive Tree where
  /-- The element at position `i`. -/
  | leaf (i : Nat)
  /-- The combination of two subtrees. -/
  | node (l r : Tree)
  deriving DecidableEq, Repr

namespace Tree

/-- The leaves, left to right. -/
def leaves : Tree → List Nat
  | leaf i => [i]
  | node l r => l.leaves ++ r.leaves

/-- `t` combines each of the positions `0, …, n-1` exactly once, in some order. -/
def Covers (n : Nat) (t : Tree) : Prop := t.leaves.Perm (List.range n)

/-- `Eval op x t r`: the tree `t` can produce `r` when leaf `i` holds `x i` and each node
combines its two operands `a`, `b` into some `c` with `op a b c`. -/
inductive Eval {α : Type} (op : α → α → α → Prop) (x : Nat → α) : Tree → α → Prop where
  /-- A leaf produces its element. -/
  | leaf (i : Nat) : Eval op x (leaf i) (x i)
  /-- A node produces any combination its operands' results allow. -/
  | node {l r : Tree} {a b c : α} :
      Eval op x l a → Eval op x r b → op a b c → Eval op x (node l r) c

/-- The depth: a leaf has depth `0`, a node one more than its deeper subtree. -/
def depth : Tree → Nat
  | leaf _ => 0
  | node l r => max l.depth r.depth + 1

/-- The left-to-right sequential tree `((x₀ + x₁) + x₂) + …` over `0, …, n-1` (for `n ≥ 1`). -/
def seq : Nat → Tree
  | 0 => leaf 0
  | 1 => leaf 0
  | n + 2 => node (seq (n + 1)) (leaf (n + 1))

/-- The sequential tree's leaves are `0, …, n-1` in order. -/
theorem seq_leaves : ∀ n, 0 < n → (seq n).leaves = List.range n
  | 0, h => absurd h (Nat.lt_irrefl 0)
  | 1, _ => rfl
  | n + 2, _ => by
    simp only [seq, leaves]
    rw [seq_leaves (n + 1) (by omega)]
    exact (List.range_succ (n := n + 1)).symm

/-- The sequential tree covers `n ≥ 1` positions: use it to show some tree exists. -/
theorem seq_covers {n : Nat} (hn : 0 < n) : (seq n).Covers n := by
  unfold Covers; rw [seq_leaves n hn]

end Tree

end TVA
