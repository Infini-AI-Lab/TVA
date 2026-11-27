import Claim
import TVA.Proof

/-!
# Trees renumbered, and their sums

A block's lane `j` is position `j + base`: `Tree.mapLeaves` renumbers a lane tree (the scan's,
the `tl.sum`'s) into positions, keeping its depth, node count and evaluation (`eval_mapLeaves`).
The sums of a tree's leaves follow any list its leaves are a permutation of (`sumOf_perm`), and
list sums over `range`/`range'` are `Finset` sums over intervals.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

/-- A tree evaluates to the same result when the leaves it uses are unchanged. -/
theorem eval_congr {α : Type} {op : α → α → α → Prop} {x y : Nat → α} {t : Tree} {r : α}
    (h : t.Eval op x r) : (∀ j ∈ t.leaves, x j = y j) → t.Eval op y r := by
  induction h with
  | leaf i =>
      intro hy
      rw [hy i (by simp [Tree.leaves])]
      exact .leaf i
  | node hl hr hop ihl ihr =>
      intro hy
      exact .node
        (ihl (fun j hj => hy j (by
          simp only [Tree.leaves, List.mem_append]; exact Or.inl hj)))
        (ihr (fun j hj => hy j (by
          simp only [Tree.leaves, List.mem_append]; exact Or.inr hj)))
        hop

/-- A renumbered tree evaluates with the renumbered leaf values. -/
theorem eval_mapLeaves {α : Type} {op : α → α → α → Prop} {x : Nat → α} {g : Nat → Nat}
    {t : Tree} {r : α} (h : t.Eval op (fun j => x (g j)) r) : (t.mapLeaves g).Eval op x r := by
  induction h with
  | leaf i => exact .leaf (g i)
  | node _ _ hop ihl ihr => exact .node ihl ihr hop

@[simp] theorem depth_mapLeaves (g : Nat → Nat) : ∀ t : Tree, (t.mapLeaves g).depth = t.depth
  | .leaf _ => rfl
  | .node l r => by
    simp only [Tree.mapLeaves, Tree.depth, depth_mapLeaves g l, depth_mapLeaves g r]

@[simp] theorem nodes_mapLeaves (g : Nat → Nat) : ∀ t : Tree, (t.mapLeaves g).nodes = t.nodes
  | .leaf _ => rfl
  | .node l r => by
    simp only [Tree.mapLeaves, Tree.nodes, nodes_mapLeaves g l, nodes_mapLeaves g r]

/-- The sum of a tree's leaves, along a list its leaves are a permutation of. -/
theorem sumOf_perm {β : Type} (val : β → ℚ) (x : Nat → β) {t : Tree} {L : List Nat}
    (h : t.leaves.Perm L) : Tree.sumOf val x t = (L.map fun i => val (x i)).sum :=
  (h.map _).sum_eq

theorem absSumOf_perm {β : Type} (val : β → ℚ) (x : Nat → β) {t : Tree} {L : List Nat}
    (h : t.leaves.Perm L) : Tree.absSumOf val x t = (L.map fun i => |val (x i)|).sum :=
  (h.map _).sum_eq

/-- A list sum over `range k` is a `Finset` sum. -/
theorem list_range_sum (f : Nat → ℚ) (k : Nat) :
    ((List.range k).map f).sum = ∑ j ∈ Finset.range k, f j := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]
    simp

/-- A list sum over `range' a k` is a `Finset` sum over `[a, a + k)`. -/
theorem list_range'_sum (f : Nat → ℚ) (a k : Nat) :
    ((List.range' a k).map f).sum = ∑ j ∈ Finset.Ico a (a + k), f j := by
  rw [List.range'_eq_map_range, List.map_map, Finset.sum_Ico_eq_sum_range]
  rw [show a + k - a = k by omega]
  exact list_range_sum _ k

/-- A sum over `range k` of a shifted function is a sum over `[c, c + k)`. -/
theorem range_shift_sum (f : Nat → ℚ) (k c : Nat) :
    ∑ j ∈ Finset.range k, f (j + c) = ∑ q ∈ Finset.Ico c (c + k), f q := by
  rw [Finset.range_eq_Ico, Finset.sum_Ico_add' f 0 k c, Nat.zero_add, Nat.add_comm k c]

/-- A sum over `[a, b)` of a shifted function is a sum over `[a + c, b + c)`. -/
theorem Ico_shift_sum (f : Nat → ℚ) (a b c : Nat) :
    ∑ j ∈ Finset.Ico a b, f (j + c) = ∑ q ∈ Finset.Ico (a + c) (b + c), f q :=
  Finset.sum_Ico_add' f a b c

end Claims.ChunkGlobalCumsum
