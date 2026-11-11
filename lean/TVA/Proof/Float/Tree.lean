import TVA.Proof.Float.Widen

/-!
# The error of a reduction tree

Triton reduces along a tree it chooses (`tl.sum` and friends: `Tree.Eval`, `TVA.Tensor.Tree`).
Each node rounds, so the result depends on the tree. This file bounds it for every tree at once,
from a bound on one node.

If each node's result `c` is within `δ·(|a| + |b|) + η` of the exact sum of its operands `a`, `b`
(a relative error `δ` and an absolute error `η` per node), the whole tree is within

`errBound δ η d N A = ((1 + δ)^d − 1) · A + η · N · (1 + δ)^d`

of the exact sum of its leaves, where `d` is the tree's depth, `N` its number of nodes and `A`
the sum of the leaves' absolute values (`eval_err_abs`). An IEEE addition has `δ = u`, `η` of the
format, and the stronger per-node bound `δ·|a + b| + η` (`eval_err`); a fusable addition has
`δ = 3u`, `η = 4η` (`TVA.Proof.Float.Fused`). The per-node bound may need room (`|a| + |b| ≤ M`,
so that no node overflows); the theorem checks it against the root's bound, and carries a
predicate `good` (such as "finite") from the leaves to the result.

The depth is `n − 1` for a sequential sum, `⌈log₂ n⌉` for a balanced one; a bound for every tree
over `n` leaves uses `depth ≤ nodes = n − 1` (`depth_le_nodes`, `nodes_of_covers`), or the
platform's depth bound (`tl.sumBound`). `errBound_mono` compares bounds.
-/
namespace TVA.Tree

/-- The number of combining nodes (one less than the number of leaves). -/
def nodes : Tree → Nat
  | leaf _ => 0
  | node l r => l.nodes + r.nodes + 1

/-- A tree is no deeper than its number of nodes: the worst case, a sequential sum. -/
theorem depth_le_nodes : ∀ t : Tree, t.depth ≤ t.nodes
  | leaf _ => le_rfl
  | node l r => by
    have := depth_le_nodes l; have := depth_le_nodes r
    simp only [depth, nodes]; omega

/-- A tree has one node fewer than leaves. -/
theorem nodes_add_one : ∀ t : Tree, t.nodes + 1 = t.leaves.length
  | leaf _ => rfl
  | node l r => by
    have := nodes_add_one l; have := nodes_add_one r
    simp only [nodes, leaves, List.length_append]; omega

/-- A tree covering `0, …, n−1` has `n − 1` nodes. -/
theorem nodes_of_covers {t : Tree} {n : Nat} (h : t.Covers n) : t.nodes + 1 = n := by
  rw [nodes_add_one, h.length_eq, List.length_range]

section
variable {β : Type} (val : β → ℚ) (x : Nat → β)

/-- The exact sum of the leaves' values. -/
def sumOf (t : Tree) : ℚ := (t.leaves.map fun i => val (x i)).sum

/-- The sum of the leaves' absolute values. -/
def absSumOf (t : Tree) : ℚ := (t.leaves.map fun i => |val (x i)|).sum

/-- The exact sum of a leaf is its value. -/
@[simp] theorem sumOf_leaf (i : Nat) : sumOf val x (leaf i) = val (x i) := by simp [sumOf, leaves]
/-- The exact sum of a node is the sum of its subtrees' sums. -/
@[simp] theorem sumOf_node (l r : Tree) : sumOf val x (node l r) = sumOf val x l + sumOf val x r := by
  simp [sumOf, leaves]
/-- The absolute sum of a leaf is its absolute value. -/
@[simp] theorem absSumOf_leaf (i : Nat) : absSumOf val x (leaf i) = |val (x i)| := by
  simp [absSumOf, leaves]
/-- The absolute sum of a node is the sum of its subtrees' absolute sums. -/
@[simp] theorem absSumOf_node (l r : Tree) :
    absSumOf val x (node l r) = absSumOf val x l + absSumOf val x r := by
  simp [absSumOf, leaves]

/-- The absolute sum is nonnegative. -/
theorem absSumOf_nonneg (t : Tree) : 0 ≤ absSumOf val x t :=
  List.sum_nonneg (by simp only [List.mem_map]; rintro _ ⟨i, -, rfl⟩; exact abs_nonneg _)

/-- The exact sum is at most the absolute sum in magnitude. -/
theorem abs_sumOf_le (t : Tree) : |sumOf val x t| ≤ absSumOf val x t := by
  induction t with
  | leaf i => simp
  | node l r ihl ihr => simp only [sumOf_node, absSumOf_node]; exact (abs_add_le _ _).trans (add_le_add ihl ihr)

/-- If `t` covers `0, …, n−1`, its leaves sum to the whole sum. -/
theorem sumOf_covers {t : Tree} {n : Nat} (h : t.Covers n) :
    sumOf val x t = ((List.range n).map fun i => val (x i)).sum := by
  unfold sumOf; exact (h.map _).sum_eq

/-- If `t` covers `0, …, n−1`, its absolute sum is the whole absolute sum. -/
theorem absSumOf_covers {t : Tree} {n : Nat} (h : t.Covers n) :
    absSumOf val x t = ((List.range n).map fun i => |val (x i)|).sum := by
  unfold absSumOf; exact (h.map _).sum_eq

end

/-- The error bound of a tree of depth `d` with `N` nodes over leaves of absolute sum `A`. -/
def errBound (δ η : ℚ) (d N : Nat) (A : ℚ) : ℚ := ((1 + δ) ^ d - 1) * A + η * N * (1 + δ) ^ d

section Bound
variable {δ η : ℚ}

/-- `(1 + δ)^d ≥ 1` for `δ ≥ 0`. -/
theorem one_le_pow (hδ : 0 ≤ δ) (d : Nat) : 1 ≤ (1 + δ) ^ d := one_le_pow₀ (by linarith)

/-- `(1 + δ)^d` grows with `d` for `δ ≥ 0`. -/
theorem pow_mono (hδ : 0 ≤ δ) {d d' : Nat} (h : d ≤ d') : (1 + δ) ^ d ≤ (1 + δ) ^ d' :=
  pow_le_pow_right₀ (by linarith) h

/-- The bound is nonnegative. -/
theorem errBound_nonneg (hδ : 0 ≤ δ) (hη : 0 ≤ η) (d N : Nat) {A : ℚ} (hA : 0 ≤ A) :
    0 ≤ errBound δ η d N A := by
  unfold errBound
  have := one_le_pow hδ d
  have : 0 ≤ ((1 + δ) ^ d - 1) * A := mul_nonneg (by linarith) hA
  positivity

/-- **The bound grows with the depth, the node count and the absolute sum**: use it to replace a tree's own depth and size by bounds that hold for every tree. -/
theorem errBound_mono (hδ : 0 ≤ δ) (hη : 0 ≤ η) {d d' N N' : Nat} {A A' : ℚ} (hd : d ≤ d')
    (hN : N ≤ N') (hA : 0 ≤ A) (hAA : A ≤ A') : errBound δ η d N A ≤ errBound δ η d' N' A' := by
  unfold errBound
  have h1 := one_le_pow hδ d
  have h2 := pow_mono hδ hd
  have hN' : (N : ℚ) ≤ N' := by exact_mod_cast hN
  have e1 : ((1 + δ) ^ d - 1) * A ≤ ((1 + δ) ^ d' - 1) * A' :=
    mul_le_mul (by linarith) hAA hA (by linarith)
  have e2 : η * N * (1 + δ) ^ d ≤ η * N' * (1 + δ) ^ d' :=
    mul_le_mul (mul_le_mul_of_nonneg_left hN' hη) h2 (by positivity) (by positivity)
  linarith

/-- One level up: a child's bound, grown by one rounding, fits the parent's. -/
theorem errBound_step (hδ : 0 ≤ δ) (hη : 0 ≤ η) {dl d Nl : Nat} {Al : ℚ} (hd : dl + 1 ≤ d)
    (hA : 0 ≤ Al) :
    (1 + δ) * errBound δ η dl Nl Al ≤ ((1 + δ) ^ d - (1 + δ)) * Al + η * Nl * (1 + δ) ^ d := by
  unfold errBound
  have h := pow_mono hδ hd
  rw [pow_succ] at h
  have e1 : ((1 + δ) ^ dl * (1 + δ) - (1 + δ)) * Al ≤ ((1 + δ) ^ d - (1 + δ)) * Al :=
    mul_le_mul_of_nonneg_right (by linarith) hA
  have e2 : η * Nl * ((1 + δ) ^ dl * (1 + δ)) ≤ η * Nl * (1 + δ) ^ d :=
    mul_le_mul_of_nonneg_left h (by positivity)
  nlinarith

end Bound

/-- **The error of a reduction tree whose nodes are bounded relative to their operands'
magnitudes.** If each node's result is within `δ·(|a| + |b|) + η` of `a + b` whenever
`|a| + |b| ≤ M`, the tree is within `errBound δ η depth nodes A` of the exact sum of its leaves
(the bound of `eval_err`). -/
theorem eval_err_abs {β : Type} (val : β → ℚ) (x : Nat → β) {op : β → β → β → Prop}
    (good : β → Prop) {δ η M : ℚ} (hδ : 0 ≤ δ) (hη : 0 ≤ η)
    (hop : ∀ a b c, op a b c → good a → good b → |val a| + |val b| ≤ M →
      good c ∧ |val c - (val a + val b)| ≤ δ * (|val a| + |val b|) + η)
    {t : Tree} {r : β} (h : t.Eval op x r) (hx : ∀ i ∈ t.leaves, good (x i))
    (hM : absSumOf val x t + errBound δ η t.depth t.nodes (absSumOf val x t) ≤ M) :
    good r ∧ |val r - sumOf val x t| ≤ errBound δ η t.depth t.nodes (absSumOf val x t) := by
  induction h with
  | leaf i =>
    refine ⟨hx i (by simp [leaves]), ?_⟩
    simp [errBound, depth, nodes]
  | @node l rt a b c hl hr hc ihl ihr =>
    have hAl := absSumOf_nonneg val x l
    have hAr := absSumOf_nonneg val x rt
    have hdl : l.depth + 1 ≤ (node l rt).depth := by simp only [depth]; omega
    have hdr : rt.depth + 1 ≤ (node l rt).depth := by simp only [depth]; omega
    have hNe : ((node l rt).nodes : ℚ) = l.nodes + rt.nodes + 1 := by
      simp only [nodes]; push_cast; ring
    rw [absSumOf_node] at hM ⊢
    rw [sumOf_node]
    have mono_l : absSumOf val x l + errBound δ η l.depth l.nodes (absSumOf val x l) ≤ M :=
      le_trans (add_le_add (le_add_of_nonneg_right hAr) (errBound_mono hδ hη (by omega)
        (by simp only [nodes]; omega) hAl (le_add_of_nonneg_right hAr))) hM
    have mono_r : absSumOf val x rt + errBound δ η rt.depth rt.nodes (absSumOf val x rt) ≤ M :=
      le_trans (add_le_add (le_add_of_nonneg_left hAl) (errBound_mono hδ hη (by omega)
        (by simp only [nodes]; omega) hAr (le_add_of_nonneg_left hAl))) hM
    obtain ⟨ga, ea⟩ := ihl (fun i hi => hx i (by simp [leaves, hi])) mono_l
    obtain ⟨gb, eb⟩ := ihr (fun i hi => hx i (by simp [leaves, hi])) mono_r
    have hstep_l := errBound_step hδ hη (Nl := l.nodes) hdl hAl
    have hstep_r := errBound_step hδ hη (Nl := rt.nodes) hdr hAr
    have hEl := errBound_mono hδ hη (d' := (node l rt).depth) (N' := l.nodes) (by omega) le_rfl hAl
      (le_refl (absSumOf val x l))
    have hEr := errBound_mono hδ hη (d' := (node l rt).depth) (N' := rt.nodes) (by omega) le_rfl hAr
      (le_refl (absSumOf val x rt))
    have hSl := abs_sumOf_le val x l
    have hSr := abs_sumOf_le val x rt
    have h1 := one_le_pow hδ (node l rt).depth
    -- `|a| + |b| ≤ |Sl| + |Sr| + |a − Sl| + |b − Sr|`
    have hab : |val a| + |val b| ≤ |sumOf val x l| + |sumOf val x rt| + |val a - sumOf val x l| +
        |val b - sumOf val x rt| := by
      have ha' : |val a| ≤ |sumOf val x l| + |val a - sumOf val x l| := by
        have := abs_add_le (sumOf val x l) (val a - sumOf val x l)
        rwa [add_sub_cancel] at this
      have hb' : |val b| ≤ |sumOf val x rt| + |val b - sumOf val x rt| := by
        have := abs_add_le (sumOf val x rt) (val b - sumOf val x rt)
        rwa [add_sub_cancel] at this
      linarith
    have key : |val c - (sumOf val x l + sumOf val x rt)| ≤ |val c - (val a + val b)| +
        |val a - sumOf val x l| + |val b - sumOf val x rt| := by
      have h := abs_add_le (val c - (val a + val b)) ((val a - sumOf val x l) + (val b - sumOf val x rt))
      have h' := abs_add_le (val a - sumOf val x l) (val b - sumOf val x rt)
      have e : val c - (val a + val b) + ((val a - sumOf val x l) + (val b - sumOf val x rt)) =
          val c - (sumOf val x l + sumOf val x rt) := by ring
      rw [e] at h; linarith
    unfold errBound at hM hEl hEr ⊢
    rw [hNe] at hM ⊢
    generalize (1 + δ) ^ (node l rt).depth = G at h1 hstep_l hstep_r hEl hEr hM ⊢
    generalize errBound δ η l.depth l.nodes (absSumOf val x l) = El at ea mono_l hstep_l hEl
    generalize errBound δ η rt.depth rt.nodes (absSumOf val x rt) = Er at eb mono_r hstep_r hEr
    generalize absSumOf val x l = Al at *
    generalize absSumOf val x rt = Ar at *
    generalize sumOf val x l = Sl at *
    generalize sumOf val x rt = Sr at *
    have hNl : (0 : ℚ) ≤ l.nodes := Nat.cast_nonneg _
    have hNr : (0 : ℚ) ≤ rt.nodes := Nat.cast_nonneg _
    generalize (l.nodes : ℚ) = Nl at *
    generalize (rt.nodes : ℚ) = Nr at *
    have hW : |val a| + |val b| ≤ M := by nlinarith
    obtain ⟨gc, ec⟩ := hop a b c hc ga gb hW
    refine ⟨gc, ?_⟩
    have hδW : δ * (|val a| + |val b|) ≤ δ * (Al + Ar + El + Er) :=
      mul_le_mul_of_nonneg_left (by linarith) hδ
    have hηG : η ≤ η * G := le_mul_of_one_le_right hη h1
    nlinarith

/-- **The error of a reduction tree whose nodes are bounded relative to their exact sum**: per
node `δ·|a + b| + η` whenever `|a + b| ≤ M` (an IEEE addition) gives the bound of `eval_err_abs`. -/
theorem eval_err {β : Type} (val : β → ℚ) (x : Nat → β) {op : β → β → β → Prop}
    (good : β → Prop) {δ η M : ℚ} (hδ : 0 ≤ δ) (hη : 0 ≤ η)
    (hop : ∀ a b c, op a b c → good a → good b → |val a + val b| ≤ M →
      good c ∧ |val c - (val a + val b)| ≤ δ * |val a + val b| + η)
    {t : Tree} {r : β} (h : t.Eval op x r) (hx : ∀ i ∈ t.leaves, good (x i))
    (hM : absSumOf val x t + errBound δ η t.depth t.nodes (absSumOf val x t) ≤ M) :
    good r ∧ |val r - sumOf val x t| ≤ errBound δ η t.depth t.nodes (absSumOf val x t) :=
  eval_err_abs val x good hδ hη (fun a b c hc ga gb hab => by
    have h1 := abs_add_le (val a) (val b)
    obtain ⟨g, e⟩ := hop a b c hc ga gb (h1.trans hab)
    exact ⟨g, e.trans (by nlinarith [mul_le_mul_of_nonneg_left h1 hδ])⟩) h hx hM

end TVA.Tree
