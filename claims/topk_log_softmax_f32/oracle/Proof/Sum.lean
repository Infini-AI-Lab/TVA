import Proof.Lane

/-!
# The row sum as one tree

The kernel's second loop adds, in trip `t`, the `tl.sum` of block `t` (a tree over its `1024`
lanes) to the carried `se`. All these additions together form one tree: leaf `0` is the initial
`se = +0`, leaf `k + 1` is lane `k` of the row (`k = t·1024 + i`), and trip `t` adds the old tree
and trip `t`'s tree, relabelled. `SumInv t se` says `se` is the result of such a tree after `t`
trips; its depth is at most `t + 10`. `sumInv_zero` and `sumInv_step` carry it along the loop;
`pow_le` bounds the growth factor `(1 + u)^d` of the tree's error bound.
-/
namespace Claims.TopkLogSoftmax

open TVA

/-! ## Relabelling a tree -/

/-- The tree `t` with every leaf `i` renamed `g i`. -/
def tmap (g : Nat → Nat) : Tree → Tree
  | .leaf i => .leaf (g i)
  | .node l r => .node (tmap g l) (tmap g r)

/-- The leaves of a relabelled tree. -/
lemma tmap_leaves (g : Nat → Nat) : ∀ t : Tree, (tmap g t).leaves = t.leaves.map g
  | .leaf _ => rfl
  | .node l r => by
    simp only [tmap, Tree.leaves, List.map_append, tmap_leaves g l, tmap_leaves g r]

/-- Relabelling keeps the depth. -/
lemma tmap_depth (g : Nat → Nat) : ∀ t : Tree, (tmap g t).depth = t.depth
  | .leaf _ => rfl
  | .node l r => by simp only [tmap, Tree.depth, tmap_depth g l, tmap_depth g r]

/-- A tree evaluated at the renamed leaves is the relabelled tree. -/
lemma eval_tmap {α : Type} {op : α → α → α → Prop} {x : Nat → α} (g : Nat → Nat) {t : Tree}
    {r : α} (h : t.Eval op (fun i => x (g i)) r) : (tmap g t).Eval op x r := by
  induction h with
  | leaf i => exact Tree.Eval.leaf (g i)
  | node _ _ hc ihl ihr => exact Tree.Eval.node ihl ihr hc

/-- A tree's result depends only on its leaves' values. -/
lemma eval_congr {α : Type} {op : α → α → α → Prop} {x y : Nat → α} {t : Tree} {r : α}
    (h : t.Eval op x r) (hxy : ∀ i ∈ t.leaves, x i = y i) : t.Eval op y r := by
  induction h with
  | leaf i =>
    have : x i = y i := hxy i (by simp [Tree.leaves])
    rw [this]; exact Tree.Eval.leaf i
  | node _ _ hc ihl ihr =>
    exact Tree.Eval.node (ihl fun i hi => hxy i (by simp [Tree.leaves, hi]))
      (ihr fun i hi => hxy i (by simp [Tree.leaves, hi])) hc

/-! ## The leaves of the row sum -/

/-- The kernel's `0.0` is the `float32` `+0`. -/
lemma zeroC_eq : zeroC = (FloatV.ofPat 0 : F32) := by decide

/-- `+0` is finite. -/
lemma zeroC_finite : zeroC.isFinite = true := by
  rw [zeroC_eq]
  have hpat0 : (0 : Nat) < 2 ^ Format.binary32.width := by positivity
  rw [FloatV.isFinite_ofPat hpat0]
  exact Format.isFinite_zero b32_reg

/-- `+0` is `0`. -/
lemma zeroC_toRat : zeroC.toRat = 0 := by
  rw [zeroC_eq]
  have hpat0 : (0 : Nat) < 2 ^ Format.binary32.width := by positivity
  rw [FloatV.toRat_ofPat hpat0]
  exact Format.toRat_of_isZero Format.isZero_zero

/-- `+0` is `0`, as a real. -/
lemma zeroC_toReal : zeroC.toReal = 0 := by
  unfold FloatV.toReal; rw [zeroC_toRat]; simp

/-- Lane `k` of the row sum: below `vocab_size`, the platform's `exp` of the rounded
`xs k − mv`; past it, `+0` (`tl.where`). -/
def LaneE (P : Platform) (V : Nat) (xs : Nat → F32) (mv : F32) (k : Nat) (e : F32) : Prop :=
  if k < V then ∃ s : F32, FloatV.NaNEq (FloatV.sub (xs k) mv) s ∧ P.math .exp .binary32 s e
  else e = zeroC

/-- After `t` trips of the sum loop, `se` is the result of a tree of IEEE additions over leaf `0`
(`+0`) and leaves `k + 1`, `k < t·1024` (lane `k`), of depth at most `t + 10`. -/
def SumInv (P : Platform) (V : Nat) (xs : Nat → F32) (mv : F32) (t : Nat) (se : F32) : Prop :=
  ∃ (X : Nat → F32) (tr : Tree), tr.Covers (1 + t * 1024) ∧ tr.depth ≤ t + 10 ∧
    tr.Eval (AddRel false) X se ∧ X 0 = zeroC ∧
    ∀ k, k < t * 1024 → LaneE P V xs mv k (X (k + 1))

/-- Before the loop: `se = 0.0`, one leaf. -/
lemma sumInv_zero (P : Platform) (V : Nat) (xs : Nat → F32) (mv : F32) :
    SumInv P V xs mv 0 zeroC := by
  refine ⟨fun _ => zeroC, TVA.Tree.leaf 0, ?_, by simp [Tree.depth], Tree.Eval.leaf 0, rfl, ?_⟩
  · show (TVA.Tree.leaf 0).leaves.Perm (List.range (1 + 0 * 1024))
    simp [Tree.leaves]
  · intro k hk; omega

/-- One trip: `se' = se + b`, where `b` is the block's sum along a tree over its `1024` lanes `y`,
of depth at most `10`, and lane `i` is lane `t·1024 + i` of the row. -/
lemma sumInv_step {P : Platform} {V : Nat} {xs : Nat → F32} {mv : F32} {t : Nat} {se b se' : F32}
    (h : SumInv P V xs mv t se) {y : Nat → F32} {trT : Tree} (hcov : trT.Covers 1024)
    (hd : trT.depth ≤ 10) (hev : trT.Eval (AddRel false) y b)
    (hy : ∀ i, i < 1024 → LaneE P V xs mv (t * 1024 + i) (y i))
    (hadd : AddRel false se b se') :
    SumInv P V xs mv (t + 1) se' := by
  obtain ⟨X, tr, hcv, hdp, hevX, hX0, hlane⟩ := h
  set N := 1 + t * 1024 with hN
  let g : Nat → Nat := fun i => N + i
  let X' : Nat → F32 := fun j => if j < N then X j else y (j - N)
  have hleaf_old : ∀ i ∈ tr.leaves, i < N := fun i hi => List.mem_range.mp (hcv.subset hi)
  have hleaf_new : ∀ i ∈ trT.leaves, i < 1024 := fun i hi => List.mem_range.mp (hcov.subset hi)
  have hev_old : tr.Eval (AddRel false) X' se :=
    eval_congr hevX fun i hi => by
      show X i = (if i < N then X i else y (i - N))
      rw [ite_eq_left (hleaf_old i hi)]
  have hev_new : (tmap g trT).Eval (AddRel false) X' b := by
    refine eval_tmap g (eval_congr hev fun i _ => ?_)
    show y i = (if N + i < N then X (N + i) else y (N + i - N))
    rw [ite_eq_right (by omega)]
    congr 1; omega
  refine ⟨X', TVA.Tree.node tr (tmap g trT), ?_, ?_, Tree.Eval.node hev_old hev_new hadd, ?_, ?_⟩
  · show (tr.leaves ++ (tmap g trT).leaves).Perm (List.range (1 + (t + 1) * 1024))
    rw [tmap_leaves, show 1 + (t + 1) * 1024 = N + 1024 by rw [hN]; ring, List.range_add]
    exact hcv.append (hcov.map g)
  · simp only [Tree.depth, tmap_depth]
    omega
  · show (if 0 < N then X 0 else y (0 - N)) = zeroC
    rw [ite_eq_left (by omega)]; exact hX0
  · intro k hk
    show LaneE P V xs mv k (if k + 1 < N then X (k + 1) else y (k + 1 - N))
    by_cases hkN : k < t * 1024
    · rw [ite_eq_left (by omega)]; exact hlane k hkN
    · rw [ite_eq_right (by omega)]
      have hk' : k = t * 1024 + (k + 1 - N) := by omega
      have := hy (k + 1 - N) (by
        have : (t + 1) * 1024 = t * 1024 + 1024 := by ring
        omega)
      rwa [← hk'] at this

/-! ## The growth of the error bound -/

/-- `(1 + u)^n ≤ 1 + n·u + n²·u²` while `n·u ≤ 1`. -/
lemma pow_le_sq : ∀ n : Nat, (n : ℝ) * uR ≤ 1 →
    ((1 : ℝ) + uR) ^ n ≤ 1 + n * uR + (n : ℝ) ^ 2 * uR ^ 2
  | 0, _ => by simp
  | n + 1, h => by
    have hu := uR_pos
    have hn : (n : ℝ) * uR ≤ 1 := by
      have : (n : ℝ) ≤ (n + 1 : Nat) := by exact_mod_cast Nat.le_succ n
      nlinarith
    have ih := pow_le_sq n hn
    rw [pow_succ]
    have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
    have h1 : ((1 : ℝ) + uR) ^ n * (1 + uR) ≤ (1 + n * uR + (n : ℝ) ^ 2 * uR ^ 2) * (1 + uR) :=
      mul_le_mul_of_nonneg_right ih (by linarith)
    have h2 : (n : ℝ) ^ 2 * uR ^ 3 ≤ (n + 1) * uR ^ 2 := by
      have : (n : ℝ) ^ 2 * uR ^ 3 = ((n : ℝ) * uR) * (n * uR ^ 2) := by ring
      rw [this]
      have h3 : (n : ℝ) * uR ^ 2 ≤ (n + 1) * uR ^ 2 := by nlinarith
      have h4 : 0 ≤ (n : ℝ) * uR ^ 2 := by positivity
      nlinarith
    push_cast
    nlinarith

/-- For a depth `n ≤ 600`: `(1 + u)^n ≤ 1 + (n + 1/32)·u`. -/
lemma pow_le {n : Nat} (hn : n ≤ 600) : ((1 : ℝ) + uR) ^ n ≤ 1 + (n + 1 / 32) * uR := by
  have hu := uR_pos
  have hn' : (n : ℝ) ≤ 600 := by exact_mod_cast hn
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hnu : (n : ℝ) * uR ≤ 1 := by
    have : (600 : ℝ) * uR ≤ 1 := by unfold uR; norm_num
    nlinarith
  have h := pow_le_sq n hnu
  have h2 : (n : ℝ) ^ 2 * uR ^ 2 ≤ 1 / 32 * uR := by
    have : (600 : ℝ) ^ 2 * uR ≤ 1 / 32 := by unfold uR; norm_num
    have h5 : (n : ℝ) ^ 2 ≤ 600 ^ 2 := by nlinarith
    have h6 : (n : ℝ) ^ 2 * uR ^ 2 = ((n : ℝ) ^ 2 * uR) * uR := by ring
    rw [h6]
    have h7 : (n : ℝ) ^ 2 * uR ≤ 1 / 32 := by nlinarith
    nlinarith
  nlinarith

end Claims.TopkLogSoftmax
