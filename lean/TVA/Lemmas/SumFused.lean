import TVA.Lemmas.Sum
import TVA.Lemmas.FusedErr
import TVA.Lemmas.Scan
import TVA.Rules.Num.Sm90
import TVA.Lemmas.DotLoop

/-!
# Error of reductions with fusable additions (library issue L33)

`tl.sumc` (`sum1c`, `sum0c`), `tl.cumsumc` and the tree form of `tl.sum_bcast_mul` add along a
tree whose nodes are `AddRel fusion`: with `enable_fp_fusion` an operand that is a rounded product
may enter the addition unrounded. Such a node is not within `δ·|a + b| + η` of `a + b` (with
`a = 1`, `b = -1` the fused result may be about `2^-24` while `a + b = 0`), so `Tree.eval_err`
does not apply. It is within `3u·(|a| + |b|) + 4η` (`FloatV.AddRel.err`), a bound relative to the
operands' magnitudes, and a tree of such nodes keeps the same shape of bound:

* `Tree.eval_err_abs`: per-node bound `δ·(|a| + |b|) + η` (for `|a| + |b| ≤ M`) gives the whole
  tree within `errBound δ η depth nodes A` of the exact sum, `A` the leaves' absolute sum, as in
  `Tree.eval_err`;
* `sumc_err`, `sumc_err_covers`, `sumc_err_list`: the float reductions with `AddRel fusion`
  nodes (any `fusion`, so also the IEEE case), with per-node constants `3u` (relative) and
  `4η` (absolute), and room `A + errBound ≤ maxVal/2 − 2η`;
* `sum_bcast_mul_tree_err`: the tree form of `tl.sum_bcast_mul` on `float32` (rounded products
  summed by fusable additions), against the exact `Σ a·b`; `sum_bcast_mul_err` adds the other
  form, the `ieee` `tl.dot`, with `Rules.Num.DotF32Ieee`;
* `sum_bcast_mul_depth_err`: the same composition along a tree of depth at most `D`
  (`bcastDepthBound`; `D = Nat.clog 2 K` with `Rules.Num.SumDepth`; library issue L159).
-/
namespace TVA

open TVA.Math.Float TVA.Math.Float.Format

namespace Tree

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

end Tree

variable {f : Format}

/-- One fusable addition: finite operands with `|a| + |b| ≤ maxVal/2 − 2η` give a finite result
within `3u·(|a| + |b|) + 4η` of `a + b` (`FloatV.AddRel.err`). -/
theorem addRel_node (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16) (fusion : Bool) :
    ∀ a b c : FloatV f, AddRel fusion a b c → a.isFinite = true → b.isFinite = true →
      |a.toRat| + |b.toRat| ≤ f.maxVal / 2 - 2 * f.eta →
      c.isFinite = true ∧ |c.toRat - (a.toRat + b.toRat)| ≤ 3 * f.u * (|a.toRat| + |b.toRat|) + 4 * f.eta :=
  fun a b c h ha hb hM => FloatV.AddRel.err hf hs hu h ha hb (by linarith)

/-- **A float reduction with fusable additions, along a given tree** (`tl.sumc`, …). -/
theorem sumc_err (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16) (fusion : Bool)
    (x : Nat → FloatV f) {t : Tree} {r : FloatV f}
    (h : t.Eval (AddRel fusion) x r) (hx : ∀ i ∈ t.leaves, (x i).isFinite = true)
    (hM : Tree.absSumOf FloatV.toRat x t +
      Tree.errBound (3 * f.u) (4 * f.eta) t.depth t.nodes (Tree.absSumOf FloatV.toRat x t) ≤
        f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - Tree.sumOf FloatV.toRat x t| ≤
        Tree.errBound (3 * f.u) (4 * f.eta) t.depth t.nodes (Tree.absSumOf FloatV.toRat x t) :=
  Tree.eval_err_abs FloatV.toRat x (fun a => a.isFinite = true)
    (mul_nonneg (by norm_num) u_nonneg) (mul_nonneg (by norm_num) eta_nonneg)
    (addRel_node hf hs hu fusion) h hx hM

/-- **A float reduction with fusable additions, along any tree over `0, …, n−1`**: the bound
with depth and node count `n − 1` and per-node constants `3u` and `4η`. This is what a kernel that sums products
(`tl.sum(x * y)` as `tl.sumc`) uses, for every tree Triton may choose. -/
theorem sumc_err_covers (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16) (fusion : Bool)
    (x : Nat → FloatV f) {n : Nat} {t : Tree} {r : FloatV f}
    (hcov : t.Covers n) (h : t.Eval (AddRel fusion) x r) (hx : ∀ i, i < n → (x i).isFinite = true)
    (hM : ((List.range n).map fun i => |(x i).toRat|).sum +
      Tree.errBound (3 * f.u) (4 * f.eta) (n - 1) (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum ≤
        f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - ((List.range n).map fun i => (x i).toRat).sum| ≤
        Tree.errBound (3 * f.u) (4 * f.eta) (n - 1) (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum := by
  have hδ : (0 : ℚ) ≤ 3 * f.u := mul_nonneg (by norm_num) u_nonneg
  have hη : (0 : ℚ) ≤ 4 * f.eta := mul_nonneg (by norm_num) eta_nonneg
  have hN := Tree.nodes_of_covers hcov
  have hD : t.depth ≤ n - 1 := by have := Tree.depth_le_nodes t; omega
  have hmono := Tree.errBound_mono hδ hη hD (show t.nodes ≤ n - 1 by omega)
    (Tree.absSumOf_nonneg FloatV.toRat x t) (le_refl (Tree.absSumOf FloatV.toRat x t))
  rw [Tree.absSumOf_covers FloatV.toRat x hcov] at hmono
  have hx' : ∀ i ∈ t.leaves, (x i).isFinite = true := fun i hi =>
    hx i (List.mem_range.mp (hcov.subset hi))
  obtain ⟨hfin, herr⟩ := sumc_err hf hs hu fusion x h hx'
    (by rw [Tree.absSumOf_covers FloatV.toRat x hcov]; linarith)
  rw [Tree.sumOf_covers FloatV.toRat x hcov, Tree.absSumOf_covers FloatV.toRat x hcov] at herr
  exact ⟨hfin, herr.trans hmono⟩

/-- **A float reduction with fusable additions along a tree over `0, …, n−1` of depth at most
`D`**: as `sumc_err_covers`, with depth `D` (library issue L149: with `Rules.Num.SumDepth`,
`D = ⌈log₂ n⌉`). -/
theorem sumc_err_depth (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16) (fusion : Bool)
    (x : Nat → FloatV f) {n D : Nat} {t : Tree} {r : FloatV f}
    (hcov : t.Covers n) (hd : t.depth ≤ D) (h : t.Eval (AddRel fusion) x r)
    (hx : ∀ i, i < n → (x i).isFinite = true)
    (hM : ((List.range n).map fun i => |(x i).toRat|).sum +
      Tree.errBound (3 * f.u) (4 * f.eta) D (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum ≤
        f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - ((List.range n).map fun i => (x i).toRat).sum| ≤
        Tree.errBound (3 * f.u) (4 * f.eta) D (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum := by
  have hδ : (0 : ℚ) ≤ 3 * f.u := mul_nonneg (by norm_num) u_nonneg
  have hη : (0 : ℚ) ≤ 4 * f.eta := mul_nonneg (by norm_num) eta_nonneg
  have hN := Tree.nodes_of_covers hcov
  have hmono := Tree.errBound_mono hδ hη hd (show t.nodes ≤ n - 1 by omega)
    (Tree.absSumOf_nonneg FloatV.toRat x t) (le_refl (Tree.absSumOf FloatV.toRat x t))
  rw [Tree.absSumOf_covers FloatV.toRat x hcov] at hmono
  have hx' : ∀ i ∈ t.leaves, (x i).isFinite = true := fun i hi =>
    hx i (List.mem_range.mp (hcov.subset hi))
  obtain ⟨hfin, herr⟩ := sumc_err hf hs hu fusion x h hx'
    (by rw [Tree.absSumOf_covers FloatV.toRat x hcov]; linarith)
  rw [Tree.sumOf_covers FloatV.toRat x hcov, Tree.absSumOf_covers FloatV.toRat x hcov] at herr
  exact ⟨hfin, herr.trans hmono⟩

/-- **A float sum with fusable additions along a tree whose leaves are the list `L` up to
order** (`L` without repeats): a position of `tl.cumsumc` (`ScanLeaves.perm`), a reduction, a
segment. The bound uses depth and node count `|L| − 1`. -/
theorem sumc_err_list (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16) (fusion : Bool)
    (x : Nat → FloatV f) {t : Tree} {L : List Nat} {r : FloatV f}
    (hL : t.leaves.Perm L) (h : t.Eval (AddRel fusion) x r) (hx : ∀ i ∈ L, (x i).isFinite = true)
    (hM : (L.map fun i => |(x i).toRat|).sum +
      Tree.errBound (3 * f.u) (4 * f.eta) (L.length - 1) (L.length - 1) (L.map fun i => |(x i).toRat|).sum ≤
        f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - (L.map fun i => (x i).toRat).sum| ≤
        Tree.errBound (3 * f.u) (4 * f.eta) (L.length - 1) (L.length - 1) (L.map fun i => |(x i).toRat|).sum := by
  have hδ : (0 : ℚ) ≤ 3 * f.u := mul_nonneg (by norm_num) u_nonneg
  have hη : (0 : ℚ) ≤ 4 * f.eta := mul_nonneg (by norm_num) eta_nonneg
  have hN : t.nodes + 1 = L.length := by rw [Tree.nodes_add_one, hL.length_eq]
  have hD : t.depth ≤ L.length - 1 := by have := Tree.depth_le_nodes t; omega
  have hsum : Tree.sumOf FloatV.toRat x t = (L.map fun i => (x i).toRat).sum := (hL.map _).sum_eq
  have habs : Tree.absSumOf FloatV.toRat x t = (L.map fun i => |(x i).toRat|).sum := (hL.map _).sum_eq
  have hmono := Tree.errBound_mono hδ hη hD (show t.nodes ≤ L.length - 1 by omega)
    (Tree.absSumOf_nonneg FloatV.toRat x t) (le_refl (Tree.absSumOf FloatV.toRat x t))
  rw [habs] at hmono
  obtain ⟨hfin, herr⟩ := sumc_err hf hs hu fusion x h (fun i hi => hx i (hL.subset hi))
    (by rw [habs]; linarith)
  rw [hsum, habs] at herr
  exact ⟨hfin, herr.trans hmono⟩

/-! ## `tl.sum_bcast_mul` on `float32` -/

section Bcast

private theorem u32 : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u

private theorem u32_le : Format.binary32.u ≤ 1 / 16 := by rw [u32]; norm_num

/-- A sum over `List.range n` is the sum over `Fin n`. -/
theorem sum_range_fin (g : ℕ → ℚ) : ∀ n, ((List.range n).map g).sum = ∑ k : Fin n, g k
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_range_fin g n, Fin.sum_univ_castSucc]
    simp

/-- The bound of the tree form of `tl.sum_bcast_mul` on `float32` with `K` products of absolute
sum `B`: the products' roundings (`u·B + K·η`), and the fusable tree over products of absolute sum
at most `(1 + u)·B + K·η`. -/
def bcastTreeBound (K : Nat) (B : ℚ) : ℚ :=
  Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (K - 1) (K - 1)
      ((1 + Format.binary32.u) * B + K * Format.binary32.eta) +
    (Format.binary32.u * B + K * Format.binary32.eta)

/-- **The tree form of `tl.sum_bcast_mul` on `float32`**, for one output element: products
`p k` of `a k · b k` (IEEE, `tl.MulRel`) summed along any tree over `0, …, K−1` with fusable
additions (`AccAdd`), are within `bcastTreeBound K B` of `Σ a k · b k`, `B = Σ |a k · b k|`, when
the operands are finite and there is room. -/
theorem sum_bcast_mul_tree_err {fusion : Bool} {K : Nat} (a b p : Nat → F32) {t : Tree} {r : F32}
    (hp : ∀ k, k < K → tl.MulRel .f32 (a k) (b k) (p k)) (hcov : t.Covers K)
    (ht : t.Eval (AccAdd fusion .f32) p r)
    (ha : ∀ k, k < K → (a k).isFinite = true) (hb : ∀ k, k < K → (b k).isFinite = true)
    (hab : ∀ k, k < K → |(a k).toRat * (b k).toRat| ≤ Format.binary32.maxVal)
    (hM : ((1 + Format.binary32.u) * ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum +
        K * Format.binary32.eta) +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (K - 1) (K - 1)
        ((1 + Format.binary32.u) * ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum +
          K * Format.binary32.eta) ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta) :
    r.isFinite = true ∧
      |r.toRat - ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum| ≤
        bcastTreeBound K ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum := by
  have hu0 := u_nonneg (f := Format.binary32)
  have he0 := eta_nonneg (f := Format.binary32)
  -- each product: finite and within `u·|ab| + η`
  have hpe : ∀ k, k < K → (p k).isFinite = true ∧
      |(p k).toRat - (a k).toRat * (b k).toRat| ≤ |(a k).toRat * (b k).toRat| * Format.binary32.u +
        Format.binary32.eta := by
    intro k hk
    obtain ⟨hfin, herr⟩ := FloatV.mul_err Format.binary32_regular (ha k hk) (hb k hk) (hab k hk)
    have hne : FloatV.NaNEq (FloatV.mul (a k) (b k)) (p k) := hp k hk
    rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
    exact ⟨hfin, herr⟩
  set B := ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum with hB
  have hP : ((List.range K).map fun k => |(p k).toRat|).sum ≤ (1 + Format.binary32.u) * B +
      K * Format.binary32.eta := by
    have : ∀ k ∈ List.range K, |(p k).toRat| ≤
        (1 + Format.binary32.u) * |(a k).toRat * (b k).toRat| + Format.binary32.eta := by
      intro k hk
      have hk' := List.mem_range.mp hk
      have h1 := (hpe k hk').2
      have h2 := abs_sub_abs_le_abs_sub (p k).toRat ((a k).toRat * (b k).toRat)
      nlinarith
    have hsum := List.sum_le_sum this
    rw [List.sum_map_add, List.sum_map_mul_left] at hsum
    simpa [hB, mul_comm] using hsum
  have hAP := List.sum_nonneg (l := (List.range K).map fun k => |(p k).toRat|)
    (by simp only [List.mem_map]; rintro _ ⟨k, -, rfl⟩; exact abs_nonneg _)
  have hδ : (0 : ℚ) ≤ 3 * Format.binary32.u := by positivity
  have hη : (0 : ℚ) ≤ 4 * Format.binary32.eta := by positivity
  have hmono := Tree.errBound_mono (d := K - 1) (d' := K - 1) (N := K - 1) (N' := K - 1) hδ hη le_rfl le_rfl hAP hP
  -- the tree over the products
  have ht' : t.Eval (AddRel fusion) p r := ht
  obtain ⟨hfin, herr⟩ := sumc_err_covers Format.binary32_regular (by decide) u32_le fusion p hcov ht'
    (fun k hk => (hpe k hk).1) (by linarith)
  refine ⟨hfin, ?_⟩
  -- the products' errors
  have hsumP : |((List.range K).map fun k => (p k).toRat).sum -
      ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum| ≤
      Format.binary32.u * B + K * Format.binary32.eta := by
    rw [sum_range_fin, sum_range_fin, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    have hle : ∀ k ∈ (Finset.univ : Finset (Fin K)), |(p k).toRat - (a k).toRat * (b k).toRat| ≤
        Format.binary32.u * |(a k).toRat * (b k).toRat| + Format.binary32.eta := fun k _ => by
      have := (hpe k k.isLt).2
      linarith [mul_comm (|(a k).toRat * (b k).toRat|) Format.binary32.u]
    refine (Finset.sum_le_sum hle).trans (le_of_eq ?_)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, hB, sum_range_fin]
  unfold bcastTreeBound
  have e : r.toRat - ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum =
      (r.toRat - ((List.range K).map fun k => (p k).toRat).sum) +
        (((List.range K).map fun k => (p k).toRat).sum -
          ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum) := by ring
  rw [e]
  exact (abs_add_le _ _).trans (add_le_add (herr.trans hmono) hsumP)

/-- The bound of the products `a k · b k` (`K` of them, absolute sum `B`), each rounded once in
`float32`, summed along a tree of fusable additions of depth at most `D`: `bcastTreeBound` with
depth `D` in place of `K − 1` (library issue L159). -/
def bcastDepthBound (D K : Nat) (B : ℚ) : ℚ :=
  Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) D (K - 1)
      ((1 + Format.binary32.u) * B + K * Format.binary32.eta) +
    (Format.binary32.u * B + K * Format.binary32.eta)

/-- `bcastTreeBound` is `bcastDepthBound` at depth `K − 1`. -/
theorem bcastTreeBound_eq (K : Nat) (B : ℚ) : bcastTreeBound K B = bcastDepthBound (K - 1) K B := rfl

/-- **Rounded `float32` products summed along a tree of depth at most `D`** (library issue L159),
for one output element: products `p k` of `a k · b k` (IEEE, `tl.MulRel`) summed along a tree over
`0, …, K−1` of depth at most `D` with fusable additions (`AccAdd`) are within
`bcastDepthBound D K B` of `Σ a k · b k`, `B = Σ |a k · b k|`, when the operands are finite and
there is room. The composition of `sum_bcast_mul_tree_err`, with `sumc_err_depth` for
`sumc_err_covers`. A float reduction of products (`tl.sum1c` of `tl.mul`) has a tree of depth at
most `tl.sumBound P K`, which `Rules.Num.SumDepth` with `SumDepth.depth_le` bounds by
`Nat.clog 2 K` (`D = Nat.clog 2 K`; `log₂ K` for a power of two, `Rules.Num.clog_two_pow`).
`tl.sum_bcast_mul` states no depth for its tree form, so this lemma is for a model that writes the
products and the reduction (`tl.mul`, then `tl.sumc` or its axis forms). -/
theorem sum_bcast_mul_depth_err {fusion : Bool} {K D : Nat} (a b p : Nat → F32) {t : Tree} {r : F32}
    (hp : ∀ k, k < K → tl.MulRel .f32 (a k) (b k) (p k)) (hcov : t.Covers K) (hd : t.depth ≤ D)
    (ht : t.Eval (AccAdd fusion .f32) p r)
    (ha : ∀ k, k < K → (a k).isFinite = true) (hb : ∀ k, k < K → (b k).isFinite = true)
    (hab : ∀ k, k < K → |(a k).toRat * (b k).toRat| ≤ Format.binary32.maxVal)
    (hM : ((1 + Format.binary32.u) * ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum +
        K * Format.binary32.eta) +
      Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) D (K - 1)
        ((1 + Format.binary32.u) * ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum +
          K * Format.binary32.eta) ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta) :
    r.isFinite = true ∧
      |r.toRat - ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum| ≤
        bcastDepthBound D K ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum := by
  have hu0 := u_nonneg (f := Format.binary32)
  have he0 := eta_nonneg (f := Format.binary32)
  have hpe : ∀ k, k < K → (p k).isFinite = true ∧
      |(p k).toRat - (a k).toRat * (b k).toRat| ≤ |(a k).toRat * (b k).toRat| * Format.binary32.u +
        Format.binary32.eta := by
    intro k hk
    obtain ⟨hfin, herr⟩ := FloatV.mul_err Format.binary32_regular (ha k hk) (hb k hk) (hab k hk)
    have hne : FloatV.NaNEq (FloatV.mul (a k) (b k)) (p k) := hp k hk
    rw [hne.eq (FloatV.isFinite_iff.mp hfin).1]
    exact ⟨hfin, herr⟩
  set B := ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum with hB
  have hP : ((List.range K).map fun k => |(p k).toRat|).sum ≤ (1 + Format.binary32.u) * B +
      K * Format.binary32.eta := by
    have : ∀ k ∈ List.range K, |(p k).toRat| ≤
        (1 + Format.binary32.u) * |(a k).toRat * (b k).toRat| + Format.binary32.eta := by
      intro k hk
      have hk' := List.mem_range.mp hk
      have h1 := (hpe k hk').2
      have h2 := abs_sub_abs_le_abs_sub (p k).toRat ((a k).toRat * (b k).toRat)
      nlinarith
    have hsum := List.sum_le_sum this
    rw [List.sum_map_add, List.sum_map_mul_left] at hsum
    simpa [hB, mul_comm] using hsum
  have hAP := List.sum_nonneg (l := (List.range K).map fun k => |(p k).toRat|)
    (by simp only [List.mem_map]; rintro _ ⟨k, -, rfl⟩; exact abs_nonneg _)
  have hδ : (0 : ℚ) ≤ 3 * Format.binary32.u := by positivity
  have hη : (0 : ℚ) ≤ 4 * Format.binary32.eta := by positivity
  have hmono := Tree.errBound_mono (d := D) (d' := D) (N := K - 1) (N' := K - 1) hδ hη le_rfl le_rfl hAP hP
  have ht' : t.Eval (AddRel fusion) p r := ht
  obtain ⟨hfin, herr⟩ := sumc_err_depth Format.binary32_regular (by decide) u32_le fusion p hcov hd ht'
    (fun k hk => (hpe k hk).1) (by linarith)
  refine ⟨hfin, ?_⟩
  have hsumP : |((List.range K).map fun k => (p k).toRat).sum -
      ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum| ≤
      Format.binary32.u * B + K * Format.binary32.eta := by
    rw [sum_range_fin, sum_range_fin, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    have hle : ∀ k ∈ (Finset.univ : Finset (Fin K)), |(p k).toRat - (a k).toRat * (b k).toRat| ≤
        Format.binary32.u * |(a k).toRat * (b k).toRat| + Format.binary32.eta := fun k _ => by
      have := (hpe k k.isLt).2
      linarith [mul_comm (|(a k).toRat * (b k).toRat|) Format.binary32.u]
    refine (Finset.sum_le_sum hle).trans (le_of_eq ?_)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, hB, sum_range_fin]
  unfold bcastDepthBound
  have e : r.toRat - ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum =
      (r.toRat - ((List.range K).map fun k => (p k).toRat).sum) +
        (((List.range K).map fun k => (p k).toRat).sum -
          ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum) := by ring
  rw [e]
  exact (abs_add_le _ _).trans (add_le_add (herr.trans hmono) hsumP)

/-- **`tl.sum_bcast_mul` on `float32`**, for one output element, whichever form the device runs:
the tree of fusable additions over the rounded products (`sum_bcast_mul_tree_err`), or the `ieee`
`tl.dot` (`Rules.Num.DotF32Ieee`). The result is finite and within the larger of the two bounds. -/
theorem sum_bcast_mul_err {P : Platform} (hdot : Rules.Num.DotF32Ieee P) {fusion : Bool} {M K N : Nat}
    {x : Tensor [M, K] F32} {y : Tensor [K, N] F32} {r : Tensor [M, N] F32}
    (hr : (∀ i j, ∃ p : Nat → F32,
        (∀ k (hk : k < K), tl.MulRel .f32 (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
        ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd fusion .f32) p (r.get2 i j)) ∨
      (DType.bytes .f32 ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
        P.dot { a := .f32, b := .f32, acc := .f32, precision := .ieee } x y
          (Tensor.full (DType.ofBits .f32 0)) r))
    (i : Fin M) (j : Fin N)
    (hx : ∀ k, (x.get2 i k).isFinite = true) (hy : ∀ k, (y.get2 k j).isFinite = true)
    (hroom : 2 * ((1 + Format.binary32.u) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
        K * Format.binary32.eta) +
      2 * Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (K - 1) (K - 1)
        ((1 + Format.binary32.u) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
          K * Format.binary32.eta) + 4 * Format.binary32.eta ≤ Format.binary32.maxVal)
    (hroom' : ((K : ℚ) + 2) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
      (K + 1) * 2 ^ (-149 : ℤ) ≤ Format.binary32.maxVal) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - ∑ k, (x.get2 i k).toRat * (y.get2 k j).toRat| ≤
        max (bcastTreeBound K (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|))
          ((K + 1) * 2 ^ (-23 : ℤ) * (∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat|) +
            (K + 1) * 2 ^ (-149 : ℤ)) := by
  have hu0 := u_nonneg (f := Format.binary32)
  have he0 := eta_nonneg (f := Format.binary32)
  set S := ∑ k, |(x.get2 i k).toRat * (y.get2 k j).toRat| with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun _ _ => abs_nonneg _
  rcases hr with htree | ⟨-, -, -, hd⟩
  · obtain ⟨p, hp, t, hcov, ht⟩ := htree i j
    let a : Nat → F32 := fun k => if h : k < K then x.get2 i ⟨k, h⟩ else default
    let b : Nat → F32 := fun k => if h : k < K then y.get2 ⟨k, h⟩ j else default
    have hak : ∀ k (hk : k < K), a k = x.get2 i ⟨k, hk⟩ := fun k hk => dite_eq_left_of_eq_true (eq_true hk)
    have hbk : ∀ k (hk : k < K), b k = y.get2 ⟨k, hk⟩ j := fun k hk => dite_eq_left_of_eq_true (eq_true hk)
    have hsumab : ((List.range K).map fun k => (a k).toRat * (b k).toRat).sum =
        ∑ k, (x.get2 i k).toRat * (y.get2 k j).toRat := by
      rw [sum_range_fin]
      exact Finset.sum_congr rfl fun k _ => by rw [hak k k.isLt, hbk k k.isLt]
    have habs : ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum = S := by
      rw [sum_range_fin]
      exact Finset.sum_congr rfl fun k _ => by rw [hak k k.isLt, hbk k k.isLt]
    have hErr0 : 0 ≤ Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) (K - 1) (K - 1)
        ((1 + Format.binary32.u) * S + K * Format.binary32.eta) :=
      Tree.errBound_nonneg (by positivity) (by positivity) _ _ (by positivity)
    have hSmax : S ≤ Format.binary32.maxVal := by nlinarith
    have hab : ∀ k, k < K → |(a k).toRat * (b k).toRat| ≤ Format.binary32.maxVal := by
      intro k hk
      rw [hak k hk, hbk k hk]
      have := Finset.single_le_sum (f := fun k : Fin K => |(x.get2 i k).toRat * (y.get2 k j).toRat|)
        (fun _ _ => abs_nonneg _) (Finset.mem_univ ⟨k, hk⟩)
      exact this.trans hSmax
    obtain ⟨hfin, herr⟩ := sum_bcast_mul_tree_err a b p
      (fun k hk => by rw [hak k hk, hbk k hk]; exact hp k hk) hcov ht
      (fun k hk => by rw [hak k hk]; exact hx _) (fun k hk => by rw [hbk k hk]; exact hy _) hab
      (by rw [habs]; linarith)
    rw [hsumab, habs] at herr
    exact ⟨hfin, herr.trans (le_max_left _ _)⟩
  · have hz : DType.val? .f32 ((Tensor.full (DType.ofBits .f32 0) : Tensor [M, N] F32).get2 i j) = some 0 :=
      DotLoop.zero_val
    obtain ⟨vr, hvr, herr⟩ := hdot x y _ r hd i j (fun k => (x.get2 i k).toRat) (fun k => (y.get2 k j).toRat) 0
      (fun k => DotLoop.val?_f32.mpr ⟨hx k, rfl⟩) (fun k => DotLoop.val?_f32.mpr ⟨hy k, rfl⟩)
      (fun _ => ⟨trivial, trivial⟩) hz (by
        simp only [abs_zero, zero_add, DType.bound]
        have h1 : ((K : ℚ) + 1) * 2 ^ (-23 : ℤ) ≤ K + 1 := by
          have : (2 : ℚ) ^ (-23 : ℤ) ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) (by norm_num)
          nlinarith [Nat.cast_nonneg (α := ℚ) K]
        nlinarith)
    obtain ⟨hfin, hval⟩ := DotLoop.val?_f32.mp hvr
    refine ⟨hfin, ?_⟩
    rw [hval]
    simp only [abs_zero, zero_add] at herr
    exact herr.trans (le_max_right _ _)

end Bcast

end TVA
