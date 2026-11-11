import TVA.Proof.Float.Fused

/-!
# The error of a float reduction

`tl.sum` (`tl.sumWith`, `TVA.Triton.Reduce`) adds along a tree Triton chooses: any tree that
covers `0, …, n−1` (`Tree.Covers`) of depth at most `tl.sumBound P n`. Each node is an IEEE
addition (`AddRel false`), or a fusable one (`AddRel fusion`) when the elements are products
(`tl.sumc`, `tl.sum1c`, …). The bounds here hold for every such tree; they are the bound of
`TVA.Proof.Float.Tree` with the per-node constants of one addition:

* IEEE additions: `sum_err` (a given tree), `sum_err_depth` (any tree over `0, …, n−1` of depth
  at most `D`), `sum_err_covers` (any tree, `D = n − 1`), with `δ = u`, `η` of the format:

  `|r − Σ xᵢ| ≤ errBound u η D (n − 1) (Σ |xᵢ|)`;

* fusable additions: `sumc_err`, `sumc_err_depth`, `sumc_err_covers`, `sumc_err_list` (leaves a
  permutation of a list `L`: a position of `tl.cumsumc`, a segment), with `δ = 3u` and `4η`.

Each assumes finite elements and room: the leaves' absolute sum plus the bound is at most
`maxVal` (`maxVal/2 − 2η` for fusable additions), so no partial sum overflows; the result is then
finite. `TVA.Proof.Float.SumBcast` composes these with the rounding of products.
-/
namespace TVA

open Format TVA.Tree

variable {f : Format}

/-- **A float reduction along a given tree.** -/
theorem sum_err (hf : f.Regular) (x : Nat → FloatV f) {t : Tree} {r : FloatV f}
    (h : t.Eval (AddRel false) x r) (hx : ∀ i ∈ t.leaves, (x i).isFinite = true)
    (hM : absSumOf FloatV.toRat x t + errBound f.u f.eta t.depth t.nodes (absSumOf FloatV.toRat x t) ≤
      f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - sumOf FloatV.toRat x t| ≤ errBound f.u f.eta t.depth t.nodes (absSumOf FloatV.toRat x t) :=
  eval_err FloatV.toRat x (fun a => a.isFinite = true) u_nonneg eta_nonneg
    (fun a b c hc ha hb hab => by
      rw [AddRel.false_iff] at hc
      obtain ⟨h1, h2⟩ := FloatV.add_err hf ha hb hab
      have hnan : (FloatV.add a b).isNaN = false :=
        (FloatV.isFinite_iff.mp h1).1
      rw [hc.eq hnan]
      exact ⟨h1, by rw [mul_comm]; exact h2⟩) h hx hM

/-- **A float reduction along a tree over `0, …, n−1` of depth at most `D`**: the bound with
depth `D` and node count `n − 1`. Use it with the platform's depth bound (`tl.sumBound`). -/
theorem sum_err_depth (hf : f.Regular) (x : Nat → FloatV f) {n D : Nat} {t : Tree} {r : FloatV f}
    (hcov : t.Covers n) (hd : t.depth ≤ D) (h : t.Eval (AddRel false) x r)
    (hx : ∀ i, i < n → (x i).isFinite = true)
    (hM : ((List.range n).map fun i => |(x i).toRat|).sum +
      errBound f.u f.eta D (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum ≤ f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - ((List.range n).map fun i => (x i).toRat).sum| ≤
        errBound f.u f.eta D (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum := by
  have hN := nodes_of_covers hcov
  have hmono := errBound_mono (δ := f.u) (η := f.eta) u_nonneg eta_nonneg hd (show t.nodes ≤ n - 1 by omega)
    (absSumOf_nonneg FloatV.toRat x t) (le_refl (absSumOf FloatV.toRat x t))
  rw [absSumOf_covers FloatV.toRat x hcov] at hmono
  have hx' : ∀ i ∈ t.leaves, (x i).isFinite = true := fun i hi =>
    hx i (List.mem_range.mp (hcov.subset hi))
  obtain ⟨hfin, herr⟩ := sum_err hf x h hx' (by rw [absSumOf_covers FloatV.toRat x hcov]; linarith)
  rw [sumOf_covers FloatV.toRat x hcov, absSumOf_covers FloatV.toRat x hcov] at herr
  exact ⟨hfin, herr.trans hmono⟩

/-- **A float reduction along any tree over `0, …, n−1`**: the bound with depth and node count
`n − 1`, which holds whatever tree Triton chooses. -/
theorem sum_err_covers (hf : f.Regular) (x : Nat → FloatV f) {n : Nat} {t : Tree} {r : FloatV f}
    (hcov : t.Covers n) (h : t.Eval (AddRel false) x r) (hx : ∀ i, i < n → (x i).isFinite = true)
    (hM : ((List.range n).map fun i => |(x i).toRat|).sum +
      errBound f.u f.eta (n - 1) (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum ≤ f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - ((List.range n).map fun i => (x i).toRat).sum| ≤
        errBound f.u f.eta (n - 1) (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum :=
  sum_err_depth hf x hcov (by have := depth_le_nodes t; have := nodes_of_covers hcov; omega) h hx hM

/-! ## Fusable additions -/

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

/-- **A float reduction with fusable additions along a tree over `0, …, n−1` of depth at most
`D`**: depth `D` (from `tl.sumBound`), node count `n − 1`, per-node constants `3u` and `4η`. -/
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

/-- **A float reduction with fusable additions along any tree over `0, …, n−1`**: the bound with
depth and node count `n − 1` and per-node constants `3u` and `4η`. This is what a kernel that
sums products (`tl.sum(x * y)` as `tl.sumc`) uses, for every tree Triton may choose. -/
theorem sumc_err_covers (hf : f.Regular) (hs : f.specials ≠ .finite) (hu : f.u ≤ 1 / 16)
    (fusion : Bool) (x : Nat → FloatV f) {n : Nat} {t : Tree} {r : FloatV f}
    (hcov : t.Covers n) (h : t.Eval (AddRel fusion) x r) (hx : ∀ i, i < n → (x i).isFinite = true)
    (hM : ((List.range n).map fun i => |(x i).toRat|).sum +
      Tree.errBound (3 * f.u) (4 * f.eta) (n - 1) (n - 1)
        ((List.range n).map fun i => |(x i).toRat|).sum ≤ f.maxVal / 2 - 2 * f.eta) :
    r.isFinite = true ∧
      |r.toRat - ((List.range n).map fun i => (x i).toRat).sum| ≤
        Tree.errBound (3 * f.u) (4 * f.eta) (n - 1) (n - 1)
          ((List.range n).map fun i => |(x i).toRat|).sum :=
  sumc_err_depth hf hs hu fusion x hcov
    (by have := depth_le_nodes t; have := nodes_of_covers hcov; omega) h hx hM

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

end TVA
