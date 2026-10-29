import TVA.Lemmas.Tree
import TVA.Lemmas.FloatErr
import TVA.Lang.Ops

/-!
# Error of a float reduction

`tl.sum` adds along a tree Triton chooses, and each node is an IEEE addition (`AddRel false`).
`sum_err` bounds the result against the exact sum of the leaves for a given tree. `sum_err_covers`
gives the bound that holds for *every* tree over `0, …, n−1`, which is what a kernel proof can
use. It uses the worst depth, `n − 1`:

`|r − Σ xᵢ| ≤ ((1 + u)^(n−1) − 1) · Σ|xᵢ| + η · (n − 1) · (1 + u)^(n−1)`

This holds when every element is finite and the bound keeps every partial sum in range.
-/
namespace TVA

open TVA.Math.Float TVA.Math.Float.Format Tree

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

/-- **A float reduction along any tree over `0, …, n−1`.** The bound uses depth and node count
`n − 1`. -/
theorem sum_err_covers (hf : f.Regular) (x : Nat → FloatV f) {n : Nat} {t : Tree} {r : FloatV f}
    (hcov : t.Covers n) (h : t.Eval (AddRel false) x r) (hx : ∀ i, i < n → (x i).isFinite = true)
    (hM : ((List.range n).map fun i => |(x i).toRat|).sum +
      errBound f.u f.eta (n - 1) (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum ≤ f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - ((List.range n).map fun i => (x i).toRat).sum| ≤
        errBound f.u f.eta (n - 1) (n - 1) ((List.range n).map fun i => |(x i).toRat|).sum := by
  have hN := nodes_of_covers hcov
  have hD : t.depth ≤ n - 1 := by have := depth_le_nodes t; omega
  have hmono := errBound_mono (δ := f.u) (η := f.eta) u_nonneg eta_nonneg hD (show t.nodes ≤ n - 1 by omega)
    (absSumOf_nonneg FloatV.toRat x t) (le_refl (absSumOf FloatV.toRat x t))
  rw [absSumOf_covers FloatV.toRat x hcov] at hmono
  have hx' : ∀ i ∈ t.leaves, (x i).isFinite = true := fun i hi =>
    hx i (List.mem_range.mp (hcov.subset hi))
  obtain ⟨hfin, herr⟩ := sum_err hf x h hx' (by rw [absSumOf_covers FloatV.toRat x hcov]; linarith)
  rw [sumOf_covers FloatV.toRat x hcov, absSumOf_covers FloatV.toRat x hcov] at herr
  exact ⟨hfin, herr.trans hmono⟩

end TVA
