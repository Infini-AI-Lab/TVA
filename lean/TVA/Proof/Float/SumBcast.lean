import TVA.Proof.Float.Sum

/-!
# Rounded products summed along a tree (`float32`)

A reduction of products, `tl.sum(a * b, axis)` written as `tl.mul` then `tl.sumc`, and the tree
form of `tl.sum_bcast_mul` (`TVA.Triton.Dot`), round each product once (`tl.MulRel`) and add the
products along a tree of fusable additions (`AccAdd`). For one output element with `K` products of
absolute sum `B = Σ |aₖ·bₖ|`:

* the products' roundings cost at most `u·B + K·η`, and leave products of absolute sum at most
  `(1 + u)·B + K·η`;
* the tree then costs `errBound (3u) (4η) D (K − 1) ((1 + u)·B + K·η)` (`sumc_err_depth`).

`bcastDepthBound D K B` is their sum; `sum_bcast_mul_depth_err` proves the result finite and
within it of the exact `Σ aₖ·bₖ`, for a tree of depth at most `D`; `sum_bcast_mul_tree_err` is
the case of any tree (`D = K − 1`, `bcastTreeBound`). The other form `tl.sum_bcast_mul` may take,
the platform's `tl.dot`, is bounded by the device's dot accuracy, not here.
-/
namespace TVA

open Format

/-- `u` of `float32`. -/
private theorem u32 : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u

/-- `float32` has at least 3 fraction bits, as the fused-addition bound needs. -/
private theorem u32_le : Format.binary32.u ≤ 1 / 16 := by rw [u32]; norm_num

/-- A sum over `List.range n` is the sum over `Fin n`. -/
theorem sum_range_fin (g : ℕ → ℚ) : ∀ n, ((List.range n).map g).sum = ∑ k : Fin n, g k
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_range_fin g n, Fin.sum_univ_castSucc]
    simp

/-- The bound of the products `a k · b k` (`K` of them, absolute sum `B`), each rounded once in
`float32`, summed along a tree of fusable additions of depth at most `D` (`K − 1` for any tree). -/
def bcastDepthBound (D K : Nat) (B : ℚ) : ℚ :=
  Tree.errBound (3 * Format.binary32.u) (4 * Format.binary32.eta) D (K - 1)
      ((1 + Format.binary32.u) * B + K * Format.binary32.eta) +
    (Format.binary32.u * B + K * Format.binary32.eta)

/-- The bound of `bcastDepthBound` for any tree over `K` products (depth `K − 1`). -/
def bcastTreeBound (K : Nat) (B : ℚ) : ℚ := bcastDepthBound (K - 1) K B

/-- **Rounded `float32` products summed along a tree of depth at most `D`**, for one output
element: products `p k` of `a k · b k` (IEEE, `tl.MulRel`) summed along a tree over `0, …, K−1`
of depth at most `D` with fusable additions (`AccAdd`) are finite and within `bcastDepthBound D K B`
of `Σ a k · b k`, `B = Σ |a k · b k|`, when the operands are finite and there is room. Use it for
a reduction of products (`tl.mul`, then `tl.sumc` or `tl.sum1c`), with `D` from `tl.sumBound`. -/
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
/-- **The tree form of `tl.sum_bcast_mul` on `float32`**, for one output element and any tree
over `0, …, K−1`: `sum_bcast_mul_depth_err` at depth `K − 1`. -/
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
        bcastTreeBound K ((List.range K).map fun k => |(a k).toRat * (b k).toRat|).sum :=
  sum_bcast_mul_depth_err a b p hp hcov
    (by have := Tree.depth_le_nodes t; have := Tree.nodes_of_covers hcov; omega) ht ha hb hab hM

end TVA
