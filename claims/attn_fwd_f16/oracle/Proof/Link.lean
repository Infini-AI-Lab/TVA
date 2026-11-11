import Proof.Defs

/-!
# The row data against the claim's reference

The exact state of a row after `T` blocks (`RowCtx.Ls`, `As`, `Ms`, `Vs`) is the claim's reference
quantities (`weight`, `attnRef`, `attnAbs`, `vAbsSum`, `lseRef`) up to the factor `exp(−m·ln 2)` of
the reference `m`, when the blocks cover the attended keys.
-/

namespace Claims.AttnFwd

open TVA Finset

noncomputable section

variable {d : Inst} {init : Mem} {bh n T : Nat} (h : HeadOk d init bh)

/-- A key is accepted by the row context exactly when it lies in `keys d n`. -/
theorem rowCtx_ok_iff (k : Nat) : (rowCtx d init bh n h).ok k = true ↔ k ∈ keys d n := by
  simp [rowCtx]

/-- The row sum over `T` blocks is `exp(−m·ln 2)` times the sum of the weights. -/
theorem rowCtx_Ls (hT : keys d n ⊆ range (T * 64)) (m : ℝ) :
    (rowCtx d init bh n h).Ls m T =
      Real.exp (-(m * Real.log 2)) * ∑ j ∈ keys d n, weight d init bh n j := by
  have hb := Softmax.lsum_blocks (fun k => k ∈ keys d n)
    (fun k => (sigma d : ℝ) * (score d init bh n k : ℝ)) 64 (m * Real.log 2) T
  have e1 : (rowCtx d init bh n h).Ls m T = Softmax.lsum
      (fun u (j : Fin 64) => (sigma d : ℝ) * (score d init bh n (u * 64 + j) : ℝ))
      (fun u => univ.filter (fun j : Fin 64 => (u * 64 + j) ∈ keys d n)) (m * Real.log 2) T := by
    unfold RowCtx.Ls RowCtx.sf RowCtx.blk
    congr 1
  rw [e1, hb]
  have e2 : (range (T * 64)).filter (fun k => k ∈ keys d n) = keys d n := by
    ext k; simp only [mem_filter, mem_range]
    exact ⟨fun h => h.2, fun hk => ⟨mem_range.mp (hT hk), hk⟩⟩
  rw [e2, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  unfold weight
  rw [← Real.exp_add]; congr 1; ring

/-- Filtering the tile range by the row context keeps exactly `keys d n`. -/
theorem filter_keys (hT : keys d n ⊆ range (T * 64)) :
    (range (T * 64)).filter (fun k => k ∈ keys d n) = keys d n := by
  ext k; simp only [mem_filter, mem_range]
  exact ⟨fun h => h.2, fun hk => ⟨mem_range.mp (hT hk), hk⟩⟩

/-- The row context `As` sum agrees with the reference sum over the kept keys. -/
theorem rowCtx_As (hT : keys d n ⊆ range (T * 64)) (e : Nat) (m : ℝ) :
    (rowCtx d init bh n h).As e m T =
      Real.exp (-(m * Real.log 2)) *
        ∑ j ∈ keys d n, weight d init bh n j * (val16 init d.v (idx d bh j e) : ℝ) := by
  have hb := Softmax.asum_blocks (fun k => k ∈ keys d n)
    (fun k => (sigma d : ℝ) * (score d init bh n k : ℝ))
    (fun k => (val16 init d.v (idx d bh k e) : ℝ)) 64 (m * Real.log 2) T
  have e1 : (rowCtx d init bh n h).As e m T = Softmax.asum
      (fun u (j : Fin 64) => (sigma d : ℝ) * (score d init bh n (u * 64 + j) : ℝ))
      (fun u (j : Fin 64) => (val16 init d.v (idx d bh (u * 64 + j) e) : ℝ))
      (fun u => univ.filter (fun j : Fin 64 => (u * 64 + j) ∈ keys d n)) (m * Real.log 2) T := by
    unfold RowCtx.As RowCtx.sf RowCtx.vf RowCtx.blk
    congr 1
  rw [e1, hb, filter_keys hT, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  unfold weight
  rw [← mul_assoc, ← Real.exp_add]; congr 2; ring

/-- The row context `Ms` sum agrees with the reference sum over the kept keys. -/
theorem rowCtx_Ms (hT : keys d n ⊆ range (T * 64)) (e : Nat) (m : ℝ) :
    (rowCtx d init bh n h).Ms e m T =
      Real.exp (-(m * Real.log 2)) *
        ∑ j ∈ keys d n, weight d init bh n j * |(val16 init d.v (idx d bh j e) : ℝ)| := by
  have hb := Softmax.asum_blocks (fun k => k ∈ keys d n)
    (fun k => (sigma d : ℝ) * (score d init bh n k : ℝ))
    (fun k => |(val16 init d.v (idx d bh k e) : ℝ)|) 64 (m * Real.log 2) T
  have e1 : (rowCtx d init bh n h).Ms e m T = Softmax.asum
      (fun u (j : Fin 64) => (sigma d : ℝ) * (score d init bh n (u * 64 + j) : ℝ))
      (fun u (j : Fin 64) => |(val16 init d.v (idx d bh (u * 64 + j) e) : ℝ)|)
      (fun u => univ.filter (fun j : Fin 64 => (u * 64 + j) ∈ keys d n)) (m * Real.log 2) T := by
    unfold RowCtx.Ms RowCtx.sf RowCtx.vf RowCtx.blk
    congr 1
  rw [e1, hb, filter_keys hT, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  unfold weight
  rw [← mul_assoc, ← Real.exp_add]; congr 2; ring

/-- The row context `Vs` sum agrees with the reference sum over the kept keys. -/
theorem rowCtx_Vs (hT : keys d n ⊆ range (T * 64)) (e : Nat) :
    (rowCtx d init bh n h).Vs e T = (vAbsSum d init bh n e : ℝ) := by
  have hb := Softmax.sum_blocks_fin_filter (fun k => k ∈ keys d n)
    (fun k => |(val16 init d.v (idx d bh k e) : ℝ)|) 64 T
  have e1 : (rowCtx d init bh n h).Vs e T = ∑ u ∈ range T,
      ∑ j ∈ univ.filter (fun j : Fin 64 => (u * 64 + j) ∈ keys d n),
        |(val16 init d.v (idx d bh (u * 64 + j) e) : ℝ)| := by
    unfold RowCtx.Vs RowCtx.vf RowCtx.blk
    congr 1
  rw [e1, hb, filter_keys hT]
  unfold vAbsSum
  push_cast
  rfl

/-- The log-sum-exp reference is `m + log₂` of the row sum at the reference `m`. -/
theorem rowCtx_lse (hT : keys d n ⊆ range (T * 64)) (hne : (keys d n).Nonempty) (m : ℝ) :
    lseRef d init bh n = m + Real.logb 2 ((rowCtx d init bh n h).Ls m T) := by
  rw [rowCtx_Ls h hT]
  unfold lseRef
  have hpos : 0 < ∑ j ∈ keys d n, weight d init bh n j :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) hne
  rw [Real.logb_mul (Real.exp_pos _).ne' hpos.ne']
  have h2 : Real.log 2 ≠ 0 := by positivity
  have : Real.logb 2 (Real.exp (-(m * Real.log 2))) = -m := by
    unfold Real.logb; rw [Real.log_exp]; field_simp
  rw [this]; ring

end

end Claims.AttnFwd
