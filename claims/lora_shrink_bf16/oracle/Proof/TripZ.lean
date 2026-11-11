import Proof.TripY

/-!
# `lora_shrink_bf16`: sums over the chunks of a part

The exact partial sums and lifted magnitudes of a part grow, trip by trip, by the sums over the
tile of the trip; the chunks of a part are disjoint, so the lifted magnitude of a part is at most
`dotMag`.
-/

namespace Claims.LoraShrink

open TVA

/-- The device's chunks of a `BLOCK_K`-long dot cover the tile once. -/
theorem sum_chunks_fin (d : Inst) (G : Fin (bk d) → ℚ) :
    ∑ x ∈ Finset.range (cpb d), ∑ kk ∈ Device.chunk16 (bk d) x, G kk = ∑ kk, G kk := by
  rw [← chunks16_bk d]
  unfold Device.chunk16
  exact Finset.sum_fiberwise_of_maps_to (g := fun k : Fin (bk d) => k.val / 64)
    (fun k _ => Finset.mem_range.mpr (Device.div_lt_chunks16 k)) _

/-- **The claim's chunks of trip `t`**, summed, are the sum over the tile. -/
theorem block_sum (d : Inst) (p t : Nat) (F : Nat → ℚ) (G : Fin (bk d) → ℚ)
    (hG : ∀ kk : Fin (bk d), G kk =
      if (p + sk d * t) * bk d + kk < d.K then F ((p + sk d * t) * bk d + kk) else 0) :
    ∑ x ∈ Finset.range (cpb d), ∑ k ∈ chunkK d p (cpb d * t + x), F k = ∑ kk, G kk := by
  rw [← sum_chunks_fin d G]
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [Nat.mul_comm, sum_chunk_eq d p t (Finset.mem_range.mp hx) F G hG]

/-- A product of two tile entries given by `hG`, `hH` is the product of the claim's values, cut at
`K`. -/
theorem tile_prod (d : Inst) (p t : Nat) (F F' : Nat → ℚ) (G H : Fin (bk d) → ℚ)
    (hG : ∀ kk : Fin (bk d), G kk =
      if (p + sk d * t) * bk d + kk < d.K then F ((p + sk d * t) * bk d + kk) else 0)
    (hH : ∀ kk : Fin (bk d), H kk =
      if (p + sk d * t) * bk d + kk < d.K then F' ((p + sk d * t) * bk d + kk) else 0)
    (kk : Fin (bk d)) :
    G kk * H kk =
      if (p + sk d * t) * bk d + kk < d.K then
        F ((p + sk d * t) * bk d + kk) * F' ((p + sk d * t) * bk d + kk) else 0 := by
  rw [hG, hH]; split_ifs <;> simp

/-- **The exact partial sum grows by the tile's dot product.** -/
theorem pSum_block (d : Inst) (va vb : Nat → BF16) (p t : Nat) (G H : Fin (bk d) → ℚ)
    (hG : ∀ kk : Fin (bk d), G kk =
      if (p + sk d * t) * bk d + kk < d.K then (va ((p + sk d * t) * bk d + kk)).toRat else 0)
    (hH : ∀ kk : Fin (bk d), H kk =
      if (p + sk d * t) * bk d + kk < d.K then (vb ((p + sk d * t) * bk d + kk)).toRat else 0) :
    pSum d va vb p (cpb d * (t + 1)) = pSum d va vb p (cpb d * t) + ∑ kk, G kk * H kk := by
  have h := block_sum d p t (fun k => (va k).toRat * (vb k).toRat) (fun kk => G kk * H kk)
    (tile_prod d p t (fun k => (va k).toRat) (fun k => (vb k).toRat) G H hG hH)
  unfold pSum
  rw [Nat.mul_succ, Finset.sum_range_add, ← h]
  rfl

/-- **The lifted magnitude grows by the tile's lifted magnitude.** -/
theorem pLift_block (d : Inst) (va vb : Nat → BF16) (p t : Nat) (G H : Fin (bk d) → ℚ)
    (hG : ∀ kk : Fin (bk d), G kk =
      if (p + sk d * t) * bk d + kk < d.K then (va ((p + sk d * t) * bk d + kk)).toRat else 0)
    (hH : ∀ kk : Fin (bk d), H kk =
      if (p + sk d * t) * bk d + kk < d.K then (vb ((p + sk d * t) * bk d + kk)).toRat else 0) :
    pLift d va vb p (t + 1) = pLift d va vb p t +
      ∑ kk, Device.liftMag .bf16 (G kk) * Device.liftMag .bf16 (H kk) := by
  have h := block_sum d p t
    (fun k => Device.liftMag .bf16 (va k).toRat * Device.liftMag .bf16 (vb k).toRat)
    (fun kk => Device.liftMag .bf16 (G kk) * Device.liftMag .bf16 (H kk)) (by
      intro kk
      rw [hG, hH]
      split_ifs <;> simp)
  unfold pLift
  rw [Nat.mul_succ, Finset.sum_range_add, ← h]

/-- The exact partial sum is at most the lifted magnitude. -/
theorem abs_pSum_le (d : Inst) (va vb : Nat → BF16) (p t : Nat) :
    |pSum d va vb p (cpb d * t)| ≤ pLift d va vb p t := by
  unfold pSum pLift cSum
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun q _ => ?_)
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
  rw [abs_mul] at *
  exact Device.abs_mul_le_liftMag _ _ _ _ |>.trans_eq' (by rw [abs_mul])

/-- The chunks of one part are disjoint. -/
theorem chunkK_disjoint (d : Inst) (p : Nat) {q q' : Nat} (hqq : q ≠ q') :
    Disjoint (chunkK d p q) (chunkK d p q') := by
  rw [Finset.disjoint_left]
  intro k h1 h2
  simp only [chunkK, Finset.mem_filter] at h1 h2
  have hs := sk_pos d
  have hc := cpb_pos d
  have e1 : q / cpb d = q' / cpb d := by
    have := h1.2.1.symm.trans h2.2.1
    have := Nat.eq_of_mul_eq_mul_left hs (by omega : sk d * (q / cpb d) = sk d * (q' / cpb d))
    exact this
  have e2 : q % cpb d = q' % cpb d := h1.2.2.symm.trans h2.2.2
  apply hqq
  rw [← Nat.div_add_mod q (cpb d), ← Nat.div_add_mod q' (cpb d), e1, e2]

/-- The lifted magnitude of a part, after any number of chunks, is at most `dotMag`. -/
theorem pLift_le_dotMag (d : Inst) (va vb : Nat → BF16) (p t : Nat) :
    pLift d va vb p t ≤ dotMag d va vb := by
  unfold pLift dotMag
  rw [← Finset.sum_biUnion (s := Finset.range (cpb d * t)) (t := chunkK d p)
    (f := fun k => Device.liftMag .bf16 (va k).toRat * Device.liftMag .bf16 (vb k).toRat)
    (fun q _ q' _ hqq => chunkK_disjoint d p hqq)]
  refine Finset.sum_le_sum_of_subset_of_nonneg ?_ fun k _ _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  intro k hk
  rw [Finset.mem_biUnion] at hk
  obtain ⟨q, -, hq⟩ := hk
  simp only [chunkK, Finset.mem_filter] at hq
  exact hq.1

/-- The lifted magnitude is nonnegative. -/
theorem pLift_nonneg (d : Inst) (va vb : Nat → BF16) (p t : Nat) : 0 ≤ pLift d va vb p t :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)

end Claims.LoraShrink
