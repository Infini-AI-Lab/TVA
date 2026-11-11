import Proof.TripX

/-!
# `lora_shrink_bf16`: the chunks of one trip

The chunk `j` of the `tl.dot` of trip `t` of part `p` (`Device.chunk16`) is the claim's chunk
`t · cpb + j` of part `p` (`chunkK`), the entries past `K` being `+0.0`.
-/

namespace Claims.LoraShrink

open TVA

/-- Membership in the claim's chunk `t · cpb + j`. -/
theorem mem_chunkK (d : Inst) (p t : Nat) {j : Nat} (hj : j < cpb d) (k : Nat) :
    k ∈ chunkK d p (t * cpb d + j) ↔
      k < d.K ∧ k / bk d = p + sk d * t ∧ k % bk d / cw d = j := by
  have h1 : (t * cpb d + j) / cpb d = t := by
    rw [Nat.mul_comm, Nat.mul_add_div (cpb_pos d), Nat.div_eq_of_lt hj, Nat.add_zero]
  have h2 : (t * cpb d + j) % cpb d = j := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hj]
  simp [chunkK, h1, h2]

/-- Membership in the device's chunk `j`. -/
theorem mem_chunk16 {K j : Nat} (kk : Fin K) : kk ∈ Device.chunk16 K j ↔ kk.val / 64 = j := by
  simp [Device.chunk16]

/-- A tile entry of the dot's chunk `j` below `K` is in the claim's chunk `t · cpb + j`. -/
theorem mem_chunkK_tile (d : Inst) (p t : Nat) {j : Nat} (hj : j < cpb d) (kk : Fin (bk d))
    (hkk : kk ∈ Device.chunk16 (bk d) j) (h : (p + sk d * t) * bk d + kk < d.K) :
    (p + sk d * t) * bk d + kk ∈ chunkK d p (t * cpb d + j) := by
  rw [mem_chunk16] at hkk
  rw [mem_chunkK d p t hj]
  have hb := bk_pos d
  have := kk.isLt
  refine ⟨h, ?_, ?_⟩
  · rw [Nat.mul_comm, Nat.mul_add_div hb, Nat.div_eq_of_lt this]; omega
  · rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt this, ← div64_eq d this]; exact hkk

/-- **The dot's chunk `j` is the claim's chunk `t · cpb + j`.** If entry `kk` of the tile is
`F k` at `k = (p + SPLIT_K · t) · BLOCK_K + kk` when `k < K` and `0` beyond, the sum over the dot's
chunk `j` is the sum of `F` over the claim's chunk. -/
theorem sum_chunk_eq (d : Inst) (p t : Nat) {j : Nat} (hj : j < cpb d) (F : Nat → ℚ)
    (G : Fin (bk d) → ℚ)
    (hG : ∀ kk : Fin (bk d), G kk =
      if (p + sk d * t) * bk d + kk < d.K then F ((p + sk d * t) * bk d + kk) else 0) :
    ∑ kk ∈ Device.chunk16 (bk d) j, G kk = ∑ k ∈ chunkK d p (t * cpb d + j), F k := by
  simp only [hG]
  rw [← Finset.sum_filter]
  refine Finset.sum_nbij (fun kk : Fin (bk d) => (p + sk d * t) * bk d + kk.val) ?_ ?_ ?_ ?_
  · intro kk hkk
    rw [Finset.mem_filter, mem_chunk16] at hkk
    rw [mem_chunkK d p t hj]
    have hb := bk_pos d
    have := kk.isLt
    refine ⟨hkk.2, ?_, ?_⟩
    · rw [Nat.mul_comm, Nat.mul_add_div hb, Nat.div_eq_of_lt this]; omega
    · rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt this, ← div64_eq d this]; exact hkk.1
  · intro a ha b hb hab
    exact Fin.ext (by simpa using hab)
  · intro k hk
    have hk' := (mem_chunkK d p t hj k).1 hk
    have hb := bk_pos d
    have hkk : k % bk d < bk d := Nat.mod_lt _ hb
    refine ⟨⟨k % bk d, hkk⟩, ?_, ?_⟩
    · rw [Finset.mem_coe, Finset.mem_filter, mem_chunk16]
      have hdiv := Nat.div_add_mod k (bk d)
      refine ⟨?_, ?_⟩
      · exact (div64_eq d hkk).trans hk'.2.2
      · show (p + sk d * t) * bk d + k % bk d < d.K
        rw [hk'.2.1] at hdiv; rw [Nat.mul_comm] at hdiv; omega
    · simp only
      have hdiv := Nat.div_add_mod k (bk d)
      rw [hk'.2.1] at hdiv; rw [Nat.mul_comm] at hdiv; omega
  · intro a ha
    rw [Finset.mem_filter, mem_chunk16] at ha

end Claims.LoraShrink
