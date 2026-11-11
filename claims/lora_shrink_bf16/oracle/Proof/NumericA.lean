import Proof.Defs
import Proof.Cfg

/-!
# `lora_shrink_bf16`: the parts of `K` partition `[0, K)`

The `SPLIT_K` programs' chunks (`chunkK`) together cover each `k < K` exactly once, so any sum over
`k < K` is the sum over parts and chunks. Also the two launch configurations (`BLOCK_K = 256`,
`SPLIT_K = 64` below 128 tokens; `32`, `8` from 128).
-/

namespace Claims.LoraShrink

open TVA

/-- The chunk of `k` within its part: `((k / BLOCK_K) / SPLIT_K) · cpb + (k % BLOCK_K) / cw`. -/
def chunkIdx (d : Inst) (k : Nat) : Nat := k / bk d / sk d * cpb d + k % bk d / cw d

/-- The elements of part `p` below `K`. -/
def partSet (d : Inst) (p : Nat) : Finset Nat :=
  (Finset.range d.K).filter fun k => k / bk d % sk d = p

/-- `BLOCK_K` is `cpb · cw`. -/
theorem bk_cpb_cw (d : Inst) : bk d = cpb d * cw d := by
  rcases cfg_cases d with h | h <;> simp [h.2.1, h.2.2.1, h.2.2.2.1]

/-- The chunk of part `p` is the elements of the part with that chunk index. -/
theorem chunkK_eq (d : Inst) {p : Nat} (hp : p < sk d) (q : Nat) :
    chunkK d p q = (partSet d p).filter fun k => chunkIdx d k = q := by
  have hcw := cw_pos d
  have hcpb := cpb_pos d
  have hsk := sk_pos d
  have hbk := bk_cpb_cw d
  apply Finset.ext
  intro k
  rw [chunkK, partSet, Finset.mem_filter, Finset.mem_filter, Finset.mem_filter, Finset.mem_range]
  unfold chunkIdx
  simp only [and_assoc]
  constructor
  · rintro ⟨hk, h1, h2⟩
    have hm : k / bk d % sk d = p := by rw [h1, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]
    have hu : k / bk d / sk d = q / cpb d := by
      rw [h1, Nat.add_mul_div_left _ _ hsk, Nat.div_eq_of_lt hp, zero_add]
    refine ⟨hk, hm, ?_⟩
    rw [hu, h2]
    have := Nat.div_add_mod q (cpb d)
    nlinarith
  · rintro ⟨hk, hm, hq⟩
    have hmlt : k % bk d / cw d < cpb d := by
      apply Nat.div_lt_of_lt_mul
      have := Nat.mod_lt k (bk_pos d)
      rw [mul_comm]; omega
    have hq1 : q / cpb d = k / bk d / sk d := by
      rw [← hq, Nat.add_comm, Nat.add_mul_div_right _ _ hcpb, Nat.div_eq_of_lt hmlt, zero_add]
    have hq2 : q % cpb d = k % bk d / cw d := by
      rw [← hq, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hmlt]
    refine ⟨hk, ?_, hq2.symm⟩
    rw [hq1]
    have := Nat.mod_add_div (k / bk d) (sk d)
    omega

/-- The chunk index of an element of part `p` is below the number of chunks of the part. -/
theorem chunkIdx_lt (d : Inst) (hK : 0 < d.K) {p k : Nat} (hk : k ∈ partSet d p) :
    chunkIdx d k < nChunks d p := by
  have hcw := cw_pos d
  have hcpb := cpb_pos d
  have hsk := sk_pos d
  have hbk := bk_cpb_cw d
  simp only [partSet, Finset.mem_filter, Finset.mem_range] at hk
  obtain ⟨hk, hm⟩ := hk
  have hmlt : k % bk d / cw d < cpb d := by
    apply Nat.div_lt_of_lt_mul
    have := Nat.mod_lt k (bk_pos d)
    rw [mul_comm]; omega
  have hb : k / bk d < blocks d := by
    unfold blocks
    have h1 : (k + bk d) / bk d ≤ (d.K + bk d - 1) / bk d :=
      Nat.div_le_div_right (by omega)
    rw [Nat.add_div_right _ (bk_pos d)] at h1
    omega
  have hu : k / bk d / sk d + 1 ≤ stepsOf d p := by
    unfold stepsOf
    rw [Nat.le_div_iff_mul_le hsk]
    have h1 := Nat.mod_add_div (k / bk d) (sk d)
    rw [hm] at h1
    have h2 : (k / bk d / sk d + 1) * sk d = sk d * (k / bk d / sk d) + sk d := by ring
    omega
  unfold nChunks chunkIdx
  nlinarith

/-- **The parts and chunks partition `[0, K)`**: a sum over all chunks of all parts is the sum over
`k < K`. -/
theorem partition_sum (d : Inst) (hK : 0 < d.K) (f : Nat → ℚ) :
    ∑ p ∈ Finset.range (sk d), ∑ q ∈ Finset.range (nChunks d p), ∑ k ∈ chunkK d p q, f k =
      ∑ k ∈ Finset.range d.K, f k := by
  have hsk := sk_pos d
  calc ∑ p ∈ Finset.range (sk d), ∑ q ∈ Finset.range (nChunks d p), ∑ k ∈ chunkK d p q, f k
      = ∑ p ∈ Finset.range (sk d), ∑ k ∈ partSet d p, f k := by
        refine Finset.sum_congr rfl fun p hp => ?_
        rw [Finset.mem_range] at hp
        calc ∑ q ∈ Finset.range (nChunks d p), ∑ k ∈ chunkK d p q, f k
            = ∑ q ∈ Finset.range (nChunks d p), ∑ k ∈ (partSet d p).filter fun k => chunkIdx d k = q,
                f k := Finset.sum_congr rfl fun q _ => by rw [chunkK_eq d hp]
          _ = ∑ k ∈ partSet d p, f k :=
            Finset.sum_fiberwise_of_maps_to (fun k hk => Finset.mem_range.mpr (chunkIdx_lt d hK hk)) f
    _ = ∑ k ∈ Finset.range d.K, f k :=
        Finset.sum_fiberwise_of_maps_to (g := fun k => k / bk d % sk d)
          (fun k _ => Finset.mem_range.mpr (Nat.mod_lt _ hsk)) f

end Claims.LoraShrink
