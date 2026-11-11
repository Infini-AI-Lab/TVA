import Proof.GlueDot

/-!
# `chunk_delta_h_bf16`: tiles read by natural-number indices

The numeric lemmas take rows and tokens as functions on `ℕ`; a tile's lane is read with `at2`
(the default outside the tile), and a dot lane is stated over `ℕ`-indexed operands.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- Lane `(i, j)` of a tile, by natural-number indices (the default outside the tile). -/
def at2 {m n : Nat} {α : Type} [Inhabited α] (T : Tensor [m, n] α) (i j : Nat) : α :=
  if h : i < m ∧ j < n then T.get2 ⟨i, h.1⟩ ⟨j, h.2⟩ else default

/-- `at2` on a lane of the tile is the lane. -/
theorem at2_eq {m n : Nat} {α : Type} [Inhabited α] (T : Tensor [m, n] α) (i : Fin m) (j : Fin n) :
    at2 T i j = T.get2 i j := by
  simp [at2, i.isLt, j.isLt]

/-- **One lane of a `tl.dot`, over `ℕ`-indexed operands**: `Device.DotBF16` at output element
`(i, n)` with the products `af k · bf k`, `k < 64`. -/
theorem dot_lane_nat {P : Platform} (hP : Device.DotBF16 P) {M N : Nat} {a : Tensor [M, 64] BF16}
    {b : Tensor [64, N] BF16} {acc r : Tensor [M, N] F32} (h : P.dot cfgBF16 a b acc r)
    (i : Fin M) (n : Fin N) (af bf : ℕ → BF16) (haf : ∀ k : Fin 64, af k = a.get2 i k)
    (hbf : ∀ k : Fin 64, bf k = b.get2 k n)
    (hfin : ∀ k, k < 64 → (af k).isFinite = true ∧ (bf k).isFinite = true)
    (hacc : (acc.get2 i n).isFinite = true)
    (hroom : |(acc.get2 i n).toReal| + ∑ k ∈ Finset.range 64, lmB (af k) * lmB (bf k) ≤ 2 ^ 120) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toReal - ((acc.get2 i n).toReal +
          ∑ k ∈ Finset.range 64, (af k).toReal * (bf k).toReal)| ≤
        rD * (|(acc.get2 i n).toReal| + ∑ k ∈ Finset.range 64, lmB (af k) * lmB (bf k)) + aD := by
  have e1 : ∑ k ∈ Finset.range 64, lmB (af k) * lmB (bf k) =
      ∑ k : Fin 64, lmB (a.get2 i k) * lmB (b.get2 k n) := by
    rw [← Fin.sum_univ_eq_sum_range (fun k => lmB (af k) * lmB (bf k)) 64]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [haf k, hbf k]
  have e2 : ∑ k ∈ Finset.range 64, (af k).toReal * (bf k).toReal =
      ∑ k : Fin 64, (a.get2 i k).toReal * (b.get2 k n).toReal := by
    rw [← Fin.sum_univ_eq_sum_range (fun k => (af k).toReal * (bf k).toReal) 64]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [haf k, hbf k]
  rw [e1] at hroom ⊢
  rw [e2]
  exact dot_lane_bound hP h i n (fun k => by rw [← haf k]; exact (hfin k k.isLt).1)
    (fun k => by rw [← hbf k]; exact (hfin k k.isLt).2) hacc hroom

end Claims.ChunkDeltaH
