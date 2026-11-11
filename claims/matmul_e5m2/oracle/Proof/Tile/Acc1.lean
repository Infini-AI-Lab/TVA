import Proof.Acc
import Proof.Room
import Proof.Basic

/-!
# The accuracy of one trip of the `K`-loop, and of the whole loop

The loop adds `2·steps K` chunks of `32` products.  `DotChunkLoss` gives each chunk of one trip
(`K = 64`, `chunks8 64 = 2`) a law on the partial values in the chunk's terms; `trip` converts it
to the claim's `prodV`/`liftV` chunks and extends the partial-value sequence `S`.  `final` runs
`Dot8Chunk.chain_errM` over all the chunks, and `acc_le` is the room the next trip needs.
-/

namespace Claims.MatmulE5m2

open TVA
open scoped BigOperators

/-- The law of chunk `j` of a partial-value sequence `S`, in the claim's terms: for every `Mj` at
least `|S j|` and the chunk's lifted products, one step is within
`min (ρ·Mj) (κ·A j + ρ'·Mj) + σ·|S j + c j| + α` of the exact sum `S j + c j`. -/
def stepOK (p q : Nat → ℚ) (S : Nat → ℚ) (j : Nat) : Prop :=
  ∀ Mj : ℚ, |S j| ≤ Mj → chunkMax q j ≤ Mj →
    |S (j + 1) - (S j + ∑ k : Fin 32, p (32 * j + k))| ≤
      min (Device.rhoChunk8 * Mj)
          (Device.kappaChunk8 * (∑ k : Fin 32, |p (32 * j + k)|) +
            Device.rhoLossChunk8 * Mj) +
        Device.sigmaChunk8 * |S j + ∑ k : Fin 32, p (32 * j + k)| + Device.alphaChunk8

/-- The first `n` chunk laws. -/
def Steps (p q : Nat → ℚ) (S : Nat → ℚ) (n : Nat) : Prop := ∀ j < n, stepOK p q S j

/-- The row values one trip reads: `segR va t k = (va (64·t + k)).toRat`. -/
def segR (v : Nat → F8E5) (t : Nat) (k : Fin 64) : ℚ := (v (64 * t + k.val)).toRat

/-- The index `64·t + 32·j + r` of a trip's segment lies below `K`. -/
theorem lt_of_seg {K : Nat} (hK64 : 64 ∣ K) {t : Nat} (ht : t < steps K)
    {j r : Nat} (hj : j < 2) (hr : r < 32) : 64 * t + 32 * j + r < K := by
  obtain ⟨m, rfl⟩ := hK64
  have hstep : steps (64 * m) = m := by
    rw [steps_eq (dvd_mul_right 64 m)]
    omega
  rw [hstep] at ht
  omega

/-- The claim's product at the index of chunk `2·t + j` is the segment's product. -/
private theorem seg_val {K : Nat} (hK64 : 64 ∣ K) {va vb : Nat → F8E5} {t j : Nat}
    (ht : t < steps K) (hj : j < 2) {r : Nat} (hr : r < 32) :
    prodV va vb K (32 * (2 * t + j) + r) =
      segR va t ⟨32 * j + r, by omega⟩ * segR vb t ⟨32 * j + r, by omega⟩ := by
  have hlt : 32 * (2 * t + j) + r < K := by
    have h := lt_of_seg hK64 ht hj hr
    omega
  have hidx : 32 * (2 * t + j) + r = 64 * t + (32 * j + r) := by omega
  unfold prodV segR
  simp only [hidx] at hlt ⊢
  simp only [hlt, ite_true]

/-- The library's chunk `j` of a `64`-element segment is the sum over the `32` indices
`32·j + r`. -/
private theorem sum_chunk8_64 {j : Nat} (hj : j < 2) (f : Fin 64 → ℚ) :
    ∑ k ∈ Device.chunk8 64 j, f k =
      ∑ r : Fin 32, f ⟨32 * j + (r : Nat), by have := r.isLt; omega⟩ := by
  unfold Device.chunk8
  refine Finset.sum_bij' (fun a _ => ⟨a.val % 32, Nat.mod_lt _ (by norm_num)⟩)
    (fun b _ => ⟨32 * j + b.val, by have := b.isLt; omega⟩) ?_ ?_ ?_ ?_ ?_
  · intro a ha; exact Finset.mem_univ _
  · intro b hb
    refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
    show (32 * j + b.val) / 32 = j
    have hb32 : b.val < 32 := b.isLt
    rw [Nat.add_comm, Nat.add_mul_div_left b.val j (by norm_num), Nat.div_eq_of_lt hb32,
      Nat.zero_add]
  · intro a ha
    apply Fin.ext
    show 32 * j + (a.val % 32) = a.val
    have hdiv : a.val / 32 = j := (Finset.mem_filter.mp ha).2
    have h := Nat.div_add_mod a.val 32
    rw [hdiv] at h
    exact h
  · intro b hb
    apply Fin.ext
    show (32 * j + b.val) % 32 = b.val
    rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt b.isLt]
  · intro a ha
    have hdiv : a.val / 32 = j := (Finset.mem_filter.mp ha).2
    have h := Nat.div_add_mod a.val 32
    rw [hdiv] at h
    exact congrArg f (Fin.ext h.symm)

/-- The library's chunk sums of a trip's segment are the claim's `prodV` chunks. -/
theorem chunkSum8_seg {K : Nat} (hK64 : 64 ∣ K) {va vb : Nat → F8E5} {t j : Nat}
    (ht : t < steps K) (hj : j < 2) :
    Device.chunkSum8 (segR va t) (segR vb t) j =
      ∑ r : Fin 32, prodV va vb K (32 * (2 * t + j) + r) := by
  unfold Device.chunkSum8
  rw [sum_chunk8_64 hj (fun k => segR va t k * segR vb t k)]
  refine Finset.sum_congr rfl (fun r _ => ?_)
  exact (seg_val hK64 ht hj r.isLt).symm

/-- The library's chunk absolute sums of a trip's segment are the claim's chunks. -/
theorem chunkAbs8_seg {K : Nat} (hK64 : 64 ∣ K) {va vb : Nat → F8E5} {t j : Nat}
    (ht : t < steps K) (hj : j < 2) :
    Device.chunkAbs8 (segR va t) (segR vb t) j =
      ∑ r : Fin 32, |prodV va vb K (32 * (2 * t + j) + r)| := by
  unfold Device.chunkAbs8
  rw [sum_chunk8_64 hj (fun k => |segR va t k * segR vb t k|)]
  refine Finset.sum_congr rfl (fun r _ => ?_)
  rw [seg_val hK64 ht hj r.isLt]

/-- A trip's lifted products are the claim's `liftV` values: the chunk `r` of the claim's chunk
`2·t + j` is the element `32·j + r` of the trip's segment. -/
theorem liftV_seg {K : Nat} (hK64 : 64 ∣ K) {va vb : Nat → F8E5} {t j : Nat}
    (ht : t < steps K) (hj : j < 2) (r : Fin 32) :
    liftV va vb K (32 * (2 * t + j) + r) =
      Device.liftMag .f8e5 ((va (64 * t + 32 * j + r.val)).toRat) *
        Device.liftMag .f8e5 ((vb (64 * t + 32 * j + r.val)).toRat) := by
  have hlt : 32 * (2 * t + j) + (r : Nat) < K := by
    have h := lt_of_seg hK64 ht hj r.isLt
    omega
  have hidx : 32 * (2 * t + j) + (r : Nat) = 64 * t + 32 * j + (r : Nat) := by omega
  unfold liftV
  simp only [hidx] at hlt ⊢
  simp only [hlt, ite_true]

/-- **One trip.**  If the partial values `S` obey the chunk laws up to chunk `2·t`, and one trip
(`tl.dot` of `64` products per output element, whose chunk laws are `hs`) produces the new
accumulator value `s 2`, then `S` extends to the partial values `S'` obeying the chunk laws up to
chunk `2·(t+1)`. -/
theorem trip {K : Nat} (hK : K ≤ 2 ^ 11) (hK64 : 64 ∣ K) {va vb : Nat → F8E5}
    {t : Nat} (ht : t < steps K) {S : Nat → ℚ}
    (hS : Steps (prodV va vb K) (liftV va vb K) S (2 * t))
    {s : Nat → ℚ} (hs0 : s 0 = S (2 * t))
    (hs : ∀ j' < 2, ∀ Mj : ℚ, |s j'| ≤ Mj →
      (∀ k ∈ Device.chunk8 64 j',
        Device.liftMag .f8e5 (segR va t k) * Device.liftMag .f8e5 (segR vb t k) ≤ Mj) →
      |s (j' + 1) - (s j' + Device.chunkSum8 (segR va t) (segR vb t) j')| ≤
        min (Device.rhoChunk8 * Mj)
            (Device.kappaChunk8 * Device.chunkAbs8 (segR va t) (segR vb t) j' +
              Device.rhoLossChunk8 * Mj) +
          Device.sigmaChunk8 * |s j' + Device.chunkSum8 (segR va t) (segR vb t) j'| +
          Device.alphaChunk8) :
    ∃ S' : Nat → ℚ, (∀ j ≤ 2 * t, S' j = S j) ∧ S' (2 * (t + 1)) = s 2 ∧
      Steps (prodV va vb K) (liftV va vb K) S' (2 * (t + 1)) := by
  have _ := hK
  refine ⟨fun j => if j ≤ 2 * t then S j else s (j - 2 * t), ?_, ?_, ?_⟩
  · intro j hj; simp [hj]
  · have h : ¬ (2 * (t + 1) ≤ 2 * t) := by omega
    simp only [h, ite_false]
    congr 1
    omega
  · intro j hj
    by_cases hlt : j < 2 * t
    · have h1 : (if j ≤ 2 * t then S j else s (j - 2 * t)) = S j := by simp only [show j ≤ 2 * t by omega, ite_true]
      have h2 : (if j + 1 ≤ 2 * t then S (j + 1) else s (j + 1 - 2 * t)) = S (j + 1) := by
        simp only [show j + 1 ≤ 2 * t by omega, ite_true]
      unfold stepOK
      intro Mj hMj hqMj
      dsimp only
      rw [h1, h2]
      exact hS j hlt Mj (by simpa [h1] using hMj) hqMj
    · have hj2 : j - 2 * t < 2 := by omega
      obtain ⟨j', rfl⟩ : ∃ j', j = 2 * t + j' := ⟨j - 2 * t, by omega⟩
      have hj' : j' < 2 := by omega
      have h1 : (if 2 * t + j' ≤ 2 * t then S (2 * t + j') else s (2 * t + j' - 2 * t)) =
          s j' := by
        have hj'' : j' = 0 ∨ j' = 1 := by omega
        rcases hj'' with rfl | rfl
        · simp only [show 2 * t + 0 ≤ 2 * t by omega, ite_true]
          simpa using hs0.symm
        · simp only [show ¬ 2 * t + 1 ≤ 2 * t by omega, ite_false]
          congr 1
          omega
      have h2 : (if 2 * t + j' + 1 ≤ 2 * t then S (2 * t + j' + 1)
          else s (2 * t + j' + 1 - 2 * t)) = s (j' + 1) := by
        simp only [show ¬ 2 * t + j' + 1 ≤ 2 * t by omega, ite_false]
        congr 1
        omega
      unfold stepOK
      intro Mj hMj hqMj
      dsimp only
      rw [h1, h2]
      have hsj : |s j'| ≤ Mj := by
        have hj'' : j' = 0 ∨ j' = 1 := by omega
        rcases hj'' with rfl | rfl
        · simpa [hs0] using hMj
        · simpa using hMj
      have hchunk : ∀ k ∈ Device.chunk8 64 j',
          Device.liftMag .f8e5 (segR va t k) * Device.liftMag .f8e5 (segR vb t k) ≤ Mj := by
        intro k hk
        have hkdiv : k.val / 32 = j' := (Finset.mem_filter.mp hk).2
        have hkval : k.val = 32 * j' + k.val % 32 := by
          have h := Nat.div_add_mod k.val 32
          rw [hkdiv] at h
          omega
        let r : Fin 32 := ⟨k.val % 32, Nat.mod_lt _ (by norm_num)⟩
        have hrv : (r : Nat) = k.val % 32 := rfl
        have hkey : Device.liftMag .f8e5 (segR va t k) *
            Device.liftMag .f8e5 (segR vb t k) =
            liftV va vb K (32 * (2 * t + j') + r) := by
          rw [liftV_seg hK64 ht hj' r]
          have hidx : 64 * t + k.val = 64 * t + 32 * j' + (r : Nat) := by
            rw [hrv]; omega
          simp only [segR, hidx]
        calc Device.liftMag .f8e5 (segR va t k) * Device.liftMag .f8e5 (segR vb t k)
            = liftV va vb K (32 * (2 * t + j') + r) := hkey
          _ ≤ chunkMax (liftV va vb K) (2 * t + j') :=
              le_chunkMax (liftV va vb K) (2 * t + j') r
          _ ≤ Mj := hqMj
      have hstep := hs j' hj' Mj hsj hchunk
      rw [chunkSum8_seg hK64 ht hj', chunkAbs8_seg hK64 ht hj'] at hstep
      exact hstep

/-- **The room for the next trip**: the partial value stays below `2^44`. -/
theorem acc_le {K : Nat} (hK : K ≤ 2 ^ 11) (hK64 : 64 ∣ K) {va vb : Nat → F8E5}
    (ha : ∀ k, k < K → |(va k).toRat| ≤ 57344) (hb : ∀ k, k < K → |(vb k).toRat| ≤ 57344)
    {t : Nat} (ht : t ≤ steps K) {S : Nat → ℚ} (h0 : S 0 = 0)
    (hS : Steps (prodV va vb K) (liftV va vb K) S (2 * t)) :
    |S (2 * t)| ≤ 2 ^ 44 := by
  have _ := hK64
  have hρ : (0:ℚ) ≤ Device.rhoChunk8 := by unfold Device.rhoChunk8; positivity
  have hσ : (0:ℚ) ≤ Device.sigmaChunk8 := by unfold Device.sigmaChunk8; positivity
  have hρ' : (0:ℚ) ≤ Device.rhoLossChunk8 := by unfold Device.rhoLossChunk8; positivity
  have hstep : ∀ j < 2 * t, ∀ Mj : ℚ, |S j| ≤ Mj →
      chunkMax (liftV va vb K) j ≤ Mj →
      |S (j + 1) - (S j + ∑ k : Fin 32, prodV va vb K (32 * j + k))| ≤
        min (Device.rhoChunk8 * Mj)
            (Device.kappaChunk8 * (∑ k : Fin 32, |prodV va vb K (32 * j + k)|) +
              Device.rhoLossChunk8 * Mj) +
          Device.sigmaChunk8 * |S j + ∑ k : Fin 32, prodV va vb K (32 * j + k)| +
          Device.alphaChunk8 := by
    intro j hj Mj h1 h2
    exact hS j hj Mj h1 h2
  have hchain := Dot8Chunk.chain_errM hρ hσ hρ'
    (s := S) (c := fun j => ∑ k : Fin 32, prodV va vb K (32 * j + k))
    (a := fun j => ∑ k : Fin 32, |prodV va vb K (32 * j + k)|)
    (m := fun j => chunkMax (liftV va vb K) j) hstep (2 * t) le_rfl
  rw [h0, psum_chunk, errM_eq_errAt] at hchain
  have hb2 := acc_bound hK ha hb (2 * t) (by omega)
  have h1 : |S (2 * t)| ≤ |S (2 * t) - partSum (prodV va vb K) (2 * t)| +
      |partSum (prodV va vb K) (2 * t)| := by
    have h := abs_add_le (S (2 * t) - partSum (prodV va vb K) (2 * t))
      (partSum (prodV va vb K) (2 * t))
    have he : S (2 * t) - partSum (prodV va vb K) (2 * t) + partSum (prodV va vb K) (2 * t)
        = S (2 * t) := by ring
    rw [he] at h
    exact h
  linarith

/-- **The whole loop**: the final accumulator value is within `tol` of the exact dot product. -/
theorem final {K : Nat} (hK64 : 64 ∣ K) {va vb : Nat → F8E5} {S : Nat → ℚ} (h0 : S 0 = 0)
    (hS : Steps (prodV va vb K) (liftV va vb K) S (2 * steps K)) :
    |S (2 * steps K) - dotSum va vb K| ≤ tol va vb K := by
  have hρ : (0:ℚ) ≤ Device.rhoChunk8 := by unfold Device.rhoChunk8; positivity
  have hσ : (0:ℚ) ≤ Device.sigmaChunk8 := by unfold Device.sigmaChunk8; positivity
  have hρ' : (0:ℚ) ≤ Device.rhoLossChunk8 := by unfold Device.rhoLossChunk8; positivity
  have hstep : ∀ j < 2 * steps K, ∀ Mj : ℚ, |S j| ≤ Mj →
      chunkMax (liftV va vb K) j ≤ Mj →
      |S (j + 1) - (S j + ∑ k : Fin 32, prodV va vb K (32 * j + k))| ≤
        min (Device.rhoChunk8 * Mj)
            (Device.kappaChunk8 * (∑ k : Fin 32, |prodV va vb K (32 * j + k)|) +
              Device.rhoLossChunk8 * Mj) +
          Device.sigmaChunk8 * |S j + ∑ k : Fin 32, prodV va vb K (32 * j + k)| +
          Device.alphaChunk8 := by
    intro j hj Mj h1 h2
    exact hS j hj Mj h1 h2
  have hchain := Dot8Chunk.chain_errM hρ hσ hρ'
    (s := S) (c := fun j => ∑ k : Fin 32, prodV va vb K (32 * j + k))
    (a := fun j => ∑ k : Fin 32, |prodV va vb K (32 * j + k)|)
    (m := fun j => chunkMax (liftV va vb K) j) hstep (2 * steps K) le_rfl
  rw [h0, psum_chunk, errM_eq_errAt] at hchain
  have hKle : K ≤ 32 * (2 * steps K) := by
    have hdiv := Nat.div_add_mod (K + 63) 64
    have hmod : (K + 63) % 64 < 64 := Nat.mod_lt _ (by norm_num)
    have h1 : K ≤ 64 * ((K + 63) / 64) := by omega
    have h2 : 32 * (2 * steps K) = 64 * ((K + 63) / 64) := by unfold steps; ring
    omega
  rw [partSum_dotSum va vb hKle] at hchain
  unfold tol
  exact hchain

end Claims.MatmulE5m2
