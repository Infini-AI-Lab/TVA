import Proof.Defs
import Proof.TripX

/-!
# `lora_shrink_bf16`: the trips of the loop over `K`

The blocks of `BLOCK_K` consecutive `k` below `K` are dealt round-robin to the `SPLIT_K` parts: block
`b` is trip `b / SPLIT_K` of part `b % SPLIT_K`. These are the arithmetic facts about that
distribution that the loop needs.
-/

namespace Claims.LoraShrink

open TVA

/-- The configuration's block size and split are one of the two host choices. -/
theorem bk_sk_cases (d : Inst) : (bk d = 256 ∧ sk d = 64) ∨ (bk d = 32 ∧ sk d = 8) := by
  unfold bk sk; split_ifs <;> simp

/-- The number of trips of the loop, `tl.cdiv(K, BLOCK_K · SPLIT_K)` (line 134). -/
def numIters (d : Inst) : Nat := (d.K + (bk d * sk d - 1)) / (bk d * sk d)

/-- `tl.cdiv` of `tl.cdiv`: the trips of part 0 are the trips of the loop. -/
theorem cdiv_cdiv (K a b : Nat) (ha : 0 < a) (hb : 0 < b) :
    ((K + (a - 1)) / a + (b - 1)) / b = (K + (a * b - 1)) / (a * b) := by
  have h1 : (K + (a - 1)) / a + (b - 1) = (K + (a - 1) + a * (b - 1)) / a := by
    rw [Nat.add_mul_div_left _ _ ha]
  have h2 : K + (a - 1) + a * (b - 1) = K + (a * b - 1) := by
    have : a * b = a * (b - 1) + a := by
      cases b with
      | zero => omega
      | succ b => simp [Nat.mul_succ]
    omega
  rw [h1, h2, Nat.div_div_eq_div_mul]

/-- Part `p`'s trips are among the loop's trips: `stepsOf p ≤ numIters`. -/
theorem stepsOf_le_numIters (d : Inst) (p : Nat) : stepsOf d p ≤ numIters d := by
  have hb := bk_pos d
  have hs := sk_pos d
  have : blocks d = (d.K + (bk d - 1)) / bk d := by
    unfold blocks; congr 1; omega
  unfold stepsOf numIters
  rw [← cdiv_cdiv d.K (bk d) (sk d) hb hs, ← this]
  exact Nat.div_le_div_right (by omega)

/-- Trip `t` of part `p < SPLIT_K` is a block below `blocks`. -/
theorem lt_stepsOf_iff (d : Inst) {p : Nat} (hp : p < sk d) (t : Nat) :
    t < stepsOf d p ↔ p + sk d * t < blocks d := by
  have hs := sk_pos d
  unfold stepsOf
  rw [← Nat.succ_le_iff, Nat.le_div_iff_mul_le hs, Nat.succ_mul, Nat.mul_comm]
  omega

/-- Block `b` starts below `K` iff it is below `blocks`. -/
theorem lt_blocks_iff (d : Inst) (b : Nat) : b < blocks d ↔ b * bk d < d.K := by
  have hb := bk_pos d
  unfold blocks
  rw [Nat.lt_div_iff_mul_lt hb]
  constructor <;> intro h <;> omega

/-- With `EVEN_K`, `K` is `c` loop strides, `numIters` is `c` and there are `SPLIT_K · c` blocks. -/
theorem evenK_decomp {d : Inst} (he : evenK d = true) :
    ∃ c, d.K = bk d * sk d * c ∧ numIters d = c ∧ blocks d = sk d * c := by
  have hb := bk_pos d
  have hs := sk_pos d
  unfold evenK at he
  have hdiv : d.K % (bk d * sk d) = 0 := of_decide_eq_true he
  obtain ⟨c, hc⟩ : ∃ c, d.K = bk d * sk d * c := ⟨d.K / (bk d * sk d), by
    have := Nat.div_add_mod d.K (bk d * sk d); omega⟩
  have hbs := Nat.mul_pos hb hs
  refine ⟨c, hc, ?_, ?_⟩
  · unfold numIters
    rw [hc, show bk d * sk d * c + (bk d * sk d - 1) = (bk d * sk d - 1) + bk d * sk d * c by ring,
      Nat.add_mul_div_left _ _ hbs, Nat.div_eq_of_lt (by omega)]
    simp
  · unfold blocks
    rw [hc, show bk d * sk d * c + bk d - 1 = (bk d - 1) + bk d * (sk d * c) by
      have : bk d * sk d * c = bk d * (sk d * c) := by ring
      omega,
      Nat.add_mul_div_left _ _ hb, Nat.div_eq_of_lt (by omega)]
    simp

/-- With `EVEN_K` every trip of every part is a whole block of `K`. -/
theorem evenK_block {d : Inst} (he : evenK d = true) {p t : Nat} (hp : p < sk d)
    (ht : t < numIters d) : ((p + sk d * t) + 1) * bk d ≤ d.K := by
  obtain ⟨c, hc, hn, -⟩ := evenK_decomp he
  rw [hn] at ht
  rw [hc]
  have : (p + sk d * t + 1) ≤ sk d * c := by nlinarith
  calc (p + sk d * t + 1) * bk d ≤ sk d * c * bk d := Nat.mul_le_mul_right _ this
    _ = bk d * sk d * c := by ring

/-- With `EVEN_K` the loop's trips are part `p`'s. -/
theorem evenK_stepsOf {d : Inst} (he : evenK d = true) {p : Nat} (hp : p < sk d) :
    stepsOf d p = numIters d := by
  obtain ⟨c, -, hn, hbl⟩ := evenK_decomp he
  have hs := sk_pos d
  unfold stepsOf
  rw [hn, hbl, show sk d * c + sk d - 1 - p = (sk d - 1 - p) + sk d * c by omega,
    Nat.add_mul_div_left _ _ hs, Nat.div_eq_of_lt (by omega)]
  simp

end Claims.LoraShrink
