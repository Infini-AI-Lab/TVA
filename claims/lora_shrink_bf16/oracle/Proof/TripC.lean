import Proof.TripZ

/-!
# `lora_shrink_bf16`: a chain of chunk steps with an initial error

The chunk steps of a `tl.dot` (`Device.DotChunk16Loss`) from a partial value that is already
within `e` of the exact partial sum stay within `cChain` of the exact partial sums.
-/

namespace Claims.LoraShrink

open TVA

/-- **A chain of chunk steps.** If `s 0` is within `e` of `P g0`, the exact partial sums grow by
`c n` (`P (g0 + n + 1) = P (g0 + n) + c n` for `n < N`) and every step `j < N` adds `c j` up to
`ρ·Mj + σ·|s j + c j| + α` for each `Mj` above `|s j|` and above the chunk's largest lifted
product (`cLift`), then after `n ≤ N` steps `s n` is within `cChain` of `P (g0 + n)`. -/
theorem cchain_bound (d : Inst) (va vb : Nat → BF16) (p g0 : Nat) (P : Nat → ℚ) (s c : Nat → ℚ)
    (e : ℚ) (N : Nat) (hP : ∀ n, n < N → P (g0 + n + 1) = P (g0 + n) + c n) (h0 : |s 0 - P g0| ≤ e)
    (hstep : ∀ j < N, ∀ Mj : ℚ, |s j| ≤ Mj → cLift d va vb p (g0 + j) ≤ Mj →
      |s (j + 1) - (s j + c j)| ≤
        Device.rhoChunk16 * Mj + Device.sigmaChunk16 * |s j + c j| + Device.alphaChunk16) :
    ∀ n ≤ N, |s n - P (g0 + n)| ≤ cChain d va vb p g0 P e n := by
  have hρ := Dot16Chunk.rhoChunk16_nonneg
  have hσ := Dot16Chunk.sigmaChunk16_nonneg
  intro n
  induction n with
  | zero => intro _; simpa [cChain] using h0
  | succ n ih =>
    intro hn
    have e0 := ih (by omega)
    set E := cChain d va vb p g0 P e n with hE
    set Pn := P (g0 + n) with hPn
    have hE0 : 0 ≤ E := (abs_nonneg _).trans e0
    have hsn : |s n| ≤ |Pn| + E := by
      have := abs_sub_abs_le_abs_sub (s n) Pn
      linarith
    have hP1 : P (g0 + n + 1) = Pn + c n := hP n (by omega)
    have hstep' := hstep n (by omega) (max (|Pn| + E) (cLift d va vb p (g0 + n)))
      (hsn.trans (le_max_left _ _)) (le_max_right _ _)
    have hsc : |s n + c n| ≤ |P (g0 + n + 1)| + E := by
      have h1 : s n + c n = P (g0 + n + 1) + (s n - Pn) := by rw [hP1]; ring
      rw [h1]
      exact (abs_add_le _ _).trans (by linarith)
    have hmax : max (|Pn| + E) (cLift d va vb p (g0 + n)) ≤
        max |Pn| (cLift d va vb p (g0 + n)) + E := by
      apply max_le
      · linarith [le_max_left |Pn| (cLift d va vb p (g0 + n))]
      · linarith [le_max_right |Pn| (cLift d va vb p (g0 + n))]
    have htri : |s (n + 1) - P (g0 + n + 1)| ≤ |s (n + 1) - (s n + c n)| + |s n - Pn| := by
      have h1 : s (n + 1) - P (g0 + n + 1) = (s (n + 1) - (s n + c n)) + (s n - Pn) := by
        rw [hP1]; ring
      rw [h1]; exact abs_add_le _ _
    have h2 := mul_le_mul_of_nonneg_left hmax hρ
    have h3 := mul_le_mul_of_nonneg_left hsc hσ
    show |s (n + 1) - P (g0 + n + 1)| ≤ cStep E Pn (cLift d va vb p (g0 + n)) (P (g0 + n + 1))
    unfold cStep
    nlinarith

end Claims.LoraShrink
