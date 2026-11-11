import Claim
import TVA.Proof

/-!
# The tolerance of the `K` loop

The accumulator of a lane after `t` trips of `acc += tl.dot(a, b)` (line 373) is within
`accErr t` of the exact partial sum of the first `t` trips, and after all trips within `tol`
(`Claim.lean`).

1. `chain_gen`, `fold_trip`, `sep_chain`, `sep_combine`: pure arithmetic on a chain of chunk
   steps of the device fact `Device.DotChunk16Loss`. A folded trip (`tl.dot(a, b, acc)`) is one
   chain from the accumulator; a separate trip (`tl.dot(a, b)` then a `float32` addition) is a
   chain from zero and one addition.
2. `dot_chain`: the device fact on the trip's tiles, in the claim's chunks `t·q + j`
   (`q = BK / 64`), with the room it needs.
3. `trip_step`: one trip, either form, from `accErr t` to `accErr (t + 1)`.
4. `accErr_final`: after the last trip the bound is at most `tol` (`tG^T ≤ 107/100` over at
   most `512` chunks); `preSum_final`: the exact partial sum is then `dotSum`.
-/


namespace Claims.FusedMoe

open TVA

/-- The chunk fact's relative constant on the largest term, `2^-13`. -/
def tRho : ℚ := 2 ^ (-13 : ℤ)
/-- The chunk fact's relative constant on the new value, `2^-20`. -/
def tSig : ℚ := 2 ^ (-20 : ℤ)
/-- The chunk fact's absolute constant, `2^-143`. -/
def tAlp : ℚ := 2 ^ (-143 : ℤ)
/-- The `float32` unit roundoff, `2^-24`. -/
def tU : ℚ := 2 ^ (-24 : ℤ)
/-- The growth of the error through one chunk, `1 + 2^-13 + 2^-20`. -/
def tLam : ℚ := 1 + tRho + tSig

/-- `2^-13 > 0`. -/
theorem tRho_pos : 0 < tRho := by unfold tRho; positivity
/-- `2^-20 > 0`. -/
theorem tSig_pos : 0 < tSig := by unfold tSig; positivity
/-- `2^-143 > 0`. -/
theorem tAlp_pos : 0 < tAlp := by unfold tAlp; positivity
/-- `2^-24 > 0`. -/
theorem tU_pos : 0 < tU := by unfold tU; positivity
/-- `1 <= 1 + 2^-13 + 2^-20`. -/
theorem tLam_ge : 1 ≤ tLam := by
  unfold tLam; linarith [tRho_pos, tSig_pos]
/-- `2^-20 ≤ 2^-13 / 128`. -/
theorem tSig_eq : tSig * 128 = tRho := by unfold tSig tRho; norm_num
/-- `2^-24 * 2048 = 2^-13`. -/
theorem tU_le : tU * 2048 = tRho := by unfold tU tRho; norm_num
/-- **A chain of chunk steps against any exact sequence.** If `s` moves from `s j` to `s (j+1)`
as `R (j+1) - R j` is added, within `ρ·Mj + σ·|s j + c| + α` for every `Mj` at least `|s j|` and
`m j`, then `|s j - R j| ≤ Λ^j (|s 0 - R 0| + Σ_{i<j} (ρ max(|R i|, m i) + σ|R (i+1)| + α))`. -/
theorem chain_gen {ρ σ α : ℚ} (hρ : 0 ≤ ρ) (hσ : 0 ≤ σ) (hα : 0 ≤ α) {s R m : ℕ → ℚ} {n : ℕ}
    (hstep : ∀ j < n, ∀ Mj : ℚ, |s j| ≤ Mj → m j ≤ Mj →
      |s (j + 1) - (s j + (R (j + 1) - R j))| ≤ ρ * Mj + σ * |s j + (R (j + 1) - R j)| + α) :
    ∀ j ≤ n, |s j - R j| ≤ (1 + ρ + σ) ^ j *
      (|s 0 - R 0| + ∑ i ∈ Finset.range j, (ρ * max |R i| (m i) + σ * |R (i + 1)| + α)) := by
  intro j
  induction j with
  | zero => intro _; simp
  | succ j ih =>
    intro hj
    have e := ih (by omega)
    set f := |s j - R j| with hf
    have hf0 : 0 ≤ f := abs_nonneg _
    have hmax0 : 0 ≤ max |R j| (m j) := (abs_nonneg _).trans (le_max_left _ _)
    have hsj : |s j| ≤ max |R j| (m j) + f := by
      have := abs_sub_abs_le_abs_sub (s j) (R j)
      have := le_max_left |R j| (m j)
      linarith
    have hst := hstep j (by omega) (max |R j| (m j) + f) hsj (by linarith [le_max_right |R j| (m j)])
    have hsc : |s j + (R (j + 1) - R j)| ≤ |R (j + 1)| + f := by
      have h1 : s j + (R (j + 1) - R j) = R (j + 1) + (s j - R j) := by ring
      rw [h1]; exact abs_add_le _ _
    have htri : |s (j + 1) - R (j + 1)| ≤ |s (j + 1) - (s j + (R (j + 1) - R j))| + f := by
      have h1 : s (j + 1) - R (j + 1) = (s (j + 1) - (s j + (R (j + 1) - R j))) + (s j - R j) := by ring
      rw [h1]; exact abs_add_le _ _
    set t := ρ * max |R j| (m j) + σ * |R (j + 1)| + α with ht
    have ht0 : 0 ≤ t := by positivity
    have hnext : |s (j + 1) - R (j + 1)| ≤ (1 + ρ + σ) * f + t := by
      have h3 := mul_le_mul_of_nonneg_left hsc hσ
      nlinarith
    set S := |s 0 - R 0| + ∑ i ∈ Finset.range j, (ρ * max |R i| (m i) + σ * |R (i + 1)| + α) with hS
    have hlam : 1 ≤ 1 + ρ + σ := by linarith
    have hS0 : 0 ≤ S := by
      have : 0 ≤ ∑ i ∈ Finset.range j, (ρ * max |R i| (m i) + σ * |R (i + 1)| + α) :=
        Finset.sum_nonneg fun i _ => by
          have := (abs_nonneg (R i)).trans (le_max_left |R i| (m i))
          positivity
      have := abs_nonneg (s 0 - R 0)
      linarith
    rw [Finset.sum_range_succ, ← add_assoc, ← hS]
    have hp : 1 ≤ (1 + ρ + σ) ^ (j + 1) := one_le_pow₀ hlam
    have e1 : (1 + ρ + σ) * f ≤ (1 + ρ + σ) * ((1 + ρ + σ) ^ j * S) :=
      mul_le_mul_of_nonneg_left e (by linarith)
    have e3 : t ≤ (1 + ρ + σ) ^ (j + 1) * t := le_mul_of_one_le_left ht0 hp
    have e4 : (1 + ρ + σ) ^ (j + 1) * (S + t) =
        (1 + ρ + σ) * ((1 + ρ + σ) ^ j * S) + (1 + ρ + σ) ^ (j + 1) * t := by ring
    show _ ≤ (1 + ρ + σ) ^ (j + 1) * (S + t)
    linarith


/-- The sum of `q` constants. -/
theorem sum_const_range (q : ℕ) (c : ℚ) : ∑ _i ∈ Finset.range q, c = q * c := by
  simp

/-- **A folded trip.** A chain of `q` chunk steps that starts within `e` of `R 0` ends within
`Λ^q (e + (ρ + 2σ) Σ A + q α)` of `R q`, when `A j` bounds `|R j|` and `m j`, and `m j` bounds
the chunk's sum. -/
theorem fold_trip {q : ℕ} {s R m A : ℕ → ℚ} {e : ℚ}
    (hstep : ∀ j < q, ∀ Mj : ℚ, |s j| ≤ Mj → m j ≤ Mj →
      |s (j + 1) - (s j + (R (j + 1) - R j))| ≤
        tRho * Mj + tSig * |s j + (R (j + 1) - R j)| + tAlp)
    (hRA : ∀ j < q, |R j| ≤ A j) (hmA : ∀ j < q, m j ≤ A j)
    (hc : ∀ j < q, |R (j + 1) - R j| ≤ m j) (he : |s 0 - R 0| ≤ e) :
    |s q - R q| ≤ tLam ^ q * (e + (tRho + 2 * tSig) * ∑ j ∈ Finset.range q, A j + q * tAlp) := by
  have h := chain_gen tRho_pos.le tSig_pos.le tAlp_pos.le hstep q le_rfl
  have hsum : ∑ i ∈ Finset.range q, (tRho * max |R i| (m i) + tSig * |R (i + 1)| + tAlp) ≤
      (tRho + 2 * tSig) * ∑ j ∈ Finset.range q, A j + q * tAlp := by
    calc _ ≤ ∑ i ∈ Finset.range q, ((tRho + 2 * tSig) * A i + tAlp) := by
          refine Finset.sum_le_sum fun i hi => ?_
          have hi := Finset.mem_range.mp hi
          have h1 : max |R i| (m i) ≤ A i := max_le (hRA i hi) (hmA i hi)
          have h2 : |R (i + 1)| ≤ 2 * A i := by
            have := abs_add_le (R i) (R (i + 1) - R i)
            have e : R i + (R (i + 1) - R i) = R (i + 1) := by ring
            rw [e] at this
            linarith [hRA i hi, hmA i hi, hc i hi]
          nlinarith [tRho_pos, tSig_pos]
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_const_range]
  have hlam : 0 ≤ tLam ^ q := pow_nonneg (by linarith [tLam_ge]) q
  have : (1 + tRho + tSig) = tLam := rfl
  rw [this] at h
  refine h.trans (mul_le_mul_of_nonneg_left ?_ hlam)
  linarith

/-- The distance of `R j` from `R 0` is at most the chunks' sums' bounds. -/
theorem abs_sub_zero_le {R m : ℕ → ℚ} (hc : ∀ j, |R (j + 1) - R j| ≤ m j) :
    ∀ j, |R j - R 0| ≤ ∑ i ∈ Finset.range j, m i
  | 0 => by simp
  | j + 1 => by
    have ih := abs_sub_zero_le hc j
    have h := abs_add_le (R j - R 0) (R (j + 1) - R j)
    have e : (R j - R 0) + (R (j + 1) - R j) = R (j + 1) - R 0 := by ring
    rw [e] at h
    rw [Finset.sum_range_succ]
    linarith [hc j]

/-- The sums of a separate trip, for `q' + 1 ≤ 4` chunks. -/
theorem sep_sum {q' : ℕ} (hq4 : q' + 1 ≤ 4) {R m A : ℕ → ℚ}
    (hRA : ∀ j < q' + 1, |R j| ≤ A j) (hZ : ∀ j < q' + 1, |R 0| ≤ A j)
    (hmA : ∀ j < q' + 1, m j ≤ A j) (hc : ∀ j < q' + 1, |R (j + 1) - R j| ≤ m j) :
    ∑ i ∈ Finset.range (q' + 1), (tRho * max |R i - R 0| (m i) + tSig * |R (i + 1) - R 0| + tAlp) ≤
      (7 / 4 * tRho + 3 * tSig) * ∑ j ∈ Finset.range (q' + 1), A j +
        ((q' + 1 : ℕ) : ℚ) * tAlp := by
  have hA0 : ∀ j < q' + 1, 0 ≤ A j := fun j hj => (abs_nonneg _).trans (hRA j hj)
  have hZ0 : 0 ≤ |R 0| := abs_nonneg _
  have hqZ : ((q' + 1 : ℕ) : ℚ) * |R 0| ≤ ∑ j ∈ Finset.range (q' + 1), A j := by
    calc ((q' + 1 : ℕ) : ℚ) * |R 0| = ∑ _j ∈ Finset.range (q' + 1), |R 0| := by simp
      _ ≤ _ := Finset.sum_le_sum fun j hj => hZ j (Finset.mem_range.mp hj)
  have hQ : ∀ j < q' + 1, |R j - R 0| ≤ A j + |R 0| := fun j hj => by
    have := abs_sub (R j) (R 0)
    linarith [hRA j hj]
  have hterm1 : ∀ i < q' + 1, max |R i - R 0| (m i) ≤ A i + |R 0| := fun i hi =>
    max_le (hQ i hi) (by linarith [hmA i hi])
  have hterm2 : ∀ i < q' + 1, |R (i + 1) - R 0| ≤ 3 * A i := fun i hi => by
    have h1 := abs_add_le (R (i + 1) - R i) (R i - R 0)
    have e : (R (i + 1) - R i) + (R i - R 0) = R (i + 1) - R 0 := by ring
    rw [e] at h1
    linarith [hc i hi, hmA i hi, hQ i hi, hZ i hi]
  have hmaxsum : ∑ i ∈ Finset.range (q' + 1), max |R i - R 0| (m i) ≤
      ∑ j ∈ Finset.range (q' + 1), A j + q' * |R 0| := by
    rw [Finset.sum_range_succ', Finset.sum_range_succ' A]
    have h0 : max |R 0 - R 0| (m 0) ≤ A 0 := by
      simp only [sub_self, abs_zero]
      exact max_le (hA0 0 (by omega)) (hmA 0 (by omega))
    have h1 : ∑ i ∈ Finset.range q', max |R (i + 1) - R 0| (m (i + 1)) ≤
        ∑ i ∈ Finset.range q', (A (i + 1) + |R 0|) :=
      Finset.sum_le_sum fun i hi => hterm1 (i + 1) (by have := Finset.mem_range.mp hi; omega)
    rw [Finset.sum_add_distrib, sum_const_range] at h1
    linarith
  have hq'Z : (q' : ℚ) * |R 0| ≤ 3 / 4 * ∑ j ∈ Finset.range (q' + 1), A j := by
    have h4 : (q' : ℚ) ≤ 3 := by
      have : q' ≤ 3 := by omega
      exact_mod_cast this
    push_cast at hqZ
    nlinarith
  have e1 : ∑ i ∈ Finset.range (q' + 1), (tRho * max |R i - R 0| (m i) +
      tSig * |R (i + 1) - R 0| + tAlp) ≤
      ∑ i ∈ Finset.range (q' + 1), (tRho * max |R i - R 0| (m i) + tSig * (3 * A i) + tAlp) :=
    Finset.sum_le_sum fun i hi => by
      have := mul_le_mul_of_nonneg_left (hterm2 i (Finset.mem_range.mp hi)) tSig_pos.le
      linarith
  refine e1.trans ?_
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    sum_const_range]
  have h3 := mul_le_mul_of_nonneg_left hmaxsum tRho_pos.le
  have h4 := mul_le_mul_of_nonneg_left hq'Z tRho_pos.le
  have h5 : tSig * ∑ i ∈ Finset.range (q' + 1), 3 * A i =
      3 * tSig * ∑ j ∈ Finset.range (q' + 1), A j := by
    rw [← Finset.mul_sum]; ring
  rw [h5]
  nlinarith

/-- **A separate trip's dot.** A chain of `q ≤ 4` chunk steps from `0` ends within
`Λ^q ((7/4 ρ + 3σ) Σ A + q α)` of `R q - R 0`, when `A j` bounds `|R j|`, `|R 0|` and `m j`. -/
theorem sep_chain {q : ℕ} (hq1 : 1 ≤ q) (hq4 : q ≤ 4) {s R m A : ℕ → ℚ}
    (hs0 : s 0 = 0)
    (hstep : ∀ j < q, ∀ Mj : ℚ, |s j| ≤ Mj → m j ≤ Mj →
      |s (j + 1) - (s j + (R (j + 1) - R j))| ≤
        tRho * Mj + tSig * |s j + (R (j + 1) - R j)| + tAlp)
    (hRA : ∀ j < q, |R j| ≤ A j) (hZ : ∀ j < q, |R 0| ≤ A j) (hmA : ∀ j < q, m j ≤ A j)
    (hc : ∀ j < q, |R (j + 1) - R j| ≤ m j) :
    |s q - (R q - R 0)| ≤
      tLam ^ q * ((7 / 4 * tRho + 3 * tSig) * ∑ j ∈ Finset.range q, A j + q * tAlp) := by
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by omega⟩
  have hstep' : ∀ j < q' + 1, ∀ Mj : ℚ, |s j| ≤ Mj → m j ≤ Mj →
      |s (j + 1) - (s j + ((R (j + 1) - R 0) - (R j - R 0)))| ≤
        tRho * Mj + tSig * |s j + ((R (j + 1) - R 0) - (R j - R 0))| + tAlp := by
    intro j hj Mj h1 h2
    have e : (R (j + 1) - R 0) - (R j - R 0) = R (j + 1) - R j := by ring
    rw [e]
    exact hstep j hj Mj h1 h2
  have h := chain_gen (R := fun i => R i - R 0) tRho_pos.le tSig_pos.le tAlp_pos.le hstep'
    (q' + 1) le_rfl
  simp only [hs0, sub_self, abs_zero, zero_add] at h
  have hsum := sep_sum hq4 hRA hZ hmA hc
  have hlam : 0 ≤ tLam ^ (q' + 1) := pow_nonneg (by linarith [tLam_ge]) _
  have hl : (1 + tRho + tSig) = tLam := rfl
  rw [hl] at h
  exact h.trans (mul_le_mul_of_nonneg_left hsum hlam)


/-- The error of the trip's final `float32` addition, `4η`. -/
def tEta4 : ℚ := 4 * 2 ^ (-150 : ℤ)

/-- **A separate trip's addition.** The dot `d` is within `L·W` of `R q - R 0`, the accumulator `X`
within `e` of `R 0`, and the sum `r` of the two within `3u(|d| + |X|) + 4η` of `d + X`: then `r` is
within `(1 + 3u)·L·(e + (9/5)ρ·S + q·(3/2)α)` of `R q`, for `W = (7/4ρ + 3σ) S + q α`,
`|R q - R 0| ≤ S`, `|R 0| ≤ S`. -/
theorem sep_combine {q : ℕ} (hq : 1 ≤ q) {L S e r d X Rq R0 : ℚ} (hL : 1 ≤ L) (hS : 0 ≤ S)
    (he : 0 ≤ e)
    (hd : |d - (Rq - R0)| ≤ L * ((7 / 4 * tRho + 3 * tSig) * S + q * tAlp))
    (hQ : |Rq - R0| ≤ S) (hX : |X - R0| ≤ e) (hZ : |R0| ≤ S)
    (hadd : |r - (d + X)| ≤ 3 * tU * (|d| + |X|) + tEta4) :
    |r - Rq| ≤ (1 + 3 * tU) * L * (e + 9 / 5 * tRho * S + q * (3 / 2 * tAlp)) := by
  have hu := tU_pos
  have hr := tRho_pos
  have hsg := tSig_pos
  have ha := tAlp_pos
  have hq' : (1 : ℚ) ≤ q := by exact_mod_cast hq
  have hη : tEta4 ≤ tAlp / 2 := by
    unfold tEta4 tAlp; norm_num
  have h1 : |r - Rq| ≤ |r - (d + X)| + |d - (Rq - R0)| + |X - R0| := by
    have e1 : r - Rq = (r - (d + X)) + (d - (Rq - R0)) + (X - R0) := by ring
    rw [e1]
    have h2 := abs_add_le ((r - (d + X)) + (d - (Rq - R0))) (X - R0)
    have h3 := abs_add_le (r - (d + X)) (d - (Rq - R0))
    linarith
  have hd' : |d| ≤ S + |d - (Rq - R0)| := by
    have := abs_sub_abs_le_abs_sub d (Rq - R0)
    linarith
  have hX' : |X| ≤ S + e := by
    have := abs_sub_abs_le_abs_sub X R0
    linarith
  set H0 := |d - (Rq - R0)|
  set c := (1 + 3 * tU) * L with hc
  have hc1 : 1 ≤ c := by
    have : 0 ≤ 3 * tU := by positivity
    nlinarith
  have hA : |r - Rq| ≤ (1 + 3 * tU) * (H0 + e) + 6 * tU * S + tEta4 := by
    have := mul_le_mul_of_nonneg_left (add_le_add hd' hX') (by positivity : (0 : ℚ) ≤ 3 * tU)
    nlinarith
  set w1 := 7 / 4 * tRho + 3 * tSig with hw1
  have hH : (1 + 3 * tU) * (H0 + e) ≤ c * (w1 * S + q * tAlp) + (1 + 3 * tU) * e := by
    have := mul_le_mul_of_nonneg_left hd (by positivity : (0 : ℚ) ≤ 1 + 3 * tU)
    rw [hc]; nlinarith
  have hUe : (1 + 3 * tU) * e ≤ c * e := by
    rw [hc]; nlinarith [mul_nonneg (by positivity : (0 : ℚ) ≤ 1 + 3 * tU) he]
  have hcoef : 6 * tU * S + w1 * S ≤ 9 / 5 * tRho * S := by
    have e1 : tSig = tRho / 128 := by have := tSig_eq; linarith
    have e2 : tU = tRho / 2048 := by have := tU_le; linarith
    rw [hw1, e1, e2]
    nlinarith [mul_nonneg hr.le hS]
  have hk : 0 ≤ (c - 1) * ((9 / 5 * tRho - w1) * S) := by
    have : 0 ≤ (9 / 5 * tRho - w1) * S := by
      have e1 : tSig = tRho / 128 := by have := tSig_eq; linarith
      rw [hw1, e1]; nlinarith [mul_nonneg hr.le hS]
    exact mul_nonneg (by linarith) this
  have hk2 : 0 ≤ (c - 1) * (q * tAlp / 2) := mul_nonneg (by linarith) (by positivity)
  have hqa : tAlp / 2 ≤ q * tAlp / 2 := by nlinarith
  nlinarith

/-! ## The loop's error sequence -/

/-- A trip's growth factor: `(1 + 3u)·Λ^q`, `q = BK / 64` chunks of `Λ` and the final addition. -/
def tG (BK : Nat) : ℚ := (1 + 3 * tU) * tLam ^ (BK / 64)

/-- The magnitude `A_g` of chunk `g` in `pathMag`: the largest of the exact partial sum before the
chunk, before its trip, and the chunk's lifted magnitude. -/
def Ag (va vb : Nat → BF16) (BK K g : Nat) : ℚ :=
  max (max |preSum va vb K g| |preSum va vb K (g - g % (BK / 64))|) (chunkLift va vb K g)

/-- The sum of `A_g` over the chunks of trip `t`. -/
def tripMag (va vb : Nat → BF16) (BK K t : Nat) : ℚ :=
  ∑ j ∈ Finset.range (BK / 64), Ag va vb BK K (t * (BK / 64) + j)

/-- **The accumulator's error bound after `t` trips**: `accErr 0 = 0` and each trip multiplies by
`tG` the error so far plus `(9/5)·2^-13` of the trip's `A_g` plus `(3/2)·2^-143` per chunk. -/
def accErr (BK K : Nat) (va vb : Nat → BF16) : Nat → ℚ
  | 0 => 0
  | t + 1 => tG BK * (accErr BK K va vb t + 9 / 5 * tRho * tripMag va vb BK K t +
      ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp))

/-- No error before the first trip. -/
theorem accErr_zero (BK K : Nat) (va vb : Nat → BF16) : accErr BK K va vb 0 = 0 := rfl

/-- The magnitude `Ag` of a chunk is nonnegative. -/
theorem Ag_nonneg (va vb : Nat → BF16) (BK K g : Nat) : 0 ≤ Ag va vb BK K g :=
  (abs_nonneg _).trans ((le_max_left _ _).trans (le_max_left _ _))

/-- The magnitude of a trip is nonnegative. -/
theorem tripMag_nonneg (va vb : Nat → BF16) (BK K t : Nat) : 0 ≤ tripMag va vb BK K t :=
  Finset.sum_nonneg fun _ _ => Ag_nonneg _ _ _ _ _

/-- The growth of one trip is at least `1`. -/
theorem tG_ge (BK : Nat) : 1 ≤ tG BK := by
  unfold tG
  have h1 : 1 ≤ tLam ^ (BK / 64) := one_le_pow₀ tLam_ge
  have := tU_pos
  nlinarith

/-- The error bound is nonnegative. -/
theorem accErr_nonneg (BK K : Nat) (va vb : Nat → BF16) : ∀ t, 0 ≤ accErr BK K va vb t
  | 0 => le_rfl
  | t + 1 => by
    have ih := accErr_nonneg BK K va vb t
    have := tG_ge BK
    have := tripMag_nonneg va vb BK K t
    have := tRho_pos
    have := tAlp_pos
    simp only [accErr]
    positivity

/-- The closed form of `accErr`: `tG^t · Σ_{s<t} ((9/5)·2^-13·tripMag s + q·(3/2)·2^-143)`. -/
theorem accErr_le_closed (BK K : Nat) (va vb : Nat → BF16) : ∀ t, accErr BK K va vb t ≤
    tG BK ^ t * ∑ s ∈ Finset.range t,
      (9 / 5 * tRho * tripMag va vb BK K s + ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp))
  | 0 => by simp [accErr]
  | t + 1 => by
    have ih := accErr_le_closed BK K va vb t
    have hG := tG_ge BK
    have hterm : 0 ≤ 9 / 5 * tRho * tripMag va vb BK K t + ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp) := by
      have := tripMag_nonneg va vb BK K t
      have := tRho_pos
      have := tAlp_pos
      positivity
    have hp : 1 ≤ tG BK ^ t := one_le_pow₀ hG
    have e3 : (9 / 5 * tRho * tripMag va vb BK K t + ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) ≤
        tG BK ^ t * (9 / 5 * tRho * tripMag va vb BK K t + ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) :=
      le_mul_of_one_le_left hterm hp
    rw [Finset.sum_range_succ, pow_succ]
    simp only [accErr]
    have e1 := mul_le_mul_of_nonneg_left ih (by linarith : (0 : ℚ) ≤ tG BK)
    have e4 := mul_le_mul_of_nonneg_left e3 (by linarith : (0 : ℚ) ≤ tG BK)
    nlinarith

/-- `(1 + x)^n ≤ 1 / (1 - b)` for `n·x ≤ b < 1`. -/
theorem pow_le_inv {x b : ℚ} (hx : 0 ≤ x) (n : Nat) (h : n * x ≤ b) (hb : b < 1) :
    (1 + x) ^ n ≤ 1 / (1 - b) := by
  have hn : (n : ℚ) * x ≤ 1 := by linarith
  have h1 := DotLoop.pow_mul_le hx n hn
  rw [le_div_iff₀ (by linarith)]
  have hp : 0 ≤ (1 + x) ^ n := by positivity
  nlinarith

/-- Fewer than `512` chunks in all: `steps · (BK/64) ≤ 512`, for `K ≤ 2^15`. -/
theorem steps_mul_le {BK K : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256) (hK15 : K ≤ 2 ^ 15) :
    steps BK K * (BK / 64) ≤ 512 := by
  unfold steps
  rcases hBK with rfl | rfl | rfl <;> omega

/-- A launch has at most `512` trips. -/
theorem steps_le {BK K : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256) (hK15 : K ≤ 2 ^ 15) :
    steps BK K ≤ 512 := by
  have := steps_mul_le hBK hK15
  rcases hBK with rfl | rfl | rfl <;> omega

/-- The trips' growth over at most `512` chunks: `tG^T ≤ 107/100`. -/
theorem tG_pow_le {BK T : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256)
    (hT : T * (BK / 64) ≤ 512) : tG BK ^ T ≤ 107 / 100 := by
  have hq : 1 ≤ BK / 64 := by rcases hBK with rfl | rfl | rfl <;> omega
  have hT512 : T ≤ 512 := by nlinarith
  have h1 : (1 + 3 * tU) ^ T ≤ 1 / (1 - 512 * (3 * tU)) := by
    refine pow_le_inv (by have := tU_pos; positivity) T ?_ ?_
    · have : (T : ℚ) ≤ 512 := by exact_mod_cast hT512
      nlinarith [tU_pos]
    · unfold tU; norm_num
  have hl : tLam = 1 + (tRho + tSig) := by unfold tLam; ring
  have h2 : tLam ^ (T * (BK / 64)) ≤ 1 / (1 - 512 * (tRho + tSig)) := by
    rw [hl]
    refine pow_le_inv (add_pos tRho_pos tSig_pos).le _ ?_ ?_
    · have : ((T * (BK / 64) : Nat) : ℚ) ≤ 512 := by exact_mod_cast hT
      nlinarith [tRho_pos, tSig_pos]
    · unfold tRho tSig; norm_num
  have e : tG BK ^ T = (1 + 3 * tU) ^ T * tLam ^ (T * (BK / 64)) := by
    unfold tG; rw [mul_pow, ← pow_mul, mul_comm (BK / 64) T]
  rw [e]
  have hA : 0 ≤ (1 + 3 * tU) ^ T := by have := tU_pos; positivity
  have hb : (1 : ℚ) / (1 - 512 * (3 * tU)) * (1 / (1 - 512 * (tRho + tSig))) ≤ 107 / 100 := by
    unfold tU tRho tSig; norm_num
  calc _ ≤ 1 / (1 - 512 * (3 * tU)) * (1 / (1 - 512 * (tRho + tSig))) :=
        mul_le_mul h1 h2 (pow_nonneg (by linarith [tLam_ge]) _) (by unfold tU; norm_num)
    _ ≤ _ := hb

/-! ## Facts about the claim's sequences -/

/-- One more chunk adds its sum. -/
theorem preSum_succ (va vb : Nat → BF16) (K g : Nat) :
    preSum va vb K (g + 1) = preSum va vb K g + chunkSum va vb K g := by
  unfold preSum; rw [Finset.sum_range_succ]

/-- A lifted product is nonnegative. -/
theorem liftAt_nonneg (va vb : Nat → BF16) (K k : Nat) : 0 ≤ liftAt va vb K k := by
  unfold liftAt magB
  split_ifs
  · exact mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  · exact le_rfl

/-- A product is at most its lifted product. -/
theorem abs_prodAt_le (va vb : Nat → BF16) (K k : Nat) : |prodAt va vb K k| ≤ liftAt va vb K k := by
  unfold prodAt liftAt magB
  split_ifs
  · exact Device.abs_mul_le_liftMag _ _ _ _
  · simp

/-- A chunk's sum is at most its lifted magnitude. -/
theorem abs_chunkSum_le (va vb : Nat → BF16) (K g : Nat) :
    |chunkSum va vb K g| ≤ chunkLift va vb K g :=
  (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun _ _ => abs_prodAt_le _ _ _ _)

/-- The partial sum before chunk `g` is the sum of the first `64·g` products. -/
theorem preSum_eq (va vb : Nat → BF16) (K g : Nat) :
    preSum va vb K g = ∑ k ∈ Finset.range (g * 64), prodAt va vb K k := by
  unfold preSum chunkSum
  rw [← DotLoop.sum_chunks (prodAt va vb K) 64 g]
  refine Finset.sum_congr rfl fun h _ => Finset.sum_congr rfl fun i _ => by rw [mul_comm]

/-- Every partial sum of lifted products is at most `dotMag`. -/
theorem sum_liftAt_le (va vb : Nat → BF16) (K n : Nat) :
    ∑ k ∈ Finset.range n, liftAt va vb K k ≤ dotMag va vb K := by
  have e : ∑ k ∈ Finset.range n, liftAt va vb K k =
      ∑ k ∈ (Finset.range n).filter (· < K), magB (va k).toRat * magB (vb k).toRat := by
    rw [Finset.sum_filter]
    exact Finset.sum_congr rfl fun k _ => by unfold liftAt; rfl
  rw [e]
  unfold dotMag
  refine Finset.sum_le_sum_of_subset_of_nonneg ?_ fun k _ _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  intro k hk
  simp only [Finset.mem_filter, Finset.mem_range] at hk ⊢
  exact hk.2

/-- A partial sum is at most `dotMag` in magnitude. -/
theorem abs_preSum_le (va vb : Nat → BF16) (K g : Nat) : |preSum va vb K g| ≤ dotMag va vb K := by
  rw [preSum_eq]
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    ((Finset.sum_le_sum fun _ _ => abs_prodAt_le _ _ _ _).trans (sum_liftAt_le _ _ _ _))

/-- A chunk's lifted magnitude is at most `dotMag`. -/
theorem chunkLift_le (va vb : Nat → BF16) (K g : Nat) : chunkLift va vb K g ≤ dotMag va vb K := by
  refine le_trans ?_ (sum_liftAt_le va vb K ((g + 1) * 64))
  have h := DotLoop.sum_chunks (liftAt va vb K) 64 (g + 1)
  rw [← h, Finset.sum_range_succ]
  have : 0 ≤ ∑ t ∈ Finset.range g, ∑ k ∈ Finset.range 64, liftAt va vb K (t * 64 + k) :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => liftAt_nonneg _ _ _ _
  unfold chunkLift
  have e : ∑ i ∈ Finset.range 64, liftAt va vb K (64 * g + i) =
      ∑ k ∈ Finset.range 64, liftAt va vb K (g * 64 + k) :=
    Finset.sum_congr rfl fun i _ => by rw [mul_comm]
  linarith

/-- `A_g` is at most `dotMag`. -/
theorem Ag_le (va vb : Nat → BF16) (BK K g : Nat) : Ag va vb BK K g ≤ dotMag va vb K :=
  max_le (max_le (abs_preSum_le _ _ _ _) (abs_preSum_le _ _ _ _)) (chunkLift_le _ _ _ _)

/-- The loop's chunks cover `K`: `64 · (steps · (BK/64)) ≥ K`. -/
theorem steps_cover {BK K : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256) :
    K ≤ steps BK K * (BK / 64) * 64 := by
  unfold steps
  rcases hBK with rfl | rfl | rfl <;> omega

/-- **The sum of the loop's chunks is the dot product.** -/
theorem preSum_final {BK K : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256) (va vb : Nat → BF16) :
    preSum va vb K (steps BK K * (BK / 64)) = dotSum va vb K := by
  rw [preSum_eq]
  unfold dotSum
  have hc := steps_cover (K := K) hBK
  have e : ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat =
      ∑ k ∈ Finset.range K, prodAt va vb K k :=
    Finset.sum_congr rfl fun k hk => by
      have := Finset.mem_range.mp hk
      simp [prodAt, this]
  rw [e]
  symm
  refine Finset.sum_subset (Finset.range_mono hc) fun k hk hk' => ?_
  have : ¬ k < K := by simpa using hk'
  simp [prodAt, this]

/-- The sum over the trips' `A_g` is `pathMag`. -/
theorem pathMag_eq_sum (va vb : Nat → BF16) (BK K : Nat) :
    pathMag va vb BK K = ∑ g ∈ Finset.range (steps BK K * (BK / 64)), Ag va vb BK K g := rfl

/-- The trips' magnitudes add up to the chunks' magnitudes. -/
theorem sum_tripMag (va vb : Nat → BF16) (BK K T : Nat) :
    ∑ s ∈ Finset.range T, tripMag va vb BK K s =
      ∑ g ∈ Finset.range (T * (BK / 64)), Ag va vb BK K g :=
  DotLoop.sum_chunks (Ag va vb BK K) (BK / 64) T

/-- **The final bound.** The error after all trips is at most the claim's tolerance. -/
theorem accErr_final {BK K : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256) (hK15 : K ≤ 2 ^ 15) (va vb : Nat → BF16) :
    accErr BK K va vb (steps BK K) ≤ tol BK K va vb := by
  have hcl := accErr_le_closed BK K va vb (steps BK K)
  have hG := tG_pow_le hBK (steps_mul_le hBK hK15)
  have hsum : ∑ s ∈ Finset.range (steps BK K),
      (9 / 5 * tRho * tripMag va vb BK K s + ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) =
      9 / 5 * tRho * pathMag va vb BK K + (steps BK K * (BK / 64) : Nat) * (3 / 2 * tAlp) := by
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_tripMag, sum_const_range]
    rw [pathMag_eq_sum]; push_cast; ring
  rw [hsum] at hcl
  have hP : 0 ≤ pathMag va vb BK K :=
    Finset.sum_nonneg fun _ _ => Ag_nonneg _ _ _ _ _
  have hT : ((steps BK K * (BK / 64) : Nat) : ℚ) ≤ 512 := by exact_mod_cast steps_mul_le hBK hK15
  have hnn : 0 ≤ 9 / 5 * tRho * pathMag va vb BK K +
      (steps BK K * (BK / 64) : Nat) * (3 / 2 * tAlp) := by
    have := tRho_pos; have := tAlp_pos; positivity
  unfold tol
  have hr : (1 : ℚ) / 2 ^ 12 = 2 ^ (-12 : ℤ) := by norm_num
  have hrho : tRho = 2 ^ (-13 : ℤ) := rfl
  have habs : (107 / 100 : ℚ) * (((steps BK K * (BK / 64) : Nat) : ℚ) * (3 / 2 * tAlp)) ≤
      2 ^ (-133 : ℤ) := by
    have : (107 / 100 : ℚ) * (512 * (3 / 2 * tAlp)) ≤ 2 ^ (-133 : ℤ) := by
      unfold tAlp; norm_num
    have h2 : (((steps BK K * (BK / 64) : Nat) : ℚ) * (3 / 2 * tAlp)) ≤ 512 * (3 / 2 * tAlp) :=
      mul_le_mul_of_nonneg_right hT (by have := tAlp_pos; positivity)
    nlinarith
  have hrel : (107 / 100 : ℚ) * (9 / 5 * tRho) ≤ 2 ^ (-12 : ℤ) := by
    unfold tRho; norm_num
  calc _ ≤ tG BK ^ steps BK K * _ := hcl
    _ ≤ 107 / 100 * (9 / 5 * tRho * pathMag va vb BK K +
        (steps BK K * (BK / 64) : Nat) * (3 / 2 * tAlp)) := mul_le_mul_of_nonneg_right hG hnn
    _ ≤ _ := by nlinarith

/-- The error before the last trip stays below `2^118` when `dotMag ≤ 2^120`. -/
theorem accErr_le_room {BK K : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256) (hK15 : K ≤ 2 ^ 15)
    (va vb : Nat → BF16) (hmag : dotMag va vb K ≤ 2 ^ 120) {t : Nat} (ht : t ≤ steps BK K) :
    accErr BK K va vb t ≤ 2 ^ 118 := by
  have hcl := accErr_le_closed BK K va vb t
  have htq : t * (BK / 64) ≤ 512 := le_trans (Nat.mul_le_mul_right _ ht) (steps_mul_le hBK hK15)
  have hG := tG_pow_le hBK htq
  have hq : (BK / 64 : Nat) ≤ 4 := by rcases hBK with rfl | rfl | rfl <;> omega
  have hTM : ∀ s, tripMag va vb BK K s ≤ ((BK / 64 : Nat) : ℚ) * 2 ^ 120 := fun s => by
    unfold tripMag
    calc _ ≤ ∑ _j ∈ Finset.range (BK / 64), (2 : ℚ) ^ 120 :=
          Finset.sum_le_sum fun j _ => (Ag_le _ _ _ _ _).trans hmag
      _ = _ := by simp
  have hsum : ∑ s ∈ Finset.range t,
      (9 / 5 * tRho * tripMag va vb BK K s + ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) ≤
      ∑ _s ∈ Finset.range t, (9 / 5 * tRho * (((BK / 64 : Nat) : ℚ) * 2 ^ 120) +
        ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) :=
    Finset.sum_le_sum fun s _ => by
      have := mul_le_mul_of_nonneg_left (hTM s) (by have := tRho_pos; positivity : (0 : ℚ) ≤ 9 / 5 * tRho)
      linarith
  rw [sum_const_range] at hsum
  have htq' : (t : ℚ) * ((BK / 64 : Nat) : ℚ) ≤ 512 := by exact_mod_cast htq
  have hq' : (0 : ℚ) ≤ ((BK / 64 : Nat) : ℚ) := Nat.cast_nonneg _
  have hfin : (t : ℚ) * (9 / 5 * tRho * (((BK / 64 : Nat) : ℚ) * 2 ^ 120) +
        ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) ≤ 2 ^ 117 + 1 := by
    have e : (t : ℚ) * (9 / 5 * tRho * (((BK / 64 : Nat) : ℚ) * 2 ^ 120) +
        ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) =
        ((t : ℚ) * ((BK / 64 : Nat) : ℚ)) * (9 / 5 * tRho * 2 ^ 120 + 3 / 2 * tAlp) := by ring
    rw [e]
    have : (512 : ℚ) * (9 / 5 * tRho * 2 ^ 120 + 3 / 2 * tAlp) ≤ 2 ^ 117 + 1 := by
      unfold tRho tAlp; norm_num
    have h0 : 0 ≤ 9 / 5 * tRho * 2 ^ 120 + 3 / 2 * tAlp := by
      have := tRho_pos; have := tAlp_pos; positivity
    nlinarith
  have hnn : 0 ≤ ∑ s ∈ Finset.range t,
      (9 / 5 * tRho * tripMag va vb BK K s + ((BK / 64 : Nat) : ℚ) * (3 / 2 * tAlp)) :=
    Finset.sum_nonneg fun s _ => by
      have := tripMag_nonneg va vb BK K s; have := tRho_pos; have := tAlp_pos; positivity
  calc _ ≤ _ := hcl
    _ ≤ 107 / 100 * (2 ^ 117 + 1) := by
        refine (mul_le_mul hG (hsum.trans hfin) hnn (by norm_num))
    _ ≤ 2 ^ 118 := by norm_num

/-! ## The device's chunks of one `tl.dot` -/

/-- A chunk of a dot of length `64·q`: the sum over the `k` with `k / 64 = j` is the sum of
the `64` consecutive terms from `64·j`. -/
theorem sum_chunk_fin (q j : Nat) (hj : j < q) (F : Nat → ℚ) :
    ∑ k ∈ (Finset.univ : Finset (Fin (64 * q))).filter (fun k => k.val / 64 = j), F k =
      ∑ i ∈ Finset.range 64, F (64 * j + i) := by
  rw [Finset.sum_filter, Fin.sum_univ_eq_sum_range (fun k => if k / 64 = j then F k else 0) (64 * q),
    mul_comm 64 q, ← DotLoop.sum_chunks (fun k => if k / 64 = j then F k else 0) 64 q]
  rw [Finset.sum_eq_single j]
  · refine Finset.sum_congr rfl fun i hi => ?_
    have hi := Finset.mem_range.mp hi
    have : (j * 64 + i) / 64 = j := by omega
    simp only [this, ite_true]
    rw [mul_comm]
  · intro t _ htj
    refine Finset.sum_eq_zero fun i hi => ?_
    have hi := Finset.mem_range.mp hi
    have : (t * 64 + i) / 64 = t := by omega
    simp [this, htj]
  · intro h; exact absurd (Finset.mem_range.mpr hj) h

/-- A lane of the trip's operand tile, with the `other = 0` past `K`: the product of two lanes
is the claim's `prodAt`. -/
theorem lane_prod (va vb : Nat → BF16) (K base k : Nat) :
    (if base + k < K then (va (base + k)).toRat else 0) *
      (if base + k < K then (vb (base + k)).toRat else 0) = prodAt va vb K (base + k) := by
  unfold prodAt
  by_cases h : base + k < K <;> simp [h]

/-- The lifted product of two lanes is the claim's `liftAt`. -/
theorem lane_lift (va vb : Nat → BF16) (K base k : Nat) :
    Device.liftMag .bf16 (if base + k < K then (va (base + k)).toRat else 0) *
      Device.liftMag .bf16 (if base + k < K then (vb (base + k)).toRat else 0) =
        liftAt va vb K (base + k) := by
  unfold liftAt magB
  by_cases h : base + k < K <;> simp [h]

/-- Lifted products of a window of `n` positions add up to at most `dotMag`. -/
theorem sum_window_le (va vb : Nat → BF16) (K base n : Nat) :
    ∑ k ∈ Finset.range n, liftAt va vb K (base + k) ≤ dotMag va vb K := by
  refine le_trans ?_ (sum_liftAt_le va vb K (base + n))
  rw [Finset.sum_range_add]
  have : 0 ≤ ∑ x ∈ Finset.range base, liftAt va vb K x :=
    Finset.sum_nonneg fun _ _ => liftAt_nonneg _ _ _ _
  linarith

/-- `float32`'s largest value is at least `2^123`. -/
theorem two_pow_123_le : (2 : ℚ) ^ 123 ≤ DType.bound .f32 := by
  simp only [DType.bound, Format.binary32_maxVal]; norm_num

/-- **One `tl.dot` of a trip, chunk by chunk.** From the device's chunk fact: for the accumulator
`acc` of value `vc` (`|vc| ≤ 2^121`) and operand tiles whose lanes hold `va`, `vb` of positions
`t·64q + kk` (`0` past `K`), the result has a value `vr` and a chain `s` from `vc` to `vr` whose
steps are the device's, over the claim's chunks `t·q + j`. -/
theorem dot_chain {P : Platform} (hdot : Device.DotBF16Chunk P) {q t K M N : Nat} (hq4 : q ≤ 4) {a : Tensor [M, 64 * q] BF16} {b : Tensor [64 * q, N] BF16}
    {acc r : Tensor [M, N] F32} (hd : P.dot ⟨.bf16, .bf16, .f32, .tf32, 0⟩ a b acc r)
    (i : Fin M) (n : Fin N) (va vb : Nat → BF16)
    (hva : ∀ kk : Fin (64 * q), DType.val? .bf16 (a.get2 i kk) =
      some (if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0))
    (hvb : ∀ kk : Fin (64 * q), DType.val? .bf16 (b.get2 kk n) =
      some (if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0))
    {vc : ℚ} (hvc : DType.val? .f32 (acc.get2 i n) = some vc) (hvc_le : |vc| ≤ 2 ^ 121)
    (hmag : dotMag va vb K ≤ 2 ^ 120) :
    ∃ (s : Nat → ℚ) (vr : ℚ), DType.val? .f32 (r.get2 i n) = some vr ∧ s 0 = vc ∧ s q = vr ∧
      ∀ j < q, ∀ Mj : ℚ, |s j| ≤ Mj → chunkLift va vb K (t * q + j) ≤ Mj →
        |s (j + 1) - (s j + chunkSum va vb K (t * q + j))| ≤
          tRho * Mj + tSig * |s j + chunkSum va vb K (t * q + j)| + tAlp := by
  have hc : Device.chunks16 (64 * q) = q := by unfold Device.chunks16; omega
  have hsum : ∑ k : Fin (64 * q),
      |(if t * (64 * q) + k < K then (va (t * (64 * q) + k)).toRat else 0) *
        (if t * (64 * q) + k < K then (vb (t * (64 * q) + k)).toRat else 0)| ≤ 2 ^ 120 := by
    have e : ∀ k : Fin (64 * q), |(if t * (64 * q) + k < K then (va (t * (64 * q) + k)).toRat else 0) *
        (if t * (64 * q) + k < K then (vb (t * (64 * q) + k)).toRat else 0)| ≤
        liftAt va vb K (t * (64 * q) + k) := fun k => by
      rw [lane_prod]; exact abs_prodAt_le _ _ _ _
    refine le_trans (Finset.sum_le_sum fun k _ => e k) ?_
    rw [Fin.sum_univ_eq_sum_range (fun k => liftAt va vb K (t * (64 * q) + k)) (64 * q)]
    exact (sum_window_le va vb K _ _).trans hmag
  have hrel : Device.relChunk16 (64 * q) ≤ 1 / 100 := by
    unfold Device.relChunk16
    rw [hc]
    have h1 : (1 + Device.rhoChunk16) ^ q ≤ (1 + Device.rhoChunk16) ^ 4 :=
      pow_le_pow_right₀ (by unfold Device.rhoChunk16; norm_num) hq4
    have : (1 + Device.rhoChunk16) ^ 4 ≤ 101 / 100 := by unfold Device.rhoChunk16; norm_num
    linarith
  have habs : Device.absChunk16 (64 * q) ≤ 1 := by
    unfold Device.absChunk16
    rw [hc]
    have h1 : (1 + Device.rhoChunk16) ^ q ≤ (1 + Device.rhoChunk16) ^ 4 :=
      pow_le_pow_right₀ (by unfold Device.rhoChunk16; norm_num) hq4
    have h2 : (1 + Device.rhoChunk16) ^ 4 ≤ 2 := by unfold Device.rhoChunk16; norm_num
    have h3 : (q : ℚ) ≤ 4 := by exact_mod_cast hq4
    have h4 : Device.alphaChunk16 ≤ 1 / 100 := by unfold Device.alphaChunk16; norm_num
    have h5 : 0 ≤ Device.alphaChunk16 := by unfold Device.alphaChunk16; norm_num
    have h6 : 0 ≤ (1 + Device.rhoChunk16) ^ q := by
      have : 0 ≤ Device.rhoChunk16 := by unfold Device.rhoChunk16; norm_num
      positivity
    calc (q : ℚ) * (1 + Device.rhoChunk16) ^ q * Device.alphaChunk16 ≤ (4 * 2) * (1 / 100) :=
          mul_le_mul (mul_le_mul h3 (h1.trans h2) h6 (by norm_num)) h4 h5 (by norm_num)
      _ ≤ 1 := by norm_num
  have hroom : |vc| + ∑ k : Fin (64 * q), |(fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0) k *
        (fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0) k| +
      (Device.relChunk16 (64 * q) * (|vc| + ∑ k : Fin (64 * q), |(fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0) k *
        (fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0) k|) +
        Device.absChunk16 (64 * q)) ≤ DType.bound .f32 := by
    have h0 : 0 ≤ Device.relChunk16 (64 * q) := by
      unfold Device.relChunk16
      rw [hc]
      have : 1 ≤ (1 + Device.rhoChunk16) ^ q :=
        one_le_pow₀ (by unfold Device.rhoChunk16; norm_num)
      linarith
    have hs0 : 0 ≤ ∑ k : Fin (64 * q), |(fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0) k *
        (fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0) k| :=
      Finset.sum_nonneg fun _ _ => abs_nonneg _
    have := two_pow_123_le
    have hm : |vc| + ∑ k : Fin (64 * q), |(fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0) k *
        (fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0) k| ≤ 2 ^ 122 := by
      have : (2 : ℚ) ^ 121 + 2 ^ 120 ≤ 2 ^ 122 := by norm_num
      linarith
    have := mul_le_mul hrel hm (by positivity) (by norm_num)
    nlinarith
  obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := (hdot .tf32) a b acc r hd i n
    (fun kk : Fin (64 * q) => if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0)
    (fun kk : Fin (64 * q) => if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0)
    vc hva hvb hvc hroom
  refine ⟨s, vr, hvr, h0, by rw [← hn, hc], ?_⟩
  intro j hj Mj hs hm
  have hj' : j < Device.chunks16 (64 * q) := by rw [hc]; exact hj
  have hcs : Device.chunkSum16 (fun kk : Fin (64 * q) =>
      if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0)
      (fun kk : Fin (64 * q) => if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0) j =
      chunkSum va vb K (t * q + j) := by
    unfold Device.chunkSum16 Device.chunk16 chunkSum
    have e1 : ∑ k ∈ (Finset.univ : Finset (Fin (64 * q))).filter (fun k => k.val / 64 = j),
        (if t * (64 * q) + k < K then (va (t * (64 * q) + k)).toRat else 0) *
        (if t * (64 * q) + k < K then (vb (t * (64 * q) + k)).toRat else 0) =
        ∑ k ∈ (Finset.univ : Finset (Fin (64 * q))).filter (fun k => k.val / 64 = j),
          (fun n => prodAt va vb K (t * (64 * q) + n)) k :=
      Finset.sum_congr rfl fun k _ => lane_prod va vb K _ _
    rw [e1]
    refine (sum_chunk_fin q j hj (fun n => prodAt va vb K (t * (64 * q) + n))).trans ?_
    refine Finset.sum_congr rfl fun i _ => ?_
    show prodAt va vb K (t * (64 * q) + (64 * j + i)) = prodAt va vb K (64 * (t * q + j) + i)
    congr 1
    ring
  have hlift : ∀ k ∈ Device.chunk16 (64 * q) j,
      Device.liftMag (DotCfg.mk .bf16 .bf16 .f32 .tf32 0).a ((fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0) k) *
      Device.liftMag (DotCfg.mk .bf16 .bf16 .f32 .tf32 0).b ((fun kk : Fin (64 * q) =>
        if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0) k) ≤ Mj := by
    intro k hk
    have hk' : k.val / 64 = j := by
      unfold Device.chunk16 at hk; simpa using hk
    show Device.liftMag .bf16 _ * Device.liftMag .bf16 _ ≤ Mj
    rw [lane_lift]
    refine le_trans ?_ hm
    unfold chunkLift
    have hidx : t * (64 * q) + (k : Nat) = 64 * (t * q + j) + k.val % 64 := by
      have : (k : Nat) = 64 * j + k.val % 64 := by omega
      rw [this]; ring_nf
      omega
    rw [hidx]
    exact Finset.single_le_sum (f := fun i => liftAt va vb K (64 * (t * q + j) + i))
      (fun _ _ => liftAt_nonneg _ _ _ _) (Finset.mem_range.mpr (Nat.mod_lt _ (by norm_num)))
  have := hstep j hj' Mj hs hlift
  rw [hcs] at this
  refine this.trans ?_
  have e1 : Device.rhoChunk16 = tRho := rfl
  have e2 : Device.sigmaChunk16 = tSig := rfl
  have e3 : Device.alphaChunk16 = tAlp := rfl
  rw [e1, e2, e3]
  have := min_le_left (tRho * Mj) (Device.kappaChunk16 * Device.chunkAbs16 (fun kk : Fin (64 * q) =>
      if t * (64 * q) + kk < K then (va (t * (64 * q) + kk)).toRat else 0)
      (fun kk : Fin (64 * q) => if t * (64 * q) + kk < K then (vb (t * (64 * q) + kk)).toRat else 0) j +
      Device.rhoLossChunk16 * Mj)
  linarith

/-- Each trip's `A_g` add up to at most `q·2^120`. -/
theorem tripMag_le (va vb : Nat → BF16) (BK K s : Nat) (hmag : dotMag va vb K ≤ 2 ^ 120) :
    tripMag va vb BK K s ≤ ((BK / 64 : Nat) : ℚ) * 2 ^ 120 := by
  unfold tripMag
  calc _ ≤ ∑ _j ∈ Finset.range (BK / 64), (2 : ℚ) ^ 120 :=
        Finset.sum_le_sum fun j _ => (Ag_le _ _ _ _ _).trans hmag
    _ = _ := by simp

/-! ## One trip of the loop -/

/-- **One trip of the loop** (`acc += tl.dot(a, b)`, line 373), at one lane `(i, n)`: whichever
form of `tl.dot_add` the device runs, if the accumulator is within `accErr t` of the exact partial
sum of the first `t` trips, the new one is within `accErr (t + 1)` of the exact partial sum of
the first `t + 1`. The tiles' lanes hold the products' operands of positions `t·BK + kk`, and
`+0` past `K` (`other=0.0`). -/
theorem trip_step {P : Platform} (hdot : Device.DotBF16Chunk P) {fusion : Bool}
    {BK K t M N : Nat} (hBK : BK = 64 ∨ BK = 128 ∨ BK = 256) (hK15 : K ≤ 2 ^ 15)
    (ht : t < steps BK K) {a : Tensor [M, BK] BF16} {b : Tensor [BK, N] BF16}
    {c z r : Tensor [M, N] F32} (i : Fin M) (n : Fin N) (va vb : Nat → BF16)
    (hmag : dotMag va vb K ≤ 2 ^ 120)
    (hva : ∀ kk : Fin BK, DType.val? .bf16 (a.get2 i kk) =
      some (if t * BK + kk < K then (va (t * BK + kk)).toRat else 0))
    (hvb : ∀ kk : Fin BK, DType.val? .bf16 (b.get2 kk n) =
      some (if t * BK + kk < K then (vb (t * BK + kk)).toRat else 0))
    (hz : DType.val? .f32 (z.get2 i n) = some 0)
    (hr : P.dot ⟨.bf16, .bf16, .f32, .tf32, 0⟩ a b c r ∨
      ∃ d, P.dot ⟨.bf16, .bf16, .f32, .tf32, 0⟩ a b z d ∧
        ∀ ix, AccAdd fusion .f32 (d.get ix) (c.get ix) (r.get ix))
    {v0 : ℚ} (hc : DType.val? .f32 (c.get2 i n) = some v0)
    (hprev : |v0 - preSum va vb K (t * (BK / 64))| ≤ accErr BK K va vb t) :
    ∃ v, DType.val? .f32 (r.get2 i n) = some v ∧
      |v - preSum va vb K ((t + 1) * (BK / 64))| ≤ accErr BK K va vb (t + 1) := by
  obtain ⟨q, hq1, hq4, hq⟩ : ∃ q, 1 ≤ q ∧ q ≤ 4 ∧ BK = 64 * q := by
    rcases hBK with rfl | rfl | rfl
    exacts [⟨1, by omega, by omega, rfl⟩, ⟨2, by omega, by omega, rfl⟩, ⟨4, by omega, by omega, rfl⟩]
  subst hq
  have hq64 : 64 * q / 64 = q := by omega
  have hroomE : accErr (64 * q) K va vb t ≤ 2 ^ 118 :=
    accErr_le_room hBK hK15 va vb hmag ht.le
  -- the exact sequences of the trip
  set R : Nat → ℚ := fun j => preSum va vb K (t * q + j) with hR
  set m : Nat → ℚ := fun j => chunkLift va vb K (t * q + j) with hm
  set A : Nat → ℚ := fun j => Ag va vb (64 * q) K (t * q + j) with hA
  have hRd : ∀ j, R (j + 1) - R j = chunkSum va vb K (t * q + j) := fun j => by
    simp only [hR]
    rw [← Nat.add_assoc, preSum_succ]; ring
  have hc' : ∀ j, |R (j + 1) - R j| ≤ m j := fun j => by
    rw [hRd]; exact abs_chunkSum_le _ _ _ _
  have hRA : ∀ j, |R j| ≤ A j := fun j =>
    (le_max_left _ _).trans (le_max_left _ _)
  have hmA : ∀ j, m j ≤ A j := fun j => le_max_right _ _
  have hZ : ∀ j < q, |R 0| ≤ A j := fun j hj => by
    have hmod : (t * q + j) % q = j := by
      rw [mul_comm t q, Nat.mul_add_mod, Nat.mod_eq_of_lt hj]
    have hidx : t * q + j - (t * q + j) % q = t * q + 0 := by omega
    have : |R 0| ≤ |preSum va vb K (t * q + j - (t * q + j) % (64 * q / 64))| := by
      rw [hq64, hidx]
    exact this.trans ((le_max_right _ _).trans (le_max_left _ _))
  have hS : tripMag va vb (64 * q) K t = ∑ j ∈ Finset.range q, A j := by
    unfold tripMag; rw [hq64]
  have hSle : tripMag va vb (64 * q) K t ≤ q * 2 ^ 120 := by
    have := tripMag_le va vb (64 * q) K t hmag
    rwa [hq64] at this
  have hS0 : 0 ≤ tripMag va vb (64 * q) K t := tripMag_nonneg _ _ _ _ _
  have hacc : accErr (64 * q) K va vb (t + 1) = (1 + 3 * tU) * tLam ^ q *
      (accErr (64 * q) K va vb t + 9 / 5 * tRho * tripMag va vb (64 * q) K t +
        (q : ℚ) * (3 / 2 * tAlp)) := by
    simp only [accErr, tG, hq64]
  have hprev' : |v0 - R 0| ≤ accErr (64 * q) K va vb t := by
    simpa [hR, hq64] using hprev
  have hRq : R q = preSum va vb K ((t + 1) * (64 * q / 64)) := by
    simp only [hR, hq64]; rw [Nat.succ_mul]
  have hv0 : |v0| ≤ 2 ^ 121 := by
    have h1 : |v0| ≤ |R 0| + |v0 - R 0| := by
      have := abs_add_le (R 0) (v0 - R 0)
      have e : R 0 + (v0 - R 0) = v0 := by ring
      rw [e] at this; linarith
    have h2 : |R 0| ≤ 2 ^ 120 := (abs_preSum_le _ _ _ _).trans hmag
    have : (2 : ℚ) ^ 118 + 2 ^ 120 ≤ 2 ^ 121 := by norm_num
    linarith
  have hq' : (q : ℚ) ≤ 4 := by exact_mod_cast hq4
  have hS122 : tripMag va vb (64 * q) K t ≤ 2 ^ 122 := by
    have : (q : ℚ) * 2 ^ 120 ≤ 4 * 2 ^ 120 := by nlinarith
    have h4 : (4 : ℚ) * 2 ^ 120 = 2 ^ 122 := by norm_num
    linarith
  have hL1 : 1 ≤ tLam ^ q := one_le_pow₀ tLam_ge
  have hu3 : 0 ≤ 3 * tU := by have := tU_pos; positivity
  rcases hr with hd | ⟨d, hd, hadd⟩
  · -- the folded form: the chain continues from the accumulator
    obtain ⟨s, vr, hvr, h0, hn, hstep⟩ := dot_chain hdot hq4 hd i n va vb hva hvb hc hv0 hmag
    have hf := fold_trip (q := q) (s := s) (R := R) (m := m) (A := A) (e := accErr (64 * q) K va vb t)
      (fun j hj Mj h1 h2 => by
        have := hstep j hj Mj h1 h2
        rw [hRd]; exact this)
      (fun j _ => hRA j) (fun j _ => hmA j) (fun j _ => hc' j) (by rw [h0]; exact hprev')
    rw [hn, ← hS] at hf
    refine ⟨vr, hvr, ?_⟩
    rw [← hRq, hacc]
    have hE0 := accErr_nonneg (64 * q) K va vb t
    have hr0 := tRho_pos
    have hs0 := tSig_pos
    have ha0 := tAlp_pos
    have hsig : (tRho + 2 * tSig) * tripMag va vb (64 * q) K t ≤
        9 / 5 * tRho * tripMag va vb (64 * q) K t := by
      have e1 : tSig = tRho / 128 := by have := tSig_eq; linarith
      rw [e1]; nlinarith
    have hqa : (q : ℚ) * tAlp ≤ (q : ℚ) * (3 / 2 * tAlp) := by
      have : (0 : ℚ) ≤ q := Nat.cast_nonneg _
      nlinarith
    have hinner : accErr (64 * q) K va vb t + (tRho + 2 * tSig) * tripMag va vb (64 * q) K t +
        q * tAlp ≤ accErr (64 * q) K va vb t + 9 / 5 * tRho * tripMag va vb (64 * q) K t +
          (q : ℚ) * (3 / 2 * tAlp) := by linarith
    have hl0 : 0 ≤ tLam ^ q := by linarith
    calc _ ≤ _ := hf
      _ ≤ tLam ^ q * (accErr (64 * q) K va vb t + 9 / 5 * tRho * tripMag va vb (64 * q) K t +
          (q : ℚ) * (3 / 2 * tAlp)) := mul_le_mul_of_nonneg_left hinner hl0
      _ ≤ _ := by
        have hpos : 0 ≤ tLam ^ q * (accErr (64 * q) K va vb t +
            9 / 5 * tRho * tripMag va vb (64 * q) K t + (q : ℚ) * (3 / 2 * tAlp)) := by
          refine mul_nonneg hl0 ?_
          have h1 : 0 ≤ 9 / 5 * tRho * tripMag va vb (64 * q) K t := by positivity
          have h2 : 0 ≤ (q : ℚ) * (3 / 2 * tAlp) := by positivity
          linarith
        nlinarith
  · -- the separate form: a dot from zero, then the addition
    obtain ⟨s, vd, hvd, h0, hn, hstep⟩ := dot_chain hdot hq4 hd i n va vb hva hvb hz
      (by norm_num) hmag
    have hsc := sep_chain (q := q) (s := s) (R := R) (m := m) (A := A) hq1 hq4 h0
      (fun j hj Mj h1 h2 => by
        have := hstep j hj Mj h1 h2
        rw [hRd]; exact this)
      (fun j _ => hRA j) hZ (fun j _ => hmA j) (fun j _ => hc' j)
    rw [hn, ← hS] at hsc
    have hQ : |R q - R 0| ≤ tripMag va vb (64 * q) K t := by
      have h1 := abs_sub_zero_le (R := R) (m := m) hc' q
      refine h1.trans ?_
      rw [hS]
      exact Finset.sum_le_sum fun j _ => hmA j
    have hZS : |R 0| ≤ tripMag va vb (64 * q) K t := by
      rw [hS]
      refine le_trans (hRA 0) ?_
      exact Finset.single_le_sum (f := A) (fun j _ => Ag_nonneg _ _ _ _ _)
        (Finset.mem_range.mpr (by omega))
    have hvdfin : (d.get2 i n).isFinite = true ∧ (d.get2 i n).toRat = vd := DType.val?_f32.mp hvd
    have hcfin : (c.get2 i n).isFinite = true ∧ (c.get2 i n).toRat = v0 := DType.val?_f32.mp hc
    have hadd' : AddRel fusion (d.get2 i n) (c.get2 i n) (r.get2 i n) := hadd (i, n, ())
    have hW : (7 / 4 * tRho + 3 * tSig) * tripMag va vb (64 * q) K t + (q : ℚ) * tAlp ≤
        2 ^ 122 + 4 := by
      have e1 : tSig = tRho / 128 := by have := tSig_eq; linarith
      have e2 : tRho ≤ 1 / 100 := by unfold tRho; norm_num
      have e3 : tAlp ≤ 1 := by unfold tAlp; norm_num
      have := tRho_pos
      rw [e1]
      nlinarith
    have hLq : tLam ^ q ≤ 2 := by
      have h1 : tLam ^ q ≤ tLam ^ 4 := pow_le_pow_right₀ tLam_ge hq4
      have : tLam ^ 4 ≤ 2 := by unfold tLam tRho tSig; norm_num
      linarith
    have hvd_le : |vd| ≤ 2 ^ 125 := by
      have h1 : |vd| ≤ |R q - R 0| + |vd - (R q - R 0)| := by
        have := abs_add_le (R q - R 0) (vd - (R q - R 0))
        have e : (R q - R 0) + (vd - (R q - R 0)) = vd := by ring
        rw [e] at this; linarith
      have h2 : |vd - (R q - R 0)| ≤ 2 * (2 ^ 122 + 4) := by
        refine hsc.trans ?_
        have := tRho_pos; have := tSig_pos; have := tAlp_pos
        exact mul_le_mul hLq hW (by positivity) (by norm_num)
      have : (2 : ℚ) ^ 122 + 2 * (2 ^ 122 + 4) ≤ 2 ^ 125 := by norm_num
      linarith
    obtain ⟨hrfin, hrerr⟩ := FloatV.AddRel.err Format.binary32_regular (by decide)
      (by rw [Format.binary32_u]; norm_num) hadd' hvdfin.1 hcfin.1
      (by
        rw [hvdfin.2, hcfin.2, Format.binary32_maxVal, Format.binary32_eta]
        have : (2 : ℚ) ^ 125 + 2 ^ 121 ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * 2 ^ (-150 : ℤ) := by
          norm_num
        linarith)
    rw [hvdfin.2, hcfin.2, Format.binary32_u, Format.binary32_eta] at hrerr
    refine ⟨(r.get2 i n).toRat, DType.val?_f32.mpr ⟨hrfin, rfl⟩, ?_⟩
    rw [← hRq, hacc]
    have hres := sep_combine (q := q) (L := tLam ^ q) (S := tripMag va vb (64 * q) K t)
      (e := accErr (64 * q) K va vb t) (r := (r.get2 i n).toRat) (d := vd) (X := v0)
      (Rq := R q) (R0 := R 0) hq1 hL1 hS0 (accErr_nonneg _ _ _ _ _) hsc hQ hprev' hZS
      (by
        have : (3 * tU * (|vd| + |v0|) + tEta4) = 3 * 2 ^ (-24 : ℤ) * (|vd| + |v0|) + 4 * 2 ^ (-150 : ℤ) := rfl
        rw [this]; exact hrerr)
    exact hres

end Claims.FusedMoe
