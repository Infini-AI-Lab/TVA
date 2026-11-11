import Claim
import TVA.Proof
import Proof.Util
import Proof.NumEntry

/-!
# The second dot and the store, in the reals

The row `a` of the rounded matrix `A` meets a column of `b_do` in `tl.dot`, whose result is
rounded to `bfloat16`. `final_err` combines the stages' errors into the claim's tolerance
`2^-7·Σ|x d| + 2^-13·Σ m D + 2^-100`, from the entries' bounds (`entry_err`, `entry_mag`).
The sums run over any finite set of lanes.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- `aMag` is `σ·Mg·W` plus at most `8`. -/
theorem aM_le_up {σ Mg W : ℝ} (hσ2 : σ ≤ 2 ^ 32) (hW0 : 0 ≤ W)
    (hW : W ≤ 2 ^ 64) (hr : σ * Mg ≤ 2 ^ 100) : aM σ Mg W ≤ σ * Mg * W + 8 := by
  unfold aM
  have h1 : σ * (Mg + 1 / 2 ^ 94) * (W + 1 / 2 ^ 112) =
      σ * Mg * W + σ * Mg * (1 / 2 ^ 112) + σ * (1 / 2 ^ 94) * W + σ * (1 / 2 ^ 94) * (1 / 2 ^ 112) := by ring
  have h2 : σ * Mg * (1 / 2 ^ 112) ≤ 2 ^ 100 * (1 / 2 ^ 112) :=
    mul_le_mul_of_nonneg_right hr (by positivity)
  have h3 : σ * (1 / 2 ^ 94) * W ≤ 2 ^ 32 * (1 / 2 ^ 94) * 2 ^ 64 := by
    have := mul_le_mul hσ2 hW hW0 (by positivity : (0 : ℝ) ≤ 2 ^ 32)
    nlinarith
  have h4 : σ * (1 / 2 ^ 94) * (1 / 2 ^ 112) ≤ 2 ^ 32 * (1 / 2 ^ 94) * (1 / 2 ^ 112) := by gcongr
  rw [h1]
  norm_num at h2 h3 h4 ⊢
  linarith

set_option maxHeartbeats 1600000 in
/-- **The tolerance.** Lanes `k` of a finite set `s`: `x k` the exact entry of `A`, `a k` the
computed one (`bfloat16`) within `2^-8·|x k| + κ·m k` of it and of lifted magnitude `la k ≤
(1 + 2^-7)·m k` (and `|x k| ≤ m k`); `d k` the `do` value with lifted magnitude `D k`. The second dot's result `v` is
within `ρ2·Σ la·D + α2` of `Σ a·d`, and the stored `o` is `v` rounded to `bfloat16`. Then `o` is
within the claim's tolerance of `Σ x·d`. -/
theorem final_err {ι : Type} (s : Finset ι) (a x m d D la : ι → ℝ) {v o ρ2 α2 : ℝ}
    (hm0 : ∀ k ∈ s, 0 ≤ m k) (hxm : ∀ k ∈ s, |x k| ≤ m k) (hD0 : ∀ k ∈ s, 0 ≤ D k) (hdD : ∀ k ∈ s, |d k| ≤ D k)
    (hax : ∀ k ∈ s, |a k - x k| ≤ 1 / 2 ^ 8 * |x k| + kap * m k)
    (hla : ∀ k ∈ s, |a k| ≤ la k) (hla' : ∀ k ∈ s, la k ≤ (1 + 1 / 2 ^ 7) * m k)
    (hρ0 : 0 ≤ ρ2) (hρ : ρ2 ≤ 66 / 2 ^ 22) (hα : α2 ≤ 66 / 2 ^ 126)
    (hv : |v - ∑ k ∈ s, a k * d k| ≤ ρ2 * ∑ k ∈ s, la k * D k + α2)
    (ho : |o - v| ≤ 1 / 2 ^ 8 * |v| + 1 / 2 ^ 133) :
    |o - ∑ k ∈ s, x k * d k| ≤ 1 / 2 ^ 7 * ∑ k ∈ s, |x k| * |d k| +
      1 / 2 ^ 13 * ∑ k ∈ s, m k * D k + 1 / 2 ^ 100 := by
  set Sx : ℝ := ∑ k ∈ s, |x k| * |d k| with hSx
  set SM : ℝ := ∑ k ∈ s, m k * D k with hSM
  have hSM0 : 0 ≤ SM := Finset.sum_nonneg fun k hk => mul_nonneg (hm0 k hk) (hD0 k hk)
  have hSx0 : 0 ≤ Sx := Finset.sum_nonneg fun k _ => mul_nonneg (abs_nonneg _) (abs_nonneg _)
  have hSxM : Sx ≤ SM := by
    refine Finset.sum_le_sum fun k hk => ?_
    exact mul_le_mul (hxm k hk) (hdD k hk) (abs_nonneg _) (hm0 k hk)
  -- the entries' errors
  have h1 : |∑ k ∈ s, a k * d k - ∑ k ∈ s, x k * d k| ≤ 1 / 2 ^ 8 * Sx + kap * SM := by
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    have : ∀ k ∈ s, |a k * d k - x k * d k| ≤ 1 / 2 ^ 8 * (|x k| * |d k|) + kap * (m k * D k) := by
      intro k hk
      rw [← sub_mul, abs_mul]
      calc |a k - x k| * |d k| ≤ (1 / 2 ^ 8 * |x k| + kap * m k) * |d k| :=
            mul_le_mul_of_nonneg_right (hax k hk) (abs_nonneg _)
        _ = 1 / 2 ^ 8 * (|x k| * |d k|) + kap * (m k * |d k|) := by ring
        _ ≤ 1 / 2 ^ 8 * (|x k| * |d k|) + kap * (m k * D k) := by
            have : m k * |d k| ≤ m k * D k := mul_le_mul_of_nonneg_left (hdD k hk) (hm0 k hk)
            have hk0 : 0 ≤ kap := by unfold kap; positivity
            nlinarith
    refine (Finset.sum_le_sum this).trans (le_of_eq ?_)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  -- the lifted magnitudes
  have h2 : ∑ k ∈ s, la k * D k ≤ (1 + 1 / 2 ^ 7) * SM := by
    rw [hSM, Finset.mul_sum]
    refine Finset.sum_le_sum fun k hk => ?_
    calc la k * D k ≤ ((1 + 1 / 2 ^ 7) * m k) * D k := mul_le_mul_of_nonneg_right (hla' k hk) (hD0 k hk)
      _ = (1 + 1 / 2 ^ 7) * (m k * D k) := by ring
  have h3 : ρ2 * ∑ k ∈ s, la k * D k ≤ 66 / 2 ^ 22 * ((1 + 1 / 2 ^ 7) * SM) := by
    have hl0 : 0 ≤ ∑ k ∈ s, la k * D k :=
      Finset.sum_nonneg fun k hk => mul_nonneg ((abs_nonneg _).trans (hla k hk)) (hD0 k hk)
    calc ρ2 * ∑ k ∈ s, la k * D k ≤ ρ2 * ((1 + 1 / 2 ^ 7) * SM) := mul_le_mul_of_nonneg_left h2 hρ0
      _ ≤ 66 / 2 ^ 22 * ((1 + 1 / 2 ^ 7) * SM) := mul_le_mul_of_nonneg_right hρ (by positivity)
  -- v against the exact sum
  have hv1 : |v - ∑ k ∈ s, x k * d k| ≤ 1 / 2 ^ 8 * Sx + kap * SM + 66 / 2 ^ 22 * ((1 + 1 / 2 ^ 7) * SM) +
      66 / 2 ^ 126 := by
    have := abs_add_le (v - ∑ k ∈ s, a k * d k) (∑ k ∈ s, a k * d k - ∑ k ∈ s, x k * d k)
    have e : v - ∑ k ∈ s, a k * d k + (∑ k ∈ s, a k * d k - ∑ k ∈ s, x k * d k) =
        v - ∑ k ∈ s, x k * d k := by ring
    rw [e] at this
    linarith
  have hxs : |∑ k ∈ s, x k * d k| ≤ Sx := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
    refine Finset.sum_congr rfl fun k _ => abs_mul _ _
  have hvv : |v| ≤ Sx + (1 / 2 ^ 8 * Sx + kap * SM + 66 / 2 ^ 22 * ((1 + 1 / 2 ^ 7) * SM) + 66 / 2 ^ 126) := by
    have := abs_sub_abs_le_abs_sub v (∑ k ∈ s, x k * d k)
    linarith
  -- the final rounding
  have hfin : |o - ∑ k ∈ s, x k * d k| ≤ |o - v| + |v - ∑ k ∈ s, x k * d k| := by
    have := abs_add_le (o - v) (v - ∑ k ∈ s, x k * d k)
    have e : o - v + (v - ∑ k ∈ s, x k * d k) = o - ∑ k ∈ s, x k * d k := by ring
    rw [e] at this; exact this
  have hk1 : kap ≤ 9 / 2 ^ 17 := by unfold kap; norm_num
  have hSMk : kap * SM ≤ 9 / 2 ^ 17 * SM := mul_le_mul_of_nonneg_right hk1 hSM0
  nlinarith

end Claims.ChunkBwdDvLocal
