import Proof.RowExp
import Proof.CoreRow

/-!
# One trip of one row, from the float facts to the core's `TripStat`

`row_stat`: the maximum, the reference, `re_scale`, `p`, the sum and the new `deno` of one row of a
trip, as the walk delivers them, give the core's `TripStat` (with the `bfloat16` rounding factors of
`p`).
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-! ## Lanes and positions -/

/-- The positions of the valid lanes of a trip starting at position `base`. -/
def laneSet (base : ℕ) (valid : Fin 64 → Bool) : Finset ℕ :=
  (Finset.univ.filter fun n => valid n = true).image fun n : Fin 64 => base + (n : ℕ)

/-- A lane function as a function of the position `base + lane` (`0` elsewhere). -/
def ext64 (base : ℕ) (f : Fin 64 → ℝ) : ℕ → ℝ :=
  fun j => if h : base ≤ j ∧ j - base < 64 then f ⟨j - base, h.2⟩ else 0

/-- Membership in `laneSet` means being `base + n` for some valid lane `n`. -/
theorem mem_laneSet {base : ℕ} {valid : Fin 64 → Bool} {j : ℕ} :
    j ∈ laneSet base valid ↔ ∃ n : Fin 64, valid n = true ∧ j = base + (n : ℕ) := by
  unfold laneSet
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨n, hn, rfl⟩; exact ⟨n, hn, rfl⟩
  · rintro ⟨n, hn, rfl⟩; exact ⟨n, hn, rfl⟩

/-- The lane set has at most 64 elements. -/
theorem laneSet_card (base : ℕ) (valid : Fin 64 → Bool) : (laneSet base valid).card ≤ 64 := by
  unfold laneSet
  refine (Finset.card_image_le).trans ?_
  calc (Finset.univ.filter fun n : Fin 64 => valid n = true).card ≤ Finset.univ.card :=
        Finset.card_le_card (Finset.filter_subset _ _)
    _ = 64 := by simp

/-- The extension `ext64 base f` of a 64-lane function agrees with `f n` at position `base + n`. -/
theorem ext64_lane (base : ℕ) (f : Fin 64 → ℝ) (n : Fin 64) : ext64 base f (base + (n : ℕ)) = f n := by
  unfold ext64
  have h : base ≤ base + (n : ℕ) ∧ base + (n : ℕ) - base < 64 := ⟨Nat.le_add_right _ _, by
    rw [Nat.add_sub_cancel_left]; exact n.isLt⟩
  rw [dite_eq_left_of_eq_true (eq_true h)]
  have : (⟨base + (n : ℕ) - base, h.2⟩ : Fin 64) = n := Fin.ext (by simp)
  rw [this]

/-- A sum over the valid lanes' positions is a sum over the lanes. -/
theorem sum_laneSet (base : ℕ) (valid : Fin 64 → Bool) (g : ℕ → ℝ) :
    ∑ j ∈ laneSet base valid, g j = ∑ n : Fin 64, if valid n = true then g (base + (n : ℕ)) else 0 := by
  unfold laneSet
  rw [Finset.sum_image (fun a _ b _ h => Fin.ext (by simpa using h)), Finset.sum_filter]

/-- The `deno` of a row is at most `2^20` in magnitude. -/
theorem RowInv.l_bound {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ} {S : Finset ℕ} {m l : ℝ} {G : Ghost}
    (hXp : Xp ≤ 2 ^ 13) (ht : t ≤ tmax) (hcard : S.card ≤ nmax)
    (hinv : RowInv sc dl Xp t S m l G) : |l| ≤ 2 ^ 20 := by
  obtain ⟨hU0, hUb⟩ := hinv.sum_u_bound hXp ht hcard
  have hl := hinv.2.2.2
  have htr : (t : ℝ) ≤ 2 ^ 13 := t_le_real ht
  have hΛ : expm1 (lam0 + lam1 * t) ≤ 1 := by
    have hx0 : 0 ≤ lam0 + lam1 * t := by have := Nat.cast_nonneg (α := ℝ) t; unfold lam0 lam1; positivity
    have hx : lam0 + lam1 * t ≤ 1 / 100 := by
      have h2 : lam1 * t ≤ lam1 * 2 ^ 13 := mul_le_mul_of_nonneg_left htr (by unfold lam1; norm_num)
      unfold lam0 lam1 at *; norm_num at h2 ⊢; linarith
    have := exp_le_quad (x := lam0 + lam1 * t) hx0 (by linarith)
    have hsq : (lam0 + lam1 * t) ^ 2 ≤ (1 / 100) ^ 2 := pow_le_pow_left₀ hx0 hx 2
    unfold expm1
    nlinarith
  have hta : (t : ℝ) * a0 ≤ 1 := by
    have := mul_le_mul_of_nonneg_right htr (by unfold a0; norm_num : (0 : ℝ) ≤ a0)
    unfold a0 at this ⊢; norm_num at this ⊢; linarith
  have hl' : |l - ∑ j ∈ S, G.u j| ≤ expm1 (lam0 + lam1 * t) * ∑ j ∈ S, G.u j + t * a0 := hl
  have h1 : |l| ≤ |l - ∑ j ∈ S, G.u j| + |∑ j ∈ S, G.u j| := by
    have := abs_add_le (l - ∑ j ∈ S, G.u j) (∑ j ∈ S, G.u j)
    rw [sub_add_cancel] at this; exact this
  rw [abs_of_nonneg hU0] at h1
  have h2 : expm1 (lam0 + lam1 * t) * ∑ j ∈ S, G.u j ≤ 1 * (2 ^ 18 * (1 + 1 / 2 ^ 7)) :=
    mul_le_mul hΛ hUb hU0 (by norm_num)
  norm_num at h2 hUb ⊢
  linarith

end

end Claims.ExtendAttn
