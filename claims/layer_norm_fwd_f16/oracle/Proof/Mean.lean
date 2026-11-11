import Proof.Row

namespace Claims.LayerNormFwdF16

namespace RowP

variable (r : RowP)

/-- `(1 + u)^D - 1 ≤ 1.00002 · D · u` for `D ≤ 15`. -/
theorem pow_sub_one_le {D : ℕ} (hD : D ≤ 15) :
    (1 + u32) ^ D - 1 ≤ 100002 / 100000 * D * u32 := by
  unfold u32
  interval_cases D <;> norm_num

/-- The arithmetic of the mean's coefficient of `aMean`. -/
theorem mean_coef {u E D : ℝ} (hu0 : 0 < u) (hu : u ≤ 1 / 10 ^ 7)
    (hEu : E ≤ 100002 / 100000 * D * u) (hD7 : 7 ≤ D) :
    (u + E * (1 + u)) + 4 * u * (1 + u + E * (1 + u)) ≤ 11 / 20 * (D + 4) * (2 * u) := by
  have h1 : E * ((1 + u) * (1 + 4 * u)) ≤ 100002 / 100000 * D * u * ((1 + u) * (1 + 4 * u)) := by
    gcongr
  have h2 : (1 + u) * (1 + 4 * u) ≤ 100001 / 100000 := by nlinarith
  have h3 : 100002 / 100000 * D * u * ((1 + u) * (1 + 4 * u)) ≤
      100002 / 100000 * D * u * (100001 / 100000) := by gcongr
  have h4 : u * u ≤ u / 10 ^ 7 := by nlinarith
  nlinarith [mul_nonneg (sub_nonneg.2 hD7) hu0.le]

/-- Sum of `xl` over the whole block is `N · mu`. -/
theorem sum_xl : ∑ j ∈ Finset.range r.B, r.xl j = r.N * r.mu := by
  have h : ∑ j ∈ Finset.range r.B, r.xl j = ∑ j ∈ Finset.range r.N, r.x j := by
    rw [← Finset.sum_range_add_sum_Ico _ r.n_le_block]
    rw [Finset.sum_eq_zero (s := Finset.Ico r.N r.B) (fun j hj => by
      simp [xl, (Finset.mem_Ico.1 hj).1.not_gt]), add_zero]
    exact Finset.sum_congr rfl fun j hj => by simp [xl, Finset.mem_range.1 hj]
  have hN : (r.N : ℝ) ≠ 0 := by have := r.n_ge; positivity
  rw [h]; unfold mu; field_simp

/-- Sum of `|xl|` over the whole block is `N · aMean`. -/
theorem sum_abs_xl : ∑ j ∈ Finset.range r.B, |r.xl j| = r.N * r.aMean := by
  have h : ∑ j ∈ Finset.range r.B, |r.xl j| = ∑ j ∈ Finset.range r.N, |r.x j| := by
    rw [← Finset.sum_range_add_sum_Ico _ r.n_le_block]
    rw [Finset.sum_eq_zero (s := Finset.Ico r.N r.B) (fun j hj => by
      simp [xl, (Finset.mem_Ico.1 hj).1.not_gt]), add_zero]
    exact Finset.sum_congr rfl fun j hj => by simp [xl, Finset.mem_range.1 hj]
  have hN : (r.N : ℝ) ≠ 0 := by have := r.n_ge; positivity
  rw [h]; unfold aMean; field_simp

/-- **The stored mean**: the lanes `l` (within one rounding of the row's values, zero on the
padding), summed along a tree of depth `D` and divided by `N`, are within `11/20 · Delta` of the
exact mean. -/
theorem mean_stage (l : ℕ → ℝ) (S m : ℝ)
    (hl : ∀ j, j < r.B → |l j - r.xl j| ≤ u32 * |r.xl j| + eta32)
    (hS : |S - ∑ j ∈ Finset.range (r.B), l j| ≤
      ((1 + u32) ^ r.D - 1) * ∑ j ∈ Finset.range (r.B), |l j| +
        eta32 * ((r.B - 1 : ℕ) : ℝ) * (1 + u32) ^ r.D)
    (hm : |m - S / r.N| ≤ 2 ^ (-22 : ℤ) * |S / r.N| + 2 ^ (-148 : ℤ)) :
    |m - r.mu| ≤ 11 / 20 * r.Delta := by
  have hN : (0 : ℝ) < r.N := by have := r.n_ge; positivity
  have hB1 : (r.N : ℝ) ≤ r.B := by exact_mod_cast r.n_le_block
  have hB2 : (r.B : ℝ) < 2 * r.N := by exact_mod_cast r.block_lt
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
  have hA := r.aMean_nonneg
  have hμA : |r.mu| ≤ r.aMean := by
    unfold mu aMean
    rw [abs_div, abs_of_pos hN]
    gcongr
    exact Finset.abs_sum_le_sum_abs _ _
  set A := r.aMean
  set n : ℝ := (r.N : ℝ) with hn
  set B : ℝ := (r.B : ℝ) with hB
  set E := (1 + u32) ^ r.D - 1 with hE
  have hE0 : 0 ≤ E := by
    have := one_le_pow₀ (n := r.D) (a := 1 + u32) (by unfold u32; norm_num)
    linarith
  have hEu : E ≤ 100002 / 100000 * r.D * u32 := pow_sub_one_le r.D_le
  have hu0 : (0 : ℝ) < u32 := by unfold u32; positivity
  have hu1 : u32 = 2 ^ (-24 : ℤ) := rfl
  have hη0 : (0 : ℝ) < eta32 := by unfold eta32; positivity
  have hP : (1 + u32) ^ r.D = 1 + E := by rw [hE]; ring
  have hcast : ((r.B - 1 : ℕ) : ℝ) ≤ B := by rw [hB]; exact_mod_cast Nat.sub_le r.B 1
  -- the lanes
  have hl1 : ∑ j ∈ Finset.range r.B, |l j| ≤ (1 + u32) * (n * A) + B * eta32 := by
    calc ∑ j ∈ Finset.range r.B, |l j|
        ≤ ∑ j ∈ Finset.range r.B, ((1 + u32) * |r.xl j| + eta32) := by
          refine Finset.sum_le_sum fun j hj => ?_
          have := hl j (Finset.mem_range.1 hj)
          have := abs_sub_abs_le_abs_sub (l j) (r.xl j)
          linarith
      _ = (1 + u32) * (n * A) + B * eta32 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, r.sum_abs_xl]
          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; rfl
  have hl2 : |∑ j ∈ Finset.range r.B, l j - n * r.mu| ≤ u32 * (n * A) + B * eta32 := by
    rw [← r.sum_xl, ← Finset.sum_sub_distrib]
    calc _ ≤ ∑ j ∈ Finset.range r.B, |l j - r.xl j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ Finset.range r.B, (u32 * |r.xl j| + eta32) :=
          Finset.sum_le_sum fun j hj => hl j (Finset.mem_range.1 hj)
      _ = _ := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, r.sum_abs_xl]
          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; rfl
  rw [hP] at hS
  have hS1 : |S - n * r.mu| ≤ n * A * (u32 + E * (1 + u32)) + 2 * B * eta32 * (1 + E) := by
    have h1 : E * ∑ j ∈ Finset.range r.B, |l j| ≤ E * ((1 + u32) * (n * A) + B * eta32) :=
      mul_le_mul_of_nonneg_left hl1 hE0
    have h2 : eta32 * ((r.B - 1 : ℕ) : ℝ) * (1 + E) ≤ eta32 * B * (1 + E) := by gcongr
    have h3 : |S - n * r.mu| ≤ |S - ∑ j ∈ Finset.range r.B, l j| +
        |∑ j ∈ Finset.range r.B, l j - n * r.mu| := abs_sub_le _ _ _
    simp only [hE] at *
    nlinarith
  -- divide by n
  have hBn : B / n < 2 := by rw [div_lt_iff₀ hN]; linarith
  have hS2 : |S / n - r.mu| ≤ A * (u32 + E * (1 + u32)) + 4 * eta32 * (1 + E) := by
    have : S / n - r.mu = (S - n * r.mu) / n := by field_simp
    rw [this, abs_div, abs_of_pos hN, div_le_iff₀ hN]
    have h4 : 2 * B * eta32 * (1 + E) ≤ (4 * eta32 * (1 + E)) * n := by
      have : B ≤ 2 * n := hB2.le
      have : 0 ≤ eta32 * (1 + E) := by positivity
      nlinarith
    nlinarith
  have hS3 : |S / n| ≤ A * (1 + u32 + E * (1 + u32)) + 4 * eta32 * (1 + E) := by
    have := abs_sub_abs_le_abs_sub (S / n) r.mu
    linarith
  have hm' : |m - S / n| ≤ 4 * u32 * |S / n| + 4 * eta32 := by
    have e1 : (2 : ℝ) ^ (-22 : ℤ) = 4 * u32 := by unfold u32; norm_num
    have e2 : (2 : ℝ) ^ (-148 : ℤ) = 4 * eta32 := by unfold eta32; norm_num
    rw [e1, e2] at hm; exact hm
  have hfin : |m - r.mu| ≤ A * ((u32 + E * (1 + u32)) + 4 * u32 * (1 + u32 + E * (1 + u32))) +
      (4 * eta32 * (1 + E) * (1 + 4 * u32) + 4 * eta32) := by
    have := abs_sub_le m (S / n) r.mu
    have h5 : 4 * u32 * |S / n| ≤ 4 * u32 * (A * (1 + u32 + E * (1 + u32)) + 4 * eta32 * (1 + E)) :=
      mul_le_mul_of_nonneg_left hS3 (by positivity)
    nlinarith
  have hc : (u32 + E * (1 + u32)) + 4 * u32 * (1 + u32 + E * (1 + u32)) ≤
      11 / 20 * ((r.D : ℝ) + 4) * (2 * u32) := by
    exact mean_coef hu0 (by rw [hu1]; norm_num) hEu hD7
  have hc2 : 4 * eta32 * (1 + E) * (1 + 4 * u32) + 4 * eta32 ≤ 11 / 20 * 2 ^ (-140 : ℤ) := by
    have hE2 : E ≤ 2 := by
      have := one_add_u_pow_le r.D_le; linarith
    have hu2 : u32 ≤ 1 / 100 := by rw [hu1]; norm_num
    have e : (2 : ℝ) ^ (-140 : ℤ) = 1024 * eta32 := by unfold eta32; norm_num
    rw [e]
    nlinarith [mul_nonneg hη0.le hE0, mul_nonneg hη0.le hu0.le,
      mul_nonneg (mul_nonneg hη0.le hu0.le) (sub_nonneg.2 hE2),
      mul_nonneg hη0.le (sub_nonneg.2 hE2)]
  have hA2 := mul_le_mul_of_nonneg_left hc hA
  unfold Delta
  have e3 : (2 : ℝ) ^ (-23 : ℤ) = 2 * u32 := by unfold u32; norm_num
  rw [e3, show r.aMean = A from rfl]
  linarith

end RowP

end Claims.LayerNormFwdF16
