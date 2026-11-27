import Proof.RowBasic

/-!
# The row's sum, mean and centered lanes

`sum_room'` and `mean_stage'` bound the `tl.sum` of the block and the mean (line 75); `c_le'` and
`Q_le'` bound the centered lanes (line 76) and the sum of their squares (line 77). `Proof.Row`
states them under their own names.
-/

namespace Claims.GeluAndMulSparse

open TVA

namespace RowP

variable (r : RowP)

/-- `u32` as a rational numeral. -/
theorem u32_eq : u32 = 1 / 16777216 := by unfold u32; norm_num

/-- `eta32` is at most `2^-149`. -/
theorem eta32_le : eta32 ≤ 2 ^ (-149 : ℤ) := by unfold eta32; norm_num

/-- `eta32 ≥ 0`. -/
theorem eta32_nonneg : 0 ≤ eta32 := by unfold eta32; positivity

/-- `meanErr ≤ 20·u·aMean + 2^-134`. -/
theorem meanErr_le : r.meanErr ≤ 20 * u32 * r.aMean + 2 ^ (-134 : ℤ) := by
  unfold meanErr
  have := r.L_le'; have := r.aMean_nonneg'
  have : ((r.L : ℝ) + 5) * u32 * r.aMean ≤ 20 * u32 * r.aMean := by
    gcongr
    · unfold u32; positivity
    · linarith
  linarith

/-- `meanErr ≥ 0`. -/
theorem meanErr_nonneg : 0 ≤ r.meanErr := by
  unfold meanErr
  have := r.aMean_nonneg'
  have : (0 : ℝ) ≤ u32 := by unfold u32; positivity
  positivity

/-- `(1 + u)^L − 1` lies in `[0, 2^-19]`. -/
theorem E_bounds : 0 ≤ (1 + u32) ^ r.L - 1 ∧ (1 + u32) ^ r.L - 1 ≤ 2 ^ (-19 : ℤ) := by
  have hE := pow_u_sub_le r.L_le
  have hL := r.L_le'
  have h1 : 1 ≤ (1 + u32) ^ r.L := one_le_pow₀ (by unfold u32; norm_num)
  refine ⟨by linarith, hE.trans ?_⟩
  have : (r.L : ℝ) * u32 * (1 + 2 ^ (-9 : ℤ)) ≤ 15 * u32 * (1 + 2 ^ (-9 : ℤ)) := by
    gcongr; unfold u32; positivity
  refine this.trans ?_
  rw [u32_eq]; norm_num

/-- The sum's error, divided by `d`: `|S/d − mu| ≤ E·aMean + 2η(1 + u)^L` with
`E = (1 + u)^L − 1`. -/
theorem sum_div_err {S : ℝ}
    (hS : |S - ∑ j ∈ Finset.range (2 ^ r.L), r.gl j| ≤
      ((1 + u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| +
        eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L) :
    |S / r.d - r.mu| ≤ ((1 + u32) ^ r.L - 1) * r.aMean + 2 * eta32 * (1 + u32) ^ r.L := by
  have hd := r.d_pos'
  rw [r.sum_gl', r.sum_abs_gl'] at hS
  have e : S / r.d - r.mu = (S - r.d * r.mu) / r.d := by field_simp
  rw [e, abs_div, abs_of_pos hd, div_le_iff₀ hd]
  have hp : 0 ≤ (1 + u32) ^ r.L := by unfold u32; positivity
  have he := eta32_nonneg
  have hb := r.bm1_le
  have h2 : eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L ≤
      eta32 * (2 * r.d) * (1 + u32) ^ r.L := by gcongr
  have e2 : (((1 + u32) ^ r.L - 1) * r.aMean + 2 * eta32 * (1 + u32) ^ r.L) * r.d =
      ((1 + u32) ^ r.L - 1) * (r.d * r.aMean) + eta32 * (2 * r.d) * (1 + u32) ^ r.L := by ring
  rw [e2]
  linarith

/-- The `tl.sum` of the block is at most `2^48`. -/
theorem sum_room' {S : ℝ}
    (hS : |S - ∑ j ∈ Finset.range (2 ^ r.L), r.gl j| ≤
      ((1 + u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| +
        eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L) :
    |S| ≤ 2 ^ 48 := by
  have h1 : |∑ j ∈ Finset.range (2 ^ r.L), r.gl j| ≤ ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| :=
    Finset.abs_sum_le_sum_abs _ _
  rw [r.sum_abs_gl'] at h1 hS
  have hA : r.d * r.aMean ≤ 2 ^ 47 := by
    have := r.d_le15; have := r.aMean_le'; have := r.aMean_nonneg'
    calc r.d * r.aMean ≤ 2 ^ 15 * 2 ^ 32 := by gcongr
      _ = 2 ^ 47 := by norm_num
  obtain ⟨hE0, hE1⟩ := r.E_bounds
  have hp := pow_u_le r.L_le
  have hb : ((2 ^ r.L - 1 : ℕ) : ℝ) ≤ 2 ^ 16 := by
    have := r.bm1_le; have := r.d_le15; linarith
  have hEt : eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L ≤ 1 := by
    have hp0 : 0 ≤ (1 + u32) ^ r.L := by unfold u32; positivity
    have : eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L ≤ eta32 * 2 ^ 16 * 2 := by
      have : (1 + u32) ^ r.L ≤ 2 := by refine hp.trans ?_; norm_num
      have := eta32_nonneg
      gcongr
    refine this.trans ?_; unfold eta32; norm_num
  have hA0 : 0 ≤ r.d * r.aMean := by have := r.aMean_nonneg'; have := r.d_pos'; positivity
  have hEA : ((1 + u32) ^ r.L - 1) * (r.d * r.aMean) ≤ 2 ^ (-19 : ℤ) * 2 ^ 47 :=
    mul_le_mul hE1 hA hA0 (by positivity)
  have h2 := abs_sub_abs_le_abs_sub S (∑ j ∈ Finset.range (2 ^ r.L), r.gl j)
  have : (2 : ℝ) ^ (-19 : ℤ) * 2 ^ 47 + 2 ^ 47 + 1 ≤ 2 ^ 48 := by norm_num
  linarith

/-- **The mean**: within `meanErr` of `mu`. -/
theorem mean_stage' {S mean0 : ℝ}
    (hS : |S - ∑ j ∈ Finset.range (2 ^ r.L), r.gl j| ≤
      ((1 + u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |r.gl j| +
        eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + u32) ^ r.L)
    (hm : |mean0 - S / r.d| ≤ 2 ^ (-22 : ℤ) * |S / r.d| + 2 ^ (-148 : ℤ)) :
    |mean0 - r.mu| ≤ r.meanErr := by
  have h1 := r.sum_div_err hS
  set E := (1 + u32) ^ r.L - 1 with hEdef
  set p := (1 + u32) ^ r.L with hpdef
  have hE := pow_u_sub_le r.L_le
  have hp := pow_u_le r.L_le
  obtain ⟨hE0, hE1⟩ := r.E_bounds
  have hA := r.aMean_nonneg'
  have hmu := r.mu_le'
  have hL := r.L_le'
  have hSd : |S / r.d| ≤ r.aMean + E * r.aMean + 2 * eta32 * p := by
    have := abs_sub_abs_le_abs_sub (S / r.d) r.mu
    linarith
  have h2 : |mean0 - r.mu| ≤ 2 ^ (-22 : ℤ) * |S / r.d| + 2 ^ (-148 : ℤ) + |S / r.d - r.mu| := by
    have := abs_sub_le mean0 (S / r.d) r.mu
    linarith
  have hEA : E * r.aMean ≤ r.L * u32 * (1 + 2 ^ (-9 : ℤ)) * r.aMean :=
    mul_le_mul_of_nonneg_right hE hA
  have hEA2 : E * r.aMean ≤ 2 ^ (-19 : ℤ) * r.aMean := mul_le_mul_of_nonneg_right hE1 hA
  have hb4 : 2 ^ (-22 : ℤ) * |S / r.d| ≤
      2 ^ (-22 : ℤ) * (r.aMean + E * r.aMean + 2 * eta32 * p) :=
    mul_le_mul_of_nonneg_left hSd (by positivity)
  have hp2 : p ≤ 2 := by refine hp.trans ?_; norm_num
  have he := eta32_le
  have he0 := eta32_nonneg
  have hep : eta32 * p ≤ 2 ^ (-149 : ℤ) * 2 := mul_le_mul he hp2 (by linarith) (by positivity)
  unfold meanErr
  rw [u32_eq] at hEA ⊢
  have hLA : (r.L : ℝ) * (1 / 16777216) * (1 + 2 ^ (-9 : ℤ)) * r.aMean =
      (1 / 16777216) * (1 + 2 ^ (-9 : ℤ)) * (r.L * r.aMean) := by ring
  have hLA2 : ((r.L : ℝ) + 5) * (1 / 16777216) * r.aMean =
      (1 / 16777216) * (r.L * r.aMean) + 5 / 16777216 * r.aMean := by ring
  rw [hLA] at hEA
  rw [hLA2]
  have hLA3 : (r.L : ℝ) * r.aMean ≤ 15 * r.aMean := mul_le_mul_of_nonneg_right hL hA
  norm_num at hEA hb4 hEA2 hep ⊢
  linarith

/-- The centered lanes are at most `2^34`. -/
theorem c_le' {mean0 : ℝ} (hm0 : |mean0 - r.mu| ≤ r.meanErr) {c : ℕ → ℝ}
    (hc : ∀ j, j < r.d → |c j - (r.g j - mean0)| ≤ 3 * u32 * (|r.g j| + |mean0|) + 4 * eta32)
    (hc0 : ∀ j, r.d ≤ j → j < 2 ^ r.L → c j = 0) :
    ∀ j, j < 2 ^ r.L → |c j| ≤ 2 ^ 34 := by
  intro j hj
  by_cases hjd : j < r.d
  · have hg := r.g_le j hjd
    have hme := r.meanErr_le
    have hA := r.aMean_le'
    have hA0 := r.aMean_nonneg'
    have hmu := r.mu_le'
    have h0 : |mean0| ≤ 2 ^ 33 := by
      have := abs_sub_abs_le_abs_sub mean0 r.mu
      have : 20 * u32 * r.aMean ≤ 20 * u32 * 2 ^ 32 := by unfold u32; gcongr
      have : (20 : ℝ) * u32 * 2 ^ 32 + 2 ^ (-134 : ℤ) ≤ 2 ^ 32 := by unfold u32; norm_num
      linarith
    have h1 := hc j hjd
    have h2 : |r.g j - mean0| ≤ |r.g j| + |mean0| := abs_sub _ _
    have h3 : |c j| ≤ |c j - (r.g j - mean0)| + |r.g j - mean0| := by
      have := abs_sub_abs_le_abs_sub (c j) (r.g j - mean0); linarith
    have h4 : 3 * u32 * (|r.g j| + |mean0|) ≤ 3 * u32 * (2 ^ 32 + 2 ^ 33) := by
      unfold u32; gcongr
    have : (3 : ℝ) * u32 * (2 ^ 32 + 2 ^ 33) + 4 * eta32 ≤ 2 ^ 20 := by unfold u32 eta32; norm_num
    linarith
  · rw [hc0 j (not_lt.mp hjd) hj]; simp

/-- The sum of squares is at most `2^90`. -/
theorem Q_le' {c q : ℕ → ℝ} {Q : ℝ} (hcb : ∀ j, j < 2 ^ r.L → |c j| ≤ 2 ^ 34)
    (hq : ∀ j, j < 2 ^ r.L → |q j - c j * c j| ≤ u32 * |c j * c j| + eta32)
    (hQ : |Q - ∑ j ∈ Finset.range (2 ^ r.L), q j| ≤
      ((1 + 3 * u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |q j| +
        4 * eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ r.L) :
    |Q| ≤ 2 ^ 90 := by
  have hqb : ∀ j, j < 2 ^ r.L → |q j| ≤ 2 ^ 69 := by
    intro j hj
    have h1 := hq j hj
    have h2 : |c j * c j| ≤ 2 ^ 68 := by
      rw [abs_mul]
      have := hcb j hj
      calc |c j| * |c j| ≤ 2 ^ 34 * 2 ^ 34 := by gcongr
        _ = 2 ^ 68 := by norm_num
    have h3 := abs_sub_abs_le_abs_sub (q j) (c j * c j)
    have : u32 * |c j * c j| ≤ u32 * 2 ^ 68 := by unfold u32; gcongr
    have : u32 * 2 ^ 68 + eta32 ≤ 2 ^ 68 := by unfold u32 eta32; norm_num
    linarith
  have hB : ((2 ^ r.L : ℕ) : ℝ) ≤ 2 ^ 15 := by
    have : 2 ^ r.L ≤ 2 ^ 15 := Nat.pow_le_pow_right (by norm_num) r.L_le
    exact_mod_cast this
  have hsum : ∑ j ∈ Finset.range (2 ^ r.L), |q j| ≤ 2 ^ 84 := by
    calc ∑ j ∈ Finset.range (2 ^ r.L), |q j| ≤ ∑ j ∈ Finset.range (2 ^ r.L), (2 : ℝ) ^ 69 :=
          Finset.sum_le_sum fun j hj => hqb j (Finset.mem_range.mp hj)
      _ = ((2 ^ r.L : ℕ) : ℝ) * 2 ^ 69 := by simp
      _ ≤ 2 ^ 15 * 2 ^ 69 := by gcongr
      _ = 2 ^ 84 := by norm_num
  have hs0 : 0 ≤ ∑ j ∈ Finset.range (2 ^ r.L), |q j| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have h1 : |∑ j ∈ Finset.range (2 ^ r.L), q j| ≤ ∑ j ∈ Finset.range (2 ^ r.L), |q j| :=
    Finset.abs_sum_le_sum_abs _ _
  have hp := pow_3u_le r.L_le
  have hE0 : 0 ≤ (1 + 3 * u32) ^ r.L - 1 := by
    have : 1 ≤ (1 + 3 * u32) ^ r.L := one_le_pow₀ (by unfold u32; norm_num)
    linarith
  have hE1 : (1 + 3 * u32) ^ r.L - 1 ≤ 1 := by
    have : (2 : ℝ) ^ (-8 : ℤ) ≤ 1 := by norm_num
    linarith
  have hb : ((2 ^ r.L - 1 : ℕ) : ℝ) ≤ 2 ^ 16 := by
    have := r.bm1_le; have := r.d_le15; linarith
  have hEt : 4 * eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ r.L ≤ 1 := by
    have h2 : (1 + 3 * u32) ^ r.L ≤ 2 := by refine hp.trans ?_; norm_num
    have hp0 : 0 ≤ (1 + 3 * u32) ^ r.L := by unfold u32; positivity
    have : 4 * eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ r.L ≤ 4 * eta32 * 2 ^ 16 * 2 := by
      have := eta32_nonneg
      gcongr
    refine this.trans ?_; unfold eta32; norm_num
  have h3 := abs_sub_abs_le_abs_sub Q (∑ j ∈ Finset.range (2 ^ r.L), q j)
  have : ((1 + 3 * u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |q j| ≤ 2 ^ 84 := by nlinarith
  linarith

end RowP

end Claims.GeluAndMulSparse
