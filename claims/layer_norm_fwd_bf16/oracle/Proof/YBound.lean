import Proof.TolRstd

/-! The bound on the output element: `y` is finite and within `yTol` of `yRef`, from the tolerances
of the mean and `rstd` (`TolMean`, `TolRstd`) and of the intermediate products (`TolY`). -/

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section

set_option maxHeartbeats 1000000

/-- The exact mean is at most the mean of absolute values. -/
theorem yb_meanRef_abs_le_absMean (d : Inst) (init : Mem) (t : Nat) :
    |meanRef d init t| ≤ absMean d init t := by
  rcases Nat.eq_zero_or_pos d.D with h0 | hpos
  · simp [meanRef, absMean, h0]
  · have hD : (0 : ℝ) < (d.D : ℝ) := by exact_mod_cast hpos
    unfold meanRef absMean
    rw [abs_div, abs_of_nonneg (le_of_lt hD)]
    exact div_le_div_of_nonneg_right (Finset.abs_sum_le_sum_abs _ _) (le_of_lt hD)

/-- One lane's squared deviation is at most `D` times the variance. -/
theorem yb_dev_sq_le {d : Inst} {init : Mem} {t j : Nat} (hD : 0 < d.D) (hj : j < d.D) :
    (sR d init t j - meanRef d init t) ^ 2 ≤ (d.D : ℝ) * varRef d init t := by
  have hmem : j ∈ Finset.range d.D := Finset.mem_range.mpr hj
  have hnn : ∀ i ∈ Finset.range d.D, (0 : ℝ) ≤ (sR d init t i - meanRef d init t) ^ 2 := by
    intro i _
    positivity
  have h1 := Finset.single_le_sum hnn hmem
  have h2 : (∑ i ∈ Finset.range d.D, (sR d init t i - meanRef d init t) ^ 2)
      = (d.D : ℝ) * varRef d init t := by
    unfold varRef
    field_simp
  linarith [h1, h2.le, h2.ge]

/-- `D ≤ 512`. -/
theorem yb_D512 {d : Inst} (hd : d.ok) : (d.D : ℝ) ≤ 512 := by
  have h := rstd_tol_BD_le hd
  exact_mod_cast (le_trans hd.d_le_bd h)

/-- `rmsS^2 = varRef + meanRef^2`. -/
theorem yb_rmsS_sq_eq {d : Inst} {init : Mem} {t : Nat} (hD : 0 < d.D) :
    rmsS d init t ^ 2 = varRef d init t + meanRef d init t ^ 2 := by
  have hD' : (0 : ℝ) < (d.D : ℝ) := by exact_mod_cast hD
  have hsum : (∑ j ∈ Finset.range d.D, sR d init t j) = (d.D : ℝ) * meanRef d init t := by
    unfold meanRef
    field_simp
  have hexp : (∑ j ∈ Finset.range d.D, (sR d init t j - meanRef d init t) ^ 2)
      = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) - (d.D : ℝ) * meanRef d init t ^ 2 := by
    have h1 : ∀ j, (sR d init t j - meanRef d init t) ^ 2
        = sR d init t j ^ 2 - 2 * meanRef d init t * sR d init t j
          + meanRef d init t ^ 2 := by
      intro j; ring
    calc (∑ j ∈ Finset.range d.D, (sR d init t j - meanRef d init t) ^ 2)
        = ∑ j ∈ Finset.range d.D, (sR d init t j ^ 2 - 2 * meanRef d init t * sR d init t j
            + meanRef d init t ^ 2) := Finset.sum_congr rfl fun j _ => h1 j
      _ = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2)
            - 2 * meanRef d init t * (∑ j ∈ Finset.range d.D, sR d init t j)
            + (d.D : ℝ) * meanRef d init t ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.mul_sum,
            Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      _ = (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) - (d.D : ℝ) * meanRef d init t ^ 2 := by
          rw [hsum]; ring
  have hV : (∑ j ∈ Finset.range d.D, (sR d init t j - meanRef d init t) ^ 2)
      = (d.D : ℝ) * varRef d init t := by
    unfold varRef
    field_simp
  have hS : (∑ j ∈ Finset.range d.D, sR d init t j ^ 2) = (d.D : ℝ) * rmsS d init t ^ 2 :=
    rstd_tol_sum_sq_eq hD
  rw [hV, hS] at hexp
  nlinarith [hexp, hD']

/-- The stored mean's tolerance is at most `14·2^-23·absMean + 2^-140`. -/
theorem yb_meanTol_le14 {d : Inst} (hd : d.ok) {init : Mem} {t : Nat} :
    meanTol d init t ≤ 14 * (2 : ℝ) ^ (-23 : ℤ) * absMean d init t + (2 : ℝ) ^ (-140 : ℤ) := by
  have hL := rstd_tol_depth_le hd
  have h1 : (depth d : ℝ) + 4 ≤ 14 := by
    have h2 : (depth d : ℝ) ≤ 10 := by exact_mod_cast hL
    linarith
  have hA0 : (0 : ℝ) ≤ absMean d init t := rstd_tol_absMean_nonneg
  have hc0 : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) := by norm_num
  have h3 : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) * absMean d init t := mul_nonneg hc0 hA0
  have h4 := mul_le_mul_of_nonneg_right h1 h3
  unfold meanTol
  linarith [h4]

/-- On an ill-conditioned row `rmsS·rstdRef` exceeds `2^17 - 1`. -/
theorem yb_ill_SR_ge {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    (hB : 1 < 4 * cond d init t) :
    (2 : ℝ) ^ (17 : ℤ) - 1 ≤ rmsS d init t * rstdRef d init t := by
  have hD : 0 < d.D := hd.d_pos
  have hR := rstd_tol_R_pos hd (init := init) (t := t)
  have hRle := rstd_tol_R_le hd (init := init) (t := t)
  have hAS := rstd_tol_absMean_le_rmsS (init := init) (t := t) hD
  have hMT := yb_meanTol_le14 hd (init := init) (t := t)
  have hc0 : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ) := by norm_num
  have hASc : 15 * (2 : ℝ) ^ (-23 : ℤ) * absMean d init t
      ≤ 15 * (2 : ℝ) ^ (-23 : ℤ) * rmsS d init t := by
    have h0 : (0 : ℝ) ≤ 15 * (2 : ℝ) ^ (-23 : ℤ) := by norm_num
    exact mul_le_mul_of_nonneg_left hAS h0
  have hMA : meanTol d init t + (2 : ℝ) ^ (-23 : ℤ) * (absMean d init t + rmsS d init t)
      ≤ 16 * (2 : ℝ) ^ (-23 : ℤ) * rmsS d init t + (2 : ℝ) ^ (-140 : ℤ) := by
    linarith [hMT, hASc]
  have hcondU : cond d init t
      ≤ (16 * (2 : ℝ) ^ (-23 : ℤ) * rmsS d init t + (2 : ℝ) ^ (-140 : ℤ))
        * rstdRef d init t := by
    have hcondEq : cond d init t
        = (meanTol d init t + (2 : ℝ) ^ (-23 : ℤ) * (absMean d init t + rmsS d init t))
          * rstdRef d init t := rfl
    rw [hcondEq]
    exact mul_le_mul_of_nonneg_right hMA (le_of_lt hR)
  have hcond4 : (1 : ℝ) / 4 < cond d init t := by linarith [hB]
  have h1 : (1 : ℝ) / 4
      < (16 * (2 : ℝ) ^ (-23 : ℤ) * rmsS d init t + (2 : ℝ) ^ (-140 : ℤ))
        * rstdRef d init t := lt_of_lt_of_le hcond4 hcondU
  have h2 : (16 * (2 : ℝ) ^ (-23 : ℤ) * rmsS d init t + (2 : ℝ) ^ (-140 : ℤ))
        * rstdRef d init t
      = 16 * (2 : ℝ) ^ (-23 : ℤ) * (rmsS d init t * rstdRef d init t)
        + (2 : ℝ) ^ (-140 : ℤ) * rstdRef d init t := by ring
  have h3 : (2 : ℝ) ^ (-140 : ℤ) * rstdRef d init t ≤ (2 : ℝ) ^ (-77 : ℤ) := by
    have h4 : (2 : ℝ) ^ (-140 : ℤ) * rstdRef d init t
        ≤ (2 : ℝ) ^ (-140 : ℤ) * (2 : ℝ) ^ (63 : ℤ) :=
      mul_le_mul_of_nonneg_left hRle (by norm_num)
    have h5 : (2 : ℝ) ^ (-140 : ℤ) * (2 : ℝ) ^ (63 : ℤ) = (2 : ℝ) ^ (-77 : ℤ) := by
      rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
    linarith [h4, h5.le, h5.ge]
  have h4 : (1 : ℝ) / 4 - (2 : ℝ) ^ (-77 : ℤ)
      < 16 * (2 : ℝ) ^ (-23 : ℤ) * (rmsS d init t * rstdRef d init t) := by
    linarith [h1, h2.le, h2.ge, h3]
  have h5 : 16 * (2 : ℝ) ^ (-23 : ℤ) = (2 : ℝ) ^ (-19 : ℤ) := by norm_num
  have h6 : (1 : ℝ) / 4 - (2 : ℝ) ^ (-77 : ℤ)
      < (2 : ℝ) ^ (-19 : ℤ) * (rmsS d init t * rstdRef d init t) := by
    have h7 : 16 * (2 : ℝ) ^ (-23 : ℤ) * (rmsS d init t * rstdRef d init t)
        = (2 : ℝ) ^ (-19 : ℤ) * (rmsS d init t * rstdRef d init t) := by rw [h5]
    linarith [h4, h7.le, h7.ge]
  have h8 : (2 : ℝ) ^ (19 : ℤ) * ((1 : ℝ) / 4 - (2 : ℝ) ^ (-77 : ℤ))
      < rmsS d init t * rstdRef d init t := by
    have h9 := mul_lt_mul_of_pos_left h6 (by norm_num : (0 : ℝ) < (2 : ℝ) ^ (19 : ℤ))
    have h10 : (2 : ℝ) ^ (19 : ℤ)
        * ((2 : ℝ) ^ (-19 : ℤ) * (rmsS d init t * rstdRef d init t))
        = rmsS d init t * rstdRef d init t := by
      rw [← mul_assoc, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
    linarith [h9, h10.le, h10.ge]
  have h11 : (2 : ℝ) ^ (17 : ℤ) - 1
      ≤ (2 : ℝ) ^ (19 : ℤ) * ((1 : ℝ) / 4 - (2 : ℝ) ^ (-77 : ℤ)) := by norm_num
  linarith [h8, h11]

/-- A pure numeric step: a root-mean-squares dominated by the mean and `2^41` at most. -/
theorem yb_sqrt_bound {S m y : ℝ} (hy0 : 0 ≤ y)
    (hS : S ^ 2 = y ^ 2 + m ^ 2) (hSy : ((2 : ℝ) ^ (17 : ℤ) - 1) * y ≤ S)
    (hm : m ^ 2 ≤ ((2 : ℝ) ^ (57 : ℤ) + 23 * y) ^ 2) :
    y < (2 : ℝ) ^ (41 : ℤ) := by
  by_contra hcon
  push Not at hcon
  have h1 : (2 : ℝ) ^ (33 : ℤ) * y ^ 2 < m ^ 2 := by
    have h2 : (((2 : ℝ) ^ (17 : ℤ) - 1) * y) ^ 2 ≤ S ^ 2 :=
      pow_le_pow_left₀ (mul_nonneg (by norm_num) hy0) hSy 2
    have h5 : (((2 : ℝ) ^ (17 : ℤ) - 1) * y) ^ 2
        = ((2 : ℝ) ^ (17 : ℤ) - 1) ^ 2 * y ^ 2 := by ring
    have h4 : (2 : ℝ) ^ (33 : ℤ) ≤ ((2 : ℝ) ^ (17 : ℤ) - 1) ^ 2 := by norm_num
    have h6 : (2 : ℝ) ^ (33 : ℤ) * y ^ 2 ≤ ((2 : ℝ) ^ (17 : ℤ) - 1) ^ 2 * y ^ 2 :=
      mul_le_mul_of_nonneg_right h4 (by positivity)
    nlinarith [h2, h5.le, h5.ge, h6, hS]
  have h7 : (2 : ℝ) ^ (33 : ℤ) * y ^ 2
      < ((2 : ℝ) ^ (57 : ℤ) + 23 * y) ^ 2 := lt_of_lt_of_le h1 hm
  have h8 : ((2 : ℝ) ^ (57 : ℤ) + 23 * y) ^ 2
      = (2 : ℝ) ^ (114 : ℤ) + 46 * (2 : ℝ) ^ (57 : ℤ) * y + 529 * y ^ 2 := by
    have h9 : (23 * y) ^ 2 = 529 * y ^ 2 := by rw [mul_pow]; norm_num
    nlinarith [h9]
  have h10 : ((2 : ℝ) ^ (33 : ℤ) - 529) * y ^ 2
      < (2 : ℝ) ^ (114 : ℤ) + 46 * (2 : ℝ) ^ (57 : ℤ) * y := by
    nlinarith [h7, h8]
  have hy : 0 < y := by
    nlinarith [hcon]
  have h11 : y * (((2 : ℝ) ^ (33 : ℤ) - 529) * y - 46 * (2 : ℝ) ^ (57 : ℤ))
      < (2 : ℝ) ^ (114 : ℤ) := by
    nlinarith [h10]
  have h12 : ((2 : ℝ) ^ (33 : ℤ) - 529) * y - 46 * (2 : ℝ) ^ (57 : ℤ)
      > (2 : ℝ) ^ (73 : ℤ) := by
    have h13 : ((2 : ℝ) ^ (33 : ℤ) - 529) * (2 : ℝ) ^ (41 : ℤ) - 46 * (2 : ℝ) ^ (57 : ℤ)
        > (2 : ℝ) ^ (73 : ℤ) := by norm_num
    nlinarith [hcon, h13]
  have h14 : (2 : ℝ) ^ (41 : ℤ) * (2 : ℝ) ^ (73 : ℤ)
      ≤ y * (((2 : ℝ) ^ (33 : ℤ) - 529) * y - 46 * (2 : ℝ) ^ (57 : ℤ)) := by
    have h15 : (0 : ℝ) ≤ ((2 : ℝ) ^ (33 : ℤ) - 529) * y - 46 * (2 : ℝ) ^ (57 : ℤ) :=
      le_of_lt (by linarith [h12])
    have h16 := mul_le_mul hcon (le_of_lt h12) (by positivity) (le_of_lt hy)
    linarith [h16]
  have h17 : (2 : ℝ) ^ (41 : ℤ) * (2 : ℝ) ^ (73 : ℤ) = (2 : ℝ) ^ (114 : ℤ) := by norm_num
  linarith [h11, h14, h17]

/-- On an ill-conditioned row `sqrt varRef < 2^41`. -/
theorem yb_ill_sqrt_varRef {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hbddj : |sR d init t j| ≤ (2 : ℝ) ^ 57) (hB : 1 < 4 * cond d init t) :
    Real.sqrt (varRef d init t) < (2 : ℝ) ^ 41 := by
  have hD : 0 < d.D := hd.d_pos
  have hV0 : 0 ≤ varRef d init t := rstd_tol_varRef_nonneg
  have hS0 : 0 ≤ rmsS d init t := rstd_tol_rmsS_nonneg
  have hR : 0 < rstdRef d init t := rstd_tol_R_pos hd
  have hW : 0 < varRef d init t + epsR d := rstd_tol_Ws_pos (init := init) (t := t) hd
  have hSR := yb_ill_SR_ge hd (init := init) (t := t) hB
  have hRV : rstdRef d init t = 1 / Real.sqrt (varRef d init t + epsR d) := rfl
  have hsqV : Real.sqrt (varRef d init t) ^ 2 = varRef d init t := Real.sq_sqrt hV0
  have hsqrtV_le : Real.sqrt (varRef d init t) ≤ Real.sqrt (varRef d init t + epsR d) :=
    Real.sqrt_le_sqrt (by linarith [epsR_pos hd])
  have hinv : (1 : ℝ) / rstdRef d init t = Real.sqrt (varRef d init t + epsR d) := by
    rw [hRV]; field_simp
  have hSy : ((2 : ℝ) ^ (17 : ℤ) - 1) * Real.sqrt (varRef d init t) ≤ rmsS d init t := by
    have h1 : ((2 : ℝ) ^ (17 : ℤ) - 1) * Real.sqrt (varRef d init t)
        ≤ ((2 : ℝ) ^ (17 : ℤ) - 1) * Real.sqrt (varRef d init t + epsR d) :=
      mul_le_mul_of_nonneg_left hsqrtV_le (by norm_num)
    have h2 : ((2 : ℝ) ^ (17 : ℤ) - 1) * Real.sqrt (varRef d init t + epsR d)
        = ((2 : ℝ) ^ (17 : ℤ) - 1) * (1 / rstdRef d init t) := by rw [hinv]
    have h3 : ((2 : ℝ) ^ (17 : ℤ) - 1) * (1 / rstdRef d init t) ≤ rmsS d init t := by
      have h4 := mul_le_mul_of_nonneg_right hSR (le_of_lt (one_div_pos.mpr hR))
      have h5 : (rmsS d init t * rstdRef d init t) * (1 / rstdRef d init t) = rmsS d init t := by
        field_simp
      linarith [h4, h5.le, h5.ge]
    linarith [h1, h2.le, h2.ge, h3]
  have hSq := yb_rmsS_sq_eq (init := init) (t := t) hD
  have hm : meanRef d init t ^ 2
      ≤ ((2 : ℝ) ^ (57 : ℤ) + 23 * Real.sqrt (varRef d init t)) ^ 2 := by
    have hdev_le : |sR d init t j - meanRef d init t| ≤ 23 * Real.sqrt (varRef d init t) := by
      have hsq := yb_dev_sq_le (init := init) (t := t) hD hj
      have h512 := yb_D512 hd
      have h529 : (23 * Real.sqrt (varRef d init t)) ^ 2 = 529 * varRef d init t := by
        rw [mul_pow, hsqV]; norm_num
      have h2 : (sR d init t j - meanRef d init t) ^ 2
          ≤ (23 * Real.sqrt (varRef d init t)) ^ 2 := by
        rw [h529]; nlinarith [hsq, hV0, h512]
      calc |sR d init t j - meanRef d init t|
          = Real.sqrt ((sR d init t j - meanRef d init t) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
        _ ≤ Real.sqrt ((23 * Real.sqrt (varRef d init t)) ^ 2) := Real.sqrt_le_sqrt h2
        _ = 23 * Real.sqrt (varRef d init t) := Real.sqrt_sq (by positivity)
    have h1 : |meanRef d init t| ≤ (2 : ℝ) ^ 57 + 23 * Real.sqrt (varRef d init t) := by
      have h2 : |meanRef d init t| ≤ |sR d init t j| + |sR d init t j - meanRef d init t| := by
        have h3 := abs_add_le (sR d init t j) (-(sR d init t j - meanRef d init t))
        have e : sR d init t j + -(sR d init t j - meanRef d init t) = meanRef d init t := by ring
        rw [e, abs_neg] at h3
        exact h3
      linarith [hbddj, hdev_le]
    have h4 := pow_le_pow_left₀ (abs_nonneg (meanRef d init t)) h1 2
    rwa [sq_abs] at h4
  have hkey := yb_sqrt_bound (S := rmsS d init t) (m := meanRef d init t)
    (y := Real.sqrt (varRef d init t)) (Real.sqrt_nonneg _) (by rw [hSq, hsqV]) hSy hm
  exact hkey

/-- The deviation is at most `23·sqrt varRef`. -/
theorem yb_dev_le_23sqrt {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D) :
    |sR d init t j - meanRef d init t| ≤ 23 * Real.sqrt (varRef d init t) := by
  have hD := hd.d_pos
  have hsq := yb_dev_sq_le (init := init) (t := t) hD hj
  have h512 := yb_D512 hd
  have hV0 := rstd_tol_varRef_nonneg (d := d) (init := init) (t := t)
  have hsqV : Real.sqrt (varRef d init t) ^ 2 = varRef d init t := Real.sq_sqrt hV0
  have h529 : (23 * Real.sqrt (varRef d init t)) ^ 2 = 529 * varRef d init t := by
    rw [mul_pow, hsqV]; norm_num
  have h2 : (sR d init t j - meanRef d init t) ^ 2
      ≤ (23 * Real.sqrt (varRef d init t)) ^ 2 := by
    rw [h529]; nlinarith [hsq, hV0, h512]
  calc |sR d init t j - meanRef d init t|
      = Real.sqrt ((sR d init t j - meanRef d init t) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt ((23 * Real.sqrt (varRef d init t)) ^ 2) := Real.sqrt_le_sqrt h2
    _ = 23 * Real.sqrt (varRef d init t) := Real.sqrt_sq (by positivity)

/-- Ill-conditioned: the exact mean is at most `2^58`. -/
theorem yb_ill_meanRef_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hbddj : |sR d init t j| ≤ 2 ^ 57) (hB : 1 < 4 * cond d init t) :
    |meanRef d init t| ≤ (2 : ℝ) ^ 58 := by
  have hV := yb_ill_sqrt_varRef hd hj hbddj hB
  have hdev := yb_dev_le_23sqrt hd (init := init) (t := t) hj
  have h1 : |meanRef d init t| ≤ |sR d init t j| + |sR d init t j - meanRef d init t| := by
    have h3 := abs_add_le (sR d init t j) (-(sR d init t j - meanRef d init t))
    have e : sR d init t j + -(sR d init t j - meanRef d init t) = meanRef d init t := by ring
    rw [e, abs_neg] at h3
    exact h3
  have h2 : 23 * Real.sqrt (varRef d init t) ≤ 23 * (2 : ℝ) ^ 41 :=
    mul_le_mul_of_nonneg_left (le_of_lt hV) (by norm_num)
  have h3 : (2 : ℝ) ^ 57 + 23 * (2 : ℝ) ^ 41 ≤ (2 : ℝ) ^ 58 := by norm_num
  linarith [h1, hdev, h2, h3]

/-- Ill-conditioned: the root mean square is at most `2^59`. -/
theorem yb_ill_rmsS_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hbddj : |sR d init t j| ≤ 2 ^ 57) (hB : 1 < 4 * cond d init t) :
    rmsS d init t ≤ (2 : ℝ) ^ 59 := by
  have hV := yb_ill_sqrt_varRef hd hj hbddj hB
  have hm := yb_ill_meanRef_le hd hj hbddj hB
  have hSq := yb_rmsS_sq_eq (init := init) (t := t) hd.d_pos
  have hV0 := rstd_tol_varRef_nonneg (d := d) (init := init) (t := t)
  have hS0 := rstd_tol_rmsS_nonneg (d := d) (init := init) (t := t)
  have hsqV : Real.sqrt (varRef d init t) ^ 2 = varRef d init t := Real.sq_sqrt hV0
  have h2 : varRef d init t + meanRef d init t ^ 2
      ≤ (Real.sqrt (varRef d init t) + |meanRef d init t|) ^ 2 := by
    have h3 := sq_abs (meanRef d init t)
    have h4 := Real.sqrt_nonneg (varRef d init t)
    have h5 := abs_nonneg (meanRef d init t)
    nlinarith [hsqV, h3, mul_nonneg h4 h5]
  have h3 : rmsS d init t ≤ Real.sqrt (varRef d init t) + |meanRef d init t| := by
    calc rmsS d init t = Real.sqrt (rmsS d init t ^ 2) := (Real.sqrt_sq hS0).symm
      _ = Real.sqrt (varRef d init t + meanRef d init t ^ 2) := by rw [hSq]
      _ ≤ Real.sqrt ((Real.sqrt (varRef d init t) + |meanRef d init t|) ^ 2) :=
          Real.sqrt_le_sqrt h2
      _ = Real.sqrt (varRef d init t) + |meanRef d init t| := Real.sqrt_sq (by positivity)
  have h4 : (2 : ℝ) ^ 41 + (2 : ℝ) ^ 58 ≤ (2 : ℝ) ^ 59 := by norm_num
  linarith [h3, hV, hm, h4]

/-- Ill-conditioned: the mean magnitude is at most `2^59`. -/
theorem yb_ill_absMean_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hbddj : |sR d init t j| ≤ 2 ^ 57) (hB : 1 < 4 * cond d init t) :
    absMean d init t ≤ (2 : ℝ) ^ 59 := by
  have h1 := rstd_tol_absMean_le_rmsS (d := d) (init := init) (t := t) hd.d_pos
  have h2 := yb_ill_rmsS_le hd hj hbddj hB
  linarith

/-- Ill-conditioned: the mean tolerance is at most `2^40`. -/
theorem yb_ill_meanTol_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hbddj : |sR d init t j| ≤ 2 ^ 57) (hB : 1 < 4 * cond d init t) :
    meanTol d init t ≤ (2 : ℝ) ^ 40 := by
  have hMT := yb_meanTol_le14 hd (init := init) (t := t)
  have hA := yb_ill_absMean_le hd hj hbddj hB
  have h1 : 14 * (2 : ℝ) ^ (-23 : ℤ) * absMean d init t
      ≤ 14 * (2 : ℝ) ^ (-23 : ℤ) * (2 : ℝ) ^ 59 :=
    mul_le_mul_of_nonneg_left hA (by norm_num)
  have h2 : (14 : ℝ) * (2 : ℝ) ^ (-23 : ℤ) * (2 : ℝ) ^ 59 + (2 : ℝ) ^ (-140 : ℤ)
      ≤ (2 : ℝ) ^ 40 := by norm_num
  linarith [hMT, h1, h2]

/-- Ill-conditioned: the deviation is at most `23·2^41`. -/
theorem yb_ill_dev_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hbddj : |sR d init t j| ≤ 2 ^ 57) (hB : 1 < 4 * cond d init t) :
    |sR d init t j - meanRef d init t| ≤ 23 * (2 : ℝ) ^ (41 : ℤ) := by
  have hV := yb_ill_sqrt_varRef hd hj hbddj hB
  have hdev := yb_dev_le_23sqrt hd (init := init) (t := t) hj
  have h2 : 23 * Real.sqrt (varRef d init t) ≤ 23 * (2 : ℝ) ^ 41 :=
    mul_le_mul_of_nonneg_left (le_of_lt hV) (by norm_num)
  linarith [hdev, h2]

/-- Ill-conditioned: the stored mean is at most `2^59`. -/
theorem yb_ill_bmean_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    {bmean : F32} (hbddj : |sR d init t j| ≤ 2 ^ 57)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hB : 1 < 4 * cond d init t) : |bmean.toReal| ≤ (2 : ℝ) ^ 59 := by
  have hm := yb_ill_meanRef_le hd hj hbddj hB
  have ht := yb_ill_meanTol_le hd hj hbddj hB
  have h1 : |bmean.toReal| ≤ |meanRef d init t| + |bmean.toReal - meanRef d init t| := by
    have h := abs_add_le (meanRef d init t) (bmean.toReal - meanRef d init t)
    have e : meanRef d init t + (bmean.toReal - meanRef d init t) = bmean.toReal := by ring
    rw [e] at h
    exact h
  have h2 : (2 : ℝ) ^ 58 + (2 : ℝ) ^ 40 ≤ (2 : ℝ) ^ 59 := by norm_num
  linarith [h1, hm, ht, h2]

/-- Ill-conditioned: the lane mean error is at most `2^41`. -/
theorem yb_ill_eps_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    {bmean : F32} (hbddj : |sR d init t j| ≤ 2 ^ 57)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hB : 1 < 4 * cond d init t) {e : ℝ}
    (he : |e| ≤ u32 * max |sR d init t j| |bmean.toReal| + 4 * eta32) :
    |(meanRef d init t - bmean.toReal) + e| ≤ (2 : ℝ) ^ (41 : ℤ) := by
  have hbm := yb_ill_bmean_le hd hj hbddj hmean hB
  have hmt := yb_ill_meanTol_le hd hj hbddj hB
  have hmax : max |sR d init t j| |bmean.toReal| ≤ (2 : ℝ) ^ 59 :=
    max_le (by linarith) hbm
  have h1 : |e| ≤ (2 : ℝ) ^ 36 := by
    have h2 : u32 * max |sR d init t j| |bmean.toReal| ≤ u32 * (2 : ℝ) ^ 59 :=
      mul_le_mul_of_nonneg_left hmax (le_of_lt u32_pos)
    have h3 : u32 * (2 : ℝ) ^ 59 ≤ (2 : ℝ) ^ 35 := by unfold u32; norm_num
    have h4 : 4 * eta32 ≤ 1 := by unfold eta32; norm_num
    linarith [he, h2, h3, h4]
  have h5 : |(meanRef d init t - bmean.toReal) + e|
      ≤ |meanRef d init t - bmean.toReal| + |e| := abs_add_le _ _
  have h6 : |meanRef d init t - bmean.toReal| ≤ meanTol d init t := by
    rw [abs_sub_comm]; exact hmean
  have h7 : (2 : ℝ) ^ 40 + (2 : ℝ) ^ 36 ≤ (2 : ℝ) ^ (41 : ℤ) := by norm_num
  linarith [h5, h6, h1, hmt, h7]


/-- `sqrt D ≤ 23`. -/
theorem yb_sqrtD_le {d : Inst} (hd : d.ok) : Real.sqrt (d.D : ℝ) ≤ 23 := by
  have h1 : (d.D : ℝ) ≤ (23 : ℝ) ^ 2 := by
    have h := yb_D512 hd
    have h2 : ((23 : ℝ) ^ 2) = 529 := by norm_num
    linarith
  calc Real.sqrt (d.D : ℝ) ≤ Real.sqrt ((23 : ℝ) ^ 2) := Real.sqrt_le_sqrt h1
    _ = 23 := Real.sqrt_sq (by norm_num)

/-- The deviation times `rstdRef` is at most `sqrt D ≤ 23`. -/
theorem yb_dev_mul_R_le {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D) :
    |sR d init t j - meanRef d init t| * rstdRef d init t ≤ 23 := by
  have hD := hd.d_pos
  have hDR : (0 : ℝ) ≤ (d.D : ℝ) := by
    have h : (0 : ℝ) < (d.D : ℝ) := by exact_mod_cast hD
    linarith
  have hsq := yb_dev_sq_le (init := init) (t := t) hD hj
  have hR2 := rstd_tol_R_sq_mul (d := d) (init := init) (t := t) hd
  have heps : (0 : ℝ) ≤ epsR d := le_of_lt (epsR_pos hd)
  have hVR : varRef d init t * rstdRef d init t ^ 2 ≤ 1 := by
    have h1 : varRef d init t * rstdRef d init t ^ 2
        ≤ (varRef d init t + epsR d) * rstdRef d init t ^ 2 := by
      apply mul_le_mul_of_nonneg_right _ (sq_nonneg (rstdRef d init t))
      linarith
    have h2 : (varRef d init t + epsR d) * rstdRef d init t ^ 2 = 1 := by
      rw [mul_comm]
      exact hR2
    linarith [h1, h2.le, h2.ge]
  have hsq2 : (|sR d init t j - meanRef d init t| * rstdRef d init t) ^ 2 ≤ (d.D : ℝ) := by
    rw [mul_pow, sq_abs]
    calc (sR d init t j - meanRef d init t) ^ 2 * rstdRef d init t ^ 2
        ≤ ((d.D : ℝ) * varRef d init t) * rstdRef d init t ^ 2 :=
          mul_le_mul_of_nonneg_right hsq (sq_nonneg _)
      _ = (d.D : ℝ) * (varRef d init t * rstdRef d init t ^ 2) := by ring
      _ ≤ (d.D : ℝ) * 1 := by
          apply mul_le_mul_of_nonneg_left hVR
          linarith
      _ = (d.D : ℝ) := by ring
  have h3 := Real.le_sqrt_of_sq_le hsq2
  calc |sR d init t j - meanRef d init t| * rstdRef d init t ≤ Real.sqrt (d.D : ℝ) := h3
    _ ≤ 23 := yb_sqrtD_le hd

/-- `gam`, the relative error of `rstd` from rounding, is at most `2^-10`. -/
theorem yb_gam_le {d : Inst} (hd : d.ok) : gam d ≤ (2 : ℝ) ^ (-10 : ℤ) := by
  have hL := rstd_tol_depth_le hd
  have heps := epsR_ge hd
  have h1 : ((depth d : ℝ) + 10) * (2 : ℝ) ^ (-23 : ℤ) ≤ 20 * (2 : ℝ) ^ (-23 : ℤ) := by
    apply mul_le_mul_of_nonneg_right _ (by positivity)
    have h : (depth d : ℝ) ≤ 10 := by exact_mod_cast hL
    linarith
  have h2 : (2 : ℝ) ^ (-136 : ℤ) / epsR d ≤ (2 : ℝ) ^ (-11 : ℤ) := by
    have h3 : (2 : ℝ) ^ (-136 : ℤ) / epsR d
        ≤ (2 : ℝ) ^ (-136 : ℤ) / (2 : ℝ) ^ (-125 : ℤ) :=
      div_le_div_of_nonneg_left (by positivity) (by positivity) heps
    have h4 : (2 : ℝ) ^ (-136 : ℤ) / (2 : ℝ) ^ (-125 : ℤ) = (2 : ℝ) ^ (-11 : ℤ) := by
      rw [← zpow_sub₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
    linarith [h3, h4.le, h4.ge]
  have h5 : 20 * (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-11 : ℤ) ≤ (2 : ℝ) ^ (-10 : ℤ) := by
    norm_num
  unfold gam
  calc ((depth d : ℝ) + 10) * (2 : ℝ) ^ (-23 : ℤ)
        + (2 : ℝ) ^ (-136 : ℤ) / epsR d
      ≤ 20 * (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-11 : ℤ) := add_le_add h1 h2
    _ ≤ (2 : ℝ) ^ (-10 : ℤ) := h5

/-- On a well-conditioned row `absMean·rstdRef ≤ 2^21`. -/
theorem yb_A_absMean_mul_R {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    (hA : 4 * cond d init t ≤ 1) :
    absMean d init t * rstdRef d init t ≤ (2 : ℝ) ^ 21 := by
  have hRpos := rstd_tol_R_pos hd (init := init) (t := t)
  have hR0 : (0 : ℝ) ≤ rstdRef d init t := le_of_lt hRpos
  have hc23 : (0 : ℝ) < 2 ^ (-23 : ℤ) := by norm_num
  have hMT0 := rstd_tol_meanTol_nonneg (d := d) (init := init) (t := t)
  have hS0 := rstd_tol_rmsS_nonneg (d := d) (init := init) (t := t)
  have hcond4 : cond d init t ≤ 1 / 4 := by linarith
  have hcondEq : cond d init t
      = (meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t))
        * rstdRef d init t := rfl
  have h1 : (absMean d init t + rmsS d init t) * rstdRef d init t ≤ (2 : ℝ) ^ 21 := by
    have h2 : 2 ^ (-23 : ℤ) * ((absMean d init t + rmsS d init t) * rstdRef d init t)
        ≤ cond d init t := by
      rw [hcondEq]
      have h3 : (meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t))
            * rstdRef d init t
          = meanTol d init t * rstdRef d init t
            + 2 ^ (-23 : ℤ) * ((absMean d init t + rmsS d init t) * rstdRef d init t) := by
        ring
      have h4 : (0 : ℝ) ≤ meanTol d init t * rstdRef d init t := mul_nonneg hMT0 hR0
      linarith [h3.le, h3.ge, h4]
    have h5 : 2 ^ (-23 : ℤ) * ((absMean d init t + rmsS d init t) * rstdRef d init t)
        ≤ 1 / 4 := le_trans h2 hcond4
    have h6 : (1 : ℝ) / 4 / (2 ^ (-23 : ℤ)) = (2 : ℝ) ^ 21 := by norm_num
    have h7 : (absMean d init t + rmsS d init t) * rstdRef d init t
        ≤ (1 : ℝ) / 4 / (2 ^ (-23 : ℤ)) := by
      rw [le_div_iff₀ hc23]
      rw [mul_comm]
      exact h5
    linarith [h7, h6.le, h6.ge]
  have h8 : absMean d init t * rstdRef d init t
      ≤ (absMean d init t + rmsS d init t) * rstdRef d init t := by
    apply mul_le_mul_of_nonneg_right _ hR0
    linarith
  linarith [h8, h1]

/-- On a well-conditioned row `meanTol·rstdRef ≤ 1/4`. -/
theorem yb_A_meanTol_mul_R {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    (hA : 4 * cond d init t ≤ 1) :
    meanTol d init t * rstdRef d init t ≤ 1 / 4 := by
  have hRpos := rstd_tol_R_pos hd (init := init) (t := t)
  have hR0 : (0 : ℝ) ≤ rstdRef d init t := le_of_lt hRpos
  have hc23 : (0 : ℝ) < 2 ^ (-23 : ℤ) := by norm_num
  have hS0 := rstd_tol_rmsS_nonneg (d := d) (init := init) (t := t)
  have hAbs0 := rstd_tol_absMean_nonneg (d := d) (init := init) (t := t)
  have hcond4 : cond d init t ≤ 1 / 4 := by linarith
  have hcondEq : cond d init t
      = (meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t))
        * rstdRef d init t := rfl
  have h1 : meanTol d init t * rstdRef d init t
      ≤ (meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t))
        * rstdRef d init t := by
    apply mul_le_mul_of_nonneg_right _ hR0
    have h2 : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t) := by positivity
    linarith
  have h2 : (meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t))
        * rstdRef d init t = cond d init t := hcondEq.symm
  linarith [h1, h2.le, h2.ge, hcond4]

/-- On a well-conditioned row the tolerance of `rstd` is at most `(1/2 + 2^-9)·rstdRef`. -/
theorem yb_A_rstdTol_le {d : Inst} (hd : d.ok) {init : Mem} {t : Nat}
    (hA : 4 * cond d init t ≤ 1) :
    rstdTol d init t ≤ (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * rstdRef d init t := by
  have hRpos := rstd_tol_R_pos hd (init := init) (t := t)
  have hR0 : (0 : ℝ) ≤ rstdRef d init t := le_of_lt hRpos
  have hcond4 : cond d init t ≤ 1 / 4 := by linarith
  have hg := yb_gam_le hd
  have hc4 : cond d init t + gam d ≤ 1 / 4 + (2 : ℝ) ^ (-10 : ℤ) := add_le_add hcond4 hg
  have h4 : 2 * (cond d init t + gam d) ≤ 2 * (1 / 4 + (2 : ℝ) ^ (-10 : ℤ)) := by linarith
  have h5 : 2 * (1 / 4 + (2 : ℝ) ^ (-10 : ℤ)) = 1 / 2 + (2 : ℝ) ^ (-9 : ℤ) := by norm_num
  unfold rstdTol
  rw [ite_eq_left hA]
  calc 2 * (cond d init t + gam d) * rstdRef d init t
      ≤ (2 * (1 / 4 + (2 : ℝ) ^ (-10 : ℤ))) * rstdRef d init t :=
        mul_le_mul_of_nonneg_right h4 hR0
    _ = (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * rstdRef d init t := by rw [h5]

/-- On a well-conditioned row the four product terms sum to at most `46`. -/
theorem yb_A_sum_bd {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hA : 4 * cond d init t ≤ 1) {bmean : F32}
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t) {e : ℝ}
    (he : |e| ≤ meanTol d init t
      + u32 * (|sR d init t j| + |bmean.toReal|) + 4 * eta32) :
    |sR d init t j - meanRef d init t| * rstdRef d init t
      + |sR d init t j - meanRef d init t| * rstdTol d init t
      + |e| * rstdRef d init t + |e| * rstdTol d init t ≤ 46 := by
  have hRpos := rstd_tol_R_pos hd (init := init) (t := t)
  have hR0 : (0 : ℝ) ≤ rstdRef d init t := le_of_lt hRpos
  have hRle := rstd_tol_R_le hd (init := init) (t := t)
  have hu32 : (0 : ℝ) ≤ u32 := le_of_lt u32_pos
  have hAR := yb_dev_mul_R_le (init := init) (t := t) hd hj
  have hA21 := yb_A_absMean_mul_R hd (init := init) (t := t) hA
  have hmR := yb_A_meanTol_mul_R hd (init := init) (t := t) hA
  have hTA := yb_A_rstdTol_le hd (init := init) (t := t) hA
  have hmeanRefR : |meanRef d init t| * rstdRef d init t ≤ (2 : ℝ) ^ 21 := by
    have h1 : |meanRef d init t| ≤ absMean d init t := yb_meanRef_abs_le_absMean d init t
    have h2 : |meanRef d init t| * rstdRef d init t
        ≤ absMean d init t * rstdRef d init t := mul_le_mul_of_nonneg_right h1 hR0
    linarith [h2, hA21]
  have hbmeanR : |bmean.toReal| * rstdRef d init t ≤ (2 : ℝ) ^ 21 + 1 / 4 := by
    have h1 : |bmean.toReal| ≤ |meanRef d init t| + meanTol d init t := by
      have h2 : |bmean.toReal| ≤ |meanRef d init t| + |bmean.toReal - meanRef d init t| := by
        have h3 := abs_add_le (meanRef d init t) (bmean.toReal - meanRef d init t)
        have e : meanRef d init t + (bmean.toReal - meanRef d init t) = bmean.toReal := by ring
        rw [e] at h3
        exact h3
      linarith [h2, hmean]
    have h2 : |bmean.toReal| * rstdRef d init t
        ≤ (|meanRef d init t| + meanTol d init t) * rstdRef d init t :=
      mul_le_mul_of_nonneg_right h1 hR0
    have h3 : (|meanRef d init t| + meanTol d init t) * rstdRef d init t
        = |meanRef d init t| * rstdRef d init t + meanTol d init t * rstdRef d init t := by
      ring
    linarith [h2, h3.le, h3.ge, hmeanRefR, hmR]
  have hsjR : |sR d init t j| * rstdRef d init t ≤ (2 : ℝ) ^ 21 + 23 := by
    have h1 : |sR d init t j| ≤ |sR d init t j - meanRef d init t| + |meanRef d init t| := by
      have h3 := abs_add_le (sR d init t j - meanRef d init t) (meanRef d init t)
      have e : (sR d init t j - meanRef d init t) + meanRef d init t = sR d init t j := by ring
      rw [e] at h3
      exact h3
    have h2 : |sR d init t j| * rstdRef d init t
        ≤ (|sR d init t j - meanRef d init t| + |meanRef d init t|) * rstdRef d init t :=
      mul_le_mul_of_nonneg_right h1 hR0
    have h3 : (|sR d init t j - meanRef d init t| + |meanRef d init t|) * rstdRef d init t
        = |sR d init t j - meanRef d init t| * rstdRef d init t
          + |meanRef d init t| * rstdRef d init t := by ring
    linarith [h2, h3.le, h3.ge, hAR, hmeanRefR]
  have heR : |e| * rstdRef d init t ≤ 1 := by
    have h1 : |e| * rstdRef d init t
        ≤ (meanTol d init t + u32 * (|sR d init t j| + |bmean.toReal|) + 4 * eta32)
          * rstdRef d init t := mul_le_mul_of_nonneg_right he hR0
    have h2 : (meanTol d init t + u32 * (|sR d init t j| + |bmean.toReal|) + 4 * eta32)
          * rstdRef d init t
        = meanTol d init t * rstdRef d init t
          + u32 * (|sR d init t j| * rstdRef d init t
            + |bmean.toReal| * rstdRef d init t)
          + 4 * eta32 * rstdRef d init t := by ring
    have h3 : u32 * (|sR d init t j| * rstdRef d init t + |bmean.toReal| * rstdRef d init t)
        ≤ u32 * ((2 : ℝ) ^ 21 + 23 + ((2 : ℝ) ^ 21 + 1 / 4)) := by
      apply mul_le_mul_of_nonneg_left _ hu32
      linarith [hsjR, hbmeanR]
    have h4 : u32 * ((2 : ℝ) ^ 21 + 23 + ((2 : ℝ) ^ 21 + 1 / 4)) ≤ 1 / 2 := by
      unfold u32; norm_num
    have h5 : 4 * eta32 * rstdRef d init t ≤ 1 / 4 := by
      have h6 : 4 * eta32 * rstdRef d init t ≤ 4 * eta32 * (2 : ℝ) ^ (63 : ℤ) :=
        mul_le_mul_of_nonneg_left hRle (by unfold eta32; norm_num)
      have h7 : 4 * eta32 * (2 : ℝ) ^ (63 : ℤ) ≤ 1 / 4 := by unfold eta32; norm_num
      linarith [h6, h7]
    linarith [h1, h2.le, h2.ge, h3, h4, h5, hmR]
  have heT : |e| * rstdTol d init t ≤ (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) := by
    have hE0 : (0 : ℝ) ≤ |e| := abs_nonneg _
    calc |e| * rstdTol d init t
        ≤ |e| * ((1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * rstdRef d init t) :=
          mul_le_mul_of_nonneg_left hTA hE0
      _ = (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * (|e| * rstdRef d init t) := by ring
      _ ≤ (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * 1 :=
          mul_le_mul_of_nonneg_left heR (by norm_num)
      _ = (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) := by ring
  have hAT : |sR d init t j - meanRef d init t| * rstdTol d init t
      ≤ 23 * (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) := by
    have hA0 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t| := abs_nonneg _
    calc |sR d init t j - meanRef d init t| * rstdTol d init t
        ≤ |sR d init t j - meanRef d init t|
          * ((1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * rstdRef d init t) :=
          mul_le_mul_of_nonneg_left hTA hA0
      _ = (1 / 2 + (2 : ℝ) ^ (-9 : ℤ))
          * (|sR d init t j - meanRef d init t| * rstdRef d init t) := by ring
      _ ≤ (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * 23 :=
          mul_le_mul_of_nonneg_left hAR (by norm_num)
      _ = 23 * (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) := by ring
  have h46 : (23 : ℝ) + 23 * (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) + 1
      + (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) ≤ 46 := by norm_num
  linarith [hAR, hAT, heR, heT, h46]


/-- The ill-conditioned tolerance `2·eps^(-1/2)` is at most `2^64`. -/
theorem yb_ill_tol {d : Inst} (hd : d.ok) : 2 / Real.sqrt (epsR d) ≤ (2 : ℝ) ^ (64 : ℤ) := by
  have heps := epsR_ge hd
  have h1 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 ≤ epsR d := by
    have h2 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = (2 : ℝ) ^ (-126 : ℤ) := by
      rw [pow_two, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
    have h3 : (2 : ℝ) ^ (-126 : ℤ) ≤ (2 : ℝ) ^ (-125 : ℤ) := by norm_num
    linarith [h2.le, h2.ge, h3, heps]
  have h2 : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt (epsR d) := Real.le_sqrt_of_sq_le h1
  have h3 : (0 : ℝ) < (2 : ℝ) ^ (-63 : ℤ) := by norm_num
  have h4 : 1 / Real.sqrt (epsR d) ≤ 1 / (2 : ℝ) ^ (-63 : ℤ) :=
    one_div_le_one_div_of_le h3 h2
  have h5 : 1 / (2 : ℝ) ^ (-63 : ℤ) = (2 : ℝ) ^ (63 : ℤ) := by norm_num
  have h6 : 2 / Real.sqrt (epsR d) = 2 * (1 / Real.sqrt (epsR d)) := by ring
  have h7 : 2 * (1 / Real.sqrt (epsR d)) ≤ 2 * (2 : ℝ) ^ (63 : ℤ) := by
    linarith [h4, h5.le, h5.ge]
  have h8 : 2 * (2 : ℝ) ^ (63 : ℤ) = (2 : ℝ) ^ (64 : ℤ) := by norm_num
  linarith [h6.le, h6.ge, h7, h8.le, h8.ge]

/-- Ill-conditioned row: the four product terms sum to at most `25·2^105`. -/
theorem yb_ill_sum_bd {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    (hbddj : |sR d init t j| ≤ (2 : ℝ) ^ 57) {bmean : F32}
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t) {e : ℝ}
    (he : |e| ≤ u32 * max |sR d init t j| |bmean.toReal| + 4 * eta32)
    (hB : 1 < 4 * cond d init t) :
    |sR d init t j - meanRef d init t| * rstdRef d init t
      + |sR d init t j - meanRef d init t| * rstdTol d init t
      + |(meanRef d init t - bmean.toReal) + e| * rstdRef d init t
      + |(meanRef d init t - bmean.toReal) + e| * rstdTol d init t
      ≤ 25 * (2 : ℝ) ^ (105 : ℤ) := by
  have hRpos := rstd_tol_R_pos hd (init := init) (t := t)
  have hRle := rstd_tol_R_le hd (init := init) (t := t)
  have hdevR := yb_dev_mul_R_le (init := init) (t := t) hd hj
  have hdev := yb_ill_dev_le hd (init := init) (t := t) hj hbddj hB
  have heps := yb_ill_eps_le hd (init := init) (t := t) hj hbddj hmean hB he
  have hT : rstdTol d init t = 2 / Real.sqrt (epsR d) := by
    unfold rstdTol
    rw [ite_eq_right (by intro hc; linarith)]
  have hTle : rstdTol d init t ≤ (2 : ℝ) ^ (64 : ℤ) := by
    rw [hT]; exact yb_ill_tol hd
  have hT0 : (0 : ℝ) ≤ rstdTol d init t := by
    rw [hT]; positivity
  have hdevT : |sR d init t j - meanRef d init t| * rstdTol d init t ≤ 23 * (2 : ℝ) ^ (105 : ℤ) := by
    have h1 : |sR d init t j - meanRef d init t| * rstdTol d init t
        ≤ (23 * (2 : ℝ) ^ (41 : ℤ)) * (2 : ℝ) ^ (64 : ℤ) :=
      mul_le_mul hdev hTle hT0 (by positivity)
    have h2 : (23 : ℝ) * (2 : ℝ) ^ (41 : ℤ) * (2 : ℝ) ^ (64 : ℤ)
        = 23 * (2 : ℝ) ^ (105 : ℤ) := by
      rw [mul_assoc, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
    linarith [h1, h2.le, h2.ge]
  have hepsR : |(meanRef d init t - bmean.toReal) + e| * rstdRef d init t
      ≤ (2 : ℝ) ^ (104 : ℤ) := by
    have h1 : |(meanRef d init t - bmean.toReal) + e| * rstdRef d init t
        ≤ (2 : ℝ) ^ (41 : ℤ) * (2 : ℝ) ^ (63 : ℤ) :=
      mul_le_mul heps hRle (le_of_lt hRpos) (by positivity)
    have h2 : (2 : ℝ) ^ (41 : ℤ) * (2 : ℝ) ^ (63 : ℤ) = (2 : ℝ) ^ (104 : ℤ) := by
      rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
    linarith [h1, h2.le, h2.ge]
  have hepsT : |(meanRef d init t - bmean.toReal) + e| * rstdTol d init t
      ≤ (2 : ℝ) ^ (105 : ℤ) := by
    have h1 : |(meanRef d init t - bmean.toReal) + e| * rstdTol d init t
        ≤ (2 : ℝ) ^ (41 : ℤ) * (2 : ℝ) ^ (64 : ℤ) :=
      mul_le_mul heps hTle hT0 (by positivity)
    have h2 : (2 : ℝ) ^ (41 : ℤ) * (2 : ℝ) ^ (64 : ℤ) = (2 : ℝ) ^ (105 : ℤ) := by
      rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      norm_num
    linarith [h1, h2.le, h2.ge]
  have hnum : (23 : ℝ) + 23 * (2 : ℝ) ^ (105 : ℤ) + (2 : ℝ) ^ (105 : ℤ)
      + (2 : ℝ) ^ (104 : ℤ) ≤ 25 * (2 : ℝ) ^ (105 : ℤ) := by norm_num
  linarith [hdevR, hdevT, hepsR, hepsT, hnum]

/-- The shared final step: the product of `bx` and `brstd` is bounded by `(1 + 2u)` times the
four-term sum. -/
theorem yb_prod_bd {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (_hj : j < d.D)
    {bx bmean brstd : F32}
    (hrstd : |brstd.toReal - rstdRef d init t| ≤ rstdTol d init t)
    {c e : ℝ} (hc : |c| ≤ 2 * u32)
    (hxeq : bx.toReal = (sR d init t j - meanRef d init t
        + ((meanRef d init t - bmean.toReal) + e)) * (1 + c))
    {B : ℝ}
    (hB : (1 + 2 * u32) * B ≤ (2 : ℝ) ^ (110 : ℤ))
    (hsum : |sR d init t j - meanRef d init t| * rstdRef d init t
      + |sR d init t j - meanRef d init t| * rstdTol d init t
      + |(meanRef d init t - bmean.toReal) + e| * rstdRef d init t
      + |(meanRef d init t - bmean.toReal) + e| * rstdTol d init t ≤ B) :
    |bx.toReal| * |brstd.toReal| ≤ (2 : ℝ) ^ (110 : ℤ) := by
  have hRpos := rstd_tol_R_pos hd (init := init) (t := t)
  have hbr : |brstd.toReal| ≤ rstdRef d init t + rstdTol d init t := by
    have h1 : |brstd.toReal| ≤ |brstd.toReal - rstdRef d init t| + |rstdRef d init t| := by
      have h := abs_add_le (brstd.toReal - rstdRef d init t) (rstdRef d init t)
      have e2 : brstd.toReal - rstdRef d init t + rstdRef d init t = brstd.toReal := by ring
      rw [e2] at h
      exact h
    have h3 : |rstdRef d init t| = rstdRef d init t := abs_of_pos hRpos
    linarith [h1, hrstd, h3.le, h3.ge]
  have hbxbd : |bx.toReal| ≤ (|sR d init t j - meanRef d init t|
      + |(meanRef d init t - bmean.toReal) + e|) * (1 + 2 * u32) := by
    have h1 : |1 + c| ≤ 1 + 2 * u32 := by
      have h2 : |1 + c| ≤ 1 + |c| := by
        simpa using abs_add_le (1 : ℝ) c
      linarith [hc]
    have h3 : |sR d init t j - meanRef d init t + ((meanRef d init t - bmean.toReal) + e)|
        ≤ |sR d init t j - meanRef d init t|
          + |(meanRef d init t - bmean.toReal) + e| := abs_add_le _ _
    have h4 : (0 : ℝ) ≤ |1 + c| := abs_nonneg _
    have h5 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t|
        + |(meanRef d init t - bmean.toReal) + e| := by
      have h6 : (0 : ℝ) ≤ |sR d init t j - meanRef d init t| := abs_nonneg _
      have h7 : (0 : ℝ) ≤ |(meanRef d init t - bmean.toReal) + e| := abs_nonneg _
      linarith
    rw [hxeq, abs_mul]
    exact mul_le_mul h3 h1 h4 h5
  have hprod1 : |bx.toReal| * |brstd.toReal| ≤ (1 + 2 * u32) * B := by
    have h1 : |bx.toReal| * |brstd.toReal|
        ≤ ((|sR d init t j - meanRef d init t|
            + |(meanRef d init t - bmean.toReal) + e|) * (1 + 2 * u32))
          * (rstdRef d init t + rstdTol d init t) :=
      mul_le_mul hbxbd hbr (abs_nonneg _) (by
        have h9 : (0 : ℝ) ≤ 1 + 2 * u32 := by linarith [u32_pos]
        exact mul_nonneg (by positivity) h9)
    have h2 : ((|sR d init t j - meanRef d init t|
          + |(meanRef d init t - bmean.toReal) + e|) * (1 + 2 * u32))
        * (rstdRef d init t + rstdTol d init t)
        = (1 + 2 * u32) *
          (|sR d init t j - meanRef d init t| * rstdRef d init t
            + |sR d init t j - meanRef d init t| * rstdTol d init t
            + |(meanRef d init t - bmean.toReal) + e| * rstdRef d init t
            + |(meanRef d init t - bmean.toReal) + e| * rstdTol d init t) := by ring
    have h3 : (1 + 2 * u32) *
          (|sR d init t j - meanRef d init t| * rstdRef d init t
            + |sR d init t j - meanRef d init t| * rstdTol d init t
            + |(meanRef d init t - bmean.toReal) + e| * rstdRef d init t
            + |(meanRef d init t - bmean.toReal) + e| * rstdTol d init t)
        ≤ (1 + 2 * u32) * B := by
      apply mul_le_mul_of_nonneg_left hsum
      linarith [u32_pos]
    linarith [h1, h2.le, h2.ge, h3]
  linarith [hprod1, hB]

/-- When the addition's operands are too large for the split bound, the row is badly scaled and
the product is nevertheless small. -/
theorem yb_gate_fail_bd {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (_hj : j < d.D)
    (hA : 4 * cond d init t ≤ 1) {bx0 bmean bx brstd : F32}
    (hval : bx0.toReal = sR d init t j) (hbddj : |sR d init t j| ≤ (2 : ℝ) ^ 57)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hrstd : |brstd.toReal - rstdRef d init t| ≤ rstdTol d init t)
    (hbxfin : bx.isFinite = true)
    (hgate : ¬ ((|bx0.toRat| + |(FloatV.neg bmean).toRat| : ℚ)
      ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta)) :
    |bx.toReal| * |brstd.toReal| ≤ (2 : ℝ) ^ (110 : ℤ) := by
  have hRpos := rstd_tol_R_pos hd (init := init) (t := t)
  have hR0 : (0 : ℝ) ≤ rstdRef d init t := le_of_lt hRpos
  -- the gate, in `ℝ`
  have hgateR : ((Format.binary32.maxVal : ℚ) : ℝ) / 2 - 2 * ((Format.binary32.eta : ℚ) : ℝ)
      < |bx0.toReal| + |(FloatV.neg bmean).toReal| := by
    have h1 : ((Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta : ℚ)
        < |bx0.toRat| + |(FloatV.neg bmean).toRat| := not_le.mp hgate
    have h4 : (((Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta : ℚ) : ℝ)
        < (((|bx0.toRat| + |(FloatV.neg bmean).toRat| : ℚ)) : ℝ) := by exact_mod_cast h1
    have h2 : (((Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta : ℚ) : ℝ)
        = ((Format.binary32.maxVal : ℚ) : ℝ) / 2 - 2 * ((Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      ring
    have h3 : (((|bx0.toRat| + |(FloatV.neg bmean).toRat| : ℚ)) : ℝ)
        = |bx0.toReal| + |(FloatV.neg bmean).toReal| := by
      push_cast
      rfl
    linarith [h2.le, h2.ge, h3.le, h3.ge, h4]
  have hx0 : |bx0.toReal| ≤ (2 : ℝ) ^ 57 := by rw [hval]; exact hbddj
  have hneg : (FloatV.neg bmean).toReal = -bmean.toReal := by
    simp [FloatV.toReal, FloatV.toRat_neg]
  have hbig : (2 : ℝ) ^ 125 ≤ |bmean.toReal| := by
    have h1 : |(FloatV.neg bmean).toReal| = |bmean.toReal| := by rw [hneg, abs_neg]
    have hnum : (2 : ℝ) ^ 125 + (2 : ℝ) ^ 57
        ≤ ((Format.binary32.maxVal : ℚ) : ℝ) / 2 - 2 * ((Format.binary32.eta : ℚ) : ℝ) := by
      rw [Format.binary32_maxVal, Format.binary32_eta]
      push_cast
      norm_num
    linarith [hgateR, hx0, hnum, h1.le, h1.ge]
  -- `absMean + meanTol` is at least `2^125`
  have hbeta : |bmean.toReal| ≤ absMean d init t + meanTol d init t := by
    have h1 : |bmean.toReal| ≤ |bmean.toReal - meanRef d init t| + |meanRef d init t| := by
      have h2 := abs_add_le (bmean.toReal - meanRef d init t) (meanRef d init t)
      have e : bmean.toReal - meanRef d init t + meanRef d init t = bmean.toReal := by ring
      rwa [e] at h2
    have h3 := yb_meanRef_abs_le_absMean d init t
    linarith
  have hA21 := yb_A_absMean_mul_R hd (init := init) (t := t) hA
  have hmR := yb_A_meanTol_mul_R hd (init := init) (t := t) hA
  have hRle : rstdRef d init t ≤ (2 : ℝ) ^ (-103 : ℤ) := by
    have h1 : (2 : ℝ) ^ 125 * rstdRef d init t
        ≤ (absMean d init t + meanTol d init t) * rstdRef d init t :=
      mul_le_mul_of_nonneg_right (by linarith [hbig, hbeta]) hR0
    have h2 : (absMean d init t + meanTol d init t) * rstdRef d init t
        = absMean d init t * rstdRef d init t + meanTol d init t * rstdRef d init t := by ring
    have h3 : (2 : ℝ) ^ 125 * rstdRef d init t ≤ (2 : ℝ) ^ 22 := by
      have h4 : absMean d init t * rstdRef d init t + meanTol d init t * rstdRef d init t
          ≤ (2 : ℝ) ^ 22 := by
        have h5 : (2 : ℝ) ^ 21 + 1 / 4 ≤ (2 : ℝ) ^ 22 := by norm_num
        linarith [hA21, hmR, h5]
      linarith [h1, h2.le, h2.ge, h4]
    have h6 : (0 : ℝ) < (2 : ℝ) ^ 125 := by norm_num
    have h7 : rstdRef d init t ≤ (2 : ℝ) ^ 22 / (2 : ℝ) ^ 125 :=
      (le_div_iff₀ h6).mpr (by rw [mul_comm]; exact h3)
    have h8 : (2 : ℝ) ^ 22 / (2 : ℝ) ^ 125 = (2 : ℝ) ^ (-103 : ℤ) := by norm_num
    calc rstdRef d init t ≤ (2 : ℝ) ^ 22 / (2 : ℝ) ^ 125 := h7
      _ = (2 : ℝ) ^ (-103 : ℤ) := h8
  -- the tolerance is smaller than `rstdRef`
  have hTA := yb_A_rstdTol_le hd (init := init) (t := t) hA
  have hT0 : (0 : ℝ) ≤ rstdTol d init t := by
    unfold rstdTol
    rw [ite_eq_left hA]
    have hc0 : (0 : ℝ) ≤ cond d init t := by
      have hMT := rstd_tol_meanTol_nonneg (d := d) (init := init) (t := t)
      have hAb := rstd_tol_absMean_nonneg (d := d) (init := init) (t := t)
      have hSb := rstd_tol_rmsS_nonneg (d := d) (init := init) (t := t)
      have hcondEq : cond d init t
          = (meanTol d init t + 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t))
            * rstdRef d init t := rfl
      rw [hcondEq]
      apply mul_nonneg _ hR0
      have h0 : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) * (absMean d init t + rmsS d init t) := by
        have h1 : (0 : ℝ) ≤ 2 ^ (-23 : ℤ) := by norm_num
        exact mul_nonneg h1 (by linarith)
      linarith
    have hg0 : (0 : ℝ) ≤ gam d := by
      unfold gam
      have h1 : (0 : ℝ) ≤ ((depth d : ℝ) + 10) * (2 : ℝ) ^ (-23 : ℤ) := by
        have h2 : (0 : ℝ) ≤ (depth d : ℝ) + 10 := by positivity
        exact mul_nonneg h2 (by norm_num)
      have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (-136 : ℤ) / epsR d :=
        div_nonneg (by norm_num) (le_of_lt (epsR_pos hd))
      exact add_nonneg h1 h2
    have h1 : (0 : ℝ) ≤ cond d init t + gam d := by linarith
    exact mul_nonneg (mul_nonneg (by norm_num) h1) hR0
  have hTle : rstdTol d init t ≤ (2 : ℝ) ^ (-103 : ℤ) := by
    have h1 : (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) ≤ 1 := by norm_num
    calc rstdTol d init t ≤ (1 / 2 + (2 : ℝ) ^ (-9 : ℤ)) * rstdRef d init t := hTA
      _ ≤ 1 * rstdRef d init t := mul_le_mul_of_nonneg_right h1 hR0
      _ = rstdRef d init t := by ring
      _ ≤ (2 : ℝ) ^ (-103 : ℤ) := hRle
  have hbr : |brstd.toReal| ≤ rstdRef d init t + rstdTol d init t := by
    have h1 : |brstd.toReal| ≤ |brstd.toReal - rstdRef d init t| + |rstdRef d init t| := by
      have h := abs_add_le (brstd.toReal - rstdRef d init t) (rstdRef d init t)
      have e2 : brstd.toReal - rstdRef d init t + rstdRef d init t = brstd.toReal := by ring
      rw [e2] at h
      exact h
    have h3 : |rstdRef d init t| = rstdRef d init t := abs_of_pos hRpos
    linarith [h1, hrstd, h3.le, h3.ge]
  have hbxle : |bx.toReal| ≤ (2 : ℝ) ^ 128 := by
    have h1 := FloatV.abs_toRat_le_maxVal (f := binary32) binary32_regular hbxfin
    have h2 : ((Format.binary32.maxVal : ℚ) : ℝ)
        = ((2 : ℝ) ^ 24 - 1) * (2 : ℝ) ^ 104 := by
      rw [Format.binary32_maxVal]
      push_cast
      norm_num
    have h3 : |(bx.toRat : ℝ)| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by exact_mod_cast h1
    have h4 : |(bx.toRat : ℝ)| = |bx.toReal| := rfl
    rw [h4] at h3
    nlinarith [h2.le, h2.ge, h3]
  have hnum2 : (2 : ℝ) ^ 128 * ((2 : ℝ) ^ (-103 : ℤ) + (2 : ℝ) ^ (-103 : ℤ))
      ≤ (2 : ℝ) ^ (110 : ℤ) := by norm_num
  have h1 : |brstd.toReal| ≤ (2 : ℝ) ^ (-103 : ℤ) + (2 : ℝ) ^ (-103 : ℤ) := by linarith
  have h2 := mul_le_mul hbxle h1 (abs_nonneg _) (by positivity)
  linarith [h2, hnum2]

/-- The magnitude of the deviation times the reciprocal standard deviation. -/
theorem y_bhat_bd {d : Inst} (hd : d.ok) {init : Mem} {t j : Nat} (hj : j < d.D)
    {bx0 bmean bx brstd : F32}
    (hval : bx0.toReal = sR d init t j) (hbx0fin : bx0.isFinite = true)
    (hbdd : |sR d init t j| ≤ 2 ^ 57)
    (hmeanfin : bmean.isFinite = true)
    (hmean : |bmean.toReal - meanRef d init t| ≤ meanTol d init t)
    (hrstdfin : brstd.isFinite = true)
    (hrstd : |brstd.toReal - rstdRef d init t| ≤ rstdTol d init t)
    (hdev : AddRel true bx0 (FloatV.neg bmean) bx) (hbxfin : bx.isFinite = true) :
    (FloatV.mul bx brstd).isFinite = true ∧ |bx.toReal| * |brstd.toReal| ≤ 2 ^ 110 := by
  have hnegfin : (FloatV.neg bmean).isFinite = true := by
    rw [FloatV.isFinite_neg]; exact hmeanfin
  have hneg : (FloatV.neg bmean).toReal = -bmean.toReal := by
    simp [FloatV.toReal, FloatV.toRat_neg]
  have hbd : |bx.toReal| * |brstd.toReal| ≤ (2 : ℝ) ^ (110 : ℤ) := by
    by_cases hA : 4 * cond d init t ≤ 1
    · by_cases hgate : (|bx0.toRat| + |(FloatV.neg bmean).toRat| : ℚ)
          ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta
      · obtain ⟨-, c, e, hc, he, hxeq⟩ :=
          rstd_tol_addrel_split_real hdev hbx0fin hnegfin hgate
        have he' : |(meanRef d init t - bmean.toReal) + e| ≤ meanTol d init t
            + u32 * (|sR d init t j| + |bmean.toReal|) + 4 * eta32 := by
          have h1 : |(meanRef d init t - bmean.toReal) + e|
              ≤ |meanRef d init t - bmean.toReal| + |e| := abs_add_le _ _
          have h2 : |meanRef d init t - bmean.toReal| ≤ meanTol d init t := by
            rw [abs_sub_comm]; exact hmean
          have h3 : |e| ≤ u32 * max |bx0.toReal| |(FloatV.neg bmean).toReal| + 4 * eta32 := he
          have h4 : max |bx0.toReal| |(FloatV.neg bmean).toReal|
              ≤ |sR d init t j| + |bmean.toReal| := by
            rw [hneg, abs_neg, hval]
            exact max_le (le_add_of_nonneg_right (abs_nonneg _))
              (le_add_of_nonneg_left (abs_nonneg _))
          have h5 : u32 * max |bx0.toReal| |(FloatV.neg bmean).toReal|
              ≤ u32 * (|sR d init t j| + |bmean.toReal|) :=
            mul_le_mul_of_nonneg_left h4 (le_of_lt u32_pos)
          linarith [h1, h2, h3, h5]
        have hxeq' : bx.toReal = (sR d init t j - meanRef d init t
            + ((meanRef d init t - bmean.toReal) + e)) * (1 + c) := by
          rw [hxeq, hneg, hval]; ring
        have hsum := yb_A_sum_bd hd hj hA hmean he'
        have hB : (1 + 2 * u32) * 46 ≤ (2 : ℝ) ^ (110 : ℤ) := by
          unfold u32; norm_num
        exact yb_prod_bd hd hj hrstd hc hxeq' hB hsum
      · exact yb_gate_fail_bd hd hj hA hval hbdd hmean hrstd hbxfin hgate
    · have hB : 1 < 4 * cond d init t := by linarith
      have hM : (|bx0.toRat| + |(FloatV.neg bmean).toRat| : ℚ)
          ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
        have h1 : |bx0.toRat| ≤ (2 : ℚ) ^ 57 := by
          have h2 : |(bx0.toRat : ℝ)| ≤ (2 : ℝ) ^ 57 := by
            have h3 : |bx0.toReal| ≤ (2 : ℝ) ^ 57 := by rw [hval]; exact hbdd
            simpa [FloatV.toReal] using h3
          exact_mod_cast h2
        have h2 : |(FloatV.neg bmean).toRat| ≤ (2 : ℚ) ^ 59 := by
          rw [FloatV.toRat_neg, abs_neg]
          have h3 : |(bmean.toRat : ℝ)| ≤ (2 : ℝ) ^ 59 := by
            have hb : (bmean.toRat : ℝ) = bmean.toReal := rfl
            rw [hb]
            exact yb_ill_bmean_le hd hj hbdd hmean hB
          exact_mod_cast h3
        have hnum : (2 : ℚ) ^ 57 + (2 : ℚ) ^ 59
            ≤ (Format.binary32.maxVal : ℚ) / 2 - 2 * Format.binary32.eta := by
          rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
        linarith
      obtain ⟨-, c, e, hc, he, hxeq⟩ :=
        rstd_tol_addrel_split_real hdev hbx0fin hnegfin hM
      have hxeq' : bx.toReal = (sR d init t j - meanRef d init t
          + ((meanRef d init t - bmean.toReal) + e)) * (1 + c) := by
        rw [hxeq, hneg, hval]; ring
      have hmaxeq : max |bx0.toReal| |(FloatV.neg bmean).toReal|
          = max |sR d init t j| |bmean.toReal| := by
        rw [hneg, abs_neg, hval]
      have he' : |e| ≤ u32 * max |sR d init t j| |bmean.toReal| + 4 * eta32 := by
        rw [← hmaxeq]; exact he
      have hsum := yb_ill_sum_bd hd hj hbdd hmean he' hB
      have hB2 : (1 + 2 * u32) * (25 * (2 : ℝ) ^ (105 : ℤ)) ≤ (2 : ℝ) ^ (110 : ℤ) := by
        unfold u32; norm_num
      exact yb_prod_bd hd hj hrstd hc hxeq' hB2 hsum
  have hmaxR : |(bx.toRat : ℝ) * (brstd.toRat : ℝ)| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    have h1 : |(bx.toRat : ℝ) * (brstd.toRat : ℝ)| = |bx.toReal| * |brstd.toReal| := by
      show |(bx.toRat : ℝ) * (brstd.toRat : ℝ)|
        = |(bx.toRat : ℝ)| * |(brstd.toRat : ℝ)|
      rw [abs_mul]
    rw [h1]
    have h2 : (2 : ℝ) ^ (110 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [Format.binary32_maxVal]
      norm_num
    linarith [hbd, h2]
  have hmaxQ : |bx.toRat * brstd.toRat| ≤ binary32.maxVal := by
    have hcast : ((|bx.toRat * brstd.toRat| : ℚ) : ℝ) ≤ ((binary32.maxVal : ℚ) : ℝ) := by
      push_cast
      exact hmaxR
    exact_mod_cast hcast
  obtain ⟨hfin, -⟩ :=
    FloatV.mul_err (f := binary32) binary32_regular hbxfin hrstdfin hmaxQ
  exact ⟨hfin, by simpa using hbd⟩

end

end Claims.LayerNormFwd
