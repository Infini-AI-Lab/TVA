import Proof.RowMean
import Proof.RowNorm

/-!
# The row's variance and standard deviation

The centered lanes `c` differ from `g − mean0` by the fusable subtractions' errors `e`. With the
Euclidean norm over the row, `‖g − mean0‖ = √d·√(sd² + (mean0 − mu)²)` and `‖e‖ ≤ √d·(3u·(sd +
|mu|) + 3u·|mean0| + 4η)`, so `√T = ‖c‖/√d` (`T = Σ c²/d`) is within `R1 = 3u·sd + 10u·aMean +
2^-136` of `sd` (`rho_bounds`). The squares, the fusable tree and `/ d` keep the variance within
`51u·T + 2^-146` of `T` (`var_T`), and `var_stage'` and `std_stage'` conclude.
-/

namespace Claims.GeluAndMulSparse

open TVA

namespace RowP

variable (r : RowP)

/-- `Σ_{j<d} (g j − a0)² = d·(sd² + (a0 − mu)²)`. -/
theorem sum_dev_sq (a0 : ℝ) :
    ∑ j ∈ Finset.range r.d, (r.g j - a0) ^ 2 = r.d * (r.sd ^ 2 + (a0 - r.mu) ^ 2) := by
  have hd := r.d_pos'
  have hsd : ∑ j ∈ Finset.range r.d, (r.g j - r.mu) ^ 2 = r.d * r.sd ^ 2 := by
    rw [r.sd_sq]; field_simp
  have hz : ∑ j ∈ Finset.range r.d, (r.g j - r.mu) = 0 := by
    rw [Finset.sum_sub_distrib, r.sum_eq]; simp
  have e : ∀ j, (r.g j - a0) ^ 2 = (r.g j - r.mu) ^ 2 - 2 * (a0 - r.mu) * (r.g j - r.mu) +
      (a0 - r.mu) ^ 2 := fun j => by ring
  simp only [e, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hz, hsd,
    Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  ring

/-- `√(sd² + δ²) ≤ sd + δ²/(2·sd)` for `sd > 0`. -/
theorem sqrt_sq_add_le {s δ : ℝ} (hs : 0 < s) : Real.sqrt (s ^ 2 + δ ^ 2) ≤ s + δ ^ 2 / (2 * s) := by
  rw [Real.sqrt_le_left (by positivity)]
  have : (s + δ ^ 2 / (2 * s)) ^ 2 = s ^ 2 + δ ^ 2 + (δ ^ 2 / (2 * s)) ^ 2 := by
    field_simp; ring
  rw [this]; nlinarith [sq_nonneg (δ ^ 2 / (2 * s))]

/-- A family vanishing on `[d, 2^L)` sums over the block as over `d`. -/
theorem sum_block' {f : ℕ → ℝ} (hf : ∀ j, r.d ≤ j → j < 2 ^ r.L → f j = 0) :
    ∑ j ∈ Finset.range (2 ^ r.L), f j = ∑ j ∈ Finset.range r.d, f j := by
  refine (Finset.sum_subset (Finset.range_subset_range.mpr r.d_le) fun j hj hj' => ?_).symm
  exact hf j (by simpa using hj') (Finset.mem_range.mp hj)

/-- The arithmetic of `rho_bounds`: `nc` within `ne` of `sq·nn`, `nn ∈ [sd, sd + Y]`, `ne ≤ sq·X`. -/
theorem norms_arith {nc nn ne sq sd X Y R1 : ℝ} (hsq : 0 < sq) (htri : |nc - sq * nn| ≤ ne)
    (hne : ne ≤ sq * X) (hnn1 : sd ≤ nn) (hnn2 : nn ≤ sd + Y) (hY0 : 0 ≤ Y)
    (hsum : Y + X ≤ R1) : |nc / sq - sd| ≤ R1 := by
  have h1 := (abs_le.mp htri)
  have e : nc / sq - sd = (nc - sq * sd) / sq := by field_simp
  rw [e, abs_div, abs_of_pos hsq, div_le_iff₀ hsq, abs_le]
  have a1 : sq * sd ≤ sq * nn := mul_le_mul_of_nonneg_left hnn1 hsq.le
  have a2 : sq * nn ≤ sq * (sd + Y) := mul_le_mul_of_nonneg_left hnn2 hsq.le
  have a3 : sq * (Y + X) ≤ sq * R1 := mul_le_mul_of_nonneg_left hsum hsq.le
  have a4 : 0 ≤ sq * Y := mul_nonneg hsq.le hY0
  constructor <;> nlinarith

/-- **`√T` against `sd`**: `T = Σ_{j<d} c²/d` is within `R1` of `sd` in square root. -/
theorem rho_bounds (hok : r.Ok) {mean0 : ℝ} (hm0 : |mean0 - r.mu| ≤ r.meanErr) {c : ℕ → ℝ}
    (hc : ∀ j, j < r.d → |c j - (r.g j - mean0)| ≤ 3 * u32 * (|r.g j| + |mean0|) + 4 * eta32) :
    |Real.sqrt ((∑ j ∈ Finset.range r.d, c j ^ 2) / r.d) - r.sd| ≤
      3 * u32 * r.sd + 10 * u32 * r.aMean + 2 ^ (-136 : ℤ) := by
  have hd := r.d_pos'
  have hsdg := hok.sd_ge
  have hsd0 : 0 < r.sd := lt_of_lt_of_le (by positivity) hsdg
  have hAsd := hok.aMean_le_sd
  have hA := r.aMean_nonneg'
  have hmu := r.mu_le'
  have hme := r.meanErr_le
  have hme0 := r.meanErr_nonneg
  have hU : u32 = 1 / 16777216 := u32_eq
  have heta := eta32_le
  have heta0 := eta32_nonneg
  set D := r.meanErr
  set δ := mean0 - r.mu with hδ
  set s := Finset.range r.d
  have hsq : Real.sqrt r.d > 0 := Real.sqrt_pos.mpr hd
  -- the norms
  have hnA : Real.sqrt (∑ j ∈ s, (r.g j - mean0) ^ 2) = Real.sqrt r.d * Real.sqrt (r.sd ^ 2 + δ ^ 2) := by
    rw [r.sum_dev_sq, Real.sqrt_mul hd.le]
  have hng : Real.sqrt (∑ j ∈ s, r.g j ^ 2) ≤ Real.sqrt r.d * (r.sd + |r.mu|) := by
    have e : ∑ j ∈ s, r.g j ^ 2 = ∑ j ∈ s, (r.g j - 0) ^ 2 := by simp
    rw [e, r.sum_dev_sq, Real.sqrt_mul hd.le]
    refine mul_le_mul_of_nonneg_left ?_ (Real.sqrt_nonneg _)
    rw [Real.sqrt_le_left (by positivity)]
    have : (0 - r.mu) ^ 2 = |r.mu| ^ 2 := by rw [sq_abs]; ring
    rw [this]; nlinarith [abs_nonneg r.mu, r.sd_nonneg']
  set K := 3 * u32 * |mean0| + 4 * eta32 with hK
  have hK0 : 0 ≤ K := by rw [hK, hU]; positivity
  have hne : Real.sqrt (∑ j ∈ s, (c j - (r.g j - mean0)) ^ 2) ≤
      Real.sqrt r.d * (3 * u32 * (r.sd + |r.mu|) + K) := by
    have h1 := sqrt_sum_sq_mono s (f := fun j => c j - (r.g j - mean0))
      (p := fun j => 3 * u32 * |r.g j| + K) fun j hj => by
        have := hc j (Finset.mem_range.mp hj); rw [hK]; linarith
    have h2 := sqrt_sum_add_sq_le s (fun j => 3 * u32 * |r.g j|) (fun _ => K)
    have h3 := sqrt_sum_sq_smul s (3 * u32) (fun j => |r.g j|)
    have h4 := sqrt_sum_sq_const r.d K
    simp only [sq_abs] at h3
    rw [h3, abs_of_nonneg (by rw [hU]; norm_num)] at h2
    rw [h4, abs_of_nonneg hK0] at h2
    have h5 : 3 * u32 * Real.sqrt (∑ j ∈ s, r.g j ^ 2) ≤ 3 * u32 * (Real.sqrt r.d * (r.sd + |r.mu|)) :=
      mul_le_mul_of_nonneg_left hng (by rw [hU]; norm_num)
    have e : Real.sqrt r.d * (3 * u32 * (r.sd + |r.mu|) + K) =
        3 * u32 * (Real.sqrt r.d * (r.sd + |r.mu|)) + K * Real.sqrt r.d := by ring
    rw [e]; linarith
  have htri := abs_sqrt_sum_sq_sub_le s c (fun j => r.g j - mean0)
  -- the square root of `T`
  have hT : Real.sqrt ((∑ j ∈ s, c j ^ 2) / r.d) = Real.sqrt (∑ j ∈ s, c j ^ 2) / Real.sqrt r.d :=
    Real.sqrt_div' _ hd.le
  rw [hT]
  rw [hnA] at htri
  set nc := Real.sqrt (∑ j ∈ s, c j ^ 2)
  set ne := Real.sqrt (∑ j ∈ s, (c j - (r.g j - mean0)) ^ 2)
  set nn := Real.sqrt (r.sd ^ 2 + δ ^ 2)
  have hnn1 : r.sd ≤ nn := by
    rw [Real.le_sqrt r.sd_nonneg' (by positivity)]; nlinarith [sq_nonneg δ]
  have hnn2 := sqrt_sq_add_le (δ := δ) hsd0
  -- numbers
  have hDsd : D ≤ 5 / 16 * r.sd + 2 ^ (-134 : ℤ) := by
    have : 20 * u32 * r.aMean ≤ 20 * u32 * (2 ^ 18 * r.sd) :=
      mul_le_mul_of_nonneg_left hAsd (by rw [hU]; norm_num)
    rw [hU] at this hme; norm_num at this hme ⊢; linarith
  have hD2 : D ≤ 5 / 16 * r.sd + 2 ^ (-72 : ℤ) * r.sd := by
    have : (2 : ℝ) ^ (-134 : ℤ) ≤ 2 ^ (-72 : ℤ) * r.sd := by
      have : (2 : ℝ) ^ (-134 : ℤ) = 2 ^ (-72 : ℤ) * 2 ^ (-62 : ℤ) := by norm_num
      rw [this]; exact mul_le_mul_of_nonneg_left hsdg (by positivity)
    norm_num at hDsd this ⊢
    linarith
  have hδD : |δ| ≤ D := hm0
  have hδ2 : δ ^ 2 / (2 * r.sd) ≤ 1563 / 10000 * D := by
    rw [div_le_iff₀ (by positivity)]
    have h1 : δ ^ 2 ≤ D * D := by rw [← sq_abs]; nlinarith [abs_nonneg δ]
    have h2 : D * D ≤ D * (5 / 16 * r.sd + 2 ^ (-72 : ℤ) * r.sd) := mul_le_mul_of_nonneg_left hD2 hme0
    have h3 : (2 : ℝ) ^ (-72 : ℤ) ≤ 1 / 10000 := by norm_num
    have h4 : D * (2 ^ (-72 : ℤ) * r.sd) ≤ D * (1 / 10000 * r.sd) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right h3 r.sd_nonneg') hme0
    nlinarith
  have hm0b : |mean0| ≤ r.aMean + D := by
    have := abs_sub_abs_le_abs_sub mean0 r.mu; linarith
  have hKb : K ≤ 3 * u32 * (r.aMean + D) + 4 * eta32 := by
    rw [hK]; have := mul_le_mul_of_nonneg_left hm0b (show (0 : ℝ) ≤ 3 * u32 by rw [hU]; norm_num)
    linarith
  have hsum : δ ^ 2 / (2 * r.sd) + (3 * u32 * (r.sd + |r.mu|) + K) ≤
      3 * u32 * r.sd + 10 * u32 * r.aMean + 2 ^ (-136 : ℤ) := by
    have h1 : 3 * u32 * |r.mu| ≤ 3 * u32 * r.aMean := mul_le_mul_of_nonneg_left hmu (by rw [hU]; norm_num)
    rw [hU] at hme hKb h1 ⊢
    norm_num at hme hKb h1 heta hδ2 ⊢
    linarith
  exact norms_arith hsq htri hne hnn1 hnn2 (by positivity) hsum

end RowP

end Claims.GeluAndMulSparse
