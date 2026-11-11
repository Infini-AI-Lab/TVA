import Proof.Row

namespace Claims.LayerNormFwdF16

namespace RowP

variable (r : RowP)

/-- **The lanes' sum of squares**, from the lanes' errors: with `κ = u(2+u)`, `δ = m − mu` and
`c = cLane m`, `Σ d² / N` is within `±(2κσ² + 2cσ)` of `σ² = sig2`, and above by at most
`(κσ + c + |δ|)²` more (`Σ (x − mu) = 0` and Cauchy–Schwarz). -/
theorem sumsq_bounds (m : ℝ) (d : ℕ → ℝ) (hd : ∀ j, j < r.N → LaneErr (r.x j) m (d j)) :
    r.N * (r.sig2 - 2 * (u32 * (2 + u32)) * r.sig2 - 2 * r.cLane m * Real.sqrt r.sig2) ≤
        ∑ j ∈ Finset.range r.N, d j ^ 2 ∧
      ∑ j ∈ Finset.range r.N, d j ^ 2 ≤
        r.N * (r.sig2 + 2 * (u32 * (2 + u32)) * r.sig2 + 2 * r.cLane m * Real.sqrt r.sig2 +
          (u32 * (2 + u32) * Real.sqrt r.sig2 + r.cLane m + |m - r.mu|) ^ 2) := by
  have hN : (0 : ℝ) < r.N := by exact_mod_cast (by have := r.n_ge; omega : 0 < r.N)
  have hσ2 : Real.sqrt r.sig2 ^ 2 = r.sig2 := Real.sq_sqrt r.sig2_nonneg
  have hσ0 : 0 ≤ Real.sqrt r.sig2 := Real.sqrt_nonneg _
  have hκ : 0 ≤ u32 * (2 + u32) := by unfold u32; positivity
  have hc : 0 ≤ r.cLane m := by unfold cLane u32 eta32; positivity
  set κ := u32 * (2 + u32) with hκdef
  set c := r.cLane m with hcdef
  set σ := Real.sqrt r.sig2 with hσdef
  set δ := m - r.mu with hδdef
  have hsa : ∑ j ∈ Finset.range r.N, (r.x j - r.mu) = 0 := by
    rw [Finset.sum_sub_distrib]
    simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    unfold mu; field_simp; ring
  have hsa2 : ∑ j ∈ Finset.range r.N, (r.x j - r.mu) ^ 2 = r.N * r.sig2 := by
    unfold sig2; field_simp
  have hsabs : ∑ j ∈ Finset.range r.N, |r.x j - r.mu| ≤ r.N * σ := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.range r.N) (fun _ => (1 : ℝ))
      (fun j => |r.x j - r.mu|)
    simp only [one_mul, sq_abs, one_pow, Finset.sum_const, Finset.card_range, nsmul_eq_mul,
      mul_one] at h
    rw [hsa2] at h
    have e : ((r.N : ℝ) * σ) ^ 2 = r.N * (r.N * r.sig2) := by rw [mul_pow, hσ2]; ring
    exact le_of_sq_le_sq (by rw [e]; exact h) (by positivity)
  set ξ : ℕ → ℝ := fun j => d j - ((r.x j - r.mu) - δ) with hξdef
  have hξ : ∀ j, j < r.N → |ξ j| ≤ κ * |r.x j - r.mu| + c := fun j hj =>
    r.lane_split (hd j hj)
  have key : ∑ j ∈ Finset.range r.N, d j ^ 2 = r.N * r.sig2 +
      2 * ∑ j ∈ Finset.range r.N, (r.x j - r.mu) * ξ j +
        ∑ j ∈ Finset.range r.N, (ξ j - δ) ^ 2 := by
    have : ∀ j ∈ Finset.range r.N, d j ^ 2 = (r.x j - r.mu) ^ 2 +
        2 * ((r.x j - r.mu) * ξ j) - 2 * δ * (r.x j - r.mu) + (ξ j - δ) ^ 2 := by
      intro j _; simp only [hξdef]; ring
    rw [Finset.sum_congr rfl this]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hsa, hsa2]
    ring
  have h1 : |∑ j ∈ Finset.range r.N, (r.x j - r.mu) * ξ j| ≤ κ * (r.N * r.sig2) + c * (r.N * σ) := by
    calc _ ≤ ∑ j ∈ Finset.range r.N, |(r.x j - r.mu) * ξ j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ Finset.range r.N, (κ * (r.x j - r.mu) ^ 2 + c * |r.x j - r.mu|) := by
          refine Finset.sum_le_sum fun j hj => ?_
          rw [abs_mul]
          have := mul_le_mul_of_nonneg_left (hξ j (Finset.mem_range.1 hj)) (abs_nonneg (r.x j - r.mu))
          have e : |r.x j - r.mu| * |r.x j - r.mu| = (r.x j - r.mu) ^ 2 := by rw [← sq, sq_abs]
          nlinarith
      _ = κ * (r.N * r.sig2) + c * ∑ j ∈ Finset.range r.N, |r.x j - r.mu| := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hsa2]
      _ ≤ _ := by nlinarith
  have h2 : ∑ j ∈ Finset.range r.N, (ξ j - δ) ^ 2 ≤ r.N * (κ * σ + c + |δ|) ^ 2 := by
    calc _ ≤ ∑ j ∈ Finset.range r.N, (κ ^ 2 * (r.x j - r.mu) ^ 2 +
          2 * κ * (c + |δ|) * |r.x j - r.mu| + (c + |δ|) ^ 2) := by
          refine Finset.sum_le_sum fun j hj => ?_
          have h3 : |ξ j - δ| ≤ κ * |r.x j - r.mu| + (c + |δ|) := by
            have := abs_sub (ξ j) δ
            have := hξ j (Finset.mem_range.1 hj)
            linarith
          have := pow_le_pow_left₀ (abs_nonneg _) h3 2
          rw [sq_abs] at this
          have e : (κ * |r.x j - r.mu| + (c + |δ|)) ^ 2 = κ ^ 2 * (r.x j - r.mu) ^ 2 +
              2 * κ * (c + |δ|) * |r.x j - r.mu| + (c + |δ|) ^ 2 := by
            rw [← sq_abs (r.x j - r.mu)]; ring
          linarith
      _ = κ ^ 2 * (r.N * r.sig2) + 2 * κ * (c + |δ|) * ∑ j ∈ Finset.range r.N, |r.x j - r.mu| +
            r.N * (c + |δ|) ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
            hsa2]
          simp
      _ ≤ _ := by
          have : 2 * κ * (c + |δ|) * ∑ j ∈ Finset.range r.N, |r.x j - r.mu| ≤
              2 * κ * (c + |δ|) * (r.N * σ) :=
            mul_le_mul_of_nonneg_left hsabs (by positivity)
          rw [← hσ2] at *
          nlinarith
  have h4 : 0 ≤ ∑ j ∈ Finset.range r.N, (ξ j - δ) ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg _
  have h5 := abs_le.1 h1
  rw [key]
  constructor <;> nlinarith

end RowP

end Claims.LayerNormFwdF16
