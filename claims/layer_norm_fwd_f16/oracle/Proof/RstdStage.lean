import Proof.SStage

namespace Claims.LayerNormFwdF16

/-- `1/w ≤ 1 + x/2 + x²` when `1 − x ≤ w²`, `0 ≤ x ≤ 1/2`, `w > 0`. -/
theorem inv_le_of_sq {w x : ℝ} (hx0 : 0 ≤ x) (hx : x ≤ 1 / 2) (hw : 0 < w) (h : 1 - x ≤ w ^ 2) :
    1 / w ≤ 1 + x / 2 + x ^ 2 := by
  rw [div_le_iff₀ hw]
  by_contra hlt
  have hlt := not_le.1 hlt
  have hc : 0 < 1 + x / 2 + x ^ 2 := by positivity
  have h1 : (w * (1 + x / 2 + x ^ 2)) ^ 2 < 1 := by
    have : 0 ≤ w * (1 + x / 2 + x ^ 2) := by positivity
    nlinarith
  have h2 : 1 ≤ (1 - x) * (1 + x / 2 + x ^ 2) ^ 2 := by
    nlinarith [mul_nonneg hx0 hx0, mul_nonneg (mul_nonneg hx0 hx0) hx0,
      mul_nonneg (mul_nonneg hx0 hx0) (sub_nonneg.2 hx), mul_nonneg hx0 (sub_nonneg.2 hx)]
  have h3 : (1 - x) * (1 + x / 2 + x ^ 2) ^ 2 ≤ w ^ 2 * (1 + x / 2 + x ^ 2) ^ 2 :=
    mul_le_mul_of_nonneg_right h (by positivity)
  nlinarith

/-- `1 − y/2 ≤ 1/w` when `w² ≤ 1 + y`, `0 ≤ y`, `w > 0`. -/
theorem inv_ge_of_sq {w y : ℝ} (hy : 0 ≤ y) (hw : 0 < w) (h : w ^ 2 ≤ 1 + y) :
    1 - y / 2 ≤ 1 / w := by
  have h1 : w ≤ 1 + y / 2 := by nlinarith [sq_nonneg y, sq_nonneg (w - 1 - y / 2)]
  rw [le_div_iff₀ hw]
  nlinarith [sq_nonneg y]

/-- The numeric core of `rstd_small`: the error budget `hiS/2 + loS² + 8.1u(1 + loS)` fits in
`thetaS` (the variables are `u`, `e = η/eps`, `c = cond`, `D`). -/
theorem theta_hat_le (u e c D : ℝ) (hu0 : 0 ≤ u) (hu : u ≤ 1 / 16000000) (he0 : 0 ≤ e)
    (he : e ≤ u / 2) (hc0 : 0 ≤ c) (hc : c ≤ 1 / 2) (hD7 : 7 ≤ D) (hD : D ≤ 15) :
    let nu := 11 / 2 * u + c / 5 + 43 / 100 * c ^ 2
    let lo := (D + 12) * u * (1 + nu) + 41 / 10 * u + 22 * e + 4001 / 1000 * u + c / 5
    let hi := (D + 12) * u * (1 + nu) + 41 / 10 * u + 22 * e + nu
    hi / 2 + lo ^ 2 + 81 / 10 * u * (1 + lo) ≤
      (D / 2 + 20) * u + c / 9 + 26 / 100 * c ^ 2 + 12 * e := by
  intro nu lo hi
  have hnu : nu ≤ 3 / 10 := by simp only [nu]; nlinarith
  have hnu0 : 0 ≤ nu := by simp only [nu]; positivity
  have hlo : lo ≤ c / 5 + 60 * u := by
    simp only [lo]
    nlinarith [mul_nonneg hu0 (by linarith : (0 : ℝ) ≤ D + 12),
      mul_nonneg (mul_nonneg hu0 (by linarith : (0 : ℝ) ≤ D + 12)) hnu0]
  have hlo0 : 0 ≤ lo := by simp only [lo]; positivity
  simp only [hi, nu] at *
  nlinarith [mul_nonneg hu0 hc0, mul_nonneg hu0 hu0, mul_nonneg hc0 hc0,
    mul_nonneg hu0 (sub_nonneg.2 hD), mul_nonneg (mul_nonneg hu0 hc0) (sub_nonneg.2 hD),
    mul_nonneg hu0 (sub_nonneg.2 hD7), mul_nonneg hc0 (sub_nonneg.2 hc),
    mul_nonneg hu0 (sub_nonneg.2 hc), mul_nonneg hlo0 hu0, mul_nonneg hlo0 hlo0]

/-- Size bounds of `loS` and `hiS` in the variables `u`, `e`, `c`, `D`. -/
theorem lo_hi_bounds (u e c D : ℝ) (hu0 : 0 ≤ u) (hu : u ≤ 1 / 16000000) (he0 : 0 ≤ e)
    (he : e ≤ u / 2) (hc0 : 0 ≤ c) (hc : c ≤ 1 / 2) (hD7 : 7 ≤ D) (hD : D ≤ 15) :
    let nu := 11 / 2 * u + c / 5 + 43 / 100 * c ^ 2
    let lo := (D + 12) * u * (1 + nu) + 41 / 10 * u + 22 * e + 4001 / 1000 * u + c / 5
    let hi := (D + 12) * u * (1 + nu) + 41 / 10 * u + 22 * e + nu
    0 ≤ lo ∧ lo ≤ hi ∧ lo ≤ 1 / 5 ∧ hi ≤ 3 / 10 := by
  intro nu lo hi
  have hnu : nu ≤ 3 / 10 := by simp only [nu]; nlinarith
  have hnu0 : 0 ≤ nu := by simp only [nu]; positivity
  have hD12 : (0 : ℝ) ≤ D + 12 := by linarith
  have h1 : (D + 12) * u * (1 + nu) ≤ 1 / 1000 := by
    have : (D + 12) * u ≤ 27 * u := by nlinarith
    nlinarith [mul_nonneg hD12 hu0]
  have h2 : 0 ≤ (D + 12) * u * (1 + nu) := by positivity
  simp only [lo, hi, nu] at *
  refine ⟨by positivity, ?_, ?_, ?_⟩ <;> nlinarith [mul_nonneg hc0 hc0]

namespace RowP

variable (r : RowP)

/-- `η/eps` is between `0` and `u/2`. -/
theorem eta_div_eps_bounds : 0 ≤ eta32 / r.eps ∧ eta32 / r.eps ≤ u32 / 2 := by
  have h := r.eps_pos
  refine ⟨by unfold eta32; positivity, ?_⟩
  rw [div_le_iff₀ h]
  have : u32 / 2 * 2 ^ (-125 : ℤ) ≤ u32 / 2 * r.eps :=
    mul_le_mul_of_nonneg_left r.eps_ge (by unfold u32; positivity)
  have e : eta32 = u32 / 2 * 2 ^ (-125 : ℤ) := by unfold eta32 u32; norm_num
  linarith

/-- The variance is at most `2^40`. -/
theorem sig2_le : r.sig2 ≤ 2 ^ 40 := by
  have hN : (0 : ℝ) < r.N := by exact_mod_cast (by have := r.n_ge; omega : 0 < r.N)
  unfold sig2
  rw [div_le_iff₀ hN]
  have hmu := r.mu_le
  have h : ∀ j ∈ Finset.range r.N, (r.x j - r.mu) ^ 2 ≤ 2 ^ 40 := by
    intro j hj
    have hx := r.x_le j (Finset.mem_range.1 hj)
    have : |r.x j - r.mu| ≤ 131008 := by
      have := abs_sub (r.x j) r.mu
      linarith
    have h2 : |r.x j - r.mu| ^ 2 ≤ 131008 ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
    rw [sq_abs] at h2
    linarith
  calc ∑ j ∈ Finset.range r.N, (r.x j - r.mu) ^ 2 ≤ ∑ _j ∈ Finset.range r.N, (2 : ℝ) ^ 40 :=
        Finset.sum_le_sum h
    _ = 2 ^ 40 * r.N := by simp [mul_comm]

/-- `sig2 + eps` is between `2^-125` and `2^127`. -/
theorem M_bounds : 2 ^ (-125 : ℤ) ≤ r.sig2 + r.eps ∧ r.sig2 + r.eps ≤ 2 ^ 127 := by
  have := r.sig2_nonneg
  have := r.sig2_le
  have := r.eps_ge
  have := r.eps_le
  constructor
  · linarith
  · have : (2 : ℝ) ^ 40 + 2 ^ 126 ≤ 2 ^ 127 := by norm_num
    linarith

/-- `R² (sig2 + eps) = 1`. -/
theorem R_sq_mul : r.R ^ 2 * (r.sig2 + r.eps) = 1 := by
  have h : 0 < r.sig2 + r.eps := by have := r.sig2_nonneg; have := r.eps_pos; linarith
  unfold R
  rw [div_pow, Real.sq_sqrt h.le]
  field_simp

end RowP

/-- The precision of `tl.sqrt` and of `1 / ·` is `4u`: the constant. -/
theorem two_pow_22 : (2 : ℝ) ^ (-22 : ℤ) = 4 * u32 := by unfold u32; norm_num

/-- `tl.sqrt` of `s ≥ 2^-126`: `r1` is within `4.001u` of `√s`. -/
theorem r1_bounds {s r1 : ℝ} (hs : 2 ^ (-126 : ℤ) ≤ s)
    (hr1 : |r1 - Real.sqrt s| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s + 2 ^ (-149 : ℤ)) :
    (1 - 4001 / 1000 * u32) * Real.sqrt s ≤ r1 ∧ r1 ≤ (1 + 4001 / 1000 * u32) * Real.sqrt s := by
  have ht : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt s := by
    apply Real.le_sqrt_of_sq_le
    have : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = 2 ^ (-126 : ℤ) := by norm_num
    linarith
  have h1 : (2 : ℝ) ^ (-149 : ℤ) ≤ 1 / 1000 * u32 * Real.sqrt s := by
    have : (2 : ℝ) ^ (-149 : ℤ) ≤ 1 / 1000 * u32 * 2 ^ (-63 : ℤ) := by unfold u32; norm_num
    have h2 : 1 / 1000 * u32 * 2 ^ (-63 : ℤ) ≤ 1 / 1000 * u32 * Real.sqrt s :=
      mul_le_mul_of_nonneg_left ht (by unfold u32; positivity)
    linarith
  rw [two_pow_22] at hr1
  obtain ⟨h3, h4⟩ := abs_le.1 hr1
  generalize (2 : ℝ) ^ (-149 : ℤ) = c at *
  constructor <;> nlinarith

/-- `rstd` is positive when `r1` is positive and below `2^65`. -/
theorem rstd_pos_of {r1 rstd : ℝ} (hr1 : 0 < r1) (hbig : r1 ≤ 2 ^ 65)
    (hrstd : |rstd - 1 / r1| ≤ 2 ^ (-22 : ℤ) * (1 / r1) + 2 ^ (-148 : ℤ)) : 0 < rstd := by
  rw [two_pow_22] at hrstd
  have hi : (2 : ℝ) ^ (-65 : ℤ) ≤ 1 / r1 := by
    rw [le_div_iff₀ hr1]
    have : (2 : ℝ) ^ (-65 : ℤ) * 2 ^ 65 = 1 := by norm_num
    nlinarith [zpow_pos (by norm_num : (0 : ℝ) < 2) (-65 : ℤ)]
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have h2 : (2 : ℝ) ^ (-148 : ℤ) < 1 / 2 * 2 ^ (-65 : ℤ) := by norm_num
  have := (abs_le.1 hrstd).1
  nlinarith

/-- The reciprocal of `tl.sqrt`'s result against the reciprocal of `√s`: within a relative `4.001u`. -/
theorem ir_facts {s r1 : ℝ} (hs : 2 ^ (-126 : ℤ) ≤ s)
    (hr1 : |r1 - Real.sqrt s| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s + 2 ^ (-149 : ℤ)) :
    0 < r1 ∧ r1 ≤ (1 + 4001 / 1000 * u32) * Real.sqrt s ∧
      (1 - 4001 / 1000 * u32) * (1 / r1) ≤ 1 / Real.sqrt s ∧
      1 / Real.sqrt s ≤ (1 + 4001 / 1000 * u32) * (1 / r1) := by
  obtain ⟨h1, h2⟩ := r1_bounds hs hr1
  have ht : 0 < Real.sqrt s := Real.sqrt_pos.2 (lt_of_lt_of_le (by positivity) hs)
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have hpos : 0 < r1 := lt_of_lt_of_le (by nlinarith) h1
  refine ⟨hpos, h2, ?_, ?_⟩
  · rw [le_div_iff₀ ht]
    have : (1 - 4001 / 1000 * u32) * Real.sqrt s * (1 / r1) ≤ r1 * (1 / r1) :=
      mul_le_mul_of_nonneg_right h1 (by positivity)
    rw [mul_one_div_cancel hpos.ne'] at this
    nlinarith
  · rw [div_le_iff₀ ht]
    have : r1 * (1 / r1) ≤ (1 + 4001 / 1000 * u32) * Real.sqrt s * (1 / r1) :=
      mul_le_mul_of_nonneg_right h2 (by positivity)
    rw [mul_one_div_cancel hpos.ne'] at this
    nlinarith

/-- `rstd ≤ (1 + 4u)/r1 + 2^-148` and `rstd ≥ (1 − 4u)/r1 − 2^-148`, unfolded. -/
theorem rstd_unfold {r1 rstd : ℝ}
    (hrstd : |rstd - 1 / r1| ≤ 2 ^ (-22 : ℤ) * (1 / r1) + 2 ^ (-148 : ℤ)) :
    rstd ≤ (1 + 4 * u32) * (1 / r1) + 2 ^ (-148 : ℤ) ∧
      (1 - 4 * u32) * (1 / r1) - 2 ^ (-148 : ℤ) ≤ rstd := by
  rw [two_pow_22] at hrstd
  obtain ⟨h1, h2⟩ := abs_le.1 hrstd
  constructor <;> linarith

/-- `rstd` is positive and within a relative `4.001u` of `1/√s`, when `2^-126 ≤ s ≤ 2^128`. -/
theorem rstd_sandwich {s r1 rstd : ℝ} (hs : 2 ^ (-126 : ℤ) ≤ s) (hsu : s ≤ 2 ^ 128)
    (hr1 : |r1 - Real.sqrt s| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s + 2 ^ (-149 : ℤ))
    (hrstd : |rstd - 1 / r1| ≤ 2 ^ (-22 : ℤ) * (1 / r1) + 2 ^ (-148 : ℤ)) :
    0 < rstd ∧ (1 - 4001 / 1000 * u32) * rstd ≤ (1 + 4001 / 1000 * u32) * (1 / Real.sqrt s) ∧
      (1 - 4001 / 1000 * u32) * (1 / Real.sqrt s) ≤ (1 + 4001 / 1000 * u32) * rstd := by
  obtain ⟨hpos, h2, h3, h4⟩ := ir_facts hs hr1
  have ht : Real.sqrt s ≤ 2 ^ 64 := by
    apply Real.sqrt_le_iff.2
    refine ⟨by positivity, ?_⟩
    have : ((2 : ℝ) ^ 64) ^ 2 = 2 ^ 128 := by norm_num
    linarith
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have hbig : r1 ≤ 2 ^ 65 := by
    have : (1 + 4001 / 1000 * u32) * Real.sqrt s ≤ 2 * 2 ^ 64 := by
      have := Real.sqrt_nonneg s
      nlinarith
    have h65 : (2 : ℝ) * 2 ^ 64 = 2 ^ 65 := by norm_num
    linarith
  have hrpos := rstd_pos_of hpos hbig hrstd
  have hi : (2 : ℝ) ^ (-65 : ℤ) ≤ 1 / r1 := by
    rw [le_div_iff₀ hpos]
    have : (2 : ℝ) ^ (-65 : ℤ) * 2 ^ 65 = 1 := by norm_num
    nlinarith [zpow_pos (by norm_num : (0 : ℝ) < 2) (-65 : ℤ)]
  have h5 : (2 : ℝ) ^ (-148 : ℤ) ≤ 1 / 1000 * u32 * 2 ^ (-65 : ℤ) := by unfold u32; norm_num
  have h6 : 1 / 1000 * u32 * 2 ^ (-65 : ℤ) ≤ 1 / 1000 * u32 * (1 / r1) :=
    mul_le_mul_of_nonneg_left hi (by positivity)
  obtain ⟨hu1, hl1⟩ := rstd_unfold hrstd
  have hir : 0 < 1 / r1 := by positivity
  have hit : 0 ≤ 1 / Real.sqrt s := by positivity
  refine ⟨hrpos, ?_, ?_⟩
  · nlinarith [mul_nonneg hu0 hir.le, mul_nonneg hu0 hu0, mul_nonneg hu0 hit]
  · nlinarith [mul_nonneg hu0 hir.le, mul_nonneg hu0 hu0, mul_nonneg hu0 hit]

/-- The numbers `1 + 8.1u` and `1 − 8.1u` bound the ratios `(1 ± 4.001u)/(1 ∓ 4.001u)`. -/
theorem ratio_bounds : (1 + 4001 / 1000 * u32) / (1 - 4001 / 1000 * u32) ≤ 1 + 81 / 10 * u32 ∧
    1 - 81 / 10 * u32 ≤ (1 - 4001 / 1000 * u32) / (1 + 4001 / 1000 * u32) := by
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  constructor
  · rw [div_le_iff₀ (by nlinarith)]
    nlinarith
  · rw [le_div_iff₀ (by nlinarith)]
    nlinarith [mul_nonneg hu0 hu0]

/-- `(1 + a)(1 + x/2 + x²) − 1 ≤ θ` once `hi/2 + x² + a(1 + x) ≤ θ`, `0 ≤ x ≤ min(1/2, hi)`. -/
theorem upper_key {a x hi θ : ℝ} (ha : 0 ≤ a) (hx0 : 0 ≤ x) (hx : x ≤ 1 / 2) (hxh : x ≤ hi)
    (h : hi / 2 + x ^ 2 + a * (1 + x) ≤ θ) : (1 + a) * (1 + x / 2 + x ^ 2) - 1 ≤ θ := by
  nlinarith [mul_nonneg (mul_nonneg ha hx0) (by linarith : 0 ≤ 1 / 2 - x)]

namespace RowP

variable (r : RowP)

/-- The key numeric fact: `hiS/2 + loS² + 8.1u(1 + loS) ≤ thetaS` on a well-conditioned row. -/
theorem hat_le (hc : r.cond ≤ 1 / 2) :
    r.hiS / 2 + r.loS ^ 2 + 81 / 10 * u32 * (1 + r.loS) ≤ r.thetaS := by
  obtain ⟨he0, he⟩ := r.eta_div_eps_bounds
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have := theta_hat_le u32 (eta32 / r.eps) r.cond r.D hu0 hu he0 he r.cond_nonneg hc hD7 hD
  unfold loS hiS thetaS nuH
  simp only [mul_div_assoc]
  simp only at this
  linarith

/-- `loS` and `hiS` are small, `loS` is nonnegative and below `hiS`, on a well-conditioned row. -/
theorem loS_hiS_bounds (hc : r.cond ≤ 1 / 2) :
    0 ≤ r.loS ∧ r.loS ≤ r.hiS ∧ r.loS ≤ 1 / 5 ∧ r.hiS ≤ 3 / 10 := by
  obtain ⟨he0, he⟩ := r.eta_div_eps_bounds
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have := lo_hi_bounds u32 (eta32 / r.eps) r.cond r.D hu0 hu he0 he r.cond_nonneg hc hD7 hD
  unfold loS hiS nuH
  simp only [mul_div_assoc]
  simp only at this
  exact this

/-- `gam ≥ 30u + 480 η/eps`. -/
theorem gam_ge : 30 * u32 + 480 * (eta32 / r.eps) ≤ r.gam := by
  have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  obtain ⟨he0, he⟩ := r.eta_div_eps_bounds
  have e1 : (2 : ℝ) ^ (-23 : ℤ) = 2 * u32 := by unfold u32; norm_num
  have e2 : (2 : ℝ) ^ (-145 : ℤ) / r.eps = 32 * (eta32 / r.eps) := by
    unfold eta32; rw [← mul_div_assoc]; norm_num
  unfold gam
  rw [e1, e2]
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ r.D - 7) hu0,
    mul_nonneg (by linarith : (0 : ℝ) ≤ r.D - 7) he0]

/-- **`rstd` on a well-conditioned row** (`cond ≤ 1/2`): `tl.sqrt` (`r1`) and `1 / ·` (`rstd`) of an
`s` with `1 − loS ≤ s R² ≤ 1 + hiS` give `rstd` within `thetaS · R` of `R`. -/
theorem rstd_small (hc : r.cond ≤ 1 / 2) (s r1 rstd : ℝ)
    (hs1 : 1 - r.loS ≤ s * r.R ^ 2) (hs2 : s * r.R ^ 2 ≤ 1 + r.hiS)
    (hr1 : |r1 - Real.sqrt s| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s + 2 ^ (-149 : ℤ))
    (hrstd : |rstd - 1 / r1| ≤ 2 ^ (-22 : ℤ) * (1 / r1) + 2 ^ (-148 : ℤ)) :
    0 < rstd ∧ |rstd - r.R| ≤ r.thetaS * r.R := by
  obtain ⟨hlo0, hlohi, hlo, hhi⟩ := r.loS_hiS_bounds hc
  have hhat := r.hat_le hc
  obtain ⟨hM1, hM2⟩ := r.M_bounds
  have hRM := r.R_sq_mul
  have hR := r.R_pos
  have hMpos : 0 < r.sig2 + r.eps := lt_of_lt_of_le (by positivity) hM1
  have hsM : s = (s * r.R ^ 2) * (r.sig2 + r.eps) := by rw [mul_assoc, hRM, mul_one]
  have hs_ge : 2 ^ (-126 : ℤ) ≤ s := by
    have h2 : (4 / 5) * (r.sig2 + r.eps) ≤ s := by
      rw [hsM]; exact mul_le_mul_of_nonneg_right (by linarith) hMpos.le
    have h3 : (4 / 5 : ℝ) * 2 ^ (-125 : ℤ) ≥ 2 ^ (-126 : ℤ) := by norm_num
    generalize (2 : ℝ) ^ (-125 : ℤ) = a at *
    generalize (2 : ℝ) ^ (-126 : ℤ) = b at *
    linarith
  have hs_le : s ≤ 2 ^ 128 := by
    have h2 : s ≤ (13 / 10) * (r.sig2 + r.eps) := by
      rw [hsM]; exact mul_le_mul_of_nonneg_right (by linarith) hMpos.le
    have h3 : (13 / 10 : ℝ) * 2 ^ 127 ≤ 2 ^ 128 := by norm_num
    nlinarith
  obtain ⟨hpos, hup, hdn⟩ := rstd_sandwich hs_ge hs_le hr1 hrstd
  refine ⟨hpos, ?_⟩
  have ht : 0 < Real.sqrt s := Real.sqrt_pos.2 (lt_of_lt_of_le (by positivity) hs_ge)
  have hw : 0 < Real.sqrt s * r.R := mul_pos ht hR
  have hw2 : (Real.sqrt s * r.R) ^ 2 = s * r.R ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (lt_of_lt_of_le (by positivity) hs_ge).le]
  have hP := inv_le_of_sq hlo0 (by linarith) hw (by rw [hw2]; linarith)
  have hQ := inv_ge_of_sq (by linarith : 0 ≤ r.hiS) hw (by rw [hw2]; linarith)
  have hit : 1 / Real.sqrt s = r.R * (1 / (Real.sqrt s * r.R)) := by
    field_simp
  obtain ⟨hA, hB⟩ := ratio_bounds
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have hq : 0 < 1 - 4001 / 1000 * u32 := by nlinarith
  have hq' : 0 < 1 + 4001 / 1000 * u32 := by nlinarith
  set P := 1 / (Real.sqrt s * r.R) with hPdef
  rw [abs_le]
  constructor
  · have h1 : (1 - 4001 / 1000 * u32) / (1 + 4001 / 1000 * u32) * (1 / Real.sqrt s) ≤ rstd := by
      rw [div_mul_eq_mul_div, div_le_iff₀ hq']; linarith
    have h2 : r.R * (1 - r.hiS / 2) ≤ 1 / Real.sqrt s := by
      rw [hit]; exact mul_le_mul_of_nonneg_left hQ hR.le
    have h3 : (1 - 81 / 10 * u32) * (r.R * (1 - r.hiS / 2)) ≤ rstd := by
      have hnn : 0 ≤ r.R * (1 - r.hiS / 2) := mul_nonneg hR.le (by linarith)
      have : (1 - 81 / 10 * u32) * (r.R * (1 - r.hiS / 2)) ≤
          (1 - 4001 / 1000 * u32) / (1 + 4001 / 1000 * u32) * (1 / Real.sqrt s) :=
        mul_le_mul hB h2 hnn (div_nonneg hq.le hq'.le)
      linarith
    nlinarith [mul_nonneg hR.le (mul_nonneg hu0 hlo0), mul_nonneg hR.le (mul_nonneg hu0 (by linarith : 0 ≤ r.hiS)),
      mul_nonneg hR.le (sq_nonneg r.loS)]
  · have h1 : rstd ≤ (1 + 4001 / 1000 * u32) / (1 - 4001 / 1000 * u32) * (1 / Real.sqrt s) := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hq]; linarith
    have h2 : 1 / Real.sqrt s ≤ r.R * (1 + r.loS / 2 + r.loS ^ 2) := by
      rw [hit]; exact mul_le_mul_of_nonneg_left hP hR.le
    have h3 : rstd ≤ (1 + 81 / 10 * u32) * (r.R * (1 + r.loS / 2 + r.loS ^ 2)) := by
      have : (1 + 4001 / 1000 * u32) / (1 - 4001 / 1000 * u32) * (1 / Real.sqrt s) ≤
          (1 + 81 / 10 * u32) * (r.R * (1 + r.loS / 2 + r.loS ^ 2)) :=
        mul_le_mul hA h2 (by positivity) (by positivity)
      linarith
    have key := upper_key (by positivity : 0 ≤ 81 / 10 * u32) hlo0 (by linarith : r.loS ≤ 1 / 2) hlohi hhat
    have := mul_le_mul_of_nonneg_left key hR.le
    linarith only [h3, this]

/-- Numeric core of `rstd_big`: `(1 + 4u)(1 + Lo/2 + Lo²) + 10^-9 ≤ (1 − 4.001u)(1 + gam)`, with
`Lo = 4.1u + 22e` and `gam ≥ 30u + 480e`. -/
theorem big_key {u e g : ℝ} (hu1 : 1 / 20000000 ≤ u) (hu : u ≤ 1 / 16000000) (he0 : 0 ≤ e) (he : e ≤ u / 2)
    (hg : 30 * u + 480 * e ≤ g) :
    (1 + 4 * u) * (1 + (41 / 10 * u + 22 * e) / 2 + (41 / 10 * u + 22 * e) ^ 2) + 1 / 10 ^ 9 ≤
      (1 - 4001 / 1000 * u) * (1 + g) := by
  have hu0 : 0 ≤ u := by linarith
  have hg0 : 0 ≤ g := by linarith
  nlinarith [mul_nonneg hu0 hu0, mul_nonneg hu0 he0, mul_nonneg he0 he0, mul_nonneg hu0 hg0,
    mul_nonneg (by linarith : 0 ≤ 1 - 4001 / 1000 * u) (by linarith : 0 ≤ g - (30 * u + 480 * e))]

/-- One step of `rstd_big_bounds`: `rstd ≤ (1 + 4u)/r1 + c` and `(1 − q)/r1 ≤ 1/√s` give
`(1 − q) rstd ≤ (1 + 4u)/√s + c`. -/
theorem big_step {rstd ir it c u q : ℝ} (h1 : rstd ≤ (1 + 4 * u) * ir + c) (h2 : (1 - q) * ir ≤ it)
    (hq0 : 0 ≤ q) (hq : q ≤ 1) (hc : 0 ≤ c) (hu : 0 ≤ u) : (1 - q) * rstd ≤ (1 + 4 * u) * it + c := by
  nlinarith [mul_le_mul_of_nonneg_left h1 (by linarith : 0 ≤ 1 - q),
    mul_le_mul_of_nonneg_left h2 (by linarith : 0 ≤ 1 + 4 * u), mul_nonneg hq0 hc]

/-- **`rstd` on any row**, the upper bound: `s` at least `eps (1 − 4.1u) − 22η` gives
`rstd ≤ (1 + gam)/√eps`, and `rstd ≥ −10^-9/√eps`. -/
theorem rstd_big_bounds (s r1 rstd : ℝ) (hs : r.eps - 41 / 10 * u32 * r.eps - 22 * eta32 ≤ s)
    (hr1 : |r1 - Real.sqrt s| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s + 2 ^ (-149 : ℤ))
    (hrstd : |rstd - 1 / r1| ≤ 2 ^ (-22 : ℤ) * (1 / r1) + 2 ^ (-148 : ℤ)) :
    2 ^ (-126 : ℤ) ≤ s ∧ -(1 / 10 ^ 9 * (1 / Real.sqrt r.eps)) ≤ rstd ∧
      rstd ≤ (1 + r.gam) / Real.sqrt r.eps := by
  obtain ⟨he0, he⟩ := r.eta_div_eps_bounds
  have hepos := r.eps_pos
  have hu : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have hg := r.gam_ge
  have hs' : r.eps * (1 - (41 / 10 * u32 + 22 * (eta32 / r.eps))) ≤ s := by
    have h22 : r.eps * (22 * (eta32 / r.eps)) = 22 * eta32 := by field_simp
    have : r.eps * (1 - (41 / 10 * u32 + 22 * (eta32 / r.eps))) =
        r.eps - 41 / 10 * u32 * r.eps - 22 * eta32 := by rw [← h22]; ring
    rw [this]; exact hs
  have hLo : 41 / 10 * u32 + 22 * (eta32 / r.eps) ≤ 1 / 2 := by linarith
  have hs_ge : 2 ^ (-126 : ℤ) ≤ s := by
    have h3 : r.eps * (1 / 2) ≥ 2 ^ (-126 : ℤ) := by
      have := r.eps_ge
      have h4 : (2 : ℝ) ^ (-125 : ℤ) * (1 / 2) = 2 ^ (-126 : ℤ) := by norm_num
      generalize (2 : ℝ) ^ (-125 : ℤ) = a at *
      generalize (2 : ℝ) ^ (-126 : ℤ) = b at *
      linarith
    have h5 : r.eps * (1 / 2) ≤ r.eps * (1 - (41 / 10 * u32 + 22 * (eta32 / r.eps))) :=
      mul_le_mul_of_nonneg_left (by linarith) hepos.le
    generalize (2 : ℝ) ^ (-126 : ℤ) = b at *
    linarith
  obtain ⟨hpos, h2, h3, h4⟩ := ir_facts hs_ge hr1
  have ht : 0 < Real.sqrt s := Real.sqrt_pos.2 (lt_of_lt_of_le (by positivity) hs_ge)
  have hse : 0 < Real.sqrt r.eps := Real.sqrt_pos.2 hepos
  have hk : (2 : ℝ) ^ (-63 : ℤ) ≤ 1 / Real.sqrt r.eps := by
    rw [le_div_iff₀ hse]
    have : Real.sqrt r.eps ≤ 2 ^ 63 := by
      apply Real.sqrt_le_iff.2
      refine ⟨by positivity, ?_⟩
      have := r.eps_le
      have h6 : ((2 : ℝ) ^ 63) ^ 2 = 2 ^ 126 := by norm_num
      linarith
    have h7 : (2 : ℝ) ^ (-63 : ℤ) * 2 ^ 63 = 1 := by norm_num
    nlinarith [zpow_pos (by norm_num : (0 : ℝ) < 2) (-63 : ℤ)]
  set k := 1 / Real.sqrt r.eps with hkdef
  have hk0 : 0 < k := by positivity
  have hw : 0 < Real.sqrt s * k := by positivity
  have hw2 : (Real.sqrt s * k) ^ 2 = s / r.eps := by
    rw [mul_pow, Real.sq_sqrt (lt_of_lt_of_le (by positivity) hs_ge).le, hkdef, div_pow,
      Real.sq_sqrt hepos.le]
    ring
  have hP := inv_le_of_sq (by positivity : 0 ≤ 41 / 10 * u32 + 22 * (eta32 / r.eps)) hLo hw (by
    rw [hw2, le_div_iff₀ hepos]; linarith)
  have hit : 1 / Real.sqrt s = k * (1 / (Real.sqrt s * k)) := by
    rw [hkdef]; field_simp
  have h148 : (2 : ℝ) ^ (-148 : ℤ) ≤ 1 / 10 ^ 9 * k := by
    have : (2 : ℝ) ^ (-148 : ℤ) ≤ 1 / 10 ^ 9 * 2 ^ (-63 : ℤ) := by norm_num
    have h8 : 1 / 10 ^ 9 * (2 : ℝ) ^ (-63 : ℤ) ≤ 1 / 10 ^ 9 * k :=
      mul_le_mul_of_nonneg_left hk (by positivity)
    linarith
  obtain ⟨hu1', hl1⟩ := rstd_unfold hrstd
  have hir : 0 < 1 / r1 := by positivity
  have hq : 0 < 1 - 4001 / 1000 * u32 := by nlinarith
  have hk148 : (0 : ℝ) ≤ 2 ^ (-148 : ℤ) := by positivity
  refine ⟨hs_ge, ?_, ?_⟩
  · nlinarith [mul_nonneg hu0 hir.le]
  · have hP' : 1 / Real.sqrt s ≤ k * (1 + (41 / 10 * u32 + 22 * (eta32 / r.eps)) / 2 +
        (41 / 10 * u32 + 22 * (eta32 / r.eps)) ^ 2) := by
      rw [hit]; exact mul_le_mul_of_nonneg_left hP hk0.le
    have hu1 : (1 : ℝ) / 20000000 ≤ u32 := by unfold u32; norm_num
    have hkey := big_key hu1 hu he0 he hg
    have h10 := big_step hu1' h3 (by positivity) (by nlinarith) hk148 hu0
    have h11 : (1 + 4 * u32) * (1 / Real.sqrt s) ≤ (1 + 4 * u32) * (k * (1 + (41 / 10 * u32 +
        22 * (eta32 / r.eps)) / 2 + (41 / 10 * u32 + 22 * (eta32 / r.eps)) ^ 2)) :=
      mul_le_mul_of_nonneg_left hP' (by positivity)
    have h12 := mul_le_mul_of_nonneg_left hkey hk0.le
    have h9 : (1 - 4001 / 1000 * u32) * rstd ≤ k * ((1 - 4001 / 1000 * u32) * (1 + r.gam)) := by
      linarith only [h10, h11, h12, h148]
    rw [div_eq_mul_inv, ← one_div, ← hkdef, mul_comm]
    exact le_of_mul_le_mul_left (by linarith) hq

/-- **`rstd` on any row**: `s` at least `eps (1 − 4.1u) − 22η` and at most `2^128` gives
`0 < rstd ≤ (1 + gam)/√eps`. -/
theorem rstd_big_of_le (s r1 rstd : ℝ) (hs : r.eps - 41 / 10 * u32 * r.eps - 22 * eta32 ≤ s)
    (hsu : s ≤ 2 ^ 128)
    (hr1 : |r1 - Real.sqrt s| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s + 2 ^ (-149 : ℤ))
    (hrstd : |rstd - 1 / r1| ≤ 2 ^ (-22 : ℤ) * (1 / r1) + 2 ^ (-148 : ℤ)) :
    0 < rstd ∧ rstd ≤ (1 + r.gam) / Real.sqrt r.eps := by
  obtain ⟨hs_ge, -, hup⟩ := r.rstd_big_bounds s r1 rstd hs hr1 hrstd
  exact ⟨(rstd_sandwich hs_ge hsu hr1 hrstd).1, hup⟩

/-- **`rstd` against its tolerance**. -/
theorem rstd_tol (s r1 rstd : ℝ) (hs : r.eps - 41 / 10 * u32 * r.eps - 22 * eta32 ≤ s)
    (hsm : r.cond ≤ 1 / 2 → 1 - r.loS ≤ s * r.R ^ 2 ∧ s * r.R ^ 2 ≤ 1 + r.hiS)
    (hr1 : |r1 - Real.sqrt s| ≤ 2 ^ (-22 : ℤ) * Real.sqrt s + 2 ^ (-149 : ℤ))
    (hrstd : |rstd - 1 / r1| ≤ 2 ^ (-22 : ℤ) * (1 / r1) + 2 ^ (-148 : ℤ)) :
    |rstd - r.R| ≤ r.rstdTol := by
  unfold rstdTol
  by_cases hc : r.cond ≤ 1 / 2
  · simp only [hc, ↓reduceIte]
    obtain ⟨h1, h2⟩ := hsm hc
    have h := (r.rstd_small hc s r1 rstd h1 h2 hr1 hrstd).2
    obtain ⟨he0, he⟩ := r.eta_div_eps_bounds
    have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
    have hc0 := r.cond_nonneg
    have hR := r.R_pos
    have e1 : (2 : ℝ) ^ (-21 : ℤ) = 8 * u32 := by unfold u32; norm_num
    have e2 : (2 : ℝ) ^ (-140 : ℤ) / r.eps = 1024 * (eta32 / r.eps) := by
      unfold eta32; rw [← mul_div_assoc]; norm_num
    have e3 : (2 : ℝ) ^ (-140 : ℤ) ≥ 0 := by positivity
    have hu0 : 0 ≤ u32 := by unfold u32; positivity
    have hθ : r.thetaS ≤ r.D * (8 * u32) + 1024 * (eta32 / r.eps) + r.cond / 4 := by
      unfold thetaS
      simp only [mul_div_assoc]
      nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ r.D - 7) hu0, mul_nonneg hc0 (by linarith : 0 ≤ 1 / 2 - r.cond)]
    rw [e1, e2]
    have := mul_le_mul_of_nonneg_right hθ hR.le
    linarith only [h, this, e3]
  · simp only [hc, ↓reduceIte]
    obtain ⟨-, hlow, hup⟩ := r.rstd_big_bounds s r1 rstd hs hr1 hrstd
    have hRk : r.R ≤ 1 / Real.sqrt r.eps := by
      have hse : 0 < Real.sqrt r.eps := Real.sqrt_pos.2 r.eps_pos
      unfold R
      apply one_div_le_one_div_of_le hse
      exact Real.sqrt_le_sqrt (by linarith [r.sig2_nonneg])
    have hR := r.R_pos
    have hg := r.gam_bounds.2
    have hk0 : 0 < 1 / Real.sqrt r.eps := by have := r.eps_pos; positivity
    have hup' : rstd ≤ (1 + r.gam) * (1 / Real.sqrt r.eps) := by rwa [mul_one_div]
    have hk2 : 2 / Real.sqrt r.eps = 2 * (1 / Real.sqrt r.eps) := by ring
    rw [abs_le, hk2]
    have := mul_le_mul_of_nonneg_right hg hk0.le
    constructor <;> linarith

end RowP

end Claims.LayerNormFwdF16
