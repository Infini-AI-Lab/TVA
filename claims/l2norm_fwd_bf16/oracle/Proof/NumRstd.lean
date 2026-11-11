import TVA.Proof

/-!
# The `rstd` chain: `addc`, `sqrt`, `div`

The kernel computes `s2 = fl(s1 + eps)`, `q = fl(√s2)`, `r = fl(1/q)` and wants `r ≈ (√(A+eps))⁻¹`
where `A` is the exact sum of squares and `s1` the computed one.
-/

set_option maxHeartbeats 1000000

namespace Claims.L2NormFwd

open TVA

/-- If `|x - y| ≤ κ y`, `0 ≤ x`, `0 < y`, `0 ≤ κ ≤ 1`, then `|√x - √y| (2-κ) ≤ κ √y`. -/
lemma sqrt_abs_sub_le {x y κ : ℝ} (hx : 0 ≤ x) (hy : 0 < y)
    (hκ0 : 0 ≤ κ) (hκ1 : κ ≤ 1) (h : |x - y| ≤ κ * y) :
    |Real.sqrt x - Real.sqrt y| * (2 - κ) ≤ κ * Real.sqrt y := by
  have hy' : 0 < Real.sqrt y := Real.sqrt_pos.2 hy
  have hsq : (Real.sqrt x - Real.sqrt y) * (Real.sqrt x + Real.sqrt y) = x - y := by
    have h1 : Real.sqrt x * Real.sqrt x = x := Real.mul_self_sqrt hx
    have h2 : Real.sqrt y * Real.sqrt y = y := Real.mul_self_sqrt hy.le
    nlinarith
  have hsum_nonneg : (0:ℝ) ≤ Real.sqrt x + Real.sqrt y := by
    have := Real.sqrt_nonneg x; linarith
  have habs : |x - y| = |Real.sqrt x - Real.sqrt y| * (Real.sqrt x + Real.sqrt y) := by
    rw [← abs_of_nonneg hsum_nonneg, ← abs_mul, hsq]
  have hxge : y - κ * y ≤ x := by
    have := abs_le.1 h; linarith [this.1]
  have hfac : (1 - κ) ^ 2 ≤ 1 - κ := by nlinarith
  have hxy : y * (1 - κ) ^ 2 ≤ x := by
    have := mul_le_mul_of_nonneg_left hfac hy.le
    linarith
  have hsqrtx : Real.sqrt y * (1 - κ) ≤ Real.sqrt x := by
    have h1 : Real.sqrt (y * (1 - κ) ^ 2) ≤ Real.sqrt x := Real.sqrt_le_sqrt hxy
    rw [Real.sqrt_mul hy.le, Real.sqrt_sq_eq_abs, abs_of_nonneg (by linarith)] at h1
    exact h1
  have hsum : Real.sqrt y * (2 - κ) ≤ Real.sqrt x + Real.sqrt y := by nlinarith
  have hstep : |Real.sqrt x - Real.sqrt y| * (Real.sqrt y * (2 - κ))
      ≤ |Real.sqrt x - Real.sqrt y| * (Real.sqrt x + Real.sqrt y) :=
    mul_le_mul_of_nonneg_left hsum (abs_nonneg _)
  rw [← habs] at hstep
  have hyy : Real.sqrt y * Real.sqrt y = y := Real.mul_self_sqrt hy.le
  have hexp : |Real.sqrt x - Real.sqrt y| * (2 - κ) * Real.sqrt y
      ≤ κ * Real.sqrt y * Real.sqrt y := by
    nlinarith [hstep, h, hyy]
  exact le_of_mul_le_mul_right (by linarith) hy'

/-- If `|x - y| ≤ κ y`, `0 ≤ x`, `0 < y`, `0 ≤ κ ≤ 1/2`, then
`|(√x)⁻¹ - (√y)⁻¹| ≤ κ/((2-κ)(1-κ)) * (√y)⁻¹`. -/
lemma inv_sqrt_abs_sub_le {x y κ : ℝ} (hx : 0 ≤ x) (hy : 0 < y)
    (hκ0 : 0 ≤ κ) (hκ1 : κ ≤ 1/2) (h : |x - y| ≤ κ * y) :
    |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹|
      ≤ κ / ((2 - κ) * (1 - κ)) * (Real.sqrt y)⁻¹ := by
  have hy' : 0 < Real.sqrt y := Real.sqrt_pos.2 hy
  have hxge : y - κ * y ≤ x := by
    have := abs_le.1 h; linarith [this.1]
  have hfac : (1 - κ) ^ 2 ≤ 1 - κ := by nlinarith
  have hxy : y * (1 - κ) ^ 2 ≤ x := by
    have := mul_le_mul_of_nonneg_left hfac hy.le
    linarith
  have hsqrtx : Real.sqrt y * (1 - κ) ≤ Real.sqrt x := by
    have h1 : Real.sqrt (y * (1 - κ) ^ 2) ≤ Real.sqrt x := Real.sqrt_le_sqrt hxy
    rw [Real.sqrt_mul hy.le, Real.sqrt_sq_eq_abs, abs_of_nonneg (by linarith)] at h1
    exact h1
  have hsqrtxpos : 0 < Real.sqrt x := by
    have h0 : 0 < Real.sqrt y * (1 - κ) := by nlinarith
    linarith
  have hkey : |Real.sqrt x - Real.sqrt y| * (2 - κ) ≤ κ * Real.sqrt y :=
    sqrt_abs_sub_le hx hy hκ0 (by linarith) h
  have hxne : Real.sqrt x ≠ 0 := ne_of_gt hsqrtxpos
  have hyne : Real.sqrt y ≠ 0 := ne_of_gt hy'
  have hid : |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| * (Real.sqrt x * Real.sqrt y)
      = |Real.sqrt x - Real.sqrt y| := by
    have hpos : 0 < Real.sqrt x * Real.sqrt y := mul_pos hsqrtxpos hy'
    rw [show (Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹
        = (Real.sqrt y - Real.sqrt x) / (Real.sqrt x * Real.sqrt y) by
      field_simp]
    rw [abs_div, abs_of_pos hpos, abs_sub_comm]
    field_simp
  have hmul : |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| * (Real.sqrt x * Real.sqrt y) * (2 - κ)
      ≤ κ * Real.sqrt y := by rw [hid]; exact hkey
  have hlow : Real.sqrt y * (1 - κ) * Real.sqrt y ≤ Real.sqrt x * Real.sqrt y :=
    mul_le_mul_of_nonneg_right hsqrtx hy'.le
  have hnn : (0:ℝ) ≤ |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| := abs_nonneg _
  have h2 : Real.sqrt y * (1 - κ) * (2 - κ) * Real.sqrt y ≤ Real.sqrt x * Real.sqrt y * (2 - κ) := by
    nlinarith [hlow, hκ1, hy'.le, mul_nonneg (mul_nonneg hy'.le (by linarith : (0:ℝ) ≤ 1 - κ)) (by linarith : (0:ℝ) ≤ 2 - κ)]
  have hcomb : |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| * (Real.sqrt y * (1 - κ) * (2 - κ)) * Real.sqrt y
      ≤ κ * Real.sqrt y := by
    have h1 : |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| * (Real.sqrt y * (1 - κ) * (2 - κ) * Real.sqrt y)
        ≤ |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| * (Real.sqrt x * Real.sqrt y * (2 - κ)) :=
      mul_le_mul_of_nonneg_left h2 hnn
    nlinarith [h1, hmul]
  have hdiv : |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| * (Real.sqrt y * ((2 - κ) * (1 - κ))) ≤ κ := by
    have hh : |(Real.sqrt x)⁻¹ - (Real.sqrt y)⁻¹| * (Real.sqrt y * (1 - κ) * (2 - κ)) * Real.sqrt y ≤ κ * Real.sqrt y := hcomb
    have := le_of_mul_le_mul_right hh hy'
    linarith [this]
  have h1κ : 0 < 1 - κ := by linarith
  have h2κ : 0 < 2 - κ := by linarith
  have hposD : 0 < ((2 - κ) * (1 - κ)) * Real.sqrt y := by positivity
  have heq : κ / (((2 - κ) * (1 - κ)) * Real.sqrt y)
      = κ / ((2 - κ) * (1 - κ)) * (Real.sqrt y)⁻¹ := by
    field_simp
  rw [← heq]
  rw [le_div_iff₀ hposD]
  nlinarith [hdiv]

/-- Reciprocal bound for `q` from `|q - √x| ≤ u√x + b`. -/
lemma inv_q_abs_sub_le {x q u b : ℝ} (_hx : 0 ≤ x) (hq : 0 < q)
    (hu : 0 < u) (hu4 : u ≤ 1/4) (hb : 0 < b) (hbx : b ≤ u * Real.sqrt x)
    (h : |q - Real.sqrt x| ≤ u * Real.sqrt x + b) :
    |1/q - (Real.sqrt x)⁻¹| ≤ 4 * u * (Real.sqrt x)⁻¹ := by
  have hxpos : 0 < Real.sqrt x := by
    have h0 : 0 < u * Real.sqrt x := lt_of_lt_of_le hb hbx
    exact pos_of_mul_pos_right h0 hu.le
  have hqge : Real.sqrt x * (1 - 2 * u) ≤ q := by
    have h1 := (abs_le.1 h).1
    nlinarith [h1, hbx]
  have hqpos2 : 0 < Real.sqrt x * (1 - 2 * u) := by nlinarith
  have hid : |1/q - (Real.sqrt x)⁻¹| * (q * Real.sqrt x) = |Real.sqrt x - q| := by
    have hpos : 0 < q * Real.sqrt x := mul_pos hq hxpos
    rw [show 1/q - (Real.sqrt x)⁻¹ = (Real.sqrt x - q) / (q * Real.sqrt x) by
      field_simp]
    rw [abs_div, abs_of_pos hpos]
    field_simp
  have hstep : |1/q - (Real.sqrt x)⁻¹| * (Real.sqrt x * (1 - 2 * u) * Real.sqrt x)
      ≤ u * Real.sqrt x + b := by
    have key : |1/q - (Real.sqrt x)⁻¹| * (q * Real.sqrt x) ≤ u * Real.sqrt x + b := by
      rw [hid, abs_sub_comm]; exact h
    have hnn : (0:ℝ) ≤ |1/q - (Real.sqrt x)⁻¹| := abs_nonneg _
    have hle : Real.sqrt x * (1 - 2 * u) * Real.sqrt x ≤ q * Real.sqrt x :=
      mul_le_mul_of_nonneg_right hqge hxpos.le
    nlinarith [mul_le_mul_of_nonneg_left hle hnn, key]
  have hu4' : 0 < 1 - 2 * u := by linarith
  have hstep2 : |1/q - (Real.sqrt x)⁻¹| * (Real.sqrt x * Real.sqrt x * (1 - 2 * u))
      ≤ 2 * u * Real.sqrt x := by
    nlinarith [hstep, hbx]
  have hstep2' : |1/q - (Real.sqrt x)⁻¹| * (Real.sqrt x * (1 - 2 * u)) * Real.sqrt x
      ≤ 2 * u * Real.sqrt x := by nlinarith [hstep2]
  have hdiv : |1/q - (Real.sqrt x)⁻¹| * (Real.sqrt x * (1 - 2 * u)) ≤ 2 * u :=
    le_of_mul_le_mul_right hstep2' hxpos
  have hhalf : |1/q - (Real.sqrt x)⁻¹| * Real.sqrt x * (1/2) ≤ 2 * u := by
    have hnn : (0:ℝ) ≤ |1/q - (Real.sqrt x)⁻¹| * Real.sqrt x := by positivity
    nlinarith [hdiv, hu4', hnn]
  have hfin : |1/q - (Real.sqrt x)⁻¹| * Real.sqrt x ≤ 4 * u := by linarith
  have heq : 4 * u * (Real.sqrt x)⁻¹ = 4 * u / Real.sqrt x := by
    rw [div_eq_mul_inv]
  rw [heq, le_div_iff₀ hxpos]
  linarith

/-- **The `rstd` chain**: the computed `1 / sqrt(s1 + eps)` is within `2^-12` (relative) of `(√(A + eps))⁻¹`, given the sum's error, the fusable `+ eps`, and the device facts for `sqrt` and `/`. -/
theorem rstd_err {A e s1 s2 q r : ℝ}
    (he0 : 0 ≤ e) (hA0 : 0 ≤ A) (he : (2 : ℝ) ^ (-125 : ℤ) ≤ e)
    (hVhi : A + e ≤ (2 : ℝ) ^ (126 : ℤ))
    (h1 : |s1 - A| ≤ (2 : ℝ) ^ (-13 : ℤ) * A + (3/4) * (2 : ℝ) ^ (-137 : ℤ))
    (h2 : |s2 - (s1 + e)| ≤ 3 * (2 : ℝ) ^ (-24 : ℤ) * (|s1| + e) + 4 * (2 : ℝ) ^ (-150 : ℤ))
    (hq0 : 0 < q)
    (h3 : |q - Real.sqrt s2| ≤ (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2 + (2 : ℝ) ^ (-149 : ℤ))
    (h4 : |r - 1 / q| ≤ (2 : ℝ) ^ (-22 : ℤ) * |1 / q| + (2 : ℝ) ^ (-148 : ℤ)) :
    |r - (Real.sqrt (A + e))⁻¹|
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * (Real.sqrt (A + e))⁻¹ := by
  obtain ⟨V, hVdef⟩ : ∃ V : ℝ, V = A + e := ⟨_, rfl⟩
  obtain ⟨u, hudef⟩ : ∃ u : ℝ, u = (2 : ℝ) ^ (-24 : ℤ) := ⟨_, rfl⟩
  obtain ⟨u3, hu3def⟩ : ∃ u3 : ℝ, u3 = (2 : ℝ) ^ (-22 : ℤ) := ⟨_, rfl⟩
  obtain ⟨c, hcdef⟩ : ∃ c : ℝ, c = (3/4) * (2 : ℝ) ^ (-137 : ℤ) := ⟨_, rfl⟩
  have hu3 : 0 < u3 := by rw [hu3def]; norm_num
  have hu34 : u3 ≤ 1 / 4 := by rw [hu3def]; norm_num
  obtain ⟨κ, hκdef⟩ : ∃ κ : ℝ, κ = (2 : ℝ) ^ (-12 : ℤ) + (2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-21 : ℤ) := ⟨_, rfl⟩
  have hu : 0 < u := by rw [hudef]; norm_num
  have hu4 : u ≤ 1 / 4 := by rw [hudef]; norm_num
  have hcpos : 0 < c := by rw [hcdef]; norm_num
  have hκ0 : 0 ≤ κ := by rw [hκdef]; norm_num
  have hκ1 : κ ≤ 1 / 2 := by rw [hκdef]; norm_num
  have hVpos : 0 < V := by
    have h0 : (0 : ℝ) < (2 : ℝ) ^ (-125 : ℤ) := by norm_num
    rw [hVdef]; linarith
  have hVlow : (2 : ℝ) ^ (-125 : ℤ) ≤ V := by rw [hVdef]; linarith
  have hVhigh : V ≤ (2 : ℝ) ^ (126 : ℤ) := by rw [hVdef]; exact hVhi
  have hAleV : A ≤ V := by rw [hVdef]; linarith
  have heV : e ≤ V := by rw [hVdef]; linarith
  have hs1 : |s1| ≤ A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + c := by
    have hle := (abs_le.1 h1).2
    have habs : |s1| ≤ A + |s1 - A| := by
      have h := abs_add_le A (s1 - A)
      rw [abs_of_nonneg hA0] at h
      have heq : A + (s1 - A) = s1 := by ring
      rwa [heq] at h
    rw [hcdef]
    nlinarith [habs, hle]
  have hs2 : |s2 - V| ≤ A * ((2 : ℝ) ^ (-13 : ℤ) + 3 * u * (1 + (2 : ℝ) ^ (-13 : ℤ)))
      + 3 * u * e + c * (1 + 3 * u) + 4 * (2 : ℝ) ^ (-150 : ℤ) := by
    have hVA : (s1 + e) - V = s1 - A := by rw [hVdef]; ring
    have htri : |s2 - V| ≤ |s2 - (s1 + e)| + |(s1 + e) - V| := abs_sub_le s2 (s1 + e) V
    rw [hVA] at htri
    have h1' : |s1 - A| ≤ (2 : ℝ) ^ (-13 : ℤ) * A + c := by rw [hcdef]; exact h1
    have h2' : |s2 - (s1 + e)| ≤ 3 * u * (|s1| + e) + 4 * (2 : ℝ) ^ (-150 : ℤ) := by
      rw [hudef]; exact h2
    have hbase : |s2 - V| ≤ 3 * u * (|s1| + e) + 4 * (2 : ℝ) ^ (-150 : ℤ)
        + ((2 : ℝ) ^ (-13 : ℤ) * A + c) := by linarith [htri, h2', h1']
    have hmul : 3 * u * |s1| ≤ 3 * u * (A * (1 + (2 : ℝ) ^ (-13 : ℤ)) + c) :=
      mul_le_mul_of_nonneg_left hs1 (by positivity)
    nlinarith [hbase, hmul]
  have hδ : |s2 - V| ≤ κ * V := by
    have hAcoef : (2 : ℝ) ^ (-13 : ℤ) + 3 * u * (1 + (2 : ℝ) ^ (-13 : ℤ))
        ≤ (2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ) := by rw [hudef]; norm_num
    have h12 : A * ((2 : ℝ) ^ (-13 : ℤ) + 3 * u * (1 + (2 : ℝ) ^ (-13 : ℤ)))
        ≤ A * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ)) :=
      mul_le_mul_of_nonneg_left hAcoef hA0
    have hAV : A * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ))
        ≤ V * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ)) :=
      mul_le_mul_of_nonneg_right hAleV (by positivity)
    have h3u : 3 * u * e ≤ (2 : ℝ) ^ (-22 : ℤ) * V := by
      have hcoef : 3 * u ≤ (2 : ℝ) ^ (-22 : ℤ) := by rw [hudef]; norm_num
      have h1 : 3 * u * e ≤ (2 : ℝ) ^ (-22 : ℤ) * e := by nlinarith [hcoef, he0]
      have h2 : (2 : ℝ) ^ (-22 : ℤ) * e ≤ (2 : ℝ) ^ (-22 : ℤ) * V :=
        mul_le_mul_of_nonneg_left heV (by norm_num)
      linarith
    have hc1 : c * (1 + 3 * u) + 4 * (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-12 : ℤ) * V := by
      have h1 : c * (1 + 3 * u) + 4 * (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-137 : ℤ) := by
        rw [hcdef, hudef]; norm_num
      have h2 : (2 : ℝ) ^ (-137 : ℤ) = (2 : ℝ) ^ (-12 : ℤ) * (2 : ℝ) ^ (-125 : ℤ) := by norm_num
      have h3 : (2 : ℝ) ^ (-12 : ℤ) * (2 : ℝ) ^ (-125 : ℤ) ≤ (2 : ℝ) ^ (-12 : ℤ) * V :=
        mul_le_mul_of_nonneg_left hVlow (by norm_num)
      linarith
    have hsum : A * ((2 : ℝ) ^ (-13 : ℤ) + (2 : ℝ) ^ (-22 : ℤ))
        + (2 : ℝ) ^ (-22 : ℤ) * V + (2 : ℝ) ^ (-12 : ℤ) * V ≤ κ * V := by
      rw [hκdef]
      have h4 : (2 : ℝ) ^ (-22 : ℤ) * V + (2 : ℝ) ^ (-22 : ℤ) * V = (2 : ℝ) ^ (-21 : ℤ) * V := by
        have h5 : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-22 : ℤ) = (2 : ℝ) ^ (-21 : ℤ) := by norm_num
        nlinarith
      linarith [hAV, h4]
    linarith [hs2, h12, hAV, h3u, hc1, hsum]
  have hs2low : V * (1 - κ) ≤ s2 := by
    have h := (abs_le.1 hδ).1; linarith
  have hs2pos : 0 < s2 := by
    have hκV : V * κ ≤ V * (1 / 2) := mul_le_mul_of_nonneg_left hκ1 hVpos.le
    nlinarith [hs2low, hκV, hVpos]
  have hVsqrtpos : 0 < Real.sqrt V := Real.sqrt_pos.2 hVpos
  have hVsqrt_low : (2 : ℝ) ^ (-63 : ℤ) ≤ Real.sqrt V := by
    have h1 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 ≤ V := by
      have h2 : ((2 : ℝ) ^ (-63 : ℤ)) ^ 2 = (2 : ℝ) ^ (-126 : ℤ) := by norm_num
      have h3 : (2 : ℝ) ^ (-126 : ℤ) ≤ (2 : ℝ) ^ (-125 : ℤ) := by norm_num
      rw [h2]; linarith
    exact (Real.le_sqrt (by norm_num) hVpos.le).2 h1
  have hVsqrt_high : Real.sqrt V ≤ (2 : ℝ) ^ (63 : ℤ) := by
    have h1 : V ≤ ((2 : ℝ) ^ (63 : ℤ)) ^ 2 := by
      have h2 : ((2 : ℝ) ^ (63 : ℤ)) ^ 2 = (2 : ℝ) ^ (126 : ℤ) := by norm_num
      rw [h2]; exact hVhigh
    exact Real.sqrt_le_iff.2 ⟨by norm_num, h1⟩
  have hs2sqrt_low : Real.sqrt V * (1 - κ) ≤ Real.sqrt s2 := by
    have h1 : (Real.sqrt V * (1 - κ)) ^ 2 ≤ s2 := by
      have h2 : (Real.sqrt V * (1 - κ)) ^ 2 = V * (1 - κ) ^ 2 := by
        rw [mul_pow, Real.sq_sqrt hVpos.le]
      rw [h2]
      have hfac : (1 - κ) ^ 2 ≤ 1 - κ := by nlinarith
      have hmul : V * (1 - κ) ^ 2 ≤ V * (1 - κ) := mul_le_mul_of_nonneg_left hfac hVpos.le
      linarith [hmul, hs2low]
    have h0 : 0 ≤ Real.sqrt V * (1 - κ) := by
      have : 0 < 1 - κ := by linarith
      positivity
    exact (Real.le_sqrt h0 hs2pos.le).2 h1
  have hinv_sqrt : |(Real.sqrt s2)⁻¹ - (Real.sqrt V)⁻¹|
      ≤ κ / ((2 - κ) * (1 - κ)) * (Real.sqrt V)⁻¹ :=
    inv_sqrt_abs_sub_le hs2pos.le hVpos hκ0 hκ1 hδ
  have hs2sqrt_ge : (2 : ℝ) ^ (-64 : ℤ) ≤ Real.sqrt s2 := by
    have hA : (2 : ℝ) ^ (-64 : ℤ) = (2 : ℝ) ^ (-63 : ℤ) * (1 / 2) := by norm_num
    have hB : (2 : ℝ) ^ (-63 : ℤ) * (1 / 2) ≤ Real.sqrt V * (1 / 2) :=
      mul_le_mul_of_nonneg_right hVsqrt_low (by norm_num)
    have hC : Real.sqrt V * (1 / 2) ≤ Real.sqrt V * (1 - κ) :=
      mul_le_mul_of_nonneg_left (by linarith : (1 : ℝ) / 2 ≤ 1 - κ) hVsqrtpos.le
    exact le_trans (hA.trans_le hB) (le_trans hC hs2sqrt_low)
  have hb : (2 : ℝ) ^ (-149 : ℤ) ≤ u3 * Real.sqrt s2 := by
    have h1 : u3 * (2 : ℝ) ^ (-64 : ℤ) ≤ u3 * Real.sqrt s2 :=
      mul_le_mul_of_nonneg_left hs2sqrt_ge hu3.le
    have h2 : (2 : ℝ) ^ (-149 : ℤ) ≤ u3 * (2 : ℝ) ^ (-64 : ℤ) := by rw [hu3def]; norm_num
    linarith
  have h3' : |q - Real.sqrt s2| ≤ u3 * Real.sqrt s2 + (2 : ℝ) ^ (-149 : ℤ) := by
    rw [hu3def]; exact h3
  have hqinv : |1 / q - (Real.sqrt s2)⁻¹| ≤ 4 * u3 * (Real.sqrt s2)⁻¹ :=
    inv_q_abs_sub_le hs2pos.le hq0 hu3 hu34 (by norm_num) hb h3'
  have hs2sqrtinv_le : (Real.sqrt s2)⁻¹ ≤ 2 * (Real.sqrt V)⁻¹ := by
    have h2 : Real.sqrt V * (1 / 2) ≤ Real.sqrt s2 := by nlinarith [hs2sqrt_low, hκ1]
    have hpos : (0 : ℝ) < Real.sqrt V * (1 / 2) := by positivity
    have h := one_div_le_one_div_of_le hpos h2
    rw [one_div, one_div] at h
    have hR : (Real.sqrt V * (1 / 2))⁻¹ = 2 * (Real.sqrt V)⁻¹ := by
      rw [mul_inv, mul_comm]; norm_num
    rwa [hR] at h
  have hqinv' : |1 / q - (Real.sqrt s2)⁻¹| ≤ 8 * u3 * (Real.sqrt V)⁻¹ := by
    have h1 : 4 * u3 * (Real.sqrt s2)⁻¹ ≤ 4 * u3 * (2 * (Real.sqrt V)⁻¹) :=
      mul_le_mul_of_nonneg_left hs2sqrtinv_le (by positivity)
    nlinarith [hqinv, h1]
  have hq_low : Real.sqrt s2 * (1 - 2 * u3) ≤ q := by
    have h1 := (abs_le.1 h3').1
    nlinarith [h1, hb]
  have hqinv_abs : |1 / q| ≤ 4 * (Real.sqrt V)⁻¹ := by
    have hqge : Real.sqrt s2 * (1 / 2) ≤ q := by nlinarith [hq_low, hu34]
    have hpos : (0 : ℝ) < Real.sqrt s2 * (1 / 2) := by
      have : 0 < Real.sqrt s2 := Real.sqrt_pos.2 hs2pos
      positivity
    have h1 : (1 : ℝ) / q ≤ 1 / (Real.sqrt s2 * (1 / 2)) := one_div_le_one_div_of_le hpos hqge
    have h2 : (1 : ℝ) / (Real.sqrt s2 * (1 / 2)) = 2 * (Real.sqrt s2)⁻¹ := by
      rw [one_div, mul_inv, mul_comm]; norm_num
    have h3 : 2 * (Real.sqrt s2)⁻¹ ≤ 4 * (Real.sqrt V)⁻¹ := by nlinarith [hs2sqrtinv_le]
    rw [abs_of_pos (by positivity : (0 : ℝ) < 1 / q)]
    linarith [h1, h2, h3]
  have hr : |r - 1 / q| ≤ 4 * u3 * (Real.sqrt V)⁻¹ + (2 : ℝ) ^ (-22 : ℤ) * (Real.sqrt V)⁻¹ := by
    have h4' : |r - 1 / q| ≤ u3 * |1 / q| + (2 : ℝ) ^ (-148 : ℤ) := by rw [hu3def]; exact h4
    have h1 : u3 * |1 / q| ≤ u3 * (4 * (Real.sqrt V)⁻¹) :=
      mul_le_mul_of_nonneg_left hqinv_abs hu3.le
    have h2 : (2 : ℝ) ^ (-148 : ℤ) ≤ (2 : ℝ) ^ (-22 : ℤ) * (Real.sqrt V)⁻¹ := by
      have h3 : (2 : ℝ) ^ (-148 : ℤ) * Real.sqrt V ≤ (2 : ℝ) ^ (-22 : ℤ) := by
        have h4 : (2 : ℝ) ^ (-148 : ℤ) * Real.sqrt V ≤ (2 : ℝ) ^ (-148 : ℤ) * (2 : ℝ) ^ (63 : ℤ) :=
          mul_le_mul_of_nonneg_left hVsqrt_high (by norm_num)
        have h5 : (2 : ℝ) ^ (-148 : ℤ) * (2 : ℝ) ^ (63 : ℤ) ≤ (2 : ℝ) ^ (-22 : ℤ) := by norm_num
        linarith
      have hdiv := (le_div_iff₀ hVsqrtpos).2 h3
      rwa [div_eq_mul_inv] at hdiv
    nlinarith [h4', h1, h2]
  have htri : |r - (Real.sqrt V)⁻¹|
      ≤ |r - 1 / q| + |1 / q - (Real.sqrt s2)⁻¹| + |(Real.sqrt s2)⁻¹ - (Real.sqrt V)⁻¹| := by
    have h1 : |r - (Real.sqrt V)⁻¹| ≤ |r - 1 / q| + |1 / q - (Real.sqrt V)⁻¹| :=
      abs_sub_le r (1 / q) ((Real.sqrt V)⁻¹)
    have h2 : |1 / q - (Real.sqrt V)⁻¹|
        ≤ |1 / q - (Real.sqrt s2)⁻¹| + |(Real.sqrt s2)⁻¹ - (Real.sqrt V)⁻¹| :=
      abs_sub_le (1 / q) ((Real.sqrt s2)⁻¹) ((Real.sqrt V)⁻¹)
    linarith
  have hnum : κ / ((2 - κ) * (1 - κ)) + 4 * u3 + (2 : ℝ) ^ (-22 : ℤ) + 8 * u3
      ≤ (2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) := by
    rw [hκdef, hu3def]; norm_num
  have hsum : |r - (Real.sqrt V)⁻¹|
      ≤ (κ / ((2 - κ) * (1 - κ)) + 4 * u3 + (2 : ℝ) ^ (-22 : ℤ) + 8 * u3) * (Real.sqrt V)⁻¹ := by
    nlinarith [htri, hr, hqinv', hinv_sqrt]
  have hlast : (κ / ((2 - κ) * (1 - κ)) + 4 * u3 + (2 : ℝ) ^ (-22 : ℤ) + 8 * u3) * (Real.sqrt V)⁻¹
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * (Real.sqrt V)⁻¹ :=
    mul_le_mul_of_nonneg_right hnum (by positivity)
  have hres : |r - (Real.sqrt V)⁻¹|
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * (Real.sqrt V)⁻¹ := le_trans hsum hlast
  rw [hVdef] at hres
  exact hres


/-! ## The host's `eps` as the device's `float32` -/

/-- `FloatV.hostF32` of a non-NaN double is its plain conversion to `float32`. -/
theorem hostF32_eq_cvt {c : F64} (h : c.isNaN = false) :
    FloatV.hostF32 c = (FloatV.cvt c : F32) := by
  simp [FloatV.hostF32, h]

/-- The device's `float32` value of the host's double is a finite `float32`, and within
`2^-24·eps + 2^-150` of it. -/
theorem hostF32_err {c : F64} (hfin : c.isFinite = true) (h0 : 0 ≤ c.toRat)
    (hmax : c.toRat ≤ (2 : ℚ) ^ (126 : ℤ)) :
    (FloatV.hostF32 c).isFinite = true ∧
      |(FloatV.hostF32 c).toRat - c.toRat|
        ≤ (2 : ℚ) ^ (-24 : ℤ) * c.toRat + (2 : ℚ) ^ (-150 : ℤ) := by
  have hnan : c.isNaN = false := Format.not_isNaN_of_isFinite hfin
  rw [hostF32_eq_cvt hnan]
  have hmax' : |c.toRat| ≤ Format.binary32.maxVal := by
    rw [abs_of_nonneg h0, Format.binary32_maxVal]
    calc c.toRat ≤ (2 : ℚ) ^ (126 : ℤ) := hmax
      _ ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
  obtain ⟨hfin', herr⟩ :=
    FloatV.cvt_err (g := Format.binary32) Format.binary32_regular hfin hmax'
  refine ⟨hfin', ?_⟩
  rw [Format.binary32_u, Format.binary32_eta, abs_of_nonneg h0] at herr
  linarith [herr]

/-- A `rstd` bound relative to the device's `eps` transfers to the host's `eps` with the full
`2^-12`. -/
theorem eps_transfer {A e e32 r : ℝ} (hA0 : 0 ≤ A) (he : (2 : ℝ) ^ (-125 : ℤ) ≤ e)
    (he32 : |e32 - e| ≤ (2 : ℝ) ^ (-24 : ℤ) * e + (2 : ℝ) ^ (-150 : ℤ))
    (hr : |r - (Real.sqrt (A + e32))⁻¹|
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * (Real.sqrt (A + e32))⁻¹) :
    |r - (Real.sqrt (A + e))⁻¹| ≤ (2 : ℝ) ^ (-12 : ℤ) * (Real.sqrt (A + e))⁻¹ := by
  set δ : ℝ := (2 : ℝ) ^ (-24 : ℤ) + (2 : ℝ) ^ (-25 : ℤ) with hδ
  have hδpos : 0 < δ := by rw [hδ]; positivity
  have hδ0 : 0 ≤ δ := le_of_lt hδpos
  have hδhalf : δ ≤ 1 / 2 := by rw [hδ]; norm_num
  have h1dpos : 0 < 1 - δ := by rw [hδ]; norm_num
  have h1d0 : 0 ≤ 1 - δ := le_of_lt h1dpos
  have h1dle1 : 1 - δ ≤ 1 := by rw [hδ]; norm_num
  have hepos : 0 < e := by
    have h : (0 : ℝ) < (2 : ℝ) ^ (-125 : ℤ) := by norm_num
    linarith
  have hVpos : 0 < A + e := by linarith
  have hkey : (2 : ℝ) ^ (-150 : ℤ) < (2 : ℝ) ^ (-125 : ℤ) * (1 - (2 : ℝ) ^ (-24 : ℤ)) := by
    norm_num
  have hmul : (2 : ℝ) ^ (-125 : ℤ) * (1 - (2 : ℝ) ^ (-24 : ℤ))
      ≤ e * (1 - (2 : ℝ) ^ (-24 : ℤ)) :=
    mul_le_mul_of_nonneg_right he (by norm_num)
  have he32pos : 0 < e32 := by
    have h := (abs_le.mp he32).1
    nlinarith [hkey, hmul]
  have hV'pos : 0 < A + e32 := by linarith
  have hdiff : |(A + e32) - (A + e)| ≤ δ * (A + e) := by
    have h1 : (A + e32) - (A + e) = e32 - e := by ring
    rw [h1]
    have h2 : (2 : ℝ) ^ (-150 : ℤ) ≤ (2 : ℝ) ^ (-25 : ℤ) * e := by
      have h := mul_le_mul_of_nonneg_left he (by norm_num : (0 : ℝ) ≤ (2 : ℝ) ^ (-25 : ℤ))
      calc (2 : ℝ) ^ (-150 : ℤ) = (2 : ℝ) ^ (-25 : ℤ) * (2 : ℝ) ^ (-125 : ℤ) := by norm_num
        _ ≤ (2 : ℝ) ^ (-25 : ℤ) * e := h
    have h3 : (2 : ℝ) ^ (-24 : ℤ) * e + (2 : ℝ) ^ (-150 : ℤ) ≤ δ * (A + e) := by
      rw [hδ]
      nlinarith [h2, hVpos, hA0]
    linarith [he32, h3]
  have hV'lb : (A + e) * (1 - δ) ≤ A + e32 := by
    have h := (abs_le.mp hdiff).1
    nlinarith [h]
  set s : ℝ := Real.sqrt (A + e) with hs
  set s' : ℝ := Real.sqrt (A + e32) with hs'
  have hs2 : s ^ 2 = A + e := by rw [hs]; exact Real.sq_sqrt (le_of_lt hVpos)
  have hs'2 : s' ^ 2 = A + e32 := by rw [hs']; exact Real.sq_sqrt (le_of_lt hV'pos)
  have hspos : 0 < s := by rw [hs]; exact Real.sqrt_pos.mpr hVpos
  have hs'pos : 0 < s' := by rw [hs']; exact Real.sqrt_pos.mpr hV'pos
  have hr' : |r - s'⁻¹| ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ)) * s'⁻¹ := by
    simpa [hs'] using hr
  -- the two square roots are within `δ·s`
  have hss : |s' - s| ≤ δ * s := by
    have hprod : |s' - s| * (s' + s) = |s' ^ 2 - s ^ 2| := by
      rw [← abs_of_nonneg (by linarith : (0 : ℝ) ≤ s' + s), ← abs_mul]
      congr 1
      ring
    have hsqle : |s' ^ 2 - s ^ 2| ≤ δ * (A + e) := by
      simpa [hs'2, hs2] using hdiff
    have h1 : |s' - s| * s ≤ |s' - s| * (s' + s) :=
      mul_le_mul_of_nonneg_left (by linarith : s ≤ s' + s) (abs_nonneg _)
    have h2 : |s' - s| * s ≤ δ * s * s := by
      calc |s' - s| * s ≤ |s' - s| * (s' + s) := h1
        _ = |s' ^ 2 - s ^ 2| := hprod
        _ ≤ δ * (A + e) := hsqle
        _ = δ * (s * s) := by rw [← hs2]; ring
        _ = δ * s * s := by ring
    exact le_of_mul_le_mul_right h2 hspos
  -- hence the inverses are within `δ·s'⁻¹`
  have hba : |s'⁻¹ - s⁻¹| ≤ δ * s'⁻¹ := by
    have hss_pos : 0 < s * s' := mul_pos hspos hs'pos
    have hkey' : s'⁻¹ - s⁻¹ = (s - s') / (s * s') := by
      field_simp
    rw [hkey', abs_div, abs_of_pos hss_pos]
    have h1 : |s - s'| / (s * s') ≤ (δ * s) / (s * s') := by
      apply div_le_div_of_nonneg_right _ (le_of_lt hss_pos)
      rwa [abs_sub_comm]
    have h2 : (δ * s) / (s * s') = δ * s'⁻¹ := by
      field_simp
    calc |s - s'| / (s * s') ≤ (δ * s) / (s * s') := h1
      _ = δ * s'⁻¹ := h2
  -- `s'` is at least `s·(1 − δ)`
  have hs'ge : s * (1 - δ) ≤ s' := by
    have hbase : 0 ≤ s * (1 - δ) := mul_nonneg (le_of_lt hspos) h1d0
    have hsq : (s * (1 - δ)) ^ 2 ≤ s' ^ 2 := by
      rw [mul_pow, hs2, hs'2]
      have h1 : (1 - δ) * (1 - δ) ≤ 1 - δ :=
        mul_le_of_le_one_left h1d0 h1dle1
      nlinarith [hV'lb, h1, h1d0]
    calc s * (1 - δ) = Real.sqrt ((s * (1 - δ)) ^ 2) := (Real.sqrt_sq hbase).symm
      _ ≤ Real.sqrt (s' ^ 2) := Real.sqrt_le_sqrt hsq
      _ = s' := Real.sqrt_sq (le_of_lt hs'pos)
  have hb_le : s'⁻¹ ≤ (1 + 2 * δ) * s⁻¹ := by
    have hinv : s'⁻¹ ≤ (s * (1 - δ))⁻¹ := by
      simpa using one_div_le_one_div_of_le (mul_pos hspos h1dpos) hs'ge
    have hinv1d : (1 - δ)⁻¹ ≤ 1 + 2 * δ := by
      rw [inv_le_iff_one_le_mul₀ h1dpos]
      have hprod : 0 ≤ δ * (1 - 2 * δ) :=
        mul_nonneg hδ0 (by rw [hδ]; norm_num)
      nlinarith [hprod]
    calc s'⁻¹ ≤ (s * (1 - δ))⁻¹ := hinv
      _ = s⁻¹ * (1 - δ)⁻¹ := by rw [mul_inv]
      _ ≤ s⁻¹ * (1 + 2 * δ) :=
          mul_le_mul_of_nonneg_left hinv1d (le_of_lt (inv_pos.mpr hspos))
      _ = (1 + 2 * δ) * s⁻¹ := by ring
  -- assemble
  have hcδ : 0 ≤ (2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ := by
    rw [hδ]; norm_num
  have htri : |r - s⁻¹| ≤ |r - s'⁻¹| + |s'⁻¹ - s⁻¹| := by
    have h : r - s⁻¹ = (r - s'⁻¹) + (s'⁻¹ - s⁻¹) := by ring
    rw [h]
    exact abs_add_le _ _
  have hb1 : |r - s⁻¹|
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ) * s'⁻¹ := by
    nlinarith [htri, hr', hba]
  have hb2 : ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ) * s'⁻¹
      ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ) * ((1 + 2 * δ) * s⁻¹) :=
    mul_le_mul_of_nonneg_left hb_le hcδ
  have hfin : ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ) * (1 + 2 * δ)
      ≤ (2 : ℝ) ^ (-12 : ℤ) := by
    rw [hδ]; norm_num
  have hsinv : 0 ≤ s⁻¹ := le_of_lt (inv_pos.mpr hspos)
  calc |r - s⁻¹| ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ) * s'⁻¹ := hb1
    _ ≤ ((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ) * ((1 + 2 * δ) * s⁻¹) := hb2
    _ = (((2 : ℝ) ^ (-12 : ℤ) - (2 : ℝ) ^ (-16 : ℤ) + δ) * (1 + 2 * δ)) * s⁻¹ := by ring
    _ ≤ (2 : ℝ) ^ (-12 : ℤ) * s⁻¹ :=
        mul_le_mul_of_nonneg_right hfin hsinv

end Claims.L2NormFwd
