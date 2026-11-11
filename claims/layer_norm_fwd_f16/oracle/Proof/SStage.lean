import Proof.SumSq
import Proof.VarChain

namespace Claims.LayerNormFwdF16

namespace RowP

variable (r : RowP)

/-- The lanes' constant error is at most a tenth of the mean's tolerance. -/
theorem cLane_le (m : ℝ) (hm : |m - r.mu| ≤ 11 / 20 * r.Delta) : r.cLane m ≤ r.Delta / 10 := by
  have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
  have hΔ := r.Delta_nonneg
  have hA := r.aMean_nonneg
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have hu2 : u32 ≤ 1 / 10 ^ 7 := by unfold u32; norm_num
  have hΔ2 : (2 : ℝ) ^ (-140 : ℤ) ≤ r.Delta := by
    unfold Delta
    have : 0 ≤ ((r.D : ℝ) + 4) * 2 ^ (-23 : ℤ) * r.aMean := by positivity
    linarith
  have hΔ3 : 11 * (2 * u32 * r.aMean) ≤ r.Delta := by
    unfold Delta
    have e : (2 : ℝ) ^ (-23 : ℤ) = 2 * u32 := by unfold u32; norm_num
    rw [e]
    have : 0 ≤ 2 * u32 * r.aMean := by positivity
    nlinarith [mul_nonneg this (by linarith : (0 : ℝ) ≤ r.D - 7), (by positivity : (0 : ℝ) ≤ 2 ^ (-140 : ℤ))]
  have hη : 3 * eta32 ≤ 3 / 1000 * r.Delta := by
    have : 3 * eta32 ≤ 3 / 1000 * 2 ^ (-140 : ℤ) := by unfold eta32; norm_num
    exact this.trans (mul_le_mul_of_nonneg_left hΔ2 (by norm_num))
  have h1 : u32 * (2 + u32) * |m - r.mu| ≤ 1 / 10 ^ 6 * r.Delta := by
    have a1 : u32 * (2 + u32) ≤ 1 / 10 ^ 6 := by nlinarith
    have a2 : u32 * (2 + u32) * |m - r.mu| ≤ 1 / 10 ^ 6 * |m - r.mu| :=
      mul_le_mul_of_nonneg_right a1 (abs_nonneg _)
    nlinarith [abs_nonneg (m - r.mu)]
  have h2 : 2 * u32 * (1 + u32) * |r.mu| ≤ (1 + u32) * (2 * u32 * r.aMean) := by
    have := r.mu_le
    have hmu : |r.mu| ≤ r.aMean := by
      unfold mu aMean
      have hN : (0 : ℝ) < r.N := by exact_mod_cast (by have := r.n_ge; omega : 0 < r.N)
      rw [abs_div, abs_of_pos hN]
      exact div_le_div_of_nonneg_right (Finset.abs_sum_le_sum_abs _ _) hN.le
    have : 0 ≤ 2 * u32 * (1 + u32) := by positivity
    nlinarith [mul_le_mul_of_nonneg_left hmu this]
  unfold cLane
  nlinarith [mul_nonneg hu hΔ]

/-- The lanes' constant error, in units of the standard deviation, is at most a tenth of the
condition number: `cLane m · R ≤ cond / 10`. -/
theorem cLane_mul_R_le (m : ℝ) (hm : |m - r.mu| ≤ 11 / 20 * r.Delta) :
    r.cLane m * r.R ≤ r.cond / 10 := by
  have := mul_le_mul_of_nonneg_right (r.cLane_le m hm) r.R_pos.le
  unfold cond
  linarith

/-- The mean's error and the lanes' constant error together are below `Delta`. -/
theorem delta_add_cLane_le (m : ℝ) (hm : |m - r.mu| ≤ 11 / 20 * r.Delta) :
    |m - r.mu| + r.cLane m ≤ r.Delta := by
  have := r.cLane_le m hm
  have := r.Delta_nonneg
  linarith

/-- On a well-conditioned row, the lanes' variance `V0 = Σ d² / N`, in units of `1/R^2`, is within
`(−(2κ + 2ĉ), nuH)` of `sig2`, where `ĉ = cLane m · R`. -/
theorem variance_scaled (m : ℝ) (hm : |m - r.mu| ≤ 11 / 20 * r.Delta) (hc : r.cond ≤ 1 / 2)
    (d : ℕ → ℝ) (hd : ∀ j, j < r.N → LaneErr (r.x j) m (d j)) :
    -(2 * (u32 * (2 + u32)) + 2 * (r.cLane m * r.R)) ≤
        ((∑ j ∈ Finset.range r.N, d j ^ 2) / r.N - r.sig2) * r.R ^ 2 ∧
      ((∑ j ∈ Finset.range r.N, d j ^ 2) / r.N - r.sig2) * r.R ^ 2 ≤ r.nuH := by
  have hN : (0 : ℝ) < r.N := by exact_mod_cast (by have := r.n_ge; omega : 0 < r.N)
  have hu0 : (0 : ℝ) < u32 := by unfold u32; positivity
  have hu2 : u32 ≤ 1 / 10 ^ 7 := by unfold u32; norm_num
  have hε0 := r.eps_pos
  obtain ⟨hL, hH⟩ := r.sumsq_bounds m d hd
  set Q := ∑ j ∈ Finset.range r.N, d j ^ 2 with hQ
  set V0 := Q / r.N with hV0def
  have hc0 := r.cond_nonneg
  have hsq : r.R ^ 2 * (r.sig2 + r.eps) = 1 := by
    unfold R
    have hM : 0 < r.sig2 + r.eps := by have := r.sig2_nonneg; linarith
    rw [div_pow, Real.sq_sqrt hM.le]
    field_simp
  have hsig := r.sig2_nonneg
  have hR := r.R_pos
  have hq0 : 0 < r.R ^ 2 := by positivity
  have hσ0 : 0 ≤ Real.sqrt r.sig2 := Real.sqrt_nonneg _
  have hσ2 : Real.sqrt r.sig2 ^ 2 = r.sig2 := Real.sq_sqrt hsig
  have hεq : r.eps * r.R ^ 2 ≤ 1 := by nlinarith
  have hqε : r.R ^ 2 ≤ 1 / r.eps := by rw [le_div_iff₀ hε0]; linarith
  have hσh2 : (Real.sqrt r.sig2 * r.R) ^ 2 ≤ 1 := by
    rw [mul_pow, hσ2]; nlinarith
  have hσh0 : 0 ≤ Real.sqrt r.sig2 * r.R := by positivity
  have hσh1 : Real.sqrt r.sig2 * r.R ≤ 1 := by nlinarith
  have hĉ := r.cLane_mul_R_le m hm
  have hĉ0 : 0 ≤ r.cLane m * r.R := by
    have : 0 ≤ r.cLane m := by unfold cLane u32 eta32; positivity
    positivity
  have hδ : |m - r.mu| * r.R ≤ 11 / 20 * r.cond := by
    have := mul_le_mul_of_nonneg_right hm hR.le
    unfold cond; linarith
  have hδ0 : 0 ≤ |m - r.mu| * r.R := by positivity
  have hκ0 : 0 ≤ u32 * (2 + u32) := by positivity
  have hκ : u32 * (2 + u32) ≤ 20001 / 10000 * u32 := by nlinarith
  have hVL : r.sig2 - 2 * (u32 * (2 + u32)) * r.sig2 - 2 * r.cLane m * Real.sqrt r.sig2 ≤ V0 := by
    rw [hV0def, le_div_iff₀ hN]; linarith
  have hVH : V0 ≤ r.sig2 + 2 * (u32 * (2 + u32)) * r.sig2 + 2 * r.cLane m * Real.sqrt r.sig2 +
      (u32 * (2 + u32) * Real.sqrt r.sig2 + r.cLane m + |m - r.mu|) ^ 2 := by
    rw [hV0def, div_le_iff₀ hN]; linarith
  set κ := u32 * (2 + u32) with hκdef
  set σh := Real.sqrt r.sig2 * r.R with hσhdef
  set ch := r.cLane m * r.R with hchdef
  set dh := |m - r.mu| * r.R with hdhdef
  have hsigq : r.sig2 * r.R ^ 2 = σh ^ 2 := by rw [hσhdef, mul_pow, hσ2]
  have hs1 : (V0 + r.eps) * r.R ^ 2 = 1 + (V0 - r.sig2) * r.R ^ 2 := by linear_combination hsq
  have h1 : κ * σh ^ 2 ≤ κ := (mul_le_mul_of_nonneg_left hσh2 hκ0).trans_eq (mul_one κ)
  have h2 : ch * σh ≤ ch := (mul_le_mul_of_nonneg_left hσh1 hĉ0).trans_eq (mul_one ch)
  have hlo : -(2 * κ + 2 * ch) ≤ (V0 - r.sig2) * r.R ^ 2 := by
    have := mul_le_mul_of_nonneg_right hVL hq0.le
    have e : (r.sig2 - 2 * κ * r.sig2 - 2 * r.cLane m * Real.sqrt r.sig2) * r.R ^ 2 =
        r.sig2 * r.R ^ 2 - 2 * κ * σh ^ 2 - 2 * ch * σh := by
      simp only [hσhdef, hchdef] at hsigq ⊢
      linear_combination (-2 * κ) * hsigq
    linarith
  have hnu : (κ * σh + ch + dh) ^ 2 ≤ 169 / 400 * r.cond ^ 2 + 27 / 10 * (r.cond * u32) +
      5 * (u32 * u32) := by
    have h3 : κ * σh + ch + dh ≤ 13 / 20 * r.cond + 20001 / 10000 * u32 := by
      linarith [(mul_le_mul_of_nonneg_left hσh1 hκ0).trans_eq (mul_one κ)]
    have h4 : (κ * σh + ch + dh) ^ 2 ≤ (13 / 20 * r.cond + 20001 / 10000 * u32) ^ 2 :=
      pow_le_pow_left₀ (by positivity) h3 2
    have e : (13 / 20 * r.cond + 20001 / 10000 * u32) ^ 2 = 169 / 400 * r.cond ^ 2 +
        2 * (13 / 20) * (20001 / 10000) * (r.cond * u32) + (20001 / 10000) ^ 2 * (u32 * u32) := by
      ring
    rw [e] at h4
    linarith [mul_nonneg hc0 hu0.le, mul_nonneg hu0.le hu0.le]
  have hhi : (V0 - r.sig2) * r.R ^ 2 ≤ r.nuH := by
    have := mul_le_mul_of_nonneg_right hVH hq0.le
    have e : (r.sig2 + 2 * κ * r.sig2 + 2 * r.cLane m * Real.sqrt r.sig2 +
        (κ * Real.sqrt r.sig2 + r.cLane m + |m - r.mu|) ^ 2) * r.R ^ 2 =
        r.sig2 * r.R ^ 2 + 2 * κ * σh ^ 2 + 2 * ch * σh + (κ * σh + ch + dh) ^ 2 := by
      simp only [hσhdef, hchdef, hdhdef] at hsigq ⊢
      linear_combination (2 * κ) * hsigq
    have hcu : r.cond * u32 ≤ u32 / 2 := by linarith [mul_le_mul_of_nonneg_right hc hu0.le]
    have huu : u32 * u32 ≤ u32 / 10 ^ 7 := by
      linarith [mul_le_mul_of_nonneg_left hu2 hu0.le]
    unfold nuH
    linarith [sq_nonneg r.cond]
  exact ⟨hlo, hhi⟩

/-- **The input of `tl.sqrt`**, from the lanes and the chain of roundings: always at least
`eps (1 − 4.1u) − 22η`, and, on a row with `cond ≤ 1/2`, within `(1 − loS, 1 + hiS)` of
`1 / R^2 = sig2 + eps`. -/
theorem s_stage (m : ℝ) (hm : |m - r.mu| ≤ 11 / 20 * r.Delta) (d v : ℕ → ℝ) (T var epsf s : ℝ)
    (hd : ∀ j, j < r.N → LaneErr (r.x j) m (d j))
    (hd0 : ∀ j, r.N ≤ j → j < r.B → d j = 0)
    (hv : ∀ j, j < r.B → |v j - d j ^ 2| ≤ 3 * u32 * d j ^ 2 + 4 * eta32)
    (hT : |T - ∑ j ∈ Finset.range (r.B), v j| ≤
      ((1 + u32) ^ r.D - 1) * ∑ j ∈ Finset.range (r.B), |v j| +
        eta32 * ((r.B - 1 : ℕ) : ℝ) * (1 + u32) ^ r.D)
    (hvar : |var - T / r.N| ≤ 2 ^ (-22 : ℤ) * |T / r.N| + 2 ^ (-148 : ℤ))
    (hε : |epsf - r.eps| ≤ u32 * r.eps + eta32)
    (hs : |s - (var + epsf)| ≤ 3 * u32 * (|var| + |epsf|) + 4 * eta32) :
    r.eps - 41 / 10 * u32 * r.eps - 22 * eta32 ≤ s ∧
      (r.cond ≤ 1 / 2 → 1 - r.loS ≤ s * r.R ^ 2 ∧ s * r.R ^ 2 ≤ 1 + r.hiS) := by
  have hvc := r.var_chain d v T var hd0 hv hT hvar
  have hN : (0 : ℝ) < r.N := by exact_mod_cast (by have := r.n_ge; omega : 0 < r.N)
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hu0 : (0 : ℝ) < u32 := by unfold u32; positivity
  have hu2 : u32 ≤ 1 / 10 ^ 7 := by unfold u32; norm_num
  have hε0 := r.eps_pos
  have hη0 : (0 : ℝ) < eta32 := by unfold eta32; positivity
  have hV0 : 0 ≤ (∑ j ∈ Finset.range r.N, d j ^ 2) / r.N :=
    div_nonneg (Finset.sum_nonneg fun j _ => sq_nonneg _) hN.le
  have hsc := r.s_chain _ var epsf s hV0 hvc hε hs
  set Q := ∑ j ∈ Finset.range r.N, d j ^ 2 with hQ
  set V0 := Q / r.N with hV0def
  have hD0 : (0 : ℝ) ≤ r.D := Nat.cast_nonneg _
  have hcoef : 0 ≤ 1 - ((r.D : ℝ) + 12) * u32 := by nlinarith
  refine ⟨?_, fun hc => ?_⟩
  · have := abs_le.1 hsc
    nlinarith [mul_nonneg hV0 hcoef]
  have hc0 := r.cond_nonneg
  have hsq : r.R ^ 2 * (r.sig2 + r.eps) = 1 := by
    have hM : 0 < r.sig2 + r.eps := by have := r.sig2_nonneg; linarith
    unfold R
    rw [div_pow, Real.sq_sqrt hM.le]
    field_simp
  have hR := r.R_pos
  have hq0 : 0 < r.R ^ 2 := by positivity
  have hsig := r.sig2_nonneg
  have hεq : r.eps * r.R ^ 2 ≤ 1 := by nlinarith
  have hqε : r.R ^ 2 ≤ 1 / r.eps := by rw [le_div_iff₀ hε0]; linarith
  obtain ⟨hlo0, hhi⟩ := r.variance_scaled m hm hc d hd
  have hĉ := r.cLane_mul_R_le m hm
  have hκ : u32 * (2 + u32) ≤ 20001 / 10000 * u32 := by nlinarith
  have hlo : -(40003 / 10000 * u32 + r.cond / 5) ≤ (V0 - r.sig2) * r.R ^ 2 := by linarith
  have hs1 : (V0 + r.eps) * r.R ^ 2 = 1 + (V0 - r.sig2) * r.R ^ 2 := by linear_combination hsq
  have hnu0 : 0 ≤ r.nuH := by unfold nuH; positivity
  have hVq : V0 * r.R ^ 2 ≤ 1 + r.nuH := by
    nlinarith [mul_nonneg hε0.le hq0.le]
  have hrem : ((r.D : ℝ) + 12) * u32 * V0 * r.R ^ 2 + 41 / 10 * u32 * (r.eps * r.R ^ 2) +
      22 * eta32 * r.R ^ 2 ≤
      ((r.D : ℝ) + 12) * u32 * (1 + r.nuH) + 41 / 10 * u32 + 22 * eta32 / r.eps := by
    have a1 : ((r.D : ℝ) + 12) * u32 * (V0 * r.R ^ 2) ≤ ((r.D : ℝ) + 12) * u32 * (1 + r.nuH) :=
      mul_le_mul_of_nonneg_left hVq (by positivity)
    have a2 : 41 / 10 * u32 * (r.eps * r.R ^ 2) ≤ 41 / 10 * u32 :=
      (mul_le_mul_of_nonneg_left hεq (by positivity)).trans_eq (mul_one _)
    have a3 : 22 * eta32 * r.R ^ 2 ≤ 22 * eta32 / r.eps := by
      calc 22 * eta32 * r.R ^ 2 ≤ 22 * eta32 * (1 / r.eps) :=
            mul_le_mul_of_nonneg_left hqε (mul_nonneg (by norm_num : (0 : ℝ) ≤ 22) hη0.le)
        _ = 22 * eta32 / r.eps := by ring
    linarith
  have hsd := mul_le_mul_of_nonneg_right (abs_le.1 hsc).2 hq0.le
  have hsd' := mul_le_mul_of_nonneg_right (abs_le.1 hsc).1 hq0.le
  unfold loS hiS
  constructor <;> linarith

end RowP

end Claims.LayerNormFwdF16
