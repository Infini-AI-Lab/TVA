import Proof.RowVar

/-!
# The variance against `T = Σ c²/d`, and the standard deviation

`var_T`: the squares, their fusable tree and `/ d` keep the kernel's variance within
`51u·T + 2^-146` of `T`; `var_stage'` and `std_stage'` then bound `√var` and `std32` against `sd`.
-/

namespace Claims.GeluAndMulSparse

open TVA

namespace RowP

variable (r : RowP)

/-- `eta32` as a rational numeral. -/
theorem eta32_eq : eta32 = 1 / 2 ^ 150 := by unfold eta32; norm_num

/-- **The variance against `T`**: within `53u·T + 2^-146`, `T = Σ_{j<d} c²/d`. -/
theorem var_T {c q : ℕ → ℝ} {Q var : ℝ}
    (hc0 : ∀ j, r.d ≤ j → j < 2 ^ r.L → c j = 0)
    (hq : ∀ j, j < 2 ^ r.L → |q j - c j * c j| ≤ u32 * |c j * c j| + eta32)
    (hQ : |Q - ∑ j ∈ Finset.range (2 ^ r.L), q j| ≤
      ((1 + 3 * u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |q j| +
        4 * eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ r.L)
    (hv : |var - Q / r.d| ≤ 2 ^ (-22 : ℤ) * |Q / r.d| + 2 ^ (-148 : ℤ)) :
    |var - (∑ j ∈ Finset.range r.d, c j ^ 2) / r.d| ≤
      53 * u32 * ((∑ j ∈ Finset.range r.d, c j ^ 2) / r.d) + 2 ^ (-146 : ℤ) := by
  have hd := r.d_pos'
  obtain ⟨C, hCdef⟩ : ∃ C, C = ∑ j ∈ Finset.range r.d, c j ^ 2 := ⟨_, rfl⟩
  rw [← hCdef]
  have hC0 : 0 ≤ C := hCdef ▸ Finset.sum_nonneg fun _ _ => sq_nonneg _
  obtain ⟨B, hBdef⟩ : ∃ B : ℝ, B = ((2 ^ r.L : ℕ) : ℝ) := ⟨_, rfl⟩
  have hB : B ≤ 2 * r.d := by
    have := r.b_lt'; rw [hBdef]; push_cast; linarith
  obtain ⟨bm, hbmdef⟩ : ∃ bm : ℝ, bm = ((2 ^ r.L - 1 : ℕ) : ℝ) := ⟨_, rfl⟩
  have hbm : bm ≤ 2 * r.d := hbmdef ▸ r.bm1_le
  have hbm0 : 0 ≤ bm := hbmdef ▸ Nat.cast_nonneg _
  rw [← hbmdef] at hQ
  have hcc : ∑ j ∈ Finset.range (2 ^ r.L), c j * c j = C := by
    rw [r.sum_block' (f := fun j => c j * c j) fun j h1 h2 => by simp [hc0 j h1 h2], hCdef]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hacc : ∑ j ∈ Finset.range (2 ^ r.L), |c j * c j| = C := by
    rw [← hcc]; exact Finset.sum_congr rfl fun j _ => abs_mul_self _
  obtain ⟨SA, hSA⟩ : ∃ SA, SA = ∑ j ∈ Finset.range (2 ^ r.L), |q j| := ⟨_, rfl⟩
  obtain ⟨SQ, hSQ⟩ : ∃ SQ, SQ = ∑ j ∈ Finset.range (2 ^ r.L), q j := ⟨_, rfl⟩
  rw [← hSA, ← hSQ] at hQ
  have hsq : SA ≤ (1 + u32) * C + B * eta32 := by
    rw [hSA]
    calc ∑ j ∈ Finset.range (2 ^ r.L), |q j| ≤
          ∑ j ∈ Finset.range (2 ^ r.L), ((1 + u32) * |c j * c j| + eta32) :=
          Finset.sum_le_sum fun j hj => by
            have h := hq j (Finset.mem_range.mp hj)
            have := abs_sub_abs_le_abs_sub (q j) (c j * c j)
            linarith
      _ = (1 + u32) * C + B * eta32 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, hacc]; simp [hBdef]
  have hqs : |SQ - C| ≤ u32 * C + B * eta32 := by
    have e : SQ - C = ∑ j ∈ Finset.range (2 ^ r.L), (q j - c j * c j) := by
      rw [Finset.sum_sub_distrib, hcc, hSQ]
    rw [e]
    calc _ ≤ ∑ j ∈ Finset.range (2 ^ r.L), |q j - c j * c j| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ Finset.range (2 ^ r.L), (u32 * |c j * c j| + eta32) :=
          Finset.sum_le_sum fun j hj => hq j (Finset.mem_range.mp hj)
      _ = u32 * C + B * eta32 := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, hacc]; simp [hBdef]
  obtain ⟨p3, hp3⟩ : ∃ p3, p3 = (1 + 3 * u32) ^ r.L := ⟨_, rfl⟩
  rw [← hp3] at hQ
  have hE3le : p3 - 1 ≤ 46 * u32 := by
    have h := pow_3u_sub_le r.L_le
    have hL := r.L_le'
    have : 3 * (r.L : ℝ) * u32 * (1 + 2 ^ (-8 : ℤ)) ≤ 3 * 15 * u32 * (1 + 2 ^ (-8 : ℤ)) := by
      gcongr; rw [u32_eq]; norm_num
    rw [← hp3] at h
    rw [u32_eq] at this h ⊢; norm_num at this h ⊢; linarith
  have hE30 : 0 ≤ p3 - 1 := by
    have : 1 ≤ p3 := hp3 ▸ one_le_pow₀ (by rw [u32_eq]; norm_num)
    linarith
  have hp3le : p3 ≤ 1 + 2 ^ (-8 : ℤ) := hp3 ▸ pow_3u_le r.L_le
  have hSA0 : 0 ≤ SA := hSA ▸ Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hB0 : 0 ≤ B := hBdef ▸ Nat.cast_nonneg _
  rw [u32_eq, eta32_eq] at *
  have h1 : (p3 - 1) * SA ≤ 46 * (1 / 16777216) * ((1 + 1 / 16777216) * C + B * (1 / 2 ^ 150)) :=
    mul_le_mul hE3le hsq hSA0 (by norm_num)
  have h2 : 4 * (1 / 2 ^ 150) * bm * p3 ≤ 4 * (1 / 2 ^ 150) * (2 * r.d) * (1 + 2 ^ (-8 : ℤ)) := by
    have : 0 ≤ p3 := by linarith
    gcongr
  have hQC : |Q - C| ≤ 48 * (1 / 16777216) * C + 11 * r.d * (1 / 2 ^ 150) := by
    have k0 := abs_sub_le Q SQ C
    have k1 : |Q - SQ| ≤ 46 * (1 / 16777216) * ((1 + 1 / 16777216) * C + B * (1 / 2 ^ 150)) +
        4 * (1 / 2 ^ 150) * (2 * r.d) * (1 + 2 ^ (-8 : ℤ)) := by linarith
    have k2 : B * (1 / 2 ^ 150 : ℝ) ≤ 2 * r.d * (1 / 2 ^ 150) :=
      mul_le_mul_of_nonneg_right hB (by norm_num)
    norm_num at k1 k2 hqs k0 ⊢
    linarith
  obtain ⟨T, hT⟩ : ∃ T, T = C / r.d := ⟨_, rfl⟩
  rw [← hT]
  have hCT : C = r.d * T := by rw [hT]; field_simp
  have hT0 : 0 ≤ T := hT ▸ div_nonneg hC0 hd.le
  have hQT : |Q / r.d - T| ≤ 48 * (1 / 16777216) * T + 11 * (1 / 2 ^ 150) := by
    have e : Q / r.d - T = (Q - C) / r.d := by rw [hT]; field_simp
    rw [e, abs_div, abs_of_pos hd, div_le_iff₀ hd]
    calc |Q - C| ≤ _ := hQC
      _ = _ := by rw [hCT]; ring
  have hQd : |Q / r.d| ≤ T + 48 * (1 / 16777216) * T + 11 * (1 / 2 ^ 150) := by
    have := abs_sub_abs_le_abs_sub (Q / r.d) T; rw [abs_of_nonneg hT0] at this; linarith
  have h4 : 2 ^ (-22 : ℤ) * |Q / r.d| ≤
      2 ^ (-22 : ℤ) * (T + 48 * (1 / 16777216) * T + 11 * (1 / 2 ^ 150)) :=
    mul_le_mul_of_nonneg_left hQd (by positivity)
  have h5 := abs_sub_le var (Q / r.d) T
  norm_num at h4 h5 hv hQT ⊢
  linarith

/-- Square roots of close values: `|var − T| ≤ ε·T` gives `|√var − √T| ≤ (ε/2 + ε²)·√T`. -/
theorem sqrt_close {var T ε : ℝ} (hT : 0 < T) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 100)
    (h : |var - T| ≤ ε * T) : |Real.sqrt var - Real.sqrt T| ≤ (ε / 2 + ε ^ 2) * Real.sqrt T := by
  obtain ⟨ρ, hρ⟩ : ∃ ρ, ρ = Real.sqrt T := ⟨_, rfl⟩
  obtain ⟨w, hw⟩ : ∃ w, w = Real.sqrt var := ⟨_, rfl⟩
  rw [← hρ, ← hw]
  have hρ0 : 0 < ρ := hρ ▸ Real.sqrt_pos.mpr hT
  have hρT : ρ ^ 2 = T := hρ ▸ Real.sq_sqrt hT.le
  have hvlo : (1 - ε) * T ≤ var := by have := (abs_le.mp h).1; linarith
  have hv0 : 0 ≤ var := by have : 0 ≤ (1 - ε) * T := by apply mul_nonneg <;> linarith
                           linarith
  have hw0 : 0 ≤ w := hw ▸ Real.sqrt_nonneg _
  have hwv : w ^ 2 = var := hw ▸ Real.sq_sqrt hv0
  have hwlo : (1 - ε) * ρ ≤ w := by
    have h1 : 0 ≤ 1 - ε := by linarith
    have h2 : ((1 - ε) * ρ) ^ 2 ≤ w ^ 2 := by
      rw [hwv, mul_pow, hρT]
      have : (1 - ε) ^ 2 ≤ 1 - ε := by nlinarith
      have := mul_le_mul_of_nonneg_right this hT.le
      linarith
    exact (pow_le_pow_iff_left₀ (mul_nonneg h1 hρ0.le) hw0 two_ne_zero).mp h2
  have hsum : (2 - ε) * ρ ≤ w + ρ := by linarith
  have hprod : |w - ρ| * (w + ρ) = |var - T| := by
    rw [← hwv, ← hρT, show w ^ 2 - ρ ^ 2 = (w - ρ) * (w + ρ) by ring, abs_mul,
      abs_of_nonneg (by linarith : 0 ≤ w + ρ)]
  have h3 : |w - ρ| * ((2 - ε) * ρ) ≤ ε * ρ ^ 2 := by
    have := mul_le_mul_of_nonneg_left hsum (abs_nonneg (w - ρ))
    rw [hρT]; linarith
  have h2e : 0 < 2 - ε := by linarith
  have h4 : |w - ρ| ≤ ε * ρ / (2 - ε) := by
    rw [le_div_iff₀ h2e]
    have e : |w - ρ| * ((2 - ε) * ρ) = (|w - ρ| * (2 - ε)) * ρ := by ring
    rw [e, sq] at h3
    have : |w - ρ| * (2 - ε) * ρ ≤ (ε * ρ) * ρ := by linarith
    exact le_of_mul_le_mul_right this hρ0
  have h5 : ε * ρ / (2 - ε) ≤ (ε / 2 + ε ^ 2) * ρ := by
    rw [div_le_iff₀ h2e]
    have : ε ≤ (ε / 2 + ε ^ 2) * (2 - ε) := by nlinarith
    nlinarith
  linarith

/-- `R1 ≤ 0.157·sd` on a row with `Ok`. -/
theorem R1_le (hok : r.Ok) :
    3 * u32 * r.sd + 10 * u32 * r.aMean + 2 ^ (-136 : ℤ) ≤ 157 / 1000 * r.sd := by
  have hsdg := hok.sd_ge
  have hAsd := hok.aMean_le_sd
  have a1 : 10 * u32 * r.aMean ≤ 10 * u32 * (2 ^ 18 * r.sd) :=
    mul_le_mul_of_nonneg_left hAsd (by rw [u32_eq]; norm_num)
  have a2 : (2 : ℝ) ^ (-136 : ℤ) ≤ 2 ^ (-74 : ℤ) * r.sd := by
    have : (2 : ℝ) ^ (-136 : ℤ) = 2 ^ (-74 : ℤ) * 2 ^ (-62 : ℤ) := by norm_num
    rw [this]; exact mul_le_mul_of_nonneg_left hsdg (by positivity)
  have hsd0 := r.sd_nonneg'
  rw [u32_eq] at a1 ⊢
  norm_num at a1 a2 ⊢
  linarith

/-- **The variance** (line 77): in `tl.sqrt`'s domain, and its square root close to `sd`. -/
theorem var_stage' (hok : r.Ok) {mean0 : ℝ} (hm0 : |mean0 - r.mu| ≤ r.meanErr) {c q : ℕ → ℝ}
    {Q var : ℝ}
    (hc : ∀ j, j < r.d → |c j - (r.g j - mean0)| ≤ 3 * u32 * (|r.g j| + |mean0|) + 4 * eta32)
    (hc0 : ∀ j, r.d ≤ j → j < 2 ^ r.L → c j = 0)
    (hq : ∀ j, j < 2 ^ r.L → |q j - c j * c j| ≤ u32 * |c j * c j| + eta32)
    (hQ : |Q - ∑ j ∈ Finset.range (2 ^ r.L), q j| ≤
      ((1 + 3 * u32) ^ r.L - 1) * ∑ j ∈ Finset.range (2 ^ r.L), |q j| +
        4 * eta32 * ((2 ^ r.L - 1 : ℕ) : ℝ) * (1 + 3 * u32) ^ r.L)
    (hv : |var - Q / r.d| ≤ 2 ^ (-22 : ℤ) * |Q / r.d| + 2 ^ (-148 : ℤ)) :
    2 ^ (-126 : ℤ) ≤ var ∧ var ≤ 2 ^ 80 ∧
      |Real.sqrt var - r.sd| ≤ ((r.L : ℝ) + 11) * u32 * r.aMean + 36 * u32 * r.sd := by
  have hrho := r.rho_bounds hok hm0 hc
  have hvT := r.var_T hc0 hq hQ hv
  have hR1 := r.R1_le hok
  obtain ⟨T, hT⟩ : ∃ T, T = (∑ j ∈ Finset.range r.d, c j ^ 2) / r.d := ⟨_, rfl⟩
  rw [← hT] at hrho hvT
  obtain ⟨R1, hR1d⟩ : ∃ R1, R1 = 3 * u32 * r.sd + 10 * u32 * r.aMean + 2 ^ (-136 : ℤ) := ⟨_, rfl⟩
  rw [← hR1d] at hrho hR1
  have hT0 : 0 ≤ T := hT ▸ div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) r.d_pos'.le
  obtain ⟨ρ, hρ⟩ : ∃ ρ, ρ = Real.sqrt T := ⟨_, rfl⟩
  rw [← hρ] at hrho
  have hρ0 : 0 ≤ ρ := hρ ▸ Real.sqrt_nonneg _
  have hρT : ρ ^ 2 = T := hρ ▸ Real.sq_sqrt hT0
  have hsdg := hok.sd_ge
  have hsd33 := r.sd_le'
  have hρlo : 843 / 1000 * r.sd ≤ ρ := by have := (abs_le.mp hrho).1; linarith
  have hρhi : ρ ≤ r.sd + R1 := by have := (abs_le.mp hrho).2; linarith
  have hsd62 : (1 / 2 ^ 62 : ℝ) ≤ r.sd := by
    have : (2 : ℝ) ^ (-62 : ℤ) = 1 / 2 ^ 62 := by norm_num
    rw [this] at hsdg; exact hsdg
  have hρlo2 : 843 / 1000 * (1 / 2 ^ 62) ≤ ρ := by nlinarith
  have hTlo : (843 / 1000 * (1 / 2 ^ 62)) ^ 2 ≤ T := by
    rw [← hρT]; exact pow_le_pow_left₀ (by norm_num) hρlo2 2
  have hTpos : 0 < T := lt_of_lt_of_le (by norm_num) hTlo
  have hU : u32 = 1 / 16777216 := u32_eq
  -- `|var − T| ≤ 59u·T`
  have hab : (2 : ℝ) ^ (-146 : ℤ) ≤ 6 * (1 / 16777216) * T := by
    have : (2 : ℝ) ^ (-146 : ℤ) ≤ 6 * (1 / 16777216) * (843 / 1000 * (1 / 2 ^ 62)) ^ 2 := by
      norm_num
    linarith
  rw [hU] at hvT
  have hvT2 : |var - T| ≤ 59 * (1 / 16777216) * T := by linarith
  have hvlo : (1 - 59 * (1 / 16777216)) * T ≤ var := by have := (abs_le.mp hvT2).1; linarith
  have hvhi : var ≤ (1 + 59 * (1 / 16777216)) * T := by have := (abs_le.mp hvT2).2; linarith
  have hsc := sqrt_close hTpos (by norm_num) (by norm_num) hvT2
  rw [← hρ] at hsc
  refine ⟨?_, ?_, ?_⟩
  · rw [show (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 by norm_num]
    have : (1 / 2 ^ 126 : ℝ) ≤ (1 - 59 * (1 / 16777216)) * (843 / 1000 * (1 / 2 ^ 62)) ^ 2 := by
      norm_num
    have := mul_le_mul_of_nonneg_left hTlo (show (0 : ℝ) ≤ 1 - 59 * (1 / 16777216) by norm_num)
    linarith
  · have hρ34 : ρ ≤ 2 ^ 34 := by nlinarith
    have hT68 : T ≤ 2 ^ 68 := by
      rw [← hρT]; calc ρ ^ 2 ≤ (2 ^ 34) ^ 2 := pow_le_pow_left₀ hρ0 hρ34 2
        _ = 2 ^ 68 := by norm_num
    have := mul_le_mul_of_nonneg_left hT68 (show (0 : ℝ) ≤ 1 + 59 * (1 / 16777216) by norm_num)
    linarith
  · have hk : ((59 * (1 / 16777216 : ℝ)) / 2 + (59 * (1 / 16777216)) ^ 2) ≤ 296 / 10 * (1 / 16777216) := by
      norm_num
    have h1 : |Real.sqrt var - ρ| ≤ 296 / 10 * (1 / 16777216) * ρ :=
      hsc.trans (mul_le_mul_of_nonneg_right hk hρ0)
    have h2 : 296 / 10 * (1 / 16777216) * ρ ≤ 296 / 10 * (1 / 16777216) * (r.sd + R1) :=
      mul_le_mul_of_nonneg_left hρhi (by norm_num)
    have h3 := abs_sub_le (Real.sqrt var) ρ r.sd
    have hA := r.aMean_nonneg'
    have hLA : 0 ≤ (r.L : ℝ) * r.aMean := by positivity
    have a2 : (2 : ℝ) ^ (-136 : ℤ) ≤ 2 ^ (-74 : ℤ) * r.sd := by
      have : (2 : ℝ) ^ (-136 : ℤ) = 2 ^ (-74 : ℤ) * 2 ^ (-62 : ℤ) := by norm_num
      rw [this]; exact mul_le_mul_of_nonneg_left hsdg (by positivity)
    have e1 : ((r.L : ℝ) + 11) * u32 * r.aMean = u32 * (r.L * r.aMean) + 11 * u32 * r.aMean := by ring
    rw [e1, hU]
    rw [hU] at hR1d
    norm_num at hR1d a2 h1 h2 ⊢
    nlinarith

/-- **The standard deviation** (line 80, `tl.sqrt`, `Device.SqrtF32`). -/
theorem std_stage' (hok : r.Ok) {var std32 : ℝ} (_hvlo : 2 ^ (-126 : ℤ) ≤ var)
    (hsv : |Real.sqrt var - r.sd| ≤ ((r.L : ℝ) + 11) * u32 * r.aMean + 36 * u32 * r.sd)
    (hstd : |std32 - Real.sqrt var| ≤ 2 ^ (-22 : ℤ) * Real.sqrt var + 2 ^ (-149 : ℤ)) :
    |std32 - r.sd| ≤ r.stdErr := by
  have hA := r.aMean_nonneg'
  have hAge := hok.aMean_ge
  have hsdA := r.sd_le_aMean
  have hsd0 := r.sd_nonneg'
  have hL := r.L_le'
  have hU : u32 = 1 / 16777216 := u32_eq
  obtain ⟨LA, hLA⟩ : ∃ LA, LA = (r.L : ℝ) * r.aMean := ⟨_, rfl⟩
  have hLA0 : 0 ≤ LA := hLA ▸ by positivity
  have hLA15 : LA ≤ 15 * r.aMean := hLA ▸ mul_le_mul_of_nonneg_right hL hA
  have e0 : ((r.L : ℝ) + 11) * u32 * r.aMean = u32 * LA + 11 * u32 * r.aMean := by rw [hLA]; ring
  rw [e0] at hsv
  obtain ⟨w, hw⟩ : ∃ w, w = Real.sqrt var := ⟨_, rfl⟩
  rw [← hw] at hsv hstd
  have hsq : w ≤ r.sd + (u32 * LA + 11 * u32 * r.aMean + 36 * u32 * r.sd) := by
    have := (abs_le.mp hsv).2; linarith
  have h1 : 2 ^ (-22 : ℤ) * w ≤ 2 ^ (-22 : ℤ) * (r.sd + (u32 * LA + 11 * u32 * r.aMean + 36 * u32 * r.sd)) :=
    mul_le_mul_of_nonneg_left hsq (by positivity)
  have h2 := abs_sub_le std32 w r.sd
  have h4 : (2 : ℝ) ^ (-149 : ℤ) ≤ 2 ^ (-79 : ℤ) * r.aMean := by
    have : (2 : ℝ) ^ (-149 : ℤ) = 2 ^ (-79 : ℤ) * 2 ^ (-70 : ℤ) := by norm_num
    rw [this]; exact mul_le_mul_of_nonneg_left hAge (by positivity)
  unfold stdErr
  have e1 : ((r.L : ℝ) + 12) * u32 * r.aMean = u32 * LA + 12 * u32 * r.aMean := by rw [hLA]; ring
  rw [e1]
  rw [hU] at hsv hsq h1 ⊢
  norm_num at hsv h1 h4 hstd h2 ⊢
  nlinarith

end RowP

end Claims.GeluAndMulSparse
