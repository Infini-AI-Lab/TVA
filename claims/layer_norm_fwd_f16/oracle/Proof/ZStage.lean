import Proof.RstdStage

namespace Claims.LayerNormFwdF16

namespace RowP

variable (r : RowP)

set_option maxHeartbeats 1600000 in
/-- **A standardised value on a well-conditioned row**: the lane `d'` (a computed `x j − m`) times
`rstd` (within `thetaS · R` of `R`), rounded once, is within `zTol j` of `zRef j`. -/
theorem z_small (j : ℕ) (_hj : j < r.N) (hc : r.cond ≤ 1 / 2) (m rstd d' z : ℝ)
    (hm : |m - r.mu| ≤ 11 / 20 * r.Delta) (hrstd : |rstd - r.R| ≤ r.thetaS * r.R)
    (hd' : LaneErr (r.x j) m d') (hz : |z - d' * rstd| ≤ u32 * |d' * rstd| + eta32) :
    |z - r.zRef j| ≤ r.zTol j := by
  have hR := r.R_pos
  have hΘ := r.thetaS_le hc
  have hΘ0 : 0 ≤ r.thetaS := by
    unfold RowP.thetaS
    have := r.cond_nonneg
    have := r.eps_pos
    have : (0 : ℝ) ≤ r.D := Nat.cast_nonneg _
    unfold u32 eta32
    positivity
  have hC := r.thetaS_add_le hc
  have hcR := r.cLane_mul_R_le m hm
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have hu1 : u32 ≤ 1 / 1000000 := by unfold u32; norm_num
  have hc0 := r.cond_nonneg
  -- names: `a = x - mu`, `δ = m - mu`, `ξ` the lane's error, `ρ = rstd`
  set a := r.x j - r.mu with ha
  set δ := m - r.mu with hδ
  set ρ := rstd with hρ
  set Θ := r.thetaS with hΘdef
  set c := r.cLane m with hcdef
  set κ := u32 * (2 + u32) with hκ
  have hκ0 : 0 ≤ κ := by positivity
  have hκ1 : κ ≤ 21 / 10 * u32 := by rw [hκ]; nlinarith
  have hξ : |d' - (a - δ)| ≤ κ * |a| + c := r.lane_split hd'
  have hδR : |δ| * r.R ≤ 11 / 20 * r.cond := by
    unfold RowP.cond
    calc |δ| * r.R ≤ (11 / 20 * r.Delta) * r.R := mul_le_mul_of_nonneg_right hm hR.le
      _ = _ := by ring
  have hρb : |ρ| ≤ (1 + Θ) * r.R := by
    have h1 := abs_le.mp hrstd
    rw [abs_le]; constructor <;> nlinarith
  have hc0' : 0 ≤ c := by
    rw [hcdef]; unfold RowP.cLane u32 eta32; positivity
  have hzRef : |r.zRef j| = |a| * r.R := by
    unfold RowP.zRef; rw [abs_mul, abs_of_pos hR]
  -- the exact identity
  have hid : d' * ρ - a * r.R = (d' - (a - δ)) * ρ + a * (ρ - r.R) - δ * ρ := by ring
  have h1 : |d' * ρ - a * r.R| ≤ (κ * |a| + c) * ((1 + Θ) * r.R) + |a| * (Θ * r.R) +
      |δ| * ((1 + Θ) * r.R) := by
    rw [hid]
    calc _ ≤ |(d' - (a - δ)) * ρ + a * (ρ - r.R)| + |δ * ρ| := abs_sub _ _
      _ ≤ |(d' - (a - δ)) * ρ| + |a * (ρ - r.R)| + |δ * ρ| := by
          gcongr; exact abs_add_le _ _
      _ ≤ _ := by
          rw [abs_mul, abs_mul, abs_mul]
          gcongr
  have h2 : |d' * ρ| ≤ ((1 + κ) * |a| + |δ| + c) * ((1 + Θ) * r.R) := by
    have hd2 : |d'| ≤ (1 + κ) * |a| + |δ| + c := by
      have e : d' = (a - δ) + (d' - (a - δ)) := by ring
      calc |d'| = |(a - δ) + (d' - (a - δ))| := by rw [← e]
        _ ≤ |a - δ| + |d' - (a - δ)| := abs_add_le _ _
        _ ≤ (|a| + |δ|) + (κ * |a| + c) := add_le_add (abs_sub _ _) hξ
        _ = _ := by ring
    rw [abs_mul]
    exact mul_le_mul hd2 hρb (abs_nonneg _) (by positivity)
  have hκ2 : κ ≤ 20001 / 10000 * u32 := by rw [hκ]; nlinarith
  have hP : (1 + Θ) ≤ 9 / 8 := by linarith
  have hP0 : 0 ≤ 1 + Θ := by linarith
  have hK : |z - a * r.R| ≤ |a| * r.R * (Θ + κ * (1 + Θ) + u32 * (1 + κ) * (1 + Θ)) +
      (1 + u32) * (1 + Θ) * (|δ| * r.R + c * r.R) + eta32 := by
    calc |z - a * r.R| ≤ |z - d' * ρ| + |d' * ρ - a * r.R| := abs_sub_le _ _ _
      _ ≤ (u32 * |d' * ρ| + eta32) + ((κ * |a| + c) * ((1 + Θ) * r.R) + |a| * (Θ * r.R) +
          |δ| * ((1 + Θ) * r.R)) := add_le_add hz h1
      _ ≤ (u32 * (((1 + κ) * |a| + |δ| + c) * ((1 + Θ) * r.R)) + eta32) +
          ((κ * |a| + c) * ((1 + Θ) * r.R) + |a| * (Θ * r.R) + |δ| * ((1 + Θ) * r.R)) := by
          gcongr
      _ = _ := by ring
  -- the pieces against `zTol`
  have hC1 : Θ + κ * (1 + Θ) + u32 * (1 + κ) * (1 + Θ) ≤ r.gam + r.mTerm := by
    have : κ * (1 + Θ) + u32 * (1 + κ) * (1 + Θ) ≤ 17 / 5 * u32 := by
      have h3 : κ * (1 + Θ) + u32 * (1 + κ) * (1 + Θ) = (1 + Θ) * (κ + u32 + u32 * κ) := by ring
      rw [h3]
      have h4 : κ + u32 + u32 * κ ≤ 30003 / 10000 * u32 := by nlinarith
      calc (1 + Θ) * (κ + u32 + u32 * κ) ≤ (9 / 8) * (30003 / 10000 * u32) :=
            mul_le_mul hP h4 (by positivity) (by norm_num)
        _ ≤ 17 / 5 * u32 := by nlinarith
    linarith
  have hE : (1 + u32) * (1 + Θ) * (|δ| * r.R + c * r.R) ≤ r.cond := by
    have h1' : (1 + u32) * (1 + Θ) ≤ 113 / 100 := by nlinarith
    have h2' : |δ| * r.R + c * r.R ≤ 13 / 20 * r.cond := by linarith
    calc (1 + u32) * (1 + Θ) * (|δ| * r.R + c * r.R) ≤ (113 / 100) * (13 / 20 * r.cond) :=
          mul_le_mul h1' h2' (by positivity) (by norm_num)
      _ ≤ r.cond := by nlinarith
  have hη : eta32 ≤ 2 ^ (-140 : ℤ) * (1 + r.R) := by
    unfold eta32
    have h1 : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-140 : ℤ) := by norm_num
    have h2 : 0 ≤ (2 : ℝ) ^ (-140 : ℤ) * r.R := by positivity
    linarith
  have hgam : 0 ≤ r.gam := by
    unfold RowP.gam
    have := r.eps_pos
    have : (0 : ℝ) ≤ r.D := Nat.cast_nonneg _
    positivity
  have hm0 : 0 ≤ r.mTerm := by unfold RowP.mTerm; positivity
  have ht0 : 0 ≤ |a| * r.R := by positivity
  have hzTol : r.zTol j = (1 + 2 * r.gam) * (|r.zRef j| * (r.gam + r.mTerm) + r.cond) +
      2 ^ (-140 : ℤ) * (1 + r.R) := by
    unfold RowP.zTol; rw [ite_eq_left hc]
  have hzr : r.zRef j = a * r.R := rfl
  have hmain : |a| * r.R * (Θ + κ * (1 + Θ) + u32 * (1 + κ) * (1 + Θ)) ≤
      |a| * r.R * (r.gam + r.mTerm) := mul_le_mul_of_nonneg_left hC1 ht0
  have hnn : 0 ≤ |a| * r.R * (r.gam + r.mTerm) + r.cond := by positivity
  have hfin : |a| * r.R * (r.gam + r.mTerm) + r.cond + 2 ^ (-140 : ℤ) * (1 + r.R) ≤
      (1 + 2 * r.gam) * (|a| * r.R * (r.gam + r.mTerm) + r.cond) + 2 ^ (-140 : ℤ) * (1 + r.R) := by
    nlinarith [mul_nonneg hgam hnn]
  rw [hzTol, hzr, abs_mul, abs_of_pos hR]
  have h6 : |a| * r.R * (Θ + κ * (1 + Θ) + u32 * (1 + κ) * (1 + Θ)) +
      (1 + u32) * (1 + Θ) * (|δ| * r.R + c * r.R) + eta32 ≤
      |a| * r.R * (r.gam + r.mTerm) + r.cond + 2 ^ (-140 : ℤ) * (1 + r.R) := by
    linarith [hmain, hE, hη]
  exact hK.trans (h6.trans hfin)

set_option maxHeartbeats 1600000 in
/-- **A standardised value on an ill-conditioned row** (`cond > 1/2`): `rstd ≤ (1 + gam)/√eps`
bounds it. -/
theorem z_big (j : ℕ) (_hj : j < r.N) (hc : ¬ r.cond ≤ 1 / 2) (m rstd d' z : ℝ)
    (hm : |m - r.mu| ≤ 11 / 20 * r.Delta) (hrstd0 : 0 < rstd)
    (hrstd : rstd ≤ (1 + r.gam) / Real.sqrt r.eps)
    (hd' : LaneErr (r.x j) m d') (hz : |z - d' * rstd| ≤ u32 * |d' * rstd| + eta32) :
    |z - r.zRef j| ≤ r.zTol j := by
  have hs := Real.sqrt_pos.mpr r.eps_pos
  obtain ⟨hg1, hg2⟩ := r.gam_bounds
  have hDc := r.delta_add_cLane_le m hm
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have hu1 : u32 ≤ 1 / 1000000 := by unfold u32; norm_num
  set a := r.x j - r.mu with ha
  set κ := u32 * (2 + u32) with hκ
  have hκ0 : 0 ≤ κ := by positivity
  have hκ2 : κ ≤ 20001 / 10000 * u32 := by rw [hκ]; nlinarith
  have hξ : |d' - (a - (m - r.mu))| ≤ κ * |a| + r.cLane m := r.lane_split hd'
  have hΔ0 := r.Delta_nonneg
  -- `|d'| ≤ (1 + κ)(|a| + Delta)`
  have hd1 : |d'| ≤ (1 + κ) * (|a| + r.Delta) := by
    have e : d' = (a - (m - r.mu)) + (d' - (a - (m - r.mu))) := by ring
    calc |d'| = |(a - (m - r.mu)) + (d' - (a - (m - r.mu)))| := by rw [← e]
      _ ≤ |a - (m - r.mu)| + |d' - (a - (m - r.mu))| := abs_add_le _ _
      _ ≤ (|a| + |m - r.mu|) + (κ * |a| + r.cLane m) := add_le_add (abs_sub _ _) hξ
      _ ≤ (1 + κ) * (|a| + r.Delta) := by nlinarith [mul_nonneg hκ0 hΔ0]
  have hd2 : |d' * rstd| ≤ (1 + κ) * (|a| + r.Delta) * ((1 + r.gam) / Real.sqrt r.eps) := by
    rw [abs_mul, abs_of_pos hrstd0]
    exact mul_le_mul hd1 hrstd hrstd0.le (by positivity)
  have hz1 : |z| ≤ (1 + u32) * ((1 + κ) * (|a| + r.Delta) * ((1 + r.gam) / Real.sqrt r.eps)) + eta32 := by
    have h1 := abs_sub_abs_le_abs_sub z (d' * rstd)
    have h2 : (1 + u32) * |d' * rstd| ≤ (1 + u32) * ((1 + κ) * (|a| + r.Delta) * ((1 + r.gam) / Real.sqrt r.eps)) :=
      mul_le_mul_of_nonneg_left hd2 (by linarith)
    nlinarith
  have hfac : (1 + u32) * (1 + κ) * (1 + r.gam) ≤ 1 + 2 * r.gam := by
    have h1 : (1 + u32) * (1 + κ) ≤ 1 + 31 / 10 * u32 := by nlinarith
    have h2 : (1 + u32) * (1 + κ) * (1 + r.gam) ≤ (1 + 31 / 10 * u32) * (1 + r.gam) :=
      mul_le_mul_of_nonneg_right h1 (by linarith)
    have h3 : 31 / 10 * u32 * r.gam ≤ 31 / 10 * u32 * (1 / 100) :=
      mul_le_mul_of_nonneg_left hg2 (by positivity)
    nlinarith
  have hq : 0 ≤ (|a| + r.Delta) / Real.sqrt r.eps := by positivity
  have hz2 : |z| ≤ (1 + 2 * r.gam) * (|a| + r.Delta) / Real.sqrt r.eps + 2 ^ (-140 : ℤ) := by
    have e : (1 + u32) * ((1 + κ) * (|a| + r.Delta) * ((1 + r.gam) / Real.sqrt r.eps)) =
        ((1 + u32) * (1 + κ) * (1 + r.gam)) * ((|a| + r.Delta) / Real.sqrt r.eps) := by ring
    have h3 : ((1 + u32) * (1 + κ) * (1 + r.gam)) * ((|a| + r.Delta) / Real.sqrt r.eps) ≤
        (1 + 2 * r.gam) * ((|a| + r.Delta) / Real.sqrt r.eps) :=
      mul_le_mul_of_nonneg_right hfac hq
    have h4 : (1 + 2 * r.gam) * ((|a| + r.Delta) / Real.sqrt r.eps) =
        (1 + 2 * r.gam) * (|a| + r.Delta) / Real.sqrt r.eps := by ring
    have h5 : eta32 ≤ 2 ^ (-140 : ℤ) := by unfold eta32; norm_num
    calc |z| ≤ (1 + u32) * ((1 + κ) * (|a| + r.Delta) * ((1 + r.gam) / Real.sqrt r.eps)) + eta32 := hz1
      _ = ((1 + u32) * (1 + κ) * (1 + r.gam)) * ((|a| + r.Delta) / Real.sqrt r.eps) + eta32 := by rw [e]
      _ ≤ (1 + 2 * r.gam) * ((|a| + r.Delta) / Real.sqrt r.eps) + eta32 := add_le_add h3 le_rfl
      _ = (1 + 2 * r.gam) * (|a| + r.Delta) / Real.sqrt r.eps + eta32 := by rw [h4]
      _ ≤ _ := add_le_add le_rfl h5
  have hzTol : r.zTol j = |r.zRef j| + (1 + 2 * r.gam) * (|r.x j - r.mu| + r.Delta) / Real.sqrt r.eps +
      2 ^ (-140 : ℤ) * (1 + 1 / Real.sqrt r.eps) := by
    unfold RowP.zTol; rw [ite_eq_right hc]
  rw [hzTol]
  have h6 := abs_sub z (r.zRef j)
  have h7 : (0 : ℝ) ≤ 2 ^ (-140 : ℤ) * (1 / Real.sqrt r.eps) := by positivity
  have h8 : 2 ^ (-140 : ℤ) * (1 + 1 / Real.sqrt r.eps) = 2 ^ (-140 : ℤ) + 2 ^ (-140 : ℤ) * (1 / Real.sqrt r.eps) := by
    ring
  linarith

end RowP

/-- `|z w| + |b| ≤ 2^15` when `z` is within `zt` of `zr` and `|w|(|zr| + zt) + |b| ≤ 2^15`. -/
theorem zw_room {zr zt w b z : ℝ} (hz : |z - zr| ≤ zt)
    (hroom : |w| * (|zr| + zt) + |b| ≤ 2 ^ 15) : |z * w| + |b| ≤ 2 ^ 15 := by
  have h1 : |z| ≤ |zr| + zt := by
    have := abs_sub_abs_le_abs_sub z zr
    linarith
  rw [abs_mul]
  nlinarith [abs_nonneg w, abs_nonneg z]

/-- **The output, before its last rounding has room**: with `z` within `zt` of `zr` and
`|w|(|zr| + zt) + |b| ≤ 2^15`, the `float32` `y32 = z w + b` stays below `65504`. -/
theorem y32_room (zr zt w b z y32 : ℝ) (hz : |z - zr| ≤ zt)
    (hroom : |w| * (|zr| + zt) + |b| ≤ 2 ^ 15)
    (hy32 : |y32 - (z * w + b)| ≤ 3 * u32 * (|z * w| + |b|) + 4 * eta32) : |y32| ≤ 65504 := by
  have hS := zw_room hz hroom
  have h1 : |z * w + b| ≤ |z * w| + |b| := abs_add_le _ _
  have h2 := abs_sub_abs_le_abs_sub y32 (z * w + b)
  have h3 : 3 * u32 * (|z * w| + |b|) ≤ 3 * u32 * 2 ^ 15 := by unfold u32; gcongr
  have h4 : (2 : ℝ) ^ 15 + 3 * u32 * 2 ^ 15 + 4 * eta32 ≤ 65504 := by unfold u32 eta32; norm_num
  linarith

/-- **The output**: `y32` within `3u(|z w| + |b|) + 4η` of `z w + b` and `y16` within a `float16`
rounding of `y32` are within `yTol` of `zr w + b`. -/
theorem y_stage (zr zt w b z y32 y16 : ℝ) (hz : |z - zr| ≤ zt)
    (_hroom : |w| * (|zr| + zt) + |b| ≤ 2 ^ 15)
    (hy32 : |y32 - (z * w + b)| ≤ 3 * u32 * (|z * w| + |b|) + 4 * eta32)
    (hy16 : |y16 - y32| ≤ |y32| * 2 ^ (-11 : ℤ) + 2 ^ (-25 : ℤ)) :
    |y16 - (zr * w + b)| ≤ 2 ^ (-9 : ℤ) * (|w| * |zr| + |b|) + 2 * |w| * zt + 2 ^ (-24 : ℤ) := by
  have hzt : 0 ≤ zt := (abs_nonneg _).trans hz
  have hz1 : |z| ≤ |zr| + zt := by
    have := abs_sub_abs_le_abs_sub z zr
    linarith
  -- the magnitudes: `yMag = |w||zr| + |b|`, `S = |z w| + |b| ≤ yMag + |w| zt`
  have hS : |z * w| + |b| ≤ (|w| * |zr| + |b|) + |w| * zt := by
    rw [abs_mul]; nlinarith [abs_nonneg w]
  have hS0 : 0 ≤ |z * w| + |b| := by positivity
  have hm0 : 0 ≤ |w| * |zr| + |b| := by positivity
  have hwz : 0 ≤ |w| * zt := mul_nonneg (abs_nonneg _) hzt
  -- `|y32 - (zr w + b)|`
  have h1 : |y32 - (zr * w + b)| ≤ |w| * zt + 3 * u32 * ((|w| * |zr| + |b|) + |w| * zt) + 4 * eta32 := by
    have e : y32 - (zr * w + b) = (y32 - (z * w + b)) + (z - zr) * w := by ring
    rw [e]
    have h2 : |(z - zr) * w| ≤ |w| * zt := by
      rw [abs_mul, mul_comm]; exact mul_le_mul_of_nonneg_left hz (abs_nonneg _)
    have h3 := abs_add_le (y32 - (z * w + b)) ((z - zr) * w)
    have h4 : 3 * u32 * (|z * w| + |b|) ≤ 3 * u32 * ((|w| * |zr| + |b|) + |w| * zt) := by
      unfold u32; gcongr
    linarith
  set A := |w| * |zr| + |b| with hA
  set Z := |w| * zt with hZ
  have hA0 : 0 ≤ A := hm0
  have h5 : |y32| ≤ (A + Z) + 3 * u32 * (A + Z) + 4 * eta32 := by
    have h6 : |zr * w + b| ≤ A := by
      calc |zr * w + b| ≤ |zr * w| + |b| := abs_add_le _ _
        _ = A := by rw [abs_mul, mul_comm]
    have h7 := abs_sub_abs_le_abs_sub y32 (zr * w + b)
    linarith
  have hc1 : 2 ^ (-11 : ℤ) * (1 + 3 * u32) + 3 * u32 ≤ 2 ^ (-9 : ℤ) := by unfold u32; norm_num
  have hK : 2 ^ (-25 : ℤ) + 4 * eta32 * (1 + 2 ^ (-11 : ℤ)) ≤ (2 : ℝ) ^ (-24 : ℤ) := by
    unfold eta32; norm_num
  have h8 : |y32| * 2 ^ (-11 : ℤ) ≤ (A + Z + 3 * u32 * (A + Z) + 4 * eta32) * 2 ^ (-11 : ℤ) :=
    mul_le_mul_of_nonneg_right h5 (by positivity)
  have h9 : |y16 - (zr * w + b)| ≤ |y16 - y32| + |y32 - (zr * w + b)| := by
    have e : y16 - (zr * w + b) = (y16 - y32) + (y32 - (zr * w + b)) := by ring
    rw [e]; exact abs_add_le _ _
  have hu0 : 0 ≤ u32 := by unfold u32; positivity
  have hAZ : 0 ≤ A + Z := by positivity
  have hcc : (2 ^ (-11 : ℤ) * (1 + 3 * u32) + 3 * u32) * (A + Z) ≤ 2 ^ (-9 : ℤ) * (A + Z) :=
    mul_le_mul_of_nonneg_right hc1 hAZ
  have hcA : 0 ≤ (2 ^ (-9 : ℤ) - (2 ^ (-11 : ℤ) * (1 + 3 * u32) + 3 * u32)) * Z :=
    mul_nonneg (by linarith) (by positivity)
  have eg : 2 * |w| * zt = 2 * Z := by rw [hZ]; ring
  rw [eg]
  have e1 : (A + Z + 3 * u32 * (A + Z) + 4 * eta32) * 2 ^ (-11 : ℤ) =
      (A + Z) * (2 ^ (-11 : ℤ) * (1 + 3 * u32)) + 4 * eta32 * 2 ^ (-11 : ℤ) := by ring
  have e2 : (2 ^ (-11 : ℤ) * (1 + 3 * u32) + 3 * u32) * (A + Z) =
      (A + Z) * (2 ^ (-11 : ℤ) * (1 + 3 * u32)) + 3 * u32 * (A + Z) := by ring
  have h10 : (2 : ℝ) ^ (-9 : ℤ) ≤ 1 := by norm_num
  nlinarith

end Claims.LayerNormFwdF16
