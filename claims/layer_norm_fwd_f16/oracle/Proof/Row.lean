import Mathlib

/-!
# One row, as plain real numbers

The numerical part of the proof does not look at floats: it works on a `RowP`, the data of one row
as reals (the length `N`, the block `2^D`, the values `x`, the real `eps`), with the same
definitions as the claim's `meanRef`, `meanTol`, `cond`, `gam`, ... (`Proof.Bridge` shows that the
claim's quantities are these).
-/

namespace Claims.LayerNormFwdF16

/-- The rounding unit of `float32`, `2^-24`. -/
noncomputable def u32 : ℝ := 2 ^ (-24 : ℤ)

/-- The underflow unit of `float32`, `2^-150`. -/
noncomputable def eta32 : ℝ := 2 ^ (-150 : ℤ)

/-- The data of one row: `N` values `x` read with a block of `2^D` lanes, and the real `eps`. -/
structure RowP where
  /-- The row length. -/
  N : ℕ
  /-- `BLOCK_SIZE`. -/
  B : ℕ
  /-- `log₂ BLOCK_SIZE`. -/
  D : ℕ
  /-- The row's values. -/
  x : ℕ → ℝ
  /-- The host's `eps`. -/
  eps : ℝ
  n_ge : 128 ≤ N
  n_le : N ≤ 32768
  block_eq : B = 2 ^ D
  n_le_block : N ≤ B
  block_lt : B < 2 * N
  x_le : ∀ j, j < N → |x j| ≤ 65504
  eps_ge : (2 : ℝ) ^ (-125 : ℤ) ≤ eps
  eps_le : eps ≤ (2 : ℝ) ^ (126 : ℤ)

/-- `d` is a computed value of `x - m` (a fusable subtraction of the exact `x` and `m`). -/
def LaneErr (x m d : ℝ) : Prop :=
  |d - (x - m)| ≤ u32 * |x - m| + u32 * (1 + u32) * (|x| + |m|) + 3 * eta32

/-- `(1 + u)^D ≤ 2` for `D ≤ 15`. -/
theorem one_add_u_pow_le {D : ℕ} (hD : D ≤ 15) : (1 + u32) ^ D ≤ 2 := by
  unfold u32
  calc (1 + (2 : ℝ) ^ (-24 : ℤ)) ^ D ≤ (1 + 2 ^ (-24 : ℤ)) ^ 15 :=
        pow_le_pow_right₀ (by norm_num) hD
    _ ≤ 2 := by norm_num

namespace RowP

variable (r : RowP)

/-- The exact mean. -/
noncomputable def mu : ℝ := (∑ j ∈ Finset.range r.N, r.x j) / r.N

/-- The mean magnitude. -/
noncomputable def aMean : ℝ := (∑ j ∈ Finset.range r.N, |r.x j|) / r.N

/-- The exact variance. -/
noncomputable def sig2 : ℝ := (∑ j ∈ Finset.range r.N, (r.x j - r.mu) ^ 2) / r.N

/-- `1 / sqrt(var + eps)`. -/
noncomputable def R : ℝ := 1 / Real.sqrt (r.sig2 + r.eps)

/-- The mean's tolerance. -/
noncomputable def Delta : ℝ := ((r.D : ℝ) + 4) * 2 ^ (-23 : ℤ) * r.aMean + 2 ^ (-140 : ℤ)

/-- The condition number. -/
noncomputable def cond : ℝ := r.Delta * r.R

/-- The relative error of `rstd`. -/
noncomputable def gam : ℝ := ((r.D : ℝ) + 8) * (2 ^ (-23 : ℤ) + 2 ^ (-145 : ℤ) / r.eps)

/-- The mean's effect on `rstd`. -/
noncomputable def mTerm : ℝ := r.cond / 8 + r.cond ^ 2 / 4

/-- The tolerance of `rstd`. -/
noncomputable def rstdTol : ℝ :=
  if r.cond ≤ 1 / 2 then
    ((r.D : ℝ) * 2 ^ (-21 : ℤ) + 2 ^ (-140 : ℤ) / r.eps + r.cond / 4) * r.R + 2 ^ (-140 : ℤ)
  else 2 / Real.sqrt r.eps

/-- The exact standardised value. -/
noncomputable def zRef (j : ℕ) : ℝ := (r.x j - r.mu) * r.R

/-- The tolerance of the standardised value. -/
noncomputable def zTol (j : ℕ) : ℝ :=
  if r.cond ≤ 1 / 2 then
    (1 + 2 * r.gam) * (|r.zRef j| * (r.gam + r.mTerm) + r.cond) + 2 ^ (-140 : ℤ) * (1 + r.R)
  else
    |r.zRef j| + (1 + 2 * r.gam) * (|r.x j - r.mu| + r.Delta) / Real.sqrt r.eps +
      2 ^ (-140 : ℤ) * (1 + 1 / Real.sqrt r.eps)

/-- The first block's lane `j`: `x j` on the row, `0` on the padding. -/
noncomputable def xl (j : ℕ) : ℝ := if j < r.N then r.x j else 0

/-- The first block's lanes are at most `65504` in magnitude. -/
theorem xl_le (j : ℕ) : |r.xl j| ≤ 65504 := by
  unfold xl
  split_ifs with h
  · exact r.x_le j h
  · simp

/-- The mean is at most `65504` in magnitude. -/
theorem mu_le : |r.mu| ≤ 65504 := by
  unfold mu
  have hN : (0 : ℝ) < r.N := by exact_mod_cast (by have := r.n_ge; omega : 0 < r.N)
  rw [abs_div, abs_of_pos hN, div_le_iff₀ hN]
  calc |∑ j ∈ Finset.range r.N, r.x j| ≤ ∑ j ∈ Finset.range r.N, |r.x j| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ Finset.range r.N, (65504 : ℝ) :=
        Finset.sum_le_sum fun j hj => r.x_le j (Finset.mem_range.mp hj)
    _ = 65504 * r.N := by simp; ring

/-- The mean magnitude is at most `65504`. -/
theorem aMean_le : r.aMean ≤ 65504 := by
  unfold aMean
  have hN : (0 : ℝ) < r.N := by exact_mod_cast (by have := r.n_ge; omega : 0 < r.N)
  rw [div_le_iff₀ hN]
  calc ∑ j ∈ Finset.range r.N, |r.x j| ≤ ∑ j ∈ Finset.range r.N, (65504 : ℝ) :=
        Finset.sum_le_sum fun j hj => r.x_le j (Finset.mem_range.mp hj)
    _ = 65504 * r.N := by simp; ring

/-- The mean magnitude is nonnegative. -/
theorem aMean_nonneg : 0 ≤ r.aMean := by
  unfold aMean
  exact div_nonneg (Finset.sum_nonneg fun j _ => abs_nonneg _) (Nat.cast_nonneg _)

/-- `D ≤ 15`. -/
theorem D_le : r.D ≤ 15 := by
  have h1 : 2 ^ r.D < 2 * 32768 := by
    have := r.block_lt; have := r.n_le; rw [← r.block_eq]; omega
  have h2 : r.D < 16 := by
    by_contra h
    have : 2 ^ 16 ≤ 2 ^ r.D := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  omega

/-- `7 ≤ D`. -/
theorem D_ge : 7 ≤ r.D := by
  by_contra h
  have h1 : 2 ^ r.D ≤ 2 ^ 6 := Nat.pow_le_pow_right (by norm_num) (by omega)
  have := r.n_le_block; have := r.n_ge; rw [r.block_eq] at *; omega

/-- The mean's tolerance is nonnegative. -/
theorem Delta_nonneg : 0 ≤ r.Delta := by
  unfold Delta
  have := r.aMean_nonneg
  positivity

/-- The mean's tolerance is at most `1`. -/
theorem Delta_le : r.Delta ≤ 1 := by
  unfold Delta
  have h1 := r.aMean_le
  have h2 : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have h3 := r.aMean_nonneg
  have : ((r.D : ℝ) + 4) * 2 ^ (-23 : ℤ) * r.aMean ≤ (15 + 4) * 2 ^ (-23 : ℤ) * 65504 := by
    gcongr
  have h4 : (19 : ℝ) * 2 ^ (-23 : ℤ) * 65504 + 2 ^ (-140 : ℤ) ≤ 1 := by norm_num
  linarith

/-- The constant part of a lane's error when it subtracts the mean `m` (`LaneErr`). -/
noncomputable def cLane (m : ℝ) : ℝ :=
  u32 * (2 + u32) * |m - r.mu| + 2 * u32 * (1 + u32) * |r.mu| + 3 * eta32

/-- The bound on the variance's relative perturbation by the lanes' errors (`cond ≤ 1/2`). -/
noncomputable def nuH : ℝ := 11 / 2 * u32 + r.cond / 5 + 43 / 100 * r.cond ^ 2

/-- `1 − loS ≤ s · R^2`: the lower side of the input of `tl.sqrt` (`cond ≤ 1/2`). -/
noncomputable def loS : ℝ :=
  ((r.D : ℝ) + 12) * u32 * (1 + r.nuH) + 41 / 10 * u32 + 22 * eta32 / r.eps + 4001 / 1000 * u32 +
    r.cond / 5

/-- `s · R^2 ≤ 1 + hiS`: the upper side of the input of `tl.sqrt` (`cond ≤ 1/2`). -/
noncomputable def hiS : ℝ :=
  ((r.D : ℝ) + 12) * u32 * (1 + r.nuH) + 41 / 10 * u32 + 22 * eta32 / r.eps + r.nuH

/-- The relative error of `rstd` (`cond ≤ 1/2`). -/
noncomputable def thetaS : ℝ :=
  ((r.D : ℝ) / 2 + 20) * u32 + r.cond / 9 + 26 / 100 * r.cond ^ 2 + 12 * eta32 / r.eps

/-- The variance is nonnegative. -/
theorem sig2_nonneg : 0 ≤ r.sig2 := by
  unfold sig2
  exact div_nonneg (Finset.sum_nonneg fun j _ => sq_nonneg _) (Nat.cast_nonneg _)

/-- `eps` is positive. -/
theorem eps_pos : 0 < r.eps := lt_of_lt_of_le (by positivity) r.eps_ge

/-- `R = 1/√(sig2 + eps)` is positive. -/
theorem R_pos : 0 < r.R := by
  unfold R
  have := r.sig2_nonneg
  have := r.eps_pos
  positivity

/-- `cond` is nonnegative. -/
theorem cond_nonneg : 0 ≤ r.cond := mul_nonneg r.Delta_nonneg r.R_pos.le

/-- `gam` is between `30u` and `1/100`. -/
theorem gam_bounds : 30 * u32 ≤ r.gam ∧ r.gam ≤ 1 / 100 := by
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
  have hx : 0 ≤ 2 ^ (-145 : ℤ) / r.eps := by have := r.eps_pos; positivity
  have hx2 : 2 ^ (-145 : ℤ) / r.eps ≤ 2 ^ (-145 : ℤ) / 2 ^ (-125 : ℤ) :=
    div_le_div_of_nonneg_left (by positivity) (by positivity) r.eps_ge
  have e1 : (2 : ℝ) ^ (-23 : ℤ) = 2 * u32 := by unfold u32; norm_num
  have h3 : (2 : ℝ) ^ (-145 : ℤ) / 2 ^ (-125 : ℤ) = 2 ^ (-20 : ℤ) := by norm_num
  unfold gam
  rw [e1]
  have hu : (0 : ℝ) ≤ u32 := by unfold u32; positivity
  have hu2 : u32 ≤ 1 / 16000000 := by unfold u32; norm_num
  constructor
  · nlinarith
  · have : (r.D + 8 : ℝ) * (2 * u32 + 2 ^ (-145 : ℤ) / r.eps) ≤ 23 * (2 * u32 + 2 ^ (-20 : ℤ)) := by
      rw [← h3] at *
      gcongr
      linarith
    have h4 : 23 * (2 * u32 + 2 ^ (-20 : ℤ)) ≤ 1 / 100 := by unfold u32; norm_num
    linarith

/-- A lane's error splits as `(x − mu) − (m − mu)` plus a term `ξ` with `|ξ| ≤ κ|x − mu| + cLane m`
(`κ = u(2+u)`), see `LaneErr`. -/
theorem lane_split {x m d' : ℝ} (hd' : LaneErr x m d') :
    |d' - ((x - r.mu) - (m - r.mu))| ≤ u32 * (2 + u32) * |x - r.mu| + r.cLane m := by
  unfold LaneErr at hd'
  have h1 : |x - m| ≤ |x - r.mu| + |m - r.mu| := by
    have := abs_sub_le x r.mu m
    rw [abs_sub_comm r.mu m] at this
    exact this
  have h2 : |x| ≤ |x - r.mu| + |r.mu| := by
    have := abs_add_le (x - r.mu) r.mu
    simpa using this
  have h3 : |m| ≤ |r.mu| + |m - r.mu| := by
    have := abs_add_le r.mu (m - r.mu)
    simpa using this
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have e : (x - r.mu) - (m - r.mu) = x - m := by ring
  rw [e]
  unfold cLane
  have a1 := mul_le_mul_of_nonneg_left h1 hu
  have a2 : u32 * (1 + u32) * (|x| + |m|) ≤
      u32 * (1 + u32) * (|x - r.mu| + |m - r.mu| + 2 * |r.mu|) := by
    have : 0 ≤ u32 * (1 + u32) := by positivity
    exact mul_le_mul_of_nonneg_left (by linarith) this
  nlinarith

/-- `Θ = thetaS ≤ 1/8` on a well-conditioned row. -/
theorem thetaS_le (hc : r.cond ≤ 1 / 2) : r.thetaS ≤ 1 / 8 := by
  unfold thetaS
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hc0 := r.cond_nonneg
  have he : 12 * eta32 / r.eps ≤ 12 * eta32 / 2 ^ (-125 : ℤ) :=
    div_le_div_of_nonneg_left (by unfold eta32; positivity) (by positivity) r.eps_ge
  have h1 : ((r.D : ℝ) / 2 + 20) * u32 ≤ (15 / 2 + 20) * u32 := by unfold u32; gcongr
  have h2 : (15 / 2 + 20 : ℝ) * u32 + 12 * eta32 / 2 ^ (-125 : ℤ) ≤ 1 / 1000 := by
    unfold u32 eta32; norm_num
  nlinarith [mul_nonneg hc0 (by linarith : 0 ≤ 1 / 2 - r.cond)]

/-- `Θ + 3.4u ≤ gam + mTerm` on a well-conditioned row. -/
theorem thetaS_add_le (hc : r.cond ≤ 1 / 2) : r.thetaS + 17 / 5 * u32 ≤ r.gam + r.mTerm := by
  unfold thetaS gam mTerm
  have hD : (r.D : ℝ) ≤ 15 := by exact_mod_cast r.D_le
  have hD7 : (7 : ℝ) ≤ r.D := by exact_mod_cast r.D_ge
  have hc0 := r.cond_nonneg
  have hx : 0 ≤ 1 / r.eps := by
    have : 0 < r.eps := lt_of_lt_of_le (by positivity) r.eps_ge
    positivity
  have e1 : (2 : ℝ) ^ (-23 : ℤ) = 2 * u32 := by unfold u32; norm_num
  have e2 : (2 : ℝ) ^ (-145 : ℤ) = 32 * eta32 := by unfold eta32; norm_num
  have e3 : 2 ^ (-145 : ℤ) / r.eps = 32 * eta32 * (1 / r.eps) := by rw [e2]; ring
  have e4 : 12 * eta32 / r.eps = 12 * eta32 * (1 / r.eps) := by ring
  rw [e1, e3, e4]
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have hη : 0 ≤ eta32 := by unfold eta32; positivity
  have a1 : 0 ≤ ((r.D : ℝ) - 7) * u32 := mul_nonneg (by linarith) hu
  have a2 : 0 ≤ ((r.D : ℝ) - 7) * (eta32 * (1 / r.eps)) := mul_nonneg (by linarith) (mul_nonneg hη hx)
  have a3 : 0 ≤ eta32 * (1 / r.eps) := mul_nonneg hη hx
  nlinarith

end RowP

end Claims.LayerNormFwdF16
