import Claim
import TVA.Proof
import Proof.Util

/-!
# An entry of the matrix `A` in the reals

Entry `(p, j)` of `b_A` is `σ̂` times the `K` loop's accumulated dot product `Ah`, times the gate
`wv` (when there is one), rounded to `float32` (`Pv`), and then to `bfloat16` (`xv`). With the
errors of the stages as hypotheses, `entry_err` bounds `xv` against the exact entry `σ·S·W`, in
the terms of the claim's tolerance: `2^-8·|σ S W|` for the final rounding and `κ·aMag` for the
rest, `aMag = σ·(Mg + 2^-94)·(W + 2^-112) + 2^-86`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The `K` loop's relative constant at the largest trip count, `(520 + 5 + 64)·2^-24`. -/
noncomputable def theta0 : ℝ := 589 / 2 ^ 24

/-- The gate's absolute error constant, `(1 + 2^-16)·2^-126`. -/
noncomputable def gam : ℝ := (1 + 1 / 2 ^ 16) / 2 ^ 126

/-- The entry's magnitude, `aMag` of the claim, in the reals. -/
noncomputable def aM (σ Mg W : ℝ) : ℝ := σ * (Mg + 1 / 2 ^ 94) * (W + 1 / 2 ^ 112) + 1 / 2 ^ 86

/-- The share of `2^-13·aMag` the entry's own error takes: `κ = 9·2^-17`. -/
noncomputable def kap : ℝ := 9 / 2 ^ 17

/-- `aMag` is nonnegative. -/
theorem aM_nonneg {σ Mg W : ℝ} (hσ : 0 ≤ σ) (hMg : 0 ≤ Mg) (hW : 0 ≤ W) : 0 ≤ aM σ Mg W := by
  unfold aM; positivity

/-- `aMag` is at least its absolute floor `2^-86`. -/
theorem aM_ge_floor {σ Mg W : ℝ} (hσ : 0 ≤ σ) (hMg : 0 ≤ Mg) (hW : 0 ≤ W) :
    1 / 2 ^ 86 ≤ aM σ Mg W := by
  unfold aM
  have : 0 ≤ σ * (Mg + 1 / 2 ^ 94) * (W + 1 / 2 ^ 112) := by positivity
  linarith

/-- `σ·Mg·W ≤ aMag`. -/
theorem sigma_mag_le_aM {σ Mg W : ℝ} (hσ : 0 ≤ σ) (hMg : 0 ≤ Mg) (hW : 0 ≤ W) :
    σ * Mg * W ≤ aM σ Mg W := by
  unfold aM
  have : 0 ≤ σ * (1 / 2 ^ 94) * (W + 1 / 2 ^ 112) := by positivity
  have h2 : 0 ≤ σ * Mg * (1 / 2 ^ 112) := by positivity
  nlinarith

set_option maxHeartbeats 1600000 in
/-- **An entry of `A` against the exact entry.** With `Ah` the `K` loop's accumulator (within
`σ·(θ₀·Mg + 2^-110)` of `σ·S`), `wv` the gate (within `Ew` of `W`, `Ew ≤ W·2^-16 + γ`), `Pv` the
`float32` product and `xv` its `bfloat16` rounding, `xv` is within `2^-8·|σ S W| + κ·aMag` of
`σ S W`. -/
theorem entry_err {σ S Mg W Ew Ah wv Pv xv : ℝ} (hσ1 : 1 / 2 ^ 32 ≤ σ) (hσ2 : σ ≤ 2 ^ 32)
    (hMg0 : 0 ≤ Mg) (hS : |S| ≤ Mg) (hW0 : 0 < W) (hEw0 : 0 ≤ Ew)
    (hEw : Ew ≤ W * (1 / 2 ^ 16) + gam)
    (hAh : |Ah - σ * S| ≤ σ * (theta0 * Mg + 1 / 2 ^ 110)) (hwv : |wv - W| ≤ Ew)
    (hP : |Pv - Ah * wv| ≤ 1 / 2 ^ 24 * |Ah * wv| + 1 / 2 ^ 150)
    (hx : |xv - Pv| ≤ 1 / 2 ^ 8 * |Pv| + 1 / 2 ^ 133) :
    |xv - σ * S * W| ≤ 1 / 2 ^ 8 * |σ * S * W| + kap * aM σ Mg W ∧
      |Pv - σ * S * W| ≤ kap * aM σ Mg W := by
  have hσ0 : 0 < σ := lt_of_lt_of_le (by positivity) hσ1
  have hX0 : 0 ≤ σ * Mg := by positivity
  have hθ : 0 ≤ theta0 := by unfold theta0; positivity
  have hg0 : 0 ≤ gam := by unfold gam; positivity
  -- the product
  have hwv' : |wv| ≤ W + Ew := by
    have := abs_sub_abs_le_abs_sub wv W
    rw [abs_of_pos hW0] at this
    linarith
  have hAbound : |σ * S * W| ≤ σ * Mg * W := by
    rw [abs_mul, abs_mul, abs_of_pos hσ0, abs_of_pos hW0]
    have := mul_le_mul_of_nonneg_left hS hσ0.le
    exact mul_le_mul_of_nonneg_right this hW0.le
  have hB1 : |Ah * wv - σ * S * W| ≤
      σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) + σ * Mg * Ew := by
    have e : Ah * wv - σ * S * W = (Ah - σ * S) * wv + σ * S * (wv - W) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul]
    have h1 : |Ah - σ * S| * |wv| ≤ σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) :=
      mul_le_mul hAh hwv' (abs_nonneg _) (by positivity)
    have h2 : |σ * S| * |wv - W| ≤ σ * Mg * Ew := by
      refine mul_le_mul ?_ hwv (abs_nonneg _) hX0
      rw [abs_mul, abs_of_pos hσ0]
      exact mul_le_mul_of_nonneg_left hS hσ0.le
    linarith
  -- expand the product bound with the gate's error
  set ε : ℝ := 1 / 2 ^ 16 with hε
  have hWEw : W + Ew ≤ W * (1 + ε) + gam := by nlinarith
  have hB1' : σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) + σ * Mg * Ew ≤
      σ * Mg * W * (theta0 * (1 + ε) + ε) + σ * Mg * gam * (theta0 + 1) +
        σ * W * (1 / 2 ^ 110 * (1 + ε)) + σ * gam * (1 / 2 ^ 110) := by
    have h1 : σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) ≤
        σ * (theta0 * Mg + 1 / 2 ^ 110) * (W * (1 + ε) + gam) :=
      mul_le_mul_of_nonneg_left hWEw (by positivity)
    have h2 : σ * Mg * Ew ≤ σ * Mg * (W * ε + gam) := mul_le_mul_of_nonneg_left hEw hX0
    calc _ ≤ σ * (theta0 * Mg + 1 / 2 ^ 110) * (W * (1 + ε) + gam) + σ * Mg * (W * ε + gam) :=
          add_le_add h1 h2
      _ = _ := by ring
  -- rounding of the product and of the result
  have hPA : |Pv - σ * S * W| ≤ (1 + 1 / 2 ^ 24) * (σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) + σ * Mg * Ew) +
      1 / 2 ^ 24 * (σ * Mg * W) + 1 / 2 ^ 150 := by
    have h1 : |Ah * wv| ≤ |σ * S * W| + |Ah * wv - σ * S * W| := by
      have := abs_sub_abs_le_abs_sub (Ah * wv) (σ * S * W)
      linarith
    have h2 : |Pv - σ * S * W| ≤ |Pv - Ah * wv| + |Ah * wv - σ * S * W| := by
      have := abs_add_le (Pv - Ah * wv) (Ah * wv - σ * S * W)
      have e : Pv - Ah * wv + (Ah * wv - σ * S * W) = Pv - σ * S * W := by ring
      rw [e] at this; exact this
    have h3 := mul_le_mul_of_nonneg_left (add_le_add hAbound hB1) (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 24)
    nlinarith
  have hxA : |xv - σ * S * W| ≤ 1 / 2 ^ 8 * |σ * S * W| + (1 + 1 / 2 ^ 8) * |Pv - σ * S * W| + 1 / 2 ^ 133 := by
    have h1 : |Pv| ≤ |σ * S * W| + |Pv - σ * S * W| := by
      have := abs_sub_abs_le_abs_sub Pv (σ * S * W)
      linarith
    have h2 : |xv - σ * S * W| ≤ |xv - Pv| + |Pv - σ * S * W| := by
      have := abs_add_le (xv - Pv) (Pv - σ * S * W)
      have e : xv - Pv + (Pv - σ * S * W) = xv - σ * S * W := by ring
      rw [e] at this; exact this
    have h3 := mul_le_mul_of_nonneg_left h1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 8)
    linarith
  -- the coefficients
  have hc1 : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 24) * (theta0 * (1 + ε) + ε) + 1 / 2 ^ 24) ≤ kap := by
    unfold theta0 kap; rw [hε]; norm_num
  have hc2 : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 24) * (gam * (theta0 + 1))) ≤ kap * (1 / 2 ^ 112) := by
    unfold theta0 kap gam; norm_num
  have hc3 : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 24) * (1 / 2 ^ 110 * (1 + ε))) ≤ kap * (1 / 2 ^ 94) := by
    unfold kap; rw [hε]; norm_num
  have hc4 : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 24) * (2 ^ 32 * gam * (1 / 2 ^ 110)) + 1 / 2 ^ 150) +
      1 / 2 ^ 133 ≤ kap * (1 / 2 ^ 86) := by
    unfold kap gam; norm_num
  have hXW0 : 0 ≤ σ * Mg * W := by positivity
  have hXg0 : 0 ≤ σ * Mg := hX0
  have hσW0 : 0 ≤ σ * W := by positivity
  have hσg : σ * gam ≤ 2 ^ 32 * gam := mul_le_mul_of_nonneg_right hσ2 hg0
  have haM : σ * Mg * W + σ * Mg * (1 / 2 ^ 112) + σ * (1 / 2 ^ 94) * W + 1 / 2 ^ 86 ≤ aM σ Mg W := by
    unfold aM
    nlinarith [mul_nonneg hσ0.le (mul_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 94) (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 112))]
  -- combine
  have hm1 := mul_le_mul_of_nonneg_right hc1 hXW0
  have hm2 := mul_le_mul_of_nonneg_right hc2 hXg0
  have hm3 := mul_le_mul_of_nonneg_right hc3 hσW0
  have hm4 : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 24) * (σ * gam * (1 / 2 ^ 110)) + 1 / 2 ^ 150) + 1 / 2 ^ 133 ≤
      kap * (1 / 2 ^ 86) := by
    have : σ * gam * (1 / 2 ^ 110) ≤ 2 ^ 32 * gam * (1 / 2 ^ 110) :=
      mul_le_mul_of_nonneg_right hσg (by positivity)
    have h5 : (1 + 1 / 2 ^ 24) * (σ * gam * (1 / 2 ^ 110)) + 1 / 2 ^ 150 ≤
        (1 + 1 / 2 ^ 24) * (2 ^ 32 * gam * (1 / 2 ^ 110)) + 1 / 2 ^ 150 := by
      have := mul_le_mul_of_nonneg_left this (by norm_num : (0 : ℝ) ≤ 1 + 1 / 2 ^ 24)
      linarith
    have h6 := mul_le_mul_of_nonneg_left h5 (by norm_num : (0 : ℝ) ≤ 1 + 1 / 2 ^ 8)
    linarith
  have hkap : 0 ≤ kap := by unfold kap; positivity
  have hfin : (1 + 1 / 2 ^ 8) * ((1 + 1 / 2 ^ 24) * (σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) + σ * Mg * Ew) +
      1 / 2 ^ 24 * (σ * Mg * W) + 1 / 2 ^ 150) + 1 / 2 ^ 133 ≤ kap * aM σ Mg W := by
    have hB1'' := mul_le_mul_of_nonneg_left hB1' (by positivity : (0 : ℝ) ≤ (1 + 1 / 2 ^ 8) * (1 + 1 / 2 ^ 24))
    have hmk := mul_le_mul_of_nonneg_left haM hkap
    nlinarith
  refine ⟨by nlinarith, ?_⟩
  have hnn : 0 ≤ (1 / 2 ^ 8 : ℝ) * ((1 + 1 / 2 ^ 24) * (σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) + σ * Mg * Ew) +
      1 / 2 ^ 24 * (σ * Mg * W) + 1 / 2 ^ 150) := by
    have : 0 ≤ σ * (theta0 * Mg + 1 / 2 ^ 110) * (W + Ew) + σ * Mg * Ew := by positivity
    positivity
  nlinarith

/-- **The magnitude of the rounded entry**: `|xv| + 2^-126 ≤ (1 + 2^-7)·aMag`. -/
theorem entry_mag {σ S Mg W Pv xv : ℝ} (hσ : 0 < σ) (hMg0 : 0 ≤ Mg) (hS : |S| ≤ Mg) (hW0 : 0 < W)
    (hPdev : |Pv - σ * S * W| ≤ kap * aM σ Mg W)
    (hx : |xv - Pv| ≤ 1 / 2 ^ 8 * |Pv| + 1 / 2 ^ 133) :
    |Pv| ≤ 2 * aM σ Mg W ∧ |xv| + 1 / 2 ^ 126 ≤ (1 + 1 / 2 ^ 7) * aM σ Mg W := by
  have hσ0 : 0 ≤ σ := hσ.le
  have haM0 := aM_nonneg hσ0 hMg0 hW0.le
  have hfl := aM_ge_floor hσ0 hMg0 hW0.le
  have hA : |σ * S * W| ≤ aM σ Mg W := by
    have h1 : |σ * S * W| ≤ σ * Mg * W := by
      rw [abs_mul, abs_mul, abs_of_pos hσ, abs_of_pos hW0]
      have := mul_le_mul_of_nonneg_left hS hσ0
      exact mul_le_mul_of_nonneg_right this hW0.le
    exact h1.trans (sigma_mag_le_aM hσ0 hMg0 hW0.le)
  have hkap : kap ≤ 1 / 2 ^ 10 := by unfold kap; norm_num
  have hP : |Pv| ≤ |σ * S * W| + kap * aM σ Mg W := by
    have := abs_sub_abs_le_abs_sub Pv (σ * S * W)
    linarith
  have hk : kap * aM σ Mg W ≤ aM σ Mg W := by nlinarith
  have hP2 : |Pv| ≤ 2 * aM σ Mg W := by linarith
  refine ⟨hP2, ?_⟩
  have hk2 : kap * aM σ Mg W ≤ 1 / 2 ^ 10 * aM σ Mg W := by nlinarith
  have hxx : |xv| ≤ (1 + 1 / 2 ^ 8) * |Pv| + 1 / 2 ^ 133 := by
    have := abs_sub_abs_le_abs_sub xv Pv
    linarith
  have h3 : (1 + 1 / 2 ^ 8) * |Pv| ≤ (1 + 1 / 2 ^ 8) * (aM σ Mg W * (1 + 1 / 2 ^ 10)) := by
    apply mul_le_mul_of_nonneg_left _ (by norm_num)
    linarith
  have h4 : (1 / 2 ^ 133 + 1 / 2 ^ 126 : ℝ) ≤ 1 / 2 ^ 8 * aM σ Mg W := by
    have : (1 / 2 ^ 133 + 1 / 2 ^ 126 : ℝ) ≤ 1 / 2 ^ 8 * (1 / 2 ^ 86) := by norm_num
    nlinarith
  nlinarith

/-- **Room for the product**: `|Ah · wv| ≤ 2^102`. -/
theorem prod_room {σ S Mg W Ew Ah wv : ℝ} (hσ2 : σ ≤ 2 ^ 32) (hσ0 : 0 < σ) (hMg0 : 0 ≤ Mg)
    (hW0 : 0 < W) (hW64 : W ≤ 2 ^ 64) (hEw0 : 0 ≤ Ew) (hEw : Ew ≤ W * (1 / 2 ^ 16) + gam)
    (hAh : |Ah - σ * S| ≤ σ * (theta0 * Mg + 1 / 2 ^ 110)) (hS : |S| ≤ Mg) (hwv : |wv - W| ≤ Ew)
    (hr : σ * Mg * (1 + W + Ew) ≤ 2 ^ 100) : |Ah * wv| ≤ 2 ^ 102 := by
  have hθ : 0 ≤ theta0 := by unfold theta0; positivity
  have hθ1 : theta0 ≤ 1 := by unfold theta0; norm_num
  have hg0 : 0 ≤ gam := by unfold gam; positivity
  have hgam : gam ≤ 1 / 2 ^ 100 := by unfold gam; norm_num
  have hwv' : |wv| ≤ W + Ew := by
    have := abs_sub_abs_le_abs_sub wv W
    rw [abs_of_pos hW0] at this
    linarith
  have hAh' : |Ah| ≤ σ * Mg * (1 + theta0) + σ * (1 / 2 ^ 110) := by
    have h1 := abs_sub_abs_le_abs_sub Ah (σ * S)
    have h2 : |σ * S| ≤ σ * Mg := by
      rw [abs_mul, abs_of_pos hσ0]; exact mul_le_mul_of_nonneg_left hS hσ0.le
    nlinarith
  rw [abs_mul]
  have h1 : |Ah| * |wv| ≤ (σ * Mg * (1 + theta0) + σ * (1 / 2 ^ 110)) * (W + Ew) :=
    mul_le_mul hAh' hwv' (abs_nonneg _) (by positivity)
  have hX0 : 0 ≤ σ * Mg := by positivity
  have h2 : σ * Mg * (W + Ew) ≤ 2 ^ 100 := by nlinarith
  have h3 : σ * (1 / 2 ^ 110) * (W + Ew) ≤ 1 := by
    have hWE : W + Ew ≤ 2 ^ 65 := by
      have : W * (1 / 2 ^ 16) ≤ 2 ^ 64 * (1 / 2 ^ 16) := by gcongr
      norm_num at this hgam ⊢
      linarith
    have := mul_le_mul hσ2 hWE (by positivity) (by positivity : (0 : ℝ) ≤ 2 ^ 32)
    nlinarith
  have h4 : σ * Mg * (1 + theta0) * (W + Ew) ≤ 2 * 2 ^ 100 := by nlinarith
  nlinarith

end Claims.ChunkBwdDvLocal
