import TVA.Lemmas.FloatMax
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Online softmax: the numeric lemmas

A flash-attention style kernel keeps, per row, a running maximum `μ`, a running sum `l` of
`2^(z_j − μ)` and an accumulator `a` of `2^(z_j − μ)·v_j`. When the maximum moves from `μ` to
`μ'` it rescales the old state by `α ≈ 2^(μ − μ')` and adds the new block. These lemmas bound
the device state against the exact sums
`Λ = Σ_j 2^(z_j − μ')` and `A = Σ_j 2^(z_j − μ')·v_j` (all over the reals):

* `two_rpow_sub_le`: `|2^a − 2^b| ≤ δ · 2^b` when `|a − b| ≤ δ ≤ 1/2` (an exponent error is a
  relative error of the same size);
* `rescale_err`: the rescaled old state `l·α` against `2^(μ − μ')·Y`;
* `accum_err`: one block, `l' ≈ l·α + (new block)` against the exact `Y_old' + Y_new`, with
  separate relative errors for the two parts and an absolute rest;
* `relify`: an absolute rest is a relative one when the exact sum is bounded below;
* `ratio_err`: the output `a / l` against `A / Λ`;
* `logb_err`: `log₂ l` against `log₂ Λ`;
* `pow_le`: `(1 + x)^n ≤ 1 + 2·n·x` for `n·x ≤ 1/2` (the depth term of a reduction tree).

`TVA.Lemmas.FloatMax` has the float maximum by value and the facts about `−∞`.
-/
namespace TVA.Softmax

open Real

/-! ## Exponent errors -/

theorem two_rpow_pos (x : ℝ) : 0 < (2 : ℝ) ^ x := Real.rpow_pos_of_pos (by norm_num) x

/-- `2^δ − 1 ≤ δ` for `0 ≤ δ ≤ 1/2`. -/
theorem two_rpow_sub_one_le {δ : ℝ} (h0 : 0 ≤ δ) (h1 : δ ≤ 1 / 2) : (2 : ℝ) ^ δ - 1 ≤ δ := by
  have hL := Real.log_two_lt_d9
  have hL0 := Real.log_two_gt_d9
  rw [Real.rpow_def_of_pos (by norm_num)]
  set y := Real.log 2 * δ with hy
  have hy0 : 0 ≤ y := mul_nonneg (by linarith) h0
  have hy1 : y ≤ 1 := by rw [hy]; nlinarith
  have h := Real.abs_exp_sub_one_sub_id_le (x := y) (by rw [abs_of_nonneg hy0]; exact hy1)
  have h' := (abs_le.mp h).2
  have : y + y ^ 2 ≤ δ := by
    rw [hy]
    have : Real.log 2 * δ + (Real.log 2 * δ) ^ 2 = δ * (Real.log 2 * (1 + Real.log 2 * δ)) := by ring
    rw [this]
    have hk : Real.log 2 * (1 + Real.log 2 * δ) ≤ 1 := by nlinarith
    calc δ * (Real.log 2 * (1 + Real.log 2 * δ)) ≤ δ * 1 := mul_le_mul_of_nonneg_left hk h0
      _ = δ := mul_one δ
  linarith

/-- `1 − 2^(−δ) ≤ δ` for `0 ≤ δ`. -/
theorem one_sub_two_rpow_neg_le {δ : ℝ} (h0 : 0 ≤ δ) : 1 - (2 : ℝ) ^ (-δ) ≤ δ := by
  have hL := Real.log_two_lt_d9
  have hL0 := Real.log_two_gt_d9
  rw [Real.rpow_def_of_pos (by norm_num)]
  have := Real.add_one_le_exp (Real.log 2 * -δ)
  nlinarith

/-- `|2^d − 1| ≤ δ` for `|d| ≤ δ ≤ 1/2`. -/
theorem abs_two_rpow_sub_one_le {d δ : ℝ} (hd : |d| ≤ δ) (hδ : δ ≤ 1 / 2) :
    |(2 : ℝ) ^ d - 1| ≤ δ := by
  rcases le_total 0 d with h | h
  · have h1 : 1 ≤ (2 : ℝ) ^ d := Real.one_le_rpow (by norm_num) h
    rw [abs_of_nonneg (by linarith)]
    have := two_rpow_sub_one_le h ((le_abs_self d).trans hd |>.trans hδ)
    linarith [le_abs_self d]
  · have h1 : (2 : ℝ) ^ d ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) h
    rw [abs_of_nonpos (by linarith)]
    have := one_sub_two_rpow_neg_le (δ := -d) (by linarith)
    rw [neg_neg] at this
    have : -d ≤ δ := by rw [← abs_of_nonpos h]; exact hd
    linarith

/-- **An exponent error is a relative error**: `|2^a − 2^b| ≤ δ·2^b` for `|a − b| ≤ δ ≤ 1/2`. -/
theorem two_rpow_sub_le {a b δ : ℝ} (h : |a - b| ≤ δ) (hδ : δ ≤ 1 / 2) :
    |(2 : ℝ) ^ a - (2 : ℝ) ^ b| ≤ δ * (2 : ℝ) ^ b := by
  have e : (2 : ℝ) ^ a = (2 : ℝ) ^ (a - b) * (2 : ℝ) ^ b := by
    rw [← Real.rpow_add (by norm_num)]; ring_nf
  have hb := two_rpow_pos b
  rw [e, show (2 : ℝ) ^ (a - b) * 2 ^ b - 2 ^ b = ((2 : ℝ) ^ (a - b) - 1) * 2 ^ b by ring, abs_mul,
    abs_of_pos hb]
  exact mul_le_mul_of_nonneg_right (abs_two_rpow_sub_one_le h hδ) hb.le

theorem two_rpow_le_two {x : ℝ} (h : x ≤ 1) : (2 : ℝ) ^ x ≤ 2 := by
  have := Real.rpow_le_rpow_of_exponent_le (x := 2) (by norm_num) h
  rwa [Real.rpow_one] at this

/-- `2^(−δ) ≥ 1/2` for `δ ≤ 1/2`. -/
theorem two_rpow_neg_ge {δ : ℝ} (h0 : 0 ≤ δ) (h1 : δ ≤ 1 / 2) : 1 / 2 ≤ (2 : ℝ) ^ (-δ) := by
  have := one_sub_two_rpow_neg_le h0
  linarith

/-! ## One block of the online softmax -/

/-- **Rescaling the old state.** If `l` is within `ρ·Λ` of `Y` (with `|Y| ≤ V·Λ`) and the
factor `α` within `κ·g + ν` of `g ≥ 0`, then `l·α` is within
`(ρ + κ·(V + ρ))·(g·Λ) + ν·(V + ρ)·Λmax` of `g·Y`. -/
theorem rescale_err {l Y Λ Λmax V ρ α g κ ν : ℝ} (hΛ : 0 ≤ Λ) (hΛm : Λ ≤ Λmax) (hρ : 0 ≤ ρ)
    (hV : 0 ≤ V) (hg : 0 ≤ g) (hν : 0 ≤ ν) (hY : |Y| ≤ V * Λ) (hl : |l - Y| ≤ ρ * Λ)
    (hα : |α - g| ≤ κ * g + ν) :
    |l * α - g * Y| ≤ (ρ + κ * (V + ρ)) * (g * Λ) + ν * (V + ρ) * Λmax := by
  have hl' : |l| ≤ (V + ρ) * Λ := by
    have := abs_sub_abs_le_abs_sub l Y
    nlinarith
  have e : l * α - g * Y = l * (α - g) + g * (l - Y) := by ring
  rw [e]
  have h1 : |l * (α - g)| ≤ (V + ρ) * Λ * (κ * g + ν) := by
    rw [abs_mul]; exact mul_le_mul hl' hα (abs_nonneg _) (by positivity)
  have h2 : |g * (l - Y)| ≤ g * (ρ * Λ) := by
    rw [abs_mul, abs_of_nonneg hg]; exact mul_le_mul_of_nonneg_left hl hg
  have h3 : (V + ρ) * Λ * ν ≤ (V + ρ) * Λmax * ν :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hΛm (by positivity)) hν
  calc |l * (α - g) + g * (l - Y)| ≤ |l * (α - g)| + |g * (l - Y)| := abs_add_le _ _
    _ ≤ (V + ρ) * Λ * (κ * g + ν) + g * (ρ * Λ) := add_le_add h1 h2
    _ ≤ (ρ + κ * (V + ρ)) * (g * Λ) + ν * (V + ρ) * Λmax := by nlinarith

/-- **One block.** The old part `la` is within `ρa·G + ν₁` of its exact value `YG`
(`|YG| ≤ V·G`), the new part `ls` within `κ·S + ν₂` of `YS`, and the combination `l'` within
`θ·(|la| + B) + ν₃` of `la + ls`, where `B ≤ (V + κB)·S + νB` (for a dot product, `B` is the
sum of the absolute products; for a sum, `|ls|`). Then `l'` is within
`(ρa + θ·(V + ρa))·G + (κ + θ·(V + κB))·S + ((1 + θ)·ν₁ + ν₂ + θ·νB + ν₃)` of `YG + YS`. -/
theorem accum_err {V G S YG YS la ls B l' ρa κ κB θ ν1 ν2 νB ν3 : ℝ} (hθ : 0 ≤ θ)
    (hYG : |YG| ≤ V * G) (h1 : |la - YG| ≤ ρa * G + ν1) (h2 : |ls - YS| ≤ κ * S + ν2)
    (hB : B ≤ (V + κB) * S + νB) (h3 : |l' - (la + ls)| ≤ θ * (|la| + B) + ν3) :
    |l' - (YG + YS)| ≤ (ρa + θ * (V + ρa)) * G + (κ + θ * (V + κB)) * S +
      ((1 + θ) * ν1 + ν2 + θ * νB + ν3) := by
  have hla : |la| ≤ (V + ρa) * G + ν1 := by
    have := abs_sub_abs_le_abs_sub la YG
    nlinarith
  have h4 : θ * (|la| + B) ≤ θ * ((V + ρa) * G + ν1 + ((V + κB) * S + νB)) :=
    mul_le_mul_of_nonneg_left (add_le_add hla hB) hθ
  have e : l' - (YG + YS) = (l' - (la + ls)) + (la - YG) + (ls - YS) := by ring
  rw [e]
  calc |(l' - (la + ls)) + (la - YG) + (ls - YS)| ≤ |l' - (la + ls)| + |la - YG| + |ls - YS| :=
        (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
    _ ≤ θ * (|la| + B) + ν3 + (ρa * G + ν1) + (κ * S + ν2) := by linarith
    _ ≤ _ := by nlinarith

/-- **An absolute rest is relative** when the exact sum is at least `lo > 0`:
`a·G + b·S + c ≤ (m + c/lo)·(G + S)` for `a, b ≤ m`, `G, S, c ≥ 0`, `G + S ≥ lo`. -/
theorem relify {e a b c m G S lo : ℝ} (hG : 0 ≤ G) (hS : 0 ≤ S) (hc : 0 ≤ c) (hlo : 0 < lo)
    (hGS : lo ≤ G + S) (ha : a ≤ m) (hb : b ≤ m) (he : e ≤ a * G + b * S + c) :
    e ≤ (m + c / lo) * (G + S) := by
  have h1 : a * G ≤ m * G := mul_le_mul_of_nonneg_right ha hG
  have h2 : b * S ≤ m * S := mul_le_mul_of_nonneg_right hb hS
  have h3 : c ≤ c / lo * (G + S) := by
    rw [div_mul_eq_mul_div, le_div_iff₀ hlo]
    exact mul_le_mul_of_nonneg_left hGS hc
  nlinarith

/-! ## The outputs -/

/-- **The normalised output.** With `|a − A| ≤ ρv·Λ`, `|l − Λ| ≤ ρ·Λ`, `|A| ≤ V·Λ`, `Λ > 0` and
`ρ < 1`: `|a/l − A/Λ| ≤ (ρv + V·ρ)/(1 − ρ)`. -/
theorem ratio_err {a A l Λ ρv ρ V : ℝ} (hΛ : 0 < Λ) (hρ1 : ρ < 1) (ha : |a - A| ≤ ρv * Λ)
    (hl : |l - Λ| ≤ ρ * Λ) (hA : |A| ≤ V * Λ) :
    |a / l - A / Λ| ≤ (ρv + V * ρ) / (1 - ρ) := by
  have hl0 : (1 - ρ) * Λ ≤ l := by
    have := (abs_le.mp hl).1; nlinarith
  have hlp : 0 < l := lt_of_lt_of_le (mul_pos (by linarith) hΛ) hl0
  have e : a / l - A / Λ = ((a - A) * Λ - A * (l - Λ)) / (l * Λ) := by
    field_simp; ring
  rw [e, abs_div, abs_of_pos (mul_pos hlp hΛ)]
  have hnum : |(a - A) * Λ - A * (l - Λ)| ≤ (ρv + V * ρ) * Λ * Λ := by
    calc |(a - A) * Λ - A * (l - Λ)| ≤ |(a - A) * Λ| + |A * (l - Λ)| := abs_sub _ _
      _ = |a - A| * Λ + |A| * |l - Λ| := by rw [abs_mul, abs_mul, abs_of_pos hΛ]
      _ ≤ ρv * Λ * Λ + V * Λ * (ρ * Λ) := by
          apply add_le_add (mul_le_mul_of_nonneg_right ha hΛ.le)
          exact mul_le_mul hA hl (abs_nonneg _) (by nlinarith [abs_nonneg A])
      _ = (ρv + V * ρ) * Λ * Λ := by ring
  have hpos : 0 ≤ ρv + V * ρ := by
    have h1 : 0 ≤ ρv * Λ := (abs_nonneg _).trans ha
    have h2 : 0 ≤ ρ * Λ := (abs_nonneg _).trans hl
    have h3 : 0 ≤ V * Λ := (abs_nonneg _).trans hA
    have : 0 ≤ ρv := nonneg_of_mul_nonneg_left h1 hΛ
    have : 0 ≤ ρ := nonneg_of_mul_nonneg_left h2 hΛ
    have : 0 ≤ V := nonneg_of_mul_nonneg_left h3 hΛ
    positivity
  rw [div_le_div_iff₀ (mul_pos hlp hΛ) (by linarith)]
  calc |(a - A) * Λ - A * (l - Λ)| * (1 - ρ) ≤ (ρv + V * ρ) * Λ * Λ * (1 - ρ) :=
        mul_le_mul_of_nonneg_right hnum (by linarith)
    _ = (ρv + V * ρ) * (((1 - ρ) * Λ) * Λ) := by ring
    _ ≤ (ρv + V * ρ) * (l * Λ) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hl0 hΛ.le) hpos

/-- **The log-sum-exp output.** With `|l − Λ| ≤ ρ·Λ`, `Λ > 0`, `ρ ≤ 1/2`:
`|log₂ l − log₂ Λ| ≤ 2ρ/(1 − ρ)`. -/
theorem logb_err {l Λ ρ : ℝ} (hΛ : 0 < Λ) (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ 1 / 2) (hl : |l - Λ| ≤ ρ * Λ) :
    |Real.logb 2 l - Real.logb 2 Λ| ≤ 2 * ρ / (1 - ρ) := by
  have hl0 : (1 - ρ) * Λ ≤ l := by have := (abs_le.mp hl).1; nlinarith
  have hl1 : l ≤ (1 + ρ) * Λ := by have := (abs_le.mp hl).2; nlinarith
  have hlp : 0 < l := lt_of_lt_of_le (mul_pos (by linarith) hΛ) hl0
  have hy : 0 < l / Λ := div_pos hlp hΛ
  have hy0 : 1 - ρ ≤ l / Λ := by rw [le_div_iff₀ hΛ]; exact hl0
  have hy1 : l / Λ ≤ 1 + ρ := by rw [div_le_iff₀ hΛ]; exact hl1
  have e : Real.logb 2 l - Real.logb 2 Λ = Real.log (l / Λ) / Real.log 2 := by
    rw [Real.log_div hlp.ne' hΛ.ne', Real.logb, Real.logb]; ring
  have hL := Real.log_two_gt_d9
  have hL0 : 0 < Real.log 2 := by linarith
  have hup : Real.log (l / Λ) ≤ ρ := by
    have := Real.log_le_sub_one_of_pos hy; linarith
  have hlo : -(ρ / (1 - ρ)) ≤ Real.log (l / Λ) := by
    have h := Real.one_sub_inv_le_log_of_pos hy
    have hinv : (l / Λ)⁻¹ ≤ 1 / (1 - ρ) := by
      rw [inv_eq_one_div]; exact one_div_le_one_div_of_le (by linarith) hy0
    have hne : 1 - ρ ≠ 0 := by linarith
    have : 1 - 1 / (1 - ρ) = -(ρ / (1 - ρ)) := by field_simp; ring
    linarith
  have hρρ : ρ ≤ ρ / (1 - ρ) := by
    rw [le_div_iff₀ (by linarith)]; nlinarith
  have habs : |Real.log (l / Λ)| ≤ ρ / (1 - ρ) := abs_le.mpr ⟨hlo, by linarith⟩
  rw [e, abs_div, abs_of_pos hL0, div_le_iff₀ hL0]
  have : 0 ≤ ρ / (1 - ρ) := div_nonneg hρ0 (by linarith)
  have e2 : 2 * ρ / (1 - ρ) = 2 * (ρ / (1 - ρ)) := by ring
  rw [e2]
  nlinarith

/-! ## Reduction depth -/

/-- `(1 + x)^n · (1 − n·x) ≤ 1` for `0 ≤ x` and `n·x ≤ 1`. -/
theorem pow_mul_le {x : ℚ} (hx : 0 ≤ x) : ∀ n : ℕ, n * x ≤ 1 → (1 + x) ^ n * (1 - n * x) ≤ 1
  | 0, _ => by simp
  | n + 1, h => by
    have hn : (n : ℚ) * x ≤ 1 := by push_cast at h; nlinarith
    have ih := pow_mul_le hx n hn
    have hp : 0 ≤ (1 + x) ^ n := by positivity
    have e : (1 + x) * (1 - ((n + 1 : ℕ) : ℚ) * x) ≤ 1 - n * x := by
      have h2 : (1 + x) * (1 - ((n + 1 : ℕ) : ℚ) * x) = 1 - n * x - ((n : ℚ) + 1) * (x * x) := by
        push_cast; ring
      rw [h2]
      have : 0 ≤ ((n : ℚ) + 1) * (x * x) := mul_nonneg (by positivity) (mul_nonneg hx hx)
      linarith
    calc (1 + x) ^ (n + 1) * (1 - ((n + 1 : ℕ) : ℚ) * x)
        = (1 + x) ^ n * ((1 + x) * (1 - ((n + 1 : ℕ) : ℚ) * x)) := by ring
      _ ≤ (1 + x) ^ n * (1 - n * x) := mul_le_mul_of_nonneg_left e hp
      _ ≤ 1 := ih

/-- `(1 + x)^n ≤ 1 + 2·n·x` for `0 ≤ x` and `n·x ≤ 1/2`. -/
theorem pow_le {x : ℚ} (hx : 0 ≤ x) (n : ℕ) (h : n * x ≤ 1 / 2) : (1 + x) ^ n ≤ 1 + 2 * n * x := by
  have h1 := pow_mul_le hx n (by linarith)
  have hpos : 0 < 1 - n * x := by linarith
  have h2 : 1 ≤ (1 + 2 * n * x) * (1 - n * x) := by nlinarith [mul_nonneg (Nat.cast_nonneg n) hx]
  exact le_of_mul_le_mul_right (h1.trans h2) hpos

end TVA.Softmax
