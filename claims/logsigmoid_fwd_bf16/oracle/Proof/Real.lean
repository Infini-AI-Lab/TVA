import Claim
import Mathlib

/-!
# The arithmetic of one element, in the reals

The kernel computes, for an input `v` (`a = |v|`, `E = e^-a`, `ℓ = log(1 + E)`):
`e ≈ E` (`tl.exp`), `z ≈ 1 + e` (one rounding), `L ≈ log z` (`tl.log`), `s ≈ m - L` with
`m = min(0, v)` (one rounding), `q ≈ s / t₃₂` (`div.full.f32`, `t₃₂` the `float32` temperature),
`c ≈ q` (the rounding to `bfloat16`). The lemmas here take each step's error bound as a hypothesis
and give:

* `z_bounds`: `1/2 ≤ z ≤ 3` (so `tl.log` is in its domain);
* `L_err`: `|L - ℓ| ≤ 2^-20·ℓ + α₀`, `α₀ = 2^-24·113/64` (the absolute part is the rounding of
  `1 + e`, about `2^-24`, and `2^-23·a·E ≤ 2^-23·3/8`);
* `s_err`: `|s - ls| ≤ (2^-20 + 2^-23)·|ls| + αs`, `ls = m - ℓ = lsRef v`, `αs = 2^-24·57/32`;
* `div_room`, `q_room`: the quotient and its rounding stay in range;
* `final`: `|c - ls/t| ≤ yTol v t`.
-/

namespace Claims.LogsigmoidFwd

open Real

/-- The difference of two logarithms: at most the difference of the arguments over a lower bound
of both. -/
theorem abs_log_sub_le {p q r : ℝ} (hr : 0 < r) (hp : r ≤ p) (hq : r ≤ q) :
    |Real.log p - Real.log q| ≤ |p - q| / r := by
  have hp0 : 0 < p := lt_of_lt_of_le hr hp
  have hq0 : 0 < q := lt_of_lt_of_le hr hq
  have key : ∀ {x y : ℝ}, 0 < x → 0 < y → r ≤ x →
      Real.log y - Real.log x ≤ |x - y| / r := by
    intro x y hx hy hrx
    have h := Real.log_le_sub_one_of_pos (div_pos hy hx)
    rw [Real.log_div hy.ne' hx.ne'] at h
    have h1 : y / x - 1 = (y - x) / x := by field_simp
    have h2 : (y - x) / x ≤ |x - y| / r := by
      rw [abs_sub_comm]
      calc (y - x) / x ≤ |y - x| / x := by gcongr; exact le_abs_self _
        _ ≤ |y - x| / r := by gcongr
    linarith
  rw [abs_le]
  constructor
  · have := key hp0 hq0 hp
    linarith
  · have := key hq0 hp0 hq
    rw [abs_sub_comm] at this
    linarith

/-- The exact log-sigmoid as the kernel writes it: `min(0, v) - log(1 + e^-|v|)`. -/
theorem lsRef_eq (v : ℝ) : lsRef v = min 0 v - Real.log (1 + Real.exp (-|v|)) := by
  unfold lsRef
  rcases le_total 0 v with h | h
  · rw [abs_of_nonneg h, min_eq_left h]
    ring
  · rw [abs_of_nonpos h, min_eq_right h, neg_neg]
    have h1 : 1 + Real.exp (-v) = Real.exp (-v) * (1 + Real.exp v) := by
      rw [mul_add, mul_one, ← Real.exp_add]
      simp only [neg_add_cancel, Real.exp_zero]
      ring
    rw [h1, Real.log_mul (Real.exp_pos _).ne' (by positivity), Real.log_exp]
    ring

/-- `a·e^-a ≤ 3/8` (its maximum is `1/e`). -/
theorem mul_exp_neg_le {a : ℝ} (ha : 0 ≤ a) : a * Real.exp (-a) ≤ 3 / 8 := by
  have h1 : a ≤ Real.exp (a - 1) := by
    have := Real.add_one_le_exp (a - 1)
    linarith
  have h2 : Real.exp a = Real.exp (a - 1) * Real.exp 1 := by
    rw [← Real.exp_add]; ring_nf
  have he : (8 : ℝ) / 3 < Real.exp 1 := by
    have := Real.exp_one_gt_d9
    linarith
  have h3 : 8 / 3 * a ≤ Real.exp a := by
    rw [h2]
    have h0 : 0 ≤ Real.exp (a - 1) := (Real.exp_pos _).le
    nlinarith
  have hpos : 0 < Real.exp a := Real.exp_pos _
  have h4 : a * Real.exp (-a) = a / Real.exp a := by
    rw [Real.exp_neg, div_eq_mul_inv]
  rw [h4, div_le_iff₀ hpos]
  linarith

/-- `E ≤ 2·log(1 + E)` for `0 ≤ E ≤ 1`. -/
theorem le_two_log {E : ℝ} (h0 : 0 ≤ E) (h1 : E ≤ 1) : E ≤ 2 * Real.log (1 + E) := by
  have h := Real.one_sub_inv_le_log_of_pos (show (0 : ℝ) < 1 + E by linarith)
  have h2 : E / 2 ≤ 1 - (1 + E)⁻¹ := by
    have hp : (0 : ℝ) < 1 + E := by linarith
    rw [show 1 - (1 + E)⁻¹ = E / (1 + E) by field_simp; ring]
    rw [div_le_div_iff₀ (by norm_num) hp]
    nlinarith
  linarith

/-- The absolute part of `L_err`, `2^-24·113/64`. -/
noncomputable def alpha0 : ℝ := 2 ^ (-24 : ℤ) * (113 / 64)

/-- The absolute part of `s_err`, `2^-24·57/32`. -/
noncomputable def alphaS : ℝ := 2 ^ (-24 : ℤ) * (57 / 32)

/-- The exponential's result, from `tl.exp`'s bound at `X = a`. -/
theorem e_bounds {a e : ℝ} (ha : 0 ≤ a)
    (he : |e - Real.exp (-a)| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * a) * Real.exp (-a) + 2 ^ (-140 : ℤ)) :
    |e - Real.exp (-a)| ≤ 2 ^ (-22 : ℤ) * Real.exp (-a) + (3 * 2 ^ (-26 : ℤ) + 2 ^ (-140 : ℤ)) ∧
      -2 ^ (-23 : ℤ) ≤ e ∧ e ≤ 1 + 2 ^ (-21 : ℤ) := by
  have hE0 : 0 < Real.exp (-a) := Real.exp_pos _
  have hE1 : Real.exp (-a) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have haE := mul_exp_neg_le ha
  have h1 : |e - Real.exp (-a)| ≤ 2 ^ (-22 : ℤ) * Real.exp (-a) + (3 * 2 ^ (-26 : ℤ) + 2 ^ (-140 : ℤ)) := by
    have : (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * a) * Real.exp (-a) =
        2 ^ (-22 : ℤ) * Real.exp (-a) + 2 ^ (-23 : ℤ) * (a * Real.exp (-a)) := by ring
    rw [this] at he
    have h23 : (2 : ℝ) ^ (-23 : ℤ) * (a * Real.exp (-a)) ≤ 2 ^ (-23 : ℤ) * (3 / 8) :=
      mul_le_mul_of_nonneg_left haE (by positivity)
    norm_num at he h23 ⊢
    linarith
  refine ⟨h1, ?_, ?_⟩
  · have := (abs_le.mp h1).1
    norm_num at this ⊢
    nlinarith
  · have := (abs_le.mp h1).2
    norm_num at this ⊢
    nlinarith

/-- The rounded `1 + e` lies in `[1/2, 3]`. -/
theorem z_bounds {e z : ℝ} (he1 : -2 ^ (-23 : ℤ) ≤ e) (he2 : e ≤ 1 + 2 ^ (-21 : ℤ))
    (hz : |z - (1 + e)| ≤ |1 + e| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) :
    1 / 2 ≤ z ∧ z ≤ 3 := by
  have hp : 0 ≤ 1 + e := by norm_num at he1 ⊢; linarith
  rw [abs_of_nonneg hp] at hz
  have h := abs_le.mp hz
  norm_num at he1 he2 h ⊢
  constructor <;> nlinarith

/-- **The logarithm's error**: `|L - ℓ| ≤ 2^-20·ℓ + α₀`, `ℓ = log(1 + e^-a)`. -/
theorem L_err {a e z L : ℝ} (ha : 0 ≤ a)
    (he : |e - Real.exp (-a)| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * a) * Real.exp (-a) + 2 ^ (-140 : ℤ))
    (hz : |z - (1 + e)| ≤ |1 + e| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ))
    (hL : |L - Real.log z| ≤ 2 ^ (-22 : ℤ) * |Real.log z| + 2 ^ (-149 : ℤ)) :
    |L - Real.log (1 + Real.exp (-a))| ≤ 2 ^ (-20 : ℤ) * Real.log (1 + Real.exp (-a)) + alpha0 := by
  set E := Real.exp (-a) with hEdef
  set ℓ := Real.log (1 + E) with hℓdef
  have hE0 : 0 < E := Real.exp_pos _
  have hE1 : E ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hℓ0 : 0 ≤ ℓ := Real.log_nonneg (by linarith)
  have hE2 : E ≤ 2 * ℓ := le_two_log hE0.le hE1
  obtain ⟨he', he1, he2⟩ := e_bounds ha he
  have hp : 0 ≤ 1 + e := by norm_num at he1 ⊢; linarith
  have hz' := hz
  rw [abs_of_nonneg hp] at hz'
  -- `log z` against `log (1 + e)`
  set r1 : ℝ := (1 + e) * (1 - 2 ^ (-24 : ℤ)) - 2 ^ (-150 : ℤ) with hr1
  have hr1pos : 0 < r1 := by rw [hr1]; norm_num at he1 ⊢; nlinarith
  have hzr : r1 ≤ z := by have := (abs_le.mp hz').1; rw [hr1]; linarith
  have hr1e : r1 ≤ 1 + e := by rw [hr1]; norm_num; nlinarith
  have hD1 : |Real.log z - Real.log (1 + e)| ≤ 2 ^ (-24 : ℤ) * (1 + 2 ^ (-20 : ℤ)) := by
    refine (abs_log_sub_le hr1pos hzr hr1e).trans ?_
    rw [div_le_iff₀ hr1pos]
    refine hz'.trans ?_
    rw [hr1]
    norm_num at he1 ⊢
    nlinarith
  -- `log (1 + e)` against `ℓ`
  have hr2 : (0 : ℝ) < 1 - 2 ^ (-23 : ℤ) := by norm_num
  have hD2 : |Real.log (1 + e) - ℓ| ≤
      (2 ^ (-22 : ℤ) * E + (3 * 2 ^ (-26 : ℤ) + 2 ^ (-140 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) := by
    refine (abs_log_sub_le hr2 (by linarith) (by norm_num; linarith)).trans ?_
    rw [div_le_iff₀ hr2]
    have h0 : 0 ≤ 2 ^ (-22 : ℤ) * E + (3 * 2 ^ (-26 : ℤ) + 2 ^ (-140 : ℤ)) := by positivity
    have : |1 + e - (1 + E)| = |e - E| := by ring_nf
    rw [this]
    refine he'.trans ?_
    norm_num
    nlinarith
  have hD : |Real.log z - ℓ| ≤ 2 ^ (-24 : ℤ) * (1 + 2 ^ (-20 : ℤ)) +
      (2 ^ (-22 : ℤ) * E + (3 * 2 ^ (-26 : ℤ) + 2 ^ (-140 : ℤ))) * (1 + 2 ^ (-22 : ℤ)) := by
    have := abs_sub_le (Real.log z) (Real.log (1 + e)) ℓ
    linarith
  have hlz : |Real.log z| ≤ ℓ + |Real.log z - ℓ| := by
    have := abs_sub_abs_le_abs_sub (Real.log z) ℓ
    rw [abs_of_nonneg hℓ0] at this
    linarith
  have hfin : |L - ℓ| ≤ |L - Real.log z| + |Real.log z - ℓ| := abs_sub_le _ _ _
  unfold alpha0
  norm_num at hD hlz hfin hL hE2 ⊢
  nlinarith

/-- **The difference `s`**: `|s - ls| ≤ (2^-20 + 2^-23)·|ls| + αs`, with `ls = m - ℓ`, `m = min(0, v)`
and `ℓ = log(1 + e^-|v|)`. -/
theorem s_err {v L s : ℝ}
    (hL : |L - Real.log (1 + Real.exp (-|v|))| ≤
      2 ^ (-20 : ℤ) * Real.log (1 + Real.exp (-|v|)) + alpha0)
    (hs : |s - (min 0 v - L)| ≤ |min 0 v - L| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) :
    |s - lsRef v| ≤ (2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ)) * |lsRef v| + alphaS ∧
      Real.log (1 + Real.exp (-|v|)) ≤ |lsRef v| ∧ |lsRef v| ≤ |v| + 7 / 10 := by
  set ℓ := Real.log (1 + Real.exp (-|v|)) with hℓdef
  have hE0 : 0 < Real.exp (-|v|) := Real.exp_pos _
  have hE1 : Real.exp (-|v|) ≤ 1 := Real.exp_le_one_iff.mpr (by have := abs_nonneg v; linarith)
  have hℓ0 : 0 ≤ ℓ := Real.log_nonneg (by linarith)
  have hℓ1 : ℓ ≤ 7 / 10 := by
    have h2 : ℓ ≤ Real.log 2 := Real.log_le_log (by linarith) (by linarith)
    have := Real.log_two_lt_d9
    linarith
  have hm0 : min 0 v ≤ 0 := min_le_left _ _
  have hmv : -|v| ≤ min 0 v := by
    rcases le_total 0 v with h | h
    · rw [min_eq_left h]; have := abs_nonneg v; linarith
    · rw [min_eq_right h, abs_of_nonpos h]; linarith
  have hls : lsRef v = min 0 v - ℓ := lsRef_eq v
  have habs : |lsRef v| = ℓ - min 0 v := by
    rw [hls, abs_of_nonpos (by linarith)]; ring
  have hmL : |min 0 v - L| ≤ |lsRef v| + |L - ℓ| := by
    have h1 : min 0 v - L = (min 0 v - ℓ) - (L - ℓ) := by ring
    rw [h1, ← hls]
    exact abs_sub _ _
  have htri : |s - lsRef v| ≤ |s - (min 0 v - L)| + |L - ℓ| := by
    have h1 : s - lsRef v = (s - (min 0 v - L)) - (L - ℓ) := by rw [hls]; ring
    rw [h1]
    exact abs_sub _ _
  refine ⟨?_, by rw [habs]; linarith, by rw [habs]; linarith⟩
  have hlsℓ : ℓ ≤ |lsRef v| := by rw [habs]; linarith
  unfold alpha0 at hL
  unfold alphaS
  norm_num at hL hs hmL htri hlsℓ ⊢
  nlinarith [abs_nonneg (lsRef v), abs_nonneg (L - ℓ)]

/-- The `float32` temperature `t₃₂` is within `2^-23·|t|` of `t`: it is nonzero, of the same
sign, and `1/|t₃₂| ≤ (1 + 2^-22)/|t|`. -/
theorem t32_bounds {t t32 : ℝ} (ht : 0 < |t|) (h : |t32 - t| ≤ 2 ^ (-23 : ℤ) * |t|) :
    |t| * (1 - 2 ^ (-23 : ℤ)) ≤ |t32| ∧ 0 < |t32| := by
  have := abs_sub_abs_le_abs_sub t t32
  rw [abs_sub_comm t t32] at this
  have h1 : |t| * (1 - 2 ^ (-23 : ℤ)) ≤ |t32| := by nlinarith
  refine ⟨h1, lt_of_lt_of_le ?_ h1⟩
  have : (0 : ℝ) < 1 - 2 ^ (-23 : ℤ) := by norm_num
  positivity

/-- **The quotient's room**: `|s / t₃₂| ≤ 2^126` when `|v| + 1 ≤ 2^125·|t|`. -/
theorem div_room {v s t t32 : ℝ} (ht : 0 < |t|) (h32 : |t32 - t| ≤ 2 ^ (-23 : ℤ) * |t|)
    (hS : |s - lsRef v| ≤ (2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ)) * |lsRef v| + alphaS)
    (hls : |lsRef v| ≤ |v| + 7 / 10) (hroom : |v| + 1 ≤ 2 ^ 125 * |t|) :
    |s / t32| ≤ 2 ^ (126 : ℤ) := by
  obtain ⟨hlo, hpos⟩ := t32_bounds ht h32
  have hs : |s| ≤ (1 + 2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ)) * (|v| + 7 / 10) + alphaS := by
    have := abs_sub_abs_le_abs_sub s (lsRef v)
    have h1 := mul_le_mul_of_nonneg_left hls
      (show (0 : ℝ) ≤ 1 + 2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ) by positivity)
    unfold alphaS at hS ⊢
    norm_num at h1 hS this ⊢
    linarith
  rw [abs_div, div_le_iff₀ hpos]
  unfold alphaS at hs
  have hv := abs_nonneg v
  norm_num at hs hlo hroom ⊢
  nlinarith

/-- **The rounding to `bfloat16` has room**: `|q| ≤ 2^127`. -/
theorem q_room {w q : ℝ} (hw : |w| ≤ 2 ^ (126 : ℤ))
    (hq : |q - w| ≤ 2 ^ (-22 : ℤ) * |w| + 2 ^ (-148 : ℤ)) : |q| ≤ 2 ^ 127 := by
  have := abs_sub_abs_le_abs_sub q w
  norm_num at hw hq this ⊢
  nlinarith

/-- **The element's error**: from the errors of the difference `s`, the temperature's rounding,
the division and the rounding to `bfloat16`, `|c - lsRef v / t| ≤ yTol v t`. -/
theorem final {v s t t32 q c : ℝ} (ht : 0 < |t|) (h32 : |t32 - t| ≤ 2 ^ (-23 : ℤ) * |t|)
    (hS : |s - lsRef v| ≤ (2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ)) * |lsRef v| + alphaS)
    (hq : |q - s / t32| ≤ 2 ^ (-22 : ℤ) * |s / t32| + 2 ^ (-148 : ℤ))
    (hc : |c - q| ≤ |q| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) :
    |c - lsRef v / t| ≤ yTol v t := by
  obtain ⟨hlo, hpos⟩ := t32_bounds ht h32
  set ls := lsRef v
  set R := |ls / t| with hR
  set A := 1 / |t| with hA
  have hRe : R = |ls| * A := by rw [hR, hA, abs_div]; ring
  have hA0 : 0 < A := by rw [hA]; positivity
  -- `w = s / t₃₂` against `ls / t`
  have hinv : 1 / |t32| ≤ A * (1 + 2 ^ (-22 : ℤ)) := by
    rw [hA, div_le_iff₀ hpos]
    have h1 : 1 / |t| * (1 + 2 ^ (-22 : ℤ)) * |t32| ≥
        1 / |t| * (1 + 2 ^ (-22 : ℤ)) * (|t| * (1 - 2 ^ (-23 : ℤ))) :=
      mul_le_mul_of_nonneg_left hlo (by positivity)
    have h2 : 1 / |t| * (1 + 2 ^ (-22 : ℤ)) * (|t| * (1 - 2 ^ (-23 : ℤ))) =
        (1 + 2 ^ (-22 : ℤ)) * (1 - 2 ^ (-23 : ℤ)) := by field_simp
    have h3 : (1 : ℝ) ≤ (1 + 2 ^ (-22 : ℤ)) * (1 - 2 ^ (-23 : ℤ)) := by norm_num
    linarith
  have hw : |s / t32 - ls / t| ≤
      ((2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ) + 2 ^ (-23 : ℤ)) * |ls| + alphaS) *
        (A * (1 + 2 ^ (-22 : ℤ))) := by
    have ht0 : t ≠ 0 := abs_pos.mp ht
    have ht320 : t32 ≠ 0 := abs_pos.mp hpos
    have hsplit : s / t32 - ls / t = (s - ls) / t32 + ls * (t - t32) / (t32 * t) := by
      field_simp; ring
    rw [hsplit]
    refine (abs_add_le _ _).trans ?_
    rw [abs_div, abs_div, abs_mul, abs_mul]
    have e1 : |s - ls| / |t32| ≤ ((2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ)) * |ls| + alphaS) *
        (A * (1 + 2 ^ (-22 : ℤ))) := by
      rw [div_eq_mul_one_div]
      exact mul_le_mul hS hinv (by positivity) (by unfold alphaS; positivity)
    have e2 : |ls| * |t - t32| / (|t32| * |t|) ≤ 2 ^ (-23 : ℤ) * |ls| * (A * (1 + 2 ^ (-22 : ℤ))) := by
      have h1 : |t - t32| ≤ 2 ^ (-23 : ℤ) * |t| := by rw [abs_sub_comm]; exact h32
      have h2 : |ls| * |t - t32| / (|t32| * |t|) ≤ |ls| * (2 ^ (-23 : ℤ) * |t|) / (|t32| * |t|) := by
        gcongr
      have h3 : |ls| * (2 ^ (-23 : ℤ) * |t|) / (|t32| * |t|) = 2 ^ (-23 : ℤ) * |ls| * (1 / |t32|) := by
        field_simp
      rw [h3] at h2
      exact h2.trans (mul_le_mul_of_nonneg_left hinv (by positivity))
    have : ((2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ) + 2 ^ (-23 : ℤ)) * |ls| + alphaS) * (A * (1 + 2 ^ (-22 : ℤ))) =
        ((2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ)) * |ls| + alphaS) * (A * (1 + 2 ^ (-22 : ℤ))) +
          2 ^ (-23 : ℤ) * |ls| * (A * (1 + 2 ^ (-22 : ℤ))) := by ring
    rw [this]
    exact add_le_add e1 e2
  -- the division's error, then the rounding
  have hwabs : |s / t32| ≤ R + |s / t32 - ls / t| := by
    have := abs_sub_abs_le_abs_sub (s / t32) (ls / t)
    rw [hR]; linarith
  have hqr : |q - ls / t| ≤ |q - s / t32| + |s / t32 - ls / t| := abs_sub_le _ _ _
  have hqabs : |q| ≤ R + |q - ls / t| := by
    have := abs_sub_abs_le_abs_sub q (ls / t)
    rw [hR]; linarith
  have hcr : |c - ls / t| ≤ |c - q| + |q - ls / t| := abs_sub_le _ _ _
  unfold yTol
  rw [← hR, show 2 ^ (-23 : ℤ) / |t| = 2 ^ (-23 : ℤ) * A by rw [hA]; ring]
  have hlsA : |ls| * A = R := hRe.symm
  unfold alphaS at hw
  have hlsA0 : 0 ≤ |ls| * A := by positivity
  -- expand the products of `|ls|` and `A` into `R`
  have hw' : |s / t32 - ls / t| ≤
      (2 ^ (-20 : ℤ) + 2 ^ (-23 : ℤ) + 2 ^ (-23 : ℤ)) * (1 + 2 ^ (-22 : ℤ)) * R +
        2 ^ (-24 : ℤ) * (57 / 32) * (1 + 2 ^ (-22 : ℤ)) * A := by
    rw [← hlsA]; nlinarith
  norm_num at hw' hq hc hwabs hqr hqabs hcr ⊢
  nlinarith [abs_nonneg (s / t32), abs_nonneg q, abs_nonneg (q - ls / t),
    abs_nonneg (s / t32 - ls / t)]

end Claims.LogsigmoidFwd
