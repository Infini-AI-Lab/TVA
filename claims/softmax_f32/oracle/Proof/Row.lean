import Proof.Lane

/-!
# One row

The row sum along a tree, the division and the final per-element inequality: `row_bound` gives the
row's factor `sc` within `sumTol` of `1`, and each output within the claim's tolerance of
`sc · softmaxRef`.
-/
namespace Claims.Softmax

open TVA

/-! ## Row-level estimates: sums, casts and the final per-lane inequality -/

/-- A list sum over `range n` is a `Finset` sum. -/
lemma list_range_sum_eq {M : Type*} [AddCommMonoid M] (f : Nat → M) (n : Nat) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ m ih =>
      rw [List.range_succ, List.map_append, List.sum_append, Finset.sum_range_succ]
      simp [ih]

/-- The cast of a list sum of rationals to the reals. -/
lemma list_sum_cast (l : List ℚ) :
    ((l.sum : ℚ) : ℝ) = (l.map (fun q => (q : ℝ))).sum := by
  induction l with
  | nil => simp
  | cons a t ih => simp [ih]

/-- The cast of the tree error bound to the reals. -/
lemma errBound_cast (d N : Nat) (Aq : ℚ) :
    ((Tree.errBound Format.binary32.u Format.binary32.eta d N Aq : ℚ) : ℝ)
      = ((1 + uR) ^ d - 1) * (Aq : ℝ) + etaR * (N : ℝ) * (1 + uR) ^ d := by
  unfold Tree.errBound
  push_cast
  rw [b32_u_cast, b32_eta_cast]

/-- A numeric fact about `uR`. -/
lemma pow_fourteen_le : ((1 : ℝ) + uR) ^ 14 ≤ 1 + 15 * uR := by
  unfold uR; norm_num

/-- A numeric fact about `uR`. -/
lemma one_add_uR_nonneg : (0 : ℝ) ≤ 1 + uR := by have := uR_pos; linarith

/-- A numeric fact about `uR`. -/
lemma pow_fourteen_le_two : ((1 : ℝ) + uR) ^ 14 ≤ 2 := by
  have h := pow_fourteen_le
  have h' : uR ≤ 1 / 128 := by unfold uR; norm_num
  linarith

/-- A numeric fact about the constant. -/
lemma uR_le_inv128 : uR ≤ 1 / 128 := by unfold uR; norm_num

/-- A numeric fact about the constant. -/
lemma uR_small_110 : 110 * uR ≤ 1 / 2 := by unfold uR; norm_num

/-- A numeric fact about the constant. -/
lemma uR_sq_small : 15620 * uR ^ 2 ≤ uR := by unfold uR; norm_num

/-- A numeric fact about `uR`. -/
lemma e15_eq : (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num

/-- A block of at most `2^14` lanes has a tree of depth at most `14`. -/
lemma clog_le_fourteen {B : Nat} (hB : B ≤ 2 ^ 14) : Nat.clog 2 B ≤ 14 :=
  Nat.clog_le_of_le_pow hB

/-- `|t| · e^t` is bounded for `t ≤ 0` by a multiple of `ER`. -/
lemma mul_exp_neg_le_ER {t : ℝ} (ht : t ≤ 0) :
    |t| * Real.exp t ≤ 14 * Real.exp t + (2 : ℝ) ^ (-15 : ℤ) := by
  have habs : |t| = -t := abs_of_nonpos ht
  rw [habs]
  have h := mul_exp_neg_le (w := -t) (by linarith)
  rwa [neg_neg] at h

/-- `|tR| · ER` is bounded for a lane at or below the max. -/
lemma abs_t_mul_ER_le {mv x : F32} (hxle : x.toRat ≤ mv.toRat) :
    |tR mv x| * ER mv x ≤ 14 * ER mv x + (2 : ℝ) ^ (-15 : ℤ) := by
  by_cases h : x = FloatV.negInf
  · rw [h, ER_eq_zero rfl, mul_zero]
    have : (0 : ℝ) < (2 : ℝ) ^ (-15 : ℤ) := by norm_num
    linarith
  · rw [ER_eq_exp h]
    have ht : tR mv x ≤ 0 := by
      have h1 : (x.toRat : ℝ) ≤ (mv.toRat : ℝ) := by exact_mod_cast hxle
      simp only [tR, mR]
      linarith
    exact mul_exp_neg_le_ER ht

/-- A numeric fact about the constant. -/
lemma uR_sq_16000 : 16000 * uR ^ 2 ≤ uR := by unfold uR; norm_num

/-- A numeric fact about the constant. -/
lemma uR_hundred : 110 * uR ≤ 1 / 4 := by unfold uR; norm_num

/-- Pointwise: a lane's rounded `exp` is at most `2·ER + 2^-33`. -/
lemma lane_abs_le {mv x num : F32}
    (hnumle : |num.toReal - ER mv x| ≤
      (4 * uR * |tR mv x| + 11 * uR) * ER mv x + (2 : ℝ) ^ (-137 : ℤ))
    (htE : |tR mv x| * ER mv x ≤ 14 * ER mv x + (2 : ℝ) ^ (-15 : ℤ)) :
    |num.toReal| ≤ 2 * ER mv x + (2 : ℝ) ^ (-33 : ℤ) := by
  have hE0 : 0 ≤ ER mv x := ER_nonneg mv x
  have hu := uR_pos
  have h1 : |num.toReal| ≤ ER mv x + ((4 * uR * |tR mv x| + 11 * uR) * ER mv x
      + (2 : ℝ) ^ (-137 : ℤ)) := by
    have h := abs_add_le (num.toReal - ER mv x) (ER mv x)
    rw [show num.toReal - ER mv x + ER mv x = num.toReal by ring,
      abs_of_nonneg hE0] at h
    linarith
  have h2 : (4 * uR * |tR mv x| + 11 * uR) * ER mv x ≤
      4 * uR * (14 * ER mv x + (2 : ℝ) ^ (-15 : ℤ)) + 11 * uR * ER mv x := by
    have h4 : 4 * uR * (|tR mv x| * ER mv x) ≤ 4 * uR * (14 * ER mv x + (2 : ℝ) ^ (-15 : ℤ)) :=
      mul_le_mul_of_nonneg_left htE (by positivity)
    nlinarith
  have h3 : 4 * uR * (14 * ER mv x + (2 : ℝ) ^ (-15 : ℤ)) + 11 * uR * ER mv x
      + (2 : ℝ) ^ (-137 : ℤ) ≤ 1 * ER mv x + (2 : ℝ) ^ (-33 : ℤ) := by
    have hu1 : 67 * uR ≤ 1 := by unfold uR; norm_num
    have hu3 : 4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ) ≤ (2 : ℝ) ^ (-33 : ℤ) := by
      unfold uR; norm_num
    nlinarith [hE0]
  linarith

/-- From `|D - A| ≤ 110·u·A` and `A ≥ 1`, the ratio `A/D` is within `sumTol`. -/
lemma sc_close {D A : ℝ} (hA1 : 1 ≤ A) (hD : |D - A| ≤ 110 * uR * A) :
    |A / D - 1| ≤ (2 : ℝ) ^ (-17 : ℤ) := by
  have hu := uR_pos
  have h110 := uR_hundred
  have hDlo : (1 - 110 * uR) * A ≤ D := by
    have h := (abs_le.mp hD).1
    nlinarith
  have hDpos : 0 < D := by nlinarith [hA1]
  have hAle : A ≤ (1 + 220 * uR) * D := by
    have h1 : (1 + 220 * uR) * ((1 - 110 * uR) * A) ≤ (1 + 220 * uR) * D :=
      mul_le_mul_of_nonneg_left hDlo (by nlinarith)
    have h3 : (1 : ℝ) ≤ (1 + 220 * uR) * (1 - 110 * uR) := by
      have h4 : 24200 * uR ^ 2 ≤ uR := by unfold uR; norm_num
      nlinarith
    have h2 : A ≤ (1 + 220 * uR) * ((1 - 110 * uR) * A) := by
      have h5 : (0 : ℝ) ≤ A := by linarith
      nlinarith [mul_le_mul_of_nonneg_right h3 h5]
    linarith
  have hAD : A / D ≤ 1 + 220 * uR := by
    rw [div_le_iff₀ hDpos]
    linarith
  have h1 : |A / D - 1| = |A - D| / D := by
    have h0 : A / D - 1 = (A - D) / D := by field_simp
    rw [h0, abs_div, abs_of_pos hDpos]
  have h2 : |A - D| / D ≤ 110 * uR * A / D :=
    div_le_div_of_nonneg_right (by rwa [abs_sub_comm]) hDpos.le
  have h3 : 110 * uR * A / D ≤ 110 * uR * (1 + 220 * uR) := by
    rw [mul_div_assoc]
    exact mul_le_mul_of_nonneg_left hAD (by positivity)
  have h4 : 110 * uR * (1 + 220 * uR) ≤ (2 : ℝ) ^ (-17 : ℤ) := by
    have h5 : 24200 * uR ^ 2 ≤ 18 * uR := by unfold uR; norm_num
    have h6 : (2 : ℝ) ^ (-17 : ℤ) = 128 * uR := e17
    rw [h6]
    nlinarith
  calc |A / D - 1| = |A - D| / D := h1
    _ ≤ 110 * uR * A / D := h2
    _ ≤ 110 * uR * (1 + 220 * uR) := h3
    _ ≤ (2 : ℝ) ^ (-17 : ℤ) := h4

/-- The `-∞` lane: everything vanishes. -/
lemma final_lane_inf {R N D E A σ t : ℝ}
    (hR : |R - N / D| ≤ (2 : ℝ) ^ (-22 : ℤ) * |N / D| + (2 : ℝ) ^ (-148 : ℤ))
    (hN : |N| ≤ (2 : ℝ) ^ (-137 : ℤ))
    (_hE : E = 0) (hσ : σ = 0) (ht : t = 0)
    (hD : 1 / 2 ≤ D) :
    |R - σ * (A / D)| ≤ (2 : ℝ) ^ (-22 : ℤ) * (|t| + 4) * σ + (2 : ℝ) ^ (-133 : ℤ) := by
  have hDpos : 0 < D := by linarith
  have h1 : |N / D| ≤ (2 : ℝ) ^ (-136 : ℤ) := by
    rw [abs_div, abs_of_pos hDpos]
    rw [div_le_iff₀ hDpos]
    calc |N| ≤ (2 : ℝ) ^ (-137 : ℤ) := hN
      _ = (2 : ℝ) ^ (-136 : ℤ) * (1 / 2) := by norm_num
      _ ≤ (2 : ℝ) ^ (-136 : ℤ) * D := mul_le_mul_of_nonneg_left hD (by norm_num)
  have h3 : |R - N / D| ≤ (2 : ℝ) ^ (-22 : ℤ) * (2 : ℝ) ^ (-136 : ℤ) + (2 : ℝ) ^ (-148 : ℤ) := by
    have h4 : (2 : ℝ) ^ (-22 : ℤ) * |N / D| ≤ (2 : ℝ) ^ (-22 : ℤ) * (2 : ℝ) ^ (-136 : ℤ) :=
      mul_le_mul_of_nonneg_left h1 (by norm_num)
    nlinarith [h4]
  have h4 : |R| ≤ (2 : ℝ) ^ (-22 : ℤ) * (2 : ℝ) ^ (-136 : ℤ) + (2 : ℝ) ^ (-148 : ℤ)
      + (2 : ℝ) ^ (-136 : ℤ) := by
    calc |R| = |R - N / D + N / D| := by ring
      _ ≤ |R - N / D| + |N / D| := abs_add_le _ _
      _ ≤ ((2 : ℝ) ^ (-22 : ℤ) * (2 : ℝ) ^ (-136 : ℤ) + (2 : ℝ) ^ (-148 : ℤ))
          + (2 : ℝ) ^ (-136 : ℤ) := add_le_add h3 h1
  have h6 : (2 : ℝ) ^ (-22 : ℤ) * (2 : ℝ) ^ (-136 : ℤ) + (2 : ℝ) ^ (-148 : ℤ)
      + (2 : ℝ) ^ (-136 : ℤ) ≤ (2 : ℝ) ^ (-133 : ℤ) := by norm_num
  have hRfin : |R| ≤ (2 : ℝ) ^ (-133 : ℤ) := le_trans h4 h6
  rw [hσ, ht]
  simpa using hRfin

/-- The numeric core of one lane's final bound. -/
lemma row_final_num {s σ t : ℝ} (hσ : 0 ≤ σ) (_hs0 : 0 ≤ s)
    (hst : s ≤ σ * (1 + 220 * uR))
    (hbig : (2 : ℝ) ^ 14 ≤ |t| → |t| * s ≤ (2 : ℝ) ^ (-99 : ℤ)) :
    (2 : ℝ) ^ (-22 : ℤ) * ((1 + 4 * uR * |t| + 11 * uR) * s + (2 : ℝ) ^ (-136 : ℤ))
        + (2 : ℝ) ^ (-148 : ℤ)
      + ((4 * uR * |t| + 11 * uR) * s + (2 : ℝ) ^ (-136 : ℤ))
      ≤ (2 : ℝ) ^ (-22 : ℤ) * (|t| + 4) * σ + (2 : ℝ) ^ (-133 : ℤ) := by
  have h99 : (2 : ℝ) ^ (-99 : ℤ) = 1 / (2 : ℝ) ^ 99 := by norm_num
  have h136 : (2 : ℝ) ^ (-136 : ℤ) = 1 / (2 : ℝ) ^ 136 := by norm_num
  have h148 : (2 : ℝ) ^ (-148 : ℤ) = 1 / (2 : ℝ) ^ 148 := by norm_num
  have h133 : (2 : ℝ) ^ (-133 : ℤ) = 1 / (2 : ℝ) ^ 133 := by norm_num
  have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / (2 : ℝ) ^ 22 := by norm_num
  have hu : uR = 1 / 16777216 := by unfold uR; norm_num
  simp only [h99, h136, h148, h133, h22, hu] at *
  have ht0 : (0 : ℝ) ≤ |t| := abs_nonneg t
  rcases lt_or_ge ((2 : ℝ) ^ 14) |t| with hb | hs
  · have hbs : |t| * s ≤ 1 / (2 : ℝ) ^ 99 := hbig (le_of_lt hb)
    have hb14 : (2 : ℝ) ^ 14 ≤ |t| := le_of_lt hb
    have hbpos : (0 : ℝ) < |t| := by positivity
    nlinarith [hbs, hb14, hσ, hst, ht0, hbpos,
      mul_le_mul_of_nonneg_left hst (le_of_lt hbpos)]
  · have hst55 : s ≤ σ * (1 + 55 / 4194304) := by
      have h : (1 : ℝ) + 220 * (1 / 16777216) = 1 + 55 / 4194304 := by norm_num
      rwa [h] at hst
    have h1 : |t| * σ ≤ (2 : ℝ) ^ 14 * σ := mul_le_mul_of_nonneg_right hs hσ
    have h2 : |t| * s ≤ |t| * (σ * (1 + 55 / 4194304)) := mul_le_mul_of_nonneg_left hst55 ht0
    nlinarith [h1, h2, hst55, hσ, ht0]

/-- One finite lane: from the division bound, the `exp` bound and the sum bound, the final inequality. -/
lemma final_lane_fin {R N D E A σ t : ℝ}
    (hR : |R - N / D| ≤ (2 : ℝ) ^ (-22 : ℤ) * |N / D| + (2 : ℝ) ^ (-148 : ℤ))
    (hN : |N - E| ≤ (4 * uR * |t| + 11 * uR) * E + (2 : ℝ) ^ (-137 : ℤ))
    (hD : 1 / 2 ≤ D) (hE : 0 ≤ E) (hA1 : 1 ≤ A)
    (hσA : σ = E / A) (hAD : A / D ≤ 1 + 220 * uR)
    (hbig : (2 : ℝ) ^ 14 ≤ |t| → |t| * (E / D) ≤ (2 : ℝ) ^ (-99 : ℤ)) :
    |R - σ * (A / D)| ≤ (2 : ℝ) ^ (-22 : ℤ) * (|t| + 4) * σ + (2 : ℝ) ^ (-133 : ℤ) := by
  have hDpos : 0 < D := by linarith
  have hApos : 0 < A := by linarith
  have hσ0 : 0 ≤ σ := by rw [hσA]; positivity
  have hED0 : 0 ≤ E / D := div_nonneg hE hDpos.le
  have hs : E / D = σ * (A / D) := by rw [hσA]; field_simp
  have hst : E / D ≤ σ * (1 + 220 * uR) := by
    calc E / D = σ * (A / D) := hs
      _ ≤ σ * (1 + 220 * uR) := mul_le_mul_of_nonneg_left hAD hσ0
  have hstep2 : |N / D - E / D| ≤ (4 * uR * |t| + 11 * uR) * (E / D) + (2 : ℝ) ^ (-136 : ℤ) := by
    have h1 : N / D - E / D = (N - E) / D := by field_simp
    rw [h1, abs_div, abs_of_pos hDpos, div_le_iff₀ hDpos]
    have h4 : (4 * uR * |t| + 11 * uR) * (E / D) * D = (4 * uR * |t| + 11 * uR) * E := by
      field_simp
    rw [add_mul, h4]
    have h2 : (2 : ℝ) ^ (-137 : ℤ) ≤ (2 : ℝ) ^ (-136 : ℤ) * D := by
      calc (2 : ℝ) ^ (-137 : ℤ) = (2 : ℝ) ^ (-136 : ℤ) * (1 / 2) := by norm_num
        _ ≤ (2 : ℝ) ^ (-136 : ℤ) * D := mul_le_mul_of_nonneg_left hD (by norm_num)
    linarith
  have hstep1 : |N| ≤ (1 + 4 * uR * |t| + 11 * uR) * E + (2 : ℝ) ^ (-137 : ℤ) := by
    have h3 : |N| ≤ |N - E| + |E| := by
      have := abs_add_le (N - E) E
      rwa [show N - E + E = N by ring] at this
    rw [abs_of_nonneg hE] at h3
    nlinarith [h3, hN]
  have hND : |N / D| ≤ (1 + 4 * uR * |t| + 11 * uR) * (E / D) + (2 : ℝ) ^ (-136 : ℤ) := by
    have h1 := hstep1
    calc |N / D| = |N| / D := by rw [abs_div, abs_of_pos hDpos]
      _ ≤ ((1 + 4 * uR * |t| + 11 * uR) * E + (2 : ℝ) ^ (-137 : ℤ)) / D :=
          div_le_div_of_nonneg_right h1 hDpos.le
      _ ≤ (1 + 4 * uR * |t| + 11 * uR) * (E / D) + (2 : ℝ) ^ (-136 : ℤ) := by
          rw [add_div]
          have h4 : ((1 + 4 * uR * |t| + 11 * uR) * E) / D
              = (1 + 4 * uR * |t| + 11 * uR) * (E / D) := by ring
          have h5 : (2 : ℝ) ^ (-137 : ℤ) / D ≤ (2 : ℝ) ^ (-136 : ℤ) := by
            rw [div_le_iff₀ hDpos]
            calc (2 : ℝ) ^ (-137 : ℤ) = (2 : ℝ) ^ (-136 : ℤ) * (1 / 2) := by norm_num
              _ ≤ (2 : ℝ) ^ (-136 : ℤ) * D := mul_le_mul_of_nonneg_left hD (by norm_num)
          linarith
  have h3 : |R - E / D| ≤ (2 : ℝ) ^ (-22 : ℤ) * ((1 + 4 * uR * |t| + 11 * uR) * (E / D)
        + (2 : ℝ) ^ (-136 : ℤ)) + (2 : ℝ) ^ (-148 : ℤ)
      + ((4 * uR * |t| + 11 * uR) * (E / D) + (2 : ℝ) ^ (-136 : ℤ)) := by
    have h4 : |R - E / D| ≤ |R - N / D| + |N / D - E / D| := by
      have := abs_add_le (R - N / D) (N / D - E / D)
      rwa [show R - N / D + (N / D - E / D) = R - E / D by ring] at this
    have h5 : (2 : ℝ) ^ (-22 : ℤ) * |N / D| ≤
        (2 : ℝ) ^ (-22 : ℤ) * ((1 + 4 * uR * |t| + 11 * uR) * (E / D) + (2 : ℝ) ^ (-136 : ℤ)) :=
      mul_le_mul_of_nonneg_left hND (by norm_num)
    linarith [h4, hR, hstep2, h5]
  have hnum := row_final_num (s := E / D) (σ := σ) (t := t) hσ0 hED0 hst hbig
  rw [← hs]
  exact le_trans h3 hnum

/-- From `|D - A| ≤ 110·u·A` and `A ≥ 1`, the ratio `A/D ≤ 1 + 220·u`. -/
lemma AD_le {D A : ℝ} (hA1 : 1 ≤ A) (hD : |D - A| ≤ 110 * uR * A) :
    A / D ≤ 1 + 220 * uR := by
  have hu := uR_pos
  have h110 : 110 * uR ≤ 1 / 4 := uR_hundred
  have hDlo : (1 - 110 * uR) * A ≤ D := by
    have h := (abs_le.mp hD).1
    nlinarith
  have hDpos : 0 < D := by nlinarith
  have hAle : A ≤ (1 + 220 * uR) * D := by
    have h1 : (1 + 220 * uR) * ((1 - 110 * uR) * A) ≤ (1 + 220 * uR) * D :=
      mul_le_mul_of_nonneg_left hDlo (by nlinarith)
    have h3 : (1 : ℝ) ≤ (1 + 220 * uR) * (1 - 110 * uR) := by
      have h4 : 24200 * uR ^ 2 ≤ uR := by unfold uR; norm_num
      nlinarith
    have h2 : A ≤ (1 + 220 * uR) * ((1 - 110 * uR) * A) := by
      have h5 : (0 : ℝ) ≤ A := by linarith
      nlinarith [mul_le_mul_of_nonneg_right h3 h5]
    linarith
  rw [div_le_iff₀ hDpos]
  linarith

/-- The cast of a mapped list sum. -/
lemma list_cast_sum_map {α : Type} (l : List α) (f : α → ℚ) :
    (((l.map f).sum : ℚ) : ℝ) = (l.map (fun i => ((f i : ℚ) : ℝ))).sum := by
  induction l with
  | nil => simp
  | cons a t ih => simp [ih]

/-- The cast of a mapped list sum of absolute values. -/
lemma list_cast_abs_sum_map {α : Type} (l : List α) (f : α → ℚ) :
    (((l.map (fun i => |f i|)).sum : ℚ) : ℝ) = (l.map (fun i => |((f i : ℚ) : ℝ)|)).sum := by
  induction l with
  | nil => simp
  | cons a t ih => simp [ih]

set_option maxHeartbeats 1000000 in
/-- **The numerics of one row.** From the device facts and the lanes' relations (max tree, rounded difference, `exp`, sum tree, division) there is one factor `sc` within `sumTol` of `1` such that every output is finite and within the claim's tolerance of `sc · softmaxRef`. -/
theorem row_bound {n B : Nat} (hn : 0 < n) (hnB : n ≤ B) (hB : B ≤ 2 ^ 14) (P : Platform)
    (hexp : Device.ExpF32 P) (hninf : Device.ExpF32NegInf P) (hdiv : Device.DivFull P)
    (hsum : Device.SumDepth P)
    {v rsm num q : Tensor [B] F32} {mv den : F32} {xs : Nat → F32}
    (hv : ∀ (i : Nat) (hi : i < B), v.get1 ⟨i, hi⟩ = if _h : i < n then xs i else FloatV.negInf)
    (hmv : ∃ t : Tree, t.Covers B ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < B then v.get1 ⟨j, h⟩ else default) mv)
    (hrsm : ∀ (i : Nat) (hi : i < B),
      FloatV.NaNEq (FloatV.sub (v.get1 ⟨i, hi⟩) mv) (rsm.get1 ⟨i, hi⟩))
    (hnum : ∀ (i : Nat) (hi : i < B),
      P.math .exp .binary32 (rsm.get1 ⟨i, hi⟩) (num.get1 ⟨i, hi⟩))
    (hden : ∃ t : Tree, t.Covers B ∧ t.depth ≤ tl.sumBound P B ∧
      t.Eval (AddRel false) (fun j => if h : j < B then num.get1 ⟨j, h⟩ else default) den)
    (hq : ∀ (i : Nat) (hi : i < B),
      P.div .binary32 (num.get1 ⟨i, hi⟩) den (q.get1 ⟨i, hi⟩))
    (hok : RowOk xs n) :
    ∃ sc : ℝ, |sc - 1| ≤ (sumTol : ℝ) ∧
      ∀ (j : Nat) (_hj : j < n) (hjb : j < B), (q.get1 ⟨j, hjb⟩).isFinite = true ∧
        |(q.get1 ⟨j, hjb⟩).toReal - sc * softmaxRef xs n j| ≤
          (elemTol : ℝ) * (softmaxGap xs n j + 4) * softmaxRef xs n j + (floorTol : ℝ) := by
  obtain ⟨i0, hi0n, hi0B, hmv_eq, hmv_fin, hmv_nan, hxsok, hxsle⟩ :=
    max_facts hn hnB hv hmv hok
  have hmv_ne : mv ≠ FloatV.negInf := finite_ne_negInf hmv_fin
  have hmv_low : -((2 : ℚ) ^ 128) ≤ mv.toRat := by
    have h := FloatV.abs_toRat_le_maxVal b32_reg (x := mv) hmv_fin
    rw [abs_le] at h
    have h2 := maxVal_lt_two_pow
    linarith [h.1]
  set xv : Nat → F32 := fun i => if h : i < B then v.get1 ⟨i, h⟩ else default with hxv
  have hxv_ap : ∀ i, xv i = (if h : i < B then v.get1 ⟨i, h⟩ else default) := fun i => by
    rw [hxv]
  have hxv_ap' : ∀ (i : Nat) (hi : i < B), xv i = v.get1 ⟨i, hi⟩ := by
    intro i hi
    rw [hxv_ap i]
    exact dite_eq_left hi
  have hxrow : ∀ (i : Nat) (hi : i < B),
      xv i = (if h : i < n then xs i else FloatV.negInf) := by
    intro i hi
    rw [hxv_ap' i hi, hv i hi]
  have hxle_all : ∀ (i : Nat) (hi : i < B), (xv i).toRat ≤ mv.toRat := by
    intro i hi
    rw [hxrow i hi]
    by_cases hin : i < n
    · rw [dite_eq_left hin]
      exact hxsle i hin
    · rw [dite_eq_right hin, negInf_toRat]
      exact hmv_low
  have hxok_all : ∀ (i : Nat) (hi : i < B),
      (xv i).isFinite = true ∨ xv i = FloatV.negInf := by
    intro i hi
    rw [hxrow i hi]
    by_cases hin : i < n
    · rw [dite_eq_left hin]
      exact hxsok i hin
    · rw [dite_eq_right hin]
      exact Or.inr rfl
  set nv : Nat → F32 := fun i => if h : i < B then num.get1 ⟨i, h⟩ else default with hnv
  have hnv_ap : ∀ i, nv i = (if h : i < B then num.get1 ⟨i, h⟩ else default) := fun i => by
    rw [hnv]
  have hnv_ap' : ∀ (i : Nat) (hi : i < B), nv i = num.get1 ⟨i, hi⟩ := by
    intro i hi
    rw [hnv_ap i]
    exact dite_eq_left hi
  set qv : Nat → F32 := fun i => if h : i < B then q.get1 ⟨i, h⟩ else default with hqv
  have hqv_ap : ∀ i, qv i = (if h : i < B then q.get1 ⟨i, h⟩ else default) := fun i => by
    rw [hqv]
  have hqv_ap' : ∀ (i : Nat) (hi : i < B), qv i = q.get1 ⟨i, hi⟩ := by
    intro i hi
    rw [hqv_ap i]
    exact dite_eq_left hi
  have hE1 : ∀ (i : Nat) (hi : i < B), ER mv (xv i) ≤ 1 := by
    intro i hi
    by_cases hneg : xv i = FloatV.negInf
    · rw [hneg, ER_eq_zero rfl]
      norm_num
    · refine ER_le_one hneg ?_
      rw [mR]
      exact_mod_cast hxle_all i hi
  have hlane : ∀ (i : Nat) (hi : i < B),
      (num.get1 ⟨i, hi⟩).isFinite = true ∧
        |(num.get1 ⟨i, hi⟩).toReal - ER mv (xv i)| ≤
          (4 * uR * |tR mv (xv i)| + 11 * uR) * ER mv (xv i) + (2 : ℝ) ^ (-137 : ℤ) := by
    intro i hi
    have hrsm' : FloatV.NaNEq (FloatV.sub (xv i) mv) (rsm.get1 ⟨i, hi⟩) := by
      rw [hxv_ap' i hi]
      exact hrsm i hi
    exact lane_bound hexp hninf hmv_fin (hxle_all i hi) (hxok_all i hi) hrsm' (hnum i hi)
  have hnvlane : ∀ (i : Nat) (hi : i < B),
      |(nv i).toReal - ER mv (xv i)| ≤
        (4 * uR * |tR mv (xv i)| + 11 * uR) * ER mv (xv i) + (2 : ℝ) ^ (-137 : ℤ) := by
    intro i hi
    have h := (hlane i hi).2
    rwa [← hnv_ap' i hi] at h
  have hnvfin : ∀ i, i < B → (nv i).isFinite = true := by
    intro i hi
    rw [hnv_ap' i hi]
    exact (hlane i hi).1
  set A : ℝ := ∑ i ∈ Finset.range B, ER mv (xv i) with hA
  have hA_eq : A = ZR mv xs n := by
    rw [hA]
    unfold ZR
    have h3 : B = n + (B - n) := (Nat.add_sub_of_le hnB).symm
    rw [h3, Finset.sum_range_add]
    have h1 : (∑ i ∈ Finset.range n, ER mv (xv i)) = ∑ i ∈ Finset.range n, ER mv (xs i) := by
      apply Finset.sum_congr rfl
      intro i hi
      have hin : i < n := Finset.mem_range.mp hi
      rw [hxrow i (lt_of_lt_of_le hin hnB), dite_eq_left hin]
    have h2 : (∑ i ∈ Finset.range (B - n), ER mv (xv (n + i))) = 0 := by
      apply Finset.sum_eq_zero
      intro i hi
      have hiB : n + i < B := by
        have := Finset.mem_range.mp hi
        omega
      have hin : ¬ (n + i < n) := by omega
      rw [hxrow (n + i) hiB, dite_eq_right hin, ER_eq_zero rfl]
    rw [h1, h2, add_zero]
  have hmax1 : ER mv (xs i0) = 1 := by
    rw [← hmv_eq]
    exact ER_self hmv_ne
  have hle1 : ∀ k, k < n → ER mv (xs k) ≤ 1 := by
    intro k hk
    by_cases hneg : xs k = FloatV.negInf
    · rw [hneg, ER_eq_zero rfl]
      norm_num
    · refine ER_le_one hneg ?_
      rw [mR]
      exact_mod_cast hxsle k hk
  have hA1 : 1 ≤ A := by
    rw [hA_eq]
    exact one_le_ZR hi0n hmax1
  have hAle : A ≤ (2 : ℝ) ^ 14 := by
    rw [hA_eq]
    have h := ZR_le_n hle1
    have hp14 : ((2 ^ 14 : Nat) : ℝ) = (2 : ℝ) ^ 14 := by norm_num
    have hnle : ((n : Nat) : ℝ) ≤ (2 : ℝ) ^ 14 := by
      have h' : ((n : Nat) : ℝ) ≤ ((2 ^ 14 : Nat) : ℝ) := by
        exact_mod_cast le_trans hnB hB
      linarith
    linarith
  have hZRpos : 0 < ZR mv xs n := by linarith
  have hZRne : ZR mv xs n ≠ 0 := hZRpos.ne'
  have hB14 : (B : ℝ) ≤ (2 : ℝ) ^ 14 := by
    have hp14 : ((2 ^ 14 : Nat) : ℝ) = (2 : ℝ) ^ 14 := by norm_num
    have h' : ((B : Nat) : ℝ) ≤ ((2 ^ 14 : Nat) : ℝ) := by
      exact_mod_cast hB
    linarith
  set S : ℝ := ∑ i ∈ Finset.range B, (nv i).toReal with hS
  set AbsS : ℝ := ∑ i ∈ Finset.range B, |(nv i).toReal| with hAbsS
  have hSA : |S - A| ≤ 70 * uR * A := by
    have hpt : ∀ i ∈ Finset.range B,
        |(nv i).toReal - ER mv (xv i)| ≤ 67 * uR * ER mv (xv i)
          + (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ)) := by
      intro i hi
      have hiB := Finset.mem_range.mp hi
      have h1 := hnvlane i hiB
      have ht := abs_t_mul_ER_le (hxle_all i hiB)
      have h3 : 4 * uR * (|tR mv (xv i)| * ER mv (xv i))
          ≤ 4 * uR * (14 * ER mv (xv i) + (2 : ℝ) ^ (-15 : ℤ)) :=
        mul_le_mul_of_nonneg_left ht (by have := uR_pos; positivity)
      nlinarith [h1, h3, ER_nonneg mv (xv i), uR_pos]
    have hSsub : S - A = ∑ i ∈ Finset.range B, ((nv i).toReal - ER mv (xv i)) := by
      rw [hS, hA, Finset.sum_sub_distrib]
    rw [hSsub]
    have h1 : |∑ i ∈ Finset.range B, ((nv i).toReal - ER mv (xv i))|
        ≤ ∑ i ∈ Finset.range B, |(nv i).toReal - ER mv (xv i)| :=
      Finset.abs_sum_le_sum_abs _ _
    have h2 : (∑ i ∈ Finset.range B, |(nv i).toReal - ER mv (xv i)|)
        ≤ ∑ i ∈ Finset.range B, (67 * uR * ER mv (xv i)
            + (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ))) :=
      Finset.sum_le_sum hpt
    have h3 : (∑ i ∈ Finset.range B, (67 * uR * ER mv (xv i)
            + (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ))))
        = 67 * uR * A
          + (B : ℝ) * (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ)) := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← hA, Finset.sum_const,
        Finset.card_range, nsmul_eq_mul]
    have h4 : (B : ℝ) * (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ))
        ≤ 3 * uR * A := by
      have hc0 : (0 : ℝ) ≤ 4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ) := by
        have hu := uR_pos
        positivity
      have h5 : (B : ℝ) * (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ))
          ≤ (2 : ℝ) ^ 14 * (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ)) :=
        mul_le_mul_of_nonneg_right hB14 hc0
      have h6 : (2 : ℝ) ^ 14 * (4 * uR * (2 : ℝ) ^ (-15 : ℤ) + (2 : ℝ) ^ (-137 : ℤ))
          ≤ 3 * uR := by
        unfold uR
        norm_num
      have h7 : 3 * uR ≤ 3 * uR * A := by nlinarith [uR_pos, hA1]
      linarith
    have h8 : 67 * uR * A + 3 * uR * A ≤ 70 * uR * A := by nlinarith [uR_pos]
    linarith
  have hAbsS0 : (0 : ℝ) ≤ AbsS := by
    rw [hAbsS]
    positivity
  have hAbsS_bound : AbsS ≤ 2 * A + (2 : ℝ) ^ (-19 : ℤ) := by
    have hpt : ∀ i ∈ Finset.range B,
        |(nv i).toReal| ≤ 2 * ER mv (xv i) + (2 : ℝ) ^ (-33 : ℤ) := by
      intro i hi
      have hiB := Finset.mem_range.mp hi
      exact lane_abs_le (hnvlane i hiB) (abs_t_mul_ER_le (hxle_all i hiB))
    rw [hAbsS]
    have h1 : (∑ i ∈ Finset.range B, |(nv i).toReal|)
        ≤ ∑ i ∈ Finset.range B, (2 * ER mv (xv i) + (2 : ℝ) ^ (-33 : ℤ)) :=
      Finset.sum_le_sum hpt
    have h2 : (∑ i ∈ Finset.range B, (2 * ER mv (xv i) + (2 : ℝ) ^ (-33 : ℤ)))
        = 2 * A + (B : ℝ) * (2 : ℝ) ^ (-33 : ℤ) := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← hA, Finset.sum_const,
        Finset.card_range, nsmul_eq_mul]
    have h3 : (B : ℝ) * (2 : ℝ) ^ (-33 : ℤ) ≤ (2 : ℝ) ^ (-19 : ℤ) := by
      have h4 : (B : ℝ) * (2 : ℝ) ^ (-33 : ℤ)
          ≤ (2 : ℝ) ^ 14 * (2 : ℝ) ^ (-33 : ℤ) :=
        mul_le_mul_of_nonneg_right hB14 (by norm_num)
      have h5 : (2 : ℝ) ^ 14 * (2 : ℝ) ^ (-33 : ℤ) = (2 : ℝ) ^ (-19 : ℤ) := by
        norm_num
      linarith
    linarith
  have hAbsS_le : AbsS ≤ (2 : ℝ) ^ 15 + 1 := by
    have h1 : (2 : ℝ) * A ≤ (2 : ℝ) * (2 : ℝ) ^ 14 :=
      mul_le_mul_of_nonneg_left hAle (by norm_num)
    have h2 : (2 : ℝ) ^ (-19 : ℤ) ≤ 1 := by norm_num
    nlinarith [hAbsS_bound, h1, h2]
  have hcast_sum : ((((List.range B).map fun i => (nv i).toRat).sum : ℚ) : ℝ) = S := by
    rw [list_cast_sum_map, list_range_sum_eq, hS]
    apply Finset.sum_congr rfl
    intro i hi
    simp only [FloatV.toReal]
  have hcast_abs : ((((List.range B).map fun i => |(nv i).toRat|).sum : ℚ) : ℝ) = AbsS := by
    rw [list_cast_abs_sum_map, list_range_sum_eq, hAbsS]
    apply Finset.sum_congr rfl
    intro i hi
    simp only [FloatV.toReal]
  obtain ⟨tden, hcov, hdepth, hden_eval⟩ := hden
  have hdepth14 : tden.depth ≤ 14 :=
    le_trans (le_trans hdepth (max_le (hsum B) le_rfl)) (clog_le_fourteen hB)
  have heta_chain : etaR * (((B - 1 : Nat)) : ℝ) * (1 + uR) ^ 14 ≤ etaR * (2 : ℝ) ^ 14 * 2 := by
    have hp0 : (0 : ℝ) ≤ (1 + uR) ^ 14 := by positivity
    have hp2 : (1 + uR) ^ 14 ≤ 2 := pow_fourteen_le_two
    have hb1 : (((B - 1 : Nat)) : ℝ) ≤ (2 : ℝ) ^ 14 := by
      have h' : ((B - 1 : Nat) : ℝ) ≤ ((B : Nat) : ℝ) := by
        exact_mod_cast Nat.sub_le B 1
      linarith
    have he0 : (0 : ℝ) ≤ etaR := etaR_pos.le
    calc etaR * (((B - 1 : Nat)) : ℝ) * (1 + uR) ^ 14
        ≤ etaR * (2 : ℝ) ^ 14 * (1 + uR) ^ 14 := by
          apply mul_le_mul_of_nonneg_right _ hp0
          exact mul_le_mul_of_nonneg_left hb1 he0
      _ ≤ etaR * (2 : ℝ) ^ 14 * 2 := by
          apply mul_le_mul_of_nonneg_left hp2
          exact mul_nonneg he0 (by positivity)
  have hMq : ((List.range B).map fun i => |(nv i).toRat|).sum
      + Tree.errBound Format.binary32.u Format.binary32.eta 14 (B - 1)
          ((List.range B).map fun i => |(nv i).toRat|).sum ≤ Format.binary32.maxVal := by
    have hMreal : ((((List.range B).map fun i => |(nv i).toRat|).sum : ℚ) : ℝ)
        + ((Tree.errBound Format.binary32.u Format.binary32.eta 14 (B - 1)
            ((List.range B).map fun i => |(nv i).toRat|).sum : ℚ) : ℝ)
        ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [errBound_cast, hcast_abs, b32_maxVal_cast]
      have hp0 : (0 : ℝ) ≤ (1 + uR) ^ 14 := by positivity
      have hp2 : (1 + uR) ^ 14 ≤ 2 := pow_fourteen_le_two
      have hprod : AbsS * (1 + uR) ^ 14 ≤ ((2 : ℝ) ^ 15 + 1) * 2 :=
        mul_le_mul hAbsS_le hp2 hp0 (by norm_num)
      have heta1 : etaR * (((B - 1 : Nat)) : ℝ) * (1 + uR) ^ 14 ≤ 1 := by
        have h7 : etaR * (2 : ℝ) ^ 14 * 2 ≤ 1 := by unfold etaR; norm_num
        linarith [heta_chain, h7]
      have hkey : AbsS + ((1 + uR) ^ 14 - 1) * AbsS
          + etaR * (((B - 1 : Nat)) : ℝ) * (1 + uR) ^ 14 ≤ (2 : ℝ) ^ 17 + 1 := by
        have hrw : AbsS + ((1 + uR) ^ 14 - 1) * AbsS = AbsS * (1 + uR) ^ 14 := by ring
        rw [hrw]
        linarith only [hprod, heta1]
      have hrhs : (2 : ℝ) ^ 17 + 1 ≤ ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := by norm_num
      linarith
    exact_mod_cast hMreal
  have hden_ok := sum_err_depth (f := Format.binary32) b32_reg nv hcov hdepth14 hden_eval
    hnvfin hMq
  obtain ⟨hden_fin, hden_err⟩ := hden_ok
  have hden_errR : |den.toReal - S| ≤ ((1 + uR) ^ 14 - 1) * AbsS
      + etaR * (((B - 1 : Nat)) : ℝ) * (1 + uR) ^ 14 := by
    have h1 : (((|den.toRat - ((List.range B).map fun i => (nv i).toRat).sum| : ℚ)) : ℝ)
        ≤ ((Tree.errBound Format.binary32.u Format.binary32.eta 14 (B - 1)
            ((List.range B).map fun i => |(nv i).toRat|).sum : ℚ) : ℝ) := by
      exact_mod_cast hden_err
    have h2 : (((|den.toRat - ((List.range B).map fun i => (nv i).toRat).sum| : ℚ)) : ℝ)
        = |den.toReal - S| := by
      rw [Rat.cast_abs, Rat.cast_sub, hcast_sum]
      simp only [FloatV.toReal]
    have h3 : ((Tree.errBound Format.binary32.u Format.binary32.eta 14 (B - 1)
            ((List.range B).map fun i => |(nv i).toRat|).sum : ℚ) : ℝ)
        = ((1 + uR) ^ 14 - 1) * AbsS
          + etaR * (((B - 1 : Nat)) : ℝ) * (1 + uR) ^ 14 := by
      rw [errBound_cast, hcast_abs]
    rw [h2, h3] at h1
    exact h1
  have hpow15 : ((1 + uR) ^ 14 - 1) ≤ 15 * uR := by
    have h := pow_fourteen_le
    linarith
  have hterm1 : ((1 + uR) ^ 14 - 1) * AbsS ≤ 31 * uR * A := by
    have h1 : ((1 + uR) ^ 14 - 1) * AbsS
        ≤ (15 * uR) * (2 * A + (2 : ℝ) ^ (-19 : ℤ)) :=
      mul_le_mul hpow15 hAbsS_bound hAbsS0 (by have := uR_pos; linarith)
    have h2 : (15 * uR) * (2 * A + (2 : ℝ) ^ (-19 : ℤ)) ≤ 31 * uR * A := by
      have hA0 : (0 : ℝ) ≤ A := by linarith
      have h3 : 15 * uR * (2 : ℝ) ^ (-19 : ℤ) ≤ uR * A := by
        have h4 : 15 * (2 : ℝ) ^ (-19 : ℤ) ≤ 1 := by norm_num
        have h5 : (0 : ℝ) ≤ uR := uR_pos.le
        have h6 : 15 * uR * (2 : ℝ) ^ (-19 : ℤ) = uR * (15 * (2 : ℝ) ^ (-19 : ℤ)) := by
          ring
        rw [h6]
        calc uR * (15 * (2 : ℝ) ^ (-19 : ℤ)) ≤ uR * 1 := mul_le_mul_of_nonneg_left h4 h5
          _ = uR := by ring
          _ ≤ uR * A := by simpa using mul_le_mul_of_nonneg_left hA1 h5
      nlinarith [h3, hA0, uR_pos]
    linarith
  have hterm2 : etaR * (((B - 1 : Nat)) : ℝ) * (1 + uR) ^ 14 ≤ uR * A := by
    have h7 : etaR * (2 : ℝ) ^ 14 * 2 ≤ uR := by unfold etaR uR; norm_num
    have hA0 : (0 : ℝ) ≤ A := by linarith
    nlinarith [heta_chain, h7, uR_pos, hA0, hA1]
  have huA : (0 : ℝ) ≤ uR * A := by nlinarith only [uR_pos, hA1]
  have hdenA : |den.toReal - A| ≤ 110 * uR * A := by
    have h1 : |den.toReal - A| ≤ |den.toReal - S| + |S - A| := by
      have h := abs_add_le (den.toReal - S) (S - A)
      rwa [show den.toReal - S + (S - A) = den.toReal - A by ring] at h
    linarith only [h1, hden_errR, hSA, hterm1, hterm2, huA]
  have hden_half : (1 / 2 : ℝ) ≤ den.toReal := by
    have h1 : A - |den.toReal - A| ≤ den.toReal := by
      have h := neg_le_abs (den.toReal - A)
      linarith only [h]
    have h2 : 110 * uR * A ≤ A / 4 := by
      have h3 : 110 * uR ≤ 1 / 4 := uR_hundred
      have hA0 : (0 : ℝ) ≤ A := by linarith only [hA1]
      nlinarith only [h3, hA0]
    linarith only [h1, hdenA, h2, hA1]
  have hAD : A / den.toReal ≤ 1 + 220 * uR := AD_le hA1 hdenA
  have hsc : |A / den.toReal - 1| ≤ (2 : ℝ) ^ (-17 : ℤ) := sc_close hA1 hdenA
  have hsumTol : (sumTol : ℝ) = (2 : ℝ) ^ (-17 : ℤ) := by
    unfold sumTol
    rw [Rat.cast_zpow]
    norm_num
  have helemTol : (elemTol : ℝ) = (2 : ℝ) ^ (-22 : ℤ) := by
    unfold elemTol
    rw [Rat.cast_zpow]
    norm_num
  have hfloorTol : (floorTol : ℝ) = (2 : ℝ) ^ (-133 : ℤ) := by
    unfold floorTol
    rw [Rat.cast_zpow]
    norm_num
  refine ⟨A / den.toReal, ?_, ?_⟩
  · rw [hsumTol]
    exact hsc
  · intro j hj hjb
    have hxsj : xv j = xs j := by
      rw [hxrow j hjb, dite_eq_left hj]
    have hqdiv : P.div .binary32 (nv j) den (qv j) := by
      rw [hnv_ap' j hjb, hqv_ap' j hjb]
      exact hq j hjb
    have hnvfin_j : (nv j).isFinite = true := hnvfin j hjb
    have hden_ne : den.toRat ≠ 0 := by
      have h1 : (0 : ℝ) < den.toReal := by linarith
      intro hzero
      have h2 : den.toReal = 0 := by
        unfold FloatV.toReal
        rw [hzero]
        norm_num
      linarith
    have hER1 : ER mv (xv j) ≤ 1 := hE1 j hjb
    have hnum_le : |(nv j).toReal| ≤ 3 := by
      have h1 := lane_abs_le (hnvlane j hjb) (abs_t_mul_ER_le (hxle_all j hjb))
      have h2 : (2 : ℝ) * ER mv (xv j) ≤ 2 := by linarith only [hER1]
      have h3 : (2 : ℝ) ^ (-33 : ℤ) ≤ 1 := by norm_num
      linarith only [h1, h2, h3]
    have hnum_q : |(nv j).toRat| ≤ 3 := by
      have h := hnum_le
      have h2 : |(nv j).toReal| = ((|(nv j).toRat| : ℚ) : ℝ) := by
        rw [Rat.cast_abs]
        simp only [FloatV.toReal]
      rw [h2] at h
      have h3 : ((|(nv j).toRat| : ℚ) : ℝ) ≤ ((3 : ℚ) : ℝ) := by simpa using h
      exact Rat.cast_le.mp h3
    have hden_q : (1 / 2 : ℚ) ≤ den.toRat := by
      have h := hden_half
      have h2 : den.toReal = ((den.toRat : ℚ) : ℝ) := by simp only [FloatV.toReal]
      rw [h2] at h
      have h3 : ((1 / 2 : ℚ) : ℝ) ≤ ((den.toRat : ℚ) : ℝ) := by
        have h1 : ((1 / 2 : ℚ) : ℝ) = (1 / 2 : ℝ) := by norm_num
        rw [h1]
        exact h
      exact Rat.cast_le.mp h3
    have hc0 : (0 : ℚ) < den.toRat := by linarith
    have hqmax : |(nv j).toRat / den.toRat| ≤ (2 : ℚ) ^ 126 := by
      rw [abs_div, abs_of_pos hc0, div_le_iff₀ hc0]
      have h1 : (3 : ℚ) ≤ (2 : ℚ) ^ 126 * (1 / 2) := by norm_num
      nlinarith [hnum_q, hden_q, hc0, h1]
    obtain ⟨hqfin_q, hqerr_q⟩ :=
      hdiv (nv j) den (qv j) hqdiv hnvfin_j hden_fin hden_ne hqmax
    have hqfin : (q.get1 ⟨j, hjb⟩).isFinite = true := by
      rw [← hqv_ap' j hjb]
      exact hqfin_q
    have hqerrR : |(q.get1 ⟨j, hjb⟩).toReal - (nv j).toReal / den.toReal|
        ≤ (2 : ℝ) ^ (-22 : ℤ) * |(nv j).toReal / den.toReal| + (2 : ℝ) ^ (-148 : ℤ) := by
      have hc22 : (((2 : ℚ) ^ (-22 : ℤ) : ℚ) : ℝ) = (2 : ℝ) ^ (-22 : ℤ) := by
        rw [Rat.cast_zpow]
        norm_num
      have hc148 : (((2 : ℚ) ^ (-148 : ℤ) : ℚ) : ℝ) = (2 : ℝ) ^ (-148 : ℤ) := by
        rw [Rat.cast_zpow]
        norm_num
      have h1 : (((|(qv j).toRat - (nv j).toRat / den.toRat| : ℚ)) : ℝ)
          ≤ (((2 : ℚ) ^ (-22 : ℤ) * |(nv j).toRat / den.toRat|
              + (2 : ℚ) ^ (-148 : ℤ) : ℚ) : ℝ) := by
        exact_mod_cast hqerr_q
      rw [hqv_ap' j hjb, Rat.cast_abs, Rat.cast_sub, Rat.cast_div, Rat.cast_add,
        Rat.cast_mul, hc22, hc148, Rat.cast_abs, Rat.cast_div] at h1
      simpa only [FloatV.toReal] using h1
    by_cases hneg : xs j = FloatV.negInf
    · have hxvneg : xv j = FloatV.negInf := by rw [hxsj, hneg]
      have hE0 : ER mv (xv j) = 0 := ER_eq_zero hxvneg
      have hN0 : |(nv j).toReal| ≤ (2 : ℝ) ^ (-137 : ℤ) := by
        have h := hnvlane j hjb
        rw [hE0] at h
        simpa using h
      have hσ0 : softmaxRef xs n j = 0 := by
        rw [softmaxRef_eq hZRne, hneg, ER_eq_zero rfl, zero_div]
      have hgap0 : softmaxGap xs n j = 0 := softmaxGap_eq_neginf hZRne hneg
      have hinf := final_lane_inf (R := (q.get1 ⟨j, hjb⟩).toReal) (N := (nv j).toReal)
        (D := den.toReal) (E := ER mv (xv j)) (A := A) (σ := softmaxRef xs n j)
        (t := 0) hqerrR hN0 hE0 hσ0 rfl hden_half
      have hinf' : |(q.get1 ⟨j, hjb⟩).toReal| ≤ (2 : ℝ) ^ (-133 : ℤ) := by
        have h := hinf
        rw [hσ0] at h
        simpa using h
      refine ⟨hqfin, ?_⟩
      rw [helemTol, hfloorTol, hgap0, hσ0]
      simpa using hinf'
    · have hne : xv j ≠ FloatV.negInf := by rw [hxsj]; exact hneg
      have hEpos : 0 < ER mv (xv j) := ER_pos hne
      have hσA : softmaxRef xs n j = ER mv (xv j) / A := by
        rw [softmaxRef_eq hZRne, ← hxsj, ← hA_eq]
      have hgap : softmaxGap xs n j = -tR mv (xv j) := by
        rw [softmaxGap_eq_fin hi0n hZRpos hmax1 hle1 hj hneg, ← hxsj]
      have hxv_fin : (xv j).isFinite = true := by
        rcases hxok_all j hjb with h | h
        · exact h
        · exact absurd h hne
      have htle : tR mv (xv j) ≤ 0 := by
        have h : ((xv j).toRat : ℝ) ≤ (mv.toRat : ℝ) := by exact_mod_cast hxle_all j hjb
        simp only [tR, mR]
        linarith
      have ht_bound : |tR mv (xv j)| ≤ (2 : ℝ) ^ 129 := by
        have h1 := FloatV.abs_toRat_le_maxVal b32_reg (x := xv j) hxv_fin
        have h2 := FloatV.abs_toRat_le_maxVal b32_reg (x := mv) hmv_fin
        have h3 : |((xv j).toRat : ℝ)| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
          rw [← Rat.cast_abs]
          exact_mod_cast h1
        have h4 : |(mv.toRat : ℝ)| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
          rw [← Rat.cast_abs]
          exact_mod_cast h2
        have h5 : ((Format.binary32.maxVal : ℚ) : ℝ) < (2 : ℝ) ^ 128 := by
          exact_mod_cast maxVal_lt_two_pow
        have h6 : |tR mv (xv j)| ≤ |((xv j).toRat : ℝ)| + |(mv.toRat : ℝ)| := by
          have h7 : tR mv (xv j) = ((xv j).toRat : ℝ) - (mv.toRat : ℝ) := by
            simp only [tR, mR]
          rw [h7]
          have h8 := abs_add_le ((xv j).toRat : ℝ) (-(mv.toRat : ℝ))
          rw [show ((xv j).toRat : ℝ) + -(mv.toRat : ℝ) = ((xv j).toRat : ℝ) - (mv.toRat : ℝ) by ring,
            abs_neg] at h8
          exact h8
        linarith
      have hERsmall : ∀ hbig : (2 : ℝ) ^ 14 ≤ |tR mv (xv j)|,
          ER mv (xv j) ≤ (2 : ℝ) ^ (-256 : ℤ) := by
        intro hbig
        have h256 : (256 : ℝ) ≤ |tR mv (xv j)| := by
          have h14 : (256 : ℝ) ≤ (2 : ℝ) ^ 14 := by norm_num
          linarith
        rw [ER_eq_exp hne]
        have h1 : (256 : ℝ) ≤ -tR mv (xv j) := by
          have h2 : |tR mv (xv j)| = -tR mv (xv j) := abs_of_nonpos htle
          linarith
        have h3 := exp_neg_le_pow2 (x := -tR mv (xv j)) (k := 256) h1
        simpa using h3
      have hbigF : (2 : ℝ) ^ 14 ≤ |tR mv (xv j)| →
          |tR mv (xv j)| * (ER mv (xv j) / den.toReal) ≤ (2 : ℝ) ^ (-99 : ℤ) := by
        intro hbig
        have hERs := hERsmall hbig
        have hED : ER mv (xv j) / den.toReal ≤ 2 * ER mv (xv j) := by
          have h1 : ER mv (xv j) / den.toReal ≤ ER mv (xv j) / (1 / 2) :=
            div_le_div_of_nonneg_left hEpos.le (by norm_num) hden_half
          have h2 : ER mv (xv j) / (1 / 2) = 2 * ER mv (xv j) := by ring
          linarith
        have h1 : |tR mv (xv j)| * (ER mv (xv j) / den.toReal)
            ≤ (2 : ℝ) ^ 129 * (2 * (2 : ℝ) ^ (-256 : ℤ)) := by
          apply mul_le_mul ht_bound _ (div_nonneg hEpos.le (by linarith)) (by positivity)
          calc ER mv (xv j) / den.toReal ≤ 2 * ER mv (xv j) := hED
            _ ≤ 2 * (2 : ℝ) ^ (-256 : ℤ) := mul_le_mul_of_nonneg_left hERs (by norm_num)
        have h2 : (2 : ℝ) ^ 129 * (2 * (2 : ℝ) ^ (-256 : ℤ)) ≤ (2 : ℝ) ^ (-99 : ℤ) := by
          norm_num
        linarith
      have hfin := final_lane_fin (R := (q.get1 ⟨j, hjb⟩).toReal) (N := (nv j).toReal)
        (D := den.toReal) (E := ER mv (xv j)) (A := A) (σ := softmaxRef xs n j)
        (t := tR mv (xv j)) hqerrR (hnvlane j hjb) hden_half hEpos.le hA1 hσA hAD hbigF
      refine ⟨hqfin, ?_⟩
      rw [helemTol, hfloorTol, hgap, ← abs_of_nonpos htle,
        mul_comm (A / den.toReal) (softmaxRef xs n j)]
      exact hfin

end Claims.Softmax
