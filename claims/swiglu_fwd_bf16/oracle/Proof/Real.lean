import Claim
import TVA.Proof

/-!
# `swiglu_fwd_bf16`: the real-number bounds of one lane

The facts about reals and formats the lane needs, before any device fact:

* `bf16_gap`: no `bfloat16` value lies in `(-89, -88.5)` (the spacing there is `1/2`), so an input
  above `-89` is at least `-88.5`, where `tl.exp(-x)` has room in `float32`;
* `exp_885`: `e^88.5 ≤ 0.82·2^128`;
* `exp_rel`: the relative error of `tl.exp(-a)` (`Device.ExpF32` at `X = |a|`) is at most
  `3·2^-18` of `1 + e^-a` for `a ≥ -88.5`;
* `den_err`: the sum `fl(1 + E)` is within `0.8·2^-16` of `1 + e^-a`, relatively;
* `sig_err`: the quotient `1 / fl(1 + E)` (`Device.DivFull`) is within `2^-16` of
  `sigmoid(a) = 1 / (1 + e^-a)`, relatively;
* `out_err`: the two products and the `bfloat16` rounding: within `zTol` of `zRef`.
-/

namespace Claims.SwigluFwd

open TVA

/-! ## Formats -/

/-- No `bfloat16` magnitude lies in `(88.5, 89)`. -/
theorem bf16_magVal_gap {M : ℕ} (h : Format.bfloat16.magVal M < 89) :
    Format.bfloat16.magVal M ≤ 177 / 2 := by
  rw [Format.magVal_eq] at h ⊢
  have hm : Format.bfloat16.m = 7 := rfl
  have hb : Format.bfloat16.bias = 127 := by decide
  have he : Format.bfloat16.emin = -126 := by decide
  simp only [hm, hb, he] at h ⊢
  split_ifs at h ⊢ with h0
  · have hF : ((M % 2 ^ 7 : ℕ) : ℚ) ≤ 127 := by
      have : M % 2 ^ 7 < 2 ^ 7 := Nat.mod_lt _ (by norm_num)
      exact_mod_cast (by omega : M % 2 ^ 7 ≤ 127)
    have hp : (2 : ℚ) ^ ((-126 : ℤ) - (7 : ℕ)) = 1 / 2 ^ 133 := by norm_num
    rw [hp]
    have h1 : ((M % 2 ^ 7 : ℕ) : ℚ) * (1 / 2 ^ 133) ≤ 127 * (1 / 2 ^ 133) :=
      mul_le_mul_of_nonneg_right hF (by positivity)
    have h2 : (127 : ℚ) * (1 / 2 ^ 133) ≤ 177 / 2 := by norm_num
    linarith
  · set E := M / 2 ^ 7 with hE
    set F := M % 2 ^ 7 with hF
    have hF128 : F < 128 := Nat.mod_lt _ (by norm_num)
    have hval : (((2 ^ 7 + F : ℕ) : ℚ)) = 128 + (F : ℚ) := by push_cast; norm_num
    rw [hval] at h ⊢
    rcases lt_trichotomy E 133 with hlt | heq | hgt
    · have hp : (2 : ℚ) ^ (((E : ℕ) : ℤ) - 127 - (7 : ℕ)) ≤ 2 ^ (-2 : ℤ) := by
        apply zpow_le_zpow_right₀ (by norm_num); push_cast; omega
      have hF' : (F : ℚ) ≤ 127 := by exact_mod_cast (by omega : F ≤ 127)
      have hp0 : (0 : ℚ) ≤ (2 : ℚ) ^ (((E : ℕ) : ℤ) - 127 - (7 : ℕ)) := by positivity
      have h4 : (2 : ℚ) ^ (-2 : ℤ) = 1 / 4 := by norm_num
      rw [h4] at hp
      nlinarith
    · rw [heq] at h ⊢
      have h2 : (2 : ℚ) ^ (((133 : ℕ) : ℤ) - 127 - (7 : ℕ)) = 1 / 2 := by norm_num
      rw [h2] at h ⊢
      have hF' : F < 50 := by
        have : (F : ℚ) < 50 := by linarith
        exact_mod_cast this
      have : (F : ℚ) ≤ 49 := by exact_mod_cast (by omega : F ≤ 49)
      linarith
    · have hp : (1 : ℚ) ≤ (2 : ℚ) ^ (((E : ℕ) : ℤ) - 127 - (7 : ℕ)) := by
        apply one_le_zpow₀ (by norm_num); push_cast; omega
      have hF0 : (0 : ℚ) ≤ (F : ℚ) := by positivity
      nlinarith

/-- A `bfloat16` above `-89` is at least `-88.5`. -/
theorem bf16_gap {a : BF16} (h : -89 < a.toReal) : -(177 / 2) ≤ a.toReal := by
  by_cases h0 : 0 ≤ a.toReal
  · linarith
  · push Not at h0
    have habs : |a.toRat| = Format.bfloat16.magVal (Format.bfloat16.mag a.pat) :=
      Format.abs_toRat a.pat
    have hq : a.toRat < 0 := by
      have : (a.toRat : ℝ) < 0 := h0
      exact_mod_cast this
    have hq' : -89 < a.toRat := by
      have : (-89 : ℝ) < (a.toRat : ℝ) := h
      exact_mod_cast this
    rw [abs_of_neg hq] at habs
    have hg := bf16_magVal_gap (M := Format.bfloat16.mag a.pat) (by rw [← habs]; linarith)
    rw [← habs] at hg
    have h1 : -(177 / 2 : ℚ) ≤ a.toRat := by linarith
    have h2 := (Rat.cast_le (K := ℝ)).mpr h1
    push_cast at h2
    exact h2

/-- A `float32` value within `2^-150` of `0` is `0` (the smallest subnormal is `2^-149`). -/
theorem f32_zero_of_small {r : F32} (h : |r.toReal| ≤ 2 ^ (-150 : ℤ)) : r.toReal = 0 := by
  by_contra hne
  have h2 := FloatV.two_eta_le_abs_toReal Format.binary32_regular hne
  rw [Format.binary32_eta_real] at h2
  have : (0 : ℝ) < 2 ^ (-150 : ℤ) := by positivity
  linarith

/-- A `bfloat16` value within `2^-134` of `0` is `0` (the smallest subnormal is `2^-133`). -/
theorem bf16_zero_of_small {r : BF16} (h : |r.toReal| ≤ 2 ^ (-134 : ℤ)) : r.toReal = 0 := by
  by_contra hne
  have h2 := FloatV.two_eta_le_abs_toReal Format.bfloat16_regular hne
  rw [Format.bfloat16_eta_real] at h2
  have : (0 : ℝ) < 2 ^ (-134 : ℤ) := by positivity
  linarith

/-- A nonzero `bfloat16` is at least `2^-133` in magnitude. -/
theorem bf16_abs_ge {a : BF16} (h : a.toReal ≠ 0) : 2 ^ (-133 : ℤ) ≤ |a.toReal| := by
  have h2 := FloatV.two_eta_le_abs_toReal Format.bfloat16_regular h
  rw [Format.bfloat16_eta_real] at h2
  have : (2 : ℝ) * 2 ^ (-134 : ℤ) = 2 ^ (-133 : ℤ) := by
    rw [show (-133 : ℤ) = -134 + 1 by norm_num, zpow_add_one₀ (by norm_num)]
    ring
  linarith

/-- A finite `bfloat16` is at most `255·2^120` in magnitude. -/
theorem bf16_abs_le {a : BF16} (h : a.isFinite = true) : |a.toReal| ≤ 255 * 2 ^ 120 := by
  have h2 := FloatV.abs_toReal_le_maxVal Format.bfloat16_regular h
  rw [Format.bfloat16_maxVal_real] at h2
  linarith

/-! ## The exponential -/

/-- `e^88.5 ≤ 0.82·2^128` (`128·ln 2 > 88.7228`). -/
theorem exp_885 : Real.exp (177 / 2) ≤ 41 / 50 * 2 ^ 128 := by
  have hl : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have h2 : Real.exp (128 * Real.log 2) = 2 ^ 128 := by
    rw [show (128 : ℝ) = ((128 : ℕ) : ℝ) by norm_num, Real.exp_nat_mul,
      Real.exp_log (by norm_num)]
  set z := 128 * Real.log 2 - 177 / 2 with hz
  have hz0 : (2228 / 10000 : ℝ) ≤ z := by rw [hz]; linarith
  have hsplit : Real.exp (177 / 2) * Real.exp z = 2 ^ 128 := by
    rw [← Real.exp_add, hz, show 177 / 2 + (128 * Real.log 2 - 177 / 2) = 128 * Real.log 2 by ring,
      h2]
  have hez : 1 + z ≤ Real.exp z := by linarith [Real.add_one_le_exp z]
  have hpos : 0 < Real.exp (177 / 2) := Real.exp_pos _
  nlinarith

/-- For `a ≥ -88.5`, `1 + e^-a ≤ 2^128`. -/
theorem one_add_exp_le {a : ℝ} (ha : -(177 / 2) ≤ a) : 1 + Real.exp (-a) ≤ 2 ^ 128 := by
  have h1 : Real.exp (-a) ≤ Real.exp (177 / 2) := Real.exp_le_exp.mpr (by linarith)
  have h2 := exp_885
  nlinarith

/-- The constants of the facts, as numerals. -/
theorem k22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
theorem k23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
theorem k24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
theorem k16 : (2 : ℝ) ^ (-16 : ℤ) = 1 / 65536 := by norm_num
theorem k18 : (2 : ℝ) ^ (-18 : ℤ) = 1 / 262144 := by norm_num
theorem k20 : (2 : ℝ) ^ (-20 : ℤ) = 1 / 1048576 := by norm_num
theorem k140 : (2 : ℝ) ^ (-140 : ℤ) = 1 / 2 ^ 140 := by norm_num
theorem k148 : (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 := by norm_num
theorem k150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num

/-- `tl.exp(-a)` (`Device.ExpF32` at `X = |a|`) has room in `float32` for every `a ≥ -88.5`
of a `bfloat16`. -/
theorem exp_room {a : ℝ} (ha : -(177 / 2) ≤ a) (ha' : |a| ≤ 255 * 2 ^ 120) :
    (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |-a|)) * |Real.exp (-a)| + 2 ^ (-140 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
  rw [Format.binary32_maxVal_real, abs_of_pos (Real.exp_pos _), abs_neg, k22, k23, k140]
  have he0 : 0 < Real.exp (-a) := Real.exp_pos _
  by_cases h0 : 0 ≤ a
  · have he : Real.exp (-a) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hk0 : 0 ≤ 1 + (1 / 4194304 + 1 / 8388608 * |a|) := by positivity
    have h1 : (1 + (1 / 4194304 + 1 / 8388608 * |a|)) * Real.exp (-a) ≤
        1 + (1 / 4194304 + 1 / 8388608 * |a|) := by
      have := mul_le_mul_of_nonneg_left he hk0; linarith
    nlinarith
  · push Not at h0
    have habs : |a| ≤ 177 / 2 := by rw [abs_of_neg h0]; linarith
    have he : Real.exp (-a) ≤ 41 / 50 * 2 ^ 128 :=
      le_trans (Real.exp_le_exp.mpr (by linarith)) exp_885
    have hk : 1 + (1 / 4194304 + 1 / 8388608 * |a|) ≤ 1 + (1 / 4194304 + 1 / 8388608 * (177 / 2)) := by
      linarith
    have := mul_le_mul hk he he0.le (by positivity)
    nlinarith

/-- The relative error of `tl.exp(-a)`: `(2^-22 + 2^-23·|a|)·e^-a ≤ 3·2^-18·(1 + e^-a)` for
`a ≥ -88.5`. -/
theorem exp_rel {a : ℝ} (ha : -(177 / 2) ≤ a) :
    (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |-a|) * Real.exp (-a) ≤
      3 * 2 ^ (-18 : ℤ) * (1 + Real.exp (-a)) := by
  rw [abs_neg, k22, k23, k18]
  have he0 : 0 < Real.exp (-a) := Real.exp_pos _
  by_cases h0 : 0 ≤ a
  · -- `e^-a ≤ 1` and `a·e^-a ≤ 1`
    have he1 : Real.exp (-a) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hae : a * Real.exp (-a) ≤ 1 := by
      have h1 : a + 1 ≤ Real.exp a := Real.add_one_le_exp a
      have h2 : Real.exp a * Real.exp (-a) = 1 := by rw [← Real.exp_add]; simp
      nlinarith
    rw [abs_of_nonneg h0]
    nlinarith
  · push Not at h0
    rw [abs_of_neg h0]
    nlinarith

/-! ## The sigmoid -/

/-- The sum `fl(1 + E)`, `E` the `tl.exp(-a)` result: within `0.8·2^-16` of `1 + e^-a`,
relatively. -/
theorem den_err {a e E Dn : ℝ} (ha : -(177 / 2) ≤ a) (he : e = Real.exp (-a))
    (hE : |E - e| ≤ (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |-a|) * e + 2 ^ (-140 : ℤ))
    (hDn : |Dn - (1 + E)| ≤ |1 + E| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) :
    |Dn - (1 + e)| ≤ 4 / 5 * 2 ^ (-16 : ℤ) * (1 + e) := by
  have hrel := exp_rel ha
  rw [← he] at hrel
  rw [k18] at hrel
  rw [k140] at hE
  rw [k24, k150] at hDn
  rw [k16]
  have he0 : 0 < e := by rw [he]; exact Real.exp_pos _
  set u := 1 + e with hu
  have hu1 : 1 ≤ u := by linarith
  have hEu : |E - e| ≤ (3 * (1 / 262144) + 1 / 2 ^ 140) * u := by nlinarith
  have h1E : |1 + E| ≤ (1 + (3 * (1 / 262144) + 1 / 2 ^ 140)) * u := by
    have : |1 + E| ≤ |u| + |E - e| := by
      have : 1 + E = u + (E - e) := by rw [hu]; ring
      rw [this]; exact abs_add_le _ _
    rw [abs_of_pos (by linarith : (0:ℝ) < u)] at this
    linarith
  have htri : |Dn - u| ≤ |Dn - (1 + E)| + |E - e| := by
    have : Dn - u = (Dn - (1 + E)) + (E - e) := by rw [hu]; ring
    rw [this]; exact abs_add_le _ _
  nlinarith

/-- The kernel's sigmoid: from the sum within `0.8·2^-16` of `u = 1 + e^-a` (with
`1 ≤ u ≤ 2^128`) and the division within `DivFull`'s bound, the quotient is within `2^-16` of
`1/u`, relatively. -/
theorem sig_err {u Dn S : ℝ} (hu1 : 1 ≤ u) (hu2 : u ≤ 2 ^ 128)
    (hD : |Dn - u| ≤ 4 / 5 * 2 ^ (-16 : ℤ) * u)
    (hS : |S - 1 / Dn| ≤ 2 ^ (-22 : ℤ) * |1 / Dn| + 2 ^ (-148 : ℤ)) :
    |S - 1 / u| ≤ 2 ^ (-16 : ℤ) * (1 / u) := by
  rw [k16] at hD ⊢
  rw [k22, k148] at hS
  have hu0 : 0 < u := by linarith
  have hDlo : (1 - 4 / 5 * (1 / 65536)) * u ≤ Dn := by
    have := (abs_le.mp hD).1; linarith
  have hDpos : 0 < Dn := by nlinarith
  set s0 := 1 / u with hs0
  set d' := 1 / Dn with hd'
  have hs0pos : 0 < s0 := by positivity
  have hd'pos : 0 < d' := by positivity
  have hus : u * s0 = 1 := by rw [hs0]; field_simp
  have hDd : Dn * d' = 1 := by rw [hd']; field_simp
  -- `(1 - c)·d' ≤ s0`
  have hd'le : (1 - 4 / 5 * (1 / 65536)) * d' ≤ s0 := by
    have h1 : (1 - 4 / 5 * (1 / 65536)) * u * d' ≤ Dn * d' :=
      mul_le_mul_of_nonneg_right hDlo hd'pos.le
    rw [hDd] at h1
    have h2 : (1 - 4 / 5 * (1 / 65536)) * d' * (u * s0) ≤ 1 * s0 := by
      nlinarith
    rw [hus] at h2
    linarith
  -- `|d' - s0| ≤ c·d'`
  have hdiff : |d' - s0| ≤ 4 / 5 * (1 / 65536) * d' := by
    have heq : d' - s0 = (u - Dn) * d' * s0 := by
      rw [hd', hs0]; field_simp
    rw [heq, abs_mul, abs_mul, abs_of_pos hd'pos, abs_of_pos hs0pos, abs_sub_comm]
    have h1 : |Dn - u| * s0 ≤ 4 / 5 * (1 / 65536) := by
      have := mul_le_mul_of_nonneg_right hD hs0pos.le
      rw [mul_assoc, hus, mul_one] at this
      exact this
    nlinarith [abs_nonneg (Dn - u)]
  have h148 : (1 : ℝ) / 2 ^ 148 ≤ 1 / 1048576 * s0 := by
    have h1 : (1 : ℝ) / 2 ^ 128 ≤ s0 := by
      rw [hs0]; exact one_div_le_one_div_of_le hu0 hu2
    nlinarith
  rw [abs_of_pos hd'pos] at hS
  have htri : |S - s0| ≤ |S - d'| + |d' - s0| := by
    have : S - s0 = (S - d') + (d' - s0) := by ring
    rw [this]; exact abs_add_le _ _
  nlinarith

/-! ## The products and the rounding -/

/-- The absolute error `2^-150` of the first product is relative for a nonzero `bfloat16` input
`a ≥ -88.5`: `2^-150 ≤ 2^-15·|a|·σ`, `σ = 1/(1 + e^-a)`. -/
theorem eta_rel {a : ℝ} (ha : -(177 / 2) ≤ a) (ha0 : 2 ^ (-133 : ℤ) ≤ |a|) :
    (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-15 : ℤ) * |a| * (1 / (1 + Real.exp (-a))) := by
  have hu1 : 1 ≤ 1 + Real.exp (-a) := by linarith [Real.exp_pos (-a)]
  have hu0 : 0 < 1 + Real.exp (-a) := by linarith
  rw [show (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 by norm_num,
    show (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 by norm_num]
  rw [show (2 : ℝ) ^ (-133 : ℤ) = 1 / 2 ^ 133 by norm_num] at ha0
  by_cases h1 : |a| ≤ 1
  · -- `e^-a ≤ e < 3`, so `σ ≥ 1/4`
    have he : Real.exp (-a) ≤ 3 := by
      have : -a ≤ 1 := by have := neg_abs_le a; linarith
      have h2 := Real.exp_le_exp.mpr this
      have h3 := Real.exp_one_lt_d9
      linarith
    have hs : (1 : ℝ) / 4 ≤ 1 / (1 + Real.exp (-a)) := one_div_le_one_div_of_le hu0 (by linarith)
    have : (1 : ℝ) / 2 ^ 150 ≤ 1 / 32768 * (1 / 2 ^ 133) * (1 / 4) := by norm_num
    have h4 : 1 / 32768 * (1 / 2 ^ 133) * (1 / 4) ≤ 1 / 32768 * |a| * (1 / (1 + Real.exp (-a))) := by
      apply mul_le_mul _ hs (by norm_num) (by positivity)
      exact mul_le_mul_of_nonneg_left ha0 (by norm_num)
    linarith
  · push Not at h1
    have hs : (1 : ℝ) / 2 ^ 128 ≤ 1 / (1 + Real.exp (-a)) :=
      one_div_le_one_div_of_le hu0 (one_add_exp_le ha)
    have : (1 : ℝ) / 2 ^ 150 ≤ 1 / 32768 * 1 * (1 / 2 ^ 128) := by norm_num
    have h4 : 1 / 32768 * 1 * (1 / 2 ^ 128) ≤ 1 / 32768 * |a| * (1 / (1 + Real.exp (-a))) := by
      apply mul_le_mul _ hs (by norm_num) (by positivity)
      exact mul_le_mul_of_nonneg_left h1.le (by norm_num)
    linarith

/-- The first product `P = fl(a·S)`: within `2^-14` of `a·σ`, relatively, when `S` is within
`2^-16` of `σ` and the absolute error is relative (`eta_rel`). -/
theorem xs_err {a σ S P : ℝ} (hσ : 0 < σ) (hS : |S - σ| ≤ 2 ^ (-16 : ℤ) * σ)
    (heta : (2 : ℝ) ^ (-150 : ℤ) ≤ 2 ^ (-15 : ℤ) * |a| * σ)
    (hP : |P - a * S| ≤ |a * S| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) :
    |P - a * σ| ≤ 2 ^ (-14 : ℤ) * |a * σ| := by
  rw [k16] at hS
  rw [k24] at hP
  rw [show (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 by norm_num] at heta
  rw [show (2 : ℝ) ^ (-14 : ℤ) = 1 / 16384 by norm_num]
  have hSle : |S| ≤ (1 + 1 / 65536) * σ := by
    have : |S| ≤ |σ| + |S - σ| := by
      have : S = σ + (S - σ) := by ring
      conv_lhs => rw [this]
      exact abs_add_le _ _
    rw [abs_of_pos hσ] at this; linarith
  have haS : |a * S| ≤ (1 + 1 / 65536) * (|a| * σ) := by
    rw [abs_mul]; nlinarith [abs_nonneg a]
  have hdiff : |a * S - a * σ| ≤ 1 / 65536 * (|a| * σ) := by
    rw [← mul_sub, abs_mul]; nlinarith [abs_nonneg a]
  have htri : |P - a * σ| ≤ |P - a * S| + |a * S - a * σ| := by
    have : P - a * σ = (P - a * S) + (a * S - a * σ) := by ring
    rw [this]; exact abs_add_le _ _
  rw [abs_mul, abs_of_pos hσ]
  nlinarith [abs_nonneg a]

/-- The second product and the `bfloat16` rounding: from `P` within `2^-14` of `w = a·σ`, the
product `Z = fl(P·b)` and its rounding `O` are within `zTol` of `w·b`. -/
theorem out_err {w b P Z O : ℝ} (hP : |P - w| ≤ 2 ^ (-14 : ℤ) * |w|)
    (hZ : |Z - P * b| ≤ |P * b| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ))
    (hO : |O - Z| ≤ |Z| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) :
    |O - w * b| ≤ zTol (w * b) := by
  unfold zTol
  rw [k24, k150] at hZ
  rw [show (2 : ℝ) ^ (-14 : ℤ) = 1 / 16384 by norm_num] at hP
  rw [show (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 by norm_num, show (2 : ℝ) ^ (-134 : ℤ) = 1 / 2 ^ 134 by norm_num] at hO
  rw [show (2 : ℝ) ^ (-8 : ℤ) = 1 / 256 by norm_num, show (2 : ℝ) ^ (-13 : ℤ) = 1 / 8192 by norm_num,
    show (2 : ℝ) ^ (-133 : ℤ) = 1 / 2 ^ 133 by norm_num]
  have hPb : |P * b - w * b| ≤ 1 / 16384 * |w * b| := by
    rw [← sub_mul, abs_mul, abs_mul]; nlinarith [abs_nonneg b]
  have hPb2 : |P * b| ≤ (1 + 1 / 16384) * |w * b| := by
    have : |P * b| ≤ |w * b| + |P * b - w * b| := by
      have : P * b = w * b + (P * b - w * b) := by ring
      conv_lhs => rw [this]
      exact abs_add_le _ _
    linarith
  have hZw : |Z - w * b| ≤ (1 / 16384 + (1 + 1 / 16384) * (1 / 16777216)) * |w * b| + 1 / 2 ^ 150 := by
    have : |Z - w * b| ≤ |Z - P * b| + |P * b - w * b| := by
      have : Z - w * b = (Z - P * b) + (P * b - w * b) := by ring
      rw [this]; exact abs_add_le _ _
    nlinarith [abs_nonneg (w * b)]
  have hZ2 : |Z| ≤ |w * b| + |Z - w * b| := by
    have : Z = w * b + (Z - w * b) := by ring
    conv_lhs => rw [this]
    exact abs_add_le _ _
  have htri : |O - w * b| ≤ |O - Z| + |Z - w * b| := by
    have : O - w * b = (O - Z) + (Z - w * b) := by ring
    rw [this]; exact abs_add_le _ _
  nlinarith [abs_nonneg (w * b)]

/-- The product's room: `|P·b| ≤ (1 + 2^-14)·|w·b|`. -/
theorem pb_le {w b P : ℝ} (hP : |P - w| ≤ 2 ^ (-14 : ℤ) * |w|) :
    |P * b| ≤ (1 + 2 ^ (-14 : ℤ)) * |w * b| := by
  have hPb : |P * b - w * b| ≤ 2 ^ (-14 : ℤ) * |w * b| := by
    rw [← sub_mul, abs_mul, abs_mul]
    have := mul_le_mul_of_nonneg_right hP (abs_nonneg b)
    linarith
  have : |P * b| ≤ |w * b| + |P * b - w * b| := by
    have : P * b = w * b + (P * b - w * b) := by ring
    conv_lhs => rw [this]
    exact abs_add_le _ _
  linarith

/-- The rounding's room: `|Z| ≤ (1 + 2^-13)·|w·b| + 2^-150`. -/
theorem z_le {w b P Z : ℝ} (hP : |P - w| ≤ 2 ^ (-14 : ℤ) * |w|)
    (hZ : |Z - P * b| ≤ |P * b| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ)) :
    |Z| ≤ (1 + 2 ^ (-13 : ℤ)) * |w * b| + 2 ^ (-150 : ℤ) := by
  have h1 := pb_le (b := b) hP
  rw [show (2 : ℝ) ^ (-14 : ℤ) = 1 / 16384 by norm_num] at h1
  rw [k24, k150] at hZ
  rw [show (2 : ℝ) ^ (-13 : ℤ) = 1 / 8192 by norm_num, k150]
  have : |Z| ≤ |P * b| + |Z - P * b| := by
    have : Z = P * b + (Z - P * b) := by ring
    conv_lhs => rw [this]
    exact abs_add_le _ _
  nlinarith [abs_nonneg (w * b)]

end Claims.SwigluFwd
