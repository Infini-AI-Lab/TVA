import Proof.Bounds
import Proof.Zero

/-!
# The state term: numeric facts

The scale of the entering state (lines 337–344) and the state term `acc₀` (lines 355–361): statements the walk uses, each with its proof.
-/

namespace Claims.SsdChunkScan

open TVA

/-! ## Rounding facts, as real numbers -/

/-! ### Casting rational error bounds to real ones -/

/-- A bound on a difference of rationals, cast to the reals. -/
private lemma rat_abs_sub_le_real {a b c : ℚ} {A B C : ℝ}
    (ha : ((a : ℚ) : ℝ) = A) (hb : ((b : ℚ) : ℝ) = B) (hc : ((c : ℚ) : ℝ) = C)
    (h : |a - b| ≤ c) : |A - B| ≤ C := by
  have h1 : ((|a - b| : ℚ) : ℝ) ≤ ((c : ℚ) : ℝ) := by exact_mod_cast h
  rw [Rat.cast_abs] at h1
  push_cast at h1
  rwa [ha, hb, hc] at h1

/-- `float32`'s `u`, as a real. -/
private lemma u32_cast : ((Format.binary32.u : ℚ) : ℝ) = (2:ℝ)^(-24:ℤ) := by
  rw [Format.binary32_u]; push_cast; norm_num

/-- `float32`'s `η`, as a real. -/
private lemma eta32_cast : ((Format.binary32.eta : ℚ) : ℝ) = (2:ℝ)^(-150:ℤ) := by
  rw [Format.binary32_eta]; push_cast; norm_num

/-- `bfloat16`'s `u`, as a real. -/
private lemma ubf_cast : ((Format.bfloat16.u : ℚ) : ℝ) = (2:ℝ)^(-8:ℤ) := by
  rw [Format.bfloat16_u]; push_cast; norm_num

/-- `bfloat16`'s `η`, as a real. -/
private lemma etabf_cast : ((Format.bfloat16.eta : ℚ) : ℝ) = (2:ℝ)^(-134:ℤ) := by
  rw [Format.bfloat16_eta]; push_cast; norm_num

/-- `FloatV.sub_err` on the reals. -/
private lemma sub32_real {x y : FloatV .binary32}
    (h : |(FloatV.sub x y).toRat - (x.toRat - y.toRat)| ≤
      |x.toRat - y.toRat| * Format.binary32.u + Format.binary32.eta) :
    |(FloatV.sub x y).toReal - ((x.toRat : ℝ) - (y.toRat : ℝ))| ≤
      |(x.toRat : ℝ) - (y.toRat : ℝ)| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
  refine rat_abs_sub_le_real ?_ ?_ ?_ h
  · simp [FloatV.toReal]
  · push_cast; ring
  · push_cast; rw [u32_cast, eta32_cast]

/-- `FloatV.mul_err` on the reals. -/
private lemma mul32_real {x y : FloatV .binary32}
    (h : |(FloatV.mul x y).toRat - x.toRat * y.toRat| ≤
      |x.toRat * y.toRat| * Format.binary32.u + Format.binary32.eta) :
    |(FloatV.mul x y).toReal - (x.toRat : ℝ) * (y.toRat : ℝ)| ≤
      |(x.toRat : ℝ) * (y.toRat : ℝ)| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
  refine rat_abs_sub_le_real ?_ ?_ ?_ h
  · simp [FloatV.toReal]
  · push_cast; ring
  · push_cast; rw [u32_cast, eta32_cast]

/-- `FloatV.cvt_err` into `bfloat16` on the reals. -/
private lemma cvt32bf_real {x : FloatV .binary32}
    (h : |(FloatV.cvtWith .rn x : FloatV .bfloat16).toRat - x.toRat| ≤
      |x.toRat| * Format.bfloat16.u + Format.bfloat16.eta) :
    |(FloatV.cvtWith .rn x : FloatV .bfloat16).toReal - (x.toRat : ℝ)| ≤
      |(x.toRat : ℝ)| * (2:ℝ)^(-8:ℤ) + (2:ℝ)^(-134:ℤ) := by
  refine rat_abs_sub_le_real ?_ ?_ ?_ h
  · simp [FloatV.toReal]
  · ring
  · push_cast; rw [ubf_cast, etabf_cast]


/-! ### Numeric facts about the formats -/

/-- `float32`'s largest value, as a real. -/
private lemma maxVal32_eq : (Format.binary32.maxVal : ℝ) = ((2:ℝ)^24 - 1) * (2:ℝ)^104 := by
  rw [Format.binary32_maxVal]; push_cast; norm_num

/-- `float32`'s largest value is below `2^128`. -/
private lemma maxVal32_le_pow128 : (Format.binary32.maxVal : ℝ) ≤ (2:ℝ)^128 := by
  rw [maxVal32_eq]
  have h : ((2:ℝ)^24 - 1) * (2:ℝ)^104 ≤ (2:ℝ)^24 * (2:ℝ)^104 :=
    mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  have h2 : (2:ℝ)^24 * (2:ℝ)^104 = (2:ℝ)^128 := by
    rw [show (128:ℕ) = 24 + 104 by norm_num, pow_add]
  linarith

/-- `2^109` is within `float32`'s range. -/
private lemma pow109_le_maxVal32 : (2:ℝ)^109 ≤ (Format.binary32.maxVal : ℝ) := by
  rw [maxVal32_eq]
  have h32 : (32:ℝ) ≤ (2:ℝ)^24 - 1 := by norm_num
  have h := mul_le_mul_of_nonneg_right h32 (show (0:ℝ) ≤ (2:ℝ)^104 by positivity)
  have h2 : (32:ℝ) * (2:ℝ)^104 = (2:ℝ)^109 := by
    rw [show (109:ℕ) = 5 + 104 by norm_num, pow_add]; norm_num
  linarith

/-- `2^126` is within `float32`'s range. -/
private lemma pow126_le_maxVal32 : (2:ℝ)^126 ≤ (Format.binary32.maxVal : ℝ) := by
  rw [maxVal32_eq]
  have h32 : (2:ℝ)^22 ≤ (2:ℝ)^24 - 1 := by norm_num
  have h := mul_le_mul_of_nonneg_right h32 (show (0:ℝ) ≤ (2:ℝ)^104 by positivity)
  have h2 : (2:ℝ)^22 * (2:ℝ)^104 = (2:ℝ)^126 := by
    rw [show (126:ℕ) = 22 + 104 by norm_num, pow_add]
  linarith

/-- `bfloat16`'s largest value, as a real. -/
private lemma maxValbf_eq : (Format.bfloat16.maxVal : ℝ) = ((2:ℝ)^8 - 1) * (2:ℝ)^120 := by
  rw [Format.bfloat16_maxVal]; push_cast; norm_num

/-- `exp 65 ≤ 2^104`. -/
private lemma exp65_le_pow104 : Real.exp 65 ≤ (2:ℝ)^104 := by
  have h : Real.exp 65 = (Real.exp 1)^65 := by
    rw [show (65:ℝ) = (65:ℕ) * 1 by norm_num, Real.exp_nat_mul]
  rw [h]
  have h1 : (Real.exp 1)^65 ≤ (3:ℝ)^65 :=
    pow_le_pow_left₀ (Real.exp_pos 1).le Real.exp_one_lt_three.le 65
  have h2 : (3:ℝ)^65 ≤ (2:ℝ)^104 := by norm_num
  linarith


/-- The exponential moves by at most `exp(D + e) − exp D` when its argument moves by `e`. -/
private lemma exp_sub_le {D t e : ℝ} (_he0 : 0 ≤ e) (ht : |t - D| ≤ e) :
    |Real.exp t - Real.exp D| ≤ Real.exp (D + e) - Real.exp D := by
  have ht1 : t ≤ D + e := by have := (abs_le.mp ht).2; linarith
  have ht2 : D - e ≤ t := by have := (abs_le.mp ht).1; linarith
  have hA : (0:ℝ) < Real.exp D := Real.exp_pos D
  have hx : (0:ℝ) < Real.exp e := Real.exp_pos e
  have hkey : 2 ≤ Real.exp e + (Real.exp e)⁻¹ := by
    have hsq : 0 ≤ (Real.exp e - 1)^2 := sq_nonneg _
    have hmul : (Real.exp e) * (Real.exp e)⁻¹ = 1 := mul_inv_cancel₀ (ne_of_gt hx)
    nlinarith [hsq, hx, hmul]
  have h3 : Real.exp D - Real.exp (D - e) ≤ Real.exp (D + e) - Real.exp D := by
    have e1 : Real.exp (D - e) = Real.exp D / Real.exp e := by rw [Real.exp_sub]
    have e2 : Real.exp (D + e) = Real.exp D * Real.exp e := Real.exp_add ..
    rw [e1, e2, div_eq_mul_inv]
    have h2 : Real.exp D * 2 ≤ Real.exp D * (Real.exp e + (Real.exp e)⁻¹) :=
      mul_le_mul_of_nonneg_left hkey hA.le
    nlinarith [h2]
  rw [abs_le]
  constructor
  · have h2 : Real.exp (D - e) ≤ Real.exp t := Real.exp_le_exp.mpr ht2
    linarith
  · have h2 : Real.exp t ≤ Real.exp (D + e) := Real.exp_le_exp.mpr ht1
    linarith

/-- **Room for `tl.exp`.** For `D ≤ 64` and `t` within the subtraction's error of `D`, the bound `ExpF32` needs (`(1 + rel)·|exp t| + 2^-140 ≤ maxVal`) holds, for very negative `D` as well (there `exp t` is tiny). -/
private lemma exp_room {D t : ℝ} (hD : D ≤ 64)
    (hDm : |D| ≤ (Format.binary32.maxVal : ℝ))
    (ht : |t - D| ≤ |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)) :
    (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) *
      (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)))))
      * |Real.exp t| + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
  set e : ℝ := |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) with he
  have he0 : 0 ≤ e := by rw [he]; positivity
  have htle : t ≤ D + e := by have := (abs_le.mp ht).2; linarith
  rw [abs_of_pos (Real.exp_pos t)]
  by_cases hcase : |D| ≤ 1024
  · have he1 : e ≤ 1 := by
      rw [he]
      have h1 : |D| * (2:ℝ)^(-24:ℤ) ≤ 1024 * (2:ℝ)^(-24:ℤ) :=
        mul_le_mul_of_nonneg_right hcase (by positivity)
      have h2 : (1024:ℝ) * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) ≤ 1 := by norm_num
      linarith
    have ht65 : t ≤ 65 := by linarith
    have hE1 : Real.exp t ≤ (2:ℝ)^104 :=
      le_trans (Real.exp_le_exp.mpr ht65) exp65_le_pow104
    have hK1 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ 1 := by
      have h1 : |D| + e ≤ (1025:ℝ) := by linarith
      have h2 : (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^(-23:ℤ) * (1025:ℝ) :=
        mul_le_mul_of_nonneg_left h1 (by positivity)
      have h3 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (1025:ℝ) ≤ 1 := by norm_num
      linarith
    have h2 : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) ≤ 2 := by linarith
    have hmul : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp t
        ≤ 2 * (2:ℝ)^104 :=
      mul_le_mul h2 hE1 (Real.exp_pos t).le (by norm_num)
    have hfin : 2 * (2:ℝ)^104 + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
      have h2 : (2:ℝ)^105 + (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^109 := by norm_num
      have h3 : (2:ℝ) * (2:ℝ)^104 = (2:ℝ)^105 := by
        rw [show (105:ℕ) = 1 + 104 by norm_num, pow_add]; norm_num
      linarith [pow109_le_maxVal32]
    linarith
  · have h1024 : (1024:ℝ) < |D| := lt_of_not_ge hcase
    have hDneg : D < 0 := by
      rcases lt_or_ge D 0 with h | h
      · exact h
      · rw [abs_of_nonneg h] at h1024; linarith
    have hDle : D ≤ -1024 := by
      rw [abs_of_neg hDneg] at h1024; linarith
    have hDabs : |D| = -D := abs_of_neg hDneg
    have hee : e = (-D) * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by rw [he, hDabs]
    have hDp : D + e < 0 := by
      rw [hee]
      have h1 : D + (-D) * (2:ℝ)^(-24:ℤ) = D * (1 - (2:ℝ)^(-24:ℤ)) := by ring
      have h2 : D * (1 - (2:ℝ)^(-24:ℤ)) ≤ (-1024) * (1 - (2:ℝ)^(-24:ℤ)) :=
        mul_le_mul_of_nonneg_right hDle (by norm_num)
      have h3 : (-1024:ℝ) * (1 - (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ) < 0 := by norm_num
      linarith
    have ht0 : t < 0 := by linarith
    have hexpt1 : Real.exp t ≤ 1 := by
      have h := Real.exp_le_exp.mpr ht0.le
      simpa using h
    have heM : e ≤ (Format.binary32.maxVal : ℝ) := by
      rw [he]
      have h1 : |D| * (2:ℝ)^(-24:ℤ) ≤ (2:ℝ)^128 * (2:ℝ)^(-24:ℤ) :=
        mul_le_mul_of_nonneg_right (le_trans hDm maxVal32_le_pow128) (by positivity)
      have h2 : (2:ℝ)^128 * (2:ℝ)^(-24:ℤ) = (2:ℝ)^104 := by norm_num
      have h3 : (2:ℝ)^104 + (2:ℝ)^(-150:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
        rw [maxVal32_eq]; norm_num
      linarith
    have hK107 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^107 := by
      have hsum : |D| + e ≤ 2 * (2:ℝ)^128 := by
        have h := maxVal32_le_pow128
        linarith
      have h2 : (2:ℝ)^(-23:ℤ) * (|D| + e) ≤ (2:ℝ)^(-23:ℤ) * (2 * (2:ℝ)^128) :=
        mul_le_mul_of_nonneg_left hsum (by positivity)
      have h3 : (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (2 * (2:ℝ)^128) ≤ (2:ℝ)^107 := by
        norm_num
      linarith
    have h1K : 1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) ≤ (2:ℝ)^108 := by
      have h : (2:ℝ)^107 + 1 ≤ (2:ℝ)^108 := by norm_num
      linarith
    have hmul : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp t
        ≤ (2:ℝ)^108 * 1 :=
      mul_le_mul h1K hexpt1 (Real.exp_pos t).le (by positivity)
    have hfin : (2:ℝ)^108 + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
      have h2 : (2:ℝ)^108 + (2:ℝ)^(-140:ℤ) ≤ (2:ℝ)^109 := by norm_num
      linarith [pow109_le_maxVal32]
    linarith

/-- A real bound on a product of `float32` values below `maxVal` gives the rational one. -/
private lemma rat_abs_mul_le {x y : FloatV .binary32} {C : ℝ}
    (h : |(x.toRat : ℝ) * (y.toRat : ℝ)| ≤ C)
    (hC : C ≤ (Format.binary32.maxVal : ℝ)) :
    |x.toRat * y.toRat| ≤ Format.binary32.maxVal := by
  have h2 : ((|x.toRat * y.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Rat.cast_abs]
    push_cast
    linarith
  exact_mod_cast h2

/-- **The exponential's error against `exp D`**: `ExpF32`'s error at `t`, with `t` within `e` of `D`, is `errE`'s. -/
private lemma exp_combined {r t D e : ℝ} (he0 : 0 ≤ e) (ht : |t - D| ≤ e)
    (hr : |r - Real.exp t| ≤
      ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * |Real.exp t| + (2:ℝ)^(-140:ℤ)) :
    |r - Real.exp D| ≤
      (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp (D + e)
        - Real.exp D + (2:ℝ)^(-140:ℤ) := by
  have h2 := exp_sub_le he0 ht
  have htle : t ≤ D + e := by have h := (abs_le.mp ht).2; linarith
  have hexpt : Real.exp t ≤ Real.exp (D + e) := Real.exp_le_exp.mpr htle
  have hK0 : (0:ℝ) ≤ (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e) := by positivity
  have h3 : ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * |Real.exp t|
      ≤ ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * Real.exp (D + e) :=
    mul_le_mul_of_nonneg_left (by rwa [abs_of_pos (Real.exp_pos t)]) hK0
  have htri : |r - Real.exp D| ≤ |r - Real.exp t| + |Real.exp t - Real.exp D| := by
    have h := abs_add_le (r - Real.exp t) (Real.exp t - Real.exp D)
    rw [show (r - Real.exp t) + (Real.exp t - Real.exp D) = r - Real.exp D by ring] at h
    exact h
  have hid : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e))) * Real.exp (D + e)
      - Real.exp D + (2:ℝ)^(-140:ℤ)
      = ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + e)) * Real.exp (D + e)
        + (Real.exp (D + e) - Real.exp D) + (2:ℝ)^(-140:ℤ) := by ring
  linarith [htri, hr, h3, h2]

/-- A `float32` value of magnitude at most `C ≤ bfloat16.maxVal` converts within range. -/
private lemma rat_abs_le_bf {x : FloatV .binary32} {C : ℝ}
    (h : |(x.toRat : ℝ)| ≤ C)
    (hC : C ≤ (Format.bfloat16.maxVal : ℝ)) :
    |x.toRat| ≤ Format.bfloat16.maxVal := by
  have h2 : ((|x.toRat| : ℚ) : ℝ) ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
    rw [Rat.cast_abs]
    linarith
  exact_mod_cast h2

/-- `2^120` is within `bfloat16`'s range. -/
private lemma pow120_le_maxValbf : (2:ℝ)^120 ≤ (Format.bfloat16.maxVal : ℝ) := by
  rw [maxValbf_eq]
  have h : (1:ℝ) ≤ (2:ℝ)^8 - 1 := by norm_num
  have := mul_le_mul_of_nonneg_right h (show (0:ℝ) ≤ (2:ℝ)^120 by positivity)
  linarith

/-! ## The scale of the entering state -/

/-- A rational of magnitude at most `float32`'s largest value, as a real. -/
private lemma real_abs_le_maxVal {q : ℚ} (h : |q| ≤ Format.binary32.maxVal) :
    |(q : ℝ)| ≤ (Format.binary32.maxVal : ℝ) := by
  have h1 : ((|q| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by exact_mod_cast h
  rwa [Rat.cast_abs] at h1

/-- **`tl.exp` of a computed exponent.** If `xt` is finite and within `δ` of `D ≤ 64`, where
`0 ≤ δ ≤ |D|·2^-24 + 2^-150`, the platform's `exp` of it is finite, within `errE D δ` of `exp D`,
and of magnitude at most `eHat D δ`. -/
private lemma scale_core {P : Platform} (hexp : Device.ExpF32 P) {D δ : ℝ} {xt e : F32}
    (hD : D ≤ 64) (hDm : |D| ≤ (Format.binary32.maxVal : ℝ)) (hδ0 : 0 ≤ δ)
    (hδF : δ ≤ |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ)) (hxf : xt.isFinite = true)
    (hxD : |(xt.toRat : ℝ) - D| ≤ δ) (hex : P.math .exp .binary32 xt e) :
    e.isFinite = true ∧ |(e.toRat : ℝ) - Real.exp D| ≤ errE D δ ∧ |(e.toRat : ℝ)| ≤ eHat D δ := by
  obtain ⟨ref, href, hprop⟩ := hexp (|D| + δ) (by positivity)
  have href' : ref = Real.exp := by
    have h1 : MathFn.ref .exp = some Real.exp := rfl
    rw [h1] at href
    cases href
    rfl
  rw [href'] at hprop
  have hmem : xt.toReal ∈ Set.Icc (-(|D| + δ)) (|D| + δ) := by
    rw [Set.mem_Icc]
    have h2 := abs_le.mp hxD
    have h3 : -|D| ≤ D := neg_abs_le _
    have h4 : D ≤ |D| := le_abs_self _
    constructor <;> simp only [FloatV.toReal] <;> linarith [h2.1, h2.2]
  have hxF : |(xt.toRat : ℝ) - D| ≤ |D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := hxD.trans hδF
  have hroomF := exp_room hD hDm hxF
  have hrel : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + δ))) ≤
      (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ))))) := by
    have : (2:ℝ)^(-23:ℤ) * (|D| + δ) ≤ (2:ℝ)^(-23:ℤ) *
        (|D| + (|D| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ))) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    linarith
  have hroom : (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|D| + δ))) * |Real.exp xt.toReal|
      + (2:ℝ)^(-140:ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    have := mul_le_mul_of_nonneg_right hrel (abs_nonneg (Real.exp xt.toReal))
    simp only [FloatV.toReal] at this ⊢
    linarith
  obtain ⟨hfin, herr⟩ := hprop xt e hex hxf hmem (by simp [MathFn.natDom]) hroom
  have hcomb := exp_combined (D := D) (e := δ) (t := xt.toReal) hδ0 hxD herr
  have herrE : |(e.toRat : ℝ) - Real.exp D| ≤ errE D δ := by
    simpa only [errE, expRel, FloatV.toReal] using hcomb
  refine ⟨hfin, herrE, ?_⟩
  have h1 : |(e.toRat : ℝ)| ≤ |Real.exp D| + |(e.toRat : ℝ) - Real.exp D| := by
    have hadd := abs_add_le (Real.exp D) ((e.toRat : ℝ) - Real.exp D)
    rw [show Real.exp D + ((e.toRat : ℝ) - Real.exp D) = (e.toRat : ℝ) by ring] at hadd
    exact hadd
  rw [abs_of_pos (Real.exp_pos _)] at h1
  unfold eHat
  linarith

/-- `errE` is nonnegative for a nonnegative error. -/
private lemma errE_nonneg' {x δ : ℝ} (hδ : 0 ≤ δ) : 0 ≤ errE x δ := by
  unfold errE expRel
  have h1 : Real.exp x ≤ Real.exp (x + δ) := Real.exp_le_exp.mpr (by linarith)
  have h2 : 0 ≤ (2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|x| + δ) := by positivity
  have h3 : Real.exp (x + δ) ≤ (1 + ((2:ℝ)^(-22:ℤ) + (2:ℝ)^(-23:ℤ) * (|x| + δ))) *
      Real.exp (x + δ) := by nlinarith [Real.exp_pos (x + δ)]
  have h4 : (0:ℝ) ≤ (2:ℝ)^(-140:ℤ) := by positivity
  linarith

/-- The exponent's error is nonnegative. -/
private lemma dErrSt_nonneg (m₀ : Mem) (d : Inst) (t h : Nat) : 0 ≤ dErrSt m₀ d t h := by
  unfold dErrSt etaF uF
  split_ifs <;> positivity

/-- `lOff` is a row of the physical chunk. -/
private lemma lOff_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) : lOff d t < d.cs := by
  have h1 := lTok_le hd ht
  have h2 : t / d.cs * d.cs + t % d.cs = t := by rw [Nat.mul_comm]; exact Nat.div_add_mod t d.cs
  have h3 : t % d.cs < d.cs := Nat.mod_lt _ (cs_pos hd)
  unfold lOff chk
  omega

/-- A token lies in a physical chunk of the launch. -/
private lemma chk_lt_nc {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) : chk d t < nc d := by
  have hcs := cs_pos hd
  unfold chk nc
  rw [Nat.div_lt_iff_lt_mul hcs]
  have h1 := Nat.div_add_mod (d.T + d.cs - 1) d.cs
  have h2 : (d.T + d.cs - 1) % d.cs < d.cs := Nat.mod_lt _ hcs
  rw [Nat.mul_comm] at h1
  omega

/-- A `dA_cumsum` read of the launch is finite and nonpositive; `dAv` is its value. -/
private lemma dA_read {m₀ : Mem} {d : Inst} (hV : Vals m₀ d) {h c i : Nat} (hh : h < d.H)
    (hc : c < nc d) (hi : i < d.cs) {a : F32}
    (ha : m₀.read .f32 (d.dA.ptr (dtIdx d h c i : Int)).addr = some a) :
    a.isFinite = true ∧ a.toRat ≤ 0 ∧ dAv m₀ d h c i = (a.toRat : ℝ) := by
  obtain ⟨v, hv, hf, hn⟩ := hV.2.2.2.2.2.2.1 h c i hh hc hi
  rw [ha] at hv
  cases hv
  exact ⟨hf, hn, by simp only [dAv, valF]; rw [ha]⟩

/-- `float32` subtraction of `+0` is exact. -/
private lemma sub_zf32 {a : F32} (ha : a.isFinite = true) :
    (FloatV.sub a zf32).isFinite = true ∧ (FloatV.sub a zf32).toRat = a.toRat := by
  have hz : (zf32 : F32).pat = 0 := by rw [zf32_eq]; exact FloatV.pat_ofPat (by decide)
  have hzfin : Format.binary32.isFinite 0 = true := Format.isFinite_zero Format.binary32_regular
  have hzr : Format.binary32.toRat 0 = 0 := F32_zero_toRat
  have hsign : Format.binary32.sign 0 = false := by decide
  have h := Format.sub_of_finite (f := Format.binary32) .rn (x := a.pat) (y := 0) ha hzfin
  have hsub : (FloatV.sub a zf32) = FloatV.ofPat (Format.binary32.sub .rn a.pat 0) := by
    unfold FloatV.sub; rw [hz]
  have hw := Format.roundRat_of_wider (f := Format.binary32) (g := Format.binary32)
    Format.binary32_regular Format.binary32_regular ⟨le_rfl, le_rfl, le_rfl⟩ .rn false ha
  rw [hsub]
  rw [h, hsign, hzr, sub_zero]
  simp only [Bool.not_false, Bool.and_true]
  have hq : |Format.binary32.toRat a.pat| ≤ Format.binary32.maxVal :=
    Format.abs_toRat_le_maxVal Format.binary32_regular ha
  obtain ⟨hlt, -, -⟩ := Format.roundRat_rn_err Format.binary32_regular false
    (Format.binary32.sign a.pat) hq
  exact ⟨by rw [FloatV.isFinite_ofPat hlt]; exact hw.1,
    by rw [FloatV.toRat_ofPat hlt]; exact hw.2⟩

/-! ## The state term (lines 337–361) -/

/-- The pointer to element `n` of channel `p` of the state the kernel reads out for token `t`:
the request's initial state when the logical chunk starts it (with initial states), else the
chunk's entering state. -/
def stSrc (d : Inst) (t h p n : Nat) : Ptr .f32 :=
  (stBuf d (d.hasInit && fromInit d t)).ptr
    ((stBase d (d.hasInit && fromInit d t) (chk d t) (reqOf d t) h + p * d.N + n : Nat) : Int)

/-- **The computed scale of the entering state** (lines 337–344) is a finite `float32` close to
`exp(dSt)`: within `eSt`, of magnitude at most `hSt`; and exactly `0` on a `gated` row. -/
def ScaleOK (m₀ : Mem) (d : Inst) (t h : Nat) (e : F32) : Prop :=
  e.isFinite = true ∧ (gated d t = true → e.toRat = 0) ∧
    (gated d t = false →
      |(e.toRat : ℝ) - Real.exp (dSt m₀ d t h)| ≤ eSt m₀ d t h ∧ |(e.toRat : ℝ)| ≤ hSt m₀ d t h)

/-- **Scale, with initial states** (line 344): `e = exp(fl(dA_cs_m − boundary))`. `a` is
`dA_cumsum` at the token's row, `b` the boundary: `dA_cumsum` one row before the logical chunk's
first row, or `+0` when that row is `0`. -/
theorem scale_init {P : Platform} (hexp : Device.ExpF32 P) {m₀ : Mem} {d : Inst} (hd : d.ok)
    (hV : Vals m₀ d) (hHas : d.hasInit = true) {t h p : Nat} (ht : t < d.T) (hh : h < d.H)
    (hp : p < d.P) {a b xt e : F32}
    (ha : m₀.read .f32 (d.dA.ptr (dtIdx d h (chk d t) (t % d.cs) : Int)).addr = some a)
    (hb : (0 < lOff d t ∧
        m₀.read .f32 (d.dA.ptr (dtIdx d h (chk d t) (lOff d t - 1) : Int)).addr = some b) ∨
      (lOff d t = 0 ∧ b = zf32))
    (hsub : FloatV.NaNEq (FloatV.sub a b) xt) (hex : P.math .exp .binary32 xt e) :
    ScaleOK m₀ d t h e := by
  have hc := chk_lt_nc hd ht
  obtain ⟨haf, han, haD⟩ := dA_read hV hh hc (Nat.mod_lt _ (cs_pos hd)) ha
  have hD64 := (hV.2.2.2.2.2.2.2.2 t h p ht hh hp).1
  have hmaxa : |a.toRat| ≤ Format.binary32.maxVal :=
    FloatV.abs_toRat_le_maxVal Format.binary32_regular haf
  have hgated : gated d t = false := by simp [gated, hHas]
  suffices hmain : e.isFinite = true ∧
      |(e.toRat : ℝ) - Real.exp (dSt m₀ d t h)| ≤ eSt m₀ d t h ∧
      |(e.toRat : ℝ)| ≤ hSt m₀ d t h from
    ⟨hmain.1, fun hg => absurd hg (by simp [hgated]), fun _ => hmain.2⟩
  rcases hb with ⟨hl, hb⟩ | ⟨hl, rfl⟩
  · obtain ⟨hbf, hbn, hbD⟩ := dA_read hV hh hc (by have := lOff_lt hd ht; omega) hb
    have hDeq : dSt m₀ d t h = (a.toRat : ℝ) - (b.toRat : ℝ) := by
      simp only [dSt, hl, ↓reduceIte, haD, hbD]
    have hmaxb : |b.toRat| ≤ Format.binary32.maxVal :=
      FloatV.abs_toRat_le_maxVal Format.binary32_regular hbf
    have hsm : |a.toRat - b.toRat| ≤ Format.binary32.maxVal := by
      rw [abs_le] at hmaxa hmaxb ⊢
      constructor <;> linarith [hmaxa.1, hmaxa.2, hmaxb.1, hmaxb.2]
    have herr := FloatV.sub_err Format.binary32_regular haf hbf hsm
    have hxt : xt = FloatV.sub a b := (FloatV.NaNEq.eq_of_isFinite hsub herr.1)
    have hxerr := sub32_real herr.2
    have hdE : dErrSt m₀ d t h = |dSt m₀ d t h| * (2:ℝ)^(-24:ℤ) + (2:ℝ)^(-150:ℤ) := by
      simp only [dErrSt, hl, ↓reduceIte, uF, etaF]; ring
    have hDm : |dSt m₀ d t h| ≤ (Format.binary32.maxVal : ℝ) := by
      rw [hDeq]
      have := real_abs_le_maxVal hsm
      push_cast at this
      exact this
    have := scale_core hexp (xt := xt) (e := e) hD64 hDm (dErrSt_nonneg m₀ d t h) hdE.le
      (by rw [hxt]; exact herr.1) (by rw [hxt, hdE, hDeq]; simpa only [FloatV.toReal] using hxerr) hex
    exact this
  · have hDeq : dSt m₀ d t h = (a.toRat : ℝ) := by
      simp only [dSt, hl, lt_self_iff_false, ↓reduceIte, haD, sub_zero]
    have hdE : dErrSt m₀ d t h = 0 := by simp [dErrSt, hl]
    obtain ⟨hf0, hv0⟩ := sub_zf32 haf
    have hxt : xt = FloatV.sub a zf32 := (FloatV.NaNEq.eq_of_isFinite hsub hf0)
    have hDm : |dSt m₀ d t h| ≤ (Format.binary32.maxVal : ℝ) := by
      rw [hDeq]; exact real_abs_le_maxVal hmaxa
    have := scale_core hexp (xt := xt) (e := e) hD64 hDm (dErrSt_nonneg m₀ d t h) (by
        rw [hdE]; positivity) (by rw [hxt]; exact hf0)
      (by rw [hdE, hxt, hDeq, hv0]; simp) hex
    exact this

/-- **Scale, without initial states** (lines 337–340): `e = where(fromInit-free, exp(dA_cs_m), 0)`;
`e0` is the platform's `exp` of the loaded `dA_cumsum` entry. -/
theorem scale_noinit {P : Platform} (hexp : Device.ExpF32 P) {m₀ : Mem} {d : Inst} (hd : d.ok)
    (hV : Vals m₀ d) (hHas : d.hasInit = false) {t h p : Nat} (ht : t < d.T) (hh : h < d.H)
    (hp : p < d.P) {a e0 e : F32}
    (ha : m₀.read .f32 (d.dA.ptr (dtIdx d h (chk d t) (t % d.cs) : Int)).addr = some a)
    (hex : P.math .exp .binary32 a e0) (he : e = if fromInit d t = true then zf32 else e0) :
    ScaleOK m₀ d t h e := by
  have hc := chk_lt_nc hd ht
  obtain ⟨haf, han, haD⟩ := dA_read hV hh hc (Nat.mod_lt _ (cs_pos hd)) ha
  have hD64 := (hV.2.2.2.2.2.2.2.2 t h p ht hh hp).1
  have hgated : gated d t = fromInit d t := by simp [gated, hHas]
  by_cases hfi : fromInit d t = true
  · have he' : e = zf32 := by rw [he]; simp [hfi]
    exact ⟨by rw [he']; exact zf32_finite, fun _ => by rw [he']; exact zf32_toRat,
      fun hg => absurd (hgated ▸ hfi) (by simp [hg])⟩
  · have hfi' : fromInit d t = false := by simpa using hfi
    have he' : e = e0 := by rw [he]; simp [hfi]
    have hl := lOff_of_not_fromInit hd ht hfi'
    have hDeq : dSt m₀ d t h = (a.toRat : ℝ) := by
      simp only [dSt, hl, lt_self_iff_false, ↓reduceIte, haD, sub_zero]
    have hdE : dErrSt m₀ d t h = 0 := by simp [dErrSt, hl]
    have hmaxa : |a.toRat| ≤ Format.binary32.maxVal :=
      FloatV.abs_toRat_le_maxVal Format.binary32_regular haf
    have hDm : |dSt m₀ d t h| ≤ (Format.binary32.maxVal : ℝ) := by
      rw [hDeq]; exact real_abs_le_maxVal hmaxa
    have := scale_core hexp (xt := a) (e := e0) hD64 hDm (dErrSt_nonneg m₀ d t h) (by
        rw [hdE]; positivity) haf (by rw [hdE, hDeq]; simp) hex
    rw [he']
    exact ⟨this.1, fun hg => absurd (hgated ▸ hg) (by simp [hfi']), fun _ => this.2⟩

/-! ## The dot of the state term -/

/-- A sum over `Fin K` of terms that vanish past `N ≤ K` is the sum over `range N`. -/
private lemma sum_pad {M : Type*} [AddCommMonoid M] {N K : Nat} (hNK : N ≤ K) (φ : Nat → M) :
    ∑ k : Fin K, (if (k : Nat) < N then φ k else 0) = ∑ n ∈ Finset.range N, φ n := by
  rw [Fin.sum_univ_eq_sum_range (fun n => if n < N then φ n else 0) K, ← Finset.sum_filter]
  congr 1
  ext n
  simp only [Finset.mem_filter, Finset.mem_range]
  omega

/-- A `C` element the kernel loaded is finite and is `Cv`. -/
private lemma C_lane {m₀ : Mem} {d : Inst} (hd : d.ok) (hV : Vals m₀ d) {t h n : Nat}
    (ht : t < d.T) (hh : h < d.H) (hn : n < d.N) {x : BF16}
    (hx : m₀.read .bf16 (d.C.ptr (cIdx d t (grp d h) n : Int)).addr = some x) :
    x.isFinite = true ∧ (x.toRat : ℝ) = Cv m₀ d t (grp d h) n := by
  obtain ⟨v, hv, hf⟩ := hV.1 t (grp d h) n ht (grp_lt hd hh) hn
  simp only [rdB] at hv
  rw [hx] at hv
  cases hv
  refine ⟨hf, ?_⟩
  simp only [Cv, valB, rdB]
  rw [hx]

/-- A state element the kernel read out is finite, at most `2^120`, and is `SvL`. -/
private lemma S_lane {m₀ : Mem} {d : Inst} (hd : d.ok) (hV : Vals m₀ d) {t h p n : Nat}
    (ht : t < d.T) (hh : h < d.H) (hp : p < d.P) (hn : n < d.N) {s : F32}
    (hs : m₀.read .f32 (stSrc d t h p n).addr = some s) :
    s.isFinite = true ∧ |s.toRat| ≤ 2 ^ 120 ∧ (s.toRat : ℝ) = SvL m₀ d t h p n := by
  by_cases hu : d.hasInit = true ∧ fromInit d t = true
  · obtain ⟨hI, hF⟩ := hu
    have hg : gated d t = false := by simp [gated, hI]
    have hsrc : stSrc d t h p n = d.initst.ptr (inIdx d (reqOf d t) h p n : Int) := by
      simp only [stSrc, hI, hF, Bool.and_self, stBuf, stBase, ite_true, inIdx]
    rw [hsrc] at hs
    obtain ⟨v, hv, hf, hb⟩ :=
      hV.2.2.2.2.1 hI (reqOf d t) h p n (reqOf_spec hd ht).1 hh hp hn
    rw [hs] at hv
    cases hv
    refine ⟨hf, hb, ?_⟩
    simp only [SvL, Sv, hg, hF, hI, ite_true, Bool.false_eq_true, ite_false, valF]
    rw [hs]
  · have hc := chk_lt_nc hd ht
    have hsrc : stSrc d t h p n = d.states.ptr (stIdx d (chk d t) h p n : Int) := by
      have : (d.hasInit && fromInit d t) = false := by
        cases hI : d.hasInit <;> cases hF : fromInit d t <;> simp_all
      simp only [stSrc, this, stBuf, stBase, stIdx, Bool.false_eq_true, ite_false]
    rw [hsrc] at hs
    obtain ⟨v, hv, hf, hb⟩ := hV.2.2.2.1 (chk d t) h p n hc hh hp hn
    rw [hs] at hv
    cases hv
    refine ⟨hf, hb, ?_⟩
    have hSv : SvL m₀ d t h p n = (valF m₀ d.states (stIdx d (chk d t) h p n) : ℝ) := by
      unfold SvL Sv gated
      by_cases hF : fromInit d t = true
      · have hI : d.hasInit = false := by
          by_contra hcon
          exact hu ⟨by simpa using hcon, hF⟩
        simp [hF, hI]
      · simp [hF]
    rw [hSv]
    simp only [valF]
    rw [hs]

/-- A `bfloat16` rounding of a state element: finite, within `uB·|s| + etaB` of it. -/
private lemma bf_lane {s : F32} {x : BF16} (hs : s.isFinite = true) (h120 : |s.toRat| ≤ 2 ^ 120)
    (hx : FloatV.NaNEq (FloatV.cvtWith .rn s) x) :
    x.isFinite = true ∧ |(x.toRat : ℝ) - (s.toRat : ℝ)| ≤ uB * |(s.toRat : ℝ)| + etaB := by
  have hmax : |s.toRat| ≤ Format.bfloat16.maxVal := by
    refine rat_abs_le_bf (C := 2 ^ 120) ?_ pow120_le_maxValbf
    have : ((|s.toRat| : ℚ) : ℝ) ≤ ((2 ^ 120 : ℚ) : ℝ) := by exact_mod_cast h120
    rw [Rat.cast_abs] at this
    simpa using this
  have herr := FloatV.cvt_err (g := Format.bfloat16) Format.bfloat16_regular hs hmax
  have hxe : x = FloatV.cvtWith .rn s := FloatV.NaNEq.eq_of_isFinite hx herr.1
  refine ⟨by rw [hxe]; exact herr.1, ?_⟩
  have := cvt32bf_real (x := s) herr.2
  rw [hxe]
  unfold uB etaB
  simp only [FloatV.toReal] at this
  linarith

/-- `Device.relDot16`, as a real, is nonnegative. -/
private lemma relD_nonneg (K : Nat) : 0 ≤ relD K := by
  unfold relD Device.relDot16; push_cast; positivity

/-- `Device.absDot16`, as a real, is nonnegative. -/
private lemma absD_nonneg (K : Nat) : 0 ≤ absD K := by
  unfold absD Device.absDot16; push_cast; positivity

/-- **The dot of line 361.** Its result is finite (with the room `lSigL`), and on a row that is
not `gated` it is within `eSig` of `sig`. -/
private theorem state_dot {P : Platform} (hdot : Device.DotBF16 P) {m₀ : Mem} {d : Inst}
    (hd : d.ok) (hV : Vals m₀ d) {t h p : Nat} (ht : t < d.T) (hh : h < d.H) (hp : p < d.P)
    {a : Tensor [BM, d.BSD] BF16} {b : Tensor [d.BSD, BN] BF16} {r : Tensor [BM, BN] F32}
    {ii : Fin BM} {jj : Fin BN}
    (hr : P.dot cfgBF16 a b (tl.zeros [BM, BN] .f32) r)
    (hA : ∀ n : Fin d.BSD,
      (n.val < d.N → m₀.read .bf16 (d.C.ptr (cIdx d t (grp d h) n : Int)).addr = some (a.get2 ii n)) ∧
      (¬ n.val < d.N → a.get2 ii n = FloatV.ofPat 0))
    (hB : ∀ n : Fin d.BSD, ∃ s : F32, FloatV.NaNEq (FloatV.cvtWith .rn s) (b.get2 n jj) ∧
      ((n.val < d.N → m₀.read .f32 (stSrc d t h p n).addr = some s) ∧
       (¬ n.val < d.N → s = zf32))) :
    (r.get2 ii jj).isFinite = true ∧
      (gated d t = false →
        |((r.get2 ii jj).toRat : ℝ) - sig m₀ d t h p| ≤ eSig m₀ d t h p) := by
  have hNB : d.N ≤ d.BSD := hd.N_le
  choose s hs1 hs2 hs3 using hB
  -- the `C` lanes
  have hCfin : ∀ n : Fin d.BSD, (a.get2 ii n).isFinite = true := fun n => by
    by_cases hn : n.val < d.N
    · exact (C_lane hd hV ht hh hn ((hA n).1 hn)).1
    · rw [(hA n).2 hn]; exact BF16_zero_finite
  have hCval : ∀ n : Fin d.BSD, ((a.get2 ii n).toRat : ℝ) =
      if n.val < d.N then Cv m₀ d t (grp d h) n else 0 := fun n => by
    by_cases hn : n.val < d.N
    · simp only [hn, ↓reduceIte]; exact (C_lane hd hV ht hh hn ((hA n).1 hn)).2
    · simp only [hn, ↓reduceIte]; rw [(hA n).2 hn]; simp
  -- the state lanes
  have hbf : ∀ n : Fin d.BSD, (b.get2 n jj).isFinite = true ∧
      (n.val < d.N → |((b.get2 n jj).toRat : ℝ) - SvL m₀ d t h p n| ≤
        uB * |SvL m₀ d t h p n| + etaB) ∧
      (¬ n.val < d.N → (b.get2 n jj).toRat = 0) := fun n => by
    by_cases hn : n.val < d.N
    · obtain ⟨hsf, hs120, hsv⟩ := S_lane hd hV ht hh hp hn (hs2 n hn)
      obtain ⟨hf, he⟩ := bf_lane hsf hs120 (hs1 n)
      refine ⟨hf, fun _ => ?_, fun h' => absurd hn h'⟩
      rw [← hsv]; exact he
    · have hz := hs3 n hn
      have hc : FloatV.cvtWith .rn (s n) = (FloatV.ofPat 0 : BF16) := by
        rw [hz, zf32_eq]; exact cvt_f32_bf16_zero
      have hfin : (FloatV.cvtWith .rn (s n) : BF16).isFinite = true := by
        rw [hc]; exact BF16_zero_finite
      have heq : b.get2 n jj = (FloatV.ofPat 0 : BF16) := by
        rw [← hc]; exact FloatV.NaNEq.eq_of_isFinite (hs1 n) hfin
      refine ⟨by rw [heq]; exact BF16_zero_finite, fun h' => absurd h' hn, fun _ => ?_⟩
      rw [heq]; exact BF16_zero_toRat
  -- the values handed to the dot
  let va : Fin d.BSD → ℚ := fun n => (a.get2 ii n).toRat
  let vb : Fin d.BSD → ℚ := fun n => (b.get2 n jj).toRat
  have hva : ∀ n, DType.val? .bf16 (a.get2 ii n) = some (va n) := fun n =>
    DType.val?_bf16.mpr ⟨hCfin n, rfl⟩
  have hvb : ∀ n, DType.val? .bf16 (b.get2 n jj) = some (vb n) := fun n =>
    DType.val?_bf16.mpr ⟨(hbf n).1, rfl⟩
  have hvc : DType.val? .f32 ((tl.zeros [BM, BN] .f32).get2 ii jj) = some 0 :=
    DType.val?_zero_f32
  -- the lifted magnitude
  have hCr : ∀ n : Fin d.BSD, (va n : ℝ) = if n.val < d.N then Cv m₀ d t (grp d h) n else 0 :=
    hCval
  have hlift : ∀ n : Fin d.BSD, ((Device.liftMag .bf16 (va n) : ℚ) : ℝ) *
      ((Device.liftMag .bf16 (vb n) : ℚ) : ℝ) ≤
      if n.val < d.N then (|Cv m₀ d t (grp d h) n| + mB) *
        ((1 + uB) * |SvL m₀ d t h p n| + etaB + mB) else 0 := fun n => by
    by_cases hn : n.val < d.N
    · simp only [hn, ↓reduceIte]
      have hCn : (va n : ℝ) = Cv m₀ d t (grp d h) n := by rw [hCr n]; simp only [hn, ↓reduceIte]
      have h1 := Device.liftMag_le .bf16 (va n)
      have h2 := Device.liftMag_le .bf16 (vb n)
      rw [Device.liftFloor_bf16] at h1 h2
      have h1' : ((Device.liftMag .bf16 (va n) : ℚ) : ℝ) ≤ |Cv m₀ d t (grp d h) n| + mB := by
        have h3 : ((Device.liftMag .bf16 (va n) : ℚ) : ℝ) ≤ ((|va n| + 2 ^ (-126 : ℤ) : ℚ) : ℝ) := by
          exact_mod_cast h1
        push_cast at h3
        rw [hCn] at h3
        exact h3
      have h2' : ((Device.liftMag .bf16 (vb n) : ℚ) : ℝ) ≤
          (1 + uB) * |SvL m₀ d t h p n| + etaB + mB := by
        have h3 : ((Device.liftMag .bf16 (vb n) : ℚ) : ℝ) ≤ ((|vb n| + 2 ^ (-126 : ℤ) : ℚ) : ℝ) := by
          exact_mod_cast h2
        push_cast at h3
        have hv := (hbf n).2.1 hn
        have hvb' : |(vb n : ℝ)| ≤ (1 + uB) * |SvL m₀ d t h p n| + etaB := by
          have h4 : |(vb n : ℝ)| ≤ |SvL m₀ d t h p n| + |(vb n : ℝ) - SvL m₀ d t h p n| := by
            have := abs_add_le (SvL m₀ d t h p n) ((vb n : ℝ) - SvL m₀ d t h p n)
            rwa [show SvL m₀ d t h p n + ((vb n : ℝ) - SvL m₀ d t h p n) = (vb n : ℝ) by ring] at this
          have hv' : |(vb n : ℝ) - SvL m₀ d t h p n| ≤ uB * |SvL m₀ d t h p n| + etaB := hv
          linarith
        have hmB : mB = (2:ℝ) ^ (-126 : ℤ) := rfl
        rw [hmB]
        linarith
      exact mul_le_mul h1' h2' (by exact_mod_cast Device.liftMag_nonneg _ _)
        (add_nonneg (abs_nonneg _) (by unfold mB; positivity))
    · simp only [hn, ↓reduceIte]
      have : va n = 0 := by
        have h0 : ((va n : ℚ) : ℝ) = 0 := by rw [hCr n]; simp only [hn, ↓reduceIte]
        exact_mod_cast h0
      rw [this, Device.liftMag_zero]; simp
  -- the sum of lifted products is at most `lSigL`
  have hL : ((∑ n, Device.liftMag .bf16 (va n) * Device.liftMag .bf16 (vb n) : ℚ) : ℝ) ≤
      lSigL m₀ d t h p := by
    push_cast
    calc _ ≤ ∑ n : Fin d.BSD, (if n.val < d.N then (|Cv m₀ d t (grp d h) n| + mB) *
            ((1 + uB) * |SvL m₀ d t h p n| + etaB + mB) else 0) :=
          Finset.sum_le_sum fun n _ => hlift n
      _ = _ := sum_pad hNB (fun n => (|Cv m₀ d t (grp d h) n| + mB) *
            ((1 + uB) * |SvL m₀ d t h p n| + etaB + mB))
  have hLmax := (hV.2.2.2.2.2.2.2.2 t h p ht hh hp).2.2.1
  have hLQ : (∑ n, Device.liftMag .bf16 (va n) * Device.liftMag .bf16 (vb n) : ℚ) ≤ 2 ^ 120 := by
    have : ((∑ n, Device.liftMag .bf16 (va n) * Device.liftMag .bf16 (vb n) : ℚ) : ℝ) ≤ 2 ^ 120 :=
      hL.trans hLmax
    exact_mod_cast this
  have hLQ0 : 0 ≤ (∑ n, Device.liftMag .bf16 (va n) * Device.liftMag .bf16 (vb n) : ℚ) :=
    Finset.sum_nonneg fun n _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hBSD : (d.BSD : ℚ) ≤ 128 := by exact_mod_cast hd.bsd_le
  have hrel : Device.relDot16 d.BSD ≤ 1 := by
    unfold Device.relDot16
    calc ((d.BSD : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ (128 + 2) * 2 ^ (-22 : ℤ) := by gcongr
      _ ≤ 1 := by norm_num
  have habs : Device.absDot16 d.BSD ≤ 1 := by
    unfold Device.absDot16
    calc ((d.BSD : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ (128 + 2) * 2 ^ (-126 : ℤ) := by gcongr
      _ ≤ 1 := by norm_num
  have hmax : (2 : ℚ) ^ 126 ≤ Format.binary32.maxVal := by
    have := pow126_le_maxVal32
    exact_mod_cast this
  have hroom : |(0 : ℚ)| + ∑ n, Device.liftMag .bf16 (va n) * Device.liftMag .bf16 (vb n) +
      (Device.relDot16 d.BSD * (|(0 : ℚ)| + ∑ n, Device.liftMag .bf16 (va n) *
        Device.liftMag .bf16 (vb n)) + Device.absDot16 d.BSD) ≤ DType.bound .f32 := by
    have h1 : Device.relDot16 d.BSD * (|(0 : ℚ)| + ∑ n, Device.liftMag .bf16 (va n) *
        Device.liftMag .bf16 (vb n)) ≤ 1 * 2 ^ 120 := by
      rw [abs_zero, zero_add]
      exact mul_le_mul hrel hLQ hLQ0 (by norm_num)
    have h2 : (2 : ℚ) ^ 126 = 2 ^ 120 * 64 := by norm_num
    rw [abs_zero, zero_add] at h1 ⊢
    show _ ≤ Format.binary32.maxVal
    linarith
  obtain ⟨vr, hvr, hbd⟩ := hdot .tf32 a b (tl.zeros [BM, BN] .f32) r hr ii jj va vb 0 hva hvb hvc hroom
  obtain ⟨hrf, hrv⟩ := DType.val?_f32.mp hvr
  refine ⟨hrf, fun hg => ?_⟩
  generalize (∑ n, Device.liftMag .bf16 (va n) * Device.liftMag .bf16 (vb n) : ℚ) = LQ at hL hbd
  have hSvL : ∀ n : Nat, SvL m₀ d t h p n = Sv m₀ d t h p n := fun n => by simp [SvL, hg]
  have hlS : lSigL m₀ d t h p = lSig m₀ d t h p := by simp only [lSigL, lSig, hSvL]
  have hres : |(vr : ℝ) - ∑ n, (va n : ℝ) * (vb n : ℝ)| ≤ relD d.BSD * (LQ : ℝ) + absD d.BSD := by
    have h' : ((|vr - (0 + ∑ n, va n * vb n)| : ℚ) : ℝ) ≤
        ((Device.relDot16 d.BSD * (|(0 : ℚ)| + LQ) + Device.absDot16 d.BSD : ℚ) : ℝ) := by
      exact_mod_cast hbd
    push_cast at h'
    simpa [relD, absD] using h'
  have hpad1 : ∑ n : Fin d.BSD, (va n : ℝ) * (vb n : ℝ) =
      ∑ n : Fin d.BSD, (if n.val < d.N then Cv m₀ d t (grp d h) n * (vb n : ℝ) else 0) :=
    Finset.sum_congr rfl fun n _ => by rw [hCr n]; split_ifs <;> simp
  have hpad2 : sig m₀ d t h p = ∑ n : Fin d.BSD,
      (if n.val < d.N then Cv m₀ d t (grp d h) n * Sv m₀ d t h p n else 0) :=
    (sum_pad hNB (fun n => Cv m₀ d t (grp d h) n * Sv m₀ d t h p n)).symm
  have herr : |∑ n, (va n : ℝ) * (vb n : ℝ) - sig m₀ d t h p| ≤
      uB * mSig m₀ d t h p + etaB * cSum m₀ d t h := by
    rw [hpad1, hpad2, ← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    refine (Finset.sum_le_sum (g := fun n : Fin d.BSD => if n.val < d.N then
      uB * (|Cv m₀ d t (grp d h) n| * |Sv m₀ d t h p n|) + etaB * |Cv m₀ d t (grp d h) n|
      else 0) fun n _ => ?_).trans ?_
    · by_cases hn : n.val < d.N
      · simp only [hn, ↓reduceIte]
        have hv := (hbf n).2.1 hn
        rw [hSvL] at hv
        have hv' : |(vb n : ℝ) - Sv m₀ d t h p n| ≤ uB * |Sv m₀ d t h p n| + etaB := hv
        rw [← mul_sub, abs_mul]
        calc |Cv m₀ d t (grp d h) n| * |(vb n : ℝ) - Sv m₀ d t h p n|
            ≤ |Cv m₀ d t (grp d h) n| * (uB * |Sv m₀ d t h p n| + etaB) :=
              mul_le_mul_of_nonneg_left hv' (abs_nonneg _)
          _ = _ := by ring
      · simp [hn]
    · rw [sum_pad hNB (fun n => uB * (|Cv m₀ d t (grp d h) n| * |Sv m₀ d t h p n|) +
        etaB * |Cv m₀ d t (grp d h) n|)]
      simp only [mSig, cSum, Finset.sum_add_distrib, Finset.mul_sum]
      exact le_rfl
  have hrR : ((r.get2 ii jj).toRat : ℝ) = (vr : ℝ) := by rw [hrv]
  have hLl : relD d.BSD * (LQ : ℝ) ≤ relD d.BSD * lSig m₀ d t h p :=
    mul_le_mul_of_nonneg_left (hL.trans hlS.le) (relD_nonneg _)
  rw [hrR]
  unfold eSig
  have htri : |(vr : ℝ) - sig m₀ d t h p| ≤
      |(vr : ℝ) - ∑ n, (va n : ℝ) * (vb n : ℝ)| +
        |∑ n, (va n : ℝ) * (vb n : ℝ) - sig m₀ d t h p| := by
    have := abs_add_le ((vr : ℝ) - ∑ n, (va n : ℝ) * (vb n : ℝ))
      (∑ n, (va n : ℝ) * (vb n : ℝ) - sig m₀ d t h p)
    rwa [show (vr : ℝ) - ∑ n, (va n : ℝ) * (vb n : ℝ) +
      (∑ n, (va n : ℝ) * (vb n : ℝ) - sig m₀ d t h p) = (vr : ℝ) - sig m₀ d t h p by ring] at this
  linarith

/-- Multiplying a finite `float32` by a finite zero gives a finite zero. -/
private lemma mul_zero_toRat {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (h0 : y.toRat = 0) :
    (FloatV.mul x y).isFinite = true ∧ (FloatV.mul x y).toRat = 0 := by
  have hy0 : Format.binary32.toRat y.pat = 0 := h0
  have hmul : Format.binary32.mul .rn x.pat y.pat = Format.binary32.roundRat .rn false
      (Format.binary32.sign x.pat != Format.binary32.sign y.pat) 0 := by
    rw [Format.mul_of_finite (rnd := .rn) hx hy, hy0, mul_zero]
  obtain ⟨hlt, hfin, hto⟩ := Format.roundRat_zero_spec Format.binary32_regular (rnd := .rn)
    (sat := false) (zs := Format.binary32.sign x.pat != Format.binary32.sign y.pat)
  unfold FloatV.mul
  rw [hmul]
  exact ⟨by rw [FloatV.isFinite_ofPat hlt]; exact hfin,
    by rw [FloatV.toRat_ofPat hlt]; exact hto⟩

/-- **The state term** `acc₀ = tl.dot(C, bf16(S)) * scale` (lines 355–361), at tile lane
`(ii, jj)` for token `t`, head `h`, channel `p`: finite, within `errSig` of `exp(dSt)·Σ_n C·S`,
of magnitude at most `aSig`. `a` is the `[BM, BSD]` tile of `C`, `b` the `bfloat16` tile of the
state, `r` the dot's result and `acc0` its product with the scale `e`. -/
theorem accA_fact {P : Platform} (hdot : Device.DotBF16 P) {m₀ : Mem} {d : Inst} (hd : d.ok)
    (hV : Vals m₀ d) {t h p : Nat} (ht : t < d.T) (hh : h < d.H) (hp : p < d.P)
    {a : Tensor [BM, d.BSD] BF16} {b : Tensor [d.BSD, BN] BF16} {r acc0 : Tensor [BM, BN] F32}
    {ii : Fin BM} {jj : Fin BN} {e : F32} (he : ScaleOK m₀ d t h e)
    (hr : P.dot cfgBF16 a b (tl.zeros [BM, BN] .f32) r)
    (hA : ∀ n : Fin d.BSD,
      (n.val < d.N → m₀.read .bf16 (d.C.ptr (cIdx d t (grp d h) n : Int)).addr = some (a.get2 ii n)) ∧
      (¬ n.val < d.N → a.get2 ii n = FloatV.ofPat 0))
    (hB : ∀ n : Fin d.BSD, ∃ s : F32, FloatV.NaNEq (FloatV.cvtWith .rn s) (b.get2 n jj) ∧
      ((n.val < d.N → m₀.read .f32 (stSrc d t h p n).addr = some s) ∧
       (¬ n.val < d.N → s = zf32)))
    (hm : FloatV.NaNEq (FloatV.mul (r.get2 ii jj) e) (acc0.get2 ii jj)) :
    (acc0.get2 ii jj).isFinite = true ∧
      |((acc0.get2 ii jj).toRat : ℝ) - Real.exp (dSt m₀ d t h) * sig m₀ d t h p| ≤
        errSig m₀ d t h p ∧
      |((acc0.get2 ii jj).toRat : ℝ)| ≤ aSig m₀ d t h p := by
  obtain ⟨hrf, hrerr⟩ := state_dot hdot hd hV ht hh hp hr hA hB
  obtain ⟨hef, hegat, hene⟩ := he
  obtain ⟨-, -, -, hHM, -⟩ := hV.2.2.2.2.2.2.2.2 t h p ht hh hp
  -- nonnegativity of the error terms
  have hδ0 := dErrSt_nonneg m₀ d t h
  have heSt0 : 0 ≤ eSt m₀ d t h := errE_nonneg' hδ0
  have hhSt0 : 0 ≤ hSt m₀ d t h := by
    unfold hSt eHat; exact add_nonneg (Real.exp_pos _).le (errE_nonneg' hδ0)
  have hm0 : 0 ≤ mSig m₀ d t h p :=
    Finset.sum_nonneg fun n _ => mul_nonneg (abs_nonneg _) (abs_nonneg _)
  have hc0 : 0 ≤ cSum m₀ d t h := Finset.sum_nonneg fun n _ => abs_nonneg _
  have hl0 : 0 ≤ lSig m₀ d t h p :=
    Finset.sum_nonneg fun n _ => mul_nonneg (add_nonneg (abs_nonneg _) (by unfold mB; positivity))
      (add_nonneg (add_nonneg (mul_nonneg (by unfold uB; positivity) (abs_nonneg _))
        (by unfold etaB; positivity)) (by unfold mB; positivity))
  have heS0 : 0 ≤ eSig m₀ d t h p := by
    unfold eSig
    have := relD_nonneg d.BSD
    have := absD_nonneg d.BSD
    have : 0 ≤ uB := by unfold uB; positivity
    have : 0 ≤ etaB := by unfold etaB; positivity
    positivity
  have hu0 : 0 ≤ uF := by unfold uF; positivity
  have hη0 : 0 ≤ etaF := by unfold etaF; positivity
  by_cases hg : gated d t = true
  · -- a `gated` row: the scale is `0`, and so is the reference's state term
    have he0 := hegat hg
    obtain ⟨hf1, h01⟩ := mul_zero_toRat hrf hef he0
    have hacc : acc0.get2 ii jj = FloatV.mul (r.get2 ii jj) e :=
      FloatV.NaNEq.eq_of_isFinite hm hf1
    have hSv0 : ∀ n, Sv m₀ d t h p n = 0 := fun n => by
      have hfi : fromInit d t = true := (by simpa [gated] using hg : _ ∧ _).2
      have hI : d.hasInit = false := (by simpa [gated] using hg : _ ∧ _).1
      simp [Sv, hfi, hI]
    have hsig0 : sig m₀ d t h p = 0 := by simp [sig, hSv0]
    refine ⟨by rw [hacc]; exact hf1, ?_, ?_⟩
    · rw [hacc, h01, hsig0]
      simp only [Rat.cast_zero, mul_zero, sub_zero, abs_zero]
      unfold errSig
      positivity
    · rw [hacc, h01]
      simp only [Rat.cast_zero, abs_zero]
      unfold aSig
      positivity
  · have hg' : gated d t = false := by simpa using hg
    obtain ⟨hse, hhat⟩ := hene hg'
    have hrs := hrerr hg'
    -- magnitudes
    have hsigM : |sig m₀ d t h p| ≤ mSig m₀ d t h p := by
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun n _ => ?_)
      rw [abs_mul]
    have hrM : |((r.get2 ii jj).toRat : ℝ)| ≤ mSig m₀ d t h p + eSig m₀ d t h p := by
      have := abs_add_le (sig m₀ d t h p) (((r.get2 ii jj).toRat : ℝ) - sig m₀ d t h p)
      rw [show sig m₀ d t h p + (((r.get2 ii jj).toRat : ℝ) - sig m₀ d t h p) =
        ((r.get2 ii jj).toRat : ℝ) by ring] at this
      linarith
    have hre : |((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ)| ≤
        (mSig m₀ d t h p + eSig m₀ d t h p) * hSt m₀ d t h := by
      rw [abs_mul]
      exact mul_le_mul hrM hhat (abs_nonneg _) (by linarith)
    have hre120 : |((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ)| ≤ 2 ^ 120 := by
      refine hre.trans ?_
      rw [mul_comm]
      exact hHM
    have hroom : |(r.get2 ii jj).toRat * e.toRat| ≤ Format.binary32.maxVal :=
      rat_abs_mul_le hre120 ((by norm_num : (2:ℝ) ^ 120 ≤ 2 ^ 126).trans pow126_le_maxVal32)
    have hmerr := FloatV.mul_err Format.binary32_regular hrf hef hroom
    have hacc : acc0.get2 ii jj = FloatV.mul (r.get2 ii jj) e :=
      FloatV.NaNEq.eq_of_isFinite hm hmerr.1
    have hmr := mul32_real hmerr.2
    rw [hacc]
    refine ⟨hmerr.1, ?_, ?_⟩
    · simp only [FloatV.toReal] at hmr
      -- split `acc₀ − exp·σ` into the product's rounding, the dot's error and the scale's
      have hsplit : (FloatV.mul (r.get2 ii jj) e).toRat - Real.exp (dSt m₀ d t h) * sig m₀ d t h p =
          (((FloatV.mul (r.get2 ii jj) e).toRat : ℝ) - ((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ)) +
          ((((r.get2 ii jj).toRat : ℝ) - sig m₀ d t h p) * (e.toRat : ℝ) +
            sig m₀ d t h p * ((e.toRat : ℝ) - Real.exp (dSt m₀ d t h))) := by ring
      rw [hsplit]
      have h1 := abs_add_le
        (((FloatV.mul (r.get2 ii jj) e).toRat : ℝ) - ((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ))
        ((((r.get2 ii jj).toRat : ℝ) - sig m₀ d t h p) * (e.toRat : ℝ) +
            sig m₀ d t h p * ((e.toRat : ℝ) - Real.exp (dSt m₀ d t h)))
      have h2 := abs_add_le
        ((((r.get2 ii jj).toRat : ℝ) - sig m₀ d t h p) * (e.toRat : ℝ))
        (sig m₀ d t h p * ((e.toRat : ℝ) - Real.exp (dSt m₀ d t h)))
      have h3 : |(((r.get2 ii jj).toRat : ℝ) - sig m₀ d t h p) * (e.toRat : ℝ)| ≤
          eSig m₀ d t h p * hSt m₀ d t h := by
        rw [abs_mul]; exact mul_le_mul hrs hhat (abs_nonneg _) heS0
      have h4 : |sig m₀ d t h p * ((e.toRat : ℝ) - Real.exp (dSt m₀ d t h))| ≤
          mSig m₀ d t h p * eSt m₀ d t h := by
        rw [abs_mul]; exact mul_le_mul hsigM hse (abs_nonneg _) hm0
      have h5 : |((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ)| * (2:ℝ) ^ (-24:ℤ) ≤
          (mSig m₀ d t h p + eSig m₀ d t h p) * hSt m₀ d t h * uF :=
        mul_le_mul_of_nonneg_right hre (by positivity)
      unfold errSig etaF uF at *
      nlinarith [h1, h2, h3, h4, h5, hmr]
    · simp only [FloatV.toReal] at hmr
      have h1 := abs_add_le (((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ))
        (((FloatV.mul (r.get2 ii jj) e).toRat : ℝ) - ((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ))
      rw [show ((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ) +
        (((FloatV.mul (r.get2 ii jj) e).toRat : ℝ) - ((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ)) =
        ((FloatV.mul (r.get2 ii jj) e).toRat : ℝ) by ring] at h1
      have h5 : |((r.get2 ii jj).toRat : ℝ) * (e.toRat : ℝ)| * (2:ℝ) ^ (-24:ℤ) ≤
          (mSig m₀ d t h p + eSig m₀ d t h p) * hSt m₀ d t h * uF :=
        mul_le_mul_of_nonneg_right hre (by positivity)
      unfold aSig etaF uF at *
      nlinarith [h1, h5, hmr, hre]

end Claims.SsdChunkScan
