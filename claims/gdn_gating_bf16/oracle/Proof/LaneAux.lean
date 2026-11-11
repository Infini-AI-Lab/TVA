import Claim
import TVA.Proof
import Proof.FloatStep
import Proof.NanFree
import Proof.Rnd

/-!
# Numeric bounds for the gating lane

Real-valued estimates for `tl.exp`, `tl.log` and the two products of the lane, together with the
ranges of `float32` and `bfloat16`.  In the nonlinear steps the numeric constants are passed as
fresh real variables, so that the arithmetic tactics only see plain atoms.
-/

set_option maxHeartbeats 1000000

namespace Claims.GdnGating

open TVA

/-! ## Casting a rational bound to the reals -/

/-- A rational error bound `|a - b| ≤ |b|·u + e` casts to the same bound over `ℝ`. -/
theorem rat_abs_le_mul_add_cast {a b u e : ℚ} (h : |a - b| ≤ |b| * u + e) :
    |(a : ℝ) - (b : ℝ)| ≤ |(b : ℝ)| * (u : ℝ) + (e : ℝ) := by
  have h' : ((|a - b| : ℚ) : ℝ) ≤ ((|b| * u + e : ℚ) : ℝ) := by exact_mod_cast h
  rwa [Rat.cast_abs, Rat.cast_sub, Rat.cast_add, Rat.cast_mul, Rat.cast_abs] at h'

/-! ## The extremes of the two formats -/

/-- `float32`'s largest finite value, as a real number. -/
theorem binary32_maxVal_real :
    (Format.binary32.maxVal : ℝ) = ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := by
  rw [Format.binary32_maxVal]
  norm_num

/-- `bfloat16`'s largest finite value, as a real number. -/
theorem bfloat16_maxVal_real : (Format.bfloat16.maxVal : ℝ) = 255 * 2 ^ 120 := by
  rw [Format.bfloat16_maxVal]
  norm_num

/-- A finite value of a regular format is at most `maxVal`, as a real number. -/
theorem abs_toRat_le_maxVal_real {f : Format} (hf : f.Regular) {x : FloatV f}
    (hx : x.isFinite = true) : |x.toReal| ≤ (f.maxVal : ℝ) := by
  have h := FloatV.abs_toRat_le_maxVal hf hx
  have h' : ((|x.toRat| : ℚ) : ℝ) ≤ ((f.maxVal : ℚ) : ℝ) := by exact_mod_cast h
  rw [FloatV.toReal]
  rwa [Rat.cast_abs] at h'

/-- A finite `float32` value is at most `maxVal`. -/
theorem abs_toReal_le_maxVal' {x : F32} (hx : x.isFinite = true) :
    |x.toReal| ≤ (Format.binary32.maxVal : ℝ) :=
  abs_toRat_le_maxVal_real Format.binary32_regular hx

/-- A finite `bfloat16` value is at most `255·2^120`. -/
theorem bf16_abs_toReal_le {x : BF16} (hx : x.isFinite = true) : |x.toReal| ≤ 255 * 2 ^ 120 := by
  have h := abs_toRat_le_maxVal_real (f := Format.bfloat16) Format.bfloat16_regular hx
  rwa [bfloat16_maxVal_real] at h

/-- A sum of two `bfloat16` values is at most `510·2^120` in absolute value. -/
theorem bf16_sum_abs_le {a b : ℝ} (ha : |a| ≤ 255 * 2 ^ 120) (hb : |b| ≤ 255 * 2 ^ 120) :
    |a + b| ≤ 510 * 2 ^ 120 := by
  have h := abs_add_le a b
  linarith

/-- A sum of two `bfloat16` values is far above `-10^40`. -/
theorem bf16_sum_lower {a b : ℝ} (ha : |a| ≤ 255 * 2 ^ 120) (hb : |b| ≤ 255 * 2 ^ 120) :
    -((10 : ℝ) ^ 40) ≤ a + b := by
  have h := bf16_sum_abs_le ha hb
  have h2 : (510 : ℝ) * 2 ^ 120 ≤ (10 : ℝ) ^ 40 := by norm_num
  have h3 := neg_abs_le (a + b)
  linarith

/-! ## Equality and finiteness across `NaNEq` -/

/-- The left side of a `NaNEq` fact is finite when the right side is. -/
theorem naneq_fin_left {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hy : y.isFinite = true) : x.isFinite = true := by
  rcases h with h | ⟨h1, h2⟩
  · rw [h]; exact hy
  · have hy' := (FloatV.isFinite_iff (x := y)).mp hy
    rw [h2] at hy'
    exact absurd hy'.1 (by simp)

/-- The left side of a `NaNEq` fact equals the right side when the latter is finite. -/
theorem naneq_eq_right {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hy : y.isFinite = true) : x = y := by
  rcases h with h | ⟨h1, h2⟩
  · exact h
  · have hy' := (FloatV.isFinite_iff (x := y)).mp hy
    rw [h2] at hy'
    exact absurd hy'.1 (by simp)

/-- The left side of a `NaNEq` fact is not finite when the right side is not finite. -/
theorem naneq_notFinite_left {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y)
    (hy : y.isFinite = false) : x.isFinite = false := by
  rcases h with h | ⟨h1, h2⟩
  · rw [h]; exact hy
  · cases hx : x.isFinite with
    | false => rfl
    | true =>
        have hx' := (FloatV.isFinite_iff (x := x)).mp hx
        rw [h1] at hx'
        exact absurd hx'.1 (by simp)

/-! ## A finite product beyond `maxVal` -/

/-- A product of two finite values that is finite is within one rounding of the exact product,
even when the exact product is past `maxVal`. -/
theorem mul_err_any {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hfin : (FloatV.mul x y).isFinite = true) :
    |(FloatV.mul x y).toReal - (x.toReal * y.toReal)| ≤
      |x.toReal * y.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  by_cases hroom : |x.toRat * y.toRat| ≤ Format.binary32.maxVal
  · obtain ⟨-, herr⟩ := mul_exact_err_fin hx hy hroom (FloatV.NaNEq.refl _)
    simpa using herr
  · push Not at hroom
    have hpat : (FloatV.mul x y).pat = Format.binary32.roundRat RoundMode.rn false
        (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat) := by
      unfold FloatV.mul
      rw [Format.mul_of_finite RoundMode.rn hx hy]
      exact FloatV.pat_ofPat (roundRat_lt_width RoundMode.rn false _ _)
    have hfin' : Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false
        (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat)) = true := by
      rw [← hpat]
      exact hfin
    have hrw : (FloatV.mul x y).toRat = Format.binary32.toRat (Format.binary32.roundRat
        RoundMode.rn false (Format.binary32.sign x.pat != Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat * Format.binary32.toRat y.pat)) := by
      have h : (FloatV.mul x y).toRat = Format.binary32.toRat (FloatV.mul x y).pat := rfl
      rw [h, hpat]
    have hrat := roundRat_rn_fin_err hfin'
    rw [← hrw] at hrat
    have key := rat_abs_le_mul_add_cast (a := (FloatV.mul x y).toRat)
      (b := x.toRat * y.toRat) (u := 2 ^ (-24 : ℤ)) (e := 2 ^ (-150 : ℤ)) hrat
    simpa [FloatV.toReal] using key

/-! ## Room for a product with `oneF32` -/

/-- A product of `oneF32` with a finite value stays below `maxVal`. -/
theorem one_mul_room {y : F32} (hy : y.isFinite = true) :
    |oneF32.toRat * y.toRat| ≤ Format.binary32.maxVal := by
  rw [oneF32_toRat, one_mul]
  exact FloatV.abs_toRat_le_maxVal Format.binary32_regular hy

/-! ## Exponent bounds -/

/-- For `t ≤ 0`, `|t|·e^t ≤ 1`. -/
theorem abs_mul_exp_le_one {t : ℝ} (h : t ≤ 0) : |t| * Real.exp t ≤ 1 := by
  have ht : |t| = -t := abs_of_nonpos h
  rw [ht]
  by_cases ht0 : t = 0
  · rw [ht0]; norm_num
  · have hneg : t < 0 := lt_of_le_of_ne h ht0
    have hmul : Real.exp t * Real.exp (-t) = 1 := by
      rw [← Real.exp_add]; norm_num
    calc -t * Real.exp t ≤ Real.exp (-t) * Real.exp t := by
          apply mul_le_mul_of_nonneg_right _ (Real.exp_pos t).le
          linarith [Real.add_one_le_exp (-t)]
      _ = 1 := by rw [mul_comm (Real.exp (-t)) (Real.exp t), hmul]

/-- `e^21 ≤ 2^31`. -/
theorem exp21_le_pow31 : Real.exp 21 ≤ (2 : ℝ) ^ 31 := by
  have h : (21 : ℝ) ≤ 31 * Real.log 2 := by nlinarith [Real.log_two_gt_d9]
  have h2 : Real.exp (31 * Real.log 2) = (2 : ℝ) ^ 31 := by
    rw [show (31 : ℝ) = ((31 : ℕ) : ℝ) by norm_num, Real.exp_nat_mul, Real.exp_log (by norm_num)]
  have h3 := Real.exp_le_exp.mpr h
  rwa [h2] at h3

/-- A rounding error of `tl.exp` at `t ≤ 0` keeps the result in `[-1/2, 2]`. -/
theorem exp_err_neg {e t2 u v c : ℝ} (hu : u = 2 ^ (-22 : ℤ)) (hv : v = 2 ^ (-23 : ℤ))
    (hc : c = 2 ^ (-140 : ℤ)) (ht2 : t2 ≤ 0)
    (he : |e - Real.exp t2| ≤ (u + v * |t2|) * Real.exp t2 + c) :
    1 / 2 ≤ 1 + e ∧ e ≤ 2 := by
  have hu0 : (0 : ℝ) ≤ u := by rw [hu]; positivity
  have hv0 : (0 : ℝ) ≤ v := by rw [hv]; positivity
  have hc0 : (0 : ℝ) ≤ c := by rw [hc]; positivity
  have hexp0 : 0 < Real.exp t2 := Real.exp_pos t2
  have hexp1 : Real.exp t2 ≤ 1 := Real.exp_le_one_iff.mpr ht2
  have hB : (u + v * |t2|) * Real.exp t2 ≤ 1 / 4 := by
    have hsplit : (u + v * |t2|) * Real.exp t2 = u * Real.exp t2 + v * (|t2| * Real.exp t2) := by
      ring
    rw [hsplit]
    have ha : u * Real.exp t2 ≤ u := by
      have := mul_le_mul_of_nonneg_left hexp1 hu0
      simpa using this
    have hb : v * (|t2| * Real.exp t2) ≤ v := by
      have := mul_le_mul_of_nonneg_left (abs_mul_exp_le_one ht2) hv0
      simpa using this
    have hsum : u + v ≤ 1 / 4 := by rw [hu, hv]; norm_num
    exact (add_le_add ha hb).trans hsum
  have hc4 : c ≤ 1 / 4 := by rw [hc]; norm_num
  have hbc : (u + v * |t2|) * Real.exp t2 + c ≤ 1 / 2 :=
    (add_le_add hB hc4).trans (by norm_num)
  have hlow : Real.exp t2 - ((u + v * |t2|) * Real.exp t2 + c) ≤ e := by
    have h1 := (abs_le.mp he).1
    linarith
  have hup : e ≤ 2 := by
    have h2 := (abs_le.mp he).2
    have h3 : e - Real.exp t2 ≤ 1 / 2 := h2.trans hbc
    linarith
  refine ⟨?_, hup⟩
  linarith

/-- A rounding error of `tl.exp` at `0 ≤ t ≤ 21` keeps the result in `(0, 2^32]`. -/
theorem exp_err_pos {e t2 u v c a : ℝ} (hu : u = 2 ^ (-22 : ℤ)) (hv : v = 2 ^ (-23 : ℤ))
    (hc : c = 2 ^ (-140 : ℤ)) (ha : a = (2 : ℝ) ^ 32) (h0 : 0 ≤ t2) (h21 : t2 ≤ 21)
    (he : |e - Real.exp t2| ≤ (u + v * |t2|) * Real.exp t2 + c) :
    0 < e ∧ e ≤ a := by
  have hv0 : (0 : ℝ) ≤ v := by rw [hv]; positivity
  have hexp0 : 0 < Real.exp t2 := Real.exp_pos t2
  have hexp1 : 1 ≤ Real.exp t2 := Real.one_le_exp h0
  have hexp31 : Real.exp t2 ≤ (2 : ℝ) ^ 31 := le_trans (Real.exp_le_exp.mpr h21) exp21_le_pow31
  have hc1 : u + v * |t2| ≤ 1 / 2 := by
    have habs : |t2| = t2 := abs_of_nonneg h0
    rw [habs]
    have hvv : v * t2 ≤ v * 21 := mul_le_mul_of_nonneg_left h21 hv0
    have hsum : u + v * 21 ≤ 1 / 2 := by rw [hu, hv]; norm_num
    calc u + v * t2 = v * t2 + u := by ring
      _ ≤ v * 21 + u := add_le_add_left hvv u
      _ = u + v * 21 := by ring
      _ ≤ 1 / 2 := hsum
  have hB : (u + v * |t2|) * Real.exp t2 ≤ (1 / 2) * Real.exp t2 :=
    mul_le_mul_of_nonneg_right hc1 hexp0.le
  have hc4 : c ≤ 1 / 4 := by rw [hc]; norm_num
  have hB2 : (u + v * |t2|) * Real.exp t2 ≤ (1 / 2) * (2 : ℝ) ^ 31 :=
    hB.trans (mul_le_mul_of_nonneg_left hexp31 (by norm_num))
  have hupper : Real.exp t2 + ((u + v * |t2|) * Real.exp t2 + c) ≤ (2 : ℝ) ^ 32 := by
    calc Real.exp t2 + ((u + v * |t2|) * Real.exp t2 + c)
        = (Real.exp t2 + (u + v * |t2|) * Real.exp t2) + c := by ring
      _ ≤ ((2 : ℝ) ^ 31 + (1 / 2) * (2 : ℝ) ^ 31) + 1 / 4 :=
          add_le_add (add_le_add hexp31 hB2) hc4
      _ ≤ (2 : ℝ) ^ 32 := by norm_num
  have hlow : Real.exp t2 - ((u + v * |t2|) * Real.exp t2 + c) ≤ e := by
    have h1 := (abs_le.mp he).1
    linarith
  have hpos : 0 < Real.exp t2 - ((u + v * |t2|) * Real.exp t2 + c) := by
    have hle : Real.exp t2 - ((u + v * |t2|) * Real.exp t2 + c) ≥
        Real.exp t2 - ((1 / 2) * Real.exp t2 + 1 / 4) := by linarith
    have h1 := hexp1
    linarith
  have hup : e ≤ (2 : ℝ) ^ 32 := by
    have h2 := (abs_le.mp he).2
    have h3 : e ≤ Real.exp t2 + ((u + v * |t2|) * Real.exp t2 + c) := by linarith
    exact h3.trans hupper
  exact ⟨by linarith, by rwa [ha]⟩

/-! ## Bounds for the logarithm's argument -/

/-- When `1 + e` is bounded below by `1/2` and above through `a`, the rounded `s` is positive and
its logarithm is at most `3a + 4` in absolute value. -/
theorem s_pos_and_log_bound {s e a u w : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hw : w = 2 ^ (-150 : ℤ))
    (hge : 1 / 2 ≤ 1 + e) (he : e ≤ a) (ha : 1 ≤ a)
    (hs : |s - (1 + e)| ≤ (1 + e) * u + w) :
    0 < s ∧ |Real.log s| ≤ 3 * a + 4 := by
  have h1e : (0 : ℝ) < 1 + e := lt_of_lt_of_le (by norm_num) hge
  have hsb := abs_le.mp hs
  have hu2 : u ≤ 1 / 2 := by rw [hu]; norm_num
  have hw8 : w ≤ 1 / 8 := by rw [hw]; norm_num
  have hw1 : w ≤ 1 := by rw [hw]; norm_num
  have hslo : (1 / 8 : ℝ) ≤ s := by
    have h1 : (1 / 2 : ℝ) * (1 - u) ≤ (1 + e) * (1 - u) :=
      mul_le_mul_of_nonneg_right hge (by linarith)
    have h3 : (1 / 8 : ℝ) ≤ (1 / 2) * (1 - u) - w := by linarith
    linarith [hsb.1]
  have h1e2 : 1 + e ≤ 2 * a := by linarith
  have hshi : s ≤ 3 * a := by
    have hu0 : (0 : ℝ) ≤ u := by rw [hu]; positivity
    have ha0 : (0 : ℝ) ≤ a := by linarith only [ha]
    have h1 : (1 + e) * (1 + u) ≤ (2 * a) * (1 + u) :=
      mul_le_mul_of_nonneg_right h1e2 (by linarith only [hu0])
    have hstep1 : (2 * a) * (1 + u) = 2 * a + 2 * a * u := by ring
    have h2u : 2 * u ≤ 1 / 2 := by rw [hu]; norm_num
    have hCa : 2 * a * u ≤ a * (1 / 2) := by
      calc 2 * a * u = a * (2 * u) := by ring
        _ ≤ a * (1 / 2) := mul_le_mul_of_nonneg_left h2u ha0
    have hC2 : a * (1 / 2) + w ≤ a := by linarith only [hw8, ha]
    have h3 : 2 * a * u + w ≤ a := by linarith only [hCa, hC2]
    have h2s : s ≤ (1 + e) * (1 + u) + w := by linarith only [hsb.2]
    calc s ≤ (1 + e) * (1 + u) + w := h2s
      _ ≤ (2 * a) * (1 + u) + w := by linarith only [h1]
      _ = 2 * a + (2 * a * u + w) := by rw [hstep1]; ring
      _ ≤ 2 * a + a := by linarith only [h3]
      _ = 3 * a := by ring
  refine ⟨by linarith, ?_⟩
  have hlog_hi : Real.log s ≤ s - 1 := Real.log_le_sub_one_of_pos (by linarith)
  have hlog_lo : Real.log (1 / 8) ≤ Real.log s := Real.log_le_log (by norm_num) hslo
  have hlog18 : -3 ≤ Real.log (1 / 8) := by
    have hlog8 : Real.log 8 ≤ 3 := by
      have h8 : Real.log 8 = 3 * Real.log 2 := by
        rw [show (8 : ℝ) = 2 ^ 3 by norm_num, Real.log_pow]
        push_cast
        ring
      rw [h8]
      nlinarith [Real.log_two_lt_d9]
    rw [show (1 / 8 : ℝ) = (8 : ℝ)⁻¹ by norm_num, Real.log_inv]
    linarith
  rw [abs_le]
  constructor <;> linarith [hlog_hi, hlog_lo, hlog18]

/-- The logarithm of the argument of the crude branch is small. -/
theorem log_room_bound {l a : ℝ} (ha : a ≤ (2 : ℝ) ^ 32) (hl : |l| ≤ 3 * a + 4) :
    (1 + 2 ^ (-22 : ℤ)) * |l| + 2 ^ (-149 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
  have h1 : |l| ≤ 3 * (2 : ℝ) ^ 32 + 4 := by linarith
  have h2 : (1 + 2 ^ (-22 : ℤ)) * |l| ≤ (1 + 2 ^ (-22 : ℤ)) * (3 * (2 : ℝ) ^ 32 + 4) :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  have h3 : (1 + 2 ^ (-22 : ℤ)) * (3 * (2 : ℝ) ^ 32 + 4) + 2 ^ (-149 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    rw [binary32_maxVal_real]
    norm_num
  exact le_trans (add_le_add_left h2 _) h3

/-- `|1 + e|` is below `maxVal` when `1 + e ≥ 1/2` and `e ≤ 2^32`. -/
theorem one_add_abs_le_maxVal {e a : ℝ} (ha : a = (2 : ℝ) ^ 32) (hge : 1 / 2 ≤ 1 + e)
    (he : e ≤ a) : |1 + e| ≤ (Format.binary32.maxVal : ℝ) := by
  have hpos : (0 : ℝ) < 1 + e := lt_of_lt_of_le (by norm_num) hge
  rw [abs_of_pos hpos]
  have h2 : 1 + e ≤ (2 : ℝ) ^ 32 + 1 := by linarith
  have h3 : (2 : ℝ) ^ 32 + 1 ≤ (Format.binary32.maxVal : ℝ) := by
    rw [binary32_maxVal_real]
    norm_num
  exact h2.trans h3

/-- `S_pos_of_big` with the hypothesis its argument needs: the error term must obey
`v * (1 + w) ≤ 1/2`.  With `w` a free real that bound does not follow from the other hypotheses. -/
theorem S_pos_of_big_of_bound {S p u v w : ℝ} (hu : u = 2 ^ (-23 : ℤ)) (_hv : v = 2 ^ (-24 : ℤ))
    (hp : 0 < p) (hS : |S - p| ≤ 52 * u * p + v * (1 + w))
    (hvw : v * (1 + w) ≤ 1 / 2) (hbig : 1 / 2 ≤ |S|) : 0 < S := by
  have hu3 : 52 * u < 1 := by rw [hu]; norm_num
  by_contra hcon
  push Not at hcon
  have habs : |S| = -S := abs_of_nonpos hcon
  rw [habs] at hbig
  have h1 : -(52 * u * p + v * (1 + w)) ≤ S - p := (abs_le.mp hS).1
  have h2 : S ≥ p - (52 * u * p + v * (1 + w)) := by linarith only [h1]
  have hp' : 52 * u * p < p := by
    have h := mul_lt_mul_of_pos_right hu3 hp
    simpa using h
  linarith only [h2, hp', hvw, hbig]

/-- The magnitude of the softplus value is at least `1/2` when the exact product overflows. -/
theorem S_abs_ge_half {E S : ℝ} (hE : 0 < E) (hEM : E ≤ (Format.binary32.maxVal : ℝ))
    (hbig : (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤ |E * S|) :
    1 / 2 ≤ |S| := by
  have hM : (0 : ℝ) < (Format.binary32.maxVal : ℝ) := by
    rw [binary32_maxVal_real]
    norm_num
  have h1 : |E * S| = E * |S| := by rw [abs_mul, abs_of_pos hE]
  rw [h1] at hbig
  have h3 : E * |S| ≤ (Format.binary32.maxVal : ℝ) * |S| :=
    mul_le_mul_of_nonneg_right hEM (abs_nonneg S)
  have h4 : (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤
      (Format.binary32.maxVal : ℝ) * |S| := hbig.trans h3
  have h5 : (1 / 2 : ℝ) ≤ 1 - 2 ^ (-22 : ℤ) := by norm_num
  have h6 : (1 / 2) * (Format.binary32.maxVal : ℝ) ≤
      (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) :=
    mul_le_mul_of_nonneg_right h5 hM.le
  have h7 : (Format.binary32.maxVal : ℝ) * (1 / 2) ≤ (Format.binary32.maxVal : ℝ) * |S| := by
    rw [mul_comm ((Format.binary32.maxVal : ℝ)) (1 / 2)]
    exact h6.trans h4
  exact (mul_le_mul_iff_of_pos_left hM).mp h7

/-- A real `y` within one rounding of `x` is at most `k·(1 + 2^-24) + 2^-150` when `|x| ≤ k`. -/
theorem abs_le_of_mul_err {y x k : ℝ} (hk : 0 ≤ k) (hx : |x| ≤ k)
    (h : |y - x| ≤ (2 : ℝ) ^ (-24 : ℤ) * |x| + (2 : ℝ) ^ (-150 : ℤ)) :
    |y| ≤ k * (1 + 2 ^ (-24 : ℤ)) + 2 ^ (-150 : ℤ) := by
  have h1 : |y| ≤ |y - x| + |x| := by
    have := abs_add_le (y - x) x
    rwa [sub_add_cancel] at this
  have h2 : (2 : ℝ) ^ (-24 : ℤ) * |x| ≤ (2 : ℝ) ^ (-24 : ℤ) * k :=
    mul_le_mul_of_nonneg_left hx (by positivity)
  have h3 : |y| ≤ ((2 : ℝ) ^ (-24 : ℤ) * |x| + (2 : ℝ) ^ (-150 : ℤ)) + |x| := by
    linarith
  have h4 : ((2 : ℝ) ^ (-24 : ℤ) * |x| + (2 : ℝ) ^ (-150 : ℤ)) + |x| ≤
      k * (1 + 2 ^ (-24 : ℤ)) + 2 ^ (-150 : ℤ) := by
    have hk0 : (0 : ℝ) ≤ k := hk
    have e1 : (2 : ℝ) ^ (-24 : ℤ) * |x| + |x| = |x| * (1 + 2 ^ (-24 : ℤ)) := by ring
    have e2 : |x| * (1 + 2 ^ (-24 : ℤ)) ≤ k * (1 + 2 ^ (-24 : ℤ)) :=
      mul_le_mul_of_nonneg_right hx (by positivity)
    linarith
  exact h3.trans h4

/-- `|t| ≤ 2^129` pushes a rounded value of the lane's sum to `|t| ≤ 2^130`. -/
theorem abs_t2_le {t x : ℝ} (hx : |x| ≤ (2 : ℝ) ^ 129)
    (h : |t - x| ≤ (2 : ℝ) ^ (-24 : ℤ) * |x| + (2 : ℝ) ^ (-150 : ℤ)) :
    |t| ≤ (2 : ℝ) ^ 130 := by
  have h1 := abs_le_of_mul_err (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ 129) hx h
  have h2 : (2 : ℝ) ^ 129 * (1 + 2 ^ (-24 : ℤ)) + 2 ^ (-150 : ℤ) ≤ (2 : ℝ) ^ 130 := by norm_num
  exact h1.trans h2

/-- The exact input of the lane is far from the exponential's ceiling. -/
theorem abs_x_le {x0 : ℝ} (h : |x0| ≤ 510 * 2 ^ 120) : |x0| ≤ (2 : ℝ) ^ 129 := by
  have h2 : (510 : ℝ) * 2 ^ 120 ≤ (2 : ℝ) ^ 129 := by norm_num
  exact h.trans h2

/-- A rounded input of the lane is far from the exponential's ceiling. -/
theorem abs_xf_le {y x0 : ℝ} (hx : |x0| ≤ 510 * 2 ^ 120)
    (h : |y - x0| ≤ (2 : ℝ) ^ (-24 : ℤ) * |x0| + (2 : ℝ) ^ (-150 : ℤ)) :
    |y| ≤ (2 : ℝ) ^ 129 := by
  have h1 := abs_le_of_mul_err (by positivity : (0 : ℝ) ≤ 510 * 2 ^ 120) hx h
  have h2 : (510 : ℝ) * 2 ^ 120 * (1 + 2 ^ (-24 : ℤ)) + 2 ^ (-150 : ℤ) ≤ (2 : ℝ) ^ 129 := by
    norm_num
  exact h1.trans h2

/-- A positive input below `20` after rounding pushes the exponent's argument to at most `21`. -/
theorem t2_le_21 {xf t1 t2 : ℝ}
    (ht1 : |t1 - xf| ≤ (2 : ℝ) ^ (-24 : ℤ) * |xf| + (2 : ℝ) ^ (-150 : ℤ))
    (ht2 : |t2 - xf| ≤ (2 : ℝ) ^ (-24 : ℤ) * |xf| + (2 : ℝ) ^ (-150 : ℤ))
    (hle : t1 ≤ 20) : t2 ≤ 21 := by
  have h2 := (abs_le.mp ht2).2
  have ht2' : t2 ≤ ((2 : ℝ) ^ (-24 : ℤ) * |xf| + (2 : ℝ) ^ (-150 : ℤ)) + xf :=
    sub_le_iff_le_add.mp h2
  by_cases hx : xf ≤ 0
  · have hx' : |xf| = -xf := abs_of_nonpos hx
    have hb : xf + (2 : ℝ) ^ (-24 : ℤ) * |xf| + (2 : ℝ) ^ (-150 : ℤ) ≤ 21 := by
      rw [hx']
      have hstep : xf + (2 : ℝ) ^ (-24 : ℤ) * -xf + (2 : ℝ) ^ (-150 : ℤ) =
          xf * (1 - (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ) := by ring
      rw [hstep]
      have h1 : xf * (1 - (2 : ℝ) ^ (-24 : ℤ)) ≤ 0 :=
        mul_nonpos_of_nonpos_of_nonneg hx (by norm_num)
      have hw : (2 : ℝ) ^ (-150 : ℤ) ≤ 21 := by norm_num
      simpa using add_le_add h1 hw
    calc t2 ≤ ((2 : ℝ) ^ (-24 : ℤ) * |xf| + (2 : ℝ) ^ (-150 : ℤ)) + xf := ht2'
      _ = xf + (2 : ℝ) ^ (-24 : ℤ) * |xf| + (2 : ℝ) ^ (-150 : ℤ) := by ring
      _ ≤ 21 := hb
  · push Not at hx
    have hx' : |xf| = xf := abs_of_pos hx
    have h1 := (abs_le.mp ht1).1
    have hxl : xf * (1 - (2 : ℝ) ^ (-24 : ℤ)) - (2 : ℝ) ^ (-150 : ℤ) ≤ 20 := by
      have hA : xf - ((2 : ℝ) ^ (-24 : ℤ) * xf + (2 : ℝ) ^ (-150 : ℤ)) ≤ t1 := by
        rw [hx'] at h1
        have h := add_le_add_right h1 xf
        calc xf - ((2 : ℝ) ^ (-24 : ℤ) * xf + (2 : ℝ) ^ (-150 : ℤ))
            = xf + (-((2 : ℝ) ^ (-24 : ℤ) * xf + (2 : ℝ) ^ (-150 : ℤ))) := by ring
          _ ≤ xf + (t1 - xf) := h
          _ = t1 := by ring
      calc xf * (1 - (2 : ℝ) ^ (-24 : ℤ)) - (2 : ℝ) ^ (-150 : ℤ)
          = xf - ((2 : ℝ) ^ (-24 : ℤ) * xf + (2 : ℝ) ^ (-150 : ℤ)) := by ring
        _ ≤ t1 := hA
        _ ≤ 20 := hle
    have hxf_le : xf ≤ (20 + (2 : ℝ) ^ (-150 : ℤ)) / (1 - (2 : ℝ) ^ (-24 : ℤ)) := by
      have hpos : (0 : ℝ) < 1 - (2 : ℝ) ^ (-24 : ℤ) := by norm_num
      rw [le_div_iff₀ hpos]
      exact sub_le_iff_le_add.mp hxl
    have hcoe : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-24 : ℤ) := by positivity
    have hst : xf * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ) ≤
        (20 + (2 : ℝ) ^ (-150 : ℤ)) / (1 - (2 : ℝ) ^ (-24 : ℤ)) * (1 + (2 : ℝ) ^ (-24 : ℤ)) +
          (2 : ℝ) ^ (-150 : ℤ) := by
      have hm := mul_le_mul_of_nonneg_right hxf_le hcoe
      exact add_le_add_left hm _
    have hnum : (20 + (2 : ℝ) ^ (-150 : ℤ)) / (1 - (2 : ℝ) ^ (-24 : ℤ)) *
        (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ) ≤ 21 := by norm_num
    have hb : xf + (2 : ℝ) ^ (-24 : ℤ) * xf + (2 : ℝ) ^ (-150 : ℤ) ≤ 21 :=
      le_trans (le_of_eq (by ring : xf + (2 : ℝ) ^ (-24 : ℤ) * xf + (2 : ℝ) ^ (-150 : ℤ) =
        xf * (1 + (2 : ℝ) ^ (-24 : ℤ)) + (2 : ℝ) ^ (-150 : ℤ))) (hst.trans hnum)
    calc t2 ≤ ((2 : ℝ) ^ (-24 : ℤ) * |xf| + (2 : ℝ) ^ (-150 : ℤ)) + xf := ht2'
      _ = xf + (2 : ℝ) ^ (-24 : ℤ) * xf + (2 : ℝ) ^ (-150 : ℤ) := by rw [hx']; ring
      _ ≤ 21 := hb

/-- The condition `t1 <= 20` on finite floats is a real inequality. -/
theorem le_cond_true {t1 : F32} (ht1 : t1.isFinite = true)
    (h : FloatV.le t1 twentyF32 = true) : t1.toReal ≤ 20 := by
  have ht1n : t1.isNaN = false := ((FloatV.isFinite_iff (x := t1)).mp ht1).1
  have h20n : twentyF32.isNaN = false := ((FloatV.isFinite_iff (x := twentyF32)).mp twentyF32_isFinite).1
  have hle := (FloatV.le_iff ht1n h20n).mp h
  rw [twentyF32_toRat] at hle
  have := (Rat.cast_le (K := ℝ)).mpr hle
  simpa [FloatV.toReal] using this

/-- The negation of the condition `t1 <= 20` on finite floats is a real inequality. -/
theorem le_cond_false {t1 : F32} (ht1 : t1.isFinite = true)
    (h : FloatV.le t1 twentyF32 = false) : 20 < t1.toReal := by
  have ht1n : t1.isNaN = false := ((FloatV.isFinite_iff (x := t1)).mp ht1).1
  have h20n : twentyF32.isNaN = false := ((FloatV.isFinite_iff (x := twentyF32)).mp twentyF32_isFinite).1
  by_contra hcon
  push Not at hcon
  have hq : t1.toRat ≤ (20 : ℚ) := by
    have h1 : ((t1.toRat : ℚ) : ℝ) ≤ ((20 : ℚ) : ℝ) := by
      simpa [FloatV.toReal] using hcon
    exact (Rat.cast_le (K := ℝ)).mp h1
  have htrue : FloatV.le t1 twentyF32 = true := by
    apply (FloatV.le_iff ht1n h20n).mpr
    rw [twentyF32_toRat]
    exact hq
  rw [htrue] at h
  exact absurd h (by simp)

end Claims.GdnGating
