import TVA.Lemmas.Mem
import TVA.Math.Float.Format.Error

/-!
# Rounding error of the IEEE operations

For a *regular* format (a fraction bit, a normal range: `Format.Regular`), every operation that
rounds to nearest is within half an ulp of the exact result, as long as the exact result is
within range (`|q| ≤ maxVal`):

`|round(q) − q| ≤ |q| · u + η`, where `u = 2^-(m+1)` is the unit roundoff and `η = 2^(emin−m−1)`
is the subnormal term. The result is then finite.

* `Format.roundRat_rn_err`: rounding a rational;
* `FloatV.add_err`, `mul_err`, `fma_err`, `divRn_err`, `cvt_err`: the operations on finite
  operands.

These build on the format-generic half-ulp theorem `Format.roundMag_rn_err`.
-/
namespace TVA.Math.Float.Format

variable (f : Format)

/-- The largest finite value of the format. -/
def maxVal : ℚ := (f.maxSig : ℚ) * 2 ^ (f.emax - f.m)

/-- The unit roundoff `2^-(m+1)`. -/
def u : ℚ := 2 ^ (-(f.m : ℤ) - 1)

/-- The absolute error term of subnormal results, `2^(emin − m − 1)`. -/
def eta : ℚ := 2 ^ (f.emin - f.m - 1)

/-- The format has a fraction bit and a normal range (all IEEE formats and the fp8 formats). -/
structure Regular : Prop where
  m_pos : 1 ≤ f.m
  range : f.emin ≤ f.emax

variable {f}

theorem u_nonneg : 0 ≤ f.u := by unfold u; positivity

theorem eta_nonneg : 0 ≤ f.eta := by unfold eta; positivity

theorem signBit_pos : 0 < f.signBit := by unfold signBit; positivity

theorem withSign_lt_two_pow (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.withSign s mag < 2 ^ f.width := by
  rw [f.withSign_of_lt s h]
  have : 2 ^ f.width = 2 * f.signBit := by unfold width signBit; rw [pow_succ]; ring
  split <;> omega

theorem isFinite_zero (hf : f.Regular) : f.isFinite 0 = true := by
  have hr := hf.range
  have hexpMax : 1 ≤ f.expMax := by
    by_cases hi : f.special = .ieee
    · have := f.emax_ieee hi; unfold emin at hr; omega
    · have := f.emax_nonieee hi; unfold emin at hr; omega
  refine f.isFinite_of 0 (Or.inl ?_)
  unfold expField; simp; omega

theorem magRat_zero : f.magRat 0 = 0 := by unfold magRat expField fracField; simp

theorem toRat_of_isZero {x : ℕ} (hz : f.isZero x = true) : f.toRat x = 0 := by
  have : f.magRat x = 0 := by
    unfold isZero at hz; simp only [Bool.and_eq_true, beq_iff_eq] at hz
    unfold magRat; simp [hz.1, hz.2]
  unfold toRat; rw [this]; simp

theorem roundRat_zero (mode : RoundMode) (sat z : Bool) : f.roundRat mode sat z 0 = f.withSign z 0 := by
  unfold roundRat
  simp only [beq_self_eq_true, ite_true]
  rw [f.withSign_of_lt z signBit_pos]; cases z <;> simp [negZero]

theorem roundRat_ne_zero (mode : RoundMode) (sat z : Bool) {q : ℚ} (hq : q ≠ 0) :
    f.roundRat mode sat z q = f.withSign (decide (q < 0)) (f.roundMag mode sat (decide (q < 0)) |q|) := by
  unfold roundRat
  rw [if_neg (by simpa using hq)]
  congr 2
  split_ifs with h
  · exact (abs_of_neg h).symm
  · exact (abs_of_nonneg (not_lt.mp h)).symm

/-- **Rounding to nearest, any rational in range.** The result is a finite pattern within
`|q| · u + η` of `q`. -/
theorem roundRat_rn_err (hf : f.Regular) (sat z : Bool) {q : ℚ} (hmax : |q| ≤ f.maxVal) :
    f.roundRat .rn sat z q < 2 ^ f.width ∧ f.isFinite (f.roundRat .rn sat z q) = true ∧
      |f.toRat (f.roundRat .rn sat z q) - q| ≤ |q| * f.u + f.eta := by
  have hu := u_nonneg (f := f)
  have he := eta_nonneg (f := f)
  by_cases hq : q = 0
  · subst hq
    rw [roundRat_zero]
    refine ⟨withSign_lt_two_pow z signBit_pos, ?_, ?_⟩
    · rw [f.isFinite_withSign z signBit_pos]; exact isFinite_zero hf
    · rw [f.toRat_withSign z signBit_pos, magRat_zero]
      cases z <;> simp <;> linarith
  · rw [roundRat_ne_zero _ _ _ hq]
    have hpos : 0 < |q| := abs_pos.mpr hq
    obtain ⟨hlt, hfin, -⟩ := f.roundMag_rn_err sat (decide (q < 0)) hpos hf.m_pos hf.range hmax
    have herr := f.roundMag_rn_err_rel sat (decide (q < 0)) hpos hf.m_pos hf.range hmax
    refine ⟨withSign_lt_two_pow _ hlt, by rw [f.isFinite_withSign _ hlt]; exact hfin, ?_⟩
    rw [f.toRat_withSign _ hlt]
    unfold u eta
    by_cases hneg : q < 0
    · rw [decide_eq_true hneg] at herr ⊢
      simp only [ite_true]
      rw [show ∀ a : ℚ, -a - q = -(a - |q|) from fun a => by rw [abs_of_neg hneg]; ring, abs_neg]
      exact herr
    · rw [decide_eq_false hneg] at herr ⊢
      simp only [Bool.false_eq_true, ite_false]
      rw [show ∀ a : ℚ, a - q = a - |q| from fun a => by rw [abs_of_nonneg (not_lt.mp hneg)]]
      exact herr

/-- **Rounding to nearest in the normal range** has no absolute term: for
`2^emin ≤ |q| ≤ maxVal` the result is within `|q| · u` of `q`. -/
theorem roundRat_rn_err_normal (hf : f.Regular) (sat z : Bool) {q : ℚ} (hmin : (2 : ℚ) ^ f.emin ≤ |q|)
    (hmax : |q| ≤ f.maxVal) :
    |f.toRat (f.roundRat .rn sat z q) - q| ≤ |q| * f.u := by
  have hpos : 0 < |q| := lt_of_lt_of_le (zpow_pos (by norm_num) _) hmin
  have hq : q ≠ 0 := abs_pos.mp hpos
  rw [roundRat_ne_zero _ _ _ hq]
  obtain ⟨hlt, -, herr⟩ := f.roundMag_rn_err sat (decide (q < 0)) hpos hf.m_pos hf.range hmax
  obtain ⟨hlo, hhi⟩ := log2Floor_rat |q| hpos
  have hE : f.expOf |q| = log2Floor |q|.num.natAbs |q|.den := by
    unfold expOf
    apply max_eq_left
    by_contra h
    push Not at h
    have : (2 : ℚ) ^ (log2Floor |q|.num.natAbs |q|.den + 1) ≤ 2 ^ f.emin :=
      zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  have hb : |(f.magRat (f.roundMag .rn sat (decide (q < 0)) |q|) - |q|)| ≤ |q| * f.u := by
    refine herr.trans ?_
    rw [hE]
    unfold u
    rw [show log2Floor |q|.num.natAbs |q|.den - (f.m : ℤ) - 1 =
        log2Floor |q|.num.natAbs |q|.den + (-(f.m : ℤ) - 1) by ring, zpow_add₀ (by norm_num)]
    exact mul_le_mul_of_nonneg_right hlo (zpow_pos (by norm_num) _).le
  rw [f.toRat_withSign _ hlt]
  by_cases hneg : q < 0
  · rw [decide_eq_true hneg] at hb ⊢
    simp only [ite_true]
    rw [show ∀ a : ℚ, -a - q = -(a - |q|) from fun a => by rw [abs_of_neg hneg]; ring, abs_neg]
    exact hb
  · rw [decide_eq_false hneg] at hb ⊢
    simp only [Bool.false_eq_true, ite_false]
    rw [show ∀ a : ℚ, a - q = a - |q| from fun a => by rw [abs_of_nonneg (not_lt.mp hneg)]]
    exact hb

/-- A rounded magnitude with the sign of `q` is the rounding of `q`. -/
theorem withSign_roundMag_eq (mode : RoundMode) (sat z : Bool) {q a : ℚ} {s : Bool} (hq : q ≠ 0)
    (hs : decide (q < 0) = s) (ha : |q| = a) :
    f.withSign s (f.roundMag mode sat s a) = f.roundRat mode sat z q := by
  rw [roundRat_ne_zero _ _ _ hq, hs, ha]

end TVA.Math.Float.Format

namespace TVA.FloatV

open TVA.Math.Float TVA.Math.Float.Format

variable {f : Format}

theorem pat_lt (x : FloatV f) : x.pat < 2 ^ f.width := x.bits.isLt

theorem toRat_def (x : FloatV f) : x.toRat = f.toRat x.pat := rfl

theorem toRat_of_isZero {x : FloatV f} (h : f.isZero x.pat = true) : x.toRat = 0 :=
  Format.toRat_of_isZero h

/-- A product with a zero factor has value zero. -/
theorem toRat_mul_of_isZero {x y : FloatV f} (h : (f.isZero x.pat || f.isZero y.pat) = true) :
    x.toRat * y.toRat = 0 := by
  simp only [Bool.or_eq_true] at h
  rcases h with h | h
  · rw [toRat_of_isZero h, zero_mul]
  · rw [toRat_of_isZero h, mul_zero]

/-- The signed zero `if z then -0 else +0` is the rounding of `0`. -/
theorem signedZero_eq (z : Bool) : (if z = true then f.negZero else 0) = f.roundRat .rn false z 0 := by
  rw [roundRat_zero, f.withSign_of_lt _ signBit_pos]; cases z <;> simp [negZero]

theorem pat_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).pat = n := by
  simp [ofPat, pat, Nat.mod_eq_of_lt h]

theorem toRat_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).toRat = f.toRat n := by
  simp only [toRat, pat_ofPat h]

theorem isFinite_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isFinite = f.isFinite n := by
  simp only [isFinite, pat_ofPat h]

theorem isFinite_iff {x : FloatV f} :
    x.isFinite = true ↔ f.isNaN x.pat = false ∧ f.isInf x.pat = false := by
  unfold isFinite Format.isFinite; simp

/-- The pattern of a rounding, read back as a `FloatV`. -/
theorem ofPat_roundRat_err (hf : f.Regular) (sat z : Bool) {q : ℚ} (hmax : |q| ≤ f.maxVal) :
    (ofPat (f.roundRat .rn sat z q) : FloatV f).isFinite = true ∧
      |(ofPat (f.roundRat .rn sat z q) : FloatV f).toRat - q| ≤ |q| * f.u + f.eta := by
  obtain ⟨hlt, hfin, herr⟩ := Format.roundRat_rn_err hf sat z hmax
  exact ⟨by rw [isFinite_ofPat hlt]; exact hfin, by rw [toRat_ofPat hlt]; exact herr⟩

/-- The pattern of a rounding in the normal range, read back as a `FloatV`. -/
theorem ofPat_roundRat_err_normal (hf : f.Regular) (sat z : Bool) {q : ℚ}
    (hmin : (2 : ℚ) ^ f.emin ≤ |q|) (hmax : |q| ≤ f.maxVal) :
    (ofPat (f.roundRat .rn sat z q) : FloatV f).isFinite = true ∧
      |(ofPat (f.roundRat .rn sat z q) : FloatV f).toRat - q| ≤ |q| * f.u := by
  obtain ⟨hlt, hfin, -⟩ := Format.roundRat_rn_err hf sat z hmax
  exact ⟨by rw [isFinite_ofPat hlt]; exact hfin,
    by rw [toRat_ofPat hlt]; exact Format.roundRat_rn_err_normal hf sat z hmin hmax⟩

theorem add_eq_roundRat {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    ∃ z, f.add .rn f.nanMag x.pat y.pat = f.roundRat .rn false z (x.toRat + y.toRat) := by
  obtain ⟨hnx, hix⟩ := isFinite_iff.mp hx
  obtain ⟨hny, hiy⟩ := isFinite_iff.mp hy
  unfold Format.add
  simp only [hnx, hny, hix, hiy, Bool.or_self, Bool.and_self, Bool.false_eq_true, ite_false]
  by_cases hz : (f.isZero x.pat && f.isZero y.pat) = true
  · rw [if_pos hz]
    simp only [Bool.and_eq_true] at hz
    refine ⟨zeroSumNeg .rn (f.sign x.pat) (f.sign y.pat), ?_⟩
    rw [toRat_of_isZero hz.1, toRat_of_isZero hz.2, add_zero, signedZero_eq]
  · rw [if_neg hz]; exact ⟨_, rfl⟩

/-- **IEEE addition.** -/
theorem add_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toRat + y.toRat| ≤ f.maxVal) :
    (add x y).isFinite = true ∧
      |(add x y).toRat - (x.toRat + y.toRat)| ≤ |x.toRat + y.toRat| * f.u + f.eta := by
  obtain ⟨z, hz⟩ := add_eq_roundRat hx hy
  unfold add; rw [hz]
  exact ofPat_roundRat_err hf false z hmax

/-- The sign of a product or quotient of nonzero values. -/
theorem sign_mul_lt {x y : FloatV f} (hx : f.isZero x.pat = false) (hy : f.isZero y.pat = false)
    {op : ℚ → ℚ → ℚ} (hop : ∀ a b : ℚ, 0 < a → 0 < b →
      0 < op a b ∧ op (-a) b = -op a b ∧ op a (-b) = -op a b ∧ op (-a) (-b) = op a b) :
    decide (op x.toRat y.toRat < 0) = (f.sign x.pat != f.sign y.pat) ∧
      |op x.toRat y.toRat| = op (f.magRat x.pat) (f.magRat y.pat) := by
  have hpx := f.magRat_pos _ hx
  have hpy := f.magRat_pos _ hy
  obtain ⟨h0, h1, h2, h3⟩ := hop _ _ hpx hpy
  unfold toRat Format.toRat
  cases f.sign x.pat <;> cases f.sign y.pat <;>
    simp only [ite_true, ite_false, Bool.false_eq_true, h1, h2, h3] <;>
    refine ⟨?_, ?_⟩ <;>
    simp [h0, h0.le, not_lt.mpr h0.le, abs_of_pos h0, abs_neg]

theorem mul_eq_roundRat {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    ∃ z, f.mul .rn f.nanMag x.pat y.pat = f.roundRat .rn false z (x.toRat * y.toRat) := by
  obtain ⟨hnx, hix⟩ := isFinite_iff.mp hx
  obtain ⟨hny, hiy⟩ := isFinite_iff.mp hy
  unfold Format.mul
  simp only [hnx, hny, hix, hiy, Bool.or_self, Bool.false_eq_true, ite_false]
  split_ifs with hz
  · refine ⟨f.sign x.pat != f.sign y.pat, ?_⟩
    rw [toRat_mul_of_isZero hz, roundRat_zero]
  · simp only [Bool.or_eq_true, not_or, Bool.not_eq_true] at hz
    obtain ⟨hs, ha⟩ := sign_mul_lt hz.1 hz.2 (op := (· * ·))
      (fun a b ha hb => ⟨mul_pos ha hb, by ring, by ring, by ring⟩)
    have hq : x.toRat * y.toRat ≠ 0 := by
      intro h; rw [h, abs_zero] at ha
      exact absurd ha (ne_of_lt (mul_pos (f.magRat_pos _ hz.1) (f.magRat_pos _ hz.2)))
    exact ⟨false, withSign_roundMag_eq .rn false false hq hs ha⟩

/-- **IEEE multiplication.** -/
theorem mul_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toRat * y.toRat| ≤ f.maxVal) :
    (mul x y).isFinite = true ∧
      |(mul x y).toRat - x.toRat * y.toRat| ≤ |x.toRat * y.toRat| * f.u + f.eta := by
  obtain ⟨z, hz⟩ := mul_eq_roundRat hx hy
  unfold mul; rw [hz]
  exact ofPat_roundRat_err hf false z hmax

/-- **IEEE division** by a nonzero value. -/
theorem divRn_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hy0 : f.isZero y.pat = false) (hmax : |x.toRat / y.toRat| ≤ f.maxVal) :
    (divRn x y).isFinite = true ∧
      |(divRn x y).toRat - x.toRat / y.toRat| ≤ |x.toRat / y.toRat| * f.u + f.eta := by
  obtain ⟨hnx, hix⟩ := isFinite_iff.mp hx
  obtain ⟨hny, hiy⟩ := isFinite_iff.mp hy
  suffices h : ∃ z, f.div .rn f.nanMag x.pat y.pat = f.roundRat .rn false z (x.toRat / y.toRat) by
    obtain ⟨z, hz⟩ := h
    unfold divRn; rw [hz]
    exact ofPat_roundRat_err hf false z hmax
  unfold Format.div
  simp only [hnx, hny, hix, hiy, hy0, Bool.or_self, Bool.false_eq_true, ite_false]
  split_ifs with hz
  · refine ⟨f.sign x.pat != f.sign y.pat, ?_⟩
    rw [toRat_of_isZero hz, zero_div, roundRat_zero]
  · have hz' : f.isZero x.pat = false := by simpa using hz
    obtain ⟨hs, ha⟩ := sign_mul_lt hz' hy0 (op := (· / ·))
      (fun a b ha hb => ⟨div_pos ha hb, by ring, by rw [div_neg], by rw [neg_div_neg_eq]⟩)
    have hq : x.toRat / y.toRat ≠ 0 := by
      intro h; rw [h, abs_zero] at ha
      exact absurd ha (ne_of_lt (div_pos (f.magRat_pos _ hz') (f.magRat_pos _ hy0)))
    exact ⟨false, withSign_roundMag_eq .rn false false hq hs ha⟩

/-- **Fused multiply-add**: one rounding of `x·y + w`. -/
theorem fma_err (hf : f.Regular) {x y w : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hw : w.isFinite = true) (hmax : |x.toRat * y.toRat + w.toRat| ≤ f.maxVal) :
    (fma x y w).isFinite = true ∧
      |(fma x y w).toRat - (x.toRat * y.toRat + w.toRat)| ≤
        |x.toRat * y.toRat + w.toRat| * f.u + f.eta := by
  obtain ⟨hnx, hix⟩ := isFinite_iff.mp hx
  obtain ⟨hny, hiy⟩ := isFinite_iff.mp hy
  obtain ⟨hnw, hiw⟩ := isFinite_iff.mp hw
  have hu := u_nonneg (f := f)
  have he := eta_nonneg (f := f)
  unfold fma Format.fma
  simp only [hnx, hny, hnw, hix, hiy, hiw, Bool.or_self, Bool.false_eq_true, ite_false]
  by_cases hz : (f.isZero x.pat || f.isZero y.pat) = true
  · rw [if_pos hz]
    have h0 := toRat_mul_of_isZero hz
    by_cases hzw : f.isZero w.pat = true
    · -- `0 · y + 0` or `x · 0 + 0`: a signed zero
      rw [if_pos hzw]
      have hq : x.toRat * y.toRat + w.toRat = 0 := by rw [h0, toRat_of_isZero hzw, add_zero]
      rw [signedZero_eq, ← hq]
      exact ofPat_roundRat_err hf false _ hmax
    · -- a zero product: exactly `w`
      rw [if_neg hzw, ofPat_pat, h0, zero_add, sub_self, abs_zero]
      exact ⟨hw, by positivity⟩
  · rw [if_neg hz]
    exact ofPat_roundRat_err hf false _ hmax

/-- **Conversion** between formats, rounding to nearest: finite and within half an ulp of the
target format when the value is in its range. -/
theorem cvt_err {g : Format} (hg : g.Regular) {x : FloatV f} (hx : x.isFinite = true)
    (hmax : |x.toRat| ≤ g.maxVal) :
    (cvt x : FloatV g).isFinite = true ∧
      |(cvt x : FloatV g).toRat - x.toRat| ≤ |x.toRat| * g.u + g.eta := by
  obtain ⟨hn, hi⟩ := isFinite_iff.mp hx
  have hlt : Format.cvt f g .rn false .canonical x.pat < 2 ^ g.width := by
    unfold Format.cvt
    simp only [hn, hi, Bool.false_eq_true, ite_false]
    split_ifs with hz
    · exact withSign_lt_two_pow _ signBit_pos
    · have hz' : f.isZero x.pat = false := by simpa using hz
      have hq := f.magRat_pos _ hz'
      have hm : f.magRat x.pat ≤ (g.maxSig : ℚ) * 2 ^ (g.emax - g.m) := by
        rw [← f.abs_toRat]; exact hmax
      exact withSign_lt_two_pow _ (g.roundMag_rn_err false _ hq hg.m_pos hg.range hm).1
  obtain ⟨hfin, herr⟩ := cvt_rn_err f g false .canonical x.pat hx hg.m_pos hg.range hmax
  unfold cvt
  exact ⟨by rw [isFinite_ofPat hlt]; exact hfin, by rw [toRat_ofPat hlt]; exact herr⟩

end TVA.FloatV

namespace TVA.Math.Float.Format

/-! ## The standard formats are regular -/

theorem binary16_regular : binary16.Regular := ⟨by decide, by decide⟩
theorem bfloat16_regular : bfloat16.Regular := ⟨by decide, by decide⟩
theorem binary32_regular : binary32.Regular := ⟨by decide, by decide⟩
theorem binary64_regular : binary64.Regular := ⟨by decide, by decide⟩
theorem e5m2_regular : e5m2.Regular := ⟨by decide, by decide⟩
theorem e4m3_regular : e4m3.Regular := ⟨by decide, by decide⟩

theorem binary16_maxVal : binary16.maxVal = 65504 := binary16_max
theorem bfloat16_maxVal : bfloat16.maxVal = 255 * 2 ^ 120 := bfloat16_max
theorem e4m3_maxVal : e4m3.maxVal = 448 := e4m3_max
theorem e5m2_maxVal : e5m2.maxVal = 57344 := e5m2_max

end TVA.Math.Float.Format

/-! ## Negation flips the sign bit -/

namespace TVA.Math.Float.Format

variable {f : Format}

theorem xor_two_pow_of {x k : ℕ} (hx : x < 2 ^ (k + 1)) :
    x ^^^ 2 ^ k = if x.testBit k then x % 2 ^ k else x + 2 ^ k := by
  have hhi : ∀ i, k < i → x.testBit i = false := fun i hi =>
    Nat.testBit_lt_two_pow (lt_of_lt_of_le hx (Nat.pow_le_pow_right (by norm_num) hi))
  split_ifs with hk
  · apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_xor, Nat.testBit_two_pow, Nat.testBit_mod_two_pow]
    rcases lt_trichotomy i k with h | rfl | h
    · simp [h, Nat.ne_of_gt h]
    · simp [hk]
    · simp [hhi i h, Nat.ne_of_lt h, Nat.not_lt.mpr h.le]
  · have hlt : x < 2 ^ k := Nat.lt_pow_two_of_testBit x fun i hi => by
      rcases Nat.eq_or_lt_of_le hi with rfl | h
      · simpa using hk
      · exact hhi i h
    rw [← or_two_pow_eq x k hlt]
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_xor, Nat.testBit_or, Nat.testBit_two_pow]
    by_cases h : k = i
    · subst h; simp [hk]
    · simp [h]

theorem sign_eq_testBit (x : ℕ) : f.sign x = x.testBit (f.e + f.m) := rfl

/-- A pattern is its sign bit over its magnitude. -/
theorem self_eq_withSign {x : ℕ} (hx : x < 2 ^ f.width) :
    x = f.withSign (f.sign x) (x % f.signBit) := by
  have hm : x % f.signBit < f.signBit := Nat.mod_lt _ signBit_pos
  rw [f.withSign_of_lt _ hm]
  have hx' : x < 2 ^ (f.e + f.m + 1) := hx
  have hsb : f.signBit = 2 ^ (f.e + f.m) := rfl
  split_ifs with hs
  · rw [sign_eq_testBit] at hs
    have := xor_two_pow_of hx'
    rw [if_pos hs] at this
    have h2 : x % 2 ^ (f.e + f.m) + 2 ^ (f.e + f.m) = x := by
      have hd := Nat.div_add_mod x (2 ^ (f.e + f.m))
      have hq : x / 2 ^ (f.e + f.m) = 1 := by
        have h1 : x / 2 ^ (f.e + f.m) < 2 := by
          rw [Nat.div_lt_iff_lt_mul (by positivity)]; rw [pow_succ] at hx'; linarith
        have h0 : x / 2 ^ (f.e + f.m) ≠ 0 := by
          intro h0
          have : x < 2 ^ (f.e + f.m) := by
            rwa [Nat.div_eq_zero_iff_lt (by positivity)] at h0
          rw [Nat.testBit_lt_two_pow this] at hs; cases hs
        exact Nat.le_antisymm (Nat.lt_succ_iff.mp h1) (Nat.one_le_iff_ne_zero.mpr h0)
      rw [hq, mul_one] at hd; omega
    rw [hsb]; exact h2.symm
  · rw [sign_eq_testBit] at hs
    have hlt : x < 2 ^ (f.e + f.m) := Nat.lt_pow_two_of_testBit x fun i hi => by
      rcases Nat.eq_or_lt_of_le hi with rfl | h
      · simpa using hs
      · exact Nat.testBit_lt_two_pow (lt_of_lt_of_le hx' (Nat.pow_le_pow_right (by norm_num) h))
    rw [hsb, Nat.mod_eq_of_lt hlt]

/-- Negation flips the sign over the same magnitude. -/
theorem negate_eq_withSign {x : ℕ} (hx : x < 2 ^ f.width) :
    f.negate x = f.withSign (!f.sign x) (x % f.signBit) := by
  have hm : x % f.signBit < f.signBit := Nat.mod_lt _ signBit_pos
  rw [f.withSign_of_lt _ hm]
  have hx' : x < 2 ^ (f.e + f.m + 1) := hx
  have hsb : f.signBit = 2 ^ (f.e + f.m) := rfl
  unfold negate
  rw [hsb, xor_two_pow_of hx', ← sign_eq_testBit]
  cases hs : f.sign x
  · simp only [Bool.false_eq_true, ite_false, Bool.not_false, ite_true]
    rw [sign_eq_testBit] at hs
    have hlt : x < 2 ^ (f.e + f.m) := Nat.lt_pow_two_of_testBit x fun i hi => by
      rcases Nat.eq_or_lt_of_le hi with rfl | h
      · simpa using hs
      · exact Nat.testBit_lt_two_pow (lt_of_lt_of_le hx' (Nat.pow_le_pow_right (by norm_num) h))
    rw [Nat.mod_eq_of_lt hlt]
  · simp

theorem negate_lt {x : ℕ} (hx : x < 2 ^ f.width) : f.negate x < 2 ^ f.width := by
  rw [negate_eq_withSign hx]; exact withSign_lt_two_pow _ (Nat.mod_lt _ signBit_pos)

theorem toRat_negate {x : ℕ} (hx : x < 2 ^ f.width) : f.toRat (f.negate x) = -f.toRat x := by
  have hm : x % f.signBit < f.signBit := Nat.mod_lt _ signBit_pos
  conv_rhs => rw [self_eq_withSign hx]
  rw [negate_eq_withSign hx, f.toRat_withSign _ hm, f.toRat_withSign _ hm]
  cases f.sign x <;> simp

theorem isFinite_negate {x : ℕ} (hx : x < 2 ^ f.width) : f.isFinite (f.negate x) = f.isFinite x := by
  have hm : x % f.signBit < f.signBit := Nat.mod_lt _ signBit_pos
  conv_rhs => rw [self_eq_withSign hx]
  rw [negate_eq_withSign hx, f.isFinite_withSign _ hm, f.isFinite_withSign _ hm]

end TVA.Math.Float.Format

namespace TVA.FloatV

open TVA.Math.Float TVA.Math.Float.Format

variable {f : Format}

theorem neg_pat (x : FloatV f) : (neg x).pat = f.negate x.pat := pat_ofPat (negate_lt x.pat_lt)

@[simp] theorem toRat_neg (x : FloatV f) : (neg x).toRat = -x.toRat := by
  rw [toRat_def, neg_pat, toRat_negate x.pat_lt]; rfl

@[simp] theorem isFinite_neg (x : FloatV f) : (neg x).isFinite = x.isFinite := by
  simp only [isFinite, neg_pat, isFinite_negate x.pat_lt]

theorem sub_eq_add_neg (x y : FloatV f) : sub x y = add x (neg y) := by
  unfold sub add Format.sub; rw [neg_pat]

/-- **IEEE subtraction.** -/
theorem sub_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toRat - y.toRat| ≤ f.maxVal) :
    (sub x y).isFinite = true ∧
      |(sub x y).toRat - (x.toRat - y.toRat)| ≤ |x.toRat - y.toRat| * f.u + f.eta := by
  rw [sub_eq_add_neg]
  have h := add_err hf hx (by rw [isFinite_neg]; exact hy)
    (by rw [toRat_neg, ← _root_.sub_eq_add_neg]; exact hmax)
  rw [toRat_neg, ← _root_.sub_eq_add_neg] at h
  exact h

end TVA.FloatV
