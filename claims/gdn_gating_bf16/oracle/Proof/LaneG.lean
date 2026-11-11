import Claim
import TVA.Proof
import Proof.GMain
import Proof.GLow
import Proof.Softplus
import Proof.SoftFalse
import Proof.SoftNeg
import Proof.FloatStep
import Proof.NanFree
import Proof.Rnd
import Proof.ExpRoom
import Proof.LaneAux

/-!
# The value stored in `g`, per lane

One lane of the `g` chain: widen both `bfloat16` inputs, add them (`float32`), apply the
`tl.where`-style comparison and the softplus (`Proof/Softplus.lean`, `Proof/SoftFalse.lean`,
`Proof/SoftNeg.lean`), and multiply by `-tl.exp A_log`.
-/

set_option maxHeartbeats 4000000

namespace Claims.GdnGating

open TVA

/-- `E` is positive when it is far from zero and near `exp A` with `|A| ≤ 89`. -/
theorem E_pos_of_abs_ge {A E : ℝ} (hA : |A| ≤ 89)
    (hE : |E - Real.exp A| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A + 2 ^ (-140 : ℤ))
    (habs : (1 : ℝ) / 8 ≤ |E|) : 0 < E := by
  have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have h23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
  rw [h22, h23] at hE
  have h140le : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 / 1000 := by norm_num
  have hepos : 0 < Real.exp A := Real.exp_pos A
  have hc : (1 / 4194304 + 1 / 8388608 * |A|) ≤ 1 / 2 := by
    have h := mul_le_mul_of_nonneg_left hA (by norm_num : (0:ℝ) ≤ 1 / 8388608)
    linarith only [h]
  by_contra hcon
  push Not at hcon
  have hEneg : E - Real.exp A < 0 := by linarith only [hcon, hepos]
  have h1 : Real.exp A ≤ |E - Real.exp A| := by
    rw [abs_of_neg hEneg]
    linarith only [hcon]
  have h3 := mul_le_mul_of_nonneg_right hc (Real.exp_nonneg A)
  have h2 : Real.exp A ≤ (1 / 2) * Real.exp A + 2 ^ (-140 : ℤ) := by
    linarith only [hE, h1, h3]
  have hexpA : Real.exp A ≤ 2 / 1000 := by linarith only [h2, h140le]
  have h5 := (abs_le.mp hE).1
  have h7 : (1 / 4194304 + 1 / 8388608 * |A|) * Real.exp A + 2 ^ (-140 : ℤ) ≤ 1 / 500 := by
    have h6 : (1 / 4194304 + 1 / 8388608 * |A|) * Real.exp A ≤ (1 / 2) * (2 / 1000) :=
      mul_le_mul hc hexpA (Real.exp_nonneg A) (by norm_num)
    linarith only [h6, h140le]
  have h9 : -(1 / 500 : ℝ) ≤ E := by linarith only [h5, h7, Real.exp_nonneg A]
  rcases le_abs.mp habs with h | h
  · linarith only [h, hcon]
  · linarith only [h, h9]

/-- For `A ≤ -89` a value within the exponential's error of `exp A` is tiny. -/
theorem E_abs_small_low {A E : ℝ} (hA : A ≤ -89)
    (hE : |E - Real.exp A| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A + 2 ^ (-140 : ℤ)) :
    |E| ≤ 1 / 8 := by
  have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have h23 : (2 : ℝ) ^ (-23 : ℤ) = 1 / 8388608 := by norm_num
  rw [h22, h23] at hE
  have h128 : (2 : ℝ) ^ (-128 : ℤ) ≤ 1 / 1000000 := by norm_num
  have h140le : (2 : ℝ) ^ (-140 : ℤ) ≤ 1 / 1000 := by norm_num
  have hepos : 0 < Real.exp A := Real.exp_pos A
  have hlog : A ≤ ((-128 : ℤ) : ℝ) * Real.log 2 := by
    push_cast
    nlinarith [Real.log_two_lt_d9]
  have h1 : Real.exp A ≤ (2 : ℝ) ^ (-128 : ℤ) := glow_exp_le_pow hlog
  have h2 : |A| * Real.exp A ≤ 1 := abs_mul_exp_le_one (by linarith only [hA])
  have h3 : |E| ≤ |E - Real.exp A| + Real.exp A := by
    have h : |E| = |(E - Real.exp A) + Real.exp A| := congrArg abs (by ring)
    rw [h]
    simpa [abs_of_pos hepos] using abs_add_le (E - Real.exp A) (Real.exp A)
  have h4 : (1 / 8388608) * (|A| * Real.exp A) ≤ 1 / 8388608 :=
    mul_le_of_le_one_right (by norm_num) h2
  have h6 : (1 / 4194304) * Real.exp A ≤ (1 / 4194304) * (1 / 1000000) := by
    have h5 : Real.exp A ≤ 1 / 1000000 := le_trans h1 h128
    exact mul_le_mul_of_nonneg_left h5 (by norm_num)
  nlinarith only [hE, h3, h4, h6, h140le, Real.exp_nonneg A, hepos, h1, h2]


/-- add error without room -/
theorem add_err_any {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hfin : (FloatV.add x y).isFinite = true) :
    |(FloatV.add x y).toReal - (x.toReal + y.toReal)| ≤
      |x.toReal + y.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
  by_cases hroom : |x.toRat + y.toRat| ≤ Format.binary32.maxVal
  · obtain ⟨-, herr⟩ := add_exact_err_fin hx hy hroom (FloatV.NaNEq.refl _)
    simpa using herr
  · push Not at hroom
    have hpat : (FloatV.add x y).pat = Format.binary32.roundRat RoundMode.rn false
        (Format.binary32.sign x.pat && Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat + Format.binary32.toRat y.pat) := by
      unfold FloatV.add
      rw [Format.add_of_finite RoundMode.rn hx hy]
      exact FloatV.pat_ofPat (roundRat_lt_width RoundMode.rn false _ _)
    have hfin' : Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false
        (Format.binary32.sign x.pat && Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat + Format.binary32.toRat y.pat)) = true := by
      rw [← hpat]
      exact hfin
    have hrw : (FloatV.add x y).toRat = Format.binary32.toRat (Format.binary32.roundRat
        RoundMode.rn false (Format.binary32.sign x.pat && Format.binary32.sign y.pat)
        (Format.binary32.toRat x.pat + Format.binary32.toRat y.pat)) := by
      have h : (FloatV.add x y).toRat = Format.binary32.toRat (FloatV.add x y).pat := rfl
      rw [h, hpat]
    have hrat := roundRat_rn_fin_err hfin'
    rw [← hrw] at hrat
    have key := rat_abs_le_mul_add_cast (a := (FloatV.add x y).toRat)
      (b := x.toRat + y.toRat) (u := 2 ^ (-24 : ℤ)) (e := 2 ^ (-150 : ℤ)) hrat
    simpa [FloatV.toReal] using key

/-- A positive-value overflow pushes `|E|` above `1/8`. -/
theorem E_abs_ge_of_big {E S : ℝ} (hS : |S| ≤ (2 : ℝ) ^ 130)
    (hbig : (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤ |E * S|) :
    (1 : ℝ) / 8 ≤ |E| := by
  have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  rw [h22] at hbig
  rw [abs_mul] at hbig
  have h1 : |E| * |S| ≤ |E| * (2 : ℝ) ^ 130 := mul_le_mul_of_nonneg_left hS (abs_nonneg E)
  have h2 : (1 - 1 / 4194304) * (Format.binary32.maxVal : ℝ) ≤ |E| * (2 : ℝ) ^ 130 :=
    hbig.trans h1
  have h3 : (0 : ℝ) < (2 : ℝ) ^ 130 := by norm_num
  have h4 : ((1 - 1 / 4194304) * (Format.binary32.maxVal : ℝ)) / (2 : ℝ) ^ 130 ≤ |E| := by
    rw [div_le_iff₀ h3]
    linarith only [h2]
  have h5 : (1 / 8 : ℝ) ≤
      ((1 - 1 / 4194304) * (Format.binary32.maxVal : ℝ)) / (2 : ℝ) ^ 130 := by
    rw [le_div_iff₀ h3]
    rw [Format.binary32_maxVal]
    norm_num
  linarith only [h4, h5]

/-- `spRef` stays below `2^129` below `2^129`. -/
theorem spRef_le {x : ℝ} (h : x ≤ (2 : ℝ) ^ 129) : spRef x ≤ (2 : ℝ) ^ 129 := by
  unfold spRef
  by_cases hx : x ≤ 20
  · rw [ite_eq_left hx]
    have h1 : 1 + Real.exp x ≤ Real.exp 21 := by
      have h2 : Real.exp x ≤ Real.exp 20 := Real.exp_le_exp.mpr hx
      have h3 : (1 : ℝ) ≤ Real.exp 20 := Real.one_le_exp (by norm_num)
      have h4 : (2 : ℝ) ≤ Real.exp 1 := by
        have h5 := Real.add_one_le_exp (1 : ℝ)
        linarith
      have h6 : Real.exp 21 = Real.exp 1 * Real.exp 20 := by
        rw [← Real.exp_add]
        norm_num
      nlinarith [h2, h3, h4, h6, Real.exp_pos 20, Real.exp_pos x]
    have h7 : Real.log (1 + Real.exp x) ≤ 21 := by
      have h8 : Real.log (1 + Real.exp x) ≤ Real.log (Real.exp 21) :=
        Real.log_le_log (by positivity) h1
      rwa [Real.log_exp] at h8
    have h9 : (21 : ℝ) ≤ (2 : ℝ) ^ 129 := by norm_num
    linarith only [h7, h9]
  · rw [ite_eq_right hx]
    linarith only [h, not_le.mp hx]

/-- An overflowing finite sum whose exact value is below the threshold is negative. -/
theorem sum_neg_big_of_notFinite {xa xb r : F32} (hxa : xa.isFinite = true)
    (hxb : xb.isFinite = true) (h : FloatV.NaNEq (FloatV.add xa xb) r)
    (hr : r.isFinite = false)
    (hub : xa.toReal + xb.toReal < (2 : ℝ) ^ (128 : ℤ) - (2 : ℝ) ^ (103 : ℤ)) :
    xa.toReal + xb.toReal < -(Format.binary32.maxVal : ℝ) := by
  set q : ℚ := Format.binary32.toRat xa.pat + Format.binary32.toRat xb.pat with hq_def
  have hqreal : ((q : ℚ) : ℝ) = xa.toReal + xb.toReal := by
    simp only [hq_def, FloatV.toReal, FloatV.toRat]
    push_cast
    ring
  have hpat : (FloatV.add xa xb).pat = Format.binary32.roundRat RoundMode.rn false
      (Format.binary32.sign xa.pat && Format.binary32.sign xb.pat) q := by
    unfold FloatV.add
    rw [Format.add_of_finite RoundMode.rn hxa hxb]
    exact FloatV.pat_ofPat (roundRat_lt_width RoundMode.rn false _ _)
  have hnf : Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false
      (Format.binary32.sign xa.pat && Format.binary32.sign xb.pat) q) = false := by
    rw [← hpat]
    exact naneq_notFinite_left h hr
  have habs : Format.binary32.maxVal < |q| := roundRat_rn_not_finite_gt_maxVal hnf
  have hqne : q ≠ 0 := by
    intro h0
    rw [h0, abs_zero] at habs
    exact absurd habs (not_lt.mpr (le_of_lt (Format.maxVal_pos Format.binary32_regular)))
  have hqnonpos : q ≤ 0 := by
    by_contra hcon
    push Not at hcon
    have hth := roundRat_rn_overflow_threshold hcon hnf
    have hubq : q < (2 : ℚ) ^ (128 : ℤ) - (2 : ℚ) ^ (103 : ℤ) := by
      have h' : ((q : ℚ) : ℝ) < (((2 : ℚ) ^ (128 : ℤ) - (2 : ℚ) ^ (103 : ℤ) : ℚ) : ℝ) := by
        rw [hqreal]
        push_cast
        linarith only [hub]
      exact_mod_cast h'
    linarith only [hth, hubq]
  have hqneg : q < 0 := lt_of_le_of_ne hqnonpos hqne
  have h1 : |q| = -q := abs_of_neg hqneg
  rw [h1] at habs
  have h2 : ((Format.binary32.maxVal : ℚ) : ℝ) < -(q : ℝ) := by
    exact_mod_cast habs
  rw [hqreal] at h2
  linarith only [h2]



/-- `oneF32` has real value `1`. -/
theorem oneF32_toReal : oneF32.toReal = 1 := by
  simp only [FloatV.toReal, oneF32_toRat]
  norm_num

/-- If `|1 + e| ≤ maxVal`, the exact sum `oneF32 + e` is within `float32`'s range. -/
theorem one_add_room {e : F32} (h : |1 + e.toReal| ≤ (Format.binary32.maxVal : ℝ)) :
    |oneF32.toRat + e.toRat| ≤ Format.binary32.maxVal := by
  have hcast : ((|oneF32.toRat + e.toRat| : ℚ) : ℝ) = |1 + e.toReal| := by
    rw [Rat.cast_abs, Rat.cast_add, oneF32_toRat]
    simp only [FloatV.toReal]
    norm_num
  have h2 : ((|oneF32.toRat + e.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [hcast]
    exact h
  exact_mod_cast h2

/-- A result equal up to NaN payload to `-∞` is `-∞`. -/
theorem eq_negInf_naneq {z r : F32} (hz : z = (FloatV.negInf : F32))
    (h : FloatV.NaNEq z r) : r = (FloatV.negInf : F32) := by
  subst hz
  have hnan : (FloatV.negInf : F32).isNaN = false :=
    FloatV.negInf_isNaN (show Format.binary32.specials = Format.Specials.ieee from rfl)
  rcases h with h | ⟨h1, h2⟩
  · exact h.symm
  · rw [hnan] at h1
    exact Bool.noConfusion h1




/-- **The tail of the `g` lane**: given the exponential `E` of `A_log` within its device bound and a softplus value `sp` within the softplus tolerance, the product `-E · sp` is a finite value within `gTol`, or `-∞` where `maxVal ≤ |gRef| + gTol`. -/
theorem lane_tail {A x0 E : ℝ} {neA sp gg : F32}
    (hroomA : ExpRoom A)
    (hE : |E - Real.exp A| ≤
      (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A + 2 ^ (-140 : ℤ))
    (hneAfin : neA.isFinite = true) (hneA : neA.toReal = -E)
    (hspfin : sp.isFinite = true)
    (hS : |sp.toReal - spRef x0| ≤ 52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)))
    (hx0 : |x0| ≤ (2 : ℝ) ^ 129)
    (hgg : FloatV.NaNEq (FloatV.mul neA sp) gg) :
    (gg.isFinite = true ∧ |gg.toReal - gRef A x0| ≤ gTol A x0) ∨
      (gg = (FloatV.negInf : F32) ∧
        (Format.binary32.maxVal : ℝ) ≤ |gRef A x0| + gTol A x0) := by
  have hA89 : A ≤ 89 := expRoom_le hroomA
  have hSle : |sp.toReal| ≤ (2 : ℝ) ^ 130 := by
    have h1 : |sp.toReal| ≤ |sp.toReal - spRef x0| + spRef x0 := by
      have h : |sp.toReal| = |(sp.toReal - spRef x0) + spRef x0| := congrArg abs (by ring)
      rw [h]
      have h2 := abs_add_le (sp.toReal - spRef x0) (spRef x0)
      rwa [abs_of_pos (spRef_pos x0)] at h2
    have h4 : spRef x0 ≤ (2 : ℝ) ^ 129 := spRef_le (le_trans (le_abs_self x0) hx0)
    have h2 : |sp.toReal - spRef x0| + spRef x0 ≤
        52 * 2 ^ (-23 : ℤ) * spRef x0 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) + (2 : ℝ) ^ 129 := by
      linarith only [hS, h4]
    have h7 : (0:ℝ) ≤ 52 * 2 ^ (-23 : ℤ) := by norm_num
    have h5 : 52 * 2 ^ (-23 : ℤ) * spRef x0 ≤ 52 * 2 ^ (-23 : ℤ) * (2 : ℝ) ^ 129 :=
      mul_le_mul_of_nonneg_left h4 h7
    have h8 : 52 * 2 ^ (-23 : ℤ) * (2 : ℝ) ^ 129 + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) +
        (2 : ℝ) ^ 129 ≤ (2 : ℝ) ^ 130 := by norm_num
    linarith only [h1, h2, h5, h8]
  by_cases hfin : (FloatV.mul neA sp).isFinite = true
  · have herr := mul_err_any hneAfin hspfin hfin
    have hggeq : gg = FloatV.mul neA sp := FloatV.NaNEq.eq_of_isFinite hgg hfin
    have hggfin : gg.isFinite = true := fin_of_naneq_fin hgg hfin
    have hgv : |gg.toReal + E * sp.toReal| ≤
        2 ^ (-24 : ℤ) * |E * sp.toReal| + 2 ^ (-150 : ℤ) := by
      have h1 : neA.toReal * sp.toReal = -(E * sp.toReal) := by rw [hneA]; ring
      rw [h1, abs_neg] at herr
      rw [hggeq]
      have h2 : (FloatV.mul neA sp).toReal - -(E * sp.toReal) =
          (FloatV.mul neA sp).toReal + E * sp.toReal := by ring
      rw [h2] at herr
      calc |(FloatV.mul neA sp).toReal + E * sp.toReal|
          ≤ |E * sp.toReal| * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := herr
        _ = 2 ^ (-24 : ℤ) * |E * sp.toReal| + 2 ^ (-150 : ℤ) := by ring
    by_cases hAlow : A ≤ -89
    · exact Or.inl ⟨hggfin, g_bound_low hAlow hE hS hgv⟩
    · push Not at hAlow
      have hAabs : |A| ≤ 89 := abs_le.mpr ⟨by linarith, hA89⟩
      exact Or.inl ⟨hggfin, g_bound hAabs hE hS hgv⟩
  · have hnf : (FloatV.mul neA sp).isFinite = false := by
      cases hm : (FloatV.mul neA sp).isFinite with
      | false => rfl
      | true => exact absurd hm hfin
    have hbig0 := mul_big_fin hneAfin hspfin hnf
    have hbig : (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) ≤ |E * sp.toReal| := by
      have h1 : neA.toReal * sp.toReal = -(E * sp.toReal) := by rw [hneA]; ring
      rw [h1, abs_neg] at hbig0
      exact hbig0
    by_cases hAlow : A ≤ -89
    · exfalso
      have hEabs := E_abs_small_low hAlow hE
      have h1 : |E * sp.toReal| ≤ (1/8) * (2 : ℝ) ^ 130 := by
        rw [abs_mul]
        exact mul_le_mul hEabs hSle (abs_nonneg _) (by norm_num)
      have h2 : (1/8 : ℝ) * (2 : ℝ) ^ 130 < (1 - 2 ^ (-22 : ℤ)) * (Format.binary32.maxVal : ℝ) := by
        have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
        rw [h22, binary32_maxVal_real]
        norm_num
      exact absurd (le_trans hbig h1) (not_le.mpr h2)
    · push Not at hAlow
      have hAabs : |A| ≤ 89 := abs_le.mpr ⟨by linarith, hA89⟩
      have habs : (1 : ℝ) / 8 ≤ |E| := E_abs_ge_of_big hSle hbig
      have hEpos : 0 < E := E_pos_of_abs_ge hAabs hE habs
      have hEM : E ≤ (Format.binary32.maxVal : ℝ) := by
        have hroom := hroomA
        unfold ExpRoom at hroom
        have h1 := (abs_le.mp hE).2
        have h3 : Real.exp A + ((2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A +
            2 ^ (-140 : ℤ)) ≤ (Format.binary32.maxVal : ℝ) := by
          rw [show Real.exp A + ((2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|) * Real.exp A +
              2 ^ (-140 : ℤ)) = (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |A|)) * Real.exp A +
              2 ^ (-140 : ℤ) by ring]
          exact hroom
        linarith only [h1, h3]
      have habsS : (1 : ℝ) / 2 ≤ |sp.toReal| := S_abs_ge_half hEpos hEM hbig
      have hSpos : 0 < sp.toReal :=
        S_pos_of_big_of_bound (u := (2:ℝ)^(-23:ℤ)) (v := (2:ℝ)^(-24:ℤ)) (w := (2:ℝ)^(-19:ℤ))
          rfl rfl (spRef_pos x0) hS (by norm_num) habsS
      have hneg : neA.toReal * sp.toReal < 0 := by rw [hneA]; nlinarith [hEpos, hSpos]
      have hggnf : gg.isFinite = false := by
        rcases hgg with h | ⟨h1, h2⟩
        · rw [← h]; exact hnf
        · exfalso
          rw [mul_isNaN_false hneAfin hspfin] at h1
          exact Bool.noConfusion h1
      have hgginf : gg = (FloatV.negInf : F32) := mul_negInf_fin hneAfin hspfin hgg hggnf hneg
      exact Or.inr ⟨hgginf, g_inf hAabs hE hS hbig⟩



/-- **One lane of `g`**: for finite inputs `A_log`, `a`, `dt_bias` meeting `Pre`, every value the chain `add, mul, exp, add, log, mul, where, exp, neg, mul` can produce satisfies the `g` clause of `Post`. -/
theorem lane_g {P : Platform} (hexp : Device.ExpF32 P) (hlog : Device.LogF32 P) (hninf : Device.ExpF32NegInf P)
    {A : F32} {av bv : BF16} (hAfin : A.isFinite = true) (hroomA : ExpRoom A.toReal)
    (hav : av.isFinite = true) (hbv : bv.isFinite = true)
    (hxub : av.toReal + bv.toReal < (2 : ℝ) ^ (128 : ℤ) - (2 : ℝ) ^ (103 : ℤ))
    {xa xb x t1 t2 e s l l2 sp eA neA gg : F32} {cond : Bool}
    (hxa : FloatV.NaNEq (FloatV.cvt (g := Format.binary32) av) xa)
    (hxb : FloatV.NaNEq (FloatV.cvt (g := Format.binary32) bv) xb)
    (hx : FloatV.NaNEq (FloatV.add xa xb) x)
    (ht1 : FloatV.NaNEq (FloatV.mul oneF32 x) t1)
    (hc : cond = FloatV.le t1 twentyF32)
    (ht2 : FloatV.NaNEq (FloatV.mul oneF32 x) t2)
    (he : P.math .exp .binary32 t2 e)
    (hsn : FloatV.NaNEq (FloatV.add oneF32 e) s)
    (hl : P.math .log .binary32 s l)
    (hl2 : FloatV.NaNEq (FloatV.mul oneF32 l) l2)
    (hsp : sp = if cond then l2 else x)
    (heA : P.math .exp .binary32 A eA)
    (hneA : FloatV.NaNEq (FloatV.neg eA) neA)
    (hgg : FloatV.NaNEq (FloatV.mul neA sp) gg) :
    (gg.isFinite = true ∧
        |gg.toReal - gRef A.toReal (av.toReal + bv.toReal)| ≤
          gTol A.toReal (av.toReal + bv.toReal)) ∨
      (gg = (FloatV.negInf : F32) ∧
        (Format.binary32.maxVal : ℝ) ≤
          |gRef A.toReal (av.toReal + bv.toReal)| + gTol A.toReal (av.toReal + bv.toReal)) := by
  -- the widening of the two `bfloat16` inputs is exact
  obtain ⟨hxafin0, hxaval0⟩ := cvt_up hav
  obtain ⟨hxbfin0, hxbval0⟩ := cvt_up hbv
  have hxaeq : xa = FloatV.cvt (g := Format.binary32) av := FloatV.NaNEq.eq_of_isFinite hxa hxafin0
  have hxbeq : xb = FloatV.cvt (g := Format.binary32) bv := FloatV.NaNEq.eq_of_isFinite hxb hxbfin0
  have hxafin : xa.isFinite = true := by rw [hxaeq]; exact hxafin0
  have hxbfin : xb.isFinite = true := by rw [hxbeq]; exact hxbfin0
  have hxaval : xa.toReal = av.toReal := by rw [hxaeq]; exact hxaval0
  have hxbval : xb.toReal = bv.toReal := by rw [hxbeq]; exact hxbval0
  have hx0abs : |av.toReal + bv.toReal| ≤ (2 : ℝ) ^ 129 :=
    abs_x_le (bf16_sum_abs_le (bf16_abs_toReal_le hav) (bf16_abs_toReal_le hbv))
  -- `tl.exp A` and its negation, needed in both branches
  obtain ⟨heAfin, heAerr⟩ := exp_apply hexp heA hAfin hroomA
  obtain ⟨hneAfin, hneAval⟩ := neg_exact heAfin hneA
  by_cases hxfin : x.isFinite = true
  · -- ============ `x` finite ============
    have hxeq : FloatV.add xa xb = x := naneq_eq_right hx hxfin
    have hxf : |x.toReal - (av.toReal + bv.toReal)| ≤
        (2 : ℝ) ^ (-24 : ℤ) * |av.toReal + bv.toReal| + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := add_err_any hxafin hxbfin (by rw [hxeq]; exact hxfin)
      rw [hxeq, hxaval, hxbval] at h
      have h2 : |av.toReal + bv.toReal| * 2 ^ (-24 : ℤ) =
          2 ^ (-24 : ℤ) * |av.toReal + bv.toReal| := by ring
      rw [h2] at h
      exact h
    have hxf_le : |x.toReal| ≤ (2 : ℝ) ^ 129 :=
      abs_xf_le (bf16_sum_abs_le (bf16_abs_toReal_le hav) (bf16_abs_toReal_le hbv)) hxf
    obtain ⟨ht1fin, ht1err⟩ := mul_exact_err_fin oneF32_isFinite hxfin (one_mul_room hxfin) ht1
    obtain ⟨ht2fin, ht2err⟩ := mul_exact_err_fin oneF32_isFinite hxfin (one_mul_room hxfin) ht2
    have ht1xf : |t1.toReal - x.toReal| ≤
        (2 : ℝ) ^ (-24 : ℤ) * |x.toReal| + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := ht1err
      rw [oneF32_toReal, one_mul] at h
      have h2 : |x.toReal| * 2 ^ (-24 : ℤ) = 2 ^ (-24 : ℤ) * |x.toReal| := by ring
      rw [h2] at h
      exact h
    have ht2xf : |t2.toReal - x.toReal| ≤
        (2 : ℝ) ^ (-24 : ℤ) * |x.toReal| + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := ht2err
      rw [oneF32_toReal, one_mul] at h
      have h2 : |x.toReal| * 2 ^ (-24 : ℤ) = 2 ^ (-24 : ℤ) * |x.toReal| := by ring
      rw [h2] at h
      exact h
    cases hb : FloatV.le t1 twentyF32 with
    | true =>
      have hcond : cond = true := by rw [hc]; exact hb
      have hle20 : t1.toReal ≤ 20 := le_cond_true ht1fin hb
      have ht2_21 : t2.toReal ≤ 21 := t2_le_21 ht1xf ht2xf hle20
      have ht2absb : |t2.toReal| ≤ (2 : ℝ) ^ 130 := abs_t2_le hxf_le ht2xf
      have hlo : -((10 : ℝ) ^ 40) ≤ t2.toReal := by
        have h1 : -(2 : ℝ) ^ 130 ≤ t2.toReal := by
          have h3 := neg_abs_le t2.toReal
          linarith only [h3, ht2absb]
        have h2 : (2 : ℝ) ^ 130 ≤ (10 : ℝ) ^ 40 := by norm_num
        linarith only [h1, h2]
      have ht2room : ExpRoom t2.toReal := expRoom_of_bounds hlo ht2_21
      obtain ⟨hefin, heerr⟩ := exp_apply hexp he ht2fin ht2room
      have hge : (1 : ℝ) / 2 ≤ 1 + e.toReal := by
        by_cases h0 : t2.toReal ≤ 0
        · have h := exp_err_neg (u := (2:ℝ)^(-22:ℤ)) (v := (2:ℝ)^(-23:ℤ)) (c := (2:ℝ)^(-140:ℤ))
            rfl rfl rfl h0 heerr
          exact h.1
        · push Not at h0
          have h := exp_err_pos (u := (2:ℝ)^(-22:ℤ)) (v := (2:ℝ)^(-23:ℤ)) (c := (2:ℝ)^(-140:ℤ))
            (a := (2:ℝ)^32) rfl rfl rfl rfl (le_of_lt h0) ht2_21 heerr
          linarith only [h.1]
      have hea : e.toReal ≤ (2 : ℝ) ^ 32 := by
        by_cases h0 : t2.toReal ≤ 0
        · have h := exp_err_neg (u := (2:ℝ)^(-22:ℤ)) (v := (2:ℝ)^(-23:ℤ)) (c := (2:ℝ)^(-140:ℤ))
            rfl rfl rfl h0 heerr
          have h2 : (2 : ℝ) ≤ (2 : ℝ) ^ 32 := by norm_num
          linarith only [h.2, h2]
        · push Not at h0
          have h := exp_err_pos (u := (2:ℝ)^(-22:ℤ)) (v := (2:ℝ)^(-23:ℤ)) (c := (2:ℝ)^(-140:ℤ))
            (a := (2:ℝ)^32) rfl rfl rfl rfl (le_of_lt h0) ht2_21 heerr
          exact h.2
      have heabs : |1 + e.toReal| ≤ (Format.binary32.maxVal : ℝ) :=
        one_add_abs_le_maxVal (a := (2:ℝ)^32) rfl hge hea
      obtain ⟨hsfin, hserr⟩ := add_exact_err_fin oneF32_isFinite hefin (one_add_room heabs) hsn
      have hsb : |s.toReal - (1 + e.toReal)| ≤
          (1 + e.toReal) * 2 ^ (-24 : ℤ) + 2 ^ (-150 : ℤ) := by
        have h := hserr
        rw [oneF32_toReal] at h
        have hpos : (0 : ℝ) < 1 + e.toReal := by linarith only [hge]
        rw [abs_of_pos hpos] at h
        exact h
      have hsb2 : |s.toReal - (1 + e.toReal)| ≤
          2 ^ (-24 : ℤ) * (1 + e.toReal) + 2 ^ (-150 : ℤ) := by
        have h2 : (1 + e.toReal) * 2 ^ (-24 : ℤ) = 2 ^ (-24 : ℤ) * (1 + e.toReal) := by ring
        rw [h2] at hsb
        exact hsb
      obtain ⟨hspos, hlogb⟩ := s_pos_and_log_bound (s := s.toReal) (e := e.toReal)
        (a := (2:ℝ)^32) (u := (2:ℝ)^(-24:ℤ)) (w := (2:ℝ)^(-150:ℤ)) rfl rfl hge hea (by norm_num) hsb
      obtain ⟨hlfin, hlerr⟩ := log_apply hlog hl hsfin hspos
        (log_room_bound (a := (2:ℝ)^32) (le_refl _) hlogb)
      obtain ⟨hl2fin, hl2err⟩ := mul_exact_err_fin oneF32_isFinite hlfin (one_mul_room hlfin) hl2
      have hl2xf : |l2.toReal - l.toReal| ≤
          (2 : ℝ) ^ (-24 : ℤ) * |l.toReal| + (2 : ℝ) ^ (-150 : ℤ) := by
        have h := hl2err
        rw [oneF32_toReal, one_mul] at h
        have h2 : |l.toReal| * 2 ^ (-24 : ℤ) = 2 ^ (-24 : ℤ) * |l.toReal| := by ring
        rw [h2] at h
        exact h
      have hSp : |l2.toReal - spRef (av.toReal + bv.toReal)| ≤
          52 * 2 ^ (-23 : ℤ) * spRef (av.toReal + bv.toReal) + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) :=
        softplus_true (x0 := av.toReal + bv.toReal) (xf := x.toReal)
          hxf ht1xf ht2xf hle20 heerr hsb2 hlerr hl2xf
      have hspval : sp = l2 := by rw [hsp, hcond]; rfl
      have hspfin : sp.isFinite = true := by rw [hspval]; exact hl2fin
      have hS : |sp.toReal - spRef (av.toReal + bv.toReal)| ≤
          52 * 2 ^ (-23 : ℤ) * spRef (av.toReal + bv.toReal) + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) := by
        rw [hspval]; exact hSp
      exact lane_tail hroomA heAerr hneAfin hneAval hspfin hS hx0abs hgg
    | false =>
      have hcond : cond = false := by rw [hc]; exact hb
      have hgt : 20 < t1.toReal := le_cond_false ht1fin hb
      have hSp : |x.toReal - spRef (av.toReal + bv.toReal)| ≤
          52 * 2 ^ (-23 : ℤ) * spRef (av.toReal + bv.toReal) + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) :=
        softplus_false (x0 := av.toReal + bv.toReal) (xf := x.toReal) hxf ht1xf hgt
      have hspval : sp = x := by rw [hsp, hcond]; rfl
      have hspfin : sp.isFinite = true := by rw [hspval]; exact hxfin
      have hS : |sp.toReal - spRef (av.toReal + bv.toReal)| ≤
          52 * 2 ^ (-23 : ℤ) * spRef (av.toReal + bv.toReal) + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) := by
        rw [hspval]; exact hSp
      exact lane_tail hroomA heAerr hneAfin hneAval hspfin hS hx0abs hgg
  · -- ============ `x` not finite ============
    have hxfnf : x.isFinite = false := by
      cases h : x.isFinite with
      | false => rfl
      | true => exact absurd h hxfin
    have hxbig : av.toReal + bv.toReal < -(Format.binary32.maxVal : ℝ) := by
      have h := sum_neg_big_of_notFinite hxafin hxbfin hx hxfnf (by rw [hxaval, hxbval]; exact hxub)
      rwa [hxaval, hxbval] at h
    have hxneg : x = (FloatV.negInf : F32) := by
      refine add_negInf_fin hxafin hxbfin hx hxfnf ?_
      have hM : (0 : ℝ) < (Format.binary32.maxVal : ℝ) := by
        rw [binary32_maxVal_real]; norm_num
      rw [hxaval, hxbval]
      linarith only [hxbig, hM]
    have hmul1 : FloatV.mul oneF32 x = (FloatV.negInf : F32) := by
      rw [hxneg]; decide +kernel
    have ht1inf : t1 = (FloatV.negInf : F32) := eq_negInf_naneq hmul1 ht1
    have ht2inf : t2 = (FloatV.negInf : F32) := eq_negInf_naneq hmul1 ht2
    have hcond : cond = true := by rw [hc, ht1inf]; decide +kernel
    have he0 : e = FloatV.ofPat 0 := by
      refine hninf e ?_
      rw [← ht2inf]; exact he
    have hadd1 : FloatV.add oneF32 e = oneF32 := by rw [he0]; decide +kernel
    have hsaddfin : (FloatV.add oneF32 e).isFinite = true := by rw [hadd1]; exact oneF32_isFinite
    have hseq : s = oneF32 := by rw [FloatV.NaNEq.eq_of_isFinite hsn hsaddfin, hadd1]
    have hsval : s.toReal = 1 := by rw [hseq]; exact oneF32_toReal
    have hsfin : s.isFinite = true := by rw [hseq]; exact oneF32_isFinite
    have hspos : 0 < s.toReal := by rw [hsval]; norm_num
    have hroomL : (1 + (2:ℝ)^(-22:ℤ)) * |Real.log s.toReal| + (2:ℝ)^(-149:ℤ) ≤
        (Format.binary32.maxVal : ℝ) := by
      rw [hsval, Real.log_one, abs_zero, mul_zero, zero_add, binary32_maxVal_real]
      norm_num
    obtain ⟨hlfin, hlerr⟩ := log_apply hlog hl hsfin hspos hroomL
    have hlabs : |l.toReal| ≤ (2 : ℝ) ^ (-149 : ℤ) := by
      simpa [hsval] using hlerr
    obtain ⟨hl2fin, hl2err⟩ := mul_exact_err_fin oneF32_isFinite hlfin (one_mul_room hlfin) hl2
    have hl2err' : |l2.toReal - l.toReal| ≤
        (2 : ℝ) ^ (-24 : ℤ) * |l.toReal| + (2 : ℝ) ^ (-150 : ℤ) := by
      have h := hl2err
      rw [oneF32_toReal, one_mul] at h
      have h2 : |l.toReal| * 2 ^ (-24 : ℤ) = 2 ^ (-24 : ℤ) * |l.toReal| := by ring
      rw [h2] at h
      exact h
    have hl2abs : |l2.toReal| ≤ (2 : ℝ) ^ (-140 : ℤ) := by
      have h := abs_le_of_mul_err (k := (2:ℝ)^(-149:ℤ)) (by positivity) hlabs hl2err'
      have h2 : (2:ℝ)^(-149:ℤ) * (1 + (2:ℝ)^(-24:ℤ)) + (2:ℝ)^(-150:ℤ) ≤
          (2:ℝ)^(-140:ℤ) := by norm_num
      linarith only [h, h2]
    have hx20 : av.toReal + bv.toReal ≤ -20 := by
      have hM : (20 : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
        rw [binary32_maxVal_real]; norm_num
      linarith only [hxbig, hM]
    have hSp : |l2.toReal - spRef (av.toReal + bv.toReal)| ≤
        52 * 2 ^ (-23 : ℤ) * spRef (av.toReal + bv.toReal) + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) :=
      softplus_neg hx20 hl2abs
    have hspval : sp = l2 := by rw [hsp, hcond]; rfl
    have hspfin : sp.isFinite = true := by rw [hspval]; exact hl2fin
    have hS : |sp.toReal - spRef (av.toReal + bv.toReal)| ≤
        52 * 2 ^ (-23 : ℤ) * spRef (av.toReal + bv.toReal) + 2 ^ (-24 : ℤ) * (1 + 2 ^ (-19 : ℤ)) := by
      rw [hspval]; exact hSp
    exact lane_tail hroomA heAerr hneAfin hneAval hspfin hS hx0abs hgg

end Claims.GdnGating
