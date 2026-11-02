import PagedAttention.Step
import PagedAttention.Model

/-!
# `paged_attention_f16`: after the loop, the output element (lines 71–72)

* The tolerances of `Model.lean` as real numbers, and their sizes at the covered extremes
  (`HEAD_DIM`, `PAGE_SIZE ≤ 256`, `X ≤ 2^10`, at most `2^16` pages): `TolFacts`.
* `final_o`: `(acc / l_i).to(float16)` against the exact attention over the sequence's tokens.
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

/-! ## The tolerances as real numbers -/

theorem rhoD_cast (D : ℕ) : ((rhoD D : ℚ) : ℝ) = (1 + 1 / 2 ^ 24) * (1 + 3 / 2 ^ 24) ^ (D - 1) - 1 := by
  unfold rhoD u32; push_cast; ring

theorem expErr_cast (D : ℕ) (X : ℚ) :
    ((expErr D X : ℚ) : ℝ) = (((rhoD D : ℚ) : ℝ) + 8 / 2 ^ 24) * (X : ℝ) + 1 / 2 ^ 50 := by
  unfold expErr u32; push_cast; ring

/-- `rhoD D ≤ 2^-14` for `D ≤ 256`. -/
theorem rhoD_bounds {D : ℕ} (hD : D ≤ 256) : 0 ≤ ((rhoD D : ℚ) : ℝ) ∧ ((rhoD D : ℚ) : ℝ) ≤ 1 / 2 ^ 14 := by
  rw [rhoD_cast]
  have h1 : 1 ≤ ((1 : ℝ) + 3 / 2 ^ 24) ^ (D - 1) := one_le_pow₀ (by norm_num)
  have h2 : ((1 : ℝ) + 3 / 2 ^ 24) ^ (D - 1) ≤ 1 + 1 / 2 ^ 14 - 1 / 2 ^ 20 := by
    have h := one_add_pow_le_exp (show (0 : ℝ) ≤ 3 / 2 ^ 24 by norm_num) (D - 1)
    set y : ℝ := ((D - 1 : ℕ) : ℝ) * (3 / 2 ^ 24) with hy
    have hn : ((D - 1 : ℕ) : ℝ) ≤ 256 := by exact_mod_cast (show D - 1 ≤ 256 by omega)
    have hy0 : 0 ≤ y := by positivity
    have hy1 : y ≤ 256 * (3 / 2 ^ 24) := mul_le_mul_of_nonneg_right hn (by norm_num)
    have hy2 : y ^ 2 ≤ (256 * (3 / 2 ^ 24)) ^ 2 := pow_le_pow_left₀ hy0 hy1 2
    have h3 := abs_exp_sub_one' (show |y| ≤ 1 by rw [abs_of_nonneg hy0]; linarith)
    rw [abs_of_nonneg hy0] at h3
    have h4 := le_abs_self (Real.exp y - 1)
    have n : (256 : ℝ) * (3 / 2 ^ 24) + (256 * (3 / 2 ^ 24)) ^ 2 ≤ 1 / 2 ^ 14 - 1 / 2 ^ 20 := by norm_num
    linarith
  constructor
  · have : (1 : ℝ) * 1 ≤ (1 + 1 / 2 ^ 24) * ((1 : ℝ) + 3 / 2 ^ 24) ^ (D - 1) :=
      mul_le_mul (by norm_num) h1 zero_le_one (by norm_num)
    linarith
  · have : (1 + 1 / 2 ^ 24) * ((1 : ℝ) + 3 / 2 ^ 24) ^ (D - 1) ≤
        (1 + 1 / 2 ^ 24) * (1 + 1 / 2 ^ 14 - 1 / 2 ^ 20) := mul_le_mul_of_nonneg_left h2 (by norm_num)
    have n : (1 + 1 / 2 ^ 24 : ℝ) * (1 + 1 / 2 ^ 14 - 1 / 2 ^ 20) - 1 ≤ 1 / 2 ^ 14 := by norm_num
    linarith

/-- The row data's tolerances: `R.Y = X`, `R.E = expErr D X`. -/
theorem psiP_cast {R : RowCtx} {D : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((psiP D X : ℚ) : ℝ) = psiPr R := by
  unfold psiP relExp psiPr relE; rw [hY, hE]; push_cast; ring

theorem psiL_cast {R : RowCtx} {D : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((psiL D R.P X : ℚ) : ℝ) = psiLr R := by
  unfold psiL psiLr u32; push_cast; rw [psiP_cast hY hE]

theorem psiA_cast {R : RowCtx} {D : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((psiA D R.P X : ℚ) : ℝ) = psiAr R := by
  unfold psiA psiAr u32; push_cast; rw [psiP_cast hY hE]; ring

theorem step_cast : ((step : ℚ) : ℝ) = stepR := by
  unfold step stepR u32; push_cast; ring

theorem drift_cast {R : RowCtx} {D : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((drift D X : ℚ) : ℝ) = 1 + 4 / 2 ^ 22 * (R.Y + R.E) := by
  unfold drift; rw [hY, hE]; push_cast; ring

theorem fL_cast {R : RowCtx} {D T : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((fL D R.P T X : ℚ) : ℝ) = (1 + psiLr R) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) * stepR ^ T := by
  unfold fL; push_cast; rw [psiL_cast hY hE, drift_cast hY hE, step_cast]

theorem fA_cast {R : RowCtx} {D T : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((fA D R.P T X : ℚ) : ℝ) = (1 + psiAr R) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) * stepR ^ T := by
  unfold fA; push_cast; rw [psiA_cast hY hE, drift_cast hY hE, step_cast]

theorem lastRel_cast : ((lastRel : ℚ) : ℝ) = 1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22 := by
  unfold lastRel; push_cast; ring

/-- **The tolerances' pieces at the covered extremes.** -/
structure TolFacts (D P T : ℕ) (X : ℚ) : Prop where
  fL1 : 1 ≤ ((fL D P T X : ℚ) : ℝ)
  fL2 : ((fL D P T X : ℚ) : ℝ) ≤ 5 / 4
  fA1 : 1 ≤ ((fA D P T X : ℚ) : ℝ)
  fA2 : ((fA D P T X : ℚ) : ℝ) ≤ 5 / 4
  wlow_eq : ((wlow D X : ℚ) : ℝ) = 1 - ((expErr D X : ℚ) : ℝ)
  wlow1 : 7 / 8 ≤ ((wlow D X : ℚ) : ℝ)
  beta_eq : ((betaL D P T X : ℚ) : ℝ) =
    (((fL D P T X : ℚ) : ℝ) - 1) + ((fL D P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ)
  beta0 : 0 ≤ ((betaL D P T X : ℚ) : ℝ)
  beta1 : ((betaL D P T X : ℚ) : ℝ) ≤ 13 / 50
  den_eq : ((denQ D P T X : ℚ) : ℝ) = 1 - ((betaL D P T X : ℚ) : ℝ)
  relQ_eq : ((relQ D P T X : ℚ) : ℝ) =
    (((fA D P T X : ℚ) : ℝ) - 1 + ((betaL D P T X : ℚ) : ℝ)) / ((denQ D P T X : ℚ) : ℝ)
  relQ0 : 0 ≤ ((relQ D P T X : ℚ) : ℝ)
  relQ1 : ((relQ D P T X : ℚ) : ℝ) ≤ 7 / 10
  absQ_eq : ((absQ D P T X : ℚ) : ℝ) =
    ((fA D P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 90)) / (((wlow D X : ℚ) : ℝ) * ((denQ D P T X : ℚ) : ℝ))
  absQ0 : 0 ≤ ((absQ D P T X : ℚ) : ℝ)
  absQ1 : ((absQ D P T X : ℚ) : ℝ) ≤ 1 / 2 ^ 70

theorem tol_facts {R : RowCtx} (hR : R.Ok) {D T : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ))
    (hE : R.E = (expErr D X : ℝ)) (hT : T ≤ 2 ^ 16) : TolFacts D R.P T X := by
  have hE0 := hR.E_nonneg
  have hE1 := hR.E_le
  have hY1 := hR.Y_le
  have hY0 := hR.Y_nonneg
  have hTr : (T : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hT
  have hd1 : 1 ≤ 1 + 4 / 2 ^ 22 * (R.Y + R.E) := by
    have : 0 ≤ 4 / 2 ^ 22 * (R.Y + R.E) := by positivity
    linarith
  have hd2 : 1 + 4 / 2 ^ 22 * (R.Y + R.E) ≤ 1 + 1 / 2 ^ 9 := by
    have : 4 / 2 ^ 22 * (R.Y + R.E) ≤ 4 / 2 ^ 22 * (2 ^ 10 + 1 / 8) :=
      mul_le_mul_of_nonneg_left (by linarith) (by norm_num)
    norm_num at this ⊢; linarith
  have hs1 : 1 ≤ stepR ^ T := one_le_pow₀ stepR_one
  have hs2 : stepR ^ T ≤ 1 + 2 * ((2 ^ 16 : ℕ) * (1 / 2 ^ 21)) :=
    pow_le_of_step stepR_one stepR_le hT (by norm_num)
  have hpL := psiLr_le hR
  have hpL0 := psiLr_nonneg hR
  have hpA := psiAr_le hR
  have hpA0 := psiAr_nonneg hR
  have f1 : ∀ psi : ℝ, 0 ≤ psi → 1 ≤ (1 + psi) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) * stepR ^ T := by
    intro psi h0
    have := mul_le_mul (by linarith : (1 : ℝ) ≤ 1 + psi) hd1 zero_le_one (by linarith)
    have := mul_le_mul this hs1 zero_le_one (by positivity)
    linarith
  have f2 : ∀ psi : ℝ, 0 ≤ psi → psi ≤ 1 / 7 →
      (1 + psi) * (1 + 4 / 2 ^ 22 * (R.Y + R.E)) * stepR ^ T ≤ 5 / 4 := by
    intro psi h0 h1
    have k1 := mul_le_mul (by linarith : 1 + psi ≤ 1 + 1 / 7) hd2 (by linarith) (by norm_num)
    have k2 := mul_le_mul k1 hs2 (by positivity) (by norm_num)
    norm_num at k2 ⊢; linarith
  have hfL1 : 1 ≤ ((fL D R.P T X : ℚ) : ℝ) := by rw [fL_cast hY hE]; exact f1 _ hpL0
  have hfL2 : ((fL D R.P T X : ℚ) : ℝ) ≤ 5 / 4 := by rw [fL_cast hY hE]; exact f2 _ hpL0 hpL
  have hfA1 : 1 ≤ ((fA D R.P T X : ℚ) : ℝ) := by rw [fA_cast hY hE]; exact f1 _ hpA0
  have hfA2 : ((fA D R.P T X : ℚ) : ℝ) ≤ 5 / 4 := by rw [fA_cast hY hE]; exact f2 _ hpA0 hpA
  have hw : ((wlow D X : ℚ) : ℝ) = 1 - ((expErr D X : ℚ) : ℝ) := by unfold wlow; push_cast; rfl
  have hw1 : 7 / 8 ≤ ((wlow D X : ℚ) : ℝ) := by rw [hw, ← hE]; linarith
  have hb : ((betaL D R.P T X : ℚ) : ℝ) =
      (((fL D R.P T X : ℚ) : ℝ) - 1) + ((fL D R.P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 100)) /
        ((wlow D X : ℚ) : ℝ) := by
    unfold betaL absL; push_cast; ring
  have hbt : 0 ≤ ((fL D R.P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) := by
    apply div_nonneg _ (by linarith); positivity
  have hbt1 : ((fL D R.P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) ≤ 1 / 2 ^ 80 := by
    rw [div_le_iff₀ (by linarith)]
    have : ((fL D R.P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 100)) ≤ 5 / 4 * (2 ^ 16 * (1 / 2 ^ 100)) :=
      mul_le_mul hfL2 (mul_le_mul_of_nonneg_right hTr (by positivity)) (by positivity) (by norm_num)
    nlinarith
  have hb0 : 0 ≤ ((betaL D R.P T X : ℚ) : ℝ) := by rw [hb]; linarith
  have hb1 : ((betaL D R.P T X : ℚ) : ℝ) ≤ 13 / 50 := by rw [hb]; norm_num at hbt1 ⊢; linarith
  have hden : ((denQ D R.P T X : ℚ) : ℝ) = 1 - ((betaL D R.P T X : ℚ) : ℝ) := by
    unfold denQ; push_cast; rfl
  have hden1 : 37 / 50 ≤ ((denQ D R.P T X : ℚ) : ℝ) := by rw [hden]; linarith
  have hrq : ((relQ D R.P T X : ℚ) : ℝ) =
      (((fA D R.P T X : ℚ) : ℝ) - 1 + ((betaL D R.P T X : ℚ) : ℝ)) / ((denQ D R.P T X : ℚ) : ℝ) := by
    unfold relQ; push_cast; rfl
  have hq : ((absQ D R.P T X : ℚ) : ℝ) =
      ((fA D R.P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 90)) /
        (((wlow D X : ℚ) : ℝ) * ((denQ D R.P T X : ℚ) : ℝ)) := by
    unfold absQ absA; push_cast; rfl
  have hwd : 1 / 2 ≤ ((wlow D X : ℚ) : ℝ) * ((denQ D R.P T X : ℚ) : ℝ) := by
    have := mul_le_mul hw1 hden1 (by norm_num) (by linarith); norm_num at this ⊢; linarith
  refine ⟨hfL1, hfL2, hfA1, hfA2, hw, hw1, hb, hb0, hb1, hden, hrq, ?_, ?_, hq, ?_, ?_⟩
  · rw [hrq]; apply div_nonneg (by linarith) (by linarith)
  · rw [hrq, div_le_iff₀ (by linarith)]; nlinarith
  · rw [hq]; apply div_nonneg (by positivity) (by linarith)
  · rw [hq, div_le_iff₀ (by linarith)]
    have : ((fA D R.P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 90)) ≤ 5 / 4 * (2 ^ 16 * (1 / 2 ^ 90)) :=
      mul_le_mul hfA2 (mul_le_mul_of_nonneg_right hTr (by positivity)) (by positivity) (by norm_num)
    nlinarith

/-! ## The sums over the sequence's tokens -/

/-- Once the pages cover the sequence (`n ≤ P·t`), `psum` is the sum over its tokens. -/
theorem psum_full {R : RowCtx} {t : ℕ} (h : R.n ≤ R.P * t) (f : ℕ → ℝ) :
    psum R t f = ∑ j ∈ Finset.range R.n, f j := by
  unfold psum
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero]
  congr 1
  ext j
  simp only [Finset.mem_filter, Finset.mem_range]
  omega

theorem wt_split (R : RowCtx) (m : ℝ) (j : ℕ) :
    wt R m j = Real.exp (-m) * Real.exp (R.c * R.sv j) := by
  unfold wt; rw [← Real.exp_add]; congr 1; ring

/-- The kernel's weights against the maximum give the exact weighted mean. -/
theorem ratio_eq {R : RowCtx} {t : ℕ} (h : R.n ≤ R.P * t) (m : ℝ) (g : ℕ → ℝ) :
    psum R t (fun j => wt R m j * g j) / psum R t (wt R m) =
      (∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) * g j) /
        ∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) := by
  rw [psum_full h, psum_full h]
  simp only [wt_split, mul_assoc]
  rw [← Finset.mul_sum, ← Finset.mul_sum, mul_div_mul_left _ _ (Real.exp_pos _).ne']

/-! ## The output element -/

/-- **The output element** (lines 71–72): after `T` pages covering the sequence, with running
maximum `m`, `l_i = l` and accumulator element `a` (column values `ve`), `(a / l).to(float16)` is
finite and within `tolO·attnAbs + tolF` of the exact attention. -/
theorem final_o {P : Platform} (hdivF : Rules.Num.DivFull P) {R : RowCtx} (hR : R.Ok)
    {D : ℕ} {X : ℚ} (hY : R.Y = (X : ℝ)) (hE : R.E = (expErr D X : ℝ))
    {T : ℕ} (hT : T ≤ 2 ^ 16) (hn : R.n ≤ R.P * T)
    {ve : ℕ → ℝ} (hve : ∀ j, |ve j| ≤ 2 ^ 15)
    {m l a o' : F32} {o16 : F16} (hM : MInv R T m) (hL : LInv R T m l) (hA : AInv R ve T m a)
    (hdv : P.div .binary32 a l o') (hc : FloatV.NaNEq (FloatV.cvt o') o16) :
    o16.isFinite = true ∧
      |o16.toReal - (∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) * ve j) /
          ∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j)| ≤
        (tolO D R.P T X : ℝ) * ((∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) * |ve j|) /
          ∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j)) + (tolF D R.P T X : ℝ) := by
  have tf := tol_facts hR hY hE hT
  have hTr : (T : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hT
  have hE0 := hR.E_nonneg
  set W := psum R T (wt R m.toReal) with hWd
  set S := psum R T (fun j => wt R m.toReal j * ve j) with hSd
  set Mg := psum R T (fun j => wt R m.toReal j * |ve j|) with hMgd
  have hWl : ((wlow D X : ℚ) : ℝ) ≤ W := by rw [tf.wlow_eq, ← hE]; exact W_ge hR hM
  have hwl0 : 0 < ((wlow D X : ℚ) : ℝ) := by linarith [tf.wlow1]
  have hW0 : 0 < W := lt_of_lt_of_le hwl0 hWl
  have hMg0 : 0 ≤ Mg := psum_nonneg fun j => mul_nonneg (wt_pos _ _ _).le (abs_nonneg _)
  have hS : |S| ≤ Mg := psum_abs_wt R m.toReal T ve
  -- `l_i`
  have hFl : ffac R (psiLr R) T m.toReal ≤ (fL D R.P T X : ℝ) := by
    refine (ffac_le_drift hR (psiLr_nonneg hR) (le_refl T) hM.bound).trans (le_of_eq ?_)
    rw [fL_cast hY hE]
  have hl : |l.toReal - W| ≤ (betaL D R.P T X : ℝ) * W := by
    have h := hL.2
    rw [tf.beta_eq]
    have hFl1 := ffac_one_le (t := T) (psiLr_nonneg hR) (m_nonneg hM.bound)
    have k1 : (ffac R (psiLr R) T m.toReal - 1) * W ≤ ((fL D R.P T X : ℝ) - 1) * W :=
      mul_le_mul_of_nonneg_right (by linarith) hW0.le
    have k2 : ffac R (psiLr R) T m.toReal * (T * absLr) ≤ (fL D R.P T X : ℝ) * (T * (1 / 2 ^ 100)) := by
      unfold absLr
      exact mul_le_mul hFl le_rfl (by positivity) (by linarith)
    have k3 : (fL D R.P T X : ℝ) * (T * (1 / 2 ^ 100)) ≤
        (fL D R.P T X : ℝ) * (T * (1 / 2 ^ 100)) / (wlow D X : ℝ) * W := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hwl0]
      have : 0 ≤ (fL D R.P T X : ℝ) * (T * (1 / 2 ^ 100)) := by have := tf.fL1; positivity
      exact mul_le_mul_of_nonneg_left hWl this
    have e : (((fL D R.P T X : ℚ) : ℝ) - 1 + ((fL D R.P T X : ℚ) : ℝ) * ((T : ℝ) * (1 / 2 ^ 100)) /
        ((wlow D X : ℚ) : ℝ)) * W =
      (((fL D R.P T X : ℚ) : ℝ) - 1) * W + ((fL D R.P T X : ℚ) : ℝ) *
        ((T : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) * W := by ring
    rw [e]; linarith
  -- `acc`
  have hFa : ffac R (psiAr R) T m.toReal ≤ (fA D R.P T X : ℝ) := by
    refine (ffac_le_drift hR (psiAr_nonneg hR) (le_refl T) hM.bound).trans (le_of_eq ?_)
    rw [fA_cast hY hE]
  have ha : |a.toReal - S| ≤ ((fA D R.P T X : ℝ) - 1) * Mg + (fA D R.P T X : ℝ) * (T * (1 / 2 ^ 90)) := by
    have h := hA.2
    have hFa1 := ffac_one_le (t := T) (psiAr_nonneg hR) (m_nonneg hM.bound)
    have k1 : (ffac R (psiAr R) T m.toReal - 1) * Mg ≤ ((fA D R.P T X : ℝ) - 1) * Mg :=
      mul_le_mul_of_nonneg_right (by linarith) hMg0
    have k2 : ffac R (psiAr R) T m.toReal * (T * absAr) ≤ (fA D R.P T X : ℝ) * (T * (1 / 2 ^ 90)) := by
      unfold absAr
      exact mul_le_mul hFa le_rfl (by positivity) (by linarith)
    linarith
  -- the ratio `acc / l_i` against the weighted mean
  obtain ⟨hlW, hq⟩ := ratio_err hWl hwl0 tf.beta0 (by linarith [tf.beta1]) hS hl ha tf.fA1 (by positivity)
  rw [← tf.den_eq] at hq hlW
  rw [← tf.relQ_eq, ← tf.absQ_eq] at hq
  rw [hSd, hMgd, hWd, ratio_eq hn, ratio_eq hn] at hq
  set ref := (∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) * ve j) /
    ∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) with hrefd
  set A := (∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) * |ve j|) /
    ∑ j ∈ Finset.range R.n, Real.exp (R.c * R.sv j) with hAd
  have hAW : A = Mg / W := by rw [hAd, hMgd, hWd, ratio_eq hn]
  have hrefW : ref = S / W := by rw [hrefd, hSd, hWd, ratio_eq hn]
  have hA0 : 0 ≤ A := by rw [hAW]; exact div_nonneg hMg0 hW0.le
  have hA15 : A ≤ 2 ^ 15 := by
    rw [hAW, div_le_iff₀ hW0, hMgd, hWd, ← psum_mul]
    exact psum_le fun j _ _ => by
      rw [mul_comm (2 ^ 15 : ℝ)]; exact mul_le_mul_of_nonneg_left (hve j) (wt_pos _ _ _).le
  have hrA : |ref| ≤ A := by
    rw [hrefW, hAW, abs_div, abs_of_pos hW0]; exact div_le_div_of_nonneg_right hS hW0.le
  have hl0 : 0 < l.toReal := by
    have := mul_pos (show (0 : ℝ) < (denQ D R.P T X : ℝ) by rw [tf.den_eq]; linarith [tf.beta1]) hW0
    linarith
  set q := a.toReal / l.toReal with hqd
  have hqb : |q| ≤ (1 + (relQ D R.P T X : ℝ)) * A + (absQ D R.P T X : ℝ) := by
    have := abs_sub_abs_le_abs_sub q ref
    linarith
  have hq56 : |q| ≤ 56000 := by
    have k1 : (1 + (relQ D R.P T X : ℝ)) * A ≤ (1 + 7 / 10) * 2 ^ 15 :=
      mul_le_mul (by linarith [tf.relQ1]) hA15 hA0 (by norm_num)
    have := tf.absQ1
    norm_num at k1 this ⊢; linarith
  -- line 71: `acc / l_i`
  have hq' : |a.toRat / l.toRat| ≤ 2 ^ 126 := by
    have : ((|a.toRat / l.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast; rw [← toReal_def, ← toReal_def]; rw [← hqd]; linarith
    exact_mod_cast this
  have hl0' : l.toRat ≠ 0 := by
    intro h; rw [toReal_def, h] at hl0; simp at hl0
  obtain ⟨ho'f, ho'e⟩ := hdivF a l o' hdv hA.1 hL.1 hl0' hq'
  have ho'r : |o'.toReal - q| ≤ 1 / 2 ^ 22 * |q| + 1 / 2 ^ 148 := by
    have := (Rat.cast_le (K := ℝ)).mpr ho'e
    push_cast at this
    rw [show (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 by norm_num,
      show (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 by norm_num] at this
    simp only [← toReal_def] at this
    exact this
  have ho'b : |o'.toReal| ≤ 65504 := by
    have := abs_sub_abs_le_abs_sub o'.toReal q
    have k : 1 / 2 ^ 22 * |q| ≤ 1 / 2 ^ 22 * 56000 := mul_le_mul_of_nonneg_left hq56 (by norm_num)
    norm_num at k ⊢; linarith
  -- line 72: `.to(float16)`
  obtain ⟨ho16f, ho16e⟩ := f16_cvt hc ho'f ho'b
  refine ⟨ho16f, ?_⟩
  -- assembling the tolerance
  have hto : (tolO D R.P T X : ℝ) = (1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22) *
      (1 + (relQ D R.P T X : ℝ)) + (relQ D R.P T X : ℝ) := by
    unfold tolO; push_cast; rw [lastRel_cast]
  have htF : (tolF D R.P T X : ℝ) = (absQ D R.P T X : ℝ) * (1 + (1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22)) +
      1 / 2 ^ 11 * (1 / 2 ^ 148) + 1 / 2 ^ 148 + 1 / 2 ^ 25 := by
    unfold tolF; push_cast; rw [lastRel_cast]
  rw [hto, htF]
  set rq := (relQ D R.P T X : ℝ)
  set aq := (absQ D R.P T X : ℝ)
  set lr : ℝ := 1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22 with hlr
  have hlr0 : 0 ≤ lr := by rw [hlr]; positivity
  have p1 := mul_le_mul_of_nonneg_left hqb hlr0
  have t1 := abs_sub_le o16.toReal o'.toReal q
  have t3 := abs_sub_le o16.toReal q ref
  have t4 := abs_sub_abs_le_abs_sub o'.toReal q
  have e1 : |o16.toReal - o'.toReal| ≤ (|q| + 1 / 2 ^ 22 * |q| + 1 / 2 ^ 148) / 2 ^ 11 + 1 / 2 ^ 25 := by
    have : |o'.toReal| ≤ |q| + 1 / 2 ^ 22 * |q| + 1 / 2 ^ 148 := by linarith
    have := div_le_div_of_nonneg_right this (by norm_num : (0 : ℝ) ≤ 2 ^ 11)
    linarith
  rw [hlr] at p1
  linarith

end Kernels.PagedAttention
