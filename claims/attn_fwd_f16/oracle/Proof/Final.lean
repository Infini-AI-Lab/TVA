import Proof.Row

/-!
# The end of a row: `acc / l`, the rounding to `float16`, and `M = m + log₂ l`

From the invariant after the last trip (`InvM`, `InvL`, `InvA`):

* `ratio_facts`: the row sum is positive and the quotient `a / l` is within
  `tolO·attnAbs + tolS·Σ|v| + tolF0` of the exact `As / Ls`;
* `out_bound`: the output (`/` then `float16`) is within `tolO2·attnAbs + tolS2·Σ|v| + tolF2`;
* `lse_bound`: `M = m + log₂ l` is within `tolM` of `m + log₂ Ls` (the exact log-sum-exp).
-/

namespace Claims.AttnFwd

open TVA Finset

noncomputable section

/-- `exp Θ ≤ 1 + Θ + Θ²` for `0 ≤ Θ ≤ 1`. -/
theorem exp_le_EA {Θ : ℝ} (h0 : 0 ≤ Θ) (h1 : Θ ≤ 1) : Real.exp Θ ≤ EA Θ := by
  have := Real.abs_exp_sub_one_sub_id_le (x := Θ) (by rw [abs_of_nonneg h0]; exact h1)
  have h2 := (abs_le.mp this).2
  unfold EA; nlinarith

/-- The key block count stays in the covered range. -/
theorem Box.of_par {D : ℕ} {X : ℝ} {T : ℕ} (hP : Par D X) (h2 : 2 ≤ T) (h1 : T ≤ 1024) :
    Box D (T : ℝ) X :=
  ⟨hP.hD, ⟨by exact_mod_cast h2, by exact_mod_cast h1⟩, hP.hX⟩

/-- The rate sums of the invariant are at most `ThL`, `ThA`. -/
theorem theta_le_ThL {X : ℝ} {T : ℕ} {m mf Θ : ℝ} (hm : |m| ≤ Bm X) (hmf : |mf| ≤ Bm X)
    (hΘ : Θ ≤ T * EPSL + CD * (m - mf)) : Θ ≤ ThL T X := by
  unfold ThL
  have : m - mf ≤ 2 * Bm X := by
    have := (abs_le.mp hm).2; have := (abs_le.mp hmf).1; linarith
  have hcd : 0 ≤ CD := by unfold CD u32; norm_num
  nlinarith

/-- Bound on the relaxed threshold `Θ` by `ThA` from bounded scores and maxima. -/
theorem theta_le_ThA {X : ℝ} {T : ℕ} {m mf Θ : ℝ} (hm : |m| ≤ Bm X) (hmf : |mf| ≤ Bm X)
    (hΘ : Θ ≤ T * EPSA + CD * (m - mf)) : Θ ≤ ThA T X := by
  unfold ThA
  have : m - mf ≤ 2 * Bm X := by
    have := (abs_le.mp hm).2; have := (abs_le.mp hmf).1; linarith
  have hcd : 0 ≤ CD := by unfold CD u32; norm_num
  nlinarith

/-- `ThL` is nonnegative for nonnegative `T` and `X`. -/
theorem ThL_nonneg {T X : ℝ} (hT : 0 ≤ T) (hX : 0 ≤ X) : 0 ≤ ThL T X := by
  unfold ThL CD Bm u32 EPSL; positivity

/-- `ThA` is nonnegative for nonnegative `T` and `X`. -/
theorem ThA_nonneg {T X : ℝ} (hT : 0 ≤ T) (hX : 0 ≤ X) : 0 ≤ ThA T X := by
  unfold ThA CD Bm u32 EPSA; positivity

/-- The computed row sum is within `lamL` (relatively) of the exact one. -/
theorem invL_rel {R : RowCtx} {X : ℝ} {T : ℕ} (hP : Par R.D X) (hT : 2 ≤ T) (hT1 : T ≤ 1024)
    {m mf l : ℝ} (hM : InvM R X T m mf) (hL : InvL R X T m mf l) :
    |l - R.Ls m T| ≤ lamL R.D T X * R.Ls m T := by
  have hbox := Box.of_par hP hT hT1
  have hlow := hM.hlow (by omega)
  have hw := hP.wlow_ge
  have hLc0 : 0 < R.Ls m T := by linarith
  have hthL := hbox.thL_le
  have hTh0 : 0 ≤ ThL T X := ThL_nonneg (Nat.cast_nonneg T) hbox.hX.1
  obtain ⟨Θl, hΘl0, hΘl, hl⟩ := hL
  have hΘl' : Θl ≤ ThL T X := theta_le_ThL hM.hm hM.hmf hΘl
  have hExpL : Real.exp Θl ≤ EA (ThL T X) :=
    (exp_le_EA hΘl0 (by linarith)).trans (EA_mono hΘl0 hΘl')
  have hr0 := hP.rbarL_nonneg
  have hη0 : 0 ≤ etaAL := by unfold etaAL; norm_num
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
  have hEA0 := EA_nonneg hTh0
  refine hl.trans ?_
  have h1 : Real.exp Θl * ((Θl + rbarL R.D X) * R.Ls m T + etaAL * T) ≤
      EA (ThL T X) * ((ThL T X + rbarL R.D X) * R.Ls m T + etaAL * T) := by
    have hnn : 0 ≤ (Θl + rbarL R.D X) * R.Ls m T + etaAL * T := by positivity
    have h2 : (Θl + rbarL R.D X) * R.Ls m T + etaAL * T ≤
        (ThL T X + rbarL R.D X) * R.Ls m T + etaAL * T := by
      have := mul_le_mul_of_nonneg_right (by linarith : Θl + rbarL R.D X ≤ ThL T X + rbarL R.D X)
        hLc0.le
      linarith
    calc _ ≤ EA (ThL T X) * ((Θl + rbarL R.D X) * R.Ls m T + etaAL * T) :=
          mul_le_mul_of_nonneg_right hExpL hnn
      _ ≤ _ := mul_le_mul_of_nonneg_left h2 hEA0
  refine h1.trans ?_
  unfold lamL
  have h3 : EA (ThL T X) * etaAL * T ≤ EA (ThL T X) * etaAL * T / wlow R.D X * R.Ls m T := by
    have hx : 0 ≤ EA (ThL T X) * etaAL * T := by positivity
    rw [div_mul_eq_mul_div, le_div_iff₀ (by linarith [hw] : 0 < wlow R.D X)]
    exact mul_le_mul_of_nonneg_left hlow hx
  nlinarith [h3]

/-- The computed accumulator (column `e`) is within the closed form with `ThA`. -/
theorem invA_rel {R : RowCtx} {X : ℝ} {T : ℕ} (hP : Par R.D X) (hT : 2 ≤ T) (hT1 : T ≤ 1024)
    {m mf : ℝ} {a : ℕ → ℝ} (hM : InvM R X T m mf) (hA : InvA R X T m mf a) {e : ℕ} (he : e < R.D) :
    |a e - R.As e m T| ≤ EA (ThA T X) * ((ThA T X + rbarA R.D X) * R.Ms e m T +
      kapA R.D X * R.Ls m T + etavA * R.Vs e T + etaAA * T) := by
  have hbox := Box.of_par hP hT hT1
  have hthA := hbox.thA_le
  have hTh0 : 0 ≤ ThA T X := ThA_nonneg (Nat.cast_nonneg T) hbox.hX.1
  obtain ⟨Θ, hΘ0, hΘ, ha⟩ := hA
  have hΘ' : Θ ≤ ThA T X := theta_le_ThA hM.hm hM.hmf hΘ
  have hExp : Real.exp Θ ≤ EA (ThA T X) :=
    (exp_le_EA hΘ0 (by linarith)).trans (EA_mono hΘ0 hΘ')
  have hr0 := hP.rbarA_nonneg
  have hk0 := hP.kapA_nonneg
  have hMs0 := R.Ms_nonneg e m T
  have hLs0 := R.Ls_nonneg m T
  have hVs0 := R.Vs_nonneg e T
  have hη1 : 0 ≤ etavA := by unfold etavA eta16 r64; norm_num
  have hη2 : 0 ≤ etaAA := by unfold etaAA; norm_num
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
  have hEA0 := EA_nonneg hTh0
  refine (ha e he).trans ?_
  have h2 : (Θ + rbarA R.D X) * R.Ms e m T + kapA R.D X * R.Ls m T + etavA * R.Vs e T + etaAA * T ≤
      (ThA T X + rbarA R.D X) * R.Ms e m T + kapA R.D X * R.Ls m T + etavA * R.Vs e T + etaAA * T := by
    have := mul_le_mul_of_nonneg_right (by linarith : Θ + rbarA R.D X ≤ ThA T X + rbarA R.D X) hMs0
    linarith
  have hnn : 0 ≤ (Θ + rbarA R.D X) * R.Ms e m T + kapA R.D X * R.Ls m T + etavA * R.Vs e T +
      etaAA * T := by positivity
  calc _ ≤ EA (ThA T X) * ((Θ + rbarA R.D X) * R.Ms e m T + kapA R.D X * R.Ls m T +
        etavA * R.Vs e T + etaAA * T) := mul_le_mul_of_nonneg_right hExp hnn
    _ ≤ _ := mul_le_mul_of_nonneg_left h2 hEA0

/-- **The quotient of two approximations**, in the form of the tolerance: with the row sum within
`lamL` (relatively) and the accumulator within the closed form, `a / l` is within
`tolO·(M/L) + tolS·V + tolF0` of `A / L`. -/
theorem ratio_alg {D : ℕ} {T X : ℝ} (hb : Box D T X) {Lc M A l a V : ℝ}
    (hw : wlow D X ≤ Lc) (hl : |l - Lc| ≤ lamL D T X * Lc) (_hM0 : 0 ≤ M) (hAM : |A| ≤ M)
    (hV0 : 0 ≤ V)
    (ha : |a - A| ≤ EA (ThA T X) * ((ThA T X + rbarA D X) * M + kapA D X * Lc + etavA * V +
      etaAA * T)) :
    0 < l ∧ |a / l - A / Lc| ≤ tolO D T X * (M / Lc) + tolS D T X * V + tolF0 D T X := by
  have hwl := hb.par.wlow_ge
  have hLc : 0 < Lc := by linarith
  have hlam0 := hb.lamL_nonneg
  have hlam1 := hb.lamL_le
  have hc : 0 < 1 - lamL D T X := by linarith
  have hlpos : 0 < l := by
    have h1 := (abs_le.mp hl).1
    nlinarith
  refine ⟨hlpos, ?_⟩
  have hLe : lamL D T X * Lc < Lc := by nlinarith
  have hrat := Softmax.ratio_err (L := Lc) (A := A) (l := l) (a := a) (eL := lamL D T X * Lc)
    (eA := EA (ThA T X) * ((ThA T X + rbarA D X) * M + kapA D X * Lc + etavA * V + etaAA * T))
    (V := M / Lc) hLc hl hLe ha (by rw [div_mul_cancel₀ _ hLc.ne']; exact hAM)
  refine hrat.trans ?_
  have hT0 : 0 ≤ T := by linarith [hb.hT.1]
  have hE0 := EA_nonneg (ThA_nonneg hT0 hb.hX.1)
  have hk0 := hb.par.kapA_nonneg
  have hr0 := hb.par.rbarA_nonneg
  have hη1 : 0 ≤ etavA := by unfold etavA eta16 r64; norm_num
  have hη2 : 0 ≤ etaAA := by unfold etaAA; norm_num
  set E := EA (ThA T X) with hE
  have hden : Lc - lamL D T X * Lc = Lc * (1 - lamL D T X) := by ring
  have e1 : (E * ((ThA T X + rbarA D X) * M + kapA D X * Lc + etavA * V + etaAA * T) +
      M / Lc * (lamL D T X * Lc)) / (Lc - lamL D T X * Lc) =
      (E * (ThA T X + rbarA D X) + lamL D T X) / (1 - lamL D T X) * (M / Lc) +
      E * kapA D X / (1 - lamL D T X) +
      (E * etavA * V + E * etaAA * T) / (Lc * (1 - lamL D T X)) := by
    rw [hden]; field_simp; ring
  rw [e1]
  have h3 : (E * etavA * V + E * etaAA * T) / (Lc * (1 - lamL D T X)) ≤
      (E * etavA * V + E * etaAA * T) / (wlow D X * (1 - lamL D T X)) := by
    refine div_le_div_of_nonneg_left (by positivity) (by positivity) ?_
    exact mul_le_mul_of_nonneg_right hw hc.le
  have e2 : (E * etavA * V + E * etaAA * T) / (wlow D X * (1 - lamL D T X)) =
      E * etavA / (wlow D X * (1 - lamL D T X)) * V + E * etaAA * T / (wlow D X * (1 - lamL D T X)) := by
    ring
  unfold tolO tolS tolF0
  rw [← hE] at *
  rw [e2] at h3
  linarith

/-- **The quotient `a / l` at the end of a row.** The row sum is within `lamL` of the exact one,
between `1/2` and `2^18`; the quotient of the computed accumulator by it is within
`tolO·attnAbs + tolS·Σ|v| + tolF0` of the exact `As / Ls` (`attnAbs = Ms / Ls`), and at most
`50000` in magnitude. -/
theorem ratio_facts {R : RowCtx} {X : ℝ} {T : ℕ} (hP : Par R.D X) (hT : 2 ≤ T) (hT1 : T ≤ 1024)
    {m mf l : ℝ} {a : ℕ → ℝ} (hM : InvM R X T m mf) (hL : InvL R X T m mf l)
    (hA : InvA R X T m mf a) {e : ℕ} (he : e < R.D) :
    |l - R.Ls m T| ≤ lamL R.D T X * R.Ls m T ∧ 1 / 2 ≤ l ∧ l ≤ 2 ^ 18 ∧
    |a e / l - R.As e m T / R.Ls m T| ≤ tolO R.D T X * (R.Ms e m T / R.Ls m T) +
      tolS R.D T X * R.Vs e T + tolF0 R.D T X ∧
    |a e / l| ≤ 50000 := by
  have hbox := Box.of_par hP hT hT1
  have hlow := hM.hlow (by omega)
  have hw := hP.wlow_ge
  have hLc0 : 0 < R.Ls m T := by linarith
  have hLup : R.Ls m T ≤ 128 * 1024 := by
    have := hM.hLup
    have : (T : ℝ) ≤ 1024 := by exact_mod_cast hT1
    nlinarith
  have hlam0 := hbox.lamL_nonneg
  have hlam1 := hbox.lamL_le
  have hel := invL_rel hP hT hT1 hM hL
  have hea := invA_rel hP hT hT1 hM hA he
  have hMs0 := R.Ms_nonneg e m T
  have hMs := R.Ms_le he m T
  have hVs0 := R.Vs_nonneg e T
  have hAM := R.abs_As_le e m T
  obtain ⟨hlpos, hrat⟩ := ratio_alg hbox hlow hel hMs0 hAM hVs0 hea
  have hlb := (abs_le.mp hel)
  refine ⟨hel, ?_, ?_, hrat, ?_⟩
  · nlinarith
  · nlinarith
  · have hVs := R.Vs_le he T
    have hT' : (T : ℝ) ≤ 1024 := by exact_mod_cast hT1
    have hVs1 : R.Vs e T ≤ 2 ^ 31 := hVs.trans (by nlinarith)
    have hMr : R.Ms e m T / R.Ls m T ≤ 2 ^ 15 := by
      rw [div_le_iff₀ hLc0]; exact hMs
    have hMr0 : 0 ≤ R.Ms e m T / R.Ls m T := div_nonneg hMs0 hLc0.le
    have hAr : |R.As e m T / R.Ls m T| ≤ 2 ^ 15 := by
      rw [abs_div, abs_of_pos hLc0]
      refine (div_le_div_of_nonneg_right hAM hLc0.le).trans hMr
    have h1 := hbox.tolO_le
    have h1' := hbox.tolO_nonneg
    have h2 := hbox.tolS_le
    have h2' := hbox.tolS_nonneg
    have h3 := hbox.tolF0_le
    have e1 : tolO R.D T X * (R.Ms e m T / R.Ls m T) ≤ 1 / 2 * 2 ^ 15 :=
      mul_le_mul h1 hMr hMr0 (by norm_num)
    have e2 : tolS R.D T X * R.Vs e T ≤ 1 / 2 ^ 22 * 2 ^ 31 := mul_le_mul h2 hVs1 hVs0 (by norm_num)
    have : |a e / l| ≤ |R.As e m T / R.Ls m T| + |a e / l - R.As e m T / R.Ls m T| := by
      have := abs_add_le (R.As e m T / R.Ls m T) (a e / l - R.As e m T / R.Ls m T)
      rwa [add_sub_cancel] at this
    norm_num at e1 e2 hAr
    linarith

/-- **The output of a row**: `acc / l` divided (`/`, within `q22`) and rounded to `float16`
(within `u16`) is within `tolO2·attnAbs + tolS2·Σ|v| + tolF2` of the exact quotient `w0`. -/
theorem out_bound {D : ℕ} {T X : ℝ} (hb : Box D T X) {r w0 qh o attn V : ℝ}
    (hattn : 0 ≤ attn) (hV : 0 ≤ V) (hw0 : |w0| ≤ attn)
    (hr : |r - w0| ≤ tolO D T X * attn + tolS D T X * V + tolF0 D T X)
    (hq : |qh - r| ≤ q22 * |r| + 1 / 2 ^ 148) (ho : |o - qh| ≤ u16 * |qh| + e16) :
    |o - w0| ≤ tolO2 D T X * attn + tolS2 D T X * V + tolF2 D T X := by
  have h1 := hb.tolO_nonneg
  have h2 := hb.tolS_nonneg
  have h3 := hb.tolF0_nonneg
  have hq0 : 0 ≤ q22 := by unfold q22; norm_num
  have hu0 : 0 ≤ u16 := by unfold u16; norm_num
  set δ := tolO D T X * attn + tolS D T X * V + tolF0 D T X with hδ
  have hδ0 : 0 ≤ δ := by positivity
  have hrr : |r| ≤ attn + δ := by
    have := abs_sub_abs_le_abs_sub r w0
    linarith
  have hqh : |qh| ≤ (1 + q22) * |r| + 1 / 2 ^ 148 := by
    have := abs_sub_abs_le_abs_sub qh r
    nlinarith
  have hlast : lastRel = u16 * (1 + q22) + q22 := rfl
  have hlr0 : 0 ≤ lastRel := by unfold lastRel u16 q22; norm_num
  have hmain : |o - w0| ≤ |o - qh| + |qh - r| + |r - w0| := by
    have h := abs_add_le (o - qh + (qh - r)) (r - w0)
    have h' := abs_add_le (o - qh) (qh - r)
    have e : o - w0 = (o - qh) + (qh - r) + (r - w0) := by ring
    rw [e]; linarith
  have e1 : u16 * |qh| ≤ u16 * ((1 + q22) * |r| + 1 / 2 ^ 148) := mul_le_mul_of_nonneg_left hqh hu0
  have e2 : lastRel * |r| ≤ lastRel * (attn + δ) := mul_le_mul_of_nonneg_left hrr hlr0
  have key : |o - w0| ≤ lastRel * (attn + δ) + δ + (u16 * (1 / 2 ^ 148) + e16 + 1 / 2 ^ 148) := by
    rw [hlast] at e2
    nlinarith
  refine key.trans (le_of_eq ?_)
  unfold tolO2 tolS2 tolF2
  rw [hδ]
  ring

/-- `|log y| ≤ lamL/(1 − lamL)` for `y` within `lamL` of `1`. -/
theorem abs_log_le {y c : ℝ} (hc0 : 0 ≤ c) (hc1 : c < 1) (hy : |y - 1| ≤ c) :
    |Real.log y| ≤ c / (1 - c) := by
  have hyb := abs_le.mp hy
  have hypos : 0 < y := by linarith
  have h1 : Real.log y ≤ y - 1 := Real.log_le_sub_one_of_pos hypos
  have h2 : Real.log (1 / y) ≤ 1 / y - 1 := Real.log_le_sub_one_of_pos (by positivity)
  rw [one_div, Real.log_inv] at h2
  have hcc : c ≤ c / (1 - c) := by
    rw [le_div_iff₀ (by linarith)]; nlinarith
  rw [abs_le]
  constructor
  · have : 1 / y - 1 ≤ c / (1 - c) := by
      rw [div_sub_one hypos.ne', div_le_div_iff₀ hypos (by linarith)]
      nlinarith
    rw [one_div] at this
    linarith
  · linarith

/-- **The log-sum-exp of a row**: `M̂ = m + log₂ l` (device `log2`, then the sum) is within `tolM`
of `m + log₂ Ls`. -/
theorem lse_bound {D : ℕ} {T X : ℝ} (hb : Box D T X) {m l Lc w Mh : ℝ} (hm : |m| ≤ Bm X)
    (hl : |l - Lc| ≤ lamL D T X * Lc) (hLc : wlow D X ≤ Lc) (hl1 : 1 / 2 ≤ l) (hl2 : l ≤ 2 ^ 18)
    (hw : |w - Real.logb 2 l| ≤ q22 * |Real.logb 2 l| + 1 / 2 ^ 149)
    (hM : |Mh - (m + w)| ≤ u32 * |m + w| + 1 / 2 ^ 150) :
    |Mh - (m + Real.logb 2 Lc)| ≤ tolM D T X := by
  have hwl := hb.par.wlow_ge
  have hLc0 : 0 < Lc := by linarith
  have hlam0 := hb.lamL_nonneg
  have hlam1 := hb.lamL_le
  have hlog2 := Real.log_two_gt_d9
  have hlog2' := Real.log_two_lt_d9
  have hlog0 : 0 < Real.log 2 := by linarith
  -- the size of `logb 2 l`
  have hlb : |Real.logb 2 l| ≤ 18 := by
    have hl0 : 0 < l := by linarith
    rw [abs_le]
    constructor
    · have : Real.logb 2 (1 / 2) ≤ Real.logb 2 l :=
        Real.logb_le_logb_of_le (by norm_num) (by norm_num) hl1
      have e : Real.logb 2 (1 / 2) = -1 := by
        rw [one_div, Real.logb_inv, Real.logb_self_eq_one (by norm_num)]
      linarith
    · have : Real.logb 2 l ≤ Real.logb 2 (2 ^ 18) :=
        Real.logb_le_logb_of_le (by norm_num) hl0 hl2
      have e : Real.logb 2 ((2 : ℝ) ^ 18) = 18 := by
        rw [show ((2 : ℝ) ^ 18) = (2 : ℝ) ^ ((18 : ℕ) : ℝ) by norm_num, Real.logb_rpow (by norm_num) (by norm_num)]
        norm_num
      linarith
  have hq0 : 0 ≤ q22 := by unfold q22; norm_num
  have hu0 : 0 ≤ u32 := by unfold u32; norm_num
  have hwa : |w| ≤ 19 := by
    have := abs_sub_abs_le_abs_sub w (Real.logb 2 l)
    have h1 : q22 * |Real.logb 2 l| ≤ q22 * 18 := mul_le_mul_of_nonneg_left hlb hq0
    have : q22 * 18 + 1 / 2 ^ 149 ≤ 1 := by unfold q22; norm_num
    linarith
  have hmw : |m + w| ≤ Bm X + 19 := (abs_add_le _ _).trans (by linarith)
  -- the change of the logarithm
  have hy : |l / Lc - 1| ≤ lamL D T X := by
    rw [show l / Lc - 1 = (l - Lc) / Lc by field_simp, abs_div, abs_of_pos hLc0, div_le_iff₀ hLc0]
    exact hl
  have hlogy := abs_log_le hlam0 (by linarith) hy
  have hdiff : |Real.logb 2 l - Real.logb 2 Lc| ≤ 14427 / 10000 * (lamL D T X / (1 - lamL D T X)) := by
    have hl0 : 0 < l := by linarith
    have e : Real.logb 2 l - Real.logb 2 Lc = Real.log (l / Lc) / Real.log 2 := by
      rw [Real.log_div hl0.ne' hLc0.ne']; unfold Real.logb; ring
    rw [e, abs_div, abs_of_pos hlog0, div_le_iff₀ hlog0]
    have h14 : (1 : ℝ) ≤ 14427 / 10000 * Real.log 2 := by linarith
    have hnn : 0 ≤ lamL D T X / (1 - lamL D T X) := by
      apply div_nonneg hlam0; linarith
    nlinarith
  have hA : |Mh - (m + w)| ≤ u32 * (Bm X + 19) + 1 / 2 ^ 150 := by
    have := mul_le_mul_of_nonneg_left hmw hu0
    linarith
  have hB : |w - Real.logb 2 l| ≤ 18 * q22 + 1 / 2 ^ 149 := by
    have h1 : q22 * |Real.logb 2 l| ≤ q22 * 18 := mul_le_mul_of_nonneg_left hlb hq0
    linarith
  have hmain : |Mh - (m + Real.logb 2 Lc)| ≤ |Mh - (m + w)| + |w - Real.logb 2 l| +
      |Real.logb 2 l - Real.logb 2 Lc| := by
    have e : Mh - (m + Real.logb 2 Lc) = (Mh - (m + w)) + (w - Real.logb 2 l) +
        (Real.logb 2 l - Real.logb 2 Lc) := by ring
    rw [e]
    have h := abs_add_le (Mh - (m + w) + (w - Real.logb 2 l)) (Real.logb 2 l - Real.logb 2 Lc)
    have h' := abs_add_le (Mh - (m + w)) (w - Real.logb 2 l)
    linarith
  unfold tolM
  linarith

end

end Claims.AttnFwd
