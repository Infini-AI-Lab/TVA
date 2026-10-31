import AttnFwdE5m2.Step

/-!
# `attn_fwd_e5m2`: the end of a row (lines 241–245)

* `ratio8`: `acc / l_i` against `S / W`;
* `final_o`: `(acc / l_i).to(float8e5)` (lines 242, 245) against the exact attention of the values
  the kernel uses, `Sx / W`;
* `final_M`: `m_i + log2(l_i)` (line 241) against `log₂ W`;
* the constants of `Model` as the real numbers of a row (`dmax_r`, `rhoA_r`, …, `Fhi_r`, `Flo_r`).
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The ratio -/

/-- **`acc / l_i` against `S / W`.** `a` is within `K·ES + EB` of `K·S`, `l` lies above `Fp·Lw`
and within `β·Lw` of `Lw = K·W`, and `Lw ≥ 1/pE`. -/
theorem ratio8 {a l K S W ES EB Lw Fp beta pE : ℝ}
    (hK : 0 < K) (hW : 0 < W) (hLw : Lw = K * W) (hpE : 1 ≤ pE * Lw)
    (hFp : 0 < Fp) (hl1 : Fp * Lw ≤ l) (hl2 : |l - Lw| ≤ beta * Lw)
    (ha : |a - K * S| ≤ K * ES + EB) (hEB : 0 ≤ EB) :
    |a / l - S / W| ≤ ES / W / Fp + pE * EB / Fp + beta / Fp * |S / W| := by
  have hLw0 : 0 < Lw := by rw [hLw]; positivity
  have hl0 : 0 < l := lt_of_lt_of_le (mul_pos hFp hLw0) hl1
  have e : a / l - S / W = (a - K * S) / l + S / W * ((Lw - l) / l) := by
    rw [hLw]; field_simp; ring
  rw [e]
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul, abs_div (a - K * S) l, abs_div S W, abs_div (Lw - l) l, abs_of_pos hl0, abs_of_pos hW]
  have h1 : |a - K * S| / l ≤ (K * ES + EB) / (Fp * Lw) := by
    have h0 : 0 ≤ K * ES + EB := (abs_nonneg _).trans ha
    calc |a - K * S| / l ≤ (K * ES + EB) / l := div_le_div_of_nonneg_right ha hl0.le
      _ ≤ (K * ES + EB) / (Fp * Lw) := div_le_div_of_nonneg_left h0 (mul_pos hFp hLw0) hl1
  have h2 : (K * ES + EB) / (Fp * Lw) ≤ ES / W / Fp + pE * EB / Fp := by
    have e2 : (K * ES + EB) / (Fp * Lw) = ES / W / Fp + EB / (Fp * Lw) := by
      rw [hLw]; field_simp
    rw [e2]
    have : EB / (Fp * Lw) ≤ pE * EB / Fp := by
      rw [div_le_div_iff₀ (mul_pos hFp hLw0) hFp]
      have := mul_le_mul_of_nonneg_left hpE (mul_nonneg hEB hFp.le)
      nlinarith
    linarith
  have h3 : |Lw - l| / l ≤ beta / Fp := by
    rw [abs_sub_comm, div_le_div_iff₀ hl0 hFp]
    have hb : 0 ≤ beta * Lw := (abs_nonneg _).trans hl2
    have := mul_le_mul hl2 hl1 (mul_nonneg hFp.le hLw0.le) hb
    nlinarith
  have h4 : |S| / W * (|Lw - l| / l) ≤ |S| / W * (beta / Fp) :=
    mul_le_mul_of_nonneg_left h3 (by positivity)
  rw [mul_comm (beta / Fp)]
  linarith

/-! ## The output -/

/-- The sum of the exact weights over the first `t` blocks is positive after a block. -/
theorem psum_ew_pos {R : RowCtx} {t : ℕ} {m : F32} (hm : MInv R t m) : 0 < psum R.al t (ew R) := by
  obtain ⟨j, hj, ha, -⟩ := hm.attained
  exact lt_of_lt_of_le (ew_pos R j) (psum_ge_single (fun i => (ew_pos R i).le) hj ha)

theorem psum_wt_eq (R : RowCtx) (t : ℕ) (m : ℝ) :
    psum R.al t (wt R m) = (2 : ℝ) ^ (-m) * psum R.al t (ew R) := by
  rw [← psum_mul]; congr 1; funext j; exact wt_eq R m j

/-- `l_i` after the loop against the weights in the running maximum's units, with factors
`Fhi ≥ FhR T`, `Flo ≤ FlR T` and an absolute `aL ≥ FhR T·T·absL`. -/
theorem l_final {R : RowCtx} (hR : R.Ok) {T : ℕ} {Fhi Flo aL : ℝ} (hFhi : FhR R T ≤ Fhi)
    (hFlo : Flo ≤ FlR R T) (haL : FhR R T * (T * absLr) ≤ aL) {m l : F32} (hM : MInv R T m)
    (hL : LInv R T m l) :
    0 ≤ aL ∧ 1 ≤ pER R * psum R.al T (wt R m.toReal) ∧
      (Flo - aL * pER R) * psum R.al T (wt R m.toReal) ≤ l.toReal ∧
      l.toReal ≤ (Fhi + aL * pER R) * psum R.al T (wt R m.toReal) ∧
      |l.toReal - psum R.al T (wt R m.toReal)| ≤
        (max (Fhi - 1) (1 - Flo) + aL * pER R) * psum R.al T (wt R m.toReal) := by
  obtain ⟨-, -, hlu, hll⟩ := hL
  set Lw := psum R.al T (wt R m.toReal) with hLwd
  have hLw0 : 0 ≤ Lw := psum_nonneg fun j => (wt_pos _ _ _).le
  have hp := pER_ge hR
  have hpL : 1 ≤ pER R * Lw := by
    have h := psum_wt_ge hM
    rw [← hLwd] at h
    have hp0 : 0 < pER R := by linarith
    rw [div_le_iff₀ hp0] at h; linarith
  have hA0 : 0 ≤ FhR R T * (T * absLr) := by
    have := FhR_ge hR T; unfold absLr; positivity
  have haL0 : 0 ≤ aL := hA0.trans haL
  have haLp : aL ≤ aL * pER R * Lw := by
    have := mul_le_mul_of_nonneg_left hpL haL0; linarith
  have hlu' : l.toReal ≤ Fhi * Lw + aL := by
    have := mul_le_mul_of_nonneg_right hFhi hLw0
    have e : FhR R T * (Lw + T * absLr) = FhR R T * Lw + FhR R T * (T * absLr) := by ring
    linarith
  have hll' : Flo * Lw - aL ≤ l.toReal := by
    have := mul_le_mul_of_nonneg_right hFlo hLw0
    linarith
  refine ⟨haL0, hpL, by nlinarith, by nlinarith, ?_⟩
  have hm1 : Fhi - 1 ≤ max (Fhi - 1) (1 - Flo) := le_max_left _ _
  have hm2 : 1 - Flo ≤ max (Fhi - 1) (1 - Flo) := le_max_right _ _
  have k1 := mul_le_mul_of_nonneg_right hm1 hLw0
  have k2 := mul_le_mul_of_nonneg_right hm2 hLw0
  rw [abs_le]; constructor <;> nlinarith

/-- **The output element** (lines 242 and 245): `(acc / l_i).to(float8e5)` against the exact
attention of the values the kernel uses over the row's keys, `Sx / W`. -/
theorem final_o {P : Platform} (hdivF : Rules.Num.DivFull P) (hcvt : Rules.Num.CvtSatF32E5m2 P)
    {R : RowCtx} (hR : R.Ok) {T : ℕ} (hT : T ≤ 1024)
    {ve : ℕ → ℝ} (hve : ∀ j, |ve j| ≤ 57344) {ES EB : ℝ} (hEB : 0 ≤ EB)
    {Fhi Flo aL : ℝ} (hFhi : FhR R T ≤ Fhi) (hFlo : Flo ≤ FlR R T) (haL : FhR R T * (T * absLr) ≤ aL)
    (hFp : 1 / 2 ^ 15 ≤ Flo - aL * pER R)
    {m l a o' : F32} {o8 : F8E5} (hM : MInv R T m) (hL : LInv R T m l) (hA : AInv R ve T m a ES EB)
    (hdv : P.div .binary32 a l o') (hc : P.cvt Format.binary32 Format.e5m2 .rn o' o8) :
    o8.isFinite = true ∧
      |o8.toReal - Sx R ve T / psum R.al T (ew R)| ≤
        9 / 8 * (1 + 1 / 2 ^ 22) *
            (ES / psum R.al T (ew R) / (Flo - aL * pER R) + pER R * EB / (Flo - aL * pER R) +
              (max (Fhi - 1) (1 - Flo) + aL * pER R) / (Flo - aL * pER R) * |Sx R ve T / psum R.al T (ew R)|) +
          (9 / 8 * (1 / 2 ^ 22) + 1 / 8) * |Sx R ve T / psum R.al T (ew R)| + 9 / 8 * (1 / 2 ^ 148) + 1 / 2 ^ 17 := by
  obtain ⟨haL0, hpL, hl1, -, hl2⟩ := l_final hR hFhi hFlo haL hM hL
  set W := psum R.al T (ew R) with hWd
  set K := (2 : ℝ) ^ (-m.toReal) with hKd
  have hW0 : 0 < W := psum_ew_pos hM
  have hK0 : 0 < K := two_rpow_pos _
  have hLw : psum R.al T (wt R m.toReal) = K * W := psum_wt_eq R T m.toReal
  rw [hLw] at hpL hl1 hl2
  set Fp := Flo - aL * pER R with hFpd
  set beta := max (Fhi - 1) (1 - Flo) + aL * pER R with hbd
  set ref := Sx R ve T / W with hrefd
  have hFp0 : 0 < Fp := lt_of_lt_of_le (by norm_num) hFp
  obtain ⟨haf, hab, hae⟩ := hA
  have hq := ratio8 hK0 hW0 rfl hpL hFp0 hl1 hl2 hae hEB
  rw [← hrefd] at hq
  -- the size of `ref`
  have href : |ref| ≤ 57344 := by
    rw [hrefd, abs_div, abs_of_pos hW0, div_le_iff₀ hW0]
    refine (abs_psum_le _ _ _).trans ?_
    rw [hWd, ← psum_mul]
    exact psum_le fun j _ _ => by
      rw [abs_mul, abs_of_pos (ew_pos R j), mul_comm (57344 : ℝ)]
      exact mul_le_mul_of_nonneg_left (hve j) (ew_pos R j).le
  -- line 242
  have hpE := pER_le hR
  have hpE1 := pER_ge hR
  have hLw1 : 1 / 2 ^ 13 ≤ K * W := by
    rw [div_le_iff₀ (by norm_num)]; nlinarith
  have hl0 : 1 / 2 ^ 28 ≤ l.toReal := by
    have := mul_le_mul hFp hLw1 (by norm_num) hFp0.le
    norm_num at this ⊢; linarith
  have hcr := crA_le hT
  have hal : |a.toReal / l.toReal| ≤ 2 ^ 86 := by
    rw [abs_div, abs_of_pos (by linarith : (0 : ℝ) < l.toReal), div_le_iff₀ (by linarith)]
    nlinarith
  have hal' : |a.toRat / l.toRat| ≤ 2 ^ 126 := by
    have : ((|a.toRat / l.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast; rw [← toReal_def, ← toReal_def]; linarith
    exact_mod_cast this
  have hl0' : l.toRat ≠ 0 := by
    intro h; rw [toReal_def, h] at hl0; norm_num at hl0
  obtain ⟨ho'f, ho'e⟩ := hdivF a l o' hdv haf hL.1 hl0' hal'
  have ho'r : |o'.toReal - a.toReal / l.toReal| ≤ 1 / 2 ^ 22 * |a.toReal / l.toReal| + 1 / 2 ^ 148 := by
    have := (Rat.cast_le (K := ℝ)).mpr ho'e
    push_cast at this
    rw [show (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 by norm_num, show (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 by norm_num] at this
    simp only [← toReal_def] at this
    exact this
  -- line 245
  obtain ⟨ho8f, ho8e⟩ := f8_cvt_sat hcvt hc ho'f
  obtain ⟨hc1, hc2⟩ := clamp_err (x := o'.toReal) href
  refine ⟨ho8f, ?_⟩
  set Rq := ES / W / Fp + pER R * EB / Fp + beta / Fp * |ref| with hRq
  have hd1 : |a.toReal / l.toReal| ≤ |ref| + Rq := by
    have := abs_sub_abs_le_abs_sub (a.toReal / l.toReal) ref; linarith
  have hd2 : |o'.toReal - ref| ≤ (1 + 1 / 2 ^ 22) * Rq + 1 / 2 ^ 22 * |ref| + 1 / 2 ^ 148 := by
    have := abs_sub_le o'.toReal (a.toReal / l.toReal) ref
    have := mul_le_mul_of_nonneg_left hd1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22)
    linarith
  have hd3 : |o'.toReal| ≤ |ref| + |o'.toReal - ref| := by
    have := abs_sub_abs_le_abs_sub o'.toReal ref; linarith
  have := abs_sub_le o8.toReal (max (-57344) (min o'.toReal 57344)) ref
  linarith

/-! ## The log-sum-exp -/

theorem logb_two_rpow (x : ℝ) : Real.logb 2 ((2 : ℝ) ^ x) = x :=
  Real.logb_rpow (by norm_num) (by norm_num)

/-- `|log₂ r| ≤ max(log₂ b, −log₂ a)` for `r ∈ [a, b]`, `a > 0`. -/
theorem abs_logb_le {r a b : ℝ} (ha : 0 < a) (h1 : a ≤ r) (h2 : r ≤ b) :
    |Real.logb 2 r| ≤ max (Real.logb 2 b) (-Real.logb 2 a) := by
  have hr : 0 < r := lt_of_lt_of_le ha h1
  have k1 : Real.logb 2 r ≤ Real.logb 2 b := Real.logb_le_logb_of_le (by norm_num) hr h2
  have k2 : Real.logb 2 a ≤ Real.logb 2 r := Real.logb_le_logb_of_le (by norm_num) ha h1
  rw [abs_le]; constructor
  · have := le_max_right (Real.logb 2 b) (-Real.logb 2 a); linarith
  · have := le_max_left (Real.logb 2 b) (-Real.logb 2 a); linarith

/-- **The log-sum-exp element** (lines 241, 244): `m_i + log2(l_i)` against `log₂ W`. -/
theorem final_M {P : Platform} (hlog : Rules.Num.Log2F32 P) {R : RowCtx} (hR : R.Ok)
    {T : ℕ} (hT : T ≤ 1024)
    {Fhi Flo aL : ℝ} (hFhi : FhR R T ≤ Fhi) (hFlo : Flo ≤ FlR R T) (haL : FhR R T * (T * absLr) ≤ aL)
    (hFp : 1 / 2 ^ 15 ≤ Flo - aL * pER R) (hFh2 : Fhi + aL * pER R ≤ 2 ^ 15)
    {m l lg Mv : F32} (hM : MInv R T m) (hL : LInv R T m l)
    (hlg : P.math .log2 .binary32 l lg) (hadd : FloatV.NaNEq (FloatV.add m lg) Mv) :
    Mv.isFinite = true ∧
      |Mv.toReal - Real.logb 2 (psum R.al T (ew R))| ≤
        (1 + 1 / 2 ^ 24) * (max (Real.logb 2 (Fhi + aL * pER R)) (-Real.logb 2 (Flo - aL * pER R)) +
          (max (Real.logb 2 (Fhi + aL * pER R)) (-Real.logb 2 (Flo - aL * pER R)) + R.E + 16) / 2 ^ 22 +
            1 / 2 ^ 149) + 1 / 2 ^ 150 + 1 / 2 ^ 24 * |Real.logb 2 (psum R.al T (ew R))| := by
  obtain ⟨haL0, hpL, hl1, hlu, -⟩ := l_final hR hFhi hFlo haL hM hL
  set Lw := psum R.al T (wt R m.toReal) with hLwd
  set W := psum R.al T (ew R) with hWd
  set lgB := max (Real.logb 2 (Fhi + aL * pER R)) (-Real.logb 2 (Flo - aL * pER R)) with hlgBd
  have hW0 : 0 < W := psum_ew_pos hM
  have hLw : Lw = (2 : ℝ) ^ (-m.toReal) * W := psum_wt_eq R T m.toReal
  have hLw0 : 0 < Lw := by rw [hLw]; exact mul_pos (two_rpow_pos _) hW0
  have hFp0 : 0 < Flo - aL * pER R := lt_of_lt_of_le (by norm_num) hFp
  have hl0 : 0 < l.toReal := lt_of_lt_of_le (mul_pos hFp0 hLw0) hl1
  -- `log₂(l/Lw)`
  have hr : |Real.logb 2 (l.toReal / Lw)| ≤ lgB := by
    refine abs_logb_le hFp0 ?_ ?_
    · rw [le_div_iff₀ hLw0]; exact hl1
    · rw [div_le_iff₀ hLw0]; exact hlu
  have hlgB : lgB ≤ 15 := by
    rw [hlgBd]
    refine max_le ?_ ?_
    · have hpos : 0 < Fhi + aL * pER R := by
        have := hFhi.trans' (FhR_ge hR T); have := pER_ge hR; nlinarith
      have := Real.logb_le_logb_of_le (b := 2) (by norm_num) hpos hFh2
      rwa [show (2 : ℝ) ^ 15 = (2 : ℝ) ^ (15 : ℝ) by norm_num, logb_two_rpow] at this
    · have := Real.logb_le_logb_of_le (b := 2) (by norm_num) (by norm_num) hFp
      rw [show (1 : ℝ) / 2 ^ 15 = (2 : ℝ) ^ (-15 : ℝ) by norm_num, logb_two_rpow] at this
      linarith
  have hlgB0 : 0 ≤ lgB := (abs_nonneg _).trans hr
  -- `log₂ Lw`
  have hE0 := hR.E_nonneg
  have hpE : pER R = (2 : ℝ) ^ R.E := rfl
  have hLw1 : -R.E ≤ Real.logb 2 Lw := by
    have h := psum_wt_ge hM
    rw [← hLwd, hpE, ← two_rpow_neg] at h
    have := Real.logb_le_logb_of_le (b := 2) (by norm_num) (two_rpow_pos _) h
    rwa [logb_two_rpow] at this
  have hLw2 : Real.logb 2 Lw ≤ 16 + R.E := by
    have h := psum_wt_le hR hM
    rw [← hLwd] at h
    have hT' : 64 * (T : ℝ) ≤ 2 ^ 16 := by
      have : (T : ℝ) ≤ 1024 := by exact_mod_cast hT
      linarith
    have h2 : Lw ≤ (2 : ℝ) ^ (16 + R.E) := by
      rw [two_rpow_add, show (2 : ℝ) ^ (16 : ℝ) = 2 ^ 16 by norm_num, ← hpE]
      exact h.trans (mul_le_mul_of_nonneg_right hT' (by linarith [pER_ge hR]))
    have := Real.logb_le_logb_of_le (b := 2) (by norm_num) hLw0 h2
    rwa [logb_two_rpow] at this
  have hsplit : Real.logb 2 l.toReal = Real.logb 2 (l.toReal / Lw) + Real.logb 2 Lw := by
    rw [Real.logb_div hl0.ne' hLw0.ne']; ring
  have hlogl : |Real.logb 2 l.toReal| ≤ lgB + R.E + 16 := by
    rw [hsplit]
    refine (abs_add_le _ _).trans ?_
    have : |Real.logb 2 Lw| ≤ 16 + R.E := abs_le.mpr ⟨by linarith, hLw2⟩
    linarith
  -- line 241: `tl.math.log2(l_i)`
  obtain ⟨g, hg, hacc⟩ := hlog
  have hg' : g = Real.logb 2 := by
    have e : MathFn.ref .log2 = some (Real.logb 2) := rfl
    rw [e] at hg; exact (Option.some.inj hg).symm
  subst hg'
  have hE13 := hR.E_le
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.logb 2 l.toReal| + 2 ^ (-149 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    rw [maxVal32]; norm_num; nlinarith
  obtain ⟨hlgf, hlge⟩ := hacc l lg hlg hL.1 (Set.mem_univ _) (show l.toReal ∈ Set.Ioi 0 from hl0) hroom
  rw [show (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 by norm_num, show (2 : ℝ) ^ (-149 : ℤ) = 1 / 2 ^ 149 by norm_num]
    at hlge
  -- `m + log₂ Lw = log₂ W`
  have hlse : m.toReal + Real.logb 2 Lw = Real.logb 2 W := by
    rw [hLw, Real.logb_mul (two_rpow_pos _).ne' hW0.ne', logb_two_rpow]; ring
  have hΔ : |m.toReal + lg.toReal - Real.logb 2 W| ≤ lgB + (lgB + R.E + 16) / 2 ^ 22 + 1 / 2 ^ 149 := by
    have e : m.toReal + lg.toReal - Real.logb 2 W = (lg.toReal - Real.logb 2 l.toReal) +
        Real.logb 2 (l.toReal / Lw) := by rw [← hlse, hsplit]; ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    have := mul_le_mul_of_nonneg_left hlogl (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22)
    have e2 : 1 / 2 ^ 22 * (lgB + R.E + 16) = (lgB + R.E + 16) / 2 ^ 22 := by ring
    linarith
  have hroom2 : |m.toReal + lg.toReal| ≤ 2 ^ 120 := by
    have := hM.bound; have := hR.Y_le
    have := abs_add_le m.toReal lg.toReal
    have := abs_sub_abs_le_abs_sub lg.toReal (Real.logb 2 l.toReal)
    have : 1 / 2 ^ 22 * |Real.logb 2 l.toReal| ≤ 1 := by nlinarith
    linarith
  obtain ⟨hMvf, hMve⟩ := f32_add hadd hM.fin hlgf hroom2
  refine ⟨hMvf, ?_⟩
  have t1 := abs_sub_le Mv.toReal (m.toReal + lg.toReal) (Real.logb 2 W)
  have t2 : |m.toReal + lg.toReal| ≤ |Real.logb 2 W| + |m.toReal + lg.toReal - Real.logb 2 W| := by
    have := abs_sub_abs_le_abs_sub (m.toReal + lg.toReal) (Real.logb 2 W); linarith
  have := div_le_div_of_nonneg_right t2 (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
  have := div_le_div_of_nonneg_right hΔ (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
  have e3 : |Real.logb 2 W| / 2 ^ 24 = 1 / 2 ^ 24 * |Real.logb 2 W| := by ring
  linarith

/-! ## The constants of `Model` as real numbers -/

section Casts

variable {R : RowCtx} {D N : ℕ} {X : ℚ}

theorem u32_r : ((u32 : ℚ) : ℝ) = 1 / 2 ^ 24 := by unfold u32; push_cast; ring

theorem rho8_r : ((rho8 : ℚ) : ℝ) = rho8r := by unfold rho8; exact relDot8_64r

theorem omega8_r : ((omega8 : ℚ) : ℝ) = om8r := by unfold omega8 om8r; push_cast; ring

theorem absA_r : ((absA : ℚ) : ℝ) = absAr := by unfold absA absAr; push_cast; ring

theorem gam63_r : ((gam63 : ℚ) : ℝ) = gamr := by unfold gam63 gamr u32; push_cast; ring

theorem dmax_r (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((dmax D X : ℚ) : ℝ) = dmaxR R := by
  unfold dmax dmaxR ybound; push_cast; rw [hY, hE]

theorem rhoA_r (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((rhoA D X : ℚ) : ℝ) = rhoAR R := by
  unfold rhoA rhoAR; push_cast; rw [rho8_r, dmax_r hY hE, u32_r]

theorem pE_r (hE : R.E = (expErr D X : ℝ)) : pE D X = pER R := by
  unfold pE pER; rw [hE]

theorem psiP_r (hE : R.E = (expErr D X : ℝ)) : psiP D X = psiPR R := by
  unfold psiP psiPR; rw [pE_r hE]

theorem psi8_r (hE : R.E = (expErr D X : ℝ)) : psi8 D X = psi8R R := by
  unfold psi8 psi8R; rw [psiP_r hE]

theorem psiA_r (hE : R.E = (expErr D X : ℝ)) : psiA D X = psiAR R := by
  unfold psiA psiAR; rw [psi8_r hE, rho8_r]

theorem stepL_r (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((stepL D X : ℚ) : ℝ) = stepLR R := by
  unfold stepL stepLR; push_cast; rw [dmax_r hY hE, u32_r]; ring

theorem qL_r (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    ((qL D X : ℚ) : ℝ) = qLR R := by
  unfold qL qLR; push_cast; rw [dmax_r hY hE, u32_r]; ring

theorem Fhi_r (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    Fhi D N X = FhR R (nblocks N) := by
  unfold Fhi FhR; rw [gam63_r, psiP_r hE, stepL_r hY hE]

theorem Flo_r (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) :
    Flo D N X = FlR R (nblocks N) := by
  unfold Flo FlR; rw [u32_r, gam63_r, pE_r hE, qL_r hY hE]; ring

theorem aL_r (N : ℕ) : ((aL N : ℚ) : ℝ) = (nblocks N : ℝ) * (1 / 2 ^ 64) := by
  unfold aL absL; push_cast; ring

end Casts

/-! ## The numeric conditions of `final_o` and `final_M` at the covered extremes -/

/-- With `T ≤ nblocks N ≤ 1024` blocks, the `Model` factors bound the row's: `FhR T ≤ Fhi`,
`Flo ≤ FlR T`, `FhR T·T·absL ≤ aL`, and the lower bound `Flo − aL·pE ≥ 2^-15`,
`Fhi + aL·pE ≤ 2^15`. -/
theorem model_factors {R : RowCtx} (hR : R.Ok) {D N : ℕ} {X : ℚ} (hY : R.Y = 3 / 2 * (X : ℝ))
    (hE : R.E = (expErr D X : ℝ)) (hN : N ≤ 2 ^ 16) {T : ℕ} (hT : T ≤ nblocks N) :
    FhR R T ≤ Fhi D N X ∧ Flo D N X ≤ FlR R T ∧ FhR R T * (T * absLr) ≤ ((aL N : ℚ) : ℝ) ∧
      1 / 2 ^ 15 ≤ Flo D N X - ((aL N : ℚ) : ℝ) * pER R ∧ Fhi D N X + ((aL N : ℚ) : ℝ) * pER R ≤ 2 ^ 15 := by
  have hnb : nblocks N ≤ 1024 := by unfold nblocks; omega
  have hnb' : (nblocks N : ℝ) ≤ 1024 := by exact_mod_cast hnb
  have hTr : (T : ℝ) ≤ nblocks N := by exact_mod_cast hT
  rw [Fhi_r hY hE, Flo_r hY hE, aL_r]
  have h1 : FhR R T ≤ FhR R (nblocks N) := by
    unfold FhR
    have := pow_le_pow_right₀ (stepLR_one hR) hT
    have : 0 ≤ (1 + gamr) * (1 + psiPR R) := by have := gamr_nonneg; have := psiPR_nonneg hR; positivity
    nlinarith
  have h2 : FlR R (nblocks N) ≤ FlR R T := by
    rw [FlR_eq, FlR_eq]
    apply div_le_div_of_nonneg_right _ (by linarith [pER_ge hR])
    have := pow_le_pow_of_le_one (qLR_nonneg hR) (qLR_le hR) hT
    have : 0 ≤ (1 - 3 / 2 ^ 24) * (1 - gamr) * (1 - 1 / 2 ^ 22) := by
      have := gamr_le; apply mul_nonneg (mul_nonneg (by norm_num) (by linarith)) (by norm_num)
    nlinarith
  have hFh := FhR_le hR (le_trans hT hnb)
  have hFhn := FhR_le hR hnb
  have hp := pER_le hR
  have hp1 := pER_ge hR
  have haL : (nblocks N : ℝ) * (1 / 2 ^ 64) * pER R ≤ 1 / 2 ^ 41 := by
    have := mul_le_mul (mul_le_mul_of_nonneg_right hnb' (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 64)) hp
      (by linarith) (by norm_num)
    norm_num at this ⊢; linarith
  refine ⟨h1, h2, ?_, ?_, ?_⟩
  · unfold absLr
    have := mul_le_mul hFh (mul_le_mul_of_nonneg_right hTr (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 80))
      (by positivity) (by norm_num)
    have k : (2 : ℝ) ^ 14 * ((nblocks N : ℝ) * (1 / 2 ^ 80)) ≤ (nblocks N : ℝ) * (1 / 2 ^ 64) := by
      have : (0 : ℝ) ≤ nblocks N := Nat.cast_nonneg _
      nlinarith
    linarith
  · have := FlR_ge hR hnb
    have h3 : (1 : ℝ) / 2 ^ 14 ≤ 1 / 2 / pER R := by
      rw [div_div, div_le_div_iff₀ (by norm_num) (by linarith)]; nlinarith
    norm_num at h3 haL ⊢; linarith
  · norm_num at haL ⊢; linarith

end Kernels.AttnFwdE5m2
