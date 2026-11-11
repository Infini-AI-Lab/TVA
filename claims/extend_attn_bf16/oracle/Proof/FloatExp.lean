import Proof.CoreLemmas

/-!
# `tl.exp` of a rounded difference, as the core analysis wants it

* `f32_sub`: the `float32` difference is finite and within one rounding.
* `exp_acc`: the device fact for `tl.exp`, for an argument at most `2^-100`.
* `exp_ok`: `tl.exp (a − b)` for `a ≤ b`: the result is `exp (a − b)` times a factor within
  `expm1 (e1 + e2·(b − a))` of `1`, up to `eps0` (`TripStat.scale`, `TripStat.weight`).
* `exp_le_one`: the result of `tl.exp` of a non-positive argument is at most `1 + 2^-20`.
* `exp_negInf`: `tl.exp (−∞ − b) = 0`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- `x - y` on `float32` (up to NaN payload): finite and within one rounding. -/
theorem f32_sub {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hb : |x.toReal - y.toReal| ≤ 2 ^ 100) (h : FloatV.NaNEq (FloatV.sub x y) r) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal - y.toReal)| ≤ 1 / 2 ^ 24 * |x.toReal - y.toReal| + 1 / 2 ^ 150 := by
  have hm : (2 : ℝ) ^ 100 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨h1, h2⟩ := FloatV.sub_err_real Format.binary32_regular hx hy (hb.trans hm)
  obtain ⟨hf, -, hr⟩ := h.exact_of_isFinite h1
  refine ⟨hf, ?_⟩
  rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
  rw [hr]
  norm_num at h2 ⊢
  linarith

/-- The device fact for `tl.exp` on `float32`, for a finite argument `d ≤ 2^-100`: finite and
within `(2^-22 + 2^-23·|d|)·exp d + 2^-140`. -/
theorem exp_acc {P : Platform} (hP : Device P) {d r : F32} (hd : d.isFinite = true)
    (hd0 : d.toReal ≤ 1 / 2 ^ 100) (hr : P.math .exp .binary32 d r) :
    r.isFinite = true ∧
      |r.toReal - Real.exp d.toReal| ≤
        (1 / 2 ^ 22 + 1 / 2 ^ 23 * |d.toReal|) * Real.exp d.toReal + 1 / 2 ^ 140 := by
  obtain ⟨ref, href, hacc⟩ := hP.expF32 |d.toReal| (abs_nonneg _)
  have href' : ref = Real.exp := by
    have h := href
    simp only [TVA.MathFn.ref] at h
    exact (Option.some.inj h).symm
  rw [href'] at hacc
  have hmem : d.toReal ∈ Set.Icc (-|d.toReal|) |d.toReal| :=
    ⟨neg_abs_le _, le_abs_self _⟩
  have hexp0 : 0 < Real.exp d.toReal := Real.exp_pos _
  have hmax : (2 : ℝ) ^ 100 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  have hroom : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |d.toReal|)) * |Real.exp d.toReal| +
      2 ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_pos hexp0]
    refine le_trans ?_ hmax
    by_cases h0 : d.toReal ≤ 0
    · have hy : |d.toReal| = -d.toReal := abs_of_nonpos h0
      have h1 : Real.exp d.toReal ≤ 1 := Real.exp_le_one_iff.mpr h0
      -- `y · exp (−y) ≤ 1`
      have h2 : |d.toReal| * Real.exp d.toReal ≤ 1 := by
        rw [hy]
        have := Real.add_one_le_exp (-d.toReal)
        have h3 : Real.exp (-d.toReal) * Real.exp d.toReal = 1 := by
          rw [← Real.exp_add]; simp
        nlinarith [Real.exp_pos d.toReal, Real.exp_pos (-d.toReal)]
      have e : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * |d.toReal|)) * Real.exp d.toReal =
          Real.exp d.toReal * (1 + 2 ^ (-22 : ℤ)) + 2 ^ (-23 : ℤ) * (|d.toReal| * Real.exp d.toReal) := by
        ring
      rw [e]
      have h4 : Real.exp d.toReal * (1 + 2 ^ (-22 : ℤ)) ≤ 1 + 2 ^ (-22 : ℤ) := by
        have : (0 : ℝ) ≤ 1 + 2 ^ (-22 : ℤ) := by positivity
        nlinarith
      have h5 : 2 ^ (-23 : ℤ) * (|d.toReal| * Real.exp d.toReal) ≤ 2 ^ (-23 : ℤ) * 1 :=
        mul_le_mul_of_nonneg_left h2 (by positivity)
      have hnum : (1 : ℝ) + 2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * 1 + 2 ^ (-140 : ℤ) ≤ 2 ^ 100 := by norm_num
      linarith
    · have h1 : d.toReal ≤ 1 / 2 := hd0.trans (by norm_num)
      have h2 := exp_le_quad (not_le.mp h0).le h1
      have h3 : |d.toReal| = d.toReal := abs_of_pos (not_le.mp h0)
      rw [h3]
      have h4 : d.toReal ≤ 1 / 2 ^ 100 := hd0
      have hd2 : d.toReal ^ 2 ≤ 1 / 4 := by nlinarith [not_le.mp h0]
      have h5 : Real.exp d.toReal ≤ 2 := by linarith
      have h6 : (1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * d.toReal)) ≤ 2 := by
        have h8 : (2 : ℝ) ^ (-23 : ℤ) * d.toReal ≤ 2 ^ (-23 : ℤ) * (1 / 2 ^ 100) :=
          mul_le_mul_of_nonneg_left h4 (by positivity)
        calc 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * d.toReal)
            ≤ 1 + (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * (1 / 2 ^ 100)) := by linarith
          _ ≤ 2 := by norm_num
      have h7 := mul_le_mul h6 h5 (Real.exp_pos _).le (by norm_num)
      have hnum : (2 : ℝ) * 2 + 2 ^ (-140 : ℤ) ≤ 2 ^ 100 := by norm_num
      linarith
  obtain ⟨hf, herr⟩ := hacc d r hr hd hmem (Set.mem_univ _) hroom
  refine ⟨hf, ?_⟩
  rw [abs_of_pos hexp0] at herr
  norm_num at herr ⊢
  exact herr

/-- Clamping a value that is within `θ·E + ζ` of `E ≥ 0` to the interval `E (1 ± θ)` moves it by
at most `ζ`. -/
theorem clamp_exists {r E θ ζ : ℝ} (hθ : 0 ≤ θ) (hE : 0 ≤ E) (hζ : 0 ≤ ζ)
    (h : |r - E| ≤ θ * E + ζ) :
    ∃ r' : ℝ, |r - r'| ≤ ζ ∧ |r' - E| ≤ θ * E := by
  have hθE : 0 ≤ θ * E := mul_nonneg hθ hE
  have hb := abs_le.mp h
  by_cases h1 : r - E ≤ θ * E
  · by_cases h2 : -(θ * E) ≤ r - E
    · exact ⟨r, by simp; linarith, abs_le.mpr ⟨by linarith, by linarith⟩⟩
    · refine ⟨E - θ * E, ?_, ?_⟩
      · rw [abs_le]; constructor <;> linarith
      · rw [abs_le]; constructor <;> linarith
  · refine ⟨E + θ * E, ?_, ?_⟩
    · rw [abs_le]; constructor <;> linarith
    · rw [abs_le]; constructor <;> linarith

/-- **`tl.exp` of a `float32` difference `a − b ≤ 0`** (gap `y = b − a ≤ 2^15`, the difference
rounded, then the device's `tl.exp`): the result is finite and `exp (a − b)` times a factor `f`
within `expm1 (e1 + e2·y)` of `1`, up to `eps0`. -/
theorem exp_ok {P : Platform} (hP : Device P) {a b d r : F32} (ha : a.isFinite = true)
    (hb : b.isFinite = true) (hab : a.toReal ≤ b.toReal) (hgap : b.toReal - a.toReal ≤ 2 ^ 15)
    (hd : FloatV.NaNEq (FloatV.sub a b) d) (hr : P.math .exp .binary32 d r) :
    r.isFinite = true ∧ ∃ f : ℝ, |f - 1| ≤ expm1 (e1 + e2 * (b.toReal - a.toReal)) ∧
      |r.toReal - Real.exp (a.toReal - b.toReal) * f| ≤ eps0 := by
  set y := b.toReal - a.toReal with hy
  have hy0 : 0 ≤ y := by rw [hy]; linarith
  have hxabs : |a.toReal - b.toReal| = y := by
    rw [abs_of_nonpos (by linarith)]; rw [hy]; ring
  obtain ⟨hdf, hde⟩ := f32_sub ha hb (by rw [hxabs]; exact hgap.trans (by norm_num)) hd
  rw [hxabs] at hde
  set x := a.toReal - b.toReal with hx
  have hxy : x = -y := by rw [hx, hy]; ring
  set xt := d.toReal with hxt
  set δ := 1 / 2 ^ 24 * y + 1 / 2 ^ 150 with hδ
  have hδ0 : 0 ≤ δ := by rw [hδ]; positivity
  have hδ1 : δ ≤ 1 / 2 ^ 8 := by
    rw [hδ]
    have : 1 / 2 ^ 24 * y ≤ 1 / 2 ^ 24 * 2 ^ 15 := mul_le_mul_of_nonneg_left hgap (by norm_num)
    norm_num at this ⊢; linarith
  have hxt0 : xt ≤ 1 / 2 ^ 100 := by
    have := (abs_le.mp hde).2
    rw [hxy] at this
    have h2 : 1 / 2 ^ 24 * y ≤ y := by nlinarith
    have h3 : (1 : ℝ) / 2 ^ 150 ≤ 1 / 2 ^ 100 := by norm_num
    nlinarith
  obtain ⟨hrf, hre⟩ := exp_acc hP hdf hxt0 hr
  refine ⟨hrf, ?_⟩
  set θ := 1 / 2 ^ 22 + 1 / 2 ^ 23 * |xt| with hθ
  have hθ0 : 0 ≤ θ := by rw [hθ]; positivity
  have hxtabs : |xt| ≤ y + δ := by
    have := abs_sub_abs_le_abs_sub xt x
    linarith [hde, hxabs]
  have hθA : θ + δ ≤ e1 + e2 * y := by
    have h3 : 1 / 2 ^ 23 * |xt| ≤ 1 / 2 ^ 23 * (y + δ) := mul_le_mul_of_nonneg_left hxtabs (by norm_num)
    have hcoef : (1 / 2 ^ 23 + 1 / 2 ^ 23 * (1 / 2 ^ 24) + 1 / 2 ^ 24 : ℝ) ≤ e2 := by unfold e2; norm_num
    have hconst : (1 / 2 ^ 22 + 1 / 2 ^ 23 * (1 / 2 ^ 150) + 1 / 2 ^ 150 : ℝ) ≤ e1 := by unfold e1; norm_num
    have h4 := mul_le_mul_of_nonneg_right hcoef hy0
    have e : θ + δ = 1 / 2 ^ 22 + 1 / 2 ^ 23 * |xt| + δ := by rw [hθ]
    rw [e]
    have e2' : 1 / 2 ^ 23 * (y + δ) + δ = (1 / 2 ^ 23 + 1 / 2 ^ 23 * (1 / 2 ^ 24) + 1 / 2 ^ 24) * y +
        (1 / 2 ^ 23 * (1 / 2 ^ 150) + 1 / 2 ^ 150) := by rw [hδ]; ring
    linarith
  have hθ1 : θ ≤ 1 / 2 := by
    have : θ ≤ e1 + e2 * y := by linarith
    have h4 : e2 * y ≤ e2 * 2 ^ 15 := mul_le_mul_of_nonneg_left hgap e2_pos.le
    unfold e1 e2 at *
    norm_num at h4 this ⊢
    linarith
  obtain ⟨r', hrr', hr'E⟩ := clamp_exists (r := r.toReal) (E := Real.exp xt) (θ := θ) (ζ := 1 / 2 ^ 140)
    hθ0 (Real.exp_pos _).le (by norm_num) hre
  have hEx : Real.exp xt = Real.exp x * Real.exp (xt - x) := by rw [← Real.exp_add]; congr 1; ring
  have hexpx : 0 < Real.exp x := Real.exp_pos _
  refine ⟨r' / Real.exp x, ?_, ?_⟩
  · -- the factor
    set A := e1 + e2 * y with hA
    have hr'b := abs_le.mp hr'E
    have hq : r' / Real.exp x ≤ Real.exp (xt - x) * (1 + θ) := by
      rw [div_le_iff₀ hexpx, hEx] at *
      nlinarith [hr'b.2]
    have hq' : Real.exp (xt - x) * (1 - θ) ≤ r' / Real.exp x := by
      rw [le_div_iff₀ hexpx]
      rw [hEx] at hr'b
      nlinarith [hr'b.1]
    have hxx : |xt - x| ≤ δ := hde
    have hup : r' / Real.exp x ≤ Real.exp A := by
      refine hq.trans ?_
      calc Real.exp (xt - x) * (1 + θ) ≤ Real.exp δ * Real.exp θ := by
            apply mul_le_mul (Real.exp_le_exp.mpr (le_trans (le_abs_self _) hxx))
              (by linarith [Real.add_one_le_exp θ]) (by linarith) (Real.exp_pos _).le
        _ = Real.exp (δ + θ) := by rw [← Real.exp_add]
        _ ≤ Real.exp A := Real.exp_le_exp.mpr (by linarith)
    have hlo : 1 - A ≤ r' / Real.exp x := by
      refine le_trans ?_ hq'
      have h1 : 1 - δ ≤ Real.exp (xt - x) := by
        have := Real.add_one_le_exp (xt - x)
        have := (abs_le.mp hxx).1
        linarith
      have h2 : 0 ≤ 1 - θ := by linarith
      nlinarith [mul_nonneg hδ0 hθ0]
    rw [abs_le]
    unfold expm1
    constructor
    · have := Real.add_one_le_exp A
      linarith
    · linarith
  · -- the absolute part
    have e : Real.exp x * (r' / Real.exp x) = r' := by field_simp
    rw [show Real.exp (a.toReal - b.toReal) = Real.exp x from rfl, e]
    refine hrr'.trans ?_
    unfold eps0; norm_num

/-- `(1 + 2^-22 + 2^-23·|z|)·exp z ≤ 1 + 2^-21` for `z ≤ 2^-100`. -/
theorem exp_mul_le {z : ℝ} (hz : z ≤ 1 / 2 ^ 100) :
    (1 + 1 / 2 ^ 22 + 1 / 2 ^ 23 * |z|) * Real.exp z ≤ 1 + 1 / 2 ^ 21 := by
  by_cases h0 : z ≤ 0
  · have hy : |z| = -z := abs_of_nonpos h0
    have h1 : Real.exp z ≤ 1 := Real.exp_le_one_iff.mpr h0
    have h2 : |z| * Real.exp z ≤ 1 := by
      rw [hy]
      have := Real.add_one_le_exp (-z)
      have h3 : Real.exp (-z) * Real.exp z = 1 := by rw [← Real.exp_add]; simp
      nlinarith [Real.exp_pos z, Real.exp_pos (-z)]
    have e : (1 + 1 / 2 ^ 22 + 1 / 2 ^ 23 * |z|) * Real.exp z =
        Real.exp z * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 23 * (|z| * Real.exp z) := by ring
    rw [e]
    have h4 : Real.exp z * (1 + 1 / 2 ^ 22) ≤ 1 + 1 / 2 ^ 22 := by
      have : (0 : ℝ) ≤ 1 + 1 / 2 ^ 22 := by positivity
      nlinarith
    have h5 : 1 / 2 ^ 23 * (|z| * Real.exp z) ≤ 1 / 2 ^ 23 * 1 :=
      mul_le_mul_of_nonneg_left h2 (by positivity)
    have hnum : (1 : ℝ) + 1 / 2 ^ 22 + 1 / 2 ^ 23 * 1 ≤ 1 + 1 / 2 ^ 21 := by norm_num
    linarith
  · have hz0 := not_le.mp h0
    have h1 : z ≤ 1 / 2 := hz.trans (by norm_num)
    have h2 := exp_le_quad hz0.le h1
    have hd2 : z ^ 2 ≤ 1 / 4 := by nlinarith
    have h3 : |z| = z := abs_of_pos hz0
    rw [h3]
    have h5 : Real.exp z ≤ 1 + 1 / 2 ^ 98 := by
      have : z ≤ 1 / 2 ^ 100 := hz
      have hz2 : z ^ 2 ≤ z * (1 / 2 ^ 100) := by nlinarith
      nlinarith
    have h6 : (1 + 1 / 2 ^ 22 + 1 / 2 ^ 23 * z) ≤ 1 + 1 / 2 ^ 22 + 1 / 2 ^ 123 := by
      have : 1 / 2 ^ 23 * z ≤ 1 / 2 ^ 23 * (1 / 2 ^ 100) := mul_le_mul_of_nonneg_left hz (by positivity)
      have hn : (1 : ℝ) / 2 ^ 23 * (1 / 2 ^ 100) = 1 / 2 ^ 123 := by norm_num
      linarith
    have h7 := mul_le_mul h6 h5 (Real.exp_pos _).le (by norm_num)
    have hnum : (1 + 1 / 2 ^ 22 + 1 / 2 ^ 123 : ℝ) * (1 + 1 / 2 ^ 98) ≤ 1 + 1 / 2 ^ 21 := by norm_num
    linarith

/-- **`tl.exp` of an argument `d ≤ 2^-100`** is finite and at most `1 + 2^-20` in magnitude. -/
theorem exp_le_one {P : Platform} (hP : Device P) {d r : F32} (hd : d.isFinite = true)
    (hd0 : d.toReal ≤ 1 / 2 ^ 100) (hr : P.math .exp .binary32 d r) :
    r.isFinite = true ∧ |r.toReal| ≤ 1 + 1 / 2 ^ 20 := by
  obtain ⟨hf, herr⟩ := exp_acc hP hd hd0 hr
  refine ⟨hf, ?_⟩
  have h1 := exp_mul_le hd0
  have h2 : |r.toReal| ≤ |Real.exp d.toReal| + |r.toReal - Real.exp d.toReal| := by
    have := abs_add_le (Real.exp d.toReal) (r.toReal - Real.exp d.toReal)
    rw [add_sub_cancel] at this; linarith
  rw [abs_of_pos (Real.exp_pos _)] at h2
  have e : (1 / 2 ^ 22 + 1 / 2 ^ 23 * |d.toReal|) * Real.exp d.toReal + Real.exp d.toReal =
      (1 + 1 / 2 ^ 22 + 1 / 2 ^ 23 * |d.toReal|) * Real.exp d.toReal := by ring
  have hnum : (1 : ℝ) + 1 / 2 ^ 21 + 1 / 2 ^ 140 ≤ 1 + 1 / 2 ^ 20 := by norm_num
  linarith

/-- **`tl.exp (−∞ − b) = +0`** for a finite `b`. -/
theorem exp_negInf_eq {P : Platform} (hP : Device P) {b d r : F32} (hb : b.isFinite = true)
    (hd : FloatV.NaNEq (FloatV.sub FloatV.negInf b) d) (hr : P.math .exp .binary32 d r) :
    r = FloatV.ofPat 0 := by
  have hs : Format.binary32.specials = .ieee := rfl
  rw [FloatV.sub_negInf_left hs hb] at hd
  have hn : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN hs
  have hdd : d = FloatV.negInf := hd.eq hn
  subst hdd
  exact hP.expF32NegInf r hr

/-- **`tl.exp (−∞ − b) = 0`** for a finite `b`. -/
theorem exp_negInf {P : Platform} (hP : Device P) {b d r : F32} (hb : b.isFinite = true)
    (hd : FloatV.NaNEq (FloatV.sub FloatV.negInf b) d) (hr : P.math .exp .binary32 d r) :
    r.isFinite = true ∧ r.toReal = 0 := by
  have := exp_negInf_eq hP hb hd hr
  subst this
  exact ⟨FloatV.isFinite_ofPat_zero Format.binary32_regular, by simp⟩

end

end Claims.ExtendAttn
