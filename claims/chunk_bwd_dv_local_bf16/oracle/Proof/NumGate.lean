import Claim
import TVA.Proof
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Proof.Util

/-!
# The gate factor `exp2(g[j] - g[p])`

The kernel computes `b_g[None, :] - b_g[:, None]` (an IEEE `float32` difference, which may
overflow to `-∞`) and `tl.exp2` of it. `gate_exp2` says the result is finite and within `wErrR` of
`2^{g[j] - g[p]}`, the bound `Claim.wErr` states; `wErrR_le` bounds `wErrR` by a relative
part and an absolute part.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The claim's gate error as a function of the exact difference `dg`. -/
noncomputable def wErrR (dg : ℝ) : ℝ :=
  (2 : ℝ) ^ dg * ((2 : ℝ) ^ (u * |dg| + 1 / 2 ^ 149) * (1 + 1 / 2 ^ 22) - 1) + 1 / 2 ^ 126

/-- The gate error is positive: at least its absolute part `2^-126`. -/
theorem wErrR_ge (dg : ℝ) : 1 / 2 ^ 126 ≤ wErrR dg := by
  unfold wErrR
  have h1 : (1 : ℝ) ≤ (2 : ℝ) ^ (u * |dg| + 1 / 2 ^ 149) := by
    apply Real.one_le_rpow (by norm_num)
    unfold u; positivity
  have h2 : (0 : ℝ) < (2 : ℝ) ^ dg := Real.rpow_pos_of_pos (by norm_num) dg
  have h3 : (1 : ℝ) ≤ (2 : ℝ) ^ (u * |dg| + 1 / 2 ^ 149) * (1 + 1 / 2 ^ 22) := by nlinarith
  nlinarith

/-- **An overflowing difference is `-∞`.** If the difference of two finite `float32` values is not
finite, it is `-∞` (when the exact difference is negative) and the exact difference is beyond
`maxVal`. -/
theorem sub_overflow {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hnf : (FloatV.sub x y).isFinite = false) (hq : x.toRat - y.toRat < 0) :
    FloatV.sub x y = FloatV.negInf ∧ Format.binary32.maxVal < |x.toRat - y.toRat| := by
  have hf := Format.binary32_regular
  have hxf : Format.binary32.isFinite x.pat = true := hx
  have hyf : Format.binary32.isFinite y.pat = true := hy
  have hsub : (FloatV.sub x y) = FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat)) := by
    unfold FloatV.sub
    rw [Format.sub_of_finite .rn hxf hyf]
    rfl
  set q := x.toRat - y.toRat with hqdef
  have hq0 : q ≠ 0 := ne_of_lt hq
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsubn⟩ := Format.roundSig_bounds (f := Format.binary32) .rn ha
  by_cases hno : (Format.roundSig .rn |q| (Format.binary32.qexp |q|) : ℚ) *
      (2 : ℚ) ^ (Format.binary32.qexp |q|) ≤ Format.binary32.maxVal
  · exfalso
    have hmag : Format.binary32.roundMag .rn false |q| =
        Format.binary32.pack (Format.roundSig .rn |q| (Format.binary32.qexp |q|))
          (Format.binary32.qexp |q|) := by
      unfold Format.roundMag Format.encode; rw [Format.pow2_eq, ite_eq_left hno]
    have hle := Format.pack_le_maxMag hf hN hQ hsubn hno
    have hlt := hle.trans_lt (Format.maxMag_lt hf)
    have : (FloatV.sub x y).isFinite = true := by
      rw [hsub]
      have hpat : (Format.binary32.roundRat .rn false
          (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) q) =
          Format.binary32.setSign true (Format.binary32.pack
            (Format.roundSig .rn |q| (Format.binary32.qexp |q|)) (Format.binary32.qexp |q|)) := by
        rw [Format.roundRat_of_ne hq0, hmag, Format.setSign_of_lt hlt]
        simp [hq]
      have hlt2 : Format.binary32.roundRat .rn false
          (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) q < 2 ^ Format.binary32.width := by
        rw [hpat]; exact Format.setSign_lt _ _
      rw [FloatV.isFinite_ofPat hlt2, hpat, Format.isFinite_setSign]
      exact Format.isFinite_of_mag_le hf (by rw [Format.mag_of_lt hlt]; exact hle)
    rw [this] at hnf
    exact Bool.noConfusion hnf
  · have hover := hno
    refine ⟨?_, ?_⟩
    · have hmag : Format.binary32.roundMag .rn false |q| = Format.binary32.infMag := by
        unfold Format.roundMag Format.encode
        rw [Format.pow2_eq, ite_eq_right hover]
        simp [Format.binary32]
      rw [hsub]
      unfold FloatV.negInf
      congr 1
      rw [Format.roundRat_of_ne hq0, hmag, Format.setSign_of_lt (FloatV.infMag_lt_signBit)]
      simp [hq]
    · by_contra hcon
      push Not at hcon
      exact hover (Format.roundSig_le_maxVal hf .rn ha hcon)

/-- **Two powers of two whose exponents are within `e` are within `2^a (2^e - 1)`** of each other. -/
theorem rpow_two_gap {a b e : ℝ} (h : |b - a| ≤ e) : |(2 : ℝ) ^ b - (2 : ℝ) ^ a| ≤ (2 : ℝ) ^ a * ((2 : ℝ) ^ e - 1) := by
  have h2a : (0 : ℝ) < (2 : ℝ) ^ a := Real.rpow_pos_of_pos (by norm_num) a
  have he : 0 ≤ e := le_trans (abs_nonneg _) h
  have hge1 : ∀ t : ℝ, 0 ≤ t → 1 ≤ (2 : ℝ) ^ t := fun t ht => Real.one_le_rpow (by norm_num) ht
  rcases le_total a b with hab | hab
  · have hb : (2 : ℝ) ^ b = (2 : ℝ) ^ a * (2 : ℝ) ^ (b - a) := by
      rw [← Real.rpow_add (by norm_num)]; congr 1; ring
    have hmono : (2 : ℝ) ^ (b - a) ≤ (2 : ℝ) ^ e :=
      Real.rpow_le_rpow_of_exponent_le (by norm_num) (by rw [abs_of_nonneg (by linarith)] at h; exact h)
    have h1 := hge1 (b - a) (by linarith)
    rw [abs_of_nonneg (by rw [hb]; nlinarith), hb]
    nlinarith
  · have hb : (2 : ℝ) ^ a = (2 : ℝ) ^ b * (2 : ℝ) ^ (a - b) := by
      rw [← Real.rpow_add (by norm_num)]; congr 1; ring
    have hmono : (2 : ℝ) ^ (a - b) ≤ (2 : ℝ) ^ e :=
      Real.rpow_le_rpow_of_exponent_le (by norm_num) (by rw [abs_of_nonpos (by linarith)] at h; linarith)
    have h1 := hge1 (a - b) (by linarith)
    have h2b : (0 : ℝ) < (2 : ℝ) ^ b := Real.rpow_pos_of_pos (by norm_num) b
    set t := (2 : ℝ) ^ (a - b) with ht
    have htpos : 0 < t := by linarith
    set v := 1 / t with hv
    have hvt : v * t = 1 := by rw [hv]; field_simp
    have hv0 : 0 < v := by positivity
    have hinv : (2 : ℝ) ^ b = (2 : ℝ) ^ a * v := by
      have : (2 : ℝ) ^ b * t = (2 : ℝ) ^ a := hb.symm
      nlinarith
    have hv1 : v ≤ 1 := by nlinarith
    have h3 : 1 - v ≤ t - 1 := by nlinarith [mul_nonneg hv0.le (sq_nonneg (t - 1))]
    rw [hinv, abs_of_nonpos (by nlinarith)]
    nlinarith

/-- A `float32` rounding to nearest is a pattern of the format's width. -/
theorem roundRat_lt_f32 (zs : Bool) (q : ℚ) :
    Format.binary32.roundRat .rn false zs q < 2 ^ Format.binary32.width := by
  have hf := Format.binary32_regular
  by_cases hq0 : q = 0
  · subst hq0
    exact (Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn) (sat := false)
      (zs := zs) hf).1
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsubn⟩ := Format.roundSig_bounds (f := Format.binary32) .rn ha
  have hlt : Format.binary32.roundMag .rn false |q| < Format.binary32.signBit := by
    by_cases hno : (Format.roundSig .rn |q| (Format.binary32.qexp |q|) : ℚ) *
        (2 : ℚ) ^ (Format.binary32.qexp |q|) ≤ Format.binary32.maxVal
    · have hmag : Format.binary32.roundMag .rn false |q| =
          Format.binary32.pack (Format.roundSig .rn |q| (Format.binary32.qexp |q|))
            (Format.binary32.qexp |q|) := by
        unfold Format.roundMag Format.encode; rw [Format.pow2_eq, ite_eq_left hno]
      rw [hmag]
      exact (Format.pack_le_maxMag hf hN hQ hsubn hno).trans_lt (Format.maxMag_lt hf)
    · exact (Format.roundMag_rn_of_over hf (by decide) hno).1
  rw [Format.roundRat_of_ne hq0]
  have : (if q < 0 then Format.binary32.signBit else 0) + Format.binary32.roundMag .rn false |q| =
      Format.binary32.setSign (decide (q < 0)) (Format.binary32.roundMag .rn false |q|) := by
    rw [Format.setSign_of_lt hlt]
    by_cases h : q < 0 <;> simp [h]
  rw [this]
  exact Format.setSign_lt _ _

/-- **The exact difference rounds with one rounding when the result is finite.** -/
theorem sub_finite_err {x y : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hfin : (FloatV.sub x y).isFinite = true) :
    |(FloatV.sub x y).toRat - (x.toRat - y.toRat)| ≤
      |x.toRat - y.toRat| * Format.binary32.u + Format.binary32.eta := by
  have hf := Format.binary32_regular
  by_cases hin : |x.toRat - y.toRat| ≤ Format.binary32.maxVal
  · exact (FloatV.sub_err hf hx hy hin).2
  · push Not at hin
    have hxf : Format.binary32.isFinite x.pat = true := hx
    have hyf : Format.binary32.isFinite y.pat = true := hy
    have hsub : (FloatV.sub x y) = FloatV.ofPat (Format.binary32.roundRat .rn false
        (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat)) := by
      unfold FloatV.sub
      rw [Format.sub_of_finite .rn hxf hyf]
      rfl
    have hfin' : Format.binary32.isFinite (Format.binary32.roundRat .rn false
        (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat)) = true := by
      by_contra hcon
      have hlt : Format.binary32.roundRat .rn false
          (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat)
          < 2 ^ Format.binary32.width := by
        exact roundRat_lt_f32 _ _
      rw [hsub, FloatV.isFinite_ofPat hlt] at hfin
      exact hcon hfin
    have herr := Format.roundRat_rn_err_of_finite hf (by decide) _ hfin'
    have hlt : Format.binary32.roundRat .rn false
        (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat)
        < 2 ^ Format.binary32.width := roundRat_lt_f32 _ _
    have hval : (FloatV.sub x y).toRat = Format.binary32.toRat (Format.binary32.roundRat .rn false
        (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat)) := by
      rw [hsub, FloatV.toRat_ofPat hlt]
    rw [hval]
    refine herr.trans ?_
    have hle : |Format.binary32.toRat (Format.binary32.roundRat .rn false
        (Format.binary32.sign x.pat && !Format.binary32.sign y.pat) (x.toRat - y.toRat))| ≤
        Format.binary32.maxVal := by
      have := FloatV.abs_toRat_le_maxVal hf (x := FloatV.sub x y) hfin
      rwa [hval] at this
    have hu0 := Format.u_nonneg (f := Format.binary32)
    nlinarith [mul_le_mul_of_nonneg_right hle hu0, mul_le_mul_of_nonneg_right hin.le hu0]

/-- **The gate factor**: for finite `float32` values `x`, `y` with `x - y ≤ 64`, `tl.exp2` of
the `float32` difference (every result the platform allows) is finite and within `wErrR` of
`2^(x - y)`. -/
theorem gate_exp2 {P : Platform} (hP : Device P) {x y : F32} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hup : (x.toRat : ℝ) - y.toRat ≤ 64) {s w : F32}
    (hs : FloatV.NaNEq (FloatV.sub x y) s) (hw : P.math .exp2 .binary32 s w) :
    w.isFinite = true ∧
      |w.toReal - (2 : ℝ) ^ ((x.toRat : ℝ) - y.toRat)| ≤ wErrR ((x.toRat : ℝ) - y.toRat) := by
  have hf := Format.binary32_regular
  have hmaxR : ((Format.binary32.maxVal : ℚ) : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 :=
    Format.binary32_maxVal_real
  have hbig : (126 : ℝ) < ((Format.binary32.maxVal : ℚ) : ℝ) := by rw [hmaxR]; norm_num
  have hnn : ∀ dg : ℝ, 0 ≤ wErrR dg - 1 / 2 ^ 126 := fun dg => sub_nonneg.mpr (wErrR_ge dg)
  by_cases hfin : (FloatV.sub x y).isFinite = true
  · -- the difference is finite
    have hnan : (FloatV.sub x y).isNaN = false := ((FloatV.isFinite_iff).mp hfin).1
    have hse : s = FloatV.sub x y := hs.eq hnan
    subst hse
    have hbd := sub_finite_err hx hy hfin
    set q : ℚ := x.toRat - y.toRat with hq
    have hdg : (q : ℝ) = (x.toRat : ℝ) - y.toRat := by rw [hq]; push_cast; ring
    rw [← hdg] at hup ⊢
    -- the bound in the reals
    have hbdR : |((FloatV.sub x y).toRat : ℝ) - (q : ℝ)| ≤ u * |(q : ℝ)| + 1 / 2 ^ 149 := by
      have h := (by exact_mod_cast hbd :
        |((FloatV.sub x y).toRat : ℝ) - (q : ℝ)| ≤ |(q : ℝ)| * ((Format.binary32.u : ℚ) : ℝ) +
          ((Format.binary32.eta : ℚ) : ℝ))
      rw [Format.binary32_u_real, Format.binary32_eta_real] at h
      unfold u
      have e1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
      have e2 : (2 : ℝ) ^ (-150 : ℤ) ≤ 1 / 2 ^ 149 := by norm_num
      rw [e1] at h
      nlinarith [abs_nonneg (q : ℝ)]
    set S : ℝ := ((FloatV.sub x y).toRat : ℝ) with hS
    have hSle : S ≤ 65 := by
      have h1 := (abs_le.mp hbdR).2
      have hu : u = 1 / 2 ^ 24 := rfl
      rcases le_total 0 (q : ℝ) with h0 | h0
      · rw [abs_of_nonneg h0] at h1
        have : u * (q : ℝ) ≤ 1 / 4 := by rw [hu]; nlinarith
        have : (1 : ℝ) / 2 ^ 149 ≤ 1 / 2 := by norm_num
        linarith
      · rw [abs_of_nonpos h0] at h1
        have : (0 : ℝ) ≤ u * (-(q : ℝ)) := by rw [hu]; nlinarith
        have : u * (-(q : ℝ)) ≤ -(q : ℝ) := by rw [hu]; nlinarith
        have : (1 : ℝ) / 2 ^ 149 ≤ 1 / 2 := by norm_num
        nlinarith
    obtain ⟨ref, href, hacc⟩ := hP.exp2F32
    have hrefeq : ref = fun t => (2 : ℝ) ^ t := by
      simp only [MathFn.ref] at href
      exact (Option.some.inj href).symm
    subst hrefeq
    have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |(2 : ℝ) ^ (FloatV.sub x y).toReal| + (2 : ℝ) ^ (-126 : ℤ) ≤
        ((Format.binary32.maxVal : ℚ) : ℝ) := by
      have h2S : (2 : ℝ) ^ S ≤ (2 : ℝ) ^ (65 : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) hSle
      have h265 : (2 : ℝ) ^ (65 : ℝ) = 2 ^ 65 := by norm_num
      rw [abs_of_pos (Real.rpow_pos_of_pos (by norm_num) _), hmaxR]
      have : (2 : ℝ) ^ (FloatV.sub x y).toReal = (2 : ℝ) ^ S := rfl
      rw [this]
      rw [h265] at h2S
      have e1 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
      have e2 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
      rw [e1, e2]
      nlinarith
    obtain ⟨hwfin, hwerr⟩ := hacc (FloatV.sub x y) w hw hfin (Set.mem_univ _) (by simp [MathFn.natDom])
      hroom
    refine ⟨hwfin, ?_⟩
    have hgap := rpow_two_gap (a := (q : ℝ)) (b := S) (e := u * |(q : ℝ)| + 1 / 2 ^ 149) hbdR
    have hSup : (2 : ℝ) ^ S ≤ (2 : ℝ) ^ (q : ℝ) * (2 : ℝ) ^ (u * |(q : ℝ)| + 1 / 2 ^ 149) := by
      rw [← Real.rpow_add (by norm_num)]
      apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
      have := (abs_le.mp hbdR).2
      linarith
    have e1 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
    have e2 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
    have hwerr2 : |w.toReal - (2 : ℝ) ^ S| ≤ 1 / 2 ^ 22 * (2 : ℝ) ^ S + 1 / 2 ^ 126 := by
      have h : |w.toReal - (2 : ℝ) ^ S| ≤ (2 : ℝ) ^ (-22 : ℤ) * |(2 : ℝ) ^ S| + (2 : ℝ) ^ (-126 : ℤ) :=
        hwerr
      rw [abs_of_pos (Real.rpow_pos_of_pos (by norm_num) _), e1, e2] at h
      exact h
    have hw' : |w.toReal - (2 : ℝ) ^ (q : ℝ)| ≤
        |w.toReal - (2 : ℝ) ^ S| + |(2 : ℝ) ^ S - (2 : ℝ) ^ (q : ℝ)| := abs_sub_le _ _ _
    unfold wErrR
    have hm := mul_le_mul_of_nonneg_left hSup (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22)
    nlinarith
  · -- the difference overflowed
    have hnf : (FloatV.sub x y).isFinite = false := by simpa using hfin
    have hqneg : x.toRat - y.toRat < 0 := by
      by_contra hcon
      push Not at hcon
      have hq64 : x.toRat - y.toRat ≤ 64 := by exact_mod_cast hup
      have hin : |x.toRat - y.toRat| ≤ Format.binary32.maxVal := by
        rw [abs_of_nonneg hcon, Format.binary32_maxVal]; linarith
      exact hfin (FloatV.sub_err hf hx hy hin).1
    obtain ⟨hneg, hover⟩ := sub_overflow hx hy hnf hqneg
    have hse : s = FloatV.negInf := by
      have hnan : (FloatV.sub x y).isNaN = false := by rw [hneg]; exact FloatV.negInf_isNaN (by decide)
      rw [← hneg]; exact hs.eq hnan
    subst hse
    have hw0 : w = FloatV.ofPat 0 := hP.exp2F32NegInf w hw
    subst hw0
    refine ⟨FloatV.isFinite_ofPat_zero hf, ?_⟩
    rw [FloatV.toReal_ofPat_zero, zero_sub, abs_neg]
    have hq : ((x.toRat - y.toRat : ℚ) : ℝ) < -((Format.binary32.maxVal : ℚ) : ℝ) := by
      have : x.toRat - y.toRat < -Format.binary32.maxVal := by
        rw [abs_of_neg hqneg] at hover; linarith
      exact_mod_cast this
    push_cast at hq
    have h1 : (2 : ℝ) ^ ((x.toRat : ℝ) - y.toRat) ≤ (2 : ℝ) ^ (-(126 : ℝ)) :=
      Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
    have h2 : (2 : ℝ) ^ (-(126 : ℝ)) = 1 / 2 ^ 126 := by
      rw [Real.rpow_neg (by norm_num)]; norm_num
    have h3 : (2 : ℝ) ^ ((x.toRat : ℝ) - y.toRat) ≤ 1 / 2 ^ 126 := h1.trans h2.le
    have := hnn ((x.toRat : ℝ) - y.toRat)
    rw [abs_of_pos (Real.rpow_pos_of_pos (by norm_num) _)]
    exact h3.trans (by linarith)

/-- **The gate error is small**: relative `2^-16` of the exact factor, plus an absolute
`(1 + 2^-16)·2^-126`, for every difference `dg ≤ 64`. -/
theorem wErrR_le {dg : ℝ} (h : dg ≤ 64) :
    wErrR dg ≤ (2 : ℝ) ^ dg * (1 / 2 ^ 16) + (1 + 1 / 2 ^ 16) / 2 ^ 126 := by
  unfold wErrR
  have h2 : (0 : ℝ) < (2 : ℝ) ^ dg := Real.rpow_pos_of_pos (by norm_num) dg
  have hu : u = 1 / 2 ^ 24 := rfl
  set e : ℝ := u * |dg| + 1 / 2 ^ 149 with he
  have he0 : 0 ≤ e := by rw [he, hu]; positivity
  have h126 : (1 : ℝ) / 2 ^ 126 ≤ (1 + 1 / 2 ^ 16) / 2 ^ 126 := by
    rw [div_le_div_iff_of_pos_right (by positivity)]; linarith
  rcases le_or_gt (-150) dg with hd | hd
  · -- moderate difference: the relative error is small
    have habs : |dg| ≤ 150 := by rw [abs_le]; constructor <;> linarith
    have he1 : e ≤ 3 / 2 ^ 18 := by
      rw [he, hu]
      have : (1 : ℝ) / 2 ^ 24 * |dg| ≤ 1 / 2 ^ 24 * 150 := by gcongr
      have : (1 : ℝ) / 2 ^ 149 ≤ 1 / 2 ^ 30 := by norm_num
      nlinarith
    have hp := rpow_one_add_le_one_add_mul_self (s := 1) (by norm_num) he0
      (by linarith [he1, (by norm_num : (3 : ℝ) / 2 ^ 18 ≤ 1)])
    have h2e : (2 : ℝ) ^ e ≤ 1 + e := by
      have : (1 + (1 : ℝ)) = 2 := by norm_num
      rw [this] at hp; linarith
    have hkey : (2 : ℝ) ^ e * (1 + 1 / 2 ^ 22) - 1 ≤ 1 / 2 ^ 16 := by
      have : (2 : ℝ) ^ e * (1 + 1 / 2 ^ 22) ≤ (1 + e) * (1 + 1 / 2 ^ 22) :=
        mul_le_mul_of_nonneg_right h2e (by norm_num)
      nlinarith
    have := mul_le_mul_of_nonneg_left hkey h2.le
    linarith
  · -- a tiny factor: the whole first term is far below `2^-142`
    have hneg : |dg| = -dg := abs_of_neg (by linarith)
    have hexp : dg + e ≤ -149 := by
      rw [he, hu, hneg]
      have : (1 : ℝ) / 2 ^ 24 * (-dg) = -dg / 2 ^ 24 := by ring
      nlinarith [(by norm_num : (1 : ℝ) / 2 ^ 149 ≤ 1 / 2 ^ 30)]
    have h1 : (2 : ℝ) ^ dg * (2 : ℝ) ^ e = (2 : ℝ) ^ (dg + e) :=
      (Real.rpow_add (by norm_num) dg e).symm
    have h3 : (2 : ℝ) ^ (dg + e) ≤ (2 : ℝ) ^ (-149 : ℝ) :=
      Real.rpow_le_rpow_of_exponent_le (by norm_num) hexp
    have h4 : (2 : ℝ) ^ (-149 : ℝ) = 1 / 2 ^ 149 := by
      rw [Real.rpow_neg (by norm_num)]; norm_num
    have h5 : (2 : ℝ) ^ dg * ((2 : ℝ) ^ e * (1 + 1 / 2 ^ 22) - 1) ≤ 2 * (1 / 2 ^ 149) := by
      have : (2 : ℝ) ^ dg * ((2 : ℝ) ^ e * (1 + 1 / 2 ^ 22) - 1) ≤
          (2 : ℝ) ^ dg * (2 : ℝ) ^ e * (1 + 1 / 2 ^ 22) := by nlinarith
      rw [h1] at this
      have h6 : (2 : ℝ) ^ (dg + e) * (1 + 1 / 2 ^ 22) ≤ 1 / 2 ^ 149 * (1 + 1 / 2 ^ 22) :=
        mul_le_mul_of_nonneg_right (h3.trans h4.le) (by norm_num)
      linarith [(by norm_num : (1 / 2 ^ 149 : ℝ) * (1 + 1 / 2 ^ 22) ≤ 2 * (1 / 2 ^ 149))]
    have h7 : (2 : ℝ) * (1 / 2 ^ 149) ≤ 1 / 2 ^ 16 * (1 / 2 ^ 126) := by norm_num
    have h8 : (0 : ℝ) ≤ (2 : ℝ) ^ dg * (1 / 2 ^ 16) := by positivity
    have h9 : (1 + 1 / 2 ^ 16 : ℝ) / 2 ^ 126 = 1 / 2 ^ 126 + 1 / 2 ^ 16 * (1 / 2 ^ 126) := by ring
    linarith

end Claims.ChunkBwdDvLocal
