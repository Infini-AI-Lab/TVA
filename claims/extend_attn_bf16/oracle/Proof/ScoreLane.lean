import Proof.ScoreReal
import Proof.Consts

/-!
# The score of a lane from the float facts

The host's scale (`hostF32 sm_scale`, `hostF32 1.0`), the scale product, and the lane's score
`fl (D · ss)`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- **The host's `float32` scale** `(float)c` of a finite double `c` in `[2^-64, 2^64]`: finite and
within `2^-24·c + 2^-150` of `c`. -/
theorem hostF32_err {c : F64} (hc : c.isFinite = true) (hlo : (1 : ℝ) / 2 ^ 64 ≤ c.toReal)
    (hhi : c.toReal ≤ 2 ^ 64) :
    (FloatV.hostF32 c).isFinite = true ∧
      |(FloatV.hostF32 c).toReal - c.toReal| ≤ 1 / 2 ^ 24 * c.toReal + 1 / 2 ^ 150 := by
  have hn : c.isNaN = false := (FloatV.isFinite_iff.mp hc).1
  have hpos : 0 < c.toReal := lt_of_lt_of_le (by positivity) hlo
  have hmax : |c.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_of_pos hpos, Format.binary32_maxVal_real]
    exact hhi.trans (by norm_num)
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.binary32) Format.binary32_regular hc hmax
  simp only [FloatV.hostF32, hn, Bool.false_eq_true, ite_false]
  rw [Format.binary32_u_real, Format.binary32_eta_real, abs_of_pos hpos] at h2
  refine ⟨h1, ?_⟩
  norm_num at h2 ⊢
  linarith

/-- The host's `k_scale = v_scale = 1.0` as a `float32`. -/
theorem one64_spec : (one64 : F64).isFinite = true ∧ (one64 : F64).toReal = 1 := by
  have hc : (one64 : F64).isFinite = true := by decide
  refine ⟨hc, ?_⟩
  have h : (one64 : F64).toRat = 1 := by
    unfold one64
    rw [FloatV.toRat_ofPat (by decide)]
    simp only [Format.toRat, Format.magVal, Format.mag, Format.pow2_eq, Format.signBit, Format.sign,
      Format.bias, Format.emin, Format.binary64]
    norm_num
  unfold FloatV.toReal
  rw [h]; norm_num

/-- The arithmetic of the scale, with the roundoffs as symbols: `u` for a `float32` rounding, `e`
for the tiny absolute part of `s32` and `ss` (at most `2^-86 · sig`), `ek` for the tiny absolute
part of `k` (at most `2^-86`). Crude: the error is below `16 u · sig`. -/
theorem scale_arith_sym {sig s32 k ss u e ek : ℝ} (hsig : 0 < sig) (hu0 : 0 ≤ u)
    (hu : u ≤ 1 / 2 ^ 24) (hulo : 1 / 2 ^ 24 ≤ u) (he0 : 0 ≤ e) (he : e ≤ sig / 2 ^ 86)
    (_hek0 : 0 ≤ ek) (hek : ek ≤ 1 / 2 ^ 86) (hs : |s32 - sig| ≤ u * sig + e)
    (hk : |k - 1| ≤ u + ek) (hss : |ss - s32 * k| ≤ u * |s32 * k| + e) :
    |ss - sig| ≤ 16 * u * sig := by
  have hu1 : u ≤ 1 := hu.trans (by norm_num)
  have hek1 : ek ≤ 1 := hek.trans (by norm_num)
  have hs1 : |s32| ≤ sig * 2 + e := by
    have := abs_sub_abs_le_abs_sub s32 sig
    rw [abs_of_pos hsig] at this
    nlinarith
  have hk1 : |k| ≤ 3 := by
    have := abs_sub_abs_le_abs_sub k 1
    simp only [abs_one] at this
    linarith
  have hsk : |s32 * k| ≤ (sig * 2 + e) * 3 := by
    rw [abs_mul]; exact mul_le_mul hs1 hk1 (abs_nonneg _) (by positivity)
  have hdiff : |s32 * k - sig| ≤ |s32 - sig| * |k| + sig * |k - 1| := by
    have e' : s32 * k - sig = (s32 - sig) * k + sig * (k - 1) := by ring
    rw [e']
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_pos hsig]
  have h1 : |s32 - sig| * |k| ≤ (u * sig + e) * 3 := mul_le_mul hs hk1 (abs_nonneg _) (by positivity)
  have h2 : sig * |k - 1| ≤ sig * (u + ek) := mul_le_mul_of_nonneg_left hk hsig.le
  have htri : |ss - sig| ≤ |ss - s32 * k| + |s32 * k - sig| := by
    have := abs_add_le (ss - s32 * k) (s32 * k - sig)
    rw [show ss - s32 * k + (s32 * k - sig) = ss - sig by ring] at this
    exact this
  have h3 : u * |s32 * k| ≤ u * ((sig * 2 + e) * 3) := mul_le_mul_of_nonneg_left hsk hu0
  have hesig : e ≤ sig * u / 2 ^ 60 := by
    have h5 : sig / 2 ^ 86 ≤ sig * u / 2 ^ 60 := by
      have : sig * (1 / 2 ^ 24) ≤ sig * u := mul_le_mul_of_nonneg_left hulo hsig.le
      have h6 : sig / 2 ^ 86 = sig * (1 / 2 ^ 24) / 2 ^ 62 := by ring
      have h7 : sig * (1 / 2 ^ 24) / 2 ^ 62 ≤ sig * u / 2 ^ 60 := by
        have : sig * (1 / 2 ^ 24) / 2 ^ 62 ≤ sig * (1 / 2 ^ 24) / 2 ^ 60 :=
          div_le_div_of_nonneg_left (by positivity) (by positivity) (by norm_num)
        have : sig * (1 / 2 ^ 24) / 2 ^ 60 ≤ sig * u / 2 ^ 60 :=
          div_le_div_of_nonneg_right ‹sig * (1 / 2 ^ 24) ≤ sig * u› (by positivity)
        linarith
      linarith
    linarith
  have hekS : sig * ek ≤ sig * u / 2 ^ 60 := by
    have h8 : ek ≤ u / 2 ^ 60 := by
      have : (1 : ℝ) / 2 ^ 86 ≤ 1 / 2 ^ 24 / 2 ^ 60 := by norm_num
      have : (1 : ℝ) / 2 ^ 24 / 2 ^ 60 ≤ u / 2 ^ 60 := div_le_div_of_nonneg_right hulo (by positivity)
      linarith
    calc sig * ek ≤ sig * (u / 2 ^ 60) := mul_le_mul_of_nonneg_left h8 hsig.le
      _ = sig * u / 2 ^ 60 := by ring
  have hsu : 0 ≤ sig * u := mul_nonneg hsig.le hu0
  have hue : u * e ≤ e := by nlinarith
  have hsum : |ss - sig| ≤ u * ((sig * 2 + e) * 3) + e + ((u * sig + e) * 3 + sig * (u + ek)) := by
    linarith [htri, hss, h3, hdiff, h1, h2]
  have hdiv : sig * u / 2 ^ 60 * 8 ≤ sig * u * 6 := by
    have : sig * u / 2 ^ 60 ≤ sig * u / 2 := div_le_div_of_nonneg_left hsu (by norm_num) (by norm_num)
    linarith
  nlinarith [hsum, hesig, hekS, hue, hdiv]

/-- The arithmetic of the scale: `ss = fl (sm32 · ks)`. -/
theorem scale_arith {sig s32 k ss : ℝ} (hsig1 : 1 / 2 ^ 64 ≤ sig)
    (hs : |s32 - sig| ≤ 1 / 2 ^ 24 * sig + 1 / 2 ^ 150) (hk : |k - 1| ≤ 1 / 2 ^ 24 + 1 / 2 ^ 150)
    (hss : |ss - s32 * k| ≤ 1 / 2 ^ 24 * |s32 * k| + 1 / 2 ^ 150) : |ss - sig| ≤ sig / 2 ^ 20 := by
  have hsig0 : 0 < sig := lt_of_lt_of_le (by positivity) hsig1
  have he : (1 : ℝ) / 2 ^ 150 ≤ sig / 2 ^ 86 := by
    have : (1 : ℝ) / 2 ^ 86 * (1 / 2 ^ 64) ≤ 1 / 2 ^ 86 * sig :=
      mul_le_mul_of_nonneg_left hsig1 (by positivity)
    have h4 : (1 : ℝ) / 2 ^ 150 ≤ 1 / 2 ^ 86 * (1 / 2 ^ 64) := by norm_num
    have : 1 / 2 ^ 86 * sig = sig / 2 ^ 86 := by ring
    linarith
  have := scale_arith_sym (u := 1 / 2 ^ 24) (e := 1 / 2 ^ 150) (ek := 1 / 2 ^ 150) hsig0 (by norm_num)
    le_rfl le_rfl (by positivity) he (by positivity) (by norm_num) hs hk hss
  have e : (16 : ℝ) * (1 / 2 ^ 24) * sig = sig / 2 ^ 20 := by ring
  linarith

/-- **The scale of a cached-part trip**, `sm_scale * k_scale`. -/
theorem scale_pref {sm : F64} {ss : F32} (hsm : sm.isFinite = true)
    (hlo : (1 : ℝ) / 2 ^ 64 ≤ sm.toReal) (hhi : sm.toReal ≤ 2 ^ 64)
    (h : FloatV.NaNEq (FloatV.mul (FloatV.hostF32 sm) (FloatV.hostF32 one64)) ss) :
    ss.isFinite = true ∧ |ss.toReal - sm.toReal| ≤ sm.toReal / 2 ^ 20 := by
  obtain ⟨hf1, he1⟩ := hostF32_err hsm hlo hhi
  obtain ⟨h64, h64v⟩ := one64_spec
  obtain ⟨hf2, he2⟩ := hostF32_err h64 (by rw [h64v]; norm_num) (by rw [h64v]; norm_num)
  rw [h64v] at he2
  have hsig0 : 0 < sm.toReal := lt_of_lt_of_le (by positivity) hlo
  have hmax : |(FloatV.hostF32 sm).toReal * (FloatV.hostF32 one64).toReal| ≤ 2 ^ 100 := by
    rw [abs_mul]
    have h1 : |(FloatV.hostF32 sm).toReal| ≤ 2 ^ 65 := by
      have h0 : 1 / 2 ^ 24 * sm.toReal ≤ 1 / 2 ^ 24 * 2 ^ 64 :=
        mul_le_mul_of_nonneg_left hhi (by positivity)
      have h00 : (1 / 2 ^ 24 * 2 ^ 64 : ℝ) + 1 / 2 ^ 150 + 2 ^ 64 ≤ 2 ^ 65 := by norm_num
      have := abs_sub_abs_le_abs_sub (FloatV.hostF32 sm).toReal sm.toReal
      rw [abs_of_pos hsig0] at this
      linarith
    have h2 : |(FloatV.hostF32 one64).toReal| ≤ 2 := by
      have h00 : (1 : ℝ) / 2 ^ 24 + 1 / 2 ^ 150 + 1 ≤ 2 := by norm_num
      have := abs_sub_abs_le_abs_sub (FloatV.hostF32 one64).toReal 1
      simp only [abs_one] at this
      linarith [he2]
    calc _ ≤ 2 ^ 65 * 2 := mul_le_mul h1 h2 (abs_nonneg _) (by positivity)
      _ ≤ 2 ^ 100 := by norm_num
  have hm : (2 : ℝ) ^ 100 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [Format.binary32_maxVal]; norm_num
  obtain ⟨hmf, hme⟩ := FloatV.mul_err_real Format.binary32_regular hf1 hf2 (hmax.trans hm)
  obtain ⟨hff, -, hr⟩ := h.exact_of_isFinite hmf
  refine ⟨hff, ?_⟩
  rw [hr]
  rw [Format.binary32_u_real, Format.binary32_eta_real] at hme
  refine scale_arith (k := (FloatV.hostF32 one64).toReal) (s32 := (FloatV.hostF32 sm).toReal) hlo
    he1 ?_ ?_
  · norm_num at he2 ⊢; linarith [he2]
  · norm_num at hme ⊢; linarith [hme]

/-- **The scale of an extend trip**, `sm_scale`. -/
theorem scale_ext {sm : F64} (hsm : sm.isFinite = true)
    (hlo : (1 : ℝ) / 2 ^ 64 ≤ sm.toReal) (hhi : sm.toReal ≤ 2 ^ 64) :
    (FloatV.hostF32 sm).isFinite = true ∧
      |(FloatV.hostF32 sm).toReal - sm.toReal| ≤ sm.toReal / 2 ^ 20 := by
  obtain ⟨hf1, he1⟩ := hostF32_err hsm hlo hhi
  refine ⟨hf1, ?_⟩
  have hsig0 : 0 < sm.toReal := lt_of_lt_of_le (by positivity) hlo
  have h3 : (1 : ℝ) / 2 ^ 150 ≤ 1 / 2 ^ 86 * sm.toReal := by
    have : (1 : ℝ) / 2 ^ 86 * (1 / 2 ^ 64) ≤ 1 / 2 ^ 86 * sm.toReal :=
      mul_le_mul_of_nonneg_left hlo (by positivity)
    have h4 : (1 : ℝ) / 2 ^ 150 ≤ 1 / 2 ^ 86 * (1 / 2 ^ 64) := by norm_num
    linarith
  have e1 : (1 : ℝ) / 2 ^ 24 * sm.toReal + 1 / 2 ^ 86 * sm.toReal ≤ sm.toReal / 2 ^ 20 := by
    have e : sm.toReal / 2 ^ 20 = sm.toReal * (1 / 2 ^ 20) := by ring
    have : (1 : ℝ) / 2 ^ 24 + 1 / 2 ^ 86 ≤ 1 / 2 ^ 20 := by norm_num
    have := mul_le_mul_of_nonneg_left this hsig0.le
    nlinarith
  linarith

/-- **The score of a lane**: `qk1 = fl (D · ss)`, `D` the dot and `ss` the scale. -/
theorem score_lane {BD : ℕ} {sig L S X : ℝ} {D ss qk1 : F32} (hBD : BD ≤ 256)
    (hsig1 : 1 / 2 ^ 64 ≤ sig) (hsig2 : sig ≤ 2 ^ 64) (hL0 : 0 ≤ L) (hLX : sig * L ≤ X)
    (hX : X ≤ 2 ^ 12) (hS : |S| ≤ L) (hDf : D.isFinite = true)
    (hD : |D.toReal - S| ≤ ((BD : ℝ) + 2) / 2 ^ 22 * L + ((BD : ℝ) + 2) / 2 ^ 126)
    (hssf : ss.isFinite = true) (hss : |ss.toReal - sig| ≤ sig / 2 ^ 20)
    (hmul : FloatV.NaNEq (FloatV.mul D ss) qk1) :
    qk1.isFinite = true ∧ |qk1.toReal - sig * S| ≤ dlt BD X ∧ |qk1.toReal| ≤ X + dlt BD X := by
  have hsig0 : 0 < sig := lt_of_lt_of_le (by positivity) hsig1
  have hBD' : (BD : ℝ) ≤ 256 := by exact_mod_cast hBD
  have hrD1 : ((BD : ℝ) + 2) / 2 ^ 22 ≤ 1 / 2 ^ 13 := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith
  have haD : ((BD : ℝ) + 2) / 2 ^ 126 ≤ 1 := by
    rw [div_le_one (by positivity)]; nlinarith
  have hDb : |D.toReal| ≤ L * (1 + 1 / 2 ^ 13) + 1 := by
    have h0 := abs_sub_abs_le_abs_sub D.toReal S
    have h1 := mul_le_mul_of_nonneg_right hrD1 hL0
    linarith
  have hss1 : |ss.toReal| ≤ sig * (1 + 1 / 2 ^ 20) := by
    have h0 := abs_sub_abs_le_abs_sub ss.toReal sig
    rw [abs_of_pos hsig0] at h0
    have e : sig / 2 ^ 20 = sig * (1 / 2 ^ 20) := by ring
    linarith
  have hprodq : |D.toReal * ss.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [abs_mul]
    have h1 := mul_le_mul hDb hss1 (abs_nonneg _) (by positivity)
    have h2 : (L * (1 + 1 / 2 ^ 13) + 1) * (sig * (1 + 1 / 2 ^ 20)) =
        (sig * L) * ((1 + 1 / 2 ^ 13) * (1 + 1 / 2 ^ 20)) + sig * (1 + 1 / 2 ^ 20) := by ring
    have h3 : (sig * L) * ((1 + 1 / 2 ^ 13) * (1 + 1 / 2 ^ 20)) ≤
        2 ^ 12 * ((1 + 1 / 2 ^ 13) * (1 + 1 / 2 ^ 20)) :=
      mul_le_mul_of_nonneg_right (hLX.trans hX) (by positivity)
    have h4 : sig * (1 + 1 / 2 ^ 20) ≤ 2 ^ 64 * (1 + 1 / 2 ^ 20) :=
      mul_le_mul_of_nonneg_right hsig2 (by positivity)
    rw [Format.binary32_maxVal_real]
    have h5 : (2 : ℝ) ^ 12 * ((1 + 1 / 2 ^ 13) * (1 + 1 / 2 ^ 20)) + 2 ^ 64 * (1 + 1 / 2 ^ 20) ≤
        (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hmf, hme⟩ := FloatV.mul_err_real Format.binary32_regular hDf hssf hprodq
  obtain ⟨hqf, -, hqr⟩ := hmul.exact_of_isFinite hmf
  refine ⟨hqf, ?_⟩
  rw [hqr]
  rw [Format.binary32_u_real, Format.binary32_eta_real] at hme
  have e24 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have e150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  rw [e24, e150] at hme
  have hq' : |(FloatV.mul D ss).toReal - D.toReal * ss.toReal| ≤
      1 / 2 ^ 24 * |D.toReal * ss.toReal| + 1 / 2 ^ 150 := by
    have := mul_comm (|D.toReal * ss.toReal|) (1 / 2 ^ 24 : ℝ)
    linarith
  have hmain := score_real hBD hsig1 hsig2 hL0 hLX hX hS hD hss hq'
  refine ⟨by unfold dlt; exact hmain, ?_⟩
  have h0 := abs_sub_abs_le_abs_sub (FloatV.mul D ss).toReal (sig * S)
  have h1 : |sig * S| ≤ sig * L := by
    rw [abs_mul, abs_of_pos hsig0]; exact mul_le_mul_of_nonneg_left hS hsig0.le
  unfold dlt
  linarith [abs_nonneg (sig * S)]

end

end Claims.ExtendAttn
