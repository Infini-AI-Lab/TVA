import TVA.Lang.Ops

/-!
# Rounding errors of the float operations

The `FloatV` forms of the bounds of `TVA.Math.Rounding`. With `u = 2^-(m+1)` and
`η = 2^(emin−m−1)` of the format (`Format.u`, `Format.eta`), every IEEE operation on finite
operands whose exact result `q` is in range (`|q| ≤ maxVal`) gives a finite result within
`|q|·u + η` of `q`:

* `FloatV.add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err` (`tl.add`, `tl.sub`, `tl.mul`,
  `tl.fma`, `tl.div_rn`), `cvt_err` (`tl.cvt`, to nearest), `ofRat_err`;
* `FloatV.sqrtRn_err` (`tl.sqrt_rn`): of a finite `x ≥ 0`, finite, nonnegative, within
  `√x·u + η` of the real `√x`;
* `FloatV.cvtWith_rz_err`: toward zero, within `2·(|q|·u + η)`;
* `FloatV.cvtSat_of_le`: the saturating conversion `cvtSat` is `cvtWith` within range;
  `cvtSat_of_gt`, `cvtSat_of_isInf`: beyond the range and for `±∞` it is `±maxVal`;
  `cvtSat_rn_err`: to nearest, it is one rounding of the clamped value
  `max (−maxVal) (min x maxVal)` (`isNaN_of_isInf`: an infinity is not a NaN);
* `FloatV.ofPat_roundRat_err`, `ofPat_roundRat_err_normal` (`|q|·u` for `|q| ≥ 2^emin`);
* `FloatV.toRat_cvt_of_wider`: a conversion into a wider format (`Format.Wider`: `float16` or
  `bfloat16` into `float32`, `float32` into `float64`, fp8 into `float16`/`bfloat16`/`float32`, …)
  keeps the value.

And the encoding: `FloatV.toRat_ofPat`, `isFinite_ofPat`, `isFinite_iff`, `toRat_of_isZero`,
`abs_toRat_le_maxVal` (a finite value is at most `maxVal`), `toRat_neg`.
-/
namespace TVA

open TVA.Math.Float

namespace FloatV

variable {f : Format}

/-! ## The encoding -/

theorem pat_lt (x : FloatV f) : x.pat < 2 ^ f.width := x.bits.isLt

theorem pat_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).pat = n := by
  simp [ofPat, pat, Nat.mod_eq_of_lt h]

theorem toRat_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).toRat = f.toRat n := by
  unfold toRat; rw [pat_ofPat h]

theorem isFinite_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isFinite = f.isFinite n := by
  unfold isFinite; rw [pat_ofPat h]

theorem isNaN_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isNaN = f.isNaN n := by
  unfold isNaN; rw [pat_ofPat h]

theorem isInf_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isInf = f.isInf n := by
  unfold isInf; rw [pat_ofPat h]

theorem isZero_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).isZero = f.isZero n := by
  unfold isZero; rw [pat_ofPat h]

theorem sign_ofPat {n : ℕ} (h : n < 2 ^ f.width) : (ofPat n : FloatV f).sign = f.sign n := by
  unfold sign; rw [pat_ofPat h]

theorem isFinite_iff {x : FloatV f} : x.isFinite = true ↔ x.isNaN = false ∧ x.isInf = false :=
  Format.isFinite_iff _

theorem toRat_of_isZero {x : FloatV f} (h : f.isZero x.pat = true) : x.toRat = 0 :=
  Format.toRat_of_isZero h

/-- A finite value is at most `maxVal` in magnitude. -/
theorem abs_toRat_le_maxVal (hf : f.Regular) {x : FloatV f} (hx : x.isFinite = true) :
    |x.toRat| ≤ f.maxVal :=
  Format.abs_toRat_le_maxVal hf hx

theorem toRat_neg (x : FloatV f) : (neg x).toRat = -x.toRat := by
  unfold neg; rw [toRat_ofPat (Format.neg_lt _), Format.toRat_neg]; rfl

theorem isFinite_neg (x : FloatV f) : (neg x).isFinite = x.isFinite := by
  unfold neg; rw [isFinite_ofPat (Format.neg_lt _), Format.isFinite_neg]; rfl

theorem isNaN_neg (x : FloatV f) : (neg x).isNaN = x.isNaN := by
  unfold neg; rw [isNaN_ofPat (Format.neg_lt _), Format.isNaN_neg]; rfl

/-! ## One rounding -/

/-- **One rounding to nearest** of a value in range: finite, within `|q|·u + η`. -/
theorem ofPat_roundRat_err (hf : f.Regular) (sat zs : Bool) {q : ℚ} (hq : |q| ≤ f.maxVal) :
    (ofPat (f.roundRat .rn sat zs q) : FloatV f).isFinite = true ∧
      |(ofPat (f.roundRat .rn sat zs q) : FloatV f).toRat - q| ≤ |q| * f.u + f.eta := by
  obtain ⟨hlt, hfin, herr⟩ := Format.roundRat_rn_err hf sat zs hq
  rw [isFinite_ofPat hlt, toRat_ofPat hlt]
  exact ⟨hfin, herr⟩

/-- **One rounding to nearest in the normal range**: within `|q|·u`. -/
theorem ofPat_roundRat_err_normal (hf : f.Regular) (sat zs : Bool) {q : ℚ}
    (hlo : (2 : ℚ) ^ f.emin ≤ |q|) (hq : |q| ≤ f.maxVal) :
    (ofPat (f.roundRat .rn sat zs q) : FloatV f).isFinite = true ∧
      |(ofPat (f.roundRat .rn sat zs q) : FloatV f).toRat - q| ≤ |q| * f.u := by
  obtain ⟨hlt, hfin, -⟩ := Format.roundRat_rn_err hf sat zs hq
  rw [isFinite_ofPat hlt, toRat_ofPat hlt]
  exact ⟨hfin, Format.roundRat_rn_err_normal hf sat zs hlo hq⟩

/-- The nearest float to an exact rational (`FloatV.ofRat`): finite, within `|q|·u + η`. (The
constant Triton builds for a Python number is `FloatV.litF32`, `litF16`, `litBF16` or `litF8`.) -/
theorem ofRat_err (hf : f.Regular) {q : ℚ} (hq : |q| ≤ f.maxVal) :
    (ofRat q : FloatV f).isFinite = true ∧ |(ofRat q : FloatV f).toRat - q| ≤ |q| * f.u + f.eta :=
  ofPat_roundRat_err hf false false hq

/-! ## The IEEE operations -/

/-- **`x + y`**, finite operands, exact sum in range: finite, within `|x + y|·u + η`. -/
theorem add_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toRat + y.toRat| ≤ f.maxVal) :
    (add x y).isFinite = true ∧
      |(add x y).toRat - (x.toRat + y.toRat)| ≤ |x.toRat + y.toRat| * f.u + f.eta := by
  unfold add; rw [Format.add_of_finite .rn hx hy]
  exact ofPat_roundRat_err hf _ _ hmax

/-- **`x - y`**: finite, within `|x − y|·u + η`. -/
theorem sub_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toRat - y.toRat| ≤ f.maxVal) :
    (sub x y).isFinite = true ∧
      |(sub x y).toRat - (x.toRat - y.toRat)| ≤ |x.toRat - y.toRat| * f.u + f.eta := by
  unfold sub; rw [Format.sub_of_finite .rn hx hy]
  exact ofPat_roundRat_err hf _ _ hmax

/-- **`x * y`**: finite, within `|x·y|·u + η`. -/
theorem mul_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toRat * y.toRat| ≤ f.maxVal) :
    (mul x y).isFinite = true ∧
      |(mul x y).toRat - x.toRat * y.toRat| ≤ |x.toRat * y.toRat| * f.u + f.eta := by
  unfold mul; rw [Format.mul_of_finite .rn hx hy]
  exact ofPat_roundRat_err hf _ _ hmax

/-- **`tl.fma(x, y, z)`**: finite, within `|x·y + z|·u + η`. -/
theorem fma_err (hf : f.Regular) {x y z : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hz : z.isFinite = true) (hmax : |x.toRat * y.toRat + z.toRat| ≤ f.maxVal) :
    (fma x y z).isFinite = true ∧
      |(fma x y z).toRat - (x.toRat * y.toRat + z.toRat)| ≤ |x.toRat * y.toRat + z.toRat| * f.u + f.eta := by
  unfold fma; rw [Format.fma_of_finite .rn hx hy hz]
  exact ofPat_roundRat_err hf _ _ hmax

/-- **`tl.div_rn(x, y)`**, `y ≠ 0`: finite, within `|x/y|·u + η`. -/
theorem divRn_err (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hz : f.isZero y.pat = false) (hmax : |x.toRat / y.toRat| ≤ f.maxVal) :
    (divRn x y).isFinite = true ∧
      |(divRn x y).toRat - x.toRat / y.toRat| ≤ |x.toRat / y.toRat| * f.u + f.eta := by
  unfold divRn; rw [Format.div_of_finite .rn hx hy hz]
  exact ofPat_roundRat_err hf _ _ hmax

/-- **A conversion to nearest** of a finite value in `g`'s range: finite, within
`|x|·u + η` of `g`. -/
theorem cvt_err {g : Format} (hg : g.Regular) {x : FloatV f} (hx : x.isFinite = true)
    (hmax : |x.toRat| ≤ g.maxVal) :
    (cvt x : FloatV g).isFinite = true ∧ |(cvt x : FloatV g).toRat - x.toRat| ≤ |x.toRat| * g.u + g.eta := by
  unfold cvt; rw [Format.cvt_of_finite g .rn false hx]
  exact ofPat_roundRat_err hg _ _ hmax

/-- **`tl.sqrt_rn`** (`FloatV.sqrtRn`) of a finite `x ≥ 0` (or `−0`): finite, nonnegative, within
`√x·u + η` of `√x`. -/
theorem sqrtRn_err (hf : f.Regular) {x : FloatV f} (hx : x.isFinite = true) (h0 : 0 ≤ x.toRat) :
    (sqrtRn x).isFinite = true ∧ 0 ≤ (sqrtRn x).toRat ∧
      |(sqrtRn x).toReal - Real.sqrt x.toReal| ≤ Real.sqrt x.toReal * f.u + f.eta := by
  obtain ⟨hlt, hfin, hnn, herr⟩ := Format.sqrt_rn_err hf hx h0
  unfold sqrtRn toReal
  rw [isFinite_ofPat hlt, toRat_ofPat hlt]
  exact ⟨hfin, hnn, herr⟩

/-- **A conversion toward zero** (`fp_downcast_rounding="rtz"`) of a finite value in `g`'s range:
finite, within `2·(|x|·u + η)` of `g`, and not larger in magnitude. -/
theorem cvtWith_rz_err {g : Format} (hg : g.Regular) {x : FloatV f} (hx : x.isFinite = true)
    (hmax : |x.toRat| ≤ g.maxVal) :
    (cvtWith .rz x : FloatV g).isFinite = true ∧
      |(cvtWith .rz x : FloatV g).toRat - x.toRat| ≤ 2 * (|x.toRat| * g.u + g.eta) ∧
      |(cvtWith .rz x : FloatV g).toRat| ≤ |x.toRat| := by
  unfold cvtWith; rw [Format.cvt_of_finite g .rz false hx]
  obtain ⟨hlt, hfin, herr, habs⟩ := Format.roundRat_rz_err hg false (f.sign x.pat) (q := f.toRat x.pat) hmax
  rw [isFinite_ofPat hlt, toRat_ofPat hlt]
  exact ⟨hfin, herr, habs⟩

/-- **Saturation changes nothing within range**: for a finite `x` with `|x| ≤ maxVal` of `g`,
`cvtSat rnd x = cvtWith rnd x`. -/
theorem cvtSat_of_le {g : Format} (hg : g.Regular) (rnd : RoundMode) {x : FloatV f}
    (hx : x.isFinite = true) (hmax : |x.toRat| ≤ g.maxVal) :
    (cvtSat rnd x : FloatV g) = cvtWith rnd x := by
  unfold cvtSat cvtWith
  rw [Format.cvt_of_finite g rnd true hx, Format.cvt_of_finite g rnd false hx]
  congr 1
  by_cases h0 : f.toRat x.pat = 0
  · rw [h0, Format.roundRat_zero, Format.roundRat_zero]
  · have ha : 0 < |f.toRat x.pat| := abs_pos.mpr h0
    rw [Format.roundRat_of_ne h0, Format.roundRat_of_ne h0, Format.roundMag_of_le hg rnd true ha hmax,
      Format.roundMag_of_le hg rnd false ha hmax]

/-- An infinite value is not a NaN. -/
theorem isNaN_of_isInf {x : FloatV f} (h : x.isInf = true) : x.isNaN = false := by
  unfold isInf Format.isInf at h
  unfold isNaN Format.isNaN
  cases hs : f.specials
  · simp only [hs, Bool.and_eq_true, beq_iff_eq] at h ⊢
    simp [h.1, h.2]
  · simp [hs] at h
  · simp

/-- **Saturation beyond the range**: for a finite `x` with `|x| > maxVal` of `g`, `cvtSat rnd x`
is finite with value `maxVal` of `x`'s sign (either mode). -/
theorem cvtSat_of_gt {g : Format} (hg : g.Regular) (rnd : RoundMode) {x : FloatV f}
    (hx : x.isFinite = true) (hmax : g.maxVal < |x.toRat|) :
    (cvtSat rnd x : FloatV g).isFinite = true ∧
      (cvtSat rnd x : FloatV g).toRat = if x.toRat < 0 then -g.maxVal else g.maxVal := by
  unfold cvtSat
  rw [Format.cvt_of_finite g rnd true hx]
  have h0 : f.toRat x.pat ≠ 0 := by
    intro h
    have : |x.toRat| = 0 := by unfold toRat; rw [h, abs_zero]
    linarith [Format.maxVal_pos hg]
  have hlt := Format.maxMag_lt hg
  have hpat : g.roundRat rnd true (f.sign x.pat) (f.toRat x.pat) =
      g.setSign (decide (f.toRat x.pat < 0)) g.maxMag := by
    have hmax' : g.maxVal < |f.toRat x.pat| := hmax
    rw [Format.roundRat_of_ne h0, Format.roundMag_sat_of_gt hg rnd hmax', Format.setSign_of_lt hlt]
    by_cases h : f.toRat x.pat < 0 <;> simp [h]
  have hw := Format.setSign_lt (f := g) (decide (f.toRat x.pat < 0)) g.maxMag
  rw [hpat, isFinite_ofPat hw, toRat_ofPat hw, Format.isFinite_setSign, Format.toRat_setSign,
    Format.mag_of_lt hlt, Format.magVal_maxMag hg]
  refine ⟨Format.isFinite_of_mag_le hg (by rw [Format.mag_of_lt hlt]), ?_⟩
  by_cases h : f.toRat x.pat < 0 <;> simp [h, toRat]

/-- **Saturation of an infinity**: `cvtSat rnd (±∞)` is `±maxVal` of `g`. -/
theorem cvtSat_of_isInf {g : Format} (hg : g.Regular) (rnd : RoundMode) {x : FloatV f}
    (hx : x.isInf = true) :
    (cvtSat rnd x : FloatV g).isFinite = true ∧
      (cvtSat rnd x : FloatV g).toRat = if x.sign then -g.maxVal else g.maxVal := by
  have hn := isNaN_of_isInf hx
  unfold isNaN at hn
  unfold isInf at hx
  unfold cvtSat Format.cvt
  rw [ite_eq_right (by rw [hn]; decide), ite_eq_left hx, ite_eq_left rfl]
  have hlt := Format.maxMag_lt hg
  have hw := Format.setSign_lt (f := g) (f.sign x.pat) g.maxMag
  rw [isFinite_ofPat hw, toRat_ofPat hw, Format.isFinite_setSign, Format.toRat_setSign,
    Format.mag_of_lt hlt, Format.magVal_maxMag hg]
  exact ⟨Format.isFinite_of_mag_le hg (by rw [Format.mag_of_lt hlt]), rfl⟩

/-- **The saturating conversion to nearest is one rounding of the clamped value.** For a finite
`x`, `cvtSat .rn x` is finite and within `|c|·u + η` of `c = max (−maxVal) (min x maxVal)`:
`x` rounded within range (`cvt_err`), and exactly `±maxVal` beyond it (`cvtSat_of_gt`). -/
theorem cvtSat_rn_err {g : Format} (hg : g.Regular) {x : FloatV f} (hx : x.isFinite = true) :
    (cvtSat .rn x : FloatV g).isFinite = true ∧
      |(cvtSat .rn x : FloatV g).toRat - Max.max (-g.maxVal) (Min.min x.toRat g.maxVal)| ≤
        |Max.max (-g.maxVal) (Min.min x.toRat g.maxVal)| * g.u + g.eta := by
  have hpos := Format.maxVal_pos hg
  by_cases hm : |x.toRat| ≤ g.maxVal
  · have hc : Max.max (-g.maxVal) (Min.min x.toRat g.maxVal) = x.toRat := by
      rw [abs_le] at hm
      rw [min_eq_left hm.2, max_eq_right hm.1]
    rw [hc, cvtSat_of_le hg .rn hx hm, cvtWith_rn]
    exact cvt_err hg hx hm
  · obtain ⟨hfin, hval⟩ := cvtSat_of_gt hg .rn hx (not_le.mp hm)
    refine ⟨hfin, ?_⟩
    have hc : Max.max (-g.maxVal) (Min.min x.toRat g.maxVal) = if x.toRat < 0 then -g.maxVal else g.maxVal := by
      rcases lt_or_ge x.toRat 0 with h | h
      · rw [abs_of_neg h] at hm
        rw [ite_eq_left h, min_eq_left (by linarith), max_eq_left (by linarith)]
      · rw [abs_of_nonneg h] at hm
        rw [ite_eq_right (not_lt.mpr h), min_eq_right (by linarith), max_eq_right (by linarith)]
    rw [hc, hval, sub_self, abs_zero]
    exact add_nonneg (mul_nonneg (abs_nonneg _) Format.u_nonneg) Format.eta_nonneg

/-- **A conversion into a wider format is exact** (both rounding modes). -/
theorem toRat_cvt_of_wider {g : Format} (hf : f.Regular) (hg : g.Regular) (hw : Format.Wider f g)
    (rnd : RoundMode) {x : FloatV f} (hx : x.isFinite = true) :
    (cvtWith rnd x : FloatV g).isFinite = true ∧ (cvtWith rnd x : FloatV g).toRat = x.toRat := by
  unfold cvtWith
  have hlt : Format.cvt f g rnd false x.pat < 2 ^ g.width := by
    rw [Format.cvt_of_finite g rnd false hx]
    by_cases h0 : f.toRat x.pat = 0
    · rw [h0, Format.roundRat_zero]; exact Format.setSign_lt _ _
    · exact (Format.roundRat_spec hg rnd false _ h0
        ((Format.abs_toRat_le_maxVal hf hx).trans hw.2.2)).1
  rw [isFinite_ofPat hlt, toRat_ofPat hlt]
  exact Format.cvt_of_wider hf hg hw rnd false hx

end FloatV

end TVA
