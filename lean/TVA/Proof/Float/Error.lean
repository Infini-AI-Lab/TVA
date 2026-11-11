import TVA.Proof.Float.Value

/-!
# Rounding errors of the float operations

The error bounds a kernel proof uses for each IEEE operation on `FloatV`. With `u = 2^-(m+1)`
and `η = 2^(emin-m-1)` of the format (`Format.u`, `Format.eta`; their values per format are in
`TVA.Proof.Float.Constants`), every operation on finite operands whose exact result `q` is in
range (`|q| ≤ maxVal`) gives a **finite** result within `|q|·u + η` of `q`: a relative error `u`
plus an absolute error `η`, which matters only for results in the subnormal range.

* `add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err` (`FloatV.add`, …: what `tl.add`,
  `tl.sub`, `tl.mul`, `tl.fma`, `tl.div_rn` return up to NaN payload), `cvt_err` (a conversion
  to nearest), `ofRat_err` (the nearest float to a rational); `ofPat_roundRat_err` for a rounding
  written out, `ofPat_roundRat_err_normal` (`|q|·u` for `|q| ≥ 2^emin`).
* `sqrtRn_err` (`tl.sqrt_rn`): of a finite `x ≥ 0`, finite, nonnegative, within `√x·u + η`.
* `cvtWith_rz_err`: toward zero, within `2·(|q|·u + η)` and not larger in magnitude.
* The saturating conversion (`cvtSat`, into fp8): `cvtSat_of_le` (in range it is `cvtWith`),
  `cvtSat_of_gt`, `cvtSat_of_isInf` (beyond the range, `±maxVal`), `cvtSat_rn_err` (one
  rounding of the clamped value).
* `toRat_cvt_of_wider`: a conversion into a wider format keeps the value.
* `add_err_real`, …, `cvtWith_rz_err_real`: the same bounds on `toReal`, with the range
  hypothesis in `ℝ`, for proofs that reason in real numbers.

The `tl.*` operation returns the IEEE result up to NaN payload; `FloatV.NaNEq.eq` (with the
finiteness these lemmas give) turns that into equality.
-/
namespace TVA

namespace FloatV

variable {f : Format}

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

/-! ## On real values -/

/-- A cast inequality of rationals in `ℝ` holds in `ℚ`. -/
private theorem le_of_real {a b : ℚ} (h : (a : ℝ) ≤ (b : ℝ)) : a ≤ b := by exact_mod_cast h

/-- **`x + y` on `toReal`**: finite operands, exact sum in range: finite, within
`|x + y|·u + η`. -/
theorem add_err_real (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hmax : |x.toReal + y.toReal| ≤ (f.maxVal : ℝ)) :
    (add x y).isFinite = true ∧
      |(add x y).toReal - (x.toReal + y.toReal)| ≤ |x.toReal + y.toReal| * (f.u : ℝ) + (f.eta : ℝ) := by
  unfold toReal at *
  obtain ⟨h1, h2⟩ := add_err hf hx hy (le_of_real (by push_cast; exact hmax))
  exact ⟨h1, by exact_mod_cast h2⟩

/-- **`x - y` on `toReal`**: finite, within `|x − y|·u + η`. -/
theorem sub_err_real (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hmax : |x.toReal - y.toReal| ≤ (f.maxVal : ℝ)) :
    (sub x y).isFinite = true ∧
      |(sub x y).toReal - (x.toReal - y.toReal)| ≤ |x.toReal - y.toReal| * (f.u : ℝ) + (f.eta : ℝ) := by
  unfold toReal at *
  obtain ⟨h1, h2⟩ := sub_err hf hx hy (le_of_real (by push_cast; exact hmax))
  exact ⟨h1, by exact_mod_cast h2⟩

/-- **`x * y` on `toReal`**: finite, within `|x·y|·u + η`. -/
theorem mul_err_real (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hmax : |x.toReal * y.toReal| ≤ (f.maxVal : ℝ)) :
    (mul x y).isFinite = true ∧
      |(mul x y).toReal - x.toReal * y.toReal| ≤ |x.toReal * y.toReal| * (f.u : ℝ) + (f.eta : ℝ) := by
  unfold toReal at *
  obtain ⟨h1, h2⟩ := mul_err hf hx hy (le_of_real (by push_cast; exact hmax))
  exact ⟨h1, by exact_mod_cast h2⟩

/-- **`tl.fma(x, y, z)` on `toReal`**: finite, within `|x·y + z|·u + η`. -/
theorem fma_err_real (hf : f.Regular) {x y z : FloatV f} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hz : z.isFinite = true)
    (hmax : |x.toReal * y.toReal + z.toReal| ≤ (f.maxVal : ℝ)) :
    (fma x y z).isFinite = true ∧
      |(fma x y z).toReal - (x.toReal * y.toReal + z.toReal)| ≤
        |x.toReal * y.toReal + z.toReal| * (f.u : ℝ) + (f.eta : ℝ) := by
  unfold toReal at *
  obtain ⟨h1, h2⟩ := fma_err hf hx hy hz (le_of_real (by push_cast; exact hmax))
  exact ⟨h1, by exact_mod_cast h2⟩

/-- **`tl.div_rn(x, y)` on `toReal`**, `y` not a zero: finite, within `|x/y|·u + η`. -/
theorem divRn_err_real (hf : f.Regular) {x y : FloatV f} (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hz : f.isZero y.pat = false)
    (hmax : |x.toReal / y.toReal| ≤ (f.maxVal : ℝ)) :
    (divRn x y).isFinite = true ∧
      |(divRn x y).toReal - x.toReal / y.toReal| ≤ |x.toReal / y.toReal| * (f.u : ℝ) + (f.eta : ℝ) := by
  unfold toReal at *
  obtain ⟨h1, h2⟩ := divRn_err hf hx hy hz (le_of_real (by push_cast; exact hmax))
  exact ⟨h1, by exact_mod_cast h2⟩

/-- **A conversion to nearest on `toReal`** of a finite value in `g`'s range: finite, within
`|x|·u + η` of `g`. -/
theorem cvt_err_real {g : Format} (hg : g.Regular) {x : FloatV f} (hx : x.isFinite = true)
    (hmax : |x.toReal| ≤ (g.maxVal : ℝ)) :
    (cvt x : FloatV g).isFinite = true ∧
      |(cvt x : FloatV g).toReal - x.toReal| ≤ |x.toReal| * (g.u : ℝ) + (g.eta : ℝ) := by
  unfold toReal at *
  obtain ⟨h1, h2⟩ := cvt_err hg hx (le_of_real (by push_cast; exact hmax))
  exact ⟨h1, by exact_mod_cast h2⟩

/-- **A conversion toward zero on `toReal`** (`fp_downcast_rounding="rtz"`) of a finite value in
`g`'s range: finite, within `2·(|x|·u + η)` of `g`, and not larger in magnitude. -/
theorem cvtWith_rz_err_real {g : Format} (hg : g.Regular) {x : FloatV f} (hx : x.isFinite = true)
    (hmax : |x.toReal| ≤ (g.maxVal : ℝ)) :
    (cvtWith .rz x : FloatV g).isFinite = true ∧
      |(cvtWith .rz x : FloatV g).toReal - x.toReal| ≤ 2 * (|x.toReal| * (g.u : ℝ) + (g.eta : ℝ)) ∧
      |(cvtWith .rz x : FloatV g).toReal| ≤ |x.toReal| := by
  unfold toReal at *
  obtain ⟨h1, h2, h3⟩ := cvtWith_rz_err hg hx (le_of_real (by push_cast; exact hmax))
  exact ⟨h1, by exact_mod_cast h2, by exact_mod_cast h3⟩

end FloatV

end TVA
