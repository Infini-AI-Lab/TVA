import TVA.Lemmas.FloatMax
import TVA.Lang.Promote
import TVA.Rules.Num

/-!
# Everyday facts about float values, formats and conversions

Pure lemmas about the library's own definitions (library issue L177); they state nothing about
the device. Kernel proofs had re-derived each of them by hand.

* **Zeros.** `FloatV.ofPat 0` (`+0`): `pat_ofPat_zero`, `toRat_ofPat_zero`, `toReal_ofPat_zero`,
  `isZero_ofPat_zero`, `sign_ofPat_zero` (every format), `isFinite_ofPat_zero`,
  `isNaN_ofPat_zero`, `isInf_ofPat_zero` (a `Format.Regular` one), and `isFinite_zero_f16` …
  `isFinite_zero_f8e4` for each format; `Format.toRat_zero` (the pattern `0`); the constants of a
  zero, `litF64_of_isZero`, `litF32_of_isZero`, `litF16_of_isZero`, `litBF16_of_isZero`,
  `litF8_of_isZero` (and the `…_ofPat_zero` forms); `cvt_ofPat_zero`, `cvtWith_ofPat_zero`,
  `cvtSat_ofPat_zero`, `ofRat_zero`; `tl.zeros_get_f16` … `tl.zeros_get_f8e4` (`tl.zeros` holds
  `+0`); `DType.ofBits_f16` … `DType.ofBits_f8e4nv` (`ofBits d n = ofPat n`, so the default
  element `d.ofBits 0` is `+0`); `DType.val?_zero_f16` … `DType.val?_zero_f8e4nv` (`= some 0`).
* **Constants of the formats.** In `ℚ`, `TVA.Math.Encoding` has `Format.binary32_u`,
  `binary32_eta`, `binary32_maxVal` and the same for `binary16`, `bfloat16`, `binary64`
  (`u`, `η`), `e5m2`, `e4m3`; this module adds `Format.binary64_maxVal`, `e2m1_u`, `e2m1_eta`. In
  `ℝ` (the cast of the rational constant): `Format.binary32_u_real`, `binary32_eta_real`,
  `binary32_maxVal_real`, the same for `binary16`, `bfloat16`, `binary64`, `e5m2`, `e4m3`, `e2m1`.
  Their right sides are closed numbers (`2 ^ (-24 : ℤ)`, `(2 ^ 24 - 1) * 2 ^ 104`, `65504`) that
  `norm_num` evaluates. They are not `@[simp]`: use them by name.
* **Bounds of a value.** `abs_toReal_le_maxVal` (finite), `abs_toRat_le_magVal_top` (any
  pattern, NaN and `±∞` included), `abs_toRat_le_f16` (`131008`), `abs_toReal_le_f16` (`2^17`);
  `two_eta_le_abs_toRat`, `two_eta_le_abs_toReal` (a nonzero value is at least the smallest
  subnormal, `2η`), `two_pow_neg24_le_abs_toRat_f16`; conversely
  `maxVal_lt_abs_toRat_of_not_isFinite`, `isFinite_of_abs_toRat_le_maxVal`,
  `isFinite_of_abs_toReal_le_maxVal` (a value at most `maxVal` in magnitude is finite).
* **Exact widening.** `cvt_of_wider` (`cvt` of a finite value into a `Format.Wider` format keeps
  finiteness, `toRat` and `toReal`), with each pair: `cvt_f16_f32_exact`, `cvt_bf16_f32_exact`,
  `cvt_f32_f64_exact`, `cvt_f16_f64_exact`, `cvt_bf16_f64_exact`, `cvt_f8e5_f16_exact` …
  `cvt_f8e4_f32_exact`; `isFinite_cvt_of_not_isFinite`, `isFinite_cvt_of_wider` (into a standard
  format NaN and `±∞` stay non-finite). For `x.to(…)`: `FloatRel.eq_cvt_of_wider`,
  `FloatRel.exact_of_wider` (every result `semantic.cast` allows for a finite value is `cvt x`,
  finite, same value), `FloatRel.isFinite_eq_of_wider` (between standard formats, finite exactly
  when `x` is), and per pair of dtypes `ConvRel.f16_f32` … `ConvRel.f8e4nv_f32`
  (`r = cvt x`, finite, same `toRat` and `toReal`), `ConvRel.isFinite_f16_f32` …
  `ConvRel.isFinite_bf16_f64`. These are derived, not definitions: `FloatRel` defines a
  conversion between standard formats as `cvt x` up to NaN payload, and one out of fp8 as `cvt x`
  on finite values only (a non-finite fp8 value may convert to any value, so an fp8 widening keeps
  finiteness only one way); exactness is `Format.cvt_of_wider`, from `Format.Wider`.
* **Conversions as `x.to(…)` allows them.** `FloatRel.iff_of_std`, `ConvRel.f32_bf16_iff` …
  (every ordered pair of distinct standard dtypes): exactly `cvt x` up to NaN payload;
  `FloatRel.cvt_of_std`, `FloatRel.cvt_of_f8`, `ConvRel.cvt_f16_f32` … (each such pair, and fp8
  into `float16`, `bfloat16`, `float32`): `cvt x` is an allowed result. `FloatRel.self_iff`: no
  conversion when the formats agree. `cvt_negInf`, `FloatRel.negInf_of_std`: `−∞` stays `−∞`.
* **`NaNEq`** is an equivalence: `NaNEq.refl` (in `TVA.Lang.Ops`), `NaNEq.symm`, `NaNEq.trans`,
  `NaNEq.equivalence`; `NaNEq.isNaN_eq`, `NaNEq.isFinite_eq`, `NaNEq.eq_of_isFinite`,
  `NaNEq.exact_of_isFinite` (the result of a `tl.*` operation whose exact result is finite).
* **`DType.val?`** of the float dtypes: `DType.val?_f16` … `DType.val?_f8e4nv`
  (`= some v ↔ finite ∧ toRat = v`).
* **Rounding errors on `toReal`** (`FloatErr` in `ℝ`, room hypothesis in `ℝ`): `add_err_real`,
  `sub_err_real`, `mul_err_real`, `fma_err_real`, `divRn_err_real`, `cvt_err_real`,
  `cvtWith_rz_err_real` (`sqrtRn_err` is already in `ℝ`).
* **The maximum of approximations.** `abs_max_sub_max_le_of`, `abs_min_sub_min_le_of` (pure);
  `FloatV.max_approx`, `FloatV.min_approx` (`FloatV.max`/`min`); `MaxRel.toRat_eq_max`,
  `MaxRel.toRat_eq_min`, `MaxRel.toReal_eq_max`, `MaxRel.toReal_eq_min`, `MaxRel.max_approx`,
  `MaxRel.min_approx`, `MaxRel.max_approx_rat`, `MaxRel.min_approx_rat` (`tl.maximum`,
  `tl.minimum`); `Tree.eval_maxRel_toReal`, `Tree.eval_minRel_toReal`, `Tree.eval_maxRel_approx`,
  `Tree.eval_minRel_approx`, `Tree.eval_maxRel_approx_sup'`, `Tree.eval_minRel_approx_inf'` (a
  `tl.max`/`tl.min` reduction tree over leaves each within `δ` of `A i` is within `δ` of the
  largest/smallest `A i`). Every one assumes the operands (leaves) are not NaN.
-/
namespace TVA

open TVA.Math.Float

/-! ## Zeros -/

namespace FloatV

variable {f : Format}

/-- The pattern of `ofPat 0` is `0`. -/
@[simp] theorem pat_ofPat_zero : (ofPat 0 : FloatV f).pat = 0 :=
  pat_ofPat (Nat.two_pow_pos _)

/-- `+0` has the value `0` (in every format). -/
@[simp] theorem toRat_ofPat_zero : (ofPat 0 : FloatV f).toRat = 0 := by
  unfold toRat; rw [pat_ofPat_zero]; exact Format.toRat_of_isZero Format.isZero_zero

/-- `+0` has the real value `0`. -/
@[simp] theorem toReal_ofPat_zero : (ofPat 0 : FloatV f).toReal = 0 := by
  unfold toReal; rw [toRat_ofPat_zero, Rat.cast_zero]

/-- `+0` is a zero. -/
@[simp] theorem isZero_ofPat_zero : (ofPat 0 : FloatV f).isZero = true := by
  unfold isZero; rw [pat_ofPat_zero]; exact Format.isZero_zero

/-- `+0` is positive. -/
@[simp] theorem sign_ofPat_zero : (ofPat 0 : FloatV f).sign = false := by
  unfold sign Format.sign; rw [pat_ofPat_zero]; simp

/-- `+0` is finite (a `Format.Regular` format). -/
theorem isFinite_ofPat_zero (hf : f.Regular) : (ofPat 0 : FloatV f).isFinite = true := by
  unfold isFinite; rw [pat_ofPat_zero]; exact Format.isFinite_zero hf

/-- `+0` is not NaN (a `Format.Regular` format). -/
theorem isNaN_ofPat_zero (hf : f.Regular) : (ofPat 0 : FloatV f).isNaN = false :=
  (isFinite_iff.mp (isFinite_ofPat_zero hf)).1

/-- `+0` is not infinite (a `Format.Regular` format). -/
theorem isInf_ofPat_zero (hf : f.Regular) : (ofPat 0 : FloatV f).isInf = false :=
  (isFinite_iff.mp (isFinite_ofPat_zero hf)).2

/-- `+0` of `float16` is finite. -/
@[simp] theorem isFinite_zero_f16 : (ofPat 0 : F16).isFinite = true :=
  isFinite_ofPat_zero Format.binary16_regular
/-- `+0` of `bfloat16` is finite. -/
@[simp] theorem isFinite_zero_bf16 : (ofPat 0 : BF16).isFinite = true :=
  isFinite_ofPat_zero Format.bfloat16_regular
/-- `+0` of `float32` is finite. -/
@[simp] theorem isFinite_zero_f32 : (ofPat 0 : F32).isFinite = true :=
  isFinite_ofPat_zero Format.binary32_regular
/-- `+0` of `float64` is finite. -/
@[simp] theorem isFinite_zero_f64 : (ofPat 0 : F64).isFinite = true :=
  isFinite_ofPat_zero Format.binary64_regular
/-- `+0` of `float8e5` is finite. -/
@[simp] theorem isFinite_zero_f8e5 : (ofPat 0 : F8E5).isFinite = true :=
  isFinite_ofPat_zero Format.e5m2_regular
/-- `+0` of `float8e4nv` is finite. -/
@[simp] theorem isFinite_zero_f8e4 : (ofPat 0 : F8E4).isFinite = true :=
  isFinite_ofPat_zero Format.e4m3_regular

/-- The `float64` constant of a zero Python number is `+0`. -/
theorem litF64_of_isZero {c : F64} (h : c.isZero = true) : litF64 c = ofPat 0 := by
  unfold litF64; rw [ite_eq_left h]

/-- The `float32` constant of a zero Python number (`0.0` or `-0.0`) is `+0`. -/
theorem litF32_of_isZero {c : F64} (h : c.isZero = true) : litF32 c = ofPat 0 := by
  unfold litF32; rw [ite_eq_left h]

/-- The `float16` constant of a zero Python number is `+0`. -/
theorem litF16_of_isZero {c : F64} (h : c.isZero = true) : litF16 c = ofPat 0 := by
  unfold litF16; rw [ite_eq_left h]

/-- The `bfloat16` constant of a zero Python number is `+0`. -/
theorem litBF16_of_isZero {c : F64} (h : c.isZero = true) : litBF16 c = ofPat 0 := by
  unfold litBF16; rw [ite_eq_left h]

/-- The fp8 constant of a zero Python number is `+0` (Triton accepts it). -/
theorem litF8_of_isZero {g : Format} {c : F64} (h : c.isZero = true) :
    (litF8 c : Option (FloatV g)) = some (ofPat 0) := by
  unfold litF8; rw [ite_eq_left h]

/-- `litF64 0.0 = +0`. -/
@[simp] theorem litF64_ofPat_zero : litF64 (ofPat 0) = ofPat 0 := litF64_of_isZero isZero_ofPat_zero
/-- `litF32 0.0 = +0`. -/
@[simp] theorem litF32_ofPat_zero : litF32 (ofPat 0) = ofPat 0 := litF32_of_isZero isZero_ofPat_zero
/-- `litF16 0.0 = +0`. -/
@[simp] theorem litF16_ofPat_zero : litF16 (ofPat 0) = ofPat 0 := litF16_of_isZero isZero_ofPat_zero
/-- `litBF16 0.0 = +0`. -/
@[simp] theorem litBF16_ofPat_zero : litBF16 (ofPat 0) = ofPat 0 :=
  litBF16_of_isZero isZero_ofPat_zero
/-- `litF8 0.0 = some +0`. -/
@[simp] theorem litF8_ofPat_zero {g : Format} : (litF8 (ofPat 0) : Option (FloatV g)) = some (ofPat 0) :=
  litF8_of_isZero isZero_ofPat_zero

/-- `+0` is the pattern `0` of every format with the sign `false`. -/
private theorem setSign_false_zero (g : Format) : g.setSign false 0 = 0 := by
  unfold Format.setSign Format.mag; simp

/-- The conversion of `+0` (either rounding mode, saturating or not) is `+0`. -/
private theorem cvt_pat_zero (hf : f.Regular) (g : Format) (rnd : RoundMode) (sat : Bool) :
    Format.cvt f g rnd sat (ofPat 0 : FloatV f).pat = 0 := by
  rw [pat_ofPat_zero]
  have hfin := Format.isFinite_zero (f := f) hf
  rw [Format.cvt_of_finite g rnd sat hfin, Format.toRat_of_isZero Format.isZero_zero,
    Format.roundRat_zero]
  have : f.sign 0 = false := by unfold Format.sign; simp
  rw [this, setSign_false_zero]

/-- **`cvt` of `+0` is `+0`** (a `Format.Regular` source). -/
theorem cvt_ofPat_zero {g : Format} (hf : f.Regular) :
    (cvt (ofPat 0 : FloatV f) : FloatV g) = ofPat 0 := by
  unfold cvt; rw [cvt_pat_zero hf]

/-- `cvtWith rnd` of `+0` is `+0`. -/
theorem cvtWith_ofPat_zero {g : Format} (hf : f.Regular) (rnd : RoundMode) :
    (cvtWith rnd (ofPat 0 : FloatV f) : FloatV g) = ofPat 0 := by
  unfold cvtWith; rw [cvt_pat_zero hf]

/-- The saturating conversion of `+0` is `+0`. -/
theorem cvtSat_ofPat_zero {g : Format} (hf : f.Regular) (rnd : RoundMode) :
    (cvtSat rnd (ofPat 0 : FloatV f) : FloatV g) = ofPat 0 := by
  unfold cvtSat; rw [cvt_pat_zero hf]

/-- The nearest float to `0` is `+0`. -/
@[simp] theorem ofRat_zero : (ofRat 0 : FloatV f) = ofPat 0 := by
  unfold ofRat; rw [Format.roundRat_zero, setSign_false_zero]

/-- `toReal` is the cast of `toRat`. -/
theorem toReal_eq (x : FloatV f) : x.toReal = (x.toRat : ℝ) := rfl

/-- Two values have one real value iff they have one rational value. -/
theorem toReal_eq_toReal_iff {g : Format} {x : FloatV f} {y : FloatV g} :
    x.toReal = y.toReal ↔ x.toRat = y.toRat := by
  unfold toReal; exact Rat.cast_inj

/-- `toReal` of `-x`. -/
theorem toReal_neg (x : FloatV f) : (neg x).toReal = -x.toReal := by
  unfold toReal; rw [toRat_neg, Rat.cast_neg]

end FloatV

/-- The pattern `0` (`+0`) has the value `0`, in every format. -/
@[simp] theorem Math.Float.Format.toRat_zero (f : Format) : f.toRat 0 = 0 :=
  Format.toRat_of_isZero Format.isZero_zero

/-- The `float16` value with bit pattern `n` (`DType.ofBits`, as in a bit cast) is `ofPat n`; so
`DType.f16.ofBits 0`, the default element of a `float16` tensor, is `+0`. -/
@[simp] theorem DType.ofBits_f16 (n : Nat) : DType.ofBits .f16 n = (FloatV.ofPat n : F16) := rfl
/-- `DType.ofBits .bf16 n` is `ofPat n`. -/
@[simp] theorem DType.ofBits_bf16 (n : Nat) : DType.ofBits .bf16 n = (FloatV.ofPat n : BF16) := rfl
/-- `DType.ofBits .f32 n` is `ofPat n`. -/
@[simp] theorem DType.ofBits_f32 (n : Nat) : DType.ofBits .f32 n = (FloatV.ofPat n : F32) := rfl
/-- `DType.ofBits .f64 n` is `ofPat n`. -/
@[simp] theorem DType.ofBits_f64 (n : Nat) : DType.ofBits .f64 n = (FloatV.ofPat n : F64) := rfl
/-- `DType.ofBits .f8e5 n` is `ofPat n`. -/
@[simp] theorem DType.ofBits_f8e5 (n : Nat) : DType.ofBits .f8e5 n = (FloatV.ofPat n : F8E5) := rfl
/-- `DType.ofBits .f8e4nv n` is `ofPat n`. -/
@[simp] theorem DType.ofBits_f8e4nv (n : Nat) :
    DType.ofBits .f8e4nv n = (FloatV.ofPat n : F8E4) := rfl

/-- `tl.zeros(…, tl.float16)` holds `+0`. -/
@[simp] theorem tl.zeros_get_f16 {s : Shape} (i : Idx s) :
    (tl.zeros s .f16).get i = (FloatV.ofPat 0 : F16) := rfl
/-- `tl.zeros(…, tl.bfloat16)` holds `+0`. -/
@[simp] theorem tl.zeros_get_bf16 {s : Shape} (i : Idx s) :
    (tl.zeros s .bf16).get i = (FloatV.ofPat 0 : BF16) := rfl
/-- `tl.zeros(…, tl.float32)` holds `+0`. -/
@[simp] theorem tl.zeros_get_f32 {s : Shape} (i : Idx s) :
    (tl.zeros s .f32).get i = (FloatV.ofPat 0 : F32) := rfl
/-- `tl.zeros(…, tl.float64)` holds `+0`. -/
@[simp] theorem tl.zeros_get_f64 {s : Shape} (i : Idx s) :
    (tl.zeros s .f64).get i = (FloatV.ofPat 0 : F64) := rfl
/-- `tl.zeros(…, tl.float8e5)` holds `+0`. -/
@[simp] theorem tl.zeros_get_f8e5 {s : Shape} (i : Idx s) :
    (tl.zeros s .f8e5).get i = (FloatV.ofPat 0 : F8E5) := rfl
/-- `tl.zeros(…, tl.float8e4nv)` holds `+0`. -/
@[simp] theorem tl.zeros_get_f8e4 {s : Shape} (i : Idx s) :
    (tl.zeros s .f8e4nv).get i = (FloatV.ofPat 0 : F8E4) := rfl

/-! ## The constants of the formats -/

namespace Math.Float.Format

/-- `maxVal` of `binary64`: `(2^53 − 1)·2^971`. -/
theorem binary64_maxVal : binary64.maxVal = (2 ^ 53 - 1) * 2 ^ 971 := by
  rw [maxVal_eq]; unfold topSig topExp bias binary64; norm_num

/-- `u` of `e2m1`: `2^-2`. -/
theorem e2m1_u : e2m1.u = 2 ^ (-2 : ℤ) := by rw [u_eq]; unfold e2m1; norm_num

/-- `η` of `e2m1`: `2^-2`. -/
theorem e2m1_eta : e2m1.eta = 2 ^ (-2 : ℤ) := by rw [eta_eq]; unfold emin bias e2m1; norm_num

/-- `u` of `binary32` in `ℝ`: `2^-24`. -/
theorem binary32_u_real : ((binary32.u : ℚ) : ℝ) = 2 ^ (-24 : ℤ) := by rw [binary32_u]; push_cast; rfl
/-- `η` of `binary32` in `ℝ`: `2^-150`. -/
theorem binary32_eta_real : ((binary32.eta : ℚ) : ℝ) = 2 ^ (-150 : ℤ) := by
  rw [binary32_eta]; push_cast; rfl
/-- `maxVal` of `binary32` in `ℝ`: `(2^24 − 1)·2^104`. -/
theorem binary32_maxVal_real : ((binary32.maxVal : ℚ) : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [binary32_maxVal]; push_cast; rfl

/-- `u` of `binary16` in `ℝ`: `2^-11`. -/
theorem binary16_u_real : ((binary16.u : ℚ) : ℝ) = 2 ^ (-11 : ℤ) := by rw [binary16_u]; push_cast; rfl
/-- `η` of `binary16` in `ℝ`: `2^-25`. -/
theorem binary16_eta_real : ((binary16.eta : ℚ) : ℝ) = 2 ^ (-25 : ℤ) := by
  rw [binary16_eta]; push_cast; rfl
/-- `maxVal` of `binary16` in `ℝ`: `65504`. -/
theorem binary16_maxVal_real : ((binary16.maxVal : ℚ) : ℝ) = 65504 := by
  rw [binary16_maxVal]; push_cast; rfl

/-- `u` of `bfloat16` in `ℝ`: `2^-8`. -/
theorem bfloat16_u_real : ((bfloat16.u : ℚ) : ℝ) = 2 ^ (-8 : ℤ) := by rw [bfloat16_u]; push_cast; rfl
/-- `η` of `bfloat16` in `ℝ`: `2^-134`. -/
theorem bfloat16_eta_real : ((bfloat16.eta : ℚ) : ℝ) = 2 ^ (-134 : ℤ) := by
  rw [bfloat16_eta]; push_cast; rfl
/-- `maxVal` of `bfloat16` in `ℝ`: `(2^8 − 1)·2^120`. -/
theorem bfloat16_maxVal_real : ((bfloat16.maxVal : ℚ) : ℝ) = (2 ^ 8 - 1) * 2 ^ 120 := by
  rw [bfloat16_maxVal]; push_cast; rfl

/-- `u` of `binary64` in `ℝ`: `2^-53`. -/
theorem binary64_u_real : ((binary64.u : ℚ) : ℝ) = 2 ^ (-53 : ℤ) := by rw [binary64_u]; push_cast; rfl
/-- `η` of `binary64` in `ℝ`: `2^-1075`. -/
theorem binary64_eta_real : ((binary64.eta : ℚ) : ℝ) = 2 ^ (-1075 : ℤ) := by
  rw [binary64_eta, Rat.cast_zpow, Rat.cast_ofNat]
/-- `maxVal` of `binary64` in `ℝ`: `(2^53 − 1)·2^971`. -/
theorem binary64_maxVal_real : ((binary64.maxVal : ℚ) : ℝ) = (2 ^ 53 - 1) * 2 ^ 971 := by
  rw [binary64_maxVal]; push_cast; rfl

/-- `u` of `e5m2` in `ℝ`: `2^-3`. -/
theorem e5m2_u_real : ((e5m2.u : ℚ) : ℝ) = 2 ^ (-3 : ℤ) := by rw [e5m2_u]; push_cast; rfl
/-- `η` of `e5m2` in `ℝ`: `2^-17`. -/
theorem e5m2_eta_real : ((e5m2.eta : ℚ) : ℝ) = 2 ^ (-17 : ℤ) := by rw [e5m2_eta]; push_cast; rfl
/-- `maxVal` of `e5m2` in `ℝ`: `57344`. -/
theorem e5m2_maxVal_real : ((e5m2.maxVal : ℚ) : ℝ) = 57344 := by rw [e5m2_maxVal]; push_cast; rfl

/-- `u` of `e4m3` in `ℝ`: `2^-4`. -/
theorem e4m3_u_real : ((e4m3.u : ℚ) : ℝ) = 2 ^ (-4 : ℤ) := by rw [e4m3_u]; push_cast; rfl
/-- `η` of `e4m3` in `ℝ`: `2^-10`. -/
theorem e4m3_eta_real : ((e4m3.eta : ℚ) : ℝ) = 2 ^ (-10 : ℤ) := by rw [e4m3_eta]; push_cast; rfl
/-- `maxVal` of `e4m3` in `ℝ`: `448`. -/
theorem e4m3_maxVal_real : ((e4m3.maxVal : ℚ) : ℝ) = 448 := by rw [e4m3_maxVal]; push_cast; rfl

/-- `u` of `e2m1` in `ℝ`: `2^-2`. -/
theorem e2m1_u_real : ((e2m1.u : ℚ) : ℝ) = 2 ^ (-2 : ℤ) := by rw [e2m1_u]; push_cast; rfl
/-- `η` of `e2m1` in `ℝ`: `2^-2`. -/
theorem e2m1_eta_real : ((e2m1.eta : ℚ) : ℝ) = 2 ^ (-2 : ℤ) := by rw [e2m1_eta]; push_cast; rfl
/-- `maxVal` of `e2m1` in `ℝ`: `6`. -/
theorem e2m1_maxVal_real : ((e2m1.maxVal : ℚ) : ℝ) = 6 := by rw [e2m1_maxVal]; push_cast; rfl

/-- `binary16` (`float16`) is `Std`. -/
theorem binary16_std : binary16.Std := Or.inl rfl
/-- `bfloat16` is `Std`. -/
theorem bfloat16_std : bfloat16.Std := Or.inr (Or.inl rfl)
/-- `binary32` is `Std`. -/
theorem binary32_std : binary32.Std := Or.inr (Or.inr (Or.inl rfl))
/-- `binary64` is `Std`. -/
theorem binary64_std : binary64.Std := Or.inr (Or.inr (Or.inr rfl))

/-- A standard format is `Regular`. -/
theorem Std.regular {f : Format} (h : f.Std) : f.Regular := by
  rcases h with rfl | rfl | rfl | rfl <;> decide

/-- A standard format has infinities (`.ieee`). -/
theorem Std.ieee {f : Format} (h : f.Std) : f.specials = .ieee := by
  rcases h with rfl | rfl | rfl | rfl <;> rfl

/-- A standard format is not an fp8 format. -/
theorem Std.not_f8 {f : Format} (h : f.Std) : ¬ f.F8 := by
  rintro (h8 | h8) <;> rcases h with rfl | rfl | rfl | rfl <;> exact absurd h8 (by decide)

end Math.Float.Format

/-! ## Bounds of a value -/

namespace FloatV

variable {f : Format}

/-- A finite value is at most `maxVal` in magnitude, in `ℝ`. -/
theorem abs_toReal_le_maxVal (hf : f.Regular) {x : FloatV f} (hx : x.isFinite = true) :
    |x.toReal| ≤ (f.maxVal : ℝ) := by
  unfold toReal; rw [← Rat.cast_abs]; exact_mod_cast abs_toRat_le_maxVal hf hx

/-- **Every pattern** (NaN and `±∞` included) has a value at most the value of the largest
magnitude `signBit − 1`. -/
theorem abs_toRat_le_magVal_top (x : FloatV f) : |x.toRat| ≤ f.magVal (f.signBit - 1) := by
  unfold toRat; rw [Format.abs_toRat]
  apply Format.magVal_le_magVal
  have := Format.mag_lt (f := f) x.pat
  omega

/-- Every `float16` pattern has a value at most `131008` in magnitude (`65504` when finite). -/
theorem abs_toRat_le_f16 (x : F16) : |x.toRat| ≤ 131008 := by
  have h := abs_toRat_le_magVal_top x
  have e : Format.binary16.magVal (Format.binary16.signBit - 1) = 131008 := by
    rw [Format.magVal_eq]
    norm_num [Format.binary16, Format.signBit, Format.bias, Format.emin]
  rwa [e] at h

/-- Every `float16` pattern has a real value at most `2^17` in magnitude. -/
theorem abs_toReal_le_f16 (x : F16) : |x.toReal| ≤ 2 ^ 17 := by
  have h := abs_toRat_le_f16 x
  unfold toReal; rw [← Rat.cast_abs]
  have : ((|x.toRat| : ℚ) : ℝ) ≤ ((131008 : ℚ) : ℝ) := by exact_mod_cast h
  linarith [show ((131008 : ℚ) : ℝ) = 131008 by norm_num]

/-- **A nonzero value is at least the smallest subnormal `2η`** in magnitude (any pattern). -/
theorem two_eta_le_abs_toRat (hf : f.Regular) {x : FloatV f} (h : x.toRat ≠ 0) :
    2 * f.eta ≤ |x.toRat| := by
  unfold toRat at h ⊢
  rw [Format.abs_toRat]
  have hm : 1 ≤ f.mag x.pat := by
    by_contra h0
    have : f.mag x.pat = 0 := by omega
    apply h
    exact Format.toRat_of_isZero (by unfold Format.isZero; simp [this])
  have h1 : f.magVal 1 = 2 * f.eta := by
    have h2 := Format.two_pow_m_ge hf
    rw [Format.magVal_eq, Format.eta_eq]
    have hd : 1 / 2 ^ f.m = 0 := Nat.div_eq_of_lt (by omega)
    have hr : 1 % 2 ^ f.m = 1 := Nat.mod_eq_of_lt (by omega)
    rw [ite_eq_left hd, hr]
    rw [show f.emin - (f.m : ℤ) = (f.emin - f.m - 1) + 1 by ring, zpow_add_one₀ (by norm_num)]
    push_cast; ring_nf
  rw [← h1]
  exact Format.magVal_le_magVal hm

/-- `two_eta_le_abs_toRat` in `ℝ`: a nonzero value is at least `2η` in magnitude. -/
theorem two_eta_le_abs_toReal (hf : f.Regular) {x : FloatV f} (h : x.toReal ≠ 0) :
    2 * (f.eta : ℝ) ≤ |x.toReal| := by
  have h' : x.toRat ≠ 0 := fun e => h (by unfold toReal; rw [e, Rat.cast_zero])
  unfold toReal; rw [← Rat.cast_abs]; exact_mod_cast two_eta_le_abs_toRat hf h'

/-- **A pattern that is not finite lies beyond `maxVal`** (a `Format.Regular` format): NaN and
`±∞` have `|toRat| > maxVal`. -/
theorem maxVal_lt_abs_toRat_of_not_isFinite (hf : f.Regular) {x : FloatV f}
    (hx : x.isFinite = false) : f.maxVal < |x.toRat| := by
  by_contra hle
  have hm : f.mag x.pat ≤ f.maxMag := by
    by_contra hlt
    apply hle
    unfold toRat; rw [Format.abs_toRat, ← Format.magVal_maxMag hf]
    exact Format.magVal_lt_magVal (not_le.mp hlt)
  have := Format.isFinite_of_mag_le hf hm
  unfold isFinite at hx; rw [this] at hx; cases hx

/-- **A value at most `maxVal` in magnitude is finite** (a `Format.Regular` format): the converse
of `abs_toRat_le_maxVal`. -/
theorem isFinite_of_abs_toRat_le_maxVal (hf : f.Regular) {x : FloatV f}
    (h : |x.toRat| ≤ f.maxVal) : x.isFinite = true := by
  cases hx : x.isFinite
  · exact absurd h (not_le.mpr (maxVal_lt_abs_toRat_of_not_isFinite hf hx))
  · rfl

/-- `isFinite_of_abs_toRat_le_maxVal` in `ℝ`. -/
theorem isFinite_of_abs_toReal_le_maxVal (hf : f.Regular) {x : FloatV f}
    (h : |x.toReal| ≤ (f.maxVal : ℝ)) : x.isFinite = true := by
  apply isFinite_of_abs_toRat_le_maxVal hf
  unfold toReal at h; rw [← Rat.cast_abs] at h; exact_mod_cast h

/-- A nonzero `float16` value is at least `2^-24` in magnitude. -/
theorem two_pow_neg24_le_abs_toRat_f16 {x : F16} (h : x.toRat ≠ 0) :
    (2 : ℚ) ^ (-24 : ℤ) ≤ |x.toRat| := by
  have := two_eta_le_abs_toRat Format.binary16_regular h
  rw [Format.binary16_eta] at this
  calc (2 : ℚ) ^ (-24 : ℤ) = 2 * 2 ^ (-25 : ℤ) := by norm_num
    _ ≤ _ := this

/-! ## Exact widening -/

/-- **`cvt` into a wider format is exact**: finite, same rational and real value. -/
theorem cvt_of_wider {g : Format} (hf : f.Regular) (hg : g.Regular) (hw : Format.Wider f g)
    {x : FloatV f} (hx : x.isFinite = true) :
    (cvt x : FloatV g).isFinite = true ∧ (cvt x : FloatV g).toRat = x.toRat ∧
      (cvt x : FloatV g).toReal = x.toReal := by
  obtain ⟨h1, h2⟩ := toRat_cvt_of_wider hf hg hw .rn hx
  rw [cvtWith_rn] at h1 h2
  exact ⟨h1, h2, by unfold toReal; rw [h2]⟩

/-- `float16 → float32` is exact. -/
theorem cvt_f16_f32_exact {x : F16} (hx : x.isFinite = true) :
    (cvt x : F32).isFinite = true ∧ (cvt x : F32).toRat = x.toRat ∧ (cvt x : F32).toReal = x.toReal :=
  cvt_of_wider Format.binary16_regular Format.binary32_regular Format.wider_binary16_binary32 hx
/-- `bfloat16 → float32` is exact. -/
theorem cvt_bf16_f32_exact {x : BF16} (hx : x.isFinite = true) :
    (cvt x : F32).isFinite = true ∧ (cvt x : F32).toRat = x.toRat ∧ (cvt x : F32).toReal = x.toReal :=
  cvt_of_wider Format.bfloat16_regular Format.binary32_regular Format.wider_bfloat16_binary32 hx
/-- `float32 → float64` is exact. -/
theorem cvt_f32_f64_exact {x : F32} (hx : x.isFinite = true) :
    (cvt x : F64).isFinite = true ∧ (cvt x : F64).toRat = x.toRat ∧ (cvt x : F64).toReal = x.toReal :=
  cvt_of_wider Format.binary32_regular Format.binary64_regular Format.wider_binary32_binary64 hx
/-- `float16 → float64` is exact. -/
theorem cvt_f16_f64_exact {x : F16} (hx : x.isFinite = true) :
    (cvt x : F64).isFinite = true ∧ (cvt x : F64).toRat = x.toRat ∧ (cvt x : F64).toReal = x.toReal :=
  cvt_of_wider Format.binary16_regular Format.binary64_regular Format.wider_binary16_binary64 hx
/-- `bfloat16 → float64` is exact. -/
theorem cvt_bf16_f64_exact {x : BF16} (hx : x.isFinite = true) :
    (cvt x : F64).isFinite = true ∧ (cvt x : F64).toRat = x.toRat ∧ (cvt x : F64).toReal = x.toReal :=
  cvt_of_wider Format.bfloat16_regular Format.binary64_regular Format.wider_bfloat16_binary64 hx
/-- `float8e5 → float16` is exact. -/
theorem cvt_f8e5_f16_exact {x : F8E5} (hx : x.isFinite = true) :
    (cvt x : F16).isFinite = true ∧ (cvt x : F16).toRat = x.toRat ∧ (cvt x : F16).toReal = x.toReal :=
  cvt_of_wider Format.e5m2_regular Format.binary16_regular Format.wider_e5m2_binary16 hx
/-- `float8e5 → bfloat16` is exact. -/
theorem cvt_f8e5_bf16_exact {x : F8E5} (hx : x.isFinite = true) :
    (cvt x : BF16).isFinite = true ∧ (cvt x : BF16).toRat = x.toRat ∧ (cvt x : BF16).toReal = x.toReal :=
  cvt_of_wider Format.e5m2_regular Format.bfloat16_regular Format.wider_e5m2_bfloat16 hx
/-- `float8e5 → float32` is exact. -/
theorem cvt_f8e5_f32_exact {x : F8E5} (hx : x.isFinite = true) :
    (cvt x : F32).isFinite = true ∧ (cvt x : F32).toRat = x.toRat ∧ (cvt x : F32).toReal = x.toReal :=
  cvt_of_wider Format.e5m2_regular Format.binary32_regular Format.wider_e5m2_binary32 hx
/-- `float8e4nv → float16` is exact. -/
theorem cvt_f8e4_f16_exact {x : F8E4} (hx : x.isFinite = true) :
    (cvt x : F16).isFinite = true ∧ (cvt x : F16).toRat = x.toRat ∧ (cvt x : F16).toReal = x.toReal :=
  cvt_of_wider Format.e4m3_regular Format.binary16_regular Format.wider_e4m3_binary16 hx
/-- `float8e4nv → bfloat16` is exact. -/
theorem cvt_f8e4_bf16_exact {x : F8E4} (hx : x.isFinite = true) :
    (cvt x : BF16).isFinite = true ∧ (cvt x : BF16).toRat = x.toRat ∧ (cvt x : BF16).toReal = x.toReal :=
  cvt_of_wider Format.e4m3_regular Format.bfloat16_regular Format.wider_e4m3_bfloat16 hx
/-- `float8e4nv → float32` is exact. -/
theorem cvt_f8e4_f32_exact {x : F8E4} (hx : x.isFinite = true) :
    (cvt x : F32).isFinite = true ∧ (cvt x : F32).toRat = x.toRat ∧ (cvt x : F32).toReal = x.toReal :=
  cvt_of_wider Format.e4m3_regular Format.binary32_regular Format.wider_e4m3_binary32 hx

/-- **`cvt` into a standard format keeps a value that is not finite not finite**: a NaN gives a
NaN and `±∞` gives `±∞`. -/
theorem isFinite_cvt_of_not_isFinite {g : Format} (hg : g.Std) {x : FloatV f}
    (hx : x.isFinite = false) : (cvt x : FloatV g).isFinite = false := by
  have hreg := hg.regular
  have hs := hg.ieee
  unfold cvt Format.cvt
  by_cases hn : f.isNaN x.pat = true
  · rw [ite_eq_left hn]
    have hlt : g.nanMag < 2 ^ g.width := by
      refine lt_trans ?_ g.signBit_lt
      unfold Format.nanMag Format.signBit
      have := (show 0 < 2 ^ (g.e + g.m) by positivity); omega
    rw [isFinite_ofPat hlt]
    exact Format.not_isFinite_nanMag hreg (by rw [hs]; decide)
  · have hi : f.isInf x.pat = true := by
      unfold isFinite Format.isFinite at hx
      cases h1 : f.isNaN x.pat <;> cases h2 : f.isInf x.pat <;> simp_all
    rw [ite_eq_right hn, ite_eq_left hi]
    simp only [Bool.false_eq_true, ite_false, Format.infPat, hs]
    rw [isFinite_ofPat (Format.setSign_lt _ _), Format.isFinite_setSign]
    exact Format.not_isFinite_infMag hs

/-- **`cvt` into a wider standard format keeps finiteness both ways**: `cvt x` is finite exactly
when `x` is. -/
theorem isFinite_cvt_of_wider {g : Format} (hf : f.Regular) (hg : g.Std) (hw : Format.Wider f g)
    (x : FloatV f) : (cvt x : FloatV g).isFinite = x.isFinite := by
  cases hx : x.isFinite
  · exact isFinite_cvt_of_not_isFinite hg hx
  · exact (cvt_of_wider hf hg.regular hw hx).1

/-- **`cvt` of `−∞` is `−∞`** between formats with infinities. -/
theorem cvt_negInf {g : Format} (hs : f.specials = .ieee) (hgs : g.specials = .ieee) :
    (cvt (negInf : FloatV f) : FloatV g) = negInf := by
  unfold cvt
  have h1 : f.isNaN (negInf : FloatV f).pat = false := negInf_isNaN hs
  have h2 : f.isInf (negInf : FloatV f).pat = true := negInf_isInf hs
  have h3 : f.sign (negInf : FloatV f).pat = true := negInf_sign
  unfold Format.cvt
  rw [h1, h2, h3]
  simp only [Bool.false_eq_true, ite_false, ite_true]
  unfold Format.infPat; rw [hgs]; rfl

/-! ## `NaNEq` -/

/-- `NaNEq` is symmetric. -/
theorem NaNEq.symm {x y : FloatV f} (h : NaNEq x y) : NaNEq y x := by
  rcases h with h | ⟨h1, h2⟩
  · exact Or.inl h.symm
  · exact Or.inr ⟨h2, h1⟩

/-- `NaNEq` is transitive. -/
theorem NaNEq.trans {x y z : FloatV f} (h1 : NaNEq x y) (h2 : NaNEq y z) : NaNEq x z := by
  rcases h1 with rfl | ⟨hx, hy⟩
  · exact h2
  · rcases h2 with rfl | ⟨-, hz⟩
    · exact Or.inr ⟨hx, hy⟩
    · exact Or.inr ⟨hx, hz⟩

/-- `NaNEq` is an equivalence. -/
theorem NaNEq.equivalence : Equivalence (NaNEq (f := f)) :=
  ⟨NaNEq.refl, NaNEq.symm, NaNEq.trans⟩

/-- Values equal up to NaN payload are both NaN or both not. -/
theorem NaNEq.isNaN_eq {x y : FloatV f} (h : NaNEq x y) : x.isNaN = y.isNaN := by
  rcases h with rfl | ⟨h1, h2⟩
  · rfl
  · rw [h1, h2]

/-- Values equal up to NaN payload are both finite or both not. -/
theorem NaNEq.isFinite_eq {x y : FloatV f} (h : NaNEq x y) : x.isFinite = y.isFinite := by
  rcases h with rfl | ⟨h1, h2⟩
  · rfl
  · have a : x.isFinite = false := by
      unfold isFinite Format.isFinite; unfold isNaN at h1; rw [h1]; rfl
    have b : y.isFinite = false := by
      unfold isFinite Format.isFinite; unfold isNaN at h2; rw [h2]; rfl
    rw [a, b]

/-- **A finite value is exact up to NaN payload**: `NaNEq x y` with `x` finite gives `y = x`. -/
theorem NaNEq.eq_of_isFinite {x y : FloatV f} (h : NaNEq x y) (hx : x.isFinite = true) : y = x :=
  h.eq (isFinite_iff.mp hx).1

/-- **What a `tl.*` operation returns for a finite exact result**: from `NaNEq x y` (`y` the
result, `x` the library's operation) and `x` finite, `y` is finite with `x`'s value. Combine it
with an error bound on `x` (`add_err_real`, …). -/
theorem NaNEq.exact_of_isFinite {x y : FloatV f} (h : NaNEq x y) (hx : x.isFinite = true) :
    y.isFinite = true ∧ y.toRat = x.toRat ∧ y.toReal = x.toReal := by
  rw [h.eq_of_isFinite hx]; exact ⟨hx, rfl, rfl⟩

/-! ## Rounding errors on `toReal` -/

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

/-! ## Conversions as `semantic.cast` allows them -/

namespace FloatRel

open Math.Float.Format

variable {P : Platform} {f g : Format}

/-- **No conversion between equal formats**: `FloatRel P f f x r` is `r = x`. -/
theorem self_iff {x r : FloatV f} : FloatRel P f f x r ↔ r = x := by
  unfold FloatRel; rw [ite_eq_left rfl]
  exact ⟨FloatV.ext_pat, fun h => by rw [h]⟩

/-- **A widening conversion is `cvt x`** on a finite value (formats differ; the target is not
fp8): between standard formats `FloatRel` gives `cvt x` up to NaN payload, and `cvt x` of a
finite value is finite; out of fp8 it gives `cvt x` on finite values. -/
theorem eq_cvt_of_wider (hf : f.Regular) (hg : g.Regular) (hw : Wider f g) (hfg : f ≠ g)
    (hg8 : ¬ g.F8) {x : FloatV f} {r : FloatV g} (hx : x.isFinite = true)
    (h : FloatRel P f g x r) : r = FloatV.cvt x := by
  unfold FloatRel at h; rw [ite_eq_right hfg] at h
  rcases h with ⟨-, -, h⟩ | ⟨-, h8, -⟩ | ⟨-, -, h⟩
  · exact h.eq_of_isFinite (FloatV.cvt_of_wider hf hg hw hx).1
  · exact absurd h8 hg8
  · exact h hx

/-- **A widening conversion is exact** (formats equal or `Wider`; the target not fp8): every
result `FloatRel` allows for a finite value is finite with the same value. -/
theorem exact_of_wider (hf : f.Regular) (hg : g.Regular) (hw : Wider f g) (hg8 : ¬ g.F8)
    {x : FloatV f} {r : FloatV g} (hx : x.isFinite = true) (h : FloatRel P f g x r) :
    r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal := by
  by_cases hfg : f = g
  · subst hfg
    rw [self_iff.mp h]; exact ⟨hx, rfl, rfl⟩
  · rw [eq_cvt_of_wider hf hg hw hfg hg8 hx h]
    exact FloatV.cvt_of_wider hf hg hw hx

/-- **Between distinct standard formats `semantic.cast` is `cvt`** up to NaN payload. -/
theorem iff_of_std (hf : f.Std) (hg : g.Std) (hfg : f ≠ g) {x : FloatV f} {r : FloatV g} :
    FloatRel P f g x r ↔ FloatV.NaNEq (FloatV.cvt x) r := by
  unfold FloatRel; rw [ite_eq_right hfg]
  constructor
  · rintro (⟨-, -, h⟩ | ⟨-, h8, -⟩ | ⟨h8, -, -⟩)
    · exact h
    · exact absurd h8 hg.not_f8
    · exact absurd h8 hf.not_f8
  · intro h; exact Or.inl ⟨hf, hg, h⟩

/-- **`−∞` converts to `−∞`** between standard formats. -/
theorem negInf_of_std (hf : f.Std) (hg : g.Std) {r : FloatV g}
    (h : FloatRel P f g FloatV.negInf r) : r = FloatV.negInf := by
  by_cases hfg : f = g
  · subst hfg; exact self_iff.mp h
  · have h' := (iff_of_std hf hg hfg).mp h
    rw [FloatV.cvt_negInf hf.ieee hg.ieee] at h'
    exact h'.eq (FloatV.negInf_isNaN hg.ieee)

/-- **`cvt x` is a result `semantic.cast` allows** between distinct standard formats. -/
theorem cvt_of_std (hf : f.Std) (hg : g.Std) (hfg : f ≠ g) (x : FloatV f) :
    FloatRel P f g x (FloatV.cvt x) :=
  (iff_of_std hf hg hfg).mpr (FloatV.NaNEq.refl _)

/-- **`cvt x` is a result `semantic.cast` allows** out of fp8 into `float16`, `bfloat16` or
`float32` (any value at all is allowed for a non-finite fp8 value). -/
theorem cvt_of_f8 (hf : f.F8) (hg : g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32)
    (x : FloatV f) : FloatRel P f g x (FloatV.cvt x) := by
  have hfg : f ≠ g := by
    rintro rfl
    rcases hf with rfl | rfl <;> rcases hg with h | h | h <;> exact absurd h (by decide)
  unfold FloatRel; rw [ite_eq_right hfg]
  exact Or.inr (Or.inr ⟨hf, hg, fun _ => rfl⟩)

/-- **A widening between standard formats keeps finiteness both ways**: every result `r` that
`FloatRel` allows is finite exactly when `x` is (NaN stays NaN, `±∞` stays `±∞`). -/
theorem isFinite_eq_of_wider (hf : f.Std) (hg : g.Std) (hw : Wider f g) {x : FloatV f}
    {r : FloatV g} (h : FloatRel P f g x r) : r.isFinite = x.isFinite := by
  by_cases hfg : f = g
  · subst hfg; rw [self_iff.mp h]
  · rw [← ((iff_of_std hf hg hfg).mp h).isFinite_eq]
    exact FloatV.isFinite_cvt_of_wider hf.regular hg hw x

end FloatRel

namespace ConvRel

open Math.Float.Format

variable {P : Platform}

/-- The per-pair widening lemmas below: `FloatRel.eq_cvt_of_wider` and `FloatV.cvt_of_wider`. -/
private theorem of_wider {f g : Format} (hf : f.Regular) (hg : g.Regular) (hw : Wider f g)
    (hg8 : ¬ g.F8) {x : FloatV f} {r : FloatV g} (hx : x.isFinite = true)
    (h : FloatRel P f g x r) (hfg : f ≠ g := by decide) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal := by
  have e := FloatRel.eq_cvt_of_wider hf hg hw hfg hg8 hx h
  have c := FloatV.cvt_of_wider hf hg hw hx
  exact ⟨e, by rw [e]; exact c.1, by rw [e]; exact c.2.1, by rw [e]; exact c.2.2⟩

/-- `x.to(tl.float32)` of a `float16`: `cvt x`, exact on a finite value. -/
theorem f16_f32 {x : F16} {r : F32} (h : ConvRel P .f16 .f32 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider binary16_regular binary32_regular wider_binary16_binary32 binary32_std.not_f8 hx h

/-- `x.to(tl.float32)` of a `bfloat16`: `cvt x`, exact on a finite value. -/
theorem bf16_f32 {x : BF16} {r : F32} (h : ConvRel P .bf16 .f32 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider bfloat16_regular binary32_regular wider_bfloat16_binary32 binary32_std.not_f8 hx h

/-- `x.to(tl.float64)` of a `float32`: `cvt x`, exact on a finite value. -/
theorem f32_f64 {x : F32} {r : F64} (h : ConvRel P .f32 .f64 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider binary32_regular binary64_regular wider_binary32_binary64 binary64_std.not_f8 hx h

/-- `x.to(tl.float64)` of a `float16`: `cvt x`, exact on a finite value. -/
theorem f16_f64 {x : F16} {r : F64} (h : ConvRel P .f16 .f64 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider binary16_regular binary64_regular wider_binary16_binary64 binary64_std.not_f8 hx h

/-- `x.to(tl.float64)` of a `bfloat16`: `cvt x`, exact on a finite value. -/
theorem bf16_f64 {x : BF16} {r : F64} (h : ConvRel P .bf16 .f64 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider bfloat16_regular binary64_regular wider_bfloat16_binary64 binary64_std.not_f8 hx h

/-- `x.to(tl.float16)` of a `float8e5`: `cvt x`, exact on a finite value. -/
theorem f8e5_f16 {x : F8E5} {r : F16} (h : ConvRel P .f8e5 .f16 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider e5m2_regular binary16_regular wider_e5m2_binary16 binary16_std.not_f8 hx h

/-- `x.to(tl.bfloat16)` of a `float8e5`: `cvt x`, exact on a finite value. -/
theorem f8e5_bf16 {x : F8E5} {r : BF16} (h : ConvRel P .f8e5 .bf16 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider e5m2_regular bfloat16_regular wider_e5m2_bfloat16 bfloat16_std.not_f8 hx h

/-- `x.to(tl.float32)` of a `float8e5`: `cvt x`, exact on a finite value. -/
theorem f8e5_f32 {x : F8E5} {r : F32} (h : ConvRel P .f8e5 .f32 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider e5m2_regular binary32_regular wider_e5m2_binary32 binary32_std.not_f8 hx h

/-- `x.to(tl.float16)` of a `float8e4nv`: `cvt x`, exact on a finite value. -/
theorem f8e4nv_f16 {x : F8E4} {r : F16} (h : ConvRel P .f8e4nv .f16 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider e4m3_regular binary16_regular wider_e4m3_binary16 binary16_std.not_f8 hx h

/-- `x.to(tl.bfloat16)` of a `float8e4nv`: `cvt x`, exact on a finite value. -/
theorem f8e4nv_bf16 {x : F8E4} {r : BF16} (h : ConvRel P .f8e4nv .bf16 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider e4m3_regular bfloat16_regular wider_e4m3_bfloat16 bfloat16_std.not_f8 hx h

/-- `x.to(tl.float32)` of a `float8e4nv`: `cvt x`, exact on a finite value. -/
theorem f8e4nv_f32 {x : F8E4} {r : F32} (h : ConvRel P .f8e4nv .f32 x r) (hx : x.isFinite = true) :
    r = FloatV.cvt x ∧ r.isFinite = true ∧ r.toRat = x.toRat ∧ r.toReal = x.toReal :=
  of_wider e4m3_regular binary32_regular wider_e4m3_binary32 binary32_std.not_f8 hx h

/-- `x.to(tl.bfloat16)` of a `float16` returns exactly `cvt x`, up to NaN payload. -/
theorem f16_bf16_iff {x : F16} {r : BF16} : ConvRel P .f16 .bf16 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary16_std bfloat16_std (by decide)
/-- `x.to(tl.float32)` of a `float16` returns exactly `cvt x`, up to NaN payload. -/
theorem f16_f32_iff {x : F16} {r : F32} : ConvRel P .f16 .f32 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary16_std binary32_std (by decide)
/-- `x.to(tl.float64)` of a `float16` returns exactly `cvt x`, up to NaN payload. -/
theorem f16_f64_iff {x : F16} {r : F64} : ConvRel P .f16 .f64 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary16_std binary64_std (by decide)
/-- `x.to(tl.float16)` of a `bfloat16` returns exactly `cvt x`, up to NaN payload. -/
theorem bf16_f16_iff {x : BF16} {r : F16} : ConvRel P .bf16 .f16 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std bfloat16_std binary16_std (by decide)
/-- `x.to(tl.float32)` of a `bfloat16` returns exactly `cvt x`, up to NaN payload. -/
theorem bf16_f32_iff {x : BF16} {r : F32} : ConvRel P .bf16 .f32 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std bfloat16_std binary32_std (by decide)
/-- `x.to(tl.float64)` of a `bfloat16` returns exactly `cvt x`, up to NaN payload. -/
theorem bf16_f64_iff {x : BF16} {r : F64} : ConvRel P .bf16 .f64 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std bfloat16_std binary64_std (by decide)
/-- `x.to(tl.float16)` of a `float32` returns exactly `cvt x`, up to NaN payload. -/
theorem f32_f16_iff {x : F32} {r : F16} : ConvRel P .f32 .f16 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary32_std binary16_std (by decide)
/-- `x.to(tl.bfloat16)` of a `float32` returns exactly `cvt x`, up to NaN payload. -/
theorem f32_bf16_iff {x : F32} {r : BF16} : ConvRel P .f32 .bf16 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary32_std bfloat16_std (by decide)
/-- `x.to(tl.float64)` of a `float32` returns exactly `cvt x`, up to NaN payload. -/
theorem f32_f64_iff {x : F32} {r : F64} : ConvRel P .f32 .f64 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary32_std binary64_std (by decide)
/-- `x.to(tl.float16)` of a `float64` returns exactly `cvt x`, up to NaN payload. -/
theorem f64_f16_iff {x : F64} {r : F16} : ConvRel P .f64 .f16 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary64_std binary16_std (by decide)
/-- `x.to(tl.bfloat16)` of a `float64` returns exactly `cvt x`, up to NaN payload. -/
theorem f64_bf16_iff {x : F64} {r : BF16} : ConvRel P .f64 .bf16 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary64_std bfloat16_std (by decide)
/-- `x.to(tl.float32)` of a `float64` returns exactly `cvt x`, up to NaN payload. -/
theorem f64_f32_iff {x : F64} {r : F32} : ConvRel P .f64 .f32 x r ↔ FloatV.NaNEq (FloatV.cvt x) r :=
  FloatRel.iff_of_std binary64_std binary32_std (by decide)

/-! `cvt x` is a result `x.to(...)` allows (every ordered pair of distinct standard dtypes, and fp8
into `float16`, `bfloat16`, `float32`). -/

/-- `cvt x` is a result of `x.to(tl.bfloat16)` of a `float16`. -/
theorem cvt_f16_bf16 (x : F16) : ConvRel P .f16 .bf16 x (FloatV.cvt x : BF16) :=
  FloatRel.cvt_of_std binary16_std bfloat16_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float32)` of a `float16`. -/
theorem cvt_f16_f32 (x : F16) : ConvRel P .f16 .f32 x (FloatV.cvt x : F32) :=
  FloatRel.cvt_of_std binary16_std binary32_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float64)` of a `float16`. -/
theorem cvt_f16_f64 (x : F16) : ConvRel P .f16 .f64 x (FloatV.cvt x : F64) :=
  FloatRel.cvt_of_std binary16_std binary64_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float16)` of a `bfloat16`. -/
theorem cvt_bf16_f16 (x : BF16) : ConvRel P .bf16 .f16 x (FloatV.cvt x : F16) :=
  FloatRel.cvt_of_std bfloat16_std binary16_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float32)` of a `bfloat16`. -/
theorem cvt_bf16_f32 (x : BF16) : ConvRel P .bf16 .f32 x (FloatV.cvt x : F32) :=
  FloatRel.cvt_of_std bfloat16_std binary32_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float64)` of a `bfloat16`. -/
theorem cvt_bf16_f64 (x : BF16) : ConvRel P .bf16 .f64 x (FloatV.cvt x : F64) :=
  FloatRel.cvt_of_std bfloat16_std binary64_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float16)` of a `float32`. -/
theorem cvt_f32_f16 (x : F32) : ConvRel P .f32 .f16 x (FloatV.cvt x : F16) :=
  FloatRel.cvt_of_std binary32_std binary16_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.bfloat16)` of a `float32`. -/
theorem cvt_f32_bf16 (x : F32) : ConvRel P .f32 .bf16 x (FloatV.cvt x : BF16) :=
  FloatRel.cvt_of_std binary32_std bfloat16_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float64)` of a `float32`. -/
theorem cvt_f32_f64 (x : F32) : ConvRel P .f32 .f64 x (FloatV.cvt x : F64) :=
  FloatRel.cvt_of_std binary32_std binary64_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float16)` of a `float64`. -/
theorem cvt_f64_f16 (x : F64) : ConvRel P .f64 .f16 x (FloatV.cvt x : F16) :=
  FloatRel.cvt_of_std binary64_std binary16_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.bfloat16)` of a `float64`. -/
theorem cvt_f64_bf16 (x : F64) : ConvRel P .f64 .bf16 x (FloatV.cvt x : BF16) :=
  FloatRel.cvt_of_std binary64_std bfloat16_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float32)` of a `float64`. -/
theorem cvt_f64_f32 (x : F64) : ConvRel P .f64 .f32 x (FloatV.cvt x : F32) :=
  FloatRel.cvt_of_std binary64_std binary32_std (by decide) x
/-- `cvt x` is a result of `x.to(tl.float16)` of a `float8e5`. -/
theorem cvt_f8e5_f16 (x : F8E5) : ConvRel P .f8e5 .f16 x (FloatV.cvt x : F16) :=
  FloatRel.cvt_of_f8 (Or.inl rfl) (Or.inl rfl) x
/-- `cvt x` is a result of `x.to(tl.bfloat16)` of a `float8e5`. -/
theorem cvt_f8e5_bf16 (x : F8E5) : ConvRel P .f8e5 .bf16 x (FloatV.cvt x : BF16) :=
  FloatRel.cvt_of_f8 (Or.inl rfl) (Or.inr (Or.inl rfl)) x
/-- `cvt x` is a result of `x.to(tl.float32)` of a `float8e5`. -/
theorem cvt_f8e5_f32 (x : F8E5) : ConvRel P .f8e5 .f32 x (FloatV.cvt x : F32) :=
  FloatRel.cvt_of_f8 (Or.inl rfl) (Or.inr (Or.inr rfl)) x
/-- `cvt x` is a result of `x.to(tl.float16)` of a `float8e4nv`. -/
theorem cvt_f8e4nv_f16 (x : F8E4) : ConvRel P .f8e4nv .f16 x (FloatV.cvt x : F16) :=
  FloatRel.cvt_of_f8 (Or.inr rfl) (Or.inl rfl) x
/-- `cvt x` is a result of `x.to(tl.bfloat16)` of a `float8e4nv`. -/
theorem cvt_f8e4nv_bf16 (x : F8E4) : ConvRel P .f8e4nv .bf16 x (FloatV.cvt x : BF16) :=
  FloatRel.cvt_of_f8 (Or.inr rfl) (Or.inr (Or.inl rfl)) x
/-- `cvt x` is a result of `x.to(tl.float32)` of a `float8e4nv`. -/
theorem cvt_f8e4nv_f32 (x : F8E4) : ConvRel P .f8e4nv .f32 x (FloatV.cvt x : F32) :=
  FloatRel.cvt_of_f8 (Or.inr rfl) (Or.inr (Or.inr rfl)) x

/-! A widening between standard dtypes keeps finiteness both ways (NaN and `±∞` too). -/

/-- `x.to(tl.float32)` of a `float16` is finite exactly when `x` is. -/
theorem isFinite_f16_f32 {x : F16} {r : F32} (h : ConvRel P .f16 .f32 x r) :
    r.isFinite = x.isFinite :=
  FloatRel.isFinite_eq_of_wider binary16_std binary32_std wider_binary16_binary32 h
/-- `x.to(tl.float32)` of a `bfloat16` is finite exactly when `x` is. -/
theorem isFinite_bf16_f32 {x : BF16} {r : F32} (h : ConvRel P .bf16 .f32 x r) :
    r.isFinite = x.isFinite :=
  FloatRel.isFinite_eq_of_wider bfloat16_std binary32_std wider_bfloat16_binary32 h
/-- `x.to(tl.float64)` of a `float32` is finite exactly when `x` is. -/
theorem isFinite_f32_f64 {x : F32} {r : F64} (h : ConvRel P .f32 .f64 x r) :
    r.isFinite = x.isFinite :=
  FloatRel.isFinite_eq_of_wider binary32_std binary64_std wider_binary32_binary64 h
/-- `x.to(tl.float64)` of a `float16` is finite exactly when `x` is. -/
theorem isFinite_f16_f64 {x : F16} {r : F64} (h : ConvRel P .f16 .f64 x r) :
    r.isFinite = x.isFinite :=
  FloatRel.isFinite_eq_of_wider binary16_std binary64_std wider_binary16_binary64 h
/-- `x.to(tl.float64)` of a `bfloat16` is finite exactly when `x` is. -/
theorem isFinite_bf16_f64 {x : BF16} {r : F64} (h : ConvRel P .bf16 .f64 x r) :
    r.isFinite = x.isFinite :=
  FloatRel.isFinite_eq_of_wider bfloat16_std binary64_std wider_bfloat16_binary64 h

end ConvRel

/-! ## `DType.val?` of the float dtypes -/

namespace DType

/-- The value of a `float16` element. -/
theorem val?_f16 {x : F16} {v : ℚ} : DType.val? .f16 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  simp only [DType.val?]; split_ifs with h <;> simp [h]
/-- The value of a `bfloat16` element. -/
theorem val?_bf16 {x : BF16} {v : ℚ} : DType.val? .bf16 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  simp only [DType.val?]; split_ifs with h <;> simp [h]
/-- The value of a `float32` element. -/
theorem val?_f32 {x : F32} {v : ℚ} : DType.val? .f32 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  simp only [DType.val?]; split_ifs with h <;> simp [h]
/-- The value of a `float64` element. -/
theorem val?_f64 {x : F64} {v : ℚ} : DType.val? .f64 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  simp only [DType.val?]; split_ifs with h <;> simp [h]
/-- The value of a `float8e5` element. -/
theorem val?_f8e5 {x : F8E5} {v : ℚ} : DType.val? .f8e5 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  simp only [DType.val?]; split_ifs with h <;> simp [h]
/-- The value of a `float8e4nv` element. -/
theorem val?_f8e4nv {x : F8E4} {v : ℚ} :
    DType.val? .f8e4nv x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  simp only [DType.val?]; split_ifs with h <;> simp [h]

/-- A `float16` `+0` (`tl.zeros`, `0.0`) has the value `0`. -/
@[simp] theorem val?_zero_f16 : DType.val? .f16 (FloatV.ofPat 0 : F16) = some 0 :=
  val?_f16.mpr ⟨FloatV.isFinite_zero_f16, FloatV.toRat_ofPat_zero⟩
/-- A `bfloat16` `+0` has the value `0`. -/
@[simp] theorem val?_zero_bf16 : DType.val? .bf16 (FloatV.ofPat 0 : BF16) = some 0 :=
  val?_bf16.mpr ⟨FloatV.isFinite_zero_bf16, FloatV.toRat_ofPat_zero⟩
/-- A `float32` `+0` has the value `0`. -/
@[simp] theorem val?_zero_f32 : DType.val? .f32 (FloatV.ofPat 0 : F32) = some 0 :=
  val?_f32.mpr ⟨FloatV.isFinite_zero_f32, FloatV.toRat_ofPat_zero⟩
/-- A `float64` `+0` has the value `0`. -/
@[simp] theorem val?_zero_f64 : DType.val? .f64 (FloatV.ofPat 0 : F64) = some 0 :=
  val?_f64.mpr ⟨FloatV.isFinite_zero_f64, FloatV.toRat_ofPat_zero⟩
/-- A `float8e5` `+0` has the value `0`. -/
@[simp] theorem val?_zero_f8e5 : DType.val? .f8e5 (FloatV.ofPat 0 : F8E5) = some 0 :=
  val?_f8e5.mpr ⟨FloatV.isFinite_zero_f8e5, FloatV.toRat_ofPat_zero⟩
/-- A `float8e4nv` `+0` has the value `0`. -/
@[simp] theorem val?_zero_f8e4nv : DType.val? .f8e4nv (FloatV.ofPat 0 : F8E4) = some 0 :=
  val?_f8e4nv.mpr ⟨FloatV.isFinite_zero_f8e4, FloatV.toRat_ofPat_zero⟩

end DType

/-! ## The maximum of approximations -/

section MaxApprox

variable {α : Type*} [AddCommGroup α] [LinearOrder α] [IsOrderedAddMonoid α]

/-- **The maximum of two approximations approximates the maximum**: from `|a − A| ≤ δ` and
`|b − B| ≤ δ`, `|max a b − max A B| ≤ δ`. -/
theorem abs_max_sub_max_le_of {a b A B δ : α} (ha : |a - A| ≤ δ) (hb : |b - B| ≤ δ) :
    |max a b - max A B| ≤ δ :=
  (abs_max_sub_max_le_max a b A B).trans (max_le ha hb)

/-- The same for the minimum: `|min a b − min A B| ≤ δ`. -/
theorem abs_min_sub_min_le_of {a b A B δ : α} (ha : |a - A| ≤ δ) (hb : |b - B| ≤ δ) :
    |min a b - min A B| ≤ δ :=
  (abs_min_sub_min_le_max a b A B).trans (max_le ha hb)

end MaxApprox

namespace FloatV

variable {f : Format}

/-- **The reference `max` of two approximations** (operands not NaN): within `δ` of `max A B`. -/
theorem max_approx {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) {A B δ : ℝ}
    (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) : |(max x y).toReal - Max.max A B| ≤ δ := by
  have e : (max x y).toReal = Max.max x.toReal y.toReal := by
    unfold toReal; rw [toRat_max hx hy, Rat.cast_max]
  rw [e]; exact abs_max_sub_max_le_of ha hb

/-- **The reference `min` of two approximations** (operands not NaN): within `δ` of `min A B`. -/
theorem min_approx {x y : FloatV f} (hx : x.isNaN = false) (hy : y.isNaN = false) {A B δ : ℝ}
    (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) : |(min x y).toReal - Min.min A B| ≤ δ := by
  have e : (min x y).toReal = Min.min x.toReal y.toReal := by
    unfold toReal; rw [toRat_min hx hy, Rat.cast_min]
  rw [e]; exact abs_min_sub_min_le_of ha hb

/-- **What `tl.maximum` gives has the larger value** (operands not NaN). -/
theorem MaxRel.toRat_eq_max {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toRat = Max.max x.toRat y.toRat := by
  have hmem := h.mem hx hy
  have hr := isNaN_of_mem hx hy hmem
  obtain ⟨h1, h2⟩ := h.le hx hy
  have a := (le_iff hx hr).mp h1
  have b := (le_iff hy hr).mp h2
  rcases hmem with rfl | rfl
  · exact (max_eq_left b).symm
  · exact (max_eq_right a).symm

/-- **What `tl.minimum` gives has the smaller value** (operands not NaN). -/
theorem MaxRel.toRat_eq_min {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toRat = Min.min x.toRat y.toRat := by
  have hmem := h.mem hx hy
  have hr := isNaN_of_mem hx hy hmem
  obtain ⟨h1, h2⟩ := h.ge hx hy
  have a := (le_iff hr hx).mp h1
  have b := (le_iff hr hy).mp h2
  rcases hmem with rfl | rfl
  · exact (min_eq_left b).symm
  · exact (min_eq_right a).symm

/-- `MaxRel.toRat_eq_max` in `ℝ`. -/
theorem MaxRel.toReal_eq_max {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toReal = Max.max x.toReal y.toReal := by
  unfold toReal; rw [h.toRat_eq_max hx hy, Rat.cast_max]

/-- `MaxRel.toRat_eq_min` in `ℝ`. -/
theorem MaxRel.toReal_eq_min {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) : r.toReal = Min.min x.toReal y.toReal := by
  unfold toReal; rw [h.toRat_eq_min hx hy, Rat.cast_min]

/-- **`tl.maximum` of two approximations**: from `|x − A| ≤ δ`, `|y − B| ≤ δ` (operands not NaN),
any result is within `δ` of `max A B`. -/
theorem MaxRel.max_approx {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℝ} (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) :
    |r.toReal - Max.max A B| ≤ δ := by
  rw [h.toReal_eq_max hx hy]; exact abs_max_sub_max_le_of ha hb

/-- `MaxRel.max_approx` in `ℚ`. -/
theorem MaxRel.max_approx_rat {x y r : FloatV f} (h : MaxRel true x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℚ} (ha : |x.toRat - A| ≤ δ) (hb : |y.toRat - B| ≤ δ) :
    |r.toRat - Max.max A B| ≤ δ := by
  rw [h.toRat_eq_max hx hy]; exact abs_max_sub_max_le_of ha hb

/-- **`tl.minimum` of two approximations**: any result is within `δ` of `min A B`. -/
theorem MaxRel.min_approx {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℝ} (ha : |x.toReal - A| ≤ δ) (hb : |y.toReal - B| ≤ δ) :
    |r.toReal - Min.min A B| ≤ δ := by
  rw [h.toReal_eq_min hx hy]; exact abs_min_sub_min_le_of ha hb

/-- `MaxRel.min_approx` in `ℚ`. -/
theorem MaxRel.min_approx_rat {x y r : FloatV f} (h : MaxRel false x y r) (hx : x.isNaN = false)
    (hy : y.isNaN = false) {A B δ : ℚ} (ha : |x.toRat - A| ≤ δ) (hb : |y.toRat - B| ≤ δ) :
    |r.toRat - Min.min A B| ≤ δ := by
  rw [h.toRat_eq_min hx hy]; exact abs_min_sub_min_le_of ha hb

end FloatV

/-- **A `tl.max` reduction (any tree) by value**: the result is one of the leaves, and no leaf has
a larger real value (leaves not NaN). -/
theorem Tree.eval_maxRel_toReal {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel true) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false) :
    (∃ i ∈ t.leaves, r = x i) ∧ ∀ i ∈ t.leaves, (x i).toReal ≤ r.toReal := by
  obtain ⟨hmem, hle⟩ := Tree.eval_maxRel h hx
  obtain ⟨i0, hi0, rfl⟩ := hmem
  refine ⟨⟨i0, hi0, rfl⟩, fun i hi => ?_⟩
  have := hle i hi
  simp only [ite_true] at this
  unfold FloatV.toReal
  exact_mod_cast (FloatV.le_iff (hx i hi) (hx i0 hi0)).mp this

/-- **A `tl.min` reduction (any tree) by value**: one of the leaves, and no leaf is smaller. -/
theorem Tree.eval_minRel_toReal {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel false) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false) :
    (∃ i ∈ t.leaves, r = x i) ∧ ∀ i ∈ t.leaves, r.toReal ≤ (x i).toReal := by
  obtain ⟨hmem, hle⟩ := Tree.eval_maxRel h hx
  obtain ⟨i0, hi0, rfl⟩ := hmem
  refine ⟨⟨i0, hi0, rfl⟩, fun i hi => ?_⟩
  have := hle i hi
  simp only [Bool.false_eq_true, ite_false] at this
  unfold FloatV.toReal
  exact_mod_cast (FloatV.le_iff (hx i0 hi0) (hx i hi)).mp this

/-- **A `tl.max` reduction of approximations approximates the maximum.** If every leaf `x i` is
within `δ` of `A i`, and `M` is the largest `A i` over the leaves (at least each, equal to one),
the result is within `δ` of `M`. -/
theorem Tree.eval_maxRel_approx {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel true) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false)
    {A : Nat → ℝ} {M δ : ℝ} (hA : ∀ i ∈ t.leaves, |(x i).toReal - A i| ≤ δ)
    (hM : ∀ i ∈ t.leaves, A i ≤ M) (hM' : ∃ i ∈ t.leaves, A i = M) :
    |r.toReal - M| ≤ δ := by
  obtain ⟨⟨i0, hi0, rfl⟩, hle⟩ := Tree.eval_maxRel_toReal h hx
  obtain ⟨j, hj, rfl⟩ := hM'
  rw [abs_le]
  have h0 := abs_le.mp (hA i0 hi0)
  have h1 := abs_le.mp (hA j hj)
  have h2 := hM i0 hi0
  have h3 := hle j hj
  constructor <;> linarith [h0.1, h0.2, h1.1, h1.2]

/-- **A `tl.min` reduction of approximations approximates the minimum**: as
`Tree.eval_maxRel_approx`, with `M` the smallest `A i`. -/
theorem Tree.eval_minRel_approx {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    (h : t.Eval (FloatV.MaxRel false) x r) (hx : ∀ i ∈ t.leaves, (x i).isNaN = false)
    {A : Nat → ℝ} {M δ : ℝ} (hA : ∀ i ∈ t.leaves, |(x i).toReal - A i| ≤ δ)
    (hM : ∀ i ∈ t.leaves, M ≤ A i) (hM' : ∃ i ∈ t.leaves, A i = M) :
    |r.toReal - M| ≤ δ := by
  obtain ⟨⟨i0, hi0, rfl⟩, hle⟩ := Tree.eval_minRel_toReal h hx
  obtain ⟨j, hj, rfl⟩ := hM'
  rw [abs_le]
  have h0 := abs_le.mp (hA i0 hi0)
  have h1 := abs_le.mp (hA j hj)
  have h2 := hM i0 hi0
  have h3 := hle j hj
  constructor <;> linarith [h0.1, h0.2, h1.1, h1.2]

/-- **`Tree.eval_maxRel_approx` for a reduction of `n` elements** (`t.Covers n`): the result is
within `δ` of `(Finset.range n).sup' hn A`, the largest `A i` for `i < n`. -/
theorem Tree.eval_maxRel_approx_sup' {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    {n : Nat} (hc : t.Covers n) (hn : (Finset.range n).Nonempty)
    (h : t.Eval (FloatV.MaxRel true) x r) (hx : ∀ i < n, (x i).isNaN = false)
    {A : Nat → ℝ} {δ : ℝ} (hA : ∀ i < n, |(x i).toReal - A i| ≤ δ) :
    |r.toReal - (Finset.range n).sup' hn A| ≤ δ := by
  have hmem : ∀ i, i ∈ t.leaves ↔ i < n := fun i => by
    rw [hc.mem_iff, List.mem_range]
  refine Tree.eval_maxRel_approx h (fun i hi => hx i ((hmem i).mp hi))
    (fun i hi => hA i ((hmem i).mp hi))
    (fun i hi => Finset.le_sup' A (Finset.mem_range.mpr ((hmem i).mp hi))) ?_
  obtain ⟨j, hj, e⟩ := Finset.exists_mem_eq_sup' hn A
  exact ⟨j, (hmem j).mpr (Finset.mem_range.mp hj), e.symm⟩

/-- **`Tree.eval_minRel_approx` for a reduction of `n` elements** (`t.Covers n`): the result is
within `δ` of `(Finset.range n).inf' hn A`, the smallest `A i` for `i < n`. -/
theorem Tree.eval_minRel_approx_inf' {f : Format} {x : Nat → FloatV f} {t : Tree} {r : FloatV f}
    {n : Nat} (hc : t.Covers n) (hn : (Finset.range n).Nonempty)
    (h : t.Eval (FloatV.MaxRel false) x r) (hx : ∀ i < n, (x i).isNaN = false)
    {A : Nat → ℝ} {δ : ℝ} (hA : ∀ i < n, |(x i).toReal - A i| ≤ δ) :
    |r.toReal - (Finset.range n).inf' hn A| ≤ δ := by
  have hmem : ∀ i, i ∈ t.leaves ↔ i < n := fun i => by
    rw [hc.mem_iff, List.mem_range]
  refine Tree.eval_minRel_approx h (fun i hi => hx i ((hmem i).mp hi))
    (fun i hi => hA i ((hmem i).mp hi))
    (fun i hi => Finset.inf'_le A (Finset.mem_range.mpr ((hmem i).mp hi))) ?_
  obtain ⟨j, hj, e⟩ := Finset.exists_mem_eq_inf' hn A
  exact ⟨j, (hmem j).mpr (Finset.mem_range.mp hj), e.symm⟩

end TVA
