import TVA.Proof.Float.Max

/-!
# Everyday facts about float values

Small facts about the definitions of `TVA.Base.Float` that kernel proofs need again and again:

* **Zeros.** `ofPat 0` is `+0` in every format: its pattern, value, sign (`pat_ofPat_zero`,
  `toRat_ofPat_zero`, `toReal_ofPat_zero`, `isZero_ofPat_zero`, `sign_ofPat_zero`, all `@[simp]`),
  finite (`isFinite_ofPat_zero`, and `isFinite_zero_f16` … `isFinite_zero_f8e4` per dtype).
  `tl.zeros` holds it (`tl.zeros_get_f16`, …), the constant of a zero Python number is it
  (`litF32_of_isZero`, …), and conversions keep it (`cvt_ofPat_zero`, `cvtWith_ofPat_zero`,
  `cvtSat_ofPat_zero`, `ofRat_zero`). `DType.ofBits_f16` … say that a float element read from
  bits is `ofPat` of them (not `@[simp]`: a claim may state its own facts about `ofBits`).
* **Bounds of a value.** A finite value is at most `maxVal` (`abs_toReal_le_maxVal`); every
  pattern, NaN and `±∞` included, is at most the top magnitude's value (`abs_toRat_le_magVal_top`,
  `abs_toRat_le_f16`); a nonzero value is at least the smallest subnormal `2η`
  (`two_eta_le_abs_toRat`). **Finiteness from magnitude**: a value at most `maxVal` in magnitude
  is finite (`isFinite_of_abs_toRat_le_maxVal`), and a non-finite one lies beyond `maxVal`
  (`maxVal_lt_abs_toRat_of_not_isFinite`).
* **`DType.val?`** of a float element is its value when it is finite (`DType.val?_f32`, …);
  `DType.exists_nan`: every float dtype has a NaN.

Exact widening conversions are in `TVA.Proof.Float.Widen`.
-/
namespace TVA

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

end FloatV

/-- The pattern `0` (`+0`) has the value `0`, in every format. -/
@[simp] theorem Format.toRat_zero (f : Format) : f.toRat 0 = 0 :=
  Format.toRat_of_isZero Format.isZero_zero

/-- The `float16` value with bit pattern `n` (`DType.ofBits`, as in a bit cast) is `ofPat n`; so
`DType.f16.ofBits 0`, the default element of a `float16` tensor, is `+0`. -/
theorem DType.ofBits_f16 (n : Nat) : DType.ofBits .f16 n = (FloatV.ofPat n : F16) := rfl
/-- `DType.ofBits .bf16 n` is `ofPat n`. -/
theorem DType.ofBits_bf16 (n : Nat) : DType.ofBits .bf16 n = (FloatV.ofPat n : BF16) := rfl
/-- `DType.ofBits .f32 n` is `ofPat n`. -/
theorem DType.ofBits_f32 (n : Nat) : DType.ofBits .f32 n = (FloatV.ofPat n : F32) := rfl
/-- `DType.ofBits .f64 n` is `ofPat n`. -/
theorem DType.ofBits_f64 (n : Nat) : DType.ofBits .f64 n = (FloatV.ofPat n : F64) := rfl
/-- `DType.ofBits .f8e5 n` is `ofPat n`. -/
theorem DType.ofBits_f8e5 (n : Nat) : DType.ofBits .f8e5 n = (FloatV.ofPat n : F8E5) := rfl
/-- `DType.ofBits .f8e4nv n` is `ofPat n`. -/
theorem DType.ofBits_f8e4nv (n : Nat) :
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

end FloatV

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

/-- Every float dtype has a NaN value (the canonical NaN). -/
theorem DType.exists_nan : (d : DType) → d.format?.isSome = true → ∃ z : d.Val, d.IsNaN z
  | .f16, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .bf16, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f32, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f64, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f8e5, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩
  | .f8e4nv, _ => ⟨FloatV.nan, show FloatV.isNaN _ = true by decide⟩

end TVA
