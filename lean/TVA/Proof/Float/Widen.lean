import TVA.Proof.Float.Facts

/-!
# Exact widening conversions and `x.to(…)`

A conversion into a format that holds every value of the source (`Format.Wider`) is exact. This
file states it for `FloatV.cvt` and for what `x.to(dtype)` may return (`ConvRel`, through
`FloatRel`, `TVA.Triton.Convert`):

* `FloatV.cvt_of_wider`: `cvt` of a finite value into a wider format is finite with the same
  value; per pair: `cvt_f16_f32_exact`, `cvt_bf16_f32_exact`, `cvt_f32_f64_exact`, …,
  `cvt_f8e4_f32_exact`. `isFinite_cvt_of_wider`: into a wider standard format, `cvt x` is finite
  exactly when `x` is; `cvt_negInf`: `−∞` stays `−∞`.
* `FloatRel.self_iff` (no conversion between equal formats), `FloatRel.iff_of_std` (between
  distinct standard formats, exactly `cvt x` up to NaN payload), `FloatRel.exact_of_wider` (every
  allowed result of a widening of a finite value is finite with the same value),
  `FloatRel.isFinite_eq_of_wider`, `FloatRel.negInf_of_std`, `FloatRel.cvt_of_std`,
  `FloatRel.cvt_of_f8` (`cvt x` is an allowed result).
* Per pair of dtypes, for `ConvRel`: `ConvRel.f16_f32` … `ConvRel.f8e4nv_f32` (a widening is
  `cvt x`, finite, same value), `ConvRel.f32_bf16_iff` … (every ordered pair of distinct standard
  dtypes: exactly `cvt x` up to NaN payload), `ConvRel.cvt_f32_f16` … (`cvt x` is an allowed
  result), `ConvRel.isFinite_f16_f32` … (a widening keeps finiteness both ways).

A conversion out of fp8 is `cvt x` on finite values only (a non-finite fp8 value may convert to
any value), so an fp8 widening keeps finiteness only one way.
-/
namespace TVA

namespace FloatV

variable {f : Format}

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

end FloatV

/-! ## Conversions as `semantic.cast` allows them -/

namespace FloatRel

open Format

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

open Format

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

end TVA
