import TVA.Math.Float.Format.Convert
import Mathlib

/-!
# fp8 with `.satfinite`: never NaN, bounded

The fp8 conversions of `cvt.rn.satfinite.{e4m3x2,e5m2x2}.*` and of `__nv_cvt_*_to_fp8(…,
__NV_SATFINITE, …)` (and the `__nv_fp8_e4m3`/`__nv_fp8_e5m2` constructors) round to nearest and
clamp an overflow to the largest finite value:

* `cvt_e4m3_satfinite`, `cvt_e4m3_satfinite_le`: converting any non-NaN value (finite or `±∞`,
  from any format) into e4m3 with saturation gives a pattern that is not the NaN, of magnitude at
  most 448;
* `cvt_e5m2_satfinite`, `cvt_e5m2_satfinite_le`: into e5m2, a finite pattern of magnitude at most
  57344.

Helpers: `roundDiv_scaled_le` (the rounded significand is at most `2^(m+1)`), `emin_le_expOf`,
`e4m3_encode_sat`, `e5m2_encode_sat`, `e4m3_roundMag_sat`, `e5m2_roundMag_sat`. A NaN input stays a
NaN (`NaNRule`), and without `.satfinite` an overflow gives the e4m3 NaN or the e5m2 `∞`
(`Format.overflow`). Proof module (Mathlib).
-/

set_option autoImplicit false

namespace TVA.Math.Float.Format

open TVA.Math.Float

/-- Under `rn`, the rounded significand of a positive value is at most `2^(m+1)` (the carry
`encode` takes back). -/
theorem roundDiv_scaled_le (f : Format) (neg : Bool) (q : ℚ) (hq : 0 < q) :
    roundDiv .rn neg (f.scaled q).1 (f.scaled q).2 ≤ 2 ^ (f.m + 1) := by
  obtain ⟨hd, hv⟩ := f.scaled_val q hq
  obtain ⟨-, hlt⟩ := log2Floor_rat q hq
  set n := (f.scaled q).1
  set d := (f.scaled q).2
  set L := log2Floor q.num.natAbs q.den
  have hE : L ≤ f.expOf q := le_max_left _ _
  have h1 : (n : ℚ) / d < 2 ^ (f.m + 1) := by
    rw [hv]
    have hp : (0 : ℚ) < 2 ^ ((f.m : ℤ) - f.expOf q) := zpow_pos (by norm_num) _
    calc q * 2 ^ ((f.m : ℤ) - f.expOf q) < 2 ^ (L + 1) * 2 ^ ((f.m : ℤ) - f.expOf q) :=
          mul_lt_mul_of_pos_right hlt hp
      _ = 2 ^ (L + 1 + ((f.m : ℤ) - f.expOf q)) := by rw [← zpow_add₀ (by norm_num)]
      _ ≤ 2 ^ (((f.m + 1 : ℕ) : ℤ)) := zpow_le_zpow_right₀ (by norm_num) (by push_cast; omega)
      _ = 2 ^ (f.m + 1) := zpow_natCast _ _
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  have h2 : (n : ℚ) < 2 ^ (f.m + 1) * d := by rwa [div_lt_iff₀ hdq] at h1
  have h3 : n < 2 ^ (f.m + 1) * d := by exact_mod_cast h2
  have h4 : n / d < 2 ^ (f.m + 1) := Nat.div_lt_of_lt_mul (by rw [mul_comm]; exact h3)
  have := roundDiv_rn_le neg n d
  omega

/-- The clamped exponent is at least `emin`. -/
theorem emin_le_expOf (f : Format) (q : ℚ) : f.emin ≤ f.expOf q := le_max_right _ _

/-! ## e4m3 with `.satfinite`: never a NaN -/

/-- `encode` into e4m3 with saturation: a significand `≤ 16` at an exponent `≥ −6` gives a
magnitude below the sign bit that is not the NaN. -/
theorem e4m3_encode_sat (neg : Bool) (m : ℕ) (e : ℤ) (hm : m ≤ 16) (he : -6 ≤ e) :
    e4m3.encode .rn true neg m e < 128 ∧ e4m3.isNaN (e4m3.encode .rn true neg m e) = false := by
  rcases lt_or_ge e 9 with h | h
  · have key : ∀ ng : Bool, ∀ mm : Fin 17, ∀ k : Fin 15,
        e4m3.encode .rn true ng mm.val ((k.val : ℤ) - 6) < 128 ∧
          e4m3.isNaN (e4m3.encode .rn true ng mm.val ((k.val : ℤ) - 6)) = false := by
      decide +kernel
    have := key neg ⟨m, by omega⟩ ⟨(e + 6).toNat, by omega⟩
    have he' : (((e + 6).toNat : ℕ) : ℤ) - 6 = e := by omega
    simp only at this
    rwa [he'] at this
  · have hemax : e4m3.emax = 8 := rfl
    have hover : ∀ (m' : ℕ) (e' : ℤ), 9 ≤ e' → e4m3.over m' e' = true := by
      intro m' e' h'
      simp only [Format.over, show e4m3.special = .fn from rfl, hemax, Bool.or_eq_true, decide_eq_true_eq]
      left; omega
    have hof : e4m3.overflow .rn true neg = 126 := rfl
    unfold encode
    by_cases h16 : m = 16
    · subst h16
      simp only [show e4m3.m = 3 from rfl, show (2 : ℕ) ^ (3 + 1) = 16 from rfl, beq_self_eq_true, ↓reduceIte,
        hover _ (e + 1) (by omega), hof]
      decide
    · have hb : (m == 2 ^ (e4m3.m + 1)) = false := by
        simp only [show e4m3.m = 3 from rfl, show (2 : ℕ) ^ (3 + 1) = 16 from rfl, beq_eq_false_iff_ne]; exact h16
      simp only [hb, Bool.false_eq_true, ↓reduceIte, hover _ e h, hof]
      decide

/-- `encode` into e5m2 with saturation: a significand `≤ 8` at an exponent `≥ −14` gives a finite
magnitude below the sign bit. -/
theorem e5m2_encode_sat (neg : Bool) (m : ℕ) (e : ℤ) (hm : m ≤ 8) (he : -14 ≤ e) :
    e5m2.encode .rn true neg m e < 128 ∧ e5m2.isFinite (e5m2.encode .rn true neg m e) = true := by
  rcases lt_or_ge e 16 with h | h
  · have key : ∀ ng : Bool, ∀ mm : Fin 9, ∀ k : Fin 30,
        e5m2.encode .rn true ng mm.val ((k.val : ℤ) - 14) < 128 ∧
          e5m2.isFinite (e5m2.encode .rn true ng mm.val ((k.val : ℤ) - 14)) = true := by
      decide +kernel
    have := key neg ⟨m, by omega⟩ ⟨(e + 14).toNat, by omega⟩
    have he' : (((e + 14).toNat : ℕ) : ℤ) - 14 = e := by omega
    simp only at this
    rwa [he'] at this
  · have hemax : e5m2.emax = 15 := rfl
    have hover : ∀ (m' : ℕ) (e' : ℤ), 16 ≤ e' → e5m2.over m' e' = true := by
      intro m' e' h'
      simp only [Format.over, show e5m2.special = .ieee from rfl, hemax, decide_eq_true_eq]
      omega
    have hof : e5m2.overflow .rn true neg = 0x7b := rfl
    unfold encode
    by_cases h8 : m = 8
    · subst h8
      simp only [show e5m2.m = 2 from rfl, show (2 : ℕ) ^ (2 + 1) = 8 from rfl, beq_self_eq_true, ↓reduceIte,
        hover _ (e + 1) (by omega), hof]
      decide
    · have hb : (m == 2 ^ (e5m2.m + 1)) = false := by
        simp only [show e5m2.m = 2 from rfl, show (2 : ℕ) ^ (2 + 1) = 8 from rfl, beq_eq_false_iff_ne]; exact h8
      simp only [hb, Bool.false_eq_true, ↓reduceIte, hover _ e h, hof]
      decide

/-- Rounding into e4m3 with `.satfinite` (`rn`): never the NaN. -/
theorem e4m3_roundMag_sat (neg : Bool) (q : ℚ) :
    e4m3.roundMag .rn true neg q < 128 ∧ e4m3.isNaN (e4m3.roundMag .rn true neg q) = false := by
  rw [roundMag_eq]
  split_ifs with hq
  · decide
  · have hq' : 0 < q := not_le.mp hq
    have hb := roundDiv_scaled_le e4m3 neg q hq'
    rcases hs : e4m3.scaled q with ⟨n, d⟩
    rw [hs] at hb
    exact e4m3_encode_sat neg _ _ hb (emin_le_expOf e4m3 q)

/-- Rounding into e5m2 with `.satfinite` (`rn`): always finite. -/
theorem e5m2_roundMag_sat (neg : Bool) (q : ℚ) :
    e5m2.roundMag .rn true neg q < 128 ∧ e5m2.isFinite (e5m2.roundMag .rn true neg q) = true := by
  rw [roundMag_eq]
  split_ifs with hq
  · decide
  · have hq' : 0 < q := not_le.mp hq
    have hb := roundDiv_scaled_le e5m2 neg q hq'
    rcases hs : e5m2.scaled q with ⟨n, d⟩
    rw [hs] at hb
    exact e5m2_encode_sat neg _ _ hb (emin_le_expOf e5m2 q)

theorem e4m3_withSign_isNaN : ∀ s : Bool, ∀ mag : Fin 128,
    e4m3.isNaN (e4m3.withSign s mag.val) = e4m3.isNaN mag.val ∧ e4m3.withSign s mag.val < 256 := by
  decide +kernel

theorem e5m2_withSign_isFinite : ∀ s : Bool, ∀ mag : Fin 128,
    e5m2.isFinite (e5m2.withSign s mag.val) = e5m2.isFinite mag.val ∧ e5m2.withSign s mag.val < 256 := by
  decide +kernel

/-- **e4m3 with `.satfinite`**: converting any non-NaN value (to nearest, saturating) gives an
e4m3 pattern that is not the NaN (`cvt.rn.satfinite.e4m3x2.*`, `__nv_cvt_*_to_fp8(…, __NV_SATFINITE,
__NV_E4M3)`, the `__nv_fp8_e4m3` constructors). -/
theorem cvt_e4m3_satfinite (src : Format) (nan : NaNRule) (x : ℕ) (hx : src.isNaN x = false) :
    e4m3.isNaN (cvt src e4m3 .rn true nan x) = false ∧ cvt src e4m3 .rn true nan x < 256 := by
  by_cases hi : src.isInf x = true
  · rw [cvt_inf src e4m3 .rn true nan hx hi]
    simp only [↓reduceIte]
    have := e4m3_withSign_isNaN (src.sign x) ⟨e4m3.maxFinite, by decide⟩
    simp only at this
    rw [this.1]; exact ⟨by decide, this.2⟩
  · have hi' : src.isInf x = false := by simpa using hi
    by_cases hz : src.isZero x = true
    · rw [cvt_zero src e4m3 .rn true nan hx hi' hz]
      have := e4m3_withSign_isNaN (src.sign x) ⟨0, by decide⟩
      simp only at this
      rw [this.1]; exact ⟨by decide, this.2⟩
    · have hz' : src.isZero x = false := by simpa using hz
      rw [cvt_finite src e4m3 .rn true nan hx hi' hz']
      obtain ⟨hlt, hnan⟩ := e4m3_roundMag_sat (src.sign x) (src.magRat x)
      have := e4m3_withSign_isNaN (src.sign x) ⟨_, hlt⟩
      simp only at this
      rw [this.1]; exact ⟨hnan, this.2⟩

/-- **e5m2 with `.satfinite`**: converting any non-NaN value gives a finite e5m2 pattern. -/
theorem cvt_e5m2_satfinite (src : Format) (nan : NaNRule) (x : ℕ) (hx : src.isNaN x = false) :
    e5m2.isFinite (cvt src e5m2 .rn true nan x) = true ∧ cvt src e5m2 .rn true nan x < 256 := by
  by_cases hi : src.isInf x = true
  · rw [cvt_inf src e5m2 .rn true nan hx hi]
    simp only [↓reduceIte]
    have := e5m2_withSign_isFinite (src.sign x) ⟨e5m2.maxFinite, by decide⟩
    simp only at this
    rw [this.1]; exact ⟨by decide, this.2⟩
  · have hi' : src.isInf x = false := by simpa using hi
    by_cases hz : src.isZero x = true
    · rw [cvt_zero src e5m2 .rn true nan hx hi' hz]
      have := e5m2_withSign_isFinite (src.sign x) ⟨0, by decide⟩
      simp only at this
      rw [this.1]; exact ⟨by decide, this.2⟩
    · have hz' : src.isZero x = false := by simpa using hz
      rw [cvt_finite src e5m2 .rn true nan hx hi' hz']
      obtain ⟨hlt, hfin⟩ := e5m2_roundMag_sat (src.sign x) (src.magRat x)
      have := e5m2_withSign_isFinite (src.sign x) ⟨_, hlt⟩
      simp only at this
      rw [this.1]; exact ⟨hfin, this.2⟩

/-- Every non-NaN e4m3 pattern has magnitude at most 448. -/
theorem e4m3_le_448 : ∀ y : Fin 256, e4m3.isNaN y.val = false → |e4m3.toRat y.val| ≤ 448 := by
  decide +kernel

/-- Every finite e5m2 pattern has magnitude at most 57344. -/
theorem e5m2_le_57344 : ∀ y : Fin 256, e5m2.isFinite y.val = true → |e5m2.toRat y.val| ≤ 57344 := by
  decide +kernel

/-- **The e4m3 saturation bound**: a saturating conversion of a non-NaN value is a number of
magnitude at most 448. -/
theorem cvt_e4m3_satfinite_le (src : Format) (nan : NaNRule) (x : ℕ) (hx : src.isNaN x = false) :
    e4m3.isNaN (cvt src e4m3 .rn true nan x) = false ∧ |e4m3.toRat (cvt src e4m3 .rn true nan x)| ≤ 448 := by
  obtain ⟨h1, h2⟩ := cvt_e4m3_satfinite src nan x hx
  exact ⟨h1, e4m3_le_448 ⟨_, h2⟩ h1⟩

/-- **The e5m2 saturation bound**: a saturating conversion of a non-NaN value is finite, of
magnitude at most 57344. -/
theorem cvt_e5m2_satfinite_le (src : Format) (nan : NaNRule) (x : ℕ) (hx : src.isNaN x = false) :
    e5m2.isFinite (cvt src e5m2 .rn true nan x) = true ∧ |e5m2.toRat (cvt src e5m2 .rn true nan x)| ≤ 57344 := by
  obtain ⟨h1, h2⟩ := cvt_e5m2_satfinite src nan x hx
  exact ⟨h1, e5m2_le_57344 ⟨_, h2⟩ h1⟩

end TVA.Math.Float.Format
