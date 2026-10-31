import AttnFwdE5m2.Real

/-!
# `attn_fwd_e5m2`: facts about single operations

* the IEEE operations on `float32` with one rounding's bound, in `ℝ` (`f32_mul`, `f32_sub`,
  `f32_add`, `f32_addRel`, `f32_mulAdd`);
* the conversion to `float8e5` (`Rules.Num.CvtSatF32E5m2`): one rounding within range (`f8_cvt`),
  and the clamped value beyond it (`f8_cvt_sat`);
* `exp2_acc`: `Rules.Num.Exp2F32` at a finite input;
* `f8_tiny`: a finite `float8e5` below `2^-16` in magnitude is zero.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

theorem toReal_def {f : Format} (x : FloatV f) : x.toReal = (x.toRat : ℝ) := rfl

theorem maxVal32 : (Format.binary32.maxVal : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [Format.binary32_maxVal]; push_cast; ring

theorem maxVal8 : (Format.e5m2.maxVal : ℝ) = 57344 := by
  rw [Format.e5m2_maxVal]; norm_num

theorem u32_eq : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u

theorem eta32_eq : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta

theorem abs_le_maxVal32 {x : F32} (hx : x.isFinite = true) : |x.toReal| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
  have := FloatV.abs_toRat_le_maxVal Format.binary32_regular hx
  rw [← maxVal32, toReal_def]; exact_mod_cast this

/-- A finite `float8e5` is at most `57344` in magnitude. -/
theorem abs_le_maxVal8 {x : F8E5} (hx : x.isFinite = true) : |x.toReal| ≤ 57344 := by
  have := FloatV.abs_toRat_le_maxVal Format.e5m2_regular hx
  rw [← maxVal8, toReal_def]; exact_mod_cast this

/-- **`x * y`** in `float32`, the result up to NaN payload. -/
theorem f32_mul {x y r : F32} (h : FloatV.NaNEq (FloatV.mul x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hroom : |x.toReal * y.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - x.toReal * y.toReal| ≤ |x.toReal * y.toReal| / 2 ^ 24 + 1 / 2 ^ 150 := by
  have hroom' : |x.toRat * y.toRat| ≤ Format.binary32.maxVal := by
    have : ((|x.toRat * y.toRat| : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; push_cast; rw [← toReal_def, ← toReal_def]; linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.mul_err Format.binary32_regular hx hy hroom'
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith [abs_nonneg ((x.toRat : ℝ) * y.toRat)]

/-- **`x − y`** in `float32`. -/
theorem f32_sub {x y r : F32} (h : FloatV.NaNEq (FloatV.sub x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hroom : |x.toReal - y.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - (x.toReal - y.toReal)| ≤ |x.toReal - y.toReal| / 2 ^ 24 + 1 / 2 ^ 150 := by
  have hroom' : |x.toRat - y.toRat| ≤ Format.binary32.maxVal := by
    have : ((|x.toRat - y.toRat| : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; push_cast; rw [← toReal_def, ← toReal_def]; linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.sub_err Format.binary32_regular hx hy hroom'
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith [abs_nonneg ((x.toRat : ℝ) - y.toRat)]

/-- **`x + y`** in `float32`. -/
theorem f32_add {x y r : F32} (h : FloatV.NaNEq (FloatV.add x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hroom : |x.toReal + y.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - (x.toReal + y.toReal)| ≤ |x.toReal + y.toReal| / 2 ^ 24 + 1 / 2 ^ 150 := by
  have hroom' : |x.toRat + y.toRat| ≤ Format.binary32.maxVal := by
    have : ((|x.toRat + y.toRat| : ℚ) : ℝ) ≤ (Format.binary32.maxVal : ℝ) := by
      rw [maxVal32]; push_cast; rw [← toReal_def, ← toReal_def]; linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.add_err Format.binary32_regular hx hy hroom'
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith [abs_nonneg ((x.toRat : ℝ) + y.toRat)]

/-- **A contractible addition** (`AddRel`) in `float32`. -/
theorem f32_addRel {fusion : Bool} {x y r : F32} (h : AddRel fusion x y r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hroom : |x.toReal| + |y.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - (x.toReal + y.toReal)| ≤
      3 * (|x.toReal| + |y.toReal|) / 2 ^ 24 + 4 / 2 ^ 150 := by
  have hroom' : |x.toRat| + |y.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have : ((|x.toRat| + |y.toRat| : ℚ) : ℝ) ≤ ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      rw [eta32_eq]; push_cast; rw [maxVal32, ← toReal_def, ← toReal_def]; norm_num; linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.AddRel.err Format.binary32_regular (by decide) (by rw [u32_eq]; norm_num) h hx hy hroom'
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith

/-- **`a * b + c`** (`tl.mul_add`) in `float32`. -/
theorem f32_mulAdd {fusion : Bool} {a b c r : F32}
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul a b) c) r ∨ (fusion = true ∧ FloatV.NaNEq (FloatV.fma a b c) r))
    (ha : a.isFinite = true) (hb : b.isFinite = true) (hc : c.isFinite = true)
    (hroom : |a.toReal * b.toReal| + |c.toReal| ≤ 2 ^ 120) :
    r.isFinite = true ∧ |r.toReal - (a.toReal * b.toReal + c.toReal)| ≤
      3 * (|a.toReal * b.toReal| + |c.toReal|) / 2 ^ 24 + 4 / 2 ^ 150 := by
  have hroom' : |a.toRat * b.toRat| + |c.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have : ((|a.toRat * b.toRat| + |c.toRat| : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      rw [eta32_eq]; push_cast; rw [maxVal32, ← toReal_def, ← toReal_def, ← toReal_def]; norm_num
      have := abs_mul a.toReal b.toReal
      linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.mulAdd_err Format.binary32_regular (by rw [u32_eq]; norm_num) h ha hb hc hroom'
  refine ⟨hf, ?_⟩
  rw [u32_eq, eta32_eq] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  have := abs_mul (a.toRat : ℝ) (b.toRat : ℝ)
  linarith

/-- **`x.to(float8e5)`** of a finite `float32` (`Rules.Num.CvtSatF32E5m2`): finite, and within one
rounding (`2^-3` relative, `2^-17` absolute) of the clamped value `max(−57344, min(x, 57344))`. -/
theorem f8_cvt_sat {P : Platform} (hcvt : Rules.Num.CvtSatF32E5m2 P) {x : F32} {r : F8E5}
    (h : P.cvt Format.binary32 Format.e5m2 .rn x r) (hx : x.isFinite = true) :
    r.isFinite = true ∧
      |r.toReal - max (-57344) (min x.toReal 57344)| ≤ |max (-57344) (min x.toReal 57344)| / 8 + 1 / 2 ^ 17 := by
  have hne := hcvt x r h
  obtain ⟨hf, he⟩ := FloatV.cvtSat_rn_err (g := Format.e5m2) Format.e5m2_regular hx
  rw [hne.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [Format.e5m2_u, Format.e5m2_eta, Format.e5m2_maxVal] at he
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  simp only [toReal_def]
  norm_num at this ⊢
  linarith

/-- **`x.to(float8e5)`** of a `float32` within range: one rounding. -/
theorem f8_cvt {P : Platform} (hcvt : Rules.Num.CvtSatF32E5m2 P) {x : F32} {r : F8E5}
    (h : P.cvt Format.binary32 Format.e5m2 .rn x r) (hx : x.isFinite = true) (hroom : |x.toReal| ≤ 57344) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ |x.toReal| / 8 + 1 / 2 ^ 17 := by
  obtain ⟨hf, he⟩ := f8_cvt_sat hcvt h hx
  have hc : max (-57344) (min x.toReal 57344) = x.toReal := by
    rw [abs_le] at hroom
    rw [min_eq_left hroom.2, max_eq_right hroom.1]
  rw [hc] at he
  exact ⟨hf, he⟩

/-- Clamping to `[−57344, 57344]` moves a value no further from a point of that interval. -/
theorem clamp_err {x a : ℝ} (ha : |a| ≤ 57344) :
    |max (-57344) (min x 57344) - a| ≤ |x - a| ∧ |max (-57344) (min x 57344)| ≤ |x| := by
  have h := abs_le.mp ha
  constructor
  · rcases le_total x 57344 with h1 | h1
    · rw [min_eq_left h1]
      rcases le_total (-57344) x with h2 | h2
      · rw [max_eq_right h2]
      · rw [max_eq_left h2]
        rw [abs_of_nonpos (by linarith : (-57344 : ℝ) - a ≤ 0), abs_of_nonpos (by linarith : x - a ≤ 0)]
        linarith
    · rw [min_eq_right h1, max_eq_right (by norm_num)]
      rw [abs_of_nonneg (by linarith : (57344 : ℝ) - a ≥ 0), abs_of_nonneg (by linarith : x - a ≥ 0)]
      linarith
  · rcases le_total x 57344 with h1 | h1
    · rw [min_eq_left h1]
      rcases le_total (-57344) x with h2 | h2
      · rw [max_eq_right h2]
      · rw [max_eq_left h2, abs_of_nonpos (by linarith : x ≤ 0)]; norm_num; linarith
    · rw [min_eq_right h1, max_eq_right (by norm_num), abs_of_nonneg (by linarith : (0 : ℝ) ≤ x)]
      norm_num; linarith

/-- **`tl.math.exp2`** on `float32` (`Rules.Num.Exp2F32`), at a finite input `x ≤ 100`. -/
theorem exp2_acc {P : Platform} (h : Rules.Num.Exp2F32 P) {x r : F32}
    (hr : P.math .exp2 .binary32 x r) (hx : x.isFinite = true) (hx1 : x.toReal ≤ 100) :
    r.isFinite = true ∧ |r.toReal - (2 : ℝ) ^ x.toReal| ≤ (2 : ℝ) ^ x.toReal / 2 ^ 22 + 1 / 2 ^ 126 := by
  obtain ⟨g, hg, hacc⟩ := h
  have hg' : g = fun y => (2 : ℝ) ^ y := by
    have e : MathFn.ref .exp2 = some (fun y => (2 : ℝ) ^ y) := rfl
    rw [e] at hg; exact (Option.some.inj hg).symm
  subst hg'
  have hp := two_rpow_pos x.toReal
  have h100 : (2 : ℝ) ^ x.toReal ≤ 2 ^ (100 : ℝ) := two_rpow_le hx1
  have h100' : (2 : ℝ) ^ (100 : ℝ) = 2 ^ (100 : ℕ) := by
    rw [show (100 : ℝ) = ((100 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  rw [h100'] at h100
  have hroom : (1 + 2 ^ (-22 : ℤ)) * |(2 : ℝ) ^ x.toReal| + 2 ^ (-126 : ℤ) ≤ (Format.binary32.maxVal : ℝ) := by
    rw [maxVal32, abs_of_pos hp]
    have : (0 : ℝ) < 2 ^ (-22 : ℤ) := by positivity
    have : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
    have : (2 : ℝ) ^ (-126 : ℤ) ≤ 1 := by norm_num
    nlinarith
  obtain ⟨hf, he⟩ := hacc x r hr hx trivial trivial hroom
  refine ⟨hf, ?_⟩
  rw [abs_of_pos hp] at he
  norm_num at he ⊢
  linarith

/-- **A finite `float8e5` below `2^-16` in magnitude is zero**: `2^-16` is `float8e5`'s smallest
positive value. -/
theorem f8_tiny {x : F8E5} (h : |x.toRat| < 1 / 2 ^ 16) : x.toRat = 0 := by
  have ha := Format.abs_toRat (f := Format.e5m2) x.pat
  have hm1 : Format.e5m2.magVal 1 = 2 ^ (-16 : ℤ) := by
    rw [Format.magVal_eq]; norm_num [Format.e5m2, Format.emin, Format.bias]
  unfold FloatV.toRat at h ⊢
  by_cases hz : Format.e5m2.mag x.pat = 0
  · rw [Format.toRat_eq, hz, Format.magVal_zero]; simp
  · exfalso
    have hle := Format.magVal_le_magVal (f := Format.e5m2) (show 1 ≤ Format.e5m2.mag x.pat by omega)
    rw [hm1] at hle
    rw [ha] at h
    norm_num at hle h
    linarith

end Kernels.AttnFwdE5m2
