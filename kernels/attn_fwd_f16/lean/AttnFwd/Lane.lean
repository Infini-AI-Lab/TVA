import AttnFwd.Real

/-!
# `attn_fwd_f16`: facts about single operations

* the IEEE operations on `float32` with one rounding's bound, in `ℝ` (`f32_mul`, `f32_sub`,
  `f32_add`, `f32_addRel`, `f32_mulAdd`), and the rounding to `float16` (`f16_cvt`);
* `exp2_acc`: `Rules.Num.Exp2F32` at a finite input;
* `f16_tiny`: a finite `float16` below `2^-24` in magnitude is zero.
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

theorem toReal_def {f : Format} (x : FloatV f) : x.toReal = (x.toRat : ℝ) := rfl

theorem maxVal32 : (Format.binary32.maxVal : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [Format.binary32_maxVal]; push_cast; ring

theorem maxVal16 : (Format.binary16.maxVal : ℝ) = 65504 := by
  rw [Format.binary16_maxVal]; norm_num

theorem u32_eq : Format.binary32.u = 2 ^ (-24 : ℤ) := Format.binary32_u

theorem eta32_eq : Format.binary32.eta = 2 ^ (-150 : ℤ) := Format.binary32_eta

theorem abs_le_maxVal32 {x : F32} (hx : x.isFinite = true) : |x.toReal| ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
  have := FloatV.abs_toRat_le_maxVal Format.binary32_regular hx
  rw [← maxVal32, toReal_def]; exact_mod_cast this

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

/-- **`x.to(float16)`** of a `float32` within `float16`'s range. -/
theorem f16_cvt {x : F32} {r : F16} (h : FloatV.NaNEq (FloatV.cvt x) r) (hx : x.isFinite = true)
    (hroom : |x.toReal| ≤ 65504) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ |x.toReal| / 2 ^ 11 + 1 / 2 ^ 25 := by
  have hroom' : |x.toRat| ≤ Format.binary16.maxVal := by
    have : ((|x.toRat| : ℚ) : ℝ) ≤ (Format.binary16.maxVal : ℝ) := by
      rw [maxVal16]; push_cast; rw [← toReal_def]; exact hroom
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.cvt_err (g := Format.binary16) Format.binary16_regular hx hroom'
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  refine ⟨hf, ?_⟩
  rw [Format.binary16_u, Format.binary16_eta] at he
  simp only [toReal_def]
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  norm_num at this ⊢
  linarith [abs_nonneg (x.toRat : ℝ)]

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

/-- **A finite `float16` below `2^-24` in magnitude is zero**: `2^-24` is `float16`'s smallest
positive value. -/
theorem f16_tiny {x : F16} (h : |x.toRat| < 1 / 2 ^ 24) : x.toRat = 0 := by
  have ha := Format.abs_toRat (f := Format.binary16) x.pat
  have hm1 : Format.binary16.magVal 1 = 2 ^ (-24 : ℤ) := by
    rw [Format.magVal_eq]; norm_num [Format.binary16, Format.emin, Format.bias]
  unfold FloatV.toRat at h ⊢
  by_cases hz : Format.binary16.mag x.pat = 0
  · rw [Format.toRat_eq, hz, Format.magVal_zero]; simp
  · exfalso
    have hle := Format.magVal_le_magVal (f := Format.binary16) (show 1 ≤ Format.binary16.mag x.pat by omega)
    rw [hm1] at hle
    rw [ha] at h
    norm_num at hle h
    linarith

end Kernels.AttnFwd
