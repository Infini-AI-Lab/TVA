import TVA.Math.Float.F32.Round

/-!
# One rounding per operation: `add_err`, `mul_err`, `fma_err` and their forms

Part of the `f32` error analysis (`TVA.Math.Float.F32.Error`); the definitions it reasons about are in
`TVA.Math.Float.F32.Bounds`. Split out of `F32/Bounds.lean`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

open Rat

/-! ### The operations: one rounding each -/

/-- [meaning] For finite operands whose exact sum is zero or normal and below overflow, the f32
sum `add` is finite and within `u·|x + y|` of the exact sum (`u = 2^-24`: one rounding to nearest).

**`add.f32`**, relative form. -/
theorem add_err (x y : Bits) (hx : isFinite x = true) (hy : isFinite y = true)
    (hr : toRat x + toRat y = 0 ∨ minNormal ≤ |toRat x + toRat y|)
    (hmax : |toRat x + toRat y| ≤ maxNormal) :
    isFinite (add x y) = true ∧ |toRat (add x y) - (toRat x + toRat y)| ≤ u * |toRat x + toRat y| := by
  obtain ⟨hxn, hxi⟩ := finite_not_nan_inf hx
  obtain ⟨hyn, hyi⟩ := finite_not_nan_inf hy
  unfold add
  simp only [hxn, hxi, hyn, hyi, Bool.false_or, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  split_ifs with hz hs
  · have hzz : isZero x = true ∧ isZero y = true := by simpa using hz
    rw [toRat_zero_of_isZero hzz.1, toRat_zero_of_isZero hzz.2, toRat_negZero]
    simp [isFinite_negZero]
  · have hzz : isZero x = true ∧ isZero y = true := by simpa using hz
    rw [toRat_zero_of_isZero hzz.1, toRat_zero_of_isZero hzz.2, toRat_posZero]
    simp [isFinite_posZero]
  · exact roundRat_err_u _ hr hmax

/-- `toRat x * toRat y` as a signed magnitude. -/
theorem toRat_mul_eq (x y : Bits) :
    toRat x * toRat y = if (sign x != sign y) then -(magRat x * magRat y) else magRat x * magRat y := by
  unfold toRat
  cases sign x <;> cases sign y <;> simp

/-- **`mul.f32`**: one rounding of the exact product. -/
theorem mul_err (x y : Bits) (hx : isFinite x = true) (hy : isFinite y = true)
    (hr : toRat x * toRat y = 0 ∨ minNormal ≤ |toRat x * toRat y|)
    (hmax : |toRat x * toRat y| ≤ maxNormal) :
    isFinite (mul x y) = true ∧ |toRat (mul x y) - toRat x * toRat y| ≤ u * |toRat x * toRat y| := by
  obtain ⟨hxn, hxi⟩ := finite_not_nan_inf hx
  obtain ⟨hyn, hyi⟩ := finite_not_nan_inf hy
  unfold mul
  simp only [hxn, hxi, hyn, hyi, Bool.false_or, Bool.false_eq_true, ↓reduceIte]
  have hz0 : (posZero).toNat < 2 ^ 31 := by decide
  split_ifs with hz
  · -- a zero factor: exact
    have hz' : isZero x = true ∨ isZero y = true := by simpa using hz
    have hprod : toRat x * toRat y = 0 := by
      rcases hz' with h | h
      · rw [toRat_zero_of_isZero h, zero_mul]
      · rw [toRat_zero_of_isZero h, mul_zero]
    rw [hprod, isFinite_withSign _ _ hz0, toRat_withSign _ _ hz0]
    refine ⟨isFinite_posZero, ?_⟩
    split_ifs <;> simp [magRat_posZero]
  · -- both nonzero: one rounding of the positive magnitude product
    have hxz : isZero x = false := by
      cases h : isZero x <;> simp_all
    have hyz : isZero y = false := by
      cases h : isZero y <;> simp_all
    set P := magRat x * magRat y with hP
    have hPpos : 0 < P := mul_pos (magRat_pos_of_not_zero hxz) (magRat_pos_of_not_zero hyz)
    have habs : |toRat x * toRat y| = P := by
      rw [abs_mul, abs_toRat, abs_toRat]
    rw [habs] at hr hmax
    have hlo : minNormal ≤ P := by
      rcases hr with h | h
      · exact absurd (show P = 0 by rw [← habs, h, abs_zero]) hPpos.ne'
      · exact h
    obtain ⟨hf, hs, herr⟩ := roundMag_spec P hPpos hmax
    have hlt := toNat_lt_of_sign_false hs
    have hr' : toRat x * toRat y = if (sign x != sign y) then -P else P := toRat_mul_eq x y
    rw [isFinite_withSign _ _ hlt]
    refine ⟨hf, ?_⟩
    rw [habs, toRat_withSign _ _ hlt, hr']
    have hu : |magRat (roundMag P) - P| ≤ u * P := by
      refine herr.trans ?_
      have hle := expOf_pow_le hlo
      have two_ne : (2 : ℚ) ≠ 0 := by norm_num
      calc (2 : ℚ) ^ (expOf P - 24) = 2 ^ (expOf P) * 2 ^ (-24 : ℤ) := by
            rw [show expOf P - 24 = expOf P + (-24 : ℤ) by ring, zpow_add₀ two_ne]
        _ ≤ P * 2 ^ (-24 : ℤ) := by gcongr
        _ = u * P := by unfold u; rw [mul_comm]; all_goals norm_num
    split_ifs
    · rw [show -magRat (roundMag P) - -P = -(magRat (roundMag P) - P) by ring, abs_neg]; exact hu
    · exact hu

/-- **`fma.rn.f32`**: a single rounding of `x·y + z`. -/
theorem fma_err (x y z : Bits) (hx : isFinite x = true) (hy : isFinite y = true) (hz : isFinite z = true)
    (hr : toRat x * toRat y + toRat z = 0 ∨ minNormal ≤ |toRat x * toRat y + toRat z|)
    (hmax : |toRat x * toRat y + toRat z| ≤ maxNormal) :
    isFinite (fma x y z) = true ∧
    |toRat (fma x y z) - (toRat x * toRat y + toRat z)| ≤ u * |toRat x * toRat y + toRat z| := by
  obtain ⟨hxn, hxi⟩ := finite_not_nan_inf hx
  obtain ⟨hyn, hyi⟩ := finite_not_nan_inf hy
  obtain ⟨hzn, hzi⟩ := finite_not_nan_inf hz
  unfold fma
  simp only [hxn, hxi, hyn, hyi, hzn, hzi, Bool.false_or, Bool.false_eq_true, ↓reduceIte]
  split_ifs with h0 hz0 hs
  · -- zero product, zero addend: ±0, exact
    have h0' : isZero x = true ∨ isZero y = true := by simpa using h0
    have hprod : toRat x * toRat y = 0 := by
      rcases h0' with h | h
      · rw [toRat_zero_of_isZero h, zero_mul]
      · rw [toRat_zero_of_isZero h, mul_zero]
    rw [hprod, toRat_zero_of_isZero hz0, toRat_negZero]; simp [isFinite_negZero]
  · have h0' : isZero x = true ∨ isZero y = true := by simpa using h0
    have hprod : toRat x * toRat y = 0 := by
      rcases h0' with h | h
      · rw [toRat_zero_of_isZero h, zero_mul]
      · rw [toRat_zero_of_isZero h, mul_zero]
    rw [hprod, toRat_zero_of_isZero hz0, toRat_posZero]; simp [isFinite_posZero]
  · -- zero product, nonzero addend: the addend, exact
    have h0' : isZero x = true ∨ isZero y = true := by simpa using h0
    have hprod : toRat x * toRat y = 0 := by
      rcases h0' with h | h
      · rw [toRat_zero_of_isZero h, zero_mul]
      · rw [toRat_zero_of_isZero h, mul_zero]
    rw [hprod, zero_add]; simp only [hz, sub_self, abs_zero, true_and]
    exact mul_nonneg u_pos.le (abs_nonneg _)
  · exact roundRat_err_u _ hr hmax

end TVA.Math.Float.F32
