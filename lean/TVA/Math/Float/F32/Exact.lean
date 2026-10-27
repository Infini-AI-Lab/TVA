import TVA.Math.Float.F32.AbsErr

/-!
# Rounding a representable value is exact

Part of the `f32` error analysis (`TVA.Math.Float.F32.Error`); the definitions it reasons about are in
`TVA.Math.Float.F32.Bounds`. Split out of `F32/Bounds.lean`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

/-! ### Rounding a representable value is exact -/

/-- `expOf` of a positive rational bracketed by consecutive powers of two. -/
theorem expOf_eq_of_bracket {q : ℚ} {E : ℤ} (hq : 0 < q) (hlo : (2 : ℚ) ^ E ≤ q) (hhi : q < 2 ^ (E + 1))
    (hE : -126 ≤ E) : expOf q = E := by
  obtain ⟨hL1, hL2⟩ := log2Floor_rat q hq
  have hLE : log2Floor q.num.natAbs q.den = E := by
    rcases lt_trichotomy (log2Floor q.num.natAbs q.den) E with h | h | h
    · exfalso
      have : (2 : ℚ) ^ (log2Floor q.num.natAbs q.den + 1) ≤ 2 ^ E :=
        zpow_le_zpow_right₀ (by norm_num) (by omega)
      linarith
    · exact h
    · exfalso
      have : (2 : ℚ) ^ (E + 1) ≤ 2 ^ (log2Floor q.num.natAbs q.den) :=
        zpow_le_zpow_right₀ (by norm_num) (by omega)
      linarith
  unfold expOf; rw [hLE, max_eq_left hE]

/-- `rne` of an exact integer fraction. -/
theorem rne_exact {num den : ℕ} (hd : 0 < den) (M : ℕ) (h : num = M * den) : rne num den = M := by
  unfold rne
  subst h
  rw [Nat.mul_div_cancel M hd, Nat.mul_mod_left M den]
  simp [hd]

/-- The scaled fraction of `q`, when `q · 2^(23 − expOf q)` is the natural `M`, is `M · den / den`. -/
theorem scaled_of_int (q : ℚ) (hq : 0 < q) (M : ℕ) (hM : q * (2 : ℚ) ^ (23 - expOf q) = M) :
    (scaled q).1 = M * (scaled q).2 := by
  obtain ⟨hden, hval⟩ := scaled_val q hq
  rw [hM] at hval
  have hd : ((scaled q).2 : ℚ) ≠ 0 := by exact_mod_cast hden.ne'
  have : ((scaled q).1 : ℚ) = (M : ℚ) * (scaled q).2 := by
    rw [div_eq_iff hd] at hval; exact hval
  exact_mod_cast this

/-- Rounding the magnitude of a finite nonzero pattern returns a pattern with the same magnitude. -/
theorem roundMag_magRat (v : Bits) (hf : isFinite v = true) (hnz : isZero v = false) :
    isFinite (roundMag (magRat v)) = true ∧ sign (roundMag (magRat v)) = false ∧
    magRat (roundMag (magRat v)) = magRat v := by
  have hq : 0 < magRat v := magRat_pos_of_not_zero hnz
  have hv32 := v.isLt
  have he_lt : expField v < 256 := by rw [expField_eq]; omega
  have he_ne : expField v ≠ 255 := by simpa [isFinite] using hf
  have hm_lt : mantField v < 2 ^ 23 := by rw [mantField_eq]; omega
  have two_ne : (2 : ℚ) ≠ 0 := by norm_num
  have hden : 0 < (scaled (magRat v)).2 := (scaled_val _ hq).1
  by_cases he : expField v = 0
  · -- subnormal: `m / 2^149`, exponent clamp `-126`
    have hm_ne : mantField v ≠ 0 := by
      intro hm; have := isZero_iff.2 ⟨he, hm⟩; rw [hnz] at this; cases this
    have hqv : magRat v = (mantField v : ℚ) / 2 ^ 149 := magRat_of_exp_zero he
    have hsub : magRat v < minNormal := by
      rw [hqv]; unfold minNormal
      have : (mantField v : ℚ) < 2 ^ 23 := by exact_mod_cast hm_lt
      rw [div_lt_div_iff₀ (by positivity) (by positivity)]
      nlinarith
    have hE : expOf (magRat v) = -126 := expOf_eq_of_sub hq hsub
    have hM : magRat v * (2 : ℚ) ^ (23 - expOf (magRat v)) = (mantField v : ℕ) := by
      rw [hE, hqv]
      rw [show (23 - (-126 : ℤ)) = ((149 : ℕ) : ℤ) by norm_num, zpow_natCast]
      field_simp
    have hsc := scaled_of_int (magRat v) hq (mantField v) hM
    rw [roundMag_eq, ite_eq_right (not_le.mpr hq)]
    rcases hsd : scaled (magRat v) with ⟨num, den⟩
    rw [hsd] at hsc hden
    dsimp only
    rw [rne_exact hden _ hsc, hE]
    obtain ⟨h1, h2, h3⟩ := encode_spec (mantField v) (-126) (le_refl _) (by omega) (by norm_num)
      (by intro h; omega) (fun _ => rfl)
    refine ⟨h1, h2, ?_⟩
    rw [h3, hqv, show ((-126 : ℤ) - 23) = -((149 : ℕ) : ℤ) by norm_num, zpow_neg, zpow_natCast, div_eq_mul_inv]
  · -- normal: `(2^23 + m) · 2^(e − 150)`
    have he_pos : 1 ≤ expField v := Nat.pos_of_ne_zero he
    set e : ℤ := (expField v : ℤ) with hedef
    set M : ℕ := 2 ^ 23 + mantField v with hMdef
    have hqv : magRat v = (M : ℚ) * (2 : ℚ) ^ (e - 150) := magRat_of_exp_ne he
    have hMlo : (2 : ℚ) ^ 23 ≤ (M : ℚ) := by
      rw [hMdef]; push_cast; linarith [(Nat.cast_nonneg (mantField v) : (0 : ℚ) ≤ _)]
    have hMhi : (M : ℚ) < 2 ^ 24 := by
      rw [hMdef]; push_cast
      have : (mantField v : ℚ) < 2 ^ 23 := by exact_mod_cast hm_lt
      linarith
    have hpos : (0 : ℚ) < 2 ^ (e - 150) := two_zpow_pos _
    have hE : expOf (magRat v) = e - 127 := by
      apply expOf_eq_of_bracket hq
      · rw [hqv, show e - 127 = (23 : ℕ) + (e - 150) by ring, zpow_add₀ two_ne, zpow_natCast]
        exact mul_le_mul_of_nonneg_right hMlo hpos.le
      · rw [hqv, show e - 127 + 1 = (24 : ℕ) + (e - 150) by ring, zpow_add₀ two_ne, zpow_natCast]
        exact mul_lt_mul_of_pos_right hMhi hpos
      · omega
    have hM : magRat v * (2 : ℚ) ^ (23 - expOf (magRat v)) = (M : ℕ) := by
      rw [hE, hqv, mul_assoc, ← zpow_add₀ two_ne, show (e - 150) + (23 - (e - 127)) = 0 by ring, zpow_zero, mul_one]
    have hsc := scaled_of_int (magRat v) hq M hM
    rw [roundMag_eq, ite_eq_right (not_le.mpr hq)]
    rcases hsd : scaled (magRat v) with ⟨num, den⟩
    rw [hsd] at hsc hden
    dsimp only
    rw [rne_exact hden _ hsc, hE]
    have hM24 : M ≤ 2 ^ 24 := by rw [hMdef]; omega
    have hMne : M ≠ 2 ^ 24 := by rw [hMdef]; omega
    have hM23 : ¬ M < 2 ^ 23 := by rw [hMdef]; omega
    obtain ⟨h1, h2, h3⟩ := encode_spec M (e - 127) (by omega) hM24 (by omega)
      (fun h => absurd h hMne) (fun h => absurd h hM23)
    refine ⟨h1, h2, ?_⟩
    rw [h3, hqv, show e - 127 - 23 = e - 150 by ring]

/-- Rounding the value of a finite pattern returns a finite pattern of the same value. -/
theorem roundRat_toRat (v : Bits) (hf : isFinite v = true) :
    isFinite (roundRat (toRat v)) = true ∧ toRat (roundRat (toRat v)) = toRat v := by
  by_cases hz : isZero v = true
  · rw [toRat_zero_of_isZero hz, roundRat_zero, toRat_posZero]
    exact ⟨isFinite_posZero, rfl⟩
  · have hnz : isZero v = false := by simpa using hz
    have hq : 0 < magRat v := magRat_pos_of_not_zero hnz
    obtain ⟨h1, h2, h3⟩ := roundMag_magRat v hf hnz
    have hlt := toNat_lt_of_sign_false h2
    have hne : toRat v ≠ 0 := by
      intro h; rw [← abs_toRat] at hq; rw [h, abs_zero] at hq; exact lt_irrefl _ hq
    have hnl : ¬ magRat v < 0 := not_lt.mpr hq.le
    have hneg : -magRat v < 0 := by linarith
    unfold roundRat
    rw [ite_eq_right (by simpa using hne)]
    have hmag : (if toRat v < 0 then -toRat v else toRat v) = magRat v := by
      unfold toRat
      cases hs : sign v
      · simp [hnl]
      · simp [hneg]
    rw [hmag, isFinite_withSign _ _ hlt, toRat_withSign _ _ hlt, h3]
    refine ⟨h1, ?_⟩
    unfold toRat
    cases hs : sign v
    · simp [hnl]
    · simp [hneg]

/-- **`add` of `+0`**: the value is unchanged (the pattern of `-0` becomes `+0`) and finite. -/
theorem toRat_add_posZero (v : Bits) (hv : isFinite v = true) :
    toRat (add posZero v) = toRat v ∧ isFinite (add posZero v) = true := by
  obtain ⟨hvn, hvi⟩ := finite_not_nan_inf hv
  have h0n : isNaN posZero = false := rfl
  have h0i : isInf posZero = false := rfl
  have h0z : isZero posZero = true := rfl
  have h0s : sign posZero = false := rfl
  unfold add
  simp only [h0n, h0i, hvn, hvi, h0z, h0s, Bool.false_or, Bool.false_and, Bool.true_and, Bool.false_eq_true,
    ↓reduceIte]
  by_cases hz : isZero v = true
  · simp only [hz, ↓reduceIte]
    rw [toRat_zero_of_isZero hz, toRat_posZero]
    exact ⟨rfl, isFinite_posZero⟩
  · have hz' : isZero v = false := by simpa using hz
    simp only [hz', Bool.false_eq_true, ↓reduceIte]
    rw [toRat_posZero, zero_add]
    obtain ⟨h1, h2⟩ := roundRat_toRat v hv
    exact ⟨h2, h1⟩

end TVA.Math.Float.F32
