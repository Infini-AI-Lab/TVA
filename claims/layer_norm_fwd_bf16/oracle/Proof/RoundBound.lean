import TVA.Proof

/-! A rounded value is no larger than the power of two that bounds it (round-to-nearest cannot
overshoot a representable value). -/

namespace TVA

namespace Format

noncomputable section

variable {f : Format}

/-- **Rounding preserves a power-of-two bound.**  If `|q| ≤ 2^E` and `2^E` is a finite value of
`f` (with `E` above the smallest exponent), the rounded value is at most `2^E` in magnitude. -/
theorem abs_roundRat_le_pow2 (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool) {q : ℚ} {E : ℕ}
    (hE : (2 : ℚ) ^ E ≤ f.maxVal) (hemin : f.emin ≤ (E : ℤ)) (hq : |q| ≤ (2 : ℚ) ^ E) :
    |f.toRat (f.roundRat rnd sat zs q)| ≤ (2 : ℚ) ^ E := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨-, -, h3⟩ := roundRat_zero_spec (f := f) (rnd := rnd) (sat := sat) (zs := zs) hf
    rw [h3, abs_zero]; positivity
  have hqmax : |q| ≤ f.maxVal := hq.trans hE
  obtain ⟨-, -, hval⟩ := roundRat_spec hf rnd sat zs hq0 hqmax
  rw [hval]
  set a : ℚ := |q| with ha_def
  have ha : 0 < a := by rw [ha_def]; exact abs_pos.mpr hq0
  have hqa : a ≤ (2 : ℚ) ^ E := by rw [ha_def]; exact hq
  have hc : |(if q < 0 then (-1 : ℚ) else 1)| = 1 := by split_ifs <;> norm_num
  have hv : (0 : ℚ) ≤ (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) := by positivity
  rw [abs_mul, hc, one_mul, abs_of_nonneg hv]
  -- Bound the quantum from above.
  have hlogE : Int.log 2 ((2 : ℚ) ^ E) = (E : ℤ) := by
    rw [← zpow_natCast]
    exact Int.log_zpow (b := 2) (by norm_num) (E : ℤ)
  have hlog : Int.log 2 a ≤ (E : ℤ) :=
    hlogE ▸ Int.log_mono_right (b := 2) ha hqa
  have hQle : f.qexp a ≤ (E : ℤ) - f.m := by
    rw [qexp_eq (f := f) ha]
    have := max_le hlog hemin
    linarith
  by_cases hcase : f.qexp a ≤ (E : ℤ) - f.m - 1
  · -- `roundSig` is at most `2^(m+1)` and the quantum is at most `2^(E-m-1)`.
    obtain ⟨hN, -, -⟩ := roundSig_bounds (f := f) rnd ha
    have hN' : (roundSig rnd a (f.qexp a) : ℚ) ≤ ((2 ^ (f.m + 1) : ℕ) : ℚ) := by
      exact_mod_cast hN
    have hcast1 : ((2 ^ (f.m + 1) : ℕ) : ℚ) = (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
      rw [show ((f.m : ℤ) + 1) = ((f.m + 1 : ℕ) : ℤ) by push_cast; ring, zpow_natCast]
      push_cast; rfl
    calc (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a)
        ≤ ((2 ^ (f.m + 1) : ℕ) : ℚ) * (2 : ℚ) ^ (f.qexp a) :=
          mul_le_mul_of_nonneg_right hN' (by positivity)
      _ = (2 : ℚ) ^ ((f.m : ℤ) + 1) * (2 : ℚ) ^ (f.qexp a) := by rw [hcast1]
      _ = (2 : ℚ) ^ ((f.m : ℤ) + 1 + f.qexp a) := by rw [← zpow_add₀ (by norm_num)]
      _ ≤ (2 : ℚ) ^ ((f.m : ℤ) + 1 + ((E : ℤ) - f.m - 1)) := by
            apply zpow_le_zpow_right₀ (by norm_num); linarith
      _ = (2 : ℚ) ^ ((E : ℤ)) := by
            rw [show ((f.m : ℤ) + 1 + ((E : ℤ) - f.m - 1)) = (E : ℤ) by ring]
      _ = (2 : ℚ) ^ E := zpow_natCast _ _
  · -- Here `qexp a = E - m`, and `roundSig ≤ 2^m`.
    rw [not_le] at hcase
    have hQe : f.qexp a = (E : ℤ) - f.m := le_antisymm hQle (by omega)
    have hscaled : a * (2 : ℚ) ^ (-(f.qexp a)) ≤ ((2 ^ f.m : ℕ) : ℚ) := by
      have hp : (0 : ℚ) < (2 : ℚ) ^ (-(f.qexp a)) := by positivity
      calc a * (2 : ℚ) ^ (-(f.qexp a)) ≤ (2 : ℚ) ^ E * (2 : ℚ) ^ (-(f.qexp a)) :=
            mul_le_mul_of_nonneg_right hqa hp.le
        _ = (2 : ℚ) ^ ((E : ℤ)) * (2 : ℚ) ^ (-(f.qexp a)) := by rw [zpow_natCast]
        _ = (2 : ℚ) ^ ((E : ℤ) + -(f.qexp a)) := by rw [zpow_add₀ (by norm_num)]
        _ = (2 : ℚ) ^ ((f.m : ℤ)) := by
              rw [show ((E : ℤ) + -(f.qexp a)) = ((f.m : ℤ)) by rw [hQe]; ring]
        _ = ((2 ^ f.m : ℕ) : ℚ) := by rw [zpow_natCast]; push_cast; rfl
    have hK : roundSig rnd a (f.qexp a) ≤ 2 ^ f.m := by
      rw [roundSig_eq]
      exact rnat_le rnd (by positivity) hscaled
    have hcast2 : ((2 ^ f.m : ℕ) : ℚ) = (2 : ℚ) ^ ((f.m : ℤ)) := by
      rw [zpow_natCast]; push_cast; rfl
    calc (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a)
        ≤ ((2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ (f.qexp a) :=
          mul_le_mul_of_nonneg_right (by exact_mod_cast hK) (by positivity)
      _ = (2 : ℚ) ^ ((f.m : ℤ)) * (2 : ℚ) ^ (f.qexp a) := by rw [hcast2]
      _ = (2 : ℚ) ^ ((f.m : ℤ) + f.qexp a) := by rw [zpow_add₀ (by norm_num)]
      _ = (2 : ℚ) ^ E := by
            rw [show ((f.m : ℤ) + f.qexp a) = (E : ℤ) by rw [hQe]; ring, zpow_natCast]

end

end Format


end TVA
