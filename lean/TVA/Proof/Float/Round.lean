import TVA.Proof.Float.Quantum

/-!
# The error of one rounding

`Format.roundRat f rnd sat zs q` is the pattern the exact value `q` rounds to. With
`u = 2^-(m+1)` (half an ulp of `1`) and `η = 2^(emin-m-1)` (half the smallest subnormal), one
rounding to nearest of a value in range is off by at most `|q|·u + η`: the *relative* error `u`
covers the normal range, the *absolute* error `η` the subnormal range, where the spacing of the
values no longer shrinks with `|q|`. This is the bound every float error of the library builds on.

* `roundRat_spec`: a value `q ≠ 0` with `|q| ≤ maxVal` rounds to a finite pattern whose value is
  `±N · 2^Q` (`TVA.Proof.Float.Quantum`); `roundRat_zero`, `roundRat_of_ne`: the pattern of `0`
  and of a nonzero value.
* **`roundRat_rn_err`**: to nearest, `|r − q| ≤ |q|·u + η` and `r` finite, for `|q| ≤ maxVal`;
  `roundRat_rn_err_normal`: `|q|·u` when `|q| ≥ 2^emin`.
* `roundRat_rz_err`: toward zero, within `2·(|q|·u + η)` and not above `|q|` in magnitude.
* `roundRat_rn_err_of_finite`: whenever a rounding to nearest gives a finite result `r` (also for
  `|q|` above `maxVal`, which then rounds down to `±maxVal`), `|r − q| ≤ |r|·u + η`. This is what
  bounds an exact value from the finite float it rounds to (`RoundsTo`, fused additions).
* `roundMag_sat_of_gt`: with saturation, a magnitude above `maxVal` gives `maxMag`.

`TVA.Proof.Float.Error` restates these for the operations on `FloatV`.
-/
namespace TVA.Format

variable {f : Format}

/-! ## Rounded patterns -/

/-- The pattern of a nonzero `q` is its sign plus its rounded magnitude. -/
theorem roundRat_of_ne {rnd : RoundMode} {sat zs : Bool} {q : ℚ} (hq : q ≠ 0) :
    f.roundRat rnd sat zs q = (if q < 0 then f.signBit else 0) + f.roundMag rnd sat |q| := by
  unfold roundRat; rw [ite_eq_right hq]
  congr 2
  split_ifs with h
  · exact (abs_of_neg h).symm
  · exact (abs_of_nonneg (not_lt.mp h)).symm

/-- The pattern of an exact zero is the zero of sign `zs`. -/
theorem roundRat_zero {rnd : RoundMode} {sat zs : Bool} :
    f.roundRat rnd sat zs 0 = f.setSign zs 0 := by
  unfold roundRat setSign mag; rw [ite_eq_left rfl]; simp

/-- The pattern of an exact zero is finite, with value `0`. -/
theorem roundRat_zero_spec (hf : f.Regular) {rnd : RoundMode} {sat zs : Bool} :
    f.roundRat rnd sat zs 0 < 2 ^ f.width ∧ f.isFinite (f.roundRat rnd sat zs 0) = true ∧
      f.toRat (f.roundRat rnd sat zs 0) = 0 := by
  rw [roundRat_zero]
  refine ⟨setSign_lt _ _, by rw [isFinite_setSign]; exact isFinite_zero hf, ?_⟩
  rw [toRat_setSign]; unfold mag; simp [magVal_zero]

/-- Rounding error of a signed magnitude: `|±v − q| = |v − |q||`. -/
private theorem abs_signed_sub {q : ℚ} (v : ℚ) :
    |(if q < 0 then (-1 : ℚ) else 1) * v - q| = abs (v - abs q) := by
  by_cases h : q < 0
  · rw [ite_eq_left h, abs_of_neg h, ← abs_neg]; congr 1; ring
  · rw [ite_eq_right h, abs_of_nonneg (not_lt.mp h)]; ring_nf

/-- The magnitude of a signed nonnegative value. -/
private theorem abs_signed {q v : ℚ} (hv : 0 ≤ v) : |(if q < 0 then (-1 : ℚ) else 1) * v| = v := by
  by_cases h : q < 0
  · rw [ite_eq_left h, abs_mul]; simp [abs_of_nonneg hv]
  · rw [ite_eq_right h, one_mul, abs_of_nonneg hv]

/-- **A value in range rounds to a finite pattern with the value `N · 2^Q`** of its rounded
significand, signed. -/
theorem roundRat_spec (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool) {q : ℚ} (hq0 : q ≠ 0)
    (hq : |q| ≤ f.maxVal) :
    f.roundRat rnd sat zs q < 2 ^ f.width ∧ f.isFinite (f.roundRat rnd sat zs q) = true ∧
      f.toRat (f.roundRat rnd sat zs q) =
        (if q < 0 then -1 else 1) * ((roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|)) := by
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsub⟩ := roundSig_bounds (f := f) rnd ha
  have hle := pack_le_maxMag hf hN hQ hsub (roundSig_le_maxVal hf rnd ha hq)
  have hlt := (hle.trans_lt (maxMag_lt hf))
  have hpat : f.roundRat rnd sat zs q = f.setSign (decide (q < 0))
      (f.pack (roundSig rnd |q| (f.qexp |q|)) (f.qexp |q|)) := by
    rw [roundRat_of_ne hq0, roundMag_of_le hf rnd sat ha hq, setSign_of_lt hlt]
    by_cases h : q < 0 <;> simp [h]
  rw [hpat]
  refine ⟨setSign_lt _ _, ?_, ?_⟩
  · rw [isFinite_setSign]; exact isFinite_of_mag_le hf (by rw [mag_of_lt hlt]; exact hle)
  · rw [toRat_setSign, mag_of_lt hlt, magVal_pack hN hQ hsub]
    by_cases h : q < 0 <;> simp [h]

/-- **One rounding to nearest** of a value in range: a finite pattern within `|q|·u + η`. -/
theorem roundRat_rn_err (hf : f.Regular) (sat zs : Bool) {q : ℚ} (hq : |q| ≤ f.maxVal) :
    f.roundRat .rn sat zs q < 2 ^ f.width ∧ f.isFinite (f.roundRat .rn sat zs q) = true ∧
      |f.toRat (f.roundRat .rn sat zs q) - q| ≤ |q| * f.u + f.eta := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨h1, h2, h3⟩ := roundRat_zero_spec (f := f) (rnd := .rn) (sat := sat) (zs := zs) hf
    refine ⟨h1, h2, ?_⟩
    rw [h3]; simp [eta_nonneg]
  obtain ⟨hlt, hfin, hval⟩ := roundRat_spec hf .rn sat zs hq0 hq
  refine ⟨hlt, hfin, ?_⟩
  have ha : 0 < |q| := abs_pos.mpr hq0
  generalize hA : |q| = a at ha hval ⊢
  have herr := roundSig_rn_err ha (f.qexp a)
  have hh := half_quantum_le (f := f) ha
  rw [hval, abs_signed_sub, hA]
  linarith

/-- **One rounding to nearest in the normal range** (`|q| ≥ 2^emin`): within `|q|·u`. -/
theorem roundRat_rn_err_normal (hf : f.Regular) (sat zs : Bool) {q : ℚ}
    (hlo : (2 : ℚ) ^ f.emin ≤ |q|) (hq : |q| ≤ f.maxVal) :
    |f.toRat (f.roundRat .rn sat zs q) - q| ≤ |q| * f.u := by
  have hq0 : q ≠ 0 := by
    intro h; rw [h, abs_zero] at hlo; exact absurd hlo (not_le.mpr (by positivity))
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨-, -, hval⟩ := roundRat_spec hf .rn sat zs hq0 hq
  rw [hval, abs_signed_sub]
  generalize hA : |q| = a at ha hlo ⊢
  have herr := roundSig_rn_err ha (f.qexp a)
  have hh := half_quantum_normal (f := f) ha (le_log_of_le ha hlo)
  linarith [mul_comm f.u a]

/-- **One rounding toward zero** of a value in range: a finite pattern within `2·(|q|·u + η)`,
not above `|q|` in magnitude. -/
theorem roundRat_rz_err (hf : f.Regular) (sat zs : Bool) {q : ℚ} (hq : |q| ≤ f.maxVal) :
    f.roundRat .rz sat zs q < 2 ^ f.width ∧ f.isFinite (f.roundRat .rz sat zs q) = true ∧
      |f.toRat (f.roundRat .rz sat zs q) - q| ≤ 2 * (|q| * f.u + f.eta) ∧
      |f.toRat (f.roundRat .rz sat zs q)| ≤ |q| := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨h1, h2, h3⟩ := roundRat_zero_spec (f := f) (rnd := .rz) (sat := sat) (zs := zs) hf
    refine ⟨h1, h2, ?_⟩
    rw [h3]; simp [eta_nonneg]
  obtain ⟨hlt, hfin, hval⟩ := roundRat_spec hf .rz sat zs hq0 hq
  refine ⟨hlt, hfin, ?_⟩
  have ha : 0 < |q| := abs_pos.mpr hq0
  rw [hval, abs_signed_sub, abs_signed (by positivity)]
  generalize hA : |q| = a at ha ⊢
  obtain ⟨h1, h2⟩ := roundSig_rz_err ha (f.qexp a)
  have hh := half_quantum_le (f := f) ha
  refine ⟨?_, h1⟩
  rw [abs_of_nonpos (by linarith)]
  linarith

/-- The overflow of a rounding to nearest without saturation is not finite (for a format with
infinities or a NaN). -/
theorem roundMag_rn_of_over (hf : f.Regular) (hs : f.specials ≠ .finite) {a : ℚ}
    (hover : ¬ (roundSig .rn a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) ≤ f.maxVal) :
    f.roundMag .rn false a < f.signBit ∧ f.isFinite (f.roundMag .rn false a) = false := by
  unfold roundMag encode
  rw [pow2_eq, ite_eq_right hover]
  simp only [Bool.false_or, beq_iff_eq, reduceCtorEq, ite_false]
  cases hsp : f.specials with
  | ieee =>
    simp only
    refine ⟨?_, not_isFinite_infMag hsp⟩
    rw [signBit_eq]; unfold infMag
    exact Nat.mul_lt_mul_of_pos_right (by have := Nat.one_le_two_pow (n := f.e); omega) (by positivity)
  | nanOnly =>
    simp only
    refine ⟨?_, not_isFinite_nanMag hf (by rw [hsp]; decide)⟩
    unfold nanMag signBit; have := Nat.one_le_two_pow (n := f.e + f.m); omega
  | finite => exact absurd hsp hs

/-- **A finite result of a rounding to nearest is within one rounding's bound of the exact value**,
also when `|q|` exceeds `maxVal` (then it rounds down to `±maxVal`), relative to the result:
`|r − q| ≤ u·|r| + η` (a bound of at least half an ulp of `r`, and up to about a whole one). A
format whose overflow is finite (`.finite`) is excluded. -/
theorem roundRat_rn_err_of_finite (hf : f.Regular) (hs : f.specials ≠ .finite) (zs : Bool) {q : ℚ}
    (h : f.isFinite (f.roundRat .rn false zs q) = true) :
    |f.toRat (f.roundRat .rn false zs q) - q| ≤ |f.toRat (f.roundRat .rn false zs q)| * f.u + f.eta := by
  by_cases hq0 : q = 0
  · subst hq0
    obtain ⟨-, -, h3⟩ := roundRat_zero_spec (f := f) (rnd := .rn) (sat := false) (zs := zs) hf
    rw [h3]; simp [eta_nonneg]
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsub⟩ := roundSig_bounds (f := f) .rn ha
  have hno : (roundSig .rn |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|) ≤ f.maxVal := by
    by_contra hover
    obtain ⟨hlt, hnf⟩ := roundMag_rn_of_over hf hs hover
    rw [roundRat_of_ne hq0, show (if q < 0 then f.signBit else 0) = if decide (q < 0) then f.signBit else 0 by simp,
      ← setSign_of_lt hlt, isFinite_setSign, hnf] at h
    cases h
  have hmag : f.roundMag .rn false |q| = f.pack (roundSig .rn |q| (f.qexp |q|)) (f.qexp |q|) := by
    unfold roundMag encode; rw [pow2_eq, ite_eq_left hno]
  have hle := pack_le_maxMag hf hN hQ hsub hno
  have hlt := hle.trans_lt (maxMag_lt hf)
  have hval : f.toRat (f.roundRat .rn false zs q) =
      (if q < 0 then -1 else 1) * ((roundSig .rn |q| (f.qexp |q|) : ℚ) * 2 ^ (f.qexp |q|)) := by
    rw [roundRat_of_ne hq0, hmag,
      show (if q < 0 then f.signBit else 0) = if decide (q < 0) then f.signBit else 0 by simp,
      ← setSign_of_lt hlt, toRat_setSign, mag_of_lt hlt, magVal_pack hN hQ hsub]
    by_cases h : q < 0 <;> simp [h]
  rw [hval, abs_signed_sub, abs_signed (by positivity)]
  generalize hA : |q| = a at ha hN hsub hno ⊢
  have herr := roundSig_rn_err ha (f.qexp a)
  -- half the quantum is at most `u·N·2^Q + η`
  have hhalf : (2 : ℚ) ^ (f.qexp a) / 2 ≤
      (roundSig .rn a (f.qexp a) : ℚ) * 2 ^ (f.qexp a) * f.u + f.eta := by
    by_cases hn : f.emin ≤ Int.log 2 a
    · have hNm := roundSig_ge (f := f) .rn ha hn
      have hNm' : (2 : ℚ) ^ (f.m : ℤ) ≤ (roundSig .rn a (f.qexp a) : ℚ) := by
        rw [zpow_natCast]; exact_mod_cast hNm
      rw [half_quantum_eq, zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
      have hp : (0 : ℚ) ≤ 2 ^ (f.qexp a) * f.u := by rw [u_eq]; positivity
      have := eta_nonneg (f := f)
      have := mul_le_mul_of_nonneg_right hNm' hp
      nlinarith
    · rw [half_quantum_sub ha (not_le.mp hn)]
      have : 0 ≤ (roundSig .rn a (f.qexp a) : ℚ) * 2 ^ (f.qexp a) * f.u :=
        mul_nonneg (by positivity) u_nonneg
      linarith
  linarith

/-- **Saturation beyond the range.** A magnitude above `maxVal`, rounded with saturation (either
mode), gives the largest finite magnitude: its rounded value is at least `maxVal`, so it is
`maxVal` itself or an overflow. -/
theorem roundMag_sat_of_gt (hf : f.Regular) (rnd : RoundMode) {a : ℚ} (ha : f.maxVal < a) :
    f.roundMag rnd true a = f.maxMag := by
  have ha0 : 0 < a := (maxVal_pos hf).trans ha
  obtain ⟨hN, hQ, hsub⟩ := roundSig_bounds (f := f) rnd ha0
  have hE : f.emin ≤ (f.topExp : ℤ) - f.bias := by rw [emin_eq]; have := topExp_pos hf; omega
  have hmlo : (2 : ℚ) ^ ((f.topExp : ℤ) - f.bias) ≤ f.maxVal := by
    rw [maxVal_eq]
    have hs : ((2 ^ f.m : ℕ) : ℚ) ≤ (f.topSig : ℚ) := by exact_mod_cast topSig_ge hf
    have e : (2 : ℚ) ^ ((f.topExp : ℤ) - f.bias) =
        ((2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ ((f.topExp : ℤ) - f.bias - f.m) := by
      push_cast; rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]; congr 1; ring
    rw [e]; exact mul_le_mul_of_nonneg_right hs (by positivity)
  have hlog : (f.topExp : ℤ) - f.bias ≤ Int.log 2 a := le_log_of_le ha0 (hmlo.trans ha.le)
  have hn : f.emin ≤ Int.log 2 a := hE.trans hlog
  have hQe : f.qexp a = Int.log 2 a - f.m := by rw [qexp_eq ha0, max_eq_left hn]
  have hp : (0 : ℚ) < (2 : ℚ) ^ (f.qexp a) := by positivity
  -- the rounded value is at least `maxVal`
  have hge : f.maxVal ≤ (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) := by
    rcases lt_or_eq_of_le hlog with hlt | heq
    · -- a binade above `maxVal`'s: the significand is at least `2^m`
      have hNm : (2 : ℚ) ^ (f.m : ℤ) ≤ (roundSig rnd a (f.qexp a) : ℚ) := by
        rw [zpow_natCast]; exact_mod_cast roundSig_ge (f := f) rnd ha0 hn
      have h1 : (2 : ℚ) ^ (Int.log 2 a) ≤ (roundSig rnd a (f.qexp a) : ℚ) * (2 : ℚ) ^ (f.qexp a) := by
        have e : (2 : ℚ) ^ (Int.log 2 a) = (2 : ℚ) ^ (f.m : ℤ) * (2 : ℚ) ^ (f.qexp a) := by
          rw [← zpow_add₀ (by norm_num), hQe]; congr 1; ring
        rw [e]; exact mul_le_mul_of_nonneg_right hNm hp.le
      have h2 : (2 : ℚ) ^ ((f.topExp : ℤ) + 1 - f.bias) ≤ (2 : ℚ) ^ (Int.log 2 a) :=
        zpow_le_zpow_right₀ (by norm_num) (by omega)
      linarith [maxVal_lt hf]
    · -- `maxVal`'s binade: the quantum is `maxVal`'s, and `a / 2^Q > topSig`
      have hQT : f.qexp a = (f.topExp : ℤ) - f.bias - f.m := by rw [hQe, ← heq]
      have hK : (f.topSig : ℚ) ≤ a * (2 : ℚ) ^ (-f.qexp a) := by
        have h := mul_lt_mul_of_pos_right ha (show (0 : ℚ) < (2 : ℚ) ^ (-f.qexp a) by positivity)
        rw [maxVal_eq, ← hQT, mul_assoc, ← zpow_add₀ (by norm_num), add_neg_cancel, zpow_zero,
          mul_one] at h
        exact h.le
      have hN' : f.topSig ≤ roundSig rnd a (f.qexp a) := by rw [roundSig_eq]; exact le_rnat rnd hK
      rw [maxVal_eq, ← hQT]
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast hN') hp.le
  unfold roundMag encode
  rw [pow2_eq]
  split_ifs with hle hsat
  · apply magVal_inj
    rw [magVal_pack hN hQ hsub, le_antisymm hle hge, magVal_maxMag hf]
  · rfl
  · simp at hsat

end TVA.Format
