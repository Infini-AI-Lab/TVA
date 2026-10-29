import TVA.Math.Defs
import Mathlib.Tactic
import Mathlib.Data.Int.Log
import Mathlib.Data.Rat.Floor

/-!
# Binary floating-point formats

A format with `e` exponent bits and `m` fraction bits stores a value in `1 + e + m` bits: the
sign (the top bit), the exponent field `E` and the fraction field `F` (the low `m` bits). With the
bias `b = 2^(e-1) - 1` (IEEE 754-2019 §3.3, and the OCP 8-bit formats):
- `E = 0`: the subnormal value `F · 2^(1 - b - m)` (both zeros when `F = 0`);
- `E > 0`: the normal value `(2^m + F) · 2^(E - b - m)`;
- the sign bit negates the value.

`Specials` says which patterns are not numbers:
- `.ieee` (IEEE 754 `binary16`, `binary32`, `binary64`, `bfloat16`, and OCP `E5M2`): an
  all-ones exponent field is `±∞` (fraction `0`) or NaN (fraction `≠ 0`);
- `.nanOnly` (OCP `E4M3`, Triton's `float8e4nv`): no infinities; NaN is the all-ones exponent
  with the all-ones fraction (`0x7f`, `0xff`); the other all-ones-exponent patterns are numbers;
- `.finite` (OCP `E2M1`, the fp4 of `tl.dot_scaled`): every pattern is a number.

Patterns are natural numbers (`FloatV` in `TVA.Lang.Scalar` holds them as `BitVec`). The
definitions are in `TVA.Math.Defs`, in core Lean, so that the library's meaning of each pattern
runs natively and is compared with the device (`evidence/float-layer`, parts A and B); this
module proves their properties.

* `toRat f n`: the exact value of a pattern (`magVal` of its magnitude, negated by the sign).
  For a finite pattern it is its number. For `±∞` it is `±2^(emax+1)`, `emax` the bias
  (`toRat_of_isInf`), beyond `±maxVal` (`maxVal_lt_inf`), so that the order of values orders the
  infinities below and above every number: `maxNum`, `minNum`, `cmp`, `orderKey`
  (`toRat_lt_iff_orderKey`) and the library's `tl.maximum`, comparisons and sort keys rely on it.
  Only a NaN's value is meaningless (the formula's; every operation tests for NaN first). Every
  error bound of the library is about finite values: it concludes that its result is finite (or
  assumes it), so the value it bounds is a number.
* `isNaN`, `isInf`, `isFinite`, `isZero`, `sign`, `mag` (the pattern without the sign), `setSign`.
* `maxVal`, `u = 2^-(m+1)` (half an ulp of `1`), `eta = 2^(emin-m-1)` (half the smallest
  subnormal), `emin = 1 - b`.
* `Regular`: `e ≥ 2` and `m ≥ 1`, which every format here meets and the error lemmas need.
-/
namespace TVA.Math.Float

theorem pow2_eq (k : ℤ) : pow2 k = (2 : ℚ) ^ k := by
  unfold pow2
  split_ifs with h
  · obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le h
    simp
  · obtain ⟨n, rfl⟩ := Int.exists_eq_neg_ofNat (le_of_lt (not_le.mp h))
    have e : (-(-(n : ℤ))).toNat = n := by simp
    rw [zpow_neg, zpow_natCast]
    have h2 : ((2 : ℚ) ^ n) = ((2 ^ n : ℕ) : ℚ) := by push_cast; rfl
    rw [h2]
    apply Rat.ext
    · rw [Rat.inv_natCast_num]
      show (1 : ℤ) = _
      exact (Int.sign_eq_one_of_pos (by positivity)).symm
    · rw [Rat.inv_natCast_den]
      show 2 ^ (-(-(n : ℤ))).toNat = _
      rw [e]; simp

namespace Format

theorem binary16_regular : binary16.Regular := by decide
theorem bfloat16_regular : bfloat16.Regular := by decide
theorem binary32_regular : binary32.Regular := by decide
theorem binary64_regular : binary64.Regular := by decide
theorem e5m2_regular : e5m2.Regular := by decide
theorem e4m3_regular : e4m3.Regular := by decide
theorem e2m1_regular : e2m1.Regular := by decide

/-! ## Basic facts -/

variable {f : Format}

theorem u_eq : f.u = (2 : ℚ) ^ (-(f.m : ℤ) - 1) := pow2_eq _
theorem eta_eq : f.eta = (2 : ℚ) ^ (f.emin - f.m - 1) := pow2_eq _

theorem u_nonneg : 0 ≤ f.u := by rw [u_eq]; positivity
theorem eta_nonneg : 0 ≤ f.eta := by rw [eta_eq]; positivity
theorem u_pos : 0 < f.u := by rw [u_eq]; positivity
theorem eta_pos : 0 < f.eta := by rw [eta_eq]; positivity

theorem signBit_pos : 0 < f.signBit := by unfold signBit; positivity

theorem two_signBit : 2 * f.signBit = 2 ^ f.width := by
  unfold signBit width; rw [pow_succ]; ring

theorem signBit_lt : f.signBit < 2 ^ f.width := by
  have := f.two_signBit; have := f.signBit_pos; omega

theorem mag_lt (n : ℕ) : f.mag n < f.signBit := Nat.mod_lt _ signBit_pos

theorem mag_le (n : ℕ) : f.mag n ≤ n := Nat.mod_le _ _

theorem mag_of_lt {n : ℕ} (h : n < f.signBit) : f.mag n = n := Nat.mod_eq_of_lt h

theorem mag_mag (n : ℕ) : f.mag (f.mag n) = f.mag n := mag_of_lt (mag_lt n)

/-- A pattern is its sign bit plus its magnitude. -/
theorem eq_setSign {n : ℕ} (hn : n < 2 ^ f.width) : f.setSign (f.sign n) n = n := by
  have h2 := f.two_signBit
  have hpos := f.signBit_pos
  unfold setSign sign mag
  have hd : n / f.signBit < 2 := by
    rw [Nat.div_lt_iff_lt_mul hpos]; linarith
  have hsplit := Nat.div_add_mod n f.signBit
  interval_cases h : n / f.signBit
  · simp; omega
  · simp; omega

theorem mag_setSign (s : Bool) (n : ℕ) : f.mag (f.setSign s n) = f.mag n := by
  unfold setSign
  cases s
  · simp [mag_mag]
  · simp only [ite_true]
    unfold mag
    rw [Nat.add_mod_left, Nat.mod_mod]

theorem sign_setSign (s : Bool) (n : ℕ) : f.sign (f.setSign s n) = s := by
  have hm := f.mag_lt n
  have hpos := f.signBit_pos
  unfold setSign sign
  cases s
  · simp [Nat.div_eq_of_lt hm]
  · simp only [ite_true]
    rw [Nat.add_comm, Nat.add_div_right _ hpos, Nat.div_eq_of_lt hm]
    rfl

theorem setSign_lt (s : Bool) (n : ℕ) : f.setSign s n < 2 ^ f.width := by
  have hm := f.mag_lt n
  have h2 := f.two_signBit
  unfold setSign
  split_ifs <;> omega

theorem setSign_of_lt {s : Bool} {M : ℕ} (hM : M < f.signBit) :
    f.setSign s M = (if s then f.signBit else 0) + M := by
  unfold setSign; rw [mag_of_lt hM]

theorem isNaN_setSign (s : Bool) (n : ℕ) : f.isNaN (f.setSign s n) = f.isNaN n := by
  unfold isNaN expField fracField; rw [mag_setSign]

theorem isInf_setSign (s : Bool) (n : ℕ) : f.isInf (f.setSign s n) = f.isInf n := by
  unfold isInf expField fracField; rw [mag_setSign]

theorem isFinite_setSign (s : Bool) (n : ℕ) : f.isFinite (f.setSign s n) = f.isFinite n := by
  unfold isFinite; rw [isNaN_setSign, isInf_setSign]

theorem isZero_setSign (s : Bool) (n : ℕ) : f.isZero (f.setSign s n) = f.isZero n := by
  unfold isZero; rw [mag_setSign]

theorem toRat_setSign (s : Bool) (n : ℕ) :
    f.toRat (f.setSign s n) = if s then -f.magVal (f.mag n) else f.magVal (f.mag n) := by
  unfold toRat; rw [sign_setSign, mag_setSign]

theorem isNaN_mag (n : ℕ) : f.isNaN (f.mag n) = f.isNaN n := by
  unfold isNaN expField fracField; rw [mag_mag]

theorem isInf_mag (n : ℕ) : f.isInf (f.mag n) = f.isInf n := by
  unfold isInf expField fracField; rw [mag_mag]

theorem isFinite_mag (n : ℕ) : f.isFinite (f.mag n) = f.isFinite n := by
  unfold isFinite; rw [isNaN_mag, isInf_mag]

theorem isFinite_iff (n : ℕ) : f.isFinite n = true ↔ f.isNaN n = false ∧ f.isInf n = false := by
  unfold isFinite; cases f.isNaN n <;> cases f.isInf n <;> simp

theorem not_isNaN_of_isFinite {n : ℕ} (h : f.isFinite n = true) : f.isNaN n = false :=
  ((isFinite_iff n).mp h).1

theorem not_isInf_of_isFinite {n : ℕ} (h : f.isFinite n = true) : f.isInf n = false :=
  ((isFinite_iff n).mp h).2

/-! ## Values of magnitudes -/

theorem magVal_eq (M : ℕ) : f.magVal M =
    if M / 2 ^ f.m = 0 then ((M % 2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ (f.emin - f.m)
    else ((2 ^ f.m + M % 2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) - f.bias - f.m) := by
  unfold magVal; rw [pow2_eq, pow2_eq]

theorem magVal_nonneg (M : ℕ) : 0 ≤ f.magVal M := by
  rw [magVal_eq]; split_ifs <;> positivity

theorem magVal_zero : f.magVal 0 = 0 := by rw [magVal_eq]; simp

theorem emin_eq : f.emin = 1 - f.bias := rfl

/-- A magnitude in exponent field `E ≥ 1` is at least `2^(E - bias)`. -/
theorem magVal_ge_of_exp {M : ℕ} (h : 1 ≤ M / 2 ^ f.m) :
    (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) - f.bias) ≤ f.magVal M := by
  rw [magVal_eq, ite_eq_right (by omega)]
  have h2 : (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) - f.bias) =
      ((2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) - f.bias - f.m) := by
    push_cast
    rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]
    ring_nf
  rw [h2]
  apply mul_le_mul_of_nonneg_right _ (by positivity)
  exact_mod_cast Nat.le_add_right _ _

/-- A magnitude in exponent field `E` is below `2^(E + 1 - bias)` (for `E = 0`, below
`2^emin`). -/
theorem magVal_lt_of_exp (M : ℕ) :
    f.magVal M < (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) + 1 - f.bias) := by
  have hF : M % 2 ^ f.m < 2 ^ f.m := Nat.mod_lt _ (by positivity)
  rw [magVal_eq]
  split_ifs with h
  · rw [h]
    have e1 : (2 : ℚ) ^ (((0 : ℕ) : ℤ) + 1 - f.bias) =
        ((2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ (f.emin - f.m) := by
      push_cast
      rw [← zpow_natCast, ← zpow_add₀ (by norm_num), emin_eq]
      congr 1; ring
    rw [e1]
    apply mul_lt_mul_of_pos_right _ (by positivity)
    exact_mod_cast hF
  · have e1 : (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) + 1 - f.bias) =
        ((2 ^ (f.m + 1) : ℕ) : ℚ) * (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) - f.bias - f.m) := by
      push_cast
      rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]
      congr 1; push_cast; ring
    rw [e1]
    apply mul_lt_mul_of_pos_right _ (by positivity)
    have : 2 ^ f.m + M % 2 ^ f.m < 2 ^ (f.m + 1) := by rw [pow_succ]; omega
    exact_mod_cast this

/-- **Magnitudes are ordered as their values.** -/
theorem magVal_lt_magVal {M M' : ℕ} (h : M < M') : f.magVal M < f.magVal M' := by
  have hm : 0 < 2 ^ f.m := by positivity
  have h1 := Nat.div_add_mod M (2 ^ f.m)
  have h2 := Nat.div_add_mod M' (2 ^ f.m)
  have hle : M / 2 ^ f.m ≤ M' / 2 ^ f.m := Nat.div_le_div_right h.le
  by_cases hE : M / 2 ^ f.m = M' / 2 ^ f.m
  · -- same exponent field: the fractions are ordered
    have hF : M % 2 ^ f.m < M' % 2 ^ f.m := by
      rw [hE] at h1
      generalize 2 ^ f.m * (M' / 2 ^ f.m) = A at h1 h2
      omega
    rw [magVal_eq, magVal_eq, ← hE]
    split_ifs
    · apply mul_lt_mul_of_pos_right _ (by positivity); exact_mod_cast hF
    · apply mul_lt_mul_of_pos_right _ (by positivity)
      exact_mod_cast (by omega : 2 ^ f.m + M % 2 ^ f.m < 2 ^ f.m + M' % 2 ^ f.m)
  · have hlt : M / 2 ^ f.m < M' / 2 ^ f.m := lt_of_le_of_ne hle hE
    have ha := magVal_lt_of_exp (f := f) M
    have hb := magVal_ge_of_exp (f := f) (M := M') ((Nat.zero_le _).trans_lt hlt)
    refine lt_of_lt_of_le ha (le_trans ?_ hb)
    apply zpow_le_zpow_right₀ (by norm_num)
    have : ((M / 2 ^ f.m : ℕ) : ℤ) + 1 ≤ ((M' / 2 ^ f.m : ℕ) : ℤ) := by exact_mod_cast hlt
    linarith

theorem magVal_le_magVal {M M' : ℕ} (h : M ≤ M') : f.magVal M ≤ f.magVal M' := by
  rcases h.lt_or_eq with h | h
  · exact (magVal_lt_magVal h).le
  · rw [h]

theorem magVal_lt_iff {M M' : ℕ} : f.magVal M < f.magVal M' ↔ M < M' :=
  ⟨fun h => by by_contra h'; exact absurd h (not_lt.mpr (magVal_le_magVal (not_lt.mp h'))),
    magVal_lt_magVal⟩

theorem magVal_inj {M M' : ℕ} (h : f.magVal M = f.magVal M') : M = M' := by
  rcases lt_trichotomy M M' with h' | h' | h'
  · exact absurd h (magVal_lt_magVal h').ne
  · exact h'
  · exact absurd h.symm (magVal_lt_magVal h').ne

theorem magVal_pos {M : ℕ} (h : 0 < M) : 0 < f.magVal M := by
  have := magVal_lt_magVal (f := f) h; rwa [magVal_zero] at this

theorem magVal_eq_zero_iff {M : ℕ} : f.magVal M = 0 ↔ M = 0 := by
  constructor
  · intro h; by_contra h'; exact absurd h (magVal_pos (Nat.pos_of_ne_zero h')).ne'
  · rintro rfl; exact magVal_zero

/-- Every magnitude's value is `S · 2^E` with a significand `S < 2^(m+1)` and `E ≥ emin − m`. -/
theorem magVal_sig_exp (M : ℕ) : ∃ S : ℕ, ∃ E : ℤ, f.magVal M = S * (2 : ℚ) ^ E ∧
    S < 2 ^ (f.m + 1) ∧ f.emin - f.m ≤ E := by
  have hF : M % 2 ^ f.m < 2 ^ f.m := Nat.mod_lt _ (by positivity)
  have h2 : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  rw [magVal_eq]
  split_ifs with h
  · exact ⟨_, _, rfl, by omega, le_rfl⟩
  · refine ⟨_, _, rfl, by omega, ?_⟩
    rw [emin_eq]
    have : (1 : ℤ) ≤ ((M / 2 ^ f.m : ℕ) : ℤ) := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr h
    linarith

/-! ## Values of patterns -/

theorem toRat_eq (n : ℕ) : f.toRat n = if f.sign n then -f.magVal (f.mag n) else f.magVal (f.mag n) :=
  rfl

theorem abs_toRat (n : ℕ) : |f.toRat n| = f.magVal (f.mag n) := by
  unfold toRat; split_ifs
  · rw [abs_neg, abs_of_nonneg (magVal_nonneg _)]
  · rw [abs_of_nonneg (magVal_nonneg _)]

theorem toRat_of_isZero {n : ℕ} (h : f.isZero n = true) : f.toRat n = 0 := by
  unfold isZero at h
  have : f.mag n = 0 := by simpa using h
  unfold toRat; rw [this, magVal_zero]; simp

theorem toRat_eq_zero_iff {n : ℕ} : f.toRat n = 0 ↔ f.isZero n = true := by
  constructor
  · intro h
    have : |f.toRat n| = 0 := by rw [h, abs_zero]
    rw [abs_toRat, magVal_eq_zero_iff] at this
    unfold isZero; simp [this]
  · exact toRat_of_isZero

theorem toRat_mag (n : ℕ) : f.toRat (f.mag n) = f.magVal (f.mag n) := by
  have : f.sign (f.mag n) = false := by
    unfold sign; rw [Nat.div_eq_of_lt (mag_lt n)]; rfl
  unfold toRat; rw [this, mag_mag]; rfl

/-- **Values are ordered as order keys**, for every pattern (infinities included: their
formula values lie beyond `maxVal`). -/
theorem toRat_lt_iff_orderKey (x y : ℕ) : f.toRat x < f.toRat y ↔ f.orderKey x < f.orderKey y := by
  unfold toRat orderKey
  have hx := magVal_nonneg (f := f) (f.mag x)
  have hy := magVal_nonneg (f := f) (f.mag y)
  have hpos : ∀ M : ℕ, 0 < M → 0 < f.magVal M := fun M h => magVal_pos h
  cases hsx : f.sign x <;> cases hsy : f.sign y <;> simp only [Bool.false_eq_true, ite_false, ite_true]
  · rw [magVal_lt_iff]; omega
  · constructor
    · intro h; exfalso; linarith
    · intro h; exfalso; omega
  · constructor
    · intro h
      by_contra h'
      have hx0 : f.mag x = 0 := by omega
      have hy0 : f.mag y = 0 := by omega
      rw [hx0, hy0, magVal_zero] at h
      simp at h
    · intro h
      rcases Nat.eq_zero_or_pos (f.mag x) with h0 | h0
      · have := hpos _ (show 0 < f.mag y by omega); linarith
      · have := hpos _ h0; linarith
  · rw [neg_lt_neg_iff, magVal_lt_iff]; omega

theorem toRat_eq_iff_orderKey (x y : ℕ) : f.toRat x = f.toRat y ↔ f.orderKey x = f.orderKey y := by
  constructor
  · intro h
    have h1 := (toRat_lt_iff_orderKey x y).not.mp (by rw [h]; exact lt_irrefl _)
    have h2 := (toRat_lt_iff_orderKey y x).not.mp (by rw [h]; exact lt_irrefl _)
    omega
  · intro h
    have h1 := (toRat_lt_iff_orderKey x y).not.mpr (by rw [h]; exact lt_irrefl _)
    have h2 := (toRat_lt_iff_orderKey y x).not.mpr (by rw [h]; exact lt_irrefl _)
    linarith [not_lt.mp h1, not_lt.mp h2]

/-! ## The largest value, the infinities and NaN -/

theorem width_eq : f.width = f.e + f.m + 1 := rfl

theorem bias_nonneg : 0 ≤ f.bias := by
  unfold bias
  have : (1 : ℤ) ≤ 2 ^ (f.e - 1) := one_le_pow₀ (by norm_num)
  linarith

theorem topExp_pos (hf : f.Regular) : 1 ≤ f.topExp := by
  obtain ⟨he, -⟩ := hf
  have : 4 ≤ 2 ^ f.e := by
    calc 4 = 2 ^ 2 := by norm_num
      _ ≤ 2 ^ f.e := Nat.pow_le_pow_right (by norm_num) he
  unfold topExp; split <;> omega

theorem topExp_lt : f.topExp < 2 ^ f.e := by
  have : 1 ≤ 2 ^ f.e := Nat.one_le_two_pow
  unfold topExp; split <;> omega

theorem topSig_ge (hf : f.Regular) : 2 ^ f.m ≤ f.topSig := by
  obtain ⟨-, hm⟩ := hf
  have h2 : 2 ≤ 2 ^ f.m := by
    calc 2 = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ f.m := Nat.pow_le_pow_right (by norm_num) hm
  have : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  unfold topSig; split <;> omega

theorem topSig_lt : f.topSig < 2 ^ (f.m + 1) := by
  have : 1 ≤ 2 ^ (f.m + 1) := Nat.one_le_two_pow
  unfold topSig; split <;> omega

theorem maxMag_div (hf : f.Regular) : f.maxMag / 2 ^ f.m = f.topExp := by
  have h1 := topSig_ge hf
  have h2 := f.topSig_lt
  have : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  unfold maxMag
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by positivity), Nat.div_eq_of_lt (by omega)]
  simp

theorem maxMag_mod (hf : f.Regular) : f.maxMag % 2 ^ f.m = f.topSig - 2 ^ f.m := by
  have h1 := topSig_ge hf
  have h2 := f.topSig_lt
  have : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  unfold maxMag
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]

theorem maxMag_lt (hf : f.Regular) : f.maxMag < f.signBit := by
  have h1 := topSig_ge hf
  have h2 := f.topSig_lt
  have h3 := f.topExp_lt
  have : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  have hs : f.signBit = 2 ^ f.e * 2 ^ f.m := by unfold signBit; rw [pow_add]
  unfold maxMag
  rw [hs]
  have : f.topExp * 2 ^ f.m + 2 ^ f.m ≤ 2 ^ f.e * 2 ^ f.m := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ h3
  omega

theorem magVal_maxMag (hf : f.Regular) : f.magVal f.maxMag = f.maxVal := by
  rw [magVal_eq, maxMag_div hf, maxMag_mod hf, ite_eq_right (by have := topExp_pos hf; omega)]
  unfold maxVal
  rw [pow2_eq]
  congr 2
  have := topSig_ge hf
  omega

theorem maxVal_pos (hf : f.Regular) : 0 < f.maxVal := by
  rw [← magVal_maxMag hf]
  apply magVal_pos
  have := topExp_pos hf
  have h := maxMag_div hf
  rcases Nat.eq_zero_or_pos f.maxMag with h0 | h0
  · rw [h0] at h; simp at h; omega
  · exact h0

/-- `maxVal = topSig · 2^(topExp - bias - m)`. -/
theorem maxVal_eq : f.maxVal = (f.topSig : ℚ) * (2 : ℚ) ^ ((f.topExp : ℤ) - f.bias - f.m) := by
  unfold maxVal; rw [pow2_eq]

/-- `maxVal < 2^(topExp + 1 - bias)`. -/
theorem maxVal_lt (hf : f.Regular) : f.maxVal < (2 : ℚ) ^ ((f.topExp : ℤ) + 1 - f.bias) := by
  have h := magVal_lt_of_exp (f := f) f.maxMag
  rw [magVal_maxMag hf, maxMag_div hf] at h
  exact h

theorem two_pow_e_ge (hf : f.Regular) : 4 ≤ 2 ^ f.e :=
  calc 4 = 2 ^ 2 := by norm_num
    _ ≤ 2 ^ f.e := Nat.pow_le_pow_right (by norm_num) hf.1

theorem two_pow_m_ge (hf : f.Regular) : 2 ≤ 2 ^ f.m :=
  calc 2 = 2 ^ 1 := by norm_num
    _ ≤ 2 ^ f.m := Nat.pow_le_pow_right (by norm_num) hf.2

theorem signBit_eq : f.signBit = 2 ^ f.e * 2 ^ f.m := by unfold signBit; rw [pow_add]

/-- `maxMag` in closed form: below the infinities (`.ieee`), below the NaN (`.nanOnly`), or the
all-ones magnitude (`.finite`). -/
theorem maxMag_eq (hf : f.Regular) : f.maxMag =
    match f.specials with
    | .ieee => f.signBit - 2 ^ f.m - 1
    | .nanOnly => f.signBit - 2
    | .finite => f.signBit - 1 := by
  have hQ := two_pow_e_ge hf
  have hP := two_pow_m_ge hf
  have h2 : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  have h4 : 4 * 2 ^ f.m ≤ 2 ^ f.e * 2 ^ f.m := Nat.mul_le_mul_right _ hQ
  rw [signBit_eq]
  unfold maxMag topExp topSig
  cases f.specials <;> simp only <;> rw [Nat.sub_mul] <;> omega

/-- Every finite magnitude is at most `maxMag`. -/
theorem mag_le_maxMag (hf : f.Regular) {n : ℕ} (h : f.isFinite n = true) : f.mag n ≤ f.maxMag := by
  have hQ := two_pow_e_ge hf
  have hP := two_pow_m_ge hf
  have hm := f.mag_lt n
  have hmax := maxMag_eq hf
  have h4 : 4 * 2 ^ f.m ≤ 2 ^ f.e * 2 ^ f.m := Nat.mul_le_mul_right _ hQ
  rw [signBit_eq] at hm hmax
  rw [isFinite_iff] at h
  obtain ⟨hnan, hinf⟩ := h
  have hdm := Nat.div_add_mod (f.mag n) (2 ^ f.m)
  have hF : f.mag n % 2 ^ f.m < 2 ^ f.m := Nat.mod_lt _ (by positivity)
  have hE : f.mag n / 2 ^ f.m < 2 ^ f.e := by
    rw [Nat.div_lt_iff_lt_mul (by positivity)]; exact hm
  unfold isNaN expField fracField at hnan
  unfold isInf expField fracField at hinf
  cases hsp : f.specials <;> simp only [hsp] at hnan hinf hmax
  · -- `.ieee`: the exponent field is not all ones
    have hne : f.mag n / 2 ^ f.m ≠ 2 ^ f.e - 1 := by
      intro hE1
      rw [hE1] at hnan hinf
      by_cases hz : f.mag n % 2 ^ f.m = 0 <;> simp [hz] at hnan hinf
    have hE2 : f.mag n / 2 ^ f.m + 1 ≤ 2 ^ f.e - 1 := by omega
    have h5 : 2 ^ f.m * (f.mag n / 2 ^ f.m + 1) ≤ 2 ^ f.m * (2 ^ f.e - 1) :=
      Nat.mul_le_mul_left _ hE2
    rw [Nat.mul_sub, mul_one, mul_add, mul_one, mul_comm (2 ^ f.m) (2 ^ f.e)] at h5
    rw [hmax]
    omega
  · have hne : f.mag n ≠ 2 ^ (f.e + f.m) - 1 := by simpa using hnan
    rw [pow_add] at hne
    rw [hmax]
    omega
  · rw [hmax]; omega

theorem abs_toRat_le_maxVal (hf : f.Regular) {n : ℕ} (h : f.isFinite n = true) :
    |f.toRat n| ≤ f.maxVal := by
  rw [abs_toRat, ← magVal_maxMag hf]
  exact magVal_le_magVal (mag_le_maxMag hf h)

theorem isFinite_of_mag_le (hf : f.Regular) {n : ℕ} (h : f.mag n ≤ f.maxMag) :
    f.isFinite n = true := by
  have hQ := two_pow_e_ge hf
  have hP := two_pow_m_ge hf
  have hmax := maxMag_eq hf
  have h4 : 4 * 2 ^ f.m ≤ 2 ^ f.e * 2 ^ f.m := Nat.mul_le_mul_right _ hQ
  rw [signBit_eq] at hmax
  rw [isFinite_iff]
  unfold isNaN isInf expField fracField
  cases hsp : f.specials <;> simp only [hsp] at hmax ⊢
  · have hlt : f.mag n < (2 ^ f.e - 1) * 2 ^ f.m := by
      rw [Nat.sub_mul, one_mul]; omega
    have hE : f.mag n / 2 ^ f.m < 2 ^ f.e - 1 := by
      rw [Nat.div_lt_iff_lt_mul (by positivity)]; exact hlt
    have : f.mag n / 2 ^ f.m ≠ 2 ^ f.e - 1 := by omega
    simp [this]
  · have : f.mag n ≠ 2 ^ (f.e + f.m) - 1 := by rw [pow_add]; omega
    simp [this]
  · simp

theorem isFinite_zero (hf : f.Regular) : f.isFinite 0 = true :=
  isFinite_of_mag_le hf (by unfold mag; simp)

theorem isZero_zero : f.isZero 0 = true := by unfold isZero mag; simp

theorem isInf_infMag (hs : f.specials = .ieee) : f.isInf f.infMag = true := by
  have hm : 0 < 2 ^ f.m := by positivity
  have hlt : f.infMag < f.signBit := by
    unfold infMag signBit; rw [pow_add]
    have : 1 ≤ 2 ^ f.e := Nat.one_le_two_pow
    exact Nat.mul_lt_mul_of_pos_right (by omega) hm
  unfold isInf expField fracField
  rw [hs, mag_of_lt hlt]
  unfold infMag
  simp [Nat.mul_div_cancel _ hm]

theorem isNaN_nanMag (hf : f.Regular) (hs : f.specials ≠ .finite) : f.isNaN f.nanMag = true := by
  have hm : 0 < 2 ^ f.m := by positivity
  have hlt : f.nanMag < f.signBit := by
    unfold nanMag signBit; have := (show 0 < 2 ^ (f.e + f.m) by positivity); omega
  unfold isNaN expField fracField
  rw [mag_of_lt hlt]
  have hs' : 2 ^ (f.e + f.m) = 2 ^ f.e * 2 ^ f.m := by rw [pow_add]
  have he : 1 ≤ 2 ^ f.e := Nat.one_le_two_pow
  have h2m : 2 ≤ 2 ^ f.m := by
    calc 2 = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ f.m := Nat.pow_le_pow_right (by norm_num) hf.2
  have hrepr : f.nanMag = (2 ^ f.e - 1) * 2 ^ f.m + (2 ^ f.m - 1) := by
    unfold nanMag; rw [hs', Nat.sub_mul, one_mul]
    have : 2 ^ f.m ≤ 2 ^ f.e * 2 ^ f.m := Nat.le_mul_of_pos_left _ (by positivity)
    omega
  cases hsp : f.specials
  · simp only
    rw [hrepr, Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt (by omega),
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    simp; omega
  · simp [nanMag]
  · exact absurd hsp hs

theorem not_isFinite_infMag (hs : f.specials = .ieee) : f.isFinite f.infMag = false := by
  unfold isFinite; rw [isInf_infMag hs]; simp

theorem not_isFinite_nanMag (hf : f.Regular) (hs : f.specials ≠ .finite) :
    f.isFinite f.nanMag = false := by
  unfold isFinite; rw [isNaN_nanMag hf hs]; simp

/-- **The value of `±∞`** (an `.ieee` format): `±2^(emax+1)`, `emax` the bias. -/
theorem toRat_of_isInf (hf : f.Regular) {n : ℕ} (h : f.isInf n = true) :
    f.toRat n = if f.sign n then -(2 : ℚ) ^ (f.bias + 1) else (2 : ℚ) ^ (f.bias + 1) := by
  have he1 : 1 ≤ f.e := le_trans (by norm_num) hf.1
  have hE : f.mag n / 2 ^ f.m = 2 ^ f.e - 1 ∧ f.mag n % 2 ^ f.m = 0 := by
    unfold isInf expField fracField at h
    split at h
    · simpa only [Bool.and_eq_true, beq_iff_eq] using h
    · simp at h
    · simp at h
  have h2e : 2 ≤ 2 ^ f.e := by
    calc 2 = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ f.e := Nat.pow_le_pow_right (by norm_num) he1
  have hv : f.magVal (f.mag n) = (2 : ℚ) ^ (f.bias + 1) := by
    rw [magVal_eq, ite_eq_right (by omega), hE.1, hE.2, Nat.add_zero]
    have hb : (((2 ^ f.e - 1 : ℕ) : ℤ) - f.bias - f.m) = f.bias + 1 - f.m := by
      unfold bias
      have h2 : (2 : ℤ) ^ f.e = 2 * 2 ^ (f.e - 1) := by
        rw [← pow_succ']; congr 1; omega
      push_cast [Nat.one_le_two_pow]
      rw [h2]; ring
    rw [hb]; push_cast
    rw [← zpow_natCast, ← zpow_add₀ (by norm_num)]; ring_nf
  rw [toRat_eq, hv]

/-- `±∞` lie beyond `±maxVal` (an `.ieee` format): `maxVal < 2^(emax+1)`. -/
theorem maxVal_lt_inf (hf : f.Regular) (hs : f.specials = .ieee) :
    f.maxVal < (2 : ℚ) ^ (f.bias + 1) := by
  have h := maxVal_lt hf
  unfold topExp at h; rw [hs] at h; dsimp only at h
  have he : 2 ≤ 2 ^ f.e := by
    calc 2 = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ f.e := Nat.pow_le_pow_right (by norm_num) (le_trans (by norm_num) hf.1)
  have hb : (((2 ^ f.e - 2 : ℕ) : ℤ) + 1 - f.bias) = f.bias + 1 := by
    unfold bias
    have h2 : (2 : ℤ) ^ f.e = 2 * 2 ^ (f.e - 1) := by
      rw [← pow_succ']; congr 1; have := hf.1; omega
    push_cast [he]
    rw [h2]; ring
  rwa [hb] at h

/-! ## The constants of the formats -/

theorem binary32_u : binary32.u = 2 ^ (-24 : ℤ) := by rw [u_eq]; unfold binary32; norm_num
theorem binary32_eta : binary32.eta = 2 ^ (-150 : ℤ) := by
  rw [eta_eq]; unfold emin bias binary32; norm_num
theorem binary32_emin : binary32.emin = -126 := by unfold emin bias binary32; norm_num
theorem binary32_maxVal : binary32.maxVal = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [maxVal_eq]; unfold topSig topExp bias binary32; norm_num
theorem binary16_u : binary16.u = 2 ^ (-11 : ℤ) := by rw [u_eq]; unfold binary16; norm_num
theorem binary16_eta : binary16.eta = 2 ^ (-25 : ℤ) := by rw [eta_eq]; unfold emin bias binary16; norm_num
theorem binary16_maxVal : binary16.maxVal = 65504 := by
  rw [maxVal_eq]; unfold topSig topExp bias binary16; norm_num
theorem bfloat16_u : bfloat16.u = 2 ^ (-8 : ℤ) := by rw [u_eq]; unfold bfloat16; norm_num
theorem bfloat16_eta : bfloat16.eta = 2 ^ (-134 : ℤ) := by rw [eta_eq]; unfold emin bias bfloat16; norm_num
theorem bfloat16_maxVal : bfloat16.maxVal = (2 ^ 8 - 1) * 2 ^ 120 := by
  rw [maxVal_eq]; unfold topSig topExp bias bfloat16; norm_num
theorem binary64_u : binary64.u = 2 ^ (-53 : ℤ) := by rw [u_eq]; unfold binary64; norm_num
theorem binary64_eta : binary64.eta = 2 ^ (-1075 : ℤ) := by rw [eta_eq]; unfold emin bias binary64; norm_num
theorem e4m3_u : e4m3.u = 2 ^ (-4 : ℤ) := by rw [u_eq]; unfold e4m3; norm_num
theorem e4m3_eta : e4m3.eta = 2 ^ (-10 : ℤ) := by rw [eta_eq]; unfold emin bias e4m3; norm_num
theorem e4m3_maxVal : e4m3.maxVal = 448 := by
  rw [maxVal_eq]; unfold topSig topExp bias e4m3; norm_num
theorem e5m2_u : e5m2.u = 2 ^ (-3 : ℤ) := by rw [u_eq]; unfold e5m2; norm_num
theorem e5m2_eta : e5m2.eta = 2 ^ (-17 : ℤ) := by rw [eta_eq]; unfold emin bias e5m2; norm_num
theorem e5m2_maxVal : e5m2.maxVal = 57344 := by
  rw [maxVal_eq]; unfold topSig topExp bias e5m2; norm_num
theorem e2m1_maxVal : e2m1.maxVal = 6 := by
  rw [maxVal_eq]; unfold topSig topExp bias e2m1; norm_num

end Format

end TVA.Math.Float
