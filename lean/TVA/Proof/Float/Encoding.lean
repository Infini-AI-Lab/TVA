import TVA.Core
import Mathlib.Tactic
import Mathlib.Data.Int.Log
import Mathlib.Data.Rat.Floor

/-!
# What a bit pattern means: the facts

`TVA.Base.Float.Format` defines how a pattern `n` of a binary format denotes a value: its sign,
its *magnitude* `mag n` (the exponent and fraction fields), the value `magVal` of the magnitude,
and `toRat n`, the magnitude's value negated by the sign. This file proves what every float
proof needs about that encoding:

* `pow2_eq`: the definitions compute `2^k` as `pow2 k`; rewrite with it to use Mathlib's powers.
  `u_eq`, `eta_eq` do the same for `u = 2^-(m+1)` and `η = 2^(emin-m-1)`.
* Sign and magnitude: `setSign`, `mag` and how `isNaN`, `isInf`, `isFinite`, `isZero`, `toRat`
  treat them (`toRat_setSign`, `isFinite_setSign`, `eq_setSign`, …).
* `magVal` is strictly increasing (`magVal_lt_magVal`), so the order of magnitudes is the order of
  values, and every value is `S · 2^E` with `S < 2^(m+1)` (`magVal_sig_exp`).
* The order of values is the order of the integer keys `orderKey`, for every pattern that is not
  NaN, infinities included (`toRat_lt_iff_orderKey`): `tl.maximum`, comparisons and sort keys
  rely on it.
* The largest finite value `maxVal`: every finite pattern is at most it in magnitude
  (`abs_toRat_le_maxVal`), and a pattern whose magnitude is at most `maxMag` is finite
  (`isFinite_of_mag_le`).
* Every format of Triton is `Regular` (`binary32_regular`, …); the error bounds assume it.

The constants of each format, and the infinities and NaN, are in
`TVA.Proof.Float.Constants`; rounding is `TVA.Proof.Float.Quantum` and `TVA.Proof.Float.Round`.
-/
namespace TVA.Format

/-- `pow2 k` (the exact rational `2^k` that the definitions compute with) is `2 ^ k`; rewrite with
it before using Mathlib's power lemmas. -/
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

/-- `binary16` (`tl.float16`) is `Regular`. -/
theorem binary16_regular : binary16.Regular := by decide
/-- `bfloat16` is `Regular`. -/
theorem bfloat16_regular : bfloat16.Regular := by decide
/-- `binary32` (`tl.float32`) is `Regular`. -/
theorem binary32_regular : binary32.Regular := by decide
/-- `binary64` (`tl.float64`) is `Regular`. -/
theorem binary64_regular : binary64.Regular := by decide
/-- `e5m2` (`tl.float8e5`) is `Regular`. -/
theorem e5m2_regular : e5m2.Regular := by decide
/-- `e4m3` (`tl.float8e4nv`) is `Regular`. -/
theorem e4m3_regular : e4m3.Regular := by decide
/-- `e2m1` (the fp4 of `tl.dot_scaled`) is `Regular`. -/
theorem e2m1_regular : e2m1.Regular := by decide

/-! ## Basic facts -/

variable {f : Format}

/-- `u = 2^-(m+1)` as a power of two. -/
theorem u_eq : f.u = (2 : ℚ) ^ (-(f.m : ℤ) - 1) := pow2_eq _
/-- `η = 2^(emin-m-1)` as a power of two. -/
theorem eta_eq : f.eta = (2 : ℚ) ^ (f.emin - f.m - 1) := pow2_eq _

/-- `u ≥ 0`. -/
theorem u_nonneg : 0 ≤ f.u := by rw [u_eq]; positivity
/-- `η ≥ 0`. -/
theorem eta_nonneg : 0 ≤ f.eta := by rw [eta_eq]; positivity
/-- `u > 0`. -/
theorem u_pos : 0 < f.u := by rw [u_eq]; positivity
/-- `η > 0`. -/
theorem eta_pos : 0 < f.eta := by rw [eta_eq]; positivity

/-- The sign bit's weight is positive. -/
theorem signBit_pos : 0 < f.signBit := by unfold signBit; positivity

/-- Twice the sign bit's weight is `2^width`. -/
theorem two_signBit : 2 * f.signBit = 2 ^ f.width := by
  unfold signBit width; rw [pow_succ]; ring

/-- The sign bit's weight is below `2^width`. -/
theorem signBit_lt : f.signBit < 2 ^ f.width := by
  have := f.two_signBit; have := f.signBit_pos; omega

/-- A magnitude is below the sign bit's weight. -/
theorem mag_lt (n : ℕ) : f.mag n < f.signBit := Nat.mod_lt _ signBit_pos

/-- A pattern's magnitude is at most the pattern. -/
theorem mag_le (n : ℕ) : f.mag n ≤ n := Nat.mod_le _ _

/-- A pattern without the sign bit is its own magnitude. -/
theorem mag_of_lt {n : ℕ} (h : n < f.signBit) : f.mag n = n := Nat.mod_eq_of_lt h

/-- Taking the magnitude twice changes nothing. -/
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

/-- Setting the sign keeps the magnitude. -/
theorem mag_setSign (s : Bool) (n : ℕ) : f.mag (f.setSign s n) = f.mag n := by
  unfold setSign
  cases s
  · simp [mag_mag]
  · simp only [ite_true]
    unfold mag
    rw [Nat.add_mod_left, Nat.mod_mod]

/-- `setSign s` gives sign `s`. -/
theorem sign_setSign (s : Bool) (n : ℕ) : f.sign (f.setSign s n) = s := by
  have hm := f.mag_lt n
  have hpos := f.signBit_pos
  unfold setSign sign
  cases s
  · simp [Nat.div_eq_of_lt hm]
  · simp only [ite_true]
    rw [Nat.add_comm, Nat.add_div_right _ hpos, Nat.div_eq_of_lt hm]
    rfl

/-- A pattern with a set sign fits in the format's width. -/
theorem setSign_lt (s : Bool) (n : ℕ) : f.setSign s n < 2 ^ f.width := by
  have hm := f.mag_lt n
  have h2 := f.two_signBit
  unfold setSign
  split_ifs <;> omega

/-- `setSign s M` of a magnitude `M` is `M` plus the sign bit when `s`. -/
theorem setSign_of_lt {s : Bool} {M : ℕ} (hM : M < f.signBit) :
    f.setSign s M = (if s then f.signBit else 0) + M := by
  unfold setSign; rw [mag_of_lt hM]

/-- Setting the sign keeps NaN-ness. -/
theorem isNaN_setSign (s : Bool) (n : ℕ) : f.isNaN (f.setSign s n) = f.isNaN n := by
  unfold isNaN expField fracField; rw [mag_setSign]

/-- Setting the sign keeps infiniteness. -/
theorem isInf_setSign (s : Bool) (n : ℕ) : f.isInf (f.setSign s n) = f.isInf n := by
  unfold isInf expField fracField; rw [mag_setSign]

/-- Setting the sign keeps finiteness. -/
theorem isFinite_setSign (s : Bool) (n : ℕ) : f.isFinite (f.setSign s n) = f.isFinite n := by
  unfold isFinite; rw [isNaN_setSign, isInf_setSign]

/-- Setting the sign keeps zeroness. -/
theorem isZero_setSign (s : Bool) (n : ℕ) : f.isZero (f.setSign s n) = f.isZero n := by
  unfold isZero; rw [mag_setSign]

/-- The value of `setSign s n`: the magnitude's value, negated when `s`. -/
theorem toRat_setSign (s : Bool) (n : ℕ) :
    f.toRat (f.setSign s n) = if s then -f.magVal (f.mag n) else f.magVal (f.mag n) := by
  unfold toRat; rw [sign_setSign, mag_setSign]

/-- A pattern is NaN exactly when its magnitude is. -/
theorem isNaN_mag (n : ℕ) : f.isNaN (f.mag n) = f.isNaN n := by
  unfold isNaN expField fracField; rw [mag_mag]

/-- A pattern is infinite exactly when its magnitude is. -/
theorem isInf_mag (n : ℕ) : f.isInf (f.mag n) = f.isInf n := by
  unfold isInf expField fracField; rw [mag_mag]

/-- A pattern is finite exactly when its magnitude is. -/
theorem isFinite_mag (n : ℕ) : f.isFinite (f.mag n) = f.isFinite n := by
  unfold isFinite; rw [isNaN_mag, isInf_mag]

/-- Finite means neither NaN nor infinite. -/
theorem isFinite_iff (n : ℕ) : f.isFinite n = true ↔ f.isNaN n = false ∧ f.isInf n = false := by
  unfold isFinite; cases f.isNaN n <;> cases f.isInf n <;> simp

/-- A finite pattern is not NaN. -/
theorem not_isNaN_of_isFinite {n : ℕ} (h : f.isFinite n = true) : f.isNaN n = false :=
  ((isFinite_iff n).mp h).1

/-- A finite pattern is not infinite. -/
theorem not_isInf_of_isFinite {n : ℕ} (h : f.isFinite n = true) : f.isInf n = false :=
  ((isFinite_iff n).mp h).2

/-! ## Values of magnitudes -/

/-- The value of a magnitude with `pow2` unfolded into Mathlib powers: subnormal when the exponent field is `0`, normal otherwise. -/
theorem magVal_eq (M : ℕ) : f.magVal M =
    if M / 2 ^ f.m = 0 then ((M % 2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ (f.emin - f.m)
    else ((2 ^ f.m + M % 2 ^ f.m : ℕ) : ℚ) * (2 : ℚ) ^ (((M / 2 ^ f.m : ℕ) : ℤ) - f.bias - f.m) := by
  unfold magVal; rw [pow2_eq, pow2_eq]

/-- A magnitude's value is nonnegative. -/
theorem magVal_nonneg (M : ℕ) : 0 ≤ f.magVal M := by
  rw [magVal_eq]; split_ifs <;> positivity

/-- The magnitude `0` has value `0`. -/
theorem magVal_zero : f.magVal 0 = 0 := by rw [magVal_eq]; simp

/-- `emin = 1 - bias`, by definition. -/
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

/-- `magVal` is monotone. -/
theorem magVal_le_magVal {M M' : ℕ} (h : M ≤ M') : f.magVal M ≤ f.magVal M' := by
  rcases h.lt_or_eq with h | h
  · exact (magVal_lt_magVal h).le
  · rw [h]

/-- Magnitudes compare as their values. -/
theorem magVal_lt_iff {M M' : ℕ} : f.magVal M < f.magVal M' ↔ M < M' :=
  ⟨fun h => by by_contra h'; exact absurd h (not_lt.mpr (magVal_le_magVal (not_lt.mp h'))),
    magVal_lt_magVal⟩

/-- `magVal` is injective. -/
theorem magVal_inj {M M' : ℕ} (h : f.magVal M = f.magVal M') : M = M' := by
  rcases lt_trichotomy M M' with h' | h' | h'
  · exact absurd h (magVal_lt_magVal h').ne
  · exact h'
  · exact absurd h.symm (magVal_lt_magVal h').ne

/-- A nonzero magnitude has a positive value. -/
theorem magVal_pos {M : ℕ} (h : 0 < M) : 0 < f.magVal M := by
  have := magVal_lt_magVal (f := f) h; rwa [magVal_zero] at this

/-- Only the magnitude `0` has value `0`. -/
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

/-- The value of a pattern, by definition: its magnitude's value, negated when the sign is
set. -/
theorem toRat_eq (n : ℕ) : f.toRat n = if f.sign n then -f.magVal (f.mag n) else f.magVal (f.mag n) :=
  rfl

/-- The absolute value of a pattern's value is its magnitude's value. -/
theorem abs_toRat (n : ℕ) : |f.toRat n| = f.magVal (f.mag n) := by
  unfold toRat; split_ifs
  · rw [abs_neg, abs_of_nonneg (magVal_nonneg _)]
  · rw [abs_of_nonneg (magVal_nonneg _)]

/-- A zero pattern (`±0`) has value `0`. -/
theorem toRat_of_isZero {n : ℕ} (h : f.isZero n = true) : f.toRat n = 0 := by
  unfold isZero at h
  have : f.mag n = 0 := by simpa using h
  unfold toRat; rw [this, magVal_zero]; simp

/-- A pattern has value `0` exactly when it is `±0`. -/
theorem toRat_eq_zero_iff {n : ℕ} : f.toRat n = 0 ↔ f.isZero n = true := by
  constructor
  · intro h
    have : |f.toRat n| = 0 := by rw [h, abs_zero]
    rw [abs_toRat, magVal_eq_zero_iff] at this
    unfold isZero; simp [this]
  · exact toRat_of_isZero

/-- The value of a magnitude (sign cleared) is its `magVal`. -/
theorem toRat_mag (n : ℕ) : f.toRat (f.mag n) = f.magVal (f.mag n) := by
  have : f.sign (f.mag n) = false := by
    unfold sign; rw [Nat.div_eq_of_lt (mag_lt n)]; rfl
  unfold toRat; rw [this, mag_mag]; rfl

/-- **Values are ordered as order keys**, for every pattern (the infinities too: their formula
values lie beyond `maxVal`). Use it to turn a comparison of values into one of integers. -/
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

/-- Two patterns have equal values exactly when their order keys are equal (`-0` and `+0` both
have key `0`). -/
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

/-! ## The largest value -/

/-- The width of a pattern, `e + m + 1` bits, by definition. -/
theorem width_eq : f.width = f.e + f.m + 1 := rfl

/-- The bias is nonnegative. -/
theorem bias_nonneg : 0 ≤ f.bias := by
  unfold bias
  have : (1 : ℤ) ≤ 2 ^ (f.e - 1) := one_le_pow₀ (by norm_num)
  linarith

/-- The largest finite exponent field is at least `1`. -/
theorem topExp_pos (hf : f.Regular) : 1 ≤ f.topExp := by
  obtain ⟨he, -⟩ := hf
  have : 4 ≤ 2 ^ f.e := by
    calc 4 = 2 ^ 2 := by norm_num
      _ ≤ 2 ^ f.e := Nat.pow_le_pow_right (by norm_num) he
  unfold topExp; split <;> omega

/-- The largest finite exponent field fits in `e` bits. -/
theorem topExp_lt : f.topExp < 2 ^ f.e := by
  have : 1 ≤ 2 ^ f.e := Nat.one_le_two_pow
  unfold topExp; split <;> omega

/-- The largest finite significand is at least `2^m`. -/
theorem topSig_ge (hf : f.Regular) : 2 ^ f.m ≤ f.topSig := by
  obtain ⟨-, hm⟩ := hf
  have h2 : 2 ≤ 2 ^ f.m := by
    calc 2 = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ f.m := Nat.pow_le_pow_right (by norm_num) hm
  have : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  unfold topSig; split <;> omega

/-- The largest finite significand is below `2^(m+1)`. -/
theorem topSig_lt : f.topSig < 2 ^ (f.m + 1) := by
  have : 1 ≤ 2 ^ (f.m + 1) := Nat.one_le_two_pow
  unfold topSig; split <;> omega

/-- `maxMag`'s exponent field is `topExp`. -/
theorem maxMag_div (hf : f.Regular) : f.maxMag / 2 ^ f.m = f.topExp := by
  have h1 := topSig_ge hf
  have h2 := f.topSig_lt
  have : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  unfold maxMag
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by positivity), Nat.div_eq_of_lt (by omega)]
  simp

/-- `maxMag`'s fraction field is `topSig - 2^m`. -/
theorem maxMag_mod (hf : f.Regular) : f.maxMag % 2 ^ f.m = f.topSig - 2 ^ f.m := by
  have h1 := topSig_ge hf
  have h2 := f.topSig_lt
  have : 2 ^ (f.m + 1) = 2 * 2 ^ f.m := by rw [pow_succ]; ring
  unfold maxMag
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]

/-- `maxMag` is a magnitude (below the sign bit). -/
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

/-- `maxMag` has the value `maxVal`. -/
theorem magVal_maxMag (hf : f.Regular) : f.magVal f.maxMag = f.maxVal := by
  rw [magVal_eq, maxMag_div hf, maxMag_mod hf, ite_eq_right (by have := topExp_pos hf; omega)]
  unfold maxVal
  rw [pow2_eq]
  congr 2
  have := topSig_ge hf
  omega

/-- `maxVal` is positive. -/
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

/-- A regular format has `2^e ≥ 4`. -/
theorem two_pow_e_ge (hf : f.Regular) : 4 ≤ 2 ^ f.e :=
  calc 4 = 2 ^ 2 := by norm_num
    _ ≤ 2 ^ f.e := Nat.pow_le_pow_right (by norm_num) hf.1

/-- A regular format has `2^m ≥ 2`. -/
theorem two_pow_m_ge (hf : f.Regular) : 2 ≤ 2 ^ f.m :=
  calc 2 = 2 ^ 1 := by norm_num
    _ ≤ 2 ^ f.m := Nat.pow_le_pow_right (by norm_num) hf.2

/-- The sign bit's weight is `2^e · 2^m`. -/
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

/-- **A finite value is at most `maxVal` in magnitude.** -/
theorem abs_toRat_le_maxVal (hf : f.Regular) {n : ℕ} (h : f.isFinite n = true) :
    |f.toRat n| ≤ f.maxVal := by
  rw [abs_toRat, ← magVal_maxMag hf]
  exact magVal_le_magVal (mag_le_maxMag hf h)

/-- A pattern whose magnitude is at most `maxMag` is finite. -/
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

/-- The pattern `0` (`+0`) is finite. -/
theorem isFinite_zero (hf : f.Regular) : f.isFinite 0 = true :=
  isFinite_of_mag_le hf (by unfold mag; simp)

/-- The pattern `0` is a zero (`+0`). -/
theorem isZero_zero : f.isZero 0 = true := by unfold isZero mag; simp

end TVA.Format
