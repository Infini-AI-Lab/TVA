import TVA.Proof.Float.Encoding

/-!
# The constants of the formats

For each format Triton uses, its `u` (half an ulp of `1`: the relative error of one rounding to
nearest), `η` (`eta`, half the smallest subnormal: the absolute error of one rounding below the
normal range) and `maxVal` (the largest finite value), as closed numbers that `norm_num`
evaluates:

| dtype | format | `u` | `η` | `maxVal` |
|---|---|---|---|---|
| `tl.float32` | `binary32` | `2^-24` | `2^-150` | `(2^24 − 1)·2^104` |
| `tl.float16` | `binary16` | `2^-11` | `2^-25` | `65504` |
| `tl.bfloat16` | `bfloat16` | `2^-8` | `2^-134` | `(2^8 − 1)·2^120` |
| `tl.float64` | `binary64` | `2^-53` | `2^-1075` | `(2^53 − 1)·2^971` |
| `tl.float8e5` | `e5m2` | `2^-3` | `2^-17` | `57344` |
| `tl.float8e4nv` | `e4m3` | `2^-4` | `2^-10` | `448` |
| fp4 of `tl.dot_scaled` | `e2m1` | `2^-2` | `2^-2` | `6` |

Each comes in `ℚ` (`binary32_u`) and in `ℝ` (`binary32_u_real`, the cast of the rational
constant, for bounds stated on `FloatV.toReal`). They are not `@[simp]`: rewrite with them by
name when a bound must become a number.

The infinities and the NaN: `infMag` is infinite and `nanMag` a NaN (`isInf_infMag`,
`isNaN_nanMag`); `±∞` has the formula value `±2^(bias+1)` (`toRat_of_isInf`), beyond `maxVal`
(`maxVal_lt_inf`), so comparing values orders them below and above every number.

Also: the standard formats (`Format.Std`: `float16`, `bfloat16`, `float32`, `float64`) are
`Regular`, have infinities and are not fp8 (`Std.regular`, `Std.ieee`, `Std.not_f8`).
-/
namespace TVA.Format

/-! ## In `ℚ` -/

/-- `u` of `binary32` (`tl.float32`): `2^-24`. -/
theorem binary32_u : binary32.u = 2 ^ (-24 : ℤ) := by rw [u_eq]; unfold binary32; norm_num

/-- `η` of `binary32` (`tl.float32`): `2^-150`. -/
theorem binary32_eta : binary32.eta = 2 ^ (-150 : ℤ) := by
  rw [eta_eq]; unfold emin bias binary32; norm_num

/-- `maxVal` of `binary32` (`tl.float32`): `(2^24-1)·2^104`. -/
theorem binary32_maxVal : binary32.maxVal = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [maxVal_eq]; unfold topSig topExp bias binary32; norm_num

/-- `emin` of `binary32`: `-126` (the exponent of the smallest normal value `2^-126`). -/
theorem binary32_emin : binary32.emin = -126 := by unfold emin bias binary32; norm_num

/-- `u` of `binary16` (`tl.float16`): `2^-11`. -/
theorem binary16_u : binary16.u = 2 ^ (-11 : ℤ) := by rw [u_eq]; unfold binary16; norm_num

/-- `η` of `binary16` (`tl.float16`): `2^-25`. -/
theorem binary16_eta : binary16.eta = 2 ^ (-25 : ℤ) := by
  rw [eta_eq]; unfold emin bias binary16; norm_num

/-- `maxVal` of `binary16` (`tl.float16`): `65504`. -/
theorem binary16_maxVal : binary16.maxVal = 65504 := by
  rw [maxVal_eq]; unfold topSig topExp bias binary16; norm_num

/-- `u` of `bfloat16` (`tl.bfloat16`): `2^-8`. -/
theorem bfloat16_u : bfloat16.u = 2 ^ (-8 : ℤ) := by rw [u_eq]; unfold bfloat16; norm_num

/-- `η` of `bfloat16` (`tl.bfloat16`): `2^-134`. -/
theorem bfloat16_eta : bfloat16.eta = 2 ^ (-134 : ℤ) := by
  rw [eta_eq]; unfold emin bias bfloat16; norm_num

/-- `maxVal` of `bfloat16` (`tl.bfloat16`): `(2^8-1)·2^120`. -/
theorem bfloat16_maxVal : bfloat16.maxVal = (2 ^ 8 - 1) * 2 ^ 120 := by
  rw [maxVal_eq]; unfold topSig topExp bias bfloat16; norm_num

/-- `u` of `binary64` (`tl.float64`): `2^-53`. -/
theorem binary64_u : binary64.u = 2 ^ (-53 : ℤ) := by rw [u_eq]; unfold binary64; norm_num

/-- `η` of `binary64` (`tl.float64`): `2^-1075`. -/
theorem binary64_eta : binary64.eta = 2 ^ (-1075 : ℤ) := by
  rw [eta_eq]; unfold emin bias binary64; norm_num

/-- `u` of `e5m2` (`tl.float8e5`): `2^-3`. -/
theorem e5m2_u : e5m2.u = 2 ^ (-3 : ℤ) := by rw [u_eq]; unfold e5m2; norm_num

/-- `η` of `e5m2` (`tl.float8e5`): `2^-17`. -/
theorem e5m2_eta : e5m2.eta = 2 ^ (-17 : ℤ) := by
  rw [eta_eq]; unfold emin bias e5m2; norm_num

/-- `maxVal` of `e5m2` (`tl.float8e5`): `57344`. -/
theorem e5m2_maxVal : e5m2.maxVal = 57344 := by
  rw [maxVal_eq]; unfold topSig topExp bias e5m2; norm_num

/-- `u` of `e4m3` (`tl.float8e4nv`): `2^-4`. -/
theorem e4m3_u : e4m3.u = 2 ^ (-4 : ℤ) := by rw [u_eq]; unfold e4m3; norm_num

/-- `η` of `e4m3` (`tl.float8e4nv`): `2^-10`. -/
theorem e4m3_eta : e4m3.eta = 2 ^ (-10 : ℤ) := by
  rw [eta_eq]; unfold emin bias e4m3; norm_num

/-- `maxVal` of `e4m3` (`tl.float8e4nv`): `448`. -/
theorem e4m3_maxVal : e4m3.maxVal = 448 := by
  rw [maxVal_eq]; unfold topSig topExp bias e4m3; norm_num

/-- `maxVal` of `e2m1` (the fp4 of `tl.dot_scaled`): `6`. -/
theorem e2m1_maxVal : e2m1.maxVal = 6 := by
  rw [maxVal_eq]; unfold topSig topExp bias e2m1; norm_num

/-- `maxVal` of `binary64`: `(2^53 − 1)·2^971`. -/
theorem binary64_maxVal : binary64.maxVal = (2 ^ 53 - 1) * 2 ^ 971 := by
  rw [maxVal_eq]; unfold topSig topExp bias binary64; norm_num

/-- `u` of `e2m1`: `2^-2`. -/
theorem e2m1_u : e2m1.u = 2 ^ (-2 : ℤ) := by rw [u_eq]; unfold e2m1; norm_num

/-- `η` of `e2m1`: `2^-2`. -/
theorem e2m1_eta : e2m1.eta = 2 ^ (-2 : ℤ) := by
  rw [eta_eq]; unfold emin bias e2m1; norm_num

/-! ## In `ℝ` -/

/-- `u` of `binary32` in `ℝ`: `2^-24`. -/
theorem binary32_u_real : ((binary32.u : ℚ) : ℝ) = 2 ^ (-24 : ℤ) := by rw [binary32_u]; push_cast; rfl
/-- `η` of `binary32` in `ℝ`: `2^-150`. -/
theorem binary32_eta_real : ((binary32.eta : ℚ) : ℝ) = 2 ^ (-150 : ℤ) := by
  rw [binary32_eta]; push_cast; rfl
/-- `maxVal` of `binary32` in `ℝ`: `(2^24 − 1)·2^104`. -/
theorem binary32_maxVal_real : ((binary32.maxVal : ℚ) : ℝ) = (2 ^ 24 - 1) * 2 ^ 104 := by
  rw [binary32_maxVal]; push_cast; rfl

/-- `u` of `binary16` in `ℝ`: `2^-11`. -/
theorem binary16_u_real : ((binary16.u : ℚ) : ℝ) = 2 ^ (-11 : ℤ) := by rw [binary16_u]; push_cast; rfl
/-- `η` of `binary16` in `ℝ`: `2^-25`. -/
theorem binary16_eta_real : ((binary16.eta : ℚ) : ℝ) = 2 ^ (-25 : ℤ) := by
  rw [binary16_eta]; push_cast; rfl
/-- `maxVal` of `binary16` in `ℝ`: `65504`. -/
theorem binary16_maxVal_real : ((binary16.maxVal : ℚ) : ℝ) = 65504 := by
  rw [binary16_maxVal]; push_cast; rfl

/-- `u` of `bfloat16` in `ℝ`: `2^-8`. -/
theorem bfloat16_u_real : ((bfloat16.u : ℚ) : ℝ) = 2 ^ (-8 : ℤ) := by rw [bfloat16_u]; push_cast; rfl
/-- `η` of `bfloat16` in `ℝ`: `2^-134`. -/
theorem bfloat16_eta_real : ((bfloat16.eta : ℚ) : ℝ) = 2 ^ (-134 : ℤ) := by
  rw [bfloat16_eta]; push_cast; rfl
/-- `maxVal` of `bfloat16` in `ℝ`: `(2^8 − 1)·2^120`. -/
theorem bfloat16_maxVal_real : ((bfloat16.maxVal : ℚ) : ℝ) = (2 ^ 8 - 1) * 2 ^ 120 := by
  rw [bfloat16_maxVal]; push_cast; rfl

/-- `u` of `binary64` in `ℝ`: `2^-53`. -/
theorem binary64_u_real : ((binary64.u : ℚ) : ℝ) = 2 ^ (-53 : ℤ) := by rw [binary64_u]; push_cast; rfl
/-- `η` of `binary64` in `ℝ`: `2^-1075`. -/
theorem binary64_eta_real : ((binary64.eta : ℚ) : ℝ) = 2 ^ (-1075 : ℤ) := by
  rw [binary64_eta, Rat.cast_zpow, Rat.cast_ofNat]
/-- `maxVal` of `binary64` in `ℝ`: `(2^53 − 1)·2^971`. -/
theorem binary64_maxVal_real : ((binary64.maxVal : ℚ) : ℝ) = (2 ^ 53 - 1) * 2 ^ 971 := by
  rw [binary64_maxVal]; push_cast; rfl

/-- `u` of `e5m2` in `ℝ`: `2^-3`. -/
theorem e5m2_u_real : ((e5m2.u : ℚ) : ℝ) = 2 ^ (-3 : ℤ) := by rw [e5m2_u]; push_cast; rfl
/-- `η` of `e5m2` in `ℝ`: `2^-17`. -/
theorem e5m2_eta_real : ((e5m2.eta : ℚ) : ℝ) = 2 ^ (-17 : ℤ) := by rw [e5m2_eta]; push_cast; rfl
/-- `maxVal` of `e5m2` in `ℝ`: `57344`. -/
theorem e5m2_maxVal_real : ((e5m2.maxVal : ℚ) : ℝ) = 57344 := by rw [e5m2_maxVal]; push_cast; rfl

/-- `u` of `e4m3` in `ℝ`: `2^-4`. -/
theorem e4m3_u_real : ((e4m3.u : ℚ) : ℝ) = 2 ^ (-4 : ℤ) := by rw [e4m3_u]; push_cast; rfl
/-- `η` of `e4m3` in `ℝ`: `2^-10`. -/
theorem e4m3_eta_real : ((e4m3.eta : ℚ) : ℝ) = 2 ^ (-10 : ℤ) := by rw [e4m3_eta]; push_cast; rfl
/-- `maxVal` of `e4m3` in `ℝ`: `448`. -/
theorem e4m3_maxVal_real : ((e4m3.maxVal : ℚ) : ℝ) = 448 := by rw [e4m3_maxVal]; push_cast; rfl

/-- `u` of `e2m1` in `ℝ`: `2^-2`. -/
theorem e2m1_u_real : ((e2m1.u : ℚ) : ℝ) = 2 ^ (-2 : ℤ) := by rw [e2m1_u]; push_cast; rfl
/-- `η` of `e2m1` in `ℝ`: `2^-2`. -/
theorem e2m1_eta_real : ((e2m1.eta : ℚ) : ℝ) = 2 ^ (-2 : ℤ) := by rw [e2m1_eta]; push_cast; rfl
/-- `maxVal` of `e2m1` in `ℝ`: `6`. -/
theorem e2m1_maxVal_real : ((e2m1.maxVal : ℚ) : ℝ) = 6 := by rw [e2m1_maxVal]; push_cast; rfl

/-! ## The standard formats -/

/-- `binary16` (`float16`) is `Std`. -/
theorem binary16_std : binary16.Std := Or.inl rfl
/-- `bfloat16` is `Std`. -/
theorem bfloat16_std : bfloat16.Std := Or.inr (Or.inl rfl)
/-- `binary32` is `Std`. -/
theorem binary32_std : binary32.Std := Or.inr (Or.inr (Or.inl rfl))
/-- `binary64` is `Std`. -/
theorem binary64_std : binary64.Std := Or.inr (Or.inr (Or.inr rfl))

/-- A standard format is `Regular`. -/
theorem Std.regular {f : Format} (h : f.Std) : f.Regular := by
  rcases h with rfl | rfl | rfl | rfl <;> decide

/-- A standard format has infinities (`.ieee`). -/
theorem Std.ieee {f : Format} (h : f.Std) : f.specials = .ieee := by
  rcases h with rfl | rfl | rfl | rfl <;> rfl

/-- A standard format is not an fp8 format. -/
theorem Std.not_f8 {f : Format} (h : f.Std) : ¬ f.F8 := by
  rintro (h8 | h8) <;> rcases h with rfl | rfl | rfl | rfl <;> exact absurd h8 (by decide)

/-! ## The infinities and NaN -/

section Specials

variable {f : Format}

/-- In a format with infinities, `infMag` (all-ones exponent, zero fraction) is infinite. -/
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

/-- The canonical NaN magnitude `nanMag` is a NaN, in every format that has a NaN. -/
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

/-- In a format with infinities, `infMag` is not finite. -/
theorem not_isFinite_infMag (hs : f.specials = .ieee) : f.isFinite f.infMag = false := by
  unfold isFinite; rw [isInf_infMag hs]; simp

/-- The canonical NaN magnitude is not finite. -/
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

end Specials

end TVA.Format
