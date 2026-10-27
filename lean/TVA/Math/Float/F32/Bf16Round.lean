import TVA.Math.Float.F32.GemmErr

/-!
# Rounding `f32 → bf16` (RNE): `bf16_round_err`, the `P` rounding of the flash-attention kernels

Part of the `f32` error analysis (`TVA.Math.Float.F32.Error`); the definitions it reasons about are in
`TVA.Math.Float.F32.Bounds`. Split out of `F32/Bounds.lean`.
-/
set_option autoImplicit false

namespace TVA.Math.Float.F32

open TVA.Math.Float

open Rat

/-! ### Rounding `f32 → bf16` (the `P` rounding of the flash-attention kernels)

`TVA.Math.Float.f32ToBf16RNE` keeps the top 16 bits of the f32 pattern after adding
`0x7FFF + lsb` (round to nearest, ties to even on the 8-bit significand);
`bf16ToF32Bits` widens back exactly. bf16 has an 8-bit significand, so the
relative error of this rounding is at most `2^-8` (not `2^-9`: the bound is
tight at `x = 1 + 2^-8 - ε`). -/

/-! The bit arithmetic of the rounding, as standalone facts (small contexts keep `omega` happy):
`v` is the f32 pattern, `c ∈ {0,1}` the rounding carry, `r = v/2^16 + c` the bf16 pattern. -/

theorem bf16_carry_le (v : ℕ) : (v % 65536 + 32767 + v / 65536 % 2) / 65536 ≤ 1 := by omega

theorem bf16_nowrap (v : ℕ) (hv : v < 4294967296) (hE : v / 8388608 % 256 ≠ 255) :
    v + 32767 + v / 65536 % 2 < 4294967296 := by omega

theorem bf16_sign (v : ℕ) (hv : v < 4294967296) (hE : v / 8388608 % 256 ≠ 255) (c : ℕ) (hc : c ≤ 1) :
    ((v / 65536 + c) * 65536 ≥ 2 ^ 31 ↔ v ≥ 2 ^ 31) := by omega

theorem bf16_exp_ne (v : ℕ) (_hv : v < 4294967296) (hE : v / 8388608 % 256 ≠ 255)
    (hE0 : v / 8388608 % 256 ≠ 0) (c : ℕ) (hc : c ≤ 1) : (v / 65536 + c) / 128 % 256 ≠ 0 := by omega

theorem bf16_carry_case (v : ℕ) (_hv : v < 4294967296) (hE : v / 8388608 % 256 ≠ 255) (c : ℕ) (hc : c ≤ 1)
    (hres : (v / 65536 + c) / 128 % 256 ≠ 255) (hover : v / 65536 % 128 + c = 128) :
    (v / 65536 + c) / 128 % 256 = v / 8388608 % 256 + 1 ∧ (v / 65536 + c) % 128 = 0 := by omega

theorem bf16_nocarry_case (v : ℕ) (c : ℕ) (hc : c ≤ 1) (hover : v / 65536 % 128 + c ≠ 128) :
    (v / 65536 + c) / 128 % 256 = v / 8388608 % 256 ∧ (v / 65536 + c) % 128 = v / 65536 % 128 + c := by
  omega

theorem bf16_close (v : ℕ) :
    (v / 65536 % 128 + (v % 65536 + 32767 + v / 65536 % 2) / 65536) * 65536 ≤ v % 8388608 + 32768 ∧
    v % 8388608 ≤ (v / 65536 % 128 + (v % 65536 + 32767 + v / 65536 % 2) / 65536) * 65536 + 32768 := by
  have h := Nat.div_div_eq_div_mul v 65536 128
  norm_num at h
  omega

theorem bf16_r16_lt (v : ℕ) (hv : v < 4294967296) (hE : v / 8388608 % 256 ≠ 255) (c : ℕ) (hc : c ≤ 1) :
    v / 65536 + c < 65536 := by
  have h := Nat.div_div_eq_div_mul v 65536 128
  norm_num at h
  omega

/-- **bf16 rounding error**: for a normal finite `x` whose bf16 rounding is finite,
`|bf16(x) − x| ≤ 2^-8·|x|`. -/
theorem bf16_round_err (x : Bits) (hx : isFinite x = true) (hn : expField x ≠ 0)
    (hres : isFinite (roundBf16 x) = true) :
    |toRat (roundBf16 x) - toRat x| ≤ (2 : ℚ) ^ (-8 : ℤ) * |toRat x| := by
  set v := x.toNat with hv
  have hv32 : v < 4294967296 := x.isLt
  have hE : expField x = v / 8388608 % 256 := expField_eq x
  have hM : mantField x = v % 8388608 := rfl
  have hx' : v / 8388608 % 256 ≠ 255 := by
    have : expField x ≠ 255 := by simpa [isFinite] using hx
    rwa [hE] at this
  have hn' : v / 8388608 % 256 ≠ 0 := by rwa [hE] at hn
  have hnotnan : f32IsNaN v = false := by
    simp only [f32IsNaN, f32Fields, floatFields, FloatFields.isNaN, Nat.shiftRight_eq_div_pow]
    simp [hx']
  -- the rounded 16-bit pattern: `hi + carry`
  set c := (v % 65536 + 32767 + v / 65536 % 2) / 65536 with hc
  have hc1 : c ≤ 1 := bf16_carry_le v
  have hr16 : f32ToBf16RNE v = v / 65536 + c := by
    unfold f32ToBf16RNE
    simp only [Nat.mod_eq_of_lt hv32, hnotnan, Bool.false_eq_true, ↓reduceIte, Nat.shiftRight_eq_div_pow]
    rw [Nat.mod_eq_of_lt (by have := bf16_nowrap v hv32 hx'; omega)]
    rw [show v + 32767 + v / 2 ^ 16 % 2 = 65536 * (v / 65536) + (v % 65536 + 32767 + v / 65536 % 2) by
      norm_num; omega]
    rw [Nat.mul_add_div (by norm_num)]
  set r16 := f32ToBf16RNE v with hr16def
  have hr16lt : r16 < 65536 := by rw [hr16]; exact bf16_r16_lt v hv32 hx' c hc1
  have hbf : bf16ToF32Bits r16 = r16 * 65536 := by
    unfold bf16ToF32Bits; rw [Nat.mod_eq_of_lt hr16lt, Nat.shiftLeft_eq]
  have hRn : (roundBf16 x).toNat = r16 * 65536 := by
    unfold roundBf16; rw [← hv, hbf]; exact toNat_ofNat_of_lt (by omega)
  have hdivR : r16 * 65536 / 2 ^ 23 = r16 / 128 := by
    rw [show (2 : ℕ) ^ 23 = 128 * 65536 by norm_num, Nat.mul_div_mul_right _ _ (by norm_num)]
  have hmodR : r16 * 65536 % 2 ^ 23 = (r16 % 128) * 65536 := by
    rw [show (2 : ℕ) ^ 23 = 128 * 65536 by norm_num, Nat.mul_mod_mul_right]
  have hER : expField (roundBf16 x) = r16 / 128 % 256 := by rw [expField_eq, hRn, hdivR]
  have hMR : mantField (roundBf16 x) = (r16 % 128) * 65536 := by rw [mantField_eq, hRn, hmodR]
  have hres' : r16 / 128 % 256 ≠ 255 := by
    have : expField (roundBf16 x) ≠ 255 := by simpa [isFinite] using hres
    rwa [hER] at this
  have hsignR : sign (roundBf16 x) = sign x := by
    rw [sign_eq, sign_eq, hRn, ← hv, hr16]
    exact decide_eq_decide.mpr (bf16_sign v hv32 hx' c hc1)
  have hER0 : expField (roundBf16 x) ≠ 0 := by rw [hER, hr16]; exact bf16_exp_ne v hv32 hx' hn' c hc1
  -- the key value identity: (2^23 + M_R) · 2^(E_R − 150) = (2^23 + (M_hi + c)·2^16) · 2^(E − 150)
  have hval : ((2 ^ 23 + mantField (roundBf16 x) : ℕ) : ℚ) * (2 : ℚ) ^ ((expField (roundBf16 x) : ℤ) - 150) =
      ((2 ^ 23 + (v / 65536 % 128 + c) * 65536 : ℕ) : ℚ) * (2 : ℚ) ^ ((expField x : ℤ) - 150) := by
    rw [hER, hMR, hE, hr16]
    by_cases hover : v / 65536 % 128 + c = 128
    · obtain ⟨h1, h2⟩ := bf16_carry_case v hv32 hx' c hc1 (hr16 ▸ hres') hover
      rw [h1, h2, hover, Nat.cast_add (R := ℤ) (v / 8388608 % 256) 1, Nat.cast_one]
      generalize ((v / 8388608 % 256 : ℕ) : ℤ) = X
      rw [show X + 1 - 150 = X - 150 + 1 by ring, zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
      norm_num; ring
    · obtain ⟨h1, h2⟩ := bf16_nocarry_case v c hc1 hover
      rw [h1, h2]
  have hclose := bf16_close v
  rw [← hc] at hclose
  -- assemble
  set q : ℚ := (2 : ℚ) ^ ((expField x : ℤ) - 150) with hq
  have hq0 : 0 < q := two_zpow_pos _
  have htx : toRat x = (if sign x then -1 else 1) * (((2 ^ 23 + mantField x : ℕ) : ℚ) * q) := by
    unfold toRat; rw [magRat_of_exp_ne hn]; split_ifs <;> ring
  have htR : toRat (roundBf16 x) = (if sign x then -1 else 1) *
      (((2 ^ 23 + (v / 65536 % 128 + c) * 65536 : ℕ) : ℚ) * q) := by
    unfold toRat; rw [hsignR, magRat_of_exp_ne hER0, hval]; split_ifs <;> ring
  have hdiff : |toRat (roundBf16 x) - toRat x| =
      |((2 ^ 23 + (v / 65536 % 128 + c) * 65536 : ℕ) : ℚ) - ((2 ^ 23 + mantField x : ℕ) : ℚ)| * q := by
    rw [htR, htx, ← mul_sub, abs_mul, ← sub_mul, abs_mul, abs_of_pos hq0]
    split_ifs <;> simp
  have habsx : |toRat x| = ((2 ^ 23 + mantField x : ℕ) : ℚ) * q := by
    rw [htx, abs_mul, abs_mul, abs_of_pos hq0]
    split_ifs <;> simp <;> (left; positivity)
  have hle : |((2 ^ 23 + (v / 65536 % 128 + c) * 65536 : ℕ) : ℚ) - ((2 ^ 23 + mantField x : ℕ) : ℚ)| ≤ 32768 := by
    rw [abs_sub_le_iff, hM]
    have e1 : (((v / 65536 % 128 + c) * 65536 : ℕ) : ℚ) ≤ ((v % 8388608 : ℕ) : ℚ) + 32768 := by
      exact_mod_cast hclose.1
    have e2 : ((v % 8388608 : ℕ) : ℚ) ≤ (((v / 65536 % 128 + c) * 65536 : ℕ) : ℚ) + 32768 := by
      exact_mod_cast hclose.2
    push_cast at e1 e2 ⊢
    constructor <;> linarith
  rw [hdiff, habsx]
  have h8 : (2 : ℚ) ^ (-8 : ℤ) * 2 ^ 23 = 32768 := by norm_num
  have hM23 : (2 ^ 23 : ℚ) ≤ ((2 ^ 23 + mantField x : ℕ) : ℚ) := by
    push_cast; linarith [(Nat.cast_nonneg (mantField x) : (0 : ℚ) ≤ mantField x)]
  calc |((2 ^ 23 + (v / 65536 % 128 + c) * 65536 : ℕ) : ℚ) - ((2 ^ 23 + mantField x : ℕ) : ℚ)| * q
      ≤ 32768 * q := mul_le_mul_of_nonneg_right hle hq0.le
    _ = (2 : ℚ) ^ (-8 : ℤ) * 2 ^ 23 * q := by rw [h8]
    _ ≤ (2 : ℚ) ^ (-8 : ℤ) * ((2 ^ 23 + mantField x : ℕ) : ℚ) * q :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hM23 (two_zpow_pos (-8 : ℤ)).le) hq0.le
    _ = (2 : ℚ) ^ (-8 : ℤ) * (((2 ^ 23 + mantField x : ℕ) : ℚ) * q) := by ring

end TVA.Math.Float.F32
