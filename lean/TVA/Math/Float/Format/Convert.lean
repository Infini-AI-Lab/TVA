import TVA.Math.Float.Format.Binary32
import TVA.Math.Float.Convert

/-!
# Bridge: the conversions of `TVA.Math.Float.Convert` are instances of `Format.cvt`

* `bf16ToF32Bits_cvt`: `bf16 → f32` is `cvt bfloat16 binary32` (exact; NaN payload widened, not
  quieted);
* `f16ToF32Bits_cvt`: `f16 → f32` is `cvt binary16 binary32` (exact; every NaN gives the
  canonical NaN `0x7fffffff`);
* `f32ToBf16RNE_cvt`: `f32 → bf16` round-to-nearest-even is `cvt binary32 bfloat16 .rn` with the
  canonical NaN `0x7fff`;
* `f32ToF16RNE_cvt`: `f32 → f16` round-to-nearest-even is `cvt binary32 binary16 .rn` with the
  canonical NaN `0x7fff`.

All four hold for every natural number (the conversions read only the low bits of their input).
The generic facts used (`encode_*`, `cvt_*`, `magRat_normal`, `magRat_sub`) are stated for any
format. Proof module (Mathlib); part of `TVA.Math.Float.Format.Bridge`.
-/
set_option autoImplicit false

namespace TVA.Math.Float

open TVA.Math.Float.Format

namespace Format

variable (f : Format)

/-! ## Encoding a rounded significand -/

theorem over_of_ieee (h : f.special = .ieee) (m : ℕ) (e' : ℤ) : f.over m e' = decide (e' > f.emax) := by
  unfold Format.over; rw [h]

theorem encode_lt (mode : RoundMode) (sat neg : Bool) {m : ℕ} {e' : ℤ} (hm : m < 2 ^ f.m)
    (hov : f.over m e' = false) : f.encode mode sat neg m e' = m := by
  have hne : (m == 2 ^ (f.m + 1)) = false := by
    rw [beq_eq_false_iff_ne]; have : 2 ^ f.m < 2 ^ (f.m + 1) := Nat.pow_lt_pow_right (by norm_num) (by omega)
    omega
  unfold encode
  simp only [hne, Bool.false_eq_true, ↓reduceIte, hov, hm]

theorem encode_mid (mode : RoundMode) (sat neg : Bool) {m : ℕ} {e' : ℤ} (hm1 : 2 ^ f.m ≤ m)
    (hm2 : m < 2 ^ (f.m + 1)) (hov : f.over m e' = false) :
    f.encode mode sat neg m e' = ((e' + f.bias).toNat <<< f.m) ||| (m - 2 ^ f.m) := by
  have hne : (m == 2 ^ (f.m + 1)) = false := by rw [beq_eq_false_iff_ne]; omega
  unfold encode
  simp only [hne, Bool.false_eq_true, ↓reduceIte, hov, not_lt.mpr hm1]

theorem encode_over (mode : RoundMode) (sat neg : Bool) {m : ℕ} {e' : ℤ} (hm : m ≠ 2 ^ (f.m + 1))
    (hov : f.over m e' = true) : f.encode mode sat neg m e' = f.overflow mode sat neg := by
  have hne : (m == 2 ^ (f.m + 1)) = false := by rw [beq_eq_false_iff_ne]; exact hm
  unfold encode
  simp only [hne, Bool.false_eq_true, ↓reduceIte, hov]

theorem encode_carry (mode : RoundMode) (sat neg : Bool) {e' : ℤ} (hov : f.over (2 ^ f.m) (e' + 1) = false) :
    f.encode mode sat neg (2 ^ (f.m + 1)) e' = (e' + 1 + f.bias).toNat <<< f.m := by
  unfold encode
  simp only [beq_self_eq_true, ↓reduceIte, hov, Bool.false_eq_true, lt_irrefl, Nat.sub_self, Nat.or_zero]

theorem encode_carry_over (mode : RoundMode) (sat neg : Bool) {e' : ℤ}
    (hov : f.over (2 ^ f.m) (e' + 1) = true) :
    f.encode mode sat neg (2 ^ (f.m + 1)) e' = f.overflow mode sat neg := by
  unfold encode
  simp only [beq_self_eq_true, ↓reduceIte, hov]

/-! ## Values of patterns -/

theorem two_ne : (2 : ℚ) ≠ 0 := by norm_num

/-- The value of a normal pattern. -/
theorem magRat_normal (x : ℕ) (hE : f.expField x ≠ 0) :
    f.magRat x = ((2 ^ f.m + f.fracField x : ℕ) : ℚ) * (2 : ℚ) ^ ((f.expField x : ℤ) - f.bias - f.m) := by
  unfold magRat
  simp only [beq_iff_eq, hE, ↓reduceIte]
  split_ifs with h
  · rw [show ((f.expField x : ℤ) - f.bias - f.m) = ((f.expField x - (f.bias + f.m) : ℕ) : ℤ) by omega,
      zpow_natCast]
    push_cast; ring
  · rw [Rat.mkRat_eq_div, show ((f.expField x : ℤ) - f.bias - f.m) = -((f.bias + f.m - f.expField x : ℕ) : ℤ) by omega,
      zpow_neg, zpow_natCast, div_eq_mul_inv]
    push_cast; ring

/-- The value of a subnormal (or zero) pattern. -/
theorem magRat_sub (x : ℕ) (hE : f.expField x = 0) :
    f.magRat x = (f.fracField x : ℚ) / 2 ^ (f.bias + f.m - 1) := by
  unfold magRat
  simp only [hE, beq_self_eq_true, ↓reduceIte, Rat.mkRat_eq_div]
  push_cast; ring

/-- `N · 2^a · 2^b` with `a + b = k ≥ 0` is the natural number `N · 2^k`. -/
theorem mul_two_zpow_nat (N : ℕ) (a b : ℤ) (k : ℕ) (h : a + b = k) :
    (N : ℚ) * 2 ^ a * 2 ^ b = ((N * 2 ^ k : ℕ) : ℚ) := by
  rw [mul_assoc, ← zpow_add₀ two_ne, h, zpow_natCast]; push_cast; ring

/-- `N · 2^a · 2^b` with `a + b = −k` is `N / 2^k`. -/
theorem mul_two_zpow_div (N : ℕ) (a b : ℤ) (k : ℕ) (h : a + b = -(k : ℤ)) :
    (N : ℚ) * 2 ^ a * 2 ^ b = (N : ℚ) / ((2 ^ k : ℕ) : ℚ) := by
  rw [mul_assoc, ← zpow_add₀ two_ne, h, zpow_neg, zpow_natCast]; push_cast; ring

/-- `N ∈ [2^p, 2^(p+1))` places `N · 2^a` in `[2^(p+a), 2^(p+a+1))`. -/
theorem bracket_of_nat {N p : ℕ} (h1 : 2 ^ p ≤ N) (h2 : N < 2 ^ (p + 1)) (a : ℤ) :
    (2 : ℚ) ^ ((p : ℤ) + a) ≤ (N : ℚ) * 2 ^ a ∧ (N : ℚ) * 2 ^ a < 2 ^ ((p : ℤ) + a + 1) := by
  have hpos : (0 : ℚ) < 2 ^ a := zpow_pos (by norm_num) a
  constructor
  · rw [zpow_add₀ two_ne, zpow_natCast]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast h1) hpos.le
  · rw [show (p : ℤ) + a + 1 = ((p + 1 : ℕ) : ℤ) + a by push_cast; ring, zpow_add₀ two_ne, zpow_natCast]
    exact mul_lt_mul_of_pos_right (by exact_mod_cast h2) hpos

/-! ## Conversions, by case -/

theorem cvt_nan (src dst : Format) (mode : RoundMode) (sat : Bool) (nan : NaNRule) {x : ℕ}
    (h : src.isNaN x = true) : cvt src dst mode sat nan x = nan.apply src dst x := by
  unfold cvt; simp [h]

theorem cvt_inf (src dst : Format) (mode : RoundMode) (sat : Bool) (nan : NaNRule) {x : ℕ}
    (h1 : src.isNaN x = false) (h2 : src.isInf x = true) :
    cvt src dst mode sat nan x = dst.withSign (src.sign x)
      (if sat then dst.maxFinite else match dst.special with
        | .ieee => dst.infMag
        | .fn => dst.nanMag
        | .finite => dst.maxFinite) := by
  unfold cvt; simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]; rfl

theorem cvt_zero (src dst : Format) (mode : RoundMode) (sat : Bool) (nan : NaNRule) {x : ℕ}
    (h1 : src.isNaN x = false) (h2 : src.isInf x = false) (h3 : src.isZero x = true) :
    cvt src dst mode sat nan x = dst.withSign (src.sign x) 0 := by
  unfold cvt; simp [h1, h2, h3]

theorem cvt_finite (src dst : Format) (mode : RoundMode) (sat : Bool) (nan : NaNRule) {x : ℕ}
    (h1 : src.isNaN x = false) (h2 : src.isInf x = false) (h3 : src.isZero x = false) :
    cvt src dst mode sat nan x =
      dst.withSign (src.sign x) (dst.roundMag mode sat (src.sign x) (src.magRat x)) := by
  unfold cvt; simp [h1, h2, h3]

/-! ## Fields as arithmetic -/

theorem expField_eq (x : ℕ) : f.expField x = x / 2 ^ f.m % 2 ^ f.e := by
  unfold expField; rw [Nat.shiftRight_eq_div_pow]

theorem fracField_eq (x : ℕ) : f.fracField x = x % 2 ^ f.m := rfl

theorem sign_eq (x : ℕ) : f.sign x = decide (x / 2 ^ (f.e + f.m) % 2 = 1) := by
  unfold sign; rw [Nat.testBit_eq_decide_div_mod_eq]

/-- A magnitude below the sign bit, with the sign as the top bit. -/
theorem withSign_eq (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.withSign s mag = (if s then 1 else 0) * f.signBit + mag := by
  rw [withSign_of_lt f s h]; cases s <;> simp; ring

theorem isNaN_ieee (h : f.special = .ieee) (x : ℕ) :
    f.isNaN x = (f.expField x == f.expMax && f.fracField x != 0) := by
  unfold isNaN; rw [h]

theorem isInf_ieee (h : f.special = .ieee) (x : ℕ) :
    f.isInf x = (f.expField x == f.expMax && f.fracField x == 0) := by
  unfold isInf; rw [h]

end Format

/-! ## Constants of bfloat16 and binary16 -/

namespace Format

@[simp] theorem bfloat16_e : bfloat16.e = 8 := rfl
@[simp] theorem bfloat16_m : bfloat16.m = 7 := rfl
@[simp] theorem bfloat16_special : bfloat16.special = .ieee := rfl
@[simp] theorem bfloat16_bias : bfloat16.bias = 127 := rfl
@[simp] theorem bfloat16_expMax : bfloat16.expMax = 255 := rfl
@[simp] theorem bfloat16_emin : bfloat16.emin = -126 := rfl
@[simp] theorem bfloat16_emax : bfloat16.emax = 127 := rfl
@[simp] theorem bfloat16_signBit : bfloat16.signBit = 2 ^ 15 := rfl
@[simp] theorem bfloat16_infMag : bfloat16.infMag = 0x7f80 := rfl
@[simp] theorem bfloat16_nanMag : bfloat16.nanMag = 0x7fff := rfl

@[simp] theorem binary16_e : binary16.e = 5 := rfl
@[simp] theorem binary16_m : binary16.m = 10 := rfl
@[simp] theorem binary16_special : binary16.special = .ieee := rfl
@[simp] theorem binary16_bias : binary16.bias = 15 := rfl
@[simp] theorem binary16_expMax : binary16.expMax = 31 := rfl
@[simp] theorem binary16_emin : binary16.emin = -14 := rfl
@[simp] theorem binary16_emax : binary16.emax = 15 := rfl
@[simp] theorem binary16_signBit : binary16.signBit = 2 ^ 15 := rfl
@[simp] theorem binary16_infMag : binary16.infMag = 0x7c00 := rfl
@[simp] theorem binary16_nanMag : binary16.nanMag = 0x7fff := rfl

end Format

/-! ## `bf16 → f32` -/

theorem bf16ToF32Bits_cvt (x : ℕ) : bf16ToF32Bits x = cvt bfloat16 binary32 .rn false (.widen false) x := by
  set S := x / 2 ^ 15 % 2 with hS
  set E := bfloat16.expField x with hE
  set M := bfloat16.fracField x with hM
  have hE' : E = x / 128 % 256 := by rw [hE, Format.expField_eq]; rfl
  have hM' : M = x % 128 := rfl
  have hs : bfloat16.sign x = decide (S = 1) := by rw [Format.sign_eq]; rfl
  have hLHS : bf16ToF32Bits x = (S * 2 ^ 15 + E * 2 ^ 7 + M) * 2 ^ 16 := by
    unfold bf16ToF32Bits; rw [Nat.shiftLeft_eq]; congr 1; omega
  have hS1 : S < 2 := Nat.mod_lt _ (by norm_num)
  have hE1 : E < 256 := by omega
  have hM1 : M < 128 := by omega
  -- the sign applied by `binary32.withSign`
  have hws : ∀ mag, mag < 2 ^ 31 →
      binary32.withSign (bfloat16.sign x) mag = S * 2 ^ 31 + mag := by
    intro mag h
    rw [Format.withSign_eq binary32 _ (by simpa using h), hs]
    simp only [binary32_signBit]
    rcases (by omega : S = 0 ∨ S = 1) with h0 | h0 <;> simp [h0]
  rw [hLHS]
  by_cases hEmax : E = 255
  · by_cases hM0 : M = 0
    · -- infinity
      have h1 : bfloat16.isNaN x = false := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0]
      have h2 : bfloat16.isInf x = true := by
        rw [Format.isInf_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_inf _ _ _ _ _ h1 h2]
      simp only [Bool.false_eq_true, ↓reduceIte, binary32_special, binary32_infMag]
      rw [hws _ (by norm_num), hEmax, hM0]; ring
    · -- NaN: the payload, shifted
      have h1 : bfloat16.isNaN x = true := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_nan _ _ _ _ _ h1]
      unfold NaNRule.apply
      simp only [Bool.false_eq_true, ↓reduceIte, binary32_infMag, binary32_m, bfloat16_m, Nat.or_zero, ← hM]
      have hor : (0x7f800000 : ℕ) ||| M <<< (23 - 7) = 255 * 2 ^ 23 + M * 2 ^ 16 := by
        rw [show (0x7f800000 : ℕ) = 255 <<< 23 by norm_num, Format.shl_or_eq _ _ _ (by
          rw [Nat.shiftLeft_eq]; omega), Nat.shiftLeft_eq]
      rw [hor, hws _ (by omega), hEmax]; ring
  · have h1 : bfloat16.isNaN x = false := by
      rw [Format.isNaN_ieee _ rfl]; simp [← hE, hEmax]
    have h2 : bfloat16.isInf x = false := by
      rw [Format.isInf_ieee _ rfl]; simp [← hE, hEmax]
    by_cases hz : E = 0 ∧ M = 0
    · have h3 : bfloat16.isZero x = true := by
        unfold Format.isZero; simp [← hE, ← hM, hz.1, hz.2]
      rw [Format.cvt_zero _ _ _ _ _ h1 h2 h3, hws _ (by norm_num), hz.1, hz.2]; ring
    · have h3 : bfloat16.isZero x = false := by
        unfold Format.isZero; simp only [← hE, ← hM]; simpa using hz
      rw [Format.cvt_finite _ _ _ _ _ h1 h2 h3]
      by_cases hE0 : E = 0
      · -- subnormal: `M · 2^-133`, an f32 subnormal
        have hM0 : M ≠ 0 := fun h => hz ⟨hE0, h⟩
        have hq : bfloat16.magRat x = (M : ℚ) * 2 ^ (-133 : ℤ) := by
          rw [Format.magRat_sub _ _ (by rw [← hE]; exact hE0)]
          simp only [bfloat16_bias, bfloat16_m, ← hM]
          rw [zpow_neg]; norm_num [div_eq_mul_inv]
        have hpos : 0 < bfloat16.magRat x := by
          rw [hq]; have : (0 : ℚ) < M := by exact_mod_cast Nat.pos_of_ne_zero hM0
          positivity
        have hexp : binary32.expOf (bfloat16.magRat x) = -126 := by
          rw [Format.expOf_eq_emin binary32 hpos]
          · rfl
          · rw [hq, binary32_emin]
            have hlt : (M : ℚ) < 2 ^ 7 := by exact_mod_cast hM1
            calc (M : ℚ) * 2 ^ (-133 : ℤ) < 2 ^ (7 : ℕ) * 2 ^ (-133 : ℤ) := by gcongr
              _ = 2 ^ (-126 : ℤ) := by rw [← zpow_natCast, ← zpow_add₀ Format.two_ne]; norm_num
        rw [Format.roundMag_of_scale binary32 .rn false (bfloat16.sign x) hpos hexp (num := M * 2 ^ 16) (den := 1)
          (by norm_num) (by rw [hq, Nat.cast_one, div_one, Format.mul_two_zpow_nat M _ _ 16 (by simp)])]
        rw [show M * 2 ^ 16 = (M * 2 ^ 16) * 1 by ring, Format.roundDiv_exact _ _ (by norm_num)]
        rw [Format.encode_lt binary32 _ _ _ (by simp; omega) (by rw [Format.over_of_ieee binary32 rfl]; decide)]
        rw [hws _ (by omega), hE0]; ring
      · -- normal: `(2^7 + M) · 2^(E − 134)`
        have hq : bfloat16.magRat x = ((2 ^ 7 + M : ℕ) : ℚ) * 2 ^ ((E : ℤ) - 134) := by
          rw [Format.magRat_normal _ _ (by rw [← hE]; exact hE0)]
          simp only [bfloat16_m, bfloat16_bias, ← hE, ← hM]
          congr 2; push_cast; ring
        have hpos : 0 < bfloat16.magRat x := by rw [hq]; positivity
        have hexp : binary32.expOf (bfloat16.magRat x) = (E : ℤ) - 127 := by
          have hb := Format.bracket_of_nat (N := 2 ^ 7 + M) (p := 7) (by omega) (by omega) ((E : ℤ) - 134)
          apply Format.expOf_eq_of_bracket binary32 hpos
          · rw [hq]; convert hb.1 using 2; push_cast; ring
          · rw [hq]; convert hb.2 using 2; push_cast; ring
          · simp; omega
        rw [Format.roundMag_of_scale binary32 .rn false (bfloat16.sign x) hpos hexp
          (num := (2 ^ 7 + M) * 2 ^ 16) (den := 1) (by norm_num)
          (by rw [hq, Nat.cast_one, div_one, Format.mul_two_zpow_nat _ _ _ 16 (by simp; ring)])]
        rw [show (2 ^ 7 + M) * 2 ^ 16 = ((2 ^ 7 + M) * 2 ^ 16) * 1 by ring, Format.roundDiv_exact _ _ (by norm_num)]
        rw [Format.encode_mid binary32 _ _ _ (by simp; omega) (by simp; omega)
          (by rw [Format.over_of_ieee binary32 rfl]; simp; omega)]
        simp only [binary32_m, binary32_bias, Nat.cast_ofNat]
        have hT : ((E : ℤ) - 127 + 127).toNat = E := by omega
        rw [hT, Format.shl_or_eq _ _ _ (by omega), hws _ (by omega)]
        omega

/-! ## `f32 → bf16` -/

/-- Rounding to nearest-even by `2^k` as a division. -/
theorem roundDiv_rn_two_pow_div (neg : Bool) (N k : ℕ) (hk : 1 ≤ k) :
    roundDiv .rn neg N (2 ^ k) = (N + 2 ^ (k - 1) - 1 + N / 2 ^ k % 2) / 2 ^ k := by
  rw [Format.roundDiv_rn_two_pow neg N k hk, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]

theorem bf16_trick_arith (S E M : ℕ) (hS : S < 2) (hE : E < 255) (hM : M < 2 ^ 23) :
    S * 2 ^ 15 + (E * 2 ^ 7 + (M + 2 ^ 15 - 1 + M / 2 ^ 16 % 2) / 2 ^ 16) =
      (S * 2 ^ 31 + E * 2 ^ 23 + M + 32767 + (S * 2 ^ 31 + E * 2 ^ 23 + M) / 2 ^ 16 % 2) / 2 ^ 16 % 2 ^ 16 := by
  have h1 : (S * 2 ^ 31 + E * 2 ^ 23 + M) / 2 ^ 16 % 2 = M / 2 ^ 16 % 2 := by omega
  rw [h1]
  have h2 : (S * 2 ^ 31 + E * 2 ^ 23 + M + 32767 + M / 2 ^ 16 % 2) / 2 ^ 16 =
      S * 2 ^ 15 + E * 2 ^ 7 + (M + 32767 + M / 2 ^ 16 % 2) / 2 ^ 16 := by omega
  rw [h2]
  have h3 : (M + 32767 + M / 2 ^ 16 % 2) / 2 ^ 16 ≤ 128 := by omega
  omega

theorem f32ToBf16RNE_cvt (x : ℕ) : f32ToBf16RNE x = cvt binary32 bfloat16 .rn false .canonical x := by
  set S := x / 2 ^ 31 % 2 with hS
  set E := binary32.expField x with hE
  set M := binary32.fracField x with hM
  have hE' : E = x / 2 ^ 23 % 256 := by rw [hE, Format.expField_eq]; rfl
  have hM' : M = x % 2 ^ 23 := rfl
  have hs : binary32.sign x = decide (S = 1) := by rw [Format.sign_eq]; rfl
  have hS1 : S < 2 := Nat.mod_lt _ (by norm_num)
  have hE1 : E < 256 := by omega
  have hM1 : M < 2 ^ 23 := by omega
  have hv : x % 2 ^ 32 = S * 2 ^ 31 + E * 2 ^ 23 + M := by omega
  -- the NaN test of the code reads the same fields
  have hnan : f32IsNaN (x % 2 ^ 32) = binary32.isNaN x := by
    rw [f32IsNaN_binary32, Format.isNaN_ieee _ rfl, Format.isNaN_ieee _ rfl, Format.expField_eq,
      Format.expField_eq, Format.fracField_eq, Format.fracField_eq]
    simp only [binary32_m, binary32_e]
    congr 2 <;> omega
  have hws : ∀ mag, mag < 2 ^ 15 → bfloat16.withSign (binary32.sign x) mag = S * 2 ^ 15 + mag := by
    intro mag h
    rw [Format.withSign_eq bfloat16 _ (by simpa using h), hs]
    simp only [bfloat16_signBit]
    rcases (by omega : S = 0 ∨ S = 1) with h0 | h0 <;> simp [h0]
  unfold f32ToBf16RNE
  simp only [Nat.shiftRight_eq_div_pow]
  rw [hnan, hv]
  by_cases hEmax : E = 255
  · by_cases hM0 : M = 0
    · have h1 : binary32.isNaN x = false := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0]
      have h2 : binary32.isInf x = true := by
        rw [Format.isInf_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_inf _ _ _ _ _ h1 h2, h1]
      simp only [Bool.false_eq_true, ↓reduceIte, bfloat16_special, bfloat16_infMag]
      rw [hws _ (by norm_num), hEmax, hM0]
      omega
    · have h1 : binary32.isNaN x = true := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_nan _ _ _ _ _ h1, h1]
      rfl
  · have h1 : binary32.isNaN x = false := by
      rw [Format.isNaN_ieee _ rfl]; simp [← hE, hEmax]
    have h2 : binary32.isInf x = false := by
      rw [Format.isInf_ieee _ rfl]; simp [← hE, hEmax]
    simp only [h1, Bool.false_eq_true, ↓reduceIte]
    by_cases hz : E = 0 ∧ M = 0
    · have h3 : binary32.isZero x = true := by
        unfold Format.isZero; simp [← hE, ← hM, hz.1, hz.2]
      rw [Format.cvt_zero _ _ _ _ _ h1 h2 h3, hws _ (by norm_num), hz.1, hz.2]
      omega
    · have h3 : binary32.isZero x = false := by
        unfold Format.isZero; simp only [← hE, ← hM]; simpa using hz
      rw [Format.cvt_finite _ _ _ _ _ h1 h2 h3]
      -- the rounded magnitude
      suffices hR : bfloat16.roundMag .rn false (binary32.sign x) (binary32.magRat x) =
          E * 2 ^ 7 + (M + 2 ^ 15 - 1 + M / 2 ^ 16 % 2) / 2 ^ 16 by
        have hT : (M + 2 ^ 15 - 1 + M / 2 ^ 16 % 2) / 2 ^ 16 ≤ 128 := by omega
        rw [hR, hws _ (by omega)]
        exact (bf16_trick_arith S E M hS1 (by omega) hM1).symm
      by_cases hE0 : E = 0
      · -- subnormal: `M · 2^-149`, rounded on the bf16 subnormal grid
        have hM0 : M ≠ 0 := fun h => hz ⟨hE0, h⟩
        have hq : binary32.magRat x = (M : ℚ) * 2 ^ (-149 : ℤ) := by
          rw [Format.magRat_sub _ _ (by rw [← hE]; exact hE0)]
          simp only [binary32_bias, binary32_m, ← hM]
          rw [zpow_neg]; norm_num [div_eq_mul_inv]
        have hpos : 0 < binary32.magRat x := by
          rw [hq]; have : (0 : ℚ) < M := by exact_mod_cast Nat.pos_of_ne_zero hM0
          positivity
        have hexp : bfloat16.expOf (binary32.magRat x) = -126 := by
          rw [Format.expOf_eq_emin bfloat16 hpos]
          · rfl
          · rw [hq, bfloat16_emin]
            have hlt : (M : ℚ) < 2 ^ 23 := by exact_mod_cast hM1
            calc (M : ℚ) * 2 ^ (-149 : ℤ) < 2 ^ (23 : ℕ) * 2 ^ (-149 : ℤ) := by gcongr
              _ = 2 ^ (-126 : ℤ) := by rw [← zpow_natCast, ← zpow_add₀ Format.two_ne]; norm_num
        rw [Format.roundMag_of_scale bfloat16 .rn false _ hpos hexp (num := M) (den := 2 ^ 16)
          (by positivity) (by rw [hq, Format.mul_two_zpow_div M _ _ 16 (by simp)])]
        rw [roundDiv_rn_two_pow_div _ _ _ (by norm_num), hE0]
        simp only [Nat.add_one_sub_one, zero_mul, zero_add]
        by_cases hR : (M + 2 ^ 15 - 1 + M / 2 ^ 16 % 2) / 2 ^ 16 < 2 ^ 7
        · rw [Format.encode_lt bfloat16 _ _ _ (by simpa using hR)
            (by rw [Format.over_of_ieee bfloat16 rfl]; decide)]
        · have hR' : (M + 2 ^ 15 - 1 + M / 2 ^ 16 % 2) / 2 ^ 16 = 2 ^ 7 := by omega
          rw [hR', Format.encode_mid bfloat16 _ _ _ (by simp) (by simp)
            (by rw [Format.over_of_ieee bfloat16 rfl]; decide)]
          rfl
      · -- normal: `(2^23 + M) · 2^(E − 150)`
        have hq : binary32.magRat x = ((2 ^ 23 + M : ℕ) : ℚ) * 2 ^ ((E : ℤ) - 150) := by
          rw [Format.magRat_normal _ _ (by rw [← hE]; exact hE0)]
          simp only [binary32_m, binary32_bias, ← hE, ← hM]
          congr 2; push_cast; ring
        have hpos : 0 < binary32.magRat x := by rw [hq]; positivity
        have hexp : bfloat16.expOf (binary32.magRat x) = (E : ℤ) - 127 := by
          have hb := Format.bracket_of_nat (N := 2 ^ 23 + M) (p := 23) (by omega) (by omega) ((E : ℤ) - 150)
          apply Format.expOf_eq_of_bracket bfloat16 hpos
          · rw [hq]; convert hb.1 using 2; push_cast; ring
          · rw [hq]; convert hb.2 using 2; push_cast; ring
          · simp; omega
        rw [Format.roundMag_of_scale bfloat16 .rn false _ hpos hexp (num := 2 ^ 23 + M) (den := 2 ^ 16)
          (by positivity) (by rw [hq, Format.mul_two_zpow_div _ _ _ 16 (by simp; ring)])]
        rw [roundDiv_rn_two_pow_div _ _ _ (by norm_num)]
        simp only [Nat.add_one_sub_one]
        have hR0 : (2 ^ 23 + M + 2 ^ 15 - 1 + (2 ^ 23 + M) / 2 ^ 16 % 2) / 2 ^ 16 =
            2 ^ 7 + (M + 2 ^ 15 - 1 + M / 2 ^ 16 % 2) / 2 ^ 16 := by omega
        rw [hR0]
        by_cases hc : (M + 2 ^ 15 - 1 + M / 2 ^ 16 % 2) / 2 ^ 16 = 2 ^ 7
        · rw [hc, show (2 : ℕ) ^ 7 + 2 ^ 7 = 2 ^ (bfloat16.m + 1) by simp]
          by_cases hE254 : E = 254
          · rw [Format.encode_carry_over bfloat16 _ _ _ (by
              rw [Format.over_of_ieee bfloat16 rfl]; simp; omega)]
            simp only [Format.overflow, bfloat16_special]
            simp [hE254]
          · rw [Format.encode_carry bfloat16 _ _ _ (by
              rw [Format.over_of_ieee bfloat16 rfl]; simp; omega)]
            simp only [bfloat16_bias, bfloat16_m, Nat.cast_ofNat, Nat.shiftLeft_eq]
            have hT : ((E : ℤ) - 127 + 1 + 127).toNat = E + 1 := by omega
            rw [hT]; ring
        · rw [Format.encode_mid bfloat16 _ _ _ (by simp) (by simp; omega)
            (by rw [Format.over_of_ieee bfloat16 rfl]; simp; omega)]
          simp only [bfloat16_bias, bfloat16_m, Nat.cast_ofNat]
          have hT : ((E : ℤ) - 127 + 127).toNat = E := by omega
          rw [hT, Format.shl_or_eq _ _ _ (by omega)]
          omega

/-! ## `f16 → f32` -/

/-- The normalisation loop of `f16ToF32Bits`: `j` doublings bring a subnormal significand into
`[2^10, 2^11)`. -/
theorem f16ToF32Bits_norm (j : ℕ) : ∀ (m e fuel : ℕ), 2 ^ 10 ≤ m * 2 ^ j → m * 2 ^ j < 2 ^ 11 → j ≤ fuel →
    f16ToF32Bits.norm m e fuel = (m * 2 ^ j, e - j) := by
  induction j with
  | zero =>
    intro m e fuel h1 _ _
    cases fuel with
    | zero => simp [f16ToF32Bits.norm.eq_1]
    | succ fuel => rw [f16ToF32Bits.norm.eq_2]; simp at h1 ⊢; omega
  | succ j ih =>
    intro m e fuel h1 h2 hf
    obtain ⟨fuel', rfl⟩ : ∃ fuel', fuel = fuel' + 1 := ⟨fuel - 1, by omega⟩
    have hj : 2 ≤ 2 ^ (j + 1) := by
      calc 2 = 2 ^ 1 := by norm_num
        _ ≤ 2 ^ (j + 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
    have hm : ¬ m ≥ 1024 := by
      intro hm
      have : 1024 * 2 ≤ m * 2 ^ (j + 1) := Nat.mul_le_mul hm hj
      omega
    rw [f16ToF32Bits.norm.eq_2, ite_eq_right hm, Nat.shiftLeft_eq,
      ih (m * 2 ^ 1) (e - 1) fuel' (by rw [mul_assoc, ← pow_add]; rwa [Nat.add_comm 1 j])
        (by rw [mul_assoc, ← pow_add]; rwa [Nat.add_comm 1 j]) (by omega)]
    rw [mul_assoc, ← pow_add, Nat.add_comm 1 j]; congr 1; omega

theorem f16ToF32Bits_cvt (x : ℕ) : f16ToF32Bits x = cvt binary16 binary32 .rn false .canonical x := by
  set S := x / 2 ^ 15 % 2 with hS
  set E := binary16.expField x with hE
  set M := binary16.fracField x with hM
  have hE' : E = x / 2 ^ 10 % 32 := by rw [hE, Format.expField_eq]; rfl
  have hM' : M = x % 2 ^ 10 := rfl
  have hE1 : E < 32 := by omega
  have hM1 : M < 2 ^ 10 := by omega
  -- the sign bit of the code is `binary32.withSign`
  have hsg : ∀ A, A < 2 ^ 31 →
      (if binary16.sign x = true then 1 <<< 31 else 0) ||| A = binary32.withSign (binary16.sign x) A := by
    intro A hA
    rw [Format.withSign_of_lt binary32 _ (by simpa using hA)]
    cases binary16.sign x
    · simp
    · simp only [↓reduceIte, binary32_signBit]
      rw [show (1 <<< 31 : ℕ) = 2 ^ 31 by norm_num, Nat.lor_comm, Format.or_two_pow_eq _ _ hA]
  unfold f16ToF32Bits
  simp only [show ∀ y, (f16Fields y).sign = binary16.sign y from fun _ => rfl,
    show ∀ y, (f16Fields y).exp = binary16.expField y from fun _ => rfl,
    show ∀ y, (f16Fields y).mant = binary16.fracField y from fun _ => rfl,
    show ∀ y, FloatFields.isNaN 5 (f16Fields y) = binary16.isNaN y from fun _ => rfl,
    show ∀ y, FloatFields.isInf 5 (f16Fields y) = binary16.isInf y from fun _ => rfl, ← hE, ← hM]
  by_cases hEmax : E = 31
  · by_cases hM0 : M = 0
    · have h1 : binary16.isNaN x = false := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0]
      have h2 : binary16.isInf x = true := by
        rw [Format.isInf_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_inf _ _ _ _ _ h1 h2, h1, h2]
      simp only [Bool.false_eq_true, ↓reduceIte, binary32_special, binary32_infMag]
      exact hsg _ (by norm_num)
    · have h1 : binary16.isNaN x = true := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_nan _ _ _ _ _ h1, h1]
      rfl
  · have h1 : binary16.isNaN x = false := by
      rw [Format.isNaN_ieee _ rfl]; simp [← hE, hEmax]
    have h2 : binary16.isInf x = false := by
      rw [Format.isInf_ieee _ rfl]; simp [← hE, hEmax]
    simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]
    by_cases hz : E = 0 ∧ M = 0
    · have h3 : binary16.isZero x = true := by
        unfold Format.isZero; simp [← hE, ← hM, hz.1, hz.2]
      rw [Format.cvt_zero _ _ _ _ _ h1 h2 h3]
      simp only [hz.1, hz.2, beq_self_eq_true, ↓reduceIte]
      rw [← hsg 0 (by norm_num), Nat.or_zero]
    · have h3 : binary16.isZero x = false := by
        unfold Format.isZero; simp only [← hE, ← hM]; simpa using hz
      rw [Format.cvt_finite _ _ _ _ _ h1 h2 h3]
      by_cases hE0 : E = 0
      · -- subnormal: `M · 2^-24`, normalised with `j = 10 − log₂ M` doublings
        have hM0 : M ≠ 0 := fun h => hz ⟨hE0, h⟩
        have hMne : (M == 0) = false := by simpa using hM0
        simp only [hE0, beq_self_eq_true, ↓reduceIte, hMne, Bool.false_eq_true]
        set p := Nat.log2 M with hp
        have hp1 : 2 ^ p ≤ M := Nat.log2_self_le hM0
        have hp2 : M < 2 ^ (p + 1) := Nat.lt_log2_self
        have hp9 : p ≤ 9 := by
          by_contra hc
          have : 2 ^ 10 ≤ 2 ^ p := Nat.pow_le_pow_right (by norm_num) (by omega)
          omega
        have hj1 : 2 ^ 10 ≤ M * 2 ^ (10 - p) := by
          calc 2 ^ 10 = 2 ^ p * 2 ^ (10 - p) := by rw [← pow_add]; congr 1; omega
            _ ≤ M * 2 ^ (10 - p) := Nat.mul_le_mul_right _ hp1
        have hj2 : M * 2 ^ (10 - p) < 2 ^ 11 := by
          calc M * 2 ^ (10 - p) < 2 ^ (p + 1) * 2 ^ (10 - p) :=
                Nat.mul_lt_mul_of_pos_right hp2 (by positivity)
            _ = 2 ^ 11 := by rw [← pow_add]; congr 1; omega
        rw [f16ToF32Bits_norm (10 - p) M _ 24 hj1 hj2 (by omega)]
        simp only
        -- the value and its rounding
        have hq : binary16.magRat x = (M : ℚ) * 2 ^ (-24 : ℤ) := by
          rw [Format.magRat_sub _ _ (by rw [← hE]; exact hE0)]
          simp only [binary16_bias, binary16_m, ← hM]
          rw [zpow_neg]; norm_num [div_eq_mul_inv]
        have hpos : 0 < binary16.magRat x := by
          rw [hq]; have : (0 : ℚ) < M := by exact_mod_cast Nat.pos_of_ne_zero hM0
          positivity
        have hexp : binary32.expOf (binary16.magRat x) = (p : ℤ) - 24 := by
          have hb := Format.bracket_of_nat hp1 hp2 (-24)
          apply Format.expOf_eq_of_bracket binary32 hpos
          · rw [hq]; exact hb.1
          · rw [hq]; exact hb.2
          · simp; omega
        rw [Format.roundMag_of_scale binary32 .rn false _ hpos hexp (num := M * 2 ^ (23 - p)) (den := 1)
          (by norm_num) (by rw [hq, Nat.cast_one, div_one, Format.mul_two_zpow_nat M _ _ (23 - p) (by simp; omega)])]
        rw [show M * 2 ^ (23 - p) = (M * 2 ^ (23 - p)) * 1 by ring, Format.roundDiv_exact _ _ (by norm_num)]
        have hsplit : M * 2 ^ (23 - p) = M * 2 ^ (10 - p) * 2 ^ 13 := by
          rw [mul_assoc, ← pow_add]; congr 2; omega
        rw [Format.encode_mid binary32 _ _ _ (by simp; omega) (by simp; omega)
          (by rw [Format.over_of_ieee binary32 rfl]; simp; omega)]
        simp only [binary32_m, binary32_bias, Nat.cast_ofNat]
        have hT : ((p : ℤ) - 24 + 127).toNat = 113 - (10 - p) := by omega
        rw [hT, Nat.lor_assoc, hsplit, Nat.shiftLeft_eq (_ % 1024),
          show M * 2 ^ (10 - p) % 1024 = M * 2 ^ (10 - p) - 2 ^ 10 by omega]
        rw [show (M * 2 ^ (10 - p) - 2 ^ 10) * 2 ^ 13 = M * 2 ^ (10 - p) * 2 ^ 13 - 2 ^ 23 by
          rw [Nat.sub_mul]; norm_num]
        apply hsg
        rw [Format.shl_or_eq _ _ _ (by omega)]
        omega
      · -- normal: `(2^10 + M) · 2^(E − 25)`
        have hEne : (E == 0) = false := by simpa using hE0
        simp only [hEne, Bool.false_eq_true, ↓reduceIte]
        have hq : binary16.magRat x = ((2 ^ 10 + M : ℕ) : ℚ) * 2 ^ ((E : ℤ) - 25) := by
          rw [Format.magRat_normal _ _ (by rw [← hE]; exact hE0)]
          simp only [binary16_m, binary16_bias, ← hE, ← hM]
          congr 2; push_cast; ring
        have hpos : 0 < binary16.magRat x := by rw [hq]; positivity
        have hexp : binary32.expOf (binary16.magRat x) = (E : ℤ) - 15 := by
          have hb := Format.bracket_of_nat (N := 2 ^ 10 + M) (p := 10) (by omega) (by omega) ((E : ℤ) - 25)
          apply Format.expOf_eq_of_bracket binary32 hpos
          · rw [hq]; convert hb.1 using 2; push_cast; ring
          · rw [hq]; convert hb.2 using 2; push_cast; ring
          · simp; omega
        rw [Format.roundMag_of_scale binary32 .rn false _ hpos hexp
          (num := (2 ^ 10 + M) * 2 ^ 13) (den := 1) (by norm_num)
          (by rw [hq, Nat.cast_one, div_one, Format.mul_two_zpow_nat _ _ _ 13 (by simp; ring)])]
        rw [show (2 ^ 10 + M) * 2 ^ 13 = ((2 ^ 10 + M) * 2 ^ 13) * 1 by ring,
          Format.roundDiv_exact _ _ (by norm_num)]
        rw [Format.encode_mid binary32 _ _ _ (by simp; omega) (by simp; omega)
          (by rw [Format.over_of_ieee binary32 rfl]; simp; omega)]
        simp only [binary32_m, binary32_bias, Nat.cast_ofNat]
        have hT : ((E : ℤ) - 15 + 127).toNat = E + 127 - 15 := by omega
        rw [hT, Nat.lor_assoc, Nat.shiftLeft_eq M,
          show (2 ^ 10 + M) * 2 ^ 13 - 2 ^ 23 = M * 2 ^ 13 by ring_nf; omega]
        apply hsg
        rw [Format.shl_or_eq _ _ _ (by omega)]
        omega

/-! ## `f32 → f16` -/

/-- The rounding of the subnormal path of `f32ToF16RNE` is `roundDiv .rn`. -/
theorem roundDiv_rn_code (neg : Bool) (n k : ℕ) (hk : 1 ≤ k) :
    (if (decide (n % 2 ^ k > 2 ^ (k - 1)) || (n % 2 ^ k == 2 ^ (k - 1) && n / 2 ^ k % 2 == 1)) = true
      then n / 2 ^ k + 1 else n / 2 ^ k) = roundDiv .rn neg n (2 ^ k) := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  rw [show 2 ^ (j + 1) = 2 * 2 ^ j by ring]
  have hpos : 0 < 2 ^ j := by positivity
  generalize 2 ^ j = h at hpos ⊢
  unfold roundDiv
  simp only
  by_cases A : n % (2 * h) > h
  · have hc : (decide (n % (2 * h) > h) || (n % (2 * h) == h && n / (2 * h) % 2 == 1)) = true := by simp [A]
    rw [ite_eq_left hc, ite_eq_left (by omega : 2 * (n % (2 * h)) > 2 * h)]
  · rw [ite_eq_right (by omega : ¬ 2 * (n % (2 * h)) > 2 * h)]
    by_cases B : n % (2 * h) = h
    · rw [ite_eq_right (by omega : ¬ 2 * (n % (2 * h)) < 2 * h)]
      rcases Nat.mod_two_eq_zero_or_one (n / (2 * h)) with hf | hf
      · have hc : (decide (n % (2 * h) > h) || (n % (2 * h) == h && n / (2 * h) % 2 == 1)) = false := by
          simp [A, hf]
        simp [hf]; omega
      · have hc : (decide (n % (2 * h) > h) || (n % (2 * h) == h && n / (2 * h) % 2 == 1)) = true := by
          simp [B, hf]
        simp [hf]; omega
    · have hc : (decide (n % (2 * h) > h) || (n % (2 * h) == h && n / (2 * h) % 2 == 1)) = false := by
        simp [A, B]
      rw [ite_eq_right (by simp [hc]), ite_eq_left (by omega : 2 * (n % (2 * h)) < 2 * h)]

/-- A fraction below one half rounds to `0`. -/
theorem roundDiv_rn_of_lt (neg : Bool) {n d : ℕ} (h : 2 * n < d) : roundDiv .rn neg n d = 0 := by
  unfold roundDiv
  have h1 : n / d = 0 := Nat.div_eq_of_lt (by omega)
  have h2 : n % d = n := Nat.mod_eq_of_lt (by omega)
  simp only [h1, h2, h, ↓reduceIte]
  simp; omega

/-- Rounding to nearest adds at most one to the floor. -/
theorem roundDiv_rn_le (neg : Bool) (n d : ℕ) : roundDiv .rn neg n d ≤ n / d + 1 := by
  unfold roundDiv; simp only; split_ifs <;> omega

theorem f32ToF16RNE_cvt (x : ℕ) : f32ToF16RNE x = cvt binary32 binary16 .rn false .canonical x := by
  set E := binary32.expField x with hE
  set M := binary32.fracField x with hM
  have hE' : E = x / 2 ^ 23 % 256 := by rw [hE, Format.expField_eq]; rfl
  have hM' : M = x % 2 ^ 23 := rfl
  have hE1 : E < 256 := by omega
  have hM1 : M < 2 ^ 23 := by omega
  have hsg : ∀ A, A < 2 ^ 15 →
      (if binary32.sign x = true then 32768 else 0) ||| A = binary16.withSign (binary32.sign x) A := by
    intro A hA
    rw [Format.withSign_of_lt binary16 _ (by simpa using hA)]
    cases binary32.sign x
    · simp
    · simp only [↓reduceIte, binary16_signBit]
      rw [show (32768 : ℕ) = 2 ^ 15 by norm_num, Nat.lor_comm, Format.or_two_pow_eq _ _ hA]
  have hsg0 : (if binary32.sign x = true then 32768 else 0) = binary16.withSign (binary32.sign x) 0 := by
    rw [← hsg 0 (by norm_num), Nat.or_zero]
  unfold f32ToF16RNE
  simp only [show ∀ y, (f32Fields y).sign = binary32.sign y from fun _ => rfl,
    show ∀ y, (f32Fields y).exp = binary32.expField y from fun _ => rfl,
    show ∀ y, (f32Fields y).mant = binary32.fracField y from fun _ => rfl,
    show ∀ y, FloatFields.isNaN 8 (f32Fields y) = binary32.isNaN y from fun _ => rfl,
    show ∀ y, FloatFields.isInf 8 (f32Fields y) = binary32.isInf y from fun _ => rfl,
    Nat.shiftRight_eq_div_pow, Nat.shiftLeft_eq, one_mul, ← hE, ← hM]
  by_cases hEmax : E = 255
  · by_cases hM0 : M = 0
    · have h1 : binary32.isNaN x = false := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0]
      have h2 : binary32.isInf x = true := by
        rw [Format.isInf_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_inf _ _ _ _ _ h1 h2]
      simp only [h1, h2, Bool.false_eq_true, ↓reduceIte, binary16_special, binary16_infMag]
      exact hsg _ (by norm_num)
    · have h1 : binary32.isNaN x = true := by
        rw [Format.isNaN_ieee _ rfl]; simp [← hE, ← hM, hM0, hEmax]
      rw [Format.cvt_nan _ _ _ _ _ h1]
      simp only [h1, ↓reduceIte]
      rfl
  · have h1 : binary32.isNaN x = false := by
      rw [Format.isNaN_ieee _ rfl]; simp [← hE, hEmax]
    have h2 : binary32.isInf x = false := by
      rw [Format.isInf_ieee _ rfl]; simp [← hE, hEmax]
    simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]
    -- the value of a normal pattern, and its exponent in binary16
    have hqn : E ≠ 0 → binary32.magRat x = ((2 ^ 23 + M : ℕ) : ℚ) * 2 ^ ((E : ℤ) - 150) := by
      intro hE0
      rw [Format.magRat_normal _ _ (by rw [← hE]; exact hE0)]
      simp only [binary32_m, binary32_bias, ← hE, ← hM]
      congr 2; push_cast; ring
    have hpn : E ≠ 0 → 0 < binary32.magRat x := by intro hE0; rw [hqn hE0]; positivity
    by_cases hbig : (E : ℤ) - 127 > 15
    · rw [ite_eq_left hbig]
      have hE0 : E ≠ 0 := by omega
      have h3 : binary32.isZero x = false := by unfold Format.isZero; simp [← hE, hE0]
      rw [Format.cvt_finite _ _ _ _ _ h1 h2 h3]
      have hexp : binary16.expOf (binary32.magRat x) = (E : ℤ) - 127 := by
        have hb := Format.bracket_of_nat (N := 2 ^ 23 + M) (p := 23) (by omega) (by omega) ((E : ℤ) - 150)
        apply Format.expOf_eq_of_bracket binary16 (hpn hE0)
        · rw [hqn hE0]; convert hb.1 using 2; push_cast; ring
        · rw [hqn hE0]; convert hb.2 using 2; push_cast; ring
        · simp; omega
      rw [Format.roundMag_of_scale binary16 .rn false _ (hpn hE0) hexp (num := 2 ^ 23 + M) (den := 2 ^ 13)
        (by positivity) (by rw [hqn hE0, Format.mul_two_zpow_div _ _ _ 13 (by simp; ring)])]
      have hov : binary16.overflow .rn false (binary32.sign x) = 0x7c00 := rfl
      by_cases hc : roundDiv .rn (binary32.sign x) (2 ^ 23 + M) (2 ^ 13) = 2 ^ (binary16.m + 1)
      · rw [hc, Format.encode_carry_over binary16 _ _ _ (by
          rw [Format.over_of_ieee binary16 rfl]; simp; omega), hov]
        exact hsg _ (by norm_num)
      · rw [Format.encode_over binary16 _ _ _ hc (by
          rw [Format.over_of_ieee binary16 rfl]; simp; omega), hov]
        exact hsg _ (by norm_num)
    · rw [ite_eq_right hbig]
      by_cases hmid : (E : ℤ) - 127 ≥ -14
      · rw [ite_eq_left hmid]
        have hE0 : E ≠ 0 := by omega
        have h3 : binary32.isZero x = false := by unfold Format.isZero; simp [← hE, hE0]
        rw [Format.cvt_finite _ _ _ _ _ h1 h2 h3]
        have hexp : binary16.expOf (binary32.magRat x) = (E : ℤ) - 127 := by
          have hb := Format.bracket_of_nat (N := 2 ^ 23 + M) (p := 23) (by omega) (by omega) ((E : ℤ) - 150)
          apply Format.expOf_eq_of_bracket binary16 (hpn hE0)
          · rw [hqn hE0]; convert hb.1 using 2; push_cast; ring
          · rw [hqn hE0]; convert hb.2 using 2; push_cast; ring
          · simp; omega
        rw [Format.roundMag_of_scale binary16 .rn false _ (hpn hE0) hexp (num := 2 ^ 23 + M) (den := 2 ^ 13)
          (by positivity) (by rw [hqn hE0, Format.mul_two_zpow_div _ _ _ 13 (by simp; ring)])]
        rw [roundDiv_rn_two_pow_div _ _ _ (by norm_num)]
        set T := (M + 2 ^ (13 - 1) - 1 + M / 2 ^ 13 % 2) / 2 ^ 13 with hT
        have hR0 : (2 ^ 23 + M + 2 ^ (13 - 1) - 1 + (2 ^ 23 + M) / 2 ^ 13 % 2) / 2 ^ 13 = 2 ^ 10 + T := by
          rw [hT]; omega
        have hT1 : T ≤ 1024 := by rw [hT]; omega
        rw [hR0]
        by_cases hcar : T ≥ 1024
        · rw [ite_eq_left hcar, show 2 ^ 10 + T = 2 ^ (binary16.m + 1) by simp; omega]
          by_cases hov : ((E : ℤ) - 127 + 15).toNat + 1 ≥ 31
          · rw [ite_eq_left hov, Format.encode_carry_over binary16 _ _ _ (by
              rw [Format.over_of_ieee binary16 rfl]; simp; omega)]
            exact hsg _ (by norm_num)
          · rw [ite_eq_right hov, Format.encode_carry binary16 _ _ _ (by
              rw [Format.over_of_ieee binary16 rfl]; simp; omega)]
            simp only [binary16_m, binary16_bias, Nat.cast_ofNat, Nat.shiftLeft_eq]
            rw [show ((E : ℤ) - 127 + 1 + 15).toNat = ((E : ℤ) - 127 + 15).toNat + 1 by omega]
            exact hsg _ (by omega)
        · rw [ite_eq_right hcar, Format.encode_mid binary16 _ _ _ (by simp) (by simp; omega)
            (by rw [Format.over_of_ieee binary16 rfl]; simp; omega)]
          simp only [binary16_m, binary16_bias, Nat.cast_ofNat, Nat.add_sub_cancel_left]
          rw [Nat.lor_assoc, Nat.shiftLeft_eq]
          apply hsg
          rw [← Nat.shiftLeft_eq, Format.shl_or_eq _ _ _ (by omega)]
          omega
      · rw [ite_eq_right hmid]
        have hshift : (-14 - ((E : ℤ) - 127)).toNat + 13 = 126 - E := by omega
        rw [hshift]
        have hexp_sub : ∀ q : ℚ, 0 < q → q < 2 ^ (-14 : ℤ) → binary16.expOf q = -14 := by
          intro q hq hlt
          rw [Format.expOf_eq_emin binary16 hq (by simpa using hlt)]; rfl
        by_cases hz : E = 0 ∧ M = 0
        · have h3 : binary32.isZero x = true := by
            unfold Format.isZero; simp [← hE, ← hM, hz.1, hz.2]
          rw [Format.cvt_zero _ _ _ _ _ h1 h2 h3, ite_eq_left (by omega)]
          exact hsg0
        · have h3 : binary32.isZero x = false := by
            unfold Format.isZero; simp only [← hE, ← hM]; simpa using hz
          rw [Format.cvt_finite _ _ _ _ _ h1 h2 h3]
          have hov0 : ∀ m, binary16.over m (-14) = false := by
            intro m; rw [Format.over_of_ieee binary16 rfl]; decide
          by_cases hfar : 126 - E ≥ 64
          · rw [ite_eq_left hfar]
            -- far below the f16 grid: rounds to `0`
            suffices h0 : binary16.roundMag .rn false (binary32.sign x) (binary32.magRat x) = 0 by
              rw [h0]; exact hsg0
            by_cases hE0 : E = 0
            · have hM0 : M ≠ 0 := fun h => hz ⟨hE0, h⟩
              have hq : binary32.magRat x = (M : ℚ) * 2 ^ (-149 : ℤ) := by
                rw [Format.magRat_sub _ _ (by rw [← hE]; exact hE0)]
                simp only [binary32_bias, binary32_m, ← hM]
                rw [zpow_neg]; norm_num [div_eq_mul_inv]
              have hpos : 0 < binary32.magRat x := by
                rw [hq]; have : (0 : ℚ) < M := by exact_mod_cast Nat.pos_of_ne_zero hM0
                positivity
              have hexp : binary16.expOf (binary32.magRat x) = -14 := by
                apply hexp_sub _ hpos
                rw [hq]
                have hlt : (M : ℚ) < 2 ^ 23 := by exact_mod_cast hM1
                calc (M : ℚ) * 2 ^ (-149 : ℤ) < 2 ^ (23 : ℕ) * 2 ^ (-149 : ℤ) := by gcongr
                  _ ≤ 2 ^ (-14 : ℤ) := by rw [← zpow_natCast, ← zpow_add₀ Format.two_ne]; norm_num
              rw [Format.roundMag_of_scale binary16 .rn false _ hpos hexp (num := M) (den := 2 ^ 125)
                (by positivity) (by rw [hq, Format.mul_two_zpow_div M _ _ 125 (by simp)])]
              rw [roundDiv_rn_of_lt _ (by omega), Format.encode_lt binary16 _ _ _ (by simp) (hov0 0)]
            · have hexp : binary16.expOf (binary32.magRat x) = -14 := by
                apply hexp_sub _ (hpn hE0)
                have hb := Format.bracket_of_nat (N := 2 ^ 23 + M) (p := 23) (by omega) (by omega) ((E : ℤ) - 150)
                rw [hqn hE0]
                calc ((2 ^ 23 + M : ℕ) : ℚ) * 2 ^ ((E : ℤ) - 150) < 2 ^ ((23 : ℕ) + ((E : ℤ) - 150) + 1) := hb.2
                  _ ≤ 2 ^ (-14 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by push_cast; omega)
              rw [Format.roundMag_of_scale binary16 .rn false _ (hpn hE0) hexp (num := 2 ^ 23 + M)
                (den := 2 ^ (126 - E)) (by positivity)
                (by rw [hqn hE0, Format.mul_two_zpow_div _ _ _ (126 - E)
                (by simp only [binary16_m]; rw [Nat.cast_sub (by omega)]; push_cast; omega)])]
              have hbig' : 2 ^ 24 ≤ 2 ^ (126 - E - 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
              have hsplit : 2 ^ (126 - E) = 2 * 2 ^ (126 - E - 1) := by
                rw [← pow_succ']; congr 1; omega
              rw [roundDiv_rn_of_lt _ (by rw [hsplit]; omega),
                Format.encode_lt binary16 _ _ _ (by simp) (hov0 0)]
          · rw [ite_eq_right hfar]
            have hE0 : E ≠ 0 := by omega
            have hexp : binary16.expOf (binary32.magRat x) = -14 := by
              apply hexp_sub _ (hpn hE0)
              have hb := Format.bracket_of_nat (N := 2 ^ 23 + M) (p := 23) (by omega) (by omega) ((E : ℤ) - 150)
              rw [hqn hE0]
              calc ((2 ^ 23 + M : ℕ) : ℚ) * 2 ^ ((E : ℤ) - 150) < 2 ^ ((23 : ℕ) + ((E : ℤ) - 150) + 1) := hb.2
                _ ≤ 2 ^ (-14 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by push_cast; omega)
            rw [Format.roundMag_of_scale binary16 .rn false _ (hpn hE0) hexp (num := 2 ^ 23 + M)
              (den := 2 ^ (126 - E)) (by positivity)
              (by rw [hqn hE0, Format.mul_two_zpow_div _ _ _ (126 - E)
                (by simp only [binary16_m]; rw [Nat.cast_sub (by omega)]; push_cast; omega)])]
            rw [show (2 ^ 23 ||| M) = 2 ^ 23 + M by
              rw [Nat.lor_comm, Format.or_two_pow_eq _ _ hM1, Nat.add_comm]]
            rw [roundDiv_rn_code (binary32.sign x) _ _ (by omega)]
            -- the rounded significand is at most `2^10`
            have hfl : (2 ^ 23 + M) / 2 ^ (126 - E) < 2 ^ 10 := by
              rw [Nat.div_lt_iff_lt_mul (by positivity)]
              calc 2 ^ 23 + M < 2 ^ 24 := by omega
                _ = 2 ^ 10 * 2 ^ 14 := by norm_num
                _ ≤ 2 ^ 10 * 2 ^ (126 - E) := Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by norm_num) (by omega))
            have hR := roundDiv_rn_le (binary32.sign x) (2 ^ 23 + M) (2 ^ (126 - E))
            by_cases hR1 : roundDiv .rn (binary32.sign x) (2 ^ 23 + M) (2 ^ (126 - E)) < 2 ^ 10
            · rw [Format.encode_lt binary16 _ _ _ (by simpa using hR1) (hov0 _)]
              exact hsg _ (by omega)
            · have hR2 : roundDiv .rn (binary32.sign x) (2 ^ 23 + M) (2 ^ (126 - E)) = 2 ^ 10 := by omega
              rw [hR2, Format.encode_mid binary16 _ _ _ (by simp) (by simp) (hov0 _)]
              exact hsg _ (by norm_num)

end TVA.Math.Float
