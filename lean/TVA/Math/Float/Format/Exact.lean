import TVA.Math.Float.Format.Bridge
import Mathlib

/-!
# Exact conversions: representable values round to themselves

* `roundMag_normal_exact`, `roundMag_nat_exact`, `roundMag_sub_exact`: a value that is a normal
  (`M · 2^(E − m)`, or any `N · 2^a` with `N < 2^(m+1)` in the normal range) or a subnormal
  (`F · 2^(emin − m)`) of a format rounds to its own pattern, under every mode; `magRat_nat`,
  `magRat_subPat`, `normalPat_fields` give that pattern's fields and value;
* `withSign_fields`, `magRat_withSign`, `toRat_withSign`, `isFinite_withSign`, `isZero_withSign`:
  the sign bit does not change the fields of a magnitude;
* **`cvt_widen`**: a conversion into an IEEE format whose range and precision contain the source's
  is exact (finite, same value, every mode, with or without saturation); instances
  `cvt_f16_f32_exact` (`__half2float`, `cvt.f32.f16`), `cvt_bf16_f32_exact`, `cvt_e4m3_f16_exact`,
  `cvt_e5m2_f16_exact` (`cvt.rn.f16x2.e4m3x2/e5m2x2`, `__nv_cvt_fp8x2_to_halfraw2`),
  `cvt_e4m3_bf16_exact`, `cvt_e4m3_f32_exact`, `cvt_e5m2_bf16_exact`, `cvt_e5m2_f32_exact`;
* `f16ToF32Bits_exact`: the library's `f16 → f32` (`convert .f32 (.f16 b)`) in `F32` terms.

Proof module (Mathlib); the definitions are in `TVA.Math.Float.Format`.
-/

set_option autoImplicit false

namespace TVA.Math.Float.Format

open TVA.Math.Float

variable (f : Format)

theorem emax_of_over {M : ℕ} {E : ℤ} (hov : f.over M E = false) : E ≤ f.emax := by
  unfold Format.over at hov
  cases h : f.special <;> simp only [h, decide_eq_false_iff_not, Bool.or_eq_false_iff] at hov
  · omega
  · exact not_lt.mp hov.1
  · omega

/-- The pattern a normal significand `M` (`2^m ≤ M < 2^(m+1)`) at exponent `E` encodes. -/
def normalPat (M : ℕ) (E : ℤ) : ℕ := ((E + f.bias).toNat <<< f.m) ||| (M - 2 ^ f.m)

theorem normalPat_eq {M : ℕ} {E : ℤ} (hM1 : 2 ^ f.m ≤ M) (hM2 : M < 2 ^ (f.m + 1)) :
    f.normalPat M E = (E + f.bias).toNat * 2 ^ f.m + (M - 2 ^ f.m) := by
  unfold normalPat
  rw [shl_or_eq _ _ _ (by rw [pow_succ] at hM2; omega)]

/-- A normal value `M · 2^(E − m)` is rounded to its own pattern, under every mode. -/
theorem roundMag_normal_exact (mode : RoundMode) (sat neg : Bool) {M : ℕ} {E : ℤ} (hM1 : 2 ^ f.m ≤ M)
    (hM2 : M < 2 ^ (f.m + 1)) (hE : f.emin ≤ E) (hov : f.over M E = false) :
    f.roundMag mode sat neg ((M : ℚ) * 2 ^ (E - f.m)) = f.normalPat M E := by
  have hMpos : 0 < M := lt_of_lt_of_le (by positivity) hM1
  have hq : 0 < (M : ℚ) * 2 ^ (E - f.m) := by positivity
  obtain ⟨hb1, hb2⟩ := bracket_of_nat hM1 hM2 (E - f.m)
  have hexp : f.expOf ((M : ℚ) * 2 ^ (E - f.m)) = E :=
    f.expOf_eq_of_bracket hq (by simpa using hb1) (by simpa using hb2) hE
  rw [f.roundMag_of_scale mode sat neg hq hexp (num := M) (den := 1) one_pos (by
    rw [Nat.cast_one, div_one, mul_assoc, ← zpow_add₀ two_ne]; simp)]
  rw [show roundDiv mode neg M 1 = M from by simpa using roundDiv_exact mode neg one_pos M]
  exact f.encode_mid mode sat neg hM1 hM2 hov


/-- A value `N · 2^a` with `0 < N < 2^(m+1)` is rounded to its own normal pattern, under every
mode, when its exponent `a + log₂ N` is in the normal range. -/
theorem roundMag_nat_exact (mode : RoundMode) (sat neg : Bool) {N : ℕ} (hN : 0 < N) (hNm : N < 2 ^ (f.m + 1))
    (a : ℤ) (hE1 : f.emin ≤ a + N.log2) (hE2 : f.over (N * 2 ^ (f.m - N.log2)) (a + N.log2) = false) :
    f.roundMag mode sat neg ((N : ℚ) * 2 ^ a) = f.normalPat (N * 2 ^ (f.m - N.log2)) (a + N.log2) := by
  have hk1 : 2 ^ N.log2 ≤ N := Nat.log2_self_le hN.ne'
  have hk2 : N < 2 ^ (N.log2 + 1) := Nat.lt_log2_self
  have hkm : N.log2 ≤ f.m := by
    by_contra h
    have : 2 ^ (f.m + 1) ≤ 2 ^ N.log2 := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  have hM1 : 2 ^ f.m ≤ N * 2 ^ (f.m - N.log2) := by
    calc 2 ^ f.m = 2 ^ N.log2 * 2 ^ (f.m - N.log2) := by rw [← pow_add]; congr 1; omega
      _ ≤ N * 2 ^ (f.m - N.log2) := Nat.mul_le_mul_right _ hk1
  have hM2 : N * 2 ^ (f.m - N.log2) < 2 ^ (f.m + 1) := by
    calc N * 2 ^ (f.m - N.log2) < 2 ^ (N.log2 + 1) * 2 ^ (f.m - N.log2) :=
          Nat.mul_lt_mul_of_pos_right hk2 (by positivity)
      _ = 2 ^ (f.m + 1) := by rw [← pow_add]; congr 1; omega
  have hval : (N : ℚ) * 2 ^ a = ((N * 2 ^ (f.m - N.log2) : ℕ) : ℚ) * 2 ^ (a + N.log2 - f.m) := by
    push_cast
    rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne]
    congr 2
    push_cast [Nat.cast_sub hkm]
    ring
  rw [hval]
  exact f.roundMag_normal_exact mode sat neg hM1 hM2 hE1 hE2

/-! ## Fields of a normal pattern, and of a signed pattern -/

theorem normalPat_fields {M : ℕ} {E : ℤ} (hM1 : 2 ^ f.m ≤ M) (hM2 : M < 2 ^ (f.m + 1))
    (hT : (E + f.bias).toNat < 2 ^ f.e) :
    f.expField (f.normalPat M E) = (E + f.bias).toNat ∧ f.fracField (f.normalPat M E) = M - 2 ^ f.m ∧
      f.normalPat M E < f.signBit := by
  have hlt : M - 2 ^ f.m < 2 ^ f.m := by rw [pow_succ] at hM2; omega
  rw [f.normalPat_eq hM1 hM2]
  refine ⟨?_, ?_, ?_⟩
  · rw [expField_eq, Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt hlt, zero_add,
      Nat.mod_eq_of_lt hT]
  · rw [fracField_eq, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hlt]
  · unfold signBit
    rw [pow_add]
    have : (E + f.bias).toNat * 2 ^ f.m + 2 ^ f.m ≤ 2 ^ f.e * 2 ^ f.m := by
      rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hT
    omega

/-- The value of a normal pattern. -/
theorem magRat_normalPat {M : ℕ} {E : ℤ} (hM1 : 2 ^ f.m ≤ M) (hM2 : M < 2 ^ (f.m + 1)) (hE : f.emin ≤ E)
    (hT : (E + f.bias).toNat < 2 ^ f.e) :
    f.magRat (f.normalPat M E) = (M : ℚ) * 2 ^ (E - f.m) := by
  obtain ⟨he, hf, -⟩ := f.normalPat_fields hM1 hM2 hT
  have hpos : 1 ≤ E + f.bias := by unfold emin at hE; omega
  rw [f.magRat_normal _ (by rw [he]; omega), he, hf, Nat.add_sub_cancel' hM1]
  congr 2
  rw [Int.toNat_of_nonneg (by omega)]; ring

theorem withSign_fields (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.expField (f.withSign s mag) = f.expField mag ∧ f.fracField (f.withSign s mag) = f.fracField mag ∧
      f.sign (f.withSign s mag) = s := by
  rw [f.withSign_eq s h]
  unfold signBit at *
  have hb : (if s then 1 else 0) < 2 := by cases s <;> simp
  refine ⟨?_, ?_, ?_⟩
  · rw [expField_eq, expField_eq, pow_add, show ∀ b : ℕ, b * (2 ^ f.e * 2 ^ f.m) + mag = mag + 2 ^ f.m * (b * 2 ^ f.e)
      from fun b => by ring, Nat.add_mul_div_left _ _ (Nat.two_pow_pos _), Nat.add_mul_mod_self_right]
  · rw [fracField_eq, fracField_eq, pow_add, show ∀ b : ℕ, b * (2 ^ f.e * 2 ^ f.m) + mag = mag + 2 ^ f.m * (b * 2 ^ f.e)
      from fun b => by ring, Nat.add_mul_mod_self_left]
  · rw [sign_eq, Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt h, zero_add]
    cases s <;> simp

theorem magRat_withSign (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.magRat (f.withSign s mag) = f.magRat mag := by
  obtain ⟨he, hf, -⟩ := f.withSign_fields s h
  unfold magRat; rw [he, hf]

theorem toRat_withSign (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.toRat (f.withSign s mag) = if s then -f.magRat mag else f.magRat mag := by
  obtain ⟨-, -, hs⟩ := f.withSign_fields s h
  unfold toRat; rw [hs, f.magRat_withSign s h]

theorem isFinite_withSign (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.isFinite (f.withSign s mag) = f.isFinite mag := by
  obtain ⟨he, hf, -⟩ := f.withSign_fields s h
  unfold isFinite isNaN isInf; rw [he, hf]

theorem isZero_withSign (s : Bool) {mag : ℕ} (h : mag < f.signBit) :
    f.isZero (f.withSign s mag) = f.isZero mag := by
  obtain ⟨he, hf, -⟩ := f.withSign_fields s h
  unfold isZero; rw [he, hf]

/-- A value `F · 2^(emin − m)` with `0 < F < 2^m` (a subnormal) is rounded to the pattern `F`. -/
theorem roundMag_sub_exact (mode : RoundMode) (sat neg : Bool) {F : ℕ} (hF : 0 < F) (hFm : F < 2 ^ f.m)
    (hov : f.over F f.emin = false) : f.roundMag mode sat neg ((F : ℚ) * 2 ^ (f.emin - f.m)) = F := by
  have hq : 0 < (F : ℚ) * 2 ^ (f.emin - f.m) := by positivity
  have hlt : (F : ℚ) * 2 ^ (f.emin - f.m) < 2 ^ f.emin := by
    calc (F : ℚ) * 2 ^ (f.emin - f.m) < (2 ^ f.m : ℕ) * 2 ^ (f.emin - f.m) :=
          mul_lt_mul_of_pos_right (by exact_mod_cast hFm) (zpow_pos (by norm_num) _)
      _ = 2 ^ f.emin := by push_cast; rw [← zpow_natCast, ← zpow_add₀ two_ne]; ring_nf
  rw [f.roundMag_of_scale mode sat neg hq (f.expOf_eq_emin hq hlt) (num := F) (den := 1) one_pos (by
    rw [Nat.cast_one, div_one, mul_assoc, ← zpow_add₀ two_ne]; simp)]
  rw [show roundDiv mode neg F 1 = F from by simpa using roundDiv_exact mode neg one_pos F]
  exact f.encode_lt mode sat neg hFm hov

/-- The pattern `F` (`0 < F < 2^m`) is the subnormal `F · 2^(emin − m)`. -/
theorem magRat_subPat {F : ℕ} (hF : 0 < F) (hFm : F < 2 ^ f.m) :
    f.expField F = 0 ∧ f.magRat F = (F : ℚ) * 2 ^ (f.emin - f.m) := by
  have hE : f.expField F = 0 := by rw [expField_eq, Nat.div_eq_of_lt hFm, Nat.zero_mod]
  refine ⟨hE, ?_⟩
  have hm1 : 1 ≤ f.m := by
    by_contra h; have : f.m = 0 := by omega
    rw [this] at hFm; omega
  rw [f.magRat_sub F hE, fracField_eq, Nat.mod_eq_of_lt hFm, div_eq_mul_inv, ← zpow_natCast, ← zpow_neg]
  congr 2
  unfold emin; push_cast [show 1 ≤ f.bias + f.m by omega]; ring

/-! ## Widening conversions are exact -/

/-- The normal pattern of `N · 2^a` (`0 < N < 2^(m+1)`) has the value `N · 2^a`. -/
theorem magRat_nat {N : ℕ} (hN : 0 < N) (hNm : N < 2 ^ (f.m + 1)) (a : ℤ) (hE1 : f.emin ≤ a + N.log2)
    (hT : (a + N.log2 + f.bias).toNat < 2 ^ f.e) :
    f.magRat (f.normalPat (N * 2 ^ (f.m - N.log2)) (a + N.log2)) = (N : ℚ) * 2 ^ a ∧
      f.normalPat (N * 2 ^ (f.m - N.log2)) (a + N.log2) < f.signBit ∧
      f.expField (f.normalPat (N * 2 ^ (f.m - N.log2)) (a + N.log2)) = (a + N.log2 + f.bias).toNat := by
  have hk1 : 2 ^ N.log2 ≤ N := Nat.log2_self_le hN.ne'
  have hk2 : N < 2 ^ (N.log2 + 1) := Nat.lt_log2_self
  have hkm : N.log2 ≤ f.m := by
    by_contra h
    have : 2 ^ (f.m + 1) ≤ 2 ^ N.log2 := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  have hM1 : 2 ^ f.m ≤ N * 2 ^ (f.m - N.log2) := by
    calc 2 ^ f.m = 2 ^ N.log2 * 2 ^ (f.m - N.log2) := by rw [← pow_add]; congr 1; omega
      _ ≤ N * 2 ^ (f.m - N.log2) := Nat.mul_le_mul_right _ hk1
  have hM2 : N * 2 ^ (f.m - N.log2) < 2 ^ (f.m + 1) := by
    calc N * 2 ^ (f.m - N.log2) < 2 ^ (N.log2 + 1) * 2 ^ (f.m - N.log2) :=
          Nat.mul_lt_mul_of_pos_right hk2 (by positivity)
      _ = 2 ^ (f.m + 1) := by rw [← pow_add]; congr 1; omega
  obtain ⟨he, -, hlt⟩ := f.normalPat_fields hM1 hM2 hT
  refine ⟨?_, hlt, he⟩
  rw [f.magRat_normalPat hM1 hM2 hE1 hT]
  push_cast
  rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne]
  congr 2
  push_cast [Nat.cast_sub hkm]
  ring

/-- The unbiased exponent of a finite normal pattern is at most `emax`. -/
theorem exp_le_emax (x : ℕ) (hfin : f.isFinite x = true) :
    (f.expField x : ℤ) - f.bias ≤ f.emax := by
  have hlt : f.expField x < 2 ^ f.e := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hle : f.expField x ≤ f.expMax := by unfold expMax; omega
  unfold emax
  unfold isFinite isNaN isInf at hfin
  cases h : f.special <;> simp only [h] at hfin ⊢
  · have hne : f.expField x ≠ f.expMax := by
      intro he
      by_cases hf : f.fracField x = 0
      · simp [he, hf] at hfin
      · simp [he, hf] at hfin
    omega
  · omega
  · omega

/-- **Widening conversions are exact.** If every finite value of `src` is a normal value of the
IEEE format `dst` (`dst` has at least as many fraction bits, its normal range reaches below the
smallest subnormal of `src` and above the largest finite value of `src`), then converting a finite
`src` pattern gives a finite `dst` pattern of the same value, under every rounding mode, with or
without saturation. Instances: `e4m3`/`e5m2 → f16, bf16, f32`, `f16 → f32`, `bf16 → f32`
(`cvt_f16_f32_exact`, ...). -/
theorem cvt_widen (src dst : Format) (mode : RoundMode) (sat : Bool) (nan : NaNRule)
    (hd : dst.special = .ieee) (he : 0 < dst.e) (hm : src.m ≤ dst.m) (hlo : dst.emin - dst.m ≤ src.emin - src.m)
    (hsub : src.emin - 1 ≤ dst.emax) (hhi : src.emax ≤ dst.emax) (hrange : dst.emin ≤ dst.emax) (x : ℕ)
    (hx : src.isFinite x = true) :
    dst.isFinite (cvt src dst mode sat nan x) = true ∧ dst.toRat (cvt src dst mode sat nan x) = src.toRat x ∧
      cvt src dst mode sat nan x < 2 * dst.signBit := by
  have hnan : src.isNaN x = false := by unfold isFinite at hx; simp only [Bool.and_eq_true, Bool.not_eq_true'] at hx; exact hx.1
  have hinf : src.isInf x = false := by unfold isFinite at hx; simp only [Bool.and_eq_true, Bool.not_eq_true'] at hx; exact hx.2
  have hexpMax : 1 ≤ dst.expMax := by
    unfold expMax; have : 2 ≤ 2 ^ dst.e := by
      calc 2 = 2 ^ 1 := rfl
        _ ≤ 2 ^ dst.e := Nat.pow_le_pow_right (by norm_num) he
    omega
  have hemax : dst.emax = (dst.expMax : ℤ) - 1 - dst.bias := by unfold emax; rw [hd]
  have hsb : 0 < dst.signBit := by unfold signBit; positivity
  by_cases hz : src.isZero x = true
  · rw [cvt_zero src dst mode sat nan hnan hinf hz]
    have h0 : dst.magRat 0 = 0 := by unfold magRat expField fracField; simp
    have hsz : src.magRat x = 0 := by
      unfold isZero at hz; simp only [Bool.and_eq_true, beq_iff_eq] at hz
      unfold magRat; simp [hz.1, hz.2]
    refine ⟨?_, ?_, ?_⟩
    · rw [dst.isFinite_withSign _ hsb]
      unfold isFinite isNaN isInf expField; rw [hd]; simp; omega
    · rw [dst.toRat_withSign _ hsb, h0]; unfold toRat; rw [hsz]
    · rw [dst.withSign_eq _ hsb]; cases src.sign x <;> simp <;> omega
  · have hz' : src.isZero x = false := by simpa using hz
    rw [cvt_finite src dst mode sat nan hnan hinf hz']
    -- the source value as `N · 2^a`
    obtain ⟨N, a, hN, hNm, hval, hA, hE2⟩ : ∃ N : ℕ, ∃ a : ℤ, 0 < N ∧ N < 2 ^ (dst.m + 1) ∧
        src.magRat x = (N : ℚ) * 2 ^ a ∧ src.emin - src.m ≤ a ∧ a + N.log2 ≤ dst.emax := by
      by_cases hE : src.expField x = 0
      · have hF : src.fracField x ≠ 0 := by
          intro hF; apply hz; unfold isZero; simp [hE, hF]
        have hFlt : src.fracField x < 2 ^ src.m := Nat.mod_lt _ (Nat.two_pow_pos _)
        have hm1 : 1 ≤ src.m := by
          by_contra h; have : src.m = 0 := by omega
          rw [this] at hFlt; omega
        have hk : (src.fracField x).log2 < src.m := (Nat.log2_lt hF).mpr hFlt
        refine ⟨src.fracField x, src.emin - src.m, Nat.pos_of_ne_zero hF, ?_, ?_, ?_, ?_⟩
        · calc src.fracField x < 2 ^ src.m := hFlt
            _ ≤ 2 ^ (dst.m + 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
        · rw [src.magRat_sub x hE, div_eq_mul_inv, ← zpow_natCast, ← zpow_neg]
          congr 2
          unfold emin; push_cast [show 1 ≤ src.bias + src.m by omega]; ring
        · omega
        · omega
      · have hFlt : src.fracField x < 2 ^ src.m := Nat.mod_lt _ (Nat.two_pow_pos _)
        have hNne : 2 ^ src.m + src.fracField x ≠ 0 := by positivity
        have hlog : (2 ^ src.m + src.fracField x).log2 = src.m := by
          apply le_antisymm
          · exact Nat.lt_succ_iff.mp ((Nat.log2_lt hNne).mpr (by rw [pow_succ]; omega))
          · exact (Nat.le_log2 hNne).mpr (by omega)
        have hmax := src.exp_le_emax x hx
        refine ⟨2 ^ src.m + src.fracField x, (src.expField x : ℤ) - src.bias - src.m, by positivity, ?_, ?_, ?_, ?_⟩
        · calc 2 ^ src.m + src.fracField x < 2 ^ (src.m + 1) := by rw [pow_succ]; omega
            _ ≤ 2 ^ (dst.m + 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
        · exact src.magRat_normal x hE
        · unfold emin; omega
        · rw [hlog]; omega
    have hX : dst.expMax < 2 ^ dst.e := by unfold expMax; have := Nat.two_pow_pos dst.e; omega
    have hsbm : 2 ^ dst.m ≤ dst.signBit := by
      unfold signBit; exact Nat.pow_le_pow_right (by norm_num) (by omega)
    by_cases hE1 : dst.emin ≤ a + N.log2
    swap
    · -- below the normal range of `dst`: a subnormal of `dst`
      have hk2 : N < 2 ^ (N.log2 + 1) := Nat.lt_log2_self
      set k := (a - (dst.emin - dst.m)).toNat with hk
      have hF : 0 < N * 2 ^ k := by positivity
      have hFm : N * 2 ^ k < 2 ^ dst.m := by
        calc N * 2 ^ k < 2 ^ (N.log2 + 1) * 2 ^ k := Nat.mul_lt_mul_of_pos_right hk2 (by positivity)
          _ = 2 ^ (N.log2 + 1 + k) := (pow_add _ _ _).symm
          _ ≤ 2 ^ dst.m := Nat.pow_le_pow_right (by norm_num) (by omega)
      have hv : (N : ℚ) * 2 ^ a = ((N * 2 ^ k : ℕ) : ℚ) * 2 ^ (dst.emin - dst.m) := by
        push_cast
        rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne]
        congr 2
        omega
      have hov : dst.over (N * 2 ^ k) dst.emin = false := by
        rw [dst.over_of_ieee hd]; simp; omega
      rw [hval, hv, dst.roundMag_sub_exact mode sat (src.sign x) hF hFm hov]
      obtain ⟨hE0, hmag⟩ := dst.magRat_subPat hF hFm
      have hlt : N * 2 ^ k < dst.signBit := lt_of_lt_of_le hFm hsbm
      refine ⟨?_, ?_, ?_⟩
      · rw [dst.isFinite_withSign _ hlt]
        unfold isFinite isNaN isInf; rw [hd, hE0]
        have : (0 : ℕ) ≠ dst.expMax := by omega
        simp [this]
      · rw [dst.toRat_withSign _ hlt, hmag]; unfold toRat; rw [hval, hv]
      · rw [dst.withSign_eq _ hlt]; cases src.sign x <;> simp <;> omega
    have hT : (a + N.log2 + dst.bias).toNat < 2 ^ dst.e := by omega
    obtain ⟨hv, hlt, hexp⟩ := dst.magRat_nat hN hNm a hE1 hT
    have hov : dst.over (N * 2 ^ (dst.m - N.log2)) (a + N.log2) = false := by
      rw [dst.over_of_ieee hd]; simp; omega
    rw [hval, dst.roundMag_nat_exact mode sat (src.sign x) hN hNm a hE1 hov]
    refine ⟨?_, ?_, ?_⟩
    · rw [dst.isFinite_withSign _ hlt]
      unfold isFinite isNaN isInf; rw [hd, hexp]
      have : (a + N.log2 + dst.bias).toNat ≠ dst.expMax := by omega
      simp [this]
    · rw [dst.toRat_withSign _ hlt, hv]; unfold toRat; rw [hval]
    · rw [dst.withSign_eq _ hlt]; cases src.sign x <;> simp <;> omega

/-! ## Instances -/

/-- The value-and-finiteness statement of an exact conversion. -/
abbrev ExactCvt (src dst : Format) (mode : RoundMode) (sat : Bool) (nan : NaNRule) : Prop :=
  ∀ x : ℕ, src.isFinite x = true →
    dst.isFinite (cvt src dst mode sat nan x) = true ∧ dst.toRat (cvt src dst mode sat nan x) = src.toRat x

theorem exactCvt_of_widen {src dst : Format} {mode : RoundMode} {sat : Bool} {nan : NaNRule}
    (hd : dst.special = .ieee) (he : 0 < dst.e) (hm : src.m ≤ dst.m) (hlo : dst.emin - dst.m ≤ src.emin - src.m)
    (hsub : src.emin - 1 ≤ dst.emax) (hhi : src.emax ≤ dst.emax) (hrange : dst.emin ≤ dst.emax) :
    ExactCvt src dst mode sat nan :=
  fun x hx => (cvt_widen src dst mode sat nan hd he hm hlo hsub hhi hrange x hx).imp_right And.left

/-- `f16 → f32` (`cvt.f32.f16`, `__half2float`) is exact. -/
theorem cvt_f16_f32_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt binary16 binary32 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- `bf16 → f32` (`cvt.f32.bf16`, `__bfloat162float`) is exact. -/
theorem cvt_bf16_f32_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt bfloat16 binary32 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- `e4m3 → f16` (`cvt.rn.f16x2.e4m3x2`, `__nv_cvt_fp8_to_halfraw`) is exact. -/
theorem cvt_e4m3_f16_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt e4m3 binary16 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem cvt_e4m3_bf16_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt e4m3 bfloat16 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem cvt_e4m3_f32_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt e4m3 binary32 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem cvt_e5m2_bf16_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt e5m2 bfloat16 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

theorem cvt_e5m2_f32_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt e5m2 binary32 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- `e5m2 → f16` (`cvt.rn.f16x2.e5m2x2`) is exact (the same exponent range: e5m2 subnormals are f16
subnormals). -/
theorem cvt_e5m2_f16_exact (mode : RoundMode) (sat : Bool) (nan : NaNRule) : ExactCvt e5m2 binary16 mode sat nan :=
  exactCvt_of_widen rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- `__half2float` on bit patterns (`f16ToF32Bits`, the library's `convert .f32 (.f16 b)`) is exact. -/
theorem f16ToF32Bits_exact (x : ℕ) (hx : binary16.isFinite x = true) :
    F32.isFinite (BitVec.ofNat 32 (f16ToF32Bits x)) = true ∧
      F32.toRat (BitVec.ofNat 32 (f16ToF32Bits x)) = binary16.toRat x := by
  obtain ⟨h1, h2, h3⟩ := cvt_widen binary16 binary32 .rn false .canonical rfl (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) x hx
  have hlt : f16ToF32Bits x < 2 ^ 32 := by rw [f16ToF32Bits_cvt]; exact h3
  rw [F32.isFinite_binary32, F32.toRat_binary32, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hlt, f16ToF32Bits_cvt]
  exact ⟨h1, h2⟩

end TVA.Math.Float.Format
