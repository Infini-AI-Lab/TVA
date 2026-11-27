import Proof.Defs

/-!
# The kernel's `float32` constants

The values of the constants the model builds: the divisor `d` (lines 75, 77), the source's
`0.044715`, `0.7978845608028654`, `2.0`, `1.0`, `0.5`, `0.0` (lines 84–90) and `tl.sigmoid`'s `1`.
The generic rounding lemmas (`vcConsts_*`) show that an integer below `2^24` converts exactly.
-/

namespace Claims.GeluAndMulSparse

open TVA
open Format

noncomputable section

/-- A rational within `maxVal` rounds to a pattern below `2^width`. -/
theorem vcConsts_roundRat_lt_of_le {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} (hq : |q| ≤ f.maxVal) : f.roundRat rnd sat zs q < 2 ^ f.width := by
  by_cases hq0 : q = 0
  · rw [hq0]; exact (Format.roundRat_zero_spec hf).1
  · exact (Format.roundRat_spec hf rnd sat zs hq0 hq).1

/-- A rational within `maxVal` rounds to a finite pattern. -/
theorem vcConsts_isFinite_roundRat_of_le {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} (hq : |q| ≤ f.maxVal) : f.isFinite (f.roundRat rnd sat zs q) = true := by
  by_cases hq0 : q = 0
  · rw [hq0]; exact (Format.roundRat_zero_spec hf).2.1
  · exact (Format.roundRat_spec hf rnd sat zs hq0 hq).2.1

/-- The real value of an `ofPat` pattern. -/
theorem vcConsts_toReal_ofPat {f : Format} {n : ℕ} (h : n < 2 ^ f.width) :
    (FloatV.ofPat n : FloatV f).toReal = ((f.toRat n : ℚ) : ℝ) := by
  unfold FloatV.toReal
  rw [FloatV.toRat_ofPat h]

/-- A nonzero value is not a zero pattern. -/
theorem vcConsts_isZero_eq_false {f : Format} {x : FloatV f} (h : x.toRat ≠ 0) :
    x.isZero = false := by
  cases hz : x.isZero
  · rfl
  · exact absurd (Format.toRat_eq_zero_iff.mpr hz) h

/-- A finite `float64` converts to the rounded pattern. -/
theorem vcConsts_hostF32_eq_ofPat {c : F64} (h : c.isFinite = true) :
    FloatV.hostF32 c = FloatV.ofPat (Format.binary32.roundRat .rn false c.sign c.toRat) := by
  have hnan : c.isNaN = false := not_isNaN_of_isFinite h
  unfold FloatV.hostF32 FloatV.cvt
  simp only [hnan, Bool.false_eq_true, ite_false]
  rw [Format.cvt_of_finite Format.binary32 .rn false h]
  rfl

/-- The real value of the `float32` host conversion of a finite value. -/
theorem vcConsts_hostF32_toReal {c : F64} (h : c.isFinite = true)
    (hq : |c.toRat| ≤ Format.binary32.maxVal) {n : ℕ}
    (hval : Format.binary32.toRat (Format.binary32.roundRat .rn false c.sign c.toRat)
      = (n : ℚ)) :
    (FloatV.hostF32 c).toReal = (n : ℝ) := by
  rw [vcConsts_hostF32_eq_ofPat h,
    vcConsts_toReal_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hq), hval]
  push_cast
  ring

/-- An integer at most `2^m` below the format's range rounds to itself. -/
theorem vcConsts_toRat_roundRat_natCast {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    (hem : f.emin ≤ 0) {n : ℕ} (hn : (n : ℚ) ≤ (2 : ℚ) ^ (f.m : ℤ))
    (hmax : (n : ℚ) ≤ f.maxVal) :
    f.toRat (f.roundRat rnd sat zs (n : ℚ)) = (n : ℚ) := by
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · subst h0
    rw [Nat.cast_zero, Format.roundRat_zero, Format.toRat_setSign]
    have hmag : f.mag 0 = 0 := by unfold Format.mag; simp
    cases zs <;> simp [hmag, Format.magVal_zero]
  · have hqpos : (0 : ℚ) < (n : ℚ) := by exact_mod_cast hpos
    have hq0 : ((n : ℚ)) ≠ 0 := ne_of_gt hqpos
    have habs : |((n : ℚ))| = (n : ℚ) := abs_of_pos hqpos
    have hq : |((n : ℚ))| ≤ f.maxVal := by rw [habs]; exact hmax
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec hf rnd sat zs hq0 hq
    rw [hval, ite_eq_right (not_lt.mpr (le_of_lt hqpos)), one_mul, habs]
    set k : ℤ := Int.log 2 (n : ℚ) with hkdef
    have hk0 : (0 : ℤ) ≤ k := by
      rw [hkdef]
      refine Format.le_log_of_le hqpos ?_
      rw [zpow_zero]
      exact_mod_cast (show (1 : ℕ) ≤ n from hpos)
    have hk_le : k ≤ (f.m : ℤ) := by
      rw [hkdef]
      have hlt : Int.log 2 (n : ℚ) < (f.m : ℤ) + 1 := by
        refine Format.log_lt_of_lt hqpos ?_
        calc (n : ℚ) ≤ (2 : ℚ) ^ (f.m : ℤ) := hn
          _ < (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
              rw [zpow_add_one₀ (by norm_num)]
              have hp : (0 : ℚ) < (2 : ℚ) ^ (f.m : ℤ) := by positivity
              linarith
      omega
    have hQ : f.qexp (n : ℚ) = k - (f.m : ℤ) := by
      rw [Format.qexp_eq hqpos, hkdef]
      rw [max_eq_left (by omega)]
    have hsub : (0 : ℤ) ≤ (f.m : ℤ) - k := by omega
    have hK : (n : ℚ) * 2 ^ (-(f.qexp (n : ℚ))) = ((n * 2 ^ ((f.m : ℤ) - k).toNat : ℕ) : ℚ) := by
      have hpow : (2 : ℚ) ^ ((f.m : ℤ) - k) = ((2 ^ ((f.m : ℤ) - k).toNat : ℕ) : ℚ) := by
        rw [show ((f.m : ℤ) - k) = (((f.m : ℤ) - k).toNat : ℤ) from
          (Int.toNat_of_nonneg hsub).symm]
        rw [zpow_natCast]
        push_cast
        rw [Int.toNat_natCast]
      rw [hQ, neg_sub, hpow]
      push_cast
      ring
    exact Format.roundSig_exact rnd hK

/-- Rounding a positive rational at least `2^E` keeps the value at least `2^E`. -/
theorem vcConsts_pow2_le_toRat_roundRat {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} {E : ℤ} (hE : f.emin ≤ E) (hq : (2 : ℚ) ^ E ≤ q) (hmax : |q| ≤ f.maxVal) :
    (2 : ℚ) ^ E ≤ f.toRat (f.roundRat rnd sat zs q) := by
  have hqpos : (0 : ℚ) < q := lt_of_lt_of_le (by positivity) hq
  have hq0 : q ≠ 0 := ne_of_gt hqpos
  have habs : |q| = q := abs_of_pos hqpos
  obtain ⟨-, -, hval⟩ := Format.roundRat_spec hf rnd sat zs hq0 hmax
  rw [hval, ite_eq_right (not_lt.mpr (le_of_lt hqpos)), one_mul, habs]
  have hlog : E ≤ Int.log 2 q := Format.le_log_of_le hqpos hq
  have hN : (2 : ℚ) ^ (f.m : ℤ) ≤ (roundSig rnd q (f.qexp q) : ℚ) := by
    have h := Format.roundSig_ge (f := f) rnd hqpos (le_trans hE hlog)
    rw [zpow_natCast]
    exact_mod_cast h
  have hQge : E - (f.m : ℤ) ≤ f.qexp q := by
    rw [Format.qexp_eq hqpos]
    omega
  have hfac : (2 : ℚ) ^ (E - (f.m : ℤ)) ≤ (2 : ℚ) ^ (f.qexp q) :=
    zpow_le_zpow_right₀ (by norm_num) hQge
  calc (2 : ℚ) ^ E = (2 : ℚ) ^ (f.m : ℤ) * (2 : ℚ) ^ (E - (f.m : ℤ)) := by
        rw [← zpow_add₀ (by norm_num)]
        congr 1
        ring
    _ ≤ (roundSig rnd q (f.qexp q) : ℚ) * (2 : ℚ) ^ (f.qexp q) :=
        mul_le_mul hN hfac (by positivity) (by positivity)

/-- The rounded magnitude of a rational within `2^E` stays within `2^E`. -/
theorem vcConsts_abs_roundRat_le_pow2 {f : Format} (hf : f.Regular) (rnd : RoundMode) (sat zs : Bool)
    {q : ℚ} {E : ℤ} (hE : f.emin ≤ E) (hEmax : (2 : ℚ) ^ E ≤ f.maxVal)
    (hq : |q| ≤ (2 : ℚ) ^ E) :
    |f.toRat (f.roundRat rnd sat zs q)| ≤ (2 : ℚ) ^ E := by
  by_cases hq0 : q = 0
  · rw [hq0, Format.roundRat_zero, Format.toRat_setSign]
    have hmag : f.mag 0 = 0 := by unfold Format.mag; simp
    have h0 : f.magVal (f.mag 0) = 0 := by rw [hmag, Format.magVal_zero]
    have hz : |(if zs then -f.magVal (f.mag 0) else f.magVal (f.mag 0))| = 0 := by
      rw [h0]
      cases zs <;> simp
    rw [hz]
    positivity
  · have hqpos : (0 : ℚ) < |q| := abs_pos.mpr hq0
    have hmax : |q| ≤ f.maxVal := le_trans hq hEmax
    obtain ⟨-, -, hval⟩ := Format.roundRat_spec hf rnd sat zs hq0 hmax
    rw [hval]
    have hsign : |(if q < 0 then (-1 : ℚ) else 1)| = 1 := by
      split_ifs <;> norm_num
    rw [abs_mul, hsign, one_mul, abs_of_nonneg (by positivity)]
    have hQle : f.qexp |q| ≤ E - (f.m : ℤ) := by
      rw [Format.qexp_eq hqpos]
      have hlog : Int.log 2 |q| ≤ E := by
        have hlt : Int.log 2 |q| < E + 1 := by
          refine Format.log_lt_of_lt hqpos ?_
          calc |q| ≤ (2 : ℚ) ^ E := hq
            _ < (2 : ℚ) ^ (E + 1) := by
                rw [zpow_add_one₀ (by norm_num)]
                have hp : (0 : ℚ) < (2 : ℚ) ^ E := by positivity
                linarith
        omega
      omega
    by_cases hQlt : f.qexp |q| < E - (f.m : ℤ)
    · have hN : (roundSig rnd |q| (f.qexp |q|) : ℚ) ≤ (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
        have hb : roundSig rnd |q| (f.qexp |q|) ≤ 2 ^ (f.m + 1) :=
          (Format.roundSig_bounds (f := f) rnd hqpos).1
        calc (roundSig rnd |q| (f.qexp |q|) : ℚ) ≤ ((2 ^ (f.m + 1) : ℕ) : ℚ) := by
              exact_mod_cast hb
          _ = (2 : ℚ) ^ ((f.m : ℤ) + 1) := by
              rw [show ((f.m : ℤ) + 1) = ((f.m + 1 : ℕ) : ℤ) from by push_cast; ring,
                zpow_natCast]
              push_cast
              ring
      have hQl : f.qexp |q| ≤ E - (f.m : ℤ) - 1 := by omega
      have hfac : (2 : ℚ) ^ (f.qexp |q|) ≤ (2 : ℚ) ^ (E - (f.m : ℤ) - 1) :=
        zpow_le_zpow_right₀ (by norm_num) hQl
      calc (roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|)
          ≤ (2 : ℚ) ^ ((f.m : ℤ) + 1) * (2 : ℚ) ^ (E - (f.m : ℤ) - 1) :=
            mul_le_mul hN hfac (by positivity) (by positivity)
        _ = (2 : ℚ) ^ E := by
            rw [← zpow_add₀ (by norm_num)]
            congr 1
            ring
    · have hQeq : f.qexp |q| = E - (f.m : ℤ) := le_antisymm hQle (not_lt.mp hQlt)
      have hxle : |q| * (2 : ℚ) ^ (-(E - (f.m : ℤ))) ≤ ((2 ^ f.m : ℕ) : ℚ) := by
        have hp : (0 : ℚ) < (2 : ℚ) ^ (-(E - (f.m : ℤ))) := by positivity
        calc |q| * (2 : ℚ) ^ (-(E - (f.m : ℤ)))
            ≤ (2 : ℚ) ^ E * (2 : ℚ) ^ (-(E - (f.m : ℤ))) :=
              mul_le_mul_of_nonneg_right hq (le_of_lt hp)
          _ = ((2 ^ f.m : ℕ) : ℚ) := by
              rw [← zpow_add₀ (by norm_num)]
              rw [show E + -(E - (f.m : ℤ)) = (f.m : ℤ) from by ring, zpow_natCast]
              push_cast
              ring
      have hrnat : rnat rnd (|q| * (2 : ℚ) ^ (-(E - (f.m : ℤ)))) ≤ 2 ^ f.m :=
        rnat_le rnd (by positivity) hxle
      have hN : (roundSig rnd |q| (f.qexp |q|) : ℚ) ≤ (2 : ℚ) ^ (f.m : ℤ) := by
        rw [hQeq, Format.roundSig_eq]
        calc (rnat rnd (|q| * (2 : ℚ) ^ (-(E - (f.m : ℤ)))) : ℚ) ≤ ((2 ^ f.m : ℕ) : ℚ) := by
              exact_mod_cast hrnat
          _ = (2 : ℚ) ^ (f.m : ℤ) := by
              rw [zpow_natCast]
              push_cast
              ring
      calc (roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|)
          = (roundSig rnd |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (E - (f.m : ℤ)) := by rw [hQeq]
        _ ≤ (2 : ℚ) ^ (f.m : ℤ) * (2 : ℚ) ^ (E - (f.m : ℤ)) :=
            mul_le_mul_of_nonneg_right hN (by positivity)
        _ = (2 : ℚ) ^ E := by
            rw [← zpow_add₀ (by norm_num)]
            congr 1
            ring


/-- Facts about the `float64` value of the divisor `d ≤ 32768`: finite, its value, its magnitude,
nonzero. -/
theorem ofRat_d_facts {n : ℕ} (hn1 : 1 ≤ n) (hn : n ≤ 32768) :
    (FloatV.ofRat (n : ℚ) : F64).isFinite = true ∧
    (FloatV.ofRat (n : ℚ) : F64).toRat = (n : ℚ) ∧
    |(FloatV.ofRat (n : ℚ) : F64).toRat| ≤ Format.binary32.maxVal ∧
    (FloatV.ofRat (n : ℚ) : F64).isZero = false := by
  have hDpos : (0 : ℚ) < (n : ℚ) := by exact_mod_cast hn1
  have hD512 : (n : ℚ) ≤ 32768 := by exact_mod_cast hn
  have h64max : (n : ℚ) ≤ Format.binary64.maxVal :=
    le_trans hD512 (le_trans (by rw [Format.binary32_maxVal]; norm_num)
      Format.wider_binary32_binary64.2.2)
  have h32max : (n : ℚ) ≤ Format.binary32.maxVal :=
    le_trans hD512 (by rw [Format.binary32_maxVal]; norm_num)
  have h64abs : |(n : ℚ)| ≤ Format.binary64.maxVal := by
    rw [abs_of_pos hDpos]; exact h64max
  have h32abs : |(n : ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos hDpos]; exact h32max
  have hlt64 := vcConsts_roundRat_lt_of_le (f := Format.binary64) Format.binary64_regular
    RoundMode.rn false false h64abs
  have hem64 : Format.binary64.emin ≤ 0 := by
    unfold Format.emin Format.bias Format.binary64; norm_num
  have htoRat : (FloatV.ofRat (n : ℚ) : F64).toRat = (n : ℚ) := by
    unfold FloatV.ofRat FloatV.toRat
    rw [FloatV.pat_ofPat hlt64]
    exact vcConsts_toRat_roundRat_natCast (f := Format.binary64) Format.binary64_regular
      RoundMode.rn false false hem64 (le_trans hD512 (by norm_num [Format.binary64])) h64max
  refine ⟨?_, htoRat, ?_, ?_⟩
  · unfold FloatV.ofRat FloatV.isFinite
    rw [FloatV.pat_ofPat hlt64]
    exact vcConsts_isFinite_roundRat_of_le Format.binary64_regular .rn false false h64abs
  · rw [htoRat]; exact h32abs
  · exact vcConsts_isZero_eq_false (by rw [htoRat]; exact ne_of_gt hDpos)

/-- **The divisor `d`** of lines 75 and 77, built in `float32`: finite, of value `d`. -/
theorem dconst {n : ℕ} (hn1 : 1 ≤ n) (hn : n ≤ 32768) :
    (FloatV.litF32 (FloatV.ofRat (n : ℚ))).isFinite = true ∧
      (FloatV.litF32 (FloatV.ofRat (n : ℚ))).toReal = (n : ℝ) := by
  obtain ⟨hc_fin, hc_toRat, hc_abs, hc_zero⟩ := ofRat_d_facts hn1 hn
  have hD512 : (n : ℚ) ≤ 32768 := by exact_mod_cast hn
  have h32max : (n : ℚ) ≤ Format.binary32.maxVal :=
    le_trans hD512 (by rw [Format.binary32_maxVal]; norm_num)
  have hval : Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false
      (FloatV.ofRat (n : ℚ) : F64).sign (FloatV.ofRat (n : ℚ) : F64).toRat)
      = (n : ℚ) := by
    rw [hc_toRat]
    exact vcConsts_toRat_roundRat_natCast (f := Format.binary32) Format.binary32_regular
      RoundMode.rn false _ (by rw [Format.binary32_emin]; norm_num)
      (le_trans hD512 (by norm_num [Format.binary32])) h32max
  unfold FloatV.litF32
  simp only [hc_zero, Bool.false_eq_true, ite_false]
  refine ⟨?_, vcConsts_hostF32_toReal hc_fin hc_abs hval⟩
  rw [vcConsts_hostF32_eq_ofPat hc_fin]
  unfold FloatV.isFinite
  rw [FloatV.pat_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hc_abs)]
  exact vcConsts_isFinite_roundRat_of_le Format.binary32_regular .rn false _ hc_abs

/-- A `float32` constant given by its exact rational value. -/
theorem const_of {x : F32} {q : ℚ} (hf : x.isFinite = true) (hq : x.toRat = q) :
    x.isFinite = true ∧ x.toReal = (q : ℝ) := ⟨hf, by unfold FloatV.toReal; rw [hq]⟩

/-- The source's `0.044715` in `float32`: `12003091 / 2^28`. -/
theorem c_a : (c32 0x3FA6E4E26D4801F7).isFinite = true ∧
    (c32 0x3FA6E4E26D4801F7).toReal = ((12003091 / 268435456 : ℚ) : ℝ) :=
  const_of (by decide +kernel) (by decide +kernel)
/-- The source's `0.7978845608028654` in `float32`: `6693141 / 2^23`. -/
theorem c_k : (c32 0x3FE9884533D43651).isFinite = true ∧
    (c32 0x3FE9884533D43651).toReal = ((6693141 / 8388608 : ℚ) : ℝ) :=
  const_of (by decide +kernel) (by decide +kernel)
/-- `2.0` in `float32`. -/
theorem c_two : (c32 0x4000000000000000).isFinite = true ∧
    (c32 0x4000000000000000).toReal = ((2 : ℚ) : ℝ) :=
  const_of (by decide +kernel) (by decide +kernel)
/-- `1.0` in `float32`. -/
theorem c_one : (c32 0x3FF0000000000000).isFinite = true ∧
    (c32 0x3FF0000000000000).toReal = ((1 : ℚ) : ℝ) :=
  const_of (by decide +kernel) (by decide +kernel)
/-- `0.5` in `float32`. -/
theorem c_half : (c32 0x3FE0000000000000).isFinite = true ∧
    (c32 0x3FE0000000000000).toReal = ((1 / 2 : ℚ) : ℝ) :=
  const_of (by decide +kernel) (by decide +kernel)
/-- `0.0` in `float32` is `+0`. -/
theorem c_zero : c32 0 = FloatV.ofPat 0 := by decide +kernel
/-- `tl.sigmoid`'s `1` (`FloatV.ofRat 1`) in `float32`. -/
theorem c_sig1 : (FloatV.ofRat 1 : F32).isFinite = true ∧ (FloatV.ofRat 1 : F32).toReal = ((1 : ℚ) : ℝ) :=
  const_of (by decide +kernel) (by decide +kernel)

end

end Claims.GeluAndMulSparse
