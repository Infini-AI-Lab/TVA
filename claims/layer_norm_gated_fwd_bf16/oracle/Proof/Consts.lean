import Claim
import TVA.Proof

/-!
# `layer_norm_gated_fwd_bf16`: the divisor `D`

The constexpr `D` of `tl.sum(...) / D`, built in `float32` (`litF32` of the double `D`): it is
finite and has the value `D` (`dv_toReal`, `dv_isFinite`), as an integer at most `512` rounds to
itself in `float64` and `float32`.
-/

namespace Claims.LayerNormGatedFwd

open TVA
open Format

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

/-- Facts about the `float64` value of the divisor: finite, its value, its magnitude, nonzero. -/
theorem vcConsts_ofRat_D_facts {d : Inst} (hd : d.ok) :
    (FloatV.ofRat (d.D : ℚ) : F64).isFinite = true ∧
    (FloatV.ofRat (d.D : ℚ) : F64).toRat = (d.D : ℚ) ∧
    |(FloatV.ofRat (d.D : ℚ) : F64).toRat| ≤ Format.binary32.maxVal ∧
    (FloatV.ofRat (d.D : ℚ) : F64).isZero = false := by
  have hDpos : (0 : ℚ) < (d.D : ℚ) := by
    have h : 0 < d.D := hd.d_pos
    exact_mod_cast h
  have hD512 : (d.D : ℚ) ≤ 512 := by exact_mod_cast hd.d_le
  have h64max : (d.D : ℚ) ≤ Format.binary64.maxVal :=
    le_trans hD512 (le_trans (by rw [Format.binary32_maxVal]; norm_num)
      Format.wider_binary32_binary64.2.2)
  have h32max : (d.D : ℚ) ≤ Format.binary32.maxVal :=
    le_trans hD512 (by rw [Format.binary32_maxVal]; norm_num)
  have h64abs : |(d.D : ℚ)| ≤ Format.binary64.maxVal := by
    rw [abs_of_pos hDpos]; exact h64max
  have h32abs : |(d.D : ℚ)| ≤ Format.binary32.maxVal := by
    rw [abs_of_pos hDpos]; exact h32max
  have hlt64 := vcConsts_roundRat_lt_of_le (f := Format.binary64) Format.binary64_regular
    RoundMode.rn false false h64abs
  have hem64 : Format.binary64.emin ≤ 0 := by
    unfold Format.emin Format.bias Format.binary64; norm_num
  have htoRat : (FloatV.ofRat (d.D : ℚ) : F64).toRat = (d.D : ℚ) := by
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

/-- The divisor's value is `D`. -/
theorem dv_toReal {d : Inst} (hd : d.ok) :
    (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))).toReal = (d.D : ℝ) := by
  obtain ⟨hc_fin, hc_toRat, hc_abs, hc_zero⟩ := vcConsts_ofRat_D_facts hd
  have hDpos : (0 : ℚ) < (d.D : ℚ) := by
    have h : 0 < d.D := hd.d_pos
    exact_mod_cast h
  have hD512 : (d.D : ℚ) ≤ 512 := by exact_mod_cast hd.d_le
  have h32max : (d.D : ℚ) ≤ Format.binary32.maxVal :=
    le_trans hD512 (by rw [Format.binary32_maxVal]; norm_num)
  have hval : Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false
      (FloatV.ofRat (d.D : ℚ) : F64).sign (FloatV.ofRat (d.D : ℚ) : F64).toRat)
      = (d.D : ℚ) := by
    rw [hc_toRat]
    exact vcConsts_toRat_roundRat_natCast (f := Format.binary32) Format.binary32_regular
      RoundMode.rn false _ (by rw [Format.binary32_emin]; norm_num)
      (le_trans hD512 (by norm_num [Format.binary32])) h32max
  unfold FloatV.litF32
  simp only [hc_zero, Bool.false_eq_true, ite_false]
  exact vcConsts_hostF32_toReal hc_fin hc_abs hval

/-- The divisor's value is finite. -/
theorem dv_isFinite {d : Inst} (hd : d.ok) :
    (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))).isFinite = true := by
  obtain ⟨hc_fin, hc_toRat, hc_abs, hc_zero⟩ := vcConsts_ofRat_D_facts hd
  unfold FloatV.litF32
  simp only [hc_zero, Bool.false_eq_true, ite_false]
  rw [vcConsts_hostF32_eq_ofPat hc_fin]
  unfold FloatV.isFinite
  rw [FloatV.pat_ofPat (vcConsts_roundRat_lt_of_le Format.binary32_regular .rn false _ hc_abs)]
  exact vcConsts_isFinite_roundRat_of_le Format.binary32_regular .rn false _ hc_abs

end Claims.LayerNormGatedFwd
