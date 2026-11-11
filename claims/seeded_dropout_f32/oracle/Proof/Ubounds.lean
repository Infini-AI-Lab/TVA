import Proof.Bounds

/-!
# The random number stays below `1`

The rounding of a non-negative integer below `2^31` is non-negative and at most `2^31`, and the
product of that rounding with `Philox.scale32 = 2^-31 · (1 - 2^-24)` rounds to a `float32` below
`1`. This is what makes `keep d j = true` imply that the kernel's `p` is below `1`.
-/

namespace Claims.SeededDropout

open TVA
open TVA.Format

/-- The magnitude a rounding to nearest of a positive value gives is below the sign bit: it is
`pack`'s (at most `maxVal`'s magnitude), `maxMag`, or the infinity's magnitude, all below
`signBit`. -/
theorem roundMag32_lt_signBit {a : ℚ} (ha : 0 < a) :
    Format.binary32.roundMag .rn false a < Format.binary32.signBit := by
  obtain ⟨hN, hQ, hsub⟩ := Format.roundSig_bounds (f := Format.binary32) .rn ha
  unfold Format.roundMag Format.encode
  rw [pow2_eq]
  split_ifs with hle
  · exact (Format.pack_le_maxMag Format.binary32_regular hN hQ hsub hle).trans_lt
      (Format.maxMag_lt Format.binary32_regular)
  · exact Format.maxMag_lt Format.binary32_regular
  · exact FloatV.infMag_lt_signBit (f := Format.binary32)

/-- The `float32` nearest to a non-negative integer is non-negative. -/
theorem ofRat_toRat_nonneg {z : Nat} : 0 ≤ (FloatV.ofRat (z : ℚ) : F32).toRat := by
  unfold FloatV.ofRat
  by_cases h0 : (z : ℚ) = 0
  · obtain ⟨h1, -, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn)
      (sat := false) (zs := false) Format.binary32_regular
    rw [h0, FloatV.toRat_ofPat h1, h3]
  · have hznn : (0 : ℚ) ≤ (z : ℚ) := by positivity
    have hz0 : ¬ ((z : ℚ) < 0) := not_lt.mpr hznn
    have hmag : 0 < |(z : ℚ)| := abs_pos.mpr h0
    have hmaglt := roundMag32_lt_signBit (a := (z : ℚ)) (by positivity)
    -- the pattern is the magnitude of the rounded value, which has no sign bit
    have hpat : Format.binary32.roundRat .rn false false (z : ℚ)
        = Format.binary32.roundMag .rn false (z : ℚ) := by
      rw [Format.roundRat_of_ne h0, ite_eq_right hz0, abs_of_nonneg hznn, Nat.zero_add]
    have hwidth : Format.binary32.roundMag .rn false (z : ℚ) < 2 ^ Format.binary32.width :=
      hmaglt.trans Format.binary32.signBit_lt
    rw [hpat, FloatV.toRat_ofPat hwidth]
    have hsign : Format.binary32.sign (Format.binary32.roundMag .rn false (z : ℚ)) = false := by
      unfold Format.sign
      rw [Nat.div_eq_of_lt hmaglt]
      rfl
    rw [Format.toRat_eq, hsign]
    simp only [Bool.false_eq_true, ite_false]
    exact Format.magVal_nonneg _

/-- The `float32` nearest to an integer below `2^31` is at most `2^31`. -/
theorem ofRat_toRat_le_two_pow_31 {z : Nat} (hz : z < 2 ^ 31) :
    (FloatV.ofRat (z : ℚ) : F32).toRat ≤ 2 ^ 31 := by
  unfold FloatV.ofRat
  rcases Nat.eq_zero_or_pos z with hz0 | hzpos
  · subst hz0
    obtain ⟨h1, -, h3⟩ := Format.roundRat_zero_spec (f := Format.binary32) (rnd := .rn)
      (sat := false) (zs := false) Format.binary32_regular
    simp only [Nat.cast_zero]
    rw [FloatV.toRat_ofPat h1, h3]
    norm_num
  · have hzq : (z : ℚ) < (2 : ℚ) ^ 31 := by
      have h : ((z : ℕ) : ℚ) < ((2 : ℕ) ^ 31 : ℚ) := by exact_mod_cast hz
      push_cast at h
      exact h
    have hq : |(z : ℚ)| ≤ Format.binary32.maxVal := by
      rw [abs_of_nonneg (by positivity)]
      exact hzq.le.trans two_pow_31_le_maxVal
    have hne : (z : ℚ) ≠ 0 := by positivity
    obtain ⟨hlt, -, hval⟩ := Format.roundRat_spec (f := Format.binary32)
      Format.binary32_regular .rn false false hne hq
    rw [FloatV.toRat_ofPat hlt, hval,
      ite_eq_right (not_lt.mpr (by positivity : (0 : ℚ) ≤ (z : ℚ))),
      one_mul, abs_of_nonneg (by positivity)]
    -- the exponent of the quantum is at most `7`, so `2^31 / 2^Q` is the integer `K`
    set Q : ℤ := Format.binary32.qexp (z : ℚ) with hQ
    have hzposq : (0 : ℚ) < (z : ℚ) := by positivity
    have hlog : Int.log 2 (z : ℚ) < 31 := by
      apply Format.log_lt_of_lt hzposq
      rw [show (31 : ℤ) = ((31 : ℕ) : ℤ) by norm_num, zpow_natCast]
      exact hzq
    have hQle : Q ≤ 7 := by
      rw [hQ, Format.qexp_eq hzposq, Format.binary32_emin]
      have hm : (Format.binary32.m : ℤ) = 23 := rfl
      rw [hm]
      have h1 : max (Int.log 2 (z : ℚ)) (-126 : ℤ) ≤ 30 := by
        apply max_le
        · omega
        · norm_num
      omega
    set n : ℕ := (31 - Q).toNat with hn
    have hnQ : (n : ℤ) = 31 - Q := by
      rw [hn]
      exact Int.toNat_of_nonneg (by omega)
    set K : ℕ := 2 ^ n with hK
    have hKn : (K : ℚ) = (2 : ℚ) ^ (n : ℤ) := by
      rw [hK]
      push_cast
      rw [zpow_natCast]
    have hKQ : (K : ℚ) * (2 : ℚ) ^ Q = (2 : ℚ) ^ 31 := by
      rw [hKn, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0), hnQ,
        show 31 - Q + Q = (31 : ℤ) by ring]
      norm_num
    have hQQ : (2 : ℚ) ^ Q * (2 : ℚ) ^ (-Q) = 1 := by
      rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0), show Q + -Q = (0 : ℤ) by ring, zpow_zero]
    have hx : (z : ℚ) * (2 : ℚ) ^ (-Q) ≤ (K : ℚ) := by
      have hp : (0 : ℚ) < (2 : ℚ) ^ (-Q) := by positivity
      have hlt2 : (z : ℚ) * (2 : ℚ) ^ (-Q) < (2 : ℚ) ^ 31 * (2 : ℚ) ^ (-Q) :=
        mul_lt_mul_of_pos_right hzq hp
      have heq : (2 : ℚ) ^ 31 * (2 : ℚ) ^ (-Q) = (K : ℚ) := by
        rw [← hKQ, mul_assoc, hQQ, mul_one]
      exact hlt2.le.trans_eq heq
    have hsig : (Format.roundSig .rn (z : ℚ) Q : ℚ) ≤ (K : ℚ) := by
      have h1 : rnat .rn ((z : ℚ) * (2 : ℚ) ^ (-Q)) ≤ K :=
        rnat_le .rn (by positivity) hx
      rw [Format.roundSig_eq]
      exact_mod_cast h1
    calc (Format.roundSig .rn (z : ℚ) Q : ℚ) * (2 : ℚ) ^ Q ≤ (K : ℚ) * (2 : ℚ) ^ Q :=
          mul_le_mul_of_nonneg_right hsig (by positivity)
      _ = (2 : ℚ) ^ 31 := hKQ

/-- **The random number is below `1`.** -/
theorem uRef_toRat_lt_one (seed : Int) (j : Nat) : (uRef seed j).toRat < 1 := by
  unfold uRef
  set A : F32 := FloatV.ofRat (rint seed j : ℚ) with hAdef
  set S : F32 := Philox.scale32 with hSdef
  have hA0 : 0 ≤ A.toRat := by rw [hAdef]; exact ofRat_toRat_nonneg
  have hA1 : A.toRat ≤ (2 : ℚ) ^ 31 := by
    rw [hAdef]; exact ofRat_toRat_le_two_pow_31 (rint_lt seed j)
  have hAfin : A.isFinite = true := by rw [hAdef]; exact ofRat_isFinite (rint_lt seed j)
  have hSfin : S.isFinite = true := by rw [hSdef]; exact scale32_isFinite
  have hSval : S.toRat = (2 : ℚ) ^ (-31 : ℤ) * (1 - (2 : ℚ) ^ (-24 : ℤ)) := by
    rw [hSdef]; exact scale32_toRat
  have hSpos : 0 < S.toRat := by rw [hSval]; norm_num
  set a : ℚ := A.toRat * S.toRat with hadef
  have ha0 : 0 ≤ a := by rw [hadef]; exact mul_nonneg hA0 hSpos.le
  -- the exact product is at most `2^31 · 2^-31 · (1 - 2^-24) = 1 - 2^-24`
  have hprod_le : a ≤ 1 - (2 : ℚ) ^ (-24 : ℤ) := by
    rw [hadef]
    have h1 : A.toRat * S.toRat ≤ (2 : ℚ) ^ 31 * S.toRat :=
      mul_le_mul_of_nonneg_right hA1 hSpos.le
    have h2 : (2 : ℚ) ^ 31 * S.toRat = 1 - (2 : ℚ) ^ (-24 : ℤ) := by
      rw [hSval, ← zpow_natCast (a := (2 : ℚ)) (n := 31), ← mul_assoc,
        ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
      norm_num
    exact h1.trans_eq h2
  have h24le : (1 : ℚ) - (2 : ℚ) ^ (-24 : ℤ) ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  have hmax : |a| ≤ Format.binary32.maxVal := by
    rw [abs_of_nonneg ha0]
    exact hprod_le.trans h24le
  obtain ⟨-, herr⟩ := FloatV.mul_err Format.binary32_regular hAfin hSfin hmax
  -- the rounding error is at most `(1 - 2^-24)·2^-24 + 2^-150`, which fits in `2^-48`
  have h48 : (2 : ℚ) ^ (-48 : ℤ) = (2 : ℚ) ^ (-24 : ℤ) * (2 : ℚ) ^ (-24 : ℤ) := by
    rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
    norm_num
  have h150 : (2 : ℚ) ^ (-150 : ℤ) < (2 : ℚ) ^ (-48 : ℤ) :=
    zpow_lt_zpow_right₀ (by norm_num : (1 : ℚ) < 2) (by norm_num : (-150 : ℤ) < -48)
  have hupper : (FloatV.mul A S).toRat
      ≤ a + a * (2 : ℚ) ^ (-24 : ℤ) + (2 : ℚ) ^ (-150 : ℤ) := by
    have hle := (abs_le.mp herr).2
    rw [abs_of_nonneg ha0, Format.binary32_u, Format.binary32_eta] at hle
    linarith
  have hmul : a * (1 + (2 : ℚ) ^ (-24 : ℤ))
      ≤ (1 - (2 : ℚ) ^ (-24 : ℤ)) * (1 + (2 : ℚ) ^ (-24 : ℤ)) := by
    apply mul_le_mul_of_nonneg_right hprod_le
    positivity
  have hsq : (1 - (2 : ℚ) ^ (-24 : ℤ)) * (1 + (2 : ℚ) ^ (-24 : ℤ))
      = 1 - (2 : ℚ) ^ (-24 : ℤ) * (2 : ℚ) ^ (-24 : ℤ) := by ring
  linarith

end Claims.SeededDropout
