import Claim
import TVA.Proof

/-!
# One `float32` rounding

A finite round-to-nearest `float32` result is within `|q| · 2^-24 + 2^-150` of the exact value, also above `maxVal`; a non-finite one lies beyond `maxVal`; the overflow threshold of rounding.
-/

namespace Claims.GdnGating

open TVA

/-- A finite round-to-nearest result is within one rounding of the exact value. -/
theorem roundRat_rn_fin_err {zs : Bool} {q : ℚ}
    (h : Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false zs q) = true) :
    |Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false zs q) - q| ≤
      |q| * (2 : ℚ) ^ (-24 : ℤ) + (2 : ℚ) ^ (-150 : ℤ) := by
  by_cases hq : |q| ≤ Format.binary32.maxVal
  · obtain ⟨-, -, herr⟩ := Format.roundRat_rn_err Format.binary32_regular false zs hq
    rwa [Format.binary32_u, Format.binary32_eta] at herr
  · push Not at hq
    have herr := Format.roundRat_rn_err_of_finite Format.binary32_regular (by decide) zs h
    have hle := Format.abs_toRat_le_maxVal Format.binary32_regular h
    rw [Format.binary32_u, Format.binary32_eta] at herr
    have hmul : |Format.binary32.toRat (Format.binary32.roundRat RoundMode.rn false zs q)| *
        (2 : ℚ) ^ (-24 : ℤ) ≤ Format.binary32.maxVal * (2 : ℚ) ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right hle (by positivity)
    have hmul2 : Format.binary32.maxVal * (2 : ℚ) ^ (-24 : ℤ) ≤
        |q| * (2 : ℚ) ^ (-24 : ℤ) :=
      mul_le_mul_of_nonneg_right hq.le (by positivity)
    linarith

/-- A non-finite round-to-nearest result has an exact value past `maxVal`. -/
theorem roundRat_rn_not_finite_gt_maxVal {zs : Bool} {q : ℚ}
    (h : Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false zs q) = false) :
    Format.binary32.maxVal < |q| := by
  by_contra hle
  push Not at hle
  obtain ⟨-, hfin, -⟩ := Format.roundRat_rn_err Format.binary32_regular false zs hle
  rw [h] at hfin
  exact Bool.noConfusion hfin

/-- A positive value that overflows round-to-nearest is at or above the overflow threshold. -/
theorem roundRat_rn_overflow_threshold {zs : Bool} {q : ℚ} (hq : 0 < q)
    (h : Format.binary32.isFinite (Format.binary32.roundRat RoundMode.rn false zs q) = false) :
    (2 : ℚ) ^ (128 : ℤ) - (2 : ℚ) ^ (103 : ℤ) ≤ q := by
  have hq0 : q ≠ 0 := ne_of_gt hq
  -- `roundRat` of a positive value is its rounded magnitude
  have hmag : Format.binary32.roundRat RoundMode.rn false zs q =
      Format.binary32.roundMag RoundMode.rn false q := by
    rw [Format.roundRat_of_ne (f := Format.binary32) hq0,
      ite_eq_right (not_lt.mpr hq.le), zero_add, abs_of_pos hq]
  rw [hmag] at h
  -- the overflow test inside `encode` fails
  have hover : ¬ (Format.roundSig RoundMode.rn q (Format.binary32.qexp q) : ℚ) *
      (2 : ℚ) ^ (Format.binary32.qexp q) ≤ Format.binary32.maxVal := by
    intro hle
    obtain ⟨hN, hQ, hsub⟩ :=
      Format.roundSig_bounds (f := Format.binary32) RoundMode.rn hq
    have hple := Format.pack_le_maxMag Format.binary32_regular hN hQ hsub hle
    have hlt := hple.trans_lt (Format.maxMag_lt Format.binary32_regular)
    have hfin : Format.binary32.isFinite
        (Format.binary32.pack (Format.roundSig RoundMode.rn q (Format.binary32.qexp q))
          (Format.binary32.qexp q)) = true :=
      Format.isFinite_of_mag_le Format.binary32_regular
        (by rw [Format.mag_of_lt hlt]; exact hple)
    have henc : Format.binary32.roundMag RoundMode.rn false q =
        Format.binary32.pack (Format.roundSig RoundMode.rn q (Format.binary32.qexp q))
          (Format.binary32.qexp q) := by
      unfold Format.roundMag Format.encode
      rw [Format.pow2_eq, ite_eq_left hle]
    rw [henc] at h
    rw [hfin] at h
    exact Bool.noConfusion h
  -- so the exact value is above `maxVal`
  have hmax_lt : Format.binary32.maxVal < q := by
    by_contra hle
    exact hover (Format.roundSig_le_maxVal Format.binary32_regular RoundMode.rn hq (not_lt.mp hle))
  have hmaxVal : Format.binary32.maxVal = ((2 : ℚ) ^ (24 : ℤ) - 1) * (2 : ℚ) ^ (104 : ℤ) := by
    rw [Format.binary32_maxVal]
    norm_num
  by_cases hbig : (2 : ℚ) ^ (128 : ℤ) ≤ q
  · have hnn : (0 : ℚ) ≤ (2 : ℚ) ^ (103 : ℤ) := by positivity
    linarith
  · push Not at hbig
    -- the value lies in `maxVal`'s binade: `2^127 ≤ q < 2^128`
    have hlo : (2 : ℚ) ^ (127 : ℤ) ≤ q := by
      have h1 : (2 : ℚ) ^ (127 : ℤ) ≤ ((2 : ℚ) ^ (24 : ℤ) - 1) * (2 : ℚ) ^ (104 : ℤ) := by
        norm_num
      rw [← hmaxVal] at h1
      linarith
    have hlog : Int.log 2 q = 127 := by
      have h2 : 127 ≤ Int.log 2 q := Format.le_log_of_le hq hlo
      have h3 : Int.log 2 q < 128 := Format.log_lt_of_lt hq hbig
      omega
    have hm : Format.binary32.m = 23 := rfl
    have hqexp : Format.binary32.qexp q = 104 := by
      rw [Format.qexp_eq (f := Format.binary32) hq, hlog, Format.binary32_emin, hm]
      norm_num
    have hover' : ¬ (Format.roundSig RoundMode.rn q 104 : ℚ) *
        (2 : ℚ) ^ (104 : ℤ) ≤ Format.binary32.maxVal := by
      rwa [hqexp] at hover
    -- the rounded significand is at least `2^24`
    have hNge : (2 : ℚ) ^ (24 : ℤ) ≤ (Format.roundSig RoundMode.rn q 104 : ℚ) := by
      have hgt : ((2 : ℚ) ^ (24 : ℤ) - 1) < (Format.roundSig RoundMode.rn q 104 : ℚ) := by
        by_contra hle
        push Not at hle
        rw [hmaxVal] at hover'
        exact hover' (mul_le_mul_of_nonneg_right hle (by positivity))
      have hgt' : ((2 ^ 24 - 1 : ℕ) : ℚ) < (Format.roundSig RoundMode.rn q 104 : ℚ) := by
        have he : ((2 ^ 24 - 1 : ℕ) : ℚ) = (2 : ℚ) ^ (24 : ℤ) - 1 := by norm_num
        linarith
      have hnat : (2 ^ 24 - 1 : ℕ) < Format.roundSig RoundMode.rn q 104 := by
        exact_mod_cast hgt'
      have hnat2 : 2 ^ 24 ≤ Format.roundSig RoundMode.rn q 104 := by omega
      have hcast : ((2 ^ 24 : ℕ) : ℚ) ≤ (Format.roundSig RoundMode.rn q 104 : ℚ) := by
        exact_mod_cast hnat2
      have he : ((2 ^ 24 : ℕ) : ℚ) = (2 : ℚ) ^ (24 : ℤ) := by norm_num
      linarith
    -- one rounding to nearest, from `2^128` or more
    have herr := Format.roundSig_rn_err hq 104
    have hhalf : (2 : ℚ) ^ (104 : ℤ) / 2 = (2 : ℚ) ^ (103 : ℤ) := by
      rw [div_eq_iff (by norm_num : (2 : ℚ) ≠ 0),
        ← zpow_add_one₀ (by norm_num : (2 : ℚ) ≠ 0)]
      norm_num
    have hupper : (Format.roundSig RoundMode.rn q 104 : ℚ) *
        (2 : ℚ) ^ (104 : ℤ) - q ≤ (2 : ℚ) ^ (103 : ℤ) := by
      have h := (abs_le.mp herr).2
      rwa [hhalf] at h
    have hlower : (2 : ℚ) ^ (128 : ℤ) ≤ (Format.roundSig RoundMode.rn q 104 : ℚ) *
        (2 : ℚ) ^ (104 : ℤ) := by
      have hmul := mul_le_mul_of_nonneg_right hNge (by positivity : (0 : ℚ) ≤ (2 : ℚ) ^ (104 : ℤ))
      calc (2 : ℚ) ^ (128 : ℤ) = (2 : ℚ) ^ (24 : ℤ) * (2 : ℚ) ^ (104 : ℤ) := by
            rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
            norm_num
        _ ≤ _ := hmul
    linarith

end Claims.GdnGating
