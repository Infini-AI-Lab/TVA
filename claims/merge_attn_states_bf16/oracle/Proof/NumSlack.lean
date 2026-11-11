import Claim
import TVA.Proof
import TVA

/-!
# A rounding above `maxVal`, by a hair

The merged lse is `log (refSum) + refMax`, computed as `l + M`. Both addends are finite and
`|l| ≤ 1`, so the exact sum is at most `maxVal + 1` in magnitude, yet it can exceed `maxVal`
(only when `M` is exactly `±maxVal`). Rounding to nearest still gives a finite value there: the
sum stays in the top binade, below `(2^(m+1) − 1/2)` units, and rounds to at most `maxMag`.
-/

namespace Claims.MergeAttnStates

open TVA

/-- Rounding to nearest of a nonnegative value below `k + 1/2` is at most `k`. -/
theorem rnat_rn_le_slack {x : ℚ} {k : ℕ} (hx : 0 ≤ x) (h : x < (k : ℚ) + 1 / 2) :
    Format.rnat .rn x ≤ k := by
  have h1 := Format.rnat_rn_err (x := x) hx
  have h2 : (Format.rnat .rn x : ℚ) < (k : ℚ) + 1 := by
    have := (abs_le.mp h1).2
    linarith
  have h3 : Format.rnat .rn x < k + 1 := by exact_mod_cast h2
  omega

/-- Auxiliary strengthening: the magnitude rounds to at most `maxMag`. -/
theorem roundMag_rn_le_maxMag_slack {a : ℚ} (h0 : 0 < a)
    (ha : a ≤ Format.binary32.maxVal + 1) :
    Format.binary32.roundMag .rn false a ≤ Format.binary32.maxMag := by
  by_cases hle : a ≤ Format.binary32.maxVal
  · have hN := Format.roundSig_bounds (f := Format.binary32) .rn h0
    have hmax := Format.roundSig_le_maxVal (f := Format.binary32) Format.binary32_regular .rn h0 hle
    have hpack := Format.pack_le_maxMag (f := Format.binary32) Format.binary32_regular
      hN.1 hN.2.1 hN.2.2 hmax
    rwa [Format.roundMag_of_le (f := Format.binary32) Format.binary32_regular .rn false h0 hle]
  · replace hle : Format.binary32.maxVal < a := not_le.mp hle
    have hm : Format.binary32.m = 23 := rfl
    have hbias : Format.binary32.bias = 127 := rfl
    have htopExp : Format.binary32.topExp = 254 := rfl
    have hem : Format.binary32.emin = -126 := Format.binary32_emin
    set T : ℤ := (Format.binary32.topExp : ℤ) - Format.binary32.bias - (Format.binary32.m : ℤ) with hTdef
    have hT104 : T = 104 := by
      rw [hTdef, htopExp, hbias, hm]
      norm_num
    have hmaxv : Format.binary32.maxVal = (Format.binary32.topSig : ℚ) * (2 : ℚ) ^ T := by
      rw [Format.maxVal_eq, ← hTdef]
    have hlog127 : Int.log 2 a = 127 := by
      have hlo : (127 : ℤ) ≤ Int.log 2 a := by
        apply Format.le_log_of_le h0
        have h2 : (2 : ℚ) ^ (127 : ℤ) ≤ Format.binary32.maxVal := by
          rw [Format.binary32_maxVal]
          norm_num
        linarith
      have hhi : Int.log 2 a < 128 := by
        apply Format.log_lt_of_lt h0
        have h4 : Format.binary32.maxVal + 1 < (2 : ℚ) ^ (128 : ℤ) := by
          rw [Format.binary32_maxVal]
          norm_num
        linarith
      omega
    have hqexp : Format.binary32.qexp a = T := by
      have hmlog : Format.binary32.emin ≤ Int.log 2 a := by
        rw [hem, hlog127]; norm_num
      rw [Format.qexp_eq h0, max_eq_left hmlog, hlog127, hTdef, htopExp, hbias, hm]
      norm_num
    set x : ℚ := a * (2 : ℚ) ^ (-T) with hxdef
    have hx0 : 0 ≤ x := by rw [hxdef]; positivity
    have hmul : Format.binary32.maxVal * (2 : ℚ) ^ (-T) = (Format.binary32.topSig : ℚ) := by
      rw [hmaxv, mul_assoc, ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
      simp
    have hxgt : (Format.binary32.topSig : ℚ) < x := by
      have hp : (0 : ℚ) < (2 : ℚ) ^ (-T) := by positivity
      have h := mul_lt_mul_of_pos_right hle hp
      rw [hmul] at h
      rwa [hxdef]
    have hxle : x ≤ (Format.binary32.topSig : ℚ) + (2 : ℚ) ^ (-T) := by
      have hp : (0 : ℚ) < (2 : ℚ) ^ (-T) := by positivity
      have h := mul_le_mul_of_nonneg_right ha hp.le
      have h1 : (Format.binary32.maxVal + 1) * (2 : ℚ) ^ (-T) =
          (Format.binary32.topSig : ℚ) + (2 : ℚ) ^ (-T) := by
        rw [add_mul, one_mul, hmul]
      rw [hxdef]
      calc a * (2 : ℚ) ^ (-T) ≤ (Format.binary32.maxVal + 1) * (2 : ℚ) ^ (-T) := h
        _ = (Format.binary32.topSig : ℚ) + (2 : ℚ) ^ (-T) := h1
    have hxlt : x < (Format.binary32.topSig : ℚ) + 1 / 2 := by
      have h2T : (2 : ℚ) ^ (-T) < 1 / 2 := by
        rw [hT104]; norm_num
      linarith
    have hNtop : Format.roundSig .rn a (Format.binary32.qexp a) = Format.binary32.topSig := by
      rw [hqexp, Format.roundSig_eq]
      show Format.rnat .rn x = Format.binary32.topSig
      have h1 : Format.binary32.topSig ≤ Format.rnat .rn x := by
        apply Format.le_rnat
        exact hxgt.le
      have h2 : Format.rnat .rn x ≤ Format.binary32.topSig := rnat_rn_le_slack hx0 hxlt
      omega
    have hval : (Format.roundSig .rn a (Format.binary32.qexp a) : ℚ) *
        (2 : ℚ) ^ (Format.binary32.qexp a) = Format.binary32.maxVal := by
      rw [hNtop, hqexp]
      exact hmaxv.symm
    have henc : Format.binary32.roundMag .rn false a =
        Format.binary32.pack (Format.roundSig .rn a (Format.binary32.qexp a))
          (Format.binary32.qexp a) := by
      unfold Format.roundMag Format.encode
      rw [Format.pow2_eq, ite_eq_left (le_of_eq hval)]
    have hpack : Format.binary32.pack (Format.roundSig .rn a (Format.binary32.qexp a))
        (Format.binary32.qexp a) = Format.binary32.maxMag := by
      rw [hNtop, hqexp]
      have hNle : Format.binary32.topSig ≤ 2 ^ (Format.binary32.m + 1) := le_of_lt Format.topSig_lt
      have hQge : Format.binary32.emin - Format.binary32.m ≤ T := by
        rw [hT104, hem, hm]; norm_num
      have hsub : Format.binary32.topSig < 2 ^ Format.binary32.m →
          T = Format.binary32.emin - Format.binary32.m := by
        intro hc
        exact absurd hc (not_lt.mpr (Format.topSig_ge Format.binary32_regular))
      have h1 : Format.binary32.magVal (Format.binary32.pack Format.binary32.topSig T) =
          (Format.binary32.topSig : ℚ) * (2 : ℚ) ^ T :=
        Format.magVal_pack (f := Format.binary32) hNle hQge hsub
      have h2 : (Format.binary32.topSig : ℚ) * (2 : ℚ) ^ T = Format.binary32.maxVal := hmaxv.symm
      have h3 : Format.binary32.magVal Format.binary32.maxMag = Format.binary32.maxVal :=
        Format.magVal_maxMag Format.binary32_regular
      apply Format.magVal_inj
      rw [h1, h2, h3]
    exact (henc.trans hpack).le

/-- **A positive value at most one unit above `float32`'s `maxVal` rounds to a finite
magnitude.** -/
theorem roundMag_rn_finite_of_le_slack {a : ℚ} (h0 : 0 < a)
    (ha : a ≤ Format.binary32.maxVal + 1) :
    Format.binary32.isFinite (Format.binary32.roundMag .rn false a) = true := by
  have hle := roundMag_rn_le_maxMag_slack h0 ha
  exact Format.isFinite_of_mag_le (f := Format.binary32) Format.binary32_regular
    ((Format.mag_le (f := Format.binary32) (Format.binary32.roundMag .rn false a)).trans hle)

/-- **A nonzero value of magnitude at most one unit above `float32`'s `maxVal` rounds to a finite
pattern.** -/
theorem roundRat_rn_finite_of_le_slack {q : ℚ} (h0 : q ≠ 0)
    (hq : |q| ≤ Format.binary32.maxVal + 1) (zs : Bool) :
    Format.binary32.isFinite (Format.binary32.roundRat .rn false zs q) = true := by
  have ha : 0 < |q| := abs_pos.mpr h0
  have hle := roundMag_rn_le_maxMag_slack ha hq
  have hlt : Format.binary32.roundMag .rn false |q| < Format.binary32.signBit :=
    hle.trans_lt (Format.maxMag_lt Format.binary32_regular)
  rw [Format.roundRat_of_ne (f := Format.binary32) h0]
  have hset : (if q < 0 then Format.binary32.signBit else 0) +
      Format.binary32.roundMag .rn false |q| =
      Format.binary32.setSign (decide (q < 0)) (Format.binary32.roundMag .rn false |q|) := by
    rw [Format.setSign_of_lt hlt]
    by_cases h : q < 0 <;> simp [h]
  rw [hset, Format.isFinite_setSign]
  exact roundMag_rn_finite_of_le_slack ha hq

/-- **Adding a small finite value to a finite `float32`**: the result is finite and within one
rounding of the exact sum (relative to itself), even when the sum exceeds `maxVal` by at most
one. -/
theorem add_err_slack {l M r : F32} (hl : l.isFinite = true) (hM : M.isFinite = true)
    (hr : FloatV.NaNEq (FloatV.add l M) r) (hq0 : l.toRat + M.toRat ≠ 0)
    (hq : |l.toRat + M.toRat| ≤ Format.binary32.maxVal + 1) :
    r.isFinite = true ∧
      |r.toRat - (l.toRat + M.toRat)| ≤
        |r.toRat| * Format.binary32.u + Format.binary32.eta := by
  have hmag := roundMag_rn_le_maxMag_slack (abs_pos.mpr hq0) hq
  have hlt : Format.binary32.roundMag .rn false
      |Format.binary32.toRat l.pat + Format.binary32.toRat M.pat| < Format.binary32.signBit :=
    hmag.trans_lt (Format.maxMag_lt Format.binary32_regular)
  have hrlt : Format.binary32.roundRat .rn false
      (Format.binary32.sign l.pat && Format.binary32.sign M.pat)
      (Format.binary32.toRat l.pat + Format.binary32.toRat M.pat) <
        2 ^ Format.binary32.width := by
    rw [Format.roundRat_of_ne (f := Format.binary32)
      (q := Format.binary32.toRat l.pat + Format.binary32.toRat M.pat) hq0]
    have hset : (if Format.binary32.toRat l.pat + Format.binary32.toRat M.pat < 0 then
          Format.binary32.signBit else 0) +
        Format.binary32.roundMag .rn false
          |Format.binary32.toRat l.pat + Format.binary32.toRat M.pat| =
        Format.binary32.setSign (decide (Format.binary32.toRat l.pat + Format.binary32.toRat M.pat < 0))
          (Format.binary32.roundMag .rn false
            |Format.binary32.toRat l.pat + Format.binary32.toRat M.pat|) := by
      rw [Format.setSign_of_lt hlt]
      by_cases h : Format.binary32.toRat l.pat + Format.binary32.toRat M.pat < 0 <;> simp [h]
    rw [hset]
    exact Format.setSign_lt _ _
  have hadd : FloatV.add l M = FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign l.pat && Format.binary32.sign M.pat)
      (Format.binary32.toRat l.pat + Format.binary32.toRat M.pat)) := by
    unfold FloatV.add
    rw [Format.add_of_finite .rn hl hM]
  have hfin_round : Format.binary32.isFinite (Format.binary32.roundRat .rn false
      (Format.binary32.sign l.pat && Format.binary32.sign M.pat)
      (Format.binary32.toRat l.pat + Format.binary32.toRat M.pat)) = true :=
    roundRat_rn_finite_of_le_slack hq0 hq _
  have hadd_fin : (FloatV.add l M).isFinite = true := by
    rw [hadd, FloatV.isFinite_ofPat hrlt]
    exact hfin_round
  have hnan : (FloatV.add l M).isNaN = false :=
    (FloatV.isFinite_iff (x := FloatV.add l M)).mp hadd_fin |>.1
  have hr_eq : r = FloatV.ofPat (Format.binary32.roundRat .rn false
      (Format.binary32.sign l.pat && Format.binary32.sign M.pat)
      (Format.binary32.toRat l.pat + Format.binary32.toRat M.pat)) := by
    rw [hr.eq hnan]
    exact hadd
  have herr := Format.roundRat_rn_err_of_finite (f := Format.binary32)
    Format.binary32_regular (by decide : Format.binary32.specials ≠ .finite)
    (Format.binary32.sign l.pat && Format.binary32.sign M.pat) hfin_round
  refine ⟨?_, ?_⟩
  · rw [hr_eq, FloatV.isFinite_ofPat hrlt]
    exact hfin_round
  · rw [hr_eq, FloatV.toRat_ofPat hrlt]
    exact herr

end Claims.MergeAttnStates
