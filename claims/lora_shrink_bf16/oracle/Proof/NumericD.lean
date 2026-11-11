import Proof.NumericC

/-!
# `lora_shrink_bf16`: the host's rounding of `scaling`

The kernel's `float32` `scaling` is the host's double rounded to nearest. When that is finite it is
within `u·|rounded| + η` of the double (also in the corner just above the largest `float32`, which
rounds down to it).
-/

namespace Claims.LoraShrink

open TVA TVA.Format

/-- A rounding to nearest (no saturation) of a format with infinities is a pattern of the width. -/
theorem roundRat_rn_lt {f : Format} (hf : f.Regular) (hs : f.specials ≠ .finite) (zs : Bool)
    (q : ℚ) : f.roundRat .rn false zs q < 2 ^ f.width := by
  by_cases hq0 : q = 0
  · subst hq0; exact (roundRat_zero_spec hf).1
  have ha : 0 < |q| := abs_pos.mpr hq0
  obtain ⟨hN, hQ, hsub⟩ := roundSig_bounds (f := f) .rn ha
  have hmlt : f.roundMag .rn false |q| < f.signBit := by
    by_cases hno : (roundSig .rn |q| (f.qexp |q|) : ℚ) * (2 : ℚ) ^ (f.qexp |q|) ≤ f.maxVal
    · have hmag : f.roundMag .rn false |q| =
          f.pack (roundSig .rn |q| (f.qexp |q|)) (f.qexp |q|) := by
        unfold roundMag encode; rw [pow2_eq, ite_eq_left hno]
      rw [hmag]
      exact (pack_le_maxMag hf hN hQ hsub hno).trans_lt (maxMag_lt hf)
    · exact (roundMag_rn_of_over hf hs hno).1
  rw [roundRat_of_ne hq0,
    show (if q < 0 then f.signBit else 0) = if decide (q < 0) then f.signBit else 0 by simp,
    ← setSign_of_lt hmlt]
  exact setSign_lt _ _

/-- **The host's `float32` `scaling`** is within `u·|scaling32| + η` of the double, when finite. -/
theorem scaling32_err (d : Inst) (hd : d.ok) :
    |s32 d - d.scaling.toRat| ≤ |s32 d| * Format.binary32.u + Format.binary32.eta := by
  obtain ⟨hfin, h32⟩ := hd.sc_fin
  have hnn : d.scaling.isNaN = false := (FloatV.isFinite_iff.mp hfin).1
  have hs : Format.binary32.specials ≠ .finite := by decide
  unfold s32 scaling32 at *
  unfold FloatV.hostF32 at *
  rw [ite_eq_right (by simp [hnn])] at *
  unfold FloatV.cvt at *
  have hfin' : Format.binary64.isFinite d.scaling.pat = true := hfin
  rw [Format.cvt_of_finite _ _ _ hfin'] at *
  have hlt := roundRat_rn_lt Format.binary32_regular hs (Format.binary64.sign d.scaling.pat)
    (Format.binary64.toRat d.scaling.pat)
  have h1 : (FloatV.ofPat (Format.binary32.roundRat .rn false (Format.binary64.sign d.scaling.pat)
      (Format.binary64.toRat d.scaling.pat)) : F32).isFinite = true := h32
  rw [FloatV.isFinite_ofPat hlt] at h1
  have := roundRat_rn_err_of_finite Format.binary32_regular hs (Format.binary64.sign d.scaling.pat) h1
  rw [FloatV.toRat_ofPat hlt]
  exact this

end Claims.LoraShrink
