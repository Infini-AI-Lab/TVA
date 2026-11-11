import Proof.FloatOps

/-!
# The float constants of the kernel

`−∞`, `−1e20` (`neg1e20`), the host's scale `1.0`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- The claim's `−∞` is `FloatV.negInf`. -/
theorem ninf32_eq : ninf32 = (FloatV.negInf : F32) := by
  unfold ninf32 FloatV.litF32
  decide

/-- The double `−1e20` of the source. -/
theorem neg1e20_double : (FloatV.ofPat 0xC415AF1D78B58C40 : F64).toRat = -100000000000000000000 := by
  rw [FloatV.toRat_ofPat (by decide)]
  simp only [Format.toRat, Format.magVal, Format.mag, Format.pow2, Format.signBit, Format.sign,
    Format.bias, Format.emin, Format.binary64]
  norm_num

/-- `neg1e20` is a finite `float32` near `−10^20`. -/
theorem neg1e20_spec : neg1e20.isFinite = true ∧ neg1e20.toReal ≤ -2 ^ 60 ∧ -2 ^ 70 ≤ neg1e20.toReal := by
  have hc : (FloatV.ofPat 0xC415AF1D78B58C40 : F64).isFinite = true := by decide
  have hn : (FloatV.ofPat 0xC415AF1D78B58C40 : F64).isNaN = false := (FloatV.isFinite_iff.mp hc).1
  have hz : (FloatV.ofPat 0xC415AF1D78B58C40 : F64).isZero = false := by decide
  have hmax : |(FloatV.ofPat 0xC415AF1D78B58C40 : F64).toReal| ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    unfold FloatV.toReal
    rw [neg1e20_double, Format.binary32_maxVal_real]; norm_num
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.binary32) Format.binary32_regular hc hmax
  have e : neg1e20 = (FloatV.cvt (FloatV.ofPat 0xC415AF1D78B58C40 : F64) : F32) := by
    unfold neg1e20 FloatV.litF32 FloatV.hostF32
    simp [hz, hn]
  rw [e]
  refine ⟨h1, ?_, ?_⟩ <;>
  · unfold FloatV.toReal at h2 ⊢
    rw [neg1e20_double] at h2
    rw [Format.binary32_u_real, Format.binary32_eta_real] at h2
    have := abs_le.mp h2
    push_cast at this
    norm_num at this ⊢
    linarith [this.1, this.2]

end

end Claims.ExtendAttn
