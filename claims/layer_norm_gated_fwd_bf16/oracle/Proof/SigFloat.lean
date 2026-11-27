import Claim
import TVA.Proof

/-!
# `layer_norm_gated_fwd_bf16`: float facts the gate needs

* `bf16_gap`: no `bfloat16` magnitude lies strictly between `88.5` and `tl.exp`'s overflow
  threshold `1453635/16384 ≈ 88.7228` (the `bfloat16` values in `[64, 128)` are multiples of
  `1/2`). So a gate's negation either overflows `tl.exp` or is at most `88.5`, where `tl.exp`
  has room;
* `add_posInf`: `x + ∞ = ∞` for a finite `x`;
* the `float32` constant `1` of `tl.sigmoid` (`FloatV.ofRat 1`) and the zeros `div.full.f32`
  returns for an infinite divisor.
-/

namespace Claims.LayerNormGatedFwd

open TVA

/-- No `bfloat16` value has a magnitude strictly between `88.5` and `1453635/16384`. -/
theorem bf16_gap (b : BF16) :
    ¬ ((177 / 2 : ℚ) < |b.toRat| ∧ |b.toRat| < 1453635 / 16384) := by
  rintro ⟨h1, h2⟩
  have habs : |b.toRat| = Format.bfloat16.magVal (Format.bfloat16.mag b.pat) :=
    Format.abs_toRat _
  obtain ⟨S, E, hv, hS, hE⟩ := Format.magVal_sig_exp (f := Format.bfloat16)
    (Format.bfloat16.mag b.pat)
  rw [habs, hv] at h1 h2
  have hm : Format.bfloat16.m = 7 := rfl
  rw [hm] at hS
  rcases le_or_gt (-1 : ℤ) E with hE1 | hE1
  · obtain ⟨k, hk⟩ : ∃ k : ℕ, E + 1 = k := ⟨(E + 1).toNat, by omega⟩
    have h2E : (2 : ℚ) * (S * (2 : ℚ) ^ E) = ((S * 2 ^ k : ℕ) : ℚ) := by
      have : (2 : ℚ) * (2 : ℚ) ^ E = (2 : ℚ) ^ (E + 1) := by
        rw [zpow_add₀ (by norm_num)]; ring
      push_cast
      rw [← zpow_natCast, ← hk, ← this]; ring
    have hlo : (177 : ℚ) < ((S * 2 ^ k : ℕ) : ℚ) := by rw [← h2E]; linarith
    have hhi : ((S * 2 ^ k : ℕ) : ℚ) < 178 := by rw [← h2E]; linarith
    have hlo' : 177 < S * 2 ^ k := by exact_mod_cast hlo
    have hhi' : S * 2 ^ k < 178 := by exact_mod_cast hhi
    omega
  · have hpow : (2 : ℚ) ^ E ≤ (2 : ℚ) ^ (-2 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have hS' : (S : ℚ) ≤ 255 := by exact_mod_cast (by omega : S ≤ 255)
    have : (S : ℚ) * (2 : ℚ) ^ E ≤ 255 * (2 : ℚ) ^ (-2 : ℤ) :=
      mul_le_mul hS' hpow (by positivity) (by norm_num)
    norm_num at this
    linarith

/-- `+∞` is not NaN (a format with infinities). -/
theorem posInf_isNaN {f : Format} (hs : f.specials = .ieee) : (FloatV.posInf : FloatV f).isNaN = false := by
  unfold FloatV.isNaN; rw [FloatV.posInf_pat, Format.isNaN_setSign]
  have := Format.isInf_infMag (f := f) hs
  unfold Format.isInf at this; unfold Format.isNaN
  rw [hs] at this ⊢
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  simp [this.1, this.2]

/-- **`x + ∞ = ∞`** for finite `x`. -/
theorem add_posInf {f : Format} (hs : f.specials = .ieee) {x : FloatV f} (hx : x.isFinite = true) :
    FloatV.add x FloatV.posInf = FloatV.posInf := by
  have hxn := (FloatV.isFinite_iff.mp hx)
  unfold FloatV.add Format.add
  have h1 : f.isNaN (FloatV.posInf : FloatV f).pat = false := posInf_isNaN hs
  have h2 : f.isInf (FloatV.posInf : FloatV f).pat = true := FloatV.posInf_isInf hs
  have h3 : f.isNaN x.pat = false := hxn.1
  have h4 : f.isInf x.pat = false := hxn.2
  rw [h1, h2, h3, h4]
  simp only [Bool.or_false, Bool.false_eq_true, ite_false, ite_true]
  unfold Format.infPat; rw [hs, FloatV.posInf_pat, Format.sign_setSign]
  rfl

/-- The `float32` `1` of `tl.sigmoid`'s body. -/
theorem one32_facts : (FloatV.ofRat 1 : F32).isFinite = true ∧ (FloatV.ofRat 1 : F32).toRat = 1 ∧
    (FloatV.ofRat 1 : F32).sign = false := by
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- The zeros of `float32` are finite and have value `0`. -/
theorem zero32_facts (b : Bool) :
    (FloatV.ofPat (Format.binary32.zeroPat b) : F32).isFinite = true ∧
      (FloatV.ofPat (Format.binary32.zeroPat b) : F32).toRat = 0 := by
  cases b <;> exact ⟨by decide +kernel, by decide +kernel⟩

end Claims.LayerNormGatedFwd
