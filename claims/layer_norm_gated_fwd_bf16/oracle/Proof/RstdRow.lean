import Proof.RstdAux
import Proof.Consts

/-!
# `layer_norm_gated_fwd_bf16`: the `rstd` of one row

From the row's sum of squares `s1` (within `(7/8)·2^-13·A + (11/16)·2^-137` of the exact
`A = Σ x_j²`, `RowSum.sum_err_bd`), the kernel computes `var = s1 / D` (`div.full.f32`),
`s2 = var + eps` (fusable), `sq = tl.sqrt(s2)`, `rstd = 1 / sq`. `var` is within
`2^-13·(A/D) + (3/4)·2^-137` of `A/D` (`var_err`), which is the hypothesis of `rstd_err_wide` with
`A/D` for `A`; `eps_transfer` moves the bound from the device's `float32` `eps` to the host's
double: `rstd` is within `2^-12` (relative) of `1 / sqrt(A/D + eps)` (`rstd_row`).
-/

set_option maxHeartbeats 2000000

namespace Claims.LayerNormGatedFwd

open TVA
open TVA TVA.Format

/-- `var = fl(s1 / D)` against `A / D`. -/
theorem var_err {A s1 v D : ℝ} (hA0 : 0 ≤ A) (hD : 1 ≤ D)
    (h1 : |s1 - A| ≤ (7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * A + (11 / 16) * (2 : ℝ) ^ (-137 : ℤ))
    (hv : |v - s1 / D| ≤ (2 : ℝ) ^ (-22 : ℤ) * |s1 / D| + (2 : ℝ) ^ (-148 : ℤ)) :
    |v - A / D| ≤ (2 : ℝ) ^ (-13 : ℤ) * (A / D) + (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by
  have hDpos : 0 < D := by linarith
  have hs1abs : |s1| ≤ A * (1 + 1 / 8192) + 1 / 2 ^ 137 := by
    have h := abs_add_le A (s1 - A)
    rw [show A + (s1 - A) = s1 by ring, abs_of_nonneg hA0] at h
    have e1 : (2 : ℝ) ^ (-13 : ℤ) = 1 / 8192 := by norm_num
    have e2 : (2 : ℝ) ^ (-137 : ℤ) = 1 / 2 ^ 137 := by norm_num
    rw [e1, e2] at h1
    nlinarith
  have hq1 : |s1 / D - A / D| ≤ (7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * (A / D) +
      (11 / 16) * (2 : ℝ) ^ (-137 : ℤ) / D := by
    rw [show s1 / D - A / D = (s1 - A) / D by ring, abs_div, abs_of_pos hDpos,
      div_le_iff₀ hDpos]
    have e : ((7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * (A / D) + (11 / 16) * (2 : ℝ) ^ (-137 : ℤ) / D) * D =
        (7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * A + (11 / 16) * (2 : ℝ) ^ (-137 : ℤ) := by
      field_simp
    rw [e]; exact h1
  have hq2 : |s1 / D| ≤ (A / D) * (1 + 1 / 8192) + 1 / 2 ^ 137 := by
    rw [abs_div, abs_of_pos hDpos, div_le_iff₀ hDpos]
    have e : ((A / D) * (1 + 1 / 8192) + 1 / 2 ^ 137) * D = A * (1 + 1 / 8192) + D / 2 ^ 137 := by
      field_simp
    rw [e]
    have : (1 : ℝ) / 2 ^ 137 ≤ D / 2 ^ 137 := by
      apply div_le_div_of_nonneg_right hD (by positivity)
    linarith
  have hinvD : (11 / 16) * (2 : ℝ) ^ (-137 : ℤ) / D ≤ (11 / 16) * (2 : ℝ) ^ (-137 : ℤ) := by
    apply div_le_self (by positivity) hD
  have htri : |v - A / D| ≤ |v - s1 / D| + |s1 / D - A / D| := by
    have h := abs_add_le (v - s1 / D) (s1 / D - A / D)
    rwa [show v - s1 / D + (s1 / D - A / D) = v - A / D by ring] at h
  have hAD : 0 ≤ A / D := div_nonneg hA0 hDpos.le
  have e3 := mul_le_mul_of_nonneg_left hq2 (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ))
  have hc1 : (2 : ℝ) ^ (-22 : ℤ) * ((A / D) * (1 + 1 / 8192)) + (7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * (A / D)
      ≤ (2 : ℝ) ^ (-13 : ℤ) * (A / D) := by
    have e : (2 : ℝ) ^ (-22 : ℤ) * (1 + 1 / 8192) + (7 / 8) * (2 : ℝ) ^ (-13 : ℤ) ≤
        (2 : ℝ) ^ (-13 : ℤ) := by norm_num
    have := mul_le_mul_of_nonneg_right e hAD
    linarith [show (2 : ℝ) ^ (-22 : ℤ) * ((A / D) * (1 + 1 / 8192)) =
      (2 : ℝ) ^ (-22 : ℤ) * (1 + 1 / 8192) * (A / D) by ring,
      show ((2 : ℝ) ^ (-22 : ℤ) * (1 + 1 / 8192) + (7 / 8) * (2 : ℝ) ^ (-13 : ℤ)) * (A / D) =
      (2 : ℝ) ^ (-22 : ℤ) * (1 + 1 / 8192) * (A / D) + (7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * (A / D) by ring]
  have hc2 : (2 : ℝ) ^ (-22 : ℤ) * (1 / 2 ^ 137) + (2 : ℝ) ^ (-148 : ℤ) +
      (11 / 16) * (2 : ℝ) ^ (-137 : ℤ) ≤ (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := by norm_num
  have e4 : (2 : ℝ) ^ (-22 : ℤ) * ((A / D) * (1 + 1 / 8192) + 1 / 2 ^ 137) =
      (2 : ℝ) ^ (-22 : ℤ) * ((A / D) * (1 + 1 / 8192)) + (2 : ℝ) ^ (-22 : ℤ) * (1 / 2 ^ 137) := by
    ring
  linarith

/-- **The `rstd` of a row**: finite, within `2^-12` (relative) of `1 / sqrt(A/D + eps)`. -/
theorem rstd_row {d : Inst} (hd : d.ok) {P : Platform} (hsqrt_acc : Device.SqrtF32 P)
    (hdiv_full : Device.DivFull P) {fusion : Bool} {A : ℝ} (hA0 : 0 ≤ A)
    (hAe : A + d.eps.toReal ≤ (2 : ℝ) ^ (125 : ℤ))
    {s1 bv s2 sq r : F32} (hs1fin : s1.isFinite = true)
    (hs1 : |s1.toReal - A| ≤ (7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * A + (11 / 16) * (2 : ℝ) ^ (-137 : ℤ))
    (hdv : P.div .binary32 s1 (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))) bv)
    (haddc : AddRel fusion bv (FloatV.hostF32 d.eps) s2)
    (hsq : P.math .sqrt .binary32 s2 sq)
    (hrd : P.div .binary32 (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)) sq r) :
    r.isFinite = true ∧
      |r.toReal - (Real.sqrt (A / d.D + d.eps.toReal))⁻¹| ≤
        (2 : ℝ) ^ (-12 : ℤ) * (Real.sqrt (A / d.D + d.eps.toReal))⁻¹ := by
  have hD1 : (1 : ℝ) ≤ d.D := by exact_mod_cast hd.d_pos
  have hDpos : (0 : ℝ) < d.D := by linarith
  have hDQ : (0 : ℚ) < d.D := by exact_mod_cast hd.d_pos
  -- the host's `eps` and its `float32`
  have he : (2 : ℝ) ^ (-125 : ℤ) ≤ d.eps.toReal := by
    have h := Rat.cast_le (K := ℝ).mpr hd.eps_lo
    push_cast at h
    simpa only [FloatV.toReal] using h
  have he0 : (0 : ℝ) ≤ d.eps.toReal := le_trans (by norm_num) he
  have hepos : (0 : ℚ) < d.eps.toRat := lt_of_lt_of_le (by norm_num) hd.eps_lo
  obtain ⟨heps32_fin, heps32_err⟩ := hostF32_err hd.eps_fin hepos.le hd.eps_hi
  have heps32_errR : |(FloatV.hostF32 d.eps).toReal - d.eps.toReal|
      ≤ (2 : ℝ) ^ (-24 : ℤ) * d.eps.toReal + (2 : ℝ) ^ (-150 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr heps32_err
    simp only [FloatV.toReal]
    push_cast at h ⊢
    simpa only [Rat.cast_sub, Rat.cast_abs] using h
  have he32_lo : (2 : ℝ) ^ (-125 : ℤ) ≤ (FloatV.hostF32 d.eps).toReal :=
    hostF32_ge hd.eps_fin hd.eps_lo hd.eps_hi
  have he32_nn : (0 : ℝ) ≤ (FloatV.hostF32 d.eps).toReal := le_trans (by norm_num) he32_lo
  have he32_hi : (FloatV.hostF32 d.eps).toReal ≤ (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 2 ^ 23) := by
    have := (abs_le.mp heps32_errR).2
    have h1 : d.eps.toReal ≤ (2 : ℝ) ^ (125 : ℤ) := by linarith
    have e : (2 : ℝ) ^ (125 : ℤ) + ((2 : ℝ) ^ (-24 : ℤ) * (2 : ℝ) ^ (125 : ℤ) + (2 : ℝ) ^ (-150 : ℤ)) ≤
        (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 2 ^ 23) := by norm_num
    have h2 := mul_le_mul_of_nonneg_left h1 (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-24 : ℤ))
    have h3 := (abs_le.mp heps32_errR).2
    clear hAe hs1 hdv haddc hsq hrd heps32_err heps32_errR
    simp only [zpow_neg, zpow_ofNat, zpow_negSucc] at *
    linarith
  -- `var = s1 / D`
  have hA125 : A ≤ (2 : ℝ) ^ (125 : ℤ) := by linarith
  have hs1abs : |s1.toReal| ≤ (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 4096) := by
    have h := abs_add_le A (s1.toReal - A)
    rw [show A + (s1.toReal - A) = s1.toReal by ring, abs_of_nonneg hA0] at h
    have := mul_le_mul_of_nonneg_left hA125 (by positivity : (0 : ℝ) ≤ (7 / 8) * (2 : ℝ) ^ (-13 : ℤ))
    have e : (2 : ℝ) ^ (125 : ℤ) + ((7 / 8) * (2 : ℝ) ^ (-13 : ℤ) * (2 : ℝ) ^ (125 : ℤ) +
        (11 / 16) * (2 : ℝ) ^ (-137 : ℤ)) ≤ (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 4096) := by norm_num
    linarith
  have hDv := dv_toReal hd
  have hDf := dv_isFinite hd
  have hDvQ : (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))).toRat = (d.D : ℚ) := by
    have h : (((FloatV.litF32 (FloatV.ofRat (d.D : ℚ))).toRat : ℚ) : ℝ) = ((d.D : ℚ) : ℝ) := by
      rw [← FloatV.toReal_eq, hDv]; push_cast; rfl
    exact_mod_cast h
  have hqb : |s1.toRat / (FloatV.litF32 (FloatV.ofRat (d.D : ℚ))).toRat| ≤ 2 ^ 126 := by
    rw [hDvQ]
    have h : ((|s1.toRat / (d.D : ℚ)| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast
      rw [← FloatV.toReal_eq, abs_div, abs_of_pos hDpos, div_le_iff₀ hDpos]
      have := mul_le_mul_of_nonneg_left hD1 (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ 126)
      have e : (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 4096) ≤ 2 ^ 126 := by norm_num
      linarith
    exact_mod_cast h
  obtain ⟨hbvfin, hbverr⟩ := hdiv_full _ _ bv hdv hs1fin hDf (by rw [hDvQ]; exact ne_of_gt hDQ) hqb
  have hbverrR : |bv.toReal - s1.toReal / d.D| ≤
      (2 : ℝ) ^ (-22 : ℤ) * |s1.toReal / d.D| + (2 : ℝ) ^ (-148 : ℤ) := by
    rw [hDvQ] at hbverr
    have h := Rat.cast_le (K := ℝ).mpr hbverr
    push_cast at h
    simpa only [FloatV.toReal] using h
  have hvar := var_err hA0 hD1 hs1 hbverrR
  -- `s2 = var + eps`
  have hAD0 : 0 ≤ A / d.D := div_nonneg hA0 hDpos.le
  have hADe : A / d.D + d.eps.toReal ≤ (2 : ℝ) ^ (125 : ℤ) := by
    have : A / d.D ≤ A := div_le_self hA0 hD1
    linarith
  have hbvabs : |bv.toReal| ≤ (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 4096) := by
    have h := abs_add_le (A / d.D) (bv.toReal - A / d.D)
    rw [show A / d.D + (bv.toReal - A / d.D) = bv.toReal by ring, abs_of_nonneg hAD0] at h
    have h1 : A / d.D ≤ (2 : ℝ) ^ (125 : ℤ) := by linarith
    have := mul_le_mul_of_nonneg_left h1 (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-13 : ℤ))
    have e : (2 : ℝ) ^ (125 : ℤ) + ((2 : ℝ) ^ (-13 : ℤ) * (2 : ℝ) ^ (125 : ℤ) +
        (3 / 4) * (2 : ℝ) ^ (-137 : ℤ)) ≤ (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 4096) := by norm_num
    linarith
  have hMq : |bv.toRat| + |(FloatV.hostF32 d.eps).toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h : ((|bv.toRat| + |(FloatV.hostF32 d.eps).toRat| : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      rw [← FloatV.toReal_eq, ← FloatV.toReal_eq, abs_of_nonneg he32_nn, Format.binary32_maxVal_real,
        Format.binary32_eta_real]
      have e : (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 4096) + (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 2 ^ 23) ≤
          (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * (2 : ℝ) ^ (-150 : ℤ) := by norm_num
      linarith
    exact_mod_cast h
  obtain ⟨hs2fin, hs2err⟩ := FloatV.AddRel.err (f := Format.binary32) Format.binary32_regular
    (by rw [Format.Std.ieee Format.binary32_std]; decide)
    (by rw [Format.binary32_u]; norm_num [zpow_neg]) haddc hbvfin heps32_fin hMq
  have hs2R : |s2.toReal - (bv.toReal + (FloatV.hostF32 d.eps).toReal)|
      ≤ 3 * (2 : ℝ) ^ (-24 : ℤ) * (|bv.toReal| + (FloatV.hostF32 d.eps).toReal)
        + 4 * (2 : ℝ) ^ (-150 : ℤ) := by
    have h := Rat.cast_le (K := ℝ).mpr hs2err
    push_cast at h
    rw [Format.binary32_u_real, Format.binary32_eta_real] at h
    simp only [FloatV.toReal] at h he32_nn ⊢
    rwa [abs_of_nonneg he32_nn] at h
  -- `tl.sqrt` and `1 / sq`
  have hADe126 : A / d.D + d.eps.toReal ≤ (2 : ℝ) ^ (126 : ℤ) :=
    hADe.trans (by norm_num)
  have hvar' : |bv.toReal - A / d.D| ≤ (2 : ℝ) ^ (-13 : ℤ) * (A / d.D) +
      (3 / 4) * (2 : ℝ) ^ (-137 : ℤ) := hvar
  obtain ⟨hs2lo, hs2hi⟩ := s2_range (A := A / d.D) (e := d.eps.toReal) (s1 := bv.toReal)
    (e32 := (FloatV.hostF32 d.eps).toReal) (s2 := s2.toReal) hAD0 he0 he hADe126 he32_nn hvar'
    heps32_errR hs2R
  obtain ⟨sref, hsref, hacc⟩ := hsqrt_acc
  have hsref' : sref = Real.sqrt := by
    have h : MathFn.ref MathFn.sqrt = some Real.sqrt := rfl
    rw [h] at hsref
    exact (Option.some.inj hsref).symm
  have hs2dom : s2.toReal ∈ Device.sqrtDom := Or.inr hs2lo
  have hs2nat : s2.toReal ∈ MathFn.natDom MathFn.sqrt := by
    have h : MathFn.natDom MathFn.sqrt = Set.Ici (0 : ℝ) := rfl
    rw [h]
    exact le_trans (by norm_num) hs2lo
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |sref s2.toReal| + (2 : ℝ) ^ (-149 : ℤ)
      ≤ (Format.binary32.maxVal : ℝ) := by
    rw [hsref']
    exact sqrt_room (le_trans (by norm_num) hs2lo) hs2hi
  obtain ⟨hsqfin, hsqerr0⟩ := hacc s2 sq hsq hs2fin hs2dom hs2nat hroom
  have hsqerr : |sq.toReal - Real.sqrt s2.toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * Real.sqrt s2.toReal + (2 : ℝ) ^ (-149 : ℤ) := by
    have h := hsqerr0
    rw [hsref'] at h
    simpa only [abs_of_nonneg (Real.sqrt_nonneg s2.toReal)] using h
  obtain ⟨hqlow, hqinv⟩ := q_low hs2lo hsqerr
  have hqposR : (0 : ℝ) < sq.toReal := lt_of_lt_of_le (by norm_num) hqlow
  have hqpos : (0 : ℚ) < sq.toRat := by
    refine (Rat.cast_lt (K := ℝ)).mp ?_
    simpa only [FloatV.toReal, Rat.cast_zero] using hqposR
  have hqne : sq.toRat ≠ 0 := ne_of_gt hqpos
  have hqbound : |(FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) : F32).toRat / sq.toRat| ≤
      2 ^ 126 := by
    rw [litF32_one_toRat]
    refine rat_le_of_cast ?_
    rw [Rat.cast_abs]
    have hc1 : (((1 : ℚ) / sq.toRat : ℚ) : ℝ) = (1 : ℝ) / sq.toReal := by
      push_cast
      rfl
    rw [hc1, abs_of_pos (one_div_pos.mpr hqposR)]
    have h2 : (2 : ℝ) ^ (65 : ℤ) ≤ (((2 : ℚ) ^ 126 : ℚ) : ℝ) := by
      push_cast
      norm_num
    linarith [hqinv, h2]
  obtain ⟨hbrfin, hbrerr⟩ := hdiv_full _ sq r hrd litF32_one_finite hsqfin hqne hqbound
  have hbrerrR : |r.toReal - (1 : ℝ) / sq.toReal|
      ≤ (2 : ℝ) ^ (-22 : ℤ) * |(1 : ℝ) / sq.toReal| + (2 : ℝ) ^ (-148 : ℤ) := by
    rw [litF32_one_toRat] at hbrerr
    have h := Rat.cast_le (K := ℝ).mpr hbrerr
    push_cast at h
    simpa only [FloatV.toReal] using h
  have hVhi : A / d.D + (FloatV.hostF32 d.eps).toReal ≤ (2 : ℝ) ^ (127 : ℤ) := by
    have h1 : A / d.D ≤ (2 : ℝ) ^ (125 : ℤ) := by linarith
    have e : (2 : ℝ) ^ (125 : ℤ) + (2 : ℝ) ^ (125 : ℤ) * (1 + 1 / 2 ^ 23) ≤ (2 : ℝ) ^ (127 : ℤ) := by
      norm_num
    linarith
  have hwide := rstd_err_wide (A := A / d.D) (e := (FloatV.hostF32 d.eps).toReal)
    (s1 := bv.toReal) (s2 := s2.toReal) (q := sq.toReal) (r := r.toReal) he32_nn hAD0 he32_lo
    hVhi hvar' hs2R hqposR hsqerr hbrerrR
  exact ⟨hbrfin, eps_transfer (A := A / d.D) (e := d.eps.toReal)
    (e32 := (FloatV.hostF32 d.eps).toReal) (r := r.toReal) hAD0 he heps32_errR hwide⟩

end Claims.LayerNormGatedFwd
