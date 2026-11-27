import Proof.SigReal
import Proof.SigFloat
import Proof.Ops

/-!
# `layer_norm_gated_fwd_bf16`: `tl.sigmoid` of one gate

`tl.sigmoid(g)` is `1 / (1 + exp(-g))`: the negation, `tl.exp`, the IEEE `+ 1`, `div.full.f32`.
For a finite `bfloat16` gate `g` (`|g| ≤ 2^56`), widened exactly to `float32`, the result `s` is
finite and within `2^-15·σ(g) + 2^-128` of `σ(g) = 1 / (1 + e^-g)` (`gate_err`):

* when `-g` reaches `tl.exp`'s overflow threshold (`Device.ExpF32Overflow`), `exp` gives `+∞`,
  the sum `+∞` and the quotient `+0` (`Device.DivF32InfDivisor`), while `σ(g) ≤ 2^-128`;
* otherwise `-g ≤ 88.5` (`bf16_gap`), `tl.exp` has room (`Device.ExpF32` with `X = |g|`), its
  error is at most `2^-16·(1 + e^-g) + 2^-140` (`exp_rel_le`), the sum is within
  `(2^-16 + 2^-23)·(1 + e^-g)` of `1 + e^-g`, and the division (`Device.DivFull`) adds `2^-22`.
-/

namespace Claims.LayerNormGatedFwd

open TVA

/-- `tl.exp`'s relative error at `X = |v|` on the input `-v`, against `1 + e^-v`, when `-v ≤ 88.5`:
`(2^-22 + 2^-23·|v|)·e^-v ≤ 2^-16·(1 + e^-v)`. -/
theorem exp_rel_le {v : ℝ} (hv : -v ≤ 177 / 2) :
    ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |v|) * Real.exp (-v) ≤
      (2 : ℝ) ^ (-16 : ℤ) * (1 + Real.exp (-v)) := by
  have hE := Real.exp_pos (-v)
  rcases le_or_gt 0 v with h0 | h0
  · rw [abs_of_nonneg h0]
    have hE1 : Real.exp (-v) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hve := mul_exp_neg_le_one v
    have e1 : (2 : ℝ) ^ (-22 : ℤ) * Real.exp (-v) ≤ (2 : ℝ) ^ (-22 : ℤ) :=
      mul_le_of_le_one_right (by positivity) hE1
    have e2 : (2 : ℝ) ^ (-23 : ℤ) * v * Real.exp (-v) ≤ (2 : ℝ) ^ (-23 : ℤ) := by
      have := mul_le_mul_of_nonneg_left hve (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-23 : ℤ))
      linarith [this]
    have e3 : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) ≤ (2 : ℝ) ^ (-16 : ℤ) := by norm_num
    have e4 : (2 : ℝ) ^ (-16 : ℤ) ≤ (2 : ℝ) ^ (-16 : ℤ) * (1 + Real.exp (-v)) := by
      have := mul_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ (-16 : ℤ)) hE
      linarith [mul_add ((2 : ℝ) ^ (-16 : ℤ)) 1 (Real.exp (-v))]
    have e5 : ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * v) * Real.exp (-v) =
        (2 : ℝ) ^ (-22 : ℤ) * Real.exp (-v) + (2 : ℝ) ^ (-23 : ℤ) * v * Real.exp (-v) := by ring
    linarith
  · rw [abs_of_neg h0]
    have hr : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * -v ≤ (2 : ℝ) ^ (-16 : ℤ) := by
      have : (2 : ℝ) ^ (-23 : ℤ) * -v ≤ (2 : ℝ) ^ (-23 : ℤ) * (177 / 2) :=
        mul_le_mul_of_nonneg_left hv (by positivity)
      have e : (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * (177 / 2) ≤ (2 : ℝ) ^ (-16 : ℤ) := by
        norm_num
      calc (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * -v
          ≤ (2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * (177 / 2) := by
            linarith
        _ ≤ _ := e
    have h1 := mul_le_mul_of_nonneg_right hr hE.le
    have h2 : (2 : ℝ) ^ (-16 : ℤ) * Real.exp (-v) ≤ (2 : ℝ) ^ (-16 : ℤ) * (1 + Real.exp (-v)) := by
      have := (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ (-16 : ℤ))
      nlinarith
    exact h1.trans h2

/-- **`tl.sigmoid` of a gate**: finite, within `2^-15·σ(g) + 2^-128` of `σ(g)`. -/
theorem gate_err {P : Platform} (hexp : Device.ExpF32 P) (hovf : Device.ExpF32Overflow P)
    (hdiv : Device.DivFull P) (hinf : Device.DivF32InfDivisor P)
    {gb : BF16} {gf nx ex den s : F32}
    (hgb : gb.isFinite = true) (hg56 : |gb.toReal| ≤ (2 : ℝ) ^ (56 : ℤ))
    (hgf : gf.isFinite = true) (hgfR : gf.toReal = gb.toReal) (hgfQ : gf.toRat = gb.toRat)
    (hnx : FloatV.NaNEq (FloatV.neg gf) nx) (hex : P.math .exp .binary32 nx ex)
    (hden : FloatV.NaNEq (FloatV.add (FloatV.ofRat 1) ex) den)
    (hs : P.div .binary32 (FloatV.ofRat 1) den s) :
    s.isFinite = true ∧
      |s.toReal - sig gb.toReal| ≤ (2 : ℝ) ^ (-15 : ℤ) * sig gb.toReal + (2 : ℝ) ^ (-128 : ℤ) := by
  set v := gb.toReal with hvdef
  obtain ⟨hone_fin, hone_q, hone_sign⟩ := one32_facts
  have hone_R : (FloatV.ofRat 1 : F32).toReal = 1 := by
    rw [FloatV.toReal_eq, hone_q]; norm_num
  -- the negation is exact
  have hnegfin : (FloatV.neg gf).isFinite = true := by rw [FloatV.isFinite_neg]; exact hgf
  obtain ⟨hnxfin, hnxQ, hnxR⟩ := hnx.exact_of_isFinite hnegfin
  rw [FloatV.toReal_neg, hgfR] at hnxR
  rw [FloatV.toRat_neg, hgfQ] at hnxQ
  have hσ0 := sig_pos v
  have hσ1 := sig_le_one v
  by_cases hthr : (1453635 / 16384 : ℚ) ≤ nx.toRat
  · -- overflow: `exp` is `+∞`, the sum `+∞`, the quotient `+0`
    have hex' : ex = FloatV.posInf := hovf nx (Or.inl ⟨hnxfin, hthr⟩) ex hex
    have hden' : den = FloatV.posInf := by
      rw [hex', add_posInf rfl hone_fin] at hden
      exact hden.eq (posInf_isNaN rfl)
    have hdinf : den.isInf = true := by rw [hden']; exact FloatV.posInf_isInf rfl
    have hs' := hinf _ _ hone_fin hdinf s hs
    obtain ⟨hzf, hzq⟩ := zero32_facts ((FloatV.ofRat 1 : F32).sign != den.sign)
    rw [hs']
    refine ⟨hzf, ?_⟩
    have hz : (FloatV.ofPat (Format.binary32.zeroPat ((FloatV.ofRat 1 : F32).sign != den.sign)) :
        F32).toReal = 0 := by rw [FloatV.toReal_eq, hzq]; norm_num
    rw [hz, zero_sub, abs_neg, abs_of_pos hσ0]
    have hthrR : (1453635 / 16384 : ℝ) ≤ -v := by
      have h := Rat.cast_le (K := ℝ).mpr hthr
      rw [hnxQ] at h
      push_cast at h
      rw [hvdef, FloatV.toReal_eq]; linarith
    have := sig_le_of_overflow hthrR
    have := mul_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ (-15 : ℤ)) hσ0
    linarith
  · -- no overflow: `-g ≤ 88.5`
    have hthr' : nx.toRat < 1453635 / 16384 := lt_of_not_ge hthr
    have hv885 : -v ≤ 177 / 2 := by
      by_contra hc
      have hc' : (177 / 2 : ℝ) < -v := lt_of_not_ge hc
      have hq : (177 / 2 : ℚ) < -gb.toRat := by
        have : ((177 / 2 : ℚ) : ℝ) < ((-gb.toRat : ℚ) : ℝ) := by
          push_cast; rw [hvdef, FloatV.toReal_eq] at hc'; linarith
        exact_mod_cast this
      have hab : |gb.toRat| = -gb.toRat := abs_of_neg (by linarith)
      exact bf16_gap gb ⟨by rw [hab]; exact hq, by rw [hab, ← hnxQ]; exact hthr'⟩
    -- `tl.exp(-g)` with `X = |g|`
    obtain ⟨ref, href, hacc⟩ := hexp |v| (abs_nonneg v)
    have hrefE : ref = Real.exp := by
      have h : MathFn.ref MathFn.exp = some Real.exp := rfl
      rw [h] at href; exact (Option.some.inj href).symm
    subst hrefE
    set E := Real.exp (-v) with hEdef
    have hEpos : 0 < E := Real.exp_pos _
    have hrel := exp_rel_le hv885
    rw [← hEdef] at hrel
    have hEle : E ≤ 19 / 10 * (2 : ℝ) ^ (127 : ℤ) :=
      (Real.exp_le_exp.mpr hv885).trans exp_885_le
    have hdom : nx.toReal ∈ Set.Icc (-|v|) |v| := by
      rw [hnxR]; exact ⟨by linarith [le_abs_self v], by linarith [neg_abs_le v]⟩
    have hroom : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |v|)) * |Real.exp nx.toReal| +
        (2 : ℝ) ^ (-140 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [hnxR, ← hEdef, abs_of_pos hEpos, Format.binary32_maxVal_real]
      have e : 19 / 10 * (2 : ℝ) ^ (127 : ℤ) * (1 + (2 : ℝ) ^ (-16 : ℤ)) + (2 : ℝ) ^ (-16 : ℤ) +
          (2 : ℝ) ^ (-140 : ℤ) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      have h1 := mul_le_mul_of_nonneg_right hEle (by positivity : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-16 : ℤ))
      have h2 : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |v|)) * E =
          E + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |v|) * E := by ring
      have h3 : (2 : ℝ) ^ (-16 : ℤ) * (1 + E) = (2 : ℝ) ^ (-16 : ℤ) + (2 : ℝ) ^ (-16 : ℤ) * E := by ring
      have h4 : E * (1 + (2 : ℝ) ^ (-16 : ℤ)) = E + (2 : ℝ) ^ (-16 : ℤ) * E := by ring
      clear hacc hexp hovf hdiv hinf
      simp only [zpow_neg, zpow_ofNat, zpow_negSucc] at *
      linarith
    obtain ⟨hexfin, hexerr⟩ := hacc nx ex hex hnxfin hdom (Set.mem_univ _) hroom
    rw [hnxR, ← hEdef, abs_of_pos hEpos] at hexerr
    have hexE : |ex.toReal - E| ≤ ((2 : ℝ) ^ (-16 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) * (1 + E) := by
      have : (2 : ℝ) ^ (-140 : ℤ) ≤ (2 : ℝ) ^ (-140 : ℤ) * (1 + E) := by
        have := mul_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ (-140 : ℤ)) hEpos
        linarith [mul_add ((2 : ℝ) ^ (-140 : ℤ)) 1 E]
      have h3 : ((2 : ℝ) ^ (-16 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) * (1 + E) =
          (2 : ℝ) ^ (-16 : ℤ) * (1 + E) + (2 : ℝ) ^ (-140 : ℤ) * (1 + E) := by ring
      linarith
    -- the sum `1 + exp(-g)`, in numerals
    clear hacc hexp hovf hroom hdom hexerr
    have hexE' : |ex.toReal - E| ≤ (1 / 65536 + 1 / 2 ^ 140) * (1 + E) := by
      have h : ((2 : ℝ) ^ (-16 : ℤ) + (2 : ℝ) ^ (-140 : ℤ)) = 1 / 65536 + 1 / 2 ^ 140 := by norm_num
      rw [← h]; exact hexE
    have hEle' : E ≤ 19 / 10 * 2 ^ 127 := by
      have h : (2 : ℝ) ^ (127 : ℤ) = 2 ^ 127 := by norm_num
      rw [← h]; exact hEle
    have hexlo : -1 / 2 ≤ ex.toReal := by
      have := (abs_le.mp hexE').1; nlinarith
    have hexhi : ex.toReal ≤ (1 + 1 / 32768) * E + 1 / 2 := by
      have := (abs_le.mp hexE').2; nlinarith
    have hsum_room : |(FloatV.ofRat 1 : F32).toReal + ex.toReal| ≤
        ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [hone_R, abs_of_nonneg (by linarith), Format.binary32_maxVal_real]
      nlinarith
    obtain ⟨hdf, hderr⟩ := FloatV.add_err_real Format.binary32_regular hone_fin hexfin hsum_room
    obtain ⟨hdenfin, -, hdenR⟩ := hden.exact_of_isFinite hdf
    rw [Format.binary32_u_real, Format.binary32_eta_real, hone_R, ← hdenR,
      abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 + ex.toReal)] at hderr
    have hderr' : |den.toReal - (1 + ex.toReal)| ≤ (1 + ex.toReal) * (1 / 2 ^ 24) + 1 / 2 ^ 150 := by
      have h1 : (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
      have h2 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
      rw [← h1, ← h2]; exact hderr
    have hδ : |den.toReal - (1 + E)| ≤ (1 / 65536 + 1 / 2 ^ 23) * (1 + E) := by
      have htri : |den.toReal - (1 + E)| ≤ |den.toReal - (1 + ex.toReal)| + |ex.toReal - E| := by
        have h := abs_add_le (den.toReal - (1 + ex.toReal)) (ex.toReal - E)
        rwa [show den.toReal - (1 + ex.toReal) + (ex.toReal - E) = den.toReal - (1 + E) by
          ring] at h
      have := (abs_le.mp hexE').2
      nlinarith
    have hdenlo : (1 + E) * (1 - (1 / 65536 + 1 / 2 ^ 23)) ≤ den.toReal := by
      have := (abs_le.mp hδ).1; nlinarith
    have hdenpos : 0 < den.toReal := by nlinarith
    have hdenQ : den.toRat ≠ 0 := by
      intro h; rw [FloatV.toReal_eq, h] at hdenpos; norm_num at hdenpos
    -- `1 / den` by `div.full.f32`
    have hinvle : 1 / den.toReal ≤ 2 := by
      rw [div_le_iff₀ hdenpos]; nlinarith
    have hqb : |(FloatV.ofRat 1 : F32).toRat / den.toRat| ≤ 2 ^ 126 := by
      rw [hone_q]
      have h : ((|1 / den.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
        push_cast
        rw [← FloatV.toReal_eq, abs_of_pos (by positivity)]
        exact hinvle.trans (by norm_num)
      exact_mod_cast h
    obtain ⟨hsfin, hserr⟩ := hdiv _ _ s hs hone_fin hdenfin hdenQ hqb
    refine ⟨hsfin, ?_⟩
    have hserrR : |s.toReal - 1 / den.toReal| ≤ (1 / 2 ^ 22) * (1 / den.toReal) + 1 / 2 ^ 148 := by
      rw [hone_q] at hserr
      have h := Rat.cast_le (K := ℝ).mpr hserr
      push_cast at h
      rw [← FloatV.toReal_eq, ← FloatV.toReal_eq, abs_of_pos (one_div_pos.mpr hdenpos)] at h
      have h1 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
      have h2 : (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 := by norm_num
      rw [h1, h2] at h; exact h
    have hσ : sig v = 1 / (1 + E) := rfl
    have hDnpos : 0 < 1 + E := by linarith
    -- `|1/den − 1/(1+E)| ≤ δ/den` and `1/den ≤ σ/(1 − δ)`
    have hinv : |1 / den.toReal - 1 / (1 + E)| ≤ (1 / 65536 + 1 / 2 ^ 23) * (1 / den.toReal) := by
      have heq : 1 / den.toReal - 1 / (1 + E) = ((1 + E) - den.toReal) / (den.toReal * (1 + E)) := by
        field_simp
      rw [heq, abs_div, abs_of_pos (mul_pos hdenpos hDnpos), abs_sub_comm,
        div_le_iff₀ (mul_pos hdenpos hDnpos)]
      have e : (1 / 65536 + 1 / 2 ^ 23 : ℝ) * (1 / den.toReal) * (den.toReal * (1 + E)) =
          (1 / 65536 + 1 / 2 ^ 23) * (1 + E) := by field_simp
      rw [e]; exact hδ
    have hinvσ : 1 / den.toReal ≤ sig v / (1 - (1 / 65536 + 1 / 2 ^ 23)) := by
      have hpos : (0 : ℝ) < (1 + E) * (1 - (1 / 65536 + 1 / 2 ^ 23)) :=
        mul_pos hDnpos (by norm_num)
      rw [hσ, div_div, div_le_div_iff₀ hdenpos hpos]
      linarith
    have htri : |s.toReal - sig v| ≤ |s.toReal - 1 / den.toReal| + |1 / den.toReal - 1 / (1 + E)| := by
      have h := abs_add_le (s.toReal - 1 / den.toReal) (1 / den.toReal - 1 / (1 + E))
      rwa [show s.toReal - 1 / den.toReal + (1 / den.toReal - 1 / (1 + E)) = s.toReal - sig v by
        rw [hσ]; ring] at h
    have hgoal : |s.toReal - sig v| ≤ 1 / 32768 * sig v + 1 / 2 ^ 128 := by
      have hc : (1 / 2 ^ 22 + (1 / 65536 + 1 / 2 ^ 23) : ℝ) *
          (sig v / (1 - (1 / 65536 + 1 / 2 ^ 23))) ≤ 1 / 32768 * sig v := by
        rw [mul_div_assoc', div_le_iff₀ (by norm_num)]
        have := hσ0
        have e : (1 / 2 ^ 22 + (1 / 65536 + 1 / 2 ^ 23) : ℝ) ≤
            1 / 32768 * (1 - (1 / 65536 + 1 / 2 ^ 23)) := by norm_num
        have := mul_le_mul_of_nonneg_right e hσ0.le
        linarith
      have hp := mul_le_mul_of_nonneg_left hinvσ
        (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22 + (1 / 65536 + 1 / 2 ^ 23))
      have e : (1 : ℝ) / 2 ^ 148 ≤ 1 / 2 ^ 128 := by norm_num
      have e2 : (1 / 2 ^ 22) * (1 / den.toReal) + (1 / 65536 + 1 / 2 ^ 23) * (1 / den.toReal) =
          (1 / 2 ^ 22 + (1 / 65536 + 1 / 2 ^ 23)) * (1 / den.toReal) := by ring
      linarith
    have h1 : (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num
    have h2 : (2 : ℝ) ^ (-128 : ℤ) = 1 / 2 ^ 128 := by norm_num
    rw [h1, h2]; exact hgoal

end Claims.LayerNormGatedFwd
