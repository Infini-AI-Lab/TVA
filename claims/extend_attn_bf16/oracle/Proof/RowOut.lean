import Proof.RowFl
import Proof.CoreFinal

/-!
# The output of a row

`row_output`: the division of an accumulator element by `deno_safe` and its store as `bfloat16`
against the claim's tolerance.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

set_option maxHeartbeats 1600000 in
/-- **One output element.** After all trips (`Tp + Te` of them, `Te ≥ 1`) the state of a row that
has seen exactly its attended positions `K` gives, for component `e < D`, a stored value within
the claim's tolerance of the exact attention. -/
theorem row_output {P : Platform} (hP : Device P) {BD Tp Te D : ℕ} {X γ : ℝ} {K : Finset ℕ}
    {sc : ℕ → ℝ} {vcol : ℕ → ℕ → ℝ} {emax deno : F32} {acc : Fin BD → F32} {e : Fin BD}
    {o : F32} {c : BF16} (hBD : BD ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 2 ^ 12) (hTe : 1 ≤ Te)
    (hT : Tp + Te ≤ 2 ^ 12 + 2) (hK : K.Nonempty) (hcard : K.card ≤ nmax) (hγ0 : 0 ≤ γ)
    (hγ : γ ≤ g2 + Tp * g1p + Te * g1e)
    (hrow : RowFl sc vcol D BD (dlt BD X) (X + dlt BD X) (Tp + Te) γ K emax deno acc)
    (he : (e : ℕ) < D)
    (hdiv : P.div .binary32 (acc e) (if FloatV.eq deno zf32 = true then one32 else deno) o)
    (hcvt : FloatV.NaNEq (FloatV.cvtWith .rn o) c) :
    c.isFinite = true ∧ |c.toReal - wref K sc (vcol e)| ≤
      tolRel BD Tp Te X * wabs K sc (vcol e) + tolAbs (∑ j ∈ K, |vcol e j|) := by
  obtain ⟨m, l, G, hinv, hdF, hdV, -, -, hcols⟩ := hrow
  obtain ⟨haF, hab, hcol⟩ := hcols e he
  have hBD' : (BD : ℝ) ≤ 256 := by exact_mod_cast hBD
  have hd0 : 0 ≤ dlt BD X := by unfold dlt; positivity
  have hd1 : dlt BD X ≤ 1 / 2 := by
    unfold dlt
    have h1 : ((BD : ℝ) + 8) * (1 / 2 ^ 22) * X ≤ (256 + 8) * (1 / 2 ^ 22) * 2 ^ 12 := by
      apply mul_le_mul (mul_le_mul_of_nonneg_right (by linarith) (by norm_num)) hX hX0 (by norm_num)
    norm_num at h1 ⊢
    linarith
  have hTp : (Tp + Te : ℕ) ≤ tmax := by unfold tmax; omega
  have hγ1 : γ ≤ 1 / 2 ^ 3 := by
    have hTs : (Tp : ℝ) + Te ≤ 4098 := by exact_mod_cast (by omega : Tp + Te ≤ 4098)
    refine hγ.trans ?_
    unfold g2 g1p g1e
    nlinarith [Nat.cast_nonneg (α := ℝ) Tp, Nat.cast_nonneg (α := ℝ) Te]
  obtain ⟨hl14, hrat⟩ := ratio_bound (dl := dlt BD X) (Xp := X + dlt BD X) hd0 hd1
    (by norm_num at hX ⊢; linarith [hd0]) (by norm_num at hX ⊢; linarith) hγ0 hγ1 hTp hK hcard hinv hcol
  have hl : (1 : ℝ) / 4 ≤ l := hl14
  -- `deno_safe = deno`
  have heq : FloatV.eq deno zf32 = false := by
    have hdn : deno.isNaN = false := (FloatV.isFinite_iff.mp hdF).1
    have hzn : zf32.isNaN = false := by
      unfold zf32; rw [FloatV.litF32_ofPat_zero]
      exact FloatV.isNaN_ofPat_zero Format.binary32_regular
    cases h : FloatV.eq deno zf32
    · rfl
    · exfalso
      have := (FloatV.eq_iff hdn hzn).mp h
      have hz0 : zf32.toRat = 0 := by
        unfold zf32; rw [FloatV.litF32_ofPat_zero]; exact FloatV.toRat_ofPat_zero
      rw [hz0] at this
      have : deno.toReal = 0 := by unfold FloatV.toReal; rw [this]; simp
      linarith
  simp only [heq, Bool.false_eq_true, ite_false] at hdiv
  -- the division
  have hdq : deno.toRat ≠ 0 := by
    intro h0
    have : deno.toReal = 0 := by unfold FloatV.toReal; rw [h0]; simp
    linarith
  have habq : |(acc e).toReal| ≤ 2 ^ 119 := hab
  have hdiv_b : |(acc e).toRat / deno.toRat| ≤ 2 ^ 126 := by
    have h1 : |((acc e).toReal : ℝ) / deno.toReal| ≤ 2 ^ 126 := by
      rw [abs_div, abs_of_pos (by linarith : 0 < deno.toReal), div_le_iff₀ (by linarith)]
      have : (2 : ℝ) ^ 119 ≤ 2 ^ 126 * (1 / 4) := by norm_num
      have := mul_le_mul_of_nonneg_left (hdV ▸ hl : (1 : ℝ) / 4 ≤ deno.toReal) (by positivity : (0 : ℝ) ≤ 2 ^ 126)
      linarith
    have h2 : (((|(acc e).toRat / deno.toRat| : ℚ)) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast
      unfold FloatV.toReal at h1
      exact h1
    exact_mod_cast h2
  obtain ⟨hof, hoe⟩ := hP.divFull (acc e) deno o hdiv haF hdF hdq hdiv_b
  have hoe' : |o.toReal - (acc e).toReal / l| ≤ 1 / 2 ^ 22 * |(acc e).toReal / l| + 1 / 2 ^ 148 := by
    have := (Rat.cast_le (K := ℝ)).mpr hoe
    push_cast at this
    unfold FloatV.toReal
    rw [← hdV]
    unfold FloatV.toReal at *
    norm_num at this ⊢
    linarith
  have hob : |o.toReal| ≤ 2 ^ 123 := by
    have h1 : |(acc e).toReal / l| ≤ 2 ^ 121 := by
      rw [abs_div, abs_of_pos (by linarith : 0 < l), div_le_iff₀ (by linarith)]
      have : (2 : ℝ) ^ 119 ≤ 2 ^ 121 * (1 / 4) := by norm_num
      have := mul_le_mul_of_nonneg_left hl (by positivity : (0 : ℝ) ≤ 2 ^ 121)
      linarith
    have h0 := abs_sub_abs_le_abs_sub o.toReal ((acc e).toReal / l)
    have h2 : (1 : ℝ) / 2 ^ 22 * |(acc e).toReal / l| ≤ 1 / 2 ^ 22 * 2 ^ 121 :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have h3 : (2 : ℝ) ^ 121 + 1 / 2 ^ 22 * 2 ^ 121 + 1 / 2 ^ 148 ≤ 2 ^ 123 := by norm_num
    linarith
  -- the store
  have hm : |o.toReal| ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
    rw [Format.bfloat16_maxVal_real]; refine hob.trans ?_; norm_num
  obtain ⟨hcf1, hce⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hof hm
  have hcv : FloatV.cvtWith .rn o = (FloatV.cvt o : BF16) := FloatV.cvtWith_rn o
  rw [hcv] at hcvt
  obtain ⟨hcf, -, hcr⟩ := hcvt.exact_of_isFinite hcf1
  have hc' : |c.toReal - o.toReal| ≤ ub * |o.toReal| + 1 / 2 ^ 134 := by
    rw [hcr]
    rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at hce
    unfold ub
    norm_num at hce ⊢
    linarith
  have hq : |o.toReal - (acc e).toReal / l| ≤ 1 / 2 ^ 22 * |(acc e).toReal / l| + 1 / 2 ^ 148 := hoe'
  have := final_bound (q := o.toReal) (c := c.toReal) (a := (acc e).toReal) (l := l) hBD hX0 hX hTe hT hK hcard hinv hcol hγ0 hγ hq hc'
  exact ⟨hcf, this.2⟩

end

end Claims.ExtendAttn
