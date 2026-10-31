import AttnFwd.Step

/-!
# `attn_fwd_f16`: the end of a row (lines 241–245)

* `wmean_err`: weights changed by factors within `e^{±θ}` move a weighted mean by at most `4θ`
  times the weighted mean of `|v|`.
* the tolerances of `Model` as real numbers, and their size at the covered extremes;
* `final_o`: `acc / l_i` (line 242) rounded to `float16` (line 245) against the exact attention;
* `final_M`: `m_i + log2(l_i)` (line 241) against the exact log-sum-exp.
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

/-! ## Real facts -/

theorem exp_two_mul_le {θ : ℝ} (h0 : 0 ≤ θ) (h1 : θ ≤ 1 / 2) : Real.exp (2 * θ) ≤ 1 + 4 * θ := by
  have := exp_le_two (y := 2 * θ) (by linarith) (by linarith); linarith

/-- **Changed weights.** If each weight `w j > 0` is scaled by `ρ j ∈ [e^-θ, e^θ]`, the weighted
mean of `v` moves by at most `4θ` times the weighted mean of `|v|`, and the weighted mean of `|v|`
grows by at most the factor `1 + 4θ`. -/
theorem wmean_err {ks : Finset ℕ} {w ρ v : ℕ → ℝ} {θ : ℝ} (hθ0 : 0 ≤ θ) (hθ : θ ≤ 1 / 2)
    (hne : ks.Nonempty) (hw : ∀ j ∈ ks, 0 < w j)
    (hρ : ∀ j ∈ ks, Real.exp (-θ) ≤ ρ j ∧ ρ j ≤ Real.exp θ) :
    |(∑ j ∈ ks, w j * ρ j * v j) / (∑ j ∈ ks, w j * ρ j) - (∑ j ∈ ks, w j * v j) / (∑ j ∈ ks, w j)| ≤
        4 * θ * ((∑ j ∈ ks, w j * |v j|) / (∑ j ∈ ks, w j)) ∧
      (∑ j ∈ ks, w j * ρ j * |v j|) / (∑ j ∈ ks, w j * ρ j) ≤
        (1 + 4 * θ) * ((∑ j ∈ ks, w j * |v j|) / (∑ j ∈ ks, w j)) := by
  set A := ∑ j ∈ ks, w j with hA
  set B := ∑ j ∈ ks, w j * ρ j with hB
  set C := ∑ j ∈ ks, w j * |v j| with hC
  have hA0 : 0 < A := Finset.sum_pos (fun j hj => hw j hj) hne
  have heθ := Real.exp_pos θ
  have heθ' := Real.exp_pos (-θ)
  have hρ0 : ∀ j ∈ ks, 0 < ρ j := fun j hj => lt_of_lt_of_le heθ' (hρ j hj).1
  have hBA : Real.exp (-θ) * A ≤ B := by
    rw [hA, hB, Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      have := (hρ j hj).1; have := hw j hj; nlinarith
  have hB0 : 0 < B := lt_of_lt_of_le (mul_pos heθ' hA0) hBA
  have hC0 : 0 ≤ C := Finset.sum_nonneg fun j hj => mul_nonneg (hw j hj).le (abs_nonneg _)
  have hmul : Real.exp θ * Real.exp (-θ) = 1 := by rw [← Real.exp_add]; simp
  have he2 : Real.exp θ * Real.exp θ = Real.exp (2 * θ) := by rw [← Real.exp_add]; ring_nf
  have he2b := exp_two_mul_le hθ0 hθ
  -- `e^θ − e^-θ ≤ e^θ·(e^{2θ} − 1)·e^-θ`, so `(e^θ − e^-θ)·e^θ = e^{2θ} − 1 ≤ 4θ`
  have hd : (Real.exp θ - Real.exp (-θ)) * Real.exp θ ≤ 4 * θ := by nlinarith
  constructor
  · rw [div_sub_div _ _ hB0.ne' hA0.ne']
    have hnum : (∑ j ∈ ks, w j * ρ j * v j) * A - B * (∑ j ∈ ks, w j * v j) =
        ∑ j ∈ ks, w j * v j * (ρ j * A - B) := by
      rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hnum, abs_div, abs_of_pos (mul_pos hB0 hA0)]
    have hterm : ∀ j ∈ ks, |w j * v j * (ρ j * A - B)| ≤
        w j * |v j| * (A * (Real.exp θ - Real.exp (-θ))) := by
      intro j hj
      rw [abs_mul, abs_mul, abs_of_pos (hw j hj)]
      refine mul_le_mul_of_nonneg_left ?_ (mul_nonneg (hw j hj).le (abs_nonneg _))
      have he : ρ j * A - B = ∑ k ∈ ks, w k * (ρ j - ρ k) := by
        rw [hA, hB, Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun k _ => by ring
      rw [he]
      refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
      rw [hA, Finset.sum_mul]
      refine Finset.sum_le_sum fun k hk => ?_
      rw [abs_mul, abs_of_pos (hw k hk)]
      refine mul_le_mul_of_nonneg_left ?_ (hw k hk).le
      have := hρ j hj; have := hρ k hk
      rw [abs_le]; constructor <;> linarith
    have hsum := (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum hterm)
    rw [← Finset.sum_mul] at hsum
    rw [div_le_iff₀ (mul_pos hB0 hA0)]
    have hk0 : 0 ≤ A * (Real.exp θ - Real.exp (-θ)) := by
      have : Real.exp (-θ) ≤ Real.exp θ := Real.exp_le_exp.mpr (by linarith)
      nlinarith
    calc |∑ j ∈ ks, w j * v j * (ρ j * A - B)| ≤ C * (A * (Real.exp θ - Real.exp (-θ))) := hsum
      _ ≤ 4 * θ * (C / A) * (B * A) := by
        rw [show 4 * θ * (C / A) * (B * A) = 4 * θ * C * B by field_simp]
        -- `A·(e^θ − e^-θ) ≤ 4θ·B` since `A ≤ e^θ·B`
        have hAB : A ≤ Real.exp θ * B := by
          have := mul_le_mul_of_nonneg_left hBA heθ.le
          rw [← mul_assoc, hmul, one_mul] at this; exact this
        have : A * (Real.exp θ - Real.exp (-θ)) ≤ 4 * θ * B := by
          have h1 : 0 ≤ Real.exp θ - Real.exp (-θ) := by
            have : Real.exp (-θ) ≤ Real.exp θ := Real.exp_le_exp.mpr (by linarith)
            linarith
          calc A * (Real.exp θ - Real.exp (-θ)) ≤ Real.exp θ * B * (Real.exp θ - Real.exp (-θ)) :=
                mul_le_mul_of_nonneg_right hAB h1
            _ = (Real.exp θ - Real.exp (-θ)) * Real.exp θ * B := by ring
            _ ≤ 4 * θ * B := mul_le_mul_of_nonneg_right hd hB0.le
        nlinarith
  · rw [div_le_iff₀ hB0]
    have h1 : ∑ j ∈ ks, w j * ρ j * |v j| ≤ Real.exp θ * C := by
      rw [hC, Finset.mul_sum]
      exact Finset.sum_le_sum fun j hj => by
        have h1 := (hρ j hj).2
        have := mul_le_mul_of_nonneg_left h1 (mul_nonneg (hw j hj).le (abs_nonneg (v j)))
        linarith
    have h2 : Real.exp θ * C ≤ (1 + 4 * θ) * (C / A) * B := by
      rw [show (1 + 4 * θ) * (C / A) * B = (1 + 4 * θ) * C * (B / A) by ring]
      have hBA' : Real.exp (-θ) ≤ B / A := by rw [le_div_iff₀ hA0]; exact hBA
      have : Real.exp θ ≤ (1 + 4 * θ) * Real.exp (-θ) := by
        have := mul_le_mul_of_nonneg_right he2b heθ'.le
        rw [← he2, mul_assoc, hmul, mul_one] at this; exact this
      have := mul_le_mul_of_nonneg_left hBA' (by linarith : (0 : ℝ) ≤ 1 + 4 * θ)
      nlinarith
    linarith

/-- `|log₂(1 + β)| ≤ 2|β|` for `|β| ≤ 1/4`. -/
theorem abs_logb_one_add {β : ℝ} (h : |β| ≤ 1 / 4) : |Real.logb 2 (1 + β)| ≤ 2 * |β| := by
  have hb := abs_le.mp h
  have hpos : 0 < 1 + β := by linarith
  have h1 := Real.log_le_sub_one_of_pos hpos
  have h2 := Real.one_sub_inv_le_log_of_pos hpos
  have hl2 := Real.log_two_gt_d9
  have hl0 : 0 < Real.log 2 := by linarith
  unfold Real.logb
  rw [abs_div, abs_of_pos hl0, div_le_iff₀ hl0]
  have h3 : |Real.log (1 + β)| ≤ 4 / 3 * |β| := by
    rw [abs_le]
    have hinv : 1 - (1 + β)⁻¹ = β / (1 + β) := by field_simp; ring
    rw [hinv] at h2
    constructor
    · rcases le_or_gt 0 β with hβ | hβ
      · have : 0 ≤ β / (1 + β) := div_nonneg hβ hpos.le
        rw [abs_of_nonneg hβ]; linarith
      · rw [abs_of_neg hβ]
        have : -(4 / 3 * -β) ≤ β / (1 + β) := by
          rw [le_div_iff₀ hpos]; nlinarith
        linarith
    · rcases le_or_gt 0 β with hβ | hβ
      · rw [abs_of_nonneg hβ]; linarith
      · rw [abs_of_neg hβ]; linarith
  have := abs_nonneg β
  nlinarith

/-- `|log₂ r| ≤ 2θ` for `r ∈ [e^-θ, e^θ]`. -/
theorem abs_logb_of_exp {r θ : ℝ} (h1 : Real.exp (-θ) ≤ r) (h2 : r ≤ Real.exp θ) :
    |Real.logb 2 r| ≤ 2 * θ := by
  have hr : 0 < r := lt_of_lt_of_le (Real.exp_pos _) h1
  have hl1 : -θ ≤ Real.log r := by rw [← Real.log_exp (-θ)]; exact Real.log_le_log (Real.exp_pos _) h1
  have hl2 : Real.log r ≤ θ := by rw [← Real.log_exp θ]; exact Real.log_le_log hr h2
  have hl := Real.log_two_gt_d9
  have hl0 : 0 < Real.log 2 := by linarith
  unfold Real.logb
  rw [abs_div, abs_of_pos hl0, div_le_iff₀ hl0]
  have hθ : 0 ≤ θ := by linarith
  have : |Real.log r| ≤ θ := abs_le.mpr ⟨by linarith, hl2⟩
  nlinarith

theorem two_rpow_le_one_add {E : ℝ} (h0 : 0 ≤ E) (h1 : E ≤ 1 / 2) : (2 : ℝ) ^ E ≤ 1 + E := by
  have := two_rpow_sub_one (show |E| ≤ 1 / 2 by rwa [abs_of_nonneg h0])
  rw [abs_of_nonneg h0] at this
  have := le_abs_self ((2 : ℝ) ^ E - 1); linarith

theorem one_sub_le_two_rpow_neg {E : ℝ} (h0 : 0 ≤ E) (h1 : E ≤ 1 / 2) : 1 - E ≤ (2 : ℝ) ^ (-E) := by
  have := two_rpow_sub_one (show |-E| ≤ 1 / 2 by rw [abs_neg, abs_of_nonneg h0]; exact h1)
  rw [abs_neg, abs_of_nonneg h0] at this
  have := neg_abs_le ((2 : ℝ) ^ (-E) - 1); linarith

/-! ## The tolerances as real numbers -/

theorem relDot16_cast (D : ℕ) : ((Rules.Num.relDot16 D : ℚ) : ℝ) = (D + 2) / 2 ^ 22 := by
  unfold Rules.Num.relDot16; push_cast; rw [show (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 by norm_num]; ring

theorem absDot16_cast (D : ℕ) : ((Rules.Num.absDot16 D : ℚ) : ℝ) = (D + 2) / 2 ^ 126 := by
  unfold Rules.Num.absDot16; push_cast; rw [show (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 by norm_num]; ring

theorem expErr_cast (D : ℕ) (X : ℚ) :
    ((expErr D X : ℚ) : ℝ) = (((Rules.Num.relDot16 D : ℚ) : ℝ) + 1 / 2 ^ 20) * (3 / 2 * (X : ℝ)) + 1 / 2 ^ 50 := by
  unfold expErr ybound; push_cast; ring

theorem psiP_cast (D : ℕ) (X : ℚ) : ((psiP D X : ℚ) : ℝ) = psiPr (expErr D X : ℝ) := by
  unfold psiP psiPr; push_cast; ring

theorem psi16_cast (D : ℕ) (X : ℚ) : ((psi16 D X : ℚ) : ℝ) = psi16r (expErr D X : ℝ) := by
  unfold psi16 psi16r; push_cast; rw [psiP_cast]

theorem psiL_cast (D : ℕ) (X : ℚ) : ((psiL D X : ℚ) : ℝ) = psiLr (expErr D X : ℝ) := by
  unfold psiL psiLr u32; push_cast; rw [psiP_cast]

theorem drift_cast (D : ℕ) (X : ℚ) :
    ((drift D X : ℚ) : ℝ) = 1 + 4 * (1 / 2 ^ 24) * (3 / 2 * (X : ℝ) + (expErr D X : ℝ)) := by
  unfold drift u32 ybound; push_cast; ring

theorem stepAcc_cast : ((stepAcc : ℚ) : ℝ) = stepAr := by
  unfold stepAcc stepAr u32; push_cast; rw [relDot16_64]

theorem stepL_cast : ((stepL : ℚ) : ℝ) = stepLr := by
  unfold stepL stepLr u32; push_cast; ring

theorem facc_cast (D N : ℕ) (X : ℚ) :
    ((facc D N X : ℚ) : ℝ) = (1 + psi16r (expErr D X : ℝ)) * ((drift D X : ℚ) : ℝ) * stepAr ^ nblocks N := by
  unfold facc; push_cast; rw [psi16_cast, stepAcc_cast]

theorem fl_cast (D N : ℕ) (X : ℚ) :
    ((fl D N X : ℚ) : ℝ) = (1 + psiLr (expErr D X : ℝ)) * ((drift D X : ℚ) : ℝ) * stepLr ^ nblocks N := by
  unfold fl; push_cast; rw [psiL_cast, stepL_cast]

/-- The exponent's error bound at the covered extremes: `E ≤ 1/8`. -/
theorem expErr_le {D : ℕ} {X : ℚ} (hD : D ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 2 ^ 10) :
    0 ≤ ((expErr D X : ℚ) : ℝ) ∧ ((expErr D X : ℚ) : ℝ) ≤ 1 / 8 := by
  rw [expErr_cast, relDot16_cast]
  have hD' : (D : ℝ) ≤ 256 := by exact_mod_cast hD
  have hX' : (X : ℝ) ≤ 2 ^ 10 := by exact_mod_cast hX
  have hX0' : (0 : ℝ) ≤ X := by exact_mod_cast hX0
  constructor
  · positivity
  · have h1 : ((D : ℝ) + 2) / 2 ^ 22 + 1 / 2 ^ 20 ≤ 262 / 2 ^ 22 := by
      have : ((D : ℝ) + 2) / 2 ^ 22 ≤ 258 / 2 ^ 22 := by apply div_le_div_of_nonneg_right <;> linarith
      norm_num at this ⊢; linarith
    have h2 : 3 / 2 * (X : ℝ) ≤ 1536 := by linarith
    have h0 : 0 ≤ ((D : ℝ) + 2) / 2 ^ 22 + 1 / 2 ^ 20 := by positivity
    have := mul_le_mul h1 h2 (by positivity) (by norm_num)
    norm_num at this ⊢; linarith

theorem stepAr_le' : stepAr ≤ 1 + 17 / 2 ^ 20 := by unfold stepAr; norm_num

theorem stepLr_le' : stepLr ≤ 1 + 1 / 2 ^ 21 := by unfold stepLr; norm_num

theorem pow_le_of_step {s : ℝ} {n : ℕ} {c : ℝ} (h1 : 1 ≤ s) (hs : s ≤ 1 + c) (hn : n ≤ 1024)
    (hc : 1024 * c ≤ 1) : s ^ n ≤ 1 + 2 * (1024 * c) := by
  have hc0 : 0 ≤ c := by linarith
  have h := one_add_pow_le_exp (show 0 ≤ s - 1 by linarith) n
  rw [show 1 + (s - 1) = s by ring] at h
  refine h.trans ((Real.exp_le_exp.mpr ?_).trans (exp_le_two (by positivity) hc))
  have : (n : ℝ) ≤ 1024 := by exact_mod_cast hn
  nlinarith

/-- **The tolerances' pieces at the covered extremes** (`HEAD_DIM ≤ 256`, `X ≤ 2^10`,
`N_CTX ≤ 2^16`). -/
structure TolFacts (D N : ℕ) (X : ℚ) : Prop where
  E0 : 0 ≤ ((expErr D X : ℚ) : ℝ)
  E1 : ((expErr D X : ℚ) : ℝ) ≤ 1 / 8
  drift1 : 1 ≤ ((drift D X : ℚ) : ℝ)
  drift2 : ((drift D X : ℚ) : ℝ) ≤ 1 + 1 / 2 ^ 11
  facc1 : 1 ≤ ((facc D N X : ℚ) : ℝ)
  facc2 : ((facc D N X : ℚ) : ℝ) ≤ 6 / 5
  fl1 : 1 ≤ ((fl D N X : ℚ) : ℝ)
  fl2 : ((fl D N X : ℚ) : ℝ) ≤ 6 / 5
  wlow_eq : ((wlow D X : ℚ) : ℝ) = 1 - ((expErr D X : ℚ) : ℝ)
  beta_eq : ((betaL D N X : ℚ) : ℝ) =
    (((fl D N X : ℚ) : ℝ) - 1) + ((fl D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ)
  beta0 : 0 ≤ ((betaL D N X : ℚ) : ℝ)
  beta1 : ((betaL D N X : ℚ) : ℝ) ≤ 1 / 4
  den_eq : ((denQ D N X : ℚ) : ℝ) = 1 - ((betaL D N X : ℚ) : ℝ)
  relQ_eq : ((relQ D N X : ℚ) : ℝ) = (((facc D N X : ℚ) : ℝ) - 1 + ((betaL D N X : ℚ) : ℝ)) / ((denQ D N X : ℚ) : ℝ)
  relQ0 : 0 ≤ ((relQ D N X : ℚ) : ℝ)
  relQ1 : ((relQ D N X : ℚ) : ℝ) ≤ 3 / 5
  absQS_eq : ((absQS D N X : ℚ) : ℝ) =
    ((facc D N X : ℚ) : ℝ) * (1 / 2 ^ 25 + 1 / 2 ^ 125) / (((wlow D X : ℚ) : ℝ) * ((denQ D N X : ℚ) : ℝ))
  absQS0 : 0 ≤ ((absQS D N X : ℚ) : ℝ)
  absQS1 : ((absQS D N X : ℚ) : ℝ) ≤ 1 / 2 ^ 23
  absQ0_eq : ((absQ0 D N X : ℚ) : ℝ) =
    ((facc D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 85)) / (((wlow D X : ℚ) : ℝ) * ((denQ D N X : ℚ) : ℝ))
  absQ00 : 0 ≤ ((absQ0 D N X : ℚ) : ℝ)
  absQ01 : ((absQ0 D N X : ℚ) : ℝ) ≤ 1

theorem tol_facts {D N : ℕ} {X : ℚ} (hD : D ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 2 ^ 10) (hN : N ≤ 2 ^ 16) :
    TolFacts D N X := by
  obtain ⟨hE0, hE1⟩ := expErr_le hD hX0 hX
  have hX' : (X : ℝ) ≤ 2 ^ 10 := by exact_mod_cast hX
  have hX0' : (0 : ℝ) ≤ X := by exact_mod_cast hX0
  have hnb : nblocks N ≤ 1024 := by unfold nblocks; omega
  have hnb' : (nblocks N : ℝ) ≤ 1024 := by exact_mod_cast hnb
  set E := ((expErr D X : ℚ) : ℝ) with hE
  have hd1 : 1 ≤ ((drift D X : ℚ) : ℝ) := by rw [drift_cast]; nlinarith
  have hd2 : ((drift D X : ℚ) : ℝ) ≤ 1 + 1 / 2 ^ 11 := by rw [drift_cast]; nlinarith
  have hsA1 : 1 ≤ stepAr ^ nblocks N := one_le_pow₀ stepAr_one
  have hsA2 := pow_le_of_step (n := nblocks N) stepAr_one stepAr_le' hnb (by norm_num)
  have hsL1 : 1 ≤ stepLr ^ nblocks N := one_le_pow₀ stepLr_one
  have hsL2 := pow_le_of_step (n := nblocks N) stepLr_one stepLr_le' hnb (by norm_num)
  have hp16 := psi16r_le hE0 hE1
  have hp160 := psi16r_nonneg hE0
  have hpL := psiLr_le hE0 hE1
  have hpL0 := psiLr_nonneg hE0
  have hfa1 : 1 ≤ ((facc D N X : ℚ) : ℝ) := by
    rw [facc_cast]
    have := mul_le_mul (by linarith : (1 : ℝ) ≤ 1 + psi16r E) hd1 zero_le_one (by linarith)
    have := mul_le_mul this hsA1 zero_le_one (by positivity)
    linarith
  have hfa2 : ((facc D N X : ℚ) : ℝ) ≤ 6 / 5 := by
    rw [facc_cast]
    have h1 := mul_le_mul (by linarith : 1 + psi16r E ≤ 1 + 1 / 7) hd2 (by linarith) (by norm_num)
    have h2 := mul_le_mul h1 hsA2 (by positivity) (by norm_num)
    norm_num at h2 ⊢; linarith
  have hfl1 : 1 ≤ ((fl D N X : ℚ) : ℝ) := by
    rw [fl_cast]
    have := mul_le_mul (by linarith : (1 : ℝ) ≤ 1 + psiLr E) hd1 zero_le_one (by linarith)
    have := mul_le_mul this hsL1 zero_le_one (by positivity)
    linarith
  have hfl2 : ((fl D N X : ℚ) : ℝ) ≤ 6 / 5 := by
    rw [fl_cast]
    have h1 := mul_le_mul (by linarith : 1 + psiLr E ≤ 1 + 1 / 7) hd2 (by linarith) (by norm_num)
    have h2 := mul_le_mul h1 hsL2 (by positivity) (by norm_num)
    norm_num at h2 ⊢; linarith
  have hw : ((wlow D X : ℚ) : ℝ) = 1 - E := by unfold wlow; push_cast; rfl
  have hw1 : 7 / 8 ≤ ((wlow D X : ℚ) : ℝ) := by rw [hw]; linarith
  have hb : ((betaL D N X : ℚ) : ℝ) =
      (((fl D N X : ℚ) : ℝ) - 1) + ((fl D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) := by
    unfold betaL absL; push_cast; ring
  have hbt : 0 ≤ ((fl D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) := by
    apply div_nonneg _ (by linarith); positivity
  have hbt1 : ((fl D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) ≤ 1 / 2 ^ 80 := by
    rw [div_le_iff₀ (by linarith)]
    have : ((fl D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 100)) ≤ 6 / 5 * (1024 * (1 / 2 ^ 100)) :=
      mul_le_mul hfl2 (mul_le_mul_of_nonneg_right hnb' (by positivity)) (by positivity) (by norm_num)
    nlinarith
  have hb0 : 0 ≤ ((betaL D N X : ℚ) : ℝ) := by rw [hb]; linarith
  have hb1 : ((betaL D N X : ℚ) : ℝ) ≤ 1 / 4 := by rw [hb]; norm_num at hbt1 ⊢; linarith
  have hden : ((denQ D N X : ℚ) : ℝ) = 1 - ((betaL D N X : ℚ) : ℝ) := by unfold denQ; push_cast; rfl
  have hden1 : 3 / 4 ≤ ((denQ D N X : ℚ) : ℝ) := by rw [hden]; linarith
  have hrq : ((relQ D N X : ℚ) : ℝ) =
      (((facc D N X : ℚ) : ℝ) - 1 + ((betaL D N X : ℚ) : ℝ)) / ((denQ D N X : ℚ) : ℝ) := by
    unfold relQ; push_cast; rfl
  have hqs : ((absQS D N X : ℚ) : ℝ) =
      ((facc D N X : ℚ) : ℝ) * (1 / 2 ^ 25 + 1 / 2 ^ 125) / (((wlow D X : ℚ) : ℝ) * ((denQ D N X : ℚ) : ℝ)) := by
    unfold absQS omega0; push_cast; rfl
  have hq0 : ((absQ0 D N X : ℚ) : ℝ) =
      ((facc D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 85)) / (((wlow D X : ℚ) : ℝ) * ((denQ D N X : ℚ) : ℝ)) := by
    unfold absQ0 absA; push_cast; rfl
  have hwd : 21 / 32 ≤ ((wlow D X : ℚ) : ℝ) * ((denQ D N X : ℚ) : ℝ) := by
    have := mul_le_mul hw1 hden1 (by norm_num) (by linarith); norm_num at this ⊢; linarith
  refine ⟨hE0, hE1, hd1, hd2, hfa1, hfa2, hfl1, hfl2, hw, hb, hb0, hb1, hden, hrq, ?_, ?_, hqs, ?_, ?_, hq0, ?_, ?_⟩
  · rw [hrq]; apply div_nonneg (by linarith) (by linarith)
  · rw [hrq, div_le_iff₀ (by linarith)]; nlinarith
  · rw [hqs]; apply div_nonneg (by positivity) (by nlinarith)
  · rw [hqs, div_le_iff₀ (by linarith)]
    have : ((facc D N X : ℚ) : ℝ) * (1 / 2 ^ 25 + 1 / 2 ^ 125) ≤ 6 / 5 * (1 / 2 ^ 25 + 1 / 2 ^ 125) :=
      mul_le_mul_of_nonneg_right hfa2 (by positivity)
    nlinarith
  · rw [hq0]; apply div_nonneg (by positivity) (by nlinarith)
  · rw [hq0, div_le_iff₀ (by linarith)]
    have : ((facc D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 85)) ≤ 6 / 5 * (1024 * (1 / 2 ^ 85)) :=
      mul_le_mul hfa2 (mul_le_mul_of_nonneg_right hnb' (by positivity)) (by positivity) (by norm_num)
    norm_num at this ⊢; linarith

/-! ## Facts about the row after the loop -/

theorem psum_ge_single {al : ℕ → Bool} {t : ℕ} {f : ℕ → ℝ} (hf : ∀ j, 0 ≤ f j) {j : ℕ} (hj : j < 64 * t)
    (ha : al j = true) : f j ≤ psum al t f := by
  unfold psum
  have := Finset.single_le_sum (f := fun i => if al i = true then f i else 0)
    (fun i _ => by split_ifs <;> simp [hf i]) (Finset.mem_range.mpr hj)
  simpa [ha] using this

/-- The sum of a row's weights against its maximum is at least `1 − E`. -/
theorem W_ge {R : RowCtx} (hR : R.Ok) {t : ℕ} {m : F32} (hm : MInv R t m) :
    1 - R.E ≤ psum R.al t (wt R m.toReal) := by
  obtain ⟨j, hj, ha, hd⟩ := hm.attained
  have h1 := psum_ge_single (f := wt R m.toReal) (fun i => (wt_pos _ _ _).le) hj ha
  have h2 : (2 : ℝ) ^ (-R.E) ≤ wt R m.toReal j := by
    unfold wt; apply two_rpow_le
    have := le_abs_self (m.toReal - R.c * R.sv j); linarith
  have := one_sub_le_two_rpow_neg hR.E_nonneg (by linarith [hR.E_le])
  linarith

/-- The relative bound after the loop, with `exp` bounded by the drift. -/
theorem ffac_le_drift {R : RowCtx} (hR : R.Ok) {psi step : ℝ} {T nb : ℕ} {m : ℝ} (hpsi : 0 ≤ psi)
    (hstep : 1 ≤ step) (hT : T ≤ nb) (hm : |m| ≤ R.Y + R.E) :
    ffac R psi step T m ≤ (1 + psi) * (1 + 4 * (1 / 2 ^ 24) * (R.Y + R.E)) * step ^ nb := by
  unfold ffac
  have hY := hR.Y_le
  have hE := hR.E_le
  have hm0 := m_nonneg hm
  have hm1 : m + R.Y + R.E ≤ 2 * (R.Y + R.E) := by have := le_abs_self m; linarith
  have h1 : Real.exp ((m + R.Y + R.E) / 2 ^ 24) ≤ 1 + 4 * (1 / 2 ^ 24) * (R.Y + R.E) := by
    have := exp_le_two (y := (m + R.Y + R.E) / 2 ^ 24) (by positivity) (by
      rw [div_le_iff₀ (by norm_num)]; linarith)
    have : 2 * ((m + R.Y + R.E) / 2 ^ 24) ≤ 4 * (1 / 2 ^ 24) * (R.Y + R.E) := by
      rw [show 2 * ((m + R.Y + R.E) / 2 ^ 24) = 2 * (1 / 2 ^ 24) * (m + R.Y + R.E) by ring]
      nlinarith
    linarith
  have h2 : step ^ T ≤ step ^ nb := pow_le_pow_right₀ hstep hT
  have h0 : 0 ≤ step ^ T := pow_nonneg (by linarith) T
  have := mul_le_mul_of_nonneg_left h1 (by linarith : (0 : ℝ) ≤ 1 + psi)
  have h3 : 0 ≤ (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) := by positivity
  calc (1 + psi) * Real.exp ((m + R.Y + R.E) / 2 ^ 24) * step ^ T
      ≤ (1 + psi) * (1 + 4 * (1 / 2 ^ 24) * (R.Y + R.E)) * step ^ T := mul_le_mul_of_nonneg_right this h0
    _ ≤ (1 + psi) * (1 + 4 * (1 / 2 ^ 24) * (R.Y + R.E)) * step ^ nb :=
        mul_le_mul_of_nonneg_left h2 (by have := hR.Y_nonneg; have := hR.E_nonneg; positivity)

/-- **`acc / l_i` against `S / W`.** -/
theorem ratio_err {a S Mg Vs l W F β wl ω Ta : ℝ}
    (hW : wl ≤ W) (hwl : 0 < wl) (hβ0 : 0 ≤ β) (hβ : β < 1) (hS : |S| ≤ Mg)
    (hl : |l - W| ≤ β * W) (ha : |a - S| ≤ (F - 1) * Mg + F * (ω * Vs + Ta))
    (hF : 1 ≤ F) (hVs : 0 ≤ Vs) (hTa : 0 ≤ Ta) (hω : 0 ≤ ω) :
    (1 - β) * W ≤ l ∧ |a / l - S / W| ≤ (F - 1 + β) / (1 - β) * (Mg / W) + F * ω / (wl * (1 - β)) * Vs +
      F * Ta / (wl * (1 - β)) := by
  have hW0 : 0 < W := lt_of_lt_of_le hwl hW
  have hlW : (1 - β) * W ≤ l := by have := neg_abs_le (l - W); nlinarith
  have h1β : 0 < 1 - β := by linarith
  have hl0 : 0 < l := lt_of_lt_of_le (mul_pos h1β hW0) hlW
  have hMg0 : 0 ≤ Mg := (abs_nonneg _).trans hS
  refine ⟨hlW, ?_⟩
  have hnum : |a * W - l * S| ≤ (|a - S| + β * Mg) * W := by
    have e : a * W - l * S = (a - S) * W - S * (l - W) := by ring
    rw [e]
    refine (abs_sub _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_pos hW0]
    have := mul_le_mul hS hl (abs_nonneg _) hMg0
    nlinarith
  have key : |a / l - S / W| ≤ (|a - S| + β * Mg) / ((1 - β) * W) := by
    rw [div_sub_div _ _ hl0.ne' hW0.ne', abs_div, abs_of_pos (mul_pos hl0 hW0),
      div_le_div_iff₀ (mul_pos hl0 hW0) (mul_pos h1β hW0)]
    have h0 : 0 ≤ (|a - S| + β * Mg) * W := by positivity
    calc |a * W - l * S| * ((1 - β) * W) ≤ (|a - S| + β * Mg) * W * ((1 - β) * W) :=
          mul_le_mul_of_nonneg_right hnum (by positivity)
      _ ≤ (|a - S| + β * Mg) * W * l := mul_le_mul_of_nonneg_left hlW h0
      _ = (|a - S| + β * Mg) * (l * W) := by ring
  refine key.trans ?_
  rw [div_le_iff₀ (mul_pos h1β hW0)]
  have e : ((F - 1 + β) / (1 - β) * (Mg / W) + F * ω / (wl * (1 - β)) * Vs + F * Ta / (wl * (1 - β))) *
      ((1 - β) * W) = (F - 1 + β) * Mg + (F * ω * Vs + F * Ta) * (W / wl) := by
    field_simp
    ring
  rw [e]
  have hWw : 1 ≤ W / wl := by rw [le_div_iff₀ hwl]; linarith
  have h0 : 0 ≤ F * ω * Vs + F * Ta := by have : 0 ≤ F := by linarith
                                          positivity
  have := mul_le_mul_of_nonneg_left hWw h0
  nlinarith

/-- A weight split into the exact one, `exp(σ·s)`, the factor `ρ = exp(qk_scale·ln 2·s − σ·s)` and
the common `2^-m`. -/
theorem wt_split (R : RowCtx) (σ m : ℝ) (j : ℕ) :
    wt R m j = Real.exp (-(Real.log 2 * m)) *
      (Real.exp (σ * R.sv j) * Real.exp (Real.log 2 * (R.c * R.sv j) - σ * R.sv j)) := by
  unfold wt
  rw [Real.rpow_def_of_pos (by norm_num), ← Real.exp_add, ← Real.exp_add]
  congr 1; ring

theorem thetaW_cast (X : ℚ) : ((thetaW X : ℚ) : ℝ) = 4 * ((X : ℝ) / 2 ^ 22) := by
  unfold thetaW; push_cast; ring

theorem lastRel_cast : ((lastRel : ℚ) : ℝ) = 1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22 := by
  unfold lastRel; push_cast; ring

/-- The weights of a row over its keys: the kernel's against the exact ones. -/
theorem row_weights {R : RowCtx} {T : ℕ} {ks : Finset ℕ} (hks : ∀ f, psum R.al T f = ∑ j ∈ ks, f j)
    (hne : ks.Nonempty) {σ θ : ℝ} (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1 / 2)
    (hθ : ∀ j ∈ ks, |Real.log 2 * (R.c * R.sv j) - σ * R.sv j| ≤ θ) (m : ℝ) (ve : ℕ → ℝ) :
    |psum R.al T (fun j => wt R m j * ve j) / psum R.al T (wt R m) -
        (∑ j ∈ ks, Real.exp (σ * R.sv j) * ve j) / ∑ j ∈ ks, Real.exp (σ * R.sv j)| ≤
      4 * θ * ((∑ j ∈ ks, Real.exp (σ * R.sv j) * |ve j|) / ∑ j ∈ ks, Real.exp (σ * R.sv j)) ∧
    psum R.al T (fun j => wt R m j * |ve j|) / psum R.al T (wt R m) ≤
      (1 + 4 * θ) * ((∑ j ∈ ks, Real.exp (σ * R.sv j) * |ve j|) / ∑ j ∈ ks, Real.exp (σ * R.sv j)) ∧
    psum R.al T (wt R m) = Real.exp (-(Real.log 2 * m)) *
      ∑ j ∈ ks, Real.exp (σ * R.sv j) * Real.exp (Real.log 2 * (R.c * R.sv j) - σ * R.sv j) := by
  set K := Real.exp (-(Real.log 2 * m)) with hK
  have hK0 : 0 < K := Real.exp_pos _
  set ρ : ℕ → ℝ := fun j => Real.exp (Real.log 2 * (R.c * R.sv j) - σ * R.sv j) with hρd
  have hρ : ∀ j ∈ ks, Real.exp (-θ) ≤ ρ j ∧ ρ j ≤ Real.exp θ := fun j hj => by
    have := abs_le.mp (hθ j hj)
    exact ⟨Real.exp_le_exp.mpr (by linarith), Real.exp_le_exp.mpr (by linarith)⟩
  have hw := wmean_err (ks := ks) (w := fun j => Real.exp (σ * R.sv j)) (ρ := ρ) (v := ve) hθ0 hθ1 hne
    (fun j _ => Real.exp_pos _) hρ
  have e1 : psum R.al T (wt R m) = K * ∑ j ∈ ks, Real.exp (σ * R.sv j) * ρ j := by
    rw [hks, Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => wt_split R σ m j
  have e2 : psum R.al T (fun j => wt R m j * ve j) = K * ∑ j ∈ ks, Real.exp (σ * R.sv j) * ρ j * ve j := by
    rw [hks, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by rw [wt_split R σ m j]; ring
  have e3 : psum R.al T (fun j => wt R m j * |ve j|) = K * ∑ j ∈ ks, Real.exp (σ * R.sv j) * ρ j * |ve j| := by
    rw [hks, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by rw [wt_split R σ m j]; ring
  rw [e1, e2, e3, mul_div_mul_left _ _ hK0.ne', mul_div_mul_left _ _ hK0.ne']
  exact ⟨hw.1, hw.2, rfl⟩

/-- **The output element** (lines 242 and 245): `(acc / l_i).to(float16)` against the exact
attention over the row's keys. -/
theorem final_o {P : Platform} (hdivF : Rules.Num.DivFull P) {R : RowCtx} (hR : R.Ok)
    {D N : ℕ} {X : ℚ} (hD : D ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 2 ^ 10) (hN : N ≤ 2 ^ 16)
    (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ))
    {T : ℕ} (hT : T ≤ nblocks N)
    {ks : Finset ℕ} (hks : ∀ f, psum R.al T f = ∑ j ∈ ks, f j) (hne : ks.Nonempty)
    {σ : ℝ} (hθ : ∀ j ∈ ks, |Real.log 2 * (R.c * R.sv j) - σ * R.sv j| ≤ X / 2 ^ 22)
    {ve : ℕ → ℝ} (hve : ∀ j, |ve j| ≤ 2 ^ 15)
    {m l a o' : F32} {o16 : F16} (hM : MInv R T m) (hL : LInv R T m l) (hA : AInv R ve T m a)
    (hdv : P.div .binary32 a l o') (hc : FloatV.NaNEq (FloatV.cvt o') o16) :
    o16.isFinite = true ∧
      |o16.toReal - (∑ j ∈ ks, Real.exp (σ * R.sv j) * ve j) / ∑ j ∈ ks, Real.exp (σ * R.sv j)| ≤
        (tolO D N X : ℝ) * ((∑ j ∈ ks, Real.exp (σ * R.sv j) * |ve j|) / ∑ j ∈ ks, Real.exp (σ * R.sv j)) +
          (tolS D N X : ℝ) * ∑ j ∈ ks, |ve j| + (tolF D N X : ℝ) := by
  have tf := tol_facts hD hX0 hX hN
  have hnb : nblocks N ≤ 1024 := by unfold nblocks; omega
  have hTr : (T : ℝ) ≤ nblocks N := by exact_mod_cast hT
  have hE0 := hR.E_nonneg
  have hX' : (X : ℝ) ≤ 2 ^ 10 := by exact_mod_cast hX
  have hX0' : (0 : ℝ) ≤ X := by exact_mod_cast hX0
  have hWl : ((wlow D X : ℚ) : ℝ) ≤ psum R.al T (wt R m.toReal) := by
    rw [tf.wlow_eq, ← hE]; exact W_ge hR hM
  have hwl0 : 0 < ((wlow D X : ℚ) : ℝ) := by rw [tf.wlow_eq]; linarith [tf.E1]
  have hW0 : 0 < psum R.al T (wt R m.toReal) := lt_of_lt_of_le hwl0 hWl
  have hMg0 : 0 ≤ psum R.al T (fun j => wt R m.toReal j * |ve j|) :=
    psum_nonneg fun j => mul_nonneg (wt_pos _ _ _).le (abs_nonneg _)
  have hVs0 : 0 ≤ psum R.al T (fun j => |ve j|) := psum_nonneg fun j => abs_nonneg _
  have hS := psum_abs_wt R m.toReal T ve
  -- `l_i`
  have hFl : ffac R (psiLr R.E) stepLr T m.toReal ≤ (fl D N X : ℝ) := by
    refine (ffac_le_drift hR (psiLr_nonneg hE0) stepLr_one hT hM.bound).trans (le_of_eq ?_)
    rw [fl_cast, drift_cast, hY, hE]
  have hl : |l.toReal - psum R.al T (wt R m.toReal)| ≤ (betaL D N X : ℝ) * psum R.al T (wt R m.toReal) := by
    have h := hL.2
    rw [tf.beta_eq]
    have k1 : (ffac R (psiLr R.E) stepLr T m.toReal - 1) * psum R.al T (wt R m.toReal) ≤
        ((fl D N X : ℝ) - 1) * psum R.al T (wt R m.toReal) :=
      mul_le_mul_of_nonneg_right (by linarith) hW0.le
    have k2 : ffac R (psiLr R.E) stepLr T m.toReal * (T * absLr) ≤ (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) := by
      unfold absLr
      have hFl1 := ffac_one_le (t := T) (psiLr_nonneg hE0) stepLr_one (m_nonneg hM.bound)
      exact mul_le_mul hFl (mul_le_mul_of_nonneg_right hTr (by positivity)) (by positivity) (by linarith)
    have k3 : (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) ≤
        (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) / (wlow D X : ℝ) * psum R.al T (wt R m.toReal) := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hwl0]
      have : 0 ≤ (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) := by have := tf.fl1; positivity
      exact mul_le_mul_of_nonneg_left hWl this
    have e : (((fl D N X : ℚ) : ℝ) - 1 + ((fl D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 100)) /
        ((wlow D X : ℚ) : ℝ)) * psum R.al T (wt R m.toReal) =
      (((fl D N X : ℚ) : ℝ) - 1) * psum R.al T (wt R m.toReal) + ((fl D N X : ℚ) : ℝ) *
        ((nblocks N : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) * psum R.al T (wt R m.toReal) := by ring
    rw [e]; linarith
  -- `acc`
  have hFa : ffac R (psi16r R.E) stepAr T m.toReal ≤ (facc D N X : ℝ) := by
    refine (ffac_le_drift hR (psi16r_nonneg hE0) stepAr_one hT hM.bound).trans (le_of_eq ?_)
    rw [facc_cast, drift_cast, hY, hE]
  have ha : |a.toReal - psum R.al T (fun j => wt R m.toReal j * ve j)| ≤
      ((facc D N X : ℝ) - 1) * psum R.al T (fun j => wt R m.toReal j * |ve j|) +
        (facc D N X : ℝ) * ((1 / 2 ^ 25 + 1 / 2 ^ 125) * psum R.al T (fun j => |ve j|) +
          nblocks N * (1 / 2 ^ 85)) := by
    have h := hA.2
    have hFa1 := ffac_one_le (t := T) (psi16r_nonneg hE0) stepAr_one (m_nonneg hM.bound)
    have k1 : (ffac R (psi16r R.E) stepAr T m.toReal - 1) * psum R.al T (fun j => wt R m.toReal j * |ve j|) ≤
        ((facc D N X : ℝ) - 1) * psum R.al T (fun j => wt R m.toReal j * |ve j|) :=
      mul_le_mul_of_nonneg_right (by linarith) hMg0
    have k2 : ffac R (psi16r R.E) stepAr T m.toReal * (om0 * psum R.al T (fun j => |ve j|) + T * absAr) ≤
        (facc D N X : ℝ) * ((1 / 2 ^ 25 + 1 / 2 ^ 125) * psum R.al T (fun j => |ve j|) + nblocks N * (1 / 2 ^ 85)) := by
      unfold om0 absAr
      refine mul_le_mul hFa ?_ (by positivity) (by linarith [tf.facc1])
      have := mul_le_mul_of_nonneg_right hTr (by positivity : (0 : ℝ) ≤ 1 / 2 ^ 85)
      linarith
    linarith
  -- the ratio `acc / l_i` against the kernel's weighted mean
  obtain ⟨hlW, hq⟩ := ratio_err hWl hwl0 tf.beta0 (by linarith [tf.beta1]) hS hl ha tf.facc1 hVs0
    (by positivity) (by positivity)
  rw [← tf.den_eq] at hq hlW
  rw [← tf.relQ_eq, ← tf.absQS_eq, ← tf.absQ0_eq] at hq
  -- the kernel's weights against the exact ones
  have hθ0 : (0 : ℝ) ≤ X / 2 ^ 22 := by positivity
  obtain ⟨hw1, hw2, -⟩ := row_weights hks hne hθ0 (by rw [div_le_iff₀ (by norm_num)]; linarith) hθ m.toReal ve
  rw [← thetaW_cast] at hw1 hw2
  -- bounds
  have hMW : psum R.al T (fun j => wt R m.toReal j * |ve j|) / psum R.al T (wt R m.toReal) ≤ 2 ^ 15 := by
    rw [div_le_iff₀ hW0, ← psum_mul]
    exact psum_le fun j _ _ => by
      rw [mul_comm (2 ^ 15 : ℝ)]; exact mul_le_mul_of_nonneg_left (hve j) (wt_pos _ _ _).le
  have hVs : psum R.al T (fun j => |ve j|) ≤ 2 ^ 31 := by
    have := psum_le_card (al := R.al) (t := T) (f := fun j => |ve j|) (B := 2 ^ 15) (by norm_num)
      fun j _ _ => hve j
    have h64 : (64 * T : ℝ) ≤ 2 ^ 16 := by
      have : 64 * T ≤ 2 ^ 16 := by unfold nblocks at hT; omega
      exact_mod_cast this
    nlinarith
  have hrW : |psum R.al T (fun j => wt R m.toReal j * ve j) / psum R.al T (wt R m.toReal)| ≤
      psum R.al T (fun j => wt R m.toReal j * |ve j|) / psum R.al T (wt R m.toReal) := by
    rw [abs_div, abs_of_pos hW0]; exact div_le_div_of_nonneg_right hS hW0.le
  have hl0 : 0 < l.toReal := by
    have := mul_pos (show (0 : ℝ) < (denQ D N X : ℝ) by rw [tf.den_eq]; linarith [tf.beta1]) hW0
    linarith
  set q := a.toReal / l.toReal with hqd
  have hqb : |q| ≤ (1 + (relQ D N X : ℝ)) * (psum R.al T (fun j => wt R m.toReal j * |ve j|) /
      psum R.al T (wt R m.toReal)) + (absQS D N X : ℝ) * psum R.al T (fun j => |ve j|) + (absQ0 D N X : ℝ) := by
    have := abs_sub_abs_le_abs_sub q (psum R.al T (fun j => wt R m.toReal j * ve j) / psum R.al T (wt R m.toReal))
    linarith
  have hq52 : |q| ≤ 52700 := by
    have k1 : (1 + (relQ D N X : ℝ)) * (psum R.al T (fun j => wt R m.toReal j * |ve j|) /
        psum R.al T (wt R m.toReal)) ≤ (1 + 3 / 5) * 2 ^ 15 :=
      mul_le_mul (by linarith [tf.relQ1]) hMW (div_nonneg hMg0 hW0.le) (by norm_num)
    have k2 : (absQS D N X : ℝ) * psum R.al T (fun j => |ve j|) ≤ 1 / 2 ^ 23 * 2 ^ 31 :=
      mul_le_mul tf.absQS1 hVs hVs0 (by norm_num)
    have := tf.absQ01
    norm_num at k1 k2 ⊢; linarith
  -- line 242: `acc / l_i[:, None]`
  have hq' : |a.toRat / l.toRat| ≤ 2 ^ 126 := by
    have : ((|a.toRat / l.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast; rw [← toReal_def, ← toReal_def]; rw [← hqd]; linarith
    exact_mod_cast this
  have hl0' : l.toRat ≠ 0 := by
    intro h; rw [toReal_def, h] at hl0; simp at hl0
  obtain ⟨ho'f, ho'e⟩ := hdivF a l o' hdv hA.1 hL.1 hl0' hq'
  have ho'r : |o'.toReal - q| ≤ 1 / 2 ^ 22 * |q| + 1 / 2 ^ 148 := by
    have := (Rat.cast_le (K := ℝ)).mpr ho'e
    push_cast at this
    rw [show (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 by norm_num, show (2 : ℝ) ^ (-148 : ℤ) = 1 / 2 ^ 148 by norm_num] at this
    simp only [← toReal_def] at this
    exact this
  have ho'b : |o'.toReal| ≤ 65504 := by
    have := abs_sub_abs_le_abs_sub o'.toReal q
    norm_num at ho'r ⊢; linarith
  -- line 245: `.to(float16)`
  obtain ⟨ho16f, ho16e⟩ := f16_cvt hc ho'f ho'b
  refine ⟨ho16f, ?_⟩
  -- assembling the tolerance
  have hto : (tolO D N X : ℝ) = (1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22) * (1 + (relQ D N X : ℝ)) *
      (1 + (thetaW X : ℝ)) + (relQ D N X : ℝ) * (1 + (thetaW X : ℝ)) + (thetaW X : ℝ) := by
    unfold tolO; push_cast; rw [lastRel_cast]
  have hts : (tolS D N X : ℝ) = (absQS D N X : ℝ) * (1 + (1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22)) := by
    unfold tolS; push_cast; rw [lastRel_cast]
  have htF : (tolF D N X : ℝ) = (absQ0 D N X : ℝ) * (1 + (1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22)) +
      1 / 2 ^ 11 * (1 / 2 ^ 148) + 1 / 2 ^ 25 + 1 / 2 ^ 148 := by
    unfold tolF; push_cast; rw [lastRel_cast]
  have hVk : psum R.al T (fun j => |ve j|) = ∑ j ∈ ks, |ve j| := hks _
  rw [hto, hts, htF, ← hVk]
  set Ab := (∑ j ∈ ks, Real.exp (σ * R.sv j) * |ve j|) / ∑ j ∈ ks, Real.exp (σ * R.sv j) with hAb
  set MW := psum R.al T (fun j => wt R m.toReal j * |ve j|) / psum R.al T (wt R m.toReal) with hMWd
  have htW0 : 0 ≤ (thetaW X : ℝ) := by rw [thetaW_cast]; positivity
  set rq := (relQ D N X : ℝ)
  set tW := (thetaW X : ℝ)
  have hrq0 : 0 ≤ rq := tf.relQ0
  set lr : ℝ := 1 / 2 ^ 11 * (1 + 1 / 2 ^ 22) + 1 / 2 ^ 22 with hlr
  have hlr0 : 0 ≤ lr := by rw [hlr]; positivity
  have p1 := mul_le_mul_of_nonneg_left hqb hlr0
  have p2 := mul_le_mul_of_nonneg_left hw2 (by linarith : (0 : ℝ) ≤ 1 + rq)
  have p3 := mul_le_mul_of_nonneg_left hw2 hrq0
  have p4 := mul_le_mul_of_nonneg_left p2 hlr0
  have t1 := abs_sub_le o16.toReal o'.toReal q
  have t2 := abs_sub_le q (psum R.al T (fun j => wt R m.toReal j * ve j) / psum R.al T (wt R m.toReal))
    ((∑ j ∈ ks, Real.exp (σ * R.sv j) * ve j) / ∑ j ∈ ks, Real.exp (σ * R.sv j))
  have t3 := abs_sub_le o16.toReal q ((∑ j ∈ ks, Real.exp (σ * R.sv j) * ve j) / ∑ j ∈ ks, Real.exp (σ * R.sv j))
  have t4 := abs_sub_abs_le_abs_sub o'.toReal q
  have e1 : |o16.toReal - o'.toReal| ≤ (|q| + 1 / 2 ^ 22 * |q| + 1 / 2 ^ 148) / 2 ^ 11 + 1 / 2 ^ 25 := by
    have : |o'.toReal| ≤ |q| + 1 / 2 ^ 22 * |q| + 1 / 2 ^ 148 := by linarith
    have := div_le_div_of_nonneg_right this (by norm_num : (0 : ℝ) ≤ 2 ^ 11)
    linarith
  rw [hlr] at p1 p4
  linarith

/-- `l_i` after the loop against the sum of the kernel's weights. -/
theorem l_final {R : RowCtx} (hR : R.Ok) {D N : ℕ} {X : ℚ} (tf : TolFacts D N X)
    (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ)) {T : ℕ} (hT : T ≤ nblocks N)
    {m l : F32} (hM : MInv R T m) (hL : LInv R T m l) :
    ((wlow D X : ℚ) : ℝ) ≤ psum R.al T (wt R m.toReal) ∧
      |l.toReal - psum R.al T (wt R m.toReal)| ≤ (betaL D N X : ℝ) * psum R.al T (wt R m.toReal) := by
  have hTr : (T : ℝ) ≤ nblocks N := by exact_mod_cast hT
  have hE0 := hR.E_nonneg
  have hWl : ((wlow D X : ℚ) : ℝ) ≤ psum R.al T (wt R m.toReal) := by
    rw [tf.wlow_eq, ← hE]; exact W_ge hR hM
  have hwl0 : 0 < ((wlow D X : ℚ) : ℝ) := by rw [tf.wlow_eq]; linarith [tf.E1]
  have hW0 : 0 < psum R.al T (wt R m.toReal) := lt_of_lt_of_le hwl0 hWl
  refine ⟨hWl, ?_⟩
  have hFl : ffac R (psiLr R.E) stepLr T m.toReal ≤ (fl D N X : ℝ) := by
    refine (ffac_le_drift hR (psiLr_nonneg hE0) stepLr_one hT hM.bound).trans (le_of_eq ?_)
    rw [fl_cast, drift_cast, hY, hE]
  have h := hL.2
  rw [tf.beta_eq]
  have k1 : (ffac R (psiLr R.E) stepLr T m.toReal - 1) * psum R.al T (wt R m.toReal) ≤
      ((fl D N X : ℝ) - 1) * psum R.al T (wt R m.toReal) :=
    mul_le_mul_of_nonneg_right (by linarith) hW0.le
  have k2 : ffac R (psiLr R.E) stepLr T m.toReal * (T * absLr) ≤ (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) := by
    unfold absLr
    have hFl1 := ffac_one_le (t := T) (psiLr_nonneg hE0) stepLr_one (m_nonneg hM.bound)
    exact mul_le_mul hFl (mul_le_mul_of_nonneg_right hTr (by positivity)) (by positivity) (by linarith)
  have k3 : (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) ≤
      (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) / (wlow D X : ℝ) * psum R.al T (wt R m.toReal) := by
    rw [div_mul_eq_mul_div, le_div_iff₀ hwl0]
    have : 0 ≤ (fl D N X : ℝ) * (nblocks N * (1 / 2 ^ 100)) := by have := tf.fl1; positivity
    exact mul_le_mul_of_nonneg_left hWl this
  have e : (((fl D N X : ℚ) : ℝ) - 1 + ((fl D N X : ℚ) : ℝ) * ((nblocks N : ℝ) * (1 / 2 ^ 100)) /
      ((wlow D X : ℚ) : ℝ)) * psum R.al T (wt R m.toReal) =
    (((fl D N X : ℚ) : ℝ) - 1) * psum R.al T (wt R m.toReal) + ((fl D N X : ℚ) : ℝ) *
      ((nblocks N : ℝ) * (1 / 2 ^ 100)) / ((wlow D X : ℚ) : ℝ) * psum R.al T (wt R m.toReal) := by ring
  rw [e]; linarith

theorem logb_two_pow (n : ℕ) : Real.logb 2 ((2 : ℝ) ^ n) = n := by
  rw [Real.logb_pow, Real.logb_self_eq_one (by norm_num), mul_one]

/-- **The log-sum-exp element** (lines 241, 244): `m_i + log2(l_i)` against the exact `log₂` of
the sum of the row's weights. -/
theorem final_M {P : Platform} (hlog : Rules.Num.Log2F32 P) {R : RowCtx} (hR : R.Ok)
    {D N : ℕ} {X : ℚ} (hD : D ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 2 ^ 10) (hN : N ≤ 2 ^ 16)
    (hY : R.Y = 3 / 2 * (X : ℝ)) (hE : R.E = (expErr D X : ℝ))
    {T : ℕ} (hT : T ≤ nblocks N)
    {ks : Finset ℕ} (hks : ∀ f, psum R.al T f = ∑ j ∈ ks, f j) (hne : ks.Nonempty)
    {σ : ℝ} (hθ : ∀ j ∈ ks, |Real.log 2 * (R.c * R.sv j) - σ * R.sv j| ≤ X / 2 ^ 22)
    {m l lg Mv : F32} (hM : MInv R T m) (hL : LInv R T m l)
    (hlg : P.math .log2 .binary32 l lg) (hadd : FloatV.NaNEq (FloatV.add m lg) Mv) :
    Mv.isFinite = true ∧
      |Mv.toReal - Real.logb 2 (∑ j ∈ ks, Real.exp (σ * R.sv j))| ≤
        (tolM D N X : ℝ) + (u32 : ℝ) * |Real.logb 2 (∑ j ∈ ks, Real.exp (σ * R.sv j))| := by
  have tf := tol_facts hD hX0 hX hN
  have hnb : nblocks N ≤ 1024 := by unfold nblocks; omega
  have hE0 := hR.E_nonneg
  have hE8 := hR.E_le
  have hX' : (X : ℝ) ≤ 2 ^ 10 := by exact_mod_cast hX
  have hX0' : (0 : ℝ) ≤ X := by exact_mod_cast hX0
  obtain ⟨hWl, hl⟩ := l_final hR tf hY hE hT hM hL
  have hwl : (7 : ℝ) / 8 ≤ (wlow D X : ℝ) := by rw [tf.wlow_eq]; linarith [tf.E1]
  have hW0 : 0 < psum R.al T (wt R m.toReal) := by linarith
  have hb1 := tf.beta1
  have hb0 := tf.beta0
  -- `W ≤ 2^16·9/8`
  have hWu : psum R.al T (wt R m.toReal) ≤ 2 ^ 16 * (9 / 8) := by
    have := psum_le_card (al := R.al) (t := T) (f := wt R m.toReal) (B := 9 / 8) (by norm_num) fun j hj ha => by
      have h1 := hM.above j hj ha
      have : wt R m.toReal j ≤ (2 : ℝ) ^ R.E := two_rpow_le (by linarith)
      have := two_rpow_le_one_add hE0 (by linarith)
      linarith
    have h64 : (64 * T : ℝ) ≤ 2 ^ 16 := by
      have : 64 * T ≤ 2 ^ 16 := by unfold nblocks at hT; omega
      exact_mod_cast this
    nlinarith
  -- `l_i ∈ [1/2, 2^17]`
  have hlb := abs_le.mp hl
  have hbW := mul_le_mul_of_nonneg_left hWu hb0
  have hbW' := mul_le_mul_of_nonneg_right hb1 hW0.le
  have hl1 : 1 / 2 ≤ l.toReal := by nlinarith
  have hl2 : l.toReal ≤ 2 ^ 17 := by nlinarith
  have hl0 : 0 < l.toReal := by linarith
  have hlog17 : |Real.logb 2 l.toReal| ≤ 17 := by
    rw [abs_le]; constructor
    · have := (Real.logb_le_logb (b := 2) (by norm_num) (by norm_num) hl0).mpr hl1
      rw [show (1 : ℝ) / 2 = (2 : ℝ) ^ (-1 : ℝ) by norm_num, Real.logb_rpow (by norm_num) (by norm_num)] at this
      linarith
    · have := (Real.logb_le_logb (b := 2) (by norm_num) hl0 (by norm_num)).mpr hl2
      rw [logb_two_pow] at this; push_cast at this; linarith
  -- line 241: `tl.math.log2(l_i)`
  obtain ⟨g, hg, hacc⟩ := hlog
  have hg' : g = Real.logb 2 := by
    have e : MathFn.ref .log2 = some (Real.logb 2) := rfl
    rw [e] at hg; exact (Option.some.inj hg).symm
  subst hg'
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.logb 2 l.toReal| + 2 ^ (-149 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    rw [maxVal32]; norm_num; nlinarith
  obtain ⟨hlgf, hlge⟩ := hacc l lg hlg hL.1 (Set.mem_univ _) (show l.toReal ∈ Set.Ioi 0 from hl0) hroom
  rw [show (2 : ℝ) ^ (-22 : ℤ) = 1 / 2 ^ 22 by norm_num, show (2 : ℝ) ^ (-149 : ℤ) = 1 / 2 ^ 149 by norm_num]
    at hlge
  have hlg1 : |lg.toReal - Real.logb 2 l.toReal| ≤ 17 / 2 ^ 22 + 1 / 2 ^ 149 := by
    have := mul_le_mul_of_nonneg_left hlog17 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 22)
    linarith
  -- `log₂ l_i` against `log₂ W`
  have hlW : |Real.logb 2 l.toReal - Real.logb 2 (psum R.al T (wt R m.toReal))| ≤ 2 * (betaL D N X : ℝ) := by
    rw [← Real.logb_div hl0.ne' hW0.ne']
    have e : l.toReal / psum R.al T (wt R m.toReal) = 1 + (l.toReal - psum R.al T (wt R m.toReal)) /
        psum R.al T (wt R m.toReal) := by field_simp; ring
    have hβ : |(l.toReal - psum R.al T (wt R m.toReal)) / psum R.al T (wt R m.toReal)| ≤ (betaL D N X : ℝ) := by
      rw [abs_div, abs_of_pos hW0, div_le_iff₀ hW0]; exact hl
    rw [e]
    refine (abs_logb_one_add (hβ.trans hb1)).trans ?_
    linarith
  -- `m + log₂ W` against `lseRef`
  have hθ0 : (0 : ℝ) ≤ X / 2 ^ 22 := by positivity
  obtain ⟨-, -, hWe⟩ := row_weights hks hne hθ0 (by rw [div_le_iff₀ (by norm_num)]; linarith) hθ m.toReal
    (fun _ => 0)
  set B := ∑ j ∈ ks, Real.exp (σ * R.sv j) * Real.exp (Real.log 2 * (R.c * R.sv j) - σ * R.sv j) with hB
  set A := ∑ j ∈ ks, Real.exp (σ * R.sv j) with hA
  have hA0 : 0 < A := Finset.sum_pos (fun j _ => Real.exp_pos _) hne
  have hBA1 : Real.exp (-(X / 2 ^ 22 : ℝ)) * A ≤ B := by
    rw [hA, hB, Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      have := Real.exp_le_exp.mpr (show -(X / 2 ^ 22 : ℝ) ≤ Real.log 2 * (R.c * R.sv j) - σ * R.sv j by
        have := abs_le.mp (hθ j hj); linarith)
      rw [mul_comm]; exact mul_le_mul_of_nonneg_left this (Real.exp_pos _).le
  have hBA2 : B ≤ Real.exp (X / 2 ^ 22 : ℝ) * A := by
    rw [hA, hB, Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      have := Real.exp_le_exp.mpr (show Real.log 2 * (R.c * R.sv j) - σ * R.sv j ≤ (X / 2 ^ 22 : ℝ) by
        have := abs_le.mp (hθ j hj); linarith)
      rw [mul_comm (Real.exp (X / 2 ^ 22 : ℝ))]; exact mul_le_mul_of_nonneg_left this (Real.exp_pos _).le
  have hB0 : 0 < B := lt_of_lt_of_le (mul_pos (Real.exp_pos _) hA0) hBA1
  have hlse : |m.toReal + Real.logb 2 (psum R.al T (wt R m.toReal)) - Real.logb 2 A| ≤ 2 * (X / 2 ^ 22) := by
    rw [hWe, Real.logb_mul (Real.exp_pos _).ne' hB0.ne']
    have hK : Real.logb 2 (Real.exp (-(Real.log 2 * m.toReal))) = -m.toReal := by
      unfold Real.logb; rw [Real.log_exp]; field_simp
    rw [hK, show -m.toReal + Real.logb 2 B = Real.logb 2 B - 0 - m.toReal + m.toReal * 0 by ring]
    rw [show m.toReal + (Real.logb 2 B - 0 - m.toReal + m.toReal * 0) - Real.logb 2 A = Real.logb 2 B - Real.logb 2 A by
      ring, ← Real.logb_div hB0.ne' hA0.ne']
    exact abs_logb_of_exp (by rw [le_div_iff₀ hA0]; exact hBA1) (by rw [div_le_iff₀ hA0]; exact hBA2)
  -- line 241: `m_i + log2(l_i)`
  have hΔ : |m.toReal + lg.toReal - Real.logb 2 A| ≤
      17 / 2 ^ 22 + 2 * (betaL D N X : ℝ) + 1 / 2 ^ 21 * X + 1 / 2 ^ 149 := by
    have e : m.toReal + lg.toReal - Real.logb 2 A = (lg.toReal - Real.logb 2 l.toReal) +
        (Real.logb 2 l.toReal - Real.logb 2 (psum R.al T (wt R m.toReal))) +
        (m.toReal + Real.logb 2 (psum R.al T (wt R m.toReal)) - Real.logb 2 A) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    refine (add_le_add (abs_add_le _ _) le_rfl).trans ?_
    linarith
  have hroom2 : |m.toReal + lg.toReal| ≤ 2 ^ 120 := by
    have := hM.bound; have := hR.Y_le
    have := abs_add_le m.toReal lg.toReal
    have := abs_sub_abs_le_abs_sub lg.toReal (Real.logb 2 l.toReal)
    linarith
  obtain ⟨hMvf, hMve⟩ := f32_add hadd hM.fin hlgf hroom2
  refine ⟨hMvf, ?_⟩
  have htM : (tolM D N X : ℝ) = (1 + 1 / 2 ^ 24) * (17 / 2 ^ 22 + 2 * (betaL D N X : ℝ) + 1 / 2 ^ 21 * X +
      1 / 2 ^ 149) + 1 / 2 ^ 150 := by
    unfold tolM u32; push_cast; ring
  have hu : ((u32 : ℚ) : ℝ) = 1 / 2 ^ 24 := by unfold u32; push_cast; ring
  rw [htM, hu]
  have t1 := abs_sub_le Mv.toReal (m.toReal + lg.toReal) (Real.logb 2 A)
  have t2 : |m.toReal + lg.toReal| ≤ |Real.logb 2 A| + |m.toReal + lg.toReal - Real.logb 2 A| := by
    have := abs_sub_abs_le_abs_sub (m.toReal + lg.toReal) (Real.logb 2 A); linarith
  have := div_le_div_of_nonneg_right t2 (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
  have := div_le_div_of_nonneg_right hΔ (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
  linarith

end Kernels.AttnFwd
