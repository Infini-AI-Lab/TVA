import Proof.CoreDefs
import Proof.CoreLemmas

/-!
# The error of the ratio against the claim's tolerance
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- **The algebra of the comparison.** With `ρ = θ ψ`, the `coreErr` expression times `1 + ε` plus
`ε` is `θ (1 + ε) ψ (1 + ub) (1 + Γ) / (1 − Λ) − 1`, and the claim's `errE` is
`θ (1 + gam (1 + rho)) / (1 − rho) − 1`; so it suffices that `Γ ≤ gam` and
`(1 + ε) ψ (1 + ub) (1 − rho) ≤ 1 − Λ`. -/
theorem core_alg {θ ψ ε Λ Γ gam rho : ℝ} (hθ : 1 ≤ θ) (_hε : 0 ≤ ε) (_hψ : 0 ≤ ψ) (hΛ : Λ < 1)
    (hrho : rho < 1) (hrho0 : 0 ≤ rho) (hΓ : Γ ≤ gam) (hΓ0 : 0 ≤ Γ)
    (hZ : (1 + ε) * ψ * (1 + ub) * (1 - rho) ≤ 1 - Λ) :
    (θ * ψ * (ub + Γ * (1 + ub) + Λ) / (1 - Λ) + (θ * ψ - 1)) * (1 + ε) + ε ≤
      θ * (rho + gam * (1 + rho)) / (1 - rho) + (θ - 1) := by
  have hub0 : 0 ≤ ub := by unfold ub; norm_num
  have h1 : 0 < 1 - Λ := by linarith
  have h2 : 0 < 1 - rho := by linarith
  have hgam : 0 ≤ gam := hΓ0.trans hΓ
  have e1' : (θ * ψ * (ub + Γ * (1 + ub) + Λ) / (1 - Λ) + (θ * ψ - 1)) * (1 + ε) + ε =
      θ * ((1 + ε) * ψ * (1 + ub) * (1 + Γ)) / (1 - Λ) - 1 := by
    field_simp
    ring
  have e2' : θ * (rho + gam * (1 + rho)) / (1 - rho) + (θ - 1) =
      θ * (1 + gam * (1 + rho)) / (1 - rho) - 1 := by
    field_simp
    ring
  rw [e1', e2']
  have : θ * ((1 + ε) * ψ * (1 + ub) * (1 + Γ)) / (1 - Λ) ≤ θ * (1 + gam * (1 + rho)) / (1 - rho) := by
    rw [div_le_div_iff₀ h1 h2]
    have hθ0 : 0 ≤ θ := by linarith
    have a1 := mul_le_mul_of_nonneg_left hZ (mul_nonneg hθ0 (by linarith : 0 ≤ 1 + Γ))
    have a2 : (1 + Γ) ≤ 1 + gam * (1 + rho) := by nlinarith
    have a3 := mul_le_mul_of_nonneg_left a2 (mul_nonneg hθ0 h1.le)
    nlinarith
  linarith

set_option maxHeartbeats 1600000 in
/-- **The key inequality**, linear in the trips `T` and the score bound `Xp` once the exponentials
are bounded: `(1 + 2^-22)·ψ·(1 + ub)·(1 − rho) ≤ 1 − Λ`, with `1 − rho = 2 − exp x`. -/
theorem core_star {T Xp : ℝ} (hT1 : 1 ≤ T) (hT : T ≤ 4098) (hXp0 : 0 ≤ Xp) (hXp : Xp ≤ 4097) :
    (1 + 1 / 2 ^ 22) * ((1 + expm1 (T * e1 + 2 * e2 * Xp)) / (1 - expm1 (T * e1 + 2 * e2 * Xp))) *
        (1 + ub) * (2 - Real.exp (1 / 2 ^ 8 + (T + 8) * (1 / 2 ^ 20) + Xp * (1 / 2 ^ 17))) ≤
      1 - (expm1 (lam0 + lam1 * T) + 1 / 2 ^ 85) := by
  have hub : ub = 1 / 2 ^ 8 := rfl
  have he1 : e1 = 257 / 2 ^ 30 := rfl
  have he2 : e2 = 771 / 2 ^ 32 := rfl
  have hl0 : lam0 = 6 / 2 ^ 24 := rfl
  have hl1 : lam1 = 3 / 2 ^ 24 := rfl
  set φ := T * e1 + 2 * e2 * Xp with hφ
  have hφ0 : 0 ≤ φ := by rw [hφ, he1, he2]; positivity
  have hφ1 : φ ≤ 1 / 200 := by
    rw [hφ, he1, he2]
    have h1 : T * (257 / 2 ^ 30) ≤ 4098 * (257 / 2 ^ 30) := mul_le_mul_of_nonneg_right hT (by norm_num)
    have h2 : 2 * (771 / 2 ^ 32) * Xp ≤ 2 * (771 / 2 ^ 32) * 4097 :=
      mul_le_mul_of_nonneg_left hXp (by norm_num)
    norm_num at h1 h2 ⊢
    linarith
  set c := 101 / 100 * φ with hc
  have hc1 : c ≤ 101 / 20000 := by rw [hc]; linarith
  have hc0 : 0 ≤ c := by rw [hc]; positivity
  have hΦ : expm1 φ ≤ c := by
    have := exp_le_quad hφ0 (by linarith)
    unfold expm1; rw [hc]; nlinarith
  have hΦ0 : 0 ≤ expm1 φ := expm1_nonneg hφ0
  have hψ : (1 + expm1 φ) / (1 - expm1 φ) ≤ (1 + c) / (1 - c) := by
    rw [div_le_div_iff₀ (by linarith) (by linarith)]
    nlinarith
  -- the other side
  set r := (T + 8) * (1 / 2 ^ 20) + Xp * (1 / 2 ^ 17) with hr
  have hr0 : 0 ≤ r := by rw [hr]; positivity
  have hx : Real.exp (1 / 2 ^ 8 + (T + 8) * (1 / 2 ^ 20) + Xp * (1 / 2 ^ 17)) ≥
      1 + (1 / 2 ^ 8 + r) := by
    have := Real.add_one_le_exp (1 / 2 ^ 8 + (T + 8) * (1 / 2 ^ 20) + Xp * (1 / 2 ^ 17))
    rw [hr]; linarith
  have h2x : (1 + ub) * (2 - Real.exp (1 / 2 ^ 8 + (T + 8) * (1 / 2 ^ 20) + Xp * (1 / 2 ^ 17))) ≤ 1 - r := by
    have hub0 : (0 : ℝ) ≤ 1 + ub := by rw [hub]; norm_num
    have := mul_le_mul_of_nonneg_left (by linarith : 2 - Real.exp (1 / 2 ^ 8 + (T + 8) * (1 / 2 ^ 20) +
      Xp * (1 / 2 ^ 17)) ≤ 1 - (1 / 2 ^ 8 + r)) hub0
    rw [hub] at this ⊢
    nlinarith
  have hε : (0 : ℝ) ≤ 1 / 2 ^ 22 := by norm_num
  -- Λ
  set y := lam0 + lam1 * T with hy
  have hy0 : 0 ≤ y := by rw [hy, hl0, hl1]; positivity
  have hy1 : y ≤ 1 / 200 := by
    rw [hy, hl0, hl1]
    have h1 : 3 / 2 ^ 24 * T ≤ 3 / 2 ^ 24 * 4098 := mul_le_mul_of_nonneg_left hT (by norm_num)
    norm_num at h1 ⊢
    linarith
  have hΛ : expm1 y ≤ 101 / 100 * y := by
    have := exp_le_quad hy0 (by linarith)
    unfold expm1; nlinarith
  -- collect
  have hA : (1 + 1 / 2 ^ 22) * ((1 + expm1 φ) / (1 - expm1 φ)) * (1 + ub) *
      (2 - Real.exp (1 / 2 ^ 8 + (T + 8) * (1 / 2 ^ 20) + Xp * (1 / 2 ^ 17))) ≤
      (1 + 1 / 2 ^ 22) * ((1 + c) / (1 - c)) * (1 - r) := by
    have hpos : 0 ≤ (1 + 1 / 2 ^ 22) * ((1 + expm1 φ) / (1 - expm1 φ)) := by
      apply mul_nonneg (by norm_num); apply div_nonneg <;> linarith
    calc _ = (1 + 1 / 2 ^ 22) * ((1 + expm1 φ) / (1 - expm1 φ)) *
          ((1 + ub) * (2 - Real.exp (1 / 2 ^ 8 + (T + 8) * (1 / 2 ^ 20) + Xp * (1 / 2 ^ 17)))) := by ring
      _ ≤ (1 + 1 / 2 ^ 22) * ((1 + expm1 φ) / (1 - expm1 φ)) * (1 - r) :=
          mul_le_mul_of_nonneg_left h2x hpos
      _ ≤ (1 + 1 / 2 ^ 22) * ((1 + c) / (1 - c)) * (1 - r) := by
          have hr1 : 0 ≤ 1 - r := by
            rw [hr]; have := hT; have := hXp
            norm_num; nlinarith
          exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hψ (by norm_num)) hr1
  refine hA.trans ?_
  have hc1' : 0 < 1 - c := by linarith
  have hfin : (1 + 1 / 2 ^ 22) * (1 + c) * (1 - r) ≤ (1 - (expm1 y + 1 / 2 ^ 85)) * (1 - c) := by
    have hlin : 1 / 2 ^ 22 + c * (1 + 1 / 2 ^ 22) + c + 101 / 100 * y + 1 / 2 ^ 85 ≤ r := by
      rw [hc, hr, hφ, hy, he1, he2, hl0, hl1]
      norm_num
      linarith
    have hA1 : 1 ≤ (1 + 1 / 2 ^ 22) * (1 + c) := by nlinarith
    have h1 : (1 + 1 / 2 ^ 22) * (1 + c) * (1 - r) ≤ (1 + 1 / 2 ^ 22) * (1 + c) - r := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hA1) hr0]
    have hΛ0 : 0 ≤ expm1 y + 1 / 2 ^ 85 := by
      have := expm1_nonneg hy0; positivity
    have h2 : 1 - (expm1 y + 1 / 2 ^ 85) - c ≤ (1 - (expm1 y + 1 / 2 ^ 85)) * (1 - c) := by
      nlinarith [mul_nonneg hΛ0 hc0]
    nlinarith
  calc (1 + 1 / 2 ^ 22) * ((1 + c) / (1 - c)) * (1 - r)
      = (1 + 1 / 2 ^ 22) * (1 + c) * (1 - r) / (1 - c) := by field_simp
    _ ≤ _ := by rw [div_le_iff₀ hc1']; exact hfin

/-- **The numeric comparison.** For a row with `Tp` cached-part trips and `Te ≥ 1` extend trips
(`Tp + Te ≤ 2^12 + 2`), a score bound `0 ≤ X ≤ 2^12` and `BD ≤ 256`, the accumulator's exponent
`g2 + Tp · g1p + Te · g1e`, the ratio's error, the `float32` division (`2^-22`) is below the
claim's `errE`. -/
theorem coreErr_le (BD Tp Te : ℕ) (X γ : ℝ) (hBD : BD ≤ 256) (hX0 : 0 ≤ X) (hX : X ≤ 2 ^ 12)
    (hTe : 1 ≤ Te) (hT : Tp + Te ≤ 2 ^ 12 + 2) (hγ0 : 0 ≤ γ)
    (hγle : γ ≤ g2 + Tp * g1p + Te * g1e) :
    coreErr (dlt BD X) (X + dlt BD X) (Tp + Te) γ * (1 + 1 / 2 ^ 22) +
        1 / 2 ^ 22 ≤ errE BD Tp Te X := by
  have hBD' : (BD : ℝ) ≤ 256 := by exact_mod_cast hBD
  have hd0 : 0 ≤ dlt BD X := by unfold dlt; positivity
  have hd1 : dlt BD X ≤ 1 / 2 := by
    unfold dlt
    have h1 : ((BD : ℝ) + 8) * (1 / 2 ^ 22) * X ≤ (256 + 8) * (1 / 2 ^ 22) * 2 ^ 12 := by
      apply mul_le_mul (mul_le_mul_of_nonneg_right (by linarith) (by norm_num)) hX hX0 (by norm_num)
    norm_num at h1 ⊢
    linarith
  have hTr : ((Tp + Te : ℕ) : ℝ) ≤ 4098 := by exact_mod_cast (by omega : Tp + Te ≤ 4098)
  have hT1 : (1 : ℝ) ≤ ((Tp + Te : ℕ) : ℝ) := by exact_mod_cast (by omega : 1 ≤ Tp + Te)
  have hXp : X + dlt BD X ≤ 4097 := by norm_num at hX; linarith
  have hstar := core_star (T := ((Tp + Te : ℕ) : ℝ)) (Xp := X + dlt BD X) hT1 hTr (by linarith) hXp
  -- the claim's constants
  set x := 1 / 2 ^ 8 + (((Tp + Te : ℕ) : ℝ) + 8) * (1 / 2 ^ 20) + (X + dlt BD X) * (1 / 2 ^ 17) with hx
  have hx0 : 0 ≤ x := by rw [hx]; have := hd0; positivity
  have hx1 : x ≤ 1 / 20 := by
    rw [hx]
    have h1 : (X + dlt BD X) * (1 / 2 ^ 17) ≤ 4097 * (1 / 2 ^ 17) :=
      mul_le_mul_of_nonneg_right hXp (by norm_num)
    have h2 : (((Tp + Te : ℕ) : ℝ) + 8) * (1 / 2 ^ 20) ≤ (4098 + 8) * (1 / 2 ^ 20) :=
      mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
    norm_num at h1 h2 ⊢
    linarith
  have hrho : rho BD (Tp + Te) X = Real.exp x - 1 := rfl
  have hrho0 : 0 ≤ rho BD (Tp + Te) X := by rw [hrho]; have := Real.one_le_exp hx0; linarith
  have hrho1 : rho BD (Tp + Te) X < 1 := by
    rw [hrho]
    have := exp_le_quad hx0 (by linarith)
    nlinarith
  have hgam0 : 0 ≤ gam Tp Te := by
    unfold gam; have := Real.one_le_exp (x := (66 + 2 * (Tp : ℝ) + 72 * (Te : ℝ)) * (1 / 2 ^ 22))
      (by positivity); linarith
  -- the accumulator's exponent
  have hγ : g2 + Tp * g1p + Te * g1e ≤ (66 + 2 * (Tp : ℝ) + 72 * (Te : ℝ)) * (1 / 2 ^ 22) := by
    have : (1 : ℝ) ≤ Te := by exact_mod_cast hTe
    have hTp : (0 : ℝ) ≤ Tp := Nat.cast_nonneg Tp
    unfold g2 g1p g1e
    nlinarith
  have hΓ : expm1 γ ≤ gam Tp Te := expm1_mono (hγle.trans hγ)
  have hΓ0 : 0 ≤ expm1 γ := expm1_nonneg hγ0
  -- Λ, Φ
  set T : ℝ := ((Tp + Te : ℕ) : ℝ) with hTdef
  set φ := T * e1 + 2 * e2 * (X + dlt BD X) with hφ
  have hφ0 : 0 ≤ φ := by
    rw [hφ]; have := e1_pos; have := e2_pos; have := hd0; positivity
  have hφ1 : φ ≤ 1 / 200 := by
    rw [hφ]; unfold e1 e2
    have h1 : T * (257 / 2 ^ 30) ≤ 4098 * (257 / 2 ^ 30) := mul_le_mul_of_nonneg_right hTr (by norm_num)
    have h2 : 2 * (771 / 2 ^ 32) * (X + dlt BD X) ≤ 2 * (771 / 2 ^ 32) * 4097 :=
      mul_le_mul_of_nonneg_left hXp (by norm_num)
    norm_num at h1 h2 ⊢
    linarith
  have hΦ : expm1 φ ≤ 1 / 100 := by
    have := exp_le_quad hφ0 (by linarith)
    unfold expm1; nlinarith
  have hΦ0 : 0 ≤ expm1 φ := expm1_nonneg hφ0
  have hy0 : 0 ≤ lam0 + lam1 * T := by unfold lam0 lam1; have := hT1; positivity
  have hy1 : lam0 + lam1 * T ≤ 1 / 200 := by
    unfold lam0 lam1
    have h1 : 3 / 2 ^ 24 * T ≤ 3 / 2 ^ 24 * 4098 := mul_le_mul_of_nonneg_left hTr (by norm_num)
    norm_num at h1 ⊢
    linarith
  have hΛ0 : 0 ≤ expm1 (lam0 + lam1 * T) + 1 / 2 ^ 85 := by
    have := expm1_nonneg hy0; positivity
  have hΛ1 : expm1 (lam0 + lam1 * T) + 1 / 2 ^ 85 < 1 := by
    have := exp_le_quad hy0 (by linarith)
    unfold expm1; norm_num; nlinarith
  have hstar' : (1 + 1 / 2 ^ 22) * ((1 + expm1 φ) / (1 - expm1 φ)) * (1 + ub) * (1 - rho BD (Tp + Te) X) ≤
      1 - (expm1 (lam0 + lam1 * T) + 1 / 2 ^ 85) := by
    have h12 : 1 - rho BD (Tp + Te) X = 2 - Real.exp x := by rw [hrho]; ring
    rw [h12]; exact hstar
  have := core_alg (θ := Real.exp (2 * dlt BD X)) (ψ := (1 + expm1 φ) / (1 - expm1 φ))
    (ε := 1 / 2 ^ 22) (Λ := expm1 (lam0 + lam1 * T) + 1 / 2 ^ 85)
    (Γ := expm1 γ) (gam := gam Tp Te) (rho := rho BD (Tp + Te) X)
    (Real.one_le_exp (by linarith)) (by norm_num) (div_nonneg (by linarith) (by linarith)) hΛ1 hrho1
    hrho0 hΓ hΓ0 hstar'
  unfold coreErr errE
  exact this

end

end Claims.ExtendAttn
