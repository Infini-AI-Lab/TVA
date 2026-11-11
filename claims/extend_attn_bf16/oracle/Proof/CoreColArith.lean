import Proof.CoreLemmas

/-!
# The accumulator's arithmetic

`col_core`: the real inequalities of one trip of the accumulator of one output component.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

set_option maxHeartbeats 1600000 in
/-- **The accumulator's arithmetic.** `a` the old accumulator, `N` the old ghost sum
`Σ u (1+δ) v`, `R` the old `Σ u |v|`, `VS` the old `Σ |v|`; `NV`, `RV`, `VV` the same for the new
lanes; `P` the computed `Σ p16 · v` and `PA` its magnitude `Σ |p16| |v|`; `q` the rescaling of the old
weights and `r` the computed `re_scale`. -/
theorem col_core {a a' r q N R VS NV RV VV P PA Γ Γ' ε1 ε2 τ : ℝ}
    (hΓ : 0 ≤ Γ) (hΓ1 : Γ ≤ 1 / 6) (hε1 : 0 ≤ ε1) (hε1' : ε1 ≤ 1 / 2 ^ 15) (hε2 : 0 ≤ ε2)
    (hε2' : ε2 ≤ 1 / 2 ^ 15)
    (hq0 : 0 ≤ q) (_hq1 : q ≤ 1 + 1 / 2 ^ 20) (hr : |r| ≤ 1 + 1 / 2 ^ 19)
    (hR0 : 0 ≤ R) (hR1 : R ≤ 2 * VS) (hVS : 0 ≤ VS) (hRV0 : 0 ≤ RV) (hRV1 : RV ≤ 2 * VV)
    (hVV : 0 ≤ VV)
    (hrR : |r| * R ≤ (q + eps0) * R) (hrq : |r - q| * |N| ≤ eps0 * |N|)
    (hN : |N| ≤ (1 + ub) * R) (hcol : |a - N| ≤ Γ * R + τ * (1 + VS))
    (hT1 : |a' - (r * a + P)| ≤ ε1 * |r * a| + ε2 * PA + eps1 * (1 + VV))
    (hPA : PA ≤ (1 + ub) * RV + (etab + eps0 * (1 + ub)) * VV)
    (hP : |P - NV| ≤ (etab + eps0 * (1 + ub)) * VV)
    (hΓ' : (1 + ε1) * Γ + ε1 * (1 + ub) ≤ Γ') (hΓ'2 : ε2 * (1 + ub) ≤ Γ')
    (hτ0 : 0 ≤ τ) (hτ : τ ≤ 2 ^ 13 * t0) :
    |a' - (q * N + NV)| ≤ Γ' * (q * R + RV) + (τ + t0) * (1 + VS + VV) := by
  have hub : ub = 1 / 2 ^ 8 := rfl
  have heps0 : eps0 = 1 / 2 ^ 135 := rfl
  have heps1 : eps1 = 1 / 2 ^ 118 := rfl
  have hetab : etab = 1 / 2 ^ 134 := rfl
  have ht0 : t0 = 1 / 2 ^ 100 := rfl
  set D := |a - N| with hD
  have hD0 : 0 ≤ D := abs_nonneg _
  have hr0 : 0 ≤ |r| := abs_nonneg _
  have hN0 : 0 ≤ |N| := abs_nonneg _
  set c3 := etab + eps0 * (1 + ub) with hc3
  have hc30 : 0 ≤ c3 := by rw [hc3, hub, heps0, hetab]; norm_num
  -- triangle
  have e1 : a' - (q * N + NV) = (a' - (r * a + P)) + (r * (a - N) + (r - q) * N) + (P - NV) := by ring
  have h1 : |a' - (q * N + NV)| ≤ |a' - (r * a + P)| + (|r| * D + |r - q| * |N|) + |P - NV| := by
    rw [e1]
    have b1 := abs_add_le ((a' - (r * a + P)) + (r * (a - N) + (r - q) * N)) (P - NV)
    have b2 := abs_add_le (a' - (r * a + P)) (r * (a - N) + (r - q) * N)
    have b3 := abs_add_le (r * (a - N)) ((r - q) * N)
    rw [abs_mul, abs_mul] at b3
    linarith
  have h2 : |r * a| ≤ |r| * (|N| + D) := by
    rw [abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ hr0
    have := abs_add_le N (a - N)
    rw [add_sub_cancel] at this
    linarith
  -- the pieces
  have hrD : |r| * D ≤ Γ * ((q + eps0) * R) + (1 + 1 / 2 ^ 19) * (τ * (1 + VS)) := by
    calc |r| * D ≤ |r| * (Γ * R + τ * (1 + VS)) := mul_le_mul_of_nonneg_left hcol hr0
      _ = Γ * (|r| * R) + |r| * (τ * (1 + VS)) := by ring
      _ ≤ Γ * ((q + eps0) * R) + (1 + 1 / 2 ^ 19) * (τ * (1 + VS)) := by
        have := mul_le_mul_of_nonneg_left hrR hΓ
        have := mul_le_mul_of_nonneg_right hr (mul_nonneg hτ0 (by linarith : (0 : ℝ) ≤ 1 + VS))
        linarith
  have hrN : |r| * |N| ≤ (q + eps0) * ((1 + ub) * R) := by
    calc |r| * |N| ≤ |r| * ((1 + ub) * R) := mul_le_mul_of_nonneg_left hN hr0
      _ = (1 + ub) * (|r| * R) := by ring
      _ ≤ (1 + ub) * ((q + eps0) * R) := mul_le_mul_of_nonneg_left hrR (by rw [hub]; norm_num)
      _ = (q + eps0) * ((1 + ub) * R) := by ring
  have hT1' : |a' - (r * a + P)| ≤ ε1 * (|r| * (|N| + D)) + ε2 * PA + eps1 * (1 + VV) := by
    refine hT1.trans ?_
    have := mul_le_mul_of_nonneg_left h2 hε1
    linarith
  have hE : ε1 * (|r| * (|N| + D)) = ε1 * (|r| * |N|) + ε1 * (|r| * D) := by ring
  have hεPA : ε2 * PA ≤ ε2 * ((1 + ub) * RV) + ε2 * (c3 * VV) := by
    have := mul_le_mul_of_nonneg_left hPA hε2
    linarith
  have hε1D : ε1 * (|r| * D) ≤ ε1 * (Γ * ((q + eps0) * R) + (1 + 1 / 2 ^ 19) * (τ * (1 + VS))) :=
    mul_le_mul_of_nonneg_left hrD hε1
  have hε1N : ε1 * (|r| * |N|) ≤ ε1 * ((q + eps0) * ((1 + ub) * R)) :=
    mul_le_mul_of_nonneg_left hrN hε1
  have hqN : |r - q| * |N| ≤ eps0 * ((1 + ub) * R) := hrq.trans (mul_le_mul_of_nonneg_left hN (by rw [heps0]; norm_num))
  -- collect
  have hmain : |a' - (q * N + NV)| ≤
      ((1 + ε1) * Γ + ε1 * (1 + ub)) * (q * R) + ε2 * (1 + ub) * RV +
      ((1 + ε1) * (1 + 1 / 2 ^ 19) * (τ * (1 + VS)) +
        ((1 + ε1) * Γ * eps0 + ε1 * eps0 * (1 + ub) + eps0 * (1 + ub)) * R +
        ε2 * (c3 * VV) + eps1 * (1 + VV) + c3 * VV) := by
    nlinarith [h1, hT1', hE, hεPA, hε1D, hε1N, hqN, hP, hrD]
  -- the relative parts
  have hqR : 0 ≤ q * R := mul_nonneg hq0 hR0
  have hrel1 : ((1 + ε1) * Γ + ε1 * (1 + ub)) * (q * R) ≤ Γ' * (q * R) :=
    mul_le_mul_of_nonneg_right hΓ' hqR
  have hrel2 : ε2 * (1 + ub) * RV ≤ Γ' * RV := mul_le_mul_of_nonneg_right hΓ'2 hRV0
  -- the absolute parts
  have hk : (1 + ε1) * (1 + 1 / 2 ^ 19) ≤ 1 + 1 / 2 ^ 14 := by nlinarith
  have hτ1 : 0 ≤ τ * (1 + VS) := mul_nonneg hτ0 (by linarith)
  have habs1 : (1 + ε1) * (1 + 1 / 2 ^ 19) * (τ * (1 + VS)) ≤
      τ * (1 + VS) + t0 / 2 * (1 + VS) := by
    have h1 := mul_le_mul_of_nonneg_right hk hτ1
    have h2 : 1 / 2 ^ 14 * τ ≤ t0 / 2 := by
      rw [ht0]; rw [ht0] at hτ; nlinarith
    have h3 : 1 / 2 ^ 14 * (τ * (1 + VS)) ≤ t0 / 2 * (1 + VS) := by
      have := mul_le_mul_of_nonneg_right h2 (by linarith : (0 : ℝ) ≤ 1 + VS)
      linarith
    nlinarith
  have hcoef : ((1 + ε1) * Γ * eps0 + ε1 * eps0 * (1 + ub) + eps0 * (1 + ub)) ≤ 3 * eps0 := by
    rw [heps0, hub]
    have : (1 + ε1) * Γ ≤ 1 / 5 := by nlinarith
    nlinarith
  have habs2 : ((1 + ε1) * Γ * eps0 + ε1 * eps0 * (1 + ub) + eps0 * (1 + ub)) * R ≤ 6 * eps0 * VS := by
    have := mul_le_mul hcoef hR1 hR0 (by rw [heps0]; norm_num)
    nlinarith
  have hsmall : ε2 * (c3 * VV) + eps1 * (1 + VV) + c3 * VV ≤ t0 / 4 * (1 + VV) := by
    have hc3' : c3 ≤ 1 / 2 ^ 120 := by rw [hc3, hub, heps0, hetab]; norm_num
    have h1 : ε2 * (c3 * VV) ≤ 1 / 2 ^ 15 * (1 / 2 ^ 120 * VV) :=
      mul_le_mul hε2' (mul_le_mul_of_nonneg_right hc3' hVV) (mul_nonneg hc30 hVV) (by norm_num)
    have h2 : c3 * VV ≤ 1 / 2 ^ 120 * VV := mul_le_mul_of_nonneg_right hc3' hVV
    have h3 : ε2 * (c3 * VV) + eps1 * (1 + VV) + c3 * VV ≤
        (1 / 2 ^ 15 * (1 / 2 ^ 120) + 1 / 2 ^ 118 + 1 / 2 ^ 120) * (1 + VV) := by
      rw [heps1]; nlinarith
    refine h3.trans ?_
    rw [ht0]
    exact mul_le_mul_of_nonneg_right (by norm_num) (by linarith)
  have h6 : 6 * eps0 * VS ≤ t0 / 4 * VS := by
    have : 6 * eps0 ≤ t0 / 4 := by rw [heps0, ht0]; norm_num
    exact mul_le_mul_of_nonneg_right this hVS
  have ht0p : 0 ≤ t0 := by rw [ht0]; norm_num
  linarith only [ht0p, hVS, hVV, hmain, hrel1, hrel2, habs1, habs2, hsmall, h6, mul_nonneg ht0p hVV,
    mul_nonneg ht0p hVS, mul_nonneg hτ0 hVV]

end

end Claims.ExtendAttn
