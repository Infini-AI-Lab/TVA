import Proof.CoreRow
import Proof.CoreCol

/-!
# The output ratio

After all trips (`S = K`, every attended position seen) the row's ratio `a / l` is the
softmax-weighted mean of the values up to a relative error `coreErr`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- **Reweighting.** Weights `ω > 0` times factors `κ ∈ [lo, hi]` change the weighted mean of `v`
by at most `(hi/lo − 1)` times the weighted mean of `|v|`; and the weighted mean of `|v|` grows by
at most `hi/lo`. -/
theorem weighted_dev {K : Finset ℕ} (hK : K.Nonempty) {ω κ v : ℕ → ℝ} {lo hi : ℝ}
    (hω : ∀ j ∈ K, 0 < ω j) (hlo : 0 < lo) (hκ : ∀ j ∈ K, lo ≤ κ j ∧ κ j ≤ hi) :
    |(∑ j ∈ K, ω j * κ j * v j) / (∑ j ∈ K, ω j * κ j) - (∑ j ∈ K, ω j * v j) / ∑ j ∈ K, ω j| ≤
        (hi / lo - 1) * ((∑ j ∈ K, ω j * |v j|) / ∑ j ∈ K, ω j) ∧
      (∑ j ∈ K, ω j * κ j * |v j|) / (∑ j ∈ K, ω j * κ j) ≤
        hi / lo * ((∑ j ∈ K, ω j * |v j|) / ∑ j ∈ K, ω j) := by
  have hL0 : 0 < ∑ j ∈ K, ω j := Finset.sum_pos hω hK
  have hKlo : lo * ∑ j ∈ K, ω j ≤ ∑ j ∈ K, ω j * κ j := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by nlinarith [(hκ j hj).1, hω j hj]
  have hKhi : ∑ j ∈ K, ω j * κ j ≤ hi * ∑ j ∈ K, ω j := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by nlinarith [(hκ j hj).2, hω j hj]
  have hKk0 : 0 < ∑ j ∈ K, ω j * κ j := lt_of_lt_of_le (mul_pos hlo hL0) hKlo
  have hhilo : lo ≤ hi := by
    obtain ⟨j, hj⟩ := hK
    exact (hκ j hj).1.trans (hκ j hj).2
  generalize hLd : ∑ j ∈ K, ω j = L at *
  generalize hKd : ∑ j ∈ K, ω j * κ j = Kk at *
  have hA : 0 ≤ ∑ j ∈ K, ω j * |v j| := Finset.sum_nonneg fun j hj => mul_nonneg (hω j hj).le (abs_nonneg _)
  constructor
  · rw [div_sub_div _ _ hKk0.ne' hL0.ne', abs_div, abs_of_pos (mul_pos hKk0 hL0)]
    have e : (∑ j ∈ K, ω j * κ j * v j) * L - Kk * (∑ j ∈ K, ω j * v j) =
        ∑ j ∈ K, ω j * v j * (κ j * L - Kk) := by
      calc (∑ j ∈ K, ω j * κ j * v j) * L - Kk * (∑ j ∈ K, ω j * v j)
          = ∑ j ∈ K, (ω j * κ j * v j * L - ω j * v j * Kk) := by
            rw [Finset.sum_sub_distrib, Finset.sum_mul, Finset.mul_sum]
            congr 1
            exact Finset.sum_congr rfl fun j _ => by ring
        _ = _ := Finset.sum_congr rfl fun j _ => by ring
    rw [e, div_le_iff₀ (mul_pos hKk0 hL0)]
    have hb : ∀ j ∈ K, |κ j * L - Kk| ≤ (hi - lo) * L := by
      intro j hj
      rw [abs_le]
      constructor <;> nlinarith [mul_le_mul_of_nonneg_right (hκ j hj).1 hL0.le,
        mul_le_mul_of_nonneg_right (hκ j hj).2 hL0.le]
    calc |∑ j ∈ K, ω j * v j * (κ j * L - Kk)|
        ≤ ∑ j ∈ K, |ω j * v j * (κ j * L - Kk)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ K, ω j * |v j| * ((hi - lo) * L) := by
          refine Finset.sum_le_sum fun j hj => ?_
          rw [abs_mul, abs_mul, abs_of_pos (hω j hj)]
          exact mul_le_mul_of_nonneg_left (hb j hj) (mul_nonneg (hω j hj).le (abs_nonneg (v j)))
      _ = (∑ j ∈ K, ω j * |v j|) * ((hi - lo) * L) := by rw [Finset.sum_mul]
      _ ≤ (hi / lo - 1) * ((∑ j ∈ K, ω j * |v j|) / L) * (Kk * L) := by
          rw [show (hi / lo - 1) * ((∑ j ∈ K, ω j * |v j|) / L) * (Kk * L) =
            (hi / lo - 1) * (∑ j ∈ K, ω j * |v j|) * Kk by field_simp]
          have : (hi - lo) * L ≤ (hi / lo - 1) * Kk := by
            rw [show (hi / lo - 1) * Kk = (hi - lo) * Kk / lo by field_simp]
            rw [le_div_iff₀ hlo]; nlinarith
          nlinarith
  · rw [div_le_iff₀ hKk0]
    have h1 : ∑ j ∈ K, ω j * κ j * |v j| ≤ hi * ∑ j ∈ K, ω j * |v j| := by
      rw [Finset.mul_sum]
      exact Finset.sum_le_sum fun j hj => by nlinarith [mul_nonneg (mul_nonneg (hω j hj).le (abs_nonneg (v j))) (sub_nonneg.mpr (hκ j hj).2)]
    calc ∑ j ∈ K, ω j * κ j * |v j| ≤ hi * ∑ j ∈ K, ω j * |v j| := h1
      _ ≤ hi / lo * ((∑ j ∈ K, ω j * |v j|) / L) * Kk := by
          have : hi * L ≤ hi / lo * Kk := by
            rw [show hi / lo * Kk = hi * Kk / lo by ring, le_div_iff₀ hlo]; nlinarith
          rw [show hi / lo * ((∑ j ∈ K, ω j * |v j|) / L) * Kk =
            (hi / lo * Kk) * (∑ j ∈ K, ω j * |v j|) / L by ring, le_div_iff₀ hL0]
          nlinarith

/-- **The ratio's arithmetic**: `a/l` against `N/U` (the ghost ratio), then against `wr`. -/
theorem ratio_assembly {a l U N R VS Γ Λ τ wr wa ρ : ℝ} (hU : 49 / 50 ≤ U) (hΛ : Λ ≤ 1 / 2)
    (hΛ0 : 0 ≤ Λ) (hl : |l - U| ≤ Λ * U) (hN : |N| ≤ R) (hR0 : 0 ≤ R) (hΓ : 0 ≤ Γ)
    (ha : |a - N| ≤ (Γ + ub) * R + τ * (1 + VS)) (hτ0 : 0 ≤ τ) (hτ : τ ≤ 1 / 2 ^ 87)
    (hVS : 0 ≤ VS) (hRU : R ≤ U * (ρ * wa)) (hNU : |N / U - wr| ≤ (ρ - 1) * wa)
    (hρ1 : 1 ≤ ρ) (hwa : 0 ≤ wa) :
    1 / 4 ≤ l ∧ |a / l - wr| ≤ (ρ * (ub + Γ * (1 + ub) + Λ) / (1 - Λ) + (ρ - 1)) * wa +
      1 / 2 ^ 85 * (1 + VS) := by
  have hU0 : 0 < U := by linarith
  have hub : ub = 1 / 2 ^ 8 := rfl
  have hub0 : 0 ≤ ub := by rw [hub]; norm_num
  have hl1 : U * (1 - Λ) ≤ l := by
    have := (abs_le.mp hl).1
    nlinarith
  have hd : 0 < U * (1 - Λ) := mul_pos hU0 (by linarith)
  have hl0 : 0 < l := lt_of_lt_of_le hd hl1
  have hl14 : 1 / 4 ≤ l := by nlinarith
  refine ⟨hl14, ?_⟩
  set D1 := (Γ + ub + Λ) * R + τ * (1 + VS) with hD1
  have hD10 : 0 ≤ D1 := by
    have := mul_nonneg (by linarith : 0 ≤ Γ + ub + Λ) hR0
    have := mul_nonneg hτ0 (by linarith : (0 : ℝ) ≤ 1 + VS)
    linarith
  have key : |a * U - l * N| ≤ D1 * U := by
    have e : a * U - l * N = (a - N) * U - N * (l - U) := by ring
    rw [e]
    refine (abs_sub _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_pos hU0]
    have h1 := mul_le_mul_of_nonneg_right ha hU0.le
    have h2 : |N| * |l - U| ≤ R * (Λ * U) := mul_le_mul hN hl (abs_nonneg _) hR0
    rw [hD1]
    nlinarith
  have h2 : |a / l - N / U| ≤ D1 / (U * (1 - Λ)) := by
    rw [div_sub_div _ _ hl0.ne' hU0.ne', abs_div, abs_of_pos (mul_pos hl0 hU0),
      div_le_div_iff₀ (mul_pos hl0 hU0) hd]
    have h3 := mul_le_mul_of_nonneg_right key hd.le
    have h4 : D1 * U * (U * (1 - Λ)) ≤ D1 * (l * U) := by
      nlinarith [mul_le_mul_of_nonneg_left hl1 (mul_nonneg hD10 hU0.le)]
    linarith
  have h1 : |a / l - wr| ≤ |a / l - N / U| + |N / U - wr| := by
    have := abs_add_le (a / l - N / U) (N / U - wr)
    rw [show a / l - N / U + (N / U - wr) = a / l - wr by ring] at this
    exact this
  have h5 : D1 / (U * (1 - Λ)) ≤ ρ * (ub + Γ * (1 + ub) + Λ) / (1 - Λ) * wa + 1 / 2 ^ 85 * (1 + VS) := by
    rw [div_le_iff₀ hd]
    have hne : (1 - Λ) ≠ 0 := by intro h; linarith
    have hρ0 : 0 ≤ ρ := by linarith
    have h6 : (Γ + ub + Λ) * R ≤ ρ * (ub + Γ * (1 + ub) + Λ) * wa * U := by
      have a1 : (Γ + ub + Λ) * R ≤ (Γ + ub + Λ) * (U * (ρ * wa)) :=
        mul_le_mul_of_nonneg_left hRU (by linarith)
      have a2 : (Γ + ub + Λ) ≤ (ub + Γ * (1 + ub) + Λ) := by nlinarith [mul_nonneg hΓ hub0]
      have a3 := mul_le_mul_of_nonneg_right a2 (mul_nonneg hU0.le (mul_nonneg hρ0 hwa))
      nlinarith
    have h7 : τ * (1 + VS) ≤ 1 / 2 ^ 85 * (1 + VS) * (U * (1 - Λ)) := by
      have : (1 : ℝ) / 2 ^ 87 ≤ 1 / 2 ^ 85 * (U * (1 - Λ)) := by
        have : (49 : ℝ) / 100 ≤ U * (1 - Λ) := by nlinarith
        nlinarith
      nlinarith [mul_le_mul_of_nonneg_right (hτ.trans this) (by linarith : (0 : ℝ) ≤ 1 + VS)]
    calc D1 = (Γ + ub + Λ) * R + τ * (1 + VS) := hD1
      _ ≤ ρ * (ub + Γ * (1 + ub) + Λ) * wa * U + 1 / 2 ^ 85 * (1 + VS) * (U * (1 - Λ)) := by linarith
      _ = (ρ * (ub + Γ * (1 + ub) + Λ) / (1 - Λ) * wa + 1 / 2 ^ 85 * (1 + VS)) * (U * (1 - Λ)) := by
        field_simp
  linarith

set_option maxHeartbeats 1600000 in
/-- **The ratio.** After `t ≤ 2^13` trips that saw the nonempty set `K` of at most `2^18`
positions, `deno` is positive and `acc / deno` is within `coreErr · wabs` plus a tiny absolute
error of `wref`. -/
theorem ratio_bound {sc v : ℕ → ℝ} {dl Xp γ m l a : ℝ} {t : ℕ} {K : Finset ℕ} {G : Ghost}
    (hdl : 0 ≤ dl) (_hdl2 : dl ≤ 1 / 2) (hXp : 0 ≤ Xp) (hXp2 : Xp ≤ 2 ^ 13) (hγ0 : 0 ≤ γ)
    (_hγ : γ ≤ 1 / 2 ^ 3) (ht : t ≤ tmax) (hK : K.Nonempty) (_hcard : K.card ≤ nmax)
    (hrow : RowInv sc dl Xp t K m l G) (hcol : ColInv v γ t K G a) :
    1 / 4 ≤ l ∧ |a / l - wref K sc v| ≤ coreErr dl Xp t γ * wabs K sc v +
      1 / 2 ^ 85 * (1 + ∑ j ∈ K, |v j|) := by
  have hrow0 := hrow
  obtain ⟨hL, hM, hE, hl⟩ := hrow
  obtain ⟨hle, ⟨j0, hj0, hjm⟩, hF⟩ := hM hK
  choose! F hF1 hFu using hF
  have htr : (t : ℝ) ≤ 2 ^ 13 := t_le_real ht
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg t
  -- the factors' exponent
  set Φ := expm1 (t * e1 + 2 * e2 * Xp) with hΦ
  have hexp : t * e1 + 2 * e2 * Xp ≤ 1 / 100 := by
    have := expo_small (t := t) (y := 2 * Xp) htr ht0 (by norm_num at hXp2 ⊢; linarith) (by linarith)
    linarith
  have hexp0 : 0 ≤ t * e1 + 2 * e2 * Xp := by
    have := e1_pos; have := e2_pos; positivity
  have hΦ0 : 0 ≤ Φ := expm1_nonneg hexp0
  have hΦ1 : Φ ≤ 1 / 50 := by
    have := exp_le_quad hexp0 (by linarith)
    rw [hΦ]; unfold expm1
    nlinarith
  have hFj : ∀ j ∈ K, 1 - Φ ≤ F j ∧ F j ≤ 1 + Φ := by
    intro j hj
    have hgap : m - G.s j ≤ 2 * Xp := by
      have h1 := abs_le.mp (hL j0 hj0).2.1
      have h2 := abs_le.mp (hL j hj).2.1
      rw [← hjm]; linarith [h1.1, h1.2, h2.1, h2.2]
    have hb : |F j - 1| ≤ Φ := (hF1 j hj).trans (expm1_mono (by
      have := mul_le_mul_of_nonneg_left hgap e2_pos.le
      linarith))
    have := abs_le.mp hb
    constructor <;> linarith [this.1, this.2]
  have hu_pos : ∀ j ∈ K, 0 < G.u j := by
    intro j hj; rw [hFu j hj]
    exact mul_pos (Real.exp_pos _) (by linarith [(hFj j hj).1])
  -- the ghost weights are `exp (−m) · ω · κ`
  set ω : ℕ → ℝ := fun j => Real.exp (sc j) with hω
  set κ : ℕ → ℝ := fun j => Real.exp (G.s j - sc j) * F j with hκdef
  have hu_eq : ∀ j ∈ K, G.u j = Real.exp (-m) * (ω j * κ j) := by
    intro j hj
    rw [hFu j hj]
    simp only [hω, hκdef]
    rw [show G.s j - m = -m + sc j + (G.s j - sc j) by ring, Real.exp_add, Real.exp_add]
    ring
  have hlo : 0 < Real.exp (-dl) * (1 - Φ) := mul_pos (Real.exp_pos _) (by linarith)
  have hκ : ∀ j ∈ K, Real.exp (-dl) * (1 - Φ) ≤ κ j ∧ κ j ≤ Real.exp dl * (1 + Φ) := by
    intro j hj
    have h1 := abs_le.mp (hL j hj).1
    have hF' := hFj j hj
    have e1' : Real.exp (-dl) ≤ Real.exp (G.s j - sc j) := Real.exp_le_exp.mpr (by linarith [h1.1, h1.2])
    have e2' : Real.exp (G.s j - sc j) ≤ Real.exp dl := Real.exp_le_exp.mpr (by linarith [h1.1, h1.2])
    simp only [hκdef]
    constructor
    · exact mul_le_mul e1' hF'.1 (by linarith) (Real.exp_pos _).le
    · exact mul_le_mul e2' hF'.2 (by linarith) (Real.exp_pos _).le
  obtain ⟨hdev1, hdev2⟩ := weighted_dev hK (ω := ω) (κ := κ) (v := v)
    (fun j _ => Real.exp_pos _) hlo hκ
  set ρ := Real.exp (2 * dl) * ((1 + Φ) / (1 - Φ)) with hρdef
  have hρ : Real.exp dl * (1 + Φ) / (Real.exp (-dl) * (1 - Φ)) = ρ := by
    have h1 : Real.exp (2 * dl) = Real.exp dl / Real.exp (-dl) := by
      rw [← Real.exp_sub]; congr 1; ring
    have h2 : (1 - Φ) ≠ 0 := by linarith
    have h3 : Real.exp (-dl) ≠ 0 := (Real.exp_pos _).ne'
    rw [hρdef, h1]
    field_simp
  have hρ1 : 1 ≤ ρ := by
    rw [← hρ, le_div_iff₀ hlo]
    have e1' : Real.exp (-dl) ≤ Real.exp dl := Real.exp_le_exp.mpr (by linarith)
    nlinarith [mul_le_mul e1' (by linarith : 1 - Φ ≤ 1 + Φ) (by linarith) (Real.exp_pos _).le]
  -- the sums
  have hUs : ∑ j ∈ K, G.u j = Real.exp (-m) * ∑ j ∈ K, ω j * κ j := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j hj => hu_eq j hj
  have hNs : ∑ j ∈ K, G.u j * v j = Real.exp (-m) * ∑ j ∈ K, ω j * κ j * v j := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j hj => by rw [hu_eq j hj]; ring
  have hRs : ∑ j ∈ K, G.u j * |v j| = Real.exp (-m) * ∑ j ∈ K, ω j * κ j * |v j| := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j hj => by rw [hu_eq j hj]; ring
  have hKk0 : 0 < ∑ j ∈ K, ω j * κ j := by
    refine Finset.sum_pos (fun j hj => mul_pos (Real.exp_pos _) ?_) hK
    exact lt_of_lt_of_le hlo (hκ j hj).1
  have hU0 : 0 < ∑ j ∈ K, G.u j := by rw [hUs]; exact mul_pos (Real.exp_pos _) hKk0
  have hU1 : 49 / 50 ≤ ∑ j ∈ K, G.u j := by
    have h1 : G.u j0 ≤ ∑ j ∈ K, G.u j :=
      Finset.single_le_sum (f := G.u) (fun j hj => (hu_pos j hj).le) hj0
    have h2 : G.u j0 = F j0 := by rw [hFu j0 hj0, hjm]; simp
    have := (hFj j0 hj0).1
    linarith
  have hN_ratio : (∑ j ∈ K, G.u j * v j) / ∑ j ∈ K, G.u j =
      (∑ j ∈ K, ω j * κ j * v j) / ∑ j ∈ K, ω j * κ j := by
    rw [hNs, hUs, mul_div_mul_left _ _ (Real.exp_pos _).ne']
  have hR_le : ∑ j ∈ K, G.u j * |v j| ≤ (∑ j ∈ K, G.u j) * (ρ * wabs K sc v) := by
    have h := hdev2
    rw [div_le_iff₀ hKk0, hρ] at h
    rw [hRs, hUs]
    have : ∑ j ∈ K, ω j * κ j * |v j| ≤ (∑ j ∈ K, ω j * κ j) * (ρ * wabs K sc v) := by
      unfold wabs
      calc _ ≤ ρ * ((∑ j ∈ K, ω j * |v j|) / ∑ j ∈ K, ω j) * ∑ j ∈ K, ω j * κ j := h
        _ = _ := by ring
    nlinarith [Real.exp_pos (-m)]
  have hwa0 : 0 ≤ wabs K sc v := by
    unfold wabs
    exact div_nonneg (Finset.sum_nonneg fun j _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _))
      (Finset.sum_pos (fun j _ => Real.exp_pos _) hK).le
  have hNU : |(∑ j ∈ K, G.u j * v j) / ∑ j ∈ K, G.u j - wref K sc v| ≤ (ρ - 1) * wabs K sc v := by
    rw [hN_ratio]
    have := hdev1
    rw [hρ] at this
    exact this
  -- the sum of the weights, the accumulator
  set U := ∑ j ∈ K, G.u j with hUdef
  set R := ∑ j ∈ K, G.u j * |v j| with hRdef
  have hR0 : 0 ≤ R := Finset.sum_nonneg fun j hj => mul_nonneg (hu_pos j hj).le (abs_nonneg _)
  have hNR : |∑ j ∈ K, G.u j * v j| ≤ R := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j hj => ?_)
    rw [abs_mul, abs_of_pos (hu_pos j hj)]
  have hΛ0 : 0 ≤ expm1 (lam0 + lam1 * t) :=
    expm1_nonneg (by have := ht0; unfold lam0 lam1; positivity)
  have hΛ1 : expm1 (lam0 + lam1 * t) ≤ 1 / 100 := by
    have hx : lam0 + lam1 * t ≤ 1 / 2 ^ 9 := by
      have h2 : lam1 * t ≤ lam1 * 2 ^ 13 := mul_le_mul_of_nonneg_left htr (by unfold lam1; norm_num)
      unfold lam0 lam1 at *; norm_num at h2 ⊢; linarith
    have := exp_le_quad (x := lam0 + lam1 * t) (by have := ht0; unfold lam0 lam1; positivity) (by linarith)
    have h3 : (lam0 + lam1 * t) ^ 2 ≤ (1 / 2 ^ 9) ^ 2 :=
      pow_le_pow_left₀ (by have := ht0; unfold lam0 lam1; positivity) hx 2
    unfold expm1
    norm_num at h3 hx this ⊢
    nlinarith
  have hta : (t : ℝ) * a0 ≤ 1 / 2 ^ 87 := by
    have := mul_le_mul_of_nonneg_right htr (by unfold a0; norm_num : (0 : ℝ) ≤ a0)
    unfold a0 at this ⊢; norm_num at this ⊢; linarith
  have hl' : |l - U| ≤ (expm1 (lam0 + lam1 * t) + 1 / 2 ^ 85) * U := by
    refine hl.trans ?_
    have : (t : ℝ) * a0 ≤ 1 / 2 ^ 85 * U := by
      have : (1 : ℝ) / 2 ^ 87 ≤ 1 / 2 ^ 85 * U := by norm_num; linarith
      linarith
    linarith
  have hτ : (t : ℝ) * t0 ≤ 1 / 2 ^ 87 := by
    have := mul_le_mul_of_nonneg_right htr (by unfold t0; norm_num : (0 : ℝ) ≤ t0)
    unfold t0 at this ⊢; norm_num at this ⊢; linarith
  have hΓ : 0 ≤ expm1 γ := expm1_nonneg hγ0
  have ha : |a - ∑ j ∈ K, G.u j * v j| ≤ (expm1 γ + ub) * R + (t : ℝ) * t0 * (1 + ∑ j ∈ K, |v j|) := by
    unfold ColInv at hcol
    have h1 : |∑ j ∈ K, G.u j * (1 + G.δ j) * v j - ∑ j ∈ K, G.u j * v j| ≤ ub * R := by
      rw [← Finset.sum_sub_distrib, hRdef, Finset.mul_sum]
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j hj => ?_)
      have e : G.u j * (1 + G.δ j) * v j - G.u j * v j = G.u j * G.δ j * v j := by ring
      rw [e, abs_mul, abs_mul, abs_of_pos (hu_pos j hj)]
      have := (hL j hj).2.2
      have := mul_le_mul_of_nonneg_left this (hu_pos j hj).le
      nlinarith [abs_nonneg (v j)]
    have := abs_add_le (a - ∑ j ∈ K, G.u j * (1 + G.δ j) * v j)
      (∑ j ∈ K, G.u j * (1 + G.δ j) * v j - ∑ j ∈ K, G.u j * v j)
    rw [show a - ∑ j ∈ K, G.u j * (1 + G.δ j) * v j +
      (∑ j ∈ K, G.u j * (1 + G.δ j) * v j - ∑ j ∈ K, G.u j * v j) = a - ∑ j ∈ K, G.u j * v j by ring] at this
    have hcol' : |a - ∑ j ∈ K, G.u j * (1 + G.δ j) * v j| ≤
        expm1 γ * R + (t : ℝ) * t0 * (1 + ∑ j ∈ K, |v j|) := hcol
    linarith
  have hVS : 0 ≤ ∑ j ∈ K, |v j| := Finset.sum_nonneg fun j _ => abs_nonneg _
  have hass := ratio_assembly (a := a) (l := l) (U := U) (N := ∑ j ∈ K, G.u j * v j) (R := R)
    (VS := ∑ j ∈ K, |v j|) (Γ := expm1 γ) (Λ := expm1 (lam0 + lam1 * t) + 1 / 2 ^ 85)
    (τ := (t : ℝ) * t0) (wr := wref K sc v) (wa := wabs K sc v) (ρ := ρ) hU1
    (by norm_num at *; linarith) (by positivity) hl' hNR hR0 hΓ ha
    (mul_nonneg ht0 (by unfold t0; norm_num)) hτ hVS hR_le hNU hρ1 hwa0
  refine ⟨hass.1, hass.2.trans (le_of_eq ?_)⟩
  unfold coreErr
  rw [hρdef, hΦ]

end

end Claims.ExtendAttn
