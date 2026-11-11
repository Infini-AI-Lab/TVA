import Proof.Num

/-!
# The end of the accuracy proof: magnitudes, the final ratio, the output's tolerance
-/
namespace Claims.PagedAttention

namespace Num

open Finset

/-- `exp z ≤ 1 + z + 0.52 z²` for `0 ≤ z ≤ 1/20`. -/
theorem exp_le_quad {z : ℝ} (h0 : 0 ≤ z) (h1 : z ≤ 1 / 20) :
    Real.exp z ≤ 1 + z + (52 / 100) * z ^ 2 := by
  have hb := Real.exp_bound (x := z) (by rw [abs_of_nonneg h0]; linarith) (n := 3) (by norm_num)
  rw [abs_of_nonneg h0] at hb
  have hs : (∑ m ∈ Finset.range 3, z ^ m / (m.factorial : ℝ)) = 1 + z + z ^ 2 / 2 := by
    simp [Finset.sum_range_succ, Nat.factorial]
  rw [hs] at hb
  have := (abs_le.mp hb).2
  norm_num [Nat.factorial] at this
  nlinarith [mul_nonneg h0 (sq_nonneg z)]

/-- `lS` is `aS` with all values `1`, and `mS` with values `1` is `lS`. -/
theorem lS_eq (S : ℕ → ℝ) (μ : ℝ) (N : ℕ) :
    aS S (fun _ => 1) μ N = lS S μ N ∧ mS S (fun _ => 1) μ N = lS S μ N := by
  unfold aS mS lS; simp

/-- Some token with score at least `μ − Δ` makes `lS ≥ exp (−Δ)`. -/
theorem lS_ge (S : ℕ → ℝ) {μ Δ : ℝ} {N k : ℕ} (hk : k < N) (h : μ ≤ S k + Δ) :
    Real.exp (-Δ) ≤ lS S μ N := by
  unfold lS
  calc Real.exp (-Δ) ≤ Real.exp (S k - μ) := Real.exp_le_exp.mpr (by linarith)
    _ ≤ _ := Finset.single_le_sum (f := fun k => Real.exp (S k - μ))
        (fun _ _ => (Real.exp_pos _).le) (Finset.mem_range.mpr hk)

/-- The weighted sums at any reference are the reference-free ones, up to the factor `e^(−μ)`. -/
theorem aS_eq_exp (S V : ℕ → ℝ) (μ : ℝ) (n : ℕ) :
    aS S V μ n = Real.exp (-μ) * ∑ k ∈ range n, Real.exp (S k) * V k := by
  unfold aS
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [sub_eq_add_neg, Real.exp_add]; ring

/-- See `aS_eq_exp`. -/
theorem lS_eq_exp (S : ℕ → ℝ) (μ : ℝ) (n : ℕ) :
    lS S μ n = Real.exp (-μ) * ∑ k ∈ range n, Real.exp (S k) := by
  unfold lS
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [sub_eq_add_neg, Real.exp_add]; ring

/-- The exact ratio is the claim's attention, whatever the reference:
`aS / lS = (Σ exp (S k)·V k) / Σ exp (S k)`. -/
theorem aS_div_lS (S V : ℕ → ℝ) (μ : ℝ) (n : ℕ) :
    aS S V μ n / lS S μ n =
      (∑ k ∈ range n, Real.exp (S k) * V k) / ∑ k ∈ range n, Real.exp (S k) := by
  rw [aS_eq_exp, lS_eq_exp]
  exact mul_div_mul_left _ _ (Real.exp_pos _).ne'

/-- The same for the weighted mean of `|V|`. -/
theorem mS_div_lS (S V : ℕ → ℝ) (μ : ℝ) (n : ℕ) :
    mS S V μ n / lS S μ n =
      (∑ k ∈ range n, Real.exp (S k) * |V k|) / ∑ k ∈ range n, Real.exp (S k) := by
  rw [mS_eq, aS_eq_exp, lS_eq_exp]
  exact mul_div_mul_left _ _ (Real.exp_pos _).ne'

/-- The claim's tolerance, `tolA T X`, as a real. -/
noncomputable def tolR (T : ℕ) (X : ℝ) : ℝ := (9 * (T : ℝ) / 8 + 8 * X + 8) / 2 ^ 21

/-- What the ratio `a / l` may be off by, relative to `Λ + |R|`, so that the division's and the
`float16` rounding's errors still fit in `tolR`. -/
noncomputable def tauMax (T : ℕ) (X : ℝ) : ℝ := (tolR T X - 1 / 2 ^ 21) / (1 + 1 / 2 ^ 10)

/-- `0 ≤ tauMax T X ≤ 1/8` on the covered range. -/
theorem tauMax_range {X : ℝ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) {T : ℕ} (hT2 : T ≤ 2 ^ 16) :
    0 ≤ tauMax T X ∧ tauMax T X ≤ 1 / 8 := by
  have hT : (T : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hT2
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
  unfold tauMax tolR
  constructor
  · apply div_nonneg _ (by norm_num)
    have : (1 : ℝ) / 2 ^ 21 ≤ (9 * (T : ℝ) / 8 + 8 * X + 8) / 2 ^ 21 := by
      apply div_le_div_of_nonneg_right _ (by norm_num); linarith
    linarith
  · rw [div_le_iff₀ (by norm_num)]
    have : (9 * (T : ℝ) / 8 + 8 * X + 8) / 2 ^ 21 ≤ 1 / 20 := by
      rw [div_le_iff₀ (by norm_num)]; norm_num; linarith
    norm_num at this ⊢; linarith

/-- **The output.** `w` within `τ·(Λ + |R|) + 2^-30` of `R`, the division `q` of `w` within
`2^-22·|w| + 2^-148`, and the `float16` rounding `c` of `q` within `2^-11·|q| + 2^-25`: then `c`
is within `(τ·(1 + 2^-10) + 2^-21)·(Λ + |R|) + 2^-11·|R| + 2^-24` of `R`. -/
theorem out_tol {R w q c τ Λ : ℝ} (hΛ : |R| ≤ Λ) (_hΛ' : Λ ≤ 2 ^ 15) (hτ : 0 ≤ τ) (_hτ1 : τ ≤ 1 / 8)
    (hw : |w - R| ≤ τ * (Λ + |R|) + 1 / 2 ^ 30) (hq : |q - w| ≤ 1 / 2 ^ 22 * |w| + 1 / 2 ^ 148)
    (hc : |c - q| ≤ 1 / 2 ^ 11 * |q| + 1 / 2 ^ 25) :
    |c - R| ≤ (τ * (1 + 1 / 2 ^ 10) + 1 / 2 ^ 21) * (Λ + |R|) + 1 / 2 ^ 11 * |R| + 1 / 2 ^ 24 := by
  have h1 : |c - R| ≤ |c - q| + |q - w| + |w - R| := by
    have := abs_add_le (c - q) (q - w)
    have h2 := abs_add_le (c - q + (q - w)) (w - R)
    rw [show c - q + (q - w) = c - w by ring, show c - w + (w - R) = c - R by ring] at h2
    rw [show c - q + (q - w) = c - w by ring] at this
    linarith
  have hqw : |q| ≤ |w| + |q - w| := by
    have := abs_sub_abs_le_abs_sub q w
    linarith
  have hwR : |w| ≤ |R| + |w - R| := by
    have := abs_sub_abs_le_abs_sub w R
    linarith
  have hR0 := abs_nonneg R
  have hε0 := abs_nonneg (w - R)
  have hτΛ : 0 ≤ τ * (Λ + |R|) := mul_nonneg hτ (by linarith)
  have e2 : (τ * (1 + 1 / 2 ^ 10) + 1 / 2 ^ 21) * (Λ + |R|) =
      τ * (Λ + |R|) * (1 + 1 / 2 ^ 10) + 1 / 2 ^ 21 * (Λ + |R|) := by ring
  rw [e2]
  nlinarith [abs_nonneg q, abs_nonneg w, abs_nonneg (q - w), abs_nonneg (c - q)]


/-- `Ex` on the covered range is `1` plus about `z`, where `z` is the largest exponent. -/
theorem Ex_quad {X e1 μ : ℝ} {T : ℕ} (_hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (he1 : 0 ≤ e1)
    (he1' : e1 ≤ eA) (hT2 : T ≤ 2 ^ 16) (hμ : μ ≤ Kx X) (hμ' : -(Kx X) ≤ μ) :
    Ex e1 gam (Kx X) T μ ≤ 1 + (1017 / 1000) * ((T : ℝ) * e1 + 2 * gam * (X + 1)) := by
  unfold Ex
  have hu := u_pos
  have hT : (T : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hT2
  have hgam : 0 ≤ gam := by unfold gam; positivity
  have hz0 : 0 ≤ (T : ℝ) * e1 + gam * (μ + Kx X) :=
    add_nonneg (mul_nonneg (Nat.cast_nonneg T) he1) (mul_nonneg hgam (by linarith))
  have hz1 : (T : ℝ) * e1 + gam * (μ + Kx X) ≤ (T : ℝ) * e1 + 2 * gam * (X + 1) := by
    unfold Kx at *
    have : gam * (μ + (X + 1)) ≤ gam * (2 * (X + 1)) := mul_le_mul_of_nonneg_left (by linarith) hgam
    linarith
  have hz2 : (T : ℝ) * e1 + 2 * gam * (X + 1) ≤ 323 / 10000 := by
    have h1 : (T : ℝ) * e1 ≤ 2 ^ 16 * eA := mul_le_mul hT he1' he1 (by positivity)
    have h2 : 2 * gam * (X + 1) ≤ 2 * gam * 1025 := by
      apply mul_le_mul_of_nonneg_left _ (by positivity); linarith
    unfold eA gam u at *; norm_num at *; linarith
  set z := (T : ℝ) * e1 + gam * (μ + Kx X) with hz
  have := exp_le_quad hz0 (by linarith)
  have h3 : (52 / 100) * z ^ 2 ≤ (52 / 100) * (323 / 10000) * z := by
    have : z ^ 2 ≤ (323 / 10000) * z := by nlinarith
    nlinarith
  nlinarith

/-- The error coefficient on the covered range, linearly bounded. -/
theorem Fc_lin {X e1 ε2 μ : ℝ} {T : ℕ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (he1 : 0 ≤ e1)
    (he1' : e1 ≤ eA) (hε2 : 0 ≤ ε2) (hε2' : ε2 ≤ 31 / 10000) (hT2 : T ≤ 2 ^ 16)
    (hμ : μ ≤ Kx X) (hμ' : -(Kx X) ≤ μ) :
    Fc e1 gam ε2 (Kx X) T μ ≤ ε2 + (1021 / 1000) * ((T : ℝ) * e1 + 2 * gam * (X + 1)) := by
  have hu := u_pos
  have hE := Ex_quad hX0 hX1 he1 he1' hT2 hμ hμ'
  have hgam : 0 ≤ gam := by unfold gam; positivity
  have hz0 : 0 ≤ (T : ℝ) * e1 + 2 * gam * (X + 1) :=
    add_nonneg (mul_nonneg (Nat.cast_nonneg T) he1) (mul_nonneg (by linarith) (by linarith))
  unfold Fc
  have h1 : (1 + ε2) * Ex e1 gam (Kx X) T μ ≤ (1 + ε2) * (1 + (1017 / 1000) * ((T : ℝ) * e1 + 2 * gam * (X + 1))) :=
    mul_le_mul_of_nonneg_left hE (by linarith)
  set z := (T : ℝ) * e1 + 2 * gam * (X + 1)
  have h2 : ε2 * ((1017 / 1000) * z) ≤ (31 / 10000) * ((1017 / 1000) * z) :=
    mul_le_mul_of_nonneg_right hε2' (mul_nonneg (by norm_num) hz0)
  nlinarith

/-- `tauMax ≤ 4/100` on the covered range. -/
theorem tauMax_le {X : ℝ} (_hX0 : 0 ≤ X) (hX1 : X ≤ 1024) {T : ℕ} (hT2 : T ≤ 2 ^ 16) :
    tauMax T X ≤ 4 / 100 := by
  have hT : (T : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hT2
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
  unfold tauMax tolR
  rw [div_le_iff₀ (by norm_num)]
  norm_num
  rw [div_le_iff₀ (by norm_num)] at *
  nlinarith

/-- **The key numbers.** On the covered range, the error coefficients of `acc` and of `l`, after
`T` trips, are at most `tauMax T X·(1 − Fc_l − 2^-62)`: the invariant's error fits the claim's
tolerance, with room for the division and the rounding to `float16`. -/
theorem key_ineq {X μ : ℝ} {T : ℕ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (hT1 : 1 ≤ T) (hT2 : T ≤ 2 ^ 16)
    (hμ : μ ≤ Kx X) (hμ' : -(Kx X) ≤ μ) :
    Fc eA gam (eps2A X) (Kx X) T μ ≤
        tauMax T X * (1 - Fc eL gam (eps2L X) (Kx X) T μ - 1 / 2 ^ 62) ∧
      Fc eL gam (eps2L X) (Kx X) T μ ≤
        tauMax T X * (1 - Fc eL gam (eps2L X) (Kx X) T μ - 1 / 2 ^ 62) := by
  have hu := u_pos
  have hT : (T : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hT2
  have hT1' : (1 : ℝ) ≤ T := by exact_mod_cast hT1
  have heL : eL ≤ eA := by unfold eL eA; nlinarith
  have heL0 : 0 ≤ eL := by unfold eL; positivity
  have heA0 : 0 ≤ eA := by unfold eA; positivity
  have h48 : 48 * u * X ≤ 48 * u * 1024 := by nlinarith
  have hε2A : eps2A X ≤ 31 / 10000 := by unfold eps2A u at *; norm_num at *; linarith
  have hε2L : eps2L X ≤ 31 / 10000 := by unfold eps2L u at *; norm_num at *; linarith
  have hA := Fc_lin hX0 hX1 heA0 le_rfl (by unfold eps2A; positivity) hε2A hT2 hμ hμ'
  have hL := Fc_lin hX0 hX1 heL0 heL (by unfold eps2L; positivity) hε2L hT2 hμ hμ'
  have hLnn : 0 ≤ Fc eL gam (eps2L X) (Kx X) T μ := by
    unfold Fc
    have := Ex_ge_one heL0 (show 0 ≤ gam by unfold gam; positivity) T hμ'
    have := mul_le_mul_of_nonneg_left this (show 0 ≤ 1 + eps2L X by unfold eps2L; positivity)
    unfold eps2L at *; nlinarith
  have hτ := tauMax_range hX0 hX1 hT2
  have hτ4 := tauMax_le hX0 hX1 hT2
  have hτeq : tauMax T X * (1 + 1 / 2 ^ 10) = (9 * (T : ℝ) / 8 + 8 * X + 8) / 2 ^ 21 - 1 / 2 ^ 21 := by
    unfold tauMax tolR; field_simp
  -- the linear inequality
  have hmain : (eps2A X + (1021 / 1000) * ((T : ℝ) * eA + 2 * gam * (X + 1)) +
        (4 / 100) * (eps2L X + (1021 / 1000) * ((T : ℝ) * eL + 2 * gam * (X + 1)) + 1 / 2 ^ 62)) *
        (1 + 1 / 2 ^ 10) ≤ (9 * (T : ℝ) / 8 + 8 * X + 8) / 2 ^ 21 - 1 / 2 ^ 21 := by
    unfold eps2A eps2L eA eL gam u; norm_num; nlinarith
  have hLb : tauMax T X * Fc eL gam (eps2L X) (Kx X) T μ ≤
      (4 / 100) * (eps2L X + (1021 / 1000) * ((T : ℝ) * eL + 2 * gam * (X + 1))) := by
    have := mul_le_mul hτ4 hL hLnn (by norm_num)
    exact this
  have hkey : Fc eA gam (eps2A X) (Kx X) T μ + tauMax T X * (Fc eL gam (eps2L X) (Kx X) T μ + 1 / 2 ^ 62) ≤
      tauMax T X := by
    have h1 : tauMax T X * (1 / 2 ^ 62) ≤ (4 / 100) * (1 / 2 ^ 62) :=
      mul_le_mul_of_nonneg_right hτ4 (by norm_num)
    have hB : Fc eA gam (eps2A X) (Kx X) T μ + tauMax T X * (Fc eL gam (eps2L X) (Kx X) T μ + 1 / 2 ^ 62) ≤
        eps2A X + (1021 / 1000) * ((T : ℝ) * eA + 2 * gam * (X + 1)) +
        (4 / 100) * (eps2L X + (1021 / 1000) * ((T : ℝ) * eL + 2 * gam * (X + 1)) + 1 / 2 ^ 62) := by
      nlinarith
    have hpos : (0 : ℝ) < 1 + 1 / 2 ^ 10 := by norm_num
    have : (Fc eA gam (eps2A X) (Kx X) T μ + tauMax T X * (Fc eL gam (eps2L X) (Kx X) T μ + 1 / 2 ^ 62)) * (1 + 1 / 2 ^ 10) ≤ tauMax T X * (1 + 1 / 2 ^ 10) := by
      rw [hτeq]
      exact (mul_le_mul_of_nonneg_right hB hpos.le).trans hmain
    exact le_of_mul_le_mul_right this hpos
  constructor
  · nlinarith
  · have hFA : Fc eL gam (eps2L X) (Kx X) T μ ≤ Fc eA gam (eps2A X) (Kx X) T μ := by
      unfold Fc Ex
      have h1 : Real.exp ((T : ℝ) * eL + gam * (μ + Kx X)) ≤ Real.exp ((T : ℝ) * eA + gam * (μ + Kx X)) :=
        Real.exp_le_exp.mpr (by nlinarith)
      have h2 : eps2L X ≤ eps2A X := by unfold eps2L eps2A; linarith
      have := Real.exp_pos ((T : ℝ) * eL + gam * (μ + Kx X))
      have h3 : (1 + eps2L X) * Real.exp ((T : ℝ) * eL + gam * (μ + Kx X)) ≤
          (1 + eps2A X) * Real.exp ((T : ℝ) * eA + gam * (μ + Kx X)) :=
        mul_le_mul (by linarith) h1 this.le (by unfold eps2A; positivity)
      linarith
    nlinarith

set_option maxHeartbeats 1000000 in
/-- **The ratio at the end of the loop.** After `T ≤ 2^16` trips over all `n ≥ 1` tokens
(`n ≤ 2^22`) of scores `|S k| ≤ X ≤ 1024` and values `|V k| ≤ 2^15`, at a reference `μ` that is
the (approximate) maximum of the scores, with the invariants of `l` and of one component `a` of
`acc`: `l > 0`, and `a / l` is within `tauMax T X·(Λ + |R|) + 2^-30` of the exact attention
`R = aS / lS`, where `Λ = mS / lS` is the weighted mean of `|V|`; and `|a / l| ≤ 40000`. -/
theorem final_core {X : ℝ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (S V : ℕ → ℝ) {n T : ℕ} {μ l a : ℝ}
    (hT1 : 1 ≤ T) (hT2 : T ≤ 2 ^ 16) (_hn1 : 1 ≤ n) (hn2 : n ≤ 2 ^ 22)
    (hV : ∀ k < n, |V k| ≤ 2 ^ 15)
    (hSμ : ∀ k < n, S k ≤ μ + dS X) (hμS : ∃ k < n, μ ≤ S k + dS X)
    (hμK : μ ≤ Kx X) (hKμ : -(Kx X) ≤ μ)
    (hl : InvL X S n T μ l) (ha : InvA X S V n T μ a) :
    0 < l ∧
      |a / l - aS S V μ n / lS S μ n| ≤
        tauMax T X * (mS S V μ n / lS S μ n + |aS S V μ n / lS S μ n|) + 1 / 2 ^ 30 ∧
      |a / l| ≤ 40000 := by
  have hu := u_pos
  obtain ⟨k0, hk0, hk0'⟩ := hμS
  have hdS0 : 0 ≤ dS X := by unfold dS; positivity
  have hdS : dS X ≤ 1 / 2 := by unfold dS u; norm_num; nlinarith
  have hexp : Real.exp (dS X) ≤ 2 := by
    refine (exp_le_inv (by linarith)).trans ?_
    rw [div_le_iff₀ (by linarith)]; linarith
  have hL1 : 1 / 2 ≤ lS S μ n := by
    have h := Real.add_one_le_exp (-dS X)
    exact (by linarith : 1 / 2 ≤ Real.exp (-dS X)).trans (lS_ge S hk0 hk0')
  have hL2 : lS S μ n ≤ 2 ^ 23 := by
    have h := mS_le S (fun _ => 1) (Vm := 1) (N := n) hdS0 (by norm_num) hSμ (fun k _ => by simp)
    rw [(lS_eq S μ n).2] at h
    have hN : (n : ℝ) ≤ 2 ^ 22 := by exact_mod_cast hn2
    calc lS S μ n ≤ n * Real.exp (dS X) * 1 := h
      _ ≤ 2 ^ 22 * 2 * 1 := by
          apply mul_le_mul (mul_le_mul hN hexp (Real.exp_pos _).le (by positivity)) le_rfl
            (by norm_num) (by positivity)
      _ = 2 ^ 23 := by norm_num
  have hLpos : 0 < lS S μ n := by linarith
  have hM0 : 0 ≤ mS S V μ n := by unfold mS; positivity
  have hAM := abs_aS_le S V μ n
  have hMV : mS S V μ n ≤ 2 ^ 15 * lS S μ n := by
    unfold mS lS
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun k hk => ?_
    have := hV k (Finset.mem_range.mp hk)
    nlinarith [Real.exp_pos (S k - μ)]
  -- the invariants
  have hl' : |l - lS S μ n| ≤ Fc eL gam (eps2L X) (Kx X) T μ * lS S μ n +
      Gabs * T * Ex eL gam (Kx X) T μ := by
    have h := hl; unfold InvL ColInv at h
    rwa [(lS_eq S μ n).1, (lS_eq S μ n).2] at h
  have ha' : |a - aS S V μ n| ≤ Fc eA gam (eps2A X) (Kx X) T μ * mS S V μ n +
      Gabs * T * Ex eA gam (Kx X) T μ := ha
  have heL0 : 0 ≤ eL := by unfold eL; positivity
  have heL : eL ≤ eA := by unfold eL eA; nlinarith
  have heA0 : 0 ≤ eA := by unfold eA; positivity
  have h48 : 48 * u * X ≤ 48 * u * 1024 := by nlinarith
  have hFL := Fc_le (ε2 := eps2L X) hX0 hX1 heL0 heL (by unfold eps2L; positivity)
    (by unfold eps2L; linarith) hT2 hμK hKμ
  have hFLn : 0 ≤ Fc eL gam (eps2L X) (Kx X) T μ := by
    unfold Fc
    have := Ex_ge_one heL0 (show 0 ≤ gam by unfold gam; positivity) T hKμ
    have := mul_le_mul_of_nonneg_left this (show 0 ≤ 1 + eps2L X by unfold eps2L; positivity)
    unfold eps2L at *; nlinarith
  have hFA : 0 ≤ Fc eA gam (eps2A X) (Kx X) T μ := by
    unfold Fc
    have := Ex_ge_one heA0 (show 0 ≤ gam by unfold gam; positivity) T hKμ
    have := mul_le_mul_of_nonneg_left this (show 0 ≤ 1 + eps2A X by unfold eps2A; positivity)
    unfold eps2A at *; nlinarith
  have hGT : ∀ {e1 : ℝ}, 0 ≤ e1 → e1 ≤ eA → Gabs * T * Ex e1 gam (Kx X) T μ ≤ 1 / 2 ^ 63 := by
    intro e1 h0 h1
    have hEx := Ex_le hX0 hX1 h0 h1 hT2 hμK hKμ
    have hT : (T : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hT2
    have hG : Gabs * T ≤ Gabs * 2 ^ 16 := mul_le_mul_of_nonneg_left hT (by unfold Gabs; positivity)
    have : Gabs * 2 ^ 16 = 1 / 2 ^ 64 := by unfold Gabs; norm_num
    calc Gabs * T * Ex e1 gam (Kx X) T μ ≤ (1 / 2 ^ 64) * (21 / 20) :=
          mul_le_mul (hG.trans this.le) hEx (Real.exp_pos _).le (by positivity)
      _ ≤ 1 / 2 ^ 63 := by norm_num
  have hgL := hGT heL0 heL
  have hgA := hGT heA0 le_rfl
  have hgL0 : 0 ≤ Gabs * T * Ex eL gam (Kx X) T μ := by unfold Ex Gabs; positivity
  have hgA0 : 0 ≤ Gabs * T * Ex eA gam (Kx X) T μ := by unfold Ex Gabs; positivity
  -- abbreviations
  set L := lS S μ n with hLdef
  set A := aS S V μ n with hAdef
  set M := mS S V μ n with hMdef
  set cL := Fc eL gam (eps2L X) (Kx X) T μ with hcL
  set cA := Fc eA gam (eps2A X) (Kx X) T μ with hcA
  set gL := Gabs * T * Ex eL gam (Kx X) T μ with hgLdef
  set gA := Gabs * T * Ex eA gam (Kx X) T μ with hgAdef
  have heLL : cL * L + gL < L := by nlinarith
  have hlpos : 0 < l := by
    have := (abs_le.mp hl').1
    nlinarith
  have hAL : |A| ≤ (|A| / L) * L := by rw [div_mul_cancel₀ _ hLpos.ne']
  have hr := TVA.Softmax.ratio_err hLpos hl' heLL ha' hAL
  have hkey := key_ineq hX0 hX1 hT1 hT2 hμK hKμ
  have hτ := tauMax_range hX0 hX1 hT2
  set τ := tauMax T X with hτdef
  set q := 1 - cL - 1 / 2 ^ 62 with hq
  have hDpos : 0 < L - (cL * L + gL) := by linarith
  have hD : L * q ≤ L - (cL * L + gL) := by
    have h1 := mul_le_mul_of_nonneg_left hL1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 62)
    have : gL ≤ 1 / 2 ^ 62 * L := by norm_num at h1 hgL ⊢; linarith
    rw [hq]; linarith
  have hq2 : 1 / 2 ≤ q := by rw [hq]; linarith
  -- Λ and ρ
  have hΛ0 : 0 ≤ M / L := div_nonneg hM0 hLpos.le
  have hρ0 : 0 ≤ |A| / L := div_nonneg (abs_nonneg _) hLpos.le
  have hΛ2 : M / L ≤ 2 ^ 15 := by rw [div_le_iff₀ hLpos]; linarith
  have hρ2 : |A| / L ≤ 2 ^ 15 := by rw [div_le_iff₀ hLpos]; linarith
  have hAabs : |A / L| = |A| / L := by rw [abs_div, abs_of_pos hLpos]
  have hMe : M = (M / L) * L := by rw [div_mul_cancel₀ _ hLpos.ne']
  have hAe : |A| = (|A| / L) * L := by rw [div_mul_cancel₀ _ hLpos.ne']
  set Λ := M / L with hΛ
  set ρ := |A| / L with hρ
  have hbound : |a / l - A / L| ≤ τ * (Λ + |A / L|) + 1 / 2 ^ 30 := by
    refine hr.trans ?_
    rw [div_le_iff₀ hDpos, hAabs]
    have t1 : cA * M ≤ τ * q * M := mul_le_mul_of_nonneg_right hkey.1 hM0
    have e3 : τ * q * M = τ * q * (Λ * L) := by rw [← hMe]
    have t2 : cL * (ρ * L) ≤ τ * q * (ρ * L) :=
      mul_le_mul_of_nonneg_right hkey.2 (mul_nonneg hρ0 hLpos.le)
    have t3 : τ * (Λ + ρ) * (L * q) ≤ τ * (Λ + ρ) * (L - (cL * L + gL)) :=
      mul_le_mul_of_nonneg_left hD (mul_nonneg hτ.1 (by linarith))
    have hD4 : 1 / 4 ≤ L - (cL * L + gL) := by
      have : L * (1 / 2) ≤ L * q := mul_le_mul_of_nonneg_left hq2 hLpos.le
      linarith
    have t4 : 1 / 2 ^ 32 ≤ 1 / 2 ^ 30 * (L - (cL * L + gL)) := by
      have := mul_le_mul_of_nonneg_left hD4 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 30)
      have h32 : (1 : ℝ) / 2 ^ 30 * (1 / 4) = 1 / 2 ^ 32 := by norm_num
      linarith
    have t5 : gA + ρ * gL ≤ 1 / 2 ^ 32 := by
      have h1 : ρ * gL ≤ 2 ^ 15 * (1 / 2 ^ 63) := mul_le_mul hρ2 hgL hgL0 (by positivity)
      have h2 : (1 : ℝ) / 2 ^ 63 + 2 ^ 15 * (1 / 2 ^ 63) ≤ 1 / 2 ^ 32 := by norm_num
      linarith
    linarith
  refine ⟨hlpos, hbound, ?_⟩
  -- |a / l| ≤ 40000
  have hR : |A / L| ≤ 32768 := by rw [hAabs]; norm_num at hρ2; exact hρ2
  have hw := abs_sub_abs_le_abs_sub (a / l) (A / L)
  have hΛ' : Λ ≤ 32768 := by norm_num at hΛ2; exact hΛ2
  have h1 : τ * (Λ + |A / L|) ≤ 4 / 100 * (32768 + 32768) :=
    mul_le_mul (tauMax_le hX0 hX1 hT2) (by linarith) (by positivity) (by norm_num)
  have h30 : (1 : ℝ) / 2 ^ 30 ≤ 1 := by norm_num
  norm_num at h1
  linarith

end Num

end Claims.PagedAttention
