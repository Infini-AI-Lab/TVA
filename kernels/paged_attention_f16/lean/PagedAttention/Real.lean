import PagedAttention.Model
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# `paged_attention_f16`: real-number facts the error analysis uses

* `exp` near `0`: `abs_exp_sub_one'` (`|e^δ − 1| ≤ |δ| + δ²` for `|δ| ≤ 1`), `exp_le_two`,
  `one_add_pow_le_exp`;
* `acc_step'`, `acc_step`: one step of a rescaled accumulation (`a' ≈ λ·a + new`), the form both
  `acc` and `l_i` follow, with the error split into a part relative to the magnitude and an
  absolute part;
* `ratio_err`: the error of `acc / l` against `S / W`.
-/

namespace Kernels.PagedAttention

open TVA

/-! ## `exp` near `0` -/

/-- `|e^δ − 1| ≤ |δ| + δ²` for `|δ| ≤ 1`. -/
theorem abs_exp_sub_one' {δ : ℝ} (h : |δ| ≤ 1) : |Real.exp δ - 1| ≤ |δ| + δ ^ 2 := by
  have hq := Real.abs_exp_sub_one_sub_id_le h
  have e : Real.exp δ - 1 = (Real.exp δ - 1 - δ) + δ := by ring
  rw [e]
  exact (abs_add_le _ _).trans (by linarith)

/-- `e^y ≤ 1 + 2y` for `0 ≤ y ≤ 1`. -/
theorem exp_le_two {y : ℝ} (h0 : 0 ≤ y) (h1 : y ≤ 1) : Real.exp y ≤ 1 + 2 * y := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := y) (by rw [abs_of_nonneg h0]; exact h1)
  have := le_abs_self (Real.exp y - 1 - y)
  nlinarith

theorem one_add_pow_le_exp {x : ℝ} (hx : 0 ≤ x) (n : ℕ) : (1 + x) ^ n ≤ Real.exp (n * x) := by
  rw [Real.exp_nat_mul]
  exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp x]) n

/-- A power `s^n` with `1 ≤ s ≤ 1 + c` and `n·c ≤ N·c ≤ 1`: at most `1 + 2·N·c`. -/
theorem pow_le_of_step {s : ℝ} {n N : ℕ} {c : ℝ} (h1 : 1 ≤ s) (hs : s ≤ 1 + c) (hn : n ≤ N)
    (hc : (N : ℝ) * c ≤ 1) : s ^ n ≤ 1 + 2 * (N * c) := by
  have hc0 : 0 ≤ c := by linarith
  have h := one_add_pow_le_exp (show 0 ≤ s - 1 by linarith) n
  rw [show 1 + (s - 1) = s by ring] at h
  refine h.trans ((Real.exp_le_exp.mpr ?_).trans (exp_le_two (by positivity) hc))
  have : (n : ℝ) ≤ N := by exact_mod_cast hn
  have : (n : ℝ) * (s - 1) ≤ N * c := mul_le_mul this (by linarith) (by linarith) (by positivity)
  exact this

/-! ## One step of a rescaled accumulation -/

/-- **One step of a rescaled accumulation.** The old value `a` is within `(F − 1)·Mg + F·Ab` of its
exact value `S` (`|S| ≤ Mg`); the step scales it by `λ ∈ [0, 1]`, which gives `a1` within
`(Φ − 1)·λ|a| + τ₁` of `λ·a`, and adds a new contribution `cn` within `ψ·Mn + An` of its exact value
`Sn` (`|Sn| ≤ Mn`), with an addition that errs by `ρ·(|a1| + Tn) + τ₂`, `Tn` the magnitude of the
added term. Then the new value is within
`(F' − 1)·(λ·Mg + Mn) + F'·(λ·Ab + An + τ₁ + τ₂)` of `λ·S + Sn`, for any `F' ≥ (1 + ρ)·Φ·F` and
`≥ (1 + ρ)(1 + ψ)`. -/
theorem acc_step' {a S Mg Ab F a1 lam Phi tau1 cn Sn Mn An psi Tn a' rho tau2 F' : ℝ}
    (hF : 1 ≤ F) (hMg : |S| ≤ Mg) (hAb : 0 ≤ Ab) (hold : |a - S| ≤ (F - 1) * Mg + F * Ab)
    (hlam : 0 ≤ lam) (_hlam1 : lam ≤ 1) (hPhi : 1 ≤ Phi)
    (ha1 : |a1 - lam * a| ≤ (Phi - 1) * (lam * |a|) + tau1)
    (hMn : |Sn| ≤ Mn) (hpsi : 0 ≤ psi) (hAn : 0 ≤ An)
    (hcn : |cn - Sn| ≤ psi * Mn + An) (hTn : Tn ≤ (1 + psi) * Mn + An)
    (hrho : 0 ≤ rho) (htau1 : 0 ≤ tau1) (htau2 : 0 ≤ tau2)
    (hcomb : |a' - (a1 + cn)| ≤ rho * (|a1| + Tn) + tau2)
    (hF1 : (1 + rho) * Phi * F ≤ F') (hF2 : (1 + rho) * (1 + psi) ≤ F') :
    |a' - (lam * S + Sn)| ≤ (F' - 1) * (lam * Mg + Mn) + F' * (lam * Ab + An + tau1 + tau2) := by
  have hlA : 0 ≤ lam * Ab := mul_nonneg hlam hAb
  have hMg0 : 0 ≤ Mg := (abs_nonneg _).trans hMg
  have hMn0 : 0 ≤ Mn := (abs_nonneg _).trans hMn
  have ha : |a| ≤ F * (Mg + Ab) := by
    have := abs_sub_abs_le_abs_sub a S
    nlinarith
  have hPF : 1 ≤ Phi * F := by nlinarith
  have h1 : |a1 - lam * S| ≤ (Phi * F - 1) * (lam * Mg) + Phi * F * (lam * Ab) + tau1 := by
    have e : a1 - lam * S = (a1 - lam * a) + lam * (a - S) := by ring
    rw [e]
    have t1 := abs_add_le (a1 - lam * a) (lam * (a - S))
    rw [abs_mul, abs_of_nonneg hlam] at t1
    have t2 : (Phi - 1) * (lam * |a|) ≤ (Phi - 1) * (lam * (F * (Mg + Ab))) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left ha hlam) (by linarith)
    have t3 : lam * |a - S| ≤ lam * ((F - 1) * Mg + F * Ab) := mul_le_mul_of_nonneg_left hold hlam
    nlinarith
  have ha1 : |a1| ≤ Phi * F * (lam * Mg) + Phi * F * (lam * Ab) + tau1 := by
    have := abs_sub_abs_le_abs_sub a1 (lam * S)
    have hS : |lam * S| ≤ lam * Mg := by
      rw [abs_mul, abs_of_nonneg hlam]; exact mul_le_mul_of_nonneg_left hMg hlam
    linarith
  have e : a' - (lam * S + Sn) = (a' - (a1 + cn)) + (a1 - lam * S) + (cn - Sn) := by ring
  rw [e]
  have t := abs_add_le ((a' - (a1 + cn)) + (a1 - lam * S)) (cn - Sn)
  have t' := abs_add_le (a' - (a1 + cn)) (a1 - lam * S)
  have hlM : 0 ≤ lam * Mg := mul_nonneg hlam hMg0
  have k1 : rho * |a1| ≤ rho * (Phi * F * (lam * Mg) + Phi * F * (lam * Ab) + tau1) :=
    mul_le_mul_of_nonneg_left ha1 hrho
  have k2 : rho * Tn ≤ rho * ((1 + psi) * Mn + An) := mul_le_mul_of_nonneg_left hTn hrho
  rw [mul_add] at hcomb
  have hB : |a' - (a1 + cn) + (a1 - lam * S) + (cn - Sn)| ≤
      rho * (Phi * F * (lam * Mg) + Phi * F * (lam * Ab) + tau1) + rho * ((1 + psi) * Mn + An) + tau2 +
        ((Phi * F - 1) * (lam * Mg) + Phi * F * (lam * Ab) + tau1) + (psi * Mn + An) := by
    linarith
  have hF'1 : 1 + rho ≤ F' := by nlinarith
  have key : 0 ≤ (F' - (1 + rho) * Phi * F) * (lam * Mg) + (F' - (1 + rho) * (1 + psi)) * Mn +
      (F' - (1 + rho) * Phi * F) * (lam * Ab) + (F' - (1 + rho)) * An + (F' - (1 + rho)) * tau1 +
      (F' - 1) * tau2 := by
    have c1 : 0 ≤ F' - (1 + rho) * Phi * F := by linarith
    have c2 : 0 ≤ F' - (1 + rho) * (1 + psi) := by linarith
    have c3 : 0 ≤ F' - (1 + rho) := by linarith
    have c4 : 0 ≤ F' - 1 := by linarith
    have := mul_nonneg c1 hlM
    have := mul_nonneg c2 hMn0
    have := mul_nonneg c1 hlA
    have := mul_nonneg c3 hAn
    have := mul_nonneg c3 htau1
    have := mul_nonneg c4 htau2
    linarith
  have iden : (F' - 1) * (lam * Mg + Mn) + F' * (lam * Ab + An + tau1 + tau2) -
      (rho * (Phi * F * (lam * Mg) + Phi * F * (lam * Ab) + tau1) + rho * ((1 + psi) * Mn + An) + tau2 +
        ((Phi * F - 1) * (lam * Mg) + Phi * F * (lam * Ab) + tau1) + (psi * Mn + An)) =
      (F' - (1 + rho) * Phi * F) * (lam * Mg) + (F' - (1 + rho) * (1 + psi)) * Mn +
      (F' - (1 + rho) * Phi * F) * (lam * Ab) + (F' - (1 + rho)) * An + (F' - (1 + rho)) * tau1 +
      (F' - 1) * tau2 := by ring
  linarith

/-- `acc_step'` with the old absolute part unscaled (`λ·Ab ≤ Ab`). -/
theorem acc_step {a S Mg Ab F a1 lam Phi tau1 cn Sn Mn An psi Tn a' rho tau2 F' : ℝ}
    (hF : 1 ≤ F) (hMg : |S| ≤ Mg) (hAb : 0 ≤ Ab) (hold : |a - S| ≤ (F - 1) * Mg + F * Ab)
    (hlam : 0 ≤ lam) (hlam1 : lam ≤ 1) (hPhi : 1 ≤ Phi)
    (ha1 : |a1 - lam * a| ≤ (Phi - 1) * (lam * |a|) + tau1)
    (hMn : |Sn| ≤ Mn) (hpsi : 0 ≤ psi) (hAn : 0 ≤ An)
    (hcn : |cn - Sn| ≤ psi * Mn + An) (hTn : Tn ≤ (1 + psi) * Mn + An)
    (hrho : 0 ≤ rho) (htau1 : 0 ≤ tau1) (htau2 : 0 ≤ tau2)
    (hcomb : |a' - (a1 + cn)| ≤ rho * (|a1| + Tn) + tau2)
    (hF1 : (1 + rho) * Phi * F ≤ F') (hF2 : (1 + rho) * (1 + psi) ≤ F') :
    |a' - (lam * S + Sn)| ≤ (F' - 1) * (lam * Mg + Mn) + F' * (Ab + An + tau1 + tau2) := by
  have h := acc_step' hF hMg hAb hold hlam hlam1 hPhi ha1 hMn hpsi hAn hcn hTn hrho htau1 htau2 hcomb hF1 hF2
  have h0 : 0 ≤ (1 + rho) * (1 + psi) := mul_nonneg (by linarith) (by linarith)
  have hlA : lam * Ab ≤ Ab := by nlinarith
  have := mul_le_mul_of_nonneg_left hlA (by linarith : (0 : ℝ) ≤ F')
  linarith

/-! ## The ratio `acc / l_i` -/

/-- **`acc / l_i` against `S / W`.** With `l` within `β·W` of `W ≥ wl > 0` and `a` within
`(F − 1)·Mg + F·Ta` of `S` (`|S| ≤ Mg`), the ratio is within
`(F − 1 + β)/(1 − β)·(Mg/W) + F·Ta/(wl·(1 − β))` of `S / W`. -/
theorem ratio_err {a S Mg l W F β wl Ta : ℝ}
    (hW : wl ≤ W) (hwl : 0 < wl) (hβ0 : 0 ≤ β) (hβ : β < 1) (hS : |S| ≤ Mg)
    (hl : |l - W| ≤ β * W) (ha : |a - S| ≤ (F - 1) * Mg + F * Ta)
    (hF : 1 ≤ F) (hTa : 0 ≤ Ta) :
    (1 - β) * W ≤ l ∧ |a / l - S / W| ≤ (F - 1 + β) / (1 - β) * (Mg / W) + F * Ta / (wl * (1 - β)) := by
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
  have e : ((F - 1 + β) / (1 - β) * (Mg / W) + F * Ta / (wl * (1 - β))) * ((1 - β) * W) =
      (F - 1 + β) * Mg + F * Ta * (W / wl) := by
    field_simp
  rw [e]
  have hWw : 1 ≤ W / wl := by rw [le_div_iff₀ hwl]; linarith
  have h0 : 0 ≤ F * Ta := mul_nonneg (by linarith) hTa
  have := mul_le_mul_of_nonneg_left hWw h0
  nlinarith

end Kernels.PagedAttention
