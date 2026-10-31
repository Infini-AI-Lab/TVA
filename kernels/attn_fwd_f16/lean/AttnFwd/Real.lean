import AttnFwd.Model
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# `attn_fwd_f16`: real-number facts the error analysis uses

* powers of two: `two_rpow_sub_one` (`|2^δ − 1| ≤ |δ|` for `|δ| ≤ 1/2`), `two_rpow_mul_err`;
* `acc_step`: one step of a rescaled accumulation (`a' ≈ λ·a + new`), the form both `acc` and
  `l_i` follow, with the error split into a part relative to the magnitude and an absolute part;
* `ratio_err`: the error of `acc / l` against `S / W`.
-/

namespace Kernels.AttnFwd

open TVA

/-! ## Powers of two -/

theorem two_rpow_pos (x : ℝ) : 0 < (2 : ℝ) ^ x := Real.rpow_pos_of_pos (by norm_num) x

theorem two_rpow_add (x y : ℝ) : (2 : ℝ) ^ (x + y) = 2 ^ x * 2 ^ y := Real.rpow_add (by norm_num) x y

theorem two_rpow_le {x y : ℝ} (h : x ≤ y) : (2 : ℝ) ^ x ≤ 2 ^ y :=
  Real.rpow_le_rpow_of_exponent_le (by norm_num) h

theorem two_rpow_zero : (2 : ℝ) ^ (0 : ℝ) = 1 := Real.rpow_zero 2

theorem one_le_two_rpow {x : ℝ} (h : 0 ≤ x) : 1 ≤ (2 : ℝ) ^ x := by
  have := two_rpow_le h; rwa [two_rpow_zero] at this

theorem two_rpow_le_one {x : ℝ} (h : x ≤ 0) : (2 : ℝ) ^ x ≤ 1 := by
  have := two_rpow_le h; rwa [two_rpow_zero] at this

/-- `|2^δ − 1| ≤ |δ|` for `|δ| ≤ 1/2`. -/
theorem two_rpow_sub_one {δ : ℝ} (h : |δ| ≤ 1 / 2) : |(2 : ℝ) ^ δ - 1| ≤ |δ| := by
  have hl1 := Real.log_two_lt_d9
  have hl0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  rw [Real.rpow_def_of_pos (by norm_num)]
  have hy : |Real.log 2 * δ| ≤ 0.6931471808 * |δ| := by
    rw [abs_mul, abs_of_pos hl0]
    exact mul_le_mul_of_nonneg_right hl1.le (abs_nonneg _)
  have hy1 : |Real.log 2 * δ| ≤ 1 := by
    have := abs_nonneg δ
    nlinarith
  have hq := Real.abs_exp_sub_one_sub_id_le hy1
  have hsq : (Real.log 2 * δ) ^ 2 = |Real.log 2 * δ| * |Real.log 2 * δ| := by
    rw [← sq_abs, sq]
  have h2 : |Real.exp (Real.log 2 * δ) - 1| ≤ |Real.log 2 * δ| + (Real.log 2 * δ) ^ 2 := by
    have e : Real.exp (Real.log 2 * δ) - 1 = (Real.exp (Real.log 2 * δ) - 1 - Real.log 2 * δ) +
        Real.log 2 * δ := by ring
    rw [e]
    exact (abs_add_le _ _).trans (by linarith)
  rw [hsq] at h2
  have hd := abs_nonneg δ
  have hya := abs_nonneg (Real.log 2 * δ)
  nlinarith [mul_le_mul hy hy hya (by positivity)]

/-- A perturbed power of two: `2^δ·(1 + e)` with `|δ| ≤ d ≤ 1/2`, `|e| ≤ ε` is within
`(1 + d)(1 + ε) − 1` of `1`. -/
theorem two_rpow_mul_err {δ e d ε : ℝ} (hδ : |δ| ≤ d) (hd : d ≤ 1 / 2) (he : |e| ≤ ε) :
    |(2 : ℝ) ^ δ * (1 + e) - 1| ≤ (1 + d) * (1 + ε) - 1 := by
  have h1 := two_rpow_sub_one (hδ.trans hd)
  have hε : 0 ≤ ε := (abs_nonneg _).trans he
  have hd0 : 0 ≤ d := (abs_nonneg _).trans hδ
  have e1 : (2 : ℝ) ^ δ * (1 + e) - 1 = ((2 : ℝ) ^ δ - 1) * (1 + e) + e := by ring
  rw [e1]
  have h2 : |((2 : ℝ) ^ δ - 1) * (1 + e)| ≤ d * (1 + ε) := by
    rw [abs_mul]
    have : |1 + e| ≤ 1 + ε := (abs_add_le _ _).trans (by rw [abs_one]; linarith)
    exact mul_le_mul (h1.trans hδ) this (abs_nonneg _) hd0
  calc |((2 : ℝ) ^ δ - 1) * (1 + e) + e| ≤ |((2 : ℝ) ^ δ - 1) * (1 + e)| + |e| := abs_add_le _ _
    _ ≤ d * (1 + ε) + ε := add_le_add h2 he
    _ = (1 + d) * (1 + ε) - 1 := by ring

/-! ## One step of a rescaled accumulation -/

/-- **One step of a rescaled accumulation.** The old value `a` is within `(F − 1)·Mg + F·Ab` of its
exact value `S` (`|S| ≤ Mg`); the step scales it by `λ ∈ [0, 1]`, which gives `a1` within
`(Φ − 1)·λ|a| + τ₁` of `λ·a`, and adds a new contribution `cn` within `ψ·Mn + An` of its exact value
`Sn` (`|Sn| ≤ Mn`), with an addition (or a `tl.dot`) that errs by `ρ·(|a1| + Tn) + τ₂`, `Tn` the
magnitude of the added terms. Then the new value is within `(F' − 1)·(λ·Mg + Mn) + F'·(Ab + An + τ₁ + τ₂)`
of `λ·S + Sn`, for any `F' ≥ (1 + ρ)·Φ·F` and `≥ (1 + ρ)(1 + ψ)`. -/
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
  have hMg0 : 0 ≤ Mg := (abs_nonneg _).trans hMg
  have hMn0 : 0 ≤ Mn := (abs_nonneg _).trans hMn
  -- `|a| ≤ F·(Mg + Ab)`
  have ha : |a| ≤ F * (Mg + Ab) := by
    have := abs_sub_abs_le_abs_sub a S
    nlinarith
  have hPF : 1 ≤ Phi * F := by nlinarith
  -- the scaled old value against `λ·S`
  have h1 : |a1 - lam * S| ≤ (Phi * F - 1) * (lam * Mg) + Phi * F * Ab + tau1 := by
    have e : a1 - lam * S = (a1 - lam * a) + lam * (a - S) := by ring
    rw [e]
    have t1 := abs_add_le (a1 - lam * a) (lam * (a - S))
    rw [abs_mul, abs_of_nonneg hlam] at t1
    have t2 : (Phi - 1) * (lam * |a|) ≤ (Phi - 1) * (lam * (F * (Mg + Ab))) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left ha hlam) (by linarith)
    have t3 : lam * |a - S| ≤ lam * ((F - 1) * Mg + F * Ab) := mul_le_mul_of_nonneg_left hold hlam
    have t4 : Phi * F * (lam * Ab) ≤ Phi * F * Ab := by
      have : lam * Ab ≤ Ab := by nlinarith
      exact mul_le_mul_of_nonneg_left this (by linarith)
    nlinarith
  have ha1 : |a1| ≤ Phi * F * (lam * Mg) + Phi * F * Ab + tau1 := by
    have := abs_sub_abs_le_abs_sub a1 (lam * S)
    have hS : |lam * S| ≤ lam * Mg := by rw [abs_mul, abs_of_nonneg hlam]; exact mul_le_mul_of_nonneg_left hMg hlam
    linarith
  have e : a' - (lam * S + Sn) = (a' - (a1 + cn)) + (a1 - lam * S) + (cn - Sn) := by ring
  rw [e]
  have t := abs_add_le ((a' - (a1 + cn)) + (a1 - lam * S)) (cn - Sn)
  have t' := abs_add_le (a' - (a1 + cn)) (a1 - lam * S)
  have hlM : 0 ≤ lam * Mg := mul_nonneg hlam hMg0
  have k1 : rho * |a1| ≤ rho * (Phi * F * (lam * Mg) + Phi * F * Ab + tau1) := mul_le_mul_of_nonneg_left ha1 hrho
  have k2 : rho * Tn ≤ rho * ((1 + psi) * Mn + An) := mul_le_mul_of_nonneg_left hTn hrho
  rw [mul_add] at hcomb
  have hB : |a' - (a1 + cn) + (a1 - lam * S) + (cn - Sn)| ≤
      rho * (Phi * F * (lam * Mg) + Phi * F * Ab + tau1) + rho * ((1 + psi) * Mn + An) + tau2 +
        ((Phi * F - 1) * (lam * Mg) + Phi * F * Ab + tau1) + (psi * Mn + An) := by
    linarith
  have hF'1 : 1 + rho ≤ F' := by nlinarith
  have key : 0 ≤ (F' - (1 + rho) * Phi * F) * (lam * Mg) + (F' - (1 + rho) * (1 + psi)) * Mn +
      (F' - (1 + rho) * Phi * F) * Ab + (F' - (1 + rho)) * An + (F' - (1 + rho)) * tau1 +
      (F' - 1) * tau2 := by
    have c1 : 0 ≤ F' - (1 + rho) * Phi * F := by linarith
    have c2 : 0 ≤ F' - (1 + rho) * (1 + psi) := by linarith
    have c3 : 0 ≤ F' - (1 + rho) := by linarith
    have c4 : 0 ≤ F' - 1 := by linarith
    have := mul_nonneg c1 hlM
    have := mul_nonneg c2 hMn0
    have := mul_nonneg c1 hAb
    have := mul_nonneg c3 hAn
    have := mul_nonneg c3 htau1
    have := mul_nonneg c4 htau2
    linarith
  have iden : (F' - 1) * (lam * Mg + Mn) + F' * (Ab + An + tau1 + tau2) -
      (rho * (Phi * F * (lam * Mg) + Phi * F * Ab + tau1) + rho * ((1 + psi) * Mn + An) + tau2 +
        ((Phi * F - 1) * (lam * Mg) + Phi * F * Ab + tau1) + (psi * Mn + An)) =
      (F' - (1 + rho) * Phi * F) * (lam * Mg) + (F' - (1 + rho) * (1 + psi)) * Mn +
      (F' - (1 + rho) * Phi * F) * Ab + (F' - (1 + rho)) * An + (F' - (1 + rho)) * tau1 +
      (F' - 1) * tau2 := by ring
  linarith

end Kernels.AttnFwd
