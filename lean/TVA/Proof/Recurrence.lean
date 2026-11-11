import Mathlib.Analysis.Complex.Exponential
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Tactic

/-!
# A state carried across loop trips with a decay

**The recurrence.** Chunked linear-attention and state-space kernels carry a state across the
trips of a loop, entry by entry,

  `S (t + 1) = g t · S t + U t`,

with a decay `g t` (often `exp` of a non-positive gate, or `exp(cs[last] − cs[i])` from a
cumulative sum `cs`) and an update `U t` (often a `tl.dot`). Unrolled, each update is decayed by
the gates of the trips after it:

  `S n = (Π_{u<n} g u)·S 0 + Σ_{t<n} (Π_{t<u<n} g u)·U t`.

**Errors.** A computed state `Ŝ` follows the same recurrence with a computed decay `ĝ`, a computed
update `Û` and a rounding per trip. Its error `|Ŝ t − S t|` then follows a recurrence of the same
shape, `E (t + 1) ≤ |ĝ t|·E t + c t`, where `c t` is what the trip adds (its rounding, the update's
error, the decay's error times the state). So the error is bounded by the same unrolled sum, with
`|ĝ t|` for the decay: when every decay is at most `1` (a gate that only forgets), the trips'
errors simply add up.

* **The closed form:** `closed` (from `0`) and `closed_Ico` (from any trip `m`).
* **The running bound:** `run G c E₀` is `E₀`, then `G t·run t + c t`. `le_run`: any `x` with
  `x (t + 1) ≤ G t·x t + c t` stays below it. Closed form `run_eq`; at most `E₀ + Σ c` when
  `0 ≤ G ≤ 1` (`run_le_sum`), at most `M^n·(E₀ + Σ c)` when `0 ≤ G ≤ M`, `1 ≤ M` (`run_le_pow`).
* **The perturbed state:** `perturb` (`|Ŝ n − S n| ≤ run G (e + ε + γ·|S|) |Ŝ 0 − S 0| n`), and its
  forms `perturb'` (gate error charged to the computed state), `perturb_le_sum` (computed decays
  at most `1`), `perturb_le_uniform` (`|S t| ≤ B`) and `perturb_le_pow` (decays at most `M ≥ 1`).
  `abs_le_run`, `abs_le_sum`, `abs_le_sum_of_le` bound the exact state itself.
* **Gates as exponentials:** the products of exponential decays telescope (`prod_exp`,
  `sum_Ico_sub`, `prod_exp_sub`), every factor is in `(0, 1]` for non-positive gates
  (`exp_mem_Ioc`, `cum_le`, `exp_sub_mem_Ioc`), and the closed form as a chunk computes it from the
  inclusive cumulative sum of the gates (`closed_exp`, `closed_exp_sub`, `closed_cumsum`).
* **A matrix state:** `closed_mat`, `closed_cumsum_mat`, `perturb_mat`, `perturb_mat_le_sum`: the
  same for a state `κ → ν → ℝ` (such as `Fin K → Fin V → ℝ`) with a scalar decay per trip.

**How a kernel uses it.** A claim usually states its reference as the trip-by-trip recurrence,
while a chunked kernel computes the chunk's closed form from `tl.cumsum` of the gates:
`closed_cumsum` (or `closed_cumsum_mat`) matches the two. For accuracy, the kernel takes `ĝ`, `Û`,
`Ŝ` from its float values and the per-trip errors `e`, `γ`, `ε` from the float error lemmas and the
device facts, then applies `perturb_le_sum` (or `perturb_mat_le_sum`), with `abs_le_sum_of_le`
for the state's size.

This is mathematics over `ℝ` (Mathlib only), in the namespace `TVA.Recurrence`.
-/

namespace TVA

namespace Recurrence

open Finset

/-! ## The closed form -/

/-- **The closed form, from trip `m`.** If `S (t + 1) = g t·S t + U t` for every trip
`m ≤ t < n`, then `S n = (Π_{m≤u<n} g u)·S m + Σ_{m≤t<n} (Π_{t<u<n} g u)·U t`. Use it for a state
carried from trip `m` (for example from one chunk boundary to the next); `closed` is the case
`m = 0`. -/
theorem closed_Ico (g U S : ℕ → ℝ) {m n : ℕ} (hmn : m ≤ n)
    (h : ∀ t, m ≤ t → t < n → S (t + 1) = g t * S t + U t) :
    S n = (∏ u ∈ Ico m n, g u) * S m +
      ∑ t ∈ Ico m n, (∏ u ∈ Ico (t + 1) n, g u) * U t := by
  induction n, hmn using Nat.le_induction with
  | base => simp
  | succ n hmn ih =>
    have hs : ∑ t ∈ Ico m n, (∏ u ∈ Ico (t + 1) (n + 1), g u) * U t =
        g n * ∑ t ∈ Ico m n, (∏ u ∈ Ico (t + 1) n, g u) * U t := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun t ht => ?_
      rw [Finset.prod_Ico_succ_top (by simp at ht; omega)]
      ring
    rw [h n hmn (Nat.lt_succ_self n), ih (fun t h1 h2 => h t h1 (by omega)),
      Finset.prod_Ico_succ_top hmn, Finset.sum_Ico_succ_top hmn, hs, Finset.Ico_self,
      Finset.prod_empty]
    ring

/-- **The closed form.** If `S (t + 1) = g t·S t + U t` for every trip `t < n`, then
`S n = (Π_{u<n} g u)·S 0 + Σ_{t<n} (Π_{t<u<n} g u)·U t`. Use it to match a kernel that computes
the state in closed form (a chunk's `exp(cs[last])·S + Σ_i exp(cs[last] − cs[i])·k_i·v_i`) with a
reference given as the trip-by-trip recurrence, or the converse. -/
theorem closed (g U S : ℕ → ℝ) {n : ℕ} (h : ∀ t, t < n → S (t + 1) = g t * S t + U t) :
    S n = (∏ u ∈ range n, g u) * S 0 +
      ∑ t ∈ range n, (∏ u ∈ Ico (t + 1) n, g u) * U t := by
  rw [Finset.range_eq_Ico]
  exact closed_Ico g U S (Nat.zero_le n) fun t _ ht => h t ht

/-! ## The running bound -/

/-- **The running bound** of a linear recursion: `run G c E₀ 0 = E₀` and
`run G c E₀ (t + 1) = G t·run G c E₀ t + c t`. It bounds the error of a carried state (`perturb`,
with `G t ≥ |ĝ t|` and `c t` the error a trip adds); `run_le_sum`, `run_le_pow` and the closed
form `run_eq` evaluate it. -/
def run (G c : ℕ → ℝ) (E₀ : ℝ) : ℕ → ℝ
  | 0 => E₀
  | t + 1 => G t * run G c E₀ t + c t

/-- The running bound before any trip is the initial error. -/
theorem run_zero (G c : ℕ → ℝ) (E₀ : ℝ) : run G c E₀ 0 = E₀ := rfl

/-- One trip of the running bound. -/
theorem run_succ (G c : ℕ → ℝ) (E₀ : ℝ) (t : ℕ) :
    run G c E₀ (t + 1) = G t * run G c E₀ t + c t := rfl

/-- **The running bound in closed form**:
`run G c E₀ n = (Π_{u<n} G u)·E₀ + Σ_{t<n} (Π_{t<u<n} G u)·c t` (an instance of `closed`). -/
theorem run_eq (G c : ℕ → ℝ) (E₀ : ℝ) (n : ℕ) :
    run G c E₀ n = (∏ u ∈ range n, G u) * E₀ +
      ∑ t ∈ range n, (∏ u ∈ Ico (t + 1) n, G u) * c t :=
  closed G c (run G c E₀) fun _ _ => rfl

/-- **A sequence below its running bound.** If `x (t + 1) ≤ G t·x t + c t` with `0 ≤ G t` for
every trip `t < n`, then `x n ≤ run G c (x 0) n`. Use it directly for an error sequence whose
per-trip bound is not of the shape `perturb` takes (a relative error on the state, say). -/
theorem le_run {x G c : ℕ → ℝ} {n : ℕ} (hG : ∀ t, t < n → 0 ≤ G t)
    (h : ∀ t, t < n → x (t + 1) ≤ G t * x t + c t) : x n ≤ run G c (x 0) n := by
  induction n with
  | zero => simp [run_zero]
  | succ n ih =>
    have e := ih (fun t ht => hG t (by omega)) (fun t ht => h t (by omega))
    rw [run_succ]
    have := mul_le_mul_of_nonneg_left e (hG n (by omega))
    linarith [h n (by omega)]

/-- The running bound is non-negative when its inputs are, on the trips `t < n`. -/
theorem run_nonneg {G c : ℕ → ℝ} {E₀ : ℝ} {n : ℕ} (hG : ∀ t, t < n → 0 ≤ G t)
    (hc : ∀ t, t < n → 0 ≤ c t) (hE : 0 ≤ E₀) : 0 ≤ run G c E₀ n := by
  induction n with
  | zero => simpa [run_zero] using hE
  | succ n ih =>
    rw [run_succ]
    have := ih (fun t ht => hG t (by omega)) (fun t ht => hc t (by omega))
    have := hG n (by omega)
    have := hc n (by omega)
    positivity

/-- **Monotonicity.** The running bound grows with its inputs: `G ≤ G'`, `c ≤ c'` on the trips
`t < n` (with `0 ≤ G`, `0 ≤ c`) and `0 ≤ E₀ ≤ E₀'` give `run G c E₀ n ≤ run G' c' E₀' n`. Use it
to replace per-trip constants by simpler upper bounds. -/
theorem run_mono {G G' c c' : ℕ → ℝ} {E₀ E₀' : ℝ} {n : ℕ} (hG : ∀ t, t < n → 0 ≤ G t)
    (hGG : ∀ t, t < n → G t ≤ G' t) (hc : ∀ t, t < n → 0 ≤ c t) (hcc : ∀ t, t < n → c t ≤ c' t)
    (hE : 0 ≤ E₀) (hEE : E₀ ≤ E₀') : run G c E₀ n ≤ run G' c' E₀' n := by
  induction n with
  | zero => simpa [run_zero] using hEE
  | succ n ih =>
    rw [run_succ, run_succ]
    have e := ih (fun t ht => hG t (by omega)) (fun t ht => hGG t (by omega))
      (fun t ht => hc t (by omega)) (fun t ht => hcc t (by omega))
    have h0 := run_nonneg (fun t ht => hG t (by omega)) (fun t ht => hc t (by omega)) hE (n := n)
    have := mul_le_mul (hGG n (by omega)) e h0 ((hG n (by omega)).trans (hGG n (by omega)))
    nlinarith [hcc n (by omega)]

/-- With no decay (`G = 1`), the running bound is the plain sum `E₀ + Σ_{t<n} c t`. -/
theorem run_one (c : ℕ → ℝ) (E₀ : ℝ) (n : ℕ) :
    run (fun _ => 1) c E₀ n = E₀ + ∑ t ∈ range n, c t := by
  induction n with
  | zero => simp [run_zero]
  | succ n ih => rw [run_succ, ih, Finset.sum_range_succ]; ring

/-- **Decays at most `1`.** If `0 ≤ G t ≤ 1` and `0 ≤ c t` on the trips `t < n` and `0 ≤ E₀`,
the running bound is at most `E₀ + Σ_{t<n} c t`: the errors of the trips add up, none is
amplified. -/
theorem run_le_sum {G c : ℕ → ℝ} {E₀ : ℝ} {n : ℕ} (hG0 : ∀ t, t < n → 0 ≤ G t)
    (hG1 : ∀ t, t < n → G t ≤ 1) (hc : ∀ t, t < n → 0 ≤ c t) (hE : 0 ≤ E₀) :
    run G c E₀ n ≤ E₀ + ∑ t ∈ range n, c t := by
  rw [← run_one]
  exact run_mono hG0 hG1 hc (fun _ _ => le_rfl) hE le_rfl

/-- **Decays at most `M`** (a computed decay that may round above `1`): if `0 ≤ G t ≤ M` with
`1 ≤ M`, `0 ≤ c t` on the trips `t < n` and `0 ≤ E₀`, the running bound is at most
`M^n·(E₀ + Σ_{t<n} c t)`. -/
theorem run_le_pow {G c : ℕ → ℝ} {E₀ M : ℝ} {n : ℕ} (hM : 1 ≤ M) (hG0 : ∀ t, t < n → 0 ≤ G t)
    (hGM : ∀ t, t < n → G t ≤ M) (hc : ∀ t, t < n → 0 ≤ c t) (hE : 0 ≤ E₀) :
    run G c E₀ n ≤ M ^ n * (E₀ + ∑ t ∈ range n, c t) := by
  induction n with
  | zero => simp [run_zero]
  | succ n ih =>
    rw [run_succ, Finset.sum_range_succ, pow_succ]
    have e := ih (fun t ht => hG0 t (by omega)) (fun t ht => hGM t (by omega))
      (fun t ht => hc t (by omega))
    have h0 := run_nonneg (fun t ht => hG0 t (by omega)) (fun t ht => hc t (by omega)) hE (n := n)
    have hGn := hGM n (by omega)
    have hcn := hc n (by omega)
    have hp : 1 ≤ M ^ n := one_le_pow₀ hM
    have hs : 0 ≤ ∑ t ∈ range n, c t :=
      Finset.sum_nonneg fun t ht => hc t (by simp at ht; omega)
    have e1 : G n * run G c E₀ n ≤ M * (M ^ n * (E₀ + ∑ t ∈ range n, c t)) :=
      mul_le_mul hGn e h0 (by linarith)
    have e2 : c n ≤ M ^ n * M * c n := by
      have : 1 ≤ M ^ n * M := by nlinarith
      nlinarith
    nlinarith

/-! ## The perturbed state -/

/-- **The perturbed state.** The exact state follows `S (t + 1) = g t·S t + U t`; the computed
state `Ŝ` is within `e t` of `ĝ t·Ŝ t + Û t` at each trip, with `|ĝ t − g t| ≤ γ t`,
`|Û t − U t| ≤ ε t` and `|ĝ t| ≤ G t`, on the trips `t < n`. Then
`|Ŝ n − S n| ≤ run G (e + ε + γ·|S|) |Ŝ 0 − S 0| n`. A kernel takes `ĝ`, `Û`, `Ŝ` from its float
values (the decay and update it computed, the state after the trip's rounding) and `e`, `γ`, `ε`
from the float error lemmas; `perturb_le_sum` is the common case. -/
theorem perturb {g gh U Uh S Sh e γ ε G : ℕ → ℝ} {n : ℕ}
    (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hSh : ∀ t, t < n → |Sh (t + 1) - (gh t * Sh t + Uh t)| ≤ e t)
    (hg : ∀ t, t < n → |gh t - g t| ≤ γ t) (hU : ∀ t, t < n → |Uh t - U t| ≤ ε t)
    (hG : ∀ t, t < n → |gh t| ≤ G t) :
    |Sh n - S n| ≤ run G (fun t => e t + ε t + γ t * |S t|) |Sh 0 - S 0| n := by
  refine le_run (x := fun t => |Sh t - S t|) (fun t ht => (abs_nonneg _).trans (hG t ht))
    fun t ht => ?_
  have eq : Sh (t + 1) - S (t + 1) = (Sh (t + 1) - (gh t * Sh t + Uh t)) + gh t * (Sh t - S t) +
      (gh t - g t) * S t + (Uh t - U t) := by rw [hS t ht]; ring
  rw [eq]
  have a1 := abs_add_le ((Sh (t + 1) - (gh t * Sh t + Uh t)) + gh t * (Sh t - S t) +
      (gh t - g t) * S t) (Uh t - U t)
  have a2 := abs_add_le ((Sh (t + 1) - (gh t * Sh t + Uh t)) + gh t * (Sh t - S t))
      ((gh t - g t) * S t)
  have a3 := abs_add_le (Sh (t + 1) - (gh t * Sh t + Uh t)) (gh t * (Sh t - S t))
  rw [abs_mul] at a2 a3
  have m1 : |gh t| * |Sh t - S t| ≤ G t * |Sh t - S t| :=
    mul_le_mul_of_nonneg_right (hG t ht) (abs_nonneg _)
  have m2 : |gh t - g t| * |S t| ≤ γ t * |S t| :=
    mul_le_mul_of_nonneg_right (hg t ht) (abs_nonneg _)
  linarith [hSh t ht, hU t ht]

/-- **The perturbed state, gate error on the computed state.** As `perturb`, but the gate's
error is charged to the computed state (`γ t·|Ŝ t|`) and the decay bound is on the exact gate
(`|g t| ≤ G t`): `|Ŝ n − S n| ≤ run G (e + ε + γ·|Ŝ|) |Ŝ 0 − S 0| n`. Use it when the kernel's
own values are easier to bound than the reference's. -/
theorem perturb' {g gh U Uh S Sh e γ ε G : ℕ → ℝ} {n : ℕ}
    (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hSh : ∀ t, t < n → |Sh (t + 1) - (gh t * Sh t + Uh t)| ≤ e t)
    (hg : ∀ t, t < n → |gh t - g t| ≤ γ t) (hU : ∀ t, t < n → |Uh t - U t| ≤ ε t)
    (hG : ∀ t, t < n → |g t| ≤ G t) :
    |Sh n - S n| ≤ run G (fun t => e t + ε t + γ t * |Sh t|) |Sh 0 - S 0| n := by
  refine le_run (x := fun t => |Sh t - S t|) (fun t ht => (abs_nonneg _).trans (hG t ht))
    fun t ht => ?_
  have eq : Sh (t + 1) - S (t + 1) = (Sh (t + 1) - (gh t * Sh t + Uh t)) + g t * (Sh t - S t) +
      (gh t - g t) * Sh t + (Uh t - U t) := by rw [hS t ht]; ring
  rw [eq]
  have a1 := abs_add_le ((Sh (t + 1) - (gh t * Sh t + Uh t)) + g t * (Sh t - S t) +
      (gh t - g t) * Sh t) (Uh t - U t)
  have a2 := abs_add_le ((Sh (t + 1) - (gh t * Sh t + Uh t)) + g t * (Sh t - S t))
      ((gh t - g t) * Sh t)
  have a3 := abs_add_le (Sh (t + 1) - (gh t * Sh t + Uh t)) (g t * (Sh t - S t))
  rw [abs_mul] at a2 a3
  have m1 : |g t| * |Sh t - S t| ≤ G t * |Sh t - S t| :=
    mul_le_mul_of_nonneg_right (hG t ht) (abs_nonneg _)
  have m2 : |gh t - g t| * |Sh t| ≤ γ t * |Sh t| :=
    mul_le_mul_of_nonneg_right (hg t ht) (abs_nonneg _)
  linarith [hSh t ht, hU t ht]

/-- **The perturbed state, decays at most `1`.** Under the hypotheses of `perturb` with every
computed decay `|ĝ t| ≤ 1` (the exact one need not be bounded):
`|Ŝ n − S n| ≤ |Ŝ 0 − S 0| + Σ_{t<n} (e t + ε t + γ t·|S t|)`. The trips' errors add up. -/
theorem perturb_le_sum {g gh U Uh S Sh e γ ε : ℕ → ℝ} {n : ℕ}
    (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hSh : ∀ t, t < n → |Sh (t + 1) - (gh t * Sh t + Uh t)| ≤ e t)
    (hg : ∀ t, t < n → |gh t - g t| ≤ γ t) (hU : ∀ t, t < n → |Uh t - U t| ≤ ε t)
    (hG : ∀ t, t < n → |gh t| ≤ 1) :
    |Sh n - S n| ≤ |Sh 0 - S 0| + ∑ t ∈ range n, (e t + ε t + γ t * |S t|) := by
  have h1 := perturb hS hSh hg hU (G := fun _ => 1) hG
  rw [run_one] at h1
  exact h1

/-- **The perturbed state, a uniform bound on the state.** As `perturb_le_sum`, with
`|S t| ≤ B` on the trips `t < n`:
`|Ŝ n − S n| ≤ |Ŝ 0 − S 0| + Σ_{t<n} (e t + ε t) + B·Σ_{t<n} γ t`. `abs_le_sum_of_le` gives a
`B`. -/
theorem perturb_le_uniform {g gh U Uh S Sh e γ ε : ℕ → ℝ} {n : ℕ} {B : ℝ}
    (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hSh : ∀ t, t < n → |Sh (t + 1) - (gh t * Sh t + Uh t)| ≤ e t)
    (hg : ∀ t, t < n → |gh t - g t| ≤ γ t) (hU : ∀ t, t < n → |Uh t - U t| ≤ ε t)
    (hG : ∀ t, t < n → |gh t| ≤ 1) (hB : ∀ t, t < n → |S t| ≤ B) :
    |Sh n - S n| ≤ |Sh 0 - S 0| + ∑ t ∈ range n, (e t + ε t) + B * ∑ t ∈ range n, γ t := by
  refine (perturb_le_sum hS hSh hg hU hG).trans ?_
  rw [Finset.mul_sum, add_assoc, ← Finset.sum_add_distrib]
  have hsum : ∑ t ∈ range n, (e t + ε t + γ t * |S t|) ≤ ∑ t ∈ range n, (e t + ε t + B * γ t) := by
    refine Finset.sum_le_sum fun t ht => ?_
    have ht' : t < n := by simpa using ht
    have hγ : 0 ≤ γ t := (abs_nonneg _).trans (hg t ht')
    nlinarith [mul_le_mul_of_nonneg_left (hB t ht') hγ]
  linarith

/-- **The perturbed state, decays at most `M`.** Under the hypotheses of `perturb`, with
`|ĝ t| ≤ M` and `1 ≤ M` (a computed decay that may round above `1`):
`|Ŝ n − S n| ≤ M^n·(|Ŝ 0 − S 0| + Σ_{t<n} (e t + ε t + γ t·|S t|))`. -/
theorem perturb_le_pow {g gh U Uh S Sh e γ ε : ℕ → ℝ} {n : ℕ} {M : ℝ} (hM : 1 ≤ M)
    (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hSh : ∀ t, t < n → |Sh (t + 1) - (gh t * Sh t + Uh t)| ≤ e t)
    (hg : ∀ t, t < n → |gh t - g t| ≤ γ t) (hU : ∀ t, t < n → |Uh t - U t| ≤ ε t)
    (hG : ∀ t, t < n → |gh t| ≤ M) :
    |Sh n - S n| ≤ M ^ n * (|Sh 0 - S 0| + ∑ t ∈ range n, (e t + ε t + γ t * |S t|)) := by
  refine (perturb hS hSh hg hU (G := fun _ => M) hG).trans ?_
  refine run_le_pow hM (fun _ _ => by linarith) (fun _ _ => le_rfl) (fun t ht => ?_) (abs_nonneg _)
  have := (abs_nonneg _).trans (hSh t ht)
  have := (abs_nonneg _).trans (hU t ht)
  have := mul_nonneg ((abs_nonneg _).trans (hg t ht)) (abs_nonneg (S t))
  linarith

/-- **The exact state's size.** If `S (t + 1) = g t·S t + U t` with `|g t| ≤ G t` on the trips
`t < n`, then `|S n| ≤ run G |U| |S 0| n`. -/
theorem abs_le_run {g U S G : ℕ → ℝ} {n : ℕ} (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hG : ∀ t, t < n → |g t| ≤ G t) : |S n| ≤ run G (fun t => |U t|) |S 0| n := by
  refine le_run (x := fun t => |S t|) (fun t ht => (abs_nonneg _).trans (hG t ht))
    fun t ht => ?_
  rw [hS t ht]
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul]
  have := mul_le_mul_of_nonneg_right (hG t ht) (abs_nonneg (S t))
  linarith

/-- **The exact state's size, decays at most `1`.** If `S (t + 1) = g t·S t + U t` with
`|g t| ≤ 1` on the trips `t < n`, then `|S n| ≤ |S 0| + Σ_{t<n} |U t|`. -/
theorem abs_le_sum {g U S : ℕ → ℝ} {n : ℕ} (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hG : ∀ t, t < n → |g t| ≤ 1) : |S n| ≤ |S 0| + ∑ t ∈ range n, |U t| := by
  have := abs_le_run hS (G := fun _ => 1) hG
  rwa [run_one] at this

/-- **One bound for every trip.** If `S (t + 1) = g t·S t + U t` with `|g t| ≤ 1` on the trips
`t < n`, then every `|S t|`, `t ≤ n`, is at most `|S 0| + Σ_{t<n} |U t|`: the `B` of
`perturb_le_uniform` in one step. -/
theorem abs_le_sum_of_le {g U S : ℕ → ℝ} {n : ℕ} (hS : ∀ t, t < n → S (t + 1) = g t * S t + U t)
    (hG : ∀ t, t < n → |g t| ≤ 1) {t : ℕ} (ht : t ≤ n) :
    |S t| ≤ |S 0| + ∑ s ∈ range n, |U s| := by
  refine (abs_le_sum (n := t) (fun s hs => hS s (by omega)) (fun s hs => hG s (by omega))).trans ?_
  have := Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr ht)
    (f := fun s => |U s|) fun _ _ _ => abs_nonneg _
  linarith

/-! ## Gates as exponentials -/

/-- A product of exponentials is the exponential of the sum: `Π_{u∈s} exp(a u) = exp(Σ_{u∈s} a u)`
(Mathlib's `Real.exp_sum`, turned around). -/
theorem prod_exp {ι : Type*} (s : Finset ι) (a : ι → ℝ) :
    ∏ u ∈ s, Real.exp (a u) = Real.exp (∑ u ∈ s, a u) :=
  (Real.exp_sum s a).symm

/-- The differences of a sequence telescope: `Σ_{t≤u<n} (A (u + 1) − A u) = A n − A t` for
`t ≤ n`. -/
theorem sum_Ico_sub (A : ℕ → ℝ) {t n : ℕ} (h : t ≤ n) :
    ∑ u ∈ Ico t n, (A (u + 1) - A u) = A n - A t := by
  induction n, h using Nat.le_induction with
  | base => simp
  | succ n hn ih => rw [Finset.sum_Ico_succ_top hn, ih]; ring

/-- **The decays of a cumulative sum telescope**: `Π_{t≤u<n} exp(A (u + 1) − A u) = exp(A n − A t)`
for `t ≤ n`. With `t + 1` in place of `t` it is the factor `Π_{t<u<n}` of `closed`. -/
theorem prod_exp_sub (A : ℕ → ℝ) {t n : ℕ} (h : t ≤ n) :
    ∏ u ∈ Ico t n, Real.exp (A (u + 1) - A u) = Real.exp (A n - A t) := by
  rw [prod_exp, sum_Ico_sub A h]

/-- A non-positive gate decays: `a ≤ 0` gives `exp a ∈ (0, 1]`. -/
theorem exp_mem_Ioc {a : ℝ} (ha : a ≤ 0) : 0 < Real.exp a ∧ Real.exp a ≤ 1 :=
  ⟨Real.exp_pos a, Real.exp_le_one_iff.mpr ha⟩

/-- A cumulative sum of non-positive gates decreases: if `A (u + 1) ≤ A u` on `t ≤ u < n`, then
`A n ≤ A t` (for `t ≤ n`). -/
theorem cum_le {A : ℕ → ℝ} {t n : ℕ} (htn : t ≤ n)
    (h : ∀ u, t ≤ u → u < n → A (u + 1) ≤ A u) : A n ≤ A t := by
  induction n, htn using Nat.le_induction with
  | base => exact le_rfl
  | succ n hn ih =>
    exact (h n hn (Nat.lt_succ_self n)).trans (ih fun u h1 h2 => h u h1 (by omega))

/-- **Every factor of a decreasing cumulative sum is in `(0, 1]`**: if `A (u + 1) ≤ A u` on
`t ≤ u < n` (each gate `A (u + 1) − A u` is non-positive) and `t ≤ n`, then
`exp(A n − A t) ∈ (0, 1]`. It bounds the exact decay, not a computed one: a kernel with
`exp(cs[last] − cs[i])` uses it for the `|g t| ≤ G t` (with `G t = 1`) of `perturb'` followed by
`run_le_sum`, and for the `|g t| ≤ 1` of `abs_le_sum`. The computed decay `ĝ t` is a float that an
approximate `exp` may round above `1`: `perturb_le_pow` covers that case. -/
theorem exp_sub_mem_Ioc {A : ℕ → ℝ} {t n : ℕ} (htn : t ≤ n)
    (h : ∀ u, t ≤ u → u < n → A (u + 1) ≤ A u) :
    0 < Real.exp (A n - A t) ∧ Real.exp (A n - A t) ≤ 1 :=
  exp_mem_Ioc (by linarith [cum_le htn h])

/-- **The closed form with exponential decays.** If `S (t + 1) = exp(a t)·S t + U t` on the
trips `t < n`, then `S n = exp(Σ_{u<n} a u)·S 0 + Σ_{t<n} exp(Σ_{t<u<n} a u)·U t`. -/
theorem closed_exp (a U S : ℕ → ℝ) {n : ℕ}
    (h : ∀ t, t < n → S (t + 1) = Real.exp (a t) * S t + U t) :
    S n = Real.exp (∑ u ∈ range n, a u) * S 0 +
      ∑ t ∈ range n, Real.exp (∑ u ∈ Ico (t + 1) n, a u) * U t := by
  rw [closed (fun t => Real.exp (a t)) U S h, prod_exp]
  simp only [prod_exp]

/-- **The closed form with the decays of a cumulative sum.** If
`S (t + 1) = exp(A (t + 1) − A t)·S t + U t` on the trips `t < n`, then
`S n = exp(A n − A 0)·S 0 + Σ_{t<n} exp(A n − A (t + 1))·U t`. -/
theorem closed_exp_sub (A U S : ℕ → ℝ) {n : ℕ}
    (h : ∀ t, t < n → S (t + 1) = Real.exp (A (t + 1) - A t) * S t + U t) :
    S n = Real.exp (A n - A 0) * S 0 +
      ∑ t ∈ range n, Real.exp (A n - A (t + 1)) * U t := by
  rw [closed (fun t => Real.exp (A (t + 1) - A t)) U S h, Finset.range_eq_Ico,
    prod_exp_sub A (Nat.zero_le n)]
  congr 1
  refine Finset.sum_congr rfl fun t ht => ?_
  rw [prod_exp_sub A (by simp at ht; omega)]

/-- **The closed form as a chunk computes it.** Let `cs` be the inclusive cumulative sum of the
gates over the trips `t < n` (`cs i = Σ_{u≤i} a u`, Triton's `tl.cumsum`), with `0 < n`. If
`S (t + 1) = exp(a t)·S t + U t` on the trips `t < n`, then
`S n = exp(cs (n − 1))·S 0 + Σ_{t<n} exp(cs (n − 1) − cs t)·U t`: the
`exp(g_cumsum[last])·S + Σ_i exp(g_cumsum[last] − g_cumsum[i])·k_i v_iᵀ` of a chunked kernel. -/
theorem closed_cumsum (a U S cs : ℕ → ℝ) {n : ℕ} (hn : 0 < n)
    (hcs : ∀ i, i < n → cs i = ∑ u ∈ range (i + 1), a u)
    (h : ∀ t, t < n → S (t + 1) = Real.exp (a t) * S t + U t) :
    S n = Real.exp (cs (n - 1)) * S 0 +
      ∑ t ∈ range n, Real.exp (cs (n - 1) - cs t) * U t := by
  set A : ℕ → ℝ := fun i => ∑ u ∈ range i, a u with hA
  have hstep : ∀ t, A (t + 1) - A t = a t := fun t => by
    simp only [hA, Finset.sum_range_succ]; ring
  have hlast : cs (n - 1) = A n := by
    rw [hcs (n - 1) (by omega), Nat.sub_add_cancel hn]
  have e := closed_exp_sub A U S (n := n) fun t ht => by rw [hstep]; exact h t ht
  rw [e, hlast]
  simp only [hA, Finset.range_zero, Finset.sum_empty, sub_zero]
  congr 1
  refine Finset.sum_congr rfl fun t ht => ?_
  rw [hcs t (by simpa using ht)]

/-! ## A matrix state -/

section Mat

variable {κ ν : Type*}

/-- **The closed form for a matrix state**, entry by entry: a state `S t : κ → ν → ℝ` (such as
`Fin K → Fin V → ℝ`) with a scalar decay per trip, `S (t + 1) k v = g t·S t k v + U t k v` on
the trips `t < n`, has `S n k v = (Π_{u<n} g u)·S 0 k v + Σ_{t<n} (Π_{t<u<n} g u)·U t k v`.
For a decay per entry (`g t k`), apply `closed` to `fun t => S t k v`. -/
theorem closed_mat (g : ℕ → ℝ) (U S : ℕ → κ → ν → ℝ) {n : ℕ}
    (h : ∀ t, t < n → ∀ k v, S (t + 1) k v = g t * S t k v + U t k v) (k : κ) (v : ν) :
    S n k v = (∏ u ∈ range n, g u) * S 0 k v +
      ∑ t ∈ range n, (∏ u ∈ Ico (t + 1) n, g u) * U t k v :=
  closed g (fun t => U t k v) (fun t => S t k v) fun t ht => h t ht k v

/-- **The chunk's closed form for a matrix state** (`closed_cumsum` entry by entry): with `cs`
the inclusive cumulative sum of the gates and `0 < n`, if
`S (t + 1) k v = exp(a t)·S t k v + U t k v` on the trips `t < n`, then
`S n k v = exp(cs (n − 1))·S 0 k v + Σ_{t<n} exp(cs (n − 1) − cs t)·U t k v`. -/
theorem closed_cumsum_mat (a cs : ℕ → ℝ) (U S : ℕ → κ → ν → ℝ) {n : ℕ} (hn : 0 < n)
    (hcs : ∀ i, i < n → cs i = ∑ u ∈ range (i + 1), a u)
    (h : ∀ t, t < n → ∀ k v, S (t + 1) k v = Real.exp (a t) * S t k v + U t k v)
    (k : κ) (v : ν) :
    S n k v = Real.exp (cs (n - 1)) * S 0 k v +
      ∑ t ∈ range n, Real.exp (cs (n - 1) - cs t) * U t k v :=
  closed_cumsum a (fun t => U t k v) (fun t => S t k v) cs hn hcs fun t ht => h t ht k v

/-- **The perturbed matrix state** (`perturb` entry by entry): with scalar decays `g`, `ĝ`
(`|ĝ t − g t| ≤ γ t`, `|ĝ t| ≤ G t`) and per-entry errors `e t k v` (the trip's rounding) and
`ε t k v` (the update's), every entry has
`|Ŝ n k v − S n k v| ≤ run G (e·k·v + ε·k·v + γ·|S·k·v|) |Ŝ 0 k v − S 0 k v| n`. -/
theorem perturb_mat {g gh γ G : ℕ → ℝ} {U Uh S Sh e ε : ℕ → κ → ν → ℝ} {n : ℕ}
    (hS : ∀ t, t < n → ∀ k v, S (t + 1) k v = g t * S t k v + U t k v)
    (hSh : ∀ t, t < n → ∀ k v, |Sh (t + 1) k v - (gh t * Sh t k v + Uh t k v)| ≤ e t k v)
    (hg : ∀ t, t < n → |gh t - g t| ≤ γ t)
    (hU : ∀ t, t < n → ∀ k v, |Uh t k v - U t k v| ≤ ε t k v)
    (hG : ∀ t, t < n → |gh t| ≤ G t) (k : κ) (v : ν) :
    |Sh n k v - S n k v| ≤
      run G (fun t => e t k v + ε t k v + γ t * |S t k v|) |Sh 0 k v - S 0 k v| n :=
  perturb (S := fun t => S t k v) (Sh := fun t => Sh t k v) (U := fun t => U t k v)
    (Uh := fun t => Uh t k v) (e := fun t => e t k v) (ε := fun t => ε t k v)
    (fun t ht => hS t ht k v) (fun t ht => hSh t ht k v) hg (fun t ht => hU t ht k v) hG

/-- **The perturbed matrix state, decays at most `1`** (`perturb_le_sum` entry by entry): with
every computed decay `|ĝ t| ≤ 1`, every entry has
`|Ŝ n k v − S n k v| ≤ |Ŝ 0 k v − S 0 k v| + Σ_{t<n} (e t k v + ε t k v + γ t·|S t k v|)`. -/
theorem perturb_mat_le_sum {g gh γ : ℕ → ℝ} {U Uh S Sh e ε : ℕ → κ → ν → ℝ} {n : ℕ}
    (hS : ∀ t, t < n → ∀ k v, S (t + 1) k v = g t * S t k v + U t k v)
    (hSh : ∀ t, t < n → ∀ k v, |Sh (t + 1) k v - (gh t * Sh t k v + Uh t k v)| ≤ e t k v)
    (hg : ∀ t, t < n → |gh t - g t| ≤ γ t)
    (hU : ∀ t, t < n → ∀ k v, |Uh t k v - U t k v| ≤ ε t k v)
    (hG : ∀ t, t < n → |gh t| ≤ 1) (k : κ) (v : ν) :
    |Sh n k v - S n k v| ≤
      |Sh 0 k v - S 0 k v| + ∑ t ∈ range n, (e t k v + ε t k v + γ t * |S t k v|) :=
  perturb_le_sum (S := fun t => S t k v) (Sh := fun t => Sh t k v) (U := fun t => U t k v)
    (Uh := fun t => Uh t k v) (e := fun t => e t k v) (ε := fun t => ε t k v)
    (fun t ht => hS t ht k v) (fun t ht => hSh t ht k v) hg (fun t ht => hU t ht k v) hG

end Mat

end Recurrence

end TVA
