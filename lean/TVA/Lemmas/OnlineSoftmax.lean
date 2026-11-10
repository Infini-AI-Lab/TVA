import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.BigOperators.Field

/-!
# Online softmax

An attention kernel with a loop over key blocks keeps, for one query row, a running reference `m`
(the maximum so far), a running sum `l` and an accumulator `acc` (one per output column). At trip
`t` it processes a block of keys: with the new reference `m_{t+1}` and `α_t = exp(m_t − m_{t+1})`,

  `l_{t+1} = α_t·l_t + Σ_{j ∈ B t} exp(s_j − m_{t+1})`,
  `acc_{t+1} = α_t·acc_t + Σ_{j ∈ B t} exp(s_j − m_{t+1})·v_j`.

This module states the mathematics of that loop over `ℝ`. A kernel connects it to its float
values itself, with `FloatErr`, `FusedErr` and the device facts. Trip `t` works on lanes
`j ∈ B t` (a `Finset`), with score `s t j` and value `v t j` (one output column); a masked lane is
left out of `B t` (`Finset.filter`), or kept with an exact weight `p = 0` (`Finset.sum_filter`
turns one form into the other). A kernel using `exp2` takes `s := s₂·log 2` and `m := m₂·log 2`
(`two_rpow_eq_exp`).

**The exact invariant** (`lsum`, `asum`; `l_eq`, `acc_eq`, `ratio_eq`): for any references
`m_t`, `l_t = Σ_{u<t} Σ_{j ∈ B u} exp(s u j − m_t)` and `acc_t = Σ_{u<t} Σ_{j ∈ B u}
exp(s u j − m_t)·v u j`, so `acc_t / l_t` is the softmax-weighted sum `wavg`, whatever the
references (`asum_div_lsum`). The loop starts from `l_0 = acc_0 = 0`; the kernel's `m_0 = −∞`
gives `α_0 = 0`, which here multiplies `0`, so `α_0` is left free. A kernel that starts from
`l = 1` (Triton's tutorial) takes `l_0 = 0` in the reals: its first `α` is `0`. For keys in blocks
of `b` (lane `j : Fin b` of trip `u` holds the key `u·b + j`) with a mask `P`, `lsum_blocks`,
`asum_blocks`, `ratio_blocks` and `wavg_blocks` give the flat sums over `{k < n·b | P k}`
(`sum_blocks`, `sum_blocks_filter`, `sum_blocks_fin_filter` for any sum).

**The perturbed form.** A computed state `x̂` follows `X_{t+1} = α_t·X_t + Y_t` (with
`|Y_t| ≤ Z_t` and the magnitude `M_{t+1} = α_t·M_t + Z_t`) when each trip is within
`ε·(α_t·|x̂_t| + Z_t) + η_t` of `α_t·x̂_t + Y_t`:
* `accum`: then `|x̂_t − X_t| ≤ bound ε α η t M_t = ((1+ε)^t − 1)·M_t + absErr ε α η t`, where
  `absErr` carries each trip's absolute error forward, scaled by later `α`s (`absErr_alpha`:
  `Σ_u (1+ε)^(t−1−u)·exp(m_{u+1} − m_t)·η_u`). `l_err` and `acc_err` are its instances, with the
  computed references `m̂_t` (`alpha m̂`), against `lsum` and `asum` at `m̂_t`.
* `bound_le`: when `α_t ≤ 1` (references that never decrease, as a running maximum) and
  `t·ε ≤ 1/2`, the bound is at most `2tε·M + 2·Σ η`.
* `ratio_err`, `ratio_err_half`: the ratio `â/l̂` against `A/L` from the two errors and `|A| ≤ V·L`;
  `err_final` and `err_final_mono` put the pieces together: `|âcc_n / l̂_n − wavg|`.
* One trip from its parts: `scale_err` (`α̂·x`), `block_err`, `block_abs_le` (a block of lanes, each
  within `ρ·p + η` of an exact `p ≥ 0`; `block_err_one`, `block_abs_le_one` for `l`), `approx_trans`
  (one approximation after another), `abs_le_of_approx`, `step_err` (the scaled state plus the
  block's contribution, through an addition, an `fma` or a `tl.dot` accumulation), and `l_trip`
  (the sum's trip `l·α̂ + Σ p̂`, assembled).

**Exponential helpers:** `abs_exp_sub_exp_le` (`|exp a − exp b| ≤ |a − b|·exp(max a b)`),
`abs_exp_sub_one_le_mul` (`|exp d − 1| ≤ |d|·(1 + |d|)` for `|d| ≤ 1`; Mathlib's
`Real.abs_exp_sub_one_le` gives `2|d|`), `abs_exp_sub_exp_le_mul` (a relative error from an argument
error), `abs_exp_round_le` (the exponential of a rounded non-positive argument: relative plus
absolute), `mul_exp_neg_le`, `exp_le_one_add_two_mul`, `one_add_pow_le`, and `exp2`:
`two_rpow_eq_exp`, `exp_le_two_rpow`, `exp_le_two_zpow`, `two_zpow_le_exp`.
-/

namespace TVA

namespace OnlineSoftmax

open Finset

/-! ## Exponential helpers -/

/-- `|exp a − exp b| ≤ |a − b|·exp (max a b)`: the exponential is `exp (max a b)`-Lipschitz between
`a` and `b`. -/
theorem abs_exp_sub_exp_le (a b : ℝ) :
    |Real.exp a - Real.exp b| ≤ |a - b| * Real.exp (max a b) := by
  wlog h : b ≤ a generalizing a b
  · rw [abs_sub_comm, abs_sub_comm a, max_comm]; exact this b a (not_le.mp h).le
  rw [max_eq_left h, abs_of_nonneg (sub_nonneg.mpr h),
    abs_of_nonneg (sub_nonneg.mpr (Real.exp_le_exp.mpr h))]
  have h1 : b - a + 1 ≤ Real.exp (b - a) := Real.add_one_le_exp _
  have h2 : Real.exp b = Real.exp a * Real.exp (b - a) := by rw [← Real.exp_add]; congr 1; ring
  rw [h2]
  have h3 := mul_le_mul_of_nonneg_left h1 (Real.exp_pos a).le
  nlinarith

/-- `|exp d − 1| ≤ |d|·(1 + |d|)` for `|d| ≤ 1`. Mathlib's `Real.abs_exp_sub_one_le` is the
weaker `|exp d − 1| ≤ 2|d|`. -/
theorem abs_exp_sub_one_le_mul {d : ℝ} (h : |d| ≤ 1) : |Real.exp d - 1| ≤ |d| * (1 + |d|) := by
  have h1 := Real.abs_exp_sub_one_sub_id_le h
  have h2 : |Real.exp d - 1| ≤ |Real.exp d - 1 - d| + |d| := by
    have := abs_add_le (Real.exp d - 1 - d) d
    rwa [sub_add_cancel] at this
  have h3 : d ^ 2 = |d| * |d| := by rw [abs_mul_abs_self d, sq]
  nlinarith

/-- `exp d ≤ 1 + 2d` for `0 ≤ d ≤ 1`. -/
theorem exp_le_one_add_two_mul {d : ℝ} (h0 : 0 ≤ d) (h1 : d ≤ 1) : Real.exp d ≤ 1 + 2 * d := by
  have := (abs_le.mp (Real.abs_exp_sub_one_le (x := d) (by rw [abs_of_nonneg h0]; exact h1))).2
  rw [abs_of_nonneg h0] at this
  linarith

/-- **A relative error from an argument error.** If `|y − x| ≤ δ ≤ 1`, then
`|exp y − exp x| ≤ δ·(1 + δ)·exp x` (so at most `2δ·exp x`). -/
theorem abs_exp_sub_exp_le_mul {x y δ : ℝ} (h : |y - x| ≤ δ) (hδ : δ ≤ 1) :
    |Real.exp y - Real.exp x| ≤ δ * (1 + δ) * Real.exp x := by
  have e : Real.exp y - Real.exp x = Real.exp x * (Real.exp (y - x) - 1) := by
    have : Real.exp y = Real.exp x * Real.exp (y - x) := by rw [← Real.exp_add]; congr 1; ring
    rw [this]; ring
  rw [e, abs_mul, abs_of_pos (Real.exp_pos x)]
  have h1 := abs_exp_sub_one_le_mul (h.trans hδ)
  have h2 : |Real.exp (y - x) - 1| ≤ δ * (1 + δ) :=
    h1.trans (mul_le_mul h (by linarith) (by positivity) ((abs_nonneg _).trans h))
  have := mul_le_mul_of_nonneg_left h2 (Real.exp_pos x).le
  linarith

/-- `y·exp(−y) ≤ K·exp(−K)` for `1 ≤ K ≤ y`: `y·exp(−y)` decreases from `y = 1`. -/
theorem mul_exp_neg_le {K y : ℝ} (hK : 1 ≤ K) (hy : K ≤ y) :
    y * Real.exp (-y) ≤ K * Real.exp (-K) := by
  have hK0 : 0 < K := by linarith
  have h1 : y ≤ K * (1 + (y - K)) := by nlinarith [mul_nonneg (sub_nonneg.mpr hK) (sub_nonneg.mpr hy)]
  have h2 : y ≤ K * Real.exp (y - K) :=
    h1.trans (mul_le_mul_of_nonneg_left (by linarith [Real.add_one_le_exp (y - K)]) hK0.le)
  have h3 : Real.exp (y - K) * Real.exp (-y) = Real.exp (-K) := by
    rw [← Real.exp_add]; congr 1; ring
  calc y * Real.exp (-y) ≤ K * Real.exp (y - K) * Real.exp (-y) :=
        mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
    _ = K * Real.exp (-K) := by rw [mul_assoc, h3]

/-- **The exponential of a rounded non-positive argument.** If `x ≤ 0` and `y` is within
`u·|x| + δ` of `x` (one rounding of `x`: relative `u ≤ 1/2`, absolute `δ ≥ 0`), then for any
`K ≥ 2` with `c = u·K + δ ≤ 1/2`, `|exp y − exp x| ≤ 2c·exp x + 2c·exp(−K/2)`: relative where
`|x| ≤ K`, absolute beyond. A kernel computing `exp(s − m)` from a rounded `s − m` (or `exp2` of it,
through `two_rpow_eq_exp`) uses it, with `K` chosen to balance the two terms. -/
theorem abs_exp_round_le {x y u δ K : ℝ} (hx : x ≤ 0) (hu : 0 ≤ u) (hu1 : u ≤ 1 / 2) (hδ : 0 ≤ δ)
    (hK : 2 ≤ K) (hc : u * K + δ ≤ 1 / 2) (h : |y - x| ≤ u * |x| + δ) :
    |Real.exp y - Real.exp x| ≤
      2 * (u * K + δ) * Real.exp x + 2 * (u * K + δ) * Real.exp (-(K / 2)) := by
  have hax : |x| = -x := abs_of_nonpos hx
  have hux0 : 0 ≤ u * |x| := mul_nonneg hu (abs_nonneg x)
  have hmax : max y x ≤ x + u * |x| + δ := by
    have : y ≤ x + u * |x| + δ := by have := (abs_le.mp h).2; linarith
    exact max_le this (by linarith)
  have h1 : |Real.exp y - Real.exp x| ≤ (u * |x| + δ) * Real.exp (x + u * |x| + δ) :=
    (abs_exp_sub_exp_le y x).trans
      (mul_le_mul h (Real.exp_le_exp.mpr hmax) (Real.exp_pos _).le (by linarith))
  have huK0 : 0 ≤ u * K := mul_nonneg hu (by linarith)
  have hc0 : 0 ≤ u * K + δ := by linarith
  have hδ1 : δ ≤ 1 / 2 := by linarith
  have hex := Real.exp_pos x
  have hA : 0 ≤ 2 * (u * K + δ) * Real.exp (-(K / 2)) :=
    mul_nonneg (by linarith) (Real.exp_pos _).le
  have hB : 0 ≤ 2 * (u * K + δ) * Real.exp x := mul_nonneg (by linarith) hex.le
  by_cases hxK : |x| ≤ K
  · have h4 : u * |x| + δ ≤ u * K + δ := by nlinarith [mul_le_mul_of_nonneg_left hxK hu]
    have h3 : Real.exp (u * |x| + δ) ≤ 2 := by
      have := exp_le_one_add_two_mul (d := u * |x| + δ) (by linarith) (by linarith)
      linarith
    have e2 : Real.exp (x + u * |x| + δ) = Real.exp x * Real.exp (u * |x| + δ) := by
      rw [← Real.exp_add]; congr 1; ring
    calc |Real.exp y - Real.exp x| ≤ (u * |x| + δ) * (Real.exp x * Real.exp (u * |x| + δ)) := by
          rw [← e2]; exact h1
      _ ≤ (u * K + δ) * (Real.exp x * 2) :=
          mul_le_mul h4 (mul_le_mul_of_nonneg_left h3 hex.le)
            (mul_nonneg hex.le (Real.exp_pos _).le) hc0
      _ ≤ _ := by linarith
  · have hxK : K < |x| := not_le.mp hxK
    have hw1 : K / 2 ≤ (1 - u) * |x| := by
      have : 0 ≤ (1 / 2 - u) * |x| := mul_nonneg (by linarith) (abs_nonneg x)
      linarith
    have he : Real.exp (x + u * |x| + δ) = Real.exp δ * Real.exp (-((1 - u) * |x|)) := by
      rw [← Real.exp_add]; congr 1; rw [hax]; ring
    have hδ2 : Real.exp δ ≤ 2 := by
      have := exp_le_one_add_two_mul hδ (by linarith); linarith
    have h5 := mul_exp_neg_le (by linarith : (1 : ℝ) ≤ K / 2) hw1
    have h6 : u * |x| ≤ 2 * u * ((1 - u) * |x|) := by
      have : 0 ≤ u * |x| * (1 - 2 * u) := mul_nonneg hux0 (by linarith)
      linarith
    have h7 : Real.exp (-((1 - u) * |x|)) ≤ Real.exp (-(K / 2)) := Real.exp_le_exp.mpr (by linarith)
    have hw0 : 0 ≤ Real.exp (-((1 - u) * |x|)) := (Real.exp_pos _).le
    have h8 : (u * |x| + δ) * Real.exp (-((1 - u) * |x|)) ≤ (u * K + δ) * Real.exp (-(K / 2)) := by
      have e1 : u * |x| * Real.exp (-((1 - u) * |x|)) ≤ u * K * Real.exp (-(K / 2)) := by
        calc u * |x| * Real.exp (-((1 - u) * |x|))
            ≤ 2 * u * ((1 - u) * |x|) * Real.exp (-((1 - u) * |x|)) :=
              mul_le_mul_of_nonneg_right h6 hw0
          _ = 2 * u * ((1 - u) * |x| * Real.exp (-((1 - u) * |x|))) := by ring
          _ ≤ 2 * u * (K / 2 * Real.exp (-(K / 2))) := mul_le_mul_of_nonneg_left h5 (by linarith)
          _ = u * K * Real.exp (-(K / 2)) := by ring
      have e2 := mul_le_mul_of_nonneg_left h7 hδ
      nlinarith
    calc |Real.exp y - Real.exp x|
        ≤ (u * |x| + δ) * (Real.exp δ * Real.exp (-((1 - u) * |x|))) := by rw [← he]; exact h1
      _ = Real.exp δ * ((u * |x| + δ) * Real.exp (-((1 - u) * |x|))) := by ring
      _ ≤ 2 * ((u * K + δ) * Real.exp (-(K / 2))) :=
          mul_le_mul hδ2 h8 (mul_nonneg (by linarith) hw0) (by norm_num)
      _ ≤ _ := by linarith

/-- `(1 + x)^n ≤ 1 + 2nx` for `0 ≤ x` and `n·x ≤ 1`. -/
theorem one_add_pow_le {x : ℝ} (hx : 0 ≤ x) {n : ℕ} (h : n * x ≤ 1) :
    (1 + x) ^ n ≤ 1 + 2 * n * x := by
  have h1 : (1 + x) ^ n ≤ Real.exp x ^ n :=
    pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp x]) n
  rw [← Real.exp_nat_mul] at h1
  have hy0 : 0 ≤ (n : ℝ) * x := mul_nonneg (Nat.cast_nonneg n) hx
  have h3 := Real.abs_exp_sub_one_sub_id_le (x := n * x) (by rw [abs_of_nonneg hy0]; exact h)
  have h4 : ((n : ℝ) * x) ^ 2 ≤ n * x := by nlinarith
  have h5 := (abs_le.mp h3).2
  linarith

/-- **`exp2` as an exponential:** `2^x = exp (x·log 2)`, for the real power `(2 : ℝ) ^ x` with which
`MathFn.ref .exp2` states `tl.math.exp2`. -/
theorem two_rpow_eq_exp (x : ℝ) : (2 : ℝ) ^ x = Real.exp (x * Real.log 2) := by
  rw [Real.rpow_def_of_pos (by norm_num), mul_comm]

/-- `exp x ≤ 2^y` when `x ≤ y·log 2`. -/
theorem exp_le_two_rpow {x y : ℝ} (h : x ≤ y * Real.log 2) : Real.exp x ≤ (2 : ℝ) ^ y := by
  rw [two_rpow_eq_exp]; exact Real.exp_le_exp.mpr h

/-- `exp x ≤ 2^k` (an integer power) when `x ≤ k·log 2`. -/
theorem exp_le_two_zpow {x : ℝ} {k : ℤ} (h : x ≤ k * Real.log 2) : Real.exp x ≤ (2 : ℝ) ^ k := by
  rw [← Real.rpow_intCast]; exact exp_le_two_rpow h

/-- `2^k ≤ exp x` (an integer power) when `k·log 2 ≤ x`. -/
theorem two_zpow_le_exp {x : ℝ} {k : ℤ} (h : k * Real.log 2 ≤ x) : (2 : ℝ) ^ k ≤ Real.exp x := by
  rw [← Real.rpow_intCast, two_rpow_eq_exp]; exact Real.exp_le_exp.mpr h

/-! ## The softmax-weighted sum -/

section Softmax

variable {κ : Type*}

/-- The softmax over a finite set of keys `K`: `exp (S k) / Σ_{k' ∈ K} exp (S k')`. -/
noncomputable def softmax (K : Finset κ) (S : κ → ℝ) (k : κ) : ℝ :=
  Real.exp (S k) / ∑ k' ∈ K, Real.exp (S k')

/-- The softmax-weighted sum `Σ_{k ∈ K} softmax K S k · V k`: one output element of attention,
for scores `S` and values `V` (one column). -/
noncomputable def wavg (K : Finset κ) (S V : κ → ℝ) : ℝ := ∑ k ∈ K, softmax K S k * V k

/-- Softmax weights are non-negative. -/
theorem softmax_nonneg (K : Finset κ) (S : κ → ℝ) (k : κ) : 0 ≤ softmax K S k :=
  div_nonneg (Real.exp_pos _).le (sum_nonneg fun _ _ => (Real.exp_pos _).le)

/-- The softmax weights of a nonempty set add up to `1`. -/
theorem sum_softmax {K : Finset κ} (hK : K.Nonempty) (S : κ → ℝ) : ∑ k ∈ K, softmax K S k = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (sum_pos (fun _ _ => Real.exp_pos _) hK).ne'

/-- `wavg` as one ratio: `(Σ exp (S k)·V k) / Σ exp (S k)`. -/
theorem wavg_eq_div (K : Finset κ) (S V : κ → ℝ) :
    wavg K S V = (∑ k ∈ K, Real.exp (S k) * V k) / ∑ k ∈ K, Real.exp (S k) := by
  unfold wavg softmax
  rw [Finset.sum_div]
  exact sum_congr rfl fun k _ => by ring

/-- **The reference does not matter:** for any `m`,
`wavg K S V = (Σ exp (S k − m)·V k) / Σ exp (S k − m)`. -/
theorem wavg_eq_div_sub (K : Finset κ) (S V : κ → ℝ) (m : ℝ) :
    wavg K S V = (∑ k ∈ K, Real.exp (S k - m) * V k) / ∑ k ∈ K, Real.exp (S k - m) := by
  rw [wavg_eq_div]
  have hn : ∑ k ∈ K, Real.exp (S k - m) * V k = (∑ k ∈ K, Real.exp (S k) * V k) * Real.exp (-m) := by
    rw [Finset.sum_mul]
    exact sum_congr rfl fun k _ => by rw [sub_eq_add_neg, Real.exp_add]; ring
  have hd : ∑ k ∈ K, Real.exp (S k - m) = (∑ k ∈ K, Real.exp (S k)) * Real.exp (-m) := by
    rw [Finset.sum_mul]
    exact sum_congr rfl fun k _ => by rw [sub_eq_add_neg, Real.exp_add]
  rw [hn, hd, mul_div_mul_right _ _ (Real.exp_pos _).ne']

/-- `|wavg K S V| ≤ c` when every `|V k| ≤ c` (`k ∈ K`). -/
theorem abs_wavg_le {K : Finset κ} {S V : κ → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hV : ∀ k ∈ K, |V k| ≤ c) : |wavg K S V| ≤ c := by
  rcases K.eq_empty_or_nonempty with hK | hK
  · simp [wavg, hK, hc]
  · calc |wavg K S V| ≤ ∑ k ∈ K, |softmax K S k * V k| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ k ∈ K, softmax K S k * c := sum_le_sum fun k hk => by
          rw [abs_mul, abs_of_nonneg (softmax_nonneg K S k)]
          exact mul_le_mul_of_nonneg_left (hV k hk) (softmax_nonneg K S k)
      _ = c := by rw [← Finset.sum_mul, sum_softmax hK, one_mul]

end Softmax

/-! ## The exact invariant -/

section Exact

variable {ι : Type*}

/-- **The running sum in closed form:** `Σ_{u<t} Σ_{j ∈ B u} exp (s u j − m)`, the sum `l` after
`t` trips at the reference `m`. Trip `u` has lanes `B u`, scores `s u j`. -/
noncomputable def lsum (s : ℕ → ι → ℝ) (B : ℕ → Finset ι) (m : ℝ) (t : ℕ) : ℝ :=
  ∑ u ∈ range t, ∑ j ∈ B u, Real.exp (s u j - m)

/-- **The accumulator in closed form:** `Σ_{u<t} Σ_{j ∈ B u} exp (s u j − m)·v u j`, the
accumulator `acc` (one output column, values `v u j`) after `t` trips at the reference `m`. -/
noncomputable def asum (s v : ℕ → ι → ℝ) (B : ℕ → Finset ι) (m : ℝ) (t : ℕ) : ℝ :=
  ∑ u ∈ range t, ∑ j ∈ B u, Real.exp (s u j - m) * v u j

variable {s v : ℕ → ι → ℝ} {B : ℕ → Finset ι}

/-- No trip, no sum. -/
@[simp] theorem lsum_zero (m : ℝ) : lsum s B m 0 = 0 := by simp [lsum]

/-- No trip, no accumulator. -/
@[simp] theorem asum_zero (m : ℝ) : asum s v B m 0 = 0 := by simp [asum]

/-- The running sum is non-negative. -/
theorem lsum_nonneg (m : ℝ) (t : ℕ) : 0 ≤ lsum s B m t :=
  sum_nonneg fun _ _ => sum_nonneg fun _ _ => (Real.exp_pos _).le

/-- **A change of reference** rescales the sum: `lsum m' = exp (m − m')·lsum m`. -/
theorem lsum_rescale (m m' : ℝ) (t : ℕ) : lsum s B m' t = Real.exp (m - m') * lsum s B m t := by
  unfold lsum
  rw [mul_sum]
  refine sum_congr rfl fun u _ => ?_
  rw [mul_sum]
  refine sum_congr rfl fun j _ => ?_
  rw [← Real.exp_add]; congr 1; ring

/-- **A change of reference** rescales the accumulator: `asum m' = exp (m − m')·asum m`. -/
theorem asum_rescale (m m' : ℝ) (t : ℕ) : asum s v B m' t = Real.exp (m - m') * asum s v B m t := by
  unfold asum
  rw [mul_sum]
  refine sum_congr rfl fun u _ => ?_
  rw [mul_sum]
  refine sum_congr rfl fun j _ => ?_
  rw [← mul_assoc, ← Real.exp_add]; congr 2; ring

/-- **One trip of the sum:** from the reference `m` to `m'`,
`lsum m' (t+1) = exp (m − m')·lsum m t + Σ_{j ∈ B t} exp (s t j − m')`. -/
theorem lsum_succ (m m' : ℝ) (t : ℕ) :
    lsum s B m' (t + 1) = Real.exp (m - m') * lsum s B m t + ∑ j ∈ B t, Real.exp (s t j - m') := by
  rw [← lsum_rescale]; unfold lsum; rw [sum_range_succ]

/-- **One trip of the accumulator:** from the reference `m` to `m'`,
`asum m' (t+1) = exp (m − m')·asum m t + Σ_{j ∈ B t} exp (s t j − m')·v t j`. -/
theorem asum_succ (m m' : ℝ) (t : ℕ) :
    asum s v B m' (t + 1) =
      Real.exp (m - m') * asum s v B m t + ∑ j ∈ B t, Real.exp (s t j - m') * v t j := by
  rw [← asum_rescale]; unfold asum; rw [sum_range_succ]

/-- `|asum v| ≤ asum |v|`. -/
theorem abs_asum_le (m : ℝ) (t : ℕ) : |asum s v B m t| ≤ asum s (fun u j => |v u j|) B m t := by
  unfold asum
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun u _ =>
    (abs_sum_le_sum_abs _ _).trans (le_of_eq (sum_congr rfl fun j _ => ?_)))
  rw [abs_mul, abs_of_pos (Real.exp_pos _)]

/-- `asum |v| ≤ V·lsum` when every `|v u j| ≤ V` on the lanes. -/
theorem asum_abs_le {m V : ℝ} {t : ℕ} (hV : ∀ u < t, ∀ j ∈ B u, |v u j| ≤ V) :
    asum s (fun u j => |v u j|) B m t ≤ V * lsum s B m t := by
  unfold asum lsum
  rw [mul_sum]
  refine sum_le_sum fun u hu => ?_
  rw [mul_sum]
  refine sum_le_sum fun j hj => ?_
  rw [mul_comm V]
  exact mul_le_mul_of_nonneg_left (hV u (mem_range.mp hu) j hj) (Real.exp_pos _).le

/-- One term is at most the sum: `exp (s u j − m) ≤ lsum m t` for `u < t`, `j ∈ B u`. -/
theorem exp_le_lsum {m : ℝ} {t u : ℕ} {j : ι} (hu : u < t) (hj : j ∈ B u) :
    Real.exp (s u j - m) ≤ lsum s B m t := by
  unfold lsum
  calc Real.exp (s u j - m) ≤ ∑ j' ∈ B u, Real.exp (s u j' - m) :=
        single_le_sum (f := fun j' => Real.exp (s u j' - m)) (fun _ _ => (Real.exp_pos _).le) hj
    _ ≤ ∑ u' ∈ range t, ∑ j' ∈ B u', Real.exp (s u' j' - m) :=
        single_le_sum (f := fun u' => ∑ j' ∈ B u', Real.exp (s u' j' - m))
          (fun _ _ => sum_nonneg fun _ _ => (Real.exp_pos _).le) (mem_range.mpr hu)

/-- `1 ≤ lsum m t` when some lane seen has a score at least `m` (for example, `m` is the maximum). -/
theorem one_le_lsum {m : ℝ} {t u : ℕ} {j : ι} (hu : u < t) (hj : j ∈ B u) (h : m ≤ s u j) :
    1 ≤ lsum s B m t :=
  (Real.one_le_exp_iff.mpr (sub_nonneg.mpr h)).trans (exp_le_lsum hu hj)

/-- `0 < lsum m t` when some trip before `t` has a lane. -/
theorem lsum_pos {m : ℝ} {t u : ℕ} (hu : u < t) (hB : (B u).Nonempty) : 0 < lsum s B m t := by
  obtain ⟨j, hj⟩ := hB
  exact (Real.exp_pos _).trans_le (exp_le_lsum hu hj)

/-- **Every term at most `1`:** when `m` is at least every score seen, `lsum m t` is at most the
number of lanes. -/
theorem lsum_le_card {m : ℝ} {t : ℕ} (h : ∀ u < t, ∀ j ∈ B u, s u j ≤ m) :
    lsum s B m t ≤ ∑ u ∈ range t, ((B u).card : ℝ) := by
  unfold lsum
  refine sum_le_sum fun u hu => ?_
  calc ∑ j ∈ B u, Real.exp (s u j - m) ≤ ∑ j ∈ B u, (1 : ℝ) :=
        sum_le_sum fun j hj => Real.exp_le_one_iff.mpr (by linarith [h u (mem_range.mp hu) j hj])
    _ = (B u).card := by simp

/-- `lsum` as one sum over the pairs (trip, lane). -/
theorem lsum_eq_sum_sigma (m : ℝ) (t : ℕ) :
    lsum s B m t = ∑ x ∈ (range t).sigma B, Real.exp (s x.1 x.2 - m) := by
  unfold lsum; rw [sum_sigma]

/-- `asum` as one sum over the pairs (trip, lane). -/
theorem asum_eq_sum_sigma (m : ℝ) (t : ℕ) :
    asum s v B m t = ∑ x ∈ (range t).sigma B, Real.exp (s x.1 x.2 - m) * v x.1 x.2 := by
  unfold asum; rw [sum_sigma]

/-- **The ratio is the softmax-weighted sum**, whatever the reference `m`:
`asum m t / lsum m t = wavg` over the pairs (trip, lane) seen in `t` trips. -/
theorem asum_div_lsum (m : ℝ) (t : ℕ) :
    asum s v B m t / lsum s B m t =
      wavg ((range t).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2) := by
  rw [wavg_eq_div_sub _ _ _ m, asum_eq_sum_sigma, lsum_eq_sum_sigma]

/-- **The exact running sum.** A sequence `l` with `l 0 = 0` and
`l (t+1) = α t·l t + Σ_{j ∈ B t} exp (s t j − m (t+1))`, where `α t = exp (m t − m (t+1))` for
`t > 0` (`α 0` is free: it multiplies `l 0 = 0`), is `lsum s B (m t) t` for every `t ≤ n`, for any
references `m`. -/
theorem l_eq {m α l : ℕ → ℝ} {n : ℕ} (h0 : l 0 = 0)
    (hα : ∀ t, 0 < t → t < n → α t = Real.exp (m t - m (t + 1)))
    (hl : ∀ t < n, l (t + 1) = α t * l t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1))) :
    ∀ t ≤ n, l t = lsum s B (m t) t := by
  intro t
  induction t with
  | zero => intro _; simp [h0]
  | succ t ih =>
    intro ht
    rw [hl t (by omega), lsum_succ (m t)]
    rcases Nat.eq_zero_or_pos t with rfl | htp
    · simp [h0]
    · rw [hα t htp (by omega), ih (by omega)]

/-- **The exact accumulator.** A sequence `acc` with `acc 0 = 0` and
`acc (t+1) = α t·acc t + Σ_{j ∈ B t} exp (s t j − m (t+1))·v t j`, where
`α t = exp (m t − m (t+1))` for `t > 0`, is `asum s v B (m t) t` for every `t ≤ n`. -/
theorem acc_eq {m α acc : ℕ → ℝ} {n : ℕ} (h0 : acc 0 = 0)
    (hα : ∀ t, 0 < t → t < n → α t = Real.exp (m t - m (t + 1)))
    (ha : ∀ t < n, acc (t + 1) = α t * acc t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1)) * v t j) :
    ∀ t ≤ n, acc t = asum s v B (m t) t := by
  intro t
  induction t with
  | zero => intro _; simp [h0]
  | succ t ih =>
    intro ht
    rw [ha t (by omega), asum_succ (m t)]
    rcases Nat.eq_zero_or_pos t with rfl | htp
    · simp [h0]
    · rw [hα t htp (by omega), ih (by omega)]

/-- **The online softmax computes the softmax-weighted sum.** With the recurrences of `l_eq` and
`acc_eq` (the same `α` and references `m`), `acc t / l t` is `wavg` over the pairs (trip, lane)
seen, for every `t ≤ n`. -/
theorem ratio_eq {m α l acc : ℕ → ℝ} {n : ℕ} (hl0 : l 0 = 0) (ha0 : acc 0 = 0)
    (hα : ∀ t, 0 < t → t < n → α t = Real.exp (m t - m (t + 1)))
    (hl : ∀ t < n, l (t + 1) = α t * l t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1)))
    (ha : ∀ t < n, acc (t + 1) = α t * acc t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1)) * v t j) :
    ∀ t ≤ n, acc t / l t = wavg ((range t).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2) := by
  intro t ht
  rw [l_eq hl0 hα hl t ht, acc_eq ha0 hα ha t ht, asum_div_lsum]

end Exact

/-! ## Keys in blocks -/

/-- Consecutive blocks of `b` make one range: `Σ_{u<n} Σ_{j<b} g (u·b + j) = Σ_{k<n·b} g k`. -/
theorem sum_blocks {M : Type*} [AddCommMonoid M] (g : ℕ → M) (b : ℕ) : ∀ n : ℕ,
    ∑ u ∈ range n, ∑ j ∈ range b, g (u * b + j) = ∑ k ∈ range (n * b), g k
  | 0 => by simp
  | n + 1 => by
    rw [sum_range_succ, sum_blocks g b n, show (n + 1) * b = n * b + b by ring, sum_range_add]

/-- The same with a mask `P` on the key `u·b + j`:
`Σ_{u<n} Σ_{j<b, P (u·b+j)} g (u·b + j) = Σ_{k<n·b, P k} g k`. -/
theorem sum_blocks_filter {M : Type*} [AddCommMonoid M] (P : ℕ → Prop) [DecidablePred P]
    (g : ℕ → M) (b n : ℕ) :
    ∑ u ∈ range n, ∑ j ∈ (range b).filter (fun j => P (u * b + j)), g (u * b + j) =
      ∑ k ∈ (range (n * b)).filter P, g k := by
  have h := sum_blocks (fun k => if P k then g k else 0) b n
  rw [sum_filter, ← h]
  exact sum_congr rfl fun u _ => sum_filter _ _

/-- The same with lanes `j : Fin b` (a tensor's lanes):
`Σ_{u<n} Σ_{j : Fin b, P (u·b+j)} g (u·b + j) = Σ_{k<n·b, P k} g k`. -/
theorem sum_blocks_fin_filter {M : Type*} [AddCommMonoid M] (P : ℕ → Prop) [DecidablePred P]
    (g : ℕ → M) (b n : ℕ) :
    ∑ u ∈ range n, ∑ j ∈ univ.filter (fun j : Fin b => P (u * b + j)), g (u * b + j) =
      ∑ k ∈ (range (n * b)).filter P, g k := by
  rw [← sum_blocks_filter]
  refine sum_congr rfl fun u _ => ?_
  rw [sum_filter, sum_filter]
  exact Fin.sum_univ_eq_sum_range (fun j => if P (u * b + j) then g (u * b + j) else 0) b

/-- **`lsum` over keys in blocks of `b`:** trip `u`, lane `j : Fin b` holds the key `u·b + j`, kept
when `P` holds (`B u = {j | P (u·b + j)}`, scores `S (u·b + j)`); the sum is the flat
`Σ_{k < n·b, P k} exp (S k − m)`. -/
theorem lsum_blocks (P : ℕ → Prop) [DecidablePred P] (S : ℕ → ℝ) (b : ℕ) (m : ℝ) (n : ℕ) :
    lsum (fun u (j : Fin b) => S (u * b + j)) (fun u => univ.filter (fun j : Fin b => P (u * b + j)))
        m n =
      ∑ k ∈ (range (n * b)).filter P, Real.exp (S k - m) :=
  sum_blocks_fin_filter P (fun k => Real.exp (S k - m)) b n

/-- **`asum` over keys in blocks of `b`:** the flat sum `Σ_{k < n·b, P k} exp (S k − m)·V k`. -/
theorem asum_blocks (P : ℕ → Prop) [DecidablePred P] (S V : ℕ → ℝ) (b : ℕ) (m : ℝ) (n : ℕ) :
    asum (fun u (j : Fin b) => S (u * b + j)) (fun u (j : Fin b) => V (u * b + j))
        (fun u => univ.filter (fun j : Fin b => P (u * b + j))) m n =
      ∑ k ∈ (range (n * b)).filter P, Real.exp (S k - m) * V k :=
  sum_blocks_fin_filter P (fun k => Real.exp (S k - m) * V k) b n

/-- **The ratio over keys in blocks of `b`** is the softmax-weighted sum over the keys
`{k < n·b | P k}` (for example `P k := k < N`, or `k < N ∧ k ≤ q` for a causal row `q`). -/
theorem ratio_blocks (P : ℕ → Prop) [DecidablePred P] (S V : ℕ → ℝ) (b : ℕ) (m : ℝ) (n : ℕ) :
    asum (fun u (j : Fin b) => S (u * b + j)) (fun u (j : Fin b) => V (u * b + j))
        (fun u => univ.filter (fun j : Fin b => P (u * b + j))) m n /
      lsum (fun u (j : Fin b) => S (u * b + j))
        (fun u => univ.filter (fun j : Fin b => P (u * b + j))) m n =
      wavg ((range (n * b)).filter P) S V := by
  rw [asum_blocks, lsum_blocks, wavg_eq_div_sub _ _ _ m]

/-- **The softmax-weighted sum over the pairs (trip, lane), for keys in blocks of `b`,** is the
one over the flat keys `{k < n·b | P k}`: it turns the target of `ratio_eq`, `err_final` and
`err_final_mono` into the flat specification. Rewriting needs `P` given explicitly
(`wavg_blocks (fun k => k < N ∧ k ≤ q)`): `rw` cannot infer it from the filter. -/
theorem wavg_blocks (P : ℕ → Prop) [DecidablePred P] (S V : ℕ → ℝ) (b n : ℕ) :
    wavg ((range n).sigma (fun u => univ.filter (fun j : Fin b => P (u * b + j))))
        (fun x => S (x.1 * b + x.2)) (fun x => V (x.1 * b + x.2)) =
      wavg ((range (n * b)).filter P) S V := by
  rw [← asum_div_lsum (s := fun u (j : Fin b) => S (u * b + j))
    (v := fun u (j : Fin b) => V (u * b + j)) 0 n, ratio_blocks]

/-! ## The perturbed recurrence -/

/-- The absolute part of the running bound: `absErr 0 = 0`,
`absErr (t+1) = (1 + ε)·α t·absErr t + η t`. Each trip's absolute error `η t` is carried to
later trips, scaled by their `α` (`absErr_alpha`). -/
noncomputable def absErr (ε : ℝ) (α η : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => (1 + ε) * α t * absErr ε α η t + η t

/-- **The running bound** after `t` trips, for a magnitude `M`:
`((1 + ε)^t − 1)·M + absErr ε α η t`. -/
noncomputable def bound (ε : ℝ) (α η : ℕ → ℝ) (t : ℕ) (M : ℝ) : ℝ :=
  ((1 + ε) ^ t - 1) * M + absErr ε α η t

/-- The rescaling factor of trip `t` for references `m`: `α t = exp (m t − m (t+1))`. -/
noncomputable def alpha (m : ℕ → ℝ) (t : ℕ) : ℝ := Real.exp (m t - m (t + 1))

/-- The rescaling factor is positive. -/
theorem alpha_pos (m : ℕ → ℝ) (t : ℕ) : 0 < alpha m t := Real.exp_pos _

/-- `α t ≤ 1` when the reference does not decrease at trip `t` (a running maximum). -/
theorem alpha_le_one {m : ℕ → ℝ} {t : ℕ} (h : m t ≤ m (t + 1)) : alpha m t ≤ 1 :=
  Real.exp_le_one_iff.mpr (by linarith)

/-- The absolute part is non-negative for non-negative factors and errors. -/
theorem absErr_nonneg {ε : ℝ} {α η : ℕ → ℝ} (hε : 0 ≤ ε) :
    ∀ {t : ℕ}, (∀ u < t, 0 ≤ α u) → (∀ u < t, 0 ≤ η u) → 0 ≤ absErr ε α η t
  | 0, _, _ => le_refl _
  | t + 1, hα, hη => by
    simp only [absErr]
    have hc := absErr_nonneg hε (t := t) (fun u hu => hα u (by omega)) (fun u hu => hη u (by omega))
    exact add_nonneg (mul_nonneg (mul_nonneg (by linarith) (hα t (by omega))) hc) (hη t (by omega))

/-- **The absolute part when `α ≤ 1`:** if `0 ≤ α u ≤ 1` (for `0 < u`; `α 0` multiplies `0`) and
`η u ≥ 0`, then `absErr ε α η t ≤ (1 + ε)^t·Σ_{u<t} η u`. -/
theorem absErr_le {ε : ℝ} {α η : ℕ → ℝ} (hε : 0 ≤ ε) :
    ∀ {t : ℕ}, (∀ u < t, 0 ≤ α u) → (∀ u, 0 < u → u < t → α u ≤ 1) → (∀ u < t, 0 ≤ η u) →
      absErr ε α η t ≤ (1 + ε) ^ t * ∑ u ∈ range t, η u
  | 0, _, _, _ => by simp [absErr]
  | t + 1, hα0, hα1, hη => by
    simp only [absErr]
    have ih := absErr_le hε (t := t) (fun u hu => hα0 u (by omega))
      (fun u hu hu' => hα1 u hu (by omega)) (fun u hu => hη u (by omega))
    have hc := absErr_nonneg hε (t := t) (fun u hu => hα0 u (by omega)) (fun u hu => hη u (by omega))
    have hP : 1 ≤ (1 + ε) ^ t := one_le_pow₀ (by linarith)
    have hηt := hη t (by omega)
    have hαc : α t * absErr ε α η t ≤ absErr ε α η t := by
      rcases Nat.eq_zero_or_pos t with rfl | ht
      · simp [absErr]
      · exact mul_le_of_le_one_left hc (hα1 t ht (by omega))
    rw [sum_range_succ, pow_succ]
    have f1 : (1 + ε) * (α t * absErr ε α η t) ≤ (1 + ε) * absErr ε α η t :=
      mul_le_mul_of_nonneg_left hαc (by linarith)
    have f2 : (1 + ε) * absErr ε α η t ≤ (1 + ε) * ((1 + ε) ^ t * ∑ u ∈ range t, η u) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have f3 : η t ≤ (1 + ε) ^ t * (1 + ε) * η t :=
      le_mul_of_one_le_left hηt (by nlinarith)
    nlinarith

/-- **The absolute part in closed form**, for the factors `alpha m` of references `m`:
`absErr ε (alpha m) η t = Σ_{u<t} (1 + ε)^(t−1−u)·exp (m (u+1) − m t)·η u`. The error of trip
`u`, made at the reference `m (u+1)`, counts at `exp (m (u+1) − m t) ≤ 1` when the references
never decrease. -/
theorem absErr_alpha (ε : ℝ) (m η : ℕ → ℝ) : ∀ t : ℕ,
    absErr ε (alpha m) η t =
      ∑ u ∈ range t, (1 + ε) ^ (t - 1 - u) * Real.exp (m (u + 1) - m t) * η u
  | 0 => by simp [absErr]
  | t + 1 => by
    simp only [absErr]
    rw [absErr_alpha ε m η t, sum_range_succ, mul_sum]
    congr 1
    · refine sum_congr rfl fun u hu => ?_
      have hu' : u < t := mem_range.mp hu
      have hpow : t + 1 - 1 - u = (t - 1 - u) + 1 := by omega
      have he : Real.exp (m (u + 1) - m (t + 1)) =
          Real.exp (m t - m (t + 1)) * Real.exp (m (u + 1) - m t) := by
        rw [← Real.exp_add]; congr 1; ring
      rw [hpow, pow_succ, he, alpha]
      ring
    · simp

/-- The running bound is non-negative for a non-negative magnitude, factors and errors. -/
theorem bound_nonneg {ε M : ℝ} {α η : ℕ → ℝ} {t : ℕ} (hε : 0 ≤ ε) (hM : 0 ≤ M)
    (hα : ∀ u < t, 0 ≤ α u) (hη : ∀ u < t, 0 ≤ η u) : 0 ≤ bound ε α η t M := by
  unfold bound
  have : 1 ≤ (1 + ε) ^ t := one_le_pow₀ (by linarith)
  exact add_nonneg (mul_nonneg (by linarith) hM) (absErr_nonneg hε hα hη)

/-- **The closed form.** When `0 ≤ α u ≤ 1` (for `0 < u`), `η u ≥ 0` and `t·ε ≤ 1/2`, the running
bound is at most `2tε·M + 2·Σ_{u<t} η u`. -/
theorem bound_le {ε M : ℝ} {α η : ℕ → ℝ} {t : ℕ} (hε : 0 ≤ ε) (ht : t * ε ≤ 1 / 2) (hM : 0 ≤ M)
    (hα0 : ∀ u < t, 0 ≤ α u) (hα1 : ∀ u, 0 < u → u < t → α u ≤ 1) (hη : ∀ u < t, 0 ≤ η u) :
    bound ε α η t M ≤ 2 * t * ε * M + 2 * ∑ u ∈ range t, η u := by
  unfold bound
  have hp := one_add_pow_le hε (n := t) (by linarith)
  have hc := absErr_le hε hα0 hα1 hη
  have hS : 0 ≤ ∑ u ∈ range t, η u := sum_nonneg fun u hu => hη u (mem_range.mp hu)
  have e1 : ((1 + ε) ^ t - 1) * M ≤ 2 * t * ε * M := mul_le_mul_of_nonneg_right (by linarith) hM
  have e2 : (1 + ε) ^ t * ∑ u ∈ range t, η u ≤ 2 * ∑ u ∈ range t, η u :=
    mul_le_mul_of_nonneg_right (by linarith) hS
  linarith

/-- **The running bound of a rescaled accumulation.** Let the exact state follow
`X (t+1) = α t·X t + Y t`, with a magnitude `M (t+1) = α t·M t + Z t`, `|Y t| ≤ Z t`, `α t ≥ 0`,
from `|X 0| ≤ M 0`. If the computed state starts at `x 0 = X 0` and, while its error so far is
within the bound, each trip is within `ε·(α t·|x t| + Z t) + η t` of `α t·x t + Y t`, then after
`t ≤ T` trips `|x t − X t| ≤ bound ε α η t (M t)`. The step may use the bound so far (to bound
`|x t|`, for an absolute error or the format's range). -/
theorem accum {ε : ℝ} {α η X M Y Z x : ℕ → ℝ} {T : ℕ} (hε : 0 ≤ ε) (hα : ∀ t < T, 0 ≤ α t)
    (hx0 : x 0 = X 0) (hXM0 : |X 0| ≤ M 0)
    (hX : ∀ t < T, X (t + 1) = α t * X t + Y t) (hM : ∀ t < T, M (t + 1) = α t * M t + Z t)
    (hYZ : ∀ t < T, |Y t| ≤ Z t)
    (hstep : ∀ t < T, |x t - X t| ≤ bound ε α η t (M t) →
      |x (t + 1) - (α t * x t + Y t)| ≤ ε * (α t * |x t| + Z t) + η t) :
    ∀ t ≤ T, |x t - X t| ≤ bound ε α η t (M t) := by
  suffices key : ∀ t ≤ T, |X t| ≤ M t ∧ |x t - X t| ≤ bound ε α η t (M t) from
    fun t ht => (key t ht).2
  intro t
  induction t with
  | zero => intro _; exact ⟨hXM0, by simp [bound, absErr, hx0]⟩
  | succ t ih =>
    intro ht
    have htT : t < T := by omega
    obtain ⟨hXM, he⟩ := ih (by omega)
    have ha := hα t htT
    have hZ := hYZ t htT
    have hZ0 : 0 ≤ Z t := (abs_nonneg _).trans hZ
    have hs := hstep t htT he
    refine ⟨?_, ?_⟩
    · rw [hX t htT, hM t htT]
      calc |α t * X t + Y t| ≤ |α t * X t| + |Y t| := abs_add_le _ _
        _ ≤ α t * M t + Z t := by
          rw [abs_mul, abs_of_nonneg ha]
          exact add_le_add (mul_le_mul_of_nonneg_left hXM ha) hZ
    · rw [hX t htT, hM t htT]
      have htri : |x (t + 1) - (α t * X t + Y t)| ≤
          |x (t + 1) - (α t * x t + Y t)| + α t * |x t - X t| := by
        have e : x (t + 1) - (α t * X t + Y t) =
            (x (t + 1) - (α t * x t + Y t)) + α t * (x t - X t) := by ring
        rw [e]
        refine (abs_add_le _ _).trans (le_of_eq ?_)
        rw [abs_mul, abs_of_nonneg ha]
      have hxabs : |x t| ≤ M t + |x t - X t| := by
        have := abs_sub_abs_le_abs_sub (x t) (X t); linarith
      unfold bound at he ⊢
      simp only [absErr]
      rw [pow_succ]
      have hP1 : 1 ≤ (1 + ε) ^ t := one_le_pow₀ (by linarith)
      have h1 : ε * (α t * |x t|) ≤ ε * (α t * (M t + |x t - X t|)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hxabs ha) hε
      have h2 : (1 + ε) * (α t * |x t - X t|) ≤
          (1 + ε) * (α t * (((1 + ε) ^ t - 1) * M t + absErr ε α η t)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left he ha) (by linarith)
      have h3 : 0 ≤ ((1 + ε) ^ t - 1) * (1 + ε) * Z t :=
        mul_nonneg (mul_nonneg (by linarith) (by linarith)) hZ0
      nlinarith

/-! ## One trip from its parts -/

/-- **One approximation after another.** If `b` is within `ρ·Z + η` of `Y`, a magnitude `Bm` of `b`
is at most `(1 + ρ)·Z + η`, and `r` is within `ρ'·Bm + η'` of `b`, then `r` is within
`(ρ + ρ'·(1 + ρ))·Z + ((1 + ρ')·η + η')` of `Y`. For example `b = Σ p̂` (lanes within `ρ` of
exact `p`, `block_err_one`) and `r` its float sum (`Bm = Σ|p̂|`, `block_abs_le_one`); or a rounding
after a conversion. -/
theorem approx_trans {ρ ρ' Y Z b Bm r η η' : ℝ} (hρ' : 0 ≤ ρ')
    (hb : |b - Y| ≤ ρ * Z + η) (hBm : Bm ≤ (1 + ρ) * Z + η) (hr : |r - b| ≤ ρ' * Bm + η') :
    |r - Y| ≤ (ρ + ρ' * (1 + ρ)) * Z + ((1 + ρ') * η + η') := by
  have e : r - Y = (r - b) + (b - Y) := by ring
  rw [e]
  have h := mul_le_mul_of_nonneg_left hBm hρ'
  have := abs_add_le (r - b) (b - Y)
  nlinarith

/-- A magnitude from an approximation: `|b| ≤ (1 + ρ)·Z + η` when `|b − Y| ≤ ρ·Z + η` and
`|Y| ≤ Z`. -/
theorem abs_le_of_approx {ρ Y Z b η : ℝ} (hb : |b - Y| ≤ ρ * Z + η) (hYZ : |Y| ≤ Z) :
    |b| ≤ (1 + ρ) * Z + η := by
  have := abs_sub_abs_le_abs_sub b Y
  linarith

/-- **The scaled state.** If `α̂` is within `ρ·α + η` of `α`, then `α̂·x` is within
`ρ·(α·|x|) + η·|x|` of `α·x`. -/
theorem scale_err {ρ η α α' x : ℝ} (h : |α' - α| ≤ ρ * α + η) :
    |α' * x - α * x| ≤ ρ * (α * |x|) + η * |x| := by
  rw [← sub_mul, abs_mul]
  have := mul_le_mul_of_nonneg_right h (abs_nonneg x)
  linarith

/-- **A block of lanes.** If each lane's `p̂ j` is within `ρ·p j + η` of an exact `p j ≥ 0`, then
`|Σ p̂·v − Σ p·v| ≤ ρ·Σ p·|v| + η·Σ |v|`. A masked lane has `p j = p̂ j = 0`. -/
theorem block_err {κ : Type*} {S : Finset κ} {p p' v : κ → ℝ} {ρ η : ℝ}
    (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    |∑ j ∈ S, p' j * v j - ∑ j ∈ S, p j * v j| ≤
      ρ * ∑ j ∈ S, p j * |v j| + η * ∑ j ∈ S, |v j| := by
  rw [← sum_sub_distrib, mul_sum, mul_sum, ← sum_add_distrib]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun j hj => ?_)
  rw [← sub_mul, abs_mul]
  have := mul_le_mul_of_nonneg_right (h j hj) (abs_nonneg (v j))
  linarith

/-- **The magnitude of a block:** under `block_err`'s conditions (and `p j ≥ 0`),
`Σ |p̂·v| ≤ (1 + ρ)·Σ p·|v| + η·Σ |v|`, the magnitude a `tl.dot` or a reduction is accurate
relative to. -/
theorem block_abs_le {κ : Type*} {S : Finset κ} {p p' v : κ → ℝ} {ρ η : ℝ}
    (hp : ∀ j ∈ S, 0 ≤ p j) (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    ∑ j ∈ S, |p' j * v j| ≤ (1 + ρ) * ∑ j ∈ S, p j * |v j| + η * ∑ j ∈ S, |v j| := by
  rw [mul_sum, mul_sum, ← sum_add_distrib]
  refine sum_le_sum fun j hj => ?_
  rw [abs_mul]
  have h1 : |p' j| ≤ (1 + ρ) * p j + η := by
    have := abs_sub_abs_le_abs_sub (p' j) (p j)
    rw [abs_of_nonneg (hp j hj)] at this
    linarith [h j hj]
  have := mul_le_mul_of_nonneg_right h1 (abs_nonneg (v j))
  linarith

/-- **A block of lanes, for the sum `l`:** `|Σ p̂ − Σ p| ≤ ρ·Σ p + η·|S|`. -/
theorem block_err_one {κ : Type*} {S : Finset κ} {p p' : κ → ℝ} {ρ η : ℝ}
    (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    |∑ j ∈ S, p' j - ∑ j ∈ S, p j| ≤ ρ * ∑ j ∈ S, p j + η * S.card := by
  have := block_err (v := fun _ => (1 : ℝ)) h
  simpa using this

/-- **The magnitude of a block, for the sum `l`:** `Σ |p̂| ≤ (1 + ρ)·Σ p + η·|S|`. -/
theorem block_abs_le_one {κ : Type*} {S : Finset κ} {p p' : κ → ℝ} {ρ η : ℝ}
    (hp : ∀ j ∈ S, 0 ≤ p j) (h : ∀ j ∈ S, |p' j - p j| ≤ ρ * p j + η) :
    ∑ j ∈ S, |p' j| ≤ (1 + ρ) * ∑ j ∈ S, p j + η * S.card := by
  have := block_abs_le (v := fun _ => (1 : ℝ)) hp h
  simpa using this

/-- **One trip.** The new state `x'` is computed from the scaled state `a` (within
`ρa·(α·|x|) + ηa` of `α·x`: `scale_err`, then `approx_trans` for a rounded product) and the block's
contribution `b` (within `ρb·Z + ηb` of the exact `Y`), through an operation within
`ρo·(|a| + Bm) + ηo` of `a + b`, where the magnitude `Bm` is at most `(1 + ρb)·Z + ηB`: an addition
or `fma` (`Bm = |b|`, `ηB = ηb` by `abs_le_of_approx`), or a `tl.dot` accumulation (`Bm = Σ|p̂·v|`
with `block_abs_le`, or the dot fact's lifted magnitude). Then `x'` is within
`ε·(α·|x| + Z) + ((1 + ρo)·ηa + ηb + ρo·ηB + ηo)` of `α·x + Y`, for any `ε` above both rates: the
step of `accum`, `l_err` and `acc_err`. -/
theorem step_err {ρa ρb ρo ε α x a b Y Z Bm ηa ηb ηB ηo x' : ℝ} (hρo : 0 ≤ ρo) (hα : 0 ≤ α)
    (hZ : 0 ≤ Z) (hεa : ρa + ρo * (1 + ρa) ≤ ε) (hεb : ρb + ρo * (1 + ρb) ≤ ε)
    (ha : |a - α * x| ≤ ρa * (α * |x|) + ηa) (hb : |b - Y| ≤ ρb * Z + ηb)
    (hBm : Bm ≤ (1 + ρb) * Z + ηB) (ho : |x' - (a + b)| ≤ ρo * (|a| + Bm) + ηo) :
    |x' - (α * x + Y)| ≤ ε * (α * |x| + Z) + ((1 + ρo) * ηa + ηb + ρo * ηB + ηo) := by
  have hαx : 0 ≤ α * |x| := mul_nonneg hα (abs_nonneg x)
  have hax : |a| ≤ (1 + ρa) * (α * |x|) + ηa := by
    have h1 := abs_sub_abs_le_abs_sub a (α * x)
    rw [abs_mul, abs_of_nonneg hα] at h1
    linarith
  have e : x' - (α * x + Y) = (x' - (a + b)) + (a - α * x) + (b - Y) := by ring
  rw [e]
  have h2 := mul_le_mul_of_nonneg_left (add_le_add hax hBm) hρo
  have h3 := (abs_add_le ((x' - (a + b)) + (a - α * x)) (b - Y)).trans
    (add_le_add (abs_add_le (x' - (a + b)) (a - α * x)) le_rfl)
  have h4 : (ρa + ρo * (1 + ρa)) * (α * |x|) ≤ ε * (α * |x|) := mul_le_mul_of_nonneg_right hεa hαx
  have h5 : (ρb + ρo * (1 + ρb)) * Z ≤ ε * Z := mul_le_mul_of_nonneg_right hεb hZ
  nlinarith

/-- **One trip of the sum `l`**, as attention kernels write it: `l' = l·α̂ + r̂` (an `fma` or a
product and a sum, within `ρo·(|α̂·l| + |r̂|) + ηo`), where `α̂` is within `ρα·α + ηα` of `α` and
`r̂` is a float sum of the lanes `p̂ j` (within `ρs·Σ|p̂| + ηs`), each within `ρp·p j + ηp` of an
exact `p j ≥ 0`. With `ρb = ρp + ρs·(1 + ρp)` and `ηb = (1 + ρs)·ηp·|S| + ηs`, `l'` is within
`ε·(α·|l| + Σ p) + ((1 + ρo)·(ηα·|l| + ηb) + ηo)` of `α·l + Σ p`, for any `ε` above
`ρα + ρo·(1 + ρα)` and `ρb + ρo·(1 + ρb)`. A masked lane has `p j = p̂ j = 0`. -/
theorem l_trip {κ : Type*} {S : Finset κ} {p p' : κ → ℝ}
    {α α' l r l' ρp ρs ρα ρo ηp ηs ηα ηo ε : ℝ} (hα : 0 ≤ α) (hρs : 0 ≤ ρs) (hρo : 0 ≤ ρo)
    (hp0 : ∀ j ∈ S, 0 ≤ p j) (hp : ∀ j ∈ S, |p' j - p j| ≤ ρp * p j + ηp)
    (hr : |r - ∑ j ∈ S, p' j| ≤ ρs * ∑ j ∈ S, |p' j| + ηs) (hα' : |α' - α| ≤ ρα * α + ηα)
    (ho : |l' - (α' * l + r)| ≤ ρo * (|α' * l| + |r|) + ηo) (hε1 : ρα + ρo * (1 + ρα) ≤ ε)
    (hε2 : (ρp + ρs * (1 + ρp)) + ρo * (1 + (ρp + ρs * (1 + ρp))) ≤ ε) :
    |l' - (α * l + ∑ j ∈ S, p j)| ≤ ε * (α * |l| + ∑ j ∈ S, p j) +
      ((1 + ρo) * (ηα * |l| + ((1 + ρs) * (ηp * S.card) + ηs)) + ηo) := by
  have hb := approx_trans hρs (block_err_one hp) (block_abs_le_one hp0 hp) hr
  have hYZ : |∑ j ∈ S, p j| ≤ ∑ j ∈ S, p j := (abs_of_nonneg (sum_nonneg hp0)).le
  have := step_err hρo hα (sum_nonneg hp0) hε1 hε2 (scale_err hα') hb (abs_le_of_approx hb hYZ) ho
  linarith

/-! ## The online softmax with computed values -/

section Perturbed

variable {ι : Type*} {s v : ℕ → ι → ℝ} {B : ℕ → Finset ι}

/-- **The running sum with errors.** Let `m̂` be the computed references and `l̂` the computed sum
(as reals), with `l̂ 0 = 0`. If, while the error so far is within the bound, each trip is within
`ε·(α t·|l̂ t| + R t) + η t` of `α t·l̂ t + R t` (with `α = alpha m̂` and the exact block sum
`R t = Σ_{j ∈ B t} exp (s t j − m̂ (t+1))`), then for `t ≤ n`
`|l̂ t − lsum s B (m̂ t) t| ≤ bound ε (alpha m̂) η t (lsum s B (m̂ t) t)`. -/
theorem l_err {mh lh η : ℕ → ℝ} {ε : ℝ} {n : ℕ} (hε : 0 ≤ ε) (hl0 : lh 0 = 0)
    (hstep : ∀ t < n, |lh t - lsum s B (mh t) t| ≤ bound ε (alpha mh) η t (lsum s B (mh t) t) →
      |lh (t + 1) - (alpha mh t * lh t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))| ≤
        ε * (alpha mh t * |lh t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1))) + η t) :
    ∀ t ≤ n, |lh t - lsum s B (mh t) t| ≤ bound ε (alpha mh) η t (lsum s B (mh t) t) :=
  accum (X := fun t => lsum s B (mh t) t) (M := fun t => lsum s B (mh t) t)
    (Y := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))
    (Z := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))
    hε (fun t _ => (alpha_pos mh t).le) (by simp [hl0]) (by simp)
    (fun t _ => lsum_succ (mh t) (mh (t + 1)) t) (fun t _ => lsum_succ (mh t) (mh (t + 1)) t)
    (fun t _ => (abs_of_nonneg (sum_nonneg fun _ _ => (Real.exp_pos _).le)).le) hstep

/-- **The accumulator with errors** (one output column). With the computed references `m̂` and
accumulator `âcc` (`âcc 0 = 0`): if each trip is within
`ε·(α t·|âcc t| + Σ_{j ∈ B t} exp (s t j − m̂ (t+1))·|v t j|) + η t` of
`α t·âcc t + Σ_{j ∈ B t} exp (s t j − m̂ (t+1))·v t j` (`α = alpha m̂`), then for `t ≤ n`
`|âcc t − asum s v B (m̂ t) t| ≤ bound ε (alpha m̂) η t (asum s |v| B (m̂ t) t)`. -/
theorem acc_err {mh ah η : ℕ → ℝ} {ε : ℝ} {n : ℕ} (hε : 0 ≤ ε) (ha0 : ah 0 = 0)
    (hstep : ∀ t < n, |ah t - asum s v B (mh t) t| ≤
        bound ε (alpha mh) η t (asum s (fun u j => |v u j|) B (mh t) t) →
      |ah (t + 1) - (alpha mh t * ah t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)| ≤
        ε * (alpha mh t * |ah t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|) + η t) :
    ∀ t ≤ n, |ah t - asum s v B (mh t) t| ≤
      bound ε (alpha mh) η t (asum s (fun u j => |v u j|) B (mh t) t) :=
  accum (X := fun t => asum s v B (mh t) t) (M := fun t => asum s (fun u j => |v u j|) B (mh t) t)
    (Y := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)
    (Z := fun t => ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|)
    hε (fun t _ => (alpha_pos mh t).le) (by simp [ha0]) (by simp)
    (fun t _ => asum_succ (mh t) (mh (t + 1)) t)
    (fun t _ => asum_succ (v := fun u j => |v u j|) (mh t) (mh (t + 1)) t)
    (fun t _ => (abs_sum_le_sum_abs _ _).trans (le_of_eq (sum_congr rfl fun j _ => by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)]))) hstep

/-- **The ratio of two approximations.** If `l̂` is within `eL < L` of `L > 0`, `â` within `eA` of
`A`, and `|A| ≤ V·L`, then `|â/l̂ − A/L| ≤ (eA + V·eL)/(L − eL)`. -/
theorem ratio_err {L A l a eL eA V : ℝ} (hL : 0 < L) (hl : |l - L| ≤ eL) (heL : eL < L)
    (ha : |a - A| ≤ eA) (hAV : |A| ≤ V * L) : |a / l - A / L| ≤ (eA + V * eL) / (L - eL) := by
  have hd : 0 < L - eL := by linarith
  have hll : L - eL ≤ l := by have := (abs_le.mp hl).1; linarith
  have hlpos : 0 < l := hd.trans_le hll
  have heL0 : 0 ≤ eL := (abs_nonneg _).trans hl
  have heA0 : 0 ≤ eA := (abs_nonneg _).trans ha
  have hV : 0 ≤ V :=
    le_of_mul_le_mul_right (by rw [zero_mul]; exact (abs_nonneg A).trans hAV) hL
  rw [div_sub_div _ _ hlpos.ne' hL.ne', abs_div, abs_of_pos (mul_pos hlpos hL),
    div_le_div_iff₀ (mul_pos hlpos hL) hd]
  have e : a * L - l * A = (a - A) * L + A * (L - l) := by ring
  have h1 : |a * L - l * A| ≤ eA * L + V * L * eL := by
    rw [e]
    calc |(a - A) * L + A * (L - l)| ≤ |(a - A) * L| + |A * (L - l)| := abs_add_le _ _
      _ = |a - A| * L + |A| * |l - L| := by rw [abs_mul, abs_mul, abs_of_pos hL, abs_sub_comm L]
      _ ≤ eA * L + V * L * eL :=
          add_le_add (mul_le_mul_of_nonneg_right ha hL.le)
            (mul_le_mul hAV hl (abs_nonneg _) (mul_nonneg hV hL.le))
  calc |a * L - l * A| * (L - eL) ≤ (eA * L + V * L * eL) * (L - eL) :=
        mul_le_mul_of_nonneg_right h1 hd.le
    _ = (eA + V * eL) * (L - eL) * L := by ring
    _ ≤ (eA + V * eL) * l * L :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hll (add_nonneg heA0 (mul_nonneg hV heL0)))
          hL.le
    _ = (eA + V * eL) * (l * L) := by ring

/-- **The ratio, when `l̂` is within half of `L`:** `|â/l̂ − A/L| ≤ 2·(eA + V·eL)/L`. -/
theorem ratio_err_half {L A l a eL eA V : ℝ} (hL : 0 < L) (hl : |l - L| ≤ eL) (heL : eL ≤ L / 2)
    (ha : |a - A| ≤ eA) (hAV : |A| ≤ V * L) : |a / l - A / L| ≤ 2 * (eA + V * eL) / L := by
  have h := ratio_err hL hl (by linarith) ha hAV
  have heL0 : 0 ≤ eL := (abs_nonneg _).trans hl
  have heA0 : 0 ≤ eA := (abs_nonneg _).trans ha
  have hV : 0 ≤ V :=
    le_of_mul_le_mul_right (by rw [zero_mul]; exact (abs_nonneg A).trans hAV) hL
  refine h.trans ?_
  rw [div_le_div_iff₀ (by linarith) hL]
  have := add_nonneg heA0 (mul_nonneg hV heL0)
  nlinarith

/-- **The output of the online softmax, with errors.** Under the trips of `l_err` (rate `εl`,
absolute errors `ηl ≥ 0`) and `acc_err` (rate `εa`, absolute errors `ηa`), with every `|v u j| ≤ V`
and room `eL < L` (`L = lsum s B (m̂ n) n`, `eL` its running bound), the computed ratio
`âcc n / l̂ n` is within `(eA + V·eL)/(L − eL)` of the softmax-weighted sum over the pairs
(trip, lane) seen, where `eA` is the accumulator's running bound. -/
theorem err_final {mh lh ah ηl ηa : ℕ → ℝ} {εl εa V : ℝ} {n : ℕ} (hεl : 0 ≤ εl) (hεa : 0 ≤ εa)
    (hηl : ∀ t < n, 0 ≤ ηl t) (hl0 : lh 0 = 0) (ha0 : ah 0 = 0)
    (hlstep : ∀ t < n, |lh t - lsum s B (mh t) t| ≤ bound εl (alpha mh) ηl t (lsum s B (mh t) t) →
      |lh (t + 1) - (alpha mh t * lh t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))| ≤
        εl * (alpha mh t * |lh t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1))) + ηl t)
    (hastep : ∀ t < n, |ah t - asum s v B (mh t) t| ≤
        bound εa (alpha mh) ηa t (asum s (fun u j => |v u j|) B (mh t) t) →
      |ah (t + 1) - (alpha mh t * ah t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)| ≤
        εa * (alpha mh t * |ah t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|) + ηa t)
    (hV : ∀ u < n, ∀ j ∈ B u, |v u j| ≤ V)
    (hroom : bound εl (alpha mh) ηl n (lsum s B (mh n) n) < lsum s B (mh n) n) :
    |ah n / lh n - wavg ((range n).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2)| ≤
      (bound εa (alpha mh) ηa n (asum s (fun u j => |v u j|) B (mh n) n) +
          V * bound εl (alpha mh) ηl n (lsum s B (mh n) n)) /
        (lsum s B (mh n) n - bound εl (alpha mh) ηl n (lsum s B (mh n) n)) := by
  have hL : 0 < lsum s B (mh n) n :=
    lt_of_le_of_lt (bound_nonneg hεl (lsum_nonneg _ _) (fun u _ => (alpha_pos mh u).le) hηl) hroom
  rw [← asum_div_lsum (mh n) n]
  exact ratio_err hL (l_err hεl hl0 hlstep n le_rfl) hroom (acc_err hεa ha0 hastep n le_rfl)
    ((abs_asum_le _ _).trans (asum_abs_le hV))

/-- **The output of the online softmax, with a running maximum.** As `err_final`, when the
references never decrease after the first trip (`m̂ t ≤ m̂ (t+1)` for `0 < t`, so `α ≤ 1`),
`n·εl ≤ 1/2`, `n·εa ≤ 1/2`, the absolute errors are non-negative, and
`2n·εl·L + 2·Σ ηl ≤ L/2` (`L = lsum s B (m̂ n) n > 0`):
`|âcc n / l̂ n − wavg| ≤ 4n·(εa + εl)·V + 4·(Σ ηa + V·Σ ηl)/L`. With `m̂ n` at most a score seen
(`one_le_lsum`), `L ≥ 1`. -/
theorem err_final_mono {mh lh ah ηl ηa : ℕ → ℝ} {εl εa V : ℝ} {n : ℕ} (hεl : 0 ≤ εl)
    (hεa : 0 ≤ εa) (hηl : ∀ t < n, 0 ≤ ηl t) (hηa : ∀ t < n, 0 ≤ ηa t)
    (hmono : ∀ t, 0 < t → t < n → mh t ≤ mh (t + 1)) (hnl : n * εl ≤ 1 / 2) (hna : n * εa ≤ 1 / 2)
    (hl0 : lh 0 = 0) (ha0 : ah 0 = 0)
    (hlstep : ∀ t < n, |lh t - lsum s B (mh t) t| ≤ bound εl (alpha mh) ηl t (lsum s B (mh t) t) →
      |lh (t + 1) - (alpha mh t * lh t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)))| ≤
        εl * (alpha mh t * |lh t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1))) + ηl t)
    (hastep : ∀ t < n, |ah t - asum s v B (mh t) t| ≤
        bound εa (alpha mh) ηa t (asum s (fun u j => |v u j|) B (mh t) t) →
      |ah (t + 1) - (alpha mh t * ah t + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * v t j)| ≤
        εa * (alpha mh t * |ah t| + ∑ j ∈ B t, Real.exp (s t j - mh (t + 1)) * |v t j|) + ηa t)
    (hV : ∀ u < n, ∀ j ∈ B u, |v u j| ≤ V) (hL : 0 < lsum s B (mh n) n)
    (hroom : 2 * n * εl * lsum s B (mh n) n + 2 * ∑ t ∈ range n, ηl t ≤ lsum s B (mh n) n / 2) :
    |ah n / lh n - wavg ((range n).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2)| ≤
      4 * n * (εa + εl) * V +
        4 * (∑ t ∈ range n, ηa t + V * ∑ t ∈ range n, ηl t) / lsum s B (mh n) n := by
  set L := lsum s B (mh n) n with hLdef
  set Ma := asum s (fun u j => |v u j|) B (mh n) n with hMa
  have hα0 : ∀ u < n, 0 ≤ alpha mh u := fun u _ => (alpha_pos mh u).le
  have hα1 : ∀ u, 0 < u → u < n → alpha mh u ≤ 1 := fun u hu hun => alpha_le_one (hmono u hu hun)
  have hel := (l_err hεl hl0 hlstep n le_rfl).trans
    (bound_le hεl (by linarith) (lsum_nonneg _ _) hα0 hα1 hηl)
  have hMa0 : 0 ≤ Ma :=
    sum_nonneg fun _ _ => sum_nonneg fun _ _ => mul_nonneg (Real.exp_pos _).le (abs_nonneg _)
  have hMaV : Ma ≤ V * L := asum_abs_le hV
  have hea := (acc_err hεa ha0 hastep n le_rfl).trans
    (bound_le hεa (by linarith) hMa0 hα0 hα1 hηa)
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hea' : |ah n - asum s v B (mh n) n| ≤
      2 * n * εa * (V * L) + 2 * ∑ t ∈ range n, ηa t := by
    refine hea.trans ?_
    have := mul_le_mul_of_nonneg_left hMaV (mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hn0) hεa)
    linarith
  rw [← asum_div_lsum (mh n) n]
  refine (ratio_err_half hL hel hroom hea' ((abs_asum_le _ _).trans (asum_abs_le hV))).trans
    (le_of_eq ?_)
  field_simp
  ring

end Perturbed

end OnlineSoftmax

end TVA
