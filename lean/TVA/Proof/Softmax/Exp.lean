import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.BigOperators.Field

/-!
# Bounds on the exponential

An online softmax evaluates `exp (s − m)` (or `exp2` of a base-2 score) from computed, rounded
arguments, and needs to know how far the result moves. The question is the same each time: the
argument moved by `δ`; how far did `exp` move? The exponential is `exp (max a b)`-Lipschitz
between `a` and `b`, so a small argument error is a small *relative* error of the result, and for
very negative arguments (where the kernel's weights vanish) it is a small *absolute* error.

* `abs_exp_sub_exp_le`: `|exp a − exp b| ≤ |a − b|·exp (max a b)`.
* `abs_exp_sub_one_le_mul`, `exp_le_one_add_two_mul`: `exp` near `0`.
* `abs_exp_sub_exp_le_mul`: an argument error `δ ≤ 1` is a relative error `δ·(1 + δ)`.
* `abs_exp_round_le`: the exponential of a rounded non-positive argument, relative plus absolute.
* `mul_exp_neg_le`, `one_add_pow_le`: the elementary inequalities behind the bounds.
* `exp2`: the device facts state `tl.math.exp2` against the real power `(2 : ℝ) ^ x`
  (`MathFn.ref`, `TVA.Device.Shapes`); `two_rpow_eq_exp` turns it into `exp (x·log 2)`, so a kernel
  using `exp2` reuses every lemma stated for `exp`. `exp_le_two_rpow`, `exp_le_two_zpow`,
  `two_zpow_le_exp` compare the two.

How a kernel uses them: a score row's weight `p̂ = exp(ŝ − m̂)` is computed from a rounded
difference; `abs_exp_round_le` turns that rounding (from the float error lemmas) into the
per-lane bound `|p̂ − p| ≤ ρ·p + η` that `TVA.Proof.Softmax.Error` consumes.

Used by `TVA.Proof.Softmax.Exact` and `TVA.Proof.Softmax.Error`.
-/

namespace TVA

namespace Softmax

/-- `|exp a − exp b| ≤ |a − b|·exp (max a b)`: the exponential is `exp (max a b)`-Lipschitz between
`a` and `b`. Use it to turn an argument error into a result error. -/
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

/-- `|exp d − 1| ≤ |d|·(1 + |d|)` for `|d| ≤ 1`: sharper than Mathlib's `Real.abs_exp_sub_one_le`
(`2|d|`) when `d` is small. -/
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
`|exp y − exp x| ≤ δ·(1 + δ)·exp x` (so at most `2δ·exp x`). Use it when the argument's error is
absolute and small, wherever the argument lies. -/
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
`|x| ≤ K`, absolute beyond. Use it for a weight `exp(s − m)` computed from a rounded `s − m` (or
`exp2` of it, through `two_rpow_eq_exp`), choosing `K` to balance the two terms. -/
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

/-- `(1 + x)^n ≤ 1 + 2nx` for `0 ≤ x` and `n·x ≤ 1`: the growth of a relative error compounded
over `n` trips stays linear while `n·x` is small. -/
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
`MathFn.ref .exp2` states `tl.math.exp2`. A kernel that works in base 2 (scores pre-multiplied by
`log₂ e`) rewrites with it and then uses the lemmas for `exp`. -/
theorem two_rpow_eq_exp (x : ℝ) : (2 : ℝ) ^ x = Real.exp (x * Real.log 2) := by
  rw [Real.rpow_def_of_pos (by norm_num), mul_comm]

/-- `exp x ≤ 2^y` when `x ≤ y·log 2`. -/
theorem exp_le_two_rpow {x y : ℝ} (h : x ≤ y * Real.log 2) : Real.exp x ≤ (2 : ℝ) ^ y := by
  rw [two_rpow_eq_exp]; exact Real.exp_le_exp.mpr h

/-- `exp x ≤ 2^k` (an integer power) when `x ≤ k·log 2`: bounds an exponential by a float
power of two, for example to show it stays below the format's maximum. -/
theorem exp_le_two_zpow {x : ℝ} {k : ℤ} (h : x ≤ k * Real.log 2) : Real.exp x ≤ (2 : ℝ) ^ k := by
  rw [← Real.rpow_intCast]; exact exp_le_two_rpow h

/-- `2^k ≤ exp x` (an integer power) when `k·log 2 ≤ x`: bounds an exponential from below, for
example to show it stays above the subnormal range. -/
theorem two_zpow_le_exp {x : ℝ} {k : ℤ} (h : k * Real.log 2 ≤ x) : (2 : ℝ) ^ k ≤ Real.exp x := by
  rw [← Real.rpow_intCast, two_rpow_eq_exp]; exact Real.exp_le_exp.mpr h

end Softmax

end TVA
