import Claim
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# `layer_norm_gated_fwd_bf16`: real facts about the gate

The logistic function `sig v = 1 / (1 + e^-v)` (`Claim.sig`) lies in `(0, 1]` and below `e^v`;
`e^88.5 ≤ 1.9·2^127` (room for `tl.exp` of a gate's negation up to `88.5`); and `tl.exp`'s overflow
threshold `1453635/16384 ≈ 88.7228` is above `128·ln 2`, so a gate at or below its negation has
`sig ≤ 2^-128`. The bounds on `ln 2` are Mathlib's (`Real.log_two_gt_d9`, `Real.log_two_lt_d9`).
-/

namespace Claims.LayerNormGatedFwd

/-- `sig v > 0`. -/
theorem sig_pos (v : ℝ) : 0 < sig v := by
  unfold sig; have := Real.exp_pos (-v); positivity

/-- `sig v ≤ 1`. -/
theorem sig_le_one (v : ℝ) : sig v ≤ 1 := by
  unfold sig
  have h := Real.exp_pos (-v)
  rw [div_le_one (by linarith)]
  linarith

/-- `sig v ≤ e^v`. -/
theorem sig_le_exp (v : ℝ) : sig v ≤ Real.exp v := by
  unfold sig
  have h := Real.exp_pos (-v)
  have hv : Real.exp v * Real.exp (-v) = 1 := by rw [← Real.exp_add]; simp
  rw [div_le_iff₀ (by linarith)]
  nlinarith [Real.exp_pos v]

/-- `e^(n·ln 2) = 2^n`. -/
theorem exp_nat_log_two (n : ℕ) : Real.exp (n * Real.log 2) = (2 : ℝ) ^ n := by
  rw [Real.exp_nat_mul, Real.exp_log (by norm_num)]

/-- `e^88.5 ≤ 1.9·2^127`. -/
theorem exp_885_le : Real.exp (177 / 2) ≤ 19 / 10 * (2 : ℝ) ^ (127 : ℤ) := by
  have hl := Real.log_two_gt_d9
  have h1 : (177 / 2 : ℝ) ≤ (127 : ℕ) * Real.log 2 + 471 / 1000 := by
    push_cast; norm_num at hl ⊢; linarith
  have h2 : Real.exp (471 / 1000) ≤ 19 / 10 := by
    have h := Real.exp_bound_div_one_sub_of_interval (x := 471 / 1000) (by norm_num) (by norm_num)
    refine h.trans ?_
    norm_num
  calc Real.exp (177 / 2) ≤ Real.exp ((127 : ℕ) * Real.log 2 + 471 / 1000) := Real.exp_le_exp.mpr h1
    _ = (2 : ℝ) ^ (127 : ℕ) * Real.exp (471 / 1000) := by rw [Real.exp_add, exp_nat_log_two]
    _ ≤ (2 : ℝ) ^ (127 : ℕ) * (19 / 10) := mul_le_mul_of_nonneg_left h2 (by positivity)
    _ = 19 / 10 * (2 : ℝ) ^ (127 : ℤ) := by norm_num

/-- `e^(-1453635/16384) ≤ 2^-128`: `tl.exp`'s overflow threshold is above `128·ln 2`. -/
theorem exp_neg_thr_le : Real.exp (-(1453635 / 16384)) ≤ (2 : ℝ) ^ (-128 : ℤ) := by
  have hl := Real.log_two_lt_d9
  have h1 : -(1453635 / 16384 : ℝ) ≤ -((128 : ℕ) * Real.log 2) := by
    push_cast; norm_num at hl ⊢; linarith
  calc Real.exp (-(1453635 / 16384)) ≤ Real.exp (-((128 : ℕ) * Real.log 2)) :=
        Real.exp_le_exp.mpr h1
    _ = ((2 : ℝ) ^ (128 : ℕ))⁻¹ := by rw [Real.exp_neg, exp_nat_log_two]
    _ = (2 : ℝ) ^ (-128 : ℤ) := by norm_num

/-- A gate `v` with `-v` at least the overflow threshold has `sig v ≤ 2^-128`. -/
theorem sig_le_of_overflow {v : ℝ} (hv : (1453635 / 16384 : ℝ) ≤ -v) : sig v ≤ (2 : ℝ) ^ (-128 : ℤ) :=
  (sig_le_exp v).trans ((Real.exp_le_exp.mpr (by linarith)).trans exp_neg_thr_le)

/-- `v·e^-v ≤ 1` for every `v` (`v ≤ e^v`, as `v + 1 ≤ e^v`). -/
theorem mul_exp_neg_le_one (v : ℝ) : v * Real.exp (-v) ≤ 1 := by
  have h := Real.add_one_le_exp v
  have he : Real.exp v * Real.exp (-v) = 1 := by rw [← Real.exp_add]; simp
  have hp := Real.exp_pos (-v)
  nlinarith

end Claims.LayerNormGatedFwd
