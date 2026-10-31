import AttnFwdE5m2.Model
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# `attn_fwd_e5m2`: real-number facts the error analysis uses

* powers of two: `two_rpow_sub_one` (`|2^δ − 1| ≤ |δ|` for `|δ| ≤ 1/2`), monotonicity, products;
* `one_add_pow_le_exp`, `exp_le_two`: powers of a factor near `1`;
* `pow6463_le`: `(64/63)^t ≤ 2^25` for `t ≤ 1024`, the growth of the accumulator's crude bound.
-/

namespace Kernels.AttnFwdE5m2

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

theorem two_rpow_neg_mul (x : ℝ) : (2 : ℝ) ^ (-x) * 2 ^ x = 1 := by
  rw [← two_rpow_add, neg_add_cancel, two_rpow_zero]

theorem two_rpow_neg (x : ℝ) : (2 : ℝ) ^ (-x) = 1 / 2 ^ x := by
  rw [Real.rpow_neg (by norm_num), one_div]

theorem two_rpow_nat (n : ℕ) : (2 : ℝ) ^ (n : ℝ) = 2 ^ n := Real.rpow_natCast 2 n

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

/-- A power of two `2^δ` with `|δ| ≤ E` lies in `[2^-E, 2^E]`. -/
theorem two_rpow_mem {δ E : ℝ} (h : |δ| ≤ E) : 1 / (2 : ℝ) ^ E ≤ 2 ^ δ ∧ (2 : ℝ) ^ δ ≤ 2 ^ E := by
  have h1 := abs_le.mp h
  refine ⟨?_, two_rpow_le h1.2⟩
  rw [← two_rpow_neg]; exact two_rpow_le h1.1

/-! ## Powers of a factor near `1` -/

theorem one_add_pow_le_exp {x : ℝ} (hx : 0 ≤ x) (n : ℕ) : (1 + x) ^ n ≤ Real.exp (n * x) := by
  rw [Real.exp_nat_mul]
  exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp x]) n

theorem exp_le_two {y : ℝ} (h0 : 0 ≤ y) (h1 : y ≤ 1) : Real.exp y ≤ 1 + 2 * y := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := y) (by rw [abs_of_nonneg h0]; exact h1)
  have := le_abs_self (Real.exp y - 1 - y)
  nlinarith

/-- A factor `s ∈ [1, 1 + c]` to a power `n ≤ 1024` with `1024·c ≤ 1`: at most `1 + 2048·c`. -/
theorem pow_le_of_step {s : ℝ} {n : ℕ} {c : ℝ} (h1 : 1 ≤ s) (hs : s ≤ 1 + c) (hn : n ≤ 1024)
    (hc : 1024 * c ≤ 1) : s ^ n ≤ 1 + 2 * (1024 * c) := by
  have hc0 : 0 ≤ c := by linarith
  have h := one_add_pow_le_exp (show 0 ≤ s - 1 by linarith) n
  rw [show 1 + (s - 1) = s by ring] at h
  refine h.trans ((Real.exp_le_exp.mpr ?_).trans (exp_le_two (by positivity) hc))
  have : (n : ℝ) ≤ 1024 := by exact_mod_cast hn
  nlinarith

/-- A factor `q ∈ [1 − c, 1]` to a power `n ≤ 1024`: at least `1 − 1024·c` (Bernoulli). -/
theorem pow_ge_of_step {q : ℝ} {n : ℕ} {c : ℝ} (hq : 1 - c ≤ q) (hc0 : 0 ≤ c) (hc : c ≤ 1)
    (hn : n ≤ 1024) : 1 - 1024 * c ≤ q ^ n := by
  have h0 : 0 ≤ 1 - c := by linarith
  have h1 : (1 - c) ^ n ≤ q ^ n := pow_le_pow_left₀ h0 hq n
  have h2 := one_add_mul_le_pow (show (-2 : ℝ) ≤ -c by linarith) n
  have h3 : (n : ℝ) ≤ 1024 := by exact_mod_cast hn
  have h4 : 1 - 1024 * c ≤ 1 + (n : ℝ) * (-c) := by nlinarith
  rw [show (1 : ℝ) + -c = 1 - c by ring] at h2
  linarith

theorem exp17_le : Real.exp 17 ≤ 2 ^ 25 := by
  have h := Real.exp_one_lt_d9
  have h17 : Real.exp 17 = Real.exp 1 ^ 17 := by rw [← Real.exp_nat_mul]; norm_num
  rw [h17]
  have h0 : 0 ≤ Real.exp 1 := (Real.exp_pos 1).le
  calc Real.exp 1 ^ 17 ≤ (2.7182818286 : ℝ) ^ 17 := pow_le_pow_left₀ h0 h.le 17
    _ ≤ 2 ^ 25 := by norm_num

/-- **`(64/63)^t ≤ 2^25`** for `t ≤ 1024`. -/
theorem pow6463_le {t : ℕ} (ht : t ≤ 1024) : ((64 : ℝ) / 63) ^ t ≤ 2 ^ 25 := by
  have h := one_add_pow_le_exp (x := (1 / 63 : ℝ)) (by norm_num) t
  rw [show (1 : ℝ) + 1 / 63 = 64 / 63 by norm_num] at h
  refine h.trans ((Real.exp_le_exp.mpr ?_).trans exp17_le)
  have : (t : ℝ) ≤ 1024 := by exact_mod_cast ht
  nlinarith

end Kernels.AttnFwdE5m2
