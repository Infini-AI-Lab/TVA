import MatmulE5m2

/-! Reviewer r-e5m2: what the tolerance of `Post` admits at covered sizes.

* `zero_meets`: from `K = 2817` to `4096`, the value `0` is within `tol K A` of every `S` with
  `|S| ≤ A`, so a kernel that stores `+0` in every element meets `OutSpec`'s accuracy clauses.
* `round8_meets`: from `K = 448` (the test's `512` included), a value within `|S|/9` of `S` (the
  largest relative error of a rounding to `float8e5`, 2 fraction bits) is within `tol K A`.
* `ord_step`: the per-step fact (`|a' - (a + X)| ≤ ρ·(|a| + B) + α`) also gives the order-aware
  recurrence `E' = (1 + ρ)·E + ρ·(|S| + B) + α`, with `|S|` the exact partial sum so far in place
  of the partial sum of magnitudes that `DotLoop.accum_step` uses.
-/

namespace Kernels.MatmulE5m2

open TVA

theorem tol_ge_rel {K : Nat} {A : ℚ} (_hA : 0 ≤ A) :
    ((1 + Rules.Num.relDot8 64) ^ steps K - 1) * A ≤ tol K A := by
  unfold tol DotLoop.bound
  have h1 : (0 : ℚ) ≤ 1 + Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have h2 : (0 : ℚ) ≤ Rules.Num.absDot8 64 := by rw [absDot8_64]; norm_num
  have : (0 : ℚ) ≤ (steps K : ℚ) * (1 + Rules.Num.relDot8 64) ^ steps K * Rules.Num.absDot8 64 := by
    positivity
  linarith

theorem zero_meets {K : Nat} (hK : 2816 < K) {S A : ℚ} (hSA : |S| ≤ A) :
    |(0 : ℚ) - S| ≤ tol K A := by
  have hA : 0 ≤ A := (abs_nonneg _).trans hSA
  have h := tol_rel_gt_one hK
  have h2 := tol_ge_rel (K := K) hA
  have h3 : A ≤ ((1 + Rules.Num.relDot8 64) ^ steps K - 1) * A := by nlinarith
  rw [zero_sub, abs_neg]
  linarith

theorem round8_meets {K : Nat} (hK : 448 ≤ K) {S A q : ℚ} (hSA : |S| ≤ A) (hq : |q - S| ≤ |S| / 9) :
    |q - S| ≤ tol K A := by
  have hA : 0 ≤ A := (abs_nonneg _).trans hSA
  have hT : 7 ≤ steps K := by unfold steps; omega
  have h1 : (1 : ℚ) ≤ 1 + Rules.Num.relDot8 64 := by rw [relDot8_64]; norm_num
  have h2 := pow_le_pow_right₀ h1 hT
  have h3 : 1 + 1 / 9 ≤ (1 + Rules.Num.relDot8 64) ^ 7 := by rw [relDot8_64]; norm_num
  have h4 := tol_ge_rel (K := K) hA
  have h5 : A / 9 ≤ ((1 + Rules.Num.relDot8 64) ^ steps K - 1) * A := by nlinarith
  linarith

theorem ord_step {ρ α a a' S X B E : ℚ} (hρ : 0 ≤ ρ) (he : |a - S| ≤ E)
    (hs : |a' - (a + X)| ≤ ρ * (|a| + B) + α) :
    |a' - (S + X)| ≤ (1 + ρ) * E + ρ * (|S| + B) + α := by
  have h1 : |a| ≤ |S| + E := by
    have := abs_sub_abs_le_abs_sub a S
    linarith
  have h2 : |a' - (S + X)| ≤ |a' - (a + X)| + |a - S| := by
    have e : a' - (S + X) = (a' - (a + X)) + (a - S) := by ring
    rw [e]; exact abs_add_le _ _
  have h3 : ρ * |a| ≤ ρ * (|S| + E) := mul_le_mul_of_nonneg_left h1 hρ
  nlinarith

end Kernels.MatmulE5m2
