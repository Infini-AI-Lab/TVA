import Matmul

/-! r-matmul, round 3: the restated `DotF16` and the specification's lifted magnitude `dotMag`. -/

set_option pp.fullNames true

#print Kernels.Matmul.mag16
#print Kernels.Matmul.dotMag
#print Kernels.Matmul.OutSpec
#print Kernels.Matmul.Assumptions
#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.DotLiftAcc
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.minNormal8
#print TVA.Rules.Num.relDot16
#print TVA.Rules.Num.absDot16
#print TVA.DotLoop.bound
#print axioms Kernels.Matmul.main
#print axioms Kernels.Matmul.consistent
#print axioms Kernels.Matmul.satisfiable

namespace R3

open Kernels.Matmul TVA

/-- The kernel's `mag16` is the library's `liftMag .f16` (definitional). -/
example (q : ℚ) : Rules.Num.liftMag .f16 q = mag16 q := rfl

/-- For zero and normal magnitudes `mag16` is `|q|`, so with no subnormal operand `dotMag` is
`Σ|va·vb|` and every tolerance figure of round 2 stands. -/
theorem mag16_of_normal {q : ℚ} (h : q = 0 ∨ 2 ^ (-14 : ℤ) ≤ |q|) : mag16 q = |q| := by
  unfold mag16
  rcases h with h | h
  · simp [h]
  · split_ifs with h0
    · simp [h0]
    · exact max_eq_left h

/-- The README's bound on one term: a lifted product exceeds `|x·y|` by at most
`2^-14·(|x| + |y|) + 2^-28`. -/
theorem mag16_mul_le (x y : ℚ) :
    mag16 x * mag16 y ≤ |x * y| + 2 ^ (-14 : ℤ) * (|x| + |y|) + 2 ^ (-28 : ℤ) := by
  have hx : mag16 x ≤ |x| + 2 ^ (-14 : ℤ) := by
    unfold mag16; split_ifs
    · positivity
    · exact max_le (by linarith [show (0:ℚ) ≤ 2 ^ (-14 : ℤ) by positivity]) (by linarith [abs_nonneg x])
  have hx0 : 0 ≤ mag16 x := mag16_nonneg x
  have hy : mag16 y ≤ |y| + 2 ^ (-14 : ℤ) := by
    unfold mag16; split_ifs
    · positivity
    · exact max_le (by linarith [show (0:ℚ) ≤ 2 ^ (-14 : ℤ) by positivity]) (by linarith [abs_nonneg y])
  have hy0 : 0 ≤ mag16 y := mag16_nonneg y
  have h28 : (2 : ℚ) ^ (-28 : ℤ) = 2 ^ (-14 : ℤ) * 2 ^ (-14 : ℤ) := by
    rw [← zpow_add₀ (by norm_num)]; norm_num
  calc mag16 x * mag16 y ≤ (|x| + 2 ^ (-14 : ℤ)) * (|y| + 2 ^ (-14 : ℤ)) :=
        mul_le_mul hx hy hy0 (by positivity)
    _ = |x * y| + 2 ^ (-14 : ℤ) * (|x| + |y|) + 2 ^ (-28 : ℤ) := by rw [abs_mul, h28]; ring

/-- Flushing subnormal operands to zero is still rejected at the largest covered `K = 2^13`: one
product `2^-15 · 1` (a subnormal `float16` times `1`) has lifted magnitude `2^-14`, so a kernel
that drops it errs by `2^-15`, half of `A`, far beyond `tol (2^13) (2^-14)`. -/
example : tol (2 ^ 13) (2 ^ (-14 : ℤ)) < 2 ^ (-15 : ℤ) := by
  have h := tol_rel_lt (K := 2 ^ 13) le_rfl
  unfold tol DotLoop.bound
  rw [show steps (2 ^ 13) = 128 by unfold steps; norm_num] at h ⊢
  have hge : (1 : ℚ) ≤ (1 + Rules.Num.relDot16 64) ^ 128 :=
    one_le_pow₀ (by unfold Rules.Num.relDot16; norm_num)
  generalize (1 + Rules.Num.relDot16 64) ^ 128 = x at h hge ⊢
  have hα : Rules.Num.absDot16 64 = 66 / 2 ^ 126 := by unfold Rules.Num.absDot16; norm_num
  rw [hα]
  have h1 : (x - 1) * 2 ^ (-14 : ℤ) ≤ 21 / 10000 * 2 ^ (-14 : ℤ) :=
    mul_le_mul_of_nonneg_right (by linarith) (by positivity)
  have h2 : (128 : ℚ) * x * (66 / 2 ^ 126) ≤ 128 * (1 + 21 / 10000) * (66 / 2 ^ 126) := by
    nlinarith
  push_cast
  have h3 : (21 : ℚ) / 10000 * 2 ^ (-14 : ℤ) + 128 * (1 + 21 / 10000) * (66 / 2 ^ 126) <
      2 ^ (-15 : ℤ) := by norm_num
  linarith

end R3
