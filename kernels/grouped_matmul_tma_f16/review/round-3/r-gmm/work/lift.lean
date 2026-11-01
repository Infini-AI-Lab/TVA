import GroupedMatmulTma

/-! r-gmm, round 3: the restated `DotF16` the claim cites, and the new tolerance's magnitude. -/

set_option pp.fullNames true

open TVA

#print Kernels.GroupedMatmulTma.dotLift
#print Kernels.GroupedMatmulTma.Assumptions
#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.minNormal8
#print axioms Kernels.GroupedMatmulTma.main
#print axioms Kernels.GroupedMatmulTma.consistent
#print axioms Kernels.GroupedMatmulTma.satisfiable

/-- For a normal `float16` value (`|q| ≥ 2^-14`) and for zero, the lifted magnitude is `|q|`, so
`dotLift` is the old `Σ |a·b|` on such inputs. -/
example (q : ℚ) (h : (2 : ℚ) ^ (-14 : ℤ) ≤ |q|) : Rules.Num.liftMag .f16 q = |q| := by
  unfold Rules.Num.liftMag Rules.Num.minNormal8
  split_ifs with h0
  · simp [h0]
  · exact max_eq_left h

example : Rules.Num.liftMag .f16 0 = 0 := by simp [Rules.Num.liftMag]

/-- The smallest `float16` subnormal, `2^-24`, counts as `2^-14`: `2^10` times its magnitude. -/
example : Rules.Num.liftMag .f16 ((2 : ℚ) ^ (-24 : ℤ)) = 2 ^ (-14 : ℤ) := by
  unfold Rules.Num.liftMag Rules.Num.minNormal8
  norm_num
