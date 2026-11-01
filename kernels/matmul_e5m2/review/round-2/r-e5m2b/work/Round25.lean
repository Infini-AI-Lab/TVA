import MatmulE5m2

/-! Library round 25: `liftMag` and `minNormal8` moved; their `float8e5` values, the chunk fact and
the other cited facts are as before. -/

set_option pp.fullNames true

#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.minNormal8
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.DotChunkAcc
#print TVA.Rules.Num.CvtF32E5m2
#print TVA.Rules.Num.CvtTotal
#print TVA.Rules.Num.DotTotal
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable

/-- The smallest normal `float8e5` magnitude is still `2^-14`. -/
example : TVA.Rules.Num.minNormal8 .f8e5 = 1 / 2 ^ 14 := by norm_num [TVA.Rules.Num.minNormal8]

/-- A subnormal `float8e5` value (`2^-16`) still lifts to `2^-14`, a normal one (`1`) to itself. -/
example : TVA.Rules.Num.liftMag .f8e5 (1 / 2 ^ 16) = 1 / 2 ^ 14 := by
  norm_num [TVA.Rules.Num.liftMag, TVA.Rules.Num.minNormal8]

example : TVA.Rules.Num.liftMag .f8e5 1 = 1 := by
  norm_num [TVA.Rules.Num.liftMag, TVA.Rules.Num.minNormal8]
