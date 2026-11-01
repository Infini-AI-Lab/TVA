import AttnFwdE5m2

/-! Reviewer r-e5fwd2, after library round 25: the claim's axioms, the assumptions' fields, the fp8
fact `dot` cites, and `liftMag` / `minNormal8` on `float8e5` (unchanged: `2^-14`). -/

set_option pp.fullNames true

#print axioms Kernels.AttnFwdE5m2.main
#print axioms Kernels.AttnFwdE5m2.consistent
#print axioms Kernels.AttnFwdE5m2.satisfiable
#print Kernels.AttnFwdE5m2.Assumptions
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.rhoChunk8
#print TVA.Rules.Num.sigmaChunk8
#print TVA.Rules.Num.alphaChunk8

example : TVA.Rules.Num.minNormal8 .f8e5 = 2 ^ (-14 : ℤ) := rfl
example (q : ℚ) : TVA.Rules.Num.liftMag .f8e5 q = if q = 0 then 0 else max |q| (2 ^ (-14 : ℤ)) := rfl
