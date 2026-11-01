import MatmulE5m2

/-! Round 25 re-check: the names the kernel cites, and `liftMag`/`minNormal8` on fp8 values. -/

open TVA.Rules.Num in
example : minNormal8 .f8e5 = 2 ^ (-14 : ℤ) := rfl

open TVA.Rules.Num in
example : minNormal8 .f8e4nv = 2 ^ (-6 : ℤ) := rfl

open TVA.Rules.Num in
example (q : ℚ) : liftMag .f8e5 q = if q = 0 then 0 else max |q| (2 ^ (-14 : ℤ)) := rfl

#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.DotChunkAcc
#print TVA.Rules.Num.rhoChunk8
#print TVA.Rules.Num.sigmaChunk8
#print TVA.Rules.Num.alphaChunk8
#print TVA.Rules.Num.CvtF32E5m2
#print TVA.Rules.Num.CvtTotal
#print TVA.Rules.Num.DotTotal
#print Kernels.MatmulE5m2.liftV
#print Kernels.MatmulE5m2.dotLift
#print Kernels.MatmulE5m2.Assumptions
#print axioms Kernels.MatmulE5m2.main
