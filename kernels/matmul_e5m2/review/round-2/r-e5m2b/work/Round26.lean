import MatmulE5m2

/-! Library round 26 added the loss facts and lemmas; the facts this kernel cites, their constants
and the lemmas its specification uses are as before. -/

set_option pp.fullNames true

#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.DotChunkAcc
#print TVA.Rules.Num.rhoChunk8
#print TVA.Rules.Num.sigmaChunk8
#print TVA.Rules.Num.alphaChunk8
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.CvtF32E5m2
#print TVA.Rules.Num.CvtTotal
#print TVA.Rules.Num.DotTotal
#check @TVA.Dot8Chunk.DotF8Chunk.err
#check @TVA.Dot8Chunk.liftMag_zero
#check @TVA.Dot8Chunk.liftMag_nonneg
#check @TVA.Dot8Chunk.abs_mul_le_liftMag
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable

example : TVA.Rules.Num.rhoChunk8 = 1 / 2 ^ 7 := by norm_num [TVA.Rules.Num.rhoChunk8]
example : TVA.Rules.Num.sigmaChunk8 = 1 / 2 ^ 12 := by norm_num [TVA.Rules.Num.sigmaChunk8]
example : TVA.Rules.Num.alphaChunk8 = 1 / 2 ^ 126 := by norm_num [TVA.Rules.Num.alphaChunk8]
example : TVA.Rules.Num.minNormal8 .f8e5 = 1 / 2 ^ 14 := by norm_num [TVA.Rules.Num.minNormal8]
