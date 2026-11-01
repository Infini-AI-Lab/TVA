import MatmulE5m2

/-! Round 26 re-check: the facts, constants and library names the kernel's specification and
assumptions use, printed with full names after the round-26 additions. -/

open TVA.Rules.Num in
example : rhoChunk8 = 2 ^ (-7 : ℤ) ∧ sigmaChunk8 = 2 ^ (-12 : ℤ) ∧ alphaChunk8 = 2 ^ (-126 : ℤ) :=
  ⟨rfl, rfl, rfl⟩

open TVA.Rules.Num in
example : minNormal8 .f8e5 = 2 ^ (-14 : ℤ) := rfl

/-- The old fact still follows from the new loss fact; the kernel cites the old one. -/
example (P : TVA.Platform) (h : TVA.Rules.Num.DotE5m2ChunkLoss P) :
    TVA.Rules.Num.DotE5m2Chunk P :=
  TVA.Rules.Num.DotF8ChunkLoss.chunk (P := P) h

#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.DotChunkAcc
#print TVA.Rules.Num.liftMag
#print TVA.Rules.Num.minNormal8
#print TVA.Rules.Num.CvtF32E5m2
#print TVA.Rules.Num.CvtTotal
#print TVA.Rules.Num.DotTotal
#print TVA.DotLoop.bound
#print Kernels.MatmulE5m2.errAt
#print Kernels.MatmulE5m2.tol
#print Kernels.MatmulE5m2.tolA
#print Kernels.MatmulE5m2.liftV
#print Kernels.MatmulE5m2.Assumptions
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.zero_rejected
#print axioms Kernels.MatmulE5m2.drop_rejected
