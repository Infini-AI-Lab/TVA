import AttnFwdE5m2

/-! Reviewer r-e5fwd2, after library round 26: the claim's axioms, the fact `dot` cites and its shape
and constants, and the library definitions the specification uses (`scoreErr`, `scoreE`), as Lean
elaborates them now. -/

set_option pp.fullNames true

#print axioms Kernels.AttnFwdE5m2.main
#print axioms Kernels.AttnFwdE5m2.consistent
#print axioms Kernels.AttnFwdE5m2.satisfiable
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.DotChunkAcc
#print TVA.Rules.Num.rhoChunk8
#print TVA.Rules.Num.sigmaChunk8
#print TVA.Rules.Num.alphaChunk8
#print TVA.Rules.Num.chunk8
#print TVA.Rules.Num.chunkSum8
#print TVA.Rules.Num.chunks8
#print TVA.Dot8Chunk.chunkLift8
#print TVA.Dot8Chunk.relLift8
#print TVA.Dot8Chunk.psum
#check @TVA.Dot8Chunk.errB
example (ρ σ α v : ℚ) (c m : ℕ → ℚ) (j : ℕ) :
    TVA.Dot8Chunk.errB ρ σ α v c m (j + 1) = (1 + ρ + σ) * TVA.Dot8Chunk.errB ρ σ α v c m j +
      ρ * max |TVA.Dot8Chunk.psum v c j| (m j) + σ * |TVA.Dot8Chunk.psum v c (j + 1)| + α := rfl
example (ρ σ α v : ℚ) (c m : ℕ → ℚ) : TVA.Dot8Chunk.errB ρ σ α v c m 0 = 0 := rfl
example : TVA.Rules.Num.minNormal8 .f8e5 = 2 ^ (-14 : ℤ) := rfl
