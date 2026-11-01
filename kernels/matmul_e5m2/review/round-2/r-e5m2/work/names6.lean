import MatmulE5m2

/-! Re-review on `DotE5m2ChunkLoss`: the new tolerance, the `dot` assumption, the library's
`errM`, and the statements of the rejection lemmas, printed with full names. -/

open TVA.Rules.Num in
example : kappaChunk8 = 2 ∧ rhoLossChunk8 = 2 ^ (-12 : ℤ) := ⟨rfl, rfl⟩

#print TVA.Rules.Num.DotE5m2ChunkLoss
#print TVA.Rules.Num.DotF8ChunkLoss
#print TVA.Rules.Num.DotChunkLoss
#print TVA.Rules.Num.chunkAbs8
#print TVA.Dot8Chunk.errM
#print Kernels.MatmulE5m2.chunkAbs
#print Kernels.MatmulE5m2.errAt
#print Kernels.MatmulE5m2.tol
#print Kernels.MatmulE5m2.Assumptions
#check @Kernels.MatmulE5m2.tol_le_tolA
#check @Kernels.MatmulE5m2.tol_spike
#check @Kernels.MatmulE5m2.zero_rejected
#check @Kernels.MatmulE5m2.drop_rejected
#check @Kernels.MatmulE5m2.chunk_step
#print axioms Kernels.MatmulE5m2.main
#print axioms Kernels.MatmulE5m2.consistent
#print axioms Kernels.MatmulE5m2.satisfiable
#print axioms Kernels.MatmulE5m2.tol_spike
#print axioms Kernels.MatmulE5m2.zero_rejected
#print axioms Kernels.MatmulE5m2.drop_rejected
