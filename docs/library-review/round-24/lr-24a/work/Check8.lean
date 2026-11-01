import TVA

/-! lr-24a: the fp8 chunk facts of round 24 (L124) and the whole-dot lemma that replaced the withdrawn
`DotE4m3`, … (L125), as Lean elaborates them in the tree of round 25, and their axioms; the per-chunk constants
against the wgmma derivation of `docs/rules/num.md`. Checked with `lake env lean` on this file only. -/

open TVA TVA.Rules.Num

#print TVA.Rules.Num.DotChunkAcc
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.DotE5m2Chunk
#print TVA.Rules.Num.chunk8
#print TVA.Rules.Num.chunks8

-- the fp8 thresholds of `liftMag` (moved to `Rules/Num.lean` in round 25), unchanged at the fp8 dtypes
example : minNormal8 .f8e5 = 2 ^ (-14 : ℤ) := rfl
example : minNormal8 .f8e4nv = 2 ^ (-6 : ℤ) := rfl

-- the withdrawn facts are gone
-- (checked in Withdrawn.lean: every withdrawn name is unknown, Withdrawn.out)

-- the whole-dot lemma (round 24's `Dot8Chunk.LiftAcc`, `Rules.Num.DotLiftAcc` since round 25)
#check @TVA.Dot8Chunk.DotF8Chunk.liftAcc
#check @TVA.Dot8Chunk.DotF8Chunk.err
#check @TVA.Dot8Chunk.errB_le
#print axioms TVA.Dot8Chunk.DotF8Chunk.liftAcc
#print axioms TVA.Dot8Chunk.DotF8Chunk.err
#print axioms TVA.Dot8Chunk.DotChunkAcc.liftErr
#print axioms TVA.Dot8Chunk.chain_err
#print axioms TVA.Dot8Chunk.errB_le
#print axioms TVA.Rules.Num.DotChunkAcc.mono
#print axioms TVA.Platform.ref.dotChunkAcc
#print axioms TVA.Platform.sm90ref.dotE4m3Chunk
#print axioms TVA.Platform.sm90ref.dotE5m2Chunk
#print axioms TVA.Platform.sm90ref.dotE4m3E5m2Chunk
#print axioms TVA.Platform.sm90ref.dotE5m2E4m3Chunk

-- the per-chunk constants against the wgmma derivation: 33 (1 + 2^-13) 2^-13 <= 2^-7, 2^-13 <= 2^-12,
-- 34 * 2^-139 <= 2^-126; and the margin on rho is under 2 (so the device's worst per-chunk loss is about half of rho)
example : (33 : ℚ) * (1 + 2 ^ (-13 : ℤ)) * 2 ^ (-13 : ℤ) ≤ rhoChunk8 := by unfold rhoChunk8; norm_num
example : rhoChunk8 ≤ 2 * ((33 : ℚ) * (1 + 2 ^ (-13 : ℤ)) * 2 ^ (-13 : ℤ)) := by unfold rhoChunk8; norm_num
example : (2 : ℚ) ^ (-13 : ℤ) ≤ sigmaChunk8 := by unfold sigmaChunk8; norm_num
example : (34 : ℚ) * 2 ^ (-139 : ℤ) ≤ alphaChunk8 := by unfold alphaChunk8; norm_num
