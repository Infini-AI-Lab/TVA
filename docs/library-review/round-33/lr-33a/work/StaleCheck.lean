import TVA
/-! lr-33a: the built `TVA.Rules.Num.Dot16Chunk` (its olean predates the source's last edit) against the source. -/
open Lean Elab Command in
elab "#doc " n:ident : command => do
  let env ← getEnv
  let some d ← findDocString? env n.getId | logInfo m!"no doc"
  logInfo m!"{d.length} chars: {(d.take 120)}"
#print TVA.Rules.Num.chunks16
#print TVA.Rules.Num.chunk16
#print TVA.Rules.Num.rhoChunk16
#print TVA.Rules.Num.sigmaChunk16
#print TVA.Rules.Num.alphaChunk16
#print TVA.Rules.Num.rhoLossChunk16
#print TVA.Rules.Num.kappaChunk16
#print TVA.Rules.Num.DotChunk16Loss
#doc TVA.Rules.Num.DotF16Chunk
#doc TVA.Rules.Num.chunk16
#doc TVA.Rules.Num.DotChunk16Loss
