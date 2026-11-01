# Review of matmul_e5m2, round 2, by r-e5m2

## Verdict

accept
Library round 26 re-opened `spec` and the assumptions `cvt`, `cvt_total`, `dot` and `dot_total`. The round only adds facts and lemmas and changes wording. Nothing the re-opened items rely on changed in meaning, so I accept every item again.

## Findings

None.

## Accepted

All items.

## Checked

- The Lean diff of the rules and lemmas only adds new things:
  - in the rules: `chunkAbs8`, `DotChunkLoss`, `kappaChunk8`, `rhoLossChunk8`, `DotF8ChunkLoss` and the four `…ChunkLoss` facts, with `DotChunkLoss.chunkAcc`, `DotChunkLoss.mono` and `DotF8ChunkLoss.chunk`;
  - in the lemmas: `errM`, `chain_errM`, `errM_le_errB`, `errL`, `DotChunkLoss.err`, `DotF8ChunkLoss.err` and the reference-platform lemmas.

  The only removed lines are in module docstrings.
- These are unchanged: `DotE5m2Chunk`, `DotF8Chunk`, `DotChunkAcc` and the constants `2^-7`, `2^-12`, `2^-126`; `liftMag` and `minNormal8` (`2^-14` for `float8e5`); `DotLoop.bound`; `CvtF32E5m2`, `CvtTotal` and `DotTotal`.
- `docs/rules/num.md`: the rows for `DotE5m2Chunk`, `CvtF32E5m2` and the total rules are unchanged. The fp8 entry of the reviewer's table now also allows the loss facts. The reworded compounding sentence (L127), the chunk-model opening (L130) and the new loss-fact sections give the cited fact no new meaning.
- `work/names5.lean`, checked with `lake env lean`:
  - the constants and `minNormal8 .f8e5` hold by `rfl`;
  - `DotE5m2ChunkLoss` gives `DotE5m2Chunk`;
  - the cited facts, `errAt`, `tol`, `tolA`, `liftV` and `Assumptions` print as before;
  - `main`, `zero_rejected` and `drop_rejected` use only the three standard axioms.
- Non-blocking, no finding: the README calls the tolerance "the device's own allowed error". After L127 it is best read as the error the cited fact allows. The library's new `DotE5m2ChunkLoss` would give a smaller bound, but this claim does not need to use it.

## Not checked

Nothing.
