# Review of matmul_e5m2, round 2, by r-e5m2

## Verdict

accept
Library round 25 (L126) re-opened `spec` and the assumptions `cvt`, `cvt_total`, `dot` and `dot_total`. Nothing they rely on changed in meaning. `liftMag` moved to `TVA.Rules.Num` with the same definition. `minNormal8` keeps its fp8 values and only gains `float16` and `bfloat16` cases, which this kernel never uses. The cited facts and their constants are unchanged. I accept every item again.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of the Lean rules: `liftMag` is still `0` for `0` and `max |q| (minNormal8 d)` otherwise. `minNormal8` is still `2^-14` for `float8e5` and `2^-6` for `float8e4nv`. The only Lean changes are `DotF16` and `DotBF16`, now `DotLiftAcc`, the new definition `DotLiftAcc` with its two lemmas, and docstrings. This kernel cites none of these.
- `DotE5m2Chunk`, `DotF8Chunk`, `DotChunkAcc` and the constants `rhoChunk8 = 2^-7`, `sigmaChunk8 = 2^-12`, `alphaChunk8 = 2^-126` print as in round 24. So do `CvtF32E5m2` (`CvtRn binary32 e5m2`), `CvtTotal` and `DotTotal`.
- The diff of `docs/rules/num.md`: the rows for `DotE5m2Chunk`, `CvtF32E5m2` and the total rules are unchanged, and the facts are still listed. The new `mma.sync` text for the fp8 chunk facts restates the per-instruction model through the `float16` evidence. It does not change the fact or its constants.
- `work/names4.lean`, checked with `lake env lean`: `minNormal8 .f8e5 = 2^-14`, `minNormal8 .f8e4nv = 2^-6` and the unfolding of `liftMag .f8e5` all hold by `rfl`. `liftV`, `dotLift` and the five assumptions print as before. `main` uses only `propext`, `Classical.choice` and `Quot.sound`.

## Not checked

Nothing.
