# Review of matmul_e5m2, round 2, by r-e5m2b

## Verdict

accept

Library round 26 re-opened assumption:dot, dot_total, cvt and cvt_total, and spec, at gate e127e9a2077f0921. The kernel's files did not change. Nothing these items rely on changed in meaning, so I accept all items again.

## Findings

None.

## Accepted

All items.

## Checked

- In `Rules.Num.Dot8Chunk`, the diff only adds. It adds a module-docstring paragraph, `chunkAbs8`, `DotChunkLoss`, `kappaChunk8`, `rhoLossChunk8`, `DotF8ChunkLoss`, the four loss facts, `DotChunkLoss.chunkAcc`, `DotChunkLoss.mono` and `DotF8ChunkLoss.chunk`. So `DotE5m2Chunk`, `DotF8Chunk`, `DotChunkAcc`, `chunk8`, `chunkSum8`, `rhoChunk8 = 2^-7`, `sigmaChunk8 = 2^-12` and `alphaChunk8 = 2^-126` are unchanged.
- In `Lemmas.Dot8Chunk`, only one module-docstring bullet changed; the rest is new lemmas. `DotF8Chunk.err`, `errB`, `chain_err`, `liftMag_zero`, `liftMag_nonneg` and `abs_mul_le_liftMag`, which the kernel's proof and spec use, are unchanged. `liftMag` and `minNormal8 .f8e5 = 2^-14` are unchanged.
- `CvtF32E5m2`, `CvtTotal` and `DotTotal` are untouched.
- In num.md, the fp8 `tl.dot` row of "Facts the library states" still lists `DotE5m2Chunk P` and adds the loss facts as an alternative. The reviewer-check rows for `DotE5m2Chunk`, the e5m2 conversion and the totality facts are unchanged. The other changes are new loss-fact text and rows, plus the L127 and L130 rewording.
- REVIEW.md lists only the two files as changed for these items, and no declaration. The round-26 author notes also say that no existing constant changed meaning.
- In work file `Round26`, `lake env lean` prints these definitions and checks the three chunk constants and `minNormal8 .f8e5`. `main`, `consistent` and `satisfiable` depend only on `propext`, `Classical.choice` and `Quot.sound`.

## Not checked

- I did not check whether the new loss facts are true on the device or what their evidence shows. This kernel does not cite them.
