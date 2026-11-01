# Review of grouped_matmul_tma_f16, round 3, by r-gmm2

## Verdict

accept
Library round 26 re-opened `assumption:dot`, `assumption:dot_total` and `spec`, through `docs/rules/num.md` only. It added fp8 loss facts and fixed wording. Nothing these items rely on changed in meaning: `DotF16`, its listing and constants, `DotLiftAcc`, `DotTotal`, the 16-bit section of the chunk model, and the lemmas the specification's proof uses. I re-accept every item and raise nothing.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of `docs/rules/num.md`:
  - new rows and a section for `DotE4m3ChunkLoss` … (`DotChunkLoss`);
  - the weakening-lemma list extended;
  - the reworded opening of "`tl.dot` (the chunk model)", still 16 products per instruction for 16-bit operands on both paths.
  
  The operation table's `DotF16 P` row, the `DotF16` row of "Facts the library states" (`DotLiftAcc`, `relDot16`, `absDot16`) and the 16-bit paragraphs are unchanged.
- The diff of `lean/TVA`: additions only, to `Rules/Num/Dot8Chunk.lean` and `Lemmas/Dot8Chunk.lean`, plus a module docstring line. `liftMag_nonneg`, `abs_mul_le_liftMag` and `liftMag_zero`, which `Basic.lean` uses, are untouched, as are `Sm90.lean`, `Num.lean`, `DotLoop.lean` and `DotLift`.
- REVIEW.md's list of what changed per item, and the brief's account of L124 and L127–L133: none concerns the float16 dot this kernel cites.

## Not checked

- The new fp8 facts and their evidence: this kernel does not cite them.
