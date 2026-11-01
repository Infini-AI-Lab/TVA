# Review of grouped_matmul_tma_f16, round 3, by r-gmm

## Verdict

accept
Library round 26 re-opened `assumption:dot`, `assumption:dot_total` and `spec` through `docs/rules/num.md`, and nothing they rely on changed in meaning. The round adds fp8 facts and lemmas and rewords text. `DotF16`, `DotTotal`, their listings and constants, and the lemmas the specification uses are as I accepted them. The kernel's own files are unchanged, so I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- The diff against HEAD of `lean/TVA`: additions only (the `…ChunkLoss` facts in `Rules/Num/Dot8Chunk.lean`, lemmas in `Lemmas/Dot8Chunk.lean`) and one removed docstring bullet. Unchanged: `DotF16`, `DotLiftAcc`, `liftMag`, `minNormal8`, `relDot16`, `absDot16`, `DotTotal`, `DotLoop` (`bound`, `stepRel`, `stepAbs`), and the `Dot8Chunk` lemmas the proof uses.
- The diff of `docs/rules/num.md`. The `DotF16`/`DotBF16` rows of the kind table and of "Facts the library states" and the 16-bit bullet are unchanged. The chunk-model opening (L130) is reworded with the same 16-bit meaning. The rest is new fp8 material and the wording of L127.
- The summaries of L124 and L127–L133 in `docs/library-review/ISSUES.md`: none concerns the 16-bit dot. L126 is now `ACCEPTED`.
- `REVIEW.md`'s list of what changed: `file:docs/rules/num.md` only. The gate is at `ea472d49ce692007`.

## Not checked

- The new fp8 facts and lemmas, which this kernel does not cite.
