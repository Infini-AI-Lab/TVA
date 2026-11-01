# Review of softmax_f32, round 3, by r-softmax2

## Verdict

accept
Library round 26 changed only fp8 `tl.dot` material, plus wording in the `tl.dot` sections of the rule page. None of the facts this kernel cites, nor their constants, listings, page statements or the lemmas its proof uses, changed. I re-accept the five items that went stale and keep the others.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of `docs/rules/num.md` and `lean/TVA`:
  - the page gains the new fp8 `ChunkLoss` facts and wording fixes in the `tl.dot` sections (L127, L130); every removed or added line concerns `tl.dot`;
  - the Lean changes are only in the two `Dot8Chunk` files.
- The five cited facts (`MathTotal`, `DivTotal`, `ExpF32`, `ExpF32NegInf`, `DivFull`): their definitions, listings, reviewer-table rows, constants and statements are all unchanged.
- The issue list (L124, L127 to L133), the round-26 author notes, and the per-item changes in `REVIEW.md`.

## Not checked

- The new fp8 loss facts and lemmas, which this kernel does not use.
- Everything outside the re-opened items. The kernel's files and the other items' digests are unchanged.
