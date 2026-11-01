# Review of softmax_f32, round 3, by r-softmax2

## Verdict

accept
Library round 25 (L126) changed only the 16-bit `tl.dot` facts and the `liftMag`/`minNormal8` support they use. None of the facts this kernel cites, nor their constants or the page's statements about them, changed. I re-accept the five items that went stale and keep the others.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of `docs/rules/num.md` and `lean/TVA/Rules`: only `DotF16` and `DotBF16` (now `DotLiftAcc`), `liftMag`, `minNormal8`, the dot lemmas, the withdrawn `Dot8Chunk.LiftAcc` and the 16-bit text and rows changed.
- The five cited facts (`MathTotal`, `DivTotal`, `ExpF32`, `ExpF32NegInf`, `DivFull`) and the definitions they rest on (`MathAcc`, `MathAt`, `DivAcc`): their Lean definitions, listings, reviewer-table rows, constants and satisfiability statements are all unchanged.
- L126 in the issue list, and the coordinator's note and the per-item changes in `REVIEW.md`.

## Not checked

- The new 16-bit dot facts and their evidence, which this kernel does not use (it has no `tl.dot`).
- Everything outside the re-opened items. The kernel's files and the other items' digests are unchanged.
