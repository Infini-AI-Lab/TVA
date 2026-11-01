# Review of softmax_f32, round 3, by r-softmax2

## Verdict

accept
Library round 24 (L124, L125) changed only the fp8 `tl.dot` facts. None of the facts this kernel cites, nor their constants or the page's statements about them, changed. I re-accept the five items that went stale and keep the others.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of `docs/rules/num.md` and `lean/TVA/Rules/Num/Sm90.lean`: only the fp8 dot facts, the new chunk facts and the moved `chunks8`, `relDot8` and `absDot8` changed.
- The five cited facts (`MathTotal`, `DivTotal`, `ExpF32`, `ExpF32NegInf`, `DivFull`): their Lean definitions, their listing in "Facts the library states", their reviewer-table rows and their constants are all unchanged.
- The coordinator's note in `REVIEW.md`, and L124 and L125 in the issue list.

## Not checked

- The new fp8 chunk facts themselves, which this kernel does not use.
- Everything outside the re-opened items. The kernel's files and the other items' digests are unchanged since round 3.
