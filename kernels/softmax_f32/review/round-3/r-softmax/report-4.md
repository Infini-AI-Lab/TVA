# Review of softmax_f32, round 3, by r-softmax

## Verdict

accept
Library round 26 added the fp8 chunk-loss facts and fixed the `tl.dot` wording in docs/rules/num.md.
Nothing this kernel cites or uses changed, so I re-accept the five re-opened assumption items, and
every item.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of docs/rules/num.md and lean/TVA, against L124 and L127–L133 in ISSUES.md and REVIEW.md's list of what changed per item.
- That `MathTotal`, `DivTotal`, `ExpF32`, `ExpF32NegInf` and `DivFull` are unchanged in Lean, in their rows of "Facts the library states" and the operation table (`tl.exp` on `float32`, `/` on `float32`), and in their constants. The satisfiability text still covers them.

## Not checked

- The kernel's own files, unchanged, and the items that did not re-open (model, map, conditions, spec, faithful).
- The new fp8 chunk-loss facts, lemmas and evidence, which this kernel does not use.
