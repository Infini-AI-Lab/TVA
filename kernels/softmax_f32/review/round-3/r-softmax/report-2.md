# Review of softmax_f32, round 3, by r-softmax

## Verdict

accept
Library round 24 changed only the fp8 `tl.dot` facts. It withdrew the whole-dot facts (L125), added
the chunk facts (L124) and moved their constants. None of these is a fact this kernel cites, so I
re-accept the five re-opened assumption items, and every item.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of docs/rules/num.md and lean/TVA/Rules/Num/Sm90.lean, against L124 and L125 in ISSUES.md and the coordinator's note in REVIEW.md.
- That `MathTotal`, `DivTotal`, `ExpF32`, `ExpF32NegInf` and `DivFull` keep the same rows in "Facts the library states" and in the operation table (`tl.exp` on `float32`, `/` on `float32`), with the same constants. Their Lean definitions are untouched, and so are the page's sections on the approximate functions and on division.

## Not checked

- The kernel's own files, unchanged since round 3, and the items that did not re-open (model, map, conditions, spec, faithful).
- The new fp8 chunk facts themselves, which this kernel does not use.
