# Review of softmax_f32, round 3, by r-softmax

## Verdict

accept
Library round 25 (L126) changed only the 16-bit `tl.dot` facts and the definitions they use
(`DotLiftAcc`, `liftMag`, `minNormal8`). None of these is a fact this kernel cites, and the kernel has no `tl.dot`.
So I re-accept the five re-opened assumption items, and every item.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of docs/rules/num.md and lean/TVA/Rules, against L126 in ISSUES.md and the coordinator's note in REVIEW.md.
- That `MathTotal`, `DivTotal`, `ExpF32`, `ExpF32NegInf` and `DivFull` are unchanged in Lean, in their rows of "Facts the library states" and the operation table (`tl.exp` on `float32`, `/` on `float32`), and in their constants. The satisfiability text still covers them.

## Not checked

- The kernel's own files, unchanged, and the items that did not re-open (model, map, conditions, spec, faithful).
- The new 16-bit dot facts and their evidence, which this kernel does not use.
