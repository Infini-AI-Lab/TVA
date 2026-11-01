# Review of attn_fwd_e5m2, round 5, by r-e5fwd2

## Verdict

accept
Library round 25 restated the 16-bit dot facts, which this kernel does not cite. It also moved
`liftMag` and `minNormal8` to Rules/Num.lean with their fp8 values unchanged. Nothing the
re-opened items rely on changed in meaning, so I accept every item again.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of docs/rules/num.md and lean/TVA/Rules:
  - liftMag's definition is the same;
  - minNormal8 .f8e5 is still 2^-14 (checked by rfl in work/lib25.lean);
  - DotE5m2Chunk, its constants and the chunk definitions are unchanged;
  - the other changes concern DotF16, DotBF16, DotLiftAcc and docstrings only.
- The Assumptions fields are as before, and main, consistent and satisfiable use only the three standard axioms (work/lib25.out).
- L126 and the coordinator's note in REVIEW.md.

## Not checked

- The new 16-bit evidence (evidence/num-dot-sub16-round25), which this kernel's facts do not rest on.
