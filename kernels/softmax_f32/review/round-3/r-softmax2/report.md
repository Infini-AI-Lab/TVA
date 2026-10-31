# Review of softmax_f32, round 3, by r-softmax2

## Verdict

accept
The prover changed only wording, and the gate digest is the same, so every item keeps the digest I accepted in round 2. The README's coverage sentence now states the `in_off` limit correctly, which settles F8, and it still matches the Lean.

## Findings

- **release F8** — 'Covered in particular' now takes the limit from in_off (the last row starts below element 2^31), with correct examples (131072 × 16384, 169467 × 12672).

## Accepted

All items.

## Checked

- The README's new 'Covered in particular' text against `Inst.ok` (`in_off`, `out_off`, `rows_grid`, `x_len`, `y_len`): the limit, the two examples, and the int32 wrap beyond the limit.
- The new `lane_facts` docstring in `Err.lean` against the theorem's statement: it now gives the bound the theorem proves, `(2^-22 + 2^-33)·a + (3·2^-24 + 2^-33)·(w·a) + 2^-139`, and `+0` for a dead lane.
- The round-2 responses to F7 and F8 in `REVIEW.md`, and that the gate digest is unchanged (`b822ec44212cc840`).

## Not checked

- Everything else is unchanged since round 2 (same gate digest, same item digests), so I did not re-check it: the model, map, conditions, specification, assumptions and proofs.
