# Review of softmax_f32, round 3, by r-softmax

## Verdict

accept
The round-2 answers changed wording only (the gate digest is still `b822ec44212cc840`). F7 is
settled: the README's coverage sentence now takes the `n_rows` limit from `in_off`, with correct
examples, and the `lane_facts` docstring states the bound the theorem proves. The README still
matches the Lean, and I still accept every item.

## Findings

- **release F7** — the README attributes the `n_rows` limit to `in_off` with correct examples, and the `lane_facts` docstring matches its statement.

## Accepted

All items.

## Checked

- The changed README sentence ("Covered in particular") against `Inst.ok.in_off` and `ptr_get`: `131071·16384 < 2^31`, `131072·16384 = 2^31`, `169466·12672 < 2^31`, `169467·12672 = 2^31 + 2176`, and one more row breaks `in_off`; the column offsets are added to the pointer in 64 bits.
- The changed `lane_facts` docstring against the theorem's statement.
- REVIEW.md's round-2 responses: wording only, the same gate digest and item digests.

## Not checked

- What did not change since round 2 (the model, map, conditions, specification, assumptions and the rest of the write-up), which I checked then; their digests are unchanged.
