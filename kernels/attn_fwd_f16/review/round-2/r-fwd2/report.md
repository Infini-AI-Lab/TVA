# Review of attn_fwd_f16, round 2, by r-fwd2

## Verdict

accept
The prover fixed my DOC finding (F4) in the docstrings and the README, and referred my library
report (F5) to L123, which records it faithfully. No item digest changed: the gate is still
`3031db2a50b0a036`. So my round-1 acceptance of every item stands, and I raise nothing new.

## Findings

- **release F4** — the line ranges, the `testInst` wording and the one score estimate (below 23 and 240, with its Chernoff and union-bound argument) are fixed in `AttnFwd.lean`, `Model.lean` (`HeadOk`) and the README.
- **release F5** — referred to L123, which states the question as I raised it.

## Accepted

All items.

## Checked

- The prover's response and notes.
- The changed text of `lean/AttnFwd.lean` (module docstring, `testInst`), `HeadOk`'s docstring and the README (Specification, Proof). I checked their figures against my round-1 `work/scores.py` and `work/tol.py`.
- The line-87-to-95 corrections in `Model.lean`, `RowAcc.lean`, `Step.lean` and `Inner.lean`, against the source.
- L123 in `docs/library-review/ISSUES.md`.
- r-fwd's round-1 filing, now open to me. Its findings agree with mine, and none adds a blocking point.

## Not checked

- Nothing new beyond round 1's list: the item digests did not change.
