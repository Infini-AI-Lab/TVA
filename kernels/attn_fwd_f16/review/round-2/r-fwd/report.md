# Review of attn_fwd_f16, round 2, by r-fwd

## Verdict

accept
The prover settled my round-1 findings with wording changes only: the gate is unchanged at `3031db2a50b0a036` and no item digest changed, so my round-1 judgment of every item stands. The line ranges are corrected, the score estimate is now one figure with a sound argument, and my library report is logged faithfully as L122. I release all three and accept every item.

## Findings

- **release F1** — the test (631–695) and benchmark (698–780) ranges in `AttnFwd.lean` and the README now match `source/`.
- **release F2** — one estimate (below 23 and 240, except with probability below `10^-6`) with its Chernoff and union-bound argument, which I re-checked.
- **release F3** — L122 records the library wording report faithfully; it does not block.

## Accepted

All items.

## Checked

- The prover's response and notes, and the changed text: `AttnFwd.lean`'s module docstring and `testInst`'s docstring, `HeadOk`'s docstring in `Model.lean`, and the README's host ranges, Specification estimate and Proof wording.
- The Chernoff bound on `Σ_e |X_e·Y_e|` at `HEAD_DIM = 128` with a union bound over every pair: at most `1.1·10^-7` (test, threshold 23) and `7.9·10^-7` (benchmark, threshold 240) (`work/chernoff.py`, `work/chernoff.out`).
- The README's tolerances at `X = 23` (`D = 128`, `N = 4096`: 0.43%) and `X = 240` (`N = 2^14`: 2.9%), against my exact evaluation of `tolO`.
- L122 in `docs/library-review/ISSUES.md` against my F3.
- r-fwd2's round-1 filing and report, now open to me: its findings agree with mine.

## Not checked

- The proofs line by line (Lean checks them; no digest changed since round 1).
