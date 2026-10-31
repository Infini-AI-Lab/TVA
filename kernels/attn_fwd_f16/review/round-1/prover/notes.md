# Prover's notes on attn_fwd_f16, round 1

## Changes

- `lean/AttnFwd.lean`: the module docstring gives the test as lines 631–695 and the benchmark as lines 698–780; `testInst`'s docstring calls it a covered instance with the test's smallest `N_CTX` and `HEAD_DIM`, one head (`Z = H = 1`, not a test shape).
- `lean/AttnFwd/Model.lean`: `HeadOk`'s docstring gives the host's score magnitudes as below `23` (test) and `240` (benchmark) except with probability below `10^-6`, with the Chernoff and union-bound argument.
- `README.md`: the benchmark's line range, the same estimate and argument (Specification), the tolerance at those values (about `0.43%` and `2.9%`), and the wording of `satisfiable` (Proof).
- `lean/AttnFwd/RowAcc.lean`, `lean/AttnFwd/Step.lean`: proof docstrings and a comment that cited line 87 for `acc * alpha[:, None]` now cite line 95.
- Docstrings and the write-up only: no item digest changed.

## Responses

- **F1** CLARIFIED — the test and benchmark line ranges corrected in AttnFwd.lean and README; other line references checked
- **F2** CLARIFIED — one estimate (below 23 and 240, except with probability below 1e-6) with its Chernoff argument in README and HeadOk's docstring
- **F3** REFERRED — library wording issue L122
- **F4** CLARIFIED — line ranges corrected; testInst called a covered one-head instance, not a test shape; one score estimate
- **F5** REFERRED — library wording issue L123

## Library requests

None.

## Gate

PASS, digest `3031db2a50b0a036`
