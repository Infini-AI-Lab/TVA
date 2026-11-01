# Prover's notes on attn_fwd_e5m2, round 4

## Changes

- README only; the Lean, the spec and the item digests are unchanged.
- The ranges that F11 lists are widened to include r-e5fwd2's round-3 outputs: the test-like `tolO`, `tolV` and `tolM` entries, the benchmark's causal `tolO` and rejection shares, the median `|attnRef|` and floor shares, and "Read this first" (5 to 18.5 times the floor on the benchmark's data).
- The table note and the per-`N_CTX` caption no longer claim full coverage: the ranges come from all rows of three heads, widened to the reviewers' independent samples, and other samples may differ by a few percent at the ends of a range.

## Responses

- **F11** CLARIFIED — every listed range is widened to include the reviewer's numbers, and the note is reworded so it does not overclaim.

## Library requests

None.

## Gate

PASS, digest `4740b3814ba92693`
