# Prover's notes on attn_fwd_e5m2, round 3

## Changes

- README only; the Lean, the spec and the item digests are unchanged.
- Every sampled figure in the README is recomputed over all rows of a head with `vIn`, for three heads, from Model.lean's expressions. Each range also covers both reviewers' round-3 evaluations, and the table note says how the figures were sampled.
- The causal rejection shares are now 4.5% to 6.6% (`N_CTX = 1024`) and 1% to 2.2% (4096). A new sentence says that the first causal query (one key, always rejected) is `1/N_CTX` of the rows.
- The benchmark figures are updated: `tolO` 1.16 to 1.54 at `D = 128`; `pE` medians 1.27 to 1.32 and 2.13 to 2.37, with a head's largest 1.46 to 1.65 and 3.05 to 3.87; rejection shares 45% to 57% and 9.5% to 19%. The other ranges move with them.

## Responses

- **F9** CLARIFIED — the figures are over all rows of three heads, with ranges covering both reviewers' numbers; the causal shares, benchmark `tolO` and `pE` maxima are corrected.
- **F10** CLARIFIED — as F9; row 0's weight is explained, and the smaller mismatches (`N_CTX = 128` shares, benchmark shares and `pE`, `tolM`, `vEff = 0` at the cap) are covered.

## Library requests

None.

## Gate

PASS, digest `4740b3814ba92693`
