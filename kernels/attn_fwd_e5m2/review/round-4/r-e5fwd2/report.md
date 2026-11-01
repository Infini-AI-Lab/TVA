# Review of attn_fwd_e5m2, round 4, by r-e5fwd2

## Verdict

accept
F10 is answered and released. The causal shares at N_CTX ≥ 1024 now agree with my outputs, a
sentence explains row 0's weight, and every smaller mismatch F10 listed is covered. No item digest
changed, so every item stays accepted. There is one new non-blocking DOC finding. Several of the
widened ranges still miss my round-3 outputs, although the table note says each range covers both
reviewers' evaluations.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — some new ranges miss my round-3 outputs, though the table note says each covers both reviewers. Examples: on the benchmark at D = 128, 0 is rejected on 28–36% of causal elements (README 34–38%), and tolV at D = 64, N = 4096 is 6.6 (README 7.3).
- **release F10** — the causal shares (4.5–6.6% and 1–2.2%) and the row-0 sentence are in, and every smaller mismatch F10 listed is now covered.

## Accepted

All items.

## Checked

- Every changed figure in the README's main table, the per-N_CTX table, the benchmark paragraph and "Read this first", against my round-3 tolerance.out (test-like and benchmark-like rows, two seeds; no new samples).

## Not checked

- The prover's three-head samples themselves. I did not re-sample.
