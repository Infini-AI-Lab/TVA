# Review of grouped_matmul_tma_f16, round 2, by r-gmm

## Verdict

accept
The prover's round-1 answers settle both of my findings. The README now compares bfloat16's error with Post's whole bound on c, last float16 rounding included, and its figures agree with my exact computation. L121 records my library report as I raised it. Nothing else changed: the gate is still `bcf71a93c1d432fe` and every item digest is the one I accepted. So I release F3 and F4 and accept every item again.

## Findings

- **release F3** — The README now gives Post's whole bound (0.253% at K = 2^13) and the thresholds K = 13633 and K = 15617 (accumulation alone), matching my round-1 work/tol.py.
- **release F4** — L121 records the scope-5 alignment gap faithfully; it is open for a later library round and does not block.

## Accepted

All items.

## Checked

- The prover's response and notes for round 1. In the README, the changed tolerance paragraph (Specification) and the Narrowings entry `K ≤ 2^13`. Their figures (5.20e-4, 7.43e-4 and 0.2528% at K = 128, 1024 and 2^13; T = 214 from K = 13633; T = 245 from K = 15617) match round 1's work/tol.py.
- L121 in `docs/library-review/ISSUES.md`, against my F4.
- r-gmm2's round-1 report. Its F1 is the same threshold slip as my F3, and the README change settles it too. Its F2 (descriptors rebuilt in a loop) is referred as L120, and I have no finding of my own on it.
- The gate digest, unchanged at `bcf71a93c1d432fe`. The Lean is unchanged (the prover's notes; the item digests are the same).

## Not checked

- L120 and L121 themselves: library matters, open for a later library round.
