# Review of grouped_matmul_tma_f16, round 2, by r-gmm2

## Verdict

accept
The prover settled both of my round-1 findings. The README now compares bfloat16's largest rounding error with Post's whole bound on the stored float16 result, and its numbers are right. Library issue L120 records my report as I made it. The Lean, the gate and every item's digest are unchanged, so my round-1 acceptance of every item stands. I raise nothing new.

## Findings

- **release F1** — the README now gives Post's whole bound (0.253% at K = 2^13) and the threshold K = 13633, with 15617 for the accumulation part alone, as recomputed exactly
- **release F2** — library issue L120 states the doubt faithfully (descriptors rebuilt per group while a TMA store through the old one may be in flight) and stays open; a LIBRARY_REPORT does not block

## Accepted

All items.

## Checked

- The prover's response.json and notes.md for round 1: F1 and F3 CLARIFIED, F2 and F4 REFERRED (L120, L121); only README.md changed, and the gate digest is still `bcf71a93c1d432fe`.
- The README's new sentences (Specification, end of the tolerance paragraph; Narrowings, `K ≤ 2^13`) against my exact recomputation (round-1 work/tol.py): `r + (1 + r)·2^-11` with `r = (1 + ρ)^T − 1` is 5.201·10^-4, 7.431·10^-4 and 2.5284·10^-3 at K = 128, 1024 and 2^13; it reaches 2^-8/(1 + 2^-8) at T = 214 (K = 13633), and `r` alone reaches it at T = 245 (K = 15617).
- `docs/library-review/ISSUES.md` L120 (its problem statement, the coordinator's observation, its status) against my F2; and L121.
- r-gmm's round-1 report and filing: its DOC is the same point as F1, and its LIBRARY_REPORT (scratch alignment) does not touch this host's fresh allocation.

## Not checked

- The coordinator's PTX reading and stress run recorded in L120: an observation for the library's later round, not part of this claim.
