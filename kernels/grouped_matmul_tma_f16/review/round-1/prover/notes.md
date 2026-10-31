# Prover's notes on grouped_matmul_tma_f16, round 1

## Changes

- README.md (Specification, tolerance paragraph; Narrowings, `K ≤ 2^13`): the `bfloat16` comparison now uses `Post`'s whole bound on `c`, with the last `float16` rounding (`0.253%` at `K = 2^13`, recomputed exactly), and the threshold `K = 13633` (`T = 214`), with `K = 15617` (`T = 245`) for the accumulation part alone (F1, F3). Nothing else changed; the Lean is unchanged.

## Responses

- **F1** CLARIFIED — README now states Post's whole relative bound (0.253% at 2^13) and the bfloat16 threshold K = 13633 (15617 for the accumulation alone).
- **F2** REFERRED — descriptors rebuilt in a runtime loop while a TMA store may be in flight: library issue L120.
- **F3** CLARIFIED — the comparison now includes Post's last float16 rounding (0.253% < 0.389% at K = 2^13; threshold K = 13633).
- **F4** REFERRED — scope 5 and the allocator's alignment: library issue L121.

## Library requests

None.

## Gate

PASS, digest `bcf71a93c1d432fe`
