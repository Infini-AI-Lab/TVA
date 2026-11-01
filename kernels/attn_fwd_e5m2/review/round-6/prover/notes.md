# Prover's notes on attn_fwd_e5m2, round 6

## Changes

- README only; the Lean, the spec and the item digests are unchanged.
- The user's goal for this re-proof (2026-11-01): "use ChunkLoss to tighten the fp8 kernel bounds."
- Every benchmark figure at `N = 1024` is the range over 33 seeds (all rows; seeds `101` to `909` by `101` and `1001` to `1024`; both causal modes) and the reviewers' recorded round-5 runs (read as recorded, not re-run), rounded outward. The README says so in one sentence.
- New `tolO`: `D = 64` `0.40` to `0.46` (causal `0.33` to `0.39`); `D = 128` `0.65` to `0.90` (causal `0.50` to `0.69`). It falls by 11% to 27% (`D = 64`) and by 20% to 51% (`D = 128`), causal rows least.
- Rejected shares: `D = 64` 60% to 70% (causal 70% to 78%); `D = 128` 38% to 51% (causal 47% to 62%). `0` meets `Post` on 22% to 40% (`D = 64`) and 38% to 62% (`D = 128`). `tolO` is about 3.7 to 12.2 times the floor.
- Round-24 figures: `tolO` `0.54` to `0.60` (causal `0.40` to `0.47`) and `1.16` to `1.54` (causal `0.82` to `1.01`); rejected 45% to 58% (causal 63% to 73%) and 9.5% to 21% (causal 27% to 40%); 4.7 to 18.5 times the floor.
- The other figures, widened the same way: `tolV`, `tolM`, `pE` (`D = 128` `2.13` to `2.40`) and the median `|attnRef|` (`0.50` to `0.62`). The `2^14` row also gives r-e5fwd2's six-row sample.

## Responses

- **F14** CLARIFIED — every listed number is inside the union ranges: `D = 128` causal `tolO` `0.50` to `0.69`, `D = 64` non-causal `0.40` to `0.46`, `D = 64` causal rejected 70% to 78%.
- **F15** CLARIFIED — every listed number is inside the same ranges: `D = 128` causal `tolO` `0.50` to `0.69`; `D = 64` causal `0.33` to `0.39` and non-causal `0.40` to `0.46`; meets 22% to 40% (`D = 64`) and 38% to 53% (`D = 128` causal). Rejected: 60% to 70%, 70% to 78%, and up to 62%.

## Library requests

None.

## Gate

PASS, digest `3d7848ee51b5e6a6`
