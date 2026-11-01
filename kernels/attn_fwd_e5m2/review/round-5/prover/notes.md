# Prover's notes on attn_fwd_e5m2, round 5

## Changes

- README only; the Lean, the spec and the item digests are unchanged.
- The user's goal for this re-proof (2026-11-01): "use ChunkLoss to tighten the fp8 kernel bounds."
- The benchmark figures at `N = 1024` are recomputed over all rows of 33 heads per `D` and causal mode (seeds `101` to `909` by `101`, and `1001` to `1024`), then widened to cover both reviewers' samples: `D = 64` `tolO` `0.35` to `0.44`, `0` meets `Post` on 25% to 39%; `D = 128` `tolO` `0.53` to `0.89` (non-causal `0.65` to `0.89`, causal `0.53` to `0.65`), `0` meets `Post` on 41% to 61%; about 4 to 12.5 times the floor; `tolO` falls by 11% to 26% (`D = 64`) and 20% to 49% (`D = 128`) on the same head, causal rows least; the loss arm is taken in 47% to 63% of the blocks.
- Updated in the Specification's table and its note, "Where `Post` does not tell `o` from `0`", "Read this first", "What changed (ChunkLoss)" and the Narrowings.

## Responses

- **F12** CLARIFIED — the `D = 128` ranges now cover your heads: `tolO` `0.53` to `0.89`, `0` meets `Post` on 41% to 61%, `tolO` about 4 to 12.5 times the floor.
- **F13** CLARIFIED — the falls are now 11% to 26% (`D = 64`) and 20% to 49% (`D = 128`), causal rows least, and the `D = 128` `tolO` range `0.53` to `0.89` covers your `0.65` to `0.84`.

## Library requests

None.

## Gate

PASS, digest `3d7848ee51b5e6a6`
