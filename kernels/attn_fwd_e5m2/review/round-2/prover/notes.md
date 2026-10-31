# Prover's notes on attn_fwd_e5m2, round 2

## Changes

- README only; the Lean, the spec and the item digests are unchanged.
- Specification, "Where `Post` does not tell `o` from `0`": the test-data figures are now given per `N_CTX` and per causal or non-causal rows (a table). The text says that the fp8 floor explains the non-causal rows at `N_CTX ≥ 1024`, and that at `N_CTX = 128` and on causal rows most of the gap is `Num.DotE5m2`'s (L124).
- The main table adds `N = 128` for test-like data. The one-key, benchmark and `N = 2^14` figures now match the reviewers' round-2 runs.
- The `tolV` bullet has a new paragraph, "What `tolV` checks": `tolV ≥ 9/8·K + |attnRef|/8 + 2^-17`, so any average of the row's own values meets it, and alone it checks the key set and the range, not the weights.
- The Narrowings are updated to match.

## Responses

- **F6** CLARIFIED — the test-data figures are given per `N_CTX` and causal or not, and the gap at `N = 128` and on causal rows is attributed to `Num.DotE5m2`.
- **F7** CLARIFIED — as F6, including `N_CTX = 128`: 87% to 89.5% at `D = 64` causal, median `|attnRef|` up to `0.094`, floor at least `|attnRef|` on 26% to 54%.
- **F8** CLARIFIED — the new paragraph says what `tolV` checks: any average of the row's values meets it; the weights are checked only through `tolO`.

## Library requests

None.

## Gate

PASS, digest `53b679fa0085b99d`
