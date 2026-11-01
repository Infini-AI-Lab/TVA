# Prover's notes on matmul_e5m2, round 2

## Changes

- `OutSpec`'s docstring (Defs.lean) now names each chunk's exact magnitude `chunkAbs` among what the tolerance uses (F6). Wording only; the gate digest is unchanged.
- README, Specification ("What it does not reject"): at `K = 4096` the tolerance on `torch.randn` data is below `|S|` on roughly 40-50% of draws (10 of my 20, about 40% of r-e5m2's 200), not on half; the sentence on fixing the sign says the same (r-e5m2's wording remark). No figure changed.

## Responses

- **F6** CLARIFIED — OutSpec's docstring now says the tolerance also uses each chunk's exact magnitude `chunkAbs`; with it, r-e5m2's remark on "half the draws" is fixed in the README.

## Library requests

None.

## Gate

PASS, digest `752f03b36e81309e`
