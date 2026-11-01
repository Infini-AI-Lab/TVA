# Review of matmul_e5m2, round 3, by r-e5m2b

## Verdict

accept

The prover answered F6 (CLARIFIED). `OutSpec`'s docstring now names each chunk's exact magnitude `chunkAbs`, so F6 is settled and I release it. The gate is unchanged at 752f03b36e81309e, and I keep my acceptance of all items.

## Findings

- **release F6** — settled: `OutSpec`'s docstring now names `chunkAbs`, so it lists all of `errAt`'s inputs.

## Accepted

All items.

## Checked

- `OutSpec`'s docstring says `E = tol va vb K` comes from the exact partial sums, each chunk's largest lifted product and each chunk's exact magnitude `chunkAbs`. Those are exactly `errAt`'s inputs (`partSum`, `chunkMax` of `liftV`, `chunkAbs` of `prodV`). No definition changed.
- The README's `torch.randn` sentence at `K = 4096` now says "roughly 40–50% of draws", and the sentence on the sign says the same. That agrees with my round-2 sampling: 9 of the first 20 seed-0 draws, and 37.5% of 120.

## Not checked

- Nothing beyond the docstring and the README sentence, as asked; no new sampling.
