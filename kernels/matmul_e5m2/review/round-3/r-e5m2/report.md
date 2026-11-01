# Review of matmul_e5m2, round 3, by r-e5m2

## Verdict

accept
F6 is settled: `OutSpec`'s docstring now names every input of the tolerance. My README remark on the `randn` draws is settled too. No definition changed, so I keep my acceptance of every item.

## Findings

None.

## Accepted

All items.

## Checked

- `OutSpec`'s new docstring: `E = tol` comes from the exact partial sums, each chunk's largest lifted product and each chunk's exact magnitude `chunkAbs`. Those are all of `errAt`'s inputs: `partSum`, `chunkMax` of `liftV`, and `chunkAbs` of `prodV`.
- The README's `randn` sentence at `K = 4096` now says the tolerance is below `|S|` on roughly 40–50% of draws (10 of its 20, about 40% of my 200). The sign sentence says the same. Both agree with my earlier figures.

## Not checked

Nothing.
