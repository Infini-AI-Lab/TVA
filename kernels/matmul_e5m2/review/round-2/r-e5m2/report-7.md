# Review of matmul_e5m2, round 2, by r-e5m2

## Verdict

accept
Both reworded README sentences are correct. My earlier remark is settled, and I accept every item as before.

## Findings

None.

## Accepted

All items.

## Checked

- "That growth of the bound, by about `1 + ρ + σ` per chunk": each step of `errAt` multiplies the previous bound by `1 + ρ + σ`.
- "the error bound on a fed-back accumulator grows by at most a factor `1 + 2^-11` per chunk": this holds for `DotE5m2ChunkLoss`. Its second arm gives the composed bound the factor `1 + σ + ρ' = 1 + 2^-11` (`errL`), and `errM` is at most `errL`.

## Not checked

Nothing.
