# Review of matmul_e5m2, round 2, by r-e5m2b

## Verdict

accept

The prover fixed the phrase about a fed-back accumulator's growth; the gate is unchanged at e127e9a2077f0921. Both reworded sentences are true, so I confirm the acceptance of all items.

## Findings

None.

## Accepted

All items.

## Checked

- "That growth of the bound, by about `1 + ρ + σ` per chunk" refers to the tolerance, `errB`, which compounds by `1 + ρ + σ` per chunk.
- "the error bound on a fed-back accumulator grows by at most a factor `1 + 2^-11` per chunk" is true. `errM ≤ errL`, and `errL` compounds by `1 + ρ' + σ = 1 + 2^-11`, as in the closed form `errL n ≤ (1 + 2^-11)^n · Σ_{j<n} (…)` (`errL_le`). num.md states the growth in the same sense.

## Not checked

- Only these two sentences, as asked.
