# Review of matmul_e5m2, round 2, by r-e5m2b

## Verdict

accept

The prover reworded the README for library issue L127; the gate digest is unchanged at e127e9a2077f0921. The new sentences are true and agree with num.md and L127. No figure or part of the claim's description changed. I confirm the acceptance of all items.

## Findings

None.

## Accepted

All items.

## Checked

- "What it does not reject" now calls the tolerance the error `DotE5m2Chunk` allows, not the device's own error. It says the fact charges each chunk `ρ·M` with `M ≥ |s|`, so a large partial sum costs the tolerance about `ρ + σ` of itself per chunk. That matches `errB (j+1) = (1 + ρ + σ)·errB j + ρ·max |S j| (m j) + σ·|S (j+1)| + α`.
- The new paragraph agrees with num.md's paragraph ending "The loss facts, next, state a second bound for each step from this", and with L127. It says the growth comes from the fact's form, each chunk relative to `M ≥ |s|`, and from its margin, `2^-7` about twice the window's worst chunk. It says the growth does not come from the device, whose truncation toward zero keeps its error at most linear in the chunks.
- The `DotE5m2ChunkLoss` sentence states the second arm correctly: `2·A_j + 2^-12·M + 2^-12·|s + c| + 2^-126`, with `A_j` the chunk's exact `Σ|a·b|`. It gives the growth as at most `1 + 2^-11` per chunk, which holds because `errM ≤ errL` and `errL` compounds by `1 + ρ' + σ`. The claim indeed cites `DotE5m2Chunk`, not the loss fact, and the tolerance is the one described.
- The "`K ≤ 4096`" narrowing and the closing paragraph now say the same in brief.
- No figure changed in the diff: `13.7%`, `2753`, `179.3%`, `6.3%`, `73.9%`, and the `0.133 → 0.137` and `1.708 → 1.793` rows. No remaining sentence calls the tolerance the device's error. Since the last commit, claim.md has changed only in its round-26 digests and timestamp.
- Wording note, not a finding: "a fed-back accumulator grows by at most `1 + 2^-11` per chunk" means the bound on its error. The paragraph's frame, the tolerance's growth, makes that clear.

## Not checked

- I did not re-check the device statement itself (linear error growth), only that the README states it as num.md and L127 do.
