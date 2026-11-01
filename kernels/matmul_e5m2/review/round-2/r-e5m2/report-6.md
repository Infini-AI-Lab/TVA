# Review of matmul_e5m2, round 2, by r-e5m2

## Verdict

accept
The README's rewording for L127 is accurate. It agrees with `docs/rules/num.md` and with L127, and it leaves the figures and the claim's description as they were. I accept every item as before.

## Findings

None.

## Accepted

All items.

## Checked

- The README diff: only text changed, and no figure. `claim.md` differs only in its generated digests and timestamps.
- "The tolerance is the error the cited fact allows, not the device's own error": `tol` is `errAt`, which is `DotE5m2Chunk` composed chunk by chunk.
- "A large partial sum costs the tolerance about `ρ + σ` of itself per chunk": `errAt`'s step has `ρ·max(|S_j|, m_j) + σ·|S_(j+1)|`.
- The growth of about `1 + ρ + σ` per chunk comes from the fact's form (`M ≥ |s|`) and its margin, not from the device. This matches num.md's paragraph ending "The loss facts, next, state a second bound for each step from this." The margin is `2^-7` against the window's `33·(1 + 2^-13)·2^-13`, and the device's truncation toward zero keeps its error linear in the chunks.
- `DotE5m2ChunkLoss`: the README's second arm, `2·A_j + 2^-12·M + 2^-12·|s + c| + 2^-126`, is what the fact states. The composed bound grows by a factor of at most `1 + 2^-11` per chunk (`errL`). The claim does not cite this fact.
- Non-blocking: "a fed-back accumulator grows by at most `1 + 2^-11` per chunk" refers to the bound composed for the accumulator, as L124 also puts it. The paragraph's opening sentence makes that clear.

## Not checked

Nothing.
