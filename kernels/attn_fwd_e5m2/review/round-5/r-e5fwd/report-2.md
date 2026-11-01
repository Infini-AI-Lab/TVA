# Review of attn_fwd_e5m2, round 5, by r-e5fwd

## Verdict

accept
Library round 25 (L126) changed only the 16-bit dot facts. It also moved liftMag and minNormal8 to
Rules/Num.lean, with the same fp8 values. Nothing that `spec` or the re-opened assumptions rely on
changed in meaning, so I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- L126 in ISSUES.md, the coordinator's note and the per-item changes in REVIEW.md, and the diff of docs/rules/num.md and lean/TVA/Rules. The diff of lean/TVA/Lemmas/Dot8Chunk.lean for errB, chunkLift8 and relLift8.
- The gate snapshots' cones before and after: only spec gains the file Rules/Num.lean.
- work/lift_check.lean (output in lift_check.out): minNormal8 .f8e5 = 2^-14 and liftMag's body on float8e5, both by rfl; DotChunkAcc printed; main's axioms.

## Not checked

- Whether the new 16-bit facts are true, which the kernel does not cite. Kernel review trusts the library.
