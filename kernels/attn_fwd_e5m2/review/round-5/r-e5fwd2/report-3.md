# Review of attn_fwd_e5m2, round 5, by r-e5fwd2

## Verdict

accept
Library round 26 added the fp8 loss facts and their lemmas and reworded docs/rules/num.md. It
changed no fact, constant or lemma that this kernel cites or its specification uses, so I accept
every item again. The notes flag, for the coordinator, a README sentence that L127 now contradicts.
It is outside the items and is not a finding.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of lean/TVA: additions only, apart from one reworded docstring bullet.
- The definitions the claim relies on, printed in work/lib26.out, with errB's equations and minNormal8 .f8e5 checked by rfl: DotE5m2Chunk, DotF8Chunk, DotChunkAcc and the three chunk constants, plus chunk8, chunkSum8, chunks8, chunkLift8, relLift8, psum and errB.
- The diff of docs/rules/num.md: the cited facts' rows and constants are kept.
- L124, L127 to L133 and the round-26 notes.
- The axioms of main, consistent and satisfiable.

## Not checked

- The new loss facts and their evidence. This kernel does not cite them.
