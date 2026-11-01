# Review of attn_fwd_e5m2, round 5, by r-e5fwd

## Verdict

accept
The README's new sentences are true, and they agree with num.md's "How a kernel uses them" paragraph
and with L127. The compounding is now called the cited DotE5m2Chunk's bound, not the device's
behaviour, whose error grows at most linearly. DotE5m2ChunkLoss is correctly described as sharper
and not cited. No figure and nothing else in the claim's description changed. Every item stays
accepted.

## Findings

None.

## Accepted

All items.

## Checked

- The README diff against HEAD (four passages), against num.md's paragraph ending "The loss facts, next, state a second bound for each step from this.", the loss-facts section, L127, and claim.md's dot assumption.

## Not checked

- Nothing else: only README.md changed, and the gate is unchanged.
