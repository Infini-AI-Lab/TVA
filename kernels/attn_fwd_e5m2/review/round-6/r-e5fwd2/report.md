# Review of attn_fwd_e5m2, round 6, by r-e5fwd2

## Verdict

accept
Every item stays accepted, since no item changed. The new benchmark ranges cover F12's headline
numbers, so I release F12. The table's D = 128 causal tolO range (0.53–0.65) still misses my 0.69,
which F12 listed. I raise that as a new, narrower non-blocking DOC.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — the table's benchmark D = 128 causal tolO range (0.53–0.65) misses my 0.689. Smaller misses: D = 64 non-causal 0.42–0.44 against my 0.448 and 0.454, and causal rejected 71–75% against my 70.1%.
- **release F12** — the headline ranges now cover my numbers (0.73 and 0.89, 53% and 61%, 8.4 and 12.2 times the floor)

## Accepted

All items.

## Checked

- The README's new benchmark ranges (table, note, "Read this first", "What changed (ChunkLoss)", "Where Post does not tell o from 0") against my round-5 tolerance.out, without new samples.

## Not checked

Nothing.
