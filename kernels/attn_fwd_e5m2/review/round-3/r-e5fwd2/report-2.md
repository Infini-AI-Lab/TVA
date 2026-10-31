# Review of attn_fwd_e5m2, round 3, by r-e5fwd2

## Verdict

accept
The prover's README-only fixes answer both wording remarks from my first round-3 filing, and the new
pE figures match my runs. No item digest changed, there are no open findings, and every item is
accepted again.

## Findings

None.

## Accepted

All items.

## Checked

- Specification and Narrowings now say "most" (78% to 100%) for the fp8 floor on the non-causal rows at N_CTX ≥ 1024. That matches the per-N_CTX table and my round-2 runs (78.2% to 100%).
- The test-like tolV at N = 4096 now reads 6.4 to 7.1 (mine 6.36 and 6.4), and the table note says the figures agree within sampling noise and gives ranges where samples differ.
- The benchmark's pE medians are about 3.4 and 94 to 100 (mine 3.44 to 3.46 and 93.9 to 99.6 at N = 1024 and 4096), and its maxima 4.5 to 4.8 and 190 to 270 (mine 4.49 to 4.79 and 190 to 228).

## Not checked

- The one-key row's tolO ranges, as in my first round-3 filing.
