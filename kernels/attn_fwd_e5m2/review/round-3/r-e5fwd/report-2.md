# Review of attn_fwd_e5m2, round 3, by r-e5fwd

## Verdict

accept
The three README fixes match my runs within sampling noise: pE's medians and maxima on the
benchmark, the fp8 floor's "78% to 100%" with "mostly" on the non-causal rows at N_CTX >= 1024, and
tolV "6.4 to 7.1" at N = 4096. Only README.md changed, and the gate and item digests are unchanged,
so every item stays accepted. A note in the filing gives my pE figures in case the prover wants the
ranges to cover them; it is not a finding.

## Findings

None.

## Accepted

All items.

## Checked

- The changed README passages: the table note, the N = 4096 tolV entries, "Where Post does not tell o from 0" (the floor's share and the benchmark's pE), and the Narrowings. I checked them against my round-2 work/tol.out.
- pE from my samples' largest X, and pE's median and maximum on fresh benchmark-like full heads (1024 rows, D = 64 and 128, three seeds), computed with Model.lean's expErr.

## Not checked

- Nothing else: the Lean is unchanged since round 2, when I checked it.
