# Review of attn_fwd_e5m2, round 4, by r-e5fwd

## Verdict

accept
The README's revised figures now cover my round-3 uniform samples for the three cases F9 named:
the causal test shares, the benchmark's D = 128 tolO, and the pE maxima. I release F9. A few
benchmark causal ranges still miss my seeds by a few percent. I note this in the filing; it is not
a finding. Only README.md changed and the gate is unchanged, so every item stays accepted.

## Findings

- **release F9** — the causal test shares, the benchmark's D = 128 tolO and the pE maxima now cover my round-3 uniform samples

## Accepted

All items.

## Checked

- The prover's round-3 response.json, and F10 in REVIEW.md.
- In the README: the table note, the main table, the per-N_CTX table with the row-0 sentence, and the benchmark paragraph of "Where Post does not tell o from 0". I checked them against my round-3 work/tol_uniform.out and work/tol.out, taking no new samples.

## Not checked

- No new samples, as the coordinator asked. The Lean is unchanged since my last filing.
