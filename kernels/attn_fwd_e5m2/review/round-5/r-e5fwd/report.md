# Review of attn_fwd_e5m2, round 5, by r-e5fwd

## Verdict

accept
Three of the four benchmark causal ranges I noted in round 4 now cover my seeds. The fourth, the
D = 128 causal rejection (28-38%), misses my 40.5% by 2.5 points. The table note no longer claims
exact coverage and says other samples may differ by a few percent at the ends of a range. That
covers the gap, so I raise no finding and every item stays accepted.

## Findings

None.

## Accepted

All items.

## Checked

- The README's table note, the per-N_CTX caption, the main table's benchmark rows and the benchmark paragraph, against my round-3 work/tol_uniform.out.

## Not checked

- No new samples, as the coordinator asked. The Lean is unchanged since my last filing.
