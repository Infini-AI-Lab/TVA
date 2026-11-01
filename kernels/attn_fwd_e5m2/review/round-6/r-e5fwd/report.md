# Review of attn_fwd_e5m2, round 6, by r-e5fwd

## Verdict

accept
I accept every item: only README.md changed and the digests are unchanged. The new falls and
D = 128's non-causal tolO now cover my eight seeds, so I release F13. The README still says its
benchmark ranges cover the reviewers' samples, but some of the numbers F13 listed are outside them.
I raise that as a new non-blocking DOC; tvr cannot record a re-raise while every item is accepted.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — The benchmark paragraph says its N = 1024 ranges cover the reviewers' samples, but some of mine are outside: D = 128 causal tolO 0.50, D = 64 causal tolO 0.335, 0 meets Post 22.6% and 38.2%
- **release F13** — the falls (11-26%, 20-49%) and D = 128's non-causal tolO (0.65-0.89) now cover my eight seeds; the rest is the new DOC

## Accepted

All items.

## Checked

- The prover's round-5 response (F12, F13) and the README's benchmark entries (the table, its note, and the benchmark paragraph), against my round-5 work/tol.out and tol_spread.out, without new samples.

## Not checked

- No new samples, as the coordinator asked.
