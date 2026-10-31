# Review of attn_bwd_pre_f16, round 2, by r-bwdpre2

## Verdict

accept
The prover answered F1 with `CLARIFIED`, and the README sentence under Narrowings, "The grid", now gives each maximum its own launch. Nothing else changed: the gate is still `74a1e051af7b1c1b` and every item digest is as I accepted it in round 1. I release F1 and accept all items again.

## Findings

- **release F1** — the sentence now gives the test the largest `Z·H`, 192 (`N_CTX = 4096`: 32 · 192 programs), and the benchmark the largest grid, 128 · 128 programs (`N_CTX = 16384`, `Z·H = 128`). Both are right.

## Accepted

All items.

## Checked

- The prover's response and notes for round 1, and the new sentence in `README.md` (Narrowings, "The grid"), against the source. The test (lines 631-634: `Z = 4`, `H = 48`, `N_CTX = 4096`) gives `4096 / 128 = 32` programs on axis 0 and `192` on axis 1, which is 6144 programs. The benchmark (lines 706, 718: `BATCH = 4`, `N_HEADS = 32`, `N_CTX` up to `2^14`) gives `128 · 128 = 16384` programs. No test launch has more programs.
- `REVIEW.md`: the gate digest and the item digests are unchanged since round 1.
- r-bwdpre's round-1 filing and report. Its one finding, F2, is the same wording point, and the same sentence settles it. It raises nothing that I missed.

## Not checked

- Anything beyond the round-1 review: no item changed.
