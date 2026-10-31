# Review of attn_fwd_e5m2, round 3, by r-e5fwd

## Verdict

accept
Only README.md changed, and the gate and item digests are those of round 2, which I accepted. The
per-N_CTX figures for the test's data now cover my round-2 evaluation at every test_op shape. The
attribution is now right: the fp8 floor for the non-causal rows at N_CTX >= 1024, Num.DotE5m2 (L124)
at N_CTX = 128 and on causal rows. So F6 is settled and I release it. The new paragraph on what tolV
checks (F8) and the per-N_CTX figures (F7) are correct, and I found no new error in the changed
text. I accept every item.

## Findings

- **release F6** — the per-N_CTX table and the Narrowings now give figures that cover my round-2 tol.out at every test_op shape, and attribute the gap correctly

## Accepted

All items.

## Checked

- The prover's round-2 response.json and notes.md, and r-e5fwd2's F7 and F8 in REVIEW.md.
- The README's Specification: the tolV bullet with "What tolV checks" (the bound tolV >= 9/8·K + |attnRef|/8 + 2^-17 re-derived from magCp, magNext and tolV), the main table, and "Where Post does not tell o from 0" with its per-N_CTX table. I checked each range and derived share against my round-2 work/tol.out (N_CTX = 128, 1024, 4096; D = 64, 128; both values of causal), and the one-key, benchmark and N = 2^14 rows against the same output.
- The Narrowings ("The tolerances' form") against the Specification.

## Not checked

- Sample-dependent maxima the README quotes that my samples do not reproduce exactly (for example pE "up to 230" on the benchmark at D = 128, where my sample reaches about 270). The conclusions do not depend on them.
- Nothing in the Lean: it is unchanged since round 2, when I checked it.
