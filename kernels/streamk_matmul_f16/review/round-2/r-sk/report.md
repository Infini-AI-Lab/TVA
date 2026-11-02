# Review of streamk_matmul_f16, round 2, by r-sk

## Verdict

accept
The prover settled F1 in the write-up; the gate and every item digest are as I accepted them in round 1, and I have no blocking finding.

## Findings

- **release F1** — README, Specification, now says A is the test's normaliser up to the subnormal operands' excess (0 to 102 per matrix, at most 1.9e-7 relative), as round-1/r-sk/work/subnormals.out recorded.

## Accepted

All items.

## Checked

- The prover's responses in REVIEW.md, and the changed README paragraph after the tolerance table, against round-1/r-sk/work/subnormals.out (no new sampling).
- The new README sentence on what `Post` does not reject: by arithmetic, `tol/A` is `4.76e-4` at `T = 30` and `4.92e-4` at `T = 31`, the first `T` above `4.88e-4` (one `float16` rounding), and `K = 1921` is the first `K` with `T = 31`.
- The gate digest `16c9926eb17e72c9` and each item digest, unchanged since round 1.

## Not checked

Nothing.
