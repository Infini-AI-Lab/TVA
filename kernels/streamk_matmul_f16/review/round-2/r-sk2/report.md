# Review of streamk_matmul_f16, round 2, by r-sk2

## Verdict

accept
The prover answered my two non-blocking DOC findings by changing the write-up and two docstrings, and both are settled. The gate (`16c9926eb17e72c9`) and every item digest are unchanged since I accepted in round 1, so I accept all items again.

## Findings

- **release F2** — the README no longer says the test's data has no subnormal operand. It says `A` exceeds the normaliser by the subnormal operands' excess, and that the figures hold to the digits given.
- **release F3** — `Inst.ok.K_le`'s docstring now says 0.21% of `A` (`dotMag`), which is what `tol_le` proves.

## Accepted

All items.

## Checked

- `REVIEW.md`'s responses to F1–F3, and the changed text: the README's Specification paragraph after the tolerance table, `Inst.ok.K_le`'s docstring (`Defs.lean`), and `testInst`'s docstring, which now cites line 100.
- The README's new sentence on what `Post` does not reject. I recomputed the threshold with my round-1 `work/tol_table.py`, in exact arithmetic, with no new sampling: `tol K A` exceeds `2^-11/(1 + 2^-11)·A` from `K = 1921` (`T = 31`, `4.916e-4·A`), and not at `K = 1920` (`T = 30`, `4.758e-4·A`). This matches the README, as do its values at `K = 2048` and `K = 4096`.
- The gate digest and the item digests are the same as at my round-1 acceptance (`REVIEW.md`, Status).

## Not checked

- The `1.9e-7` excess the README quotes from r-sk's sample. My round-1 sample counted the subnormal operands but did not compute that excess.
