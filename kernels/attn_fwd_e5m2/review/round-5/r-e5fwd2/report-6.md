# Review of attn_fwd_e5m2, round 5, by r-e5fwd2

## Verdict

accept
- `assumption:dot` cites DotE5m2ChunkLoss, the library's listed fact for the model's fp8 tl.dot.
- The new tolO is explicit. Each block's arm is chosen from the row's exact data alone. The tolerance is never above the old one: tolOW(0) is the old recursion, and errM ≤ errB, so the new o clause implies the old.
- My own transcription reproduces most of the README's figures: v = 1 at 2^16 falls from 1.17e6 to 9.9, the test data are unchanged on non-causal rows, and the zero heads stay pinned.
- One non-blocking DOC finding: the benchmark's D = 128 "now" ranges miss my two heads by more than the stated few percent.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — the benchmark's D = 128 "now" ranges miss my heads by more than a few percent: tolO is 0.73 and 0.89 (README 0.70–0.77), 0 meets Post on 53% and 61% (README 42–56%), and tolO is 8.4 and 12.2 times the floor (README 4 to 8.5).

## Accepted

All items.

## Checked

- **The Lean side** (work/chunkloss.out):
  - Assumptions.dot is DotE5m2ChunkLoss.
  - The new definitions as Lean elaborates them: scoreErr with errM, stepC, stC, errStep, accStep, tolOW, tolO, blkV, blkAbsV, lS8, lC8, lP8.
  - main, consistent and satisfiable use only the three standard axioms.
- **The new tolO against the old.** At W = 0 the new recursion is the old one, term by term. The min therefore never exceeds it, and the Model diff leaves Post, HeadOk, tolV and tolM as they were.
- **The README's figures**, against my transcription of the new and old tolO (work/tolerance.py, work/tolerance.out):
  - test-like data: all rows at N = 128 and 96 rows otherwise, two seeds;
  - benchmark-like data at N = 1024 (two seeds) and at 2^14;
  - one key;
  - q = 0, v = 1 up to 2^16;
  - vEff = 0 at X = 0 and at the cap.

## Not checked

Nothing.
