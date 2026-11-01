# Review of attn_fwd_e5m2, round 5, by r-e5fwd

## Verdict

accept
`assumption:dot` now cites DotE5m2ChunkLoss, the library's fact for both fp8 tl.dots. tolO's new
parts are explicit and computable. Each block's choice of arm is a function of the row's data and
of X and Z. tolO is the minimum with the round-24 recursion, which tolOW at W = 0 reproduces term
for term, so it is never above the old claim's.

Re-evaluated on data like the host's, the tightening is real. At N = 2^14, tolO falls from 20 and
75 to 1.65 and 3.7; for v = 1 at 2^16 it falls from 1.2e6 to 9.9. The test's figures are unchanged
or about 1.5% lower. One non-blocking DOC finding: the benchmark N = 1024 falls and ranges come
from narrow samples, and causal rows gain less than stated. I accept every item.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — Benchmark N = 1024 figures: over eight seeds tolO falls 11-22% (D = 64) and 20-38.5% (D = 128), not 16-25% and 34-49%; causal rows gain least, and several ranges are narrower than the seed spread

## Accepted

All items.

## Checked

- REVIEW.md's per-item changes, the new claim.md (`dot`), num.md's loss-facts section and listings, and errM in lean/TVA/Lemmas/Dot8Chunk.lean.
- The Lean diff of Model.lean in full: scoreErr, blkV, blkAbsV, lS8, lC8, lP8, lA8, gD, coefD, accStep, errStep, stepC, stC, errC, accA, tolOW, tolO, and the dot field. tolOW at W = 0 compared term for term with the round-24 recursion. A grep of the kernel's Lean found no instance, notation, attribute or renaming.
- work/print_spec.lean (output in print_spec.out): the changed names with full names, and the axioms of main, consistent and satisfiable.
- work/tol.py on the new formulas, with the old recursion beside it. tol.out covers every test_op shape and the benchmark at N = 1024 (two seeds), v = 1 up to 2^16, and vEff = 0 at the cap. tol_spread.out covers the benchmark at N = 1024 over six more seeds, and tol_big.out the benchmark at N = 2^14.
- The README: "Read this first", "What changed (ChunkLoss)", the Specification's tolO text, the table, "Where Post does not tell o from 0".

## Not checked

- The proof bodies of Cen, Row, Step, Inner, Program and RowAcc: Lean checks them.
- Whether DotE5m2ChunkLoss is true of the device (no library review followed round 26; kernel review trusts the library).
- The test's data at N = 2^14 under the new formulas.
