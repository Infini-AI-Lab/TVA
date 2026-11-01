# Review of attn_fwd_e5m2, round 3, by r-e5fwd

## Verdict

accept
The re-proof on DotE5m2Chunk is right as far as my checks reach. `assumption:dot` now cites the
library's fact for both fp8 tl.dots, and the other re-opened assumptions are unchanged in meaning.
The new spec quantifies a per-key score error bound (scoreErr, the chunk facts' own bound) next to
the lifted magnitude, centres tolO on attnRef, and keeps tolV.

Re-evaluated on data like the host's, the tolerances are what the README says, apart from some
small-sample figures (a DOC finding). The tightening is real: on the benchmark, tolO is about 0.55
(D = 64) and 1.2-1.5 (D = 128), against 13.9 and 1.2e4 before, and Post rejects 0 on 10-69% of the
elements. HeadOk's new cap on σ·scoreLift covers the host's data. The heads without vIn are still
pinned at ±0 by tolV (7.63e-6) at every covered size. I accept every item.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — Some sample-based figures in the new tables do not hold over larger uniform row samples, e.g. 0 rejected on causal rows at N = 4096: README 4-5%, three 128-row samples 1.7-1.9%

## Accepted

All items.

## Checked

- REVIEW.md (the changed items and the coordinator's note), L124 and L125 in ISSUES.md, and the new fact in lean/TVA/Rules/Num/Dot8Chunk.lean. In lean/TVA/Lemmas/Dot8Chunk.lean: errB, chain_err, chunkLift8, DotChunkAcc.err, DotF8Chunk.err and relLift8. num.md's rows for every cited fact, and Sm90.lean's remaining facts.
- claim.md §5: every assumption against the operation the model uses. `dot` is DotE5m2Chunk for lines 73 and 103 (fp8 x fp8 into float32, 2^30, any precision).
- Model.lean's new specification in full: scoreLift, scoreErr, HeadOk, partW, centC, blkC, blkA, blkLW, blkVL, scoreE, expErr, pE, FhT, Flo, Fp, cS8, cC8, cP8, cA8, eps8, etaA, gA, a8, b8, kap, hw, denX, denS, coefS, accNext, errNext, accA, errC, tolO, tolM, iotaV, magCp, magNext, magA, magD, tolV and Post.
- Cen.lean in full: the statements of cstep_real, lane_cen, block_cen, cl2_cen and CInv, and the definitions Cx, bC, bA, bLW, bVL, acR and ecR. A grep of the kernel's Lean found no instance, notation, attribute or renaming; the only options are maxHeartbeats and the linter.
- work/print_spec.lean (output in print_spec.out): the new spec's names with full names, and the axioms of main, consistent and satisfiable.
- work/tol.py on the new formulas (work/tol.out, and work/tol_uniform.out with three seeds of 128 evenly spread rows). It covers every test_op shape and the benchmark at N = 1024, both values of causal; v = 1 up to 2^16; and the heads without vIn at the cap. It reports tolO, tolV, tolM, pE, the share where Post accepts 0, the fp8 floor, causal row 0, and the largest σ·scoreLift against Xh.
- The README: "What changed (library round 24)", the Specification, the tables, "Where Post does not tell o from 0", the host cap paragraph, and the Narrowings.

## Not checked

- The proof bodies of Cen, Row, Step, Program and the other changed modules beyond the statements listed: Lean checks them.
- Whether DotE5m2Chunk is true of the device (library round 24 had no library review; kernel review trusts the library).
- The benchmark at N above 1024 with the new formulas (only v = 1 at 2^14 and 2^16, and the cap from round 1's samples at 2^14).
- Part (B) of Compiler.Faithful's scope.
