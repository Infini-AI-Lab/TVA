# Review of attn_fwd_e5m2, round 2, by r-e5fwd

## Verdict

accept
Only `item:spec` changed. Post now holds a second bound on o, tolV, which compares acc with
attnRef·l_i over the kernel's own p. pE does not enter it. I re-evaluated both tolerances on the new
formulas. tolV brings the benchmark's D = 128 tolerance from about 10^4 down to about 7, close to the
magnitude estimate I gave in round 1. On a one-key causal row tolV is 0.31·|attnRef|, so Post
rejects 0 there. Post still accepts o = 0 on most of the test's and the benchmark's elements. The
write-up now says so plainly and refers the library's part to L124. Rejecting 0 there would need a
sharper library fact, so I release F4. F5 is settled by the new title, opening and table. One new
non-blocking DOC finding: the write-up's percentages for the test's data overstate how much of this
the fp8 rounding explains. I accept every item.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — README's figures on the test's data overstate the fp8 floor: at N = 128 and on causal rows it covers 27-73% of elements, not 81-100%, and Post accepts 0 on 89.5-100%, not 97.7-100%
- **release F4** — tolV (benchmark about 7 instead of 10^4, one-key rows 0.31·|attnRef|), with the rest stated as a limitation citing L124: the route F4 offered
- **release F5** — the title, the README's opening and the root docstring state the zero heads; "meaningful" is gone; the benchmark's D = 128 is in the table

## Accepted

All items.

## Checked

- The prover's response.json and notes.md, REVIEW.md, the new claim.md's specification, and r-e5fwd2's round-1 report.
- The gate snapshots: only `spec`'s digest changed. Its cone gained only magCp, magNext, magA, magD and tolV.
- The new definitions in Model.lean (magCp, magNext, magA, magD, tolV, Post, and Post's docstring) against the README's Specification. work/print_spec.lean (output in print_spec.out) prints them and Post with full names. main, consistent and satisfiable still use only propext, Classical.choice and Quot.sound.
- Mag.lean in full (lminB, lmin_real, p_big, lane_pq, lane_E, estep_real, final_v, ref_le). RowSt and mag_tail in Row.lean, OutO and row_out in Program.lean. A grep of the changed files found no instance, notation, attribute or renaming; the only options are maxHeartbeats and the linter.
- work/tol.py (output in tol.out) on the new formulas. It covers every test_op shape (N = 128, 1024, 4096; D = 64, 128; both values of causal), the benchmark at N = 1024 and at 2^14, the test at 2^14, and v = 1 up to 2^16. It computes tolO, tolV, the share where Post accepts 0, the fp8-rounding floor, the one-key rows, and the largest K/|attnRef| for which tolV < |attnRef|. Every tolV figure in the README's table and its kappa claims agree with mine.
- The README's new opening, "Where Post does not tell o from 0", the table and the Narrowings. kernel.json's title. The root module's docstring.

## Not checked

- The proof bodies of the changed lemmas (mstep_plain, mstep_masked, astep, mag_tail, row_out) beyond their statements: Lean checks them.
- Whether a sharper DotE5m2 is true of the device (L124, a library matter).
- Part (B) of Compiler.Faithful's scope (the environment), which is assumed.
