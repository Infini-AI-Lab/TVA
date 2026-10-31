# Review of attn_fwd_e5m2, round 2, by r-e5fwd2

## Verdict

accept
The revision answers my blocking finding F1 the way round 1 asked, and goes further. The README now
says plainly, in Specification and in Narrowings, where Post's accuracy part does not tell o from 0:
the test's data, the benchmark's data, and rows spread over many blocks. "Meaningful on the test's
shapes and data" is gone. The new second bound, tolV, is proved and is what the README says. It
adds real content: about 6.6 instead of 1.4e4 on the benchmark at D = 128, and the zero heads are
pinned to ±0 at every covered size. Every item is accepted at its current digest, and I release
F1, F2 and F3. Two new non-blocking DOC findings remain. First, the README's test-data figures do
not hold at test_op's N_CTX = 128. Second, tolV is met by any average of the row's own values,
whatever q and k are, and the README should say so.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — the test-data figures (97.7–100%, median |attnRef| 0.009–0.04, fp8 floor ≥ |attnRef| on 81–100%) do not hold at test_op's N_CTX = 128 (87–89% at D = 64 causal; 26–54%), and Narrowings' "that is the fp8 rounding of p itself" fails there
- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — say plainly that tolV ≥ 9/8·K + |attnRef|/8 at every nblk, so it accepts every output in the range of the row's own values whatever q, k, sm_scale: by itself it checks the key set and the range, not the weights
- **release F1** — the README states the limitation plainly in Specification and Narrowings, with its causes and L124; "meaningful" dropped; tolV is proved, re-computed, and as described
- **release F2** — referred to library issue L124, which states the request as raised
- **release F3** — the title, the README opening, the kernel paragraph and the root docstring all state the zero heads

## Accepted

All items.

## Checked

- That item:spec (the only item whose digest changed) is what the README says. I printed Post and the new definitions tolV, magA, magD, magNext, magCp and rho8 as Lean elaborates them (work/axioms.out). I also read the statements of Mag (lminB, lmin_real, p_big, lane_pq, lane_E, estep_real, final_v), Row.RowSt and Row.mag_tail, Step.astep's added conjunct, RowMax.mstep_* and Program.row_out. The second bound compares acc with r·l_i over the computed p, and uses l_i ≥ 1/2. main, consistent and satisfiable still depend only on propext, Classical.choice and Quot.sound.
- tolV re-run against the new formulas (work/tolerance.py, work/tolerance.out), with K = max_j |vEff_j − attnRef| over the row's keys. It reproduces the README:
  - q = 0, v = 1: 0.644, 1.13, 2.55, 71 and 1.13e7 at N = 1024 to 2^16;
  - v = 1 on the last 64 keys at 2^16: tolO 8.45e-4 against attnRef 9.77e-4;
  - benchmark-like data: 6.64 and 6.69 at N = 1024, 360 and 359 at 2^14;
  - test-like data at 2^14: 144 and 145;
  - one key: 0.308·|attnRef|; vEff = 0: 7.6e-6;
  - the README's K/R thresholds: 0.529 at 2 blocks, 0.217 at 16, none from 31 blocks on.
- Where 0 meets Post with both bounds, on every test_op shape (three seeds at N_CTX = 128 and 1024). Post holds for 0 on 87–100% of the elements on the test's data and 100% on the benchmark's data. On the benchmark's data pE has median 3.4 and 94–120, and maximum 4.5–4.8 and 190–228, at D = 64 and 128 (the README's "up to 4.7 and 190").
- That every element of L124 matches F2, and that F3's text is in place in kernel.json, the README, the root module docstring and kernels/README.md's listing.

## Not checked

- The proofs of the new lemmas line by line. The gate's build and replay check them, and I read only the statements.
- Whether Row.RowSt's K hypothesis over R.al matches Post's keys d n. main's proof checks this, and I did not trace rowCtx's al.
