# Review of attn_fwd_e5m2, round 3, by r-e5fwd2

## Verdict

accept
The re-proof on the library's chunk facts is sound in form and much tighter.

- `assumption:dot` now cites `DotE5m2Chunk`, the library's fact for the model's fp8 `tl.dot`s. The other num facts are unchanged.
- The new `Post` is what the README says.
- On the test's data, tolO is now 1.75 to 5.5 times the fp8 floor.
- On the benchmark's data, Post rejects o = 0 on many elements: tolO is about 0.6 and 1.4 instead of 13.9 and 1.2·10^4.
- The heads without vIn are still pinned at ±0.
- The new cap still covers the host's data.

I accept every item. One non-blocking DOC finding remains: the README's causal rejection shares at N_CTX ≥ 1024 depend on row 0 being 1/24 of its sample.

## Findings

- **DOC** `doc:kernels/attn_fwd_e5m2/README.md` — the causal test-data figures at N_CTX ≥ 1024 (0 rejected on 7–8% and 4–5%) come from 24-row samples in which row 0 (one key, always rejected) is 1/24 of the sample; over 96 rows they are 5.4–5.8% and 1.7–2.2%. A few other ranges miss my samples slightly.

## Accepted

All items.

## Checked

- **The assumptions** (work/axioms.out).
  - `Assumptions.dot` is `DotE5m2Chunk P`, that is `DotF8Chunk f8e5 f8e5`: every precision, `maxImpreciseAcc = 2^30`, a float32 accumulator. That is the model's cfg8 `tl.dot` at lines 73 and 103.
  - The other num fields print unchanged. Library round 24's notes say their meaning did not change.
  - main, consistent and satisfiable use only the three standard axioms.
- **The spec's new definitions** (HeadOk, scoreLift, scoreErr, scoreE, tolO, accNext, errNext, blkLW, iotaV, magCp, magD) and the library's errB and relLift8, as Lean prints them. I also read the statements of Cen (cstep_real, block_cen, CInv, acR, ecR).
- **The README's figures**, against my own transcription of the new tolO, tolV and tolM (work/tolerance.py, work/tolerance.out). I checked:
  - test-like data, all rows at N = 128 and 96 rows otherwise, two seeds;
  - benchmark-like data at N = 1024, causal or not;
  - one-key rows;
  - q = 0, v = 1 up to 2^16;
  - vEff = 0 at X = 0 and at the cap;
  - the README's own 24-row causal samples.
  
  The medians, ratios and limits agree. The causal shares do not; that is the finding.
- **The heads without vIn**: at K = R = 0 tolV is 7.63e-6, below 2^-16, at every size up to D = 256, N = 2^16, so they are pinned at ±0.
- **HeadOk's new cap on σ·scoreLift**: a Chernoff bound on the exact distribution of the lifted fp8 products (work/score_cap.py) gives 10^-19.78 per benchmark pair at D = 128. That is the same as for the old cap, against 10^10.5 pairs.

## Not checked

- The proofs of Cen and of the changed lemmas line by line. The gate checks them; I read the statements.
- The library's chunk facts and their evidence. Kernel review trusts the library.
