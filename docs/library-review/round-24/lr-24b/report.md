# Library review, round 24, by lr-24b

## Verdicts

- **L124** REOPEN — The chunk facts `DotE4m3Chunk` … are true as far as I can test them, and
  they tighten a score's bound. But they do not tighten L124's own case, the accumulator fed back
  once per block, and the reason the batch gives for that is false. **What holds.**
  evidence/num-dot-chunk re-ran identical. work/fp8_adv.out adds 749568 elements on setups the
  evidence lacks: `m64n16`, `m64n32` and `m64n256` tiles; `M = 64` and `M = 256` with 8 warps;
  `mma.sync` with 1, 2, 4 and 8 warps; `a` transposed, both operands transposed, and `a` converted in
  registers; tutorial-03 K loops (num_stages 3 and 4, `acc = tl.dot(a, b, acc)` and
  `acc += tl.dot(a, b)`). Inputs include mixed-sign window cases, all-negative subnormal products,
  operands at the subnormal/normal edge, cancellation and subnormal accumulators. No element is
  outside the interval the facts allow. The largest ratio is `0.44`, and my own `wgmma` chunk model,
  written from rules/num.md, equals the device on all 466944 `wgmma` elements. A zero operand does
  not set the window (round 25's work/zero_probe.out). `input_precision` changes no fp8 bit (round
  25's work/misc_paths.out). work/Check24.lean applies a fact to a one-chunk dot.
  **What is wrong.** Each step still charges `2^-7·Mj` with `Mj ≥ |s_j|`, so a dominating
  accumulator grows by `1 + 2^-7 + 2^-12` per chunk. Before, it grew by `1 + 2^-7` per chunk.
  attn_fwd_e5m2's own table shows the effect: its tolerance at `N_CTX = 2^16` went from `9.7·10^5`
  to `1.2·10^6`. ISSUES' round-24 entry and docs/rules/num.md ("How a kernel uses them") call this
  "the device's behaviour under the window, not slack in the fact". The facts' own model, which
  the evidence shows exact, says otherwise:
  - a term truncated toward zero to a multiple of `u` loses at most `min(|term|, u)`;
  - so a chunk's error is below `(1+2^-13)·(Σ_chunk|a·b| + u) + 2^-13·|s_j + c_j|`, with
    `u ≤ 2^-13·Mj + 2^-139`;
  - that is, it is within `B_alt = (1+2^-13)·Σ_chunk|a·b| + 2^-12·Mj + 2^-12·|s_j + c_j| + 2^-126`,
    which charges a dominating accumulator about `2^-11` per chunk.

  work/fp8_alt.out finds every chunk within `B_alt`:
  - single-chunk dots on both paths, checked on the device with no model: largest `0.82`, with
    products of one sign just below the unit under a dominating accumulator, where the facts' bound
    is 4 times looser (`0.23`);
  - with products far below the accumulator, the device is at `0.06` to `0.09` of the facts' step
    bound;
  - tutorial-06 chains: the model equals the device on 32768 dots, largest `0.34`.

  **Fix:** bound each step by the smaller of the present bound and a margin version of `B_alt` (for
  example `2·Σ_chunk|a·b| + 2^-11·Mj + σ·|s_j + c_j| + α`), derived from the same model, and extend
  the evidence's per-chunk check to it. Give `Dot8Chunk.errB` and `chain_err` the minimum, and
  correct the sentence in num.md, ISSUES and the num-dot-chunk README. A fed-back accumulator then
  grows by about `1 + 3·2^-12` per chunk. Round 25 changed some of what the round-24 notes describe:
  - `liftMag` and `minNormal8` moved to `Rules/Num.lean`. `minNormal8` gained values for `float16`
    and `bfloat16`, but its fp8 values are unchanged, so the facts mean the same;
  - `Dot8Chunk.LiftAcc` became `Rules.Num.DotLiftAcc`, the same statement;
  - the `mma.sync` derivation is now "25 bits below the largest field, truncated", which still
    bounds it.
- **L125** ACCEPT — The four whole-dot facts and `DotF8` are gone from the Lean, from "Facts the
  library states", from the reviewer's table and from `tools/tvr.py`'s example. Their `sm90ref`
  lemmas are removed too, and grep finds the names only in history notes. The replacement is
  `Dot8Chunk.DotF8Chunk.liftAcc`, a theorem of the chunk facts with standard axioms
  (work/Check24.lean). Its lifted room covers the facts' room (`relDot8 ≤ relLift8`,
  `absDot8 ≤ absLift8`, `|a·b| ≤ liftMag a·liftMag b`), and it keeps subnormal operands covered. On
  the device it holds at no more than `0.44` of `relLift8 K·L + absLift8 K` (work/fp8_adv.out). The
  constant list matches the change. Since round 25 the lemma states `Rules.Num.DotLiftAcc`, the same
  proposition, and `matmul_e5m2` and `attn_fwd_e5m2` cite `DotE5m2Chunk`.

## New issues

None.

## Re-ran

- evidence/num-dot-chunk `run.py`, from work/rerun: `out.txt` is identical to the evidence's, apart
  from the run time.
- work/fp8_adv.py (`fp8_adv.out`): my model and interval checks of the chunk facts and of
  `DotF8Chunk.liftAcc` on setups and inputs the evidence lacks (above), all four pairs, both paths.
- work/fp8_alt.py (`fp8_alt.out`): the device against `B_alt`, and against the facts' step bound.
- work/Check24.lean, run with `lake env lean`: the axioms of the new theorems, and a one-chunk use
  of `DotE5m2Chunk`.

## Not checked

- I did not write a bit-level model of the fp8 `mma.sync` path (two `float16` instructions with
  interleaved `k`). There I checked only the facts' interval, which held below `0.0002`.
- `max_num_imprecise_acc` other than `2^30`, which the library does not state.
- The two kernels' re-proofs, which belong to kernel review.
