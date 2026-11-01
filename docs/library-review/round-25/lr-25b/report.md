# Library review, round 25, by lr-25b

## Verdicts

- **L126** ACCEPT — `DotF16` and `DotBF16`, restated as `DotLiftAcc`, are true on the H200 for
  every operand value I could reach.
  - **The evidence re-ran identical:** evidence/num-dot-sub16-round25 (the whole run), and the
    coordinator's evidence/num-dot-sub16 `probe.py` and `run.py`.
  - **My own runs** (work/dot16_adv.out) cover 585083 elements on setups the evidence lacks:
    - tiles and warps: `m64n16`, `m64n32` and `m64n256` `wgmma` tiles; `M = 64` and `M = 256`
      with 8 warps; `mma.sync` with 1, 2, 4 and 8 warps at `M` from 16 to 64;
    - layouts: `a` transposed, both operands transposed, and `a` converted in registers from
      `float32`;
    - tutorial-03 K loops: num_stages 3 and 4, with `acc = tl.dot(a, b, acc)` and with
      `acc += tl.dot(a, b)`;
    - eleven kinds of input: mixed-sign window cases with the subnormal operand in `a` or in `b`;
      all-negative subnormal products; operands at the subnormal/normal edge; both operands
      subnormal; accumulator-dominated chunks; cancellation; subnormal accumulators; near-maximal
      operands; random bit patterns; and `bfloat16` products below `float32`'s subnormal range.

    No element is outside the bound. The largest ratio is `0.130` (`bfloat16`, edge operands,
    `K = 16`), below the derivation's limit of `0.146`. The old bound fails on 103 of these
    elements, by up to `21.9` times. My own bit-level model, written from num.md, equals the device
    on every element on both paths.
  - **Zero operands** do not set the window (work/zero_probe.out). A product `0 × 2^15`,
    `0 × 2^127`, `0 × 448` or `0 × 57344`, or one with `−0`, leaves the other products exact on both
    paths. The control, with the smallest subnormal in place of the zero, loses their bits. So
    `liftMag 0 = 0` is right, for the fp8 facts too.
  - **Code-generation options** change no bit (work/misc_paths.out): `warp_specialize` (it
    compiles with 8 warps; with 4 the launch is rejected on the host) and `num_ctas = 2`, for
    `float16`, `bfloat16` and the fp8 pairs.
  - **The derivation** in num.md checks: `5.25·2^-23` per chunk; no growth, since truncation is
    toward zero; and `⌈K/16⌉·5.25 ≤ 2(K+2)`.
  - **Usability** (work/Check25.lean): on zero and normal values `DotF16` gives the old `DotAcc`
    bound; the two thresholds come out as num.md says; and `sum_liftMag_mul_le` applies. The new
    theorems use only the standard axioms.
  - **Reach:** the list names `DotF16`, `DotBF16`, `minNormal8`, `liftMag` and `LiftAcc`. It does
    not say that the generic `DotChunkAcc` and the `Dot8Chunk.liftMag_*` lemmas also change meaning
    at `.f16` and `.bf16`, through `liftMag`. No fact or kernel uses them there, and the digests
    re-open the items anyway.

## New issues

- **NEW** wording — docs/LIBRARY.md, section "Kernels": the table still says "(none yet)" and
  that the kernels "are rewritten under the protocol, beginning with `vec_add_f32`". Nine kernel
  folders exist, listed in kernels/README.md. Fix: point to kernels/README.md, or list the kernels.
- **NEW** wording — evidence/num-dot-sub16/README.md: conclusion 1 ("`DotF16` and `DotBF16` do not
  hold") and the table's "error / bound" column mean the facts as they were stated before round 25,
  relative to `Σ|a·b|`. Since round 25 those names denote the restated facts, which hold on the
  same inputs (`0.104` and `0.1025` of the bound in evidence/num-dot-sub16-round25, Part F). Fix:
  add "as stated before round 25", and point to the round-25 evidence.
- **NEW** wording — docs/rules/num.md, the first sentence of "`tl.dot` (the chunk model)", is
  garbled: "evidence `evidence/num-dot`, whose parts "Part 1" … "Part 4" below are, and for the
  16-bit facts `evidence/num-dot-sub16-round25`, cited by name". Fix: "evidence `evidence/num-dot`
  (its "Part 1" … "Part 4", cited by name below) and, for the 16-bit facts,
  `evidence/num-dot-sub16-round25`".

## Re-ran

- evidence/num-dot-sub16-round25 `run.py`, from work/rerun: `out.txt` is identical, apart from the
  run time.
- evidence/num-dot-sub16 `probe.py` and `run.py`, from work/rerun-sub16: both outputs are identical.
- work/dot16_adv.py (`dot16_adv.out`), work/zero_probe.py (`zero_probe.out`) and
  work/misc_paths.py (`misc_paths.out`), as above. The last two cover the fp8 facts as well.
- work/Check25.lean, run with `lake env lean`.
- Survey (`lemmas`):
  - I read lean/TVA/Lemmas/Dot8Chunk.lean, DotLift.lean, DotLoop.lean and Sm90.lean in full. Each
    statement says what its name and docstring say, and the K-loop, chunk-chain and lift lemmas are
    what the restated facts need (Check24.lean and Check25.lean apply them).
  - For the other 25 modules I read every statement and docstring, and found no mismatch.
  - I read docs/LIBRARY.md against the modules; it gave the first new issue.

## Not checked

- In the lemma modules other than the four above, I did not read the proofs. Lean checks them, and
  I did not run `lake build`.
- `tf32`, `tf32x3`, `bf16x3` and `bf16x6` dots of `float32` operands, which the library does not
  state.
