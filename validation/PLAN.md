# Validating the semantic model

Goal (Yi Zhou, with zhuominc, 2026-11-09): check that the library's formal semantics
(`lean/TVA/Lang`, `lean/TVA/Sem`) says what Triton 3.8.0 does on the device, by comparing the
Lean definitions themselves with the GPU.

## Where validation stands (2026-11-09)

| Layer | What it is | Validated by |
|---|---|---|
| scalar floats (`Math/Defs.lean`, encodings) | IEEE operations, rounding, conversions, bytes | the Lean definitions, compiled and run against the H200 ([evidence/float-layer](../evidence/float-layer/README.md)) |
| device facts (`Rules.Num`, `Platform`) | accuracy of `exp`, `/`, `tl.dot`, … | GPU runs, exhaustive or adversarial |
| tensor operations, `Step` (`Lang/Ops.lean`, `Sem/Step.lean`) | reductions, scans, sort, reshapes, masked accesses, promotion, integer operations, Philox, the conditions for undefined behaviour | the GPU against Python references written by hand (`argmax`, `permute`, `int-reductions`, `promotion`, …): these test a reading of SEMANTICS.md, not the Lean terms |
| front-end facts (SEMANTICS.md) | how the source becomes a `Prog` | GPU runs, and reading the compiler sources |
| transcriptions (`kernels/`) | each kernel's model | review only |

The model is relational (`Prog.pick`, the `Platform` relations, `AddRel`), so it cannot simply be
run. Validation is refinement checking: is what the device did one of the model's outcomes?

## Decisions

- 2026-11-09: scope stays as is. zhuominc is confident the current semantics covers most Triton
  programs in the wild. The gaps below are for discussion, not planned work.
- 2026-11-09: the work lives in `validation/`. The relational semantics stays the trusted base; an
  executable semantics is added beside it, with a proof that links the two (`exec_sound`), and does
  not replace it.
- 2026-11-09: the review protocol stays as it is. A mismatch is brought to zhuominc, not filed as
  a library issue.

## Phases

0. **Trust inventory.** For every definition in the trusted base, record how it is validated:
   Lean definition executed, Python reference, source reading, or not at all.
1. **An executor with an oracle.** `exec : Prog α → Mem → Oracle → Option (Out α × Mem)`: the
   oracle answers each `pick` and `env`, and `exec` decides the predicate on the concrete instance.
   Prove `exec_sound`: if `exec` succeeds, `Step` reaches its result. A lemma, not a new construct.
2. **Operation-level tests of `Ops.lean`.** Reuse the float-layer harness. Swap the Python
   references of the existing evidence for the compiled Lean definitions. Add reductions and scans
   (tree shape, NaN, ±0, ties), `sort`, `histogram`, `join`/`split`/`interleave`/`reshape`,
   broadcasting, masked `load` with `other`, integer `//`/`%`/shifts, and Philox (bit-exact).
   Promotion (`Promote.lean`) exhaustively against `tl.static_assert` and TTIR (no GPU needed).
3. **Undefined-behaviour predictions.** Where the model says `Safe`, `compute-sanitizer`
   (memcheck, racecheck, initcheck) reports nothing. Cases at the boundaries: masks at the end of a
   buffer, offsets off by one.
4. **Kernel-level tests.** Run each kernel's model through `exec` and compare its final memory with
   the device's: transcription and semantics at once. Bit-exact through a platform calibrated on the
   device (the oracle answers with the device's result for each operation) and through inputs that
   remove the nondeterminism (small-integer `float16` operands make `tl.dot` exact in any order;
   fusion off). Programs of a tile-parallel kernel run one after another, as `independent`
   justifies.
5. **Atomics across programs.** Litmus tests (message passing, store buffering, coherence) for
   `Sem/Launch.lean`. Low priority given the scope decision.

In every phase:
- **Tightness:** a relation that holds of everything passes every test. For operations meant to be
  exact, also check that the relation admits only the device's value.
- **Mutation checks:** seed bugs into copies of the definitions and confirm the tests catch them.

## Possible gaps, for discussion with zhuominc

1. A program that updates its input in place (loads bytes, then stores to them) is excluded
   (`IntraRace`, library issue L15). Common in the wild: fused optimizers, vLLM's
   `fused_add_rms_norm`, in-place rotary embeddings. A narrower rule (each lane stores only to bytes
   it read, a value that depends on that read) might be sound.
2. Missing operations: `tl.extra.cuda.libdevice` (`tanh`, `pow`, …), `tl.gather`, `tl.cat`,
   `tl.umulhi`.
3. A survey of `tl.*` usage and in-place patterns in vLLM, FlashAttention, Liger,
   flash-linear-attention and Inductor output would ground (1) and (2) in data (collaborator
   question 6). No GPU needed.

## GPU

The device facts are for sm_90: an H200, or an H100 if zhuominc accepts it. A recent driver
(the evidence used 610.57) and `compute-sanitizer` are needed. First choice: the H200 NVL the
existing evidence ran on. Otherwise an internal AWS account (P5 is H100, P5e/P5en is H200) or a
cloud billed by the hour. Phases 0–1, the promotion part of 2, and the survey need no GPU.

## Open questions

- Do results from an H100 count for the H200 target?
- Access to the existing H200 machine.
