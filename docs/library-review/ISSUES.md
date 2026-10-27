# Library issues

The library's open questions and defects, from kernel review (`LIBRARY_REPORT`s and `REFERRED`
findings, collected in [REPORTS.md](REPORTS.md)), from library review, and from the author.
The process is in [../LIBRARY_REVIEW.md](../LIBRARY_REVIEW.md); kernel findings are cited as
`<kernel>#F<n>` (each kernel's `REVIEW.md`).

The kernels of the first round (vec-add, grouped GEMM, flash attention) were cleared on
2026-10-27 at the user's request, to restart from scratch. The issues they raised stand on their
own; the cleared kernels' records are in the git history (commit `089d4c5`).

| Issue | State | Round | Source | Summary |
|---|---|---|---|---|
| [L1](#l1) | OPEN | | cleared round 1: grouped GEMM F1, F2 | A hint inside a loop body on a loop-carried name is not a no-op |
| [L2](#l2) | OPEN | | cleared round 1: grouped GEMM F2 | What a hint on a rank ≥ 2 tensor promises, in terms a kernel can check |
| [L3](#l3) | ACCEPTED | 2 | cleared round 1: vec-add F1, grouped GEMM F3, flash attention F4 | A launch rejected before it runs is outside the claim (user, 2026-10-27) |
| [L4](#l4) | OPEN | | cleared round 1: flash attention F1, F2, F7, grouped GEMM F6 | Device facts with constants, owned by the library, on TVA's own evidence |
| [L5](#l5) | OPEN | | cleared round 1: flash attention F5 | A descriptor built inside the kernel writes launch scratch memory |
| [L6](#l6) | ACCEPTED | 2 | cleared round 1: flash attention F8 | A drop reason for `tl.static_assert` |
| [L7](#l7) | OPEN | 3 | user, 2026-10-27 | TVA cites no other project's results, lemmas or measurements, and the tool enforces it |
| [L8](#l8) | ACCEPTED | 2 | cleared vec-add prover, round 1 | `Block.Fits` is stricter than `int32` indexing needs |
| [L9](#l9) | ACCEPTED | 2 | library review round 1 (lr-1) | `Launch.Valid` admits grids the launcher silently skips (program count ≥ 2^31) |
| [L10](#l10) | ACCEPTED | 2 | library review round 1 (lr-1) | `no-op:print` drops arguments that may have effects |
| [L11](#l11) | ACCEPTED | 2 | library review round 1 (lr-1) | `rules/num.md` asks kernel reviewers to check device truth |
| [L12](#l12) | OPEN | 3 | library author, round 2 | No digest covers the transcription rules, so a change to them re-opens no kernel `map` |
| [L13](#l13) | OPEN | 3 | library review round 2 (lr-2a) | soundness: a kernel's Lean can add `main` without the kernel checking it |
| [L14](#l14) | OPEN | 3 | library review round 2 (lr-2a) | soundness: "pure" is unsound for aggregate operands (overloaded operators run code) |
| [L15](#l15) | OPEN | | library review round 2 (lr-2b) | soundness: one program is not sequential for global memory (needs the user's decision) |
| [L16](#l16) | OPEN | 3 | library review round 2 (lr-2b) | gap: `tl.max`/`tl.min` widen small integers to `int32` |
| [L17](#l17) | OPEN | 3 | library review round 2 (lr-2b) | gap: no `tl.permute` (rank ≥ 3), PDL (`gdc_*`, `launch_pdl`) or `tl.argmax` |
| [L18](#l18) | OPEN | 3 | library review round 2 (lr-2b) | gap: `TRITON_F32_DEFAULT` changes `tl.dot`'s default precision |

## L1

**A hint inside a loop body on a loop-carried name is not a no-op.**

- **Problem.** The library said such a hint is a no-op, and tutorial 08 relied on that. But to
  find the loop-carried names, Triton's front end first visits a `for` or `while` body once with
  every name still bound to its pre-loop value. So the hint marks the pre-loop value, and the
  compiled loads then assume 16-byte alignment. With `lda = 36` the kernel faults.
- **Evidence.** [evidence/hints-dry-run](../../evidence/hints-dry-run/README.md).
- **To do.** State the meaning (a check on the pre-loop value at the loop's entry, plus no effect
  on the block argument) and the transcription rule. Cover the induction variable, and loops
  inside loops.
- **Affects.** The hint operations' documentation, [TRANSCRIPTION.md](../TRANSCRIPTION.md) §8,
  and the protocol's drop reasons (`no-op:block-arg-hint` is withdrawn).

## L2

**What a hint on a rank ≥ 2 tensor promises.**

- **Problem.** `tl.multiple_of(a_ptrs, [16, 16])` on a 2-D pointer tensor, read per element,
  would claim every element is 16-byte aligned, which is false for tutorial 08. Triton 3.8.0
  uses the value along the fastest-varying dimension only.
- **Problem.** The `contig` argument of `multiple_of` is defined by "Triton's contiguity", which
  a kernel reviewer cannot check.
- **To do.** Define both in terms of the tensor's values.

## L3

**A launch rejected before it runs is outside the claim** (user decision, 2026-10-27).

- **Problem.** Too much shared memory, too many registers, or a `tl.dot` below its minimum
  size: Triton reports these on the host before any program runs. The rule page asked kernel
  reviewers to check that every covered instance compiles.
- **To do.** State in `TVA.Claim` and `Rules.Compiler.Faithful` (docstrings and the rule page)
  that such a launch has no outcome. Remove the compile-log check from the rule page.
- **Fixed in round 1** ([notes](round-1/author/notes.md)); **reopened** by library review ([lr-1](round-1/lr-1/report.md)): a rejection must be *reported to the host with an error*, and stale text remains. Round 2.

## L4

**Device facts with constants, owned by the library.**

- **Problem.** Kernels chose their own accuracy constants, and justified them with another
  project's measurements. Under the protocol, device facts are the library's, on TVA's own
  evidence (Triton 3.8.0, the H200), and kernels cite them by name.
- **To do,** for sm_90 with Triton 3.8.0:
  - `tl.math.exp2` on `float32`: the lowering (libdevice, with flush-to-zero), the constants,
    and `exp2(-∞)`;
  - `tl.math.log2` on `float32`: the lowering (libdevice), the constants;
  - `/` on `float32` (`DivFull`): TVA's own measurement;
  - `tl.dot` with `float16` and `bfloat16` operands into `float32`: the constants;
  - each as a named fact with a page and evidence.

## L5

**A descriptor built inside the kernel writes launch scratch memory.**

- **Problem.** With `tl.make_tensor_descriptor`, each program writes its descriptors into
  memory the launcher allocates. The semantics has no such memory, so "every other byte is
  unchanged" would be false.
- **To do.** Model launch scratch, or exclude in-kernel descriptors (with host-built
  descriptors still covered).

## L6

**A drop reason for `tl.static_assert`.**

- **Problem.** None of the drop reasons fit a compile-time assertion.
- **Fixed in round 1.** The drop reason `compile-check`, whose note names the condition of
  `Inst.ok` that makes the assertion hold: in the protocol (§7), the tool's drop list and
  [TRANSCRIPTION.md](../TRANSCRIPTION.md) §10.
- **Reopened** by library review ([lr-1](round-1/lr-1/report.md)): the condition is evaluated as an
  ordinary expression, so an operation inside it runs. The drop applies only to conditions free of
  operations. Round 2.

## L7

**TVA cites no other project's results, lemmas or measurements** (user, 2026-10-27: "let's make
sure the independence of tva, provers and reviewers cannot cite other project's results or
lemmas").

- **Rule.** TVA's results come from TVA and Mathlib, and its evidence is its own. No library
  file, kernel file or review record refers to another project.
- **Enforced** by `tvr audit` and the gate's `independence` check:
  - Lean imports only `TVA`, the kernel's own modules and `Mathlib`;
  - the lake configuration requires only Mathlib;
  - no text names another project or a path into one.

  `tvr file` and `tvr respond` refuse a filing or a response that does.
- **Fixed in round 1** ([notes](round-1/author/notes.md)). One consequence is open: `DivFull`
  had rested on another project's measurement, and now waits for TVA's own (L4).
- **Fixed in round 2**; **reopened** by library review round 2 ([lr-2a](round-2/lr-2a/report.md), [lr-2b](round-2/lr-2b/report.md)): home paths written as `$HOME/…`, `${HOME}/…`, `file:///home/…` or `~<user>/…` pass the audit. Round 3.
- **Reopened** by library review ([lr-1](round-1/lr-1/report.md)): 14 of 21 probes passed the audit.
  Examples: `public import`/`meta import`, a `[[require]]` whose `name` is not its first key, a
  `[[lean_lib]]` outside TVA, unscanned file types, names inside identifiers, `fpv/tva*` and `../`
  paths. Round 2.

## L8

**`Block.Fits` is stricter than `int32` indexing needs** (a library request by the cleared
vec-add kernel's prover).

- **Problem.** `TVA.Block.Fits p B` asks `p*B + B < 2^31`. The lane lemmas need only the largest
  offset, `p*B + B − 1`, to fit `int32`, that is `p*B + B ≤ 2^31`. So a kernel on `n` elements
  cannot cover `2^31 − B < n < 2^31`, though its indexing is exact there.
- **To do.** Relax `Fits` to `p*B + B ≤ 2^31`. Add a `fits_of_grid` form for `n < 2^31` when
  `B` divides `2^31`.

## L9

**`Launch.Valid` admits grids the launcher silently skips** (library review, round 1).

- **Problem.** Triton 3.8.0's launcher guards the launch with `gridX * gridY * gridZ > 0` computed
  in C `int` (`driver.c`). A program count in `[2^31, 2^32]` wraps to a value `≤ 0`, and the launch
  returns normally with no program run. Grids `(2^30, 2, 1)` and `(2^16, 2^15, 2)` satisfy
  `Launch.Valid`, so `Compiler.Faithful` is false there for any kernel that writes.
- **To do.** Add `L.size < 2^31` to `Launch.Valid` (docstring, SEMANTICS.md, the rule page), pin
  `driver.c`, and put the test in `evidence/`.

## L10

**`no-op:print` drops arguments that may have effects** (library review, round 1).

- **Problem.** `tl.static_print`, `tl.device_assert` and `tl.device_print` evaluate their arguments
  as ordinary expressions. With `debug` off, an atomic inside any of them still runs. A Python
  `assert` statement lowers to `device_assert`, and the drop list does not name it.
- **To do.** Allow the drop only for pure arguments (the same test as L6); otherwise transcribe the
  argument operations. Name `assert`.

## L11

**`rules/num.md` asks kernel reviewers to check device truth** (library review, round 1).

- **Problem.** Its section "What the reviewer judges" asks for the PTX or a test. Kernel reviewers
  read that page, and the protocol forbids them that check.
- **To do.** Address the page's device questions to library review, and tell kernel reviewers what
  they do check (the right fact for the operation).

## L12

**No digest covers the transcription rules** (the library author's notes, round 2).

- **Problem.** A kernel item's digest covers the Lean constants it reaches, and the claim reader
  hashes the docstrings of `TVA.Claim`, `TVA.Rules` and `TVA.Platform`. Nothing covers
  `docs/TRANSCRIPTION.md` or the drop list in `tools/tvr.py`. So when a rule or a drop reason
  changes (as L6 and L10 did), no kernel's `map` or `model` re-opens, and a map accepted under the
  old rule stays accepted.
- **To do.** Give each transcription rule and drop reason an identity, have the map entries name
  the ones they use, and fold their text into the `map` (and `model`) digests.

## L13

**A kernel's Lean can add `main` without the kernel checking it** (soundness; library review round 2,
[lr-2a](round-2/lr-2a/report.md)).

- **Problem.** Meta code in a kernel file (`run_cmd`, `#eval`, `elab`, `macro`, `initialize`) can
  call `addDeclCore … (doCheck := false)` and add `main` and `consistent` unchecked. A bundle with
  `Post = False` passes build, shape, axioms, usage and hygiene (`work/lean/GateHygiene.lean`).
- **To do.** Re-check the kernel's compiled output with an independent kernel checker, and/or
  forbid meta-level environment changes in kernel files.

## L14

**"Pure" is unsound for aggregate operands** (soundness; [lr-2a](round-2/lr-2a/report.md)).

- **Problem.** On an aggregate value, `+`, `==`, `<` and the like dispatch to its `@triton.jit`
  dunder methods, which are inlined and may run atomics. So `compile-check`, `no-op:print` and `dead`
  can drop a line with an effect (`work/l6_operator_dispatch.py`: 4 of 4 programs).
- **To do.** Restrict purity to operands of constexpr, dtype, shape, tuple or plain scalar type, or
  exclude aggregates.

## L15

**One program is not sequential for global memory** (soundness; [lr-2b](round-2/lr-2b/report.md)).

- **Problem.** The library says a program's later steps see its earlier writes (SEMANTICS.md,
  `Prog`/`Launch`, S1), and drops `tl.debug_barrier` and `num_stages`. On the H200:
  - a program that stores a block and loads it back reversed reads stale data in 24–40% of
    elements;
  - a loop that loads the row the previous iteration stored reads stale data when Triton pipelines
    it (`num_stages = 3`, or a `tl.dot` in the body at the default).

  A barrier, the same offsets (without pipelining), or `num_stages = 1` gives the sequential result.
  So `Compiler.Faithful` is false for such models.
- **Decision.** It needs the user's decision, because S1 settled the barrier the other way. Held out of round 3 until the user decides (asked 2026-10-27): exclude such
  models, or give `tl.debug_barrier` and `num_stages` a meaning. Until then, kernel claims should not
  rely on a program reading global bytes it wrote in an earlier step.

## L16

**`tl.max`/`tl.min` widen small integers to `int32`** (gap; [lr-2b](round-2/lr-2b/report.md)).

- **To do.** State it in SEMANTICS.md and by the integer reductions (to `int32` for any
  signedness; `uint8`/`uint16` zero-extended).

## L17

**Missing operations** (gap; [lr-2b](round-2/lr-2b/report.md)).

- **Problem.** These are missing:
  - `tl.permute`, and `trans` of rank 3 or more;
  - PDL: `gdc_wait`, `gdc_launch_dependents` and `launch_pdl`, which `Launch.Valid` does not
    exclude;
  - `tl.argmax`, which the docs name but nothing defines.
- **To do.** Add `permute`. Define PDL, or require `launch_pdl = false` and list it in
  exclusions.md. Define the index reductions, or drop the references to them.

## L18

**The environment changes defaults the model fixes** (gap; [lr-2b](round-2/lr-2b/report.md)).

- **Problem.** `TRITON_F32_DEFAULT` sets `tl.dot`'s `input_precision` when the source omits it;
  `DotCfg.precision` defaults to `tf32`.
- **To do.** Record it in the launch, or state that claims assume it unset.
