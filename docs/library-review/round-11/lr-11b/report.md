# Library review, round 11, by lr-11b

## Verdicts

- **L5** ACCEPT — `TensorDesc.Valid` now asks `0 < shape[k]` for every dimension, so every access
  through `Access` or `RowAccess` with a zero entry is undefined, and building such a descriptor
  stays pure. My rerun of evidence/desc-zero-shape gave the same output as `output.txt`: Z2, Z3,
  Z5–Z7, Z10 and Z11 fault, Z1, Z4 and Z12 match the model, Z13 is rejected on the host, and
  gather and scatter fail in ptxas on sm_90a (Z8, Z9, Z14–Z16). SEMANTICS "Tensor descriptors" and
  the docstrings say the same. I also tested the strides that `Valid` still allows
  (work/desc_strides.py): a zero non-last stride, overlapping rows, and strides of 2^38 and 2^40
  bytes on a dimension of size 1. All five in-kernel loads matched the model and none faulted.
- **L37** ACCEPT — exclusions.md has both rows of the user's 2026-10-28 decisions, each citing L37.
  Tutorial 08's `grouped_matmul_kernel` is excluded, with the reason SEMANTICS gives: dimension 0
  of `a_ptrs` has bound 1, so `[16, 16]` promises that every pointer is aligned. Tutorial 09's
  pointer kernels are covered on full tiles only. On a full tile, `tl.where(offs_am < M, offs_am,
  0)` is the identity, so `max_contiguous_multiple_of offs_am BM BM` holds. SEMANTICS "Compiler
  hints" points to both rows.
- **L57** ACCEPT — PROVER_GUIDE's bullet now matches SEMANTICS "Constants" point by point. The
  constant is built in the operation's dtype with an arithmetic or bitwise operator, as a value of
  `tl.where`, and in `tl.full`. It is a tensor of its own type in a comparison, `other=`, a stored
  value and other `tl.*` functions. The bullet also points to L59's fact.
- **L58** ACCEPT — The readings match the compiler. `AxisInfo.cpp:1244-1259` replaces only the
  hinted parts of the lattice. `AxisInfo.h` defines divisibility per group of the contiguity, and
  it raises every dimension's divisibility to that of any dimension with contiguity 1. A `C = 1`
  reading already implies that raise, since every element is then a group start. Contiguities are
  powers of two, so a lower bound `C` divides the compiler's contiguity, and `FirstsDiv` with that
  `C` implies the compiler's belief. The pair is exactly the compiler's belief. My rerun of
  evidence/hint-contig (H1–H21) gave the same output as `output.txt`.
- **L59** ACCEPT — `visit_Assign`'s `_sanitize_value` (`code_generator.py:727-760`) is what
  SEMANTICS now states: a target annotated `: tl.constexpr` keeps the value, tuples are converted
  element by element, `None` and dtypes are kept, and other values go through `to_tensor`. My rerun
  of evidence/python-numbers (A1–A14, B1–B4) gave the same output as `output.txt`. The
  TRANSCRIPTION rows `e = 0.1` and `i = 0` follow from the fact.
- **L60** ACCEPT — "Check each result" is sound whether or not a pass keeps the mark (M1, M7–M10).
  The inlined-call drop rests on the inliner removing the marked `tt.call`, which M3, M12 and M13
  confirm. My rerun of evidence/hint-multi gave the same output as `output.txt`.
- **L61** ACCEPT — Each item now says what is true:
  - index promotion by an `int64` or `uint64` operand, and `[-2^31, 2^31)`;
  - a name that only one branch of an `if` defines (evidence/if-names N1–N6, my rerun identical);
  - builtin `min` and `max`, against `core.py:3879-3929`: compile-time arguments only refuse NaN and
    `-0.0`, and with a tensor argument the call folds `minimum`/`maximum`;
  - the arguments the last hook leaves, in Faithful's "Which launch" and SEMANTICS "One launch";
  - "not necessarily `0`";
  - PROVER_GUIDE's hint sentence.
- **L71** ACCEPT — The reader elaborates `TVA.Claim.Satisfiable NS.Inst.ok NS.Pre := NS.satisfiable`
  on the same constants that the `Spec` check uses, and `axioms` checks its axiom list. My rerun of
  evidence/claim-satisfiable gave the same S1 and S2 results. The check is complete for what it
  states. `Consistent`'s total witness, applied to the `satisfiable` instance, has an outcome, and
  `main` makes it a normal end with `Post`. With one `Compiler.Faithful` field, `main` and
  `consistent` also force `Launch.Valid`, `debug = false` and well-behavedness on every covered
  instance: otherwise a behaviour that faults there satisfies the assumptions.
  `TVA.Claim`, PROVER_GUIDE and protocol §8 say that the witness is one instance and memory, and
  that coverage of real use is the reviewer's judgment. The witness may be degenerate (an empty
  grid). REVIEWER_GUIDE §2's "Are the conditions satisfiable" remains the place for that check.
- **L72** ACCEPT — Scope 4 now covers options given at `kernel.warmup(…)` (`jit.py:362-363`) and
  in `triton.compile`'s `options` (`compiler.py:226-243`). The rule page, SEMANTICS "Launches",
  exclusions.md, TRANSCRIPTION's host pitfall and REVIEWER_GUIDE name both. My rerun of
  evidence/compiled-options (O1–O7) gave the same output as `output.txt`. One more route exists:
  `@triton.heuristics` can supply options. It is filed below as a new issue, since L72 was about
  precompiled kernels.
- **L73** ACCEPT — The frame follows from the trust in the compiler, and the rule's hashed text
  states it. My L71 argument shows that `main` with `consistent` forces validity and
  well-behavedness on every covered instance, so a claim does imply the frame there. `Post` cannot
  state it, and `TVA.Claim` says so. My rerun of evidence/launch-frame matched `output.txt`; only
  the counts of changed bytes inside the outputs differ, because the inputs are random. I widened
  the evidence (work/frame_battery.py, W1–W12). No byte outside the written buffers and scratch
  changed under either sentinel, and only the in-kernel descriptor kernel (W9) had a global
  scratch. The eleven kernels that ran cover:
  - host descriptors with a `warp_specialize` `tl.dot` loop;
  - `tl.gather`;
  - an emulated mxfp8 `tl.dot_scaled`;
  - `tl.device_print`;
  - a custom `tl.reduce` and `tl.associative_scan`, and `tl.argmax`;
  - `tl.join`, `split`, `permute`, `trans` and `reshape`;
  - a 128×256×64 `num_stages=4` dot;
  - a persistent `flatten=True` loop over in-kernel descriptors;
  - `atomic_cas` and `atomic_xchg`;
  - a masked `float16` `atomic_add` ending inside a pair, with `-0.0` beside it;
  - 2-D cumsum, sort and flip.

  W2, a `warp_specialize` loop with pointer loads, was rejected on the host.
- **L74** ACCEPT — Scope 7 names the H200 and code compiled for it, and so do `TVA.Claim`,
  `Behaviour`, the rule page, SEMANTICS and exclusions.md. The grid sentence matches
  `nvidia-driver.c:968-972` and protocol §6. Scope 6's host descriptors match `specialize.cc:165-226`
  (`tensordesc<dtype[block]>`) and `make_tensordesc_arg` (`nvidia-driver.py:209-264`), which takes
  the block from the compiled metadata and the base, shape, strides and padding from the argument
  (O5–O7 reproduced).
- **L75** ACCEPT — `builtin_namespace` (`code_generator.py:357-365`) holds exactly the eleven names,
  and `name_lookup` tries local, then global, then builtin names (`:381-427`). So SEMANTICS's "a
  name that neither the function nor its module defines" is the right condition, and other
  builtins raise `NameError` on the host. My rerun of evidence/py-builtins (F1–F10) gave the same
  output as `output.txt`.

## New issues

- **NEW** soundness — `@triton.heuristics` gives compile options, and the library names no such
  route. `Heuristics.run` (`primitives/triton-3.8.0-autotuner.py:480-483`) sets `kwargs[name]` for
  each heuristic, replacing a launch keyword of the same name, then calls the JIT's `run`. That
  `run` parses options out of those keywords (`jit.py:726-741`). In work/heur_options.py, with no
  launch keyword:
  - H1: a heuristic `enable_reflect_ftz: False` gave `exp2(-130) = 2^-130` (compiled with
    `enable_reflect_ftz=False`);
  - H3: a heuristic `enable_fp_fusion: False` removed the `fma`.

  H5: a heuristic `enable_fp_fusion: True` overrode a launch keyword `enable_fp_fusion=False`. The
  device fused, and `x*y+z` gave `2^-24`. A model that follows `Launch.fusion`'s docstring ("the
  source's or the launch's value") has `fusion = false`, so `tl.mul_add` can only give the unfused
  `0`, and `Faithful`'s conclusion fails for a well-behaved, valid model. `debug`, `num_ctas`,
  `launch_pdl` and the excluded options (`ir_override`, …) reach the compile the same way. Scope 4
  (`lean/TVA/Rules/Compiler.lean`, docs/rules/compiler-faithful.md), SEMANTICS "Launches", the
  options row of exclusions.md, REVIEWER_GUIDE's launch and host checks, and the docstrings of
  `Launch.fusion` and `Launch.debug` (`lean/TVA/Sem/Launch.lean`) name only:
  - the launch;
  - `warmup`;
  - `triton.compile`;
  - a `triton.Config`.

  **Fix:** state that a field of `Launch` is the value the JIT receives: the launch keyword, an
  autotune config's, or a `@triton.heuristics` value, which replaces both. Name heuristics among
  the places scope 4 forbids other options, in those docstrings, SEMANTICS, exclusions.md and the
  reviewer's checks.
- **NEW** wording — `triton.compile(src)` with `src` a file path compiles that IR file
  (`IRSource`, `primitives/triton-3.8.0-compiler.py:87-112, 233-240`: `.ttir`, `.ttgir`, `.ptx`, …),
  not the kernel's Python source. Yet scope 6 (`Compiler.Faithful`'s docstring, the rule page,
  SEMANTICS "Launches") presents any `CompiledKernel` from `triton.compile` as covered, with the
  specialisation of "its signature and attributes". **Fix:** say that a `CompiledKernel` from
  `triton.compile` is covered only when `src` is an `ASTSource` of the kernel's function. A kernel
  compiled from an IR file runs that IR and, like `ir_override`, is outside the scope.

## Re-ran

- In full, on GPU 0 (H200 NVL, Triton 3.8.0, torch 2.11.0+cu130). Each run matched the
  author's `output.txt` exactly, except that launch-frame's counts of changed bytes inside the
  outputs differ, because the inputs are random:
  - evidence/compiled-options;
  - evidence/launch-frame;
  - evidence/desc-zero-shape;
  - evidence/hint-contig;
  - evidence/hint-multi;
  - evidence/python-numbers;
  - evidence/if-names;
  - evidence/py-builtins.

  The outputs are in work/rerun/.
- evidence/claim-satisfiable (Lean, no GPU): S1 and S2 as recorded.
- New tests, outputs beside each script:
  - work/frame_battery.py (W1–W12): the L73 frame and the global scratch, over eleven more kernel
    shapes;
  - work/desc_strides.py (S1–S5): in-kernel descriptor strides that `TensorDesc.Valid` allows (L5);
  - work/heur_options.py (H1–H5): compile options given by `@triton.heuristics` (new issue).

## Not checked

- `tvr check` (not to be run this cycle), and the release check, which the coordinator runs.
- TRANSCRIPTION.md beyond the rows and pitfalls the batch changed (lr-13b's survey area).
- `lean/TVA/Math` and the numeric lemmas (being replaced in round 12). I did not check the
  device facts' constants in `TVA.Rules.Num`/`Sm90`, only that `Platform.sm90ref` satisfies them
  together.
- Whether the device's printf buffer, which `tl.device_print` writes (W5), counts as "global
  memory" in the frame's sense. Settling it needs CUDA's documentation, which is outside the
  sources this review may use.
