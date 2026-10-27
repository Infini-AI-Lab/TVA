# Rule `Compiler.Faithful`

Lean: `TVA.Rules.Compiler.Faithful` (`lean/TVA/Rules/Compiler.lean`).

```lean
def Faithful (P : Platform) (k : Kernel) (L : Launch) (K : Mem → Outcome → Prop) : Prop :=
  L.debug = false → L.Valid → ∀ init, WellBehaved (L.env P) k L init → ∀ o, K init o →
    match o with
    | .ok fin => Outcomes (L.env P) k L init (.ok fin)
    | .fault => False
    | .hang => Outcomes (L.env P) k L init .hang
```

## What it says

Take the compiled kernel `K`, launched as `L` from global memory `init`, without `debug`, with
a valid launch (`Launch.Valid`: CUDA's grid limits, fewer than `2^31` programs, `num_ctas = 1`,
no programmatic dependent launch, and block shapes Triton accepts), and with Triton's environment
at its defaults (below). If the model `k`'s launch is well behaved from `init` (no undefined
behaviour and no race between programs on any interleaving), then:

- a normal end of the device run leaves a memory that some interleaving of the model also
  leaves;
- the device run does not fault;
- the device run hangs only if some interleaving of the model runs forever.

A launch is *rejected* when Triton or the driver reports it to the host with an error before
any program runs. A rejected launch has no outcome at all (`TVA.Claim.Behaviour`). So neither
this rule nor any claim says whether a kernel compiles (user, 2026-10-27).

A launch that returns without an error has an outcome, even if no program ran. With a zero grid
dimension, that outcome is a normal end with memory unchanged, which is also what the semantics
gives. Triton 3.8.0's launcher returns without an error, and runs no program, when the grid has
`2^31` to `2^32` programs: it tests `gridX * gridY * gridZ > 0` in C `int`
(`primitives/triton-3.8.0-nvidia-driver.c:968-972`). `Launch.Valid` excludes such grids by
asking for fewer than `2^31` programs, so a valid launch that returns without an error has run its
whole grid ([evidence/launch-outcomes](../../evidence/launch-outcomes/README.md); library
issue L9).

### Triton's environment at its defaults

Triton 3.8.0 reads environment variables that change what it compiles or how it launches. The
rule covers a kernel compiled and launched with none of them set, except those that only name
Triton's home, cache and dump directories, only dump, print or time, or only change debug
information: `TRITON_HOME`, `TRITON_CACHE_DIR`, `TRITON_DUMP_DIR`, `TRITON_KERNEL_DUMP`,
`TRITON_ALWAYS_COMPILE`, `TRITON_PRINT_AUTOTUNING`, `TRITON_DUMP_PTXAS_LOG`, `MLIR_ENABLE_DUMP`,
`MLIR_DUMP_PATH`, `LLVM_IR_ENABLE_DUMP`, `NVPTX_ENABLE_DUMP`, `MLIR_ENABLE_TIMING`,
`LLVM_ENABLE_TIMING`, `TRITON_DISABLE_LINE_INFO`, `USE_IR_LOC`. So, among others, these are unset:
- `TRITON_F32_DEFAULT`, which sets the `input_precision` of a `tl.dot` that gives none: with it
  unset, that is `tf32`, as the model's `DotCfg` says;
- `TRITON_DEFAULT_FP_FUSION`, which sets the default of `enable_fp_fusion`: with it unset, `true`;
- `TRITON_DEBUG` (`debug`), `TRITON_INTERPRET` (the interpreter, not the device),
  `TRITON_KERNEL_OVERRIDE` (replaces the compiled code with files), `TRITON_OVERRIDE_ARCH`,
  `TRITON_LIBDEVICE_PATH`, `TRITON_PTXAS_PATH`, `PTXAS_OPTIONS`, `DISABLE_PTXAS_OPT`,
  `DISABLE_LLVM_OPT`, `TRITON_CACHE_MANAGER` and `TRITON_REMOTE_CACHE_BACKEND`.

[evidence/env-knobs](../../evidence/env-knobs/README.md) shows the first two at work. The library
reads the list from Triton's `knobs.py` and its C++ list of environment variables
(`primitives/triton-3.8.0-knobs.py:343-513`, `primitives/triton-3.8.0-GetEnv.h:15-57`).

## What it trusts

- The Triton compiler (3.8.0) and its documentation: the device runs the Triton program as
  `TVA.Sem` reads the documentation. User, 2026-10-26: "we trust triton compiler and
  documentations".
- The device's meaning of approximate operations, which is the platform `P` the claim
  quantifies over. Accuracy facts about `P` are separate rules (`Rules.Num.*`), not part of this
  one.

## What it does not cover

- **The model's correspondence to the source.** The model is the prover's transcription, and
  the reviewer checks it line by line (`MODEL_MISMATCH`). The rule is about the model.
- **Launches that are not well behaved, or not valid.** For those the rule claims nothing. A
  proof must show the launch is well behaved, for example with `TVA.independent`, and valid
  (`meets_of_faithful` asks for `(launch d).Valid`; `Launch.valid_grid1` proves it for a
  one-dimensional grid).
- **Launches with programmatic dependent launch** (`launch_pdl=True`, `Launch.pdl`): the kernel
  may start before the previous kernel on the stream has finished writing, so its programs may
  read memory other than `init`. `Launch.Valid` asks for `pdl = false`.
- **A non-default environment** (above).
- **Whether the kernel compiles.** A rejected launch has no outcome, so the rule covers only
  launches that run. The limits behind a rejection include (A1–A8 of
  [evidence/launch-outcomes](../../evidence/launch-outcomes/README.md) show several):
  - shared memory, registers and threads;
  - dtypes the architecture lacks (`float8e4nv` before sm_89, the AMD fp8 formats on NVIDIA);
  - `tl.dot` minimum sizes;
  - a failing `tl.static_assert`;
  - a grid dimension above CUDA's limits;
  - a descriptor block whose last dimension is under 16 bytes;
  - a gather or scatter of fewer than 8 rows.

  A claim may cover instances that never compile. Nothing is promised there, and for a valid
  launch nothing goes wrong silently: the host gets the error, no program runs, and memory is
  unchanged.
  `Launch.Valid` checks only the block shapes listed in `blocks`.
- **The host.** Arguments, grid and allocation are the instance `d`: `launch d`, and the buffers
  in `Pre`.
- **The [exclusions](../exclusions.md).**

## How a bundle cites it

```lean
structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful` -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
```

The reviewer checks four things:

- `model d` is the kernel's source, with every implicit conversion, broadcast and mask;
- `launch d` is the host's launch: the grid formula, `enable_fp_fusion`, `num_ctas`, `launch_pdl`
  (not set), and the source's block shapes in `blocks` (the bundle proves `Launch.Valid`, so the
  grid has fewer than `2^31` programs on every covered instance);
- `d.ok` covers the instances the claim is about;
- the bundle proves `Claim.Consistent Assumptions` (`exists_total_faithful` gives the behaviour
  for this rule).

## Consistency

The rule holds of some total behaviour for every model and launch:
`TVA.exists_total_faithful` (`lean/TVA/Lemmas/Consistency.lean`). The behaviour follows the
model where the launch is well behaved and faults elsewhere. It is total because a
well-behaved launch always ends normally or runs forever (`WellBehaved.outcome`): no state of
the semantics is stuck (`Step.progress`).
