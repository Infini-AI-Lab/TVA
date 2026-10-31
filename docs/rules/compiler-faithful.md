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

Take the compiled kernel `K`, launched as `L` on an NVIDIA H200 (sm_90), driven from an x86-64 host
running CPython 3.13 (without `-O` or `-OO` and with `PYTHONOPTIMIZE` unset, in the default floating-point environment), from global memory `init`, without `debug`, with a valid launch (`Launch.Valid`: CUDA's grid limits, fewer than
`2^31` programs, `num_ctas = 1`, no programmatic dependent launch, and block shapes Triton
accepts), within the rule's scope (below; (A), the kernel and its launches, checked: only the
options `Launch` records, wherever they are given, and no `@triton.heuristics`, the scratch of
descriptors built in the kernel its own, a precompiled kernel compiled from the kernel's source and
given arguments in its specialisation, no host descriptor that rounds to tf32, no CUDA graph
capture; (B), the environment, assumed and not checked: nothing else writes the launch's bytes,
Triton's knobs at their defaults, no hook, no `debug` attribute set from outside and no `preload`,
the H200 and the host as the target, Triton and the kernel's module as installed and written, and
the kernel new: the code Triton compiles for the launch's own argument values and settings). If
the model `k`'s launch is well behaved from `init` (on no interleaving undefined behaviour, a race between programs, or a conflict within one program: two
accesses of a program to the same bytes, at least one writing them, unless both are atomics on
the same bytes, library issue L15), then, with the semantics performing a program's atomics in
every order their dependences allow (`Step.defer`):

- a normal end of the device run leaves a memory that some interleaving of the model also
  leaves;
- the device run does not fault;
- the device run hangs only if some interleaving of the model runs forever;
- the device run writes no byte of global memory other than those `init` gives, except the scratch
  of the descriptors the kernel builds (scope 5; "Memory", below).

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
issue L9). A call made on a stream that is capturing into a CUDA graph also returns without an error
and runs no program, and its programs run later, at each replay of the graph; a call made on another
stream during the capture runs at once. Scope 9 excludes every call made while a capture is in
progress, on any stream, and every replay (library issue L91).

## Memory

`init` gives the launch its bytes: the bytes present in it (`Mem` is a partial map), the buffers of
the claim's `Pre`. The outcome `ok fin` gives the same bytes, with their contents when the launch
ends. Bytes `init` does not give are no part of the outcome, so `Post` cannot speak of them, and the
model may not access them (library issue L49).

The rule also states a frame (library issue L73): a valid launch within the scope, from an `init`
from which it is well behaved, writes no byte of global memory other than those `init` gives, except
the scratch of the descriptors the kernel builds (scope 5). The device writes what the model writes,
and the launcher passes the kernel no other buffer of its own but that scratch and those of
instrumentation (`instrumentation_mode`: a profile scratch, a sanitizer's state), which scopes 2 and
4 exclude (`primitives/triton-3.8.0-nvidia-driver.py:280-333`). In
[evidence/launch-frame](../../evidence/launch-frame/README.md), every buffer was carved out of one
sentinel-filled pool with guards between them: a vector add with a masked last tile, a row softmax,
a pipelined `tl.dot`, integer and float atomics, a sort, a scan and a histogram changed no byte
outside their output buffers, and a copy through in-kernel descriptors none outside its output and
its scratch; the global scratch size was `0` for every kernel but that one (128 bytes per
descriptor), and the profile scratch size `0` for all. So a claim that cites
the rule implies that, on its covered instances, the launch leaves every other byte the host holds
as it was ([`TVA.Claim`](../../lean/TVA/Claim.lean), "Other memory"). What anything else does to
those bytes is outside the claim.

## Scope

The rule covers a launch only under the conditions below, in two parts (library issues L5, L17, L18,
L19, L47, L49, L72, L74, L76, L77, L78, L91, L103, L104, L105, L107, L109, L110, L116, L117). They are
about the host, which `Launch` does not describe, so the claim's text states them (the hashed
docstring of `Rules.Compiler.Faithful`). The user's decisions, 2026-10-29, under L36: "To make the
verifier easy, we do not verify triton cache.", "we assume the kernel is new." and "So basically, we
want to verify the kernel itself, rather than triton installation or other code programmed by
users."; on the autotuner ("Which launch"), "actually we do not neccessarily cover autotune. we do
verification basically for the kernel." and "our principle is, we are verifying kernel itself.
[...] actually not the prover picks a config, it is we pick a config. prover is not aware of the
autotune." Each condition keeps its number of earlier rounds, so that a citation of "scope `n`" or
"point `n`" still names it.

### (A) The kernel and its launches: checked

What the claim's instances describe and the kernel reviewer checks against the source: the
`@triton.jit` function, its helpers and its decorators as the kernel's source writes them
(`triton.heuristics`, and `triton.autotune`, which is outside the supported subset: "Which
launch"); the launch call, a direct launch of the `@triton.jit` function (the grid, the arguments,
the compile-time values and the launch options the call passes); and what the kernel needs from its
host to run at all (an allocator for descriptors, the host `TensorDescriptor`s the launch is given).
The checklist ("How a bundle cites it") lists these.

- **4.** **Only the launch options `Launch` records.** Every field of the NVIDIA backend's
   `CUDAOptions` (`primitives/triton-3.8.0-nvidia-compiler.py:108-135`) is accepted as a launch
   keyword (`jit.py:713-715`), in an autotune `triton.Config`, by `kernel.warmup(…)`, which takes
   the same keywords (`primitives/triton-3.8.0-jit.py:362-363`), and in `triton.compile`'s
   `options` (`primitives/triton-3.8.0-compiler.py:226-243`); the launch of the `CompiledKernel`
   these two return takes none. The rule covers a launch only when none is given to the call that
   compiles the kernel (a JIT launch, `warmup`, `triton.compile`) or in a config, except
   `enable_fp_fusion`, `debug`, `num_ctas` and `launch_pdl` (`Launch.fusion`,
   `debug`, `numCtas`, `pdl`) and the four that do not change the meaning: `num_warps`,
   `num_stages`, `maxnreg` and `launch_cooperative_grid` (no exact result changed under any of
   them, evidence/launch-options B). So `ir_override` (gave `x + 2` for `x + 1`, A1),
   `allowed_dot_input_precisions` (switched a `tl.dot` to `ieee`, D1),
   `default_dot_input_precision`, `enable_reflect_ftz` (turned libdevice's flush-to-zero off, E1),
   `extern_libs`, `ptx_options`, `ptx_version`, `arch`, `warp_size`, `supported_fp8_dtypes`,
   `deprecated_fp8_dot_operand_dtypes`, `max_num_imprecise_acc_default`, `sanitize_overflow`,
   `instrumentation_mode` and `backend_name` are not given. `enable_reflect_ftz=False` changed
   `tl.math.exp2(-130)` from `0` to `2^-130` given at a JIT launch, at `warmup` (and then a launch
   with no keyword) and in `triton.compile`'s options alike
   ([evidence/compiled-options](../../evidence/compiled-options/README.md) O1–O4; library issue L72).
   And the kernel is not under `@triton.heuristics`: a heuristic sets launch keywords before the JIT
   reads them, compile options included, and replaces a launch keyword or a config's value of the
   same name (`primitives/triton-3.8.0-autotuner.py:473-483`). A heuristic `enable_fp_fusion: True`
   overrode a launch's `enable_fp_fusion=False`, so `x*y+z` was fused, and `enable_reflect_ftz: False`
   changed `tl.math.exp2(-130)` (`docs/library-review/round-11/lr-11b/work/heur_options.py` H1, H5;
   library issue L78, excluded under L88, [exclusions.md](../exclusions.md)).
   The decorator's `@triton.jit(debug=True)` sets the `JITFunction`'s attribute `debug`, which a
   launch that gives no `debug` takes (`kwargs.get("debug", self.debug)`,
   `primitives/triton-3.8.0-jit.py:727`): it gives `Launch.debug`.
- **5.** **The scratch of descriptors built in the kernel is the launch's own.** A kernel that builds
   tensor descriptors (`tl.make_tensor_descriptor`) needs an allocator (`triton.set_allocator`;
   the default one raises, `primitives/triton-3.8.0-allocation.py:17-21`), and without one the launch
   is rejected on the host with a `RuntimeError` before any program runs. The launcher asks the
   allocator for `global_scratch_size` bytes per program (128 per descriptor) once per launch, on
   the launch's stream (`primitives/triton-3.8.0-nvidia-driver.py:297-333`), and each program
   writes its own descriptors there (`primitives/triton-3.8.0-TMALowering.cpp:166-184`), and nothing else
   ([evidence/desc-scratch](../../evidence/desc-scratch/README.md) D2, D3). The model does not
   access that buffer (`tl.make_tensor_descriptor` is pure). The rule covers such a launch only
   when the buffer holds no byte `init` gives the launch and nothing else writes it until the
   launch ends: an allocator that returned the start of an input buffer had 1024 of its elements
   overwritten (D4).
- **6.** **A precompiled kernel gets arguments of its specialisation.** A launch through a
   `CompiledKernel` (`kernel.warmup(…)` or `triton.compile`, then `compiled[grid](…)`, as tutorial
   02 does) runs the code compiled for the warm-up's arguments (for `triton.compile`, its signature
   and attributes), with the warm-up's constexprs (a constexpr position of the launch is ignored),
   and the launcher checks nothing against them (`primitives/triton-3.8.0-compiler.py:407-515`). Outside that specialisation, a misaligned
   pointer faulted, `n = 1001` on a kernel compiled for `n = 1024` wrote three elements past `n`, a
   stride of `3` on a kernel compiled with `1` was read as `1`, an integer of `2^31` or more passed
   to an `int32` parameter wrapped, and a `float16` tensor in a `float32` slot was read as raw bytes
   ([evidence/compiled-launch](../../evidence/compiled-launch/README.md) K2). The rule covers the
   launch, as the launch of the warm-up's specialisation, only when each argument is in it: the
   same dtypes, each pointer 16-byte aligned exactly when the warm-up's was, and each integer in
   the same class (equal to `1`; otherwise a multiple of 16 or not, and in `int32`, `int64` or
   `uint64` range; [SEMANTICS.md](../SEMANTICS.md), "Scalar arguments"), and each host tensor
   descriptor (`TensorDescriptor`) with the warm-up's dtype and block shape, its specialisation
   (`primitives/triton-3.8.0-specialize.cc:165-226`). The launcher builds the TMA descriptor with the
   compiled kernel's block shape and element type, not the argument's
   (`primitives/triton-3.8.0-nvidia-driver.py:209-264`): a `[16, 16]` descriptor given to a kernel
   compiled for `[32, 32]` had 1024 elements copied (evidence/compiled-options O6; library issue
   L74). Tutorial 02 launches with the warm-up's own arguments. A kernel `triton.compile` built is
   covered only when `triton.compile`'s source is an `ASTSource` of the kernel's `@triton.jit`
   function: given a file path, it compiles that IR file (a `.ttir`, `.ttgir` or `.ptx` file,
   `IRSource`, `primitives/triton-3.8.0-compiler.py:87-112, 233-240`), which replaces the source as
   `ir_override` does (library issue L77).
- **8.** **Host descriptors load what memory holds.** No host tensor descriptor (`TensorDescriptor`) the
   launch is given has `round_f32_to_tf32=True`. With it, the launcher builds the TMA descriptor
   with the element type `TMA_TF32` (`primitives/triton-3.8.0-nvidia-driver.py:231-232`), and its
   loads round `float32` to tf32: `1 + 2^-23` was loaded as `1.0` and `0x3dcccccd` as `0x3dccc000`
   ([evidence/desc-tf32](../../evidence/desc-tf32/README.md)), where `tl.desc_load` returns the stored
   values (library issue L76; no pinned tutorial sets the field).
- **9.** **No CUDA graph capture.** The call that launches the kernel is not made while a CUDA stream
   capture is in progress, on any stream (`torch.cuda.graph(…)`, `torch.cuda.CUDAGraph.capture_begin`,
   `triton.testing.do_bench_cudagraph`, a `cudaStreamBeginCapture` by other code), and the launch is
   not a replay of a captured graph (`graph.replay()`). A call on the capturing stream (the current
   stream inside `torch.cuda.graph(…)`) is captured: it returns without an error, compiling the kernel
   first if needed, and runs no program; the programs run at each replay, with the arguments the call
   captured, on the memory at the replay (`docs/library-review/round-16/lr-16b/work/graph_capture.py`
   G1-G3: the output kept its sentinel after the captured call, held `x + 1` after the first replay
   and `11` after the host set `x = 10` and replayed, and a captured `n = 16` wrote 16 elements
   whatever the host's `n` was later). A call on another stream while a capture is in progress is not
   captured: it runs at once, as any launch, and a replay does not run it
   ([evidence/host-round20](../../evidence/host-round20/README.md) G0–G3, in the capture modes
   `global`, `thread_local` and `relaxed`). The rule covers no call made while a capture is in
   progress, captured or not (the captured call's outcome, "ok, memory unchanged", is not the
   model's), and no replay, which is not a call through Triton: an exclusion wider than the calls
   capture changes (library issue L91; excluded by the user's decision, 2026-10-28,
   [exclusions.md](../exclusions.md); no in-scope tutorial's host captures).

### (B) The environment: assumed, not verified

TVA verifies the kernel, not the Triton installation or other code users write. These conditions
are assumptions of the rule: the gate, the prover and the kernel reviewer do not check them, and
the kernel reviewer does not read host code for them. A claim's own launch or kernel source that
visibly breaks one (the source sets a knob, or rebinds a Triton name) is still a finding.

- **1.** **Nothing else writes the launch's bytes.** From the launch's start until it ends, nothing but
   the launch writes a byte `init` gives it: not the host, not another kernel on any stream or
   device, and not a later kernel that programmatic dependent launch starts early. A kernel whose
   programs call `gdc_launch_dependents()` lets the next launch on its stream, if that launch has
   `launch_pdl=True`, start while it still runs; that launch then wrote the kernel's input in 20 of
   20 runs ([evidence/pdl](../../evidence/pdl/README.md) part D), and if it writes a byte the
   kernel is given the case is outside the rule. Writes to other bytes do not matter, whenever
   they happen: the device's stores are byte-granular. Another stream's kernel that wrote the
   bytes between the claimed ones, in the same sectors, lines and 4-byte words, while the claimed
   launch ran (19 of 20 trials overlapped) changed none of its bytes, and a writer to the claimed
   bytes themselves corrupted up to all of them
   ([evidence/other-writers](../../evidence/other-writers/README.md)). So the condition is about the
   bytes `init` gives: no other stream, thread or device writes the claim's buffers until the launch
   has finished.
- **2.** **Triton's knobs are at their defaults.** Every knob of `triton.knobs`
   (`primitives/triton-3.8.0-knobs.py:320-513`) and every environment variable Triton's C++ reads
   (`primitives/triton-3.8.0-GetEnv.h:15-57`) has its default, however it would be set: by the
   environment variable, or by assigning the knob from Python. A knob's environment variable is
   only its default (`knobs.py:68-100`): `triton.knobs.language.fp32_default = "ieee"` changes a
   `tl.dot` with no environment variable set when `triton.knobs.propagate_env` is `False`, and sets
   the variable otherwise; `triton.knobs.runtime.debug` is a plain attribute
   ([evidence/launch-options](../../evidence/launch-options/README.md) D3, D4, F1, S1). The
   exceptions only name Triton's home, cache and dump directories, only dump, print or time, or
   only change debug information: `TRITON_HOME`, `TRITON_CACHE_DIR`, `TRITON_DUMP_DIR`,
   `TRITON_KERNEL_DUMP`, `TRITON_ALWAYS_COMPILE`, `TRITON_PRINT_AUTOTUNING`,
   `TRITON_DUMP_PTXAS_LOG`, `MLIR_ENABLE_DUMP`, `MLIR_DUMP_PATH`, `LLVM_IR_ENABLE_DUMP`,
   `NVPTX_ENABLE_DUMP`, `MLIR_ENABLE_TIMING`, `LLVM_ENABLE_TIMING`, `TRITON_DISABLE_LINE_INFO`,
   `USE_IR_LOC`, and the knobs they set. So, among others, these are at their defaults:
   - `TRITON_F32_DEFAULT` / `knobs.language.fp32_default`, which sets the `input_precision` of a
     `tl.dot` that gives none: at its default, that is `tf32`, as the model's `DotCfg` says;
   - `TRITON_DEFAULT_FP_FUSION` / `knobs.language.default_fp_fusion`, the default of
     `enable_fp_fusion`: `true`;
   - `TRITON_DEBUG` / `knobs.runtime.debug` (`debug`), `TRITON_INTERPRET` (the interpreter, not
     the device), `TRITON_KERNEL_OVERRIDE` (replaces the compiled code with files),
     `TRITON_OVERRIDE_ARCH`, `TRITON_LIBDEVICE_PATH`, `TRITON_PTXAS_PATH`, `PTXAS_OPTIONS`,
     `DISABLE_PTXAS_OPT`, `DISABLE_LLVM_OPT`, `TRITON_CACHE_MANAGER`, `TRITON_REMOTE_CACHE_BACKEND`
     and `knobs.compilation.instrumentation_mode`.
- **3.** **No hook is installed:** `knobs.runtime.launch_enter_hook`, `launch_exit_hook`,
   `kernel_load_start_hook`, `kernel_load_end_hook`, `kernel_unload_hook`, `jit_cache_hook`,
   `jit_post_compile_hook`, `add_stages_inspection_hook`, `knobs.compilation.listener`,
   `knobs.autotuning.listener` (`knobs.py:470-492`, `:372`, `:385`), and no pre-run hook of the
   kernel's `JITFunction` (`primitives/triton-3.8.0-jit.py:735-736`). A stages hook rewrote
   `x + 1` into `x - 1`, and a `jit_cache_hook` returning true made the call return with nothing
   run (`jit.py:757-759`, `:880-882`; evidence/launch-options A2, A3).
- **4.** **(the environment's part) No `debug` attribute set from outside, and no `preload`** (library
   issue L105). No code but the kernel's source sets the kernel's `JITFunction` attribute `debug`:
   `kernel.debug = True` from Python makes the launch's `debug` true (`Launch.debug`), and with it an
   `int32` overflow trapped ([evidence/host-round20](../../evidence/host-round20/README.md) D1). And no
   `JITFunction.preload` is called on the kernel in the process: it compiles the kernel with the
   options its data names and stores it under the cache key its data names
   (`primitives/triton-3.8.0-jit.py:823-875`), so that a later launch with that key runs that code with
   no option given: after a `preload` of code compiled with `enable_reflect_ftz=False`, a plain launch
   gave `tl.math.exp2(-130)` as `2^-130`, not `0` (P1).
- **7.** **The target is the H200.** The launch runs on an NVIDIA H200 (sm_90), TVA's only target
   ([exclusions.md](../exclusions.md)), with code Triton compiles for the device it runs on: no
   `arch` option and no `TRITON_OVERRIDE_ARCH` (points 2 and 4), and no `triton.compile` target
   other than the device's (one for sm_80 was rejected on the host, "no kernel image is available",
   evidence/compiled-options O5). The model's defaults are the H200's (`DotCfg`: `tf32` when
   `input_precision` is omitted, `maxImpreciseAcc = 2^30` for an fp8 × fp8 `tl.dot`), and the device
   facts a claim cites are those of `Rules.Num.Sm90` ([num.md](num.md)) (library issue L74). The host
   that runs Triton is an x86-64 machine with CPython 3.13, as TVA's evidence ran (library issue
   L107): some values [SEMANTICS.md](../SEMANTICS.md) states are the host's, such as the bits of a NaN
   passed as a `tl.float16` argument (CPython 3.13's `PyFloat_Pack2` drops the payload, `0x7e00` or
   `0xfe00`; CPython 3.14 keeps its top bits, `docs/library-review/round-19/lr-19b/work/host_pack2.py`)
   and the sign of Python's `float('inf') - float('inf')` (x86-64's default NaN,
   `0xfff8000000000000`). The host process is CPython 3.13 run without `-O` or `-OO` and with
   `PYTHONOPTIMIZE` unset, in the default floating-point environment (round to nearest even;
   flush-to-zero and denormals-are-zero off: no `fesetround`, no `_mm_setcsr`, no loaded library that
   changes them; library issue L110), for two reasons
   ([evidence/host-round21](../../evidence/host-round21/README.md) H1–H3):
   - `-O` removes the Python `assert`s by which Triton's front end rejects some kernels on the host
     (`primitives/triton-3.8.0-code-generator.py:497, 872, 886, 1020, 1117-1121`), and such a kernel
     then compiles and runs: a loop body that gives a carried name holding `tl.float16` the dtype
     `tl.float32`, rejected under `python -I` ([SEMANTICS.md](../SEMANTICS.md), "Names after a loop"),
     ran under `-O`, `-OO` and `PYTHONOPTIMIZE=1` and stored `1.0` (H1);
   - the host builds the constants (`FloatV.litF32`, `litF16`, `litBF16`), converts a `float` argument
     (`FloatV.hostF32`) and computes compile-time float arithmetic in the process's floating-point
     environment, whose rounding mode and MXCSR flush-to-zero and denormals-are-zero bits host code can
     change (`fesetround`, `_mm_setcsr`, or a shared library built with `-ffast-math`, whose start-up
     code sets both bits): after `fesetround(FE_TOWARDZERO)`, `tl.full([1], 0.1, tl.float32)` and the
     argument `0.1` stored `0x3dcccccc`, not `0x3dcccccd`, and the constexprs `0.1 + 0.2` gave
     `0x3fd3333333333333`, not Python's `0x3fd3333333333334` (H2); with both bits set, by `_mm_setcsr`
     or by loading such a library, `tl.full([1], 1e-40, tl.float32)` stored `0` (H3).
- **10.** **Triton and the kernel's module as installed and written** (library issue L104).
   - (a) The process imports Triton 3.8.0's installed package, unmodified: its files as installed, no
     other `triton` earlier on `sys.path`, and no substitute put in `sys.modules`.
   - (b) No code but Triton's own (the kernel's module, another module, a library it imports), at
     any time before the launch ends, changes an attribute of a Triton module (`tl.exp = tl.log`,
     `tl.arange = …`), of a class Triton defines or of an object of one, or of the kernel's or a
     helper's `JITFunction` (its `fn`, `src`, `noinline`), whether by assignment, by `setattr` or
     `delattr`, or by a call made to change it (a `JITFunction`'s `_unsafe_update_src`,
     `primitives/triton-3.8.0-jit.py:561-569`; `triton.runtime.driver.set_active`; `importlib.reload`
     of a Triton module),
   - (c) other than through Triton's API as the in-scope tutorials use it: launches,
     `kernel.warmup(…)`, `triton.autotune` with its `triton.Config`s and hooks, the `JITFunction`'s
     `debug` (point 4), `triton.set_allocator` (point 5; it sets Triton's `_allocator`,
     `primitives/triton-3.8.0-allocation.py`), the knobs point 2 allows, and the fields of a host
     `TensorDescriptor` the launch is given (a config's `pre_hook` may set its `block_shape`, as
     tutorials 06 and 09 do, "Which launch"; point 8 still bars `round_f32_to_tf32=True`); every
     other point still holds of these (point 3 bars a hook an API call installs).
   - (d) No code binds a name of the kernel's module other than as the module's own statements bind
     it (`kmod.double = kmod.triple` in another module, an assignment into `kernel.fn.__globals__`).

   The front end compiles what these objects and names hold when it compiles the kernel, so such a
   change gives the launch another computation than its source's: `3·x` for the source's `2·x`,
   `log x` for `exp x`, `5·x` after `_unsafe_update_src` for `2·x`
   ([evidence/host-round20](../../evidence/host-round20/README.md) M1–M4).
- **11.** **The kernel is new** (library issues L98, L103, L109, L116, L117). The code the claimed
   launch runs is the code Triton 3.8.0 compiles for that launch, from the kernel's own source, the
   launch's own argument values, compile-time and runtime, as the claim's instance gives them (the
   compile-time ones are its `tl.constexpr` arguments, and an integer `1` or `None` passed to a
   parameter without annotation, which `primitives/triton-3.8.0-specialize.cc` makes compile-time,
   [SEMANTICS.md](../SEMANTICS.md), "Scalar arguments"), and the settings the other conditions
   give. Code compiled earlier counts when it is that code: code that a cache returns, the
   `JITFunction`'s cache in the process (`primitives/triton-3.8.0-jit.py:592-614`) or the cache
   directory (`TRITON_CACHE_DIR`, else `.triton/cache` under `TRITON_HOME` or the home directory,
   `primitives/triton-3.8.0-knobs.py:343-354`, `primitives/triton-3.8.0-compiler.py:244-270`), for
   example for a repeated launch of the same instance, and the code a `CompiledKernel` runs within
   point 6. TVA does not verify Triton's caches (the user's decisions, 2026-10-29, under L36: "To
   make the verifier easy, we do not verify triton cache." and "we assume the kernel is new.").

   A runtime argument's value is part of the claim's instance, and the prover's conditions constrain
   it, for example `n ≤ x.len` in `Inst.ok` for a length `n_elements` (the user's decision,
   2026-10-29, under L36: "for the viriable n_elements, its value should be merged into the
   assumptions of the prover. e.g., x.len >= n_elements, etc."). So code compiled for another value
   of a runtime argument counts only when it is the code compiled for the launch's own: `n = 1001` on
   code compiled for `n = 1024` wrote three elements past `n`
   ([evidence/compiled-launch](../../evidence/compiled-launch/README.md) K2, point 6).

   The assumption is needed. The caches find compiled code by keys that equate some different
   compile-time values (in the process, values Python's `==` equates and every NaN; on disk, the
   `str`s of the values joined with `-`, `primitives/triton-3.8.0-compiler.py:71-76`) and that record
   no knob set from Python, no `TRITON_KERNEL_OVERRIDE`, no hook, no change to Triton in memory and no
   state of the host process (`primitives/triton-3.8.0-GetEnv.h:15-57`). So code taken from them may
   have been compiled for other values or under other settings, and may not be that code. The record:
   [evidence/compile-keys](../../evidence/compile-keys/README.md) K1–K4 (`C = 1.0` after `C = 1` ran
   the integer kernel; `MODE = "1"` after `MODE = 1` ran `MODE = 1`'s code; `C = -float('nan')` after
   `C = float('nan')` stored `0x7fc00000`); [evidence/host-round21](../../evidence/host-round21/README.md)
   K1–K4 (`C = 2.0` after `C = 2`, also with a fresh cache directory set in the process between them;
   an autotuner's chosen config ran the code of its benchmark run of another) and X1–X5 (a process with
   Triton at its defaults, on a cache directory an earlier process had used, ran that process's code:
   `log x` for `tl.exp(x)` after `tl.exp = tl.log` there, the `ieee` `tl.dot` after a Python-set
   `fp32_default`, a constant built after `fesetround`, a kernel compiled under `-O`); and the library
   reviews of round 20 ([lr-20a](../library-review/round-20/lr-20a/report.md),
   [lr-20b](../library-review/round-20/lr-20b/report.md),
   [lr-20c](../library-review/round-20/lr-20c/report.md),
   [lr-20d](../library-review/round-20/lr-20d/report.md),
   [lr-20f](../library-review/round-20/lr-20f/report.md)). Calls of a helper are the front end's
   matter, within one compile, and the supported subset's ([SEMANTICS.md](../SEMANTICS.md),
   "Compile-time values Triton keys apart").

## Which launch

The rule is about one direct launch of the kernel's `@triton.jit` function (`kernel[grid](…)`, or
the launch of a `CompiledKernel` within point 6), made through Triton outside CUDA graph capture
(scope 9), with the grid, the arguments, the compile-time values and the options the launch call
gives. Its `init` is the memory when it starts; what the caller's memory was before is the host's
part, outside the claim, and a claim about the caller's buffers before the call must be stated on
the host (it is not a kernel claim).

**The autotuner is not verified** (library issue L116; the user's decisions, 2026-10-29, above).
TVA verifies the kernel itself. `@triton.autotune` is outside the supported subset of a kernel's
source ([SEMANTICS.md](../SEMANTICS.md), "The supported subset"; [exclusions.md](../exclusions.md)),
and the rule covers no call of an autotuned kernel. Such a call is host code
(`primitives/triton-3.8.0-autotuner.py:217-282`): on a key it has not timed, it prunes the configs
(`prune_configs_by`), runs the kernel many times on the caller's buffers to time each config left
(`_bench`, 133-173, through `do_bench`), and runs `reset_to_zero`, `restore_value`, the autotuner's
`pre_hook` and `post_hook` and each config's `pre_hook`, which write those buffers; then it launches
the config it chose, after that config's `pre_hook`. In
[evidence/autotune-launch](../../evidence/autotune-launch/README.md) an in-place `+1` ran about
4000 times on the first call, and `reset_to_zero` made the final launch start from zeros; whether a
call benchmarks depends on the process's cache, the number of configs left after pruning, and the
disk cache. Its benchmark runs, its key, pruning, `reset_to_zero` and `restore_value`, `do_bench`
and its choice are outside every claim.

**A kernel a tutorial autotunes** is covered through one config, which the user and the coordinator
pick when they set up the kernel's folder, before any prover starts
([REVIEW_PROTOCOL.md](../REVIEW_PROTOCOL.md) §3). The folder's source is the tutorial's with the
`@triton.autotune` decorator removed, and its launch passes the config's meta-parameters and launch
options (`num_warps`, `num_stages`, …), as the autotuner's own launch passes them
(`Config.all_kwargs`, `primitives/triton-3.8.0-autotuner.py:273-280, 365-377`). Where the config has
a `pre_hook`, the host code does what the hook does before the launch: tutorials 06 and 09 set a
host `TensorDescriptor`'s `block_shape` there. That launch is a direct launch, which the rule
covers as any other. The prover proves the kernel as given in `source/`, and the kernel reviewer
judges that kernel, not the choice of config.

[evidence/env-knobs](../../evidence/env-knobs/README.md) shows `TRITON_F32_DEFAULT` and
`TRITON_DEFAULT_FP_FUSION` at work.

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
- **Launches that are not well behaved, or not valid.** For those the rule claims nothing. In
  particular, a kernel whose program reads back what it wrote, writes the same bytes twice, or
  updates an input in place is not covered: on the device a program does not see its own global
  accesses in source order ([evidence/intra-program](../../evidence/intra-program/README.md)). A
  proof must show the launch is well behaved, for example with `TVA.independent`, and valid
  (`meets_of_faithful` asks for `(launch d).Valid`; `Launch.valid_grid1` proves it for a
  one-dimensional grid).
- **Launches with programmatic dependent launch** (`launch_pdl=True`, `Launch.pdl`): the kernel
  may start before the previous kernel on the stream has finished writing, so its programs may
  read memory other than `init`. `Launch.Valid` asks for `pdl = false`. The other side, a next
  launch with `launch_pdl=True` after a kernel that calls `gdc_launch_dependents()`, is outside the
  scope (point 1).
- **Anything outside the scope** (above). Part (A), checked: a compile option `Launch` does not
  record (at the launch, at `warmup`, in `triton.compile` or in a config) or `@triton.heuristics`, a
  descriptor scratch that holds or shares the launch's bytes, a precompiled kernel given arguments
  outside its specialisation or compiled from an IR file, a host descriptor with
  `round_f32_to_tf32=True`, a call made during a CUDA graph capture or a replay of the graph. Part
  (B), assumed not to happen and not checked: another writer of the launch's bytes during the
  launch, a knob not at its default, a hook, a `debug` attribute set from outside or a `preload`, a
  target other than the H200 or a host other than x86-64 with CPython 3.13 (or one run with `-O` or
  `-OO` or with `PYTHONOPTIMIZE` set, or in another floating-point environment), Triton or the
  kernel's module changed by other code, and code other than the code Triton compiles for the
  launch's own argument values and settings (Triton's caches may return such code).
- **The autotuner and the host code around the launch**: a call of a `@triton.autotune`d kernel,
  with its benchmark runs, key, pruning, `reset_to_zero` and `restore_value`, hooks, `do_bench` and
  choice, which TVA does not verify (above, "Which launch"), and what anything but the launch does
  to the bytes `init` does not give (the launch itself writes none of them but the descriptor
  scratch, "Memory").
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

The reviewer checks five things:

- `model d` is the kernel's source, with every implicit conversion, broadcast and mask, and the
  source stays inside the supported subset ([SEMANTICS.md](../SEMANTICS.md), "The supported
  subset");
- `launch d` is the host's launch: the grid formula, `enable_fp_fusion`, `num_ctas`, `launch_pdl`
  (not set), and the source's block shapes in `blocks` (the bundle proves `Launch.Valid`, so the
  grid has fewer than `2^31` programs on every covered instance);
- the kernel and its launches are within part (A) of the scope, point by point ("Scope" above):
  - 4. no option other than the eight listed, at the launch, at `warmup`, in `triton.compile`'s
    options or in a `triton.Config`, and no `@triton.heuristics`; the decorator's `debug` is
    `Launch.debug`;
  - 5. when the kernel builds descriptors, an allocator whose buffer holds none of the claim's buffers
    and that nothing else writes during the launch;
  - 6. for a `CompiledKernel`, the kernel's own source (an `ASTSource`, not an IR file) and arguments
    in its specialisation (the dtypes, each pointer's alignment, each integer's class, each host
    descriptor's dtype and block shape);
  - 8. no host descriptor with `round_f32_to_tf32=True`;
  - 9. no CUDA graph capture: no call made while a CUDA stream capture is in progress, on the
    capturing stream or any other, and no replay;

  and that the launch is a direct launch of the `@triton.jit` function: a source that has
  `@triton.autotune` is outside the supported subset ("Which launch"). Part (B) (points 1, 2, 3, the
  environment's part of 4, 7, 10 and 11) is assumed: the reviewer does not read host code for it,
  and files a finding only when the claim's own launch or kernel source visibly breaks it;
- `d.ok` covers the instances the claim is about: a runtime argument's value is part of the
  instance, and `d.ok` constrains it (for example `n ≤ x.len` for a length `n_elements`; point 11);
- the bundle proves `Claim.Consistent Assumptions` (`exists_total_faithful` gives the behaviour
  for this rule), and `Claim.Satisfiable Inst.ok Pre`, that some covered instance and memory exist
  (library issue L71).

## Consistency

The rule holds of some total behaviour for every model and launch:
`TVA.exists_total_faithful` (`lean/TVA/Lemmas/Consistency.lean`). The behaviour follows the
model where the launch is well behaved and faults elsewhere. It is total because a
well-behaved launch always ends normally or runs forever (`WellBehaved.outcome`): no state of
the semantics is stuck (`Step.progress`).
