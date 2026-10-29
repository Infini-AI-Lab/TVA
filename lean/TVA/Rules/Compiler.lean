import TVA.Claim
import TVA.Sem.Launch

/-!
# Rule `Compiler.Faithful`: the compiled kernel behaves as its Triton program

Page: `docs/rules/compiler-faithful.md`.

This is the trust the project places in the Triton compiler and its documentation (user,
2026-10-26: "we trust triton compiler and documentations"), on TVA's target, the NVIDIA H200
(sm_90). If a launch of the kernel's model is
well behaved, so that no interleaving reaches undefined behaviour, a data race between programs,
or a conflict within one program (two accesses of a program to the same bytes, at least one
writing, unless both are atomics on the same bytes; library issue L15), then every device outcome
is an outcome of the model's semantics, in which a program's atomics may be performed in any order
their dependences allow. A normal end has the
same final memory, the launch never faults, and it hangs only if some interleaving runs forever.

The rule is about a *model*, and the model is the prover's transcription of the source. Whether
the model is the source, with every implicit broadcast, conversion and mask, is a separate
reviewed claim (`MODEL_MISMATCH`). The accuracy of the operations the platform `P` decides is
not part of this rule but of the `Rules.Num` rules about `P`.
-/
namespace TVA.Rules.Compiler

open TVA.Claim

/-- **Rule `Compiler.Faithful`.** The device's behaviour `K` on one instance (launch `L`, on an
NVIDIA H200, scope 7) is included in the semantics of the model `k` on platform `P`, for every
initial memory from which that launch is well behaved (`WellBehaved`: no undefined behaviour, no
data race between programs, and no conflict within a program, across loop iterations included),
when the kernel is compiled without `debug` and the launch is valid (`Launch.Valid`, which includes
fewer than `2^31` programs: Triton's launcher skips a grid of `2^31` to `2^32` programs without an
error, running no program, and a larger grid may run, `2^32 + 1` programs wrapping to one; and no
programmatic dependent launch). A launch that Triton or the driver reports to the host with an
error before any program runs (shared memory, registers, threads, other compile or launch limits)
has no outcome (`TVA.Claim.Behaviour`), so the rule says nothing about whether the kernel compiles.

**Memory.** `init` gives the launch its bytes: the bytes present in it (`Mem`, a partial map), the
buffers of the claim's `Pre`. The outcome `ok fin` gives the same bytes, with their contents when
the launch ends; the model may not access any other byte. A valid launch within the scope, from an
`init` from which it is well behaved, writes no byte of global memory other than those `init`
gives, except the scratch of the descriptors the kernel builds (scope 5): the device writes what the
model writes, and the launcher's only other buffers are that scratch and those of instrumentation
(`instrumentation_mode`: a profile scratch, a sanitizer's state), which scopes 2 and 4 exclude
(`triton-3.8.0-nvidia-driver.py:280-333`;
`evidence/launch-frame`: no byte outside the written buffers and the scratch changed; library issue
L73). Bytes `init` does not give are no part of the outcome, so `Post` cannot speak of them; this
frame is part of the rule, and what anything else does to them is outside the claim.

**Scope.** The rule covers a launch only under the conditions below, in two parts (library issues
L5, L17, L18, L19, L47, L49, L72, L74, L76, L77, L78, L91, L103, L104, L105, L107, L109, L110). The
user's decisions, 2026-10-29: "To make the verifier easy, we do not verify triton cache.", "we
assume the kernel is new." and "So basically, we want to verify the kernel itself, rather than
triton installation or other code programmed by users." Each condition keeps its number of earlier
rounds, so that a citation of "scope `n`" still names it.

**(A) The kernel and its launches: checked.** What the claim's instances describe and the kernel
reviewer checks against the source: the `@triton.jit` function, its helpers and its decorators as
the kernel's source writes them (`triton.autotune` with its `triton.Config`s and their hooks,
`triton.heuristics`); the launch call (the grid, the arguments, the compile-time values and the
launch options the call passes); and what the kernel needs from its host to run at all (an
allocator for descriptors, the host `TensorDescriptor`s the launch is given). The claim's text
states these conditions:

- **4.** *Only the launch options `Launch` records.* No compile or launch option of the NVIDIA backend
   (`CUDAOptions`, `triton-3.8.0-nvidia-compiler.py:108-135`; the launcher accepts every field as a
   keyword, `triton-3.8.0-jit.py:713-715`) is given to the call that compiles the kernel (a JIT
   launch, `kernel.warmup(…)`, which takes the same keywords, `triton-3.8.0-jit.py:362-363`, or
   `triton.compile`'s `options`, `triton-3.8.0-compiler.py:226-243`) or in an autotune
   `triton.Config` (`evidence/compiled-options` O1, O2, O4: `enable_reflect_ftz=False` given at a
   launch, at `warmup` or to `triton.compile` changed `tl.math.exp2(-130)` from `0` to `2^-130`),
   except `enable_fp_fusion`, `debug`, `num_ctas` and
   `launch_pdl` (`Launch.fusion`, `debug`, `numCtas`, `pdl`) and the four that do not change the
   meaning: `num_warps`, `num_stages`, `maxnreg` and `launch_cooperative_grid`. So `ir_override`,
   `allowed_dot_input_precisions`, `default_dot_input_precision`, `enable_reflect_ftz`,
   `extern_libs`, `ptx_options`, `ptx_version`, `arch`, `warp_size`, `supported_fp8_dtypes`,
   `deprecated_fp8_dot_operand_dtypes`, `max_num_imprecise_acc_default`, `sanitize_overflow`,
   `instrumentation_mode` and `backend_name` are not given (`evidence/launch-options`). And the
   kernel is not under `@triton.heuristics`, which sets launch keywords before the JIT reads them,
   compile options included, replacing the launch's and a config's of the same name
   (`triton-3.8.0-autotuner.py:473-483`; library issue L78). The
   decorator's `@triton.jit(debug=True)` sets the `JITFunction`'s attribute `debug`, which a launch
   that gives no `debug` reads (`triton-3.8.0-jit.py:727`), so it makes the launch's `debug` true
   (`Launch.debug`).
- **5.** *The scratch of descriptors built in the kernel is the launch's own.* A kernel that builds
   tensor descriptors (`tl.make_tensor_descriptor`) needs an allocator (`triton.set_allocator`);
   without one the launch is rejected on the host. The launch writes the buffer the allocator
   returns for it, each program its own `global_scratch_size` bytes (128 per descriptor), which the
   model does not access (`tl.make_tensor_descriptor` is pure). The rule covers such a launch only
   when that buffer holds no byte `init` gives the launch and nothing else writes it until the
   launch ends (`evidence/desc-scratch`: a scratch inside an input overwrote 1024 of its elements).
- **6.** *A precompiled kernel gets arguments of its specialisation.* A launch through a
   `CompiledKernel` (`kernel.warmup(…)` or `triton.compile`, then `compiled[grid](…)`, as tutorial
   02 does) runs the code compiled for the warm-up's arguments (or `triton.compile`'s signature and
   attributes), and the launcher checks nothing against it. The rule covers it, as the launch
   `Launch` describes with the warm-up's constexprs, only when every argument is in the
   specialisation it was compiled for: the same dtypes; each pointer 16-byte aligned exactly when
   the warm-up's was; each integer in the same class (equal to 1, which was a constexpr; a multiple
   of 16 or not; in `int32`, `int64` or `uint64` range); each host tensor descriptor
   (`TensorDescriptor`) with the warm-up's dtype and block shape, its specialisation
   (`triton-3.8.0-specialize.cc:165-226`; the launcher builds the TMA descriptor with the compiled
   block shape, `triton-3.8.0-nvidia-driver.py:209-264`, so a `[16, 16]` descriptor given to a kernel
   compiled for `[32, 32]` had 1024 elements copied, `evidence/compiled-options` O6); and a constexpr
   position is ignored (`evidence/compiled-launch`: outside the specialisation a launch faulted,
   wrote past `n`, or used a constexpr's compiled value). A kernel `triton.compile` built is covered
   only when its source is an `ASTSource` of the kernel's `@triton.jit` function: given a file path,
   `triton.compile` compiles that IR file (a `.ttir`, `.ttgir` or `.ptx` file, `IRSource`,
   `triton-3.8.0-compiler.py:87-112, 233-240`), which replaces the source, as `ir_override` does
   (library issue L77).
- **8.** *Host descriptors load what memory holds.* No host tensor descriptor (`TensorDescriptor`) the
   launch is given has `round_f32_to_tf32=True`: with it, the launcher builds a TMA descriptor whose
   loads round `float32` to tf32 (`triton-3.8.0-nvidia-driver.py:231-232`; `evidence/desc-tf32`:
   `1 + 2^-23` was loaded as `1.0`), which `tl.desc_load` does not model (library issue L76).
- **9.** *No CUDA graph capture.* The call that launches the kernel is not made while a CUDA stream
   capture is in progress, on any stream (`torch.cuda.graph(…)`, `torch.cuda.CUDAGraph.capture_begin`,
   `triton.testing.do_bench_cudagraph`, a `cudaStreamBeginCapture`), and the launch is not a replay
   of a captured graph. A call on the capturing stream (the current stream inside
   `torch.cuda.graph(…)`) is captured: it returns without an error, compiling the kernel first if
   needed, and runs no program, and the programs run at each `graph.replay()`, with the arguments the
   call captured, on the memory at the replay. A call on another stream while a capture is in
   progress is not captured: it runs at once, as any launch, and a replay does not run it
   (`evidence/host-round20` G0–G3, in the three capture modes). The rule covers no call made while a
   capture is in progress, captured or not, and no replay (library issue L91; excluded by the user's
   decision, 2026-10-28, `docs/exclusions.md`).

**(B) The environment: assumed, not verified.** TVA verifies the kernel, not the Triton
installation or other code users write. These conditions are assumptions of the rule: the claim's
text states them, and the gate, the prover and the kernel reviewer do not check them. A claim's own
launch or kernel source that visibly breaks one (the source sets a knob, or rebinds a Triton name)
is still a finding:

- **1.** *Nothing else writes the launch's bytes.* From the launch's start until it ends, nothing but
   the launch writes a byte `init` gives it: not the host, not another kernel on any stream or
   device, and not a later kernel that programmatic dependent launch starts early. A kernel whose
   programs call `gdc_launch_dependents()` lets the next launch on its stream, when that launch
   has `launch_pdl=True`, start while it still runs; if that launch writes such a byte (it wrote
   the kernel's input in 20 of 20 runs, `evidence/pdl` part D), the case is outside the rule.
   Writes to other bytes do not matter, whenever they happen: the device's stores are
   byte-granular (`evidence/other-writers`: concurrent writes to the neighbouring bytes, in the
   same 4-byte words, changed nothing in 20 of 20 overlapping runs, and a concurrent writer to the
   launch's own bytes corrupted it).
- **2.** *Triton's knobs are at their defaults.* Every knob of `triton.knobs`
   (`triton-3.8.0-knobs.py:320-513`) and every environment variable Triton's C++ reads
   (`triton-3.8.0-GetEnv.h:15-57`) has its default, however it would be set: by the environment
   variable, or by assigning the knob from Python (`triton.knobs.language.fp32_default = "ieee"`,
   with or without `triton.knobs.propagate_env`). The exceptions only name Triton's home, cache and
   dump directories, only dump, print or time, or only change debug information: `TRITON_HOME`,
   `TRITON_CACHE_DIR`, `TRITON_DUMP_DIR`, `TRITON_KERNEL_DUMP`, `TRITON_ALWAYS_COMPILE`,
   `TRITON_PRINT_AUTOTUNING`, `TRITON_DUMP_PTXAS_LOG`, `MLIR_ENABLE_DUMP`, `MLIR_DUMP_PATH`,
   `LLVM_IR_ENABLE_DUMP`, `NVPTX_ENABLE_DUMP`, `MLIR_ENABLE_TIMING`, `LLVM_ENABLE_TIMING`,
   `TRITON_DISABLE_LINE_INFO`, `USE_IR_LOC`, and the knobs they set. In particular
   `TRITON_F32_DEFAULT` / `knobs.language.fp32_default` (the `input_precision` of a `tl.dot` that
   gives none), `TRITON_DEFAULT_FP_FUSION` / `knobs.language.default_fp_fusion` (the default of
   `enable_fp_fusion`), `TRITON_DEBUG` / `knobs.runtime.debug`, `TRITON_INTERPRET`,
   `TRITON_KERNEL_OVERRIDE`, `TRITON_OVERRIDE_ARCH`, `TRITON_LIBDEVICE_PATH`, `TRITON_PTXAS_PATH`,
   `PTXAS_OPTIONS`, `DISABLE_PTXAS_OPT`, `DISABLE_LLVM_OPT`, `TRITON_CACHE_MANAGER`,
   `TRITON_REMOTE_CACHE_BACKEND` and `knobs.compilation.instrumentation_mode` are at their
   defaults.
- **3.** *No hook is installed.* None of `knobs.runtime.launch_enter_hook`, `launch_exit_hook`,
   `kernel_load_start_hook`, `kernel_load_end_hook`, `kernel_unload_hook`, `jit_cache_hook` (which
   can skip the launch), `jit_post_compile_hook`, `add_stages_inspection_hook` (which can rewrite
   any compiler stage), `knobs.compilation.listener` and `knobs.autotuning.listener`, and no
   pre-run hook of the kernel's `JITFunction` (`triton-3.8.0-knobs.py:470-492`,
   `triton-3.8.0-jit.py:735-736`, `:880-882`).
- **4.** *(the environment's part) No `debug` attribute set from outside and no `preload`* (library
   issue L105). No code but the kernel's source sets the kernel's `JITFunction` attribute `debug`
   (`kernel.debug = True` from Python makes the launch's `debug` true, `Launch.debug`;
   `evidence/host-round20` D1: with the attribute set, an `int32` overflow trapped), and no
   `JITFunction.preload` is called on the kernel in the process: it compiles the kernel with the
   options its data names and stores it under the cache key its data names
   (`triton-3.8.0-jit.py:823-875`), so that a later launch with that key runs that code (P1: after a
   `preload` of code compiled with `enable_reflect_ftz=False`, a launch with no keyword gave
   `tl.math.exp2(-130)` as `2^-130`, not `0`).
- **7.** *The target.* The launch runs on an NVIDIA H200 (sm_90), TVA's only target
   (`docs/exclusions.md`), with code Triton compiles for the device it runs on: no `arch` option and
   no `TRITON_OVERRIDE_ARCH` (scopes 2 and 4), and no `triton.compile` target other than the
   device's (one for sm_80 was rejected on the host, `evidence/compiled-options` O5). The model's
   defaults are the H200's (`DotCfg`: `tf32` when `input_precision` is omitted, `maxImpreciseAcc =
   2^30` for an fp8 × fp8 `tl.dot`), and the device facts a claim cites are those of
   `Rules.Num.Sm90`. The host that runs Triton is an x86-64 machine with CPython 3.13, as TVA's
   evidence ran (library issue L107): some values `docs/SEMANTICS.md` states are the host's, such as
   the bits of a NaN passed as a `tl.float16` argument (CPython's `PyFloat_Pack2`: under 3.13 the
   payload is dropped, `0x7e00` or `0xfe00`; CPython 3.14 keeps its top bits,
   `docs/library-review/round-19/lr-19b/work/host_pack2.py`) and the sign of Python's
   `float('inf') - float('inf')` (x86-64's default NaN, `0xfff8000000000000`). The host process is
   CPython 3.13 run without `-O` or `-OO` and with `PYTHONOPTIMIZE` unset, in the default
   floating-point environment (round to nearest even; flush-to-zero and denormals-are-zero off: no
   `fesetround`, no `_mm_setcsr`, no loaded library that changes them; library issue L110). `-O`
   removes the Python `assert`s by which Triton's front end rejects some kernels on the host
   (`triton-3.8.0-code-generator.py:497, 872, 886, 1020, 1117-1121`), and such a kernel then runs
   (`evidence/host-round21` H1: a loop body that gives a carried name holding `tl.float16` the dtype
   `tl.float32`, rejected under `python -I`, ran under `-O`, `-OO` and `PYTHONOPTIMIZE=1` and stored
   `1.0`). The host computes the constants Triton builds, the `float32` it passes for a `float`
   argument (`FloatV.litF32`, `FloatV.hostF32`, `FloatV.litBF16`) and compile-time float arithmetic
   in the process's floating-point environment (H2: after `fesetround(FE_TOWARDZERO)`,
   `tl.full([1], 0.1, tl.float32)` and the argument `0.1` stored `0x3dcccccc`, not `0x3dcccccd`, and
   the constexprs `0.1 + 0.2` gave `0x3fd3333333333333`; H3: with flush-to-zero and
   denormals-are-zero set, by `_mm_setcsr` or by loading a shared library built with `-ffast-math`,
   `tl.full([1], 1e-40, tl.float32)` stored `0`).
- **10.** *Triton and the kernel's module as installed and written* (library issue L104). (a) The process
   imports Triton 3.8.0's installed package, unmodified: its files as installed, no other `triton`
   earlier on `sys.path`, and no substitute put in `sys.modules`. (b) No code but Triton's own (the
   kernel's module, another module, a library it imports), at any time before the launch ends,
   changes an attribute of a Triton module (`tl.exp = tl.log`, `tl.arange = …`), of a class Triton
   defines or of an object of one, or of the kernel's or a helper's `JITFunction` (its `fn`, `src`,
   `noinline`), whether by assignment, by `setattr` or `delattr`, or by a call made to change it (a
   `JITFunction`'s `_unsafe_update_src`, `triton.runtime.driver.set_active`, `importlib.reload` of a
   Triton module), (c) other than through Triton's API as the in-scope tutorials use it: launches,
   `kernel.warmup(…)`, `triton.autotune` with its `triton.Config`s and hooks, the `JITFunction`'s
   `debug` (scope 4), `triton.set_allocator` (scope 5; it sets Triton's `_allocator`,
   `triton-3.8.0-allocation.py`), the knobs scope 2 allows, and the fields of a host
   `TensorDescriptor` the launch is given (a config's `pre_hook` may set its `block_shape`, as
   tutorials 06 and 09 do, "Which launch"; scope 8 still bars `round_f32_to_tf32=True`); every other
   scope still holds of these. (d) No code binds a name of the kernel's module other than as the
   module's own statements bind it (`kmod.double = kmod.triple` in another module, an assignment into
   `kernel.fn.__globals__`). The front end compiles what these objects and names hold when it
   compiles the kernel, so such a change gives the launch another computation than its source's
   (`evidence/host-round20` M1–M4: `3·x` for the source's `2·x`, `log x` for `exp x`, `5·x` after
   `_unsafe_update_src` for `2·x`).
- **11.** *The kernel is new* (library issues L98, L103, L109). The code the launch runs was compiled
   for that launch: from the kernel's own source, its compile-time values (its `tl.constexpr`
   arguments, an autotune config's values, and an integer `1` or `None` passed to a parameter
   without annotation, which `triton-3.8.0-specialize.cc` makes compile-time) and the settings the
   other conditions give. It is not code compiled earlier, for other values or under other
   settings, and taken from Triton's caches, the `JITFunction`'s cache in the process
   (`triton-3.8.0-jit.py:592-614`) or the cache directory (`triton-3.8.0-compiler.py:244-270`). Code
   compiled earlier for the same source, values and settings (an autotuner's benchmark run of the
   chosen config, an earlier launch of the same instance) counts as compiled for it. TVA does not
   verify Triton's caches. The assumption is needed because the caches' keys equate some different
   compile-time values and record no knob set from Python, no `TRITON_KERNEL_OVERRIDE`, no hook, no
   change to Triton in memory and no state of the host process, so code taken from them may have
   been compiled for other values or under other settings (`evidence/compile-keys`,
   `evidence/host-round21` K1–K4 and X1–X5, and the library reviews of round 20;
   `docs/rules/compiler-faithful.md`, "Scope" 11).

**Which launch.** The rule is about one launch of the kernel, made through Triton outside CUDA
graph capture (scope 9). A call of a `@triton.autotune`d
kernel is host code: on a new key it runs the kernel many times on the caller's buffers to time
each config, and runs `reset_to_zero`, `restore_value`, the autotuner's hooks and each config's
`pre_hook`, which write those buffers; then it makes one launch with the chosen config, after that
config's `pre_hook` (`triton-3.8.0-autotuner.py:217-282`, `evidence/autotune-launch`). The launch
the rule covers is that last one, with the arguments the last hook leaves (a hook may change them:
tutorials 06 and 09 set a host `TensorDescriptor`'s `block_shape` in a config's `pre_hook`), and its
`init` is the memory after the last hook. What the caller's memory was before the call is the host's
part, outside the claim. -/
def Faithful (P : Platform) (k : Kernel) (L : Launch) (K : Mem → Outcome → Prop) : Prop :=
  L.debug = false → L.Valid → ∀ init, WellBehaved (L.env P) k L init → ∀ o, K init o →
    match o with
    | .ok fin => Outcomes (L.env P) k L init (.ok fin)
    | .fault => False
    | .hang => Outcomes (L.env P) k L init .hang

end TVA.Rules.Compiler
