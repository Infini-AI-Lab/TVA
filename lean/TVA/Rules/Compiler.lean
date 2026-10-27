import TVA.Claim
import TVA.Sem.Launch

/-!
# Rule `Compiler.Faithful`: the compiled kernel behaves as its Triton program

Page: `docs/rules/compiler-faithful.md`.

This is the trust the project places in the Triton compiler and its documentation (user,
2026-10-26: "we trust triton compiler and documentations"). If a launch of the kernel's model is
well behaved, so that no interleaving reaches undefined behaviour or a data race between
programs, then every device outcome is an outcome of the model's semantics. A normal end has the
same final memory, the launch never faults, and it hangs only if some interleaving runs forever.

The rule is about a *model*, and the model is the prover's transcription of the source. Whether
the model is the source, with every implicit broadcast, conversion and mask, is a separate
reviewed claim (`MODEL_MISMATCH`). The accuracy of the operations the platform `P` decides is
not part of this rule but of the `Rules.Num` rules about `P`.
-/
namespace TVA.Rules.Compiler

open TVA.Claim

/-- **Rule `Compiler.Faithful`.** The device's behaviour `K` on one instance (launch `L`) is
included in the semantics of the model `k` on platform `P`, for every initial memory from which
that launch is well behaved, when the kernel is compiled without `debug`, the launch is valid
(`Launch.Valid`, which includes fewer than `2^31` programs, since Triton's launcher skips larger
grids without an error, and no programmatic dependent launch), and Triton's environment is at its
defaults. A launch that Triton or the driver reports to the host with an error before any program
runs (shared memory, registers, threads, other compile or launch limits) has no outcome
(`TVA.Claim.Behaviour`), so the rule says nothing about whether the kernel compiles.

*Triton's environment at its defaults* means: none of the environment variables Triton 3.8.0
reads to compile or launch a kernel is set (`triton-3.8.0-knobs.py:343-513`,
`triton-3.8.0-GetEnv.h:15-57`), except those that only name Triton's home, cache and dump
directories, only dump, print or time, or only change debug information: `TRITON_HOME`,
`TRITON_CACHE_DIR`, `TRITON_DUMP_DIR`, `TRITON_KERNEL_DUMP`, `TRITON_ALWAYS_COMPILE`,
`TRITON_PRINT_AUTOTUNING`, `TRITON_DUMP_PTXAS_LOG`, `MLIR_ENABLE_DUMP`, `MLIR_DUMP_PATH`,
`LLVM_IR_ENABLE_DUMP`, `NVPTX_ENABLE_DUMP`, `MLIR_ENABLE_TIMING`, `LLVM_ENABLE_TIMING`,
`TRITON_DISABLE_LINE_INFO`, `USE_IR_LOC`. In particular `TRITON_F32_DEFAULT` (the `input_precision`
of a `tl.dot` that gives none), `TRITON_DEFAULT_FP_FUSION` (the default of `enable_fp_fusion`),
`TRITON_DEBUG`, `TRITON_INTERPRET`, `TRITON_KERNEL_OVERRIDE`, `TRITON_OVERRIDE_ARCH`,
`TRITON_LIBDEVICE_PATH`, `TRITON_PTXAS_PATH`, `PTXAS_OPTIONS`, `DISABLE_PTXAS_OPT`,
`DISABLE_LLVM_OPT`, `TRITON_CACHE_MANAGER` and `TRITON_REMOTE_CACHE_BACKEND` are unset. -/
def Faithful (P : Platform) (k : Kernel) (L : Launch) (K : Mem → Outcome → Prop) : Prop :=
  L.debug = false → L.Valid → ∀ init, WellBehaved (L.env P) k L init → ∀ o, K init o →
    match o with
    | .ok fin => Outcomes (L.env P) k L init (.ok fin)
    | .fault => False
    | .hang => Outcomes (L.env P) k L init .hang

end TVA.Rules.Compiler
