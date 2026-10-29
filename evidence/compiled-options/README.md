# Options and host descriptors of a precompiled kernel, and its target

**Question.** `Compiler.Faithful`'s scope 4 excluded compile options given "as a launch keyword" or
"in an autotune `triton.Config`". A `CompiledKernel` (tutorial 02) gets its options where it is
compiled, at `kernel.warmup(…)` or `triton.compile(…, options=…)`, and its launch takes none. Do
options given there reach its launches? Scope 6 did not say how a host tensor descriptor is
specialised. And which target does `triton.compile` build for? Library issues L72 (raised by lr-10b,
`docs/library-review/round-10/lr-10b/work/warmup_options.py`) and L74 (round 11).

**Sources** (Triton 3.8.0).
- `primitives/triton-3.8.0-jit.py` lines 362-363: `warmup(*args, grid, **kwargs)` is `run` with
  `warmup=True` and the same keywords as a launch.
- `primitives/triton-3.8.0-compiler.py` lines 226-243: `triton.compile(src, target=None,
  options=None)` parses `options` with the backend; `target` defaults to the current device's.
- `primitives/triton-3.8.0-specialize.cc` lines 165-226: a host `TensorDescriptor` is specialised as
  `tensordesc<dtype[block_shape]>`.
- `primitives/triton-3.8.0-nvidia-driver.py` lines 209-264 (`make_tensordesc_arg`): the launcher
  builds the TMA descriptor with the compiled kernel's `block_size` and element type, and the
  argument's base, shape, strides and padding.

**Method.** `run.py` runs each case on GPU 0 in its own process:

- O1: a JIT launch with `enable_reflect_ftz=False`: `tl.math.exp2(-130)` is `2^-130` without
  libdevice's flush to zero and `0` with it (the default).
- O2, O3: `k.warmup(…, enable_reflect_ftz=False)` (O3: the defaults), then `compiled[grid](…)` with no
  keyword.
- O4: `triton.compile(ASTSource(k, …), options={"enable_reflect_ftz": False})`, then the launch.
- O5: `triton.compile` for `GPUTarget("cuda", 80, 32)`, launched on the H200.
- O6: a kernel that copies its host descriptor's block (`xd.block_shape`) into a `32 x 32` output,
  warmed up with a `[32, 32]` descriptor and launched through the `CompiledKernel` with a `[16, 16]`
  one; O7: the same `[16, 16]` descriptor through the JIT (control).

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/compiled-options/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | result |
|---|---|
| O1 | `exp2(-130) = 2^-130` |
| O2 | `exp2(-130) = 2^-130`: the option given at `warmup` reached a launch with no keyword |
| O3 | `exp2(-130) = 0` (control) |
| O4 | `exp2(-130) = 2^-130`: the option given to `triton.compile` reached the launch |
| O5 | rejected on the host: "no kernel image is available for execution on the device" |
| O6 | 1024 elements written: the compiled `[32, 32]` block, not the argument's `[16, 16]` |
| O7 | 256 elements written (the JIT recompiled for `[16, 16]`) |

**Conclusions**, used by `Rules.Compiler.Faithful` (scopes 4, 6 and 7), its rule page, `docs/SEMANTICS.md`
("Launches") and `docs/exclusions.md`:

1. A compile option changes every launch of the code it compiles, wherever it is given: at a JIT
   launch, at `warmup` or in `triton.compile`'s options (O1, O2, O4). Scope 4 names all three.
2. A host tensor descriptor given to a `CompiledKernel` is read with the compiled block shape (O6):
   its specialisation is its dtype and block shape, and scope 6 asks the warm-up's.
3. Code compiled for another target than the device was rejected on the host (O5); the scope names
   the H200 as the target, and code compiled for it.
