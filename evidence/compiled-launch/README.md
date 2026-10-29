# Launching a precompiled `CompiledKernel`, and `launch_metadata=`

**Question.** Tutorial 02 compiles its kernel with `softmax_kernel.warmup(...)` and launches the
returned `CompiledKernel` with `kernel[(num_programs, 1, 1)](...)` (`tutorials/02-fused-softmax.py`
lines 143-174). Through the JIT, every call binds the arguments and specialises the kernel to them
(a pointer 16-byte aligned or not, an integer equal to 1, divisible by 16, at or above `2^31`);
a `CompiledKernel` launch does neither. What does such a launch do with arguments that do not match
what it was compiled for? And when does a `launch_metadata=` function given to `@triton.jit` run?
Library issue L47 (round 8), and the `launch_metadata=` item of L44/L46.

**Sources** (Triton 3.8.0).
- `primitives/triton-3.8.0-jit.py` lines 362-363 (`warmup` compiles without launching), 741 (the
  binder computes the specialisation on every JIT call), 778-780 (the JIT's launch).
- `primitives/triton-3.8.0-specialize.cc` lines 274-276 (an integer equal to 1 becomes a constexpr), 280-284
  (`i32` or `i64` by value; divisibility 16 when the value is), 363 (a pointer's divisibility 16).
- `primitives/triton-3.8.0-compiler.py` lines 493-514: `CompiledKernel.launch_metadata` and
  `__getitem__`. The launch passes `grid[0]`, `grid[1]`, `grid[2]` and the arguments as given to
  the launcher; nothing compares them with the specialisation.
- `primitives/triton-3.8.0-nvidia-driver.c`: `extractArgs` (lines 1360-1390) walks the compiled
  signature and drops constexpr positions without checking the number of arguments given (1383);
  `extractI32` (1100-1103) stores `PyLong_AsLong(obj)` into an `int32_t` (wraps modulo `2^32`);
  pointers must be device-accessible (1083); `launchHook` (1406-1415) calls the enter hook with the
  launch metadata object before the launch (1447) and abandons the launch if it raises.
- `primitives/triton-3.8.0-knobs.py` lines 477-478: `knobs.runtime.launch_enter_hook` is a
  `HookChain`, never `None`, so `CompiledKernel.launch_metadata` always builds a `LazyDict`
  (`primitives/triton-3.8.0-compiler.py` lines 374-387); the user's `launch_metadata` function runs
  only when something calls that dict's `get()`.

**Method.** `run.py`, each case in its own process on GPU 0:

- K1: tutorial 02's kernel (copied verbatim) and its two calls, on its unit-test shape (1823 x 781):
  the result against `torch.softmax`; the specialisation the warm-up compiled; the launch with other
  values (1 and 99; `None` and a string) in the two constexpr positions; without them; with a
  1-element grid.
- K2: a copy kernel (`y[i] = x[i]` for `i < n`, or `x[i * s]`) compiled by `warmup` and launched
  as a `CompiledKernel` with arguments outside its specialisation: (a) `x` 4 bytes past a 16-byte
  boundary where the compile saw an aligned `x`; (b) `n = 1001` where it saw `n = 1024`
  (divisible by 16); (c) `s = 3` where it saw `s = 1` (a constexpr); (d) `s = 1` where it saw
  `s = 3`; (e) `n = 2^31 + 5` and `2^32 + 5` where it saw an `int32` `n`; (f) a `float16` `x` where
  it saw `float32`; (g) two of the three runtime arguments (five processes); (h) a CPU tensor.
- K3: `@triton.jit(launch_metadata=f)`, launched through the JIT and as a `CompiledKernel`: with
  the default hooks, with an enter hook that does not read the metadata, with one that calls
  `get()`, and with `f` raising.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/compiled-launch/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

- K1: the tutorial's launch is correct (`allclose` to `torch.softmax`). The warm-up compiled
  `output_ptr`, `input_ptr` with divisibility 16, the strides and sizes as `int32` without
  divisibility (781, 1823), `BLOCK_SIZE = 1024` and `num_stages = 4` as constexprs. The values in
  the constexpr positions are ignored (1 and 99, `None` and a string give bit-identical results),
  and the launch without them runs and gives the same result. A grid of length 1 raises
  `IndexError` on the host.
- K2 (compiled for aligned `x`, `n = 1024`; its PTX uses `ld.global.v4.b32` / `st.global.v4.b32`):
  - (a) misaligned `x`: `CUDA error: misaligned address` (the JIT recompiles and is correct);
  - (b) `n = 1001`: 1004 elements written, indices 1001-1003 beyond `n` (the JIT writes 1001);
  - (c) `s = 3` to a kernel compiled with `s = 1`: the kernel computes `x[i]`, as if `s = 1`;
  - (d) `s = 1` to a kernel compiled with an `int32` `s = 3`: correct;
  - (e) `n = 2^31 + 5`: nothing written (the value wraps to a negative `int32`); `n = 2^32 + 5`:
    `n` wraps to 5, and with the compiled divisibility of `n` the mask is applied per group of four,
    so 8 elements are written (the JIT compiles an `i64` `n` for both and writes all 128);
  - (f) a `float16` `x`: `y` holds `x`'s bytes read as `float32`; no error;
  - (g) too few arguments: `SystemError: ... bad argument to internal function` in 5 of 5 processes
    (an earlier variant of this case that went on in the same process died with a segmentation
    fault): the launcher reads past the arguments it was given;
  - (h) a CPU tensor: `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`,
    nothing written.
- K3: `launch_enter_hook` is a `HookChain`, not `None`. `f` is not called with the default hooks
  or with a hook that does not read the metadata, through the JIT or the `CompiledKernel`; with a
  hook that calls `get()`, `f` is called once per launch and the hook sees its entries; the result is
  correct in every case. If `f` raises, the exception reaches the caller and the launch does not run
  (`y` untouched).

**Conclusions.**

1. A `CompiledKernel` launch runs the code compiled for the warm-up's arguments, with no check of
   the new arguments against that specialisation. With arguments that match it (same dtypes, the
   same pointer alignment class, integers in the same class: equal to 1, divisible by 16 or not,
   `int32` or `int64`), it is the launch the JIT would make; with others, its results can be
   wrong (b, c, e, f), it can fault (a), or the host can fail (g). Tutorial 02's launch passes the
   warm-up's own arguments, so it matches.
2. A constexpr position of a `CompiledKernel` launch is ignored: the compiled value is used.
3. `launch_metadata=` is host code that runs only when a launch-enter hook reads the metadata; it
   does not change the launch. If it raises, the launch is abandoned on the host (no outcome).
