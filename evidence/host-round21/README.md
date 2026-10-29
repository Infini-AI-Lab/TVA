# Code compiled earlier, and the host process's own state

Library evidence (not a kernel's), for library issues L103, L109 and L110. Round 21.

**Question.**
- L103: `Compiler.Faithful`'s scope 11 said a fresh cache directory, or each compile-time parameter given
  values Triton keys apart, meets it. Does a launch on a fresh cache directory run the code compiled for its
  own values when an earlier launch of its process had values Python's `==` equates (`2` and `2.0`), and when
  the values are keyed apart parameter by parameter but not as one tuple? Is an autotuner's benchmark run an
  earlier launch?
- L109: the scopes on knobs (2), hooks (3) and Triton as installed (10) were stated of the claimed launch's own
  process at the launch. Does a launch run code compiled earlier outside those scopes, in its own process or
  in another process that used the same cache directory, and does `TRITON_ALWAYS_COMPILE=1` stop it?
- L110: scope 7 named the host as x86-64 with CPython 3.13. Do `python -O` (which removes `assert`s) and the
  process's floating-point environment (rounding mode, flush-to-zero, denormals-are-zero) change what a launch
  does?

**Sources** (pinned in `primitives/`). `triton-3.8.0-jit.py:592-614` (`compute_cache_key`: the process's key,
a dictionary lookup by `==`, then the `str` of the specialisation), `:726-757` (`JITFunction.run`: the process's
`kernel_cache` first, then a compile); `triton-3.8.0-compiler.py:71-76` (`ASTSource.hash`: each constant's `str`,
joined with `-`), `:244-270` (`compile`: the disk key, `get_cache_key`, then `always_compile` skips the disk
entry); `triton-3.8.0-GetEnv.h:15-57` (`CACHE_INVALIDATING_ENV_VARS`, which holds no `TRITON_KERNEL_OVERRIDE`);
`triton-3.8.0-knobs.py:343-354` (the cache directory: `TRITON_CACHE_DIR`, else `.triton/cache` under
`TRITON_HOME` or the home directory), `:362` (`TRITON_ALWAYS_COMPILE`); `triton-3.8.0-autotuner.py:133-170`
(the benchmark runs are launches of the kernel), `:393-405` (`triton.Config` is compared by its keywords and its
`pre_hook`); `triton-3.8.0-code-generator.py:497, 872, 886, 1020, 1117-1121` (rejections made by `assert`);
`triton-3.8.0-ir.cc:952-967` (the constant builders take C++ `float`s), `triton-3.8.0-nvidia-driver.c:1152-1157`
(a `float` argument is C's `(float)` of the double).

**Method** ([run.py](run.py), output [out.txt](out.txt)). Every case runs in processes of their own (new
interpreters, every `TRITON_` and `PYTHON` variable removed from the environment), on cache directories made for
the case; the kernel module is written to a temporary file and imported from the child.
- K1–K5 (the cache keys): `kc` stores `x * C` for the `int32` `x = 2^24 + 1` and a `tl.constexpr` `C` (the integer
  product for an `int`, the `float32` product for a `float`). K1: one process on a fresh cache directory,
  `C = 2` then `C = 2.0`, and the reverse. K2: `C = 1`, then `triton.knobs.cache.dir` set to a second, fresh
  directory, then `C = 1.0`. K3: `kab` (stores `10·len(A) + len(B)`) with `(A, B) = ("a-b", "c")` then
  `("a", "b-c")`. K4: a fresh process and directory, `triton.autotune` over `{"C": 1}` and `{"C": 1.0}` (the
  second with a `pre_hook` that does nothing, so the autotuner does not take the two configs for one) of a copy
  of `kc`, with a `do_bench` that runs each config once and times the second faster. K5: each value alone, the
  first launch of its process on a fresh directory (the controls).
- X1–X5 (the cache and other settings): a process A outside the scope launches on a directory S; then B, with Triton at its
  defaults, launches the same kernel on S; C, as B, on a fresh directory; D, as B, on S with
  `TRITON_ALWAYS_COMPILE=1`. X1: A assigns `tl.exp = tl.log`, the kernel stores `tl.exp(2.0)`. X2: A sets
  `triton.knobs.propagate_env = False` and `knobs.language.fp32_default = "ieee"`, the kernel stores
  `tl.dot(a, I)[0, 0]` for `16 x 16` `float32` blocks with `a = 1 + 2^-12` (tf32 keeps `1.0`). X3: A calls
  `fesetround(FE_TOWARDZERO)`, the kernel `envk` (below). X4: A runs as `python -I -O`, the kernel `kd` (below).
  X5: one process sets the knob of X2, launches, deletes the knob (its default) and launches again; once
  without and once with `TRITON_ALWAYS_COMPILE=1` in its environment from its start.
- H1–H3 (scope 7), each on a fresh directory: H1 `kd`, whose loop body gives the name `d`, holding
  `tl.float16` before the loop, the dtype `tl.float32` (rejected on the host, SEMANTICS.md "Names after a
  loop"), then stores `tl.full([1], 1 + 2**-11, d)` as `float32`, with `n = 3`, under `python -I`, `-I -O`,
  `-I -OO`, and `python -s` with `PYTHONOPTIMIZE=1`. H2: `fesetround(FE_TOWARDZERO)` (glibc's `libm`), then
  `envk`: `tl.full([1], 0.1, tl.float32)`, the `float32` argument `y = 0.1`, `tl.full([1], 1e-40, tl.float32)`
  and `tl.full([1], A + B, tl.float64)` for the constexprs `A = 0.1`, `B = 0.2`. H3: flush-to-zero and
  denormals-are-zero set by a C function calling `_mm_setcsr`, and by loading a shared library built with
  `gcc -shared -ffast-math` (its start-up code), then `envk`; the MXCSR bits are read back each time.

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9, x86-64, gcc 11.5.0).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28 to 2026-10-29 (the final run).

**Result** (`out.txt`).

| case | result | alone (the control) |
|---|---|---|
| K1 `C = 2` then `C = 2.0`, fresh directory | `33554434`, `33554434` | `C = 2.0`: `33554432` |
| K1 `C = 2.0` then `C = 2` | `33554432`, `33554432` | `C = 2`: `33554434` |
| K2 `C = 1`, a fresh directory set in the process, `C = 1.0` | `16777217`, `16777217` | `C = 1.0`: `16777216` |
| K3 `("a-b", "c")` then `("a", "b-c")`, fresh directory | `31`, `31` | `("a", "b-c")`: `13` |
| K4 autotune, fresh process and directory | chose `{"C": 1.0}` after 2 benchmark runs, stored `16777217` | `C = 1.0`: `16777216` |
| X1 `tl.exp = tl.log` in A | B `log 2 = 0.6931…` | C, D `exp 2 = 7.389…` |
| X2 `fp32_default = "ieee"` from Python in A | B `0x1.001p+0` (ieee) | C, D `0x1.0p+0` (tf32) |
| X3 `fesetround(FE_TOWARDZERO)` in A | B: the constant `0.1` `0x3dcccccc`, `A + B` `0x3fd3333333333333` (the argument, converted at the launch: `0x3dcccccd`) | C, D: `0x3dcccccd`, `0x3fd3333333333334` |
| X4 A under `python -I -O` (it stored `1.0`) | B `1.0`: it ran A's code | C, D rejected on the host |
| X5 knob `"ieee"`, launch, knob deleted, launch | second launch `0x1.001p+0` (ieee), with and without `TRITON_ALWAYS_COMPILE=1` | `0x1.0p+0` |
| H1 `python -I` | rejected on the host (`CompilationError`) | |
| H1 `-I -O`, `-I -OO`, `PYTHONOPTIMIZE=1` | compiled, ran, stored `1.0` (Python's value: `1.00048828125`) | |
| H2 `fesetround(FE_TOWARDZERO)` | `0.1`: `0x3dcccccc`, `y = 0.1`: `0x3dcccccc`, `A + B`: `0x3fd3333333333333` | `0x3dcccccd`, `0x3dcccccd`, `0x3fd3333333333334` |
| H3 `_mm_setcsr`, and a `-ffast-math` library loaded | MXCSR FTZ and DAZ set; `1e-40`: `0` | `0x000116c2` |

Every launch of K1–K4 and X1–X5 ran code compiled for other values or outside the scope (11 of 11).

**Conclusions.**
1. A fresh cache directory does not keep a launch from running code compiled for other values: the process
   keeps its own cache (K2), which equates values Python's `==` equates (K1), and the disk key, which a launch
   also reaches in its own process, joins the `str`s of all the compile-time parameters (K3). An autotuner's
   benchmark run is an earlier launch (K4).
2. A launch runs code compiled earlier, in its process or in any process that used its cache directory, and
   neither key records a knob set from Python, a change to Triton in memory, `-O` or the floating-point
   environment (X1–X5; `TRITON_KERNEL_OVERRIDE` likewise,
   `docs/library-review/round-20/lr-20d/work/cache_cross_process.py` X3).
3. After the user's decisions of 2026-10-29 ("To make the verifier easy, we do not verify triton cache." and
   "we assume the kernel is new."), TVA does not verify Triton's caches: `Compiler.Faithful`'s scope 11
   assumes the kernel is new (the code the launch runs was compiled for it, not taken from a cache), and
   nobody checks it. Conclusions 1 and 2 are the record of why that assumption is needed (library issues
   L103, L109; [rules/compiler-faithful.md](../../docs/rules/compiler-faithful.md), "Scope" 11). The output's
   headings ("scope 11", "scope 12") name the scopes of the round's first draft, before those decisions; the
   runs are the same.
4. `-O`, `-OO` and `PYTHONOPTIMIZE` turn a host rejection made by an `assert` into a launch that runs (H1), and
   the rounding mode and the MXCSR bits change the constants Triton builds, the `float32` of a `float` argument
   and compile-time float arithmetic (H2, H3). Scope 7 names the host process: CPython 3.13 without `-O`/`-OO`
   and with `PYTHONOPTIMIZE` unset, in the default floating-point environment (library issue L110), an
   assumption of the scope's part (B), which nobody checks (the user's decision, 2026-10-29: "So
   basically, we want to verify the kernel itself, rather than triton installation or other code
   programmed by users.").
