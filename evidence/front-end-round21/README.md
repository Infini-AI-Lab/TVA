# A helper's equal `return`s, and arithmetic in an fp8 dtype

Library evidence (not a kernel's), for library issues L102 and L111. Round 21.

**Question.**
- L102: SEMANTICS admitted, as a helper whose call is the value of the `return` its program takes,
  "`return`s that all give one compile-time value, equal values of one Python type". lr-20b found that
  `(1, 2)` and `(1.0, 2)` are such values, and that the call is the first visited `return`'s value on
  every path. Is that so, and is it so for `0.0` and `-0.0`, which Python's `==` also finds equal?
- L111: SEMANTICS said that arithmetic in an fp8 dtype is always rejected on the host. lr-20e found that
  MLIR's folders remove some such operations first, which then run. Which forms run, which are rejected?

**Sources** (pinned in `primitives/`). `triton-3.8.0-code-generator.py:556-626` (`decide_return_type`,
`cast_to`, `handle_returns`: the call's type from every visited `return`), `triton-3.8.0-core.py:187-188`
(`constexpr_type.__eq__`, Python's `==` on the values). `llvm-ArithOps.cpp` (the `arith` folders).

**Method** ([run.py](run.py), output [out.txt](out.txt)). Each kernel is written to a file of its own in a
temporary folder and imported from there, on a fresh Triton cache directory.
- T1–T4: a helper with a runtime `if c > 0:` and two `return`s of compile-time values that `==` finds
  equal, called with the runtime `int32` `c = 5` (the `if` taken) and `c = -5` (never `1`, which Triton
  passes as a compile-time constant). T1 `return (1, 2)` then `return (1.0, 2)`, the kernel storing
  `x * th(c)[0]` for the `int32` `x = 2^24 + 1` (`16777217` for the integer product, `16777216` for the
  `float32` one); T2 the two `return`s the other way round; T3 as T1 through `a, b = th(c)`; T4
  `return 0.0` then `return -0.0`, the kernel storing `2` when `f"{h(c)}" == "-0.0"`, else `1`.
- F: on an fp8 tensor `x` of each fp8 dtype, loaded by bit cast from all 256 patterns, each of 20 forms,
  its result stored by bit cast: `x * 1.0`, `1.0 * x`, `x * 1`, `1 * x`, `x / 1.0`, `x / 1`, `x - 0.0`,
  `x - 0`, `x + 0.0`, `x * 2.0`, `x * -1.0`, `x / 2.0`, `x + x`, `x - x`, `x * x`, `-x`, the comparison
  `x < x`, and three operations on two fp8 constants (`tl.full([256], 1.5, d) * tl.full([256], 2.0, d)`,
  `tl.full([256], 1.5, d) + 0.5`, `tl.full([256], 1.5, d) - tl.full([256], 0.5, d)`).

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`out.txt`).

| case | `c = 5` | `c = -5` | the `return` taken gives |
|---|---|---|---|
| T1 `(1, 2)`, then `(1.0, 2)`; `x * th(c)[0]` | `16777217` | `16777217` | `16777217`, `16777216` |
| T2 `(1.0, 2)`, then `(1, 2)` | `16777216` | `16777216` | `16777216`, `16777217` |
| T3 as T1, through `a, b = th(c)` | `16777217` | `16777217` | `16777217`, `16777216` |
| T4 `0.0`, then `-0.0`; `f"{h(c)}" == "-0.0"` | `1` | `1` | `1`, `2` |

| F, both fp8 dtypes | result |
|---|---|
| `x * 1.0`, `1.0 * x`, `x * 1`, `1 * x`, `x / 1.0`, `x / 1`, `x - 0.0`, `x - 0` | compiled and ran; every one of the 256 patterns stored unchanged |
| two fp8 constants: `1.5 * 2.0`, `1.5 + 0.5`, `1.5 - 0.5` | compiled and ran (computed at compile time): `3.0` (`0x42` in `float8e5`, `0x44` in `float8e4nv`), `2.0` (`0x40`), `1.0` (`0x3c`, `0x38`) |
| `x + 0.0`, `x * 2.0`, `x * -1.0`, `x / 2.0`, `x + x`, `x - x`, `x * x`, `-x`, `x < x` | rejected on the host (`PassManager::run failed`) |

22 of the 40 fp8 forms compiled and ran.

**Conclusions.**
1. A helper's visited `return`s of compile-time values that `==` finds equal give the first visited
   `return`'s value on every path: tuples are compared element by element, and `0.0` with `-0.0` too
   (T1–T4). So the subset admits only `return`s of the same value of one Python type: for a float bit
   for bit (no NaN, not `0.0` with `-0.0`), for a tuple element by element (library issue L102;
   SEMANTICS.md "The supported subset", "A helper's value").
2. Arithmetic whose computation dtype is an fp8 dtype is rejected on the host except where MLIR's
   folders remove the operation first: `x * 1`, `1 * x`, `x / 1` and `x - 0` give `x`, and an operation
   on two fp8 constants is computed at compile time (F). Every operator whose computation dtype is an
   fp8 dtype, and unary `-` on an fp8 tensor, are outside the subset (library issue L111;
   SEMANTICS.md "Operators and promotion", exclusions.md).
