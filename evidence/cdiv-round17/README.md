# `tl.cdiv`: Triton's `(x + (div - 1)) // div`, not ceiling division

**Question.** The in-scope tutorial kernels call `tl.cdiv` 23 times (tutorials 03, 06, 08, 09 and 10;
evidence/subset-calls-round17), and the supported subset admits only the `tl.*` functions
`docs/SEMANTICS.md` and `TVA.Lang` give meaning to, which did not include it (library issue L88,
reopened by lr-15a, whose `docs/library-review/round-15/lr-15a/work/probes.py` K1, K2 showed that it is
not ceiling division on a runtime `int32`). What does `tl.cdiv(x, div)` compute, on compile-time values
and on tensors? Round 17.

**Sources** (Triton 3.8.0, pinned in `primitives/`).
- `triton-3.8.0-standard.py` lines 32-43: `cdiv` is a `@jit` function, also a tensor method
  (`_tensor_member_fn`), whose body is `return (x + (div - 1)) // div`.
- `triton-3.8.0-code-generator.py` lines 1358-1396 (`call_JitFunction`): a call of a `@jit` function
  is a helper call: its arguments are bound to its parameters, a Python number stays a compile-time
  value (`normalize_value`), and the body is compiled as a function the compiler inlines. Lines
  556-585 (`decide_return_type`): a body that returns one compile-time value has a compile-time
  return type, so the call's result is that value; `unflatten_ir_values` (1396) gives it back as a
  `constexpr`.
- So on two compile-time values the body is Python's `(x + (div - 1)) // div` (floor division), and
  with a tensor operand it is the tensor operators of SEMANTICS "Operators and promotion" and
  "Integer `//`, `%`, `<<`, `>>`": wrapping `+` and `-` in the promoted dtype, a Python number built
  in that dtype, and a truncating `//` (`tl.idiv`), undefined when `div` is `0` or the quotient
  overflows.

**Method.** `run.py`, each case in its own process with its own Triton cache:
- C1: `tl.cdiv(x, D)` on an `int32` block loaded from memory, `D` a `tl.constexpr` in
  `{1, 2, 3, 4, 7, 128, 1024, -4}`, for **every** `int32` value of `x` (`2^32` inputs per divisor),
  compared with `wrap32(x + (D - 1))` divided by `D` with truncation;
- C2: both operands `int32` blocks from memory: `2^24` random pairs (half with a small divisor) and
  every pair of 33 edge values, except `div = 0` and `x = -2^31` with `div = -1`, against the same
  formula;
- C3: `x` a runtime `int32` scalar argument (`do_not_specialize`), `D = 4` and `128`, the edge values;
- C4: an `int64` block, `D = 128`, edge values up to `2^63 - 1`, against the formula at 64 bits;
- C5: the method form `x.cdiv(4)` on the edge values;
- C6: `tl.cdiv(A, B)` with `A`, `B` `tl.constexpr`s, stored, with
  `tl.static_assert(tl.cdiv(A, B) == E)` for Python's `E = (A + (B - 1)) // B` (the front end
  evaluates `tl.static_assert` only on a compile-time condition), and `tl.arange(0, tl.cdiv(13, 4))`
  (the front end accepts only compile-time bounds there).

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/cdiv-round17/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | what | result |
|---|---|---|
| C1 | every `int32` `x`, eight constant divisors | 0 of `2^32` differ, for each divisor |
| C2 | both runtime, 16764531 pairs | 0 differ |
| C3 | runtime scalar `x`, `D = 4`, `128` | 0 of 33 differ; `tl.cdiv(2^31 - 1, 128)` is `-16777215` (the addition wraps), `tl.cdiv(-5, 128)` is `0` (`//` truncates) |
| C4 | `int64`, `D = 128` | 0 of 19 differ; `2^31 - 1` gives `16777216` (no wrap at 64 bits), `2^63 - 1` gives `-72057594037927935` |
| C5 | `x.cdiv(4)` | 0 of 33 differ |
| C6 | compile-time values | `tl.cdiv(-5, 4) = -1`, `(5, 4) = 2`, `(7, -2) = -2`, `(2^31 - 1, 128) = 16777216`, `(0, 3) = 0`, `(-7, -2) = 5`, each Python's and each passing the `tl.static_assert`; `tl.arange(0, tl.cdiv(13, 4))` accepted |

**Conclusions**, used by `docs/SEMANTICS.md` ("`tl.cdiv`"):

1. `tl.cdiv(x, div)` and `x.cdiv(div)` are the expression `(x + (div - 1)) // div`, by the rules of the
   page: on two compile-time values Python computes it at compile time, and the call is a
   compile-time value (C6: the `tl.static_assert` and the `tl.arange` bound accepted it), so
   `tl.cdiv(N, BLOCK)` on `tl.constexpr`s is Python's floor division, which is the ceiling of
   `N / BLOCK` for a positive `BLOCK` and not otherwise (`(-7, -2)` gives `5`, the ceiling is `4`);
2. with a tensor operand it is the wrapping addition and the truncating division in the operands'
   promoted dtype (C1-C5): not the ceiling when `x + (div - 1)` overflows (C3: `-16777215`) or when it
   is negative (`-5` with `4` gives `0`; the ceiling is `-1`).
