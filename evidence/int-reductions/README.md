# The result types and values of integer reductions

Library evidence (not a kernel's), for library issue L16. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question.** What type, and what value, do `tl.max`, `tl.min`, `tl.sum`, `tl.max(…,
return_indices=True)` and `tl.argmax` give for integers narrower than 32 bits?

**What the library reads** (`primitives/triton-3.8.0-standard.py`):
- `tl.max` (`:173-191`) and `tl.min` (`:232-250`) without `return_indices`: an input narrower
  than 32 bits is converted first, floats to `float32` and integers to `int32` whatever their
  signedness (`:185-190`, `:244-249`). The conversion `.to(int32)` sign-extends a signed and
  zero-extends an unsigned input (`tl.int1` is unsigned).
- With `return_indices=True`, and in `tl.argmax` (`:194-199`) and `tl.argmin` (`:253-258`), only
  `bfloat16` is converted (`:178`, `:237`); the value keeps the input's type and the index is
  `int32`.
- `tl.sum` (`:284-293`) converts an integer narrower than 32 bits to `int32` if it is signed and to
  `uint32` if it is unsigned, `tl.int1` included (`_pick_sum_dtype`, `:269-281`).

**Method.** One program reduces one block. Each 8-bit type's block holds all 256 values, each
16-bit type's all 65536, shuffled with a fixed seed; `int1` has 256 booleans; the 32-bit types have
65536 values with the extremes, `0`, `±1` and random values. The script reads the `tt.reduce`
result type from the TTIR (MLIR integers carry no sign), stores the result with `.to(tl.int64)`
(sign-extending a signed result, zero-extending an unsigned one), and stores `r + 1000` in an
`int32` buffer. Part B sums 2^17 copies of a large value, so that the sum leaves 32 bits.

**Result.**

| Input | `tl.max`, `tl.min` | `tl.sum` | `tl.max(…, return_indices=True)`, `tl.argmax` |
|---|---|---|---|
| `int8`, `int16` | `i32`, sign-extended: exact max/min (`127 + 1000 = 1127`) | `i32`, wraps as `int32` | value `i8`/`i16`, index `i32` |
| `uint8`, `uint16` | `i32`, zero-extended: `255`, `65535` | `i32`, read back zero-extended: `uint32` (`65535 · 2^17` gives `4294836224`) | value `i8`/`i16` (unsigned), index `i32` |
| `int1` | `i32`: `1`, `0` | `uint32`: the count of `True` | value `i1`, index `i32` |
| `int32` | `i32` | `i32`, wraps | value `i32` |
| `uint32` | unsigned, not widened (`4294967295`) | `uint32`, wraps | value unsigned 32-bit |

- Part A: all 35 results equal the statement above (`A: 0 result(s) differ`); for each argmax the
  index is the first position of the maximum.
- Part B: the 8 sums equal the `int32` (signed inputs) or `uint32` (unsigned inputs) wrap of the
  exact sum.
- Part C: on the value of `tl.max(x, 0, return_indices=True)`, `r + 1000` is rejected at compile
  time for `int8` and `uint8` ("Scalar 1000 is out of range"), and for `int16` it wraps in 16 bits
  (`32767 + 1000 = -31769`). So that value is not widened.

**Conclusions.**
1. `tl.max`/`tl.min` without indices reduce `int32` values for every integer input narrower than 32
   bits, sign-extended from `int8`/`int16`, zero-extended from `uint8`/`uint16`/`int1`, and the
   result is `int32`. A model writes that conversion before the reduction.
2. `tl.sum` reduces in `int32` for signed and `uint32` for unsigned inputs narrower than 32 bits,
   and wraps there.
3. `tl.max`/`tl.min` with `return_indices`, `tl.argmax` and `tl.argmin` keep the input's integer
   type for the value; the index is `int32`.
