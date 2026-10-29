# Index reductions: `tl.argmax`, `tl.argmin`, and `tl.max`/`tl.min` with indices

Library evidence (not a kernel's), for library issue L17. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27 (rerun and extended for round 3). Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question.** Does the device's `(value, index)` result agree with the library's definition of the
index reductions?

**What the library reads** (`primitives/triton-3.8.0-standard.py`, `primitives/triton-3.8.0-core.py`).
- `tl.max(x, axis, return_indices=True, return_indices_tie_break_left=t)` (`standard.py:173-183`)
  and `tl.argmax(x, axis, tie_break_left=t)` (`:194-199`) reduce the pairs `(x[j], j)` with
  `_argmax_combine` (`:146-155`); `tl.min`/`tl.argmin` (`:232-242`, `:253-258`) with
  `_argmin_combine` (`:205-214`). The index is `tl.arange(0, n)`, `int32`
  (`_reduce_with_indices`, `core.py:3093-3107`). Only `bfloat16` is converted first (`:178`, `:237`).
- The combine, as the library states it (`tl.argCombine`): `tie = t and v1 == v2 and i1 < i2`;
  `win = (v1 > v2 if max else v1 < v2) or tie`; the result is `(v1, i1)` if `win`, else
  `(v2, i2)`. Float comparisons are ordered (false with a NaN). The reduction is this combine along
  some binary tree over all positions, in any leaf order.

**Method.** Rows of a 2-D block are reduced along axis 1, or columns along axis 0, by all four
operations, with `tie_break_left` true and false. Checks per result: (a) the index is in range and
the value is `x[index]` bit for bit; (b) without NaN and with `tie_break_left`, the index is the
first position whose value `==` the maximum (minimum); (c) without NaN and without
`tie_break_left`, the value `==` the maximum (minimum) at the index; (d) for rows of at most 8
elements, the index is one that some tree gives (every tree and leaf order enumerated in Python).
Rows: every row of length 4 over `NaN, -inf, -1, -0, +0, 1, +inf` (floats) or over the type's
extremes, `-1`, `0`, `1` (integers); 512 random rows of length 8 over the same values; 128 rows of
length 1024 with many ties (`±0` among them), with and without 1% NaN. Types: `float32`, `float16`,
`int32`, `int8`, `uint8`.

Two further parts look at the value and the index of one result when the slice holds a NaN:
- 20 × 256 slices of length 16, 5% NaN, axis 0, `float16` and `bfloat16`;
- the last part (`value_index`): `[16, 64]` blocks reduced along axis 0 (slices of 16) and
  `[32, 256]` blocks along axis 1 (slices of 256), 256 programs × 8 launches per configuration,
  values in `-4…4`, NaNs at a rate that puts one in about half the slices, and the same without
  NaN; `float16`, `float32`, `bfloat16`; `tl.max`/`tl.min` with `return_indices=True` and
  `tl.argmax`/`tl.argmin` in the same kernel, `tie_break_left` true and false.

In both, values are compared bit for bit, except that any NaN equals any NaN (`bfloat16` is reduced
in `float32` and converted back, which may change a NaN's payload).

**Result.**

| Case | Result |
|---|---|
| value type (TTIR) | the input's type: `f32`, `f16`, `i32`, `i8` (for `int8` and `uint8`); index `i32` |
| every result, every type, layout and operation | the index is in range (0 out of range) |
| (a) on every row without NaN, all types, operations, axes, `tie_break_left` | 0 failures |
| (b), (c) on every row without NaN | 0 failures: with `tie_break_left` the first maximal (minimal) position, `±0` counting as equal; without it a maximal (minimal) position |
| (d) on every row of length 4 and 8, NaN included | 0 failures: the index is one some tree gives |
| `value_index`, slices without NaN, `float16`, `float32`, `bfloat16`, both layouts | value = `x[index]` in all 786432 slices of each type; `tl.argmax`'s (`tl.argmin`'s) index = `tl.max`'s (`tl.min`'s) index in every slice, with NaN or without |
| (a) on rows with NaN, `float32` (all parts, all layouts) | 0 failures (in `value_index`, 392873 slices with NaN) |
| (a) on rows with NaN, `float16`, n ≤ 8 | 0 failures |
| `float16`, axis 0, n = 1024, 1% NaN | 8 failures (2 rows, each in 4 configurations) |
| `float16`, axis 0, 20 × 256 slices of 16, 5% NaN | value ≠ `x[index]` in 585 of 2901 slices with NaN (`tl.max`), 590 of 2909 (`tl.min`) |
| `bfloat16`, the same | 557 of 2869 (`tl.max`), 625 of 2873 (`tl.min`) |
| `value_index`, `float16`, `[16, 64]` axis 0, slices with NaN | value ≠ `x[index]` in 50857 of 262138 (19.4%): value NaN and `x[index]` a number 21329; value a number and `x[index]` NaN 21354; two different numbers 8174 |
| `value_index`, `float16`, `[32, 256]` axis 1, slices with NaN | 1394 of 131125 (1.1%): value NaN, `x[index]` a number 726; value a number, `x[index]` NaN 668; two different numbers 0 |
| `value_index`, `bfloat16`, `[16, 64]` axis 0 / `[32, 256]` axis 1 | 50890 of 261838 (19.4%; 21344, 21272, 8274) / 1368 of 130991 (1.0%; 684, 684, 0) |
| in every slice with NaN, every type | the value is an element of the slice (a number of the slice, or NaN), and the index is in range |
| `float8e5`, `float8e4nv` operands with `return_indices=True` | the compiler fails on the host (an MLIR pass error, `RuntimeError`): a rejected launch |
| `float32`, n = 8, values 1…8 with one NaN at position p, `tl.argmax` | the device gives index 7 for every p; the definition allows every index |

An earlier version of this page said `bfloat16` showed no disagreement. That came from a bug in the
script: it tested `dtype.startswith("float")`, so `bfloat16` inputs went through `int64` and every NaN
became `-2^63`. The script now treats `bfloat16` as a float (`is_float`), and `bfloat16` behaves like
`float16`.

**Conclusions.**
1. Without NaN in the reduced slice, the device agrees with the definition, and the result is
   determined: the value is the element at the index, for every type tested; with
   `tie_break_left` the index is the first position of the maximum (minimum) under `==`.
   `tl.argmax`/`tl.argmin` give the same index as `tl.max`/`tl.min` with `return_indices=True`.
2. With a NaN in the slice, for `float32` the value is still the element at the index, and the
   index is one some tree gives; the result depends on the tree (axis 0 and axis 1 differ on the
   same data).
3. **For `float16` and `bfloat16` with a NaN in the slice, the value and the index may come from
   different positions**, along either axis: the value NaN with the index of a number, the value a
   number with the index of a NaN, or (axis 0) two different numbers. The value is always an
   element of the slice and the index is always in range. So for a slice with a NaN the library may
   promise only: the index is a position of the slice, and the value is an element of the slice
   (a NaN when it is a NaN), not necessarily the element at the index.
4. fp8 operands do not compile with indices (a rejected launch).
