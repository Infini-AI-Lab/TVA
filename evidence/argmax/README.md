# Index reductions: `tl.argmax`, `tl.argmin`, and `tl.max`/`tl.min` with indices

Library evidence (not a kernel's), for library issue L17. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script: [run.py](run.py); output:
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

**Result.**

| Case | Result |
|---|---|
| value type (TTIR) | the input's type: `f32`, `f16`, `i32`, `i8` (for `int8` and `uint8`); index `i32` |
| (a) on every row without NaN, all types, operations, axes, `tie_break_left` | 0 failures |
| (b), (c) on every row without NaN | 0 failures: with `tie_break_left` the first maximal (minimal) position, `±0` counting as equal; without it a maximal (minimal) position |
| (d) on every row of length 4 and 8, NaN included | 0 failures: the index is one some tree gives |
| (a) on rows with NaN, `float32` (all axes), `float16` along axis 1, n ≤ 8 | 0 failures |
| (a) on rows with NaN, `float16` along **axis 0**, n = 1024 | 8 failures (2 rows, each in 4 configurations) |
| `float16`, axis 0, 20 × 256 rows of length 16 with 5% NaN | with NaN, the value is not `x[index]` in 585 of 2901 rows (`tl.max`) and 590 of 2909 (`tl.min`); in every such row the value is some element of the row and the index is in range; every row without NaN agrees |
| `bfloat16` (converted to `float32` first) | no disagreement (exploratory run, not in the script) |
| `float8e5`, `float8e4nv` operands with `return_indices=True` | rejected on the host at compile time |
| `float32`, n = 8, values 1…8 with one NaN at position p, `tl.argmax` | the device gives index 7 for every p; the definition allows every index |

**Conclusions.**
1. Without NaN, the device agrees with the definition, and the result is determined: with
   `tie_break_left` the first position of the maximum (minimum) under `==`, and the value is the
   element at the index.
2. With a NaN in the reduced slice, the index is one some tree gives, and the result depends on the
   tree (axis 0 and axis 1 differ on the same data).
3. **The definition is too strong for `float16` with a NaN in the slice.** Along axis 0 the value
   and the index may come from different positions: e.g. value `4.0` with the index of a NaN, or a
   NaN value with the index of a number. The value is still an element of the slice and the index is
   in range. So for a slice with a NaN, the library may promise only: the index is a position of the
   slice, and the value is an element of the slice (not necessarily the element at the index).
4. fp8 operands do not compile with indices (a rejected launch).
