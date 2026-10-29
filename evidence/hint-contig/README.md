# `tl.multiple_of` alone, and with `tl.max_contiguous`, on the shapes kernels hint

**Question.** Before round 11 the library read `tl.multiple_of(x, v)` alone as "every element of
`x` is a multiple of `v`", which no launch meets for the common 1-D idiom
`tl.multiple_of(s + tl.arange(0, B), v)`, and `tl.max_contiguous` with a `multiple_of` on the same
value as `ContigIs` (`C` is exactly the value's contiguity). Triton 3.8.0 relies on less. What does
the compiler believe after each hint, and for which shapes can the library state the contiguity it
pairs a `multiple_of` with? Library issue L58 (round 11; raised by lr-8a, whose
`docs/library-review/round-8/lr-8a/work/hint_weaker.py` W1-W8 first showed it).

**Sources** (`primitives/triton-3.8.0-AxisInfo.cpp`, `AxisInfo.h`).
- The analysis keeps, per value and dimension, a contiguity, the divisibility of the first element
  of each contiguous group, and a constancy (`AxisInfo.h` lines 20-139). A dimension of contiguity
  1 makes its divisibility apply to every element and to every dimension (lines 31-40, 92-101).
- `visitOperation` (`AxisInfo.cpp` lines 1232-1259) computes the defining operation's value, then
  replaces the contiguity with a `tt.contiguity` hint (`max_contiguous`), the divisibility with a
  `tt.divisibility` hint (`multiple_of`) and the constancy with a `tt.constancy` hint
  (`max_constancy`). So `multiple_of` alone pairs `v` with the contiguity the analysis inferred, and
  the pair `max_contiguous(multiple_of(x, v), C)` replaces both: the compiler then believes
  "groups of `C` are consecutive, and `v` divides the first element of each".
- The contiguity inferred: `tt.make_range` (`tl.arange(a, b)`) has `b - a` (lines 169-183); an
  addition or subtraction (`arith.addi`, `subi`, `tt.addptr`) has
  `max(gcd(lhs.constancy, rhs.contiguity), gcd(lhs.contiguity, rhs.constancy))`, or for `x - y`
  `gcd(x.contiguity, y.constancy)` (lines 242-279); a splat scalar has constancy the whole shape
  (lines 553-575); `tt.expand_dims` inserts contiguity 1 (lines 607-640); `tt.broadcast` gives a
  broadcast dimension contiguity 1 and constancy the whole dimension, and keeps the others (lines
  642-668); integer casts (`extsi`, `extui`, `trunci`) pass the operand's through (lines 138-147,
  1180-1185). The library states bounds for these only; other operations may give more (a
  select, a product with the constant 1), which the library does not rely on, and a value with no
  visitor gets 1.

**Method.** `run.py` loads a block of 128 `float32` through `x_ptr + offs` (or a hinted pointer),
with one warp, so each thread holds four consecutive elements and the load is vectorised
(`ld.global.v4.b32`) exactly when the compiler believes each thread's first address is 16-byte
aligned. `x_ptr` is 16-byte aligned; the offsets `s` and strides `S` are chosen so that the launcher
does not mark them divisible by 16 itself (a runtime integer that is a multiple of 16, `0` included,
is marked, which is why `s = 0` would vectorise without any hint). Each case is "true" or "false"
under the library's reading after round 11: `multiple_of` alone is `FirstsDiv x k C v` with `C` the
lower bound stated for the shape (`B = 128` for the 1-D sums, `32` along dimension 1 of the 2-D
ones, `1` along dimension 0), and the pair is `ContigAlong x k C ∧ FirstsDiv x k C v`. Each case
runs in its own process (a misaligned access ends the CUDA context), on GPU 0.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-contig/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | hinted value | hint | under the reading | loads | result |
|---|---|---|---|---|---|
| H1, H2 | `s + tl.arange(0, 128)`, `s = 4`, `1` | `multiple_of(·, 4)` | true, false | `v4` | correct; misaligned address |
| H3, H4 | `tl.arange(0, 128) + s`, `s = 4`, `2` | `multiple_of(·, 4)` | true, false | `v4` | correct; misaligned address |
| H5, H6 | `tl.arange(0, 128) - s`, `s = -8`, `-3` | `multiple_of(·, 4)` | true, false | `v4` | correct; misaligned address |
| H7, H8 | a loaded scalar `+ tl.arange(0, 128)`, `8`, `1` | `multiple_of(·, 4)` | true, false | `v4` | correct; misaligned address |
| H9, H10 | the pointer `x_ptr + s + tl.arange(0, 128)`, `s = 4`, `1` | `multiple_of(·, 16)` (bytes) | true, false | `v4` | correct; misaligned address |
| H11, H12 | `(s + tl.arange(0, 128)).to(tl.int64)`, `4`, `1` | `multiple_of(·, 4)` | true, false | `v4` | correct; misaligned address |
| H13, H14 | `(tl.arange(0, 4) * S)[:, None] + tl.arange(0, 32)[None, :]`, `S = 36`, `34` | `multiple_of(·, [1, 4])` | true, false | `v4` | correct; misaligned address |
| H15, H16 | a loaded column `[:, None] + tl.arange(0, 32)[None, :]`, rows `[0, 36, 72, 108]`, `[0, 34, 68, 102]` | `multiple_of(·, [1, 4])` | true, false | `v4` | correct; misaligned address |
| H17 | `s + tl.arange(0, 128)`, `s = 4` | `max_contiguous(multiple_of(·, 4), 4)` | true (`C = 4` below the contiguity 128) | `v4` | correct |
| H18 | the same | `max_contiguous(multiple_of(·, 4), 8)` | true | `v4` | correct |
| H19 | `s = 2` | `max_contiguous(multiple_of(·, 4), 4)` | false | `v4` | misaligned address |
| H20 | `s + tl.arange(0, 128)`, `s = 4` | none (control) | - | `b32` | correct |
| H21 | the 2-D offsets, `S = 36` | none (control) | - | `b32` | correct |

**Conclusions**, used by `docs/SEMANTICS.md` and `TVA.Lang.Ops` ("Compiler hints": what a hint
promises; the contiguity the analysis infers):

1. `tl.multiple_of(x, v)` alone is a promise about the first element of each group of the
   contiguity the analysis infers for `x`, not about every element: on each listed shape the hint
   vectorised the loads (against scalar loads without it, H20, H21), was correct when `v` divides
   the first element of each group of the stated bound, and faulted when it does not. So
   `FirstsDiv x k C v`, with `C` the stated lower bound, is the reading: it implies the compiler's
   belief (a smaller `C` is a stronger promise), and it is true exactly on the launches where the
   device was correct. For a shape the library lists no bound for, `C = 1` (every element) remains
   sound.
2. The shapes with a stated bound, each confirmed here: `tl.arange(0, B)` plus or minus a scalar
   (an argument, a loaded scalar), with the arange on either side of `+`; a pointer plus such
   offsets; an integer conversion of them; and, along the dimension that is not broadcast, the sum
   of a column `[:, None]` (computed or loaded) and a row `tl.arange(0, BN)[None, :]`, where
   dimension 0 has the bound 1.
3. With both hints on one value, the compiler believes exactly `ContigAlong x k C` and
   `FirstsDiv x k C v`, also for `C` below the value's own contiguity (H17, H18 correct, H19
   faults): `ContigIs` is needed only for `max_contiguous` alone (`evidence/hint-rank2` C1).
