# What a compiler hint promises, in terms of the tensor's values

**Question.** `docs/SEMANTICS.md` ("Compiler hints") must say what `tl.multiple_of`,
`tl.max_contiguous` and `tl.max_constancy` promise about the values of the hinted tensor, for any
rank, so that a kernel reviewer can check a hint from the values alone (library issue L2). The
documentation (`primitives/triton-3.8.0-core.py` lines 3307-3382) defines `multiple_of`'s value by
"contiguous groups", and the compiler combines a hinted value with what its own analysis infers
for the others (`primitives/triton-3.8.0-AxisInfo.cpp` lines 1244-1259 override only the hinted
attribute; the `AxisInfo` constructor, `primitives/triton-3.8.0-AxisInfo.h` lines 31-40 and
92-101, applies the divisibility of a dimension with contiguity 1 to every element; the
elementwise lowering computes one value per group of the constancy it is given,
`primitives/triton-3.8.0-ElementwiseOpToLLVMBase.h` lines 54-104). Is the documented reading a sound one, and
what happens on tutorial 08's and tutorial 09's hints?

**Method.** `run.py` compiles small kernels with Triton 3.8.0 and launches each once, in its own
process, on GPU 0, with one warp so that each thread holds several consecutive elements and a
load is vectorised when the compiler believes the address aligned. The reference is the source's
meaning with every hint dropped. Each case lists the library's reading of its hints (the reading
adopted in `docs/SEMANTICS.md` in round 8: `multiple_of` without `max_contiguous` on the same value
promises every element; `max_contiguous` promises the tensor's own contiguity; `max_constancy`
promises groups of equal values; round 11 refined the first two, see conclusion 2).

- R1-R6: tutorial 08's pattern outside a loop: a `[16, 64]` block of float16 pointers
  `a + r[:, None] * S + c[None, :]`. R1, R2 `tl.multiple_of(p, [16, 16])`, the tutorial's hint;
  R4, R5 `tl.max_contiguous(tl.multiple_of(p, [1, 16]), [1, 64])`; R3, R6 no hint. `S = 64` rows
  are 128 bytes apart (the launcher marks `S` divisible by 16), `S = 36` 72 bytes.
- C1-C3: `rr = tl.arange(8, 16) - 8` (`[0, …, 7]`, built so that the compiler infers its
  contiguity 8 and divisibility 8), then a `[8, 64]` float32 load at `x + rr[:, None] + c[None, :]`.
  C1 `tl.max_contiguous(rr, 1)`, true if read as "groups of 1 consecutive values"; C2
  `tl.max_contiguous(rr, 8)`; C3 no hint.
- T1-T3: tutorial 09's pattern: `offs = tl.where(offs < N, offs, 0)` and
  `tl.max_contiguous(tl.multiple_of(offs, BN), BN)`, `BN = 64` float16, on a full tile (T1,
  `N = 64`) and a partial one (T2, `N = 40`); T3 the partial tile without the hints.
- K1-K3: `tl.max_constancy(offs, 4)` on `offs = [0, …, 127]` (false), on `offs // 4` (true), and
  no hint, followed by `y * 3 + 1`.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-rank2/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | library's reading | loads | result |
|---|---|---|---|
| R1 `[16, 16]`, S = 64 | false (`p[0, 1]` is 2 bytes past `a`) | `ld.global.v4.b32` | correct |
| R2 `[16, 16]`, S = 36 | false | `ld.global.v4.b32` | misaligned address |
| R3 no hint, S = 64 | - | `ld.global.v4.b32` | correct |
| R4 `[1, 16]` with `max_contiguous [1, 64]`, S = 64 | true | `ld.global.v4.b32` | correct |
| R5 the same, S = 36 | false (row 1 starts 72 bytes in) | `ld.global.v4.b32` | misaligned address |
| R6 no hint, S = 36 | - | `ld.global.b16` | correct |
| C1 `max_contiguous(rr, 1)` | false (rr's contiguity is 8) | `ld.global.v4.b32` | misaligned address |
| C2 `max_contiguous(rr, 8)` | true | `ld.global.b32` | correct |
| C3 no hint | - | `ld.global.b32` | correct |
| T1 full tile | true | `ld.global.b32` (two float16) | correct |
| T2 partial tile | false | `ld.global.b32` (two float16) | wrong: lanes 41, 43, …, 63 hold `x[1]`, not `x[0]` |
| T3 partial tile, no hint | - | `ld.global.b16` | correct |
| K1 `max_constancy(offs, 4)` | false | - | wrong at 96 of 128 elements |
| K2 `max_constancy(offs // 4, 4)` | true | - | correct |
| K3 no hint | - | - | correct |

Conclusions, used by `docs/SEMANTICS.md` ("Compiler hints"):

1. **`max_contiguous` must give the tensor's own contiguity.** C1 holds under the reading
   "groups of `C` consecutive values" (every tensor is groups of 1), yet the launch faults: the
   compiler keeps the divisibility its analysis inferred for groups of 8 (`rr[0] = 0`) and, with
   contiguity 1, applies it to every element, so it believes each row of the 2-D load starts
   16-byte aligned. A sound reading of `max_contiguous(x, C)` along a dimension is: the groups of
   `C` positions hold consecutive values, and the groups of `2C` do not (or `2C` exceeds the
   dimension). Then `C` is at least whatever the analysis inferred, and the divisibility it keeps
   holds for the hinted groups.
2. **`multiple_of` without `max_contiguous` on the same value promises every element.** The
   compiler pairs the hinted divisibility with the contiguity its analysis infers, which a reviewer
   cannot compute and which can be 1 (then the divisibility is applied to every element). With
   `max_contiguous` on the same value, the pair is the compiler's whole belief, so `multiple_of`
   promises the first element of every hinted group (R4 true and correct, R5 false and faulting).
   Round 11 (library issue L58, `evidence/hint-contig`): the library now states a lower bound of
   the inferred contiguity for the shapes kernels hint (`s + tl.arange(0, B)` has `B`, …), and reads
   `multiple_of` alone as the first element of each group of that bound; where it states none, the
   reading is every element, as here. And with the pair, `max_contiguous` promises only that the
   hinted groups are consecutive (`ContigAlong`); conclusion 1 is about `max_contiguous` alone.
3. **Tutorial 08's `[16, 16]` is false** under the documented meaning and under the library's
   reading, for every `S` (`p[0, 1]` is 2 bytes past a 16-byte boundary): the hint promises every
   element along dimension 0, whose contiguity is 1. With `S = 64` the device result is correct
   (R1), because the compiled loads use only the row starts; with `S = 36` it faults (R2).
4. **A partial tile in tutorial 09 makes the hints false**, and the compiled loads then read
   other bytes than the source names (T2: each pair of lanes is one 4-byte access from the pair's
   first offset, so the odd lanes past `N` read `x[1]`). The tutorial discards those lanes at its
   masked store, which is why its own `validate(32, 32, 32)` passes; a claim covers only the
   launches on which its hints hold (full tiles).
5. **`max_constancy` is read as documented**: a false one gives wrong values (K1: the elementwise
   code computes one value per hinted group), a true one is harmless (K2).
