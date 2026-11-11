# The integer `tl.dot`: `int8` by `int8` into `int32`

Library evidence (not a kernel's), for library issue L141. Round 31.

**Question.** `docs/rules/num.md` listed the integer `tl.dot` under "What the library does not state", so a
claim on an `int8` GEMM (vLLM's `scaled_mm_kernel`, `tl.dot(a, b, acc, out_dtype=tl.int32)`) had to write
`DotAcc` with zero error itself, a fact with constants of the kernel's own. Is `tl.dot` of `int8` operands into
an `int32` accumulator exact on sm_90, with and without an accumulator, under every `input_precision` the
front end accepts? What happens when the sum leaves the `int32` range? Which forms does the front end accept?

**Sources** (Triton 3.8.0, pinned in `primitives/`).
- `triton-3.8.0-semantic.py` 1429-1520 (`dot`): integer operands must both be `int8` (`int8` with `uint8`:
  "Both operands must be same dtype"; `uint8` with `uint8`: "only int8 supported!"); the result is `int32`
  whatever `out_dtype` says; an accumulator must have the `out_dtype`, which defaults to the accumulator's
  dtype; `max_num_imprecise_acc` is `0` unless the source gives it (it is `2^30` only for fp8 by fp8).
- `triton-3.8.0-core.py` 2355-2408: `input_precision`, when omitted, is `tf32` (knobs at their defaults).
- `triton-3.8.0-nvidia-compiler.py` 19-31 (`min_dot_size`): for 8-bit operands `K ≥ 32`, any `M` and `N`.

**Method** ([run.py](run.py), output [output.txt](output.txt)). A kernel loads an `int32` accumulator, then
loops over `K` in tiles of `BK`, each a `tl.dot` of an `int8` `[BM, BK]` tile by a `[BK, BN]` tile, and stores
the `int32` result. The exact result is computed in `float64` on the GPU (every partial sum is an integer below
`2^53`, so it is exact). Each case runs in its own process with its own empty Triton cache.
- I1: 14 tile configurations `(BM, BN, BK, num_warps)`, from `(1, 1, 32, 1)` to `(128, 128, 128, 8)`, so
  `M` and `N` from 1 to 128 and both instruction paths (I6); `K ∈ {BK, 4·BK, 2048}`; four spellings:
  `tl.dot(a, b, acc)`, `tl.dot(a, b, acc, out_dtype=tl.int32)`, `tl.dot(a, b, acc=acc)`,
  `acc += tl.dot(a, b)`. Operands uniform in `[-128, 127]`, with a row of `-128` and a row of `127` in `a`
  and two columns of `-128` in `b`; the accumulator uniform within its room, so that
  `|acc| + Σ|a·b| ≤ 2^31 - 1` for every output (checked).
- I2: `input_precision` omitted, `"tf32"`, `"tf32x3"`, `"ieee"`, `"bf16x3"`, `"bf16x6"`, and
  `allow_tf32=False`, on three configurations (`mma.sync` at `(16, 16)` and `(1, 16)`, `wgmma` at
  `(64, 128)`), `K = 1024`, as I1.
- I3: the room at its edge, `K = 131040`: `a = -128`, `b = -128` (products `16384`) from
  `acc = 2^31 - 1 - 16384·K`, and `b = 127` (products `-16256`) from `acc = -(2^31 - 1) + 16256·K`; the exact
  results are `±(2^31 - 1)`. On three configurations, both paths.
- I4: beyond the room, which the fact excludes: a final sum `2^31 - 1 - 100 + 64·16129` past `2^31 - 1`; and
  a partial sum past it (the first 32 products `+16129`, the next 32 `-16256`, from `2^31 - 1 - 1000`) whose
  final sum is in range. At `(16, 16, 32, 1)` (`mma.sync`) and `(64, 64, 32, 4)` (`wgmma`), `num_stages=1`.
- I5: the front end: `int8 × uint8`; `uint8 × uint8`; `K = 16`; `tl.dot(a, b, out_dtype=tl.float16)` (no
  accumulator); an `int32` accumulator with `out_dtype=tl.float32`; a `float32` accumulator;
  `max_num_imprecise_acc=32` (`K = 64`).
- I6: the PTX of each configuration of I1: which `mma.sync`/`wgmma` instruction.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-int8/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-11-08.

**Result** (`output.txt`).

| case | what | result |
|---|---|---|
| I1 | 14 configurations × 3 `K` × 4 spellings, 2142192 outputs with room | 0 differ from the exact result |
| I2 | 6 precisions and `allow_tf32=False` × 3 configurations, 236992 outputs | 0 differ |
| I3 | `\|acc\| + Σ\|a·b\| = 2^31 - 1`, `K = 131040`, both signs, 3 configurations | exact (`±2147483647`) everywhere |
| I4 | a final sum past `2^31 - 1` | `mma.sync`: `2147483647` (saturated); `wgmma`: `-2146451493` (wrapped) |
| I4 | a partial sum past `2^31 - 1`, the final sum in range | `mma.sync`: `2146963455` (the first chunk saturated); `wgmma`: exact |
| I5 | `int8 × uint8`, `uint8 × uint8`, `K = 16` | rejected on the host |
| I5 | `out_dtype=tl.float16`, no accumulator | accepted: an `int32` result, exact |
| I5 | an `int32` accumulator with `out_dtype=tl.float32`; a `float32` accumulator | rejected on the host |
| I5 (round 32) | a `uint32` accumulator, `out_dtype` omitted or `tl.uint32`; every other accumulator dtype | accepted, as the accumulator's bits read as `int32`; rejected on the host ([num-dot-int8-round32](../num-dot-int8-round32/README.md) A1, A2) |
| I5 | `max_num_imprecise_acc=32` | accepted: exact |
| I6 | `BM < 64` or `num_warps < 4` | `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` |
| I6 | `BM ≥ 64` and `num_warps ≥ 4` | `wgmma.mma_async.sync.aligned.m64nNk32.s32.s8.s8` (`N` the instruction's width) |

**Conclusions**, used by `docs/rules/num.md` (`DotI8`, "The integer `tl.dot`") and `docs/SEMANTICS.md`
(`tl.dot`):

1. Triton 3.8.0's only integer `tl.dot` on sm_90 is `int8 × int8`, into `int32` whatever `out_dtype` says, with
   `K ≥ 32` (I5). (Corrected in round 32, library issue L141: I5 tried only `int32` and `float32`
   accumulators. A `uint32` accumulator is accepted too, and computes as its bits read as `int32`; every
   other accumulator dtype is rejected on the host, [num-dot-int8-round32](../num-dot-int8-round32/README.md).) It runs as a chain along `K` of integer tensor-core instructions of 32 products each:
   `mma.sync` with `.satfinite`, which clamps an `int32` result that leaves the range, or `wgmma`, which wraps
   it (I4, I6).
2. While `|acc| + Σ|a·b| ≤ 2^31 - 1`, every partial sum, in any order and any grouping, has magnitude at most
   `|acc| + Σ|a·b|`, so neither path saturates or wraps, and the result is `acc + Σ a·b` exactly, for every
   `K`: so on every output of I1–I3, at the edge of the room included, on both paths, in every spelling and
   under every `input_precision`. This is `DotAcc` with `rel = abs = 0` on all operand values (`DotI8`).
3. Beyond the room the two paths differ, even when the final sum is in range (I4), so the library states
   nothing there.
4. `max_num_imprecise_acc=32` gave the exact result too (I5), but `DotI8` is stated only for the default `0`,
   which is what a source that does not give it has.
