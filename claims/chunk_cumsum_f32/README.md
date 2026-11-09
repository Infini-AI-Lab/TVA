# `chunk_cumsum_f32`: prepared claim

fla's `chunk_local_cumsum_scalar_kernel` (`source/chunk_cumsum.py`, kernel
lines 80–120 with the helper `unflatten_program_id` 70–77; entry point `chunk_local_cumsum`
209–247 through `chunk_local_cumsum_scalar` 162–206). `Claim.lean`, namespace `Claims.ChunkCumsum`;
the claim to prove is `Statement`.

## Specification

Sequences (`nSeq`, `lo`, `hi`) are the rows `[b·T, b·T + T)` of a fixed-length batch, or
`[cu_seqlens[n], cu_seqlens[n+1])` of a variable-length one, in the flat time axis. For every
sequence `n`, position `p` in it and head `h < H`, `Post` says:
- `o[p, h]` holds a `float32` value `v`;
- if the element is `InRange` (every term of `rng n p` finite, `A = Σ|s| ≤ 2^126`, and with a scale
  the double and its `float32` value finite and `|σ₃₂|·A ≤ 2^126`), `v` is finite and
  `|v − σ·S| ≤ tol`, where `S = Σ_{q ∈ rng n p} s[q, h]`, `rng n p` running from the start of `p`'s
  chunk (`lo + ⌊(p − lo)/BT⌋·BT`) to `p` (forward) or from `p` to the chunk's end
  `min(start + BT, hi)` (reverse), and `σ` the host's double scale (`1` without one);
- frame: every byte outside the written elements (`written`: the sequences' positions × heads) is
  unchanged.

The reference is computed from `cu_seqlens` (or the rows) and `s` alone, never from
`chunk_indices` or the kernel's indexing.

## Conditions

`Inst`: `IS_VARLEN`, `B`, `T`, `H`, `BT`, `REVERSE`, `HAS_SCALE`, the scale (double), `s` (contents
`sv`), `o`, and for variable length the tables' dtype, `N`, `cu_seqlens` (`cuVal`), `NT`,
`chunk_indices` (`ciVal`).

`Inst.ok`: `BT` a power of two `≤ 2^20` (not only 16–128); `B, H ≥ 1`; `B·H < 2^31` (the
compile-time divisor `B * H` of line 96 is an `int32`-range constant, whatever the grid size);
`T < 2^31`; variable length: `B = 1`, tables `int32` or `int64`, `N < 2^31`, table lengths, `o` disjoint from the tables; the grid
`NT·B·H` (`NT = cdiv(T, BT)` on fixed length) `< 2^31`; `s`, `o` hold `B·T·H` elements; `o` disjoint
from `s`.

`Pre`: buffers valid, holding the contents; variable length: `TablesOk` — `cu_seqlens` nonnegative,
nondecreasing, at most `T` (not necessarily `0` and `T` at the ends); each row of `chunk_indices`
names a sequence `n < N` and a chunk `c ≥ 0` with `c·BT < len n`; rows distinct; every such chunk
has a row (what `prepare_chunk_indices` builds, in any row order). Every `float32` gate is covered.

The launch: `Launch.grid1 (NT·B·H) [[BT]]`, `debug = false`, `USE_GRAPH = False`. Part (A) of
`Compiler.Faithful`'s scope: `num_warps=2`, `num_stages=3` only. The claim relies on part (B).

## Assumptions

One `[rule]` field, `faithful`. Float operations: `tl.cumsum` (IEEE additions; no product feeds
it) and an IEEE `*`: no device fact. No `[kernel]` field.

## Coverage

76 instances `setup_*` / `setup_*_ok`: every launch of the host's test (lines 267–300): the five
fixed-length shapes and the seven variable-length ones (`[0, 15]` … the 65,536-token batch of 32
heads), forward and reverse, without a scale and with `RCP_LN2`; the variable-length ones with
`int32` tables (the test's) and `int64` tables (`SETUP.md`), `NT` as `prepare_chunk_indices` gives
it. `Pre`'s `TablesOk` is about contents, so it is checked on the host's tables by `check_gpu.py`
(every table passes).

## Tolerances

`u = 2^-24`, `η = 2^-150` (`Format.binary32`; `u32_eq`, `eta32_eq`). `depth n p = p − start`
(forward) or `start + BT − 1 − p` (reverse): one less than the number of lanes the scan combines at
`p`'s lane (lanes past the sequence's end hold the load's `+0`, of magnitude `0`, so the magnitude
sum of the leaves is `A`).
- Without a scale, `tol = E = Tree.errBound u η m m A = ((1+u)^m − 1)·A + η·m·(1+u)^m`: by
  `safe_cumsum` each lane is a tree over its `ScanLeaves` with IEEE additions, and `sum_err_list`
  bounds it by exactly this (`L.length − 1 = m`), in range since `A + E ≤ 1.07·2^126`.
- With a scale, `tol = |σ₃₂|·E + (u·|σ₃₂|·(|S| + E) + η) + |σ − σ₃₂|·|S|`: the scan's error scaled,
  `FloatV.mul_err` for `b_o·σ₃₂` (in range: `|σ₃₂|·(A + E) < 2^127`), and the scale's `float32`
  conversion, written exactly so that the reference uses the host's double.

Meaningful: the bound is per position (`m` from `0` to `BT − 1`), relative to the magnitudes summed
(for log-decay gates, `A = |S|`). At depth `0` the tolerance is `0` (one term, exact). Measured on the
H200: worst error / tol 1.0000 (just below 1, at shallow depths where the bound is one rounding);
at depth 64–127 the measured error is at most about `3.2u·A` against `≈ 127u·A` — the library only
says the scan is some tree over the leaves, so the depth bound is `m` (see Library requests).

## Exclusions

- **A grid of `2^31` programs or more** — a limit of the launch (CUDA): `NT·B·H ≥ 2^31`, e.g.
  `BT = 16`, `H = 64`, `T ≥ 2^29` tokens. Real use does not reach it.
- **`B·H ≥ 2^31`** — outside real use (batch rows times heads in the thousands at most); the
  compile-time constant `B * H` is then outside `int32`, and beyond `2^63` Triton rejects it at
  compile time, even on a launch with an empty grid (`T = 0`).
- **`T ≥ 2^31`** — outside real use (a ≥ 8 GiB gate per head); `T` would be an `int64` argument and
  `int32` tables would wrap.
- **Elements not `InRange`** — a limit of `float32`, per element: non-finite gates in the summed
  range, or magnitudes above `2^126` (the cumulative sum or the product can overflow near `2^128`);
  a scale beyond `float32`'s range makes `σ₃₂` infinite. Real use (log-decays, `RCP_LN2`) is far
  inside.
- **`USE_GRAPH = True`** — outside `SETUP.md`'s use (the layers call with the default `False`).
- **Variable-length tables not `TablesOk`** (a hand-made table with repeated or missing chunks, or
  entries outside the sequences) — outside real use: `SETUP.md` asks for `prepare_chunk_indices`'s
  table or the same table passed by the caller.
- The 4-D path (`chunk_local_cumsum_vector_kernel`) is another kernel.

## Modelling decisions

- `unflatten_program_id` inlined (`Y = None`: the 2-D branch).
- `if IS_VARLEN:` is a compile-time `if`; the model's Lean `if` returns `(i_t, bos, T)` from each
  branch: on fixed length `bos = i_b·T` with `T` promoted to `int64` (as line 109 promotes it);
  `eos` there is dead.
- The table loads are `tl.load1` of `idxD` with `tl.convert idxD .i32/.i64` (identity for the same
  dtype; exact for the other since entries are below `2^31`).
- `tl.cumsum(…, reverse=REVERSE)`: the two calls of lines 115/117 are one scan with the flag.
- `scale` is the runtime `float32` (`FloatV.hostF32` of the double); when `HAS_SCALE` is false the
  host passes `None` and the model's value is unused.

## Library requests

- **The depth of `tl.cumsum`'s combine tree.** The library states only that each lane is some tree
  over its leaves (`ScanLeaves`), so the provable depth is the number of leaves less one, about 40×
  the measured error at `BT = 128`. A fact bounding the depth Triton 3.8.0's scan uses on sm_90
  (per-thread sequential part plus a logarithmic warp/block part) would allow a tolerance near the
  kernel's real error.

## Checks

- Builds with no errors or warnings, no `sorry`/`axiom`; `consistent`, `satisfiable`, `setup_*_ok`
  use only the three standard axioms.
- `check_gpu.py` (H200): 30 cases, each forward and reverse with each scale, through the host and
  through a guarded direct launch (`o` prefilled with a sentinel): the host's 12 shapes with `int32`
  and `int64` tables, and edges (empty and length-1 sequences, sequences not covering `[0, T)`, a
  permuted `chunk_indices`, `BT` 512 and 2048, gates near `2^118`, NaN/±inf gates, scales `−2.5`,
  `±0`, `1e-45`, `1e-40`, `1e30`, `1e39`, `2^20` tokens × 64 heads, 64 fixed rows). Every host table
  is `TablesOk`; worst error / tol 1.0000 (below 1) over every in-range element, all finite;
  inputs unchanged; every unwritten element keeps the sentinel.
