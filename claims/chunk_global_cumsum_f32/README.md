# `chunk_global_cumsum_f32`

**The kernel.** `chunk_global_cumsum_scalar_kernel` of flash-linear-attention (fla) v0.6.0
(`source/chunk_global_cumsum.py`, kernel lines 60–101; entry point `chunk_global_cumsum` 142–173,
which launches it through `chunk_global_cumsum_scalar` 104–139). A 1-D grid of `N·H` programs, one
per (sequence, head); each walks its sequence in blocks of `BT` positions (from the last block
backwards when reversed), loads a masked block of gate values of its head, takes `tl.cumsum`
along it (forward or reverse), adds the running sum `b_z` of the blocks it has done, multiplies by
the scale if there is one, stores the same lanes of `o`, and adds the block's `tl.sum` to `b_z`.
The claim is `Claims.ChunkGlobalCumsum.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
sequence `n` (a row `[b·T, b·T + T)` of a fixed-length batch, or `[cu_seqlens[n], cu_seqlens[n+1])`
of a variable-length one), every position `p` in it and every head `h`:
- `o[p, h]` holds a `float32` value `v`;
- if the element is `InRange` (every summed term finite, `A = Σ|s| ≤ 2^126`, and with a scale the
  double and its `float32` value `σ₃₂` finite and `|σ₃₂|·A ≤ 2^126`), `v` is finite and
  `|v − σ·S| ≤ tol`, where `S` is the exact sum of `s[q, h]` from the sequence's start to `p`
  (forward), or from `p` to the sequence's end (reverse), and `σ` is the host's double scale
  (`σ = σ₃₂ = 1` without one);
- every byte outside the written elements is unchanged.

The reference is computed from `cu_seqlens` (or the rows) and `s` alone, never from the kernel's
blocks or indexing.

**The conditions** (`Inst.ok`, `Pre`):
- `BT` is a power of two at most `2^20` (`tl.arange`; the autotune list of fla's host has 32 to
  256, the source's host 256); `B, H ≥ 1`; `H ≤ 2^17`; `T < 2^31` (`T` is a runtime `int32`) and
  `T ≤ 2^22`; `B ≤ 8192`;
- `s` and `o` hold `B·T·H` elements, are valid, and `o` shares no byte with `s`;
- on a variable-length launch: `B = 1`; `cu_seqlens` is `int32` or `int64`, has `N + 1` entries,
  `N ≤ 8192`, is valid and shares no byte with `o`; and it is `TablesOk`: nonnegative,
  nondecreasing and at most `T` (sequences may be empty, `N` may be `0`, and they need not cover
  `[0, T)`).

Every `float32` gate is covered.

**The launches covered** (`SETUP.md`): 70 instances `setup_*_ok`, every launch of the host's test
(the five fixed-length and six variable-length shapes, forward and reverse, without a scale and
with `RCP_LN2`; the variable-length ones with `int32` and `int64` tables), and two `float32`
launches of fla's own tests (`2050` sequences, `2050` rows), all with the host's `BT = 256`.
`TablesOk` is about contents, so `check_gpu.py` checks it on the host's tables.

## Tolerances

`tol = (depth + 1)·2^-23·|σ₃₂|·A + |σ − σ₃₂|·A + 2^-125·(1 + |σ₃₂|)` (`σ = σ₃₂ = 1` without a
scale, so `tol = (depth + 1)·2^-23·A + 2^-124`).

The output at `p` is one tree of IEEE `float32` additions over the summed terms: the block's scan
at `p`'s lane, joined by `b_o + b_z` with the running sum, itself a chain of additions of the
earlier blocks' `tl.sum`s. `depth` bounds that tree's depth:
- the scan's tree at lane `j = p − blkStart` (`tl.cumsum` may combine in any tree) combines `j + 1`
  lanes forward and `BT − j` reverse (lanes past the sequence's end hold `+0`), so its depth is at
  most `laneDepth = j` (forward) or `BT − 1 − j` (reverse);
- after `k = before` blocks (the earlier ones forward, the later ones reverse) the running sum is
  `+0` when `k = 0`, and otherwise the result of `k` additions, the first after a `tl.sum` whose
  tree has depth at most `⌈log₂ BT⌉`: `carryDepth = ⌈log₂ BT⌉ + k`. That `tl.sum` bound is the one
  device fact the claim uses, `Device.SumDepth` (on sm_90, a float reduction's tree is balanced);
- `depth = max(laneDepth, carryDepth) + 1`.

A tree of IEEE additions of depth `D` over terms whose magnitudes sum to `A` is within
`((1 + u)^D − 1)·A + η·N·(1 + u)^D` of the exact sum (`u = 2^-24`, `η = 2^-150`, `N` nodes), at most
`D·2^-23·A + 2^-126` for `D, N ≤ 2^23` (covered launches have `D ≤ 2^22 + 21` and
`N < 2^22 + 2^20`). The scale's IEEE product adds one rounding, `2^-24` relative, and the scale's
conversion to `float32` the term `|σ − σ₃₂|·A`; together these stay within `tol`. The relative part
`(depth + 1)·2^-23` is about twice the first-order bound `(depth + 1)·2^-24`, and counts the first
addition into the running sum (onto `+0`, which is exact).

## Exclusions

- **Contexts longer than `2^22` tokens, more than `2^17` heads, and batches of more than `8192`
  rows or sequences**: the claim covers real model shapes only. Every launch of `SETUP.md` meets
  them.
- **`T ≥ 2^31`**: outside real use (`T` would be an `int64` argument).
- **Elements not `InRange`**: a limit of `float32` (non-finite gates, or magnitudes near
  overflow); only that the output holds a value is claimed.
- **Gates or outputs of other dtypes** (`bfloat16` or `float16` `s`, or `output_dtype` other than
  `torch.float`), **tables not `TablesOk`**, and the 4-D path (`chunk_global_cumsum_vector_kernel`,
  another kernel): outside `SETUP.md`'s use.

`check_gpu.py` (one H200) checks `Post` on the host's launches and on edge cases.
