# `chunk_cumsum_f32`

**The kernel.** `chunk_local_cumsum_scalar_kernel` of flash-linear-attention (fla) v0.6.0
(`source/chunk_cumsum.py`, kernel lines 80–120 with the helper `unflatten_program_id` 70–77; entry
point `chunk_local_cumsum` 209–247, which launches it through `chunk_local_cumsum_scalar` 162–206).
A 1-D grid of `NT·B·H` programs, one per (chunk, batch row, head); each loads a masked column of
`BT` gate values of one head, takes `tl.cumsum` along it (forward or reverse), multiplies by the
scale if there is one, and stores the same lanes of `o`. The claim is `Claims.ChunkCumsum.claim`
in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
sequence `n` (a row `[b·T, b·T + T)` of a fixed-length batch, or `[cu_seqlens[n], cu_seqlens[n+1])`
of a variable-length one), every position `p` in it and every head `h`:
- `o[p, h]` holds a `float32` value `v`;
- if the element is `InRange` (every summed term finite, `A = Σ|s| ≤ 2^126`, and with a scale the
  double and its `float32` value `σ₃₂` finite and `|σ₃₂|·A ≤ 2^126`), `v` is finite and
  `|v − σ·S| ≤ tol = (depth + 1)·2^-23·|σ₃₂|·A + |σ − σ₃₂|·A + 2^-128·(1 + |σ₃₂|)`, where `S` is
  the exact sum of `s[q, h]` from the start of `p`'s chunk (chunks of `BT` counted from the
  sequence's start) to `p` (forward), or from `p` to the chunk's end (reverse), `σ` is the host's
  double scale (`σ = σ₃₂ = 1` without one) and `depth + 1` is the number of lanes the scan
  combines at `p`;
- every byte outside the written elements is unchanged.

The reference is computed from `cu_seqlens` (or the rows) and `s` alone, never from
`chunk_indices` or the kernel's indexing.

**The conditions** (`Inst.ok`, `Pre`):
- `BT` is a power of two at most `2^20`; `B, H ≥ 1`; `B·H < 2^31`; `T < 2^31` and `T ≤ 2^22`;
  `B ≤ 8192`; the grid `NT·B·H` (`NT = cdiv(T, BT)` on fixed length) has fewer than `2^31`
  programs;
- `s` and `o` hold `B·T·H` elements, are valid, and `o` shares no byte with `s`;
- on a variable-length launch: `B = 1`; the tables are `int32` or `int64`; `N < 2^31` and
  `N ≤ 8192` sequences; the tables are long enough, valid, and share no byte with `o`; and they are
  `TablesOk`: `cu_seqlens` nonnegative, nondecreasing and at most `T`, each row of `chunk_indices`
  names a sequence and a chunk that starts inside it, rows are distinct, and every such chunk has a
  row (what `prepare_chunk_indices` builds, in any row order).

Every `float32` gate is covered.

**The launches covered** (`SETUP.md`): 76 instances `setup_*_ok`, every launch of the host's test
(the five fixed-length and seven variable-length shapes, forward and reverse, without a scale and
with `RCP_LN2`; the variable-length ones with `int32` and `int64` tables). `TablesOk` is about
contents, so `check_gpu.py` checks it on the host's tables.

**Device facts.** None: `tl.cumsum` (IEEE additions; no product feeds it) and `*` are IEEE
operations.

## Exclusions

- **Contexts longer than `2^22` tokens, and batches of more than `8192` rows or sequences**: the
  claim covers real model shapes only. Every launch of `SETUP.md` meets them.
- **A grid of `2^31` programs or more**: a limit of the launch (CUDA). Real use does not reach it.
- **`B·H ≥ 2^31`**, **`T ≥ 2^31`**: outside real use.
- **Elements not `InRange`**: a limit of `float32` (non-finite gates, or magnitudes near
  overflow); only that the output holds a value is claimed.
- **`USE_GRAPH = True`**, **tables not `TablesOk`**, the 4-D path (another kernel): outside
  `SETUP.md`'s use.

`check_gpu.py` (one H200) checks `Post` on the host's launches and on edge cases.
