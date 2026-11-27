# `mean_pooling_fwd_bf16`

**The kernel.** `mean_pooling_fwd_kernel` of flash-linear-attention (fla) v0.6.0
(`source/pooling.py`, kernel lines 81–117 with the helper `unflatten_program_id` 71–78; entry point
`mean_pooling` 191–211, which launches it through `mean_pooling_fwd` 159–188). A 2-D grid
`(cdiv(D, BD)·NT, B·H)`, one program per (feature block, chunk) and (batch row, head); each loads a
masked `[BT, BD]` tile of `x` (the chunk's positions, `BD` features of one head), sums it along the
chunk in `float32` (`tl.sum`), divides by the chunk's length `L = min(BT, T − c·BT)` and stores the
`bfloat16` rounding into one row of `o`. The claim is `Claims.MeanPoolingFwd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
output row `g` (chunk `c` of row `b`, `g = b·cdiv(T, BT) + c`, on a fixed-length launch; on a
variable-length one the chunk `c = chunk_indices[g, 1]` of sequence `n = chunk_indices[g, 0]`),
every head `h` and every feature `j < D`:
- `o[g, h, j]` holds a `bfloat16` value `v`;
- if the element is `InRange` (the chunk's `L` inputs `x[p, h, j]` finite, `A = Σ|x| ≤ 2^125`), `v`
  is finite and `|v − μ| ≤ tol = 2^-8·|μ| + (log₂ BT + 5)·2^-24·A/L + 2^-132`, where
  `μ = (1/L)·Σ x[p, h, j]` is the exact mean over the chunk's positions `p`
  (`[start, start + L)`, `start = b·T + c·BT` or `cu_seqlens[n] + c·BT`);
- every byte outside the output's `nRows·H·D` elements is unchanged (`x` and the tables included).

The reference is computed from `x`, `cu_seqlens` and the row's own entry of `chunk_indices` only:
output row `g` is the chunk that row `g` of `chunk_indices` names, as the kernel writes it. With the
table `prepare_chunk_indices` builds (the entry point's), that is the `g`-th chunk of the batch in
order, sequence by sequence.

**The conditions** (`Inst.ok`, `Pre`):
- `BT` is a power of two; `BD` is a power of two in `[16, 128]` (the values of the source's autotune
  list; the host passes `64`); `BT·BD ≤ 2^20` (so `BT ≤ 2^16`);
- `B, H, D ≥ 1`; `T < 2^31` and `T ≤ 2^22`; `B ≤ 8192`; `D ≤ 2^17`; `B·H < 2^16`; the grid
  `cdiv(D, BD)·NT·B·H` has fewer than `2^31` programs (`NT = cdiv(T, BT)` on fixed length,
  `len(chunk_indices)` on variable length);
- `x` holds `B·T·H·D` elements and `o` the `nRows·H·D` of the output (`nRows = B·cdiv(T, BT)` or
  `NT`); both are valid and `o` shares no byte with `x`;
- on a variable-length launch: `B = 1`; the tables are `int32` or `int64`; `N < 2^31` and
  `N ≤ 8192` sequences; the tables are long enough, valid, and share no byte with `o`; and they are
  `TablesOk`: `cu_seqlens` nonnegative, nondecreasing and at most `T`, and each row of
  `chunk_indices` names a sequence `n < N` and a chunk `c ≥ 0` that starts inside it
  (`c·BT < cu_seqlens[n+1] − cu_seqlens[n]`). Rows may come in any order or repeat; the table
  `prepare_chunk_indices` builds is such a table.

Every `bfloat16` input is covered. `T = 0` (fixed length) and a batch of empty sequences (`NT = 0`)
are covered: the grid is empty.

**The launches covered** (`SETUP.md`): 30 instances `setup_*_ok`: fla's pooling test (five
fixed-length and five variable-length shapes, the latter with `int32` and `int64` tables), the
pooling of the keys and values in fla's NSA tests (five fixed-length and four variable-length
shapes, both table dtypes) and a packed NSA batch of 65,536 tokens, all in `bfloat16` with the
host's `BD = 64`. `mk_ok` covers the host's launch at every shape `Inst.ok` allows. `TablesOk` is
about contents, so `check_gpu.py` checks it on the host's tables.

**Device facts.** `Device.SumDepth` (the tree of a `tl.sum` over `BT` lanes has depth at most
`log₂ BT`) and `Device.DivFull` (`/` on `float32`: within `2^-22·|q| + 2^-148` of the quotient `q`
for `|q| ≤ 2^126`); the rest is IEEE arithmetic (the tree's additions, the exact widening of
`bfloat16`, the exact conversion of the integer `L`, the rounding to `bfloat16`).

## Tolerances

With `u = 2^-24`, `η = 2^-150` (`float32`), `u_bf = 2^-8`, `η_bf = 2^-134` (`bfloat16`),
`e = log₂ BT ≤ 16`, `S = Σ x` and `A = Σ|x|` over the chunk (`|S| ≤ A`):
- the column sum `s` runs along a tree of depth at most `e` over the `BT` lanes (the `BT − L` lanes
  past the chunk hold `+0`), within `E = ((1+u)^e − 1)·A + η·(BT − 1)·(1+u)^e` of `S`
  (`sum_err_depth`); `(1+u)^e ≤ 1 + e·u·(1 + 2^-10)`, so `E ≤ e·u·(1 + 2^-10)·A + 2^-134·(1 + 2^-10)`;
- `L` converts to `float32` exactly, and the division gives `q` within `2^-22·|s/L| + 2^-148` of
  `s/L` (`Device.DivFull`; `|s/L| ≤ |s| ≤ A + E ≤ 2^126` since `A ≤ 2^125`);
- the rounding to `bfloat16` gives `v` within `u_bf·|q| + η_bf` of `q` (`|q|` is within the format).

Adding up, with `|q| ≤ |s/L| + 2^-22·|s/L| + 2^-148` and `|s/L| ≤ |μ| + E/L`:
`|v − μ| ≤ 2^-8·|μ| + (1 + 2^-8)·((1 + 2^-22)·E/L + 2^-22·A/L + 2^-148) + 2^-134`. The
coefficient of `A/L` is at most `(1 + 2^-8)·((1 + 2^-22)(1 + 2^-10)·e + 4)·u ≤ (e + 5)·u` for
`e ≤ 16`, and the absolute terms are below `2^-132` (`L ≥ 1`). The dominant term is the one rounding
to `bfloat16`, `2^-8·|μ|`; the `float32` terms matter only where the chunk's values cancel.
`Num.mean_err` proves the bound; `check_gpu.py` measures a worst error / tolerance of `0.996`.

## Exclusions

- **Contexts longer than `2^22` tokens, batches of more than `8192` rows or sequences, and heads of
  more than `2^17` features**: the claim covers real model shapes only. Every launch of `SETUP.md`
  meets them.
- **`B·H ≥ 2^16`**: CUDA's limit on the grid's axis 1 (`B * H`); such a launch fails on the host.
  **A grid of `2^31` programs or more**: a limit of the launch. Real use does not reach either.
- **`BD` outside the autotune list's `16 … 128`**, and the autotuner itself: the claim fixes the
  configuration the host passes, and covers every `BD` of the list.
- **Elements not `InRange`**: a limit of `float32` (non-finite inputs, or `Σ|x|` near overflow); only
  that the output holds a value is claimed.
- **Tables not `TablesOk`** (a row naming a chunk outside its sequence, `cu_seqlens` decreasing or
  past `T`), and **`B > 1` with `cu_seqlens`** (the entry point raises on it).
- **Other dtypes** (`float16`, `float32` inputs) and the backward kernel `mean_pooling_bwd_kernel`:
  other claims.

`check_gpu.py` (one H200) checks `Post` on the host's launches and on edge cases.
