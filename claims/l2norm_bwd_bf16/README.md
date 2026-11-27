# `l2norm_bwd_bf16`

**The kernel.** `l2norm_bwd_kernel` of fla (`source/l2norm.py`, kernel lines 51–67; host
`l2norm_bwd` lines 70–95, which launches it with `BT = 16`, `num_warps = 4` on the grid
`(cdiv(T, 16),)`). It is the backward of L2 normalisation `y = x / sqrt(Σ x² + eps)`: from the
forward's output `y`, its factors `rstd` and the incoming gradient `dy`, program `p` takes the
rows `p·BT + [0, BT)` of `y` and `dy` viewed as `[T, D]` (`int64` row offsets, masked by `T`) and
the columns `[0, BD)` (masked by `D`), loads both tiles with `other = 0` and widens them to
`float32`, loads the rows' `rstd`, and stores `dx = dy·rstd − (Σ_j dy_j·y_j)·y·rstd` (rounded to
`bfloat16`, or kept `float32`). The claim is `Claims.L2NormBwd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for each
row `t < T`, with `a_j`, `g_j` the row's initial `y` and `dy` values, `r` its initial `rstd`,
`S = Σ_{j<D} g_j·a_j` and `A = Σ_{j<D} |g_j·a_j|`, the exact value
`ref_j = g_j·r − S·a_j·r`, the scale `M_j = |r|·(|g_j| + A·|a_j|)` and the floor factor
`F_j = (1 + |r|)·(1 + |a_j|)`:
- each `dx[t, j]`, `j < D`, is a finite value of `dx`'s dtype with
  `|dx[t, j] − ref_j| ≤ 2^-18·M_j + 2^-138·F_j` for a `float32` `dx`, and
  `≤ 2^-8·|ref_j| + 2^-17·M_j + 2^-133·F_j` for a `bfloat16` `dx`;
- every byte outside the first `T·D` elements of `dx` is unchanged (so `y`, `dy` and `rstd` are
  too).

The reference is the kernel's formula in the reals on the stored inputs: `rstd` is taken as given,
not recomputed from the forward's `x`. The bound is relative to `M_j`, the size of the two terms
the kernel subtracts, not to `|ref_j|`: when `dy` is nearly parallel to `y` the two terms cancel
and `float32` cannot do better.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ D ≤ 512` (this kernel's path; the host asserts `D ≤ 512`); `BD` a power of two with
  `D ≤ BD ≤ 512` (the host's `next_power_of_2(D)`);
- `BT` any power of two with `BT·BD ≤ 2^20` (the host's `16`, and every `BT` of fla's autotune
  list);
- fewer than `2^31` programs (`cdiv(T, BT) < 2^31`); `T = 0` is covered. `T` is passed as an
  `int32` below `2^31` and as an `int64` from `2^31`; both are covered;
- each of `y`, `dy` and `dx` is `bfloat16` or `float32`, in any combination (the host's `dx` has
  `y`'s dtype); `rstd` is `float32`; `eps` is any double (the kernel never reads it);
- `y`, `dy` and `dx` hold at least `T·D` elements, `rstd` at least `T`, all valid in the initial
  memory; the written elements of `dx` share no byte with the read elements of `y`, `dy` and
  `rstd` (the host allocates `dx` fresh);
- every element read is finite, and on every row, for each `j < D`,
  `(1 + |r|)·(1 + |a_j|)·(1 + |g_j| + A) ≤ 2^124`.

**The launches covered** (`SETUP.md`): the host's nine test shapes `(B, T, H, D)`, GatedDeltaNet's
6 heads of 256 on 4 × 4096 tokens, PaTH attention's `float32` `w` (32 heads of 64 on 4 × 4096
tokens, `y`, `dy`, `dx` all `float32`) and a `float32` `dy` into a `bfloat16` `y`
(`setup_*_ok`). `mk_ok` covers the host's launch for every `D ≤ 512` with a power-of-two
`BD ≥ D`, with `T ≤ 2^27` and `T·D ≤ 2^28`, and each dtype of `y` (with `dx` the same) and of `dy`.

**Device facts** the tolerances rest on (all in `Device P`): only `Device.SumDepth` (a float
reduction's tree is balanced, so the row sum over `BD ≤ 512` lanes has depth at most `9`). The
other operations are IEEE products, fusable additions and roundings, whose errors the library
proves from the IEEE definitions.

## Tolerances

With `u = 2^-24`, `η = 2^-150` (one `float32` rounding):
- the row sum: the products `dy_k·y_k` cost `u·A + 512η`; the fusable sum along a tree of depth
  at most `9` costs `((1 + 3u)^9 − 1)·((1 + u)·A + 512η) + 4η·511·(1 + 3u)^9`; together
  `|s − S| ≤ 29u·A + 2600η` (`(1 + 3u)^9 − 1 ≤ 27.01u`);
- per element, `t1 = fl(g·r)`, `t2 = fl(s·a)`, `t3 = fl(t2·r)` cost one rounding each, and the
  fusable `t1 − t3` costs `3u·(|t1| + |t3|) + 4η`. Collected: at most `4.01u` times `|g|·|r|`,
  `34u` times `A·|a|·|r|`, `2600·1.0001·η` times `|a|·|r|`, `1.0001·η` times `|r|`, and `7η`.
  So a `float32` `dx` is within `2^-18·M + 2^-138·F` (`2^-18 = 64u`, `2^-138 = 4096η`, and
  `F = 1 + |r| + |a| + |a|·|r|`);
- a `bfloat16` `dx` adds one rounding, `2^-8·|dx| + 2^-134`, with `|dx| ≤ |ref| + 2^-18·M +
  2^-138·F`: `2^-8·|ref| + (1 + 2^-8)·(2^-18·M + 2^-138·F) + 2^-134 ≤ 2^-8·|ref| + 2^-17·M +
  2^-133·F`. The `2^-8·|ref|` term is the `bfloat16` rounding the kernel cannot avoid.

## Exclusions

- **Rows with `(1 + |r|)·(1 + |a_j|)·(1 + |g_j| + A) > 2^124`.** Near `2^128` the `float32`
  products overflow: `r = 2^100`, `dy = 2^40`, `y = 0.5` on a row of 4 gives `dx = −inf` where
  the exact value is `0` (`dy·rstd = 2^140`). Below that the kernel is correct, but the error
  bounds need room below `float32`'s largest value. Real values are far smaller: `y` is a unit
  row, `rstd ≤ eps^-1/2` (`10^3` at fla's default `eps = 1e-6`), gradients are small.
- **Non-finite inputs** (in `y`, `dy` or `rstd`): the reference is undefined.
- **`cdiv(T, BT) ≥ 2^31`**: CUDA rejects the grid at launch.
- **Other dtypes** (`float16`, `float64`, fp8) of `y`, `dy` or `dx`, a `rstd` of another dtype
  than `float32`, **`D > 512`** (fla launches another kernel) and **`D = 0`**: outside the file's
  launches.

`check_gpu.py` (one H200) checks `Post` on the `SETUP.md` launches and on the edges of `Inst.ok`
and `Pre`, with every dtype combination, and runs the overflow of the first exclusion.
