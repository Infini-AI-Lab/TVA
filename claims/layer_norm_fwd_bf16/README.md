# `layer_norm_fwd_bf16`

**The kernel.** fla's `layer_norm_fwd_kernel` (`source/layernorm.py`, kernel lines 50–114; host
`layer_norm_fwd` lines 117–182) on the paths `SETUP.md` asks for: LayerNorm with weight and bias
(`IS_RMS_NORM = False`, `HAS_WEIGHT = HAS_BIAS = True`), with or without a residual
(`HAS_RESIDUAL = STORE_RESIDUAL_OUT`), over `G ≥ 1` groups (fla's `GroupNorm`: row `t` uses the
weight and bias of group `t % G`; `LayerNorm` is `G = 1`), `bfloat16` tensors, `float32` `mean`
and `rstd`. A 1-D grid of `cdiv(T, 32)` programs; program `p` handles rows `32p … 32p + 31` as one
`[32, BD]` block: it loads `x` (and adds `residual`), stores `res_out`, computes the mean and the
variance with two `tl.sum`s, `rstd = 1 / sqrt(var + eps)`, and stores `mean`, `rstd` and
`y = (x − mean)·rstd·w + b`. The claim is `Claims.LayerNormFwd.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). For every row
`t < T`, with `s j` the `float32` sum `float32(x[t, j]) + float32(residual[t, j])` (the IEEE
addition) with a residual and `float32(x[t, j])` without, `w j`, `b j` the weight and bias of row
`t`'s group, and the exact real LayerNorm of `s` as the reference (`meanRef = Σ s/D`,
`varRef = Σ (s − meanRef)²/D`, `rstdRef = 1/sqrt(varRef + eps)` with the host's double `eps`,
`yRef j = (s j − meanRef)·rstdRef·w j + b j`):
- with a residual, `res_out[t, j]` is `bfloat16(s j)` (round to nearest even), bit for bit;
- `mean[t]` is a finite `float32` within `meanTol` of `meanRef`;
- `rstd[t]` is a finite `float32` within `rstdTol` of `rstdRef`;
- `y[t, j]` is a finite `bfloat16` within `yTol j` of `yRef j`;
- every byte outside the outputs' touched elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ D ≤ 512` (this kernel's path, the host's assert, line 151), `BD` its next power of two;
- `1 ≤ T < 2^31` (`T` is a runtime `int32`);
- `1 ≤ G ≤ 2^16`; `T` need not be a multiple of `G`;
- `eps` finite, `2^-125 ≤ eps ≤ 2^126`;
- every tensor holds what the launch touches (`x`, `y`: `T·D`; `residual`, `res_out`: `T·D` with a
  residual; `weight`, `bias`: `G·D`; `mean`, `rstd`: `T`), and is valid in the initial memory;
- the outputs' touched elements share no byte with the inputs' or with each other's;
- every element of `x` and `residual` read is finite with `|v| ≤ 2^56`, every element of `weight`
  and `bias` finite with `|v| ≤ 2^16`.

**The launches covered** (`SETUP.md`, `setup_*_ok`): the host's test launches (`LayerNorm` with a
residual, `eps = 1e-5`, nine shapes from `1 × 1` to `65536 × 256`), RWKV-7's `g_norm` (`G = 32`,
`D = 64`, `eps = 6.4e-4`, training and decoding), RWKV-6's `g_norm` (`G = 4`, `D = 512`), and fla's
GroupNorm and LayerNorm tests' shapes in `bfloat16`, without a residual: twenty-three in all.

**Device facts** the tolerances rest on (all in `Device P`): `Device.SqrtF32` (`tl.sqrt` on
`float32`, on `0` and inputs `≥ 2^-126`: `2^-22·√x + 2^-149`), `Device.DivFull` (`/` on `float32`:
relative `2^-22`, absolute `2^-148`, for quotients up to `2^126`), `Device.SumDepth` (the tree of a
float `tl.sum` over `BD` positions has depth at most `⌈log₂ BD⌉`), and the totality facts for
`sqrt` and `/`.

## Tolerances

With `L = log₂ BD` (`depth`), `A = Σ|s|/D`, `rmsS = sqrt(Σ s²/D)`, a lane's mean bound
`eTol j = meanTol + 2^-23·(|s j| + A)` and the row's condition number
`cond = (meanTol + 2^-23·(A + rmsS))·rstdRef`:

| Output | Tolerance |
|---|---|
| `mean` | `meanTol = (L + 4)·2^-23·A + 2^-140` |
| `rstd` | `rstdTol = 2·(cond + gam)·rstdRef` when `4·cond ≤ 1`, else `2·eps^(-1/2)`; `gam = (L + 10)·2^-23 + 2^-136/eps` |
| `y` | `yTol = 2^-7·|yRef| + 4·|w|·zTol + 2^-21·|b| + 2^-120`, `zTol = |s − meanRef|·rstdTol + eTol·(rstdRef + rstdTol)` |

The derivation, with `u = 2^-24`:
1. **Mean**: the sum along a tree of depth `L`, then `/ D` (`DivFull`): relative `(L + 4)·u` to
   first order, within half of `(L + 4)·2^-23`.
2. **Deviations**: each fusable subtraction (lines 94 and 108, each its own choice) sees a lane mean
   error `|e j| ≤ eTol j`; their root mean square times `rstdRef` is at most `cond`.
3. **Variance**: with `V0 = Σ (d j − e j)²/D`, `V0 + eps` lies between `(σ² + eps)(1 − 2·cond)` and
   `(σ² + eps)(1 + cond)²`; the later steps are relative on nonnegative terms, and the input of
   `tl.sqrt` is at least `2^-126` (from `eps ≥ 2^-125`).
4. **rstd**: `tl.sqrt` and `1/·` add `(1 ± 4u)` each; the relative rounding error is at most `gam`;
   on an ill-conditioned row (`4·cond > 1`) `rstd` is only known to be at most about `eps^(-1/2)`.
5. **Standardised value** and **output**: the product by `rstd`, by `w`, `tl.addc` with `b`, and the
   `bfloat16` rounding (`2^-7` is twice its `2^-8`).
6. **`res_out`**: `tl.cvt` of the finite `s j`, exact.
7. The fine bounds relax to `rstdTol` and `yTol` on every row.

On well-conditioned rows (the covered launches' data) the tolerances are about `1.2·10^-6·A` to
`1.5·10^-6·A` for the mean, `6.2·10^-6` to `7.4·10^-6` relative for `rstd`, and one `bfloat16` ulp
plus `≈ 3·10^-5·|w|` for `y`.

## Exclusions

- **`eps < 2^-125`**: below `2^-126` a kernel limit (the `float32` `eps` is subnormal, `tl.sqrt`
  flushes `var + eps` of a constant row to `0`, `rstd = inf`, `y = NaN`); in `[2^-126, 2^-125)` the
  library's division and fused-add facts admit a subnormal sum. Real use (`1e-12` to `6.4e-4`)
  reaches neither.
- **`eps > 2^126`**: outside real use; near the `float32` maximum a kernel limit (`var + eps`
  overflows, `rstd = 0`).
- **`|x|`, `|residual|` above `2^56`**: from `2^59` a kernel limit (the `float32` sum of squares
  overflows); between, outside real use and the sum lemma needs room below `maxVal/2`.
- **`|weight|`, `|bias|` above `2^16`**: outside real use (trained scales of order 1); a limit of the
  library's fusable-subtraction model on ill-conditioned rows, not of the kernel.
- **Non-finite inputs**: there is no reference value.
- **`T ≥ 2^31`**: the host then passes `T` as an `int64`, a different compiled kernel.
- **`G > 2^16`**: outside real use (one group per head).
- **`T = 0`, `D > 512`** (another kernel), **`IS_RMS_NORM`**, **no weight or no bias, a `res_out`
  without a residual, other dtypes**: not part of `SETUP.md`.

`check_gpu.py` (one H200) checks `Post` on the twenty-three setup launches and edge cases.
