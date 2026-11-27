# `layer_norm_gated_fwd_bf16`

**The kernel.** `layer_norm_gated_fwd_kernel` of fla (`source/fused_norm_gate.py`, kernel lines
55–127; host `layer_norm_gated_fwd` lines 131–200, which launches it with `BT = 16`,
`num_warps = 4` on the grid `(cdiv(T, 16),)`), on the configuration fla's `FusedRMSNormGated` runs:
RMS norm (`IS_RMS_NORM`), a weight, no bias, no residual, and the gate activation `"swish"` /
`"silu"` or `"sigmoid"`, every tensor `bfloat16` except `rstd` (`float32`). Program `p` takes the
rows `p·BT + [0, BT)` of `x` and `g` viewed as `[T, D]` (`int64` row offsets, masked by `T`) and
the columns `[0, BD)` (masked by `D`). It loads the tile of `x` with `other = 0` and widens it to
`float32`, sums the squares of each row, divides by `D`, adds `eps`, takes `tl.sqrt` and
`1 / sqrt`, and stores `rstd`. It then loads `weight` and the tile of `g`, computes
`x·rstd·w·g·sigmoid(g)` (swish) or `x·rstd·w·sigmoid(g)` (sigmoid), and stores that to `y`
rounded to `bfloat16`. The claim is `Claims.LayerNormGatedFwd.claim` in `Claim.lean`.

**Why this configuration.** Every `FusedRMSNormGated` layer of fla calls the host with
`is_rms_norm = True`, `bias = None` and no residual. Its weight is a `[D]` parameter. The layer's
`activation` is `"swish"` by default, and that is what GatedDeltaNet, GLA, DeltaNet,
GatedDeltaProduct, Mesa-Net and others use. KDA, Comba, GDN-2 and MoBA pass `"sigmoid"`. Both
activations are covered (`Inst.swish`). `"silu"` takes the same branch as `"swish"`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). For each row
`t < T`, let `x_j`, `g_j` and `w_j` be the row's initial `bfloat16` inputs and the weight. Let
`eps` be the host's double. Define these exactly, in the reals:
`rstdRef = 1 / sqrt(Σ_{j<D} x_j² / D + eps)`, `Z_j = x_j·rstdRef`, `σ(v) = 1 / (1 + e^-v)`,
`gate_j = g_j·σ(g_j)` (swish) or `σ(g_j)` (sigmoid), and `yRef_j = Z_j·w_j·gate_j`. Then:
- `rstd[t]` is a finite `float32` with `|rstd[t] − rstdRef| ≤ 2^-12 · rstdRef`;
- each `y[t, j]`, `j < D`, is a finite `bfloat16` with
  `|y[t, j] − yRef_j| ≤ 2^-7·|yRef_j| + 2^-126·(1 + |Z_j|)·(1 + |w_j|)·gMag_j`, where
  `gMag_j = 1 + |g_j|` for swish and `1` for sigmoid;
- every byte outside the first `T·D` elements of `y` and the first `T` elements of `rstd` is
  unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `1 ≤ D ≤ 512` (this kernel's path; for `D > 512` fla launches another kernel, and the host here
  asserts `D ≤ 512`); `BD` a power of two with `D ≤ BD ≤ 512` (the host's `next_power_of_2(D)`);
- `BT` any power of two with `BT·BD ≤ 2^20` (the host's `16`, and every `BT` of fla's autotune
  list `16, 32, 64`);
- fewer than `2^31` programs (`cdiv(T, BT) < 2^31`). This is a kernel limit: CUDA rejects a larger
  grid. `T = 0` is covered. `T` is passed as an `int32` below `2^31` and as an `int64` from
  `2^31`, and both are covered;
- `x` and `g` hold at least `T·D` elements, `weight` at least `D`, `y` at least `T·D`, `rstd` at
  least `T`, all valid in the initial memory; the written elements of `y` and `rstd` share no byte
  with each other or with the read elements of `x`, `g`, `weight` (the host allocates both fresh);
- `eps` is a finite double in `[2^-125, 2^126]` (fla's layers pass `1e-5` or `1e-6`);
- every element of `x` read is finite. Every element of `g` and `weight` read is finite with
  `|·| ≤ 2^56`. Every row has `Σ_j x[t, j]² + eps ≤ 2^125`.

The lower bound on `eps` and the upper bounds on `Σ x² + eps`, `|g|` and `|w|` are numeric
limits. The kernel is wrong past them, or its products overflow (Exclusions). The other conditions
are the host's own launches.

**The launches covered** (`SETUP.md`): the host test's seven cases (`setup_4x64_silu_ok` …
`setup_33x300_swish_ok`), plus the output norms of GatedDeltaNet (6 heads of 512, swish), GLA
(4 heads of 512, swish) and KDA (16 heads of 128, sigmoid). Each layer is covered on a training
batch of 4 × 4096 tokens and on one decoding token of a batch of 8 (`setup_gdn_train_ok` …
`setup_kda_decode_ok`). `mkInst_ok` covers the host's launch for every `D ≤ 512` with a
power-of-two `BD ≥ D`, with `T·D ≤ 2^30`, either activation and any `eps` in `[2^-125, 1]`. **Not covered**:
- fla's own test of this path as it stands: it passes `float32` tensors and `eps = 0`. With
  `eps = 0` a zero row has no finite reference, and the kernel stores `rstd = inf`, `y = NaN`. The
  host test's cases take its shapes in `bfloat16` with `eps = 1e-5`;
- a `float32` weight (a model kept in `float32` under autocast);
- `elementwise_affine = False` (no weight: YOCO's norm);
- the LayerNorm path (`FusedLayerNormGated`), a bias, a residual input or output, and an
  `out_dtype` other than `bfloat16`;
- `D > 512`, which goes to `layer_norm_gated_fwd_kernel_row`, another kernel.

**Device facts** the tolerances rest on (all in `Device P`):
- `Device.SqrtF32`: `tl.sqrt` on `float32` is within `2^-22·√x + 2^-149` on `sqrtDom`;
- `Device.DivFull`: `/` on `float32` has relative error `2^-22` and absolute error `2^-148` for
  quotients up to `2^126`;
- `Device.DivF32InfDivisor`: a finite value divided by `±inf` is `±0`, so `1 / +inf = +0`;
- `Device.ExpF32`: `tl.exp` on `float32` has relative error `2^-22 + 2^-23·|v|` and absolute
  error `2^-140`, with room below `maxVal`;
- `Device.ExpF32Overflow`: inputs `≥ 1453635/16384 ≈ 88.72` give `+inf`;
- the totality facts for `sqrt`, `exp` and `/`.

## Tolerances

Here `u = 2^-24`, `η = 2^-150` (one `float32` rounding), `S = Σ_j x_j²` and `V = S/D + eps`.

**`rstd`, `2^-12` relative.**
- The products and the fusable row sum over at most `512` lanes, along any tree, are within
  `(7/8)·2^-13·S + (11/16)·2^-137` of `S`.
- `/ D` (`DivFull`, the quotient below `2^126` since `S + eps ≤ 2^125`) brings this within
  `2^-13·S/D + (3/4)·2^-137` of `S/D`.
- The fusable `+ eps` gives a normal number, so it is in `tl.sqrt`'s domain (this needs
  `eps ≥ 2^-125`).
- `tl.sqrt` and `1 / ·` add `2^-22` each.
- Moving from the device's `float32` `eps` to the host's double adds `2^-24`.
- The total is below `2^-12 − 2^-16`, plus that last step.

**`y`, `2^-7` relative.** The chain is `Z = x·rstd`, then `·w`, then `·g` (swish), then `·s` with
`s` the kernel's `sigmoid(g)`, then the `bfloat16` rounding. Its relative part is the sum of:
- `rstd`'s `2^-12`;
- the gate's `2^-15`;
- the `float32` roundings (`4u`);
- the `bfloat16` rounding `2^-8`.

The total is below `2^-8 + 2^-11 < 2^-7`. The relative part `2^-7` is twice the `bfloat16`
rounding the kernel cannot avoid.

The gate. `s = 1 / (1 + exp(−g))` is within `2^-15·σ(g) + 2^-128` of `σ(g)`:
- For `−g ≤ 88.5`, from `ExpF32` and `DivFull`. As `g` is a `bfloat16`, `|g|` never lies strictly
  between `88.5` and `88.72`, so `ExpF32`'s room condition holds.
- For `−g ≥ 88.72`, `exp` overflows to `+inf` and `s = +0`, while `σ(g) ≤ 2^-128`.

**The floor `2^-126·(1 + |Z|)·(1 + |w|)·gMag`.** It covers the absolute errors that a relative
bound cannot carry:
- the `float32` underflow of each product (`2^-150` each, scaled by the later factors);
- the gate's `2^-128` (times `|Z·w|` for sigmoid, `|Z·w·g|` for swish);
- the `bfloat16` underflow `2^-134`.

`|Z| ≤ 23` (since `Z² ≤ D ≤ 512`) and `|g|, |w| ≤ 2^56` keep every product below `2^127`, so no
`float32` operation overflows.

## Exclusions

- **`eps < 2^-125`**, and **`eps = 0`** in particular. Below `2^-126` the kernel is wrong on a zero
  row: `var + eps` is `0` or subnormal, `tl.sqrt` flushes it to `0`, and the kernel stores
  `rstd = +inf`, `y = NaN`. Two launches show it: `eps = 0` (fla's test) and `eps = 1e-39`, each
  with `T = 1`, `D = 4`, `x = 0`. In `[2^-126, 2^-125)` the claim leaves the kernel out because the
  fusable addition `AddRel` admits an operand below the `float32` `eps`, and so a subnormal sum.
  fla's layers use `eps ≥ 1e-6`.
- **Rows with `Σ x² + eps > 2^125`**, and **`eps > 2^126`**. Near `2^128` the `float32` sum of
  squares overflows, and the kernel stores `rstd = 0`, `y = 0` (`T = 1`, `D = 1`, `x = 2^64`).
  Below that the error bounds need room under `maxVal / 2`. Real activations are far smaller.
- **`|g| > 2^56` or `|w| > 2^56`.** The products can overflow `float32` and the stored `y` is
  `inf`: `w = g = 2^100` at `D = 1` gives `y = inf`, with the exact value `1.6·10^60` beyond
  `bfloat16` anyway. Real gates and weights are below `2^10`.
- **Non-finite inputs** (in `x`, `g`, `weight` or `eps`): the reference is undefined.
- **`cdiv(T, BT) ≥ 2^31`**: CUDA rejects the grid at launch.
- **The configurations listed as not covered above**: their flags select other code in the
  kernel, or they launch another kernel.

`check_gpu.py` (one H200) checks `Post` on the `SETUP.md` launches and on the edges of `Inst.ok`
and `Pre`, with both activations. It also runs the launches of the exclusions.
