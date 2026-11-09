# Prepared claim: `gdn_gating_bf16`

Kernel: `source/fused_gdn_gating.py`, SGLang v0.5.23's
`fused_gdn_gating_kernel` (lines 53-84), launched by its host `fused_gdn_gating` (lines 87-119).
Claim: `Claim.lean`, namespace `Claims.GdnGating`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. The model map is `model-map.json`.

One program per (token, block of 8 heads): it loads `A_log`, `a`, `b`, `dt_bias`, stores
`g = -exp(A_log) · softplus(a + dt_bias)` (softplus with `beta = 1`, `threshold = 20`) and
`beta_output = sigmoid(b)` (rounded to `bfloat16`, stored as `float32`).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.GdnGating.main : Statement` (`Statement`: `Claim.lean`
line 299). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 139), and every initial memory `init` that `Pre` accepts (line 216), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 251).

**What `Post` asks** (line 251), three parts, for every token `i < batch` and head `h < H`, with
`A = A_log[h]`, `av = a[i·stride_a + h]`, `bv = dt_bias[h]`, `bb = b[i·stride_b + h]` read in `init`,
and `x = av.toReal + bv.toReal` (the exact sum):
1. element `i·H + h` of `g` reads in `fin` as `c`, and either `c` is finite with
   `|c.toReal - gRef A.toReal x| ≤ gTol A.toReal x`, or `c = FloatV.negInf` and
   `maxVal ≤ |gRef A.toReal x| + gTol A.toReal x` (the exact value is at the edge of `float32`'s
   range). `gRef A x = -(e^A · spRef x)` (line 226) is the exact decay in the reals, with
   `spRef x = log(1 + e^x)` for `x ≤ 20` and `x` above (line 223); `gTol` (line 236) is relative
   `2^-19 + 2^-23·(|A| + 2·min(max(-x, 0), 20))` of `|gRef|`, plus `2^-23·e^A`, `2^-139·spRef x`
   and `2^-148`;
2. element `i·H + h` of `beta_output` reads in `fin` as a finite `c` with
   `|c.toReal - sigRef bb.toReal| ≤ bTol bb.toReal`: `sigRef b = 1 / (1 + e^(-b))` (line 229),
   exact in the reals; `bTol b = 2^-7·|sigRef b| + 2^-133` (line 242);
3. the frame: every byte outside the first `batch·H` elements of `g` and of `beta_output` is
   unchanged.

**What you are given** (`Assumptions`, line 274):
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `exp_total` (`Num.MathTotal P .exp .binary32`): `tl.exp` has a result on every input (source
  lines 79, 81, 83; masked lanes included);
- `exp_acc` (`Num.ExpF32`): for `X ≥ 0` and finite `x ∈ [-X, X]` with room, `tl.exp` is within
  `(2^-22 + 2^-23·X)·e^x + 2^-140` (source lines 79, 81, 83);
- `exp_ninf` (`Num.ExpF32NegInf`): `tl.exp(-∞) = +0` (source line 79, when `a + dt_bias` rounds to `-∞`);
- `log_total` (`Num.MathTotal P .log .binary32`) and `log_acc` (`Num.LogF32`, `2^-22·|ln x| +
  2^-149`): `tl.log` of source line 79;
- `div_total` (`Num.DivTotal .binary32`) and `div_full` (`Num.DivFull`, `2^-22` relative,
  `2^-148`): the `/` of source line 83 (`tl.sigmoid`).

**The kernel, as the model runs it** (`fused_gdn_gating_kernel` line 47, `model` line 178,
`launch` line 186): a 3-D grid `(batch, 1, Bn d)`, `Bn d = (H + 7) / 8` (line 133). Program
`(i, 0, k)` takes the heads `8k + [0, 8)` masked by `< H`: it loads (masked, `tl.load`) `A_log`,
`a` and `b` at row `i`, and `dt_bias`; widens `a`, `dt_bias` to `float32` and adds; computes the
softplus as `tl.where (1·x ≤ 20) (1·log(1 + exp(1·x))) x`; stores `g = (-exp(A_log))·softplus`
at `i·H + 8k + [0, 8)`; then computes `1 / (1 + exp(-b))`, rounds it to `bfloat16`, widens it back
and stores it to `beta_output` at the same offsets. No loop, no reduction. Programs write disjoint
pieces of the outputs and use no atomics: the route is `TVA.meets_independent` (PROVER_GUIDE §4).
The offsets are not the 1-D block pattern: use `safe_load` and `safe_store`.

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.Valid` unfolded: `batch_lt`, `grid_z`, the
  program count `batch·Bn d < 2^31` from `batch_lt` and `flat_le`, the block `[8]`);
- each program is `Safe`: the enabled lanes' `int32` offsets do not wrap (`flat_le`, `arow_le`,
  `brow_le`, `stride_a_lt`, `stride_b_lt`) and lie in the buffers (`A_len`, `bias_len`, `a_len`,
  `b_len`, `g_len`, `bout_len`); `tl.exp`, `tl.log` and `/` have results on every lane
  (`exp_total`, `log_total`, `div_total`);
- the write footprints of different programs are disjoint (distinct `(i, k)` give distinct
  elements), and no program reads what any program writes (`g_disj`, `bout_disj`);
- the value of `g` gives part 1: the IEEE steps by `FloatV.add_err`, `mul_err`, `cvt_err` (the
  products by `1.0` and the negation are exact), `exp_acc` at `X = |x_f|` and at `X = |A|` (room
  from `ExpRoom`, line 192, and `InOk`, line 202), `exp_ninf` when the sum is `-∞`, `log_acc`,
  with the two branches `x_f ≤ 20` and `x_f > 20`; README, Tolerances gives the derivation and the
  overflow case;
- the value of `beta_output` gives part 2: `exp_acc` at `X = |b|` (room from `InOk`), the `1 +`
  rounding, `div_full`, then the `bfloat16` rounding (`cvt_err`) and the exact widening;
- the bytes no program writes give part 3.

The real-shape condition `batch_ctx` (`batch ≤ 2^22`) gives the grid's first axis (`batch_lt`) at
once; the `int32` offsets still take their bounds from `flat_le`, `arow_le` and `brow_le` (the
head count is not bounded by it).

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `bytes_disj` (line 329), `rows_fit` (line 337) and `maxVal_ge` (line 394)
can be reused.

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`. `Post d init fin`: for every token `i < batch` and head
`h < num_heads`, with `A = A_log[h]`, `x = a[i, h] + dt_bias[h]` (the exact real sum of the two
`bfloat16` inputs) and `bb = b[i, h]`, all read from `init` at the host's layout (row strides
`stride_a`, `stride_b`, outputs `[1, batch, num_heads]` contiguous):

- `g[i * num_heads + h]` is a finite `float32` with `|g - gRef A x| ≤ gTol A x`, where
  `gRef A x = -e^A · spRef x`, `spRef x = ln(1 + e^x)` for `x ≤ 20` and `x` above it, and
  `gTol A x = (2^-19 + 2^-23·(|A| + 2·min(max(-x, 0), 20)))·|gRef| + 2^-23·e^A + 2^-139·spRef x + 2^-148`;
  or it is `-∞` and `maxVal ≤ |gRef| + gTol` (the exact value is at the top of `float32`'s range:
  the product overflows, as IEEE rounding requires);
- `beta_output[i * num_heads + h]` is a finite `float32` with
  `|beta - σ(bb)| ≤ 2^-7·σ(bb) + 2^-133`, `σ(b) = 1 / (1 + e^-b)`;
- every byte outside the two outputs' first `batch * num_heads` elements is unchanged.

The references are computed from the host's inputs (the `bfloat16`/`float32` values at the host's
indices), with the source's own softplus threshold, as the host's `reference` does.

## Conditions

`Inst`: `batch`, `H = num_heads`, the row strides `stride_a`, `stride_b` (elements), and the six
buffers. `model d` passes `seq_len = 1`, `NUM_HEADS = H`, `BLK_HEADS = 8` and the compile-time
`beta = 1.0`, `threshold = 20.0` (the wrapper's defaults, which every caller uses) as constants
in the model. The model computes `beta * x` twice, as the source does (line 79: in the condition
and inside `tl.exp`).

`Inst.ok`:
- `batch < 2^31`, `cdiv(H, 8) < 2^16`: the grid within CUDA's limits;
- `batch * H ≤ 2^31`, `(batch - 1) * stride_a + H ≤ 2^31`, `(batch - 1) * stride_b + H ≤ 2^31`:
  every enabled lane's `int32` offset is below `2^31`;
- `stride_a, stride_b < 2^31`: `int32` arguments;
- `A_log`, `dt_bias` hold `H` elements; row `i` of `a` (`b`) fits (`i * stride + H ≤ len` for
  `i < batch`; any stride, so contiguous rows and the views split from one `[batch, 2H]` projection);
  `g`, `beta_output` hold `batch * H`;
- `g` shares no byte with the inputs or `beta_output`, and `beta_output` none with the inputs (the
  wrapper allocates both fresh). `a` and `b` may overlap (they are only read);
- `batch ≤ 2^22` (`batch_ctx`): `batch` is the number of tokens of the extend batch, a context
  length (out of scope beyond, Exclusions). `num_heads` is a number of heads, not a model
  dimension, and is not bounded by it.

`Pre`: the six buffers valid, and every input element read is finite with
- `ExpRoom A_log[h]`: `(1 + 2^-22 + 2^-23·|A|)·e^A + 2^-140 ≤ maxVal`, i.e. `A_log ≤ 88.72282`;
- `a[i, h] + dt_bias[h] < 2^128 - 2^103`: the `float32` sum does not round to `+∞` (it may round
  to `-∞`: a sum below `-(2^128 - 2^103)` is covered);
- `ExpRoom (-b[i, h])`: `b ≥ -88.72282`.

Every other finite value is covered: `A_log` down to `-maxVal`, `a`, `dt_bias`, `b` anywhere in
`bfloat16`'s finite range (subject only to the three conditions above).

Part (A) of `Compiler.Faithful`'s scope: the launch passes only `num_warps=1` and `num_stages=3`;
no `@triton.heuristics` or autotune; no descriptors (no allocator); a direct launch of the
`@triton.jit` function; no host `TensorDescriptor`. Part (B), the environment, is assumed.

## Assumptions

`faithful` (`Compiler.Faithful`); `exp_total`, `log_total` (`Num.MathTotal`: `tl.exp`/`tl.log` have
a result on every input, masked lanes and the discarded `tl.where` side included); `exp_acc`
(`Num.ExpF32`); `exp_ninf` (`Num.ExpF32NegInf`, for sums rounding to `-∞`); `log_acc`
(`Num.LogF32`); `div_total` (`Num.DivTotal .binary32`); `div_full` (`Num.DivFull`). All are library
facts named as `docs/rules/num.md` lists them; `consistent` witnesses them with `Platform.sm90ref`.

## Coverage

`mk batch H split` builds the host's tensors at fixed disjoint addresses (contiguous: stride `H`;
split: `b`, `a` the halves of one `[batch, 2H]` tensor, stride `2H`); `mk_ok` proves `Inst.ok` for
every `1 ≤ batch ≤ 2^22`, `H ≤ 2^16`, `batch * H ≤ 2^26`. Every launch of `SETUP.md` has at most
65,536 tokens, so none is left out.

| SETUP launch | Instance |
|---|---|
| test `(1, 32)`, `(1, 48, split)`, `(4, 8, split)`, `(16, 48, split)`, `(32, 24, split)`, `(7, 4)`, `(513, 16)`, `(8192, 32)`, `(32768, 32, split)` | `setup_b1_h32` … `setup_b32768_h32_split` |
| Qwen3-Next, 32 heads at TP 1, 2, 4, 8 (32, 16, 8, 4 heads), prefill chunks of 32768-65536 tokens | `setup_qwen3next_tp1` … `_tp8` |
| Qwen3.5-27B, 48 and 24 heads (the test's) | `setup_qwen35_tp1`, `_tp2` |
| `beta = 1.0`, `threshold = 20.0`, dtypes, any finite values | fixed in `model`; `Pre` (above) |

## Tolerances

Units: `U = 2^-23`, `u = 2^-24` (one `float32` rounding, `add_err`/`mul_err`: `u·|q| + 2^-150`).
The library facts used: `ExpF32` at `X` (`rel 2^-22 + 2^-23·X`, `abs 2^-140`, with room),
`ExpF32NegInf`, `LogF32` (`2^-22`, `2^-149`), `DivFull` (`2^-22`, `2^-148`), the IEEE `+ * -` and
conversions (`FloatV.add_err`, `mul_err`, `cvt_err`; the products by `1.0` and the negation are
exact). Write `xf = fl(x)` (the `float32` sum, `|xf - x| ≤ u|x| + 2^-150`), `S` the kernel's softplus,
`E = tl.exp(A)`.

**Softplus, `xf ≤ 20`** (`S = log(fl(1 + exp(xf)))`).
1. `tl.exp(xf)` with `X = |xf|`: within `(2 + |xf|)U·e^xf + 2^-140`; room holds since `xf ≤ 20`
   (and for `xf ≪ 0`, `(1 + 2^-23|xf|)e^xf` is tiny). Its effect on `ln(1 + ·)` is at most
   `ε·e^x/(1+e^x)/(1-ε)`, so relative to `sp(xf)` at most `(2 + |xf|)U·σ/sp`: `≤ 1.69U` for
   `xf ∈ [0, 20]` (`max (2+x)σ(x)/sp(x) = 1.683`), `≤ (2 + |xf|)U` for `xf ∈ [-20, 0)`, and an
   absolute `≤ 22U·e^-20 < 2^-47` below `-20`. `xf = -∞` gives `+0` (`ExpF32NegInf`), then
   `log(1) ∈ [-2^-149, 2^-149]`.
2. `1 + exp`: one rounding, `|δ| ≤ u`, so `|ln fl(1+E') - ln(1+E')| ≤ u(1 + 2^-20)`: **absolute**,
   whatever `x` (this is the kernel's own error for `x ≪ 0`, where `1 + e^x` rounds to `1`).
3. `tl.log`: `2U` relative and `2^-149`.
4. `x → xf`: `|sp(xf) - sp(x)| ≤ σ(ξ)·|xf - x|`; for `x ≥ 0`, `σ ≤ 1` and `|x| ≤ sp(x)`: `0.5U`
   relative; for `x ∈ [-20, 0)`, `σ ≤ sp`: `0.5|x|U` relative; below `-20`: absolute `< 2^-48`.
   At the threshold (`xf ≤ 20 < x` or the reverse, only within `20u` of `20`):
   `|ln(1+e^20) - 20| < e^-20`, under `2^-32` relative.
Sum: relative `ρ_S ≤ (5.2 + 1.5·m)U` with `m = min(max(-x, 0), 20)`, absolute
`α_S ≤ 2^-24·(1 + 2^-19)`.

**Softplus, `xf > 20`** (`S = xf`): `0.5U` relative (plus the threshold case above).

**`g = fl(-E·S)`**. `E` with `X = |A|` (room by `Pre`): `|E - e^A| ≤ r_A e^A + 2^-140`,
`r_A = (2 + |A|)U`. Then
`|E·S - e^A·sp| ≤ e^A·sp·(ρ_S + r_A(1 + ρ_S)) + e^A·α_S·(1 + r_A) + 2^-140·(sp·(1 + ρ_S) + α_S)`,
and the product's rounding adds `u·|E·S| + 2^-150`. Collecting:
- relative: `ρ_S + r_A + 0.5U + (cross terms) ≤ (7.7 + 1.5m + |A|)U` for `|A| ≤ 2^20`, against
  `gTol`'s `(16 + 2m + |A|)U` (a factor `≥ 2.08` on the constant part, `1.33` on the `m` part);
  for `|A| > 2^20`, `e^A·sp ≤ e^(-2^20)·2^128`, far below `2^-148`;
- absolute: `e^A·α_S·(1 + r_A)(1 + u) ≤ 1.13·2^-24·e^A < 2^-23·e^A` (`r_A ≤ 1/8` for
  `A ≥ -2^20`; `A ≤ 88.73` gives `r_A < 2^-16`); `2^-140·sp·(1 + ρ_S)(1 + u) ≤ 2^-139·sp`; the rest
  (`2^-140·α_S`, `2^-150`, and the `A < -2^20` terms) `≤ 2^-148`.
- overflow: when `fl(-E·S)` is `-∞`, `|E·S| ≥ maxVal`, and `|E·S| ≤ |gRef| + gTol` by the same
  bounds (`S ≥ -2^-148`, `E > 0` up to `2^-140`, so the sign is negative and `+∞` is impossible).
So `gTol` is provable from the cited facts. Meaningful: at `|A_log| ≤ 20` the GPU's largest
relative error is `3.4e-7` for `x ∈ [0, 20)` (tolerance up to `4.5e-6`), `1.8e-4` for `x ∈ [-8, 0)`
(`3.6e-4`), and where the relative term is at least half of `gTol` the worst error is `0.375` of it.
The absolute `2^-23·e^A` is twice the kernel's real worst error (`1 + e^x` rounded: the GPU reaches
`0.4997` of `gTol` there). It is below `|g|` for `x > -16`; for `x < -16` the kernel's own result
is about `0` (it computes `log(1 + exp(x))`, not `log1p`), a relative error up to `100 %`, and the
tolerance says `|g - gRef| ≤ 2^-23·e^A`, which is what the kernel achieves.

**`beta_output`**: `tl.exp(-b)` (`X = |b|`, room by `Pre`): relative `≤ (2 + 88.73)U < 1.1·10^-5`
on `σ` (as above, `≤ ε` relative), `1 +`: `u`, `DivFull`: `2^-22` and `2^-148`, then the `bfloat16`
rounding: `2^-8` relative and `2^-134` (its grid below `2^-126` is `2^-133`), and the widening back
is exact. Total relative `2^-8 + 1.2·10^-5 < 2^-7`, absolute `2^-134 + 2^-148·(1 + 2^-8) < 2^-133`.
The GPU reaches `0.498` of it.

## Exclusions

- **`A_log > 88.72282` (`ExpRoom` fails): a limit of the kernel.** `exp(A_log)` overflows to `+∞`
  (from `88.72284` on the device), and `g = -∞·softplus(x)` is `-∞` where the exact `g` is finite
  (`A_log = 88.75`, `a = -5`, `dt_bias = 0`: exact `g ≈ -2.4e36`, checked on the H200) and NaN where
  the kernel's softplus is `0` (`x < -16.6`, where `1 + e^x` rounds to `1`; `A_log = 88.72284`,
  `a = -20` gives NaN on the H200). (The one `float32` value `88.72283`, below the device's overflow, is outside
  the library's room.) Real use: `A_log = log(A)` with `A ∈ (0, 16]` at initialisation (about
  `≤ 2.8`); never reached.
- **`a + dt_bias ≥ 2^128 - 2^103` (the `float32` sum rounds to `+∞`): a limit of the kernel.** Then
  `softplus = +∞` and `g = -∞` (NaN if `exp(A_log)` is `0`), where the exact `g` is finite for
  small `e^A`: `a = dt_bias = 3.39e38` (the `bfloat16` maximum), `A_log = -10`: exact
  `g ≈ -3.1e34` (the H200 gives `-∞`). Real use (projection outputs and a learned bias of magnitude far below `2^127`)
  never reaches it.
- **`b < -88.72282` (`ExpRoom (-b)` fails).** `exp(-b)` overflows and the device returns
  `beta = 1/∞ = 0`. For `b ∈ (-92.19, -88.72)`, `σ(b)` is a `bfloat16` subnormal above `2^-133`,
  so `0` is outside `bTol` (`b = -88.75`: `σ ≈ 2.2e-39`, the H200 gives `0`): a limit of the kernel (an
  intermediate overflow). Below `-92.19` the
  device's `0` is within `bTol`, but the library states nothing for `tl.exp` beyond its room or for
  `/` by `∞` (a library gap, below). Real use: `b` is a projection output of magnitude in the tens;
  never reached.
- **`batch * num_heads > 2^31`: a limit of the kernel.** The `int32` output offset
  `i_b * num_heads + h` wraps and the store goes below `g`: first at `num_heads = 32`,
  `batch = 2^26 + 1` (67,108,865 tokens; 8 GiB per output). Real use (tens of thousands of tokens,
  at most 48 heads) is below `2^22`. Within `batch_ctx` (`batch ≤ 2^22`) it takes
  `num_heads ≥ 513`.
- **`(batch - 1) * stride + num_heads > 2^31`: a limit of the kernel.** The `int32` row offset of
  `a` (`b`) wraps, reading below the buffer: split views with `num_heads = 32` (stride 64), from
  `batch = 33,554,433`. Never reached in real use. Within `batch_ctx` it takes split views with
  `num_heads ≥ 257`.
- **`batch ≥ 2^31` or `cdiv(num_heads, 8) ≥ 2^16`: the CUDA grid limit** (a launch error); from
  `num_heads = 524,281`. Never reached.
- **Strides `≥ 2^31`**: Triton would pass them as `int64` (another model); outside real use
  (strides are `num_heads` or `2·num_heads`).
- **Non-finite inputs** are outside SETUP.md ("any finite values").
- **out of scope**: dimensions that are not multiples of 128 or exceed 2^17, contexts longer
  than 2^22 tokens, and batches of more than 8192 requests: the claim covers real model shapes
  only. The field: `batch`, the number of tokens (`batch_ctx`: a context length, not a batch
  size). The kernel has no model-dimension field (`num_heads` is a head count) and no field that
  counts requests.

## Library requests

None needed for real use. The library states `ExpF32Overflow` (`tl.exp` on `float32` is `+∞` from `1453635/16384` on) and `DivF32InfDivisor` (`1/∞ = +0`); with them, and an absolute `beta` term of `2^-126`, the claim could cover every `b`.

## Checks

- **Builds**: `lake env lean ../Claim.lean` from `lean/`: no error, no
  warning, no `sorry`, no `axiom`; `consistent`, `satisfiable` and every `setup_*_ok` use only
  `propext`, `Classical.choice`, `Quot.sound`.
- **Not vacuous**: `consistent` (`Platform.sm90ref`, `exists_total_faithful`); `satisfiable` (the
  first test case, `batch = 1`, `num_heads = 32`, on the all-zero memory).
- **Coverage**: the 15 instances above, `Inst.ok` proved for each (`mk_ok`).
- **GPU** (`check_gpu.py`, H200, Triton 3.8.0): the 15 setup launches through the host (the test's
  inputs with `a` past the threshold; the configurations with `A ~ U(0, 16)`), edge launches (every
  finite `bfloat16` `a` against 16 values of `A_log` from `-3e38` to `89`, with `dt_bias` `0`,
  random, and `±3.38e38`, and every finite `b`; sums within a few ulps of the threshold `20`), and
  two direct launches with the host's launch line on outputs carved from a random pool with guards.
  11,068,444 covered cells: `g` worst `0.4997` of `gTol` (`0.375` where its relative term is at least
  half of it), `beta_output` worst `0.498` of `bTol`; 245,348 cells `-∞` by the overflow clause;
  every input and guard unchanged. Of the 1,050,128 cells outside `Pre` (the exclusions), 132,826
  violate `Post` (`A_log ≥ 88.72284`, sums to `+∞`, `b < -88.72`), confirming the limits.
