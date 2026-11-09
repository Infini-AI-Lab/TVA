# Claim: `layer_norm_fwd_bf16`

fla's `layer_norm_fwd_kernel` (`source/layernorm.py`, lines 50–114) on
the paths `SETUP.md` asks for: LayerNorm with weight and bias (`IS_RMS_NORM = False`,
`HAS_WEIGHT = HAS_BIAS = True`), with or without a residual
(`HAS_RESIDUAL = STORE_RESIDUAL_OUT`, either value), over `G ≥ 1` groups (fla's `GroupNorm`:
row `t` uses the weight and bias of group `t % G`; `LayerNorm` is `G = 1`), `bfloat16` tensors,
`float32` `mean` and `rstd`, launched by the host `layer_norm_fwd` (lines 117–182) with `BT = 32`,
`num_warps = 4`, grid `(cdiv(T, 32),)`. `Claim.lean` holds the model, the conditions, the
specification and the assumptions; `Statement` is the claim to prove.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.LayerNormFwd.main : Statement` (`Statement`:
`Claim.lean` line 484). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 218), and every initial memory `init` that `Pre` accepts (line 323), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 454).

**What `Post` asks** (line 454): for every row `t < T`, `RowDone d init fin t` (line 442), and the
frame. Row `t`'s values are `s j = sv d init t j` (line 356: with a residual the IEEE `float32` sum
`FloatV.add` of the two `bfloat16` inputs converted to `float32`, without one `x[t, j]` converted), and the
reference is the exact real LayerNorm of `s` (`meanRef` line 364, `varRef` 372, `rstdRef` 379,
`zRef` 383, `yRef` 387; `w`, `b` of group `t % G`, `gidx` line 190, read from `init`). Element
`(t, j)` of a `[T, D]` tensor is `idx d t j = t·D + j` (line 187). `RowDone` has four parts:
1. with a residual (`d.hasRes = true`), for every `j < D`: `res_out[idx t j]` reads in `fin` as
   `FloatV.cvt (sv d init t j)`, bit for bit (no tolerance); without one, nothing is asked;
2. `mean[t]` reads as a finite `float32` `c` with `|c.toReal - meanRef d init t| ≤ meanTol d init t`
   (`meanTol`, line 395);
3. `rstd[t]` reads as a finite `float32` `c` with `|c.toReal - rstdRef d init t| ≤ rstdTol d init t`
   (`rstdTol`, line 419: `2·(cond + gam)·rstdRef` when `4·cond ≤ 1`, else `2·eps^(-1/2)`;
   `gam` 413, `cond` 409);
4. for every `j < D`: `y[idx t j]` reads as a finite `bfloat16` `c` with
   `|c.toReal - yRef d init t j| ≤ yTol d init t j` (`yTol`, line 432:
   `2^-7·|yRef| + 4·|w|·zTol + 2^-21·|b| + 2^-120`, with `zTol` 426 and `eTol` 404).

These hold on every row; there is no guard on the data's spread. `depth` (line 392) is
`Nat.clog 2 BD`. The frame: every byte outside `outBytes d` (line 210: the `T·D` elements of `y`,
those of `res_out` only with a residual, the `T` elements of `mean` and `rstd`) is unchanged.

**What you are given** (`Assumptions`, line 461):
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `sqrt_total` (`Num.MathTotal P .sqrt .binary32`): `tl.sqrt` (source line 99, `Claim.lean`
  line 117) has a result;
- `sqrt_acc` (`Num.SqrtF32`): on `0` and inputs `≥ 2^-126`, `tl.sqrt` is within `2^-22·√x + 2^-149`;
- `div_total` (`Num.DivTotal P .binary32`): `/` (source lines 91, 95, 99; `Claim.lean` lines 97, 109,
  116) has a result;
- `div_full` (`Num.DivFull`): `/` is within `2^-22·|q| + 2^-148` of the quotient `q`, for
  `|q| ≤ 2^126`;
- `sum_depth` (`Num.SumDepth`): the trees of `tl.sum1` (source line 91, `Claim.lean` line 97)
  and `tl.sum1c` (source line 95, `Claim.lean` line 109) over their `BD` positions have depth at
  most `Nat.clog 2 BD` (`SumDepth.depth_le`).

**The kernel, as the model runs it** (`layer_norm_fwd_kernel` line 50, `model` line 280, `launch`
line 291): a 1-D grid of `cdiv(T, 32)` programs. Program `p` handles the rows
`o_t = 32p + [0, 32)` (`int64`), the row groups `o_g = o_t % G` (`tl.irem`), the columns
`o_d = [0, BD)`, the masks `m_t` (`o_t < T`), `m_d` (`o_d < D`), `m_x` (both). It loads the `[32, BD]`
block of `x` (and of `residual`, adding it, when `hasRes`) with `other = 0`; stores `res_out` when
`hasRes`; reduces each row with `tl.sum1` and divides by `D` (the mean), stores `mean` under `m_t`;
forms the deviations with `tl.subc` (zero outside `m_d`), squares them, reduces with `tl.sum1c`,
divides by `D`, adds `eps` (`tl.addc`), takes `tl.sqrt` and `1 / ·`, stores `rstd` under `m_t`; loads
`w` and `b` at row group `o_g` under `m_d` only (no `other`), forms `tl.subc`, `tl.mul` by `rstd`,
`tl.mul` by `w`, `tl.addc` with `b`, converts to `bfloat16` and stores `y` under `m_x`. There is no
loop. Programs write disjoint row blocks, all read the same `weight` and `bias`, and use no atomics:
the route is `TVA.meets_independent` (PROVER_GUIDE §4). The loads and stores go through computed
pointer blocks: `safe_load`, `safe_store` (distinct lanes), not the 1-D block rules.

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: fewer than `2^31` programs from
  `t_lt`; the block shapes `[32]`, `[BD]`, `[32, 1]`, `[1, BD]`, `[32, BD]` from `bd_pow2`, `d_le`,
  `d_le_bd`);
- each program is `Safe`: the `int64` offsets `o_t·D + o_d` do not wrap (`t_lt`, `d_le`); every
  enabled lane of `x`, `residual`, `y`, `res_out` is below `T·D` (`x_len`, `res_len`, `y_len`,
  `res_out_len`), of `mean`, `rstd` below `T` (`mean_len`, `rstd_len`); the loads of `w` and `b` are
  enabled on rows `t ≥ T` too, and stay in bounds because `o_g < G` (`g_pos`, `w_len`, `b_len`;
  `safe_irem` with a nonzero divisor); every float step has a result (`sqrt_total`, `div_total`);
- the write footprints of distinct programs (their rows of `y`, `res_out`, `mean`, `rstd`) are
  disjoint, and no program's reads meet any program's writes (`out_in`, `y_res_out`, `y_mean`,
  `y_rstd`, `res_out_mean`, `res_out_rstd`, `mean_rstd`); this also gives `hRW`;
- part 1: `safe_add` and `safe_cvt` on the finite inputs (`Pre`, `InOk`), exact;
- parts 2–4: the rounding of each step, the sums by `sum_err_depth` (`tl.sum1`) and `sumc_err_depth`
  (`tl.sum1c`) with `SumDepth.depth_le`, `div_full`, `sqrt_acc` (its input is at least `2^-126`
  from `eps_ge`), the conversion of the host's `eps` (`FloatV.hostF32`), and the magnitude bounds
  `InOk` (`2^56`), `InOkW` (`2^16`) that keep every value finite: derive the fine bounds of README
  `## Tolerances`, steps 1–7, then relax them to `rstdTol` and `yTol` by step 8;
- the frame: the bytes no program writes are `init`'s.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mkInst_ok`,
`mkInst_pre`, `eps1e5_facts`, `eps64e5_facts` and the `setup_*_ok` theorems; `bytes_disj` (line 541)
and `elems_disj` (line 549) can be reused.

## Specification

For each row `t < T`, with `s j` the `float32` sum `float32(x[t, j]) + float32(residual[t, j])`
(the IEEE addition of the exactly converted `bfloat16` inputs, as the host's `reference` forms it)
with a residual and `float32(x[t, j])` without one, `w j = weight[(t % G)·D + j]` and
`b j = bias[(t % G)·D + j]` (row `t`'s group), and the exact real LayerNorm of `s` as the
reference (`meanRef = Σ s/D`, `varRef = Σ (s − meanRef)²/D`, `rstdRef = 1/sqrt(varRef + eps)` with
the host's double `eps`, `zRef j = (s j − meanRef)·rstdRef`, `yRef j = zRef j·w j + b j`), `Post`
says:

- with a residual, `res_out[t, j]` is `bfloat16(s j)` (round to nearest even), bit for bit;
- `mean[t]` is a finite `float32` within `meanTol` of `meanRef`;
- `rstd[t]` is a finite `float32` within `rstdTol` of `rstdRef`;
- `y[t, j]` is a finite `bfloat16` within `yTol j` of `yRef j`;
- every byte outside the `T·D` elements of `y` (and of `res_out`, with a residual) and the `T`
  elements of `mean` and `rstd` is unchanged (frame; without a residual `res_out` is not written).

The reference reads the host's tensors at the host's layout (`t·D + j`, `g·D + j`), never through
the kernel's indexing. The tolerances (Tolerances, below), with `L = log₂ BD` (`depth`: the depth
of the reductions' trees over their `BD` positions, `Rules.Num.SumDepth`), `A = Σ|s|/D`,
`rmsS = sqrt(Σ s²/D)`, a lane's mean bound `eTol j = meanTol + 2^-23·(|s j| + A)` and the row's
condition number `cond = (meanTol + 2^-23·(A + rmsS))·rstdRef`:

| Output | Tolerance |
|---|---|
| `mean` | `meanTol = (L + 4)·2^-23·A + 2^-140` |
| `rstd` | `rstdTol = 2·(cond + gam)·rstdRef` when `4·cond ≤ 1`, else `2·eps^(-1/2)`; `gam = (L + 10)·2^-23 + 2^-136/eps` |
| `y` | `yTol = 2^-7·|yRef| + 4·|w|·zTol + 2^-21·|b| + 2^-120`, with `zTol = |s − meanRef|·rstdTol + eTol·(rstdRef + rstdTol)` |

Each is a relative bound on the output's natural magnitude: `mean` relative to `A` (with the
`2^-140` floor), `rstd` relative to `rstdRef` with `ε = 2·(cond + gam)` on a well-conditioned row,
and `y` the output's `bfloat16` rounding (`2^-7·|yRef|`, twice it) plus the standardised value's
error `zTol` through the weight. `rstd` and `y` keep the row's conditioning: `cond` (the mean's
rounding in units of the row's standard deviation) and, on an ill-conditioned row (`4·cond > 1`),
`eps^(-1/2)`, the most the kernel's `rstd` is then known to be; no single constant `ε` holds
there.

They hold on every row, with no condition on the data's spread. On a well-conditioned row (the
covered launches' rows: `A ≈ 0.8`–`1.1`, `rstdRef ≈ 1`, `cond ≈ 1.3·10^-6`) they are
`(L + 4)·2^-23·A ≈ 1.2·10^-6·A` (`D = 64`) to `1.5·10^-6·A` (`D = 512`) for the mean,
`6.2·10^-6` to `7.4·10^-6` relative for `rstd` (`2·(cond + gam)`), and `2^-7·|y| + ≈ 3·10^-5·|w|`
for `y` (one `bfloat16` ulp of the output: `tol / |y|` has median `7.8·10^-3` at RWKV-7's and
RWKV-6's `g_norm` shapes). They grow with `cond`, which is bounded by about `(meanTol + 2^-22·rmsS)/sqrt(eps)`: a
row whose values share a large offset and differ by little (`|mean| ≫ std`, `std² ≲ eps`) loses
the variance to the computed mean's rounding (the two-pass algorithm in `float32`, whose
subtractions may fuse with the mean's division), and the tolerances then say what the kernel still
guarantees: `rstd` at most about `1/sqrt(eps)` (`rstdTol = 2·eps^(-1/2)`), and the standardised
value within `zTol = |s − meanRef|·rstdTol + eTol·(rstdRef + rstdTol)`. With
`eps = 1e-5`, `cond` stays below about `15·2^-23·rmsS·316 ≈ 6·10^-4·rmsS`: rows of entries of order
1 are always well conditioned, constant rows included (`D = 1`, a constant row: `zRef = 0` and
`y = b` within `4·|w|·zTol`).

## Conditions

`Inst` is everything the host chooses: `T`, `D`, `BD`, `G` (`num_groups`), `hasRes` (a residual
is given: `HAS_RESIDUAL = STORE_RESIDUAL_OUT = hasRes`, lines 139, 174–175), `eps` (the Python
double), and the eight tensors' storage (`Buf`; without a residual the host passes `None` for
`res` and `res_out`, and their buffers are not touched). `Inst.ok`:
- `1 ≤ D ≤ 512` (this kernel's path, the host's assert, line 151), `BD` the next power of two
  (`BD = 2^e`, `D ≤ BD < 2D`);
- `1 ≤ T < 2^31` (`T` is a runtime `int32` argument: `do_not_specialize`, and below `2^31`);
- `1 ≤ G ≤ 2^16` (Exclusions); `T` need not be a multiple of `G`;
- `eps` finite, `2^-125 ≤ eps ≤ 2^126` (Exclusions);
- every tensor holds what the launch touches (`x`, `y`: `T·D`; `residual`, `res_out`: `T·D` with a
  residual; `weight`, `bias`: `G·D`, the host's asserts, lines 133–136; `mean`, `rstd`: `T`);
- the outputs' touched elements share no byte with the inputs' or with each other's (`x` and
  `residual`, `weight` and `bias` may overlap: they are only read).

`Pre`: the buffers the launch touches valid (aligned, present; `residual` and `res_out` only with
a residual), every element of `x` and `residual` (with one) the launch reads finite with
`|v| ≤ 2^56`, and every element of `weight` and `bias` (all `G·D`) finite with `|v| ≤ 2^16`
(Exclusions).

Scope of `Compiler.Faithful`, part (A): the launch passes only `num_warps` besides the kernel's
arguments (no other option, no `@triton.heuristics`; the source's `@triton.autotune` is removed,
`SETUP.md`), the decorator has no `debug` (`Launch.debug = false`), no descriptors, no
`CompiledKernel`, no host `TensorDescriptor`. Part (B), the environment, is
assumed: the claim relies on it.

## Assumptions

- `[rule] Compiler.Faithful`, as `docs/rules/compiler-faithful.md` states it.
- `[rule] Num.MathTotal P .sqrt .binary32` and `Num.SqrtF32 P`: `tl.sqrt` on `float32` (line 99).
- `[rule] Num.DivTotal P .binary32` and `Num.DivFull P`: `/` on `float32` (lines 91, 95, 99).
- `[rule] Num.SumDepth P`: the trees of the two float `tl.sum`s (lines 91, 95) over their `BD`
  positions have depth at most `⌈log₂ BD⌉ = L` (`depth`). The kernel has no loop, so neither
  reduction is one the loop rewrite reassociates.

`consistent` witnesses them with `Platform.sm90ref`.

## The model

`model-map.json` maps every code line. `+ eps` (line 99) is `tl.addc` (at `D = 1` the division
`/ 1.0` folds away and the one-lane variance is the product `b_xbar * b_xbar` itself, which the
backend may contract with `+ eps`); `other=0.0` is the `float32` `+0` converted to `bfloat16`
(SEMANTICS.md, `tl.load`); `eps` is the host's double, `FloatV.hostF32` in the model; the constant
`D` of lines 91 and 95 is `FloatV.litF32` of its double; the pointer offsets add the `int32`
`o_d[None, :]` as the source does (`addptr` takes it as it is); `o_t % G` (line 76) is `tl.irem` of
`int64` tensors, the constexpr `G` built in `int64`. `G`, `D`, `BD`, `HAS_RESIDUAL` and
`STORE_RESIDUAL_OUT` are Lean parameters, the two flags' branches (lines 84, 87) Lean `if`s;
`IS_RMS_NORM = False`, `HAS_WEIGHT = HAS_BIAS = True` and `BT = 32` are fixed and the branches
they do not take are not modelled (`config`).

## Coverage

Each launch is an instance `mkInst T D BD G hasRes eps` (the tensors at `k·2^36`) with a proof
`setup_*_ok`:

| launch | `(T, D, BD, G)`, residual, `eps` | instance |
|---|---|---|
| host test (source lines 203–213), `LayerNorm` with a residual | `(1, 1, 1, 1)`, `(1, 64, 64, 1)`, `(31, 50, 64, 1)`, `(2048, 50/64/128, 64/64/128, 1)`, `(33, 512, 512, 1)`, `(1000, 300, 512, 1)`, `(65536, 256, 256, 1)`; yes; `1e-5` | `setup_1x1`, `setup_1x64`, `setup_31x50`, `setup_2048x50`, `setup_2048x64`, `setup_2048x128`, `setup_33x512`, `setup_1000x300`, `setup_65536x256` |
| RWKV-7 `g_norm`, 4 × 4096 tokens | `(524288, 64, 64, 32)`; no; `6.4e-4` | `setup_rwkv7_gn_train` |
| RWKV-7 `g_norm`, one decoding token | `(32, 64, 64, 32)`; no; `6.4e-4` | `setup_rwkv7_gn_decode` |
| RWKV-6 `g_norm`, 4 × 4096 tokens | `(65536, 512, 512, 4)`; no; `1e-5` | `setup_rwkv6_gn_train` |
| fla's GroupNorm test's shapes (2 × 512 tokens), `G = 4`, hidden 64, 128, 512, 1024, 2048 | `(4096, 16, 16, 4)`, `(4096, 32, 32, 4)`, `(4096, 128, 128, 4)`, `(4096, 256, 256, 4)`, `(4096, 512, 512, 4)`; no; `1e-5` | `setup_gn_test_64_g4`, `setup_gn_test_128_g4`, `setup_gn_test_512_g4`, `setup_gn_test_1024_g4`, `setup_gn_test_2048_g4` |
| fla's GroupNorm test's shapes, `G = 1`, hidden 64, 128, 512 | `(1024, 64/128/512, 64/128/512, 1)`; no; `1e-5` | `setup_gn_test_64_g1`, `setup_gn_test_128_g1`, `setup_gn_test_512_g1` |
| fla's LayerNorm test's shapes, `D` 50, 64, 128 | `(2048, 50/64/128, 64/64/128, 1)`; no; `1e-5` | `setup_ln_2048x50`, `setup_ln_2048x64`, `setup_ln_2048x128` |

fla's two tests run in `float32`; their setups here are the `bfloat16` launches at the tests'
shapes. `SETUP.md`'s ranges (`D` 1..512, `T` from 1 up, `G` from 1 up, with or without a residual, any
positive `eps`, training batches of hundreds of thousands of rows) are covered by `Inst.ok` as
above, up to the exclusions below; `Pre` puts no condition on the data beyond finiteness,
`|x|, |residual| ≤ 2^56` and `|weight|, |bias| ≤ 2^16`. `satisfiable` uses `mkInst 2048 64 64 1 true 1e-5` in a memory of zeros.

## Tolerances

Notation: `u = 2^-24`, `η = 2^-150` (`float32`; `FloatV.add_err`, `sub_err`, `mul_err`, `cvt_err`:
`u·|q| + η`), `u16 = 2^-8`, `η16 = 2^-134` (`bfloat16`), `n = BD`, `L = log₂ n`,
`e_m = mean' − meanRef` (the kernel's mean `mean'`), `d j = s j − meanRef`, `σ = sqrt(varRef)`. The
facts used:
`SumDepth` with `SumDepth.depth_le` (each `tl.sum`'s tree over its `n` positions has depth at most
`Nat.clog 2 n = L`), `sum_err_depth` (`tl.sum1` along a tree of depth `L` with `n − 1` nodes:
`((1+u)^L − 1)·Σ|leaf| + η(n−1)(1+u)^L`), `sumc_err_depth` (`tl.sum1c`: per node `3u`, `4η`,
depth `L`), `FloatV.AddRel.err` (`tl.addc`: `3u(|a| + |b|) + 4η`), `DivFull` (`2^-22·|q| + 2^-148`,
`|q| ≤ 2^126`), `SqrtF32` (`2^-22`, `2^-149`, on `0` and `x ≥ 2^-126`), and the fusable
subtraction (`tl.subc`, `AddRel` on `b_x` and `−mean'`: SEMANTICS.md, "Float `+` and `-`", a
`float32` quotient is a product): the rounding of `p − q` with `p = s j` and `q` any exact value
rounding to `mean'`, or `q = mean'` and `p` any exact value rounding to `s j` (the IEEE difference
among them). The masked lanes `j ≥ D`
are `+0` (`other=0.0`, and `tl.where` puts `0.0` in the deviations). Nothing below depends on the
group or the residual: `s` is a finite `float32` row with `|s| ≤ 2^57` either way, and `w`, `b`
are the row's group's (`InOkW`).

1. **Mean.** `|S' − S| ≤ E0·D·A + η(n−1)(1+u)^L` with `E0 = (1+u)^L − 1`, then the division:
   `|mean' − meanRef| ≤ (E0(1 + 4u) + 4u(1 + E0))·A + 2^-141 + 2^-148`. The relative part is
   `(L + 4)·u` to first order (composed exactly: `4u` at `L = 0`, `13.0000u` at `L = 9`), at most
   `0.50` of `(L + 4)·2^-23 = (2L + 8)·u` at every `L ≤ 9`, and the absolute part is below `2^-140`.
2. **The deviations.** Each of the two subtractions (lines 94 and 108, each its own choice) forms
   `fl(d j − e j)` for a lane mean error `e j = e_m − ε j`, where `ε j` is the amount the fusable
   subtraction moves `s j` or `mean'` by: `|ε j| ≤ u·max(|s j|, |mean'|) + η` (half a unit of the
   operand), with `|mean'| ≤ 1.00001·A + 2^-140`. So `|e j| ≤ eTol j` (`2^-23 = 2u` covers
   `1.00001u`; the absolute parts, `2^-147` and `2^-149.9`, are below `2^-140`), and by Minkowski the
   root mean square `E` of line 94's `e j` over `j < D` is at most
   `meanTol + 2^-23·(A + rmsS)`, so `y := E·rstdRef ≤ cond`.
3. **Variance.** `V0 = Σ_j (d j − e j)²/D` (line 94's `e j`) lies between `(σ − E)²` (`0` when
   `E ≥ σ`) and `(σ + E)²`; so `V0 + eps ≥ (σ² + eps)(1 − 2y)` (`2σE ≤ 2y·(σ² + eps)`), `V0 + eps ≥
   eps`, and `V0 + eps ≤ (σ² + eps)(1 + y)²`. Every factor after the subtraction is relative on
   nonnegative terms: the subtraction's rounding `(1 ± u)` squared, the product `(1 ± u)`, the fused
   sum `(1 ± E1)` with `E1 = (1+3u)^L − 1`, the division `(1 ± 4u)`, the conversion of the host's
   double `eps` to `float32` `(1 ± u)` and `tl.addc` `(1 ± 3u)`. So the input of `tl.sqrt` is
   `(V0 + eps)·(1 + Θ)` with `|Θ| ≤ E1 + 11u` (to first order), plus absolute terms below `2^-138.4`
   in all (the `η` of the products and of the two sums, the divisions' `2^-148`, `tl.addc`'s `4η`;
   the subtraction's `η` enters only through `2η·Σ|dev| ≤ u·Σdev² + n·η²/u`). Relative to the
   input, which is at least `eps·(1 − u)`, they are below `2^-138.4/eps`. Room: `|s| ≤ 2^57`, each
   deviation at most `(|s j| + |mean'|)(1 + 2u) ≤ 2^58·1.0001`, its square at most `2^116.001`, the
   sum at most `2^125.001 ≤ maxVal/2 − 2η`; the input of `tl.sqrt` is at least
   `eps(1 − 4u) − 2^-147 ≥ 2^-126` (`eps ≥ 2^-125`: `SqrtF32`'s domain).
4. **rstd.** `tl.sqrt` and `1/·` add `(1 ± 4u)` each (their absolute terms are below `2^-84` relative,
   since `sqrt(var + eps) ≥ 2^-63` and `rstd ≥ 2^-64`). With `r* = 1/sqrt(V0 + eps)`,
   `|rstd' − r*| ≤ ρ·r*`, where `ρ ≤ (1.5L + 13.5)·u + 2^-139.4/eps` (composed exactly, with
   `Θ`'s envelope `(1+u)^4(1+E1)(1+4u)(1+3u) − 1`: `13.50u` at `L = 0`, `27.00u` at `L = 9`), at most
   `0.72` of `gam`'s rounding part `(L + 10)·2^-23 = (2L + 20)·u` at every `L ≤ 9`, and the `eps`
   part a tenth of `2^-136/eps`; so `ρ ≤ gam`. By step 3, `rstdRef/(1 + y) ≤ r* ≤ eps^(-1/2)`, and
   when `4y ≤ 1` also `r* ≤ rstdRef/sqrt(1 − 2y) ≤ rstdRef·(1 + 2y)` (convexity: `0.414 ≤ 0.5` at
   `y = 1/4`); with `y ≤ cond`, `r* ≤ rUp`, where `rUp = min(rstdRef·(1 + 2·cond), eps^(-1/2))` when
   `4·cond ≤ 1`, else `eps^(-1/2)`. Upward, `rstd' − rstdRef ≤ (1 + ρ)·rUp − rstdRef =
   (1 + ρ)(rUp − rstdRef) + ρ·rstdRef`; downward, `rstdRef − rstd' ≤ rstdRef − (1 − ρ)·rstdRef/(1 + y)
   = rstdRef·(y + ρ)/(1 + y) ≤ rstdRef·(min(1, cond) + ρ)`. Both are within the fine bound
   `rstdTol₀ = (1 + gam)·max(rUp − rstdRef, rstdRef·min(1, cond)) + gam·rstdRef + 2^-140`.
5. **Standardised value.** `z' = fl(fl(d j − e j)·rstd')` with line 108's `e j`, so
   `z' − zRef = d j·((1+a)(1+c)·rstd' − rstdRef) − e j(1+a)(1+c)·rstd'` (`|a|, |c| ≤ u`). The first
   term is at most `|d j|·((1+u)²·rstdTol₀ + (2u + u²)·rstdRef) ≤ (1 + 2^-22)·|d j|·rstdTol₀ +
   2^-22·|zRef|`; the second at most `(1 + 2^-22)·eTol j·(rstdRef + rstdTol₀)` (`rstd' ≤ rstdRef +
   rstdTol₀`); the absolute terms (the `η` of the subtraction times `rstd'`, of the product) are below
   `2^-140·(1 + rstdRef + rstdTol₀)`. So the fine bound `zTol₀ = (1 + 2^-22)·(|d j|·rstdTol₀ + eTol j·(rstdRef + rstdTol₀)) +
   2^-22·|zRef| + 2^-140·(1 + rstdRef + rstdTol₀)` bounds `|z' − zRef|`. Room: line 108's deviation is
   within `2^-23·(|s j| + |mean'|)·1.0001 ≤ 2^35.001` of line 94's at the same lane, which is at most
   `sqrt(D·V0)` before rounding; with `r* ≤ 2^62.5`, `|z'| ≤ (1 + 10u)(sqrt(512) + 2^35.001·2^62.5)
   ≤ 2^97.6`.
6. **Output.** `p = fl(z'·w)` (`u|z'w| + η`), `y32 = addc(p, b)` (`3u(|p| + |b|) + 4η`, the fused and
   the unfused results alike), `y' = bfloat16(y32)` (`u16·|y32| + η16`). With
   `|y32| ≤ |yRef| + |w|·zTol₀ + (errors)` and `|z'w| ≤ |w|(|zRef| + zTol₀)`:
   `|y' − yRef| ≤ 2^-8|yRef| + (1 + 2^-8)(1 + 4.01u)·|w|·zTol₀ + (1 + 2^-8)·(4.01u·|w||zRef| + 3u|b|)
   + 2^-133`, within the fine bound `yTol₀ = 2^-7·|yRef| + (1 + 2^-7)·|w|·zTol₀ + 2^-21·(|w|·|zRef| +
   |b|) + 2^-130` (`2^-7` is twice the rounding's `2^-8`; `(1 + 2^-8)(1 + 4.01u) ≤ 1 + 2^-7`;
   `4.01u(1 + 2^-8)` and `3u(1 + 2^-8)` are below `2^-21`). Room: `|p| + |b| ≤ 2^97.6·2^16 + 2^16
   ≤ 2^114`, and `|y32| ≤ 2^114.1`, inside `bfloat16`'s range.
7. **`res_out`**: `tl.cvt` of the finite `s j` is `FloatV.cvt`, exactly.
8. **The relaxation** of the fine bounds to `Post`'s. `meanTol` is used as it is.
   *`rstd`:* if `4·cond ≤ 1`, `rUp − rstdRef ≤ 2·cond·rstdRef` and `rstdRef·min(1, cond) ≤
   2·cond·rstdRef`, so `rstdTol₀ ≤ (1 + gam)·2·cond·rstdRef + gam·rstdRef + 2^-140`, and
   `rstdTol − rstdTol₀ ≥ gam·rstdRef·(1 − 2·cond) − 2^-140 ≥ gam·rstdRef/2 − 2^-140 > 0`
   (`rstdRef ≥ 2^-63.5` since `varRef ≤ 2^115` and `eps ≤ 2^126`, and `gam ≥ 10·2^-23`). Otherwise
   `rUp = eps^(-1/2)` and `rstdRef ≤ eps^(-1/2)`, so `rstdTol₀ ≤ (1 + 2·gam)·eps^(-1/2) + 2^-140 ≤
   2·eps^(-1/2)` (`gam ≤ (L + 10)·2^-23 + 2^-11 ≤ 2^-10`, from `eps ≥ 2^-125`). So
   `rstdTol₀ ≤ rstdTol` on every row, and also `rstdTol ≥ gam·rstdRef ≥ 10·2^-23·rstdRef`.
   *`z`:* with `r = rstdRef`, `rT = rstdTol` (`zTol₀` grows with `rstdTol₀ ≤ rT`) and `d = |s j −
   meanRef|`: `2^-22·|zRef| = 2^-22·d·r ≤ 0.2·d·rT`, and `2^-140·(r + rT) ≤ eTol·(r + rT)`
   (`eTol ≥ meanTol ≥ 2^-140`), so `zTol₀ ≤ 1.2000003·d·rT + 2.0000003·eTol·(r + rT) + 2^-140`.
   *`y`:* `2^-21·|w|·|zRef| ≤ 0.4·|w|·d·rT` the same way, so `(1 + 2^-7)·|w|·zTol₀ + 2^-21·|w|·|zRef|
   ≤ |w|·(1.61·d·rT + 2.016·eTol·(r + rT)) + 1.008·2^-140·|w| ≤ 4·|w|·zTol + 2^-123.9` (`zTol = d·rT +
   eTol·(r + rT)`, `|w| ≤ 2^16`), and `yTol₀ ≤ 2^-7·|yRef| + 4·|w|·zTol + 2^-21·|b| + 2^-123.9 +
   2^-130 ≤ yTol`.

On ill-conditioned rows the fine tolerances are reached almost exactly, and stay below by the
derivation's margin: with `D = 3`, an offset of `3·10^4`, noise of `10^-3` and `eps = 2^-125`
(`cond ≈ 10^17`), the variance is lost to the mean's rounding and the kernel's `rstd` is far below
`rstdRef`: `rstdRef − rstd' ≤ rstdRef·(y + ρ)/(1 + y) < rstdRef` against `rstdTol₀ ≥ (1 + gam)·rstdRef`,
a ratio at most `1/(1 + gam)` (`0.99902` of `rstdTol₀`, `0.500` of `rstdTol = 2·eps^(-1/2)`, on
`check_gpu.py`'s launch). The `y` tolerance has a factor of 2 over the output's `bfloat16`
rounding, which dominates the kernel's error on the GPU (ratio `0.498`). `meanTol` and `rstdTol`
reach `0.12` and `0.036` of themselves on well-conditioned rows (`check_gpu.py`): they take every
rounding of a depth-`L` tree, and every choice of the two fusable subtractions, at its worst. At
real shapes `rstdTol / rstdRef` has median `6.2·10^-6` (RWKV-7's `g_norm`) and `7.4·10^-6`
(RWKV-6's), `1.44` times the fine bound's `4.3·10^-6` and `5.2·10^-6`; `y`'s median `tol / |y|`
is `7.8·10^-3` for both.

## Exclusions

Launches `SETUP.md` allows that `Inst.ok`/`Pre` leave out:

- **`eps < 2^-125`.** Below `2^-126` this is a **kernel limit**. The kernel's `float32` `eps` is
  subnormal, so on a row of zero variance `var + eps` is subnormal, `sqrt.approx.ftz` flushes it to
  `0`, `rstd = +inf`, and `y = 0·inf = NaN`. The first launch to hit it is `T = 1`, `D = 1`,
  `eps = 1e-39`, any input; on the GPU it gives `rstd = inf` and `y = nan`. In `[2^-126, 2^-125)`
  the library's semantics fail it: `DivFull` allows `−2^-148` for a zero variance, `tl.addc` admits
  an `eps` operand just below the `float32` `eps`, and the sum can then round to a subnormal. On
  the device the kernel is fine there. Real use (`eps` from `1e-12` to `6.4e-4`) reaches neither.
- **`eps > 2^126`.** This is outside real use (`eps` is a stabiliser far below 1). Near the
  `float32` maximum it is also a **kernel limit**: `var + eps` overflows, giving `rstd = 0` where
  the reference is `2^-64`, and `y = b`. The first launch to hit it is `eps = 3.4028234663852886e38`,
  `D = 2`, `x = [2^52, −2^52]`; on the GPU it gives `rstd = 0`.
- **`|x|`, `|residual|` above `2^56`.** From `2^59` this is a **kernel limit**: the `float32` sum
  of squared deviations overflows, so `var = inf`, `rstd = 0` and `y = b` instead of `±w + b`. The
  first launch to hit it is `T = 1`, `D = 512`, `x = residual = ±2^59` alternating (`s = ±2^60`).
  On the GPU it gives `rstd = 0` and `y = 0`; at `2^58` it is correct. Between `2^56` and `2^59`,
  activations of `7·10^16` or more are outside real use, and the library's sum lemma needs room
  below `maxVal/2`. Real use never reaches it.
- **`|weight|`, `|bias|` above `2^16`.** This is outside real use (trained scales of order 1). It
  is a **library gap, not a kernel limit**: the library lets the two subtractions of lines 94 and
  108 each take their own fusable result, so on an ill-conditioned row (a large offset, `eps` near
  `2^-125`) line 108's deviation may be `2^35` where line 94's is `0`, the standardised value up to
  `2^97.6`, and `y = z'·w` would overflow for `|w|` near `2^32.5`; `2^16` keeps `|y32| ≤ 2^114`. On a
  well-conditioned row the kernel is correct until `y` overflows `bfloat16`, near `|w| ≈ 2^123`.
- **Non-finite inputs.** There is no reference value.
- **`T ≥ 2^31`.** The host then passes `T` as a runtime `int64`, a different compiled kernel (the
  model's `T : I32`). It needs at least `2^31` rows: `4 GiB` per tensor at `D = 1`, more than the
  device's memory beyond `D ≈ 16`. This is outside real use (tens of thousands of rows). It is
  not known to be a kernel limit, since the offsets are `int64`.
- **`G > 2^16`.** Outside real use (one group per head: 4 to 64 heads in fla's models). Not known
  to be a kernel limit (`weight` and `bias` hold `G·D` elements; the offsets are `int64`).
- **`T = 0`**, **`D > 512`** (another kernel, `layer_norm_fwd_kernel_row`), **`IS_RMS_NORM`**
  (this kernel's RMSNorm mode, which the `GroupNorm(..., is_rms_norm=True)` query and key norms of
  `fla/layers/forgetting_attn.py:72-81` and `fla/layers/wall_attn.py:122-131` run, without a bias),
  **no weight or no bias, a `res_out` without a residual (`residual_in_fp32`), other dtypes**: not part
  of the setup (`SETUP.md`).

## Library requests

None needed for this claim.

## Checks

1. **Builds**: `lake env lean ../claims/layer_norm_fwd_bf16/Claim.lean` from `lean/` gives no error
   or warning, no `sorry`, no `axiom`. `consistent`, `satisfiable` and the twenty-three `setup_*_ok`
   depend only on `propext`, `Classical.choice` and `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (`mkInst 2048 64 64 1 true
   1e-5`, zeros).
3. **Covers the setup**: the twenty-three `setup_*_ok` above.
4. **GPU** (`check_gpu.py`, H200, Triton 3.8.0, 53 launches): the twenty-three setup launches with
   inputs drawn as the host draws them (weights `1 + 0.1·N(0,1)` and biases `0.1·N(0,1)` for the
   models' shapes, `N(0,1)` for fla's tests), and these edges:
   - `D` = 1, 2, 7, 257, 512, with `T = 33`, `T = 100` and `T = 1`;
   - constant rows (offset 1000, `cond = 0.49`) and near-constant rows (`cond` up to `0.016`), also
     without a residual and `G = 3`; ill-conditioned rows (`D = 3`, offset `3·10^4`, noise `10^-3`,
     `eps` `1e-5` and `2^-125`);
   - rows of `1e-30`, and subnormal rows (also without a residual, `G = 4`);
   - `|x| = |res| = 2^56`, `|x| = 2^56` without a residual (`G = 2`), and `|w| = |b| = 2^16`
     (also without a residual, `G = 5`, and on ill-conditioned rows with `eps = 2^-125`);
   - `w = b = 0`;
   - `eps = 2^-125` (random, constant and tiny rows, with and without a residual), `eps = 1` and
     `eps = 2^126`;
   - `T = 37` with `G = 7` (not a multiple), `T = 5` with `G = 32` (fewer rows than groups), and
     `G = 1024` at `D = 1`.

   With a residual `res_out` is exact everywhere; without one it is not written. The frame holds
   everywhere: guard bands around every output, and the inputs unchanged. The worst error /
   tolerance is `0.122` for `mean` and `0.500` for `rstd`, on the ill-conditioned rows with
   `eps = 2^-125` (Tolerances; elsewhere at most `0.075` for `rstd`), and `0.498` for `y` (the
   `bfloat16` rounding of the output). Seven wrong outputs at RWKV-7's `g_norm` shape (8192 rows) fail
   `Post`: `y = 0` (ratio `128`), group 0's weight and bias for every row (`4.1·10^4`), no bias
   (`2.5·10^4`), `x` for `x − mean` (`5.0·10^4`), `mean·(1 + 2^-12)` (`130`), `rstd·(1 + 2^-14)`
   (`10`), `rstd` without `eps` (`125`). `tol / |output|`: `mean` `1.2·10^-6·A` (RWKV-7, `D = 64`)
   and `1.55·10^-6·A` (RWKV-6, `D = 512`), `rstd` `6.2·10^-6` and `7.4·10^-6` (median), `y` median
   `7.8·10^-3` (worst near `y = 0`, where the absolute part dominates). PASS.
