# `attn_bwd_f16`: the prepared claim

Kernel: `source/06-fused-attention.py`, `_attn_bwd` (lines 366–500) with its helpers
`_attn_bwd_dkdv` (266–315) and `_attn_bwd_dq` (319–363), which the front end inlines; the kernel
lines are 20–21, 266–315, 319–363 and 366–500 ([`claim.json`](claim.json)). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.AttnBwd`, `Statement := ∀ K P, Assumptions K P → Spec K`.
Model map: [`model-map.json`](model-map.json). GPU check: [`check_gpu.py`](check_gpu.py). The
launches to cover: [`SETUP.md`](SETUP.md).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.AttnBwd.main : Statement` (`Statement`:
`Claim.lean` line 769). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 393), and every initial memory `init` that `Pre` accepts (line 474), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 735).

**What `Post` asks** (line 735), two parts:
1. for every head `b < G d` (line 364), row `r < N_CTX` and column `j < HEAD_DIM`, three
   `OutSpec` clauses (line 726), for `DV` (`dVE`, `dVTol`, lines 676, 682), `DK` (`dKE`, `dKTol`,
   lines 696, 700) and `DQ` (`dQE`, `dQTol`, lines 714, 718). Each says: element `eIdx d b r j`
   (line 367) of the output reads in `fin` as a `float16` `c`, for every element. Then, for every
   `hv : HeadV` (line 519) that `HeadV.Holds d init b hv` (line 534: the head's values in `init`)
   and `HeadV.Ok d hv` (line 646: values finite, `|L|, |Δ| ≤ 2^126`, and `fits` for every
   `Allowed` pair): there is a finite `float32` `v` with `|v.toReal − E hv| ≤ T hv` and
   `FloatV.NaNEq (FloatV.cvt v) c`; and when `|E hv| + T hv ≤ 65504`, `c` is finite and
   `|c.toReal − E hv| ≤ T hv + (|E hv| + T hv)·u16 + eta16`. The exact values `E` are in the reals
   (`pE`, `dsE`, lines 602, 606, from `hv`); the tolerances use `accB` (line 511), `stepsKV`,
   `stepsQ` (lines 667, 672). Outside `HeadV.Ok` only "holds a `float16` value" is asked;
2. the frame: every byte outside `outBytes d` (line 384) is unchanged.

**What you are given** (`Assumptions`, line 746), four fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotF16`, a `DotLiftAcc`): each `tl.dot` and `tl.dot_add` (model lines 94, 113,
  118, 125, 170, 185, 191) is within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, `L` the lifted
  magnitude, with room in `float32`;
- `dot_total` (`Rules.Num.DotTotal` at `dotCfg`, line 50): every `tl.dot` has a result;
- `exp2` (`Rules.Num.Exp2F32`): `tl.exp2` (model lines 97, 173) within `2^-22·2^x + 2^-126`;
  `exp2_total` (`Rules.Num.MathTotal`): it has a result.

**The kernel, as the model runs it** (`attn_bwd` line 208, helpers `attn_bwd_dkdv` line 62 and
`attn_bwd_dq` line 142, `model` line 449, `launch` line 469): a grid `(N_CTX/128, 1, G d)`.
Program `(pid, bhid)` offsets every pointer by its head (`adj`, `off_chz`, with `tl.irem`,
`tl.idiv` by `H`). It loads the key and value tiles of rows `pid·128 + [0, 128)`, runs
`attn_bwd_dkdv` (with `CAUSAL`, first 8 masked steps of 16 query rows from `start_n`, then the
unmasked steps of 32 rows to `N_CTX`; without, `N_CTX/32` steps from `0`), each a
`tl.forRange 0 num_steps 1` loop that loads `qT`, `m`, `do`, `Di` and accumulates `dv`, `dk` with
`tl.dot_add`; it stores `dv` and `dk·sm_scale` as `float16`. Then for query rows
`pid·128 + [0, 128)` it loads `q`, `do`, `m`, runs `attn_bwd_dq` (with `CAUSAL`, 8 masked steps
of 16 key rows on the diagonal block, then `start_m/32` steps from `0`), and stores `dq·LN2` as
`float16`. Every load and store is unmasked. Programs write disjoint tiles and use no atomics: the
route is `TVA.meets_independent`; each loop is `safe_forRange` with an invariant (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (the `Launch` default; `Launch.Valid` from `g_lt`,
  `n_le`, `n_mod`, and the block shapes `blocks d`, line 460, from `hd`);
- each program is `Safe`: the divisor `H` is nonzero (a program exists only when `G d = Z·H > 0`),
  and so are the block sizes; the `int32` values (`bhid·N_CTX`, `adj`, the pointer offsets, `curr_m`,
  `curr_n`, `num_steps`) do not wrap (`size_lt`; `TVA.Lemmas.Int`); every lane of every load and
  store is inside its tensor (`q_len` … `d_len`) and the store lanes are distinct (`safe_load`,
  `safe_store`); `safe_dot` and `safe_dot_add` (with `dot_total`), `safe_math` (with `exp2_total`),
  `safe_subc`, `safe_mul`, `safe_convert`, `safe_cvt`; the loop bodies write nothing;
- the programs' write tiles of `DV`, `DK`, `DQ` are disjoint (distinct `(pid, bhid)`; `dq_dk`,
  `dq_dv`, `dk_dv`), and no program writes what any program reads (`out_in`);
- the value part of part 1, from `HeadV.Holds` and `HeadV.Ok`: per pair, the computed exponent and
  `dP − Δ` within `eX`, `eY` (`eSub`, line 576), `P` and `dS` in `float16` within `p16Err`,
  `ds16Err`; over the steps, the accumulators within `accB` (`dot` with `DotLoop.dotAdd_step`, one
  step of `K = 32` or `16` products); the masked lanes exactly `0`; then the products by `smF` or
  `ln2F` (line 514) and the rounding to `float16` (`FloatV.cvt_err`; README `## Tolerances`);
- the bytes no program writes give part 2.

`z_le` (`BATCH ≤ 8192`, real batch sizes) is not needed by the derivation: `g_lt` and `size_lt`
already bound the heads and the offsets.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `canon_ok`,
`smTest_ok`, `smBench_ok` and the `setup_*_ok` theorems; `elems_bytes` (line 802) can be reused.

## Specification

For head `b < BATCH·N_HEAD` write `Q`, `K`, `V`, `dO` for its `N_CTX × HEAD_DIM` blocks of `q`,
`arg_k`, `v`, `do`, and `L`, `Δ` for its rows of `M` and `delta`, all as `float16`/`float32` values
in the initial memory (`HeadV`, `HeadV.Holds`). For a key row `n` and a query row `m`, `Allowed n m`
is `CAUSAL = false ∨ n ≤ m`, and in ℝ, exactly:

- `P[n, m] = 2^(Σ_c K[n, c]·Q[m, c] − L[m])` when `Allowed`, else `0` (`pE`);
- `dS[n, m] = P[n, m]·(Σ_c V[n, c]·dO[m, c] − Δ[m])` (`dsE`);
- `dV[n, j] = Σ_m P[n, m]·dO[m, j]` (`dVE`);
- `dK[n, j] = s·Σ_m dS[n, m]·Q[m, j]`, `s` the `float32` `sm_scale` the kernel receives (`dKE`);
- `dQ[m, j] = LN2·Σ_n dS[n, m]·K[n, j]`, `LN2` the kernel's `float32` constant
  `0x3f317218 = 1453635/2097152` (`dQE`).

`Post`: for every head `b`, row `r < N_CTX` and column `j < HEAD_DIM`, each of `dv`, `dk`, `dq`
holds at element `(b, r, j)` some `float16` value `c` (`OutSpec`); whenever head `b`'s values are
covered (`HeadV.Ok`), `c` is the `float16` rounding (nearest even, up to NaN payload) of a finite
`float32` value `v` with `|v − E| ≤ T` (`E` the exact value, `T` its tolerance `dVTol`, `dKTol`,
`dQTol`), and when `|E| + T ≤ 65504`, `c` is finite and `|c − E| ≤ T + (|E| + T)·2^-11 + 2^-25`.
Every byte outside the three outputs' elements is unchanged.

With the host's inputs these are the gradients of attention: `arg_k = k·sm_scale/ln 2` (rounded
to `float16`), `M` is the row's base-2 log-sum-exp of the scaled scores, so `P` is the softmax
probability, `delta = Σ_j O·dO`, and `LN2·K = sm_scale·k`. The claim does not assume these
relations: it speaks of the values in memory, whatever they are.

`HeadV.Ok`, the premise of the accuracy clauses (not of `Pre`: the launch ends normally and the
frame holds for any values): `Q`, `K`, `V`, `dO`, `L`, `Δ` finite, `|L|, |Δ| ≤ 2^126`, and for every
`Allowed` pair `xE + eX ≤ 15` (the exponent with its error bound) and `pBr·yB ≤ 2^15` (the bound
on `|P·(dP − Δ)|`). On the host's inputs `P ≤ 1` and `dS` is small, so they hold with room (the GPU
check finds them true on every head it checks).

## Conditions

`Inst`: the shapes `Z`, `H`, `N_CTX`, `HEAD_DIM`, `CAUSAL`, the double `sm_scale`, and the nine
buffers, in the contiguous layout (element `(z, h, r, c)` is `((z·H + h)·N_CTX + r)·HEAD_DIM + c`,
`eIdx`; of `M`/`delta` `(z·H + h)·N_CTX + r`, `rIdx`); the model passes the strides
`H·N_CTX·HEAD_DIM`, `N_CTX·HEAD_DIM`, `HEAD_DIM`, `1`. `Inst.ok`:

- `hd`: `HEAD_DIM ∈ {16, 32, 64, 128, 256}` (the forward's assertion, line 512);
- `n_mod`: `N_CTX % 128 = 0` (line 599); `n_le`: `N_CTX ≤ 2^14` (the benchmark's largest);
- `z_le`: `BATCH ≤ 8192` (`BATCH` is a batch size, the number of sequences): real model shapes only
  (Exclusions);
- `g_lt`: `BATCH·N_HEAD < 2^16` (the grid's axis 2); `size_lt`:
  `(BATCH·N_HEAD − 1)·N_CTX·HEAD_DIM < 2^31` (the kernel's `int32` offsets, a limit of the kernel)
  and `N_HEAD·N_CTX·HEAD_DIM < 2^31` (the model's `int32` argument `stride_z`; outside real use
  beyond);
- `sm`: the `float32` of `sm_scale` finite, `|s| ≤ 2^16` (the host's `0.5`, `1.3`);
- each storage holds its tensor; the outputs share no byte with the inputs or with each other
  (`torch.empty_like`).

`Pre`: the nine buffers valid. Values unconstrained.

The model follows lines 266–500: the two helpers as Lean functions applied at their calls, the
compile-time `if CAUSAL` and `if MASK` as Lean `if`s, names assigned Python integers as `int32`
scalars, `LN2` built in `float32`, `tl.dot(…) − x` as the fusable `tl.subc`, `acc += tl.dot(a, b)`
as `tl.dot_add`, `.to(tl.float16)` as `tl.cvt`, `tl.where(mask, x, 0.0)` with `+0.0`; `stride_d = 1`
(and `N_HEAD = 1`) is the compile-time `1`, which each use builds in `int32`.

`launch d`: the grid `(N_CTX/128, 1, BATCH·N_HEAD)`, the source's block shapes, `fusion = true`,
`num_ctas = 1`, no `launch_pdl`, `debug = false`. `num_warps=4`, `num_stages=5` are the only
options (scope 4's options that do not change the meaning); a direct `kernel[grid](…)` launch (no
`@triton.autotune` or `@triton.heuristics` on this kernel), no descriptors (scope 5, 8), no
`CompiledKernel` (scope 6): within part (A) of `Compiler.Faithful`'s scope. The claim relies on part (B)
(the environment) as the rule states it.

## Assumptions

All `[rule]`: `faithful` (`Compiler.Faithful`); `dot` (`Num.DotF16`, the seven `tl.dot`s of
`float16` operands into `float32`, lines 293, 303, 307, 310, 345, 353, 358); `dot_total`
(`Num.DotTotal` for the model's `DotCfg`); `exp2` (`Num.Exp2F32`, lines 294, 346); `exp2_total`
(`Num.MathTotal P .exp2 .binary32`). `consistent` uses `Platform.sm90ref`.

## Coverage

`canon Z H N D c sm` is the host's launch (contiguous tensors, fresh outputs); `canon_ok` proves
`Inst.ok` from numeric facts. `SETUP.md`'s launches: `setup_test_ok` (all 48 test launches, with
`sm_scale = 0.5`, `smTest_ok`), `setup_bench_ok` (all 20 benchmark launches at `BATCH = 4`,
`N_HEADS = 32`, `sm_scale = 1.3`, `smBench_ok`), `setup_test_large_ok`, `setup_bench_large_ok`;
`setup_headdims_ok` covers `HEAD_DIM` 16, 32, 256 at the benchmark's largest shape.

## Tolerances

From the library's facts (`docs/rules/num.md`): `DotF16` (`relDot16 K = (K+2)·2^-22` relative to
`|acc| + Σ liftMag a·liftMag b`, plus `absDot16 K = (K+2)·2^-126`), `Exp2F32` (`2^-22·2^x + 2^-126`),
and the IEEE rounding lemmas (`u32 = 2^-24`, `eta32 = 2^-150`, `u16 = 2^-11`, `eta16 = 2^-25`):

- per pair, `eSub` bounds a computed `tl.dot(a, b) − x` (the dot over `HEAD_DIM` products,
  `rH = relDot16 HEAD_DIM` of the lifted magnitude `aQK` or `aVD` plus `aH`, then a fusable
  subtraction, `3u32` of the operands plus `4eta32`); `p16Err` adds `tl.math.exp2`'s error at the
  largest exponent the error allows (`pBr − 2^xE`) and one `float16` rounding; `ds16Err` the same
  for `dP − Δ` (`eY`), the IEEE product with `P` and the rounding of `dS`; `0` on masked pairs
  (exactly `0` in the kernel);
- the accumulation over `T` steps of `acc += tl.dot(a, b)` (32 or 16 products each):
  `accB T B = ((1 + ρ)^T − 1)·B + T·(1 + ρ)^T·α`, `ρ = DotLoop.stepRel relDot16 32 ≈ 8.3·10^-6`,
  `α = DotLoop.stepAbs absDot16 32`, `B` the sum of the bounds on the operands' lifted magnitudes
  (`p16L·lm dO`, `ds16L·lm Q`, `ds16L·lm K`, with `lift16` counting a `float16` that may be a
  nonzero subnormal at `2^-14`); `T` is `stepsKV`/`stepsQ`, the steps the program takes for that
  row (`N_CTX/32`, or with `CAUSAL` `8` masked steps plus the rest);
- for `dk` and `dq`, the product by `s` or `LN2` scales that, plus one `float32` rounding.

Provable: each is a composition of the cited facts with the library's `DotLift.dotAdd_step` (one
step of a `K` loop) and the rounding lemmas, and `HeadV.Ok` keeps every intermediate in range (the
sums are below `2^47`, so the products by `s ≤ 2^16` and `LN2` have room in `float32`).
Meaningful: against the outputs themselves, `tol / |output|` (the tolerance over the exact value,
per element) has median `1.8–2.1%` (dV, dK, dQ) at the test's `Z = 1, H = 2, N_CTX = 1024,
HEAD_DIM = 64` (90th percentile `0.095–0.11`) and `2.7–2.8%` at `N_CTX = 4096, HEAD_DIM = 128`,
causal (90th percentile `0.15–0.16`); on the benchmark's `randn` inputs (`sm_scale = 1.3`) the median
is `1.0–1.9%` at `N_CTX = 4096`, causal, and `1.6–2.8%` at `2^14`, but the 90th percentile reaches
`2.2` (dV) and `2.6` (dQ) at `4096` and `1.2` (dQ) at `2^14`: an output that cancels to near `0` keeps
the floor of the `float16` roundings of `P` and `dS`, `2^-25` per key (about `N_CTX·2^-25` times
`|dO|`, `|Q|` or `|K|`), so on those elements the tolerance exceeds the output. Against the sums it bounds, `dVTol` is about `0.2%`
of `Σ_m P·|dO|` at the test's sizes and
`0.6–0.9%` at the benchmark's (`ln 2·(HEAD_DIM + 2)·2^-22·A` from the exponent's dot error, `A` the
weighted mean of `aQK`; `2^-11` from `P`'s rounding; `(1 + ρ)^T − 1 ≤ 0.43%` at `T ≤ 516`), and the
same relative part bounds `dKTol`, `dQTol` against `|s|·Σ|dS·Q|`, `LN2·Σ|dS·K|` plus the error of
`dP − Δ`, which does not shrink when `dS` cancels. The GPU check finds the kernel's float32 error at
up to `0.73` of `T`: the bound is within a factor `1.4` of the kernel's worst error.

## Exclusions

* **Batches of more than 8192 requests** — out of scope: the claim covers real model shapes only.
  The field: `Z` (`BATCH`, `z_le`: at most `8192`). Every launch of the file is covered
  (`BATCH ≤ 4`). No other field takes these bounds: `N_CTX`, the context length, is at most `2^14`
  (`n_le`), within `2^22`; `HEAD_DIM` is a head dimension and `N_HEAD` a number of heads, not
  model dimensions.
* **`int32` offsets (a kernel limit).** `bhid * N_CTX` and `adj = stride_h·(bhid % H) +
  stride_z·(bhid // H)` (lines 384–385) are `int32`; `adj` reaches the last head's first element,
  `(BATCH·N_HEAD − 1)·N_CTX·HEAD_DIM`, and wraps once that is `2^31` or more (`size_lt`). The first
  host launch that hits it has `BATCH·N_HEAD = 1025` heads at `16384 × 128`; the file's launches
  have at most `128` heads of `16384 × 128`.
* **`N_HEAD·N_CTX·HEAD_DIM ≥ 2^31`** (`size_lt`'s second part): outside real use (the file's
  largest is `32` heads of `16384 × 128`, `2^26` elements). Not a kernel limit: `stride_z =
  q.stride(0)` is then an `int64` argument, so `stride_z·(bhid // H)` is an `int64` product that does
  not wrap; with `BATCH ≥ 2` the first part already excludes it, and at `BATCH = 1` the product is
  `0`. The claim's model takes `stride_z` as an `int32` argument.
* **`N_CTX > 2^14`**: outside the file's launches (its largest is `2^14`). Not a kernel limit.
* **Layouts other than contiguous**: the host asserts `do.is_contiguous()` and equal strides; the
  two strides those checks leave free (`stride(0)` when `BATCH = 1`, `stride(1)` when `N_HEAD = 1`,
  and any strides of an empty tensor) are multiplied by `0` or run no program, but are outside the
  claim's `Inst`: outside real use.
* **`BATCH·N_HEAD ≥ 2^16`** (CUDA's grid limit): no launch.
* **An `int` `sm_scale`** (a different parameter type) and a `float32` `sm_scale` above `2^16`:
  outside real use (the file passes `0.5` and `1.3`).
* **Values outside `HeadV.Ok`** (non-finite inputs, `P` beyond `2^15`): covered with a weaker
  `Post` (a `float16` value is stored, nothing else changes); outside real use.
* **Other dtypes** (`float8e5`): the file never runs the backward pass in fp8 (the test skips it,
  line 643; the benchmark's `triton-fp8` provider runs the `float16` backward).

## Library requests

None.

## Checks

* **Builds**: `lake env lean ../claims/attn_bwd_f16/Claim.lean`: no errors or warnings; `consistent`,
  `satisfiable` and the coverage facts use only `propext`, `Classical.choice`, `Quot.sound`.
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (the test's `Z = 1`, `H = 2`,
  `N_CTX = 128`, `HEAD_DIM = 64`, not causal, a memory of zeros).
* **Covers the setup**: `canon_ok`, `setup_test_ok`, `setup_bench_ok`, `setup_test_large_ok`,
  `setup_bench_large_ok`, `setup_headdims_ok`.
* **Model map**: every code line covered, every model text found in `Claim.lean` (the gate's
  `map_problems`).
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 48 test launches and the benchmark's 20 sizes
  and modes (at `1·2` heads, and one at `4·32` heads), through the host's `attention(...)
  .backward(dout)`; `HEAD_DIM` 16 and 32 through `_attention.backward` on host-computed forward
  tensors (the forward kernel rejects them); `Z·H = 1`; sharper scores (inputs of std 2 at
  `sm_scale = 1.3`) and a small `sm_scale = 2^-6`. On 411 heads, `HeadV.Ok` holds, every output
  is the `float16` rounding of a value within `T` of `E` (the float32 part at most `0.73·T`: dV
  `0.73`, dK `0.64`, dQ `0.54`), and the worst `|c − E|` over `T + (|E| + T)·2^-11 + 2^-25` is
  `0.98` (dV, dK; outputs near `10^-8`, where the store's subnormal rounding is most of the bound)
  and `0.66` (dQ). `HEAD_DIM = 256` is rejected on the host (shared memory). Inputs unchanged bit
  for bit; the bytes past `dq`, `dk`, `dv` and `delta` keep their sentinel.
