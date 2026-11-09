# Claim: `moe_router_bf16`

Kernel: `moe_router_bf16` (SGLang's `fused_moe_router_cudacore_kernel`: per token, the router
logits `x · W^T` summed in `float32`, the soft cap `30·tanh(l/30)` written with `exp`, the top-2
experts by `tl.argmax`, and their softmax weights). Claim: [`Claim.lean`](Claim.lean), namespace
`Claims.MoeRouter`; the statement to prove is `Statement : ∀ K P, Assumptions K P → Spec K`. Model
map: [`model-map.json`](model-map.json). GPU check: [`check_gpu.py`](check_gpu.py).

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`. For each token `t < bs` (`TokenPost`), with
`mu t e = Σ_h x[t,h]·W[e,h]` (exact, from the initial memory) and `absS t e = Σ_h |x[t,h]·W[e,h]|`:
there are reals `C 0 … C 7` (the capped logits the run computes) with

* `|C e − 30·tanh(mu t e / 30)| ≤ capTol (mu t e) (absS t e)` for every expert (`capRef`);
* `topk_ids[t, 0] = i0`, the first index of the largest `C` (Triton's `tie_break_left`), and
  `topk_ids[t, 1] = i1 ≠ i0`, the first index of the largest `C` among the other experts;
* `topk_weights[t, 0]`, `topk_weights[t, 1]` finite, within `wRel0·σ0 + wFloor` and
  `wRel1·σ1 + wFloor` of `σ0 = 1/Σ_k exp(C k − C i0)` and `σ1 = exp(C i1 − C i0)/Σ_k exp(C k − C i0)`
  (`softRef`), the softmax of `C` over all 8 experts (no renormalisation, as the kernel);

and every byte outside the `2·bs` elements of each output is unchanged. The reference is the
host's inputs alone (`mu`, `absS`, `capRef`); `C` is pinned to it by `capTol`, and the ids and
weights are pinned to `C`. Composed: the ids are a top-2 of the exact capped logits up to `capTol`,
and each weight is within `wRel·σ + wFloor` of the softmax of logits within `capTol` of the exact
ones, so within a relative `e^{2·max capTol} − 1` (plus `wRel`) of the exact softmax: at Grok's
shape about `10^-3` (Tolerances).

The tolerances (`Claim.lean`, "The tolerances"; `u = 2^-24`, `η = 2^-150`, `B = BLOCK_SIZE`,
`h = hidden_dim`):
* `tolL S = ((1+3u)^D − 1)·((1+u)·S + h·η) + 4η·(B−1)·(1+3u)^D + u·S + h·η`, `D = ⌈log₂ B⌉`
  (`Nat.clog 2 B`; `13` at `B = 8192`);
* `capTol μ S = (1 + 2^-21)·tolL S + 2^-21·|μ| + 2^-14`;
* `wRel0 = wRel1 = 2^-14`, `wFloor = 2^-130`.

## Conditions

`Inst`: `bs`, `hidden_dim`, `BLOCK_SIZE`, and the buffers `x` (`bfloat16 [bs, hidden_dim]`),
`router_weight` (`float32 [8, hidden_dim]`), `topk_weights` (`float32 [bs, 2]`), `topk_ids`
(`int32 [bs, 2]`). The compile-time values are SETUP.md's: `num_experts = 8`, `topk = 2`,
`moe_softcapping = 30.0`, `is_correction_bias = False` (`correction_bias = None`),
`moe_renormalize = False`. `launch`: the grid `(bs,)`, the block values `[8, BLOCK_SIZE]`,
`[BLOCK_SIZE]` and `[8]`, `enable_fp_fusion` at its default; `num_warps` (the host's formula) is
performance-only.

`Inst.ok`: `1 ≤ hidden_dim ≤ BLOCK_SIZE ≤ 2^13`, `BLOCK_SIZE` a power of two (the host passes
`next_power_of_2(hidden_dim)`; any larger power of two is covered too); `bs ≥ 1`,
`(bs − 1)·hidden_dim < 2^31` and `2·bs ≤ 2^31` (the `int32` products and offsets: line 80 is
`(input_ptr + pid * hidden_dim) + offsets`, so only `pid * hidden_dim` is computed in `int32`); the buffers
hold their tensors; the outputs share no byte with the inputs or each other.

`Pre`: the four buffers are valid; every input element finite; and, the kernel's `float32` limits
(Exclusions), for every token and expert `absS ≤ 2^120` and `mu + capTol ≤ 1320` (`= 44·30`).

The launches are within part (A) of `Compiler.Faithful`'s scope: a bare `@triton.jit`, no autotune
or heuristics, the launch gives only `num_warps`, no descriptors, no `CompiledKernel`, no host
`TensorDescriptor`. Part (B), the environment, is assumed and not checked.

## Assumptions

`[rule]` fields: `faithful` (`Rules.Compiler.Faithful P (model d) (launch d) (K d)`),
`exp_total` (`Rules.Num.MathTotal P .exp .binary32`), `exp_acc` (`Rules.Num.ExpF32 P`, lines 90,
109, 127), `div_total` (`Rules.Num.DivTotal P .binary32`), `div_full` (`Rules.Num.DivFull P`, lines
89, 93, 109), `sum_depth` (`Rules.Num.SumDepth P`: the float `tl.sum`s of lines 83, 109 and 124 are
trees of depth at most `⌈log₂ n⌉`). `consistent` instantiates them at `Platform.sm90ref`. The
kernel has no loop around a reduction (the loop of line 139 runs no trip at `topk = 2`), so the
reductions are not in the loop form the library excludes (a loop updating a carried value with a
reduction of a loaded tile).

## Coverage

`SETUP.md` asks for Grok's launches: the compile-time values above, `hidden_dim` 6144 and 8192
(`BLOCK_SIZE = 8192`), `bs` 1 to 511. The source's eight Grok test cases (line 215: `bs` 1, 7, 128,
511 × `hidden_dim` 6144, 8192) are `setup_<bs>_<hidden>`, each with `setup_<bs>_<hidden>_ok`
proved; `satisfiable` uses `setup_1_6144` in a zero memory. `Inst.ok` also covers every
`hidden_dim ≤ 8192` and every `bs` up to the `int32` limit (262,144 tokens at 8192).

## Tolerances

From the library's facts (`FloatV.mul_err`, `sub_err`, `add_err`; `Tree.errBound` through
`sumc_err_depth` and `sum_err_depth` with the depth `Rules.Num.SumDepth` gives
(`SumDepth.depth_le`); `Rules.Num.ExpF32`, `DivFull`), on finite inputs with `absS ≤ 2^120` (so
every sum stays finite) and `mu + capTol ≤ 1320`:

1. **Logit** (`tolL`). Each lane's product `fl(w·x)` is within `u·|wx| + η`; the masked lanes are
   `0·0 = +0`. The row sum (`tl.sum1c`, fusable) over the `B` lanes is a tree of depth at most
   `D = ⌈log₂ B⌉` (`SumDepth`), so within `Tree.errBound (3u) (4η) D (B−1) A` of the sum of the
   products, `A ≤ (1+u)·S + h·η` their magnitudes (`sumc_err_depth`). Adding the products' errors
   gives `tolL`: at `B = 8192`, `(1+3u)^13 − 1 ≈ 39u`, so `tolL ≈ 40u·S = 2.4·10^-6·S`.
2. **Capped logit** (`capTol`). `s = l/30` (`DivFull`: `2^-22·|l/30| + 2^-148`), `t = fl(2s)`
   (`u·|2s| + η`), so `|t − 2μ/30| ≤ τ = tolL/15 + 2^-21·(|μ| + tolL)/15 + 2^-146`. The map
   `t ↦ 30·tanh(t/2)` is 15-Lipschitz, giving `15τ`. For the formula at the computed `t`: `t ≤ 88`
   (from `mu + capTol ≤ 1320`) gives `ExpF32`'s room at `X = |t|`; `|ẽ − e^t| ≤ rel·e^t + 2^-140`,
   and `g(y) = (y−1)/(y+1)` moves by `2|ẽ − e^t|/((ẽ+1)(e^t+1))`, at most `60·max_t
   (2^-22 + 2^-23|t|)·e^t/(e^t+1)^2 ≤ 60·7.2e-8 = 4.3e-6` after the factor 30; `ẽ ∓ 1` and the
   division add at most `30·(2.0001u + 2^-22·(1 + 2^-20)) = 1.07e-5`; `·30` adds `30u·1.0001 + η =
   1.8e-6`. Total formula error `1.7e-5 < 2^-14 = 6.1e-5` (margin 3.6), so
   `|C − capRef μ| ≤ 15τ + 1.7e-5 ≤ capTol`. On the way, `|C| ≤ 30·(1 + 8e-7)`.
3. **First weight** (`wRel0`). `top1_v = C i0` exactly (`tl.max`). `C k − top1_v` with `tl.subc`
   (each operand may enter unrounded): within `≈ 4u·30 = 120u = 7.2e-6`. `tl.exp` at `X ≤ 60.0001`:
   relative `2^-22 + 2^-23·60.0001 = 7.39e-6`; with the input error, each exponential within a
   relative `1.459e-5` of `exp(C k − C i0)` (plus `2^-140`). The sum of the 8 positive terms
   (`tl.sum`, a tree of depth 3 by `SumDepth`, `sum_err_depth`) adds `(1+u)^3 − 1 = 1.8e-7`, `1.0 /`
   adds `2^-22 = 2.4e-7`: relative `1.50e-5` in all, `wRel0 = 2^-14 = 6.10e-5` (margin 4); absolute
   terms below `2^-136 ≤ wFloor`.
4. **Second weight** (`wRel1`). `top2_v` (`tl.sumc` of `C·onehot`, the products exact, the leaves
   `C i1` and seven zeros) along a tree of depth 3 over the 8 leaves (`SumDepth`, `sumc_err_depth`):
   within `((1+3u)^3 − 1)·|C i1| + 28η·(1+3u)^3 ≤ 9.0001u·30.00003 = 1.61e-5`; `top2_v − top1_v`
   (IEEE) adds `60u = 3.6e-6`; `tl.exp` adds `7.39e-6`; so `exp(…)` is within a relative
   `2.71e-5` (second-order terms below `10^-9`); times `invsumexp` (relative `1.50e-5`) and the
   product's rounding (`u = 6.0e-8`): `4.22e-5`, `wRel1 = 2^-14 = 6.10e-5` (margin 1.4).

**Meaningful at the covered sizes.** `check_gpu.py` measures, at Grok's shape (`bs = 511`,
`hidden_dim` 6144 and 8192, `BLOCK_SIZE = 8192`):

| data | `capTol / |capRef|` median (p90) | composed weight tolerance, median | tokens rejecting: no soft cap / swapped ids |
|---|---|---|---|
| logits of std 3, hidden 6144 | `2.1·10^-4` (`1.1·10^-3`) | `9.1·10^-4` | `0.97` / `1.00` |
| logits of std 3, hidden 8192 | `2.4·10^-4` (`1.2·10^-3`) | `1.0·10^-3` | `0.96` / `0.998` |
| the source's test data, hidden 6144 | `4.7·10^-4` (`2.5·10^-3`) | `2.6·10^-4` | `0.014` / `1.00` |
| the source's test data, hidden 8192 | `4.6·10^-4` (`2.6·10^-3`) | `2.8·10^-4` | `0.025` / `1.00` |

`capTol` is about `10^-4` absolute there (`tolL ≈ 2.4·10^-6·absS` plus the cap formula's `2^-14`),
against the kernel's real error of about `10^-6`: the capped logits reach `2·10^-2` of `capTol`, the
weights `0.017` and `0.036` of theirs. The composed weight tolerance is `e^{2·max capTol} − 1 +
wRel1`. A kernel without the soft cap (the ids and weights of the raw logits) fails `Post`: on a
token it is rejected when a weight lies outside the range of the softmax over every `C` within
`capTol`, which happens on almost every token of logits of standard deviation 3 and on a few tokens
of the test's small logits (below 2, where the cap changes a logit by less than `3·10^-4`); one
rejected token rejects the run. Swapped ids are rejected on every token whose two largest logits
are more than `2·capTol` apart. The bound along any tree (depth `B − 1`), which the library states
without `SumDepth`, gives at the same data `capTol / |capRef|` median `0.11`–`0.13` (p90
`0.59`–`0.72`), a composed weight tolerance of `0.05`–`0.67`, rejects a kernel without the soft cap
on no token, and swapped ids on `76`–`83%` of them.

## Exclusions

What `SETUP.md` allows and the claim leaves out:

* **Logits above `1320 − capTol`** (`Pre`: `mu + capTol ≤ 1320`): a limit of the kernel's soft cap.
  The source computes `30·tanh(l/30)` as `(e^{2l/30} − 1)/(e^{2l/30} + 1)·30`; for a computed logit
  above about `1330.8` (`= 88.72·15`, about `44.4·cap`) `exp` overflows to `inf` and `inf/inf` is a
  NaN, which `tl.argmax` and the weights propagate. The claim also leaves out the band where the
  exact logit is below that but its proven error bound `capTol` reaches past `1320`, since the
  computed logit may be past the limit there. First launch
  that hits the kernel limit: any token whose logit exceeds about `1331`. On the H200, with `x = 1`
  over 8192 columns and one gate row at `l/8192`, `l = 1330.7` gives the right ids and weights and
  `l = 1331.0` gives `topk_ids = [0, 1]` and NaN weights. Real use does not reach it: Grok's router logits are of order
  1 to 10 (the cap at 30 bounds the *output*, the inputs stay small), and the test's are below 2.
* **`absS > 2^120`**, non-finite inputs: the `float32` sum (or an input) is not finite; a limit of
  `float32`, never reached by real hidden states and gate weights.
* **`(bs − 1)·hidden_dim ≥ 2^31`**: a limit of the kernel, the `int32` product `pid * hidden_dim`
  (line 80) wraps. First launch: `hidden_dim = 8192`, `bs = 262,145` (a 4 GiB `x`). Real use:
  `bs < 512` on this path.
* **`hidden_dim > 8192`** (`BLOCK_SIZE > 2^13`) and the other compile-time values (a correction
  bias, cap `0`, `topk` other than 2, other expert counts), including the five CPU-grid cases of the
  source's test (lines 216–217: `(17, 64, 8, top-1, cap 0)`, `(128, 257, 16, top-4, bias)`,
  `(17, 257, 32, top-2, cap 0, bias)`, `(128, 64, 8, top-4, cap 0, bias)`, `(1, 4096, 8, top-8)`):
  not part of the setup (`SETUP.md`; SGLang's only GPU caller is Grok).

## Checks

* **Build**: `lake env lean ../claims/moe_router_bf16/Claim.lean` from `lean/` builds with no error, no
  `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (at `Platform.sm90ref`) and `satisfiable` (`setup_1_6144` in a zero
  memory) are proved.
* **Coverage**: the eight `setup_<bs>_<hidden>_ok` are proved.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 27 launches
  hold: the eight Grok cases; `hidden_dim` 1, 3, 5, 64, 257, 4097, 8191 and 1024; `bs` 2048, 32,768
  and 262,144; all-zero inputs (every expert tied); two identical gate rows; a logit near `1300`
  and one near `−98,000`; `x` at `2^-126` with `W` near `2^120`; hidden states scaled by 64 (logits
  up to 86); four launches at `bs = 511`, hidden 6144 and 8192, with the test's data and with
  logits of standard deviation 3, which also run the negative tests (Tolerances). The witness `C` is
  the probe kernel's capped logits (the source's lines 66–93, verbatim): worst error / tolerance
  `3.2e-2` (capped logits), `0.017` (weight 0), `0.036` (weight 1); the ids are the first argmaxes of `C` on every token; `Pre` holds; the byte frame
  holds (the inputs unchanged, and guard bands of 4096 elements on each side of both outputs, in
  which the check places the host's two allocations, unchanged).

## Library requests

None.

## Notes

* The cap chain's constant `2^-14` and `wRel0 = wRel1 = 2^-14` are derived above by hand (margins
  3.6, 4 and 1.4); a proof carries the second-order terms within those margins.
* The witness `C` of the GPU check is the probe kernel's output, not the kernel's internal value;
  it is a valid witness because every clause is checked against it.
