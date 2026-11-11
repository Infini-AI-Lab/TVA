# `moe_router_bf16`

**The kernel.** `fused_moe_router_cudacore_kernel` of SGLang (`source/moe_router.py`, kernel lines
51–152; host `fused_moe_router_cudacore` lines 155–195). One program per token: the 8 router logits
`x · W^T` summed in `float32`, the soft cap `30·tanh(l/30)` written with `exp`, the top-2 experts by
`tl.argmax`, and their softmax weights. The claim is `Claims.MoeRouter.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for each
token `t < bs`, with `μ_e = Σ_h x[t,h]·W[e,h]` and `S_e = Σ_h |x[t,h]·W[e,h]|` (exact, from the
initial memory), there are reals `C 0 … C 7` (the capped logits the run computes) with
- `|C e − 30·tanh(μ_e / 30)| ≤ capEps S_e = 2^-18·S_e + 2^-14` for every expert;
- `topk_ids[t, 0] = i0`, the first index of the largest `C` (`tie_break_left`), and
  `topk_ids[t, 1] = i1 ≠ i0`, the first index of the largest `C` among the other experts;
- `topk_weights[t, 0]`, `topk_weights[t, 1]` finite, within `2^-14·σ + 2^-130` of
  `σ0 = 1/Σ_k exp(C k − C i0)` and `σ1 = exp(C i1 − C i0)/Σ_k exp(C k − C i0)`, the softmax of `C`
  over all 8 experts (no renormalisation, as the kernel);

and every byte outside the `2·bs` elements of each output is unchanged.

## Conditions

`Inst.ok` and `Pre`, for the compile-time values of `SETUP.md` (`num_experts = 8`, `topk = 2`,
`moe_softcapping = 30.0`, no correction bias, `moe_renormalize = False`):
- `1 ≤ hidden_dim ≤ BLOCK_SIZE ≤ 2^13`, `BLOCK_SIZE` a power of two; `hidden_dim` a multiple of
  `128`, at most `2^17`;
- `1 ≤ bs ≤ 2^22` (a number of tokens), `(bs − 1)·hidden_dim < 2^31` and `2·bs ≤ 2^31` (the
  `int32` product `pid * hidden_dim` of line 80 and the output offsets do not wrap);
- the buffers hold their tensors, are valid in the initial memory, and the outputs share no byte
  with the inputs or with each other;
- every input element is finite, and for every token and expert `S_e ≤ 2^120` and
  `μ_e + capTol μ_e S_e ≤ 1320` (`= 44·30`), with the fine bound
  `capTol μ S = (1 + 2^-21)·tolL S + 2^-21·|μ| + 2^-14`,
  `tolL S = ((1+3u)^D − 1)·((1+u)·S + h·η) + 4η·(B−1)·(1+3u)^D + u·S + h·η`, `D = ⌈log₂ B⌉`.

**The launches covered** (`SETUP.md`): the source's eight Grok test cases (`bs` 1, 7, 128, 511 ×
`hidden_dim` 6144, 8192, `BLOCK_SIZE = 8192`), `setup_<bs>_<hidden>_ok`.

**Device facts** the tolerances rest on (all in `Device P`): `Device.ExpF32`, `Device.DivFull`,
`Device.SumDepth` (the float `tl.sum`s of lines 83, 109 and 124 are trees of depth at most
`⌈log₂ n⌉`), and the totality facts for `exp` and `/`.

## Tolerances

From IEEE rounding (`u = 2^-24`, `η = 2^-150` per `float32` operation), the error of a tree sum of
depth `D` (`3u` per level for a fusable sum, `u` for an IEEE one), the depth bound
`Device.SumDepth`, and `Device.ExpF32`, `Device.DivFull`, on finite inputs with `absS ≤ 2^120` (so
every sum stays finite) and `mu + capTol ≤ 1320`:

1. **Logit** (`tolL`). Each lane's product `fl(w·x)` is within `u·|wx| + η`; the masked lanes are
   `0·0 = +0`. The row sum (`tl.sum1c`, fusable) over the `B` lanes is a tree of depth at most
   `D = ⌈log₂ B⌉` (`SumDepth`), so within `errBound (3u) (4η) D (B−1) A` of the sum of the
   products, `A ≤ (1+u)·S + h·η` their magnitudes. Adding the products' errors
   gives `tolL`: at `B = 8192`, `(1+3u)^13 − 1 ≈ 39u`, so `tolL ≈ 40u·S = 2.4·10^-6·S`.
2. **Capped logit** (`capTol`). `s = l/30` (`DivFull`: `2^-22·|l/30| + 2^-148`), `t = fl(2s)`
   (`u·|2s| + η`), so `|t − 2μ/30| ≤ τ = tolL/15 + 2^-21·(|μ| + tolL)/15 + 2^-146`. The map
   `t ↦ 30·tanh(t/2)` is 15-Lipschitz, giving `15τ`. For the formula at the computed `t`: `t ≤ 88`
   (from `mu + capTol ≤ 1320`) gives `ExpF32`'s room at `X = |t|`; `|ẽ − e^t| ≤ rel·e^t + 2^-140`,
   and `g(y) = (y−1)/(y+1)` moves by `2|ẽ − e^t|/((ẽ+1)(e^t+1))`, at most `60·max_t
   (2^-22 + 2^-23|t|)·e^t/(e^t+1)^2 ≤ 60·7.2e-8 = 4.3e-6` after the factor 30; `ẽ ∓ 1` and the
   division add at most `30·(2.0001u + 2^-22·(1 + 2^-20)) = 1.07e-5`; `·30` adds `30u·1.0001 + η =
   1.8e-6`. Total formula error `φ ≤ 1.71e-5 < 2^-14 = 6.1e-5` (margin 3.6, second-order terms
   included), so `|C − capRef μ| ≤ 15τ + φ ≤ (1 + 2^-21)·tolL + 2^-21·|μ| + φ`, the fine bound
   (`capTol` is it with `2^-14` for `φ`). On the way, `|C| ≤ 30·(1 + 8e-7)`.
2'. **Relaxation to `capEps = 2^-18·S + 2^-14`.** `B ≤ 2^13` gives `D ≤ 13`, so
   `(1+3u)^D − 1 ≤ 3Du·(1 + 3Du) ≤ 39.0001u`, and `tolL ≤ (39.0001u·(1+u) + u)·S + a₀ ≤
   40.0002u·S + a₀` with `a₀ = 39.0001u·hη + 4η·(B−1)·1.0000024 + hη < 2^-134` (`h ≤ B ≤ 2^13`).
   With `|μ| ≤ S` (the triangle inequality), the fine bound is at most
   `(1 + 2^-21)·40.0002u·S + 8u·S + 2^-133.9 + φ ≤ 48.001u·S + 1.72e-5`, below
   `64u·S + 2^-14 = capEps` (`2^-18 = 64u`; margin `1.33` on the `S` part at `D = 13`, `3.5` on the
   constant). `capTol ≤ capEps` too whenever `S ≥ 2^-113` (the `S` part's slack `16u·S` covers `a₀`);
   below that, `capTol` exceeds `capEps` by at most `2^-133`, and the route goes through `φ`, not
   through `capTol`'s `2^-14`.
3. **First weight** (`wRel0`). `top1_v = C i0` exactly (`tl.max`). `C k − top1_v` with `tl.subc`
   (each operand may enter unrounded): within `≈ 4u·30 = 120u = 7.2e-6`. `tl.exp` at `X ≤ 60.0001`:
   relative `2^-22 + 2^-23·60.0001 = 7.39e-6`; with the input error, each exponential within a
   relative `1.459e-5` of `exp(C k − C i0)` (plus `2^-140`). The sum of the 8 positive terms
   (`tl.sum`, a tree of depth 3 by `SumDepth`) adds `(1+u)^3 − 1 = 1.8e-7`, `1.0 /`
   adds `2^-22 = 2.4e-7`: relative `1.50e-5` in all, `wRel0 = 2^-14 = 6.10e-5` (margin 4); absolute
   terms below `2^-136 ≤ wFloor`.
4. **Second weight** (`wRel1`). `top2_v` (`tl.sumc` of `C·onehot`, the products exact, the leaves
   `C i1` and seven zeros) along a tree of depth 3 over the 8 leaves (`SumDepth`):
   within `((1+3u)^3 − 1)·|C i1| + 28η·(1+3u)^3 ≤ 9.0001u·30.00003 = 1.61e-5`; `top2_v − top1_v`
   (IEEE) adds `60u = 3.6e-6`; `tl.exp` adds `7.39e-6`; so `exp(…)` is within a relative
   `2.71e-5` (second-order terms below `10^-9`); times `invsumexp` (relative `1.50e-5`) and the
   product's rounding (`u = 6.0e-8`): `4.22e-5`, `wRel1 = 2^-14 = 6.10e-5` (margin 1.4).

## Exclusions

- **Logits above `1320 − capTol`**: a limit of the kernel's soft cap. The source computes
  `30·tanh(l/30)` as `(e^{2l/30} − 1)/(e^{2l/30} + 1)·30`; above about `1330.8` `exp` overflows
  and `inf/inf` is a NaN. The band where the exact logit is below that but `capTol` reaches past
  `1320` is left out too. Real router logits are of order 1 to 10.
- **`S_e > 2^120`**, non-finite inputs: the `float32` sum (or an input) is not finite.
- **`(bs − 1)·hidden_dim ≥ 2^31`**: a limit of the kernel, the `int32` product `pid * hidden_dim`
  wraps (first at `hidden_dim = 8192`, `bs = 262,145`).
- **`hidden_dim` not a multiple of 128 or above `2^17`, `bs > 2^22`**: outside real model shapes.
- **`hidden_dim > 8192`** and the other compile-time values (a correction bias, cap `0`, `topk`
  other than 2, other expert counts), including the five CPU-grid cases of the source's test
  (lines 216–217): not part of `SETUP.md`.

`check_gpu.py` (one H200) checks `Post` on the eight Grok launches and on edge launches, with the
witness `C` taken from a probe kernel running the source's lines 66–93.
