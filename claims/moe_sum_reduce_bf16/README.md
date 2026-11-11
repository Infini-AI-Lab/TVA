# `moe_sum_reduce_bf16`

**The kernel.** SGLang's `_moe_sum_reduce_kernel` (`source/moe_sum_reduce.py`, kernel lines 50–104;
host `moe_sum_reduce` lines 107–146). A 2-D grid `(cdiv(token_num, BLOCK_M), cdiv(hidden_dim,
BLOCK_DIM))`; each program sums, in `float32`, the `topk_num` expert rows of its tile of the input
`[token_num, topk_num, hidden_dim]`, multiplies by the constexpr `routed_scaling_factor` (a `float32`
constant) and stores the `bfloat16` rounding into the output `[token_num, hidden_dim]`. The claim is
`Claims.MoeSumReduce.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
output element `(t, j)`:
- it holds a `bfloat16` value `c`;
- if the element is `InRange` (its `topk_num` inputs `x_k` finite, the scale finite as a double `σ`
  and as its `float32` constant `σ₃₂`, `A = Σ|x_k| ≤ 2^126`, `|σ₃₂|·A ≤ 2^126`), `c` is finite and
  `|c − σ·S| ≤ 2^-8·|σ·S| + K·2^-22·(|σ| + 2^-126)·(A + 2^-126) + 2^-133` (`tol`), with
  `S = Σ_k x_k` exact and `K = topk_num`;
- every byte outside the output's `token_num·hidden_dim` elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`): `BLOCK_M`, `BLOCK_DIM` powers of two with
`BLOCK_M·BLOCK_DIM ≤ 2^20`; `1 ≤ token_num, topk_num, hidden_dim < 2^31`; `topk_num ≤ 2^20`;
`hidden_dim ≤ 2^17`; `token_num ≤ 2^22`; `topk_num·hidden_dim < 2^31`; `cdiv(hidden_dim, BLOCK_DIM)
< 2^16` and fewer than `2^31` programs; the buffers hold the tensors, are valid, and the output
shares no byte with the input. The model passes the host's contiguous strides. Every `bfloat16` input
and every scale is covered.

**The launches covered** (`SETUP.md`): the host's nine test launches and `(65536, 9, 7168, 2.5)`
for the Marlin/LoRA runners, with `BLOCK_M = 1`, `BLOCK_DIM = 2048` (`setup_*_ok`). `mk_ok` covers
the host's contiguous launch at every shape and block size `Inst.ok` allows.

**Device facts.** None: the float operations are IEEE `+`, `*` and `bf16 ↔ f32` conversions.

## Tolerances

With `u = 2^-24`, `η = 2^-150` (`float32`), `u_bf = 2^-8`, `η_bf = 2^-134` (`bfloat16`) and
`s = |σ₃₂|`: the accumulation from `+0` is a chain of `K` IEEE additions, within
`E1 = ((1+u)^K − 1)·A + η·K·(1+u)^K` of `S`; the product with `σ₃₂` adds `u·s·(|S| + E1) + η`; the
rounding to `bfloat16` adds `u_bf·Q + η_bf` (`Q` the product's magnitude bound); the scale's own
conversion adds `|σ − σ₃₂|·|S| ≤ (u·|σ| + η)·|S|`. With `|S| ≤ A` and `K ≤ 2^20`
(`(1+u)^K ≤ 1.0645`), the sum is at most `tol`: the dominant term is one `bfloat16` rounding of the
result, `2^-8·|σ·S|`; the terms in `|σ|·A` have coefficient at most `4·K·u = K·2^-22`; the floors
`2^-126` absorb the absolute parts in `|σ|` and in `A`; the remaining absolute terms are below
`2^-133`.

## Exclusions

- `hidden_dim > 2^17` and `token_num > 2^22`: the claim covers real model shapes only (any
  `hidden_dim` up to `2^17` is covered, a multiple of `128` or not, e.g. gpt-oss's `2880`).
- `cdiv(hidden_dim, BLOCK_DIM) ≥ 2^16`, or `2^31` programs or more: launch limits.
- Elements not `InRange`: a non-finite input or scale, or magnitudes near `float32`/`bfloat16`
  overflow; only that the output holds a value is claimed.
- `topk_num > 2^20`: outside real use (`≤ 9`); keeps the accumulation bound within `float32`.
- `topk_num·hidden_dim ≥ 2^31`: `input.stride(0)` would be an `int64` argument.
- `token_num`, `topk_num` or `hidden_dim` `= 0` or `≥ 2^31`: outside real use.
- `USE_GDC = True` / `launch_pdl`: not covered.

`check_gpu.py` (one H200) checks `Post` on the host's launches and on edge cases.
