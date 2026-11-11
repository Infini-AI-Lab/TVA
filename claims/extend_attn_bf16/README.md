# `extend_attn_bf16`

**The kernel.** `_fwd_kernel` of SGLang's Triton extend attention (`source/extend_attention.py`,
kernel lines 89–90, 93–96 and 99–638; host `extend_attention_fwd` lines 687–857). A grid
`(batch, num_heads, cdiv(max_len_extend, BLOCK_M))`; program `(b, h, m)` takes `BLOCK_M` new tokens
of request `b` and head `h`, runs an online softmax first over the request's cached prefix (keys and
values gathered from the paged cache `k_buffer`, `v_buffer` through `kv_indices`), then over its new
tokens (`k_extend`, `v_extend`) with the causal mask or a target verification's tree mask
(`custom_mask`), each optionally limited to a sliding window, and stores
`o_extend = acc / deno` as `bfloat16`. The claim is `Claims.ExtendAttn.claim` in `Claim.lean`.

The covered paths are `SETUP.md`'s: an extend batch (`int64` `qo_indptr`, causal) and a target
verification (`int32` `qo_indptr`, the tree mask), each with or without a sliding window; the other
flags as SGLang's backend passes them (no logit cap, sinks, xai temperature or LSE, page size 1,
`k_scale = v_scale = 1.0`, `SKIP_PREFIX_CUSTOM_MASK = True`); `bfloat16` data, head dims up to 256,
H200 block sizes (`BLOCK_M = 128` for `D ≤ 128`, else `64`; `BLOCK_N = BLOCK_N_PREFIX = 64`).

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- every `o_extend[qo_indptr[b] + i, h, e]` (`b < batch`, `i < ext_b`, `h < num_heads`, `e < D`)
  is a finite `bfloat16` `c` with
  `|c − attnRef| ≤ tolRel(BD, tripsPre, tripsExt, Xrow)·attnAbs + tolAbs(Σ|v|)`, where `attnRef` is
  the exact softmax-weighted mean of the values over the positions the row attends (`keys`: its
  prefix, then its new tokens up to itself or as the tree mask allows, with a window `W` only those
  with `pre_b + i ≤ j + W`), computed in the reals from the initial inputs, `attnAbs` the same mean
  of `|v|`, `Xrow` the row's largest `sm_scale·Σ liftMag q·liftMag k`, and `tolAbs(s) =
  2^-80·(1 + s)`;
- every byte outside those output elements is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- the grid within CUDA's limits (`num_heads < 2^16`, `cdiv(max_len_extend, BLOCK_M) < 2^16`, fewer
  than `2^31` programs); `num_kv_heads ≥ 1` divides `num_heads`;
- `1 ≤ D ≤ BLOCK_DMODEL ≤ 256`, `BLOCK_DMODEL` a power of two, `D ≠ 192`; the H200's `BLOCK_M`;
- `max_len_extend ≤ 2^18` (`≤ 2^12` with the tree mask), at most `2^22` new tokens and `8192`
  requests, a window of at most `2^18`, `sm_scale` a finite double in `[2^-64, 2^64]`;
- the twelve strides below `2^31`; the kernel's `int32` head offsets do not wrap (`heads_i32`), nor,
  with an `int32` `qo_indptr`, the element offsets of `q_extend`, `k_extend`, `v_extend`,
  `o_extend` (`cm_i32`);
- each buffer holds its tensor; `o_extend` row-major in one order or the other and disjoint from
  every input;
- the requests well formed (`ReqOk`: `0 ≤ qo_indptr[b] ≤ qo_indptr[b+1] ≤ T`, `ext_b ≤
  max_len_extend`, `0 ≤ kv_indptr[b] ≤ kv_indptr[b+1] ≤ NI`, `pre_b + ext_b ≤ 2^18`, every prefix
  slot in the cache);
- with the tree mask (`MaskOk`): `mask_indptr[b] ≥ 0`, with a window `0 ≤ window_kv_offsets[b] ≤
  2^40`, each request's block within `custom_mask`, and every new token attends itself and no later
  new token;
- the values (`ValsOk`): every query, key, value and prefix-slot component a finite `bfloat16`,
  `|v| ≤ 2^100`, and every attended pair's `sm_scale·Σ liftMag q·liftMag k ≤ 2^12`.

**The launches covered** (`SETUP.md`): `canon_ok` covers the host's launch on contiguous tensors for
an extend batch or a target verification, with or without a window, up to `8192` requests,
`T ≤ 2^20`, `S ≤ 2^22`, `1 ≤ H ≤ 128`, `HKV ∣ H`, head dims up to 256 other than 192, and any
`sm_scale` in range. `setup_*_ok`: the code base's test at its largest draw for head dims 256, 128,
96, 80, 13 and its sliding-window test; Llama-3-8B, Qwen2.5-7B, Qwen2.5-0.5B, Gemma-3-12B (global
and local layers), one KV head per rank, a 128K and a 256K context; target verifications with 64
draft tokens, a chain of 4, and Gemma-3's local layers. `claim_satisfiable` uses one request with no
prefix and one new token, head dim 16, zero inputs.

**Device facts** the tolerances rest on (all in `Device P`): `Device.DotBF16` (`tl.dot` of
`bfloat16` operands into `float32`, relative `relDot16 K = (K+2)·2^-22` of the lifted magnitudes),
`Device.ExpF32`, `Device.ExpF32NegInf`, `Device.DivFull`, and the totality facts for `tl.dot`,
`exp` and `/`.

## Tolerances

Fix a row (`b`, `h`, `i`), `X = Xrow`, `B = BLOCK_DMODEL`, `Tp = tripsPre`, `Te = tripsExt`,
`T = Tp + Te` (a window or a skipped tile only lowers the trips the row runs; every term grows with
them):
- `dlt = (B + 8)·2^-22·X + 2^-40` bounds each computed score's error (the score `tl.dot`, the
  `float32` rounding of `sm_scale`, the products);
- `rho = exp(2^-8 + (T + 8)·2^-20 + (X + dlt)·2^-17) − 1` bounds the relative error of each
  position's coefficient and of `deno` (one `bfloat16` rounding of `p`, the per-trip `float32`
  roundings and `tl.exp`/`/` errors, `tl.exp`'s error growing with its argument);
- `gam = exp((66 + 2·Tp + 72·Te)·2^-22) − 1` bounds the `p·v` `tl.dot` errors and the
  accumulator's roundings (a prefix trip's dot has a zero accumulator; a new-token trip's
  `tl.dot_add` is charged `66·2^-22` of its accumulator by `DotBF16`);
- `errE = e^{2·dlt}·(rho + gam·(1 + rho))/(1 − rho) + (e^{2·dlt} − 1)`, and
  `tolRel = 2^-8 + (1 + 2^-8)·errE` (the store's `bfloat16` rounding);
- the absolute terms (subnormal lifts, `tl.exp`'s `2^-140`, roundings' absolute parts) total at most
  `2^-80·(1 + Σ|v|)` for `T ≤ 2^12 + 2`.

`tolRel ≈ 0.85·10^-2` at short lengths, about `0.014` at a 128K context with an 8K chunk, `0.047`
for 128K new tokens with no prefix.

## Exclusions

- **`int32` head offsets** (`cur_head * stride_qh`, …, lines 241, 317, 412, 513, 577, 623) and,
  with an `int32` `qo_indptr`, the `int32` element offsets of `q_extend`, `k_extend`, `v_extend`,
  `o_extend`: a limit of the kernel, kept out by `heads_i32` and `cm_i32`.
- **More than `2^18` positions in a request**, `max_len_extend > 2^18`: a limit of the derivation
  (the bound grows with the trips). More than `2^12` draft tokens per request with the tree mask.
- **Contexts longer than `2^22` tokens, batches of more than `8192` requests**: real model shapes
  only.
- **Head dim `192`** (the host runs the RoPE split `BLOCK_DMODEL = 128`, `BLOCK_DPE = 64`, lines
  653–655, another configuration), head dims above 256, MLA head dims, `Lq ≠ Lv`. Head dims of 8 or
  less are rejected by Triton (`tl.dot` needs `K ≥ 16`): no outcome.
- **Input values**: non-finite inputs, `|v| > 2^100` (the `float32` accumulator could overflow),
  attended scores above `2^12` (where the bound is close to vacuous), `sm_scale` outside
  `[2^-64, 2^64]`.
- **Malformed requests or masks**: decreasing `qo_indptr`, a prefix slot outside the cache,
  `ext_b > max_len_extend`, a tree mask with an entry above the diagonal or without its diagonal,
  `o_extend` overlapping the inputs.
- **The other paths and dtypes**: logit cap, sinks, xai temperature, LSE, pages larger than 1,
  `identity_kv_indices`, `score_mod`, non-causal layers, `SKIP_PREFIX_CUSTOM_MASK = False`,
  `k_scale`/`v_scale ≠ 1`, an fp8 KV cache, `int32` `kv_indices`: not `SETUP.md`'s launches.

`check_gpu.py` (one H200) checks `Post` on the code base's tests, `SETUP.md`'s shapes, sliding
windows, target verifications and edge cases, with negative tests.
