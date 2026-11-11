# `prefill_attn_varlen_bf16`

**The kernel.** `_fwd_kernel` of SGLang's prefill attention (`source/prefill_attention.py`,
kernel lines 51–184; host `context_attention_fwd`, lines 187–243). Attention over a packed batch
of variable-length sequences, causal or not: program `(b, h, m)` loads the `[128, BLOCK_DMODEL]`
block `m` of sequence `b`'s queries at head `h`, loops over the key blocks it attends (`m + 1`
blocks when causal, all `cdiv(len_b, 128)` when not), computing the scores with `tl.dot`, the
online softmax (`tl.exp`, row maxima and sums, `/`), rounding the weights to `bfloat16` and
accumulating `tl.dot(p, v)`, and stores the block of `o` in `bfloat16`. The claim is
`Claims.PrefillAttnVarlen.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
sequence `b < batch` (tokens `start_b … start_b + len_b − 1`, from `b_start_loc`, `b_seq_len`),
query `p < len_b`, head `h < num_heads` and component `e < Lk`:
- `o[start_b + p, h, e]` is a finite `bfloat16` `c` with
  `|c − attnRef| ≤ tolRel(BLOCK_DMODEL, T, Xrow)·attnAbs + 2^-80·(1 + vSum)`, where, over the keys
  `j ≤ L` the query attends (`L = p` when causal, `L = len_b − 1` when not) and with `σ = sm_scale`
  and key/value head `g = h // (num_heads // num_kv_heads)`:
  - `score_j = σ·Σ_e q[start_b+p, h, e]·k[start_b+j, g, e]`;
  - `attnRef = Σ_j e^{score_j}·v_j / Σ_j e^{score_j}` (exact, in the reals), `attnAbs` the same mean
    of `|v_j|`, `vSum = Σ_j |v_j|`;
  - `Xrow = max_j σ·Σ_e liftMag q·liftMag k` (`Device.liftMag .bf16`), `T = L // 128 + 1`;
  - `tolRel(D, T, X) = 2^-6 + T·2^-14 + (D + 16)·X·2^-20`;
- every byte outside those elements of `o` is unchanged.

`2^-6` covers the two `bfloat16` roundings (`p.to(bfloat16)` and the store) with the other per-row
roundings; `2^-14` per trip the second `tl.dot`'s error and the online softmax's per-trip roundings;
`(D + 16)·2^-20` per unit of `X` the score `tl.dot`'s error and `tl.exp`'s error, which grows with
its argument. At the host's sizes `tolRel ≈ 1.7·10^-2`.

## Conditions

`Inst.ok` and `Pre`:
- the grid `(batch, num_heads, cdiv(max_input_len, 128))` within CUDA's limits, fewer than `2^31`
  programs;
- `num_kv_heads ≥ 1` divides `num_heads`; `1 ≤ Lk ≤ BLOCK_DMODEL ≤ 256`, `BLOCK_DMODEL` a power of
  two;
- `max_input_len ≤ 2^17`, `total_tokens ≤ 2^22`, `batch ≤ 8192` (see Exclusions);
- `sm_scale` a finite double in `[2^-64, 2^64]`;
- the eight strides are `int32` arguments, and every element index of `q`, `k`, `v`, `o` is below
  `2^31` (see Exclusions);
- each storage holds its tensor; `o` is row-major in one order or the other (`empty_like(q)`) and
  shares no byte with the five inputs; the six buffers are valid;
- `SeqOk`: each `b_start_loc[b]`, `b_seq_len[b]` with `0 ≤ start`, `0 ≤ len ≤ max_input_len`,
  `start + len ≤ total_tokens`, no two non-empty sequences sharing a token;
- `ValsOk`: in every sequence, every `q`, `k`, `v` component finite, `|v| ≤ 2^120`, and every
  query–key pair of the sequence (causal-masked pairs included) with `σ·Σ liftMag q·liftMag k ≤ 2^12`.

**The launches covered** (`SETUP.md`, `setup_*_ok`): the host's eight test cases (causal), the
model shapes at their largest (64/8 heads of 128, two prompts of 8192), strided `q`, `k`, `v` views
of a fused `qkv` tensor (`setup_fused`), and the vision encoders' non-causal launches on fused-`qkv`
views (Qwen2.5-VL full and window attention, its largest image and a video, Qwen3-VL, MiMo-VL, and
`sm_scale = 1.0`). `canon_ok` and `fusedQKV_ok` cover the host's contiguous and fused-`qkv`
launches at every size within their bounds, causal or not.

**Device facts** the tolerance rests on (all in `Device P`): `Device.DotBF16` (`tl.dot` of
`bfloat16` into `float32`: relative `relDot16 n = (n + 2)·2^-22`, absolute `(n + 2)·2^-126`, on the
lifted magnitude), `Device.ExpF32` (`tl.exp` on `[−X, X]`: `(2^-22 + 2^-23·X)·e^x + 2^-140`),
`Device.ExpF32NegInf` (`tl.exp(−∞) = +0`: masked lanes, the first trip's `alpha`),
`Device.DivFull` (`/` on `float32`: `2^-22` relative, `2^-148` absolute, quotients up to `2^126`),
and the totality facts for `tl.dot`, `exp` and `/`.

## Tolerances

With `D = BLOCK_DMODEL`, `u = 2^-24`, the fine bound is `fineRel·attnAbs + 2^-80·(1 + vSum)`:
- `δ = (D + 8)·2^-22·X + 2^-40` bounds each computed score's error (the score `tl.dot`, the
  rounding of `hostF32(σ)`, the multiply-add with the mask);
- the scores' error moves the softmax weights by at most `e^{±2δ}`;
- the online softmax's factors telescope, so each key's coefficient is exactly its softmax weight
  up to the roundings: `tl.exp`'s errors (the `α`'s arguments sum to at most `2(X + δ)`), the row
  sum's `127u`, the products, `tl.addc`, `DivFull`, and `p.to(bfloat16)`'s `2^-8`; together
  `ρ = exp(2^-8 + (T + 8)·2^-18 + (X + δ)·2^-17) − 1`;
- the second `tl.dot`, carried over `T` trips: `γ = exp(T·136·2^-22) − 1`;
- `E = e^{2δ}·(ρ + γ·(1 + ρ))/(1 − ρ) + (e^{2δ} − 1)`, and with the store's rounding
  `fineRel = 2^-8 + (1 + 2^-8)·E`; the absolute terms stay below `2^-80·(1 + vSum)` for
  `T ≤ 1024`.

On every covered row (`T ≤ 1024`, `X ≤ 2^12`, `D ≤ 256`): `ρ ≤ 0.04`, and
`fineRel ≤ 2^-8 + 1.0668·s + 1.1058·G + 1.4169·2δ` (`s` the exponent of `ρ`, `G = T·136·2^-22`),
whose constant is at most `0.00811 ≤ 2^-6`, per-trip coefficient at most `4.0·10^-5 ≤ 2^-14`, and
per-unit-of-`X` coefficient at most `8.14·10^-6 + 6.76·10^-7·(D + 8) ≤ (D + 16)·2^-20`: so
`fineRel ≤ tolRel`. On the H200 the worst error is `0.434` of the bound (an adversarial uniform
causal case; `0.40` on the host's tests).

## Exclusions

- **`int32` element offsets**: a limit of the kernel. Its element offsets are `int32` and wrap
  beyond `2^31`; `Inst.ok` keeps every element index below `2^31` (`q_i32` … `o_i32`). The first
  host launch that hits it has `total_tokens·num_heads·head_dim ≥ 2^31` (e.g. `524,289` tokens at
  32 heads of 128); the wrapper does not check.
- **`max_input_len > 2^17`**: not a kernel limit; the bound grows with the number of trips.
- **Contexts longer than `2^22` tokens, batches of more than `8192`**: the claim covers real model
  shapes only (`total_le`, `batch_le`). `Lk` is a head dimension and `H`, `HKV` head counts, so no
  field is a model dimension.
- **Head dims of 8 or less** are rejected on the host (`tl.dot` needs `K ≥ 16`): no outcome; head
  dims above 256 are outside real use.
- **Input values** outside `ValsOk` (non-finite inputs, `|v| > 2^120`, scores' magnitudes above
  `2^12`), **`sm_scale`** outside `[2^-64, 2^64]`, **overlapping sequences**, `o` overlapping the
  inputs, **a grid** beyond CUDA's limits, **`int64` `b_start_loc`/`b_seq_len`**: outside real use
  or the setup.

`check_gpu.py` (one H200) checks `Post` on the setup's launches, causal and not, on the edges of
`Inst.ok` and `Pre`, and runs negative tests (the other mode, a missing key block, the wrong
key/value head), which `Post` rejects.
