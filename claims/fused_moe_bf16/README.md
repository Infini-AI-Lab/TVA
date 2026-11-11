# `fused_moe_bf16`

**The kernel.** vLLM's `fused_moe_kernel` (`source/fused_moe.py`, lines 109–420, with the helper
`write_zeros_to_output`, lines 89–106) on the unquantised `bfloat16` path. Its hosts are vLLM's two
expert GEMMs, `fused_moe_gemm1` (lines 704–766) and `fused_moe_gemm2` (lines 769–833), through
`invoke_fused_moe_triton_kernel` (lines 609–690), with the expert assignment of
`_prepare_expert_assignment` (lines 573–606: `moe_align_block_size`, or the naive assignment for
decode-sized batches) and the tile `try_get_optimal_moe_config` picks on an H200 (vLLM's tuned
files in `source/configs/`, else `get_default_config`). A 1-D grid of
`cdiv(EM, BLOCK_SIZE_M)·cdiv(N, BLOCK_SIZE_N)` programs; each maps to a block of `BLOCK_SIZE_M`
positions of the assignment and a tile of columns (grouped ordering), returns if the block is past
the padded total, stores zeros if the block's expert is `-1`, and otherwise accumulates
`tl.dot`s of `bfloat16` tiles of `A` and the expert's `B` into `float32` over `cdiv(K, BLOCK_SIZE_K)`
trips, multiplies by the routing weight with `MUL_ROUTED_WEIGHT`, rounds to `bfloat16` and stores
the rows of `C`. The claim is `Claims.FusedMoe.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race). Write `ntp`
for the padded total in `num_tokens_post_padded`, and `entryOf pos` for the flat index at position
`pos` (entry `pos` of `sorted_token_ids`; with the naive assignment `pos / BLOCK_SIZE_M` in a block's
first lane and `M·top_k`, masked, in the others). For every flat index `f < M·top_k` and column
`c < N`, the element `f·N + c` of `C`:
- **not routed** (no position below `ntp` holds `f`): unchanged;
- **routed, the block's expert `-1`**: the `bfloat16` `+0.0` (up to NaN payload);
- **routed, the block's expert `e ∈ [0, E)`**: when row `f / top_k` of `A` and column `(e, c)` of
  `B` hold finite values and `dotMag ≤ 2^120`, the element is the `bfloat16` rounding of a finite
  `float32` value `u` (of the `float32` product of `u` with the finite routing weight at `f`, with
  `MUL_ROUTED_WEIGHT`), with `|u − S| ≤ T = 2^-12·pathMag + 2^-133`, `S` the exact dot product;
  with room (`|S| + T` within the `bfloat16` range, or `(|S| + T)·|w| ≤ 2^126` with the weight) the
  output is finite. Otherwise it only holds some `bfloat16`;
- every byte outside the matrix `C` is unchanged.

`pathMag` sums, over the chunks of 64 positions the loop's `tl.dot`s cover, the largest of the
exact partial sum before the chunk, the exact partial sum before the chunk's trip, and the chunk's
lifted magnitude (`Device.liftMag .bf16`: `|q|`, a nonzero subnormal counted as `2^-126`);
`pathMag ≥ dotMag ≥ |S|`.

**The conditions** (`Inst.ok`, `Pre`):
- the tile `(BM, BN, BK, GM)` is one of the 56 tiles of vLLM's tuned `bfloat16` H200 files, which
  include `get_default_config`'s (`cfgs`);
- `top_k`, `E`, `N`, `K` positive; `K ≤ 2^15`, `N ≤ 2^17`; `K + BK`, `N + BN`, `EM + BM ≤ 2^31`,
  `cdiv(N, BN)·GM < 2^31`, fewer than `2^31` programs, `M·top_k < 2^31`, `B`'s strides and `N`
  below `2^31`, and the `int64` offsets of `B` below `2^63` (the kernel's integer arithmetic does
  not wrap);
- the buffers hold the matrices; `topk_weights` is contiguous (`stride(0) = size(1)`); the matrix
  `C` shares no byte with the other six buffers; the seven buffers are valid;
- the expert assignment is as the host builds it: `ntp` a multiple of `BLOCK_SIZE_M`, at most `EM`;
  the entries below `ntp` nonnegative and the valid ones pairwise distinct; `expert_ids` covering
  the blocks below `ntp`, each `-1` or in `[0, E)`; with `MUL_ROUTED_WEIGHT`, the weights of the
  valid flat indices finite.

Any `bfloat16` values in `A` and `B` are covered, NaN and infinities included.

**The launches covered** (`SETUP.md`): 64 `setup_*_ok`, both GEMMs: the host's nine test cases
(source lines 865–873) and model layers (Mixtral-8x7B at TP1/2/8, Mixtral-8x22B, Grok-1,
DeepSeek-V3, Qwen3-30B-A3B, Llama-4-Scout at TP8; decode with the naive assignment, batches,
prefill up to `M = 40000`, the weight on the input), and an edge (one expert, `top_k = 1`,
`GROUP_SIZE_M = 16`). `hostInst_ok` covers the host's instance for any sizes within the bounds.

**Device facts** the tolerance uses (all in `Device P`): `Device.DotBF16Chunk` (a `bfloat16`
`tl.dot` into `float32`, chunk by chunk of 64 products: chunk `j` within
`min (2^-13·Mj) (2·A_j + 2^-22·Mj) + 2^-20·|s_j + c_j| + 2^-143` of `s_j + c_j`, with room), and the
totality fact for `tl.dot`. The weight's product, the conversions and the integer arithmetic are
exact or IEEE.

## Tolerances

Position `k` holds `p_k = va k · vb k` (`0` past `K`: the last trip's lanes load `+0`); chunk `g` is
the 64 positions from `64·g`, `c_g` its exact sum, `m_g` its lifted magnitude, `P_g` the exact
partial sum before it; trip `t` holds chunks `t·q … t·q + q − 1`, `q = BK/64 ≤ 4`; there are at
most `G = 512` chunks (`K ≤ 2^15`).
- **One chunk step** (`DotBF16Chunk`, with `ρ = 2^-13`, `σ = 2^-20`, `α = 2^-143`): from a value
  within `e` of `P_g`, the next is within `(1 + ρ + σ)·e + ρ·max(|P_g|, m_g) + σ·|P_{g+1}| + α` of
  `P_{g+1}`.
- **One trip** (`tl.dot_add`, line 373): either `tl.dot(a, b, acc)` (the accumulator's chain
  continues) or `tl.dot(a, b)` followed by a `float32` addition (`3u·(|d| + |acc|) + 4η`); the trip's
  error is the larger.
- **Relaxation.** With `A_g = max(|P_g|, |P_{tq}|, m_g)` (`pathMag = Σ_g A_g`), both forms give
  `tripErr(e) ≤ γ·e + γ·(1.7763·ρ·M_t + q·α) + 4η`, `γ = (1 + 3u)·(1 + ρ + σ)^q`; over the trips
  `γ^T ≤ 1.0651`, so the error is at most `1.892·2^-13·pathMag + 2^-133 ≤ 2^-12·pathMag + 2^-133`.

The weight's product and the `bfloat16` rounding are stated exactly in `Post` (`FloatV.mul`,
`FloatV.cvt`, up to NaN payload). The bound grows with `K` (median `T/|S|` `0.0026` at `K = 128`,
`0.094` at `K = 2^15` on `check_gpu.py`'s data), about twice the fine bound everywhere; it rejects
the next expert's weights, a wrong scale, a dropped `K` block and a `bfloat16` accumulator.

## Exclusions

- **Real shapes only**: `N > 2^17` (`N_le17`) and `K > 2^15` (`K_le15`; the largest real `K` is
  `32768`, and the tolerance grows with `K`). `N` and `K` need not be multiples of `128`.
- **`int32` limits**: `EM + BM`, `N + BN` or `K + BK` above `2^31` (`tl.cdiv` wraps), `M·top_k ≥
  2^31`, `stride_be = N·K ≥ 2^31`: not reached by real use.
- **A non-contiguous `topk_weights`** with `MUL_ROUTED_WEIGHT` (a kernel limit): the kernel's flat
  read (line 404) takes other pairs' weights; `w_contig` excludes it. vLLM's routers return a
  contiguous `[M, top_k]` tensor.
- **Non-finite routing weights** with `MUL_ROUTED_WEIGHT`: outside real use (softmax probabilities).
- **Not among the launches**: the batch-invariant tile, other devices' or user tuned files, an
  override configuration, `int64` `topk_ids` with the naive assignment, quantised paths, bias,
  `SWAP_AB`, `USE_TD`.

`check_gpu.py` (one H200) checks `Post` on 41 host launches and a direct launch with a guarded `C`,
the `-1` expert path, and negative tests.
