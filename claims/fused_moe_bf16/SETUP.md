# Setup of `fused_moe_bf16`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/model_executor/layers/fused_moe/fused_moe.py:298`,
the kernel `fused_moe_kernel` (with its helper `write_zeros_to_output`), launched by the two expert
GEMMs of vLLM's `fused_experts_impl` on the unquantised `bfloat16` path.

## The source

`source/fused_moe.py` holds:
- `write_zeros_to_output` (`fused_moe.py:44-61`) and `fused_moe_kernel` (`fused_moe.py:297-608`),
  the `@triton.jit` helper and kernel, verbatim but for the kernel's one parameter default
  (below). A program computes one `BLOCK_SIZE_M × BLOCK_SIZE_N` tile of the output: it reads
  `num_tokens_post_padded` and returns if its block of sorted rows is past it; it takes the block's
  `BLOCK_SIZE_M` flat indices (`t · top_k + j`, or the padding value `M · top_k`) from
  `sorted_token_ids`, or with `naive_block_assignment` the block's own index in the first lane and
  the padding in the others; masks those not below `num_valid_tokens`; loads the block's expert from
  `expert_ids` (writing zeros and returning on `-1`); runs the K loop over the rows
  `flat // top_k` of `A` and the expert's slice of `B`; multiplies by the routing weights if
  `MUL_ROUTED_WEIGHT`, casts to `compute_type` and stores to the rows `flat` of `C`. The kernel's
  other constexpr paths (fp8/int8 quantisation, bias, `SWAP_AB`, the tensor-descriptor path
  `USE_TD`) stay in the source;
- `moe_align_block_size`, a torch-only equivalent of vLLM's `moe_align_block_size`
  (`fused_moe/moe_align_block_size.py:55-169`), whose work vLLM does in a CUDA op
  (`csrc/moe/moe_align_sum_kernels.cu`): it builds `sorted_token_ids`, `expert_ids` and
  `num_tokens_post_padded` with the same sizes and contents (within one expert, the flat indices in
  ascending order, where vLLM's op leaves the order to its threads);
- `get_default_config` (`fused_moe.py:1298-1418`), its `bfloat16` branch on CUDA;
- `_prepare_expert_assignment` (`fused_moe.py:1549-1599`), for weights in device memory: the naive
  assignment or `moe_align_block_size`;
- `invoke_fused_moe_triton_kernel` (from `fused_moe.py:761-910`), the launcher, for the unquantised
  path with no bias;
- the hosts `fused_moe_gemm1`, the part of vLLM's `fused_experts_impl` (`fused_moe.py:1664-1868`) up
  to and including its first GEMM, `hidden_states @ w1[e]ᵀ` into `intermediate_cache1`, and
  `fused_moe_gemm2`, its second GEMM, `intermediate_cache2 @ w2[e]ᵀ` into `intermediate_cache3`
  (`fused_moe.py:1836-1861`, the kernel's `top_k = 1`, `MUL_ROUTED_WEIGHT = not
  apply_router_weight_on_input`);
- `get_device_name_as_file_name`, `get_config_file_name`, `get_moe_configs` and
  `try_get_optimal_moe_config` (`utils/platform_utils.py:69-74`, `fused_moe.py:1095-1165, 1421-1450`):
  the configuration lookup, reading `source/configs/`, which holds vLLM's 22 tuned `bfloat16` files
  for `NVIDIA_H200` (`model_executor/layers/fused_moe/configs/E=<E>,N=<N>,device_name=NVIDIA_H200.json`,
  verbatim);
- `reference` (the routed products in `float64`) and the test under `if __name__ == "__main__":`.

Changes from vLLM's source (each with its `file:line` in the file's docstring): the alignment
helper in torch; `get_default_config` reduced to its `bfloat16` branch on CUDA; the configuration
lookup without an override configuration or `VLLM_TUNED_CONFIG_FOLDER`, the device name from
`torch.cuda.get_device_name`; `_prepare_expert_assignment` for weights in device memory and unquantised weights;
`intermediate_cache1` and `intermediate_cache3` allocated fresh (vLLM takes views of one buffer);
`fused_moe_gemm2` takes `intermediate_cache2` as its input (vLLM computes it with the activation);
the quantised and wna16 branches of the wrapper dropped; the grid computed directly. The kernel's
signature loses the default of `USE_TD` (TVA's subset has none; the host passes it); its other
lines are unchanged.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. vLLM's `try_get_optimal_moe_config`
(`fused_moe.py:1421-1450`), which the hosts copy, picks the configuration once per call, from the
number of tokens `M` and `w2`'s shape `[E, K, N_w2]` (`N_w2` the intermediate size per rank); both
GEMMs of one call use it:
- the tuned file `E=<E>,N=<N_w2>,device_name=NVIDIA_H200.json` when it exists (every H200 maps to
  that name, `fused_moe.py:1099-1101`; the `bfloat16` files have no `dtype` in their name): its
  entry for the batch size nearest `M`. vLLM has 22 such files (`E` 8 to 512), with 56 distinct
  tiles: `BLOCK_SIZE_M` 16, 32, 64, 128; `BLOCK_SIZE_N` 32, 64, 128, 256; `BLOCK_SIZE_K` 64, 128,
  256; `GROUP_SIZE_M` 1, 16, 32, 64; `num_warps` 4 or 8, `num_stages` 2 to 5. Mixtral-8x7B
  (`E=8,N=14336` at TP1, `N=7168` at TP2, `N=1792` at TP8), Qwen3-30B-A3B (`E=128,N=768`) and
  Llama-4-Scout at TP8 (`E=16,N=1024`) have one;
- otherwise `get_default_config(M, E)`'s `bfloat16` branch (`fused_moe.py:1372-1416`), e.g. for
  DeepSeek-V3 (`E = 256`), Mixtral-8x22B and Grok-1:

| `M` | `BLOCK_SIZE_M` | `BLOCK_SIZE_N` | `BLOCK_SIZE_K` | `num_warps` | `num_stages` |
|---|---|---|---|---|---|
| `≤ 32` | 16 | 64 | 128 | 4 | 4 |
| `33–64` | 32 | 64 | 128 | 4 | 3 |
| `65–96` | 32 | 128 | 64 | 4 | 3 |
| `97–128` | 64 | 128 | 64 | 4 | 3 |
| `129–512` | 64 | 128 | 64 | 8 | 3 |
| `> 512` | 128 | 128 | 64 | 8 | 3 |

with `GROUP_SIZE_M = 16` when `M // E > 128` and `1` otherwise, `SPLIT_K = 1` (the wrapper sets it;
the kernel does not use it). The grid is `(cdiv(EM, BLOCK_SIZE_M) · cdiv(N, BLOCK_SIZE_N),)`, with
`EM = sorted_token_ids.size(0)`, or, when `A.size(0) < BLOCK_SIZE_M`,
`EM = min(sorted_token_ids.size(0), A.size(0) · top_k · BLOCK_SIZE_M)`, or, with the naive
assignment, `EM = A.size(0) · top_k · BLOCK_SIZE_M` (`fused_moe.py:822-833`).

The flags, as vLLM's GEMMs set them for unquantised `bfloat16` weights (`fused_moe.py:1800-1858` and
`invoke_fused_moe_triton_kernel`):
- the first GEMM: `MUL_ROUTED_WEIGHT = apply_router_weight_on_input` (`False` by default; e.g. Llama
  4 applies the weights on the input), `top_k = topk_ids.size(1)`; the second GEMM:
  `MUL_ROUTED_WEIGHT = not apply_router_weight_on_input`, `top_k = 1`;
- `naive_block_assignment = (sorted_token_ids is None)`: the naive assignment when there is no expert
  map and `M · top_k · 4 ≤ E_global` (`fused_moe.py:1570-1591`; e.g. DeepSeek-V3 up to `M = 8`,
  Qwen3-30B-A3B up to `M = 4`, Mixtral at `M = 1`), `moe_align_block_size` otherwise;
- `compute_type = tl.bfloat16`; `use_fp8_w8a8 = use_int8_w8a8 = use_int8_w8a16 = per_channel_quant
  = False`; `HAS_BIAS = False` (`w1_bias`, `w2_bias` are `None` by default); `SWAP_AB = False`
  (computed only for fp8); `USE_TD = False` (`resolve_moe_use_td`, `fused_moe/utils.py:736-748`: off
  on CUDA unless `VLLM_TRITON_USE_TD=1`);
- `b_bias_ptr`, `a_scale_ptr`, `b_scale_ptr` are `None`, their nine strides `0`, and
  `group_n = group_k = 0`;
- the parameters' defaults removed: TVA's subset has no kernel parameter defaults; the host passes
  each (`fused_moe.py:353`: `USE_TD: tl.constexpr = False` becomes `USE_TD: tl.constexpr`, passed
  `False`).

## The launches

The claim covers the hosts' launches, of both GEMMs:
- the first GEMM: `hidden_states` (`A`) `bfloat16` `[M, K]` contiguous, `w1` (`B`) `bfloat16`
  `[E_local, N, K]` with `stride(-1) = 1` (contiguous in vLLM), `C` `bfloat16` `[M, top_k, N]`
  contiguous; the second GEMM: `intermediate_cache2` (`A`) `bfloat16` `[M · top_k, N']` contiguous,
  `w2` (`B`) `[E_local, K', N']`, `C` `[M, top_k, K']` contiguous (zeroed first when there is an
  expert map); `topk_weights` `float32` `[M, top_k]` contiguous, `topk_ids` `int32` `[M, top_k]`
  (vLLM's routers give `int32`);
- the expert assignment of `_prepare_expert_assignment`: `moe_align_block_size` with the tile's
  `BLOCK_SIZE_M` (each routed flat index once, at its expert's padded offset; padding entries
  `M · top_k`; `expert_ids` the expert of each block below `num_tokens_post_padded`, `-1` after it),
  with or without an expert map, or the naive assignment;
- the routed experts: each `topk_ids` entry in `[0, E)` with `E = global_num_experts` (`E_local` by
  default); with expert parallelism, an `expert_map` `int32` `[E]` sending each global expert to its
  local index in `[0, E_local)` or to `-1` (entries routed to `-1`, or outside `[0, E)`, are skipped,
  and their rows of `C` are not written, as in vLLM);
- every configuration of `get_default_config` above, so every batch: decode (`M = 1` …), batches up
  to 512, prefill;
- shapes as variables, with the ranges vLLM uses: `M ≥ 1` (tests `tests/kernels/moe/test_moe.py:247-253`:
  1, 33, 32768, 40000); `E` from 8 to 256 in tests and models (`test_moe.py:169-170`: 8, 64, 192,
  128, 256; the alignment op needs `E` rounded up to 32 below 1024); `top_k` 1, 2, 6, 8
  (`test_moe.py:172-173, 1909`); the hidden size 128, 511, 1024 in tests and 2048 to 7168 in models;
  the intermediate size per TP rank up to 32768; e.g. Mixtral-8x7B (E 8, top 2, hidden 4096,
  intermediate 14336: the first GEMM's `N = 28672 / TP`, the second's `K = 14336 / TP`),
  Mixtral-8x22B (E 8, top 2, hidden 6144, intermediate 16384), Grok-1 (E 8, top 2, hidden 6144,
  intermediate 32768), DeepSeek-V3 (E 256, top 8, hidden 7168, intermediate 2048), Qwen3-30B-A3B
  (E 128, top 8, hidden 2048, intermediate 768); `K` and `N` multiples of the blocks or not.

vLLM also launches this kernel with the batch-invariant tile `(64, 64, 32, 8)`
(`VLLM_BATCH_INVARIANT`, `fused_moe.py:1307-1314`), with a user's tuned files
(`VLLM_TUNED_CONFIG_FOLDER`), an override configuration or another device's files, with `int64`
`topk_ids` in its tests (with the naive assignment `expert_ids` is then `int64`), and with the
quantised paths; those launches are not part of the setup.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 41 launches of the hosts and one
guarded direct launch, each under 3.7 GiB of device memory, every one with the tile the host picks:
the host's nine test cases `(M, N, K, E, top_k)` (`(1, 256, 128, 8, 2)` to `(2, 512, 7168, 256, 8)`,
one with `MUL_ROUTED_WEIGHT` on the input, one with an expert map of 16 local experts, two with the
naive assignment, three with tuned files' tiles) and their second GEMMs; model-sized cases (Qwen3 at
`M = 4096`, `4` (naive) and `80`; DeepSeek-V3 at TP 8 at `M = 2048`, `1`, `8` (naive) and `9`;
Mixtral at TP 2; the second GEMM at `K = 14336` (Mixtral, `M = 1` naive, `16`, `64` and `512`),
`16384` (Mixtral-8x22B) and `32768` (Grok-1); Mixtral at TP 8 (`M = 1`, naive) and TP 2
(`M = 1024`); Llama-4-Scout at TP 8 (`M = 1`, naive); `M = 40000`); edges (`K = 32767` with
`N = 1000` and the weight, `E = 1`/`top_k = 1` with `GROUP_SIZE_M = 16`, `K = 64` with `N = 129`,
inputs scaled by 4); and the `-1` expert path. Among the tiles run: `BLOCK_SIZE_K = 256`
(`(16, 64, 256, 16)`, `(16, 64, 256, 1)`), `GROUP_SIZE_M = 64` (`(32, 128, 128, 64)`,
`(128, 256, 64, 64)`), `BLOCK_SIZE_N = 32` (`(16, 32, 64, 1)`) and `256`, and every
`get_default_config` row above runs. The largest error over the tolerance of `Post` is 0.934 (at `K = 64`,
where the `bfloat16` rounding dominates; 0.18 to 0.47 at `K ≥ 7168`); zeros are exact on the `-1`
path; unrouted rows keep their initial value; the inputs, the alignment buffers and the guard bytes
around `C` are unchanged. The tolerance over the reference, `tol / |ref|`, has median (90th
percentile) 0.005 (0.011) at `K = 128`, 0.009 (0.032) at `K = 2048`, 0.017 (0.063) at `K = 7168`,
0.027 (0.10) at `K = 14336`, 0.030 (0.11) at `K = 16384` and 0.053 (0.20) at `K = 32768`. The
negative tests (a result with the next expert's weights, without or with half the scale, all zero,
without the last `K` block, or accumulated in `bfloat16`) fail `Post` on every case that runs them,
from `K = 768` to `K = 32768`.
