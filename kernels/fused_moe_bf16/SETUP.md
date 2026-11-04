# Setup of `fused_moe_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from vLLM
v0.33.0 (commit `4ba787ec99`), `vllm/model_executor/layers/fused_moe/fused_moe.py:298`
`fused_moe_kernel`.

## The source

`source/fused_moe.py` holds:
- `write_zeros_to_output` (`fused_moe.py:44-61`) and `fused_moe_kernel` (`fused_moe.py:297-608`),
  the `@triton.jit` helper and kernel, verbatim but for the kernel's one parameter default
  (below). A program computes one `BLOCK_SIZE_M × BLOCK_SIZE_N`
  tile of the output: it reads `num_tokens_post_padded` and returns if its block of sorted rows is
  past it; it loads the block's `BLOCK_SIZE_M` entries of `sorted_token_ids` (flat indices
  `t · top_k + j`, or the padding value `M · top_k`), masks those not below `num_valid_tokens`, loads
  the block's expert from `expert_ids` (writing zeros and returning on `-1`), and runs the K loop
  over the rows `sorted_token_ids // top_k` of `A` and the expert's slice of `B`; it multiplies by the
  routing weights if `MUL_ROUTED_WEIGHT`, casts to `compute_type` and stores to the rows
  `sorted_token_ids` of `C`. The kernel's other constexpr paths (fp8/int8 quantisation, bias,
  `SWAP_AB`, the tensor-descriptor path `USE_TD`, `naive_block_assignment`) stay in the source;
- `moe_align_block_size`, a torch-only equivalent of vLLM's `moe_align_block_size`
  (`fused_moe/moe_align_block_size.py:55-169`), whose work vLLM does in a CUDA op
  (`csrc/moe/moe_align_sum_kernels.cu`): it builds `sorted_token_ids`, `expert_ids` and
  `num_tokens_post_padded` with the same sizes and contents (within one expert, the flat indices in
  ascending order, where vLLM's op leaves the order to its threads);
- `invoke_fused_moe_triton_kernel` (from `fused_moe.py:761-910`), the launcher, for the unquantised
  path with no bias;
- `fused_moe_gemm1`, the host: the part of vLLM's `fused_experts_impl` (`fused_moe.py:1664-1822`)
  up to and including its first GEMM, `hidden_states @ w1[e]ᵀ` into `intermediate_cache1`;
- `reference` (the routed products in `float64`) and the test under `if __name__ == "__main__":`.

Changes from vLLM's source (each with its `file:line` in the file's docstring): the alignment
helper in torch; `intermediate_cache1` allocated fresh as `[M, top_k, N]` (vLLM takes a view of a
buffer it reuses for the third cache); the expert assignment always by `moe_align_block_size` (the
`naive_block_assignment` branch of `_prepare_expert_assignment`, `fused_moe.py:1570-1591`, is not
taken); the quantised and wna16 branches of the wrapper dropped; one fixed configuration; the grid
function computed directly. The kernel's signature loses the default of `USE_TD` (TVA's subset has
none; the host passes it); its other lines are unchanged.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. vLLM's `try_get_optimal_moe_config`
(`fused_moe.py:1421-1450`) takes a tuned configuration from a JSON file when one exists for
`(E, N, device, dtype)`, and otherwise `get_default_config` (`fused_moe.py:1298-1418`), which
depends on the batch. The host passes one configuration: `get_default_config`'s branch for `bfloat16`
(`fused_moe.py:1372-1416`) at large batches (`M > 512`, and `M // E > 128` for `GROUP_SIZE_M`):
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 16`,
  `SPLIT_K = 1` (the wrapper sets it, `fused_moe.py:841`; the kernel does not use it);
- `num_warps = 8`, `num_stages = 3`;
- grid `(cdiv(EM, 128) · cdiv(N, 128),)`, with `EM = sorted_token_ids.size(0)`, or, when
  `M < BLOCK_SIZE_M`, `EM = min(sorted_token_ids.size(0), M · top_k · BLOCK_SIZE_M)`
  (`fused_moe.py:822-833`).

The flags, as vLLM's first GEMM sets them for unquantised `bfloat16` weights
(`fused_moe.py:1800-1822` and `invoke_fused_moe_triton_kernel`):
- `MUL_ROUTED_WEIGHT = apply_router_weight_on_input`, `False` by default (vLLM's first GEMM applies
  the routing weights only when the model asks for them on the input, e.g. Llama 4; the second GEMM
  passes `True` with `top_k = 1`); `top_k = top_k_num` (`topk_ids.size(1)`);
- `compute_type = tl.bfloat16`; `use_fp8_w8a8 = use_int8_w8a8 = use_int8_w8a16 = per_channel_quant
  = False`; `HAS_BIAS = False` (`w1_bias` is `None` by default); `SWAP_AB = False` (computed only
  for fp8); `USE_TD = False` (`resolve_moe_use_td`, `fused_moe/utils.py:736-748`: off on CUDA unless
  `VLLM_TRITON_USE_TD=1`); `naive_block_assignment = False` (a `sorted_token_ids` tensor is passed);
- `b_bias_ptr`, `a_scale_ptr`, `b_scale_ptr` are `None`, their nine strides `0`, and
  `group_n = group_k = 0`;
- the parameters' defaults removed: TVA's subset has no kernel parameter defaults; the host passes
  each (`fused_moe.py:353`: `USE_TD: tl.constexpr = False` becomes `USE_TD: tl.constexpr`, passed
  `False`).

## The launches

The claim should cover the host's launches:
- `hidden_states` (`A`) `bfloat16` `[M, K]` contiguous; `w1` (`B`) `bfloat16` `[E_local, N, K]`
  with `stride(-1) = 1` (contiguous in vLLM); `C` `bfloat16` `[M, top_k, N]` contiguous; `topk_weights`
  `float32` `[M, top_k]` with `stride(1) = 1`, `topk_ids` `int32` `[M, top_k]` (vLLM's routers give
  `int32`; the helper also accepts `int64`, as vLLM's test `test_moe.py:1934` does);
- `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded` as `moe_align_block_size` builds them
  with block size 128: each routed flat index once, at its expert's padded offset; padding entries
  `M · top_k`; `expert_ids` the expert of each block below `num_tokens_post_padded`, `-1` after it;
- the routed experts: each `topk_ids` entry in `[0, E)` with `E = global_num_experts` (`E_local` by
  default); with expert parallelism, an `expert_map` `int32` `[E]` sending each global expert to its
  local index in `[0, E_local)` or to `-1` (entries routed to `-1`, or outside `[0, E)`, are skipped,
  and their rows of `C` are not written, as in vLLM);
- shapes as variables, with the ranges vLLM uses: `M ≥ 1` (tests `tests/kernels/moe/test_moe.py:247-253`:
  1, 33, 32768, 40000); `E` from 8 to 256 in tests and models (`test_moe.py:169-170`: 8, 64, 192,
  128, 256; the alignment op needs `E` rounded up to 32 below 1024); `top_k` 1, 2, 6, 8
  (`test_moe.py:172-173, 1909`); `K` (hidden size) 128, 511, 1024 in tests and 2048 to 7168 in models;
  `N` (twice the intermediate size over the tensor-parallel degree) 256 to 4096 in tests and, e.g.,
  Mixtral-8x7B (E 8, top 2, hidden 4096, intermediate 14336: `N = 28672 / TP`), DeepSeek-V3 (E 256,
  top 8, hidden 7168, intermediate 2048: `N = 4096 / TP`), Qwen3-30B-A3B (E 128, top 8, hidden 2048,
  intermediate 768); `K` and `N` multiples of the blocks or not.

vLLM also launches this kernel with `naive_block_assignment = True` (no sorted ids) when
`M · top_k · 4 ≤ E` and there is no expert map (`fused_moe.py:1570-1591`), with tuned
configurations from its JSON files, and with the quantised paths; those launches are not this
task's.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes on eight cases
`(M, N, K, E, top_k)`: `(1, 256, 128, 8, 2)`, `(33, 4096, 128, 64, 6)`, `(100, 1022, 511, 8, 2)`,
`(222, 512, 7168, 256, 8)`, `(1024, 1536, 2048, 128, 8)`, `(2048, 3584, 4096, 8, 2)`, `(64, 2048,
5120, 16, 1)` with `MUL_ROUTED_WEIGHT = True`, and `(300, 1024, 1024, 64, 6)` with an expert map of 16
local experts (only the routed rows compared), with inputs drawn as vLLM's test draws them
(`randn / 10`, softmax routing). Against the `float64` reference the largest error over
`|ref| + mean |ref|` is 2.6e-3 to 3.3e-3 (the `bfloat16` rounding of the output). The alignment
helper reproduces vLLM's documented example (`moe_align_block_size.py:112-127`).
