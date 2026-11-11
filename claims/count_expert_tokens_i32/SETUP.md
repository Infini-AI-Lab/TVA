# Setup of `count_expert_tokens_i32`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/model_executor/layers/fused_moe/utils.py:171`, the
kernel `_count_expert_num_tokens`, launched by the wrapper `count_expert_num_tokens`.

## The source

`source/count_expert_tokens.py` holds:
- `_count_expert_num_tokens`, the `@triton.jit` kernel, verbatim (`utils.py:170-200`): one program
  per local expert counts the entries of `topk_ids` routed to it, optionally through an expert map;
- `count_expert_num_tokens`, the host wrapper (`utils.py:202-238`), with the same assert,
  allocation, grid and arguments;
- `make_topk_ids`, `make_expert_map` (after vLLM's `determine_expert_map`,
  `fused_moe/expert_map_manager.py:60-112`) and `reference` (exact counts on the CPU), test helpers;
- the test under `if __name__ == "__main__":`.

Changes from the code base: none to the kernel. The wrapper's docstring is shortened, and the
launch options it leaves to Triton's defaults are written out (`num_warps=4`, `num_stages=3`).

## The config

No autotuning and no heuristics in the source. The wrapper's own choice: grid
`(num_local_experts,)`, `BLOCK_SIZE = next_power_of_2(min(topk_ids.numel(), 1024))`,
`HAS_EXPERT_MAP = expert_map is not None`; `num_warps=4`, `num_stages=3` (Triton's defaults, which
the wrapper uses).

## The launches

The claim covers the wrapper's launches, both constexpr paths of `HAS_EXPERT_MAP`, with
shapes as variables:
- `topk_ids`: `[num_tokens, topk]`, contiguous, `int32` (the default of vLLM's routers,
  `fused_moe/router/fused_topk_router.py:158-161`) or `int64` (DeepEP's dispatch,
  `prepare_finalize/deepep_ht.py:110`, and vLLM's test); entries are `-1` or global expert ids;
- `expert_map` (with the map): `[num_global_experts]`, `int32`, each entry a local id in
  `[0, num_local_experts)` or `-1`, as `determine_expert_map` builds it (linear or round-robin
  placement); without the map, `topk_ids` holds local ids or `-1`;
- `expert_num_tokens`: `[num_local_experts]`, `int32`, allocated by the wrapper;
- the code base's sizes: vLLM's test (`tests/kernels/moe/test_count_expert_num_tokens.py`):
  `num_tokens` 1 to 7317, `topk` 1, 2, 6, 8, 32 or 64 global experts, `ep_size` 1, 2, 4, and
  `numel` from 1 to 8191; model configs: DeepSeek-V3 (256 experts, top-8), Qwen3-MoE
  (128, top-8), Mixtral (8, top-2), with expert parallelism up to 16 or more ranks; the number of
  tokens a rank receives can be tens of thousands.

The caller in the code base is `deepgemm_moe_permute` (`fused_moe/deep_gemm_utils.py:538`), which
counts locally when the dispatch did not supply the counts.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 126 launches: the host's 112
(vLLM's test shapes with `int64` ids, 64 experts, `ep_size` 1/2/4, 1 to 7317 tokens; its `numel`
cases from 1 to 8191; `int32` model configurations up to 16384 tokens with 8, 128 and 256 experts,
`ep_size` up to 16, linear and round-robin maps, an uneven split of 128 experts on 3 ranks, each
with and without the expert map) and 14 edges (arbitrary `int32`/`int64` values without a map,
arbitrary map entries, `numel = 1`, one expert, `2^26` entries, 4096 programs). Every count equals
the reference exactly, the inputs are unchanged, and in a guarded relaunch every byte outside the
counts is unchanged.
