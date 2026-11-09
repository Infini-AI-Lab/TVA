# Setup of `lora_shrink_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from vLLM
v0.33.0 (commit `4ba787ec99`), `vllm/lora/ops/triton_ops/lora_shrink_op.py:102`
`_lora_shrink_kernel`, with `do_shrink_kernel` and `mm_k` from
`vllm/lora/ops/triton_ops/kernel_utils.py`.

## The source

`source/lora_shrink.py` holds:
- `mm_k` (`kernel_utils.py:8-107`), `do_shrink_kernel` (`kernel_utils.py:236-356`) and
  `_lora_shrink_kernel` (`lora_shrink_op.py:101-216`), the `@triton.jit` helpers and kernel,
  verbatim but for the kernel's four parameter defaults (below). A program reads its adapter group's id (`lora_ids[program_id(2)]`, returning on `-1`),
  the group's size and start, and returns if its tile of rows is past the group's size; it loads the
  tile's token indices from `token_indices_sorted_by_lora_ids` (wrapping within the tile's rows),
  and `do_shrink_kernel` multiplies those rows of `inputs` by the adapter's slice of the LoRA A
  weights (`mm_k`, the K loop: unmasked tiles when `EVEN_K`, otherwise skipped, unmasked or masked
  per tile), scales by `scaling` and stores the `float32` tile, or with `SPLIT_K > 1` adds it with a
  relaxed `tl.atomic_add`. The other constexpr paths (`SPLIT_K > 1`, several slices through a table
  of weight addresses, `USE_GDC`, `STORE_PARTIALS`) stay in the source;
- `_get_lora_a_ptr` (from `lora/ops/triton_ops/utils.py:25-71`), the weights' address and strides;
- `lora_kernel_meta`, the LoRA mapping metadata in torch as vLLM's `LoRAKernelMeta.make`,
  `prepare_tensors` and `meta_args` build it (`lora/ops/triton_ops/lora_kernel_metadata.py:47-205`):
  the token indices sorted stably by adapter id, the distinct ids sorted (`-1` first when present)
  with their counts, the counts' cumulative sums from 0, the no-LoRA flag, and the grid's number of
  adapter groups;
- `lora_shrink`, the host, from vLLM's `_lora_shrink` (`lora_shrink_op.py:219-408`);
- `reference` (in `float64`) and the test under `if __name__ == "__main__":`.

Changes from vLLM's source (each with its `file:line` in the file's docstring): the host allocates
its output fresh (`torch.zeros`, `[num_slices, M, rank]`, `float32`, the buffer vLLM's
`PunicaWrapperGPU` passes, `lora/punica_wrapper/punica_gpu.py:248-253`) and returns it, where vLLM
zeroes the caller's buffer and writes into it; the weight-address cache is dropped; the
batch-invariant path (a second kernel) is dropped; one fixed configuration. The kernel's
signature loses its four parameter defaults (TVA's subset has none; the host passes each); its
other lines are unchanged.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. vLLM's wrapper takes its configuration
from `get_lora_op_configs("shrink", …)` (`lora/ops/triton_ops/utils.py:195-312`): tuned JSON files
only when `VLLM_TUNED_CONFIG_FOLDER` is set, otherwise the default (`utils.py:217-231`), whose
`split_k` and `block_k` depend on the batch (`split_k = 64`, `block_k = 256` below 128 tokens;
`split_k = 8`, `block_k = 32` from 128). The host passes the default for fewer than 128 tokens with
`SPLIT_K = 1` (the configuration the test set proves for this kernel):
- `BLOCK_M = 32`, `BLOCK_N = 16`, `BLOCK_K = 256`, `SPLIT_K = 1`, `GROUP_SIZE_M = 8`;
- `num_warps = 4`, `num_stages = 2`, `num_ctas = 1`;
- `EVEN_K = (K % (BLOCK_K · SPLIT_K) == 0)`, as the wrapper computes it (`lora_shrink_op.py:327`);
- `SLICE_NUM = len(lora_a_weights)`; `USE_GDC = False` and `launch_pdl = False` (the wrapper's
  `use_gdc` needs `VLLM_LORA_ENABLE_DUAL_STREAM`, off by default, `lora_shrink_op.py:339-343`);
  `STORE_PARTIALS = False`, `output_split_stride = 0`, `PARTIAL_N = 0`, `PARTIAL_K = 0`, the
  values of their defaults in vLLM's source (the batch-invariant path is off by default);
- the parameters' defaults removed: TVA's subset has no kernel parameter defaults; the host passes
  each (`lora_shrink_op.py:131-134`: `STORE_PARTIALS`, `output_split_stride`, `PARTIAL_N`,
  `PARTIAL_K`, with the values above);
- grid `(SPLIT_K · cdiv(M, 32) · cdiv(rank, 16), num_slices, num_active_loras)`, with
  `num_active_loras = max_loras + 1` (vLLM's default, `specialize_active_lora = False`,
  `config/lora.py:67`) or, with `specialize_active_lora`, the number of distinct ids in the mapping.

## The launches

The claim should cover the host's launches with one slice (`SLICE_NUM = 1`, the kernel reading the
weight tensor itself, as for vLLM's single-slice layers: row-parallel `o_proj` and `down_proj`,
non-merged column-parallel layers), `SPLIT_K = 1`, and either value of `EVEN_K`:
- `inputs` `bfloat16` `[M, hidden]` contiguous; the LoRA A weights `bfloat16`
  `[max_loras, 1, rank, hidden]` (or `[max_loras, rank, hidden]`) contiguous, as vLLM's
  `lora_a_stacked`; the output `float32` `[1, M, rank]` contiguous; `scaling` a `float` (vLLM's
  layers pass `1.0`, `lora/layers/base_linear.py:231-233`; its test `0.5`);
- the metadata as `lora_kernel_meta` builds it from a `token_lora_mapping` `int32` `[M]` with entries
  in `[-1, max_loras)`: `token_indices_sorted_by_lora_ids` `int32` `[M]`, a permutation of
  `0 … M − 1`; `lora_ids` `int32` `[max_loras + 1]` the distinct ids in increasing order then `-1`;
  `num_tokens_per_lora` `int32` `[max_loras + 1]` their counts then `0`; `lora_token_start_loc`
  `int32` `[max_loras + 2]` the counts' cumulative sums from `0`; a mapping of only `-1` launches
  nothing;
- shapes as variables, with the ranges vLLM's test uses (`tests/lora/test_punica_ops.py:320-421`):
  `M ≥ 1` (batches of 1 to 32 sequences of 128 tokens there; in serving, the step's tokens);
  `hidden` from 2 (`128 / 64`) to 128256, the hidden sizes of the LoRA models vLLM supports and their
  shards for tensor-parallel degrees 1 to 64, and 2049; `rank` 16, 32, 64, 128, 256
  (multiples of `BLOCK_N = 16`); `max_loras` 1, 4, 8, 32, 128; and LoRA A weights of fewer than `2^31`
  elements (`max_loras · rank · hidden < 2^31`). Ranks that are not a multiple of 16 and larger weight
  tensors are not this task's.

vLLM's launches with two or three slices (merged `qkv_proj`, `gate_up_proj`) take the `SLICE_NUM > 1`
path, which loads each slice's weight address from a `uint64` tensor and converts it to a pointer;
those, and the default `split_k` of 64 or 8, are not this task's.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes on nine cases
`(M, hidden, rank, max_loras, slices)`: `(1, 4096, 16, 4, 1)`, `(5, 2049, 8, 8, 1)`,
`(37, 2049, 1, 1, 1)`, `(128, 4096, 64, 32, 1)`, `(512, 2049, 256, 8, 2)`, `(300, 14336, 32, 4, 3)`,
`(2048, 8192, 16, 128, 1)`, `(4096, 5120, 128, 8, 1)` and `(64, 4096, 16, 4, 1)` with no token
having an adapter, with 0 to 50 % of the tokens mapped to `-1`, `specialize_active_lora` on in two
cases, and inputs drawn as vLLM's test draws them (`torch.rand`). Against the `float64` reference,
the largest error over `|ref| + mean |ref|` is 3.0e-6 to 3.4e-5 (`float32` accumulation; the largest
for `hidden = 14336`), and exactly 0 with no adapter.
