# Setup of `lora_shrink_bf16`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/lora/ops/triton_ops/lora_shrink_op.py:102`, the
function `_lora_shrink_kernel`, with `do_shrink_kernel` and `mm_k` from
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
  per tile) over the program's part of `K` (`SPLIT_K` parts), scales by `scaling` and adds the
  `float32` tile into the output with a relaxed `tl.atomic_add` (or, with `SPLIT_K = 1`, stores it);
  with several slices it loads its slice's weight address from a `uint64` table and converts it to a
  pointer. The other constexpr paths (`USE_GDC`, `STORE_PARTIALS`) stay in the source;
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
zeroes the caller's buffer and writes into it; the weight-address cache is removed; the
batch-invariant path (a second kernel) is removed; `get_lora_op_configs` is replaced by its default
configuration, written in the host. The kernel's
signature loses its four parameter defaults (TVA's subset has none; the host passes each); its
other lines are unchanged.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. vLLM's wrapper takes its configuration
from `get_lora_op_configs("shrink", …)` (`lora/ops/triton_ops/utils.py:195-312`): tuned JSON files
only when `VLLM_TUNED_CONFIG_FOLDER` is set, otherwise the default (`utils.py:217-231`), whose
`split_k` and `block_k` depend on the batch `M`. The host computes that default (source lines
552–560):
- `BLOCK_M = 32`, `BLOCK_N = 16`, `GROUP_SIZE_M = 8`; `BLOCK_K = 256`, `SPLIT_K = 64` for fewer than
  128 tokens, `BLOCK_K = 32`, `SPLIT_K = 8` from 128 (`is_batch_invariant` off);
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

The claim should cover the host's launches with any number of slices (one: the kernel reads the
weight tensor itself, as for vLLM's single-slice layers, row-parallel `o_proj` and `down_proj`,
non-merged column-parallel layers, the logits LoRA; several: the kernel loads each slice's weight
address from the `uint64` table, as for vLLM's merged `qkv_proj`, three slices, and `gate_up_proj`,
two, `lora/layers/column_parallel_linear.py:204, 467`), the default configuration for every batch
(`SPLIT_K = 64` below 128 tokens, `SPLIT_K = 8` from 128), and either value of `EVEN_K`:
- `inputs` `bfloat16` `[M, hidden]` contiguous; each slice's LoRA A weights `bfloat16`
  `[max_loras, 1, rank, hidden]` (or `[max_loras, rank, hidden]`) contiguous, as vLLM's
  `lora_a_stacked`, all slices of one shape; the output `float32` `[num_slices, M, rank]` contiguous;
  `scaling` a `float` (vLLM's layers pass `1.0`, `lora/layers/base_linear.py:231-233`; its test
  `0.5`);
- the metadata as `lora_kernel_meta` builds it from a `token_lora_mapping` `int32` `[M]` with entries
  in `[-1, max_loras)`: `token_indices_sorted_by_lora_ids` `int32` `[M]`, a permutation of
  `0 … M − 1`; `lora_ids` `int32` `[max_loras + 1]` the distinct ids in increasing order then `-1`;
  `num_tokens_per_lora` `int32` `[max_loras + 1]` their counts then `0`; `lora_token_start_loc`
  `int32` `[max_loras + 2]` the counts' cumulative sums from `0`; a mapping of only `-1` launches
  nothing;
- shapes as variables, with the ranges vLLM's test uses (`tests/lora/test_punica_ops.py:320-421`):
  `M ≥ 1` (batches of 1 to 32 sequences of 128 tokens there; in serving, the step's tokens, a few in
  decode and thousands in prefill); `hidden` from 2 (`128 / 64`) to 128256, the hidden sizes of the
  LoRA models vLLM supports and their shards for tensor-parallel degrees 1 to 64, and 2049; `rank`
  16, 32, 64, 128, 256 (multiples of `BLOCK_N = 16`); `max_loras` 1, 4, 8, 32, 128; one to three
  slices; and LoRA A weights of fewer than `2^31` elements (`max_loras · rank · hidden < 2^31`).

Not part of the setup: ranks that are not a multiple of 16 (vLLM's test also uses 1, 4 and 8; the
host's test in `source/` runs ranks 8 and 1, its cases 2 and 3); weight tensors of `2^31` elements or
more; tuned configurations (`VLLM_TUNED_CONFIG_FOLDER`); the batch-invariant path.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host on 29 covered launches: the host test's cases 1 and
4 to 8 (cases 5 and 6 with their two and three slices); decode and prefill launches of vLLM's layers
with `scaling = 1.0` (`qkv_proj` with three slices at 8 tokens, `gate_up_proj` with two at 2048
tokens, `down_proj` at 64, 1024 and 4096 tokens), on `torch.rand` and signed `torch.randn` data;
127 and 128 tokens (the two configurations); `scaling` `0.1`, `1/3`, `2^-7` and `−3.75`; and edges
(hidden 2, 255, 2049, 16384, 128256, `2^17` and `2^17 − 1`; 128 adapters with
`num_active_loras = 129`; one token at rank 256 with three slices; `specialize_active_lora`). Every
one is within the claim's tolerance: the largest error / tolerance is about `2·10^-2` (`torch.rand`
data, 128 adapters), at most `1.3·10^-3` on signed data; every element of a token with no adapter is
`+0.0` in every slice; the guard bands around the output and every input buffer are unchanged. On
signed data at hidden 14336 the tolerance is `2.8·10^-3` of the output (median) below 128 tokens and
`1.5·10^-2` from 128 tokens. Wrong outputs (all zero, the last `BLOCK_K` block of `k` dropped, one of
the `SPLIT_K` parts dropped, the scale missing, another adapter, swapped slices) are rejected. It also
runs ranks 8, 1 and 4 (cases 2 and 3, not part of the setup) for information: error / tolerance at
most `6.4·10^-3`.
