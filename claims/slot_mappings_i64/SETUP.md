# Setup of `slot_mappings_i64`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/v1/worker/gpu/block_table.py:334`, the kernel
`_compute_slot_mappings_kernel`, launched by `BlockTables.compute_slot_mappings`
(`block_table.py:229-260`).

## The source

`source/block_table.py` holds:
- `_load_ptr`, the `@triton.jit` helper the kernel calls (verbatim from
  `vllm/v1/worker/gpu/buffer_utils.py:353-357`): loads a `uint64` address and casts it to a
  pointer, with the hint `tl.multiple_of(ptr, 16)`;
- `_compute_slot_mappings_kernel`, the kernel, verbatim with its decorator `@triton.jit`
  (`block_table.py:333-414`);
- `BlockTables`, the host class, keeping what `compute_slot_mappings` uses (`block_table.py:17-271`):
  the constructor's arguments `block_sizes`, `max_num_reqs`, `max_num_batched_tokens`,
  `max_num_blocks_per_group`, `device`, `cp_size`, `cp_rank`, `cp_interleave`,
  `slot_mapping_enabled`, `dcp_sharded`; the per-group block tables, `[max_num_reqs,
  max_num_blocks]` `int32`, as plain `torch.zeros` tensors (vLLM's `StagedWriteTensor.gpu`,
  `buffer_utils.py:161`, is one), written by `set_rows`; `self.slot_mappings`; the layout tensors
  the kernel reads (`init_block_table_layout_tensors`, `block_table.py:105-123`: the tables'
  addresses as `uint64`, their row strides as `int64`, the block sizes as `int32`, the flags as
  `bool`); and `compute_slot_mappings` verbatim;
- `reference`, a test helper (the slots computed token by token on the CPU);
- the test under `if __name__ == "__main__":`.

The changes, listed in the source's module docstring: `_load_ptr` is copied into the kernel's file
(vLLM imports it from `buffer_utils.py`); `PAD_SLOT_ID = -1` is defined in the file (vLLM imports it
from `vllm/v1/attention/backends/utils.py:54`; only the host reads it); `triton` and
`triton.language as tl` are imported directly (vLLM's `vllm.triton_utils` re-exports them); the
host keeps only `compute_slot_mappings` and what it reads (the staged writes, the gather of block
tables and its kernel `_gather_block_tables_kernel`, the dummy tables and the sliding-window masks
are dropped).

## The config

No autotuning and no heuristics. The launch of `compute_slot_mappings`: grid `(num_groups,
num_reqs + 1)`, `max_num_tokens = slot_mappings.shape[1]` and `slot_mappings.stride(0)` as runtime
integers, `cp_rank` a runtime integer, and the constexprs `CP_SIZE = cp_size`,
`CP_INTERLEAVE = cp_interleave`, `PAD_ID = PAD_SLOT_ID = -1`, `TRITON_BLOCK_SIZE = 1024`. The
launch passes no option: Triton 3.8.0's defaults, `num_warps = 4`, `num_stages = 3`. With
`num_groups = 0` the host launches nothing.

## The launches

The callers (vLLM's `vllm/v1/worker/gpu/`):
- the model runner, `model_runner.py:1609-1614` (`BlockTables` built at `:718-730` with
  `cp_size = decode_context_parallel_size`, `cp_rank` the rank in the DCP group, `cp_interleave =
  cp_kv_cache_interleave_size`, and per KV-cache group the block size, `uses_slot_mapping` and
  `dcp_sharded`, `:644-674`), on the batch's `idx_mapping` (`int64`, `:1406-1408`),
  `query_start_loc` (`int32`, the running sums of the scheduled tokens) and `positions` (`int64`,
  `input_batch.py:30`); dummy batches (`InputBatch.make_dummy`, `input_batch.py:140-200`:
  `idx_mapping = arange(num_reqs)`, positions `0`);
- the context-parallel manager, `pcp_manager.py:791-797`, on its global batch with its own output
  (`torch.empty(num_groups, max_num_tokens, dtype=torch.int64)`, `:134-140`);
- the speculators: `spec_decode/multi_module_mtp/speculator.py:222-227`,
  `spec_decode/standalone_ar/speculator.py:210-216` (an output of `max_num_tokens + max_num_reqs`
  columns, `:94-98`) and `:316-318`, `spec_decode/target_dependent_ar/speculator.py:511-516`,
  `:570-575`, `:615-620`; their `idx_mapping` is `int64` (`spec_decode/speculator.py:154-156`) and
  may hold `-1` for dummy or padding requests (the kernel's comment, source lines 116-117).

The claim covers these launches with shapes as variables, the instances `setup_*_ok`:
- `setup_runner_one_group`: one attention group, block size 16, 1024 rows of 8192 blocks
  (128K-token contexts), 256 requests, 8192 tokens;
- `setup_runner_hybrid`: a hybrid model's groups, block sizes 16 (full attention), 32 (sliding
  window) and 1056 (a recurrent group without slot mappings, one block per row), 300 requests,
  16384 tokens;
- `setup_dcp2_rank0`, `setup_dcp2_rank1`, `setup_dcp8_rank5_interleave64` (a second group
  replicated, `dcp_sharded = False`), `setup_dcp4_rank1_interleave16`: decode context parallelism,
  block size 64, 100 requests, 8192 tokens;
- `setup_dummy_batch`, `setup_dummy_requests`: a dummy batch (100 requests over 4096 tokens), and
  a batch with dummy requests (`idx_mapping = -1`, any positions);
- `setup_speculator_expanded`: the standalone speculator's output of `4096 + 256` columns;
- `setup_source_test_cp1`, `setup_source_test_cp2`, `setup_source_test_cp4`: the source's own test
  (source lines 299-326) at `CP_SIZE` 1, 2 (rank 1) and 4 (rank 2, interleave 8);
- `setup_edge_max`: 8 groups, 8192 requests, `2^22` tokens.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 20 launches: the setup's
twelve host launches (the runner with one group and with hybrid groups, DCP at 2 ranks rank 0 and
rank 1, 8 ranks interleave 64 with an unsharded group, 4 ranks interleave 16 rank 1, the dummy
batch, dummy requests with random `int64` positions, the speculator's wider output, the source's
test at `CP_SIZE` 1, 2, 4), and eight guarded direct launches at the edges (no request, a batch
filling the row, `max_num_tokens = 1`, `query_start_loc[0] = 777` with rows of stride 2400 wider
than `max_num_tokens = 2100`, 8192 requests, `2^22` tokens, block sizes 1 and 7 with `CP_SIZE = 3`,
`CP_SIZE = 64` with interleave 128 at rank 63). The block tables are random over the whole `int32`
range except the rows of real requests, whose entries are block ids up to `(2^31 − 1)/B − 1` (the
largest slots that fit `int32`); the outputs start from random contents. All 4,381,544 written
elements are exact as `Post` states; every input, every other output element and every guard is
unchanged; 38 wrong outputs (the context-parallel local position ignored, the rank's filter
dropped, dummy requests mapped through row 0, the groups without slot mappings mapped, the padding
started one token late) all fail `Post`.
