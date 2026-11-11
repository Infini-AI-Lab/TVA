# Setup of `kv_indices_i32`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/attention/kv_indices.py:59`,
the kernel `create_flashinfer_kv_indices_triton`, launched by `create_kv_indices`, the launch of
`ForwardBatchDeepSeekMHAMixin.fetch_mha_one_shot_kv_indices`, which also takes the
`kv_start_idx` FlashInfer's planners pass.

## The source

`source/kv_indices.py` holds:
- `create_flashinfer_kv_indices_triton`, the `@triton.jit` kernel, copied verbatim but for
  its parameters' defaults: one program per request loads the request's row of `req_to_token`,
  its start (with `kv_start_idx`), its length and its output offset `kv_indptr[b]`, and copies
  the row's entries to `kv_indices` in blocks of 512;
- `create_kv_indices`, the host function, adapted from the launch in
  `ForwardBatchDeepSeekMHAMixin.fetch_mha_one_shot_kv_indices`
  (`srt/model_executor/forward_batch_deepseek_mha_mixin.py:265-296`);
- `reference`, a test helper (the rows' slices concatenated);
- the test under `if __name__ == "__main__":`.

The kernel keeps all its paths: a page-granular table (`ENTRY_PAGE_SIZE > 1`), several programs
per request (`TOKEN_BLOCK_PARALLEL`), and the remapping `token_mapping`. The host builds
`kv_indptr` (zeroed, then the cumulative sums of the lengths) and a fresh `kv_indices`, as that
caller and FlashInfer's planners do (`srt/layers/attention/flashinfer_backend.py:1788-1809`),
launches the kernel and returns both; the module docstring lists the changes with their source
lines.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. The
parameters' defaults are removed (`ENTRY_PAGE_SIZE = 1`, `TOKEN_BLOCK_PARALLEL = False`,
`token_mapping = None`, `kv_indices.py:71-73`): TVA's subset has no kernel parameter defaults;
the host passes each. The configuration is that of the code base's launch and test: grid `(bs,)`, the table's row stride
`req_to_token.shape[1]` as a runtime argument, and the constexpr parameters at the kernel's
defaults, passed explicitly: `ENTRY_PAGE_SIZE=1`, `TOKEN_BLOCK_PARALLEL=False`,
`token_mapping=None`. `BLOCK_SIZE = 512` is set in the kernel's body. The launch options are
`num_warps=4, num_stages=3`: the launch passes none, and these are Triton 3.8.0's defaults.

## The launches

The covered caller is the one-shot launch (`create_kv_indices`, grid `(bs,)`), with or without
`kv_start_idx`. The claim covers the host's launches with shapes as variables:
- `int32` `req_to_token` `[num_slots, max_context_len]` (`srt/mem_cache/memory_pool.py:406`,
  contiguous: the row stride is `max_context_len`); `req_pool_indices` and `seq_lens` each `int64`
  (`srt/managers/schedule_batch.py:2545, 2786`) or `int32` (the code base's test),
  independently; `int32` `kv_indptr` and `kv_indices`;
- `ENTRY_PAGE_SIZE=1`, `TOKEN_BLOCK_PARALLEL=False`, `token_mapping=None`, grid `(bs,)`;
- `kv_start_idx=None` (the one-shot launch above, and the code base's test,
  `test/registered/attention/test_create_kvindices.py:58-66`), or `kv_start_idx` a `[bs]` tensor,
  `int64` or `int32`, of each request's first position in `[0, 2^31)`: FlashInfer's planners pass
  it for sliding-window layers (`srt/layers/attention/flashinfer_backend.py:1697`: decode,
  `seq_lens - min(seq_lens, window + 1)`; `:2045-2058`: prefill, the window over the prefix or
  `seq_lens - paged_kernel_lens`; `:2071`: the full-attention wrapper, all `0`), through
  `KVIndexTranslator.pack_read_stream` (`srt/mem_cache/kv_index_translator.py:600`) and
  `call_begin_forward` (`flashinfer_backend.py:1807`), with the same arguments on a
  token-granular table; the code base's test runs it once (the last host test case);
- `bs` from 1 to the batch size (the code base's test: 1, 37 and 1786 requests from a pool of
  4096), each request's length from 0 to `max_context_len` (the test: random in `[0, 4096)`;
  model contexts to 128K tokens and beyond), not a multiple of 512;
- `req_pool_indices[b]` a row of the table (rows may be shared between requests) and
  `kv_start_idx[b] + seq_lens[b] ≤ max_context_len`; `kv_indptr` the prefix sums of `seq_lens`,
  and `kv_indices` of `sum(seq_lens)` elements.

The kernel's other paths are not part of the setup: `pack_read_stream`'s page-granular read table
(`ENTRY_PAGE_SIZE > 1`, a translating KV pool), its block-parallel grid (`TOKEN_BLOCK_PARALLEL`,
`num_token_blocks > 1`), and `token_mapping`.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 24 launches: the host's seven
test cases (a pool of 4096 × 4096 with batches of 1, 37 and 1786, `int32` and `int64` indices;
lengths 1 to 8192 in a pool of 513 × 8192; lengths up to 131072 in a pool of 9 × 131072; 256
requests in a pool of 2049 × 32768; 100 sliding windows with `kv_start_idx` in a pool of
300 × 4096), FlashInfer's `kv_start_idx` launches (sliding-window decode with a 4096-token window
over contexts to 131072, the full-attention wrapper's all-zero starts, the ragged prefill's window
over the prefix, `int32` lengths and starts with either row dtype, windows past the end of their
row), and edges (mixed index dtypes both ways, one row shared by several requests,
`max_context_len` 1 and 16, all lengths 0, `2^20`-token rows, 65536 requests, and three guarded
direct launches with `kv_indptr` starting at 0, 1234 and 99, the last with `kv_start_idx`). All
15,093,146 written indices are exact as `Post` states; every other output element, every input
and every guard is unchanged; 33 wrong outputs (the start ignored, the start off by one, a
request's last block dropped) all fail `Post`.
