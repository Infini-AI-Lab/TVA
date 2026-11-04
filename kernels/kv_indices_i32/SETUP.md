# Setup of `kv_indices_i32`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/attention/kv_indices.py:59`
`create_flashinfer_kv_indices_triton`.

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

No decorator was removed: the kernel has no `@triton.autotune` or `@triton.heuristics`. The
parameters' defaults are removed (`ENTRY_PAGE_SIZE = 1`, `TOKEN_BLOCK_PARALLEL = False`,
`token_mapping = None`, `kv_indices.py:71-73`): TVA's subset has no kernel parameter defaults;
the host passes each. The configuration is that of the code base's launch and test: grid `(bs,)`, the table's row stride
`req_to_token.shape[1]` as a runtime argument, and the constexpr parameters at the kernel's
defaults, passed explicitly: `ENTRY_PAGE_SIZE=1`, `TOKEN_BLOCK_PARALLEL=False`,
`token_mapping=None`. `BLOCK_SIZE = 512` is set in the kernel's body. The launch options are
`num_warps=4, num_stages=3`: the launch passes none, and these are Triton 3.8.0's defaults.

## The launches

The claim should cover the host's launches with shapes as variables:
- `int32` `req_to_token` `[num_slots, max_context_len]` (`srt/mem_cache/memory_pool.py:406`,
  contiguous: the row stride is `max_context_len`); `int64` `req_pool_indices` and `seq_lens`
  (`srt/model_executor/cuda_graph_buffer_registry.py:648-660`; the code base's test passes
  `int32`); `int32` `kv_indptr` and `kv_indices`;
- `ENTRY_PAGE_SIZE=1`, `TOKEN_BLOCK_PARALLEL=False`, `token_mapping=None`, grid `(bs,)`, and
  `kv_start_idx=None` (the one-shot launch above, and the code base's test,
  `test/registered/attention/test_create_kvindices.py:58-66`). The other paths, and
  `kv_start_idx` given (sliding windows, which the host also accepts), stay in the source and are
  not part of the claim;
- `bs` from 1 to the batch size (the code base's test: 1, 37 and 1786 requests from a pool of
  4096), each request's length from 0 to `max_context_len` (the test: random in `[0, 4096)`;
  model contexts to 128K tokens and beyond), not a multiple of 512;
- `req_pool_indices[b]` a row of the table and `seq_lens[b] ≤ max_context_len`; `kv_indptr` the
  prefix sums of `seq_lens`, and `kv_indices` of `sum(seq_lens)` elements.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes, `kv_indices` equal to the
reference and `kv_indptr` to the prefix sums in every case: the code base's test's shapes
(a pool of 4096 × 4096, batches of 1, 37 and 1786 with random lengths in `[0, 4096)`, `int32`
and `int64` indices); lengths 1, 511, 512, 513, 1025 and 8192 in a pool of 513 × 8192; lengths
131072, 1, 70000, 4097, 0, 33, 99999 and 512 in a pool of 9 × 131072; 256 requests in a pool of
2049 × 32768; and, outside the claim, 100 requests with `kv_start_idx` (windows below 1024).
