# Setup of `write_req_to_token_pool_i32`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/memory/common.py:9`, the
kernel `write_req_to_token_pool_triton`, launched by `write_cache_indices`
(`python/sglang/srt/mem_cache/allocation.py:56-103`), whose one caller is `alloc_for_extend`
(`allocation.py:374-488`): the scheduler's allocation of an extend (prefill) batch
(`ScheduleBatch.prepare_for_extend`).

## The source

`source/write_req_to_token_pool.py` holds:
- `write_req_to_token_pool_triton`, the `@triton.jit` kernel, verbatim (`common.py:8-56`), with
  the module's imports (`common.py:1-5`): one program per request writes its cached prefix
  (positions `0 … prefix_lens[b] - 1` of its row of `req_to_token`, from its prefix tensor) and
  its new locations (positions `prefix_lens[b] … seq_lens[b] - 1`, from `out_cache_loc` at the
  sum of the preceding requests' `extend_lens`), in blocks of 512;
- `write_cache_indices`, the host function, adapted from `allocation.py:56-90`;
- `reference`, a test helper (the fallback's per-request writes, `allocation.py:91-103`);
- the test under `if __name__ == "__main__":`.

Changes from the code base: none to the kernel; the module's other kernels (`get_last_loc_*`,
`common.py:59-163`) are left out. The host keeps the Triton branch of `write_cache_indices` only
(the attention-backend test and the per-request fallback are dropped), takes the table
`req_to_token` instead of the pool object and the device tensors only (the CPU mirrors were used
by the fallback), builds the `uint64` pointer tensor `prefix_pointers` as there but without pinned
memory and with a blocking copy, and writes out the launch options it leaves to Triton's
defaults (`num_warps=4`, `num_stages=3`). The source's module docstring lists the changes with
their lines.

## The config

No autotuning and no heuristics in the source. `BLOCK_SIZE = 512` is set in the kernel's body;
`req_to_token_ptr_stride` is a compile-time argument, the host's `req_to_token.shape[1]`. The
grid is `(bs,)`, `bs = req_pool_indices.shape[0]`; `num_warps=4`, `num_stages=3`.

## The launches

The claim covers the host's launches with shapes as variables:
- `req_to_token`: `[size + 1, max_context_len]`, `int32`, contiguous (`ReqToTokenPool`,
  `srt/mem_cache/memory_pool.py:406`; row `0` is a padding row), `max_context_len` the model's
  context length plus `4`, plus the draft tokens under speculative decoding
  (`srt/mem_cache/allocation_sizing.py:96-115`, `srt/mem_cache/kv_cache_configurator.py:1572`);
- `req_pool_indices`, `prefix_lens`, `extend_lens`: `[bs]`, `int64` (`allocation.py:395-411`),
  each request a distinct row; `seq_lens`: `[bs]`, `int64` (`srt/managers/schedule_batch.py:2876`),
  `seq_lens[b] = prefix_lens[b] + extend_lens[b] ≤ max_context_len`;
- the prefix tensors (`_prefix_kv_indices`, `allocation.py:363-371`): `int64` device tensors, the
  radix cache's indices of the matched prefix or the request's own row converted to `int64` (a
  copy), and for a request without a prefix the CPU tensor `_EMPTY_PREFIX` (its pointer is never
  followed); their data pointers in a `[bs]` `uint64` tensor;
- `out_cache_loc`: `[sum(extend_lens)]`, `int64`, the new KV slots of the allocators
  (`srt/mem_cache/allocator/`);
- batch sizes from 1 to the scheduler's running requests, prefixes up to the context, extends up
  to the chunked-prefill size.

The `setup_*_ok` theorems of `Claim.lean` cover:
- the host test's five cases (`source/write_req_to_token_pool.py:158-184`): pools of 64, 256 and
  4096 requests with a 4096-token context and batches of 1, 37 and 512; a 32768-token context
  with 8 requests; a 131072-token context with 4 requests (`setup_test_*`);
- serving shapes: a chunked prefill over 1024 requests of a 128K-context pool of 2048
  (`setup_serve_ctx128k_bs1024`), a 1M-token context (`setup_serve_ctx1m`), 8192 requests in one
  batch (`setup_serve_bs8192`), and the row length of speculative decoding with 8 draft tokens
  (`setup_spec_ctx32k`).

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 19 launches: the host's five
test cases; the four serving shapes (the 128K pool's prefixes cut to 16384 tokens to keep the case
under 4 GB); edges (every pair of prefix and extend lengths among `0, 1, 511, 512, 513, 1023,
1024, 1025, 1536, 2047`, no prefixes, no new tokens, whole rows on the first and last rows,
`max_context_len = 1`, requests that write nothing sharing a row, one request); and three direct
launches with the host's launch line (`extend_lens` larger than the extends with a longer
`out_cache_loc`, empty device prefixes whose pointer is `0`, the empty batch). The table starts
from random contents inside random guards, and the locations are random `int64` values over the
whole range. All 16,580,926 written elements are exact as `Post` states; every other element of
the table, the guards and every input are unchanged; 65 wrong outputs (the prefix not written,
the new locations one element late, every request reading `out_cache_loc` from `0`, a request's
last block dropped) all fail `Post`.
