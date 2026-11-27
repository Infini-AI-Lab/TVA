# Setup of `compute_position_i64`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/attention/position.py:33`,
the kernel `compute_position_kernel`, launched by `compute_position_triton` in the same file.

## The source

`source/position.py` is the whole upstream file with a module docstring added:
- `compute_position_triton`, the host function, verbatim: it allocates `positions`
  (`extend_seq_lens_sum` `int64` elements) and `extend_start_loc` (`bs` `int32` elements) with
  `torch.empty`, sets `has_prefix = extend_prefix_lens.shape[0] == bs`, and launches the kernel on
  the grid `(bs,)`;
- `compute_position_kernel`, the `@triton.jit` kernel, verbatim.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. Its one `tl.constexpr` parameter is
`has_prefix`, set by the host; `BLOCK_SIZE = 512` is set in the kernel's body. The launch passes no
options, so they are Triton 3.8.0's defaults: `num_warps=4, num_stages=3`.

## The launches

The host is called through `compute_position` (`srt/model_executor/forward_batch_info.py:2378`,
when the attention backend supports Triton), with `extend_seq_lens_sum` the sum of the lengths:
- `ForwardBatch.init_new` (`forward_batch_info.py:1249`) for every extend batch: prefill, mixed
  chunked prefill and speculative draft extends. The lengths and prefixes are `int32`, built from
  the batch's lists (`:1234-1239`) or handed in as device tensors by the speculative workers
  (`srt/speculative/eagle_worker_common.py:170-185`, `int32` from
  `kernels/ops/speculative/eagle.py:298-326` or built there); prefixes are given per request
  (`has_prefix`). The batch has `bs` from 1 to the running batch size, each length from 1 to the
  chunked-prefill budget (draft extends: the number of draft tokens), each prefix the request's
  cached length;
- two-batch overlap's split child batch (`srt/batch_overlap/two_batch_overlap.py:658`), with the
  device fields rebuilt in their own `int32` dtype (`:176-193`);
- DFlash and DSpark prefills (`srt/speculative/dflash_worker_v2.py:2401`,
  `srt/speculative/dspark_components/dspark_kv_inject.py:80`), `int32` lengths and prefixes;
- the code base's test (`test/registered/kernels/ops/speculative/test_next_verify_prep.py:112-134`):
  `bs` in `{1, 3, 129}`, every length the width in `{1, 4, 7}`, every prefix the value in
  `{0, 1, 131000}`, `int32`.

The claim covers these with shapes as variables (`mk_ok`: every `bs < 2^31`, with or without
prefixes); the setup theorems are the test's three batch sizes, a 64-request prefill in an
8192-token budget, one `2^20`-token request, 4096 requests in 65536 tokens, 256 draft extends of 8
tokens, the host without prefixes (`extend_prefix_lens` of another length, which no caller passes
but the host accepts), and an empty batch (`bs = 0`, no program).

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0) passes on 41 launches: the code base's 27 test cases
(three batch sizes, three widths, three prefix values), prefill batches (64 random requests, one
request of `2^20` tokens after a `2^20` prefix, 4096 requests with zero lengths among them, lengths
at the block edges 0 to 8192), 256 draft extends of 8 tokens, the host without prefixes (an empty
`extend_prefix_lens`, and one of another length), an empty batch, and edges (a prefix with
`p(b) + extend_seq_lens[b] = 2^31`, negative prefixes down to `-2^31`, 8192 requests, and three
guarded direct launches with random initial contents). All 1,445,877 written positions and every
start are exact as `Post` states; every other output element, every input and every guard is
unchanged; 112 wrong outputs (the prefix ignored, the inclusive running sum as the start, a
request's last block dropped, positions off by one) all fail `Post`.
