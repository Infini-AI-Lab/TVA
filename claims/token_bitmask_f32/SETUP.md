# Setup of `token_bitmask_f32`

From SGLang v0.5.23 (commit `297e54fd10`),
`python/sglang/kernels/ops/grammar/bitmask_ops.py:14` `apply_token_bitmask_inplace_kernel`
(adapted by SGLang from xgrammar v0.1.17).

## The source

`source/bitmask_ops.py` holds:
- `apply_token_bitmask_inplace_kernel`, the `@triton.jit` kernel, copied verbatim
  (`bitmask_ops.py:13-81`): a persistent loop over the work items (a row and a block of `4096`
  tokens) that sets to `-inf` the logits whose bit in the packed `int32` bitmask is `0`;
- `apply_token_bitmask_inplace_triton`, the host (`bitmask_ops.py:84-141`), copied verbatim;
- `get_device_core_count`, the host's helper (`python/sglang/srt/utils/common.py:1161-1167`), with
  its MUSA and XPU branches dropped (it replaces the file's import of it): the CUDA device's
  multiprocessor count, else `0`.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. The host's configuration:
`BLOCK_SIZE = 4096`; on a CUDA device `NUM_SMS = get_device_core_count()` (`132` on the H200) and
`grid = (NUM_SMS,)`; on a device that reports no multiprocessors `grid = (num_rows * cdiv(vocab_size,
4096),)` and `NUM_SMS = next_power_of_2(grid[0])`; `num_warps = 4096 // 32 // (16 //
logits.element_size())` (`32` for `float32`), `num_stages = 3`. The kernel's arguments:
`num_rows` (`len(indices)`, or the logits' rows), `vocab_size = min(width, 32 * bitmask width)`,
`logits_strides = width` (the logits are contiguous), `bitmask_strides` the bitmask's width. A 1-D
logits or bitmask is one row. No flags.

## The launches

SGLang calls the host from the xgrammar backend's `apply_vocab_mask`
(`srt/constrained/xgrammar_backend.py:130-135, 301-306`; CUDA, not ROCm), through
`SamplingBatchInfo.apply_logits_bias` (`srt/sampling/sampling_batch_info.py:443-449`) on the
sampler's logits, and in speculative decoding on the draft and verify logits
(`srt/speculative/spec_utils.py:760-789`). The claim covers these launches, with shapes as
variables:
- `logits`: `[bs, vocab_size]`, `float32`, contiguous: the logits processor slices the lm head's
  output to the model's vocabulary and converts it to a fresh `float32` tensor
  (`srt/layers/logits_processor.py:1346-1369`);
- `bitmask`: `[bs, cdiv(vocab_size, 32)]`, `int32` (xgrammar's `get_bitmask_shape` and
  `bitmask_dtype`), filled by the grammars (`-1`, every token allowed, for rows without a grammar,
  `xgrammar_backend.py:66-75`) and moved to the device; any words;
- no `indices`;
- vocabularies of real models: 32000 (Llama 2, Mistral), 128256 (Llama 3), 129280 (DeepSeek-V3),
  151936 (Qwen 2 and 3), 152064 (Qwen 2.5), 202048 (Llama 4), 262144 (Gemma 3);
- batches from 1 to the maximum number of running requests, times the candidate tokens per request
  in speculative decoding (up to 8192 rows).

The host's interface also accepts (and the claim covers): a logits row wider than
`32 * bitmask width` (a padded lm head; only `vocab_size` tokens are processed), a 1-D logits and
bitmask, `indices` (a list or tensor of distinct rows, made `int32` by the host), and its branch
for a device without multiprocessors.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 33 launches: the setup's SGLang calls through the host
(vocabularies 32000 to 262144, batches 1 to 1024, a sparse grammar-like mask; the setup's 8192 rows
at vocabularies 129280 and 262144 cut to 2048 and 1024 rows to stay under 4 GB), a padded row,
indices, and edges (every token masked, none masked, 1-D tensors, vocabularies 0, 1, 31, 4096 and
4097, a bitmask narrower than the row, batch 0, 1000 rows of one token, 8192 rows of 64 tokens,
reversed and empty indices) and six direct launches with the host's launch line inside guarded
buffers (`NUM_SMS` 1, 7, 132 and 200, the no-multiprocessor branch's `grid = items < NUM_SMS`,
`vocab_size` below both strides, indices). The logits hold random bit patterns and the bitmasks
random words. All 825,223,555 processed logits are bit-exact as `Post` states, and every other
byte (the rest of the logits, the bitmask, the indices, the guards) is unchanged; 39 of 39 wrong
outputs (bits read most significant first, the polarity inverted, the next word's bits, tokens past
`vocab_size` written, `+inf` stored, another row's bits) fail `Post`.
