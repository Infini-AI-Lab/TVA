# Setup of `topk_log_softmax_f32`

From vLLM (commit `4ba787ec99`), `vllm/v1/worker/gpu/sample/logprob.py:17`
`_topk_log_softmax_kernel`, and its host `compute_token_logprobs` (`logprob.py:84`).

## The source

`source/logprob.py` holds, from that file:
- `_MAX_TOPK_BLOCK` (line 27) and `_topk_log_softmax_kernel`, the `@triton.jit` kernel (lines
  30–72), copied verbatim;
- `compute_token_logprobs` (lines 75–99), the host wrapper, copied verbatim;
- `compute_topk_scores` (lines 102–116), adapted: only the fast path that builds the sampler's
  token-id matrix `[sampled, top-k ids]` and calls `compute_token_logprobs`.

The changes from the upstream file (also in the source's module docstring):
- `from vllm.triton_utils import tl, triton` is `import triton` and `import triton.language as tl`
  (vLLM's `triton_utils` re-exports these two modules); the imports used only by removed code
  (`numpy`, `MAX_LOGPROB_TOKEN_IDS`, `SamplingParams`, `LogprobsTensors`, `StagedWriteTensor`,
  `UvaBackedTensor`) are dropped;
- the other kernels (`_ranks_kernel`, `_fill_logprob_token_ids_kernel`) and the class
  `LogprobTokenIdsState` are removed;
- `compute_topk_scores` keeps its fast path (no custom `logprob_token_ids`,
  `logits_mode=False`) and returns `(logprob_token_ids, scores)`; its custom-id path (which fills
  the id matrix with `_fill_logprob_token_ids_kernel`), the ranks (`_ranks_kernel`) and the
  `LogprobsTensors` wrapper are removed.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. `compute_token_logprobs` converts the
ids to `int64` (`token_ids.to(torch.int64)`), allocates a fresh `float32` output
`[batch_size, num_logprobs]`, and launches on the grid `(batch_size,)` with
`logits.stride(0)`, `token_ids.stride(0)`, `num_logprobs`, `vocab_size`, `BLOCK_SIZE=1024` and
`TOPK_BLOCK_SIZE = min(next_power_of_2(num_logprobs), 1024)`. No launch option; no flags.

## The launches

The callers of `compute_token_logprobs` in vLLM (`v1/worker/gpu/sample/sampler.py:177`,
`prompt_logprob.py:112, 306`, `spec_decode/rejection_sampler.py:217`, through
`compute_topk_scores` but for `prompt_logprob.py:112`) launch it with:
- `logits` a `float32` matrix `[batch_size, vocab_size]` with unit column stride: the processed
  logits (`sampler.py:242` copies them to a fresh `float32` tensor before the logits processors,
  temperature, min-p and top-k/top-p, which may set entries to `-inf`), or the model's logits when
  the head is `float32`; the sharded sampler passes a column slice `logits[:, :vocab_size]`
  (`model_runner.py:1653`), whose row stride exceeds `vocab_size`. The grammar bitmask sets
  entries to `-inf` in place (`model_runner.py:1664`); NaN logits can occur (the sampler counts
  them, `sampler.py:150`);
- `vocab_size` a model's vocabulary: `32000` (Llama 2), `50257` (GPT-2), `128256` (Llama 3),
  `129280` (DeepSeek-V3), `151936` (Qwen), `262208` (Gemma 3), up to about `2^18`;
- `batch_size` the rows of logits: the requests of a step (up to `max_num_seqs`), `1 + k` rows
  per request with `k` speculative tokens (the rejection sampler), or a chunk of at most `1024`
  prompt positions (`prompt_logprob.py:20`, `CHUNK_SIZE`); `0` rows is possible;
- the `int64` id matrix `[batch_size, num_logprobs]`, contiguous:
  - the sampler's fast path: column `0` the sampled token, then `num_logprobs` top-k ids
    (`logprobs=k`, `k = 0` gives one column); with `prompt_logprobs=-1` the top-k is the whole
    vocabulary, `1 + vocab_size` columns;
  - the custom path: `1 + max(num_logprobs, max_per_req)` columns (`max_per_req` up to
    `MAX_LOGPROB_TOKEN_IDS = 128`), the sampled token, then a request's own ids or its top-k, and
    `0` in the unused columns;
  - prompt-token scoring (`prompt_logprob.py:112`): a chunk of the request's label ids, negative
    ids clamped to `0` (`prompt_logprob.py:62`);
  every id is in `[0, vocab_size)`: sampled tokens and top-k indices are columns, and requests'
  own ids are checked against the vocabulary (`sampling_params.py:939-1030`);
- `TOPK_BLOCK_SIZE = min(next_power_of_2(num_logprobs), 1024)`, from `1` to `1024`.

The setup instances of `Claim.lean` (`setup_*_ok`) are: sampling without logprobs (Qwen,
`1024 × 151936`, `K = 1`), with `logprobs=20` (Llama 3, `256 × 128256`, `K = 21`), with
`logprobs=5` (Gemma 3, `128 × 262208`, `K = 6`), custom ids (DeepSeek-V3, `64 × 129280`,
`K = 129`), speculative decoding (GPT-2, `5120 × 50257`, `K = 2`), prompt logprobs over the whole
vocabulary (Llama 2, `1024 × 32000`, `K = 32001`), prompt-token scoring (Qwen, `1024 × 151936`,
`K = 8`), a column slice (Qwen, row stride `152064`) and an empty batch.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, one H200 NVL) runs all 9 instances above through the host (the
sampler's id matrices built by `compute_topk_scores`, the custom-id matrix as
`_fill_logprob_token_ids_kernel` fills it), plus direct launches with other `TOPK_BLOCK_SIZE`s and
edges of `Inst.ok` (README.md, "Checked on the H200"): `Post` holds on every row, worst error /
tolerance `0.49`; 32 of 32 cases pass.
