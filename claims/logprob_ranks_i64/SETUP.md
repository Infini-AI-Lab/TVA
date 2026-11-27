# Setup of `logprob_ranks_i64`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/v1/worker/gpu/sample/logprob.py:62`, the kernel
`_ranks_kernel`, launched by `compute_topk_scores` (`logprob.py:111`).

## The source

`source/logprob_ranks.py` holds:
- `_ranks_kernel`, the `@triton.jit` kernel, verbatim (`logprob.py:61-81`): one program per row
  counts the vocabulary entries whose logit is at least the sampled token's;
- `compute_token_ranks`, the host: the part of `compute_topk_scores` that computes the ranks
  (`batch_size, vocab_size = logits.shape`, `logprob.py:122`; the allocation of `token_ranks` and
  the launch, `logprob.py:173-181`), with the same grid and arguments;
- `reference` (exact on the CPU for a token whose logit is not `-inf`), a test helper;
- the test under `if __name__ == "__main__":`.

Changes from the code base: none to the kernel. The host keeps only the ranks part of
`compute_topk_scores`: the top-k selection, the logprob kernels it calls
(`_topk_log_softmax_kernel`, `_fill_logprob_token_ids_kernel`, with their hosts) and the
`LogprobsTensors` it returns do not touch `token_ranks` and are dropped, with the imports only they
use. The launch options the source leaves to Triton's defaults are written out (`num_warps=4`,
`num_stages=3`).

## The config

No autotuning and no heuristics in the source. The host's choice: grid `(batch_size,)`,
`BLOCK_SIZE = 8192`; `num_warps=4`, `num_stages=3` (Triton's defaults, which the host uses).
`logits_stride = logits.stride(0)` and `vocab_size` are runtime integers.

## The launches

The claim covers the host's launches, with shapes as variables:
- `logits`: `[batch_size, vocab_size]`, last dimension contiguous, row stride
  `logits.stride(0)`. Its dtype is `float32` when the sampler has processed the logits
  (`vllm/v1/worker/gpu/sample/sampler.py:242` copies them to a new `float32` tensor) and the
  model's dtype (`bfloat16` or `float16`) for raw logits (`apply_sampling_params` returns them
  unchanged when no request needs processing, `sampler.py:238-239`). The stride is `vocab_size`
  for a contiguous tensor, and larger for the slice `logits[:, :self.vocab_size]` of a padded
  vocabulary (`vllm/v1/worker/gpu/model_runner.py:1653`);
- `sampled_token_ids`: `[batch_size]`, `int64`, contiguous, token ids in `[0, vocab_size)`;
- `token_ranks`: `[batch_size]`, `int64`, allocated by the host.

The callers of `compute_topk_scores` in the code base:
- the sampler (`sampler.py:177`): one row per sampled logit row, the sampled tokens (`int64`,
  `gumbel.py:414`, "Use int64 for later indexing"), raw or processed logits;
- the rejection sampler of speculative decoding (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:217`):
  `num_reqs · (k + 1)` rows, the flattened sampled tokens (`int64`; rows past a request's accepted
  tokens hold `0`, a valid id);
- prompt logprobs (`vllm/v1/worker/gpu/sample/prompt_logprob.py:306`): chunks of at most
  `CHUNK_SIZE = 1024` prompt positions, the prompt's token ids (`int64`);
- the diffusion model `diffusion_gemma` (`vllm/model_executor/models/diffusion_gemma.py:1644`):
  the rows of one slot (`src[pos : pos + k_i]`, contiguous), its argmax tokens (`int64`).

Sizes in the code base: vocabularies of 32000 (Llama-2), 50257 (GPT-2, padded to 50304),
128256 (Llama-3), 151936 (Qwen3) and 262144 (Gemma-3); batches of one to thousands of rows;
vLLM's regression test of the `int64` row offset (`tests/v1/sample/test_logprobs.py:1573`) runs
`16448` rows of `131072` (`batch_size · vocab_size > 2^31`).

`Claim.lean`'s `setup_*` instances: the host's six test cases; the sampler on `1024 × 128256`
`float32`, `256 × 151936` `bfloat16`, `256 × 32000` `float16` and the slice `64 × 50257` with stride
`50304`; a speculative batch of `2560 × 128256`; prompt chunks `1024 × 151936` `bfloat16` and
`37 × 262144` `float32`; vLLM's `16448 × 131072` regression shape; an empty batch.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 42 launches: the 15 setup launches through the host (the
`16448 × 131072` regression shape cut to `4100` rows, 2.1 GB, where the byte offsets of the last
rows pass `2^31`), each followed by a guarded direct launch, and 27 edges: ties on a coarse
`bfloat16` grid, the token at `0` and at `vocab_size − 1`; NaN, `±inf` and `±0` token logits and
entries in all three dtypes, rows of `-inf` only, vocabularies with and without padding lanes
(`8192`, `8193`, `32000`, `49153`; 12 launches); `vocab_size = 1` (the scalar arguments
compile-time constants, twice) and `2`; strides `0` (an expanded row), `9000 < vocab_size`
(overlapping rows) and `2^22`; `vocab_size = 2^20`; 65536 rows; direct launches with `BLOCK_SIZE`
`1`, `2`, `16`, `1024`, `2^16`, `2^17`. The largest block tested is `2^17`: a block of `2^20`
lanes takes too long to compile (more than 30 minutes). Every rank is exactly `Post`'s (error 0,
tolerance 0), the inputs are unchanged bit for bit, and in each guarded launch every element
outside the output keeps the sentinel. Each wrong output fails `Post` on some launch: a strict
`>` (41 of 42 launches), no padding term (15), the padding term always (32), NaN entries counted
(18), a 0-based rank (41), the first block only (30).
