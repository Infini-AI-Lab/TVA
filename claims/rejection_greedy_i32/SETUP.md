# Setup of `rejection_greedy_i32`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/v1/sample/rejection_sampler.py:793`
`rejection_greedy_sample_kernel`.

## The source

`source/rejection_greedy.py` holds:
- `rejection_greedy_sample_kernel`, the kernel, verbatim with its decorator
  `@triton.jit(do_not_specialize=["max_spec_len"])` (`rejection_sampler.py:791-847`): one program per
  request accepts draft tokens while they equal the target's argmax, writes the target's tokens, and
  appends the bonus token if every draft was accepted;
- `rejection_greedy_sample`, the host, the greedy part of vLLM's `rejection_sample`
  (`rejection_sampler.py:432-536`) and the launch of `_rejection_greedy_sample`
  (`rejection_sampler.py:874-891`);
- `reference` (exact, on the CPU) and `make_batch`, test helpers;
- the test under `if __name__ == "__main__":`.

Changes from the code base, all in the host: it takes `target_argmax` (vLLM computes it as
`target_logits.argmax(dim=-1)`, line 511) instead of the logits, keeps the asserts on the inputs it
takes, allocates the output with `torch.full(..., PLACEHOLDER_TOKEN_ID, dtype=torch.int32)` as vLLM
does (lines 474-479), with `PLACEHOLDER_TOKEN_ID = -1` a local of the host (vLLM's module constants
`PLACEHOLDER_TOKEN_ID` and `GREEDY_TEMPERATURE`, lines 43-44, are not module-level in this file;
the kernel reads neither, and `GREEDY_TEMPERATURE` only serves vLLM's computation of `is_greedy`), and launches the kernel directly instead of through vLLM's warm-up dispatcher
(`vllm/model_executor/warmup/jit_warmup_triton_helper.py:585-605`), which passes the same arguments
by name with Triton's default launch options.

## The config

No autotuning and no heuristics in the source. The launch `_rejection_greedy_sample` makes: grid
`(batch_size,)`, `SYNTHETIC_MODE = synthetic_conditional_rates is not None`; launch options
Triton's defaults, written out as `num_warps=4`, `num_stages=3`. `max_spec_len` is a runtime scalar
(`do_not_specialize`).

The flags are those of vLLM's main caller, `RejectionSampler.forward`
(`rejection_sampler.py:125`, its call at line 207), with synthetic mode off
(`synthetic_mode=False`, the default), so `synthetic_conditional_rates = None` and
`SYNTHETIC_MODE = False`, on both kinds of batch it launches the kernel for (lines 481-484, 509):
- an all-greedy batch (every request at temperature 0, as in vLLM's greedy tests):
  `sampling_metadata.all_greedy` makes `is_greedy = None` and `uniform_probs = None`, so every
  program runs the greedy path;
- a mixed batch (some requests greedy, some random): `is_greedy = sampling_metadata.temperature ==
  GREEDY_TEMPERATURE` (a `[batch_size]` `bool` tensor, `GREEDY_TEMPERATURE = 0`), and
  `uniform_probs` is the `float32` tensor of `num_tokens` uniforms (lines 491-497), which the
  kernel does not read with `SYNTHETIC_MODE = False`; the programs of requests that are not greedy
  return at once.

An all-random batch does not launch the kernel (line 509). The source keeps the synthetic-mode
path, which is not part of the setup.

## The launches

The claim covers the host's launches with `SYNTHETIC_MODE = False`, with `is_greedy = None` or an
`is_greedy` mask, with shapes as variables:
- `batch_size ≥ 1` requests, request `r` with `n_r ≥ 0` draft tokens; `max_spec_len = max_r n_r`
  (`SpecDecodeMetadata`, `vllm/v1/spec_decode/metadata.py:27`), at most `MAX_SPEC_LEN = 128`
  (`rejection_sampler.py:47`);
- `cu_num_draft_tokens`: `[batch_size]`, `int32`, the inclusive prefix sums of the `n_r`
  (`vllm/v1/worker/gpu_model_runner.py:2629-2639`);
- `draft_token_ids`: `[sum_r n_r]`, `int32` (gathered from the `int32` `input_ids`,
  `gpu_model_runner.py:2653-2654`); `target_argmax`: `[sum_r n_r]`, `int64`, token ids in
  `[0, vocab_size)`; `bonus_token_ids`: `[batch_size, 1]`, `int32`, contiguous (the sampler's
  output, `vllm/v1/sample/sampler.py:200-208`);
- `is_greedy`: `None`, or `[batch_size]`, `bool`, contiguous (`temperature == 0`, any mix of
  greedy and random requests); `uniform_probs`: `None`, or `[num_tokens]`, `float32` (not read);
- `output_token_ids`: `[batch_size, max_spec_len + 1]`, `int32`, contiguous, allocated by the host
  and filled with `-1`;
- the host's own test (source lines 219-265): vLLM's hand-written cases with `is_greedy = None`,
  and random batches of 1 to 1024 requests, each run with `is_greedy = None` and with a random mask;
- the code base's sizes: vLLM's test (`tests/v1/sample/test_rejection_sampler.py`) uses batches of
  1 to 8 requests with 0 to 5 drafts and vocabularies of 100 to 1000; in serving,
  `num_speculative_tokens` is typically 1 to 8 (EAGLE, MTP, n-gram), batches up to the maximum
  number of sequences (hundreds to a thousand), vocabularies of 32000 to 151936 or more.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 34 launches: the 22 setup launches through the host (vLLM's
hand-written cases, and random batches of 1 to 1024 requests with up to 128 drafts, accept rates 0
to 1, each with `is_greedy = None` and with the test's mask) and edges (`S = 128` all accepted; `S`
above every `n_r`; `int64` targets with high bits; `S = 0` with 70000 requests; `2^21` requests; a
1024-request `S = 128` mixed batch with `is_greedy = temperature == 0` and `uniform_probs`; masks
all true, all false and of bytes `0..255`; three guarded direct launches, one with a mask). All
19,327,627 output cells are exactly as `Post` states, every input and guard is unchanged, and the
host-launched cases also equal the host's `reference`; 123 of 123 wrong outputs (the mask ignored,
a dropped bonus token, an all-zero output, a shifted packing, a missing column) fail `Post`.
