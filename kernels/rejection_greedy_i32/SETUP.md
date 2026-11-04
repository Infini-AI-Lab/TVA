# Setup of `rejection_greedy_i32`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from vLLM
v0.33.0 (commit `4ba787ec99`), `vllm/v1/sample/rejection_sampler.py:793`
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

The flags proved are those of vLLM's main caller, `RejectionSampler.forward`
(`rejection_sampler.py:125`, its call at line 207), on an all-greedy batch (every request at
temperature 0, as in vLLM's greedy tests): `sampling_metadata.all_greedy` makes
`is_greedy = None` (lines 481-482), so every program runs the greedy path; synthetic mode is off
(`synthetic_mode=False`, the default), so `uniform_probs = None`,
`synthetic_conditional_rates = None` and `SYNTHETIC_MODE = False`. The source keeps the other
paths (an `is_greedy` mask for mixed batches, synthetic mode).

## The launches

The claim should cover the host's launches with `is_greedy = None` and `SYNTHETIC_MODE = False`,
with shapes as variables:
- `batch_size ≥ 1` requests, request `r` with `n_r ≥ 0` draft tokens; `max_spec_len = max_r n_r`
  (`SpecDecodeMetadata`, `vllm/v1/spec_decode/metadata.py:27`), at most `MAX_SPEC_LEN = 128`
  (`rejection_sampler.py:47`);
- `cu_num_draft_tokens`: `[batch_size]`, `int32`, the inclusive prefix sums of the `n_r`
  (`vllm/v1/worker/gpu_model_runner.py:2629-2639`);
- `draft_token_ids`: `[sum_r n_r]`, `int32` (gathered from the `int32` `input_ids`,
  `gpu_model_runner.py:2653-2654`); `target_argmax`: `[sum_r n_r]`, `int64`, token ids in
  `[0, vocab_size)`; `bonus_token_ids`: `[batch_size, 1]`, `int32`, contiguous (the sampler's
  output, `vllm/v1/sample/sampler.py:200-208`);
- `output_token_ids`: `[batch_size, max_spec_len + 1]`, `int32`, contiguous, allocated by the host
  and filled with `-1`;
- the code base's sizes: vLLM's test (`tests/v1/sample/test_rejection_sampler.py`) uses batches of
  1 to 8 requests with 0 to 5 drafts and vocabularies of 100 to 1000; in serving,
  `num_speculative_tokens` is typically 1 to 8 (EAGLE, MTP, n-gram), batches up to the maximum
  number of sequences (hundreds to a thousand), vocabularies of 32000 to 151936 or more.

## Checked on the H200

On an H200 (Triton 3.8.0, torch 2.11), the test passes: every output equals the reference exactly
on vLLM's six hand-written cases (a perfect match, an early mismatch, two requests, one draft, no
draft, several mismatches) and on 16 random batches: 1 to 1024 requests, up to 8 drafts per request
(and one batch up to 128), the same or a varying number of drafts per request (0 included),
acceptance rates 0 to 1, some draft ids `-1`, each with `is_greedy = None` and with a random mask.
