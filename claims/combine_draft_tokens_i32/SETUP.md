# Setup of `combine_draft_tokens_i32`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/v1/worker/gpu/input_batch.py:415`, the kernel
`_combine_sampled_and_draft_tokens_kernel`, launched by `combine_sampled_and_draft_tokens`
(`input_batch.py:475`), whose one caller is `GPUModelRunner.prepare_inputs`
(`vllm/v1/worker/gpu/model_runner.py:1510-1521`).

## The source

`source/combine_draft_tokens.py` holds:
- `_combine_sampled_and_draft_tokens_kernel`, the `@triton.jit` kernel (`input_batch.py:414-472`),
  verbatim but for the default of its last parameter: one program per request writes the
  request's logits indices and, past its prefill, its last sampled token and its draft tokens into
  `input_ids`;
- `combine_sampled_and_draft_tokens`, the host (`input_batch.py:475-513`), verbatim.

Changes from the code base: `from vllm.triton_utils import tl, triton` (vLLM's re-export of Triton,
`input_batch.py:10`) is written `import triton` and `import triton.language as tl`; the default
`= 1` of `NUM_NEW_SAMPLED_TOKENS: tl.constexpr` (`input_batch.py:427`) is removed: TVA's subset has
no kernel parameter defaults, and the host passes the parameter on every launch. The module's
other classes and kernels are not copied. The module docstring lists the changes with their source
lines.

## The config

No autotuning and no heuristics. The launch the host makes: grid `(num_reqs,)` with `num_reqs =
idx_mapping.shape[0]`; `NUM_NEW_SAMPLED_TOKENS = num_new_sampled_tokens` (`0` or `1`, asserted);
`BLOCK_SIZE = triton.next_power_of_2(num_speculative_steps + num_new_sampled_tokens)` with
`num_speculative_steps = draft_tokens.shape[-1]`; `draft_tokens_stride = draft_tokens.stride(0)`, a
runtime integer (`1` when `num_speculative_steps` is `0` or `1`, which Triton then compiles as the
constant `1`). Launch options: none given, so Triton 3.8.0's defaults (`num_warps=4`,
`num_stages=3`).

## The launches

The caller (`model_runner.py:1388-1521`) passes, on every step:
- `input_ids`: the runner's `[max_num_tokens]` `int32` buffer (`input_batch.py:29`;
  `max_num_tokens = max_num_batched_tokens`, `model_runner.py:231`);
- `idx_mapping`: `[num_reqs]` `int64`, each request's row in the request state
  (`model_runner.py:1406-1408`), distinct rows below `max_num_reqs`;
- `last_sampled_tokens`: `[max_num_reqs, 1]` `int64`; `draft_tokens`: `[max_num_reqs,
  num_speculative_steps]` `int64`, contiguous; `prefill_len`: `[max_num_reqs]` `int32` in host
  memory mapped for the device (`vllm/v1/worker/gpu/states.py:53, 68-80`);
- `query_start_loc`: `[num_reqs_padded + 1]` `int32`, `0` then the running sums of the scheduled
  tokens, padded with the token count (`model_runner.py:1461-1468`, `1484`); `seq_lens`:
  `[num_reqs_padded]` `int32`, each request's computed tokens plus its scheduled tokens
  (`prepare_pos_seq_lens`, `model_runner.py:1499-1506`);
- `cu_num_logits`: `[num_reqs + 1]` `int32`: `0, 1, …, num_reqs` without drafts and with
  `num_new_sampled_tokens_per_step = 1` (`model_runner.py:1414-1424`); otherwise `0` then the
  running sums of `num_draft_tokens + num_bonus_tokens`, after the assert that every request
  schedules at least that many tokens (`model_runner.py:1425-1441`); with adaptive verification,
  both running sums are recomputed on the device from the admitted draft counts, `cu_num_logits`
  from `capacities + num_bonus_tokens` and `query_start_loc` from `capacities +
  num_non_draft_tokens` (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:455-468`);
- `num_logits = total_num_logits` (`num_reqs · N + total draft tokens`), the length of the fresh
  `logits_indices`; `num_new_sampled_tokens = model_state.num_new_sampled_tokens_per_step`: `1`
  by default (`vllm/v1/worker/gpu/model_states/interface.py:254`), `0` for models that sample no
  bonus token (`vllm/model_executor/models/diffusion_gemma.py:1122`).

A request's draft count is at most `num_speculative_steps`, so `nl ≤ S + N ≤ BLOCK_SIZE`. The
batch mixes decoding requests (query `nd + N` rows, past their prefill), prefill chunks
(`seq_lens ≤ prefill_len`), chunks whose first logit is a prompt position, and requests with more
query rows than logits. The sizes: `num_speculative_steps` `0` (no speculative decoding), `1`
(MTP), `3` to `7` (EAGLE), up to `8` or more (n-gram); `num_reqs` up to `max_num_seqs` (256 to
1024 by default); `max_num_batched_tokens` from 2048 to 16384 and beyond.

The claim's `setup_*` instances: `setup_decode_b256`, `setup_decode_b1024`, `setup_single`
(`S = 0`, `N = 1`, `BLOCK_SIZE = 1`), `setup_mtp1` (`S = 1`, `BLOCK_SIZE = 2`), `setup_eagle3`
(`S = 3`, `4`), `setup_eagle7` (`S = 7`, `8`), `setup_ngram8` (`S = 8`, `16`), `setup_spec128`
(`S = 128`, `256`), `setup_n0_s4` (`N = 0`, `S = 4`, `4`), `setup_n0_s7` (`N = 0`, `S = 7`, `8`),
`setup_large` (8192 requests, `S = 3`, a `2^20`-token budget) and `setup_empty` (`num_reqs = 0`).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 21 launches: the 12 setup launches through the host, on
batches mixing decoding requests, prefill chunks, prompt-tail chunks and padded query slots, with
`int64` tokens carrying high bits; and 9 guarded direct launches at the edges (`BLOCK_SIZE = 64`
above `next_power_of_2(S + N)`, `nl = BLOCK_SIZE` for every request, repeated request-state rows,
`N = 0` with requests of no logit, stride `1` (`S = 1`), `S = 0`, negative and maximal
`prefill_len` entries with `seq_lens` near `2^31`, `seq_lens = 0`, an empty batch). All 1,844,783
output cells are exactly as `Post` states, every input and every guard byte is unchanged, and 84
of 84 wrong outputs (prefill requests treated as decoding, the prompt-tail test ignored, the last
sampled token dropped, logits indices shifted by one, the tokens not truncated, drafts read from
another row) fail `Post`.
