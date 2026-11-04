# Setup of `merge_attn_states_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (`sandbox/TESTSET.md`, kernel
13), from vLLM v0.33.0 (commit `4ba787ec99`),
`vllm/v1/attention/ops/triton_merge_attn_states.py:73` `merge_attn_states_kernel`.

## The source

`source/merge_attn_states.py` holds:
- `merge_attn_states_kernel` (lines 65-216), the `@triton.jit` kernel, verbatim from the source's
  lines 72-223 except for its parameters' defaults: one program per (token, head) merges the two partial outputs by their
  log-sum-exps, or, for a token at or after `prefill_tokens_with_context`, copies the suffix's
  output and lse;
- `merge_attn_states`, the host wrapper, from the source's `merge_attn_states` (lines 14-69),
  with the checks of the dispatcher that calls it (`vllm/v1/attention/ops/merge_attn_states.py:113-128`);
- `reference`, the merge in `float64`, and the test under `if __name__ == "__main__":`.

What changed from the code base's source (the module docstring lists each with its line): the
vLLM imports are removed; the defaults of the kernel's parameters `FP8_MIN` and `FP8_MAX`
(`= float8_info.min`, `= float8_info.max`, source lines 94-95) are removed; the module global
`float8_info` (source line 9) is a local of the host, with `current_platform.fp8_dtype()` =
`torch.float8_e4m3fn`, and the host passes `FP8_MIN = -448.0` and `FP8_MAX = 448.0`, the values the
defaults gave; the host allocates `output` and `output_lse` fresh and
returns them, where the source's wrapper writes into tensors the caller passes. The kernel's
body is unchanged.

## The config

No autotuning and no heuristics: the wrapper launches the kernel with grid
`(num_tokens, num_heads)` and Triton's default launch options (`num_warps = 4`,
`num_stages = 3`). The compile-time values are `HEAD_SIZE = head_size`,
`PADDED_HEAD_SIZE = next_power_of_2(head_size)`, `OUTPUT_LSE = output_lse is not None`,
`USE_FP8 = output_scale is not None`, `FP8_MIN = -448.0`, `FP8_MAX = 448.0` (the min and max of
`float8_e4m3fn`). The parameters' defaults removed: TVA's subset has no kernel parameter
defaults; the host passes each.

The proved path: `OUTPUT_LSE = True` (the host's `return_lse=True`, as the code base's test and
its prefill callers use it) and `USE_FP8 = False` (no `output_scale`: a `bfloat16` output).

## The launches

The claim should cover the wrapper's launches with, as variables:
- `prefix_output`, `suffix_output`: `bfloat16`, `[num_tokens, num_heads, head_size]`, last
  dimension contiguous, one head stride for both (`prefix_output.stride(1)`, at least
  `head_size`: the backends may pad the heads), and token stride `num_heads` times it;
- `prefix_lse`, `suffix_lse`: `float32`, `[num_heads, num_tokens]`, any strides (the backends pass
  transposes of `[num_tokens, num_heads]` tensors), holding finite values, `-inf` (an empty
  side) and `+inf` (FlashAttention-2's empty side, read as `-inf`);
- `prefill_tokens_with_context`: `None` (all tokens merged) or an integer in `[0, num_tokens]`;
- the code base's shapes (`tests/kernels/attention/test_merge_attn_states.py:154-156`): tokens
  256, 512, 613, 1024, 1536 and 4096 (6 in the both-empty test), heads 4, 8, 16, 32, 48 and 64,
  head sizes 32, 48, 64, 96, 128 and 256, `prefill_tokens_with_context` `None` or 128.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes on 9 cases (1 to 4096
tokens, 4 to 64 heads, head sizes 32 to 256, `prefill_tokens_with_context` `None`, 0, 1 and 128,
padded head strides, transposed lse views, 10% `+inf` and 5% both-empty entries): the output's
largest error relative to the `float64` reference is 3.9e-3 (within one `bfloat16` rounding,
2^-8), the lse's largest absolute error 1.9e-7, the infinite lse entries are those of the
reference, no output is NaN, and the launch without the lse gives the same output.
