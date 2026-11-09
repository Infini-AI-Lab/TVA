# Setup of `merge_attn_states_bf16`

From vLLM v0.33.0 (commit `4ba787ec99`), `vllm/v1/attention/ops/triton_merge_attn_states.py:73`, the
function `merge_attn_states_kernel`.

## The source

`source/merge_attn_states.py` holds:
- `merge_attn_states_kernel` (lines 65-216), the `@triton.jit` kernel, verbatim from the source's
  lines 72-223 except for its parameters' defaults: one program per (token, head) merges the two partial outputs by their
  log-sum-exps, or, for a token at or after `prefill_tokens_with_context`, copies the suffix's
  output and lse;
- `merge_attn_states`, the host wrapper, from the source's `merge_attn_states` (lines 14-69),
  with the checks of the dispatcher that calls it (`vllm/v1/attention/ops/merge_attn_states.py:113-128`);
- `reference`, the merge in `float64`, and the test under `if __name__ == "__main__":`.

Changes from vLLM's source (the module docstring lists each with its line): the
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

The paths the claim covers: `OUTPUT_LSE` `True` or `False` (the host's `return_lse`: an
`output_lse` tensor, or `None` as the backends' calls pass it) and `USE_FP8 = False` (no
`output_scale`: a `bfloat16` output). The fp8 output (`USE_FP8 = True`, an `output_scale`) is not
part of the setup.

## The launches

The claim should cover the wrapper's launches with, as variables:
- `return_lse` `True` or `False`: the code base's callers mostly pass no `output_lse`
  (`vllm/v1/attention/backends/flash_attn.py:1636`, `vllm/v1/attention/backends/flashinfer.py:544`,
  `vllm/model_executor/layers/attention/mla_attention.py:3297`,
  `vllm/model_executor/layers/attention/sparse_mla_attention.py:1245`), so `OUTPUT_LSE = False`;
  the code base's test and the host's test also run it with `output_lse`;
- `prefix_output`, `suffix_output`: `bfloat16`, `[num_tokens, num_heads, head_size]`, last
  dimension contiguous, one head stride for both (`prefix_output.stride(1)`, at least
  `head_size`: the backends may pad the heads), and token stride `num_heads` times it;
- `prefix_lse`, `suffix_lse`: `float32`, `[num_heads, num_tokens]`, any strides (the backends pass
  transposes of `[num_tokens, num_heads]` tensors), holding finite values, `-inf` (an empty
  side) and `+inf` (FlashAttention-2's empty side, read as `-inf`);
- `prefill_tokens_with_context`: `None` (all tokens merged) or an integer in `[0, num_tokens]`;
- the code base's shapes (`tests/kernels/attention/test_merge_attn_states.py:154-156`): tokens
  256, 512, 613, 1024, 1536 and 4096 (6 in the both-empty test), heads 4, 8, 16, 32, 48 and 64,
  head sizes 32, 48, 64, 96, 128 and 256, `prefill_tokens_with_context` `None` or 128;
- `output` and `output_lse` fresh, sharing no byte with the inputs. The in-place merge of
  `vllm/model_executor/layers/attention/mla_attention.py:2882` (`output=prefix_output`,
  `output_lse=prefix_lse`) is not part of the setup: there each program reads and then writes the
  same bytes, and a program that reads and then writes the same bytes is out of scope.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host, each launch with `return_lse` `True` and
`False`, on its nine test cases (1 to 4096 tokens, 4 to 64
heads, head sizes 32 to 256, `prefill_tokens_with_context` `None`, 0, 1 and 128, padded head
strides, transposed lse views), on vLLM's test's largest shape (4096 tokens, 64 heads of 256,
`prefill_tokens_with_context = 128`), on four edges of the conditions (one token, one head, head size
1, head size 3 padded by 5, 64 heads of 256 padded by 64) and on four edges of the inputs (lse of
every bit pattern but NaN, huge, subnormal and infinite; `bfloat16` inputs of every finite pattern;
exact cancellation; both lse at the `float32` maximum; any bits, NaN included, in the copied tokens
and in both-empty rows). Every output is within the claim's tolerance: the worst output error /
tolerance is `0.978` (the bound is one `bfloat16` rounding, margin `1.02×`), the worst lse error /
tolerance `6.7·10^-3` (margin `149×`). On every case a both-empty pair gives lse `−∞` and output
`+0`, the copied tokens are the suffix's bit for bit, and every input is unchanged; without `output_lse` the
output is the same. Seven wrong outputs (all zero, the suffix's output alone, the prefix's output
alone, an unweighted average, swapped weights, the lse's log term dropped, zeroed copied tokens)
are rejected.
