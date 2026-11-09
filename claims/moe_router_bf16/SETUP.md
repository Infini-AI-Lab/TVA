# Setup of `moe_router_bf16`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/moe/router.py:13`, the function
`fused_moe_router_cudacore_kernel`.

## The source

`source/moe_router.py` holds:
- `fused_moe_router_cudacore_kernel` (lines 51-152), the `@triton.jit` kernel, verbatim from the
  source's lines 12-113: one program per token computes the router logits (the token's hidden
  state times each expert's gate weights, an elementwise product summed in `float32`), the soft
  cap, the correction bias, and the `topk` experts by repeated `tl.argmax` with their softmax
  weights;
- `fused_moe_router_cudacore`, the host wrapper, copied from the source's lines 116-155;
- `reference`, the logits and their softmax in `float64`, and the test under
  `if __name__ == "__main__":`.

Changes from SGLang's source (the module docstring lists each with its line): the
module's `_is_hip = is_hip()` is a local of the host, `False` (an H200), so the file has no
module-level assignments; and `typing.Optional` is written with `|`.
The kernel's body and the wrapper's code are otherwise unchanged. The tensor-core router and the
dispatcher `fused_moe_router_shim` (source lines 338-387) are not copied. `reference` and the test
under `__main__` are new.

## The config

No autotuning and no heuristics: the wrapper picks the launch itself (source lines 132-138),
`BLOCK_SIZE = next_power_of_2(hidden_dim)` and
`num_warps = max(min(next_power_of_2(cdiv(hidden_dim, 256)), 32), 4)` (32 with `_is_hip` false),
grid `(bs,)`, Triton's default `num_stages`. The kernel declares no parameter defaults; the
host passes each parameter. It passes `moe_renormalize = False` always.

The path the claim covers is the one SGLang runs: its only GPU caller is Grok
(`python/sglang/srt/models/grok.py:139-160`), through `fused_moe_router_shim`, which takes this
kernel when `not ((bs >= 512 or num_experts > 8) and hidden_dim % BLOCK_SIZE_K == 0 and not
enable_deterministic_inference)`, so for Grok's 8 experts when `bs < 512`. Its flags:
`topk = 2`, `moe_softcapping = 30.0` (Grok's `router_logit_softcapping`), no correction bias
(`is_correction_bias = False`), `num_experts = 8`. For Grok-1 (`hidden_dim = 6144`) and Grok-2
(`8192`), `BLOCK_SIZE = 8192` and `num_warps = 32`.

## The launches

The claim should cover the wrapper's launches with:
- `x`: `bfloat16`, `[bs, hidden_dim]`, contiguous (the hidden states);
- `router_weight`: `float32`, `[num_experts, hidden_dim]`, contiguous (Grok's gate is created
  with `params_dtype=torch.float32`);
- `topk_weights` (`float32`) and `topk_ids` (`int32`), `[bs, topk]`, allocated by the wrapper;
- the compile-time values `num_experts = 8`, `topk = 2`, `moe_softcapping = 30.0`,
  `is_correction_bias = False`, `moe_renormalize = False`, and `hidden_dim` (Grok's 6144 and
  8192), with `BLOCK_SIZE` and `num_warps` as above;
- `bs` from 1 to 511 (the cudacore path's range for 8 experts).

Not part of the setup: the other flags (no cap, a correction bias, `topk` 1 or above 2, other powers
of two of experts), which stay in the source, and so the five cases of the host's test taken from
the CPU router's test grid; that test (`test/registered/cpu/test_topk.py:412-427`)
runs 17 or 128 tokens, hidden sizes 64 and 257, 8, 16 or 32 experts, top-1, 2 or 4, caps 0 and
30, with and without a bias.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host on 27 launches with Grok's compile-time values:
the source's eight Grok test cases (`bs` 1, 7, 128 and 511, hidden 6144 and 8192), `hidden_dim` 1, 3,
5, 64, 257, 4097, 8191 and 1024, `bs` 2048, 32,768 and 262,144, all-zero inputs (every expert tied),
two identical gate rows, a logit near `1300` and one near `−98,000`, `x` at `2^-126` with gate
weights near `2^120`, hidden states scaled by 64 (logits up to 86), and `bs = 511` at hidden 6144
and 8192 with the test's data and with logits of standard deviation 3. Every launch meets the
claim: the worst error / tolerance is `3.2·10^-2` for the capped logits, `0.017` and `0.036` for the
two weights; the ids are the first argmaxes on every token; the inputs are unchanged and the guard
bands around both outputs keep their bytes. On the last four launches a kernel without the soft cap
and swapped ids fail `Post` (README, Tolerances).
