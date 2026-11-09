# Setup of `moe_sum_reduce_bf16`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py:1240`,
the function `_moe_sum_reduce_kernel`.

## The source

`source/moe_sum_reduce.py` holds:
- `_moe_sum_reduce_kernel`, the `@triton.jit` kernel, copied verbatim but for its
  parameters' defaults: each program sums, for
  `BLOCK_M` tokens and `BLOCK_DIM` hidden columns, the token's `topk_num` rows in `float32`,
  multiplies by `routed_scaling_factor` and stores in the input's dtype;
- `moe_sum_reduce`, the host function, adapted from `moe_sum_reduce_triton` (`:1296`);
- `reference`, a test helper (the sum and scale in `float64`);
- the test under `if __name__ == "__main__":`.

The host differs from the wrapper in allocating the output itself (`[token_num, hidden_dim]` in
the input's dtype, the shape the wrapper asserts) and returning it, in launching without
programmatic dependent launch, and in passing `num_stages` explicitly; the kernel's parameter
default (`USE_GDC = False`) is removed (the host passes it); `reference` and the test are new. The
module docstring lists the changes with their source lines.

## The config

No decorator was removed: the kernel has no `@triton.autotune` or `@triton.heuristics`. The
parameters' defaults are removed (`USE_GDC = False`, `fused_moe_triton_kernels.py:1255`): TVA's
subset has no kernel parameter defaults; the host passes each. The configuration is the wrapper's own (`fused_moe_triton_kernels.py:1305-1308`): `BLOCK_M = 1`,
`BLOCK_DIM = 2048`, `NUM_STAGE = 1` (the `tl.range` loop's `num_stages`), `num_warps = 16`, grid
`(cdiv(token_num, 1), cdiv(hidden_dim, 2048))`, the strides from `input.stride()` and
`output.stride()`, and `routed_scaling_factor` as a compile-time value. The flag: `USE_GDC=False`
and no `launch_pdl`; the wrapper passes `USE_GDC=True, launch_pdl=True` when
`is_arch_support_pdl()` holds (`:1315`), and programmatic dependent launch is outside TVA's
launches. The launch option `num_stages=3` is Triton 3.8.0's default, which the wrapper's launch
gets.

## The launches

The claim should cover the wrapper's launches, with shapes as variables:
- `bfloat16` input and output, both contiguous;
- `USE_GDC=False`;
- `token_num` from 1 up: the Triton MoE runner calls it for at most 32 tokens on the H200
  (`srt/layers/moe/moe_runner/triton_utils/fused_moe.py:1005-1012`, with
  `_use_moe_sum_reduce_torch_compile`, `:159`), and the Marlin and LoRA runners for any batch
  (`srt/lora/lora_moe_runner_marlin.py:222`, `srt/lora/marlin_lora_temp/moe_runner.py:857`);
- `topk_num` 1 to 9 (top-8 plus a fused shared expert; the code base's benchmark uses 9,
  `kernels/aot/benchmark/bench_sum_scale.py:135`), `hidden_dim` the model's hidden size (2048 to
  7168: Qwen3-MoE 2048 and 4096, DeepSeek-V3 7168, the benchmark's 4096), not necessarily a
  multiple of 2048;
- `routed_scaling_factor` any float (1.0 when the caller passes none, `fused_moe.py:985-986`;
  DeepSeek-V3's 2.5; the benchmark's 0.3).

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host on 25 cases: the host's nine test launches
(`(token_num, topk_num, hidden_dim, scale)` from `(1, 8, 7168, 2.5)` to `(8192, 8, 7168, 2.5)`,
`(1, 1, 1, 1.0)` included) and edges: cancellation, sums near `2^126`, `bfloat16` subnormal inputs,
scales `1e-40`, `0`, `−0`, `−7.75`, `1e30` and `1e39`, NaN and infinities in some elements, top-1 and
top-64, hidden 1 and 2049, and `2^15` tokens; each case is also launched with other block sizes
(`(4, 256)`, `(2, 1024)`, `(8, 64)`, `(1, 4096)`). Every element in range is finite and within the
claim's tolerance (worst error / tolerance `0.9961`, one `bfloat16` rounding), the elements out of
range are counted (for example 9218 in the near-overflow case with scale 3), the input is unchanged
bit for bit, and the guard bands around the output keep their bytes.
