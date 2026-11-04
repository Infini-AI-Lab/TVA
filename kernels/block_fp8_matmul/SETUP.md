# Setup of `block_fp8_matmul`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from vLLM
v0.33.0 (commit `4ba787ec99`), `vllm/model_executor/layers/quantization/utils/fp8_utils.py:841`
`_w8a8_triton_block_scaled_mm`.

## The source

`source/block_fp8_matmul.py` holds:
- `_w8a8_triton_block_scaled_mm`, the `@triton.jit` kernel, verbatim (`fp8_utils.py:840-921`): one
  program per `BLOCK_SIZE_M × BLOCK_SIZE_N` tile of `C`, the tiles grouped along M by
  `GROUP_SIZE_M`; along K, each `BLOCK_SIZE_K` slice's `tl.dot` of the `e4m3` tiles is multiplied
  by the slice's activation scales (one per row) and weight scales (one per column) and added to a
  `float32` accumulator, which is cast to `C`'s dtype and stored;
- `_upcast_e8m0_to_fp32`, a torch helper of the wrapper, verbatim (`fp8_utils.py:1157-1166`);
- `w8a8_triton_block_scaled_mm`, the host, from vLLM's wrapper of the same name
  (`fp8_utils.py:961-1093`);
- `make_inputs` and `reference` (the dequantised product in `float64`), and the test under
  `if __name__ == "__main__":`.

Changes from vLLM's source (each listed with its `file:line` in the file's docstring): the wrapper's
ROCm `gfx1250` branch, a torch fallback that launches no kernel, is removed; the tuned-config lookup
is removed and the wrapper's default configuration always used; the grid function is computed
directly. The kernel's lines are unchanged.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. vLLM's wrapper takes a tuned
configuration from a JSON file named by `(N, K, device, block_size)` when one exists
(`get_w8a8_block_fp8_configs`, `fp8_utils.py:924-958`), and otherwise, or under
`VLLM_BATCH_INVARIANT`, its default configuration (`fp8_utils.py:1052-1062`). The host always takes
the default configuration:
- `BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = block_size[0] = 128`, `BLOCK_SIZE_K = block_size[1] = 128`,
  `GROUP_SIZE_M = 32`;
- `num_warps = 4`, `num_stages = 2`;
- grid `(cdiv(M, 64) · cdiv(N, 128),)`;
- runtime `group_n = block_size[0] = 128`, `group_k = block_size[1] = 128`, and the strides as the
  wrapper passes them: `A.stride(-2), A.stride(-1)`, `B.stride(1), B.stride(0)` (so `stride_bk` is
  1 and `stride_bn` is K), `C.stride(-2), C.stride(-1)`, `As.stride(-2), As.stride(-1)`,
  `Bs.stride(1), Bs.stride(0)`.

## The launches

The claim should cover the host's launches with block size `[128, 128]` (DeepSeek-V3's format):
- `A` `float8_e4m3fn` (`tl.float8e4nv`) `[M, K]` contiguous (the wrapper accepts leading dimensions
  `[..., K]`, with `M` their product); `B` `float8_e4m3fn` `[N, K]` contiguous; `As` `float32`
  `[M, cdiv(K, 128)]`, row-major as vLLM's `QuantFP8` makes it for this path
  (`model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:68-73`; the wrapper also
  accepts other strides); `Bs` `float32` `[cdiv(N, 128), cdiv(K, 128)]` contiguous; `C` `bfloat16`,
  the output dtype vLLM's model layers pass (`out_dtype` of the layer; the constexpr branches for a
  `float16` or `float32` `C` stay in the source);
- shapes as variables: `M ≥ 1` (the batch's tokens; the wrapper sets no upper limit), `N ≥ 1`,
  `K ≥ 1`, each a multiple of 128 or not. vLLM's test (`tests/kernels/quantization/test_block_fp8.py:53-58`)
  runs `M ∈ {1, 7, 8, 83, 4096}`, `N ∈ {128, 512, 576, 7168, 13824}`, `K ∈ {256, 3884, 4096, 13824,
  16384}`; DeepSeek-V3 (hidden size 7168, intermediate size 18432, so `N = 4608` at TP8) runs
  `(N, K)` pairs such as those of the H200 configuration files the code base ships
  (`fp8_utils.py`'s `configs/` directory: `N` from 512 to 36864, `K` from 256 to 18432, e.g.
  `(4608, 7168)`, `(7168, 2304)`, `(24576, 1536)`, `(2112, 7168)`);
- scales finite and positive (vLLM's test draws them in `[0, 0.01)`; real scales are a block's
  absolute maximum over 448).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes on nine shapes
`(M, N, K)`: `(1, 128, 256)`, `(7, 576, 3884)`, `(8, 512, 4096)`, `(83, 7168, 256)`,
`(83, 13824, 16384)`, `(100, 2112, 7168)`, `(257, 4608, 7168)`, `(4096, 7168, 2304)` and
`(4096, 4608, 7168)`, with inputs drawn as vLLM's test draws them. Against the dequantised product
(in `float64`, or `float32` for the three cases with `M·N·K > 2^34`), the mean relative error is 1.4e-3 in every case (the
`bfloat16` rounding of the output), and the largest error over `|ref| + mean |ref|` is at most
3.4e-3.
