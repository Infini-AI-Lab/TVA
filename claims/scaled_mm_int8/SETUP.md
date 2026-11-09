# Setup of `scaled_mm_int8`

From vLLM v0.33.0 (commit `4ba787ec99`),
`vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py:21-172`, the kernel
`scaled_mm_kernel`, launched by its host `triton_scaled_mm` (`triton_scaled_mm.py:175-284`) on the
`int8` path: the compressed-tensors W8A8 `int8` linear layer when its Triton kernel is used.

## The source

`source/triton_scaled_mm.py` holds:
- `scaled_mm_kernel` (lines 55-206 here, `triton_scaled_mm.py:21-172` upstream), the
  `@triton.jit` kernel, verbatim but for the defaults of its last two parameters (below). A program
  computes one `BLOCK_SIZE_M × BLOCK_SIZE_N` tile of `C`: it runs the `K` loop of masked `int8`
  loads and `tl.dot`s into an `int32` accumulator, converts it to `float32`, multiplies by
  `scale_a` (one value per row, or one for all) and by `scale_b` (one per column, or one for all),
  adds the bias if a bias pointer is passed, converts to the output's dtype and stores the tile;
  the tensor-descriptor branches (`USE_TD`) stay in the source;
- `is_weak_contiguous` (lines 47-52) and `triton_scaled_mm` (lines 211-302), the host: its
  assertions, the output `torch.empty((M, N), dtype=out_dtype)`, the tile heuristic, the scales'
  block sizes, the accumulator dtype and the launch;
- `reference` (the result in `float64`) and a test under `if __name__ == "__main__":`.

Changes from vLLM's source (each with its upstream line in the file's docstring):
- the imports `from vllm.triton_utils import tl, triton, use_tensor_descriptor` and
  `from vllm.triton_utils.allocation import set_triton_allocator` become `import triton` and
  `import triton.language as tl` (`vllm.triton_utils` re-exports Triton's modules); the module-level
  set `_TD_ALLOCATOR_DEVICES` (upstream line 10), used only by the host's descriptor path, is removed;
- the kernel's parameters `USE_TD: tl.constexpr = False` and `B_T: tl.constexpr = False`
  (upstream lines 44-45) lose their defaults (TVA's subset has no kernel parameter defaults); the
  host passes both, as upstream; the kernel's other lines are unchanged;
- the host's parameter `use_td` is removed and `use_td = False` (upstream lines 188, 237-254):
  `use_tensor_descriptor(None)` (`vllm/triton_utils/tensor_descriptor.py:5-14`) is
  `current_platform.is_xpu()` when `VLLM_TRITON_USE_TD` is unset, so `False` on CUDA, and no caller
  of the W8A8 path passes `use_td`; the allocator set-up for descriptors goes with it; `b_t` is still
  computed and passed as `B_T`;
- `reference` and the test are new (after vLLM's `tests/kernels/quantization/test_triton_scaled_mm.py`).

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`. The host chooses the tile itself
(lines 249-263, `use_heuristic=True`, the default every caller uses), from `M` and `N`:

| `M` | `N` | `(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)` |
|---|---|---|
| `≤ 32` | `< 8192` | `(64, 64, 256)` |
| `≤ 32` | `≥ 8192` | `(64, 128, 256)` |
| `33 … 64` | any | `(64, 64, 256)` |
| `65 … 128` | any | `(64, 128, 128)` |
| `> 128` | any | `(128, 128, 128)` |

With `use_heuristic=False` and no block sizes (vLLM's rounding test), the tile is the host's
defaults `(32, 32, 32)`. The other constexprs, as the host sets them on the `int8` path:
- `BLOCK_SIZE_SCALE_A = 1` when `scale_a` is `[1, 1]` (after the host's `reshape(-1, 1)` of a 0-d or
  1-d scale), `BLOCK_SIZE_M` otherwise (`[M, 1]`); `BLOCK_SIZE_SCALE_B` likewise with
  `BLOCK_SIZE_N` (lines 265-266);
- `ACCUMULATOR_DTYPE = tl.int32` (line 268: the inputs are not floating point);
- `USE_TD = False` (line 272); `B_T = (weight.stride(1) != 1 and weight.stride(0) == 1)`, which the
  kernel reads only under `USE_TD`.

The launch (lines 276-300): grid `(cdiv(M, BLOCK_SIZE_M) · cdiv(N, BLOCK_SIZE_N),)` (the grid
function of lines 241-243), no `num_warps` or `num_stages` (Triton's defaults, `4` and `3`), no
other launch option.

## The launches

The claim covers the host's launches as vLLM's `int8` W8A8 path makes them:
- **Callers.** `TritonInt8ScaledMMLinearKernel.apply_weights`
  (`vllm/model_executor/kernels/linear/scaled_mm/triton.py:123-156`) through the custom op
  `int8_scaled_mm_func` (`triton.py:223-258`), the compressed-tensors W8A8 `int8` linear
  (`CompressedTensorsW8A8Int8`) when this kernel is selected (`_POSSIBLE_INT8_KERNELS`,
  `vllm/model_executor/kernels/linear/__init__.py:446-454`: on CUDA after Cutlass, e.g. with
  `VLLM_DISABLED_KERNELS=CutlassInt8ScaledMMLinearKernel`; first on ROCm after AITER and on XPU); and
  `cutlass_scaled_mm` (`vllm/_custom_ops.py:784-830`), which calls it when the weight's sizes are not
  multiples of 16. With asymmetric activations the zero-point correction is applied after the kernel
  (`triton.py:138-154`); the kernel's launch is the same.
- **Inputs.** `A` (`x_q`): `int8` `[M, K]`, contiguous, from `ops.scaled_int8_quant`
  (`vllm/_custom_ops.py:1908-1962`), any `int8` values; `scale_a`: `float32`, `[M, 1]` per token
  (dynamic quantisation) or a scalar (static, `[1, 1]` after the host's reshape); `B` (`w_q`):
  `int8` `[K, N]`, the transposed view `w_q.t()` of the checkpoint's contiguous `[N, K]` weight
  (strides `(1, K)`, `triton.py:42-50`), or a contiguous `[K, N]` (strides `(N, 1)`, vLLM's test);
  `scale_b`: `float32`, `[N, 1]` per channel (fused modules with per-tensor scales are converted to
  per channel, `triton.py:56-64`) or a scalar; `bias`: `None` or `[N]` of the output's dtype (the
  layer's bias); `out_dtype`: the activations' dtype, `bfloat16` or `float16`; `C` the fresh
  contiguous `[M, N]` the host allocates. Scales are finite (`float32` quantisation scales).
- **Shapes.** `M` the batch's tokens, `1 … 16384` (vLLM's `int8` GEMM benchmark,
  `benchmarks/kernels/benchmark_int8_gemm.py:96`); `(K, N)` the Llama-class layers of
  `benchmarks/kernels/weight_shapes.py:16-63` (Mistral-7B, Llama-2-7B, -13B, -70B, Llama-3-8B,
  Llama-3.1-8B, -405B, Llama-3.3-70B) and the W8A8 test models of
  `tests/quantization/test_compressed_tensors.py:67-79` (with their model configurations: TinyLlama-1.1B: hidden 2048, intermediate
  5632, qkv 2560; Llama-3.2-1B: hidden 2048, intermediate 8192, qkv 3072), at tensor-parallel degrees
  1, 2, 4 and 8 (`K` from 256 to 53248, `N` from 320 to 106496); vLLM's kernel tests
  (`tests/kernels/quantization/test_triton_scaled_mm.py`): `(M, N, K)` in `(1, 256, 128)`,
  `(33, 256, 496)`, `(64, 971, 1024)`, `(64, 20486, 128)`, `(512, 256, 496)`, `(512, 20486, 1024)`
  with scalar or vector scales and with or without a bias (lines 66-123), the descriptor test's plain
  launches `(1, 4096, 4096)`, `(64, 4096, 4096)`, `(256, 2048, 4096)` (lines 150-178), and the rounding
  test `(1, 16, 32)` with and without the heuristic (lines 126-142).

Of these, the claim covers every launch whose `K` is a multiple of the tile's `BLOCK_SIZE_K`
(`Inst.ok.K_div`): every Llama-class layer with `K` a multiple of 256, for every `M`; the layers with
`K` a multiple of 128 but not of 256 (Llama-2-7B's down projection at TP2, `K = 5504`; Llama-2-13B's
down projection at TP4, `K = 3456`, and its attention output at TP8, `K = 640`; TinyLlama's down
projection at TP4, `K = 1408`) for `M > 64`; the kernel tests
`(64, 971, 1024)` and `(512, 20486, 1024)`; the three plain descriptor-test launches; and the
rounding test with `use_heuristic=False`. The launches with a partial last `K` block (the other
kernel tests, the rounding test with the heuristic, those layers at `M ≤ 64`, and Llama-2-7B's down
projection at TP4 and TP8, `K = 2752`, `1376`, Llama-2-13B's at TP8, `K = 1728`, TinyLlama's at TP8,
`K = 704`, for every `M`) are not part of the setup: there the kernel's result depends on values the
library leaves unspecified (a kernel limit, README, Exclusions). The tensor-descriptor path (`use_td=True`, XPU's default) and the `fp8` path of
the same kernel are not part of the setup either.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) runs the host on 94 covered launches and
two guarded direct launches: vLLM's two covered test shapes with every flag and both output dtypes
(32 launches, the test's draws), the three plain descriptor-test launches (12), the rounding test
with `use_heuristic=False` (the bias added before the rounding: `1 + 2^-7` exactly), eight
Llama-class layers with the full `int8` range, four flag sets each (32: Llama-3-8B's four layers,
its down projection at TP8, Llama-3.2-1B's gate-up, Llama-2-70B's down, Llama-3.1-405B's down at TP8;
`M` from 1 to 4096; `bfloat16` and `float16`), Llama-2-7B's down at TP2 (`K = 5504`, `M = 128`,
`2000`), the largest `K` and `N` (Llama-3.1-405B's down at TP1, `K = 53248`; its gate-up at TP2,
`N = 53248`), and edges (`K = 2^17 − 256` with every value `−128`, the largest accumulator;
`M = N = 1`; `M`, `N` not multiples of the tile; the scales multiplied by `2^20` and `2^10`, or `scale_a` by `2^-60`; a bias
cancelling the product). Every element meets both clauses of `Post`: the largest error / tolerance
is 0.4998 (vLLM's test, `M = 512`, `N = 20486`, `float16`). Four wrong results (the scaled product rounded before the bias
is added, on the rounding test's data; the last `BLOCK_SIZE_K` products left out; the next row's
scale; the bias left out) each break `Post`. The inputs are unchanged and, in the two direct
launches (padded rows, column-major), no byte outside the matrix `C` is written. Outside the
claim, the three kernel-test shapes with a partial last `K` block also match the reference on the
device (largest relative error `0.0039`), where Triton 3.8.0 happens to load `0` on the masked lanes. The peak device memory of the run is 2.5 GiB.
