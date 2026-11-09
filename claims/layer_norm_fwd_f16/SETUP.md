# Setup of `layer_norm_fwd_f16`

From Triton 3.8.0's tutorial 05, `python/tutorials/05-layer-norm.py`: the forward kernel
`_layer_norm_fwd_fused` (lines 48–95) and its host `LayerNorm.forward` (lines 235–260).

## The source

`source/05-layer-norm.py` is the tutorial file, unchanged: the `@triton.jit` forward kernel
`_layer_norm_fwd_fused` (lines 48–95), the backward kernels `_layer_norm_bwd_dx_fused` and
`_layer_norm_bwd_dwdb` (lines 132–221, not this claim's kernels), the autograd function
`LayerNorm` (lines 233–294) and `layer_norm = LayerNorm.apply` (line 297), the test
`test_layer_norm` (lines 300–323), the benchmark `bench_layer_norm` (lines 326–371), and, at the
module's top level, a run of the test and of the benchmark (lines 374–375). One program normalises
one row: it loops over the row in blocks of `BLOCK_SIZE` three times (the mean, the variance, the
output), stores the row's `mean` and `rstd = 1/sqrt(var + eps)`, and writes
`y = (x − mean)·rstd·w + b`.

## The config

Launched directly: the forward kernel has no `@triton.autotune` and no `@triton.heuristics`.
`LayerNorm.forward` allocates `y = torch.empty_like(x)`, `mean` and `rstd` (`float32`, `[M]`),
sets `BLOCK_SIZE = min(65536 // x.element_size(), next_power_of_2(N))` (`min(32768, …)` for
`float16`), raises `RuntimeError` when `N > BLOCK_SIZE`, sets
`num_warps = min(max(BLOCK_SIZE // 256, 1), 8)`, and launches the grid `(M,)` with
`(x_arg, y, weight, bias, mean, rstd, x_arg.stride(0), N, eps)`, `BLOCK_SIZE`, `num_warps` and
`num_ctas=1` (lines 252–255).

## The launches

Every forward launch the file makes, through `layer_norm(x, w_shape, weight, bias, eps)`:
`x` a fresh contiguous `float16` tensor `[M, N]` (`-2.3 + 0.5·randn`), `weight` and `bias` fresh
`float16` tensors `[N]` (`rand`), `eps = 1e-5`:

- **The test** (`test_layer_norm(1151, 8192, torch.float16)`, line 374): `M = 1151`, `N = 8192`
  (`BLOCK_SIZE = 8192`, `num_warps = 8`).
- **The benchmark** (`bench_layer_norm.run`, line 375; `x_vals`, line 329, and `args`, line 336):
  `M = 4096`, `N = 512·i` for `i = 2 … 31` (`1024 … 15872`), `float16`, `mode = 'backward'`: the
  `triton` provider runs the forward once (`y = y_fwd()`, line 367), which is this kernel's launch;
  the timed part is the backward (`_layer_norm_bwd_dx_fused`, `_layer_norm_bwd_dwdb`), which is not
  part of the setup. The `torch` and `apex` providers do not launch the kernel.

Also covered, as the host allows them: every `N` from `2` to `32768` (`BLOCK_SIZE` the next power of
two), any `M ≥ 1` with `(M − 1)·N < 2^31`, any `eps` in `[2^-125, 2^126]`, any finite inputs whose
output has room in `float16` (README, Conditions).

Not part of the setup: the backward kernels, other dtypes, `N = 1`.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, torch, an H200) runs the test's launch and all 30 benchmark forward
launches with the file's input distributions, and compares `mean`, `rstd` and `y` with the float64
reference on every row. All are within the claim's tolerances: the largest error / tolerance on
these launches is `0.499` for `y` (the output's `float16` rounding, against a tolerance of twice that
rounding), `0.056` for `rstd` and `0.092` for `mean`; over the edge launches of README, Checks, it is
`0.499`, `0.060` and `0.101`. Every element of `y`, `mean` and `rstd` is written and finite, the
guard bands around them are unchanged, and the inputs are unchanged. Wrong outputs at the test's
shape and at `N = 15872` (a mean or a variance with one term dropped, the variance divided by
`N − 1`, an all-zero `y`, shifted weights) all fail `Post`. The module's own
`test_layer_norm(1151, 8192, torch.float16)` passes at import.
