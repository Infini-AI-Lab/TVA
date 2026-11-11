# Setup of `l2norm_fwd_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/modules/norm/l2norm.py:69`,
the function `l2norm_fwd_kernel`.

## The source

`source/l2norm.py` holds:
- `l2norm_fwd_kernel`, the `@triton.jit` kernel (lines 44-60), verbatim from
  `fla/modules/norm/l2norm.py:68-84`: each row of `x` times `1 / sqrt(sum x^2 + eps)`, one
  program per `BT` rows, with that factor saved per row (`rstd`);
- `l2norm_fwd`, the host, adapted from fla's wrapper `l2norm_fwd` (`l2norm.py:111-139`);
- `reference`, the normalisation in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla's source (each listed with its `file:line` in the module docstring): the
autotuner `@fla_cache_autotune(...)` is removed from the kernel; the host drops fla's `@dispatch`
(its backend registry, which falls back to this function on NVIDIA GPUs), passes one
configuration explicitly, and replaces the branch for `D > 512`, which launches another kernel
(`l2norm_fwd_kernel_row`), by an assert. `reference` and the test under `__main__` are new.

## The config

Removed: `@fla_cache_autotune(configs=[triton.Config({"BT": BT}, num_warps=num_warps) for
num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST], key=["D", "NB"], ...)`
(`fla/modules/norm/l2norm.py:63-67`, `BT_LIST = [8, 16, 32, 64, 128]` at line 17).

Chosen: `BT = 16`, `num_warps = 4`, the 12th of the list's 25 configurations (`num_warps = 4`,
`BT = 16`); `num_stages` is Triton's default. The wrapper's own values: grid
`(triton.cdiv(T, BT),)`, `BD = min(65536 // x.element_size(), triton.next_power_of_2(D))`,
`NB = triton.cdiv(T, 2048 * 32)` (an autotune key only; the body does not read it), `eps` and `T`
passed as runtime scalars (`T` under `do_not_specialize`). No heuristics.

## The launches

The claim should cover the host's launches with shapes as variables:
- `x` `bfloat16`, contiguous, viewed as `[T, D]` (fla's callers pass inputs through its
  `input_guard`, which makes them contiguous); `y` of the dtype of `x` (`output_dtype = None`, the
  default), or `float32` (`output_dtype = torch.float32`: PaTH attention normalises its `w` so,
  `fla/layers/path_attn.py:124, 147, 203`, heads of 64 at its default 32 heads of hidden size
  2048); `rstd` `float32`; `eps = 1e-6` (the default; any positive `float`);
- `D` from 1 to 512 (this kernel's path; fla's delta-rule layers normalise queries and keys of
  head dim 64, 128 or 256: `fla/ops/gated_delta_rule/chunk.py:283-284`,
  `fla/ops/delta_rule/chunk.py:171-172`; DeltaNet's default config has 16 heads of 128,
  GatedDeltaNet's 6 heads of 256), `BD` the next power of two;
- `T` (the rows: batch × tokens × heads) from 1 up, not a multiple of `BT` in general; fla's tests
  (`tests/modules/test_l2norm.py:20-29`) use up to 3 × 1024 × 4 rows on this path, and a training
  batch of 4 × 4096 tokens × 16 heads is about 2.6·10^5 rows.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host on 230 launches: the host test's nine shapes
`(B, T, H, D)`, from `(1, 1, 1, 1)` to `(4, 4096, 16, 128)`, GatedDeltaNet's 6 heads of 256 on
4 × 4096 tokens, and PaTH attention's 32 heads of 64 with a `float32` `y` on 4 × 4096 tokens and on
one token of a batch of 8; `D` in 17 classes from 1 to 512 at `T` 1, 15, 17 and 1001; `eps` from
`2^-125` to `2^100` (and `2^126` on zero rows) with zero rows, subnormal inputs and rows scaled by
`2^±40`; rows with `Σ x² + eps` up to `2^126`; direct launches at `BT` 1, 2, 8, 16, 32, 64 and 128
into sentinel-filled buffers; and `T = 2^31 + 7` (the `int64` class) on the first two programs'
rows; the edges with both output dtypes. Every output is finite and within the claim's tolerance:
the worst error / tolerance is `8.8·10^-4` for `rstd` (margin `1136×`), `0.498` for a `bfloat16` `y`
(margin `2.0×`) and `0.34` for a `float32` `y` (margin `2.9×`); `x` is unchanged and every sentinel
intact. Eleven wrong outputs (`y = 0`, `y = x`, the next row's `rstd`, `rstd·(1 + 2^-11)`,
`rstd = 1/Σ x²`, a `float32` `y` rounded to `bfloat16`) exceed their tolerances. It also shows the
two kernel limits outside the claim: `eps = 1e-39` on a zero row gives `rstd = inf`, `y = NaN`, and
`x = 2^64` at `D = 1` gives `rstd = 0`, `y = 0`.
