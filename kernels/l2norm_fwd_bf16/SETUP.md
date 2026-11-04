# Setup of `l2norm_fwd_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/modules/norm/l2norm.py:69`
`l2norm_fwd_kernel`.

## The source

`source/l2norm.py` holds:
- `l2norm_fwd_kernel`, the `@triton.jit` kernel (lines 44-60), verbatim from
  `fla/modules/norm/l2norm.py:68-84`: each row of `x` times `1 / sqrt(sum x^2 + eps)`, one
  program per `BT` rows, with that factor saved per row (`rstd`);
- `l2norm_fwd`, the host, adapted from fla's wrapper `l2norm_fwd` (`l2norm.py:111-139`);
- `reference`, the normalisation in `float64`, and the test under `if __name__ == "__main__":`.

Changed from fla's source (each listed with its `file:line` in the module docstring): the
autotuner `@fla_cache_autotune(...)` is removed from the kernel; the host drops fla's `@dispatch`
(its backend registry, which falls back to this function on NVIDIA GPUs), passes one
configuration explicitly, and replaces the branch for `D > 512`, which launches another kernel
(`l2norm_fwd_kernel_row`), by an assert.

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
  default); `rstd` `float32`; `eps = 1e-6` (the default; any positive `float`);
- `D` from 1 to 512 (this kernel's path; fla's delta-rule layers normalise queries and keys of
  head dim 64, 128 or 256: `fla/ops/gated_delta_rule/chunk.py:283-284`,
  `fla/ops/delta_rule/chunk.py:171-172`; DeltaNet's default config has 16 heads of 128,
  GatedDeltaNet's 6 heads of 256), `BD` the next power of two;
- `T` (the rows: batch × tokens × heads) from 1 up, not a multiple of `BT` in general; fla's tests
  (`tests/modules/test_l2norm.py:20-29`) use up to 3 × 1024 × 4 rows on this path, and a training
  batch of 4 × 4096 tokens × 16 heads is about 2.6·10^5 rows.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes on nine shapes `(B, T, H, D)`,
from `(1, 1, 1, 1)` to `(4, 4096, 16, 128)`, with `D` in 1, 60, 64, 100, 128, 256 and 512: max abs
error of `y` against the `float64` reference at most 9.8e-4 (bfloat16 rounding), max relative
error of `rstd` at most 1.9e-7.
