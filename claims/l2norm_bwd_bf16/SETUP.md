# Setup of `l2norm_bwd_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/modules/norm/l2norm.py:93`,
the function `l2norm_bwd_kernel`.

## The source

`source/l2norm.py` holds:
- `l2norm_bwd_kernel`, the `@triton.jit` kernel (lines 51-67), verbatim from
  `fla/modules/norm/l2norm.py:92-108`: each row's
  `dx = dy * rstd - (sum dy * y) * y * rstd`, one program per `BT` rows;
- `l2norm_bwd`, the host, adapted from fla's wrapper `l2norm_bwd` (`l2norm.py:142-168`);
- `reference`, the formula in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla's source (each listed with its `file:line` in the module docstring): the
autotuner `@fla_cache_autotune(...)` is removed from the kernel; the host drops fla's `@dispatch`
(its backend registry, which falls back to this function on NVIDIA GPUs), passes one
configuration explicitly, and replaces the branch for `D > 512`, which launches another kernel
(`l2norm_bwd_kernel_row`), by an assert. The file's other kernels and functions (the forward, the
row kernels, the autograd function and the module) are left out. `reference` and the test under
`__main__` are new.

## The config

Removed: `@fla_cache_autotune(configs=[triton.Config({"BT": BT}, num_warps=num_warps) for
num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST], key=["D", "NB"], ...)`
(`fla/modules/norm/l2norm.py:87-91`, `BT_LIST = [8, 16, 32, 64, 128]` at line 17).

Chosen: `BT = 16`, `num_warps = 4`, the 12th of the list's 25 configurations (`num_warps = 4`,
`BT = 16`); `num_stages` is Triton's default. The wrapper's own values: grid
`(triton.cdiv(T, BT),)`, `BD = min(65536 // y.element_size(), triton.next_power_of_2(D))`,
`NB = triton.cdiv(T, 2048 * 32)` (an autotune key only; the body does not read it), `eps` and `T`
passed as runtime scalars (`T` under `do_not_specialize`; the body does not read `eps`). No
heuristics.

## The launches

The claim should cover the host's launches with shapes as variables. fla calls `l2norm_bwd` from
the backward of `l2norm` (`L2NormFunction.backward`, `l2norm.py:181-186`) and from the backward of
its delta-rule layers with `use_qk_l2norm_in_kernel` (`fla/ops/gated_delta_rule/chunk.py:384-385`,
`fla/ops/delta_rule/chunk.py:221-222`, `fla/ops/kda/chunk.py:204-205`, and the other chunked
delta-rule variants), on the normalised queries and keys and their gradients:
- `y` `bfloat16`, contiguous, viewed as `[T, D]` (the forward's output for a `bfloat16` input),
  or `float32` (the forward with `output_dtype = torch.float32`: PaTH attention normalises its `w`
  so, `fla/layers/path_attn.py:124, 147, 203`, heads of 64 at its default 32 heads of hidden size
  2048); `rstd` `float32`, the forward's factors; `dy` the gradient of `y`, of `y`'s dtype through
  autograd, and as the delta-rule backward kernels produce it (`bfloat16` for `bfloat16` models);
  `dx = empty_like(y)`; `eps = 1e-6` (the default; the kernel does not read it). The claim covers
  each of `y`, `dy`, `dx` in `bfloat16` or `float32`, in any combination;
- `D` from 1 to 512 (this kernel's path; fla's delta-rule layers normalise queries and keys of
  head dim 64, 128 or 256; DeltaNet's default config has 16 heads of 128, GatedDeltaNet's 6 heads
  of 256), `BD` the next power of two;
- `T` (the rows: batch × tokens × heads) from 1 up, not a multiple of `BT` in general; fla's tests
  (`tests/modules/test_l2norm.py:20-29`) use up to 3 × 1024 × 4 rows on this path, and a training
  batch of 4 × 4096 tokens × 16 heads is about 2.6·10^5 rows.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs 188 launches: the host test's nine shapes
`(B, T, H, D)`, from `(1, 1, 1, 1)` to `(4, 4096, 16, 128)`, GatedDeltaNet's 6 heads of 256 on
4 × 4096 tokens, PaTH attention's 32 heads of 64 in `float32` on 4 × 4096 tokens, a `float32` `dy`
into a `bfloat16` `y` and a `bfloat16` `dy` into a `float32` `y`; `D` in 17 classes from 1 to 512
at `T` 1, 15, 17 and 1001, with the four dtype pairs of `y` and `dy` at `T = 17`; rows with `dy`
along `y` (`dx ≈ 0`, the cancellation the tolerance's `M` covers), zero, scaled by `2^±40` and
subnormal gradients, zero rows with `rstd = 2^60`, `rstd = 2^-60`, subnormal `y`, and rows near
the room bound `2^124`; direct launches at `BT` 1, 2, 8, 16, 32, 64 and 128 into sentinel-filled
buffers, with a `dx` of either dtype; and `T = 2^31 + 7` (the `int64` class) on the first two
programs' rows. Every input satisfies `Pre`, every output is finite and within the claim's
tolerance: the worst error / tolerance is `0.994` for a `bfloat16` `dx` (the `2^-8·|ref|`
rounding term is the exact half-ulp bound, so a correctly rounded value can come close to it)
and `0.034` for a `float32` `dx` (margin `29×`); `y`, `dy`, `rstd` are unchanged and every
sentinel intact. Eleven wrong outputs (`dx = 0`, `dx = dy·rstd`, the projection added, the next
row's `rstd`, `dx·(1 + 2^-6)`, a `float32` `dx` rounded to `bfloat16`) exceed their tolerances,
by 5× at least. It also shows the kernel limit outside the claim: `rstd = 2^100`, `dy = 2^40`,
`y = 0.5` on a row of 4 gives `dx = −inf` (exact value `0`). Peak device memory is 2.7 GiB.
