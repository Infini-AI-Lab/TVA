# Setup of `layer_norm_fwd_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/modules/norm/layernorm.py:48`
`layer_norm_fwd_kernel`.

## The source

`source/layernorm.py` holds:
- `layer_norm_fwd_kernel`, the `@triton.jit` kernel (lines 50-114), verbatim from
  `fla/modules/norm/layernorm.py:47-111`: one program per `BT` rows; it adds the residual, writes
  the sum out, and normalises it (LayerNorm, or RMSNorm under `IS_RMS_NORM`), saving `mean` and
  `rstd`, then applies the weight and bias of the row's group. All its compile-time paths are kept;
- `layer_norm_fwd`, the host, adapted from fla's wrapper `layer_norm_fwd`
  (`layernorm.py:388-471`);
- `reference`, LayerNorm in `float64`, and the test under `if __name__ == "__main__":`.

Changed from fla's source (each listed with its `file:line` in the module docstring): the
`@triton.autotune(...)` decorator is removed from the kernel; the host drops fla's `@dispatch`
(its backend registry, which falls back to this function on NVIDIA GPUs), passes one
configuration explicitly, and replaces the branch for `D > 512` (and `D == 512` on Intel GPUs),
which launches another kernel (`layer_norm_fwd_kernel_row`), by an assert.

## The config

Removed: `@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for BT in
[32, 64, 128] for num_warps in [2, 4, 8]], key=['D', 'NB', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT',
'IS_RMS_NORM'], ...)` (`fla/modules/norm/layernorm.py:38-46`).

Chosen: `BT = 32`, `num_warps = 4`, the 2nd of the list's 9 configurations; `num_stages` is
Triton's default. The wrapper's own values: grid `(triton.cdiv(T, BT),)`,
`BD = min(65536 // x.element_size(), triton.next_power_of_2(D))`, `NB = triton.cdiv(T, 2048)` (an
autotune key only; the body does not read it), `G = num_groups`, `eps` and `T` passed as runtime
scalars (`T` under `do_not_specialize`). The flags, as the wrapper computes them from its
arguments: `IS_RMS_NORM = is_rms_norm`, `HAS_RESIDUAL = residual is not None`,
`STORE_RESIDUAL_OUT = res_out is not None`, `HAS_WEIGHT = weight is not None`,
`HAS_BIAS = bias is not None`. No heuristics.

## The launches

The claim should cover the LayerNorm path with a residual, as fla's `LayerNorm` module runs it in
a pre-norm block (`fla/modules/norm/layernorm.py:964-1014`; RWKV-6 and RWKV-7 call it with the
residual and `prenorm=True`, `fla/models/rwkv7/modeling_rwkv7.py:191`), with shapes as variables:
- the flags: `IS_RMS_NORM = False`, `HAS_RESIDUAL = True`, `STORE_RESIDUAL_OUT = True`,
  `HAS_WEIGHT = True`, `HAS_BIAS = True` (the module's `elementwise_affine=True` default, and the
  bias that RWKV's configs enable, `norm_bias=True`, `fla/models/rwkv7/configuration_rwkv7.py:38`;
  the module's own default is `bias=False`);
- `G = 1` (`num_groups = 1`, LayerNorm; fla's GroupNorm launches the same kernel with `G > 1`);
- `x`, `residual`, `weight`, `bias`, and so `y` and `res_out`, `bfloat16`; `mean` and `rstd`
  `float32`; `x` and `residual` contiguous `[T, D]` (`LayerNormFunction` reshapes them to
  `[T, D]` under `input_guard`, which makes them contiguous); `weight` and `bias` `[D]`;
  `eps = 1e-5` (the module's default, as RWKV's `norm_eps`; any positive `float`);
- `D` from 1 to 512 (this kernel's path; wider rows go to `layer_norm_fwd_kernel_row`), `BD` the
  next power of two; fla's tests use `D` in 50, 64 and 128 (`tests/modules/test_layernorm.py:20-25`);
- `T` (the rows: batch × tokens) from 1 up, not a multiple of `BT` in general; the tests use
  2 × 2 × 512 rows, a training batch tens of thousands.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes on nine shapes `(T, D)`, from
`(1, 1)` to `(65536, 256)`, with `D` in 1, 50, 64, 128, 256, 300 and 512: `res_out` equal bit for
bit to the `float32` sum rounded to `bfloat16`; `y` within 3.1e-3 of the `float64` reference,
relative to `1 + |ref|` (bfloat16 rounding); `mean` within 4.3e-8 absolute; `rstd` within 1.9e-7
relative.
