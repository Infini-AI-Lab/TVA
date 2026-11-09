# Setup of `layer_norm_fwd_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/modules/norm/layernorm.py:48`,
the function `layer_norm_fwd_kernel`.

## The source

`source/layernorm.py` holds:
- `layer_norm_fwd_kernel`, the `@triton.jit` kernel (lines 50-114), verbatim from
  `fla/modules/norm/layernorm.py:47-111`: one program per `BT` rows; it adds the residual (if
  any), writes the sum out (if asked), and normalises it (LayerNorm, or RMSNorm under
  `IS_RMS_NORM`), saving `mean` and `rstd`, then applies the weight and bias of the row's group
  (`t % G`). All its compile-time paths are kept;
- `layer_norm_fwd`, the host, adapted from fla's wrapper `layer_norm_fwd`
  (`layernorm.py:388-471`);
- `reference`, LayerNorm in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla's source (each listed with its `file:line` in the module docstring): the
`@triton.autotune(...)` decorator is removed from the kernel; the host drops fla's `@dispatch`
(its backend registry, which falls back to this function on NVIDIA GPUs), passes one
configuration explicitly, and replaces the branch for `D > 512` (and `D == 512` on Intel GPUs),
which launches another kernel (`layer_norm_fwd_kernel_row`), by an assert. `reference` and the test under `__main__` are new.

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

The claim should cover the kernel's real launches on the LayerNorm path (`IS_RMS_NORM = False`)
with weight and bias, with shapes as variables. fla's `LayerNormFunction`
(`fla/modules/norm/layernorm.py:578-620`) reshapes its input to `[-1, hidden // num_groups]`, so
a row is one group of one token, `T = tokens × G` and `D = hidden / G`; the hidden sizes of the
RWKV models (2048 and wider) make `LayerNorm` rows of `D > 512`, which take
`layer_norm_fwd_kernel_row`, so on this kernel the real use is `GroupNorm`:
- **`GroupNorm` without a residual**, fla's `GroupNorm` module (`layernorm.py:1017-1078`):
  RWKV-7's `g_norm` (`fla/layers/rwkv7.py:126-133`: `num_groups = num_heads`, head size
  `D = 64`, `eps = head_dim * norm_eps = 64 * 1e-5`, `bias=True`; 32 heads at the default hidden
  size 2048, `fla/models/rwkv7/configuration_rwkv7.py:25-29`) and RWKV-6's `g_norm`
  (`fla/layers/rwkv6.py:87`: 4 heads of `D = 512` at hidden size 2048, `eps = 1e-5`,
  `bias=True`); the shapes of fla's GroupNorm test (`tests/modules/test_layernorm.py:57-62`: `G` 1
  and 4, hidden 64 to 2048 on this path; the test runs `float32`, the setup its shapes in
  `bfloat16`). The flags: `HAS_RESIDUAL = STORE_RESIDUAL_OUT = False` (no residual and
  `residual_in_fp32 = False`: no `res_out`), `HAS_WEIGHT = HAS_BIAS = True`; `G` from 1 up (one
  group per head);
- **`LayerNorm`** (`G = 1`) with hidden size at most 512, as fla's `LayerNorm` module
  (`layernorm.py:964-1014`) runs it: without a residual (`attn_norm`, the final `norm`,
  `fla/models/rwkv7/modeling_rwkv7.py:179`; fla's LayerNorm test,
  `tests/modules/test_layernorm.py:20-25`, `D` in 50, 64 and 128 on 2 × 2 × 512 rows, in
  `float32` there and in `bfloat16` here) and with a
  residual in a pre-norm block (`ffn_norm(hidden_states, residual, True)`, `modeling_rwkv7.py:191`:
  `HAS_RESIDUAL = STORE_RESIDUAL_OUT = True`, `res_out` of the residual's dtype); the weight and
  the bias that RWKV's configs enable (`norm_bias=True`, `configuration_rwkv7.py:38`);
- `x`, `residual`, `weight`, `bias`, and so `y` and `res_out`, `bfloat16`; `mean` and `rstd`
  `float32`; `x` and `residual` contiguous `[T, D]` (`input_guard` makes them contiguous);
  `weight` and `bias` `[G·D]`; `eps` any positive `float` (`1e-5` and `6.4e-4` above);
- `D` from 1 to 512 (this kernel's path; wider rows go to `layer_norm_fwd_kernel_row`), `BD` the
  next power of two;
- `T` from 1 up, not a multiple of `BT` in general: one decoding token of RWKV-7 is 32 rows, a
  training batch of 4 × 4096 tokens 524288 rows (RWKV-7) or 65536 (RWKV-6).

RMSNorm (`IS_RMS_NORM = True`: the same kernel's RMSNorm mode, without a bias, which the
`GroupNorm(..., is_rms_norm=True)` query and key norms of `fla/layers/forgetting_attn.py:72-81` and
`fla/layers/wall_attn.py:122-131` run), launches without weight or bias, and a `res_out` without
a residual (`residual_in_fp32 = True`) are not part of the setup.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host on 53 launches: the host test's nine shapes
`(T, D)` with a residual, from `(1, 1)` to `(65536, 256)`; RWKV-7's `g_norm` on 524288 rows and on
one token (`G = 32`, `D = 64`, `eps = 6.4e-4`), RWKV-6's on 65536 rows (`G = 4`, `D = 512`), fla's
GroupNorm (eight shapes) and LayerNorm tests' shapes without a residual; and edges: `D` = 1, 2, 7, 257 and 512;
constant, near-constant (also with `G = 3`) and ill-conditioned rows (`D = 3`, offset `3·10^4`,
noise `10^-3`, `eps` `1e-5` and `2^-125`); rows of `1e-30` and subnormal rows;
`|x| = |res| = 2^56` and `|w| = |b| = 2^16` (also without a residual, `G` 2 and 5, and on
ill-conditioned rows); `w = b = 0`;
`eps = 2^-125` (random, constant and tiny rows), `eps = 1` and `eps = 2^126`; `T = 33`, `T = 37`
with `G = 7`, `T = 5` with `G = 32`, and `G = 1024` at `D = 1`. With a residual `res_out` equals
the `float32` sum rounded to `bfloat16` bit for bit; without one `res_out` is not written; the
guard bands around every output and the inputs are unchanged; the worst error / tolerance is
`0.122` for `mean` and `0.99902` for `rstd`, on the ill-conditioned rows with `eps = 2^-125`
(`cond ≈ 10^17`; elsewhere at most `0.149` for `rstd`), and `0.498` for `y`. Seven wrong outputs at RWKV-7's shape (`y = 0`, group 0's weight for every row,
no bias, `x` for `x − mean`, `mean·(1 + 2^-12)`, `rstd·(1 + 2^-14)`, `rstd` without `eps`) exceed
their tolerances by factors of 15 to `1.5·10^5`.
