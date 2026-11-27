# Setup of `layer_norm_gated_fwd_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`),
`fla/modules/norm/fused_norm_gate.py:36`, the function `layer_norm_gated_fwd_kernel`.

## The source

`source/fused_norm_gate.py` holds:
- `layer_norm_gated_fwd_kernel`, the `@triton.jit` kernel (lines 55–127), verbatim from
  `fla/modules/norm/fused_norm_gate.py:35-107`. It computes LayerNorm or RMSNorm of each row of
  `x`, optionally times a weight and plus a bias, times a gate `g` passed through `swish` / `silu`
  or `sigmoid`. One program handles `BT` rows, and `rstd` (and LayerNorm's `mean`) is saved per
  row;
- `layer_norm_gated_fwd`, the host (lines 131–200), adapted from fla's wrapper
  (`fused_norm_gate.py:447-525`);
- `reference`, the RMS-norm-gated output in `float64`, and the test under
  `if __name__ == "__main__":`.

Changes from fla's source (each listed with its `file:line` in the module docstring):
- The decorators `@triton.heuristics(...)` and `@triton.autotune(...)` are removed from the kernel.
  The heuristics computed `STORE_RESIDUAL_OUT`, `HAS_RESIDUAL`, `HAS_WEIGHT` and `HAS_BIAS` from
  which tensors are `None`.
- The host drops fla's `@dispatch` (its backend registry, which falls back to this function on
  NVIDIA GPUs).
- The host passes those four flags and one configuration explicitly.
- The branch for `D > 512`, which launches another kernel (`layer_norm_gated_fwd_kernel_row`), is
  replaced by an assert.
- `reference` and the test under `__main__` are new.

## The config

Removed: `@triton.autotune(configs=[triton.Config({"BT": BT}, num_warps=num_warps) for BT in
[16, 32, 64] for num_warps in [4, 8, 16]], key=["D", "NB", "IS_RMS_NORM", "STORE_RESIDUAL_OUT",
"HAS_RESIDUAL", "HAS_WEIGHT"], **autotune_cache_kwargs)` (`fused_norm_gate.py:30-34`), and
`@triton.heuristics(...)` (`fused_norm_gate.py:22-29`).

Chosen: `BT = 16`, `num_warps = 4`, the first of the list's 9 configurations; `num_stages` is
Triton's default. The wrapper's own values:
- the grid `(triton.cdiv(T, BT),)`;
- `BD = min(65536 // x.element_size(), triton.next_power_of_2(D))`;
- `NB = triton.cdiv(T, 2048 * 32)`, which is an autotune key only (the body does not read it);
- `eps` and `T` passed as runtime scalars (`T` under `do_not_specialize`).

The claim covers every power-of-two `BT` with `BT·BD ≤ 2^20`, so every `BT` of the list. `num_warps`
does not enter the model.

## The launches

fla's `FusedRMSNormGated` (`fla/modules/norm/fused_norm_gate.py:967`) calls `rms_norm_gated`. That
reshapes `x` and `g` to `[-1, D]` and calls the host with `is_rms_norm = True`, its weight, and
`bias = None`. The linear-attention layers call it as `o_norm(o, g)`, with no residual. The claim
should cover these launches with shapes as variables:
- `x`, `g`, `y` `bfloat16`, contiguous `[T, D]`; `weight` `bfloat16` `[D]` (the model's dtype;
  initialised to ones and trained); `rstd` `float32` `[T]`; `mean` `None`;
- `activation` `"swish"` (the default), `"silu"` or `"sigmoid"`. The default is used by
  GatedDeltaNet (`fla/layers/gated_deltanet.py:195`, `output_gate_activation = 'swish'`), GLA
  (`gla.py:163`), DeltaNet (`delta_net.py:164`), GatedDeltaProduct, Mesa-Net, MoM and others.
  `"sigmoid"` is used by KDA (`kda.py:191`), Comba (`comba.py:207`), GDN-2 (`gdn2.py:197`) and MoBA
  (`moba.py:154`);
- `eps` the configuration's `norm_eps`: `1e-6` in the model configurations of GatedDeltaNet, GLA
  and KDA (`fla/models/*/configuration_*.py`), `1e-5` the layers' default;
- `D` the value head dimension, at most 512 on this path:
  - GatedDeltaNet: 6 heads of `256·2 = 512`;
  - GLA: 4 heads of `2048 / 4 = 512`;
  - KDA: 16 heads of 128;
- `T` (the rows: batch × tokens × heads) from 1 up, not a multiple of `BT` in general. A training
  batch of 4 × 4096 tokens gives `98304` rows for GatedDeltaNet, `65536` for GLA and `262144` for
  KDA. One decoding token of a batch of 8 gives `48`, `32` and `128`.

fla's own test (`tests/modules/test_layernorm_gated.py:71-93`) runs this path with `float32`
tensors and `eps = 0`, on `(B, H, T, D)` `(2, 2, 1, 64, silu)`, `(2, 2, 512, 128, sigmoid)` and
`(2, 2, 50, 50, sigmoid)` (its `D = 1200` case takes the other kernel). The host test here takes
those shapes in `bfloat16` with `eps = 1e-5`. `eps = 0` is outside the claim (README, Exclusions).

Not launched by these layers, and so not covered: a `float32` weight (a model kept in `float32`
under autocast), `elementwise_affine = False` (YOCO, `yoco.py:131`), `FusedLayerNormGated` (the
LayerNorm path), a bias, a residual, and `D > 512`.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs 330 checks:
- the host test's seven cases;
- the three layers on 4 × 4096 tokens and on one decoding token of a batch of 8;
- `D` in its classes from 1 to 512 at several `T` not a multiple of 16, with both activations;
- `eps` from `2^-125` to `2^100` with zero and tiny rows, and `eps = 2^125` on zero rows;
- rows with `Σ x² + eps` up to `2^125`, and mixed scales;
- gates and weights from `0` to `±2^56`, with gates that overflow `tl.exp` (`−88.5`, `−89`,
  `−100`), subnormal gates and subnormal products;
- direct launches at `BT` 1 to 64 and `num_warps` 8 and 16 into sentinel-filled buffers;
- `T = 2^31 + 7` (the `int64` class), on the first two programs' rows.

Every output is finite and within the claim's tolerance:
- the worst error / tolerance is `9.2·10^-4` for `rstd` (margin `1088×`) and `0.498` for `y`
  (margin `2.0×`);
- the inputs are unchanged and every sentinel is intact.

Nine wrong outputs per activation exceed their tolerances: `y = 0`, `y` without the gate, with the
other activation, gated by `g` alone, with the mean subtracted (LayerNorm), without the weight, or
with the next row's `rstd`; `rstd·(1 + 2^-11)`; and `rstd` without `/ D`.

It also shows the kernel limits outside the claim:
- `eps = 0` and `eps = 1e-39` on a zero row give `rstd = inf`, `y = NaN`;
- `x = 2^64` at `D = 1` gives `rstd = 0`, `y = 0`;
- `w = g = 2^100` gives `y = inf`.
