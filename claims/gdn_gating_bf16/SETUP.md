# Setup of `gdn_gating_bf16`

From SGLang v0.5.23 (commit `297e54fd10`),
`python/sglang/kernels/ops/attention/linear/fused_gdn_gating.py:11`, the kernel
`fused_gdn_gating_kernel`, launched by the wrapper `fused_gdn_gating`.

## The source

`source/fused_gdn_gating.py` holds:
- `fused_gdn_gating_kernel`, the `@triton.jit` kernel, copied verbatim (with the two comment
  lines above it): one program per (token, block of 8 heads) loads `A_log`, `a`, `b` and
  `dt_bias`, stores `g = -exp(A_log) · softplus(a + dt_bias)` (softplus with `beta` and
  `threshold`) and `beta_output = sigmoid(b)`, converted to `b`'s dtype before the store;
- `fused_gdn_gating`, the host wrapper (`fused_gdn_gating.py:44`), copied verbatim but for its
  return annotation (`tuple` for `typing.Tuple`) and one added launch option;
- `reference`, a test helper (`g` and `sigmoid(b)` in `float64`);
- the test under `if __name__ == "__main__":`.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`, and its
parameters have no defaults. The configuration is the wrapper's own: grid
`(batch, 1, cdiv(num_heads, 8))`, `seq_len = 1`, `BLK_HEADS = 8`, `num_warps=1`, the row strides
`a.stride(0)` and `b.stride(0)`, `NUM_HEADS = a.shape[1]`, and `beta = 1.0`,
`threshold = 20.0` (the wrapper's defaults, which every caller uses) as compile-time values. The
launch option `num_stages=3` is added: the wrapper passes none, and it is Triton 3.8.0's default.
No flags.

## The launches

The claim covers the wrapper's launches as SGLang's Gated DeltaNet backend makes them
(`srt/layers/attention/linear/gdn_backend.py:946` and `:1348`, both
`fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)`, on the extend/prefill paths), with shapes
as variables:
- dtypes, as Qwen3-Next's layer creates them (`srt/models/qwen3_next.py:215-219`): `A_log`
  `float32` (created with `dtype=torch.float32`); `dt_bias` `bfloat16` (created with
  `torch.zeros` and no dtype, under `set_default_torch_dtype(model_config.dtype)`,
  `srt/model_loader/loader.py:838`, so the model's dtype; the code base's test also uses
  `bfloat16`, `test/registered/attention/test_gdn_noncontiguous_stride.py:45-46`); `a` and `b`
  `bfloat16`, the layer's `in_proj_ba` output; `g` and `beta_output` `float32`;
- `beta = 1.0`, `threshold = 20.0`;
- `a`, `b` of shape `[batch, num_heads]` with unit stride along the heads and a row stride of
  `num_heads` (the fused split, `kernels/ops/attention/triton_gdn_fused_proj.py:130-135`) or more
  (views split from one `[batch, 2 · num_heads]` tensor: row stride `2 · num_heads`, as the code
  base's test checks, `test_gdn_noncontiguous_stride.py:21-36`);
- `batch` the number of tokens of the extend batch, from 1 to the prefill chunk (thousands to
  tens of thousands); `num_heads` the value heads per tensor-parallel rank: Qwen3-Next's 32
  (`srt/configs/qwen3_next.py:208`) divided by TP 1 to 8, and the test's 48, 24 and 8
  (Qwen3.5-27B at TP 1 and 2), not necessarily a multiple of 8;
- `A_log`, `dt_bias` of `num_heads` elements; any finite values of `a`, `b`, `dt_bias` and
  `A_log`.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes: the host's nine test launches
(`(batch, num_heads)` from (1, 32) to (32768, 32), contiguous and split `a`/`b`) and six
model-sized ones (up to 65,536 tokens with 4 to 48 heads), edge launches (every finite `bfloat16`
`a` against 16 values of `A_log` from `-3e38` to `89`, with `dt_bias` zero, random and huge, and
every finite `b`; sums near the threshold `20`), and two direct launches into a guarded random
pool. Over 11,068,444 cells inside `Pre`, `Post` holds on every one: `g` at worst 0.4997 of
`gTol` (0.375 where its relative term is at least half of it), `beta_output` at worst 0.498 of its
bound, 245,348 cells `-∞` by the overflow clause; every input and guard is unchanged. Of the
1,050,128 cells outside `Pre`, 132,826 violate `Post` (`A_log ≥ 88.72284`, sums rounding to `+∞`,
`b < -88.72`).
