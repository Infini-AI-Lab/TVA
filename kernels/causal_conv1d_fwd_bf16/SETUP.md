# Setup of `causal_conv1d_fwd_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/modules/causal_conv1d/ops.py:51`
`causal_conv1d_fwd_kernel`.

## The source

`source/causal_conv1d.py` holds:
- `causal_conv1d_fwd_kernel`, the `@triton.jit` kernel, verbatim: the depthwise causal
  convolution of a `[B, T, D]` input with a `[D, W]` filter, with an optional initial state (the
  `W - 1` tokens before each sequence), bias, SiLU and residual, over fixed-length batches or
  packed variable-length ones; one program per (block of `BD` channels, chunk of `BT` tokens,
  batch row). It calls no helper;
- the host: `causal_conv1d` (the `backend='triton'` path of fla's entry point, as
  `ShortConvolution.forward` calls it, `ops.py:995-1115`, with `CausalConv1dFunction.forward`,
  `ops.py:934-972`) and `causal_conv1d_fwd` (the launch, `ops.py:604-662`), with fla's torch
  helpers `prepare_lens`, `_segmented_arange` and `prepare_chunk_indices`
  (`fla/ops/utils/index.py`), which build the `[NT, 2]` table of (sequence, chunk) pairs;
- `reference`, the convolution in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring: the kernel's
decorators `@triton.heuristics` and `@fla_cache_autotune` removed (`@triton.jit` kept); the host
passes the flags and one configuration explicitly; `causal_conv1d_fwd` loses its
`output_final_state` argument (the final state is another kernel's, `causal_conv1d_update_states`)
and its unused `layout_fallback`; `einops.rearrange` written as a `reshape`; the decorators
`@deprecate_kwarg`, `@dispatch` (backend selection; Triton on NVIDIA GPUs) and
`@input_guard(no_guard_contiguous=["x"])` replaced by `.contiguous()` calls on every tensor
argument but `x`; the entry point keeps only the Triton backend without context parallelism
(`'cuda'`, `'mix'` and context parallelism run other code); the index helpers copied without their
`@tensor_cache` memo.

## The config

- Removed: `@triton.heuristics({'HAS_WEIGHT': …, 'HAS_BIAS': …, 'HAS_RESIDUAL': …,
  'USE_INITIAL_STATE': …, 'IS_VARLEN': …})` (each `args[name] is not None`) and
  `@fla_cache_autotune(configs=[triton.Config({'BD': BD}, num_warps=num_warps) for BD in [16, 32,
  64, 128] for num_warps in NUM_WARPS_AUTOTUNE], key=['D', 'W'])` (`ops.py:34-49`;
  `NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]` off AMD, `ops.py:30`).
- Chosen: `BD = 64`, `num_warps=8`, the list's tenth entry, the autotuner's choice on the H200 for
  `D` = 2048, 4096 and 8192 with `W = 4` (variable-length, initial state, bias, SiLU);
  `num_stages=3`, a `triton.Config`'s default. The grid is the wrapper's `(cdiv(D, BD), NT, B)`.
- Flags set by the host: `HAS_WEIGHT`, `HAS_BIAS`, `HAS_RESIDUAL`, `USE_INITIAL_STATE`,
  `IS_VARLEN`, each `True` when `weight`, `bias`, `residual`, `initial_state`, `cu_seqlens` is not
  `None`. The wrapper itself passes `BT = chunk_size` (64 by default), `BW = next_power_of_2(W)`,
  `ACTIVATION = activation`, `D`, `W`, `B`, `T` and `x`'s strides.

## The launches

The claim should cover the launches of fla's `ShortConvolution` in a prefill or training step with
a cache and a bias, over a packed batch (`ShortConvolution.forward`, `ops.py:1205-1281`, with
`bias=True`, `cache` given as `initial_state`, `cu_seqlens` given; its other arguments at their
defaults; fla's test of this path: `tests/modules/test_conv.py:462-514`,
`test_conv_varlen_initial_state`):
- `bfloat16` `x`, `weight`, `bias`, `initial_state` and `y` (the module's and the cache's dtype);
  `int32` or `int64` `cu_seqlens`, and `chunk_indices` of its dtype, as `prepare_chunk_indices`
  builds it or as the caller passes it, the same table;
- `HAS_WEIGHT`, `HAS_BIAS`, `USE_INITIAL_STATE`, `IS_VARLEN` true, `HAS_RESIDUAL` false (the
  default `residual=None`), `ACTIVATION = 'silu'` (`ShortConvolution`'s default; `'swish'` runs
  the same code), `BT = 64`;
- `x` of shape `[1, T, D]` as `ShortConvolution` gets it from a linear projection (contiguous:
  strides `(T·D, D, 1)`), `weight` `[D, W]`, `bias` `[D]`, `initial_state` `[N, D, W]`, contiguous;
- shapes as variables: `W` from 2 to 4 (`conv_size = 4` in fla's model configs,
  `fla/models/gated_deltanet/configuration_gated_deltanet.py:30`; fla's tests use 3 and 4; its
  CUDA context-parallel path documents widths 2 to 4, `ops.py:1046`), `D`
  from fla's tests (100, 128, 200, 1024, 4096, 8192) and models (gated DeltaNet's defaults,
  `configuration_gated_deltanet.py:25-33`: 6 heads of 256 key channels and 512 value channels,
  so `D` = 1536 and 3072, and 6144 for the fused q/k/v convolution, `fla/layers/gated_deltanet.py:247-266`),
  `N` sequences with lengths `≥ 0` summing to `T` (fla's tests: 128 sequences of 32 tokens and one
  of 8192; serving and training batches of up to some 10^5 tokens).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes on the proved path
(variable-length, initial state, bias, SiLU) for `D` = 128, 200, 256, 2048, 6144 and 8192, `W` = 2, 3
and 4, a one-token sequence, 129 packed sequences (128 of 32 tokens and one of 8192), sequences of
1, 2 and 62 tokens, and on the defaults of `ShortConvolution` (no bias, no cache) and fixed-length
batches for `D` = 100 to 4096. The largest error of `y` against the `float64` reference,
relative to `1 + |ref|`, was 3.7e-3 (`bfloat16`'s unit roundoff is 2^-8 ≈ 3.9e-3).
