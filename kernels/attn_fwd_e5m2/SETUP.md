# Setup of `attn_fwd_e5m2`

Set up by the coordinator on 2026-10-31, before any prover started (protocol §3, "Setting up a
kernel folder", library issue L116).

## The config

Tutorial 06 (`tutorials/06-fused-attention.py`, Triton 3.8.0) autotunes `_attn_fwd` over
`list(filter(keep, configs))` (lines 135-153), pruned by `prune_invalid_configs` (lines 156-165),
with the decorator at lines 176-177. The config chosen is the one the tutorial's own test uses
(lines 142-146: under pytest, `configs` is this single config, "for reproducibility"):

```python
triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
```

It is also in the full list (lines 135-141: `BM = 128`, `BN = 64`, `s = 2`, `w = 4`). `keep` keeps
it on the H200, and `prune_invalid_configs` keeps it whenever `N_CTX ≥ 128`.

Its `pre_hook` is `_host_descriptor_pre_hook` (lines 113-125). When the launch is given host
`TensorDescriptor`s, it sets their `block_shape`s from `BLOCK_M`, `BLOCK_N` and `HEAD_DIM`.
Otherwise it does nothing.

## What changed from the tutorial's source

`source/06-fused-attention.py` is the tutorial's file with three changes, and nothing else:

1. The `@triton.autotune(configs=list(filter(keep, configs)), key=[...], prune_configs_by=...)`
   decorator of `_attn_fwd` (the tutorial's lines 176-177) is removed. `_attn_fwd` is now a plain
   `@triton.jit` function, launched directly.
2. In `_attention.forward`, right before the launch, the host does what the config's `pre_hook`
   does, as the autotuner does before its launch: it calls `_host_descriptor_pre_hook` with the
   values the autotuner would give it (`BLOCK_M`, `BLOCK_N`, `HEAD_DIM`, `FP8_OUTPUT` and the four
   descriptor arguments `desc_q`, `desc_k`, `desc_v`, `desc_o`).
3. The launch of `_attn_fwd` in `_attention.forward` passes the config's meta-parameters and launch
   options: `BLOCK_M=128, BLOCK_N=64` and `num_stages=2, num_warps=4`.

The config list `configs`, `keep` and `prune_invalid_configs` are left in the file. Nothing uses
them now. `_attn_bwd_preprocess` and `_attn_bwd` were never autotuned, and their launches in
`_attention.backward` are unchanged.

## The folder

`attn_fwd_e5m2` is for `_attn_fwd`'s `float8e5` run (`FP8_OUTPUT=True`), as in the test's `fwd`
mode with the `triton-fp8` provider (the tutorial's lines 666-672) and in the benchmark's
`triton-fp8` lines (lines 742-747). `q`, `k` and the output `o` are `float8e5`. `v` is `float8e5`
and is stored transposed: it is the `(Z, H, N_CTX, HEAD_DIM)` view of a contiguous
`(Z, H, HEAD_DIM, N_CTX)` tensor. The `float16` run is `attn_fwd_f16`.
