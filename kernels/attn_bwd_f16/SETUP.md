# Setup of `attn_bwd_f16`

Set up by the coordinator on 2026-10-31, before any prover started (protocol §3, "Setting up a
kernel folder", library issue L116).

## The kernel

`_attn_bwd` is not autotuned. `_attention.backward` launches it directly (the tutorial's lines
603-616) on the grid `(N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)`, with `BLOCK_M1=32, BLOCK_N1=128,
BLOCK_M2=128, BLOCK_N2=32, BLK_SLICE_FACTOR=2` and `num_warps=4, num_stages=5`. No config was
chosen for it.

## The source

`source/06-fused-attention.py` is the same file as the one `attn_fwd_f16` and `attn_fwd_e5m2` are
set up from. Those folders need its changes, and one file serves all four folders of tutorial 06.
It is the tutorial's file with three changes, all in `_attn_fwd` and its launch:

1. The `@triton.autotune(...)` decorator of `_attn_fwd` (the tutorial's lines 176-177) is removed.
2. In `_attention.forward`, right before the launch of `_attn_fwd`, the host calls
   `_host_descriptor_pre_hook` with the values the autotuner would give it.
3. That launch passes the meta-parameters and launch options of the config chosen for `_attn_fwd`
   (the one the tutorial's test uses, lines 142-146): `BLOCK_M=128, BLOCK_N=64`, `num_stages=2`,
   `num_warps=4`.

Nothing else changed: `_attn_bwd_preprocess`, `_attn_bwd`, their helpers and their launches in
`_attention.backward` are the tutorial's.

## The folder

`attn_bwd_f16` is for the tutorial's run of `_attn_bwd`: `Q`, `K` (the host's `arg_k`, which is `k`
scaled by `sm_scale / ln 2`), `V`, `DO`, `DQ`, `DK` and `DV` are `float16`, and `M` and `D` are
`float32`. The tutorial runs no other: its backward pass does not support `float8e5` ("Backward
pass with FP8 is not supported", line 638).
