# Setup of `attn_bwd_f16`

From Triton 3.8.0's tutorial `python/tutorials/06-fused-attention.py` (fused attention, Flash
Attention v2), the kernel `_attn_bwd` (`source/06-fused-attention.py:366-500`) with its helpers
`_attn_bwd_dkdv` (266-315) and `_attn_bwd_dq` (319-363), launched by `_attention.backward`.

## The source

`source/06-fused-attention.py` is the tutorial's file with three changes, all in `_attn_fwd` and its
launch (the forward kernel, which this kernel's launches do not run):

1. The `@triton.autotune(...)` decorator of `_attn_fwd` is removed.
2. In `_attention.forward`, right before the launch of `_attn_fwd`, the host calls
   `_host_descriptor_pre_hook` with the values the autotuner would give it.
3. That launch passes the meta-parameters and launch options of the config the tutorial's test
   uses: `BLOCK_M=128, BLOCK_N=64`, `num_stages=2`, `num_warps=4`.

`_attn_bwd_preprocess`, `_attn_bwd`, their helpers and their launches in `_attention.backward` are
the tutorial's.

## The config

Launched directly: `_attn_bwd` is not autotuned. `_attention.backward` launches it (lines 608–621)
on the grid `(N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)` with `BLOCK_M1=32, BLOCK_N1=128,
BLOCK_M2=128, BLOCK_N2=32, BLK_SLICE_FACTOR=2`, `HEAD_DIM=ctx.HEAD_DIM`, `CAUSAL=ctx.causal` and
`num_warps=4, num_stages=5`.

## The launches

The launches the file's test and benchmark make in `mode = "bwd"`, shapes as variables. Every one
is a `float16` launch: the test skips `bwd` with the `triton-fp8` provider (line 643), and the
benchmark's `triton-fp8` provider in `bwd` mode converts nothing (its conversion to `float8e5`,
line 747, is for `fwd` only), so it makes the same `float16` launches as `triton-fp16`:
- `q`, `arg_k = k * (sm_scale * RCP_LN2)` (line 597), `v`, `do` `float16`
  `(BATCH, N_HEAD, N_CTX, HEAD_DIM)`, contiguous (`do.is_contiguous()` and equal strides are
  asserted, lines 585–586); `dq`, `dk`, `dv` fresh `float16` (`torch.empty_like`); `M` (the
  forward's base-2 log-sum-exp) and `delta` (`_attn_bwd_preprocess`'s `Σ o·do`) `float32`
  `(BATCH, N_HEAD, N_CTX)`, contiguous;
- the test (`test_op`, lines 631–695): `Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`,
  `HEAD_DIM ∈ {64, 128}`, `causal ∈ {False, True}`, `q`, `k`, `v` drawn `normal(0, 0.5)`,
  `dout = randn_like(q)`, `sm_scale = 0.5`;
- the benchmark (`bench_flash_attention`, lines 705–775): `BATCH = 4`, `N_HEADS = 32`,
  `N_CTX = 2^10 … 2^14`, `HEAD_DIM ∈ {64, 128}`, `causal ∈ {False, True}`, `randn` inputs,
  `sm_scale = 1.3`.

The forward admits `HEAD_DIM ∈ {16, 32, 256}` too (line 512), but on the H200 the file's own path
never runs this kernel there: at `16` and `32` the forward kernel fails a static assertion, and at
`256` this kernel's launch is rejected on the host (it needs more shared memory than the device
has). They are not part of the setup (the claim still covers them: a rejected launch has no
outcome).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs all 48 test launches and the benchmark's 20 sizes and modes (at
`BATCH = 1`, `N_HEAD = 2`, plus one launch at `4·32` heads: the full benchmark tensors exceed the
4 GB budget) through the host's `attention(...).backward(dout)`, and the head dims 16 and 32 through
`_attention.backward` on host-computed forward tensors. On 411 heads (all heads of the small
launches, four of the large ones), every output is the `float16` rounding of a value within `Post`'s
tolerance of the exact gradient: the float32 part of the error reaches `0.73` of the tolerance at
most, and the worst error / (tolerance plus the store's rounding) is `0.98` (outputs near zero, where
the store's `float16` rounding, at most `2^-25`, is most of it). `HEAD_DIM = 256` is rejected on
the host (out of shared memory). Inputs are unchanged and the bytes past `dq`, `dk`, `dv`, `delta`
keep their sentinel.
