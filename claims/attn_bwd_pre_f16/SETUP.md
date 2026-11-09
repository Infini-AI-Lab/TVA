# Setup of `attn_bwd_pre_f16`

From Triton 3.8.0's tutorial `python/tutorials/06-fused-attention.py` (fused attention, Flash
Attention v2), the kernel `_attn_bwd_preprocess` (`source/06-fused-attention.py:248-262`), launched by
`_attention.backward`.

## The source

`source/06-fused-attention.py` is the tutorial's file with three changes, all in `_attn_fwd` and its
launch (another kernel of the file, which this kernel's launches do not run):

1. The `@triton.autotune(...)` decorator of `_attn_fwd` is removed.
2. In `_attention.forward`, right before the launch of `_attn_fwd`, the host calls
   `_host_descriptor_pre_hook` with the values the autotuner would give it.
3. That launch passes the meta-parameters and launch options of the config the tutorial's test
   uses: `BLOCK_M=128, BLOCK_N=64`, `num_stages=2`, `num_warps=4`.

`_attn_bwd_preprocess`, `_attn_bwd`, their helpers and their launches in `_attention.backward` are
the tutorial's. The file also holds the forward kernel, the backward kernel `_attn_bwd`, the host
class `_attention`, the test `test_op` and the benchmark `bench_flash_attention`.

## The config

Launched directly: `_attn_bwd_preprocess` is not autotuned. `_attention.backward` launches it
(lines 600–607) on the grid `(N_CTX // PRE_BLOCK, BATCH * N_HEAD)` with `BLOCK_M = PRE_BLOCK = 128`,
`HEAD_DIM = ctx.HEAD_DIM` and no launch option.

## The launches

The launches the file's test and benchmark make in `mode = "bwd"`, shapes as variables. Every one
is a `float16` launch: the test skips `bwd` with the `triton-fp8` provider (line 643), and the
benchmark's `triton-fp8` provider in `bwd` mode converts nothing (its conversion to `float8e5`,
line 747, is for `fwd` only), so it makes the same `float16` launches as `triton-fp16`:
- `o` (the forward's output) and `do` `float16`, `(BATCH, N_HEAD, N_CTX, HEAD_DIM)`, contiguous
  (`do.is_contiguous()` and equal strides are asserted, lines 585–586); `delta` a fresh `float32`
  `(BATCH, N_HEAD, N_CTX)` (`torch.empty_like(M)`);
- the test (`test_op`, lines 631–695): `Z ∈ {1, 4}`, `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`,
  `HEAD_DIM ∈ {64, 128}`, `causal` both (this kernel does not read it), `q`, `k`, `v` drawn
  `normal(0, 0.5)`, `dout = randn_like(q)`, `sm_scale = 0.5`;
- the benchmark (`bench_flash_attention`, lines 705–775): `BATCH = 4`, `N_HEADS = 32`,
  `N_CTX = 2^10 … 2^14`, `HEAD_DIM ∈ {64, 128}`, `causal` both, `randn` inputs, `sm_scale = 1.3`;
- `HEAD_DIM = 256`, which the forward admits (line 512): the file's path runs this kernel there
  (the backward kernel launched after it is then rejected on the host for lack of shared memory);
- every `N_CTX` the backward admits (`N_CTX % 128 == 0`, line 599) at those head counts, up to the
  kernel's `int32` limit: its offsets `off_hz * HEAD_DIM * N_CTX` and `off_m * HEAD_DIM` must stay
  below `2^31` (`(BATCH·N_HEAD − 1)·HEAD_DIM·N_CTX < 2^31`), so at the test's `4·48` heads of head
  dim 128 the first `N_CTX` left out is `87,936`; the file's largest is `16,384`.

The forward admits `HEAD_DIM ∈ {16, 32}` too, but there its kernel fails a static assertion and the
file's path never reaches this kernel: those are not part of the setup (the claim still covers
them).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs the 48 test launches through the host's
`attention(...).backward(dout)`, and the benchmark's 10 sizes, the head dims 16, 32 and 256 and the
edges of the claim through the launch line of lines 600–607. Against `Post`'s exact reference, the
worst error / tolerance is `0.053` on the test's launches, `0.065` on the benchmark's sizes and
`0.063` at `HEAD_DIM = 16`; every row with finite inputs is finite, and the inputs and the
bytes past `delta` are unchanged. Wrong outputs (`delta` rounded to `float16`, a column dropped or
doubled, the next row of `do`) fail `Post`.
