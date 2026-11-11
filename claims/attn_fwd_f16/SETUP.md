# Setup of `attn_fwd_f16`

From Triton 3.8.0's tutorial 06, `python/tutorials/06-fused-attention.py`: the forward kernel
`_attn_fwd` (tutorial lines 176–247) and its helpers, with the host, test and benchmark of the same
file.

## The source

`source/06-fused-attention.py` is the tutorial's file with three changes, and nothing else:

1. The `@triton.autotune(configs=list(filter(keep, configs)), key=[...], prune_configs_by=...)`
   decorator of `_attn_fwd` (the tutorial's lines 176–177) is removed. `_attn_fwd` is a plain
   `@triton.jit` function, launched directly.
2. In `_attention.forward`, right before the launch, the host does what the config's `pre_hook`
   does, as the autotuner does before its launch: it calls `_host_descriptor_pre_hook` with
   `BLOCK_M`, `BLOCK_N`, `HEAD_DIM`, `FP8_OUTPUT` and the four descriptor arguments.
3. The launch of `_attn_fwd` passes the config's meta-parameters and launch options:
   `BLOCK_M=128, BLOCK_N=64` and `num_stages=2, num_warps=4`.

The config list `configs`, `keep` and `prune_invalid_configs` stay in the file, unused.
`_attn_bwd_preprocess` and `_attn_bwd` (other folders' kernels) are unchanged. The file holds the
kernel `_attn_fwd` (lines 176–245) with its helpers `_attn_fwd_inner` (47–110) and
`_maybe_make_tensor_desc` (168–173), the host `_attention.forward` (505–580), the test `test_op`
(631–695) and the benchmark (698–780).

## The config

The tutorial's own test config (lines 142–146: under pytest, `configs` is this single config):
`triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4,
pre_hook=_host_descriptor_pre_hook)`. It is in the full list too, `keep` keeps it on the H200, and
`prune_invalid_configs` keeps it whenever `N_CTX ≥ 128`. The launch passes it (The source); the
`pre_hook` sets the host `TensorDescriptor`s' block shapes (`[128, HEAD_DIM]` for `q` and `o`,
`[64, HEAD_DIM]` for `k` and `v`) and does nothing when the tensors are passed. The
folder is `_attn_fwd`'s `float16` run: `q`, `k`, `v`, `o` are `float16` (`FP8_OUTPUT=False`); the
`float8e5` run is `attn_fwd_e5m2`. The device is the H200 (`is_hopper()`, so `IS_HOPPER=True`, no
`maxnreg`; `supports_host_descriptor()`).

## The launches

Every launch of `_attn_fwd` the file makes on `float16` inputs, through
`attention(q, k, v, causal, sm_scale, warp_specialize)` (`_attention.forward`, lines 505–580):
`q`, `k`, `v` fresh contiguous `float16` tensors of shape `[Z, H, N_CTX, HEAD_DIM]`,
`o = torch.empty_like(q)`, `M = torch.empty((Z, H, N_CTX), float32)`, the grid
`(cdiv(N_CTX, 128), Z·H, 1)`, `STAGE = 3` when causal and `1` otherwise. The setup is the
host-descriptor path, `warp_specialize = False`: on the H200 the host then passes host
`TensorDescriptor`s (line 523), and the allocator it sets with `triton.set_allocator` (line 546) is
never called.

- **The test** (`test_op`, lines 631–695). The `float16` forward runs in the `bwd` mode (the `fwd`
  mode skips `triton-fp16`; the `triton-fp8` provider's `bwd` mode is skipped): `Z ∈ {1, 4}`,
  `H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`, `HEAD_DIM ∈ {64, 128}`, `causal ∈ {False, True}`,
  `warp_specialize = False` (off Blackwell), `sm_scale = 0.5`, inputs `normal(0, 0.5)` (48 launches).
- **The benchmark** (lines 698–780, `bench_flash_attention`). `BATCH = Z = 4`, `H = 32`,
  `N_CTX ∈ {2^10, …, 2^14}`, `HEAD_DIM ∈ {64, 128}`, `causal ∈ {True, False}`, `sm_scale = 1.3`,
  inputs `torch.randn`. The `triton-fp16` line runs the `float16` forward in both modes, and so does
  the `triton-fp8` line in the `bwd` mode (its inputs are converted only in the `fwd` mode).
  The setup is its launches with `warp_specialize = False`: all of them for the causal forward and
  the `bwd` mode, and one of the two for the non-causal `fwd`-mode forward, which `enable_ws`
  (line 713) runs with `warp_specialize` `False` and `True` on Hopper.

Value ranges: the inputs are finite, `|v|` far below `2^15`, and every `sm_scale·Σ_e |q_e·k_e|` is
below about `23` (test) and `240` (benchmark) at `HEAD_DIM = 128` (README, Specification).

Not part of the setup: the benchmark's launches with `warp_specialize = True` (the host passes the
raw tensors and the kernel builds its descriptors; out of scope: a tensor descriptor built inside
the kernel; README, Exclusions), `HEAD_DIM ∈ {16, 32}` (the wrapper accepts them; line 188's
`tl.static_assert(BLOCK_N <= HEAD_DIM)` rejects them at compile time), `N_CTX` not a multiple of
`128`, other devices and configs, the `float8e5` run.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, torch, an H200) runs every test launch (48) and every `float16`
benchmark launch with `warp_specialize = False` (20: `N_CTX = 2^10 … 2^14`, both `HEAD_DIM`, both
modes), with the host's own input distributions, and compares `o` and `M` with the float64
reference. All are within the claim's tolerance: the largest error / tolerance for `o` is `0.459`
on the test's launches and `0.112` on the benchmark's, and for `M` `0.0053` (on the benchmark's
launches); every element of `o`
and `M` is written and finite, the guard bands around them are unchanged, the inputs are unchanged,
and no other tensor is allocated (the allocator the host sets is never called).
