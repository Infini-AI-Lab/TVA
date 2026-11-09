# Setup of `ssd_chunk_state_bf16`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_state.py:217`
`_chunk_state_fwd_kernel`, launched by `_chunk_state_fwd` (`ssd_chunk_state.py:686-755`) as
`_mamba_chunk_scan_combined_fwd` calls it (`ssd_combined.py:133`).

## The source

`source/ssd_chunk_state.py` holds:
- `_chunk_state_fwd_kernel`, the `@triton.jit` kernel (lines 77-224; upstream
  `ssd_chunk_state.py:216-363`, a kernel line here is the upstream line minus 139): for each
  physical chunk `c` of a packed prefill batch, head `h` and tile of `headdim × dstate`, the
  chunk's state `states[0, c, h, p, n] = Σ_t x[0, t, h, p] · B[0, t, g, n] · exp(dA_cumsum[0, h, c,
  chunk_size − 1] − dA_cumsum[0, h, c, t − c·chunk_size]) · dt[0, h, c, t − c·chunk_size]` over the
  chunk's tokens `t` of the request running at its last token (`seq_idx`), `g = h // (nheads //
  ngroups)`. The kernel scales `B` in `float32`, rounds it to `bfloat16` and accumulates `x^T ·
  B_scaled` with `tl.dot` in `float32`, `BLOCK_SIZE_K` tokens per step;
- `chunk_state_fwd`, the host, adapted from SGLang's `_chunk_state_fwd` (lines 227-303);
- `make_inputs` (the layer's inputs: `x` and `B` as views of one `bfloat16` row per token, or
  contiguous as in SGLang's tests; `dt`, `dA_cumsum` as `_chunk_cumsum_fwd` lays them out;
  `seq_idx`) and `reference` (the chunk states in `float64`), test helpers;
- the test under `if __name__ == "__main__":`.

Changes from the code base: `@triton.autotune` removed (below); the defaults of the kernel's
parameters `BLOCK_SIZE_M: tl.constexpr = 16`, `BLOCK_SIZE_N: tl.constexpr = 16`, `BLOCK_SIZE_K:
tl.constexpr = 16` removed (upstream lines 258-260), the host passing all three; no other line of
the kernel changes. In the host: one configuration passed explicitly, the grid a tuple, and the
starred `seq_idx` stride group of the launch written out as two arguments; the asserts, the
allocation of `states`, the argument order (the `dt` and `dA_cumsum` strides in the order `0, 2, 1,
3`, `B.stride(-1)`), the device context and the flag are as in the wrapper. The kernel reads only
the module-level names `triton` and `tl`; the file's other kernels (`_chunk_cumsum_fwd_kernel`,
whose comment mentions `tl.clamp`, `_chunk_state_varlen_kernel`) are not part of `source/`.

## The config

Removed: `@triton.autotune(configs=[14 configs], key=["hdim", "dstate", "chunk_size"],
prune_configs_by=prune_oversized_tiles(...), **autotune_cache_kwargs)` (`ssd_chunk_state.py:128-215`).
The host passes `BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 64`, `num_stages = 2`,
`num_warps = 4`, the fifth config (`ssd_chunk_state.py:152-156`, listed for `dstate = 128`). On the
H200, for the layer's shapes with `chunk_size = 256` and 2048 to 8192 tokens, it was the fastest of
the configs the pruning keeps for 128 heads × `headdim` 64 or 80 × `dstate` 128 (8 groups), for
`dstate` 256, and within 2% of the fastest for 80 heads × 64 × 128 (one group); for 64 heads × 64 ×
128 another config was faster. The autotuner's pruning would not pick it for SGLang's small test
shapes (`headdim` 5, `chunk_size` 16); the claim fixes this one config for every launch by
convention, and the kernel's meaning does not depend on the config. Grid
`(cdiv(headdim, 64)·cdiv(dstate, 128), batch·nchunks, nheads)`.

Flags, as SGLang's Mamba-2 prefill sets them (`python/sglang/srt/layers/attention/mamba/mamba.py:
577-644`, through `mamba_chunk_scan_combined` and `_mamba_chunk_scan_combined_fwd`,
`python/sglang/kernels/ops/mamba/triton_ops/ssd_combined.py:127-133`):
- `HAS_SEQ_IDX = True`: SGLang's only serving caller (`mamba.py:616`) always passes `seq_idx`,
  which is built for every extend batch
  (`python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:268-275`). The launch without
  `seq_idx` (`HAS_SEQ_IDX = False`) is compiled only by SGLang's unit tests and is not part of the
  setup; their shapes are covered here with the `seq_idx` of one request;
- `states_in_fp32 = True` (`ssd_combined.py:133`): `states` is `float32`, allocated by the host.

## The launches

The claim covers the host's launches with `HAS_SEQ_IDX = True`, `batch = 1`, `x` and `B`
`bfloat16`, `dt`, `dA_cumsum`, `states` `float32`, with shapes as variables:
- the prefill tokens of `num_requests ≥ 1` requests packed one after another, `seqlen` their total
  (each ≥ 1); `nchunks = cdiv(seqlen, chunk_size)` physical chunks, which may cross request
  boundaries; `seq_idx`: `[1, seqlen]`, `int32`, the request of each token
  (`torch.repeat_interleave(arange(num_requests), seqlens)`, `mamba2_metadata.py:268-275`);
- `x`: `[1, seqlen, nheads, headdim]` and `B`: `[1, seqlen, ngroups, dstate]`, any strides (within
  `int32`): in SGLang's layer both are views of the convolution's output, one row of `conv_dim =
  nheads·headdim + 2·ngroups·dstate` elements per token (`mamba.py:592-604, 616-622`; `B.stride(-1)
  = 1`, as the caller ensures, `ssd_combined.py:91-92`); contiguous in SGLang's tests;
- `dt`, `dA_cumsum`: `[1, nheads, nchunks, chunk_size]`, contiguous, as `_chunk_cumsum_fwd`
  (`ssd_chunk_state.py:634-683`) gives them with the layer's `dt_softplus = True`, `dt_limit = (0,
  inf)`: `dt ≥ 0`, `0` past `seqlen`; `dA_cumsum` the cumulative sum of `A·dt` (`A < 0`) within each
  physical chunk, across request boundaries, constant past `seqlen`;
- `states`: `[1, nchunks, nheads, headdim, dstate]`, `float32`, allocated by the host;
- the code base's sizes: `chunk_size` a power of two (asserted, `ssd_combined.py:77`), 256 in the
  model configs (Mamba-2 `configs/mamba2.py`; Nemotron-H `configs/nemotron_h.py:275-288`, default
  256; Granite 4.0 hybrid `configs/granitemoehybrid.py:160-166`; Falcon-H1
  `configs/falcon_h1.py:160-166`), 8 to 256 in SGLang's tests; per tensor-parallel rank `nheads`
  up to 128, `headdim` 64 or 80 (and down to 5 in the tests), `dstate` 128 or 256 (5 to 128 in the
  tests), `ngroups` 1 or 8 (`ngroups = nheads` in the tests); up to the prefill token budget
  (`chunked_prefill_size`, 8192 by default on an H200, `python/sglang/srt/arg_groups/memory_hook.py:
  146-157`; larger when raised, and with `--chunked-prefill-size -1` one request of any length,
  `python/sglang/srt/managers/schedule_policy.py:1627-1632`, e.g. a few hundred thousand tokens,
  where `x.stride(0) = seqlen·conv_dim` is an `int64` argument).

SGLang's tests (`test/registered/layers/mamba/test_mamba_ssm_ssd.py`): one request of 112 or 128
tokens with chunks of 16 or 32, `nheads` 3 to 32, `headdim = dstate` 5 to 128 (lines 219-228,
the `bfloat16` cases); packed batches with chunks of 8, 128 and 256 (lines 292-326, 397-405), run
here in `bfloat16`.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 24 launches (and the negative tests' two), all holding: the source's ten test cases
(SGLang's test shapes: one request of 112 or 128 tokens, packed batches of 2 to 4 requests with
chunks of 8 to 256; the layer's shapes: 80 heads × 64 × 128 with one group, 128 heads × 64 × 128
and × 80 × 128 with eight groups, 64 heads × 64 × 256, up to 8192 tokens and 41 requests), six edges
of `Inst.ok` (32768 tokens at 128 heads; 200 one-token requests; chunks of 1 and 2; `headdim =
dstate = 1`; `headdim` 65 and `dstate` 129), one with `x.stride(0)` and `B.stride(0)` above `2^31`
(`int64` arguments), and seven of `Vals` (strong decay `dA × 200`, no decay,
`dt × 1000`, a non-monotone `dA_cumsum`, values near `10^15` and near `10^-20`, and a coherent
worst case for the `bfloat16` rounding). The worst error / tolerance is `0.992` (the coherent worst
case; `0.98` on the others, from elements dominated by one token); every element is finite, `Vals`
holds, the byte frame holds, and the host's `seq_idx` equals the claim's `reqOf` on every launch.
Eight negative tests are rejected: one element moved by 1.01 × its tolerance, the launch without
`seq_idx`, `B`'s groups swapped, and the states of kernels that drop the last token of each chunk,
decay to position `chunk_size − 2`, read `dt` one position off, drop a middle token of each chunk,
or drop the first token of each chunk's request segment under a weak decay (`dA × 0.01`); a
kernel that drops that token under the inputs' own decay (weight `e^-20.1` to `e^-9.5` in chunks 0
and 1, `e^-12.8` to `e^-7.5` in chunk 2, where it is token 600 at position 88) is accepted.
