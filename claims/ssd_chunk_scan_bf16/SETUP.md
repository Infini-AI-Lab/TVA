# Setup of `ssd_chunk_scan_bf16`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/mamba/triton_ops/ssd_chunk_scan.py:162`
`_chunk_scan_fwd_kernel`, launched by `_chunk_scan_fwd` (`ssd_chunk_scan.py:560`) as SGLang's Mamba-2
mixer calls it through `mamba_chunk_scan_combined` (`python/sglang/srt/layers/attention/mamba/mamba.py:616-642`,
`kernels/ops/mamba/triton_ops/ssd_combined.py:177-191`).

## The source

`source/ssd_chunk_scan.py` holds:
- `_chunk_scan_fwd_kernel`, the `@triton.jit` kernel (lines 92–488; upstream `ssd_chunk_scan.py:161-557`):
  for each logical chunk (the part of a physical chunk of `chunk_size` tokens that belongs to one
  request), head and tile of `64 × 64` (tokens × channels), the output of Mamba-2's chunked scan:
  the state entering the logical chunk (the request's initial state when the chunk starts it, `0`
  without initial states, else the chunk's entering state from state passing) read out by `C` and
  decayed by
  `exp(dA_cumsum)`, plus the causal intra-chunk term `Σ_s cb·exp(dA_t − dA_s)·dt_s·x_s`, plus
  `D·x`, stored in `bfloat16`;
- `chunk_scan_fwd`, the host, adapted from SGLang's `_chunk_scan_fwd` (lines 491–644; upstream
  `ssd_chunk_scan.py:560-700`);
- `query_start_loc_to_chunk_indices_offsets`, SGLang's torch-only metadata helper
  (`python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:111-192`), and `make_inputs` and
  `reference` (the output in `float64` from `query_start_loc`, with or without initial states),
  test helpers;
- the test under `if __name__ == "__main__":`, each case with and without initial states.

Changes from the code base: `@triton.autotune` removed (below); the defaults of the kernel's
parameters `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` (`tl.constexpr = 16`, upstream lines
237–239) removed, the host passing them; no other line of the kernel changes. The module's imports
of `packaging.version` and of SGLang's autotune helpers and its `TRITON_22` (upstream lines 14, 16,
18) are removed; the host passes `IS_TRITON_22=True`, `TRITON_22`'s value under Triton 3.8.0. In
the host: one configuration passed explicitly, the grid a tuple, and the starred stride groups of
the launch (`seq_idx`'s and the initial states', upstream lines 670, 680–689) written out one
argument each; the asserts, the argument order (with the `dt` and `dA_cumsum` strides `(0, 2, 1,
3)`), the flags and `out_x` are as in the wrapper. `out` is given by the caller, as in SGLang
(a slice of the layer's preallocated output, `mamba.py:549-565`).

## The config

Removed: `@triton.autotune(configs=[24 configs], key=["chunk_size", "hdim", "dstate",
"IS_CAUSAL"], prune_configs_by=prune_oversized_tiles(...), **autotune_cache_kwargs)`
(`ssd_chunk_scan.py:21-160`). The host passes `BLOCK_SIZE_M = 64`, `BLOCK_SIZE_N = 64`,
`BLOCK_SIZE_K = 32`, `num_warps = 4`, `num_stages = 1` (the seventh config,
`ssd_chunk_scan.py:61-65`). Timed on the H200 with SGLang's autotuner, on launches with initial
states, 8192 tokens: `[4096, 4096]` tokens at 64 heads × 64, chunks of 256; `[3000, 2000, 3192]`
at 128 × 64, chunks of 256; `[8192]` at 80 × 80, chunks of 128; `[6000, 2192]` at 128 × 64, one
group, chunks of 256 (`dstate` 128, 8 groups unless stated). It picked this config on each, 7% to
11% ahead of the next (`BLOCK_SIZE_K = 64`) and 3 to 5 times faster than the configs with more
stages; other timings rank it differently (without initial states it leads; with them
`(128, 64, 32)` or `(64, 64, 64)` can come first). The autotune key (`chunk_size`, `hdim`,
`dstate`, `IS_CAUSAL`) leaves out `HAS_INITSTATES`, so in SGLang the first launch per key decides
the config for both values of the flag. The function the kernel computes does not depend on the
config (only the order of the dots' roundings does); the claim fixes this one.
Grid `(cdiv(chunk_size, 64)·cdiv(headdim, 64), N, nheads)`, `N = len(chunk_offsets)` with initial
states, `nchunks` without.

Flags, as SGLang's Mamba-2 prefill sets them (`mamba.py:616-642`, through
`_mamba_chunk_scan_combined_fwd`, `ssd_combined.py:177-191`):
- `HAS_SEQ_IDX = True`: `seq_idx` is always built for an extend batch (`mamba2_metadata.py:268-275`);
- `HAS_INITSTATES`, both values (`prep_initial_states = any(extend_prefix_lens > 0)`,
  `mamba2_metadata.py:260-262`): `True` when some request of the batch has a cached prefix:
  `initial_states` is passed (requests without one get zeros, `mamba.py:606-612`) with
  `chunk_indices`, `chunk_offsets` (`mamba2_metadata.py:281-286`); `False` when none has (fresh
  prompts, the most common prefill): `initial_states = None`, and the host drops the metadata
  (`chunk_meta_num = 0`, the grid's `y` `nchunks`);
- `HAS_D = True`, `D_HAS_HDIM = False`: `D = self.D`, a `[nheads]` parameter (`mamba.py:412, 625`);
- `HAS_Z = False`: `z = None` (`mamba.py:626`; the gate is applied by the gated RMS norm after the
  scan);
- `IS_CAUSAL = True` (the wrapper's literal), `IS_TRITON_22 = True`;
- `BLOCK_SIZE_DSTATE = max(next_power_of_2(dstate), 16) ≤ 128`, so the single-`tl.dot` branch
  (lines 347–361) runs; `dstate > 128` (the larger Falcon-H1 models' 256) takes the other branch
  (lines 362–381) and is not part of the setup.

Dtypes: `x`, `C` in the model's `bfloat16` (slices of the convolution's output); `D` in
`bfloat16` (created under the model's default dtype, `srt/model_loader/loader.py:838`); `cb` in
`float32` (`_bmm_chunk_fwd(..., output_dtype=torch.float32)`, `ssd_combined.py:165`); `dt` and
`dA_cumsum` in `float32` (`_chunk_cumsum_fwd`, `ssd_chunk_state.py:642-647`); `states` in
`state_dtype = ssm_state.dtype`, `float32` by default (`mamba2_state_dtype`,
`srt/configs/mamba_utils.py:47-105`), and `initial_states` from the `float32` state cache.

## The launches

The claim covers the host's launches with the flags above, with shapes as variables:
- `batch = 1`: the prefill tokens of `num_requests ≥ 1` requests packed one after another,
  `seqlen` the sum of their lengths (each ≥ 1); `nchunks = cdiv(seqlen, chunk_size)`;
- `x`: `[1, seqlen, nheads, headdim]` and `C`: `[1, seqlen, ngroups, dstate]`, `bfloat16`, each a
  slice of a `[seqlen, conv_dim]` row-major buffer (`conv_dim = nheads·headdim +
  2·ngroups·dstate`, `mamba.py:499-507, 604`; token stride `conv_dim`, last dimension contiguous),
  or contiguous (SGLang's tests);
- `cb`: `[1, nchunks, ngroups, chunk_size, chunk_size]`, `float32`, contiguous (`C_i·B_j` per
  chunk, zero across requests; entries past `seqlen` are never written by `_bmm_chunk_fwd` and may
  hold anything);
- `dt`, `dA_cumsum`: `[1, nheads, nchunks, chunk_size]`, `float32`, contiguous;
- `states`: `[1, nchunks, nheads, headdim, dstate]`, `float32`, contiguous;
- `D`: `[nheads]`, `bfloat16`;
- `initial_states`: `[num_requests, nheads, headdim, dstate]`, `float32`, contiguous, or `None`;
- `seq_idx`: `[1, seqlen]`, `int32` (`0, …, 0, 1, …`); with initial states `chunk_indices`,
  `chunk_offsets`: `[N]`, `int32`, from `_query_start_loc_to_chunk_indices_offsets`:
  `N = cdiv(seqlen, chunk_size)` plus one per request that starts inside a chunk (`None` without);
- `out`: `[1, seqlen, nheads, headdim]`, `bfloat16`, contiguous, given by the caller;
- the code base's sizes: `chunk_size` a power of two (asserted, `ssd_combined.py:77`): 128 or 256
  in the Mamba-2 hybrids (Mamba-2, Nemotron-H, Nemotron Nano 2, Granite 4.0-H), 8 to 256 in
  SGLang's tests (`test/registered/layers/mamba/test_mamba_ssm_ssd.py:289-327, 397-405`, with
  request lengths such as `(16, 2, 8, 13)`, `(270, 88, 212, 203)`, `(138, 225)`); `headdim` 64
  (Mamba-2, Nemotron-H, Granite 4.0-H) or 80 (Nemotron Nano 2), 5 to 32 in the tests; `dstate`
  128 in the models, 5 to 32 in the tests; `ngroups` 1 or 8 (after tensor parallelism, down to
  1), `= nheads` in the tests; `nheads` up to 128 per tensor-parallel rank; up to the prefill
  token budget (8192 tokens per forward pass on the H200 with chunked prefill, the default,
  `memory_hook.py:146-153`).

The models that run this kernel on the H200: Mamba-2, Nemotron-H, Nemotron Nano 2 and Granite
4.0-H; Falcon-H1-0.5B (`dstate` 128) and Falcon-H1-Tiny-90M (`dstate` 64) take this branch too,
their layers calling the mixer directly (`falcon_h1.py:341-353`). Their prefill launches, Falcon-H1's
included, are covered.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 76 covered launches, all holding: the source's eleven test
cases (1 to 41 requests, 1 to 8192 tokens, 4 to 64 heads, `headdim` 5 to 80, `dstate` 5 to 128,
`chunk_size` 8 to 256) and 27 edges, each with and without initial states (chunk sizes 2, 16,
1024; `dstate` 1, 17, 100; `headdim` 1, 65, 128; one group for 128 heads; 200 one-token requests;
chunk-aligned starts; seven starts in one chunk; contiguous `x` and `C`; a Nemotron-H and a
Nemotron Nano 2 rank at 8192 tokens and 128 heads; 24 heads in one group at 8192 tokens; 32768
tokens; strong and no decay; a non-monotone `dA_cumsum`; values near `10^15`; states at `2^120`;
subnormal `x` and `C`; states whose `bfloat16` roundings all go one way; `cb` at the `float32`
maximum and NaN above the diagonal; `dt` at the `float32` maximum where `cb` is `0`). The worst
error / tolerance is `0.480`, a sample (the store's half-ulp allows about `0.5`); `Pre`, `Vals` and the frame hold. Eight launches outside
`Vals` (the `float32` range: states or `cb` at the `float32` maximum, decays above `64`, a state
read-out above the `float32` range, `x` near the `bfloat16` maximum) fail `Vals` and give
non-finite outputs. `Post` rejects seven wrong outputs (wrong `chunk_offsets`, an output moved by
twice its tolerance with and without initial states, and four wrong readings of the kernel). The
source's own test passes on its eleven cases with and without initial states (largest error 0.49%
of the largest output).
