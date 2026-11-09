# Setup of `ssd_state_passing_f32`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/mamba/triton_ops/ssd_state_passing.py:34`
`_state_passing_fwd_kernel`.

## The source

`source/ssd_state_passing.py` holds:
- `_state_passing_fwd_kernel`, the `@triton.jit` kernel (`ssd_state_passing.py:33-185`): for each
  head and block of state elements, the state is carried across the physical chunks of a packed
  prefill batch, `s = exp(dA) * s + states[c]`, switching to a request's initial state when a new
  request reaches the end of a chunk and then counting only that request's part of the chunk's
  decay (through `chunk_offsets`); without initial states (`initial_states = None`) the state
  starts from `0` and restarts from `0` there instead; the state entering each chunk is stored, and
  the last one goes to `final_states`;
- `state_passing_fwd`, the host, adapted from SGLang's `_state_passing_fwd`
  (`ssd_state_passing.py:188-281`);
- `query_start_loc_to_chunk_indices_offsets`, SGLang's torch-only metadata helper
  (`python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:111-192`), and `make_inputs` and
  `reference` (the recurrence in `float64` from per-token decays, each request starting from its
  initial state or from `0`), test helpers;
- the test under `if __name__ == "__main__":`, which runs each case with and without initial
  states.

Changes from the code base: `@triton.autotune` removed (below); the default of the kernel's
parameter `BLOCK_SIZE: tl.constexpr = 16` removed (line 74), the host passing `BLOCK_SIZE`; no other
line of the kernel changes. In the host: one configuration passed explicitly, the grid a tuple, and
the starred stride groups of the launch written out one argument each; the asserts, allocations,
argument order and flags are as in the wrapper. Both outputs are allocated fresh by the wrapper
itself.

## The config

Removed: `@triton.autotune(configs=[BLOCK_SIZE 64, 128, 256, 512, 1024, 2048], key=["dim"],
prune_configs_by=prune_oversized_tiles(...), **autotune_cache_kwargs)` (`ssd_state_passing.py:18-32`).
The host passes `BLOCK_SIZE = 1024` (the fifth config), with `triton.Config`'s defaults
`num_warps=4`, `num_stages=3`; on the H200, for `dim` 8192 and 10240 with 64 to 128 heads, 1024 is
the fastest or within 1% of it (256 to 2048 within 10%). Grid `(cdiv(dim, 1024), batch, nheads)`.

Flags, as SGLang's Mamba-2 prefill sets them (`python/sglang/srt/layers/attention/mamba/mamba.py:
606-642`, through `mamba_chunk_scan_combined` and `_mamba_chunk_scan_combined_fwd`,
`kernels/ops/mamba/triton_ops/ssd_combined.py:146-159`):
- `HAS_SEQ_IDX = True`: `seq_idx` is always built for an extend batch (`mamba2_metadata.py:268-275`);
- `IS_CONT_BATCHED = True`: `cu_seqlens = query_start_loc_p` is always passed;
- `HAS_INITSTATES` either value. `True`: `initial_states` is passed when some request of the batch
  has a cached prefix (`prep_initial_states = any(extend_prefix_lens > 0)`,
  `mamba2_metadata.py:260-262`; requests without one get zeros, `mamba.py:607-612`), and then
  `chunk_offsets` is built (`mamba2_metadata.py:280-286`). `False`: when no request of the extend
  batch has a cached prefix (fresh prompts, no radix-cache hit), SGLang passes
  `initial_states = None` and `chunk_offsets = None` (`mamba.py:606-612, 630-632`;
  `ssd_combined.py:146-159`), and the host passes `chunk_meta_num = 0` and the initial-state
  strides `0`.

## The launches

The claim covers the host's launches with `HAS_SEQ_IDX = IS_CONT_BATCHED = True` and
`HAS_INITSTATES` either value, all states `float32`, with shapes as variables:
- `batch = 1`: the prefill tokens of `num_requests ≥ 1` requests packed one after another, `seqlen
  = sum` of their lengths (each ≥ 1); `nchunks = cdiv(seqlen, chunk_size)` physical chunks, which
  may cross request boundaries;
- `states`: `[1, nchunks, nheads, dim]`, `float32`, contiguous (`_chunk_state_fwd(...,
  states_in_fp32=True)`, flattened `headdim × dstate`);
- `dA_cumsum`: `[1, nheads, nchunks, chunk_size]`, `float32`, contiguous: the cumulative sum of
  `A·dt` within each physical chunk, across request boundaries, constant past `seqlen`
  (`_chunk_cumsum_fwd`);
- `initial_states`: `[num_requests, nheads, dim]`, `float32`, contiguous (the cached state or zero,
  `torch.where(has_initial_states, ssm_state[state_indices], 0)`), or `None`;
- `seq_idx`: `[1, seqlen]`, `int32`, the request of each token (`0, …, 0, 1, …`);
- `chunk_offsets`: `[N]`, `int32`, from `_query_start_loc_to_chunk_indices_offsets`
  (`mamba2_metadata.py:111-192`): `N = cdiv(seqlen, chunk_size)` plus one per request that starts
  inside a chunk, each the logical chunk's start offset in its physical chunk (`None` without
  initial states);
- `out_dtype = ssm_state.dtype`: `float32`, SGLang's default for Mamba-2 state
  (`mamba2_state_dtype`, `python/sglang/srt/configs/mamba_utils.py:45-105`; a model config or
  `SGLANG_MAMBA_SSM_DTYPE` can change it); `final_states` is `float32` always;
- `out`: `[1, nchunks, nheads, dim]` and `final_states`: `[1, nheads, dim]`, allocated by the host;
- the code base's sizes: `chunk_size` a power of two (asserted, `ssd_combined.py:80`), 128 or 256
  in Mamba-2 models, 8 to 256 in SGLang's tests (`test/registered/layers/mamba/test_mamba_ssm_ssd.py`,
  with request lengths such as `(16, 20)` and `(270, 88, 212, 203)`); `dim = headdim × dstate`:
  8192 (headdim 64, dstate 128: Mamba-2, Nemotron-H), 10240 (headdim 80: Nemotron Nano 2), down to
  `5 × 16` in the tests; `nheads` up to 128 per tensor-parallel rank; up to the prefill token budget
  (thousands to tens of thousands of tokens).

The models that run this kernel on the H200: Mamba-2, Nemotron-H, Nemotron Nano 2,
Granite-4.0-H and Falcon-H1, whose layers call the mixer directly (`falcon_h1.py:344-352`, through
`MambaMixer2.forward`, `mamba.py:616-644`). Their prefill launches, Falcon-H1's included, are
covered.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 50 launches, all holding: the source's nine test cases
(1 to 41 requests, 1 to 8192 tokens, 4 to 128 heads, `dim` 80 to 10240, `chunk_size` 8 to 256)
and 16 edges (chunk sizes 1, 2 and 1024; `dim` 1, 1025, 2049; 200 one-token requests; aligned
starts; several starts in one chunk; 4096 chunks; 32768 tokens at 128 heads × 8192; strong decay;
no decay; values near `10^30`; a non-monotone `dA_cumsum`), each with and without initial states.
The worst error / tolerance is `0.499` (`out`) and `0.498` (`final_states`); `out[0]` is bit-exact
(`+0` without initial states), the frame holds, and the host's metadata equal the claim's `reqOf` /
`hostOff` / `hostN` on every launch. Three negative tests (an output moved by twice its tolerance,
a launch with initial states checked as one without, all-zero `chunk_offsets`) are rejected. The
source's own test (`python -I source/ssd_state_passing.py`) passes in both modes, with relative
errors at most `2.1·10^-7` against its `float64` reference.
