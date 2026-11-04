# Setup of `chunk_kkt_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/common/chunk_scaled_dot_kkt.py:33`
`chunk_scaled_dot_kkt_fwd_kernel`.

## The source

`source/chunk_kkt.py` holds:
- `chunk_scaled_dot_kkt_fwd_kernel`, the `@triton.jit` kernel, verbatim: within each chunk of `BT`
  tokens, `β·K·Kᵀ` (the keys of value head `h`'s key head `h // (HV // H)`, by `tl.dot` over blocks
  of `BK` channels into `float32`), gated by `exp2(g_i − g_j)` when `g` is given, kept strictly
  lower triangular and inside the sequence; one program per (chunk, batch row and value head);
- `exp2`, the `@triton.jit` helper it calls, copied verbatim from `fla/ops/utils/op.py:46-47`
  (the branch taken without `FLA_USE_FAST_OPS`, fla's default, on an H200 too: `tl.math.exp2` in
  `float32`);
- the host `chunk_scaled_dot_kkt_fwd` (`chunk_scaled_dot_kkt.py:83-135`), with fla's torch helpers
  `prepare_lens`, `_segmented_arange` and `prepare_chunk_indices` (`fla/ops/utils/index.py`) for
  variable-length batches;
- `reference`, the same matrix in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring: the kernel's
decorators `@triton.heuristics` and `@triton.autotune` removed (`@triton.jit(do_not_specialize=['T'])`
kept); the host passes the flags and one configuration explicitly; its decorator
`@dispatch('common')` (backend selection; Triton on NVIDIA GPUs) removed; the index helpers copied
without their `@tensor_cache` memo.

## The config

- Removed: `@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'IS_VARLEN': lambda
  args: args['cu_seqlens'] is not None})` and `@triton.autotune(configs=[triton.Config({'BK': BK},
  num_warps=num_warps, num_stages=num_stages) for BK in [32, 64, 128] for num_warps in [2, 4, 8]
  for num_stages in [2, 3, 4]], key=['H', 'HV', 'K', 'BT', 'IS_VARLEN'])`
  (`chunk_scaled_dot_kkt.py:18-31`).
- Chosen: `BK = 64`, `num_warps=2`, `num_stages=3`, the list's eleventh entry, the autotuner's
  choice on the H200 for 16 key heads, 32 value heads, `K = 128` and chunks of 32 and 64 tokens
  (for 16-token chunks it chose `num_stages=2` with the same `BK` and `num_warps`).
- Flags set by the host: `USE_G = g is not None`, `IS_VARLEN = cu_seqlens is not None`. The wrapper
  itself passes `BT = chunk_size`, `H`, `HV` (`beta.shape[2]`), `K` and `T`, and the grid
  `(NT, B * HV)`.

## The launches

The claim should cover the wrapper's launches with a gate, as the gated delta rules call it
(`fla/ops/gated_delta_rule/chunk_fwd.py:402-410`, the chunked gated delta rule's intra-chunk step
for chunks of 16 and 32 tokens, and `fla/ops/gated_delta_product/chunk.py:67-74`, chunks of 64),
on fixed-length batches (the default `cu_seqlens=None`):
- `bfloat16` `k` and `beta`, `float32` `g` (the within-chunk cumulative log2 decay that
  `chunk_local_cumsum(…, scale=RCP_LN2)` returns, `fla/ops/gated_delta_rule/chunk.py:63-69`),
  `float32` `A` (the default `output_dtype`);
- `USE_G = True`, `IS_VARLEN = False`;
- `k` `[B, T, H, K]`, `g` and `beta` `[B, T, HV]`, `A` `[B, T, HV, BT]`, contiguous (the callers
  run under `@input_guard`, which makes them so);
- shapes as variables: `BT` in {16, 32, 64} (`chunk_fwd.py:369-370` admits these), `K` the key head
  dimension (fla's tests: 64, 100, 128; gated DeltaNet's default `head_dim = 256`,
  `fla/models/gated_deltanet/configuration_gated_deltanet.py:31`; 128 in Qwen3-Next-style
  configurations), `H` key heads and `HV` value heads with `HV` a multiple of `H` (the wrapper's
  docstring; fla's tests: `H` from 1 to 8, `HV` up to 8; gated DeltaNet's default 6 and 6), any
  batch size `B` and length `T ≥ 1` (fla's tests: `T` from 63 to 2048, `B` up to 4).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes for fla's test shapes
and larger ones (`B` up to 4, `T` from 1 to 2048, `H`/`HV` from 1/1 to 16/32, `K` = 64, 100, 128
and 256, `BT` = 16, 32 and 64), fixed-length and variable-length, each with and without `g`: the
largest absolute error of `A` against the `float64` reference was 8.2e-8 (inputs: L2-normalised
keys, `β` in (0.5, 0.74), gates from `logsigmoid`).
