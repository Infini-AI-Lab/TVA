# Setup of `chunk_cumsum_f32`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/utils/cumsum.py:34`,
the kernel `chunk_local_cumsum_scalar_kernel`, launched by fla's entry point `chunk_local_cumsum`.

## The source

`source/chunk_cumsum.py` holds:
- `chunk_local_cumsum_scalar_kernel`, the `@triton.jit` kernel, verbatim but for the default of
  its parameter `USE_GRAPH` (below): the cumulative sum of a
  `[B, T, H]` gate along time within each chunk of `BT` tokens, forward or reverse, optionally
  times `scale`, over fixed-length batches or packed variable-length ones; one program per
  (chunk, batch row, head);
- `unflatten_program_id`, the `@triton.jit` helper it calls, copied verbatim from
  `fla/ops/utils/op.py:22-29`;
- the host: `chunk_local_cumsum` (fla's entry point, `cumsum.py:451-496`) and
  `chunk_local_cumsum_scalar` (the launch, `cumsum.py:253-293`), with fla's torch helpers
  `prepare_lens`, `_segmented_arange` and `prepare_chunk_indices` (`fla/ops/utils/index.py`), which
  build the `[NT, 2]` table of (sequence, chunk) pairs of a variable-length batch;
- `reference`, the chunk-local cumulative sum in `float64`, and the test under
  `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring: the kernel's
decorators `@triton.heuristics` and `@fla_cache_autotune` removed (`@triton.jit(do_not_specialize=['T'])`
kept); the default of the kernel's parameter `USE_GRAPH` removed (`USE_GRAPH: tl.constexpr = False`
written `USE_GRAPH: tl.constexpr`); the host passes the flags and one configuration explicitly; the decorators `@input_guard`
(tensor arguments made contiguous) and `@dispatch` (backend selection; Triton on NVIDIA GPUs) of
the entry point replaced by `.contiguous()` calls; its 4-D branch (another kernel,
`chunk_local_cumsum_vector_kernel`) raises; the index helpers copied without their `@tensor_cache`
memo.

## The config

- Removed: `@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None, 'IS_VARLEN':
  lambda args: args['cu_seqlens'] is not None})` and `@fla_cache_autotune(configs=[triton.Config({},
  num_warps=num_warps) for num_warps in [1, 2, 4, 8]], key=['B', 'H', 'BT', 'IS_VARLEN',
  'REVERSE'])` (`cumsum.py:21-32`).
- Removed: the default `= False` of the kernel's parameter `USE_GRAPH` (`cumsum.py:47`): TVA's
  subset has no kernel parameter defaults, and the host passes `USE_GRAPH` on every launch.
- Chosen: `num_warps=2`, the list's second entry, the autotuner's choice on the H200 for a
  variable-length batch of 32 heads in 64-token chunks; `num_stages=3`, a `triton.Config`'s default.
- Flags set by the host: `HAS_SCALE = scale is not None`, `IS_VARLEN = cu_seqlens is not None`.
  The wrapper itself passes `REVERSE = reverse`, `USE_GRAPH = use_graph` (default `False`), `BT =
  chunk_size` (asserted a power of two), `B`, `H` and `T`.

## The launches

The claim covers the entry point's launches as fla's linear-attention layers make them on
variable-length batches (the chunked gated delta rule's forward without the in-kernel gate,
`fla/ops/gated_delta_rule/chunk.py:63-69`: `chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2,
cu_seqlens=…)`; its backward, `chunk.py:246`: `reverse=True` with no scale; gated DeltaProduct,
`fla/ops/gated_delta_product/chunk.py:55-62`), and on fixed-length batches (`cu_seqlens = None`,
as the host's test runs them):
- `float32` `s` (`[B, T, H]`) and `o` (the default `output_dtype=torch.float`);
- `USE_GRAPH = False`, `REVERSE` either way and `HAS_SCALE` either way (a `float` scale;
  `RCP_LN2 = 1.4426950216`, `fla/ops/utils/constant.py:10`, in gated DeltaNet);
- variable length (`IS_VARLEN = True`, `B = 1`): `cu_seqlens` (`int32` or `int64`), nonnegative,
  nondecreasing and at most `T`, with `N ≥ 1` sequences of lengths `≥ 0`, and `chunk_indices` as
  `prepare_chunk_indices` builds it (the wrapper's default) or as the caller passes it, the same
  table in any row order;
- fixed length (`IS_VARLEN = False`): `B ≥ 1` rows of `T` tokens;
- shapes as variables: `H ≥ 1` heads (fla's tests: 1 to 8; gated DeltaNet's value heads, 6 by
  default, `fla/models/gated_deltanet/configuration_gated_deltanet.py:32`, and up to 32 or 64 in
  larger models), `BT` a power of two up to `2^20` (the entry point asserts a power of two; 64 in gated DeltaNet; fla's
  tests, `tests/ops/utils/test_cumsum.py`, use 16 to 128), total length `T` up to the packed
  batch's token count (fla's tests: up to 2048; training batches of 2^16 tokens and more),
  sequence lengths from 0 (fla's tests: `[0, 1, 100, 300, 1200, 2048]`, `[0, 15]`).

`USE_GRAPH = True` and the 4-D path (another kernel) are not part of the setup.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11, peak device memory 2.09 GiB) passes on
30 cases, each forward and reverse, without and with a scale, through the host and through a
direct launch with `o` prefilled with a sentinel: the host's five fixed-length and seven
variable-length test shapes (the latter with `int32` and `int64` tables, up to a packed batch of
65,536 tokens with 32 heads), and edges (empty and short sequences, sequences not covering
`[0, T)`, a permuted `chunk_indices`, `BT` 512 and 2048, gates near `2^126`, NaN/±inf gates,
scales from `1e-45` to `1e39`, `2^20` tokens × 64 heads, 64 fixed rows). Every table is
`TablesOk`; over every in-range element the output is finite and the worst error / tolerance is
1.0000 (at most 1); the inputs are unchanged and every unwritten element keeps the sentinel.
