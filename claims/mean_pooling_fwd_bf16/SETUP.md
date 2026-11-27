# Setup of `mean_pooling_fwd_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/utils/pooling.py:30`, the
kernel `mean_pooling_fwd_kernel`, launched by fla's entry point `mean_pooling` through its forward
host `mean_pooling_fwd`.

## The source

`source/pooling.py` holds:
- `mean_pooling_fwd_kernel`, the `@triton.jit` kernel, verbatim: the mean of every chunk of `BT`
  tokens of a `[B, T, H, D]` input, per head and feature, over fixed-length batches or packed
  variable-length ones; one program per (feature block, chunk) and (batch row, head);
- `unflatten_program_id`, the `@triton.jit` helper it calls, copied verbatim from
  `fla/ops/utils/op.py:22-29`;
- the host: `mean_pooling` (fla's entry point, `pooling.py:207-224`) and `mean_pooling_fwd` (the
  launch, `pooling.py:119-143`), with fla's torch helpers `prepare_lens`, `_segmented_arange` and
  `prepare_chunk_indices` (`fla/ops/utils/index.py`), which build the `[NT, 2]` table of
  (sequence, chunk) pairs of a variable-length batch;
- `reference`, the chunk means in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring: the kernel's decorators
`@triton.heuristics` and `@triton.autotune` removed (`@triton.jit(do_not_specialize=['T'])` kept);
the host passes the flag and one configuration explicitly; the entry point calls
`mean_pooling_fwd` directly in place of the autograd function `MeanPoolingFunction` (whose forward
is `mean_pooling_fwd` under `@input_guard`, replaced by `.contiguous()` calls, and
`@autocast_custom_fwd`, which changes nothing outside an autocast region); the backward kernel, its
host and the autograd function are not kept; the index helpers copied without their
`@tensor_cache` memo.

## The config

- Removed: `@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})` and
  `@triton.autotune(configs=[triton.Config({'BD': BD}, num_warps=num_warps) for BD in [16, 32, 64,
  128] for num_warps in [1, 2, 4, 8]], key=['BT'], **autotune_cache_kwargs)`
  (`pooling.py:17-28`).
- Chosen: `BD = 64`, `num_warps = 2`, the autotuner's choice on the H200 for a packed batch of
  65,536 tokens, 4 heads of 128 features and 64-token chunks in `bfloat16` (NSA's default block
  size and a common head size); `num_stages = 3`, a `triton.Config`'s default. The claim covers
  every `BD` of the list (`16 ≤ BD ≤ 128`, a power of two); `num_warps` is performance-only.
- Flag set by the host: `IS_VARLEN = cu_seqlens is not None`. The host itself passes `T`
  (`x.shape[1]`, a runtime `int32`: `do_not_specialize=['T']`), `H`, `D` and `BT = chunk_size`.

## The launches

The claim covers the entry point's launches as fla makes them: native sparse attention compresses
its keys and values with `mean_pooling(k, block_size, cu_seqlens_k)` and
`mean_pooling(v, block_size, cu_seqlens_k)` (`fla/ops/nsa/parallel.py:913`, `block_size = 64` by
default; `fla/ops/nsa/naive.py:383`), and fla's tests call the entry point directly:
- `bfloat16` `x` (`[B, T, H, D]`) and the fresh `o = x.new_empty(B, NT, H, D)`;
- variable length (`IS_VARLEN = True`, `B = 1`, which the entry point checks): `cu_seqlens`
  (`int32` or `int64`), nonnegative, nondecreasing and at most `T`, with `N ≥ 1` sequences of
  lengths `≥ 0`, and `chunk_indices` as `prepare_chunk_indices` builds it (the host's default) or
  as a caller of `mean_pooling_fwd` passes it;
- fixed length (`IS_VARLEN = False`): `B ≥ 1` rows of `T` tokens;
- shapes as variables: `H ≥ 1` heads (NSA's key/value heads; fla's tests: 1 to 8), `D ≥ 1`
  features (the head size, a power of two or not; fla's tests: 30 to 1024), `chunk_size` a power of
  two (`tl.arange`; NSA's `block_size`: 16 to 64 in fla's tests, 64 by default; fla's pooling test:
  16 to 128), total length `T` up to the packed batch's token count (fla's tests: up to 2048).

The covered instances (`setup_*_ok` in `Claim.lean`, all `bfloat16`, `BD = 64`):
- fla's pooling test (`tests/ops/utils/test_pooling.py`, run there in `float32`/`float16`):
  fixed `(B, T, H, chunk, D)` = `(1, 63, 1, 16, 30)`, `(2, 500, 4, 32, 60)`,
  `(2, 1000, 5, 64, 128)`, `(3, 1024, 6, 64, 500)`, `(4, 2048, 8, 128, 1024)`; variable
  `(H, chunk, D, cu_seqlens)` = `(2, 32, 60, [0, 15])`, `(3, 64, 100, [0, 256, 500, 1000])`,
  `(4, 64, 256, [0, 15, 100, 300, 1200, 2000])`, `(4, 128, 500, [0, 1, 100, 300, 1200, 2048])`,
  `(2, 128, 1024, [0, 200, 512, 1200, 2048])`, each with `int32` and `int64` tables;
- fla's NSA tests (`tests/ops/test_nsa.py`, `test_parallel` and `test_parallel_varlen`, the pooling
  of `k` and `v`): fixed `(B, T, H, block_size, D)` = `(1, 63, 1, 32, 64)`, `(3, 111, 1, 32, 100)`,
  `(3, 1024, 2, 32, 60)`, `(3, 1024, 2, 32, 128)`, `(4, 2048, 2, 32, 64)`; variable
  `(H, block_size, D, cu_seqlens)` = `(1, 32, 64, [0, 15])`, `(1, 16, 64, [0, 15, 205, 550, 800])`,
  `(2, 32, 64, [0, 256, 500, 1000])`, `(2, 32, 100, [0, 15, 100, 300, 1200, 2000])`, each with
  `int32` and `int64` tables;
- a packed NSA batch: `(H, block_size, D, cu_seqlens)` = `(4, 64, 128, [0, 1, 64, 65, 4096, 30000,
  65536])`, `int32` and `int64` tables.

The backward (`mean_pooling_bwd_kernel`) and other dtypes are not part of the setup.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11, peak device memory 1.27 GiB) passes on
50 cases, through the host and through a direct launch with `o` inside a sentinel-filled buffer:
every launch above, and edges (`BT = 1`, empty and one-token sequences, sequences not covering
`[0, T)`, a reversed and a repeated `chunk_indices`, all sequences empty, `T = 0`, `BT` 1024 and
4096, `BD` 16, 32 and 128 with other `num_warps`, inputs near `2^125` and subnormal, NaN/±inf
inputs, `2^22` tokens, 8192 fixed rows, `D = 2^17`). Every table is `TablesOk`; over every in-range
element the output is finite and the worst error / tolerance is `0.996`; the inputs are unchanged
and every byte outside `o` keeps the sentinel. Three wrong outputs (the sum divided by `BT` in place
of `L`, the output times `1 + 2^-6`, chunks shifted by one position) fail the tolerance.
