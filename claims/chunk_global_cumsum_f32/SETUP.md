# Setup of `chunk_global_cumsum_f32`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/utils/cumsum.py:151`,
the kernel `chunk_global_cumsum_scalar_kernel`, launched by fla's entry point
`chunk_global_cumsum`.

## The source

`source/chunk_global_cumsum.py` holds:
- `chunk_global_cumsum_scalar_kernel`, the `@triton.jit` kernel, verbatim: the cumulative sum of a
  `[B, T, H]` gate along time over each whole sequence, forward or reverse, optionally times
  `scale`, over fixed-length batches or packed variable-length ones; one program per
  (sequence, head), walking the sequence in blocks of `BT` positions with a running sum;
- the host: `chunk_global_cumsum` (fla's entry point, `cumsum.py:412-448`) and
  `chunk_global_cumsum_scalar` (the launch, `cumsum.py:347-375`);
- `reference`, the cumulative sum in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring: the kernel's
decorators `@triton.heuristics` and `@triton.autotune` removed (`@triton.jit(do_not_specialize=['T'])`
kept); the host passes the flags and one configuration explicitly; the decorators `@input_guard`
(tensor arguments made contiguous) of both host functions and `@dispatch` (backend selection;
Triton on NVIDIA GPUs) of the entry point replaced by `.contiguous()` calls; the entry point's 4-D
branch (another kernel, `chunk_global_cumsum_vector_kernel`) raises.

## The config

- Removed: `@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None, 'IS_VARLEN':
  lambda args: args['cu_seqlens'] is not None})` and `@triton.autotune(configs=[triton.Config({'BT':
  BT}, num_warps=num_warps, num_stages=num_stages) for BT in [32, 64, 128, 256] for num_warps in
  [2, 4, 8] for num_stages in [1, 2, 3, 4]], key=['B', 'H', 'IS_VARLEN', 'REVERSE'])`
  (`cumsum.py:136-149`).
- Chosen: `BT=256`, `num_warps=8`, `num_stages=3`, the autotuner's choice on the H200 for a packed
  batch of 65,536 tokens with 32 heads, forward and reverse. The claim covers every power of two
  `BT ≤ 2^20` (every entry of the removed list); `num_warps` and `num_stages` change no outcome.
- Flags set by the host: `HAS_SCALE = scale is not None`, `IS_VARLEN = cu_seqlens is not None`.
  The wrapper itself passes `REVERSE = reverse`, `B`, `H` and `T`.

## The launches

The claim covers the entry point's launches as fla's attention layers make them on
variable-length batches (forgetting attention, `fla/ops/attn/parallel.py:737`:
`chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2)`, and its backward,
`parallel.py:775`: `reverse=True` with no scale; PaTH attention, `fla/ops/path_attn/parallel.py:35`
and `:217`; WALL attention, `fla/ops/wall_attn/parallel.py:961-1045`; simple GLA's backward,
`fla/ops/simple_gla/parallel.py:633`), and on fixed-length batches (`cu_seqlens = None`, as the
host's test and `fla/layers/wall_attn.py:299` run them):
- `float32` `s` (`[B, T, H]`) and `z` (the default `output_dtype=torch.float`);
- `REVERSE` either way and `HAS_SCALE` either way (a `float` scale; `RCP_LN2 = 1.4426950216`,
  `fla/ops/utils/constant.py:10`);
- variable length (`IS_VARLEN = True`, `B = 1`): `cu_seqlens` (`int32` or `int64`), nonnegative,
  nondecreasing and at most `T`, with `N ≥ 0` sequences of lengths `≥ 0`;
- fixed length (`IS_VARLEN = False`): `B ≥ 1` rows of `T ≥ 0` tokens;
- shapes as variables: `H ≥ 1` heads (fla's tests, `tests/ops/utils/test_cumsum.py`: 1 to 32; up
  to 64 in larger models), `B` up to `2050` rows and `N` up to `2050` sequences in fla's tests,
  total length `T` up to the packed batch's token count (training batches of `2^16` tokens and
  more), sequence lengths from 0.

The host's test (source lines 192–224) launches the five fixed-length shapes `(B, T, H)` =
`(1, 63, 1)`, `(2, 500, 4)`, `(2, 1000, 5)`, `(3, 1024, 6)`, `(4, 2048, 8)` and the six
variable-length ones `(T, H, cu_seqlens)` = `(15, 2, [0, 15])`, `(1000, 3, [0, 256, 500, 1000])`,
`(2000, 4, [0, 15, 100, 300, 1200, 2000])`, `(2048, 4, [0, 1, 100, 300, 1200, 2048])`,
`(1, 32, [0, 1])`, `(65536, 32, [0, 1, 64, 65, 4096, 30000, 65536])`, each forward and reverse,
without a scale and with `RCP_LN2`, with `int32` tables (and here also `int64` ones): the
`setup_*_ok` theorems of `Claim.lean`. fla's own tests of this kernel have the same fixed-length
shapes, and two `float32` launches more, also covered (`setup_fla_*_ok`): `2050` sequences of `16`
tokens with `32` heads (`cu_seqlens = range(0, 32801, 16)`), and the reversed sum over `2050` rows
of `64` tokens with `32` heads.

Other gate dtypes (fla's `float16` test cases, `output_dtype=k.dtype` in
`fla/ops/linear_attn/utils.py:22`) and the 4-D path (another kernel) are not part of the setup.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11, peak device memory 1.54 GiB) passes on
56 cases: the host's five fixed-length and six variable-length test shapes (the latter with
`int32` and `int64` tables), each forward and reverse, without and with a scale, through the host
and through a direct launch with `o` prefilled with a sentinel; edges (empty and short sequences
and block-sized ends with `BT` 1, 32, 64, 128, 256 and 2048, sequences not covering `[0, T)`, no
sequence (an empty grid), fixed rows of `T = 0`, `2^22` tokens with `BT` 32 and 256, gates near
`2^126`, NaN/±inf gates, scales from `1e-45` to `1e39`, `2^20` tokens × 64 heads, 64 fixed rows; fla's two test launches above);
and 20 negatives (outputs perturbed by `2^-18` relative, an exclusive scan, a block-local scan, a
sign flip, a running sum missing a block), each rejected. Every table is `TablesOk`; over every
in-range element the output is finite and the worst error / tolerance is `0.30`; the inputs are
unchanged and every unwritten element keeps the sentinel.
