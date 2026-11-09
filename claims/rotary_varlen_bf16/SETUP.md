# Setup of `rotary_varlen_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/modules/rotary/ops.py:31`
`rotary_embedding_kernel`.

## The source

`source/rotary.py` holds:
- `rotary_embedding_kernel`, the `@triton.jit` kernel (lines 63-154), verbatim from
  `fla/modules/rotary/ops.py:30-121`: one program per (chunk of `BT` tokens, batch, head); it
  rotates the first `2 R` channels of each token's head vector by the cosines and sines of the
  token's position, not interleaved (channels `i` and `i + R`) or interleaved (`2i` and `2i + 1`),
  over a padded batch or packed variable-length sequences (`cu_seqlens`, with `chunk_indices`
  listing each program's sequence and chunk). All its compile-time paths are kept;
- `rotary_embedding_fwd`, the host, adapted from fla's wrapper `rotary_embedding_fwdbwd`
  (`ops.py:144-241`), out of place;
- the torch helpers the host calls: `get_multiprocessor_count` (`fla/utils/hardware.py:30-43`),
  `prepare_lens`, `_segmented_arange` and `prepare_chunk_indices` (`fla/ops/utils/index.py`);
- `cos_sin_cache` (the tables of fla's `RotaryEmbedding`), `reference`, the rotation in `float64`,
  and the test under `if __name__ == "__main__":`.

Changed from fla's source (each listed with its `file:line` in the module docstring): the
`@triton.autotune(...)` decorator is removed from the kernel; the host drops fla's `@dispatch`
(its backend registry, which falls back to this function on NVIDIA GPUs), drops the argument
`inplace` and its path (`y = x`), so `y` is always a fresh `torch.zeros_like(x)` (with the
channels past `2 R` copied from `x`), and passes one configuration explicitly; the helpers lose
their caching decorators (`@cache`, `@tensor_cache`) and the NPU fallback.

## The config

Removed: `@triton.autotune(configs=[triton.Config({}, num_warps=num_warps,
num_stages=num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 3, 4]],
key=['B', 'H', 'D', 'INTERLEAVED'], ...)` (`fla/modules/rotary/ops.py:21-29`;
`NUM_WARPS_AUTOTUNE = [2, 4, 8, 16, 32]` on NVIDIA GPUs, line 18).

Chosen: `num_warps = 4`, `num_stages = 2`, the 4th of the list's 15 configurations. The
wrapper's own values: `BD = triton.next_power_of_2(2 R)`;
`BT = min(128, triton.next_power_of_2(triton.cdiv(T, SMs)))` with `SMs` the device's
multiprocessor count (132 on the H200), so `BT` is a power of two from 1 to 128; grid
`(NT, B, H)` with `NT = triton.cdiv(T, BT)`, or `len(chunk_indices)` with `cu_seqlens`. The
flags, as the wrapper computes them: `IS_SEQLEN_OFFSETS_TENSOR = isinstance(seqlen_offsets,
torch.Tensor)`, `IS_VARLEN = cu_seqlens is not None`, `INTERLEAVED = interleaved`,
`CONJUGATE = conjugate`. `T` is a runtime scalar (`do_not_specialize`). No heuristics.

## The launches

The claim covers the forward launches of fla's `RotaryEmbedding` on queries and keys
(`fla/modules/rotary/ops.py:481-559`), with shapes as variables:
- `x` `bfloat16`, contiguous `[B, T, H, D]` (`RotaryEmbeddingFunction` runs under `input_guard`,
  which makes inputs contiguous); `cos`, `sin` `bfloat16`, contiguous `[TR, R]` (the module's
  tables, cast to the dtype of `x`); `y` fresh, of the shape and dtype of `x`;
- both values of `INTERLEAVED` (the module's `interleaved`, `False` by default: GPT-NeoX; `True`:
  GPT-J);
- both values of `IS_VARLEN`: a padded batch (`cu_seqlens = None`, `1 <= B < 65536`; larger padded batches
  are not part of the setup), and packed
  variable-length sequences (`B = 1`, `x` `[1, T, H, D]`, `cu_seqlens` `[N + 1]` `int32` or
  `int64`, non-decreasing from 0 to `T`, as fla's tests and layers pass it:
  `tests/modules/test_rotary.py:92-130`, `fla/layers/attn.py:125`; `chunk_indices` from
  `prepare_chunk_indices(cu_seqlens, BT)`, the default `None` argument);
- both values of `IS_SEQLEN_OFFSETS_TENSOR`: one `int` offset for every sequence,
  `seqlen_offsets >= 0` with `seqlen_offsets + T <= TR` (the wrapper's assert; 0 in training, the
  cache's length in decoding, `fla/layers/attn.py:113-115`); and a tensor of one offset per
  sequence, `[N]` `int32` or `int64` with `length + offset <= TR` for each sequence (the wrapper's
  asserts and `torch._assert_async`), which fla's attention layers pass in batched generation with a
  cache and an `attention_mask` (`fla/layers/attn.py:117-119`: `int32` offsets equal to the cache's
  length minus the sequence's left padding, so negative in the prefill, with `TR` the larger of
  `T + max(offset)` and `max_position_embeddings`); its tokens at a negative position are left at
  the host's `0` (source line 106 masks them). Offsets are at least `-2^31`;
- `CONJUGATE = False` (the forward);
- `D <= 256` and `2 R <= D` (the wrapper's asserts), `R >= 1`: `2 R = D` for the module built
  with `dim = head_dim` (`fla/layers/attn.py:83`), `2 R < D` for partial rotary (the tests use
  `rotary_dim = 64` with `D = 128, 256`, `tests/modules/test_rotary.py:165-172`); head dims 64
  (fla's Transformer config: hidden 2048, 32 heads), 128 and 256 (the tests);
- `H` the query or the key heads (the tests use 4 query heads and 1 or 4 key heads);
- `T` from 1 up (the tests use `T = 2048, 4096` with `B = 2, 4` and `N = 4` packed sequences;
  `TR >= T`), `N` from 1 up; with a tensor of offsets, the left-padded prefill (`B = 4`, `T = 512`,
  `H = 32`, `D = 64`, `TR = 2048`) and decoding step (`B = 8`, `T = 1`) of fla's Transformer config.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 63 launches, all passing: the host's 22 test cases (11 per
value of `INTERLEAVED`, padded and packed, `cu_seqlens` `int32` and `int64`, up to `T = 32768`),
the 13 other setup configurations (decoding at offset 4095, head dim 64 with 32 heads, one key head,
partial rotary 64 of `D = 256`, one packed sequence; with a tensor of offsets, the left-padded
prefill and decoding step in both layouts, `int64` offsets padded and packed) and 28 edges
(`B = 65535`, `R = 1`, `R = 3`, offset `2^20`, empty sequences, 132 one-token sequences,
`|x| = 2^126`, subnormal inputs, exact cancellation, NaN where the kernel does not read; tensor
offsets with `BT = 128` and a whole row padded, offsets `-2^31` in `int32` and `int64`, 132
one-token sequences; each in both layouts). The worst `|y − ref| / tol` is `0.9960`; every rotated
value at a non-negative position is finite, the tokens at a negative position keep their bytes
(`0` with the host's `torch.zeros_like`), and the frame holds. Five wrong outputs are rejected
(tensor offsets ignored, `conjugate = True`, offset `1` for `0`, all zeros, left-padding tokens
written). A padded batch of `B = 65536` fails to launch (`invalid argument`), outside the claim.
