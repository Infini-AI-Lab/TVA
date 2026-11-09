# Setup of `recompute_wu_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/gated_delta_rule/wy_fast.py:43`
`recompute_w_u_fwd_kernel`, launched by `recompute_w_u_fwd` (`wy_fast.py:254-294`).

## The source

`source/wy_fast.py` holds:
- `recompute_w_u_fwd_kernel`, the `@triton.jit` kernel (lines 69-133), verbatim from
  `fla/ops/gated_delta_rule/wy_fast.py:42-106`: one program per (chunk of 64 tokens, batch entry and
  value head); it loads the chunk's rows of `A` and `beta`, then for each block of 64 columns computes
  `u = A @ bf16(beta * v)` and, with the gate, `w = A @ bf16(beta * exp2(g) * k)`, each rounded to
  `bfloat16`, over a padded batch or packed variable-length sequences (`cu_seqlens`, with
  `chunk_indices` listing each program's sequence and chunk). All its compile-time paths are kept;
- `exp2`, the `@triton.jit` helper the kernel calls (lines 65-66), from `fla/ops/utils/op.py:46-47`;
- `recompute_w_u_fwd`, the host (lines 175-220), from `fla/ops/gated_delta_rule/wy_fast.py:254-294`;
- the torch helpers the host calls: `prepare_lens`, `_segmented_arange` and `prepare_chunk_indices`
  (`fla/ops/utils/index.py:40-42, 113-134, 155-164`).

Changed from fla's source (each listed with its `file:line` in the module docstring):
- the decorators `@triton.heuristics({'USE_G': ..., 'IS_VARLEN': ...})` and
  `@fla_cache_autotune(...)` are removed from the kernel (`wy_fast.py:29-41`);
  `@triton.jit(do_not_specialize=['T'])`, the signature and the body are verbatim;
- `exp2` is defined at the top level of the file; upstream it is defined in `fla/ops/utils/op.py` under
  `if os.environ.get('FLA_USE_FAST_OPS', '0') == '1': ... else:` (this is the `else` branch, taken by
  default) and imported;
- the host drops fla's `@dispatch` (its backend registry, which falls back to this function on NVIDIA
  GPUs) and passes the flags the heuristics computed and one configuration explicitly; the rest of
  the wrapper is verbatim;
- the index helpers lose their caching decorator (`@tensor_cache`), and `_segmented_arange` calls
  `torch.nn.functional.pad` for `F.pad`.

## The config

Removed: `@fla_cache_autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages)
for num_warps in RECOMPUTE_W_U_NUM_WARPS for num_stages in [2, 3, 4]], key=['H', 'HV', 'K', 'V',
'BT', 'BK', 'BV', 'IS_VARLEN'], **autotune_cache_kwargs)` (`wy_fast.py:33-41`;
`RECOMPUTE_W_U_NUM_WARPS = [2, 4, 8]` on NVIDIA GPUs, line 26), and
`@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'IS_VARLEN': lambda args:
args['cu_seqlens'] is not None})` (lines 29-32).

Chosen: `num_warps = 4`, `num_stages = 3`, the 5th of the list's 9 configurations: the fastest of the
nine on the H200 at Qwen3-Next's shape (`B = 4`, `T = 4096`, 16 key and 32 value heads of 128,
`bfloat16`; 0.32 ms against 0.36-0.49 ms for the others). The flags, as the heuristics compute
them: `USE_G = g is not None`, `IS_VARLEN = cu_seqlens is not None`. The wrapper's own values:
`BT = A.shape[-1]`, `BK = BV = 64`; grid `(NT, B * HV)` with `NT = triton.cdiv(T, BT)`, or
`len(chunk_indices)` with `cu_seqlens`. `T` is a runtime scalar (`do_not_specialize`).

## The launches

The claim covers the launches of the gated DeltaNet's chunked forward and backward
(`chunk_gated_delta_rule_fwd_intra`, `fla/ops/gated_delta_rule/chunk_fwd.py:419`, and
`chunk_gated_delta_rule_bwd`, `fla/ops/gated_delta_rule/chunk.py:147`) and of the models built on it
(fla's `GatedDeltaNet`, `fla/layers/gated_deltanet.py:302`; `MoM`, `fla/layers/mom.py:596, 736`;
`GatedDeltaProduct`, `fla/layers/gated_deltaproduct.py:229` through
`fla/ops/gated_delta_product/chunk.py:82`; `PrecondGatedDeltaNet`,
`fla/layers/precond_gated_deltanet.py:321` through `fla/ops/precond_gated_delta_rule/chunk.py:87`), with
shapes as variables:
- `k` `bfloat16` `[B, T, H, K]`, `v` `bfloat16` `[B, T, HV, V]`, contiguous (`chunk_gated_delta_rule`
  runs under `input_guard`); `A` `[B, T, HV, 64]` of `k`'s dtype, `bfloat16` (chunk size 64: the fused
  `kkt + solve_tril` kernel allocates it with `dtype=k.dtype`, `chunk_fwd.py:385`); `beta`
  `[B, T, HV]`, `float32` or `bfloat16`: `chunk_gated_delta_rule` passes the caller's `beta` unchanged
  unless `use_beta_sigmoid_in_kernel=True` (default `False`, `chunk.py:407`), whose
  `fused_beta_sigmoid` returns `float32` (`fla/ops/common/gate.py:59`; `GatedDeltaNet` uses it);
  `MoM`, `GatedDeltaProduct` and `PrecondGatedDeltaNet` pass `b_proj(h).sigmoid()` in the model's
  `bfloat16` (`mom.py:498, 503`; `gated_deltaproduct.py:229`; `precond_gated_deltanet.py:321`), as do
  the docstring example (`chunk.py:491`) and the kernel's unit tests (`tests/ops/test_gdn_kernels.py:185,
  1120`); `test_chunk` and `test_chunk_varlen` pass `float32`; `g` `float32` `[B, T, HV]`, always
  given (the gate's chunk-local cumulative sum
  in base 2, `chunk_local_cumsum(..., scale=RCP_LN2)` or `gdn_gate_chunk_cumsum`, `float32`, so
  `USE_G = True`); `w`, `u` fresh;
- `H >= 1` key heads and `HV` value heads, `HV` a multiple of `H` (grouped value attention,
  `chunk.py:526-530`); head dims `K, V >= 1` (the tests use 32, 60, 64, 100, 128; Qwen3-Next's layers
  128 and 128; fla's `GatedDeltaNet` defaults 256 and 512, `fla/layers/gated_deltanet.py:93-96`; the
  docstring of `chunk_gated_delta_rule` 512 and 512, `chunk.py:487`; `MoM`, `GatedDeltaProduct` and
`PrecondGatedDeltaNet` 256 and 512 by default);
- both values of `IS_VARLEN`: a padded batch (`cu_seqlens = None`, `B >= 1` with `B * HV < 65536`,
  see the kernel limit in README.md), and packed variable-length sequences (`B = 1`, `cu_seqlens`
  `[N + 1]` `int64` as fla's tests and layers pass it, or `int32`, non-decreasing from 0 to `T`;
  `chunk_indices` from `prepare_chunk_indices(cu_seqlens, 64)`);
- `T` from 1 up, `N` from 1 up; the inputs' values as the model produces them (`k` L2-normalised,
  `beta` in `(0, 1)`, or `(0, 2)` with `allow_neg_eigval`, `g <= 0`, `A` the inverse of a unit
  lower-triangular matrix).

The concrete launches (each a `setup_*` instance of `Claim.lean` and a case of `check_gpu.py`), with
the code base's shapes, `k`, `v`, `A` in `bfloat16`:
- `tests/ops/test_gdn.py:157-167` (`test_chunk`, padded; the test runs `float16`, here its shapes in
  `bfloat16`, with its `float32` `beta`): `(B, T, H, HV, D)` = `(4, 1024, 4, 4, 128)`,
  `(2, 75, 4, 4, 64)`, `(2, 500, 3, 3, 60)`, `(2, 1000, 3, 3, 64)`, `(3, 1024, 4, 4, 100)`,
  `(2, 1500, 4, 4, 128)`, `(4, 2048, 8, 8, 64)`, `(2, 256, 2, 4, 64)`, `(2, 512, 2, 8, 64)`,
  `(2, 1024, 4, 8, 128)`, with `K = V = D`;
- `tests/ops/test_gdn.py:758-763` (`test_chunk_varlen`, packed; the test runs `float16`, here its
  shapes in `bfloat16`, with its `float32` `beta`): `(H, HV, D)` = `(4, 4, 60)` with
  `cu_seqlens = [0, 15]`, `(4, 4, 64)`, `(2, 4, 64)`, `(2, 8, 64)` with `[0, 256, 500, 1000]`,
  `(4, 4, 100)` with `[0, 15, 100, 300, 1200, 2000]`;
- `tests/ops/test_gdn_kernels.py:165-168` (`test_recompute_w_u_fwd`, `bfloat16` `beta` as the test
  passes it; the last case runs `float16` there, here `bfloat16`): `(2, 128, 2, 2, 64)`,
  `(2, 128, 2, 4, 64)`, `(1, 256, 4, 4, 32)`; `tests/ops/test_gdn_kernels.py:1099-1102`
  (`test_recompute_w_u_fwd_varlen`, `bfloat16` `beta`): `(H, HV, D)` = `(2, 2, 64)`, `(1, 1, 64)` with
  `[0, 64, 128]`, `(2, 4, 64)` with `[0, 128, 256]`;
- the models with a `float32` `beta`: Qwen3-Next's heads (16 key, 32 value heads of 128), padded
  `B = 2, T = 4096` and packed (`int32` `cu_seqlens`, 8 sequences, 16384 tokens); fla's `GatedDeltaNet`
  defaults (6 heads, `K = 256`, `V = 512`), `B = 2, T = 4096`;
- with a `bfloat16` `beta`: the docstring example (`H = 4`, `HV = 8`, `K = V = 512`), padded
  `B = 4, T = 2048` and packed `[0, 2048, 4096, 6144, 8192]` (`chunk.py:487, 491, 502`);
  `GatedDeltaProduct`'s defaults (6 heads, `K = 256`, `V = 512`, two Householder steps, so `T = 2·2048`
  interleaved tokens), padded `B = 2` and packed `[0, 1000, 2600, 4096]`; `PrecondGatedDeltaNet`'s
  defaults, `B = 2, T = 2048`; `MoM`'s defaults (4 heads, `K = 256`, `V = 512`), its memories' tokens
  packed as `[0, 700, 1500, 2048, 2900, 4096]`;
- edges: one token with `K = V = 1`; `B * HV = 65535` (`B = 13107`, `HV = 5`); packed with empty
  sequences (`[0, 0, 5, 5, 70]`, `int32`); 132 one-token sequences.

Not part of the setup: `g = None` (`USE_G = False`, which only the kernel's unit tests and a
no-gate smoke test use, `tests/ops/test_gdn_kernels.py:165-166`, `tests/ops/test_gdn.py:1327`;
the gated DeltaNet always passes `g`), `float16` `beta`, `float16` or `float32` `k`, `v`, `A` (the tests'
other dtypes; this claim is the `bfloat16` kernel), and chunk sizes 16 and 32
(`A.shape[-1] != 64`, the unfused path of `chunk_fwd.py:400-416`).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 47 launches through the host, all passing: the 34 launches above
(padded and packed, `beta` `float32` and `bfloat16`, `cu_seqlens` `int32` and `int64`, up to 16384
tokens and 32 value heads); 6 edges of `Pre`, each with a `float32` and a `bfloat16` `beta` (dense
`A` of order 1 with `|g|` up to 30, padded and packed; equal operands whose roundings to `bfloat16`
add up; the largest magnitudes `Pre` allows, `|A| = 2^64`, `|k| = |v| = 2^32`, `beta = 2^8`, `g = 8`,
padded and packed; subnormal values with gates far below, where `exp2` flushes to 0); and one edge
for a `bfloat16` `beta` whose roundings at lines 129 and 132 all go up and add up (`k = 1.15625`,
`beta = 1.75`, `g = -0.0055`, `A` all `1`: `w` reaches `0.6499` of `tolW`); plus a direct launch writing `w` and `u` between guard regions.
The largest
measured `|out - ref| / tol` is `0.9921` for `u`, and for `w` `0.9676` (`float32` `beta`) and `0.9483`
(`bfloat16` `beta`); every output is finite, the inputs
(and `cu_seqlens`, `chunk_indices`) are unchanged, and the guards keep their pattern. Every negative
test is rejected (`w` without `2^g`, `A` transposed in its chunk, the wrong key head, a packed
batch's chunks not restarted at its sequences, one `u` element off by 2%, `w` scaled by
`1 + 2^-6`). A padded launch with `B * HV = 65536` fails (`invalid argument`), outside the claim.
