# Setup of `ssd_bmm_chunk_bf16`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/mamba/triton_ops/ssd_bmm.py:81`
`_bmm_chunk_fwd_kernel`, launched by `_bmm_chunk_fwd` (`ssd_bmm.py:193`) as step 4 of Mamba-2's
chunked scan calls it (`ssd_combined.py:165`: `CB = _bmm_chunk_fwd(C, B, chunk_size,
seq_idx=seq_idx, output_dtype=torch.float32)`).

## The source

`source/ssd_bmm.py` holds:
- `_bmm_chunk_fwd_kernel`, the `@triton.jit` kernel (lines 80–190, the upstream line numbers): for
  each chunk `c` of `chunk_size` tokens, group `g` and a `64 × 128` tile of positions `(m, n)`, the
  dot product over `dstate` of row `c·chunk_size + m` of `a` and row `c·chunk_size + n` of `b`,
  accumulated in `float32` by `bfloat16` `tl.dot`s of 32 terms; rows past `seqlen` load as zero;
  with `seq_idx`, an element whose two tokens belong to different requests (or lie past `seqlen`,
  where `seq_idx` loads as `-1` for rows and `-2` for columns) is set to `0.0`; every element of the
  `chunk_size × chunk_size` block is stored;
- `bmm_chunk_fwd`, the host, adapted from SGLang's `_bmm_chunk_fwd` (lines 193–282);
- `make_inputs` (`C`, `B` contiguous or as views of a packed `xBC` tensor, `seq_idx` as SGLang
  builds it), `reference` (the product in `float64`) and the test under `if __name__ == "__main__":`.

Changes from the code base (each listed with its upstream `file:line` in the file's docstring):
lines 1–19 (header comments, imports) replaced by a docstring and the imports `math`, `torch`,
`triton`, `triton.language as tl` (the import of SGLang's autotune helpers removed), and blank lines
that keep the kernel at its upstream line numbers; `@triton.autotune` removed (lines 20–79, below);
the defaults `= 16` of the kernel's parameters `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`
removed (lines 111–113); no other line of the kernel changes. In the host: one configuration passed
explicitly and the grid a tuple; the checks, the `.contiguous()` fallbacks, the allocation of
`out`, `dot_dtype`, the argument order, the device context and the flags are as in the wrapper.

## The config

Removed: `@triton.autotune(configs=[…nine configs…], key=["chunk_size", "K", "IS_CAUSAL"],
prune_configs_by={"early_config_prune": prune_oversized_tiles(...)}, **autotune_cache_kwargs)`
(`ssd_bmm.py:20-79`). The host passes the fifth config (`ssd_bmm.py:42-46`): `BLOCK_SIZE_M = 64`,
`BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 32`, `num_stages = 4`, `num_warps = 4`. On the H200, at
`chunk_size = 256` and `dstate = 128` with 8192 tokens, it is the fastest of the nine for
`ngroups = 1, 4, 8` (9.5, 24.2, 41.2 µs; the others up to 23% slower), and within 2% of the fastest
at `dstate = 256` (the pruning keeps all nine at these sizes). It is SGLang's choice only for
chunk sizes 128 to 256: for `chunk_size ≤ 64`, `prune_oversized_tiles`
(`python/sglang/kernels/ops/mamba/triton_ops/autotune.py:19-43`) removes `BLOCK_SIZE_N = 128`. The kernel's meaning
does not depend on the config, and the claim covers every chunk size with this one. Grid `(cdiv(chunk_size, 64) ·
cdiv(chunk_size, 128), batch, nchunks · ngroups)`.

Flags, as SGLang's Mamba-2 prefill sets them (`python/sglang/srt/layers/attention/mamba/mamba.py:
604-642`, through `mamba_chunk_scan_combined` and `_mamba_chunk_scan_combined_fwd`,
`ssd_combined.py:165`):
- `IS_CAUSAL = False`: `ssd_combined.py:165` passes no `causal`, the wrapper's default `False`;
  every element of each chunk's `chunk_size × chunk_size` block is computed and stored;
- `HAS_SEQ_IDX = True`: `seq_idx` is always built for an extend batch
  (`python/sglang/srt/layers/attention/mamba/mamba2_metadata.py:265-275`);
- `dot_dtype = tl.bfloat16` (the inputs are `bfloat16`); `out` `float32` (`output_dtype`).

## The launches

The claim covers the host's launches with `IS_CAUSAL = False`, `HAS_SEQ_IDX = True`, `bfloat16`
inputs and a `float32` output, with shapes as variables:
- `batch = 1`: the prefill tokens of the requests packed one after another, `seqlen ≥ 1`;
  `nchunks = cdiv(seqlen, chunk_size)` chunks, which may cross request boundaries;
- `a = C`, `b = B`: `[1, seqlen, ngroups, dstate]` `bfloat16` with `stride(-1) = 1`
  (`ssd_combined.py:94-97` makes them so). In the Mamba-2 mixer they are views of one packed
  `xBC` tensor `[seqlen, conv_dim]` (`mamba.py:499-507, 604, 622-623`: `torch.split` along the last
  axis, then `.view(1, seqlen, ngroups, -1)`), strides `(seqlen·conv_dim, conv_dim, dstate, 1)` with
  `conv_dim = intermediate + 2·ngroups·dstate` per tensor-parallel rank
  (`python/sglang/srt/configs/mamba_utils.py:222`); in SGLang's tests they are contiguous. The
  claim allows any strides within its bounds (`stride(0)` an `int32` or `int64` argument, the
  others `int32`);
- `seq_idx`: `[1, seqlen]` `int32`, contiguous, the request of each token
  (`repeat_interleave(arange(num_prefills), seqlens)`, `mamba2_metadata.py:268-275`); the claim
  allows any non-negative entries;
- `out`: `[1, nchunks, ngroups, chunk_size, chunk_size]` `float32`, allocated by the host;
- the code base's sizes: `chunk_size` a power of two (asserted, `ssd_combined.py:77`): the model
  config's `mamba_chunk_size`, 256 by default in SGLang's Nemotron-H, Falcon-H1 and Granite-4.0-H
  configs (`python/sglang/srt/configs/nemotron_h.py:41`, `falcon_h1.py:166`,
  `granitemoehybrid.py:166`), 128 in some checkpoints, 64 when a config gives none
  (`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:137-141`); 8 to 256 in the
  tests. `ngroups` per rank: 8 (Mamba-2 / Codestral, Nemotron-H at TP 1), divided by the
  tensor-parallel size, down to 1 (Falcon-H1, Granite-4.0-H: `n_groups = 1`); 3 to 32 in the tests
  (`ngroups = n_heads`). `dstate`: 128 (Mamba-2, Nemotron-H, Granite-4.0-H), 256 (Falcon-H1); 5 to
  128 in the tests. `seqlen` up to the prefill token budget (thousands to tens of thousands of
  tokens), and one long prefill when chunked prefill is disabled
  (`python/sglang/srt/managers/scheduler.py:1372`), up to the grid limit (2,096,896 tokens at 8
  groups and chunks of 256);
- SGLang's tests (`test/registered/layers/mamba/test_mamba_ssm_ssd.py`), whose `bfloat16` cases run
  this kernel without `seq_idx`, here with `seq_idx` and in `bfloat16`: the single example (112
  tokens in chunks of 16, `ngroups ∈ {3, 32}`, `dstate ∈ {5, 128}`; outside CI also 128 tokens in
  chunks of 32, `ngroups` 3 to 32, `dstate` 5 to 128; lines 219–286), the intermediate states (128
  tokens in chunks of 32 and 256 in chunks of 64, `ngroups ∈ {4, 16}`, `dstate ∈ {32, 64}`, lines
  632–692), the continuous batches (64 tokens in chunks of 8 or 256, 768 in chunks of 128; `ngroups ∈ {4, 13}`,
  `dstate ∈ {5, 32}`; the request patterns of lines 292–326), the prefill-chunking cases
  (`ngroups = 16`, `dstate = 64`, chunks of 8 and 256, requests `(16, 2, 8, 13)`,
  `(270, 88, 212, 203)`, `(16, 20)`, lines 397–431).

Not part of the setup: `causal=True`; `seq_idx=None`, which SGLang's single-example and
intermediate-states tests use (`test_mamba_ssm_ssd.py:262, 661`) and serving never does; 3-D
`a`/`b` (no groups), `batch > 1`, `float16`/`float32` inputs and other output dtypes.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 38 launches, all holding: the tests' 17 patterns, five Mamba-2
mixer launches with `C`, `B` as views of a packed `xBC` tensor (`ngroups` 8, 4, 1; `dstate` 128 and
256; chunks of 128 and 256; up to 32768 tokens), and 16 edges (`dstate` 1, 33 and 1024; chunks of 2
and 2048; one token; 200 one-token requests; 2^20 tokens; a grid `z` of 65535; a random
non-monotone `seq_idx`; token-major `a`, `b`; `a.stride(0) = b.stride(0)` of `2^31 + 7` and
`262144·10240`, `int64` arguments; values near 2^54; subnormal operands; NaN and infinities). The worst error / tolerance is `0.0087`; every element outside a same-request pair is
`+0.0` bit for bit, the inputs are unchanged and the guard bands around `out` hold. Four wrong
outputs are rejected: the exact result rounded to `bfloat16`, the result
missing the last `tl.dot` step, a launch without the cross-request zeroing, and the causal launch,
which leaves tiles unwritten. The source's own test (`python -I source/ssd_bmm.py`) passes on its
nine cases (largest absolute error `3.9e-5`).
