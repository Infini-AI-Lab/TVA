# Setup of `chunk_fwd_o_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/common/chunk_o.py:58`,
the kernel `chunk_fwd_kernel_o`, launched by fla's `chunk_fwd_o` (`chunk_o.py:543-586`).

## The source

`source/chunk_o.py` holds:
- `chunk_fwd_kernel_o` (lines 83-181), the `@triton.jit` kernel, verbatim but for its decorators
  (below): the output of chunked linear attention, one program per (block of `BV` value columns,
  64-token chunk, batch row and value head). For a token `t` of a chunk with state `S` (`h`'s
  chunk), value head `j` and query/key head `j // (HV // H)`, it computes
  `scale·(2^g[t]·q[t]·S + Σ_{s ≤ t in the chunk} (q[t]·k[s])·2^(g[t]−g[s])·v[s])`, with
  `γ[j]·(i+1)` for `g[t]` (`i` the local position of `t` in the chunk) when `g_gamma` is given,
  and the gates `1` without either; the two products with `tl.dot` over `K` in steps of `BK`, the scores rounded to
  `bfloat16` before their product with `v`;
- `unflatten_program_id` (lines 69-76), copied verbatim from `fla/ops/utils/op.py:22-29`, and
  `exp2` (lines 79-80), `fla/ops/utils/op.py:46-47` as fla defines it when `FLA_USE_FAST_OPS` is
  unset (its default, `op.py:32,43`): the `@triton.jit` helpers the kernel calls;
- the host: `chunk_fwd_o` (lines 223-273, fla's `chunk_o.py:543-586`) and fla's torch helpers
  `prepare_lens`, `_segmented_arange` and `prepare_chunk_indices` (`fla/ops/utils/index.py:40-42,
  113-134, 155-164`), which build the `[NT, 2]` table of (sequence, chunk) pairs of a
  variable-length batch;
- `reference`, the output in `float64` chunk by chunk, and the test under `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring: the kernel's
decorators `@triton.heuristics` and `@fla_cache_autotune` removed
(`@triton.jit(do_not_specialize=['T'])` kept); the module-level config lists removed; the host
passes the heuristics' flags and, for the autotuner's choice, the configuration its new argument
`BK` names (default 64) explicitly; the host's decorator
`@dispatch('common')` (backend selection; Triton on NVIDIA GPUs) removed; the index helpers copied
without their `@tensor_cache` memo. The kernel's signature and body are unchanged.

## The config

- Removed: `@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'USE_G_GAMMA':
  lambda args: args['g_gamma'] is not None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not
  None})` and `@fla_cache_autotune(configs=_O_CONFIGS, key=['H', 'HV', 'K', 'V', 'BT',
  'STATE_V_FIRST'], ...)` (`chunk_o.py:51-56`).
- Covered: every entry of the source's list on an H200, `BK = BV = 128` with 8 warps,
  `BK = BV = 64` with 4 warps, `BK = BV = 32` with 2 warps, all with 3 stages
  (`chunk_o.py:34-42`; Hopper is not Blackwell, so the first entry is kept). The autotuner's
  choice depends on the shape: `BK = BV = 128` with 8 warps is the faster on fla's benchmark
  shapes with 64 and 96 heads (`B1_T8192_H96_D128`, `B4_T4096_H64_D128`), `BK = BV = 64` on
  others. The host's argument `BK` (32, 64 or 128; default 64) names the entry, and the host
  passes `BV = BK` and its `num_warps`.
- Flags set by the host: `USE_G = g is not None`, `USE_G_GAMMA = g_gamma is not None`,
  `IS_VARLEN = cu_seqlens is not None`. The host itself passes `STATE_V_FIRST = state_v_first`,
  `BT = chunk_size = 64`, `T = q.shape[1]`, `H`, `HV`, `K`, `V`, and `scale = K ** -0.5` when none
  is given.

## The launches

The claim covers the host's launches as fla's linear-attention ops make them:
- the delta-rule family: the gated DeltaNet chunk forward (`fla/ops/gated_delta_rule/chunk.py:111-121`:
  gates `g` from `chunk_local_cumsum(…, scale=RCP_LN2)` or `gdn_gate_chunk_cumsum`,
  `chunk.py:51-68`: `float32`, cumulative within each 64-token chunk, nonpositive), the delta rule
  (`fla/ops/delta_rule/chunk.py:51-61`, `g=None`), comba (`fla/ops/comba/chunk.py:76-85`), the
  preconditioned gated delta rule (`fla/ops/precond_gated_delta_rule/chunk.py:109-119`) and cyfa
  (`fla/ops/cyfa/chunk.py:621-632`);
- simple GLA (`fla/ops/simple_gla/chunk.py:45-57`): with `g` (`chunk_local_cumsum(…,
  scale=RCP_LN2)`, `chunk.py:173-179`; fla's simple GLA layer, `fla/layers/simple_gla.py:244-253`,
  `state_v_first=True`), with `g_gamma` (`[H]` `float32` per-head decays, scaled by `RCP_LN2`,
  `chunk.py:180-181`), or with neither (linear attention, `fla/ops/linear_attn/chunk.py:73`); its
  `g_gamma` callers are RetNet (`fla/ops/retention/chunk.py:68-79`: `log(1 − 2^(−5−h))`, used by
  `fla/layers/multiscale_retention.py:256`) and lightning attention
  (`fla/ops/lightning_attn/chunk.py:74-85`: `−(8/H)·(1 − layer_idx/num_layers)·h`); fla's
  benchmark registry runs both (`benchmarks/ops/registry.py:317-322, 576-582`). No caller passes
  both `g` and `g_gamma` (`chunk_simple_gla`'s docstring: "Only one of `g` or `g_gamma` should be
  provided", `chunk.py:264-267`);

with:
- `bfloat16` `q`, `k` (`[B, T, H, K]`), `v` (`[B, T, HV, V]`), `h` (`[B, NT, HV, K, V]`, or
  `[B, NT, HV, V, K]` with `state_v_first=True`, as fla's gated DeltaNet and simple GLA layers and
  cyfa call it: `fla/layers/gated_deltanet.py:302-317`, `fla/ops/cyfa/chunk.py:619-631`), and the
  output; `float32` `g`, or `float32` `g_gamma` with at least `HV` entries, or neither;
  contiguous tensors; `chunk_size = 64` (the default);
- `state_v_first` either way;
- each of the three configurations `BK = BV = 32`, `64`, `128` ("The config": the autotuner's
  choice depends on the shape);
- fixed length (`cu_seqlens = None`), `B ≥ 1` rows of `T` tokens, `B·HV < 65536` (README,
  Exclusions); or variable length (`B = 1`, which the gated delta rule and simple GLA assert,
  `fla/ops/gated_delta_rule/chunk.py:550-553`, `fla/ops/simple_gla/chunk.py:328-333`):
  `cu_seqlens` `int32` (the layers' `prepare_cu_seqlens_from_mask`,
  `fla/ops/utils/index.py:58-63`) or `int64` (fla's tests, `torch.LongTensor`), nonnegative,
  nondecreasing and at most `T`, sequences of any length from 0, and `chunk_indices` as
  `prepare_chunk_indices` builds it (the host's default) or as the caller passes it (the same rows
  in any order, `h`'s chunks in that order);
- shapes as variables: `H ≥ 1` query/key heads and `HV` a multiple of `H` value heads (grouped
  value heads), `K ≤ 256`, `V ≤ 512`; fla's tests (`tests/ops/test_gdn_kernels.py:451-479`:
  `(B, T, H, HV, D) = (2, 128, 2, 2, 64)`, `(2, 128, 2, 4, 64)` with gates, `(2, 128, 2, 2, 64)`
  without, in `bfloat16`; `:706-722`: one packed batch of 8192 tokens, `cu_seqlens = [0, 5000,
  8192]`, 8 heads, `K = V = 128`; `tests/ops/test_gdn.py:74-81, 157-167`: head dims 60, 64, 100,
  128); fla's op benchmarks (`benchmarks/ops/registry.py:203-209`: up to `B = 8`, `T = 16384`,
  96 heads, head dim 256); fla's gated DeltaNet defaults
  (`fla/models/gated_deltanet/configuration_gated_deltanet.py:26-33`: 6 heads, `K = 256`,
  `V = 512`); fla's multiscale retention defaults (`fla/layers/multiscale_retention.py:74-77`: 8
  heads, `K = 128`, `V = 256`); Qwen3-Next's gated DeltaNet layers, whose integration fla's
  `README.md:55` announces (16 query/key heads, 32 value heads, `K = V = 128`), from prefills of a
  few tokens to training batches of `2^20` tokens.

Chunk sizes 16 and 32 are not part of the setup: they are an opt-in keyword of the gated delta
rule and simple GLA (`fla/ops/gated_delta_rule/chunk.py:538-540`), which no layer or model config
passes; fla tests them only in `float32` and `float16` (`tests/ops/test_gdn.py:245-250`,
`tests/ops/test_retention.py:70-79`). Nor are `float16` or `float32` inputs.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, peak device memory 2.28 GiB) passes on 52
launches, each through the host and through a direct launch into a guarded output prefilled with
a sentinel. They are the claim's coverage launches except the two `2^20`-token ones, which need
more than 4 GB. At `BK = BV = 64`: fla's three `bfloat16` test cases, its packed deterministic
test, Qwen3-Next's heads on a packed prefill of 8 × 4096 tokens with `state_v_first=True`, fla's
gated DeltaNet defaults `K = 256`, `V = 512` with `int64` tables, the benchmark shapes with 96
heads, 64 heads without gates and head dim 256, head dims 60/100, RetNet's `g_gamma` at fla's
retention defaults on a fixed-length and a packed (`int64`) batch, lightning attention's
`g_gamma` at `B1_T8192_H96_D128` and `B4_T2048_H16_D128`, and 2047 one-token rows at 32 value
heads; and edges: `K = V = 1`, `K = 17`, `V = 200` with empty and one-token sequences, a permuted
`chunk_indices`, tokens after `cu_seqlens[N]`, scales `2^-64` and `2^64`, fla's test gates
`0.1·randn`, gates from `−3000` to `60`, decays `γ = 0, 1, −64, −2^-6` with `state_v_first`,
values near `2^60`, subnormal inputs, and a NaN key, an infinite value and a NaN gate. At
`BK = BV = 32`: fla's gated test case, the deterministic test, head dims 60/100. At
`BK = BV = 128`: the three benchmark shapes, Qwen3-Next's 8 × 4096 prefill, the gated DeltaNet
defaults, head dims 60/100, RetNet's fixed-length batch, lightning attention at
`B1_T8192_H96_D128`. At both: the edges `K = V = 1`, `K = 17`/`V = 200`, the permuted table, the
decay edges, the large gates, the large values and the subnormals. And two bases of the negative
tests in each configuration. On every launch the host's and the direct launch's outputs agree
bit for bit, every table is `TablesOk`, every in-range element is finite and within `tol` (worst
error / tolerance `0.906` in each configuration; the median tolerance is `0.008` to `0.030` of
`|ref|`), every element outside the written ones keeps the sentinel, and the inputs are
unchanged. Nine outputs of other meanings (no causal mask, `e` for `2` in the gates, the wrong
query head, the state read untransposed, the scale squared, the previous chunk's state, one
element moved by twice its tolerance, the `g_gamma` gate at the global token position, and
without its `+ 1`) are rejected in each configuration. A fixed-length batch of `B = 2048` rows
at 32 value heads (`B·HV = 65536`) fails on the host (`Triton Error [CUDA]: invalid argument`);
`B = 2047` runs.
