# Setup of `chunk_bwd_dv_local_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/common/chunk_o.py:466`,
the kernel `chunk_bwd_kernel_dv_local`, launched by fla's `chunk_bwd_dv_local`
(`fla/ops/common/chunk_o.py:654`).

## The source

`source/chunk_o.py` holds:
- `chunk_bwd_kernel_dv_local` (lines 74–149), the `@triton.jit(do_not_specialize=['T'])` kernel,
  with its signature and body verbatim from the source's lines 465–540: one program per (chunk of
  `BT` tokens, batch row and value head) forms the causal-masked `[BT, BT]` matrix
  `A = (k @ q^T)·scale`, times the gate `2^(g[j] − g[i])` when a gate is given (`USE_G`), in
  `float32`, rounds it to `bfloat16` and stores `dv = A @ do` for its chunk, over fixed-length
  rows or packed variable-length sequences;
- `exp2` (lines 70–71), the `@triton.jit` helper it calls, from `fla/ops/utils/op.py:46-47`;
- the host: `chunk_bwd_dv_local` (lines 191–241, from `chunk_o.py:653-706`), with fla's torch
  helpers `prepare_lens`, `_segmented_arange` and `prepare_chunk_indices`
  (`fla/ops/utils/index.py`), which build the `[NT, 2]` table of (sequence, chunk) pairs of a
  variable-length batch;
- `reference`, the value gradient in `float64` chunk by chunk, and the test under
  `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring:
- the kernel's decorators `@triton.heuristics` and `@fla_cache_autotune` removed
  (`chunk_o.py:450-464`; `@triton.jit(do_not_specialize=['T'])` kept);
- the helper `exp2` copied from the default branch of `fla/ops/utils/op.py:43-53` (the one taken
  when the environment variable `FLA_USE_FAST_OPS` is not `'1'`) as a plain top-level `def`;
- the host: the decorator `@dispatch('common')` (backend selection; the Triton default on NVIDIA
  GPUs) removed; the device query `check_shared_mem('hopper', …)` replaced by its value on an H200
  (`CONST_TILING = 128`); the launch passes the flags the heuristics computed and one
  configuration explicitly (below);
- the index helpers copied without their `@tensor_cache` memo, `F.pad` written
  `torch.nn.functional.pad`.

The kernel's body is unchanged.

## The config

- Removed: `@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'USE_G_GAMMA':
  lambda args: args['g_gamma'] is not None, 'USE_A': lambda args: args['A'] is not None,
  'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})` and
  `@fla_cache_autotune(configs=[triton.Config({}, num_warps=w, num_stages=s) for w in
  CHUNK_BWD_DV_LOCAL_NUM_WARPS for s in [2, 3, 4]], key=['H', 'HV', 'K', 'V', 'BT', 'BK', 'BV',
  'USE_G'])` (`chunk_o.py:450-464`); on an H200 (Hopper) `CHUNK_BWD_DV_LOCAL_NUM_WARPS = [2, 4]`
  (`chunk_o.py:28-31`).
- Chosen: `num_warps=4, num_stages=3`, the fastest of the six on an H200 at gated DeltaNet's
  default head dims (`K = 256`, `V = 512`, 64-token chunks, with a gate: 46.8 µs against 50.5 to
  59.6 µs for the others); at head dims 128 the six are within 1% of one another.
- Flags set by the host: `USE_G = g is not None`, `USE_G_GAMMA = g_gamma is not None`,
  `USE_A = A is not None`, `IS_VARLEN = cu_seqlens is not None`.
- The wrapper's own launch values: `BT = chunk_size`; `BK = min(max(next_power_of_2(K), 16), 128)`,
  `BV` the same for `V` (`CONST_TILING = 128` on an H200); the grid `(NT, B·HV)` with
  `NT = cdiv(T, BT)` (fixed length) or `len(chunk_indices)` (variable length); `T = k.shape[1]`;
  `dv = torch.empty_like(do)`.

The covered path: `USE_G` either way, `USE_G_GAMMA = USE_A = False` (`g_gamma = A = None`: no caller
in fla v0.6.0 passes them to `chunk_bwd_dv_local`), `IS_VARLEN` either way.

## The launches

The claim covers `chunk_bwd_dv_local` as fla's chunked delta-rule backward passes call it
(`fla/ops/gated_delta_rule/chunk.py:172-181`, gated DeltaNet, with `g`, the chunk-local cumulative
log decay in base 2 that the forward builds with `chunk_local_cumsum(g, chunk_size,
scale=RCP_LN2)`, `chunk.py:63-69`; `fla/ops/delta_rule/chunk.py:98-106`, DeltaNet, `g=None`;
`fla/ops/comba/chunk.py:124` and `fla/ops/precond_gated_delta_rule/chunk.py:185`, with `g`), with:
- `bfloat16` `q`, `k` (`[B, T, H, K]`), `do` and `dv` (`[B, T, HV, V]`), contiguous; `float32` `g`
  (`[B, T, HV]`) or none; `HV` a multiple of `H` (grouped value heads, `chunk.py:526-531`);
- `scale` a Python `float` (the callers' default `K ** -0.5`; fla's tests use `0.1` and `1`);
- `chunk_size` 64 (the callers' default), and 16 and 32 (`tests/ops/test_gdn.py:244-251`);
- fixed-length rows (`cu_seqlens = None`) with `B·HV < 65536` (a kernel limit: README,
  Exclusions), or a packed variable-length batch (`B = 1`), `cu_seqlens` `int64` (fla's tests) or
  `int32` (`prepare_cu_seqlens_from_lens`'s default), nonnegative, nondecreasing and at most `T`,
  sequences of any length from 0, and `chunk_indices` as `prepare_chunk_indices` builds it (or the
  same table in any row order);
- head dims `1 ≤ K ≤ 256` (the chunked forward asserts `K ≤ 256`,
  `fla/ops/common/chunk_delta_h.py:716`) and any `V ≥ 1`.

The concrete launches:
- the host's test cases (source lines 278–290), each without and with a gate: the shapes of
  `tests/ops/test_gdn.py` `test_chunk` (lines 149–167: `B, T, H, HV, D` = `(2, 500, 4, 4, 60)`,
  `(3, 1024, 4, 4, 100)`, `(2, 1000, 2, 8, 128)`, …), `test_chunk_with_chunk_size` (lines
  244–251: `(1, 64, 2, 4, 32)` with chunk sizes 16, 32 and 64) and `test_chunk_varlen` (lines
  753–764: `[0, 15]`, `[0, 256, 500, 1000]`, `[0, 15, 100, 300, 1200, 2000]` with 4/4 heads and
  head dims 60, 64, 100, and 2/8 heads), gated DeltaNet's default heads and Qwen3-Next's grouped
  value heads, and a packed batch with sequences of 1, 62, 1, 1, 935 and 3096 tokens;
- the code base's benchmark shapes (`benchmarks/ops/registry.py:203-210`, `K = V = D`):
  `(B, T, H, D)` = `(1, 8192, 96, 128)`, `(2, 16384, 16, 128)`, `(4, 2048, 16, 128)`,
  `(4, 4096, 64, 128)`, `(8, 2048, 32, 256)`, `(8, 1024, 8, 64)`;
- gated DeltaNet's default layer (`fla/models/gated_deltanet/configuration_gated_deltanet.py:25-33`,
  `fla/layers/gated_deltanet.py:121-128`: 6 heads, `head_dim = 256`, `expand_v = 2`, so `K = 256`,
  `V = 512`) at its default context of 2048 tokens, 8 rows;
- a packed training batch of `2^16` tokens with grouped value heads (16 query/key heads, 32 value
  heads, head dims 128, Qwen3-Next's gated DeltaNet layers), `int32` and `int64` tables;
- the largest fixed-length grid the kernel launches, `B·HV = 65535`.

Not part of the setup: `g_gamma` or `A` given (`USE_G_GAMMA`, `USE_A`), `float16` or `float32`
`q`, `k`, `do` (fla's tests mostly run `float16`), an `int` `scale` (an integer `1` is a
compile-time constant, another specialisation), chunk sizes other than 16, 32 and 64.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0, `CUDA_VISIBLE_DEVICES=3`, peak device memory 2.17 GiB) passes
all 46 checks: the host's 13 test cases, the six benchmark shapes, gated DeltaNet's default layer,
the `2^16`-token packed batch with `int32` and `int64` tables and the `B·HV = 65535` grid, each
without and with a gate; `Inst.ok` edges (head dims 1, 15/17 and 256/1000, eight value heads per key
head, empty and short sequences, chunk sizes 16 and 32 on packed batches, sequences not covering
`[0, T)`, a permuted `chunk_indices`, direct launches with a sentinel-filled `dv`); `Pre` edges
(inputs scaled to `kqMag` near `2^47` and `|do|` near `2^40`, subnormal `do`, all-positive inputs,
strong decays with gate differences far below `−126`, gates rising by up to 63 within a chunk,
`scale` `2^-32` and `2^32`). Every written element is finite and within `tol` of `ref`; the worst
error / tolerance is `0.934`; `ValsOk` and `TablesOk` hold on every input, the inputs are unchanged, and
every element of `dv` outside the sequences keeps its sentinel. The four negative tests are rejected
(one element moved, the ungated output against the gated specification, chunk size 32 against 64, an
unwritten element changed), and a launch with `B·HV = 65536` (`B = 4096`, `HV = 16`) fails on the
host with `Triton Error [CUDA]: invalid argument`.
