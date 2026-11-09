# Setup of `chunk_delta_h_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`), `fla/ops/common/chunk_delta_h.py:59`
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, launched by `chunk_gated_delta_rule_fwd_h`
(`chunk_delta_h.py:687`), the state recurrence of the chunked gated delta rule's forward.

## The source

`source/chunk_delta_h.py` holds:
- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, the `@triton.jit` kernel (lines 58–349, at
  their upstream line numbers): for each sequence, value head and block of 32 value columns, the
  `float32` state `h` (`[K, V]`, in up to four blocks of 64 rows) is carried across the chunks of
  64 tokens; in each chunk the state entering it is stored to `h[chunk]` (`bfloat16`),
  `v_new = u − w·h` is computed (two-step `tl.dot` chain over the `K` blocks) and stored, and the
  state is updated, `h ← h·exp2(g_last) + kᵀ·(v_new·exp2(g_last − g))` (`tl.dot` with the state as
  accumulator); the state starts from `h0` or zero, and the last one goes to `ht`;
- `exp2`, the `@triton.jit` helper it calls (lines 47–48), copied from `fla/ops/utils/op.py:46-47`
  (fla's default branch, `FLA_USE_FAST_OPS` unset);
- the host `chunk_gated_delta_rule_fwd_h` (lines 398–467), from `chunk_delta_h.py:687-748`, with
  fla's torch helpers `prepare_lens`, `_segmented_arange`, `prepare_chunk_indices`,
  `prepare_chunk_offsets` (`fla/ops/utils/index.py:40-42, 113-134, 155-172`; lines 352–395);
- `make_inputs` (the inputs as the gated delta rule's forward builds them: L2-normalized keys, the
  gate's chunk-local cumulative sum in log2 units, the WY representation `w`, `u` of each chunk),
  `reference` (the recurrence in `float64`) and the test under `if __name__ == "__main__":`.

Changes from the code base (each listed with its `file:line` in the module docstring): the
license header, the imports of fla's modules and the module-level warp lists
(`chunk_delta_h.py:1-37`) replaced by the docstring, the imports of `torch`, `triton`,
`triton.language` and the helper `exp2`; the decorators `@triton.heuristics({...})` and
`@fla_cache_autotune(...)` (`chunk_delta_h.py:40-57`) removed (blank lines in their place),
`@triton.jit(do_not_specialize=['T'])` kept; no line of the kernel changes. The host drops its
`@dispatch('common')` decorator (backend selection; Triton on NVIDIA GPUs), passes the flags the
heuristics computed and one configuration (below) as launch arguments, and the grid as a tuple;
its checks, allocations and arguments are the code base's. The index helpers are copied without
their `@tensor_cache` memo. The backward kernel (`chunk_delta_h.py:352-685`) is not copied.

## The config

Removed: `@fla_cache_autotune(configs=[triton.Config({'BV': BV}, num_warps=w, num_stages=s) for w
in [2, 4] for s in [2, 3, 4] for BV in [32, 64]], key=['H', 'HV', 'K', 'V', 'BT',
'STATE_V_FIRST'])` (the lists on an H200: Hopper, Ampere-class and Ada-class shared memory) and
`@triton.heuristics` (`chunk_delta_h.py:40-57`).

Chosen: `BV = 32`, `num_warps = 4`, `num_stages = 3`. Timed on the H200 (median of five
`do_bench` runs, 8192 tokens, every list entry): at fla's default GatedDeltaNet layer (`K = 256`,
`V = 512`, 6 heads, `fla/models/gated_deltanet/configuration_gated_deltanet.py:26-33`) it is the
fastest entry in both state layouts (0.302 ms `[K, V]`, 0.297 ms `[V, K]`; next `num_stages = 2`,
0.318 and 0.346 ms); at Qwen3-Next's heads (`K = V = 128`, 16 key and 32 value heads) it is second
to `num_stages = 4` (0.185 vs 0.177 ms, 0.189 vs 0.183 ms), which needs more shared memory than the
H200 has at `K = 256`. At `K = 256` four entries do not compile with the `[K, V]` layout (`BV = 32`
or `64` with `num_warps = 4`, `num_stages = 4`; `BV = 64`, `num_warps = 2`, `num_stages = 4`;
`BV = 64`, `num_warps = 4`, `num_stages = 3`) and two with the `[V, K]` layout (`BV = 32` or `64`
with `num_warps = 4`, `num_stages = 4`). The same configuration serves every launch, with or
without the gate. Grid `(cdiv(V, 32)·N·HV,)`.

Flags, as the gated delta rule's forward sets them (`fla/ops/gated_delta_rule/chunk.py:95-105`,
`chunk_gated_delta_rule_fwd`) and the delta rule's (`fla/ops/delta_rule/chunk.py:39-49`,
`chunk_delta_rule_fwd`, and its backward's recomputation of the states, `:87-96`):
`USE_G = g is not None` (either way: the gated delta rule passes its gate `g`; the delta rule,
fla's DeltaNet, passes `g = None`), `USE_GK = False` (no `gk`), `SAVE_NEW_VALUE = True`
(`save_new_value` at its default), `STATE_V_FIRST = state_v_first` (either way: `False` is its default, as `chunk_gated_delta_rule` and its direct
callers use it; `True`, the `[V, K]` layout of the state, `h`, `initial_state` and the final state,
is what fla's layers that reach this kernel without `gk` pass: `fla/layers/gated_deltanet.py:313,334`,
`fla/layers/mom.py:605,631,745,758`; `fla/layers/kda.py:278,297` passes it too, but KDA reaches
the kernel only with `gk`, outside this setup),
`USE_INITIAL_STATE = initial_state is not None` and `STORE_FINAL_STATE = output_final_state`
(each either way), `IS_VARLEN = cu_seqlens is not None` (either way); `BT = chunk_size = 64` (the
default; fla's layers and tests). Triton's knobs at their defaults.

Not part of this setup: `gk` (KDA's per-key gate), `save_new_value=False`, other chunk sizes,
`bfloat16` or `float16` initial states, and `float16` inputs.

## The launches

The claim covers the host's launches with the flags above, `bfloat16` `k`, `w`, `u` (the models'
dtype), `float32` `g` (or `None`), `h0`, `ht`, all contiguous, with shapes as variables:
- `k`: `[B, T, H, K]`; `w`: `[B, T, HV, K]`; `u`: `[B, T, HV, V]`; `g`: `[B, T, HV]` (the
  chunk-local cumulative sum of the per-token log decay in log2 units, `chunk_local_cumsum(...,
  scale=RCP_LN2)`, `chunk.py:63-69`); `h0`, `ht`: `[N, HV, K, V]` (`[N, HV, V, K]` with
  `state_v_first`); the outputs `h` (`[B, NT, HV, K, V]`, or `[B, NT, HV, V, K]` with
  `state_v_first`, `bfloat16`), `v_new` (like `u`) and `ht` (zeros, then written), allocated by the
  host;
- fixed length (padded batches): `B ≥ 1` rows of `T` tokens each (`T ≥ 1`), `N = B`;
- variable length (packed batches): `B = 1`, `cu_seqlens` (`int64`, as fla's tests pass it, or
  `int32`, as fla's layers build it) from `0` to `T`, nondecreasing (sequences of any length,
  `0` included), `N` sequences, and `chunk_offsets` as `prepare_chunk_offsets` builds it;
- heads: `H ≥ 1` key heads, `HV` value heads a multiple of `H` (grouped value heads); `1 ≤ K ≤ 256`
  (the host asserts `K ≤ 256`), `V ≥ 1`, `HV·V < 2^31`: fla's tests (`tests/ops/test_gdn.py:149-169, 244-253,
  752-771, 848-863`): `D = K = V` in `{32, 60, 64, 100, 128}`, `H` 2 to 8, `HV` 3 to 8, `T` up to 2048
  per row and 8192 per packed batch (`test_chunk_state_v_first`, `test_gdn.py:414-425`, for the
  `[V, K]` layout: `(B, T, H, D)` = `(1, 63, 1, 64)`, `(2, 500, 3, 60)`, `(3, 1024, 4, 128)`,
  `(4, 2048, 8, 64)`), sequences `[0, 15]`, `[0, 256, 500, 1000]`,
  `[0, 15, 100, 300, 1200, 2000]`, `[0, 8192]`; Qwen3-Next's linear attention (16 key heads,
  32 value heads, `K = V = 128`); fla's default GatedDeltaNet layer (6 heads, `K = 256`,
  `V = 512`); batches of thousands to hundreds of thousands of tokens;
- without the gate (`g = None`, the delta rule): fla's delta-rule tests
  (`tests/ops/test_delta.py:21-26, 95, 167-170`: `(B, T, H, D)` = `(1, 63, 1, 64)`,
  `(2, 100, 4, 60)`, `(2, 1000, 3, 128)`, `(2, 1024, 4, 128)`, `(3, 2000, 4, 128)`,
  `(4, 2048, 8, 64)`, `(1, 64, 2, 32)`; sequences `[0, 15]`, `[0, 111, 500]`,
  `[0, 256, 500, 900, 1000]`, `[0, 15, 100, 300, 1200, 1599, 1800, 2000]` with `int32`
  `cu_seqlens`); fla's default DeltaNet layer (`fla/layers/delta_net.py:259`,
  `fla/models/delta_net/configuration_delta_net.py:26-35`: 16 heads, `K = V = 128`), the state in
  the `[K, V]` layout;
- the input values real use produces: `k` L2-normalized per token, `w`, `u` the WY representation
  (`A·(β·k·2^g)`, `A·(β·v)`, `2^g = 1` without the gate, `β ∈ (0, 1)` or `(0, 2)` with negative eigenvalues allowed), `g`
  nonincreasing within each chunk and at most `0` (log decays), `h0` a previous final state.

The claim leaves out (README, Exclusions): `T`, grid or index sizes beyond `int32` (outside real
use).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, peak device memory 1.8 GiB) runs 77 launches through the host, all
holding `Post`: fla's eleven fixed-length test shapes and six variable-length ones (fla's tests run
them in `float16`; here in the setup's `bfloat16`, with an initial and a final state), one with
`int32` `cu_seqlens`; Qwen3-Next's heads on `2 × 4096` tokens and on a packed batch of five
sequences (lengths 1000, 3000, 0, 1, 4096), each with the four combinations of initial and final
state; fla's default GatedDeltaNet layer (`K = 256`, `V = 512`), fixed and packed; with
`state_v_first=True`, the four `test_chunk_state_v_first` shapes (in `bfloat16`; the test runs
`float16`), a packed batch with grouped value heads, the GatedDeltaNet layer packed (`int32`
`cu_seqlens`, the four flag combinations) and fixed, Qwen3-Next's heads packed, and three edges;
without the gate (`g = None`), fla's seven delta-rule test shapes and four variable-length ones
(`int32` `cu_seqlens`), fla's default DeltaNet layer fixed and packed (the four flag combinations),
grouped value heads, the `[V, K]` layout and three edges; and edges (`K = V = T = 1`; `K = 192`,
`V = 33`; `K = 256`; only empty sequences; no decay over 4096 tokens; slow decay over 16,384
tokens; strong decay; `β` up to 2; initial states of order `5·10^3` and `10^-30`, and with `-0` entries (both layouts); 256 sequences of
1 to 64 tokens; a gate that is not monotone within its chunks). `Post`'s `float32` states are the
kernel's own, read by a copy of the kernel with one more store per state block (its outputs equal
the host's bit for bit): `Y 0` is the initial state (as a value: an initial `-0` is the state `+0`), every stored state is the `bfloat16` rounding
of its state and `ht` the last state, bit for bit; the worst error / tolerance is `0.926` (a step)
and `0.996` (`v_new`); every stored state is in range (`StOk`), every output is finite, `Vals`
holds, the host's `chunk_offsets` equal `cbase`, and a direct launch with the outputs between guard
regions leaves the guards and inputs unchanged, writes every output element and equals the host's
outputs bit for bit. Forty-seven corrupted outputs on four launches (an element moved by 4 times
its tolerance, two sequences' chunks exchanged, and the exact recurrence with one bug: no decay, a
dropped chunk, no gate, the wrong key head, the first token's gate, `g_last` one token early, `ht`
one chunk early, every state times `1 ± 2^-7` each chunk) are refuted, a changed guard element is
rejected, and `Vals` fails on an input whose padded state rows grow past its bound (`K = 60`,
`g = 8` over 33 chunks).
