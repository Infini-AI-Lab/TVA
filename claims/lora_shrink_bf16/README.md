# `lora_shrink_bf16`

**The kernel.** vLLM's `_lora_shrink_kernel` (`source/lora_shrink.py`, lines 308–423) with its
helpers `do_shrink_kernel` (185–305) and `mm_k` (83–182), inlined at their calls; the host is
`lora_shrink` (503–610) with `lora_kernel_meta` (462–500) and `_get_lora_a_ptr` (426–459). The grid
is `(SPLIT_K·cdiv(M, 32)·cdiv(R, 16), S, num_active_loras)`. A program takes one adapter group
(axis 2), one slice (axis 1), one `32 × 16` tile of the group's tokens and of the rank, and one of
the `SPLIT_K` parts of `K` (axis 0); it gathers the tile's token indices, multiplies those rows of
`inputs` by the adapter's LoRA A rows over its part of `K` (`bfloat16` operands, `float32`
accumulator), scales by `scaling`, and adds the tile into the zeroed `float32` output with a relaxed
`tl.atomic_add` whose result is unused. The claim is `Claims.LoraShrink.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
slice `s < S`, token `t < M`, rank column `j < R`, the output element `(s, t, j)` holds a `float32`
value `v`:
- a token with no adapter (`-1`): `v` is `+0.0`;
- a token on adapter `l ≥ 0`: when the `K` values of row `t` of `inputs` and of row `j` of adapter
  `l`'s weights of slice `s` are finite, `dotMag ≤ 2^120` and there is room in `float32`
  (`vMag + atomTol ≤ maxVal`), `v` is finite and within `tol` of `scaling·S`, with
  `S = Σ_k x[t, k]·A_{s,l}[j, k]` exact and `scaling` the host's double, whatever order the atomic
  additions ran in;
- every byte outside the output's `[S, M, R]` tensor is unchanged.

With `dotMag = Σ_k liftMag x·liftMag w` (a `bfloat16` subnormal counted at `2^-126`) and
`L = BLOCK_K·cdiv(K, BLOCK_K·SPLIT_K)` (`partLen`):
`tol = 2^-21·L·|scaling|·dotMag + 2^-100·(1 + |scaling| + dotMag)`.

## Conditions

`Inst.ok` and `Pre`:
- `R` a multiple of `16` (Exclusions), `K ≥ 1`, `K ≤ 2^17`, `M ≤ 2^22`, `R·K < 2^31`, `S ≥ 1`;
- the kernel's `int32` offsets do not wrap: `(M − 1)·R < 2^31`; for each token's adapter `l ≥ 0`,
  `l·R·K < 2^31`; with several slices, `(S − 1)·M·R < 2^31`;
- the grid within CUDA's limits (`n ≤ G ≤ maxL + 1`, `G < 2^16`, `S < 2^16`, fewer than `2^31`
  programs);
- `scaling` finite, and its `float32` rounding finite;
- the metadata is the host's (`lora_kernel_meta`): adapters in `[-1, maxL)`, the distinct ids
  increasing with their counts and cumulative sums, `token_indices_sorted_by_lora_ids` a permutation
  whose range for each group is that group's tokens; in memory, `lora_ids`, `num_tokens_per_lora`,
  `lora_token_start_loc` and the sorted indices hold these values (then `-1`, `0`, `0`);
- the buffers hold their tensors and are valid; with several slices the `uint64` table holds each
  slice's weight address; the output's `[S, M, R]` tensor is `+0.0`;
- the output shares no byte with `inputs`, any slice's weights, the table or the metadata.

Any `bfloat16` values in `inputs` and the weights are covered.

## Launches covered

`SETUP.md`: one to three slices, vLLM's default configuration (`BLOCK_K = 256`, `SPLIT_K = 64`
below 128 tokens; `32`, `8` from 128), either `EVEN_K`, ranks 16 to 256: `setup_c1`, `setup_c4` to
`setup_c8`, `setup_decode_qkv`, `setup_prefill_gate_up`, `setup_layer`, `setup_hidden2`,
`setup_hidden_max` (each with `setup_*_ok`; one fixed token-to-adapter map each). The host test's
cases 2 and 3 (ranks 8 and 1) are outside (Exclusions); case 9 launches nothing.
`claim_satisfiable` uses one token, hidden 512, rank 16, with a memory holding the host's metadata.

**Device facts** the tolerances rest on (all in `Device P`): `Device.DotBF16` (a whole `tl.dot` of
`bfloat16` operands into `float32`, `relDot16`, `absDot16`), `Device.DotBF16Chunk` (the same dot,
chunk by chunk, `rhoChunk16 = 2^-13`, `sigmaChunk16 = 2^-20`, `alphaChunk16 = 2^-143`),
`Device.AtomicAddF32` (`2^-23`, `2^-124`), and the totality facts for the dot and the atomic.

## Tolerances

For one element, each part `p` of the `SPLIT_K` parts accumulates its blocks of `BLOCK_K` with
`tl.dot_add`; its error against the part's exact sum is at most `errPart p`, the smaller of a chunk
by chunk bound (`errChunk`, `DotBF16Chunk`'s first arm per chunk of 64 `k`, under both forms of the
`+=`) and a step by step bound (`errStep = DotLoop.bound ρ α T (partLift p)`, `DotBF16` per step with
its addition). The scaled parts' roundings add `mulTol`, the `SPLIT_K` atomic additions into `+0.0`
add `atomTol = DotLoop.bound 2^-23 2^-124 SPLIT_K vMag`, and the host's double `scaling` differs from
the kernel's `float32` one by at most one rounding. Together they are at most `tol` on every covered
instance (`T·2^-13` of `|scaling|·dotMag` below 128 tokens, `T·2^-16` from 128, `T = cdiv(K,
BLOCK_K·SPLIT_K)`).

## Exclusions

- **Ranks not a multiple of 16**: line 237's `tl.max_contiguous(tl.multiple_of(offset_n % N, 16),
  16)` is false on the last tile, and a false hint is undefined behaviour: a limit of the kernel.
- **An adapter `l` with `l·R·K ≥ 2^31`**, **`(M − 1)·R ≥ 2^31`**, **`(S − 1)·M·R ≥ 2^31`**: `int32`
  products of the kernel wrap (lines 249, 290, 286): limits of the kernel, outside real use.
- **`R·K ≥ 2^31`**, **`K > 2^17`**, **`M > 2^22`**: outside `SETUP.md` and real shapes.
- **A grid beyond CUDA's limits**, **a non-finite `scaling`**, **an output sharing bytes with an
  input**: outside real use.
- Other configurations, `USE_GDC`, `STORE_PARTIALS`: not part of the setup.

`check_gpu.py` (one H200) checks `Post` on 29 covered launches and rejects corrupted outputs.
