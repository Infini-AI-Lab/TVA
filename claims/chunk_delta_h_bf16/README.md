# `chunk_delta_h_bf16`

**The kernel.** fla's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (`source/chunk_delta_h.py`,
kernel lines 58–349, with the `@triton.jit` helper `exp2`, lines 47–48; host
`chunk_gated_delta_rule_fwd_h`, lines 398–467): the state recurrence across chunks of the chunked
gated delta rule's forward, `bfloat16` inputs, `float32` state, `BT = 64`, `BV = 32`. The grid is
`(cdiv(V, 32)·N·HV,)`; each program takes one sequence, one value head and one block of `32` value
columns, and runs over the sequence's chunks of `64` tokens: it stores the state rounded to
`bfloat16` into `h`, computes the new values `v_new = u − w·h` (stored as `bfloat16`), gates them and
decays the state with `exp2` of the gate (`USE_G`), and adds `kᵀ·v_new` to the state. With
`STORE_FINAL_STATE` the last state is stored to `ht` in `float32`. The claim is
`Claims.ChunkDeltaH.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
sequence `n < N`, value head `hv < HV`, value column `j < V`, with `h[t]` the stored column
`h[cbase n + t, hv, :, j]`, `t < nch n = cdiv(len n, 64)`:
- every `v_new` element of the sequence holds a `bfloat16` value; when `Vals` holds and `StOk t'`
  for every `t' ≤ t`, it is finite and within `tolV t i = 2^-8·vAbs t i + 2^-100` of
  `vrefS t i = u[i, j] − Σ_c w[i, c]·h[t][c]`;
- every element of every `h[t]` holds a `bfloat16` value;
- there are `float32` states `Y 0, …, Y (nch n)` such that: with `Vals`, `Y 0` is the initial state
  (`0` without one), finite; for each `t`, a finite `Y t c` gives `h[t][c] = bf16(Y t c)` exactly;
  for each `t`, with `Vals` and `StOk t'` for every `t' ≤ t`, `Y t` and `Y (t + 1)` are finite and
  `‖Y (t + 1) − Step t (Y t)‖ ≤ errStep t`, where
  `Step t y c = 2^g_last·y c + Σ_i k[i, c]·2^(g_last − g_i)·vrefS t i` is one exact step of the
  recurrence (real arithmetic, `2^x` the real power; both powers `1` without the gate); with
  `STORE_FINAL_STATE`, `ht[n, hv, :, j]` is `Y (nch n)` exactly;
- every byte outside `h`'s, `v_new`'s and (when stored) `ht`'s elements is unchanged.

The steps are bounded one at a time: the drift of `h[t]` from the exact recurrence started at `h0`
is not bounded (Exclusions).

## Conditions

`Inst.ok`, `Pre`, and the value conditions of `Post`'s numeric clauses:
- `1 ≤ H` divides `HV`, `1 ≤ HV < 2^16`; `1 ≤ K ≤ 256` (the host asserts it); `1 ≤ V < 2^16`;
  `HV·V < 2^31`; `N ≥ 1`; fewer than `2^31` programs;
- `T + 64 < 2^31` and, with the gate, `T·HV < 2^31` (`int32` arguments and products);
- `T ≤ 2^22` tokens, `N ≤ 8192` sequences, `B ≤ 8192` rows (real shapes, Exclusions);
- fixed length: `N = B`; variable length: `B = 1`, `cu_seqlens` `int32` or `int64`, from `0`,
  nondecreasing, ending at `T`;
- the buffers hold their tensors (`g` when given) and are valid in the initial memory; `h`, `v_new`,
  `ht` share no byte with each other or with the inputs; on a variable-length launch `cu_seqlens`
  holds its entries and `chunk_offsets[n] = cbase n` (what `prepare_chunk_offsets` builds);
- `Vals` (the inputs): `k`, `w`, `u`, `h0` finite and at most `2^16`, `g` finite and at most
  `2^24`; in every chunk `g_last ≤ 8` and `g_last − g_i ≤ 8`; the padded state rows' bound
  `padB t ≤ 2^100`;
- `StOk t` (the stored state): every element of `h[t]` a finite `bfloat16` and `‖h[t]‖ ≤ 2^40`.

Any float values are covered for termination, the presence of every output and the frame.

## Launches covered

`SETUP.md`: fixed and variable length, with and without the gate, `int32` and `int64` `cu_seqlens`,
both state layouts (`STATE_V_FIRST`), the four combinations of initial and final state, fla's test
shapes, Qwen3-Next's heads, fla's GatedDeltaNet and DeltaNet layers: 56 instances `setup_*` with
`setup_*_ok`, and `setup_min` (one token, `K = V = 1`), which `claim_satisfiable` uses with a memory
of zeros.

**Device facts** the tolerances rest on (all in `Device P`): `Device.DotBF16` at `K = 64`
(`relDot16 64 = 66·2^-22`, `absDot16 64 = 66·2^-126`, relative to `|acc| + Σ liftMag a·liftMag b`),
`Device.Exp2F32` (`2^-22·2^x + 2^-126`), and the totality facts for `tl.dot` and `exp2`.

## Tolerances

With `vAbs t i = |vrefS t i| + 2^-5·(|u[i, j]| + ‖w[i]‖·‖h[t]‖)` (norms of `liftMag` values) and
`Gᵢ = 2^(g_last − g_i)·vAbs t i`:
- `tolV = 2^-8·vAbs + 2^-100`: the `w` dots' error, the fusable `-` and the `bfloat16` rounding of
  the store;
- `errStep = 2^-8·‖kᵀ‖₂·‖G‖ + 2^-15·(‖k‖_F·‖G‖ + 2^g_last·‖h[t]‖) + 2^-36`: the `bfloat16`
  rounding of each gated value carried through `kᵀ`, the `k` dot's error, the decay's and the
  scaling's roundings of the state, and a floor.

Each is a function of the chunk's inputs and of `h[t]` alone, derived from the facts above as a fine
bound and then relaxed. The padded state rows (`K ≤ c < 64·cdiv(K, 64)`) are never stored but feed
the `w` dot; they stay within `padB t`, a bound from the inputs alone, while every earlier chunk's
stored state is in range. On real inputs `errStep` is about 0.4–0.7% of `‖h‖` (median).

## Exclusions

- `T > 2^22`, `N > 8192` or `B > 8192`: the claim covers real shapes only.
- `T + 64 ≥ 2^31`, `T·HV ≥ 2^31` with the gate, `HV ≥ 2^16`, `V ≥ 2^16`, `HV·V ≥ 2^31`, `2^31`
  programs or more: `int32` limits and CUDA's grid, outside real use.
- Inputs outside `Vals` and stored states outside `StOk`: only termination, presence and the frame
  are claimed.
- The drift over many chunks: `Post` bounds each step, not the composition of the steps.
- `gk`, `save_new_value=False`, other chunk sizes, other dtypes: not part of the setup.

`check_gpu.py` (one H200) checks `Post` on 77 launches, with the kernel's own `float32` states as
the witnesses `Y`, and refutes corrupted outputs.
