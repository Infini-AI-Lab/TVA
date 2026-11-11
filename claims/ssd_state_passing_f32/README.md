# `ssd_state_passing_f32`

**The kernel.** SGLang's `_state_passing_fwd_kernel`, the Mamba-2 SSD state passing across the
physical chunks of a packed prefill batch, `float32` states (`source/ssd_state_passing.py`, kernel
lines 79–231; host `state_passing_fwd` lines 234–333, metadata
`query_start_loc_to_chunk_indices_offsets` lines 336–368). A grid `(cdiv(dim, 1024), 1, nheads)`;
program `(pid_m, ·, pid_h)` owns lanes `pid_m·1024 + [0, 1024)` of head `pid_h`, stores the start
state to `out[0, 0]`, then for each chunk `c` loads the chunk's state and decay, restarts the state
where a request starts (with initial states: reloads the request's initial state and, when it
starts inside the chunk, subtracts the decay before its start; without: scales by `+0`), computes
`exp(dA)·state + states[0, c]` (`tl.mul_add`) and stores it to `out[0, c + 1]` or, after the last
chunk, to `final_states`. Two models: `ssd_state_passing_kernel` (`HAS_INITSTATES = True`) and
`ssd_state_passing_kernel_noinit` (`False`). The claim is `Claims.SsdStatePassing.claim` in
`Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
head `h < nheads` and state element `m < dim`:
- `out[0, 0, h, m]` is `initial_states[0, h, m]` up to NaN payload, on every input; without initial
  states it is the `float32` `+0`;
- for `1 ≤ c < nchunks`, `out[0, c, h, m]` holds a `float32` value; when the inputs' values are
  covered (`Vals`), it is finite and within `tol c` of the exact recurrence `sRef c`;
- `final_states[0, h, m]` the same at `c = nchunks`;
- every byte outside `out`'s `nchunks·nheads·dim` and `final_states`' `nheads·dim` elements is
  unchanged.

`sRef` is the Mamba-2 recurrence in the reals, from the host's inputs alone: request `r` starts from
`initial_states[r]` (or `0`); `sRef 0 = startV 0`; `sRef (c+1) = exp(dRef c)·xRef c +
states[0, c]`, where `xRef c` is the start state of the request at chunk `c`'s last token when chunk
`c` starts it (`isReset`) and `sRef c` otherwise, and `dRef c` is `dA_cumsum[0, h, c, chunk − 1]`,
less `dA_cumsum[0, h, c, off − 1]` when that request starts at offset `off > 0` in the chunk. The
request starts come from `query_start_loc`, not from `seq_idx` or `chunk_offsets`.

## Conditions

(`Inst.ok`, `Pre`.)
- `HAS_SEQ_IDX = IS_CONT_BATCHED = True`, `HAS_INITSTATES` either value, `batch = 1`,
  `BLOCK_SIZE = 1024`, contiguous tensors with the strides the host passes;
- `1 ≤ nheads < 2^16`; `dim ≥ 1`, `dim + 1023 < 2^31`; fewer than `2^31` programs; `chunk_size` a
  power of two;
- a packed batch: `1 ≤ num_requests ≤ 8192`, `qsl 0 = 0`, every request at least one token, the
  last ending at `seqlen ≤ 2^22`; `nchunks = cdiv(seqlen, chunk_size)`; `chunk_meta_num` the host's
  `N` (`0` without initial states), `< 2^31`;
- the `int32` arguments and products do not wrap: `nchunks·chunk_size`, `nheads·nchunks·chunk_size`,
  `nchunks·nheads·dim` below `2^31`, and with initial states `(num_requests − 1)·nheads·dim < 2^31`;
- the buffers hold their tensors and are valid in the initial memory; `out` and `final_states`
  share no byte with the inputs or each other;
- the metadata are the host's (`MetaOk`): `seq_idx[0, t]` is the request of token `t`, and
  `chunk_offsets` is what the host's loop builds (`hostOff`).

Any `float32` values are covered; `Post`'s numeric clauses hold under `Vals`: every `states` and
(with initial states) `initial_states` entry finite, every `dA_cumsum` entry finite and `≤ 0`, each
chunk's `dRef c ≤ 64` and update `eHat c·(|xRef c| + tolIn c) + |states[0, c]| ≤ 2^126`.

**The launches covered** (`SETUP.md`): the source's nine test cases (lines 432–441), each with and
without initial states (`setup_1 … setup_18`, `setup_*_ok`); `mk_ok` covers the host's launch for
any request lengths, heads, `dim` and `chunk_size` within the conditions.

**Device facts** the tolerance rests on (in `Device P`): `Device.ExpF32` (`tl.exp` on `float32`:
relative `2^-22 + 2^-23·X` on `[−X, X]`, absolute `2^-140`) and the totality of `exp`. The
subtraction and `tl.mul_add` are IEEE operations.

## Tolerances

Per chunk and lane: `tol 0 = 0`;
`tol (c+1) = eHat c·tolIn c + errE c·|xRef c| + ρ·(eHat c·(|xRef c| + tolIn c) + |states[0, c]|) + 3·2^-150`,
`tolIn c = 0` on a chunk that starts its request, `ρ = 2^-23 + 2^-47`.
- The decay exponent: line 215's IEEE `-` (when it runs) errs by at most
  `dErr = 2^-24·|dRef| + 2^-150`.
- `tl.exp` at `X = |dRef| + dErr`: `|ẽ − e^dRef| ≤ (1 + rel)·e^(dRef + dErr) − e^dRef + 2^-140 =
  errE`, `rel = 2^-22 + 2^-23·X`; `eHat = e^dRef + errE` bounds `|ẽ|`.
- `tl.mul_add`: two IEEE roundings or one `fma`, both within `ρ·(|p| + |n|) + 3·2^-150`.
- Without initial states, where the request changes the scale is the exact `+0`, so the update is
  `states[0, c]` exactly.

On `check_gpu.py`'s launches the kernel's error reaches about half of `tol`.

## Exclusions

- With initial states, `(num_requests − 1)·nheads·dim ≥ 2^31`: line 181's `int32` product wraps (a
  kernel limit; needs more than 2048 requests at `nheads·dim = 2^20`).
- `nchunks·nheads·dim ≥ 2^31`, `nheads·nchunks·chunk_size ≥ 2^31`, `nchunks·chunk_size ≥ 2^31`,
  `dim + 1023 ≥ 2^31`, `N ≥ 2^31`, `nheads ≥ 2^16`: `int64` arguments, `int32` wraps or CUDA's grid
  limit; outside real use.
- Values outside `Vals`: non-finite inputs, a positive `dA_cumsum` entry, a decay exponent above
  `64` or an update above `2^126` (the kernel's `float32` range). Frame and termination hold on
  every input.
- Contexts longer than `2^22` tokens and batches of more than `8192` requests: real shapes only.
- Other flag combinations and `batch > 1`: not part of `SETUP.md`.

`check_gpu.py` (one H200) checks `Post` on the nine test cases and edge cases, in both modes.
