# `ssd_state_passing_f32`: the prepared claim

Kernel: `ssd_state_passing_f32` (SGLang's `_state_passing_fwd_kernel`, the Mamba-2 SSD state
passing across the physical chunks of a packed prefill batch, `float32` states). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.SsdStatePassing`; the statement to prove is
`Statement : ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.SsdStatePassing.main : Statement` (`Statement`:
`Claim.lean` line 673). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 395), and every initial memory `init` that `Pre` accepts (line 509: the seven
buffers valid, the metadata the host's, `MetaOk` line 502), every outcome of the launch is a normal
end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 642).

**What `Post` asks** (line 642), for every head `h < nheads` and element `m < dim`:
1. `out[0, 0, h, m]` (element `stIdx d 0 h m`, line 357) holds a value `v`, on every input: with
   `hasInit`, `init` reads `initst`'s element `inIdx d 0 h m` as some `a` and `FloatV.NaNEq a v`
   (a copy up to NaN payload); without, `v = FloatV.ofPat 0` (the `float32` `+0`);
2. for every `1 ≤ c < nchunks`, `out[0, c, h, m]` holds some `v` with `OutO init d h m c v`
   (line 632): when `Vals init d` holds (line 618), `v` is finite and
   `|v.toReal - sRef init d h m c| ≤ tol init d h m c`. Without `Vals`, only "holds a value" is
   asked. `sRef` (line 558) is the exact recurrence in the reals from the host's inputs:
   `sRef (c+1) = exp(dRef c) · xRef c + newV c`, with `dRef`, `xRef`, `startV`, `newV` (lines 545,
   566, 550, 539) and the request structure `eReq`, `isReset`, `offR`, `hasSub` (lines 521–532)
   computed from `qsl`, not from the metadata. `tol` (line 601) is a recursion: `tol 0 = 0`,
   `tol (c+1) = eHat c · tc + errE c · |xRef c| + rho · (eHat c · (|xRef c| + tc) + |newV c|) +
   3·2^-150`, `tc = 0` on a reset chunk (`tolIn`, line 610), with `dErr`, `expRel`, `errE`,
   `eHat`, `rho` (lines 577–595);
3. `final_states[0, h, m]` (element `finIdx d h m`, line 366) holds some `v` with
   `OutO init d h m nchunks v`;
4. the frame: every byte outside `out`'s first `nchunks·nheads·dim` elements and `finalst`'s first
   `nheads·dim` elements is unchanged.

**What you are given** (`Assumptions`, line 660):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, when the launch is
  valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `exp_acc` (`Rules.Num.ExpF32`): for every `X ≥ 0`, `tl.exp` on `float32` (source line 223) on
  `[−X, X]` is within relative `2^-22 + 2^-23·X`, absolute `2^-140` (a `MathAcc`);
- `exp_total` (`Rules.Num.MathTotal P .exp .binary32`): that `tl.exp` has a result.

**The kernel, as the model runs it** (`model` line 458: `ssd_state_passing_kernel`, line 46, when
`hasInit`, else `ssd_state_passing_kernel_noinit`, line 206; `launch` line 484): a 3-D grid
`(gx d, 1, nheads)` (`gx`, line 334), block shape `[1024]`. Program `(pid_m, ·, pid_h)` owns lanes
`pid_m·1024 + [0, 1024)` below `dim` of head `pid_h`. It stores the start state (the loaded
`initst` row `0`, or zeros) to `out[0, 0]`, then runs `tl.forRange 0 nchunks 1` with seven (or
five) carried values. Each trip loads `states[0, c]` and the scalar `dA_cs`, loads `seq_idx` at the
chunk's last token, and either (with `hasInit`, request changed) reloads the initial state of that
request, reads `chunk_offsets[logical_chunk_idx]` (masked) and, when `c_off > 0`, subtracts a masked
boundary `dA` entry (`tl.sub`); or (without) sets `scale_mask` from the request comparison. Then
`tl.exp`, `tl.where`, `tl.mul_add`, and one masked store to `out[0, c+1]` (`c < nchunks − 1`) or to
`final_states`. Programs write disjoint lanes and use no atomics: the route is
`TVA.meets_independent` (PROVER_GUIDE §4); the loop rule is `safe_forRange`, with trip `c`'s write
footprint the lanes of `out[0, c+1]` or of `final_states`, disjoint across trips.

**What the proof must establish.**
- each launch is valid and has `debug = false`: the grid is 3-D, so `Launch.Valid` is proved
  directly (`x < 2^31` from `dim_lt`, `z < 2^16` from `heads_lt`, the count from `grid_lt`, the
  shape `[1024]`);
- each program is `Safe`: no `int32` offset wraps (`dim_lt`, `tok_lt`, `dA_lt`, `st_lt`,
  `init_lt`, `nmeta_lt`; `seqlen_ctx` keeps every token index and the requests' starts at most
  `2^22`); every enabled load is in bounds (`states_len`, `dA_len`, `init_len`,
  `seq_len`, `offs_len`, using `MetaOk` for the `seq_idx` and `chunk_offsets` values, and
  `qsl_*`, `nchunks_eq`, `nmeta_eq` for their range); the stores are in bounds (`out_len`,
  `fin_len`) on distinct lanes;
- the kernel's metadata path decides the reference's structure: `seq_idx` changes at the end of
  chunk `c` exactly when `isReset d c`, the loaded `seq_idx_chunk_end` is `eReq d c`, and
  the masked `chunk_offsets[logical_chunk_idx]` is `offR d c` when `hasSub d c`, and not positive
  on a reset chunk without it. This needs `reqOf`, `hostN`, `hostOff` (lines 376–392);
- footprints: different programs' lanes are disjoint, and the outputs are disjoint from every
  read (`disj`);
- the values: part 1 is a copy (`safe_load`, `safe_store`); parts 2–3 follow by induction on `c`,
  with the step bound of README `## Tolerances`: `FloatV.sub_err` (source line 215), `exp_acc` at
  `X = |dRef| + dErr` (`MathAcc.mono` if needed), `FloatV.mul_err`, `add_err`, `fma_err`
  (`safe_mul_add`), room from `Vals`' last conjunct; without `hasInit` a reset chunk gives `n`
  exactly;
- the frame: the bytes no program writes.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `bytes_disj` (line 704), `disj_far` (line 713), `zero_i32` (line 846) and
`valid_zero` (line 853) can be reused.

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance, from every memory `Pre`
allows, every launch ends normally and leaves `Post`. For each head `h < nheads` and state element
`m < dim`:

* `out[0, 0, h, m]` is `initial_states[0, h, m]` (bit for bit, up to NaN payload), for every input;
  without initial states (`hasInit = false`, the host given `initial_states = None`) it is the
  `float32` `+0`;
* for `1 ≤ c < nchunks`, `out[0, c, h, m]` holds a `float32` value; when the inputs' values are
  covered (`Vals`), it is finite and `|out[0, c, h, m] − sRef c| ≤ tol c`;
* `final_states[0, h, m]` the same at `c = nchunks`;
* every byte outside `out`'s `nchunks·nheads·dim` elements and `final_states`' `nheads·dim`
  elements is unchanged.

**The reference** `sRef` is the Mamba-2 recurrence computed from the host's inputs alone: the
request boundaries `query_start_loc` (`Inst.qsl`), the chunk states `states`, the initial states and
`dA_cumsum`, read as exact values from the initial memory. Each request starts from its start
state `startV r`: `initial_states[r, h, m]` with initial states, `0` without. With `e_c` the
request of chunk `c`'s last token `min((c+1)·chunk_size, seqlen) − 1` (`eReq`, through `reqOf`, which counts the requests that
start at or before a token), chunk `c` *starts* that request when its first token
`qsl e_c ≥ c·chunk_size` (`isReset`), at offset `off_c = qsl e_c − c·chunk_size` (`offR`):

* `sRef 0 = startV 0`;
* `sRef (c+1) = exp(dRef c) · xRef c + states[0, c, h, m]`, where `xRef c` is `startV e_c` when
  chunk `c` starts `e_c` (so without initial states a request starting in chunk `c` has
  `sRef (c+1) = states[0, c, h, m]`), and `sRef c` otherwise;
* `dRef c = dA_cumsum[0, h, c, chunk_size − 1] − dA_cumsum[0, h, c, off_c − 1]` when chunk `c` starts
  `e_c` at `off_c > 0` (only that request's tokens decay its state), and
  `dA_cumsum[0, h, c, chunk_size − 1]` (the chunk's cumulative total) otherwise.

`exp` is the real exponential and the arithmetic is exact. Nothing in the reference is what the
kernel reads or how it indexes: the request starts come from `query_start_loc`, not from
`seq_idx` or `chunk_offsets`, and the kernel's `logical_chunk_idx` appears nowhere. A kernel that
reads the wrong `chunk_offsets` entry, or misses a reset (with initial states, or without: it
would carry the previous request's state into the next), fails `Post`.

**The tolerance** `tol` is a recursion over the chunks, from the same inputs (`Claim.lean`, "The
tolerance"; derivation under Tolerances): `tol 0 = 0`, and

`tol (c+1) = eHat c · tolIn c + errE c · |xRef c| + ρ · (eHat c · (|xRef c| + tolIn c) + |states[0,c]|) + 3·2^-150`

with `tolIn c = 0` when chunk `c` starts its request (the state restarts from an exact initial
state, or from `0`) and `tol c` otherwise; `errE c` bounds the computed `exp` against
`exp(dRef c)`; `eHat c = exp(dRef c) + errE c`; `ρ = 2^-23 + 2^-47`. It is decay-aware: the carried error is multiplied by the
chunk's decay, so a request's tolerance is a few `float32` roundings of the decayed magnitudes it has
accumulated since its start, not a sum over all chunks.

## Conditions

`Inst`: `hasInit` (`HAS_INITSTATES`), `nheads`, `dim`, `nchunks`, `seqlen`, `chunk_size`,
`num_requests`, `query_start_loc` (`qsl`), `N = len(chunk_offsets)` (`chunk_meta_num`, `0` without
initial states), and the seven tensors by their storage (`Buf`; without initial states
`initial_states` and `chunk_offsets` are the host's `None`s, empty buffers). The flags are
`HAS_SEQ_IDX = IS_CONT_BATCHED = True` and `HAS_INITSTATES` either value, `batch = 1`,
`BLOCK_SIZE = 1024` (the host's line 290), all tensors contiguous with the strides the host passes
(`sSb`, …; note the launch passes `dA_cumsum.stride(0), stride(2), stride(1), stride(3)`). `model`
is `ssd_state_passing_kernel` with initial states and `ssd_state_passing_kernel_noinit` without:
the source's `HAS_INITSTATES = False` branch starts from `tl.zeros` (line 150), never reads
`initial_states` or `chunk_offsets`, takes the chunk's whole decay `dA_cumsum[…, chunk_size − 1]`,
and on a chunk where the request at the end changes sets `scale_mask` to `false` (line 220), so
`scale = tl.where(…, tl.exp(dA_cs), 0.0)` is `+0` and the state restarts as `0·s + states[c]`.
`launch` is the grid `(cdiv(dim, 1024), 1, nheads)`, the block `[1024]`, `enable_fp_fusion` at its
default.

`Inst.ok`:
* `nheads ≥ 1`, `nheads < 2^16` (the grid's `z`); `dim ≥ 1`, `dim + 1023 < 2^31`; fewer than `2^31`
  programs; `chunk_size` a power of two (asserted by the code base);
* the packed batch: `num_requests ≥ 1`, `qsl 0 = 0`, every request at least one token
  (`qsl r < qsl (r+1)`), `qsl num_requests = seqlen`; `nchunks = cdiv(seqlen, chunk_size)`; `N` the
  host's `hostN` with initial states, `0` without (and `< 2^31`);
* `seqlen`, the packed batch's tokens, is a context length: **`seqlen ≤ 2^22`** (`seqlen_ctx`);
* `num_requests`, the batch's requests, is a batch size: **`num_requests ≤ 8192`** (`nreq_le`);
* the `int32` arguments and arithmetic: `nchunks·chunk_size < 2^31`, `nheads·nchunks·chunk_size <
  2^31` (`dA_cumsum.stride(0)`), `nchunks·nheads·dim < 2^31` (`states.stride(0)`, `out.stride(0)`),
  with initial states `(num_requests − 1)·nheads·dim < 2^31` (line 181's product, a kernel limit:
  Exclusions); so every integer argument is below `2^31` and Triton types it `int32` (one equal to
  `1` is the compile-time `1`, built in `int32` at each use), the model's class; the constexpr ints
  the kernel uses (`BLOCK_SIZE = 1024`, the literals `0`, `1`, `-1`) are `int32` too;
* the buffers hold their tensors; `out` and `final_states` share no byte with the inputs or each
  other (the host allocates both).

`Pre`: the seven buffers are valid (aligned, present), and the metadata are the host's (`MetaOk`):
`seq_idx[0, t] = reqOf t` (the source's `repeat_interleave(arange(nreq), seqlens)`), and
`chunk_offsets[k] = hostOff k`, where `hostOff` transcribes the host's loop
(`query_start_loc_to_chunk_indices_offsets`, lines 350–366: `p` counts the requests not starting on
a chunk boundary, `chunk_offsets[s // chunk_size + p] = s % chunk_size`; empty without initial
states). No condition on the `float32` values in `Pre`.

`Vals` (the hypothesis of `Post`'s numeric clauses): every `states` entry and (with initial states)
every `initial_states` entry finite; every `dA_cumsum` entry finite and `≤ 0` (real use: cumulative
sums of `A·dt` with `A < 0`, `dt ≥ 0`, non-positive in any summation order); and the kernel's `float32` range: every chunk's
`dRef c ≤ 64` and `eHat c · (|xRef c| + tolIn c) + |states[0, c]| ≤ 2^126` (Exclusions). Within a
chunk `dA_cumsum` need not be monotone (a parallel scan's rounding may break monotonicity), so a
chunk's decay exponent may be slightly positive; the recursion covers it.

The launches are within part (A) of `Compiler.Faithful`'s scope: the kernel is the `@triton.jit`
function of the source (without its autotuner, `SETUP.md`); the launch gives `num_warps = 4`,
`num_stages = 3` and no other option, no `@triton.heuristics`, no descriptors, no `CompiledKernel`,
no host `TensorDescriptor`. Part (B), the environment, is
assumed and not checked.

## Assumptions

* `[rule] faithful`: `Rules.Compiler.Faithful P (model d) (launch d) (K d)` on every covered `d`;
* `[rule] exp_acc`: `Rules.Num.ExpF32 P` (`tl.exp` on `float32`, line 223);
* `[rule] exp_total`: `Rules.Num.MathTotal P .exp .binary32`.

`consistent` instantiates them at `Platform.sm90ref`.

## Coverage

`SETUP.md` asks for the host's launches with `HAS_SEQ_IDX = IS_CONT_BATCHED = True`, with and
without initial states, `float32` states, `batch = 1`, any number of requests of any lengths ≥ 1,
and the code base's sizes. Each of the source's nine test cases (lines 432–441), which the source
runs in both modes, is an instance built by `mk` from the mode, its request lengths, heads, `dim`
and `chunk_size` (the host's shapes, `N = hostN` or `0`, the tensors at distinct aligned addresses):
`setup_1 … setup_9` with initial states, `setup_10 … setup_18` without, with `setup_i_ok :
setup_i.ok` proved by `decide`. `satisfiable` uses `setup_1` in a memory of zeros. The SETUP sizes beyond the tests (`dim` up to 10240, `nheads` up to 128, chunk
sizes 8 to 256, tens of thousands of tokens) are inside `Inst.ok` (for example 128 heads × 10240
with 128-token chunks up to 210,000 tokens), in both modes. Every setup has at most 8192 tokens
(`seqlen ≤ 2^22`); none is dropped.

## Tolerances

Each chunk's update on one lane, from the library's facts (`FloatV.sub_err`, `Rules.Num.ExpF32`,
`FloatV.mul_err`, `add_err`, `fma_err`; `docs/rules/num.md`):

1. The decay exponent: `dA_cs` is loaded exactly; when the kernel subtracts the boundary entry
   (line 215, on a chunk with `hasSub`), the IEEE `-` of two finite non-positive entries does not
   overflow and errs by at most `dErr = 2^-24·|dRef| + 2^-150` (`sub_err`); otherwise `dErr = 0`.
2. `tl.exp`: the computed input lies in `[−X, X]` for `X = |dRef| + dErr`; `ExpF32` at that `X`
   gives the result within `rel·e^{d̃} + 2^-140`, `rel = 2^-22 + 2^-23·X` (`expRel`); the room
   condition holds since `d̃ ≤ 64 + dErr`. Then `|ẽ − e^{dRef}| ≤ (1 + rel)·e^{dRef + dErr} −
   e^{dRef} + 2^-140 = errE`: for `d̃ ≥ dRef` directly; for `d̃ < dRef` and `rel ≤ 1`, the error is
   at most `e^{dRef} − (1 − rel)·e^{dRef − δ} + 2^-140`, below `errE` because
   `2 ≤ (1+rel)e^δ + (1−rel)e^{−δ}`; for `d̃ < dRef` and `rel > 1` (only when `|dRef|` exceeds
   about `2^23`, which `Vals` allows), the error is at most `rel·e^{d̃} + 2^-140 + e^{dRef} − e^{d̃}
   = e^{dRef} + (rel − 1)·e^{d̃} + 2^-140 ≤ rel·e^{dRef} + 2^-140`, below `errE` since
   `(1 + rel)·e^{dRef + dErr} ≥ (1 + rel)·e^{dRef}`. In every case `|ẽ| ≤ eHat`. `ExpF32`'s family is instantiated per chunk
   at that chunk's `X`, so very negative exponents (strong decay) are covered: there `rel` grows but
   `rel·e^{dRef}` vanishes.
3. `tl.mul_add(ẽ, x̃, n)`: either `fl(fl(ẽx̃) + n)`, within `(2^-23 + 2^-48)·|ẽx̃| + 2^-24·|n| +
   (2 + 2^-24)·2^-150`, or `fma`, within `2^-24·|ẽx̃ + n| + 2^-150`; both within `ρ·(|ẽx̃| + |n|) +
   3·2^-150`, `ρ = 2^-23 + 2^-47` (room: the second `Vals` bound keeps every intermediate below
   `2^126·(1 + 2^-23)`).
4. Composition: `s̃ − sRef(c+1) = ẽ·(x̃ − xRef) + (ẽ − e^{dRef})·xRef + rounding`, so
   `|s̃ − sRef(c+1)| ≤ eHat·tolIn + errE·|xRef| + ρ·(eHat·(|xRef| + tolIn) + |n|) + 3·2^-150 =
   tol(c+1)`, by induction from `tol 0 = 0` (the copied initial state, or the stored `+0`); a
   chunk that starts its request loads the initial state exactly, so its carried error is `0`.
5. Without initial states, a chunk that starts its request: where the request at the chunk's end
   changes, `tl.where` gives the exact `+0`, and `tl.mul_add(+0, s̃, n)` is `n` exactly (`s̃` is
   finite by the induction: `0·s̃ = ±0`, and `±0 + n = n`, or the exact `fma`); on chunk `0` with
   no change, `s̃` is the `+0` block and `ẽ` is finite (`dRef 0 ≤ 64`), so again the result is `n`.
   Either way the error is `0 ≤ tol(c+1) = ρ·|n| + 3·2^-150` (`xRef = 0`, `tolIn = 0`). For
   `c ≥ 1` a chunk starts its request (`isReset`) exactly when the request at its end differs from
   the one at the end of chunk `c − 1` (the kernel's `scale_mask` is `false`). The other chunks are
   as with initial states (the whole chunk's decay, `dErr = 0`).

**Meaningful at the covered sizes.** On every launch of `check_gpu.py` (the nine test cases, up to
4096 chunks, 128 heads × 8192, strong and no decay, in both modes) the kernel's error reaches 0.499
of `tol`: the
tolerance is within a factor of about 2 of the kernel's worst actual error, because it is the
recursion of the kernel's own roundings. For one chunk it is about `2^-23` relative to the state;
with no decay over 4096 chunks it grows to at most about `4096·2^-22` of the accumulated magnitude
(`0.1%`), still far below the outputs.

## Exclusions

What `SETUP.md` allows and the claim leaves out:

* **With initial states, `(num_requests − 1)·nheads·dim ≥ 2^31`** (`init_lt`): a limit of the
  kernel. Line 181 computes
  `seq_idx_chunk_end * stride_initstates_batch` in `int32`, which wraps; the kernel then reads the
  wrong initial state. First launch that hits it: `nheads = 128`, `dim = 8192`
  (`nheads·dim = 2^20`) and 2049 requests (an `initial_states` tensor of 8 GiB). Real use reaches it
  only with a prefill batch of more than 2048 requests at that head size (possible in principle with
  many short requests, not in the code base's tests or typical batches).
* **`nchunks·nheads·dim ≥ 2^31`** (`st_lt`) and **`nheads·nchunks·chunk_size ≥ 2^31`** (`dA_lt`):
  `states.stride(0)` (resp. `dA_cumsum.stride(0)`) would be an `int64` argument (the pointer `+=`
  itself is 64-bit and does not wrap), a different
  specialisation than the model's `int32` one; for `dA_cumsum`, `pid_h * stride_dA_cs_head` (an
  `int32` product) also wraps from `(nheads − 1)·nchunks·chunk_size + chunk_size − 1 ≥ 2^31`, a
  kernel limit. These need an 8 GiB `states` tensor (210,000 tokens at 128 heads × 10240 with
  128-token chunks) or 16.7 million tokens: outside real use (prefill budgets of thousands to tens
  of thousands of tokens). Likewise `dim + 1023 ≥ 2^31`, `nchunks·chunk_size ≥ 2^31`, `N ≥ 2^31`,
  `nheads ≥ 2^16` (CUDA's grid `z`).
* **Values** (`Vals`, a hypothesis of `Post`'s numeric clauses; frame and termination hold on every
  input): non-finite inputs; a positive `dA_cumsum` entry (outside real use: `A < 0`, `dt ≥ 0`); and,
  a limit of the kernel's `float32` arithmetic, a chunk decay exponent above `64` or an update above
  `2^126`, where `exp` or the state would overflow. Real use (states of order 1 to 10^3, decays
  `≤ 0`) is far inside.
* **Out of scope: dimensions that are not multiples of 128 or exceed `2^17`, contexts longer than
  `2^22` tokens, and batches of more than `8192` requests: the claim covers real model shapes
  only.** The context length: `seqlen` (`seqlen_ctx`; the request starts `qsl` follow). The batch
  size: `num_requests` (`nreq_le`). No field is a model dimension: `dim` is a per-head state
  (`headdim × dstate`, flattened), `nheads` a head count, `chunk_size` a block size; they keep
  their bounds above. Every launch of `SETUP.md` meets them (at most `41` requests).
* The other flag combinations (no `seq_idx`, not continuous batching) and `batch > 1`: not part
  of the setup (`SETUP.md`).

## Checks

* **Build**: `Claim.lean` builds with no error, no
  `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (at `Platform.sm90ref`) and `satisfiable` (`setup_1` in a zero
  memory) are proved.
* **Coverage**: `setup_1_ok … setup_18_ok` (the source's nine test cases, with and without initial
  states) are proved.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 50 launches
  hold: the nine test cases and 16 edges (chunk sizes 1, 2 and 1024; `dim` 1, 1025, 2049; one head;
  200 one-token requests; chunk-aligned starts; seven starts in one chunk; one chunk with three
  requests; 4096 chunks; 32,768 tokens at 128 heads × 8192; `dA × 200`; no decay over 4096 chunks;
  values near `10^30`; a non-monotone `dA_cumsum` with positive decays), each with and without
  initial states. Worst error / tolerance: `out` 0.499, `final_states` 0.498 (without initial
  states, the chunks where a request starts are exact). `out[0]` is bit-exact (the `+0` without
  initial states), `Vals` holds, the byte frame holds (the inputs unchanged, and guard bands of 4096
  elements on each side of `out` and `final_states`, in which the check places the host's two
  allocations, unchanged), and the host's metadata equal `reqOf`/`hostOff`/`hostN` on every launch.
  Three negative tests are rejected: an output moved by twice its tolerance on a chunk where a
  request starts (without initial states, error / tolerance 2), the outputs of a launch with
  nonzero initial states checked as one without (`1.6·10^11`), and a launch with all-zero
  `chunk_offsets` (`1.6·10^6`).

## Library requests

None. (The edges `(num_requests − 1)·nheads·dim` near `2^31` and `nchunks·nheads·dim` near `2^31`
need tensors of 8 GiB and more and are not run on the GPU.)
