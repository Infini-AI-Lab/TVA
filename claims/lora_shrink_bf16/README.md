# Claim: `lora_shrink_bf16`

The kernel is vLLM's `_lora_shrink_kernel` (`source/lora_shrink.py`, lines
308–423) with its helpers `do_shrink_kernel` (185–305) and `mm_k` (83–182), inlined at their calls; the
host is `lora_shrink` (503–610) with `lora_kernel_meta` (462–500) and `_get_lora_a_ptr` (426–459). The
claim is `Claims.LoraShrink.Statement : ∀ K P, Assumptions K P → Spec K` in [`Claim.lean`](Claim.lean),
which also proves `consistent`, `satisfiable` and the coverage facts `setup_*_ok`, and does not prove
`Statement`. The model map is [`model-map.json`](model-map.json); the GPU check is
[`check_gpu.py`](check_gpu.py).

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance, from every memory satisfying
`Pre`, every launch that runs ends normally and leaves `Post`:

* **a token `t` with an adapter `l = adapter t ≥ 0`**, element `(s, t, j)` of the output (slice
  `s < S`, `j < R`): when the `K` values of row `t` of `inputs` and of row `j` of adapter `l`'s LoRA A
  matrix of slice `s` are finite, `dotMag ≤ 2^120`, and there is room in `float32`
  (`vMag + atomTol ≤ maxVal`), the element is finite and within `tol` of `scaling·S`, with `S` the
  exact dot product `Σ_k x[t,k]·A_{s,l}[j,k]` and `scaling` the host's double (`Inst.scaling : F64`),
  whatever order the atomic additions ran in. The reference is the host's own `scaling · S`. A row
  holding a NaN or an infinity is not bounded (the element is some `float32`); `dotMag > 2^120`
  (products that can overflow the `float32` accumulator) is not bounded either.
* **a token with no adapter** (`-1`): the element is exactly `+0.0` in every slice, the value the
  host's `torch.zeros` gave it.
* **frame**: every byte outside the output's `[S, M, R]` tensor is unchanged (so every input buffer).

The reference is computed from the host's own inputs: `adapter` is the host's `token_lora_mapping`,
and `S` reads `inputs` and slice `s`'s weight tensor at the indices of the token and its adapter, not
at the rows the kernel gathers through the metadata nor at the address the kernel loads from the
table. `dotMag = Σ_k liftMag x·liftMag w` (a `bfloat16` subnormal counted at `2^-126`); `tol`,
`accTol`, `mulTol`, `vMag` and `atomTol` are in Tolerances.

## Conditions

`Inst` carries the sizes `M`, `K` (hidden), `R` (rank), `maxL` (`max_loras`), `S` (slices), `n`
(distinct adapter values, `-1` included), `G` (`num_active_loras`, the grid's axis 2), the host's
mapping `adapter`, the metadata it builds (`groupId`, `groupCnt`, `groupStart`, `tIdx`), `scaling`
(the host's Python `float`, a double; the kernel receives `scaling32`, its `float32` rounding), the
inputs, the `S` weight tensors `lora s`, the `uint64` address table `tab` (used when `S > 1`), the
output and the four metadata tensors. The configuration is the host's: `bk = 256`, `sk = 64` below 128
tokens, `bk = 32`, `sk = 8` from 128 (`BLOCK_K`, `SPLIT_K`). `Inst.ok`:

* the shapes: `R % 16 = 0` (a limit of the kernel, Exclusions), `0 < R`, `0 < K`, `K ≤ 2^17`,
  `R·K < 2^31`, `0 < S`;
* the kernel's `int32` offsets (limits of the kernel, Exclusions): `(M − 1)·R < 2^31` (line 290);
  for every token's adapter `l ≥ 0`, `l·R·K < 2^31` (line 249); with `S > 1`, `(S − 1)·M·R < 2^31`
  (line 286);
* the grid: axis 0 `sk·cdiv(M, 32)·cdiv(R, 16) < 2^31`, `S < 2^16`, `n ≤ G ≤ maxL + 1`, `G < 2^16`
  (CUDA's limits), fewer than `2^31` programs;
* `scaling` finite, and its `float32` rounding finite;
* the metadata is the host's (`lora_kernel_meta`): adapters in `[-1, maxL)`; `groupId` the distinct
  values in increasing order (`-1` first when present), each with a token; `groupCnt` the counts;
  `groupStart` their cumulative sums from `0`; `tIdx` a permutation whose range for group `g` is
  exactly group `g`'s tokens;
* the buffers hold their tensors (each slice's weights `[maxL, R, K]`, the table `S` entries when
  `S > 1`, the output `[S, M, R]`), and the output shares no byte with the inputs, any slice's
  weights, the table (when `S > 1`) or the metadata.

`Pre`: the buffers are valid (the table too when `S > 1`, holding slice `s`'s weight address in entry
`s`, as `_get_lora_a_ptr` writes it); `lora_ids` holds the distinct ids then `-1`,
`num_tokens_per_lora` the counts then `0`, `lora_token_start_loc` the cumulative sums then `0` (the
host's `torch.zeros`), `token_indices_sorted_by_lora_ids` holds `tIdx`; the output's `[S, M, R]`
tensor is `+0.0`. Any `bfloat16` values in `inputs` and the weights.

`launch`: the grid `(sk·cdiv(M, 32)·cdiv(R, 16), S, G)` (lines 564–568), the source's block shapes,
`enable_fp_fusion` at its default. The model takes `BLOCK_M = 32`, `BLOCK_N = 16`, `BLOCK_K = bk`,
`SPLIT_K = sk`, `GROUP_SIZE_M = 8`, `SLICE_NUM = S` and `EVEN_K = (K % (bk·sk) = 0)`, as the host
passes. With one slice the kernel's `lora_ptr` is the weight tensor; with several it is the `uint64`
table, whose entry `slice_id` the kernel loads and converts to a `*bf16` (`tl.intToPtr`, lines
231–233); the model takes the two forms as the parameters `lora_ptr` and `lora_tab`. Each program
adds its scaled accumulator into the output with a relaxed `tl.atomic_add` (line 305, `SPLIT_K > 1`),
whose result the source never uses (Modelling decisions).
Its pointer arithmetic follows the source's grouping: each `int32` offset of lines 241–252 and
285–292 is added to the pointer in turn (Python's left-to-right `+`), not summed first.

The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope: one direct launch of
the `@triton.jit` kernel (no `@triton.autotune`, no `@triton.heuristics`); the options `num_warps = 4`,
`num_stages = 2`, `num_ctas = 1`, `launch_pdl = False` only, `debug` unset; no descriptors (no
allocator); no `CompiledKernel`; no host `TensorDescriptor`. Part (B), the
environment (an H200, Triton 3.8.0 as installed, CPython 3.13 without `-O`, the default
floating-point environment, the kernel new), is relied on.

## Assumptions

* `faithful` — `[rule]` `Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`.
* `dot` — `[rule]` `Rules.Num.DotBF16 P`: the model's `tl.dot` of `bfloat16` operands into `float32`
  (`maxImpreciseAcc = 0`, any `input_precision`), a whole `tl.dot` relative to lifted magnitudes;
  used per step through `DotLift.dotAdd_step`, which covers both forms of
  `accumulator += tl.dot(a, b)` (`tl.dot_add`).
* `dotChunk` — `[rule]` `Rules.Num.DotBF16Chunk P`: the same `tl.dot`, chunk by chunk (chunks of 64
  consecutive `k`, all `K` when `K < 64`; `Rules.Num.DotChunk16Loss` with `2^-13`, `2^-20`,
  `2^-143`, `2`, `2^-22`); its first arm, composed per `tl.dot_add` step under both forms
  (`tripErr`, Tolerances).
* `dot_total` — `[rule]` `Rules.Num.DotTotal P dotCfg`: the same `tl.dot` has a result.
* `atomic` — `[rule]` `Rules.Num.AtomicAddF32 P`: the `float32` atomic addition of line 305, the
  `tl.atomic_add` inside the model's `tl.atomic_add_unused` (`AtomicAddAcc`, `2^-23`, `2^-124`, every
  `sem` and `scope`).
* `atomic_total` — `[rule]` `Rules.Num.AtomicAddTotal P binary32`: the same atomic has a result.

`consistent` holds on `Platform.sm90ref`.

## Modelling decisions

- **Line 305 is `tl.atomic_add_unused c_ptr accumulator (some c_mask) .relaxed`.** The call is an
  expression statement: no operation reads the value it returns. The library gives such an atomic
  the order `MemSem.unused .relaxed .gpu`, which is `acq_rel` at `gpu` scope (SEMANTICS.md,
  "Atomics", an atomic whose result is unused). The `SPLIT_K` programs of a tile therefore add into
  the same output elements without a race, in any order.
- **The output is reached by atomics only during the launch.** The host zeroes the output before
  the launch (`torch.zeros`; `Pre`: every element of the `[S, M, R]` tensor is `+0.0`), and the
  kernel never loads it; with `SPLIT_K > 1` its only access to the output is line 305 (line 303's
  plain store is the `SPLIT_K = 1` branch, which no program of such a launch takes). `Inst.ok` keeps
  the output apart from every buffer the kernel loads (`out_inp`, `out_lora`, `out_tab`, `out_tid`,
  `out_cnt`, `out_start`, `out_id`), so no plain load or store touches its bytes.

## Coverage

`SETUP.md` asks for the host's launches: one to several slices, vLLM's default configuration (which
depends on the batch), either `EVEN_K`, ranks 16 to 256 (multiples of 16) and LoRA A weights of fewer
than `2^31` elements (`max_loras·rank·hidden < 2^31`). Each instance below has `SETUP.md`'s shapes and
one fixed mapping (the host's test draws it at random): `oneInst` maps every token to adapter `a`,
`twoInst` the first `nm` tokens to `-1` and the rest to `a`.

| `SETUP.md` launch `(M, hidden, rank, max_loras, slices)` | Instance | `Inst.ok` |
|---|---|---|
| case 1 `(1, 4096, 16, 4, 1)`; `BLOCK_K = 256`, `SPLIT_K = 64` | `setup_c1` | `setup_c1_ok` |
| case 2 `(5, 2049, 8, 8, 1)` | — | not part of the setup (rank 8); outside `Inst.ok` (Exclusions 1) |
| case 3 `(37, 2049, 1, 1, 1)`, specialize | — | not part of the setup (rank 1); outside `Inst.ok` (Exclusions 1) |
| case 4 `(128, 4096, 64, 32, 1)`, 10 % `-1`; `BLOCK_K = 32`, `SPLIT_K = 8` | `setup_c4` (13 tokens `-1`, the rest on adapter 31) | `setup_c4_ok` |
| case 5 `(512, 2049, 256, 8, 2)`, `EVEN_K = False` | `setup_c5` | `setup_c5_ok` |
| case 6 `(300, 14336, 32, 4, 3)`, specialize | `setup_c6`, `G = n = 1` | `setup_c6_ok` |
| case 7 `(2048, 8192, 16, 128, 1)`, 10 % `-1` | `setup_c7` (adapter 127, `G = 129`) | `setup_c7_ok` |
| case 8 `(4096, 5120, 128, 8, 1)` | `setup_c8` | `setup_c8_ok` |
| case 9 `(64, 4096, 16, 4, 1)`, every token `-1` | — | no launch: the host returns before it (line 526) |
| a decode step on `qkv_proj`: 8 tokens, hidden 4096, rank 16, 4 adapters, 3 slices, `scaling = 1.0` | `setup_decode_qkv` | `setup_decode_qkv_ok` |
| a prefill step on `gate_up_proj`: 2048 tokens, hidden 4096, rank 64, 8 adapters, 2 slices | `setup_prefill_gate_up` | `setup_prefill_gate_up_ok` |
| a layer: `scaling = 1.0`, hidden 14336, rank 256, 32 adapters, 4096 tokens | `setup_layer` | `setup_layer_ok` |
| hidden 2 (the smallest named) | `setup_hidden2` | `setup_hidden2_ok` |
| hidden 128256 (the largest named), rank 256, 2 adapters (`check_gpu.py`'s edge) | — (inside `Inst.ok`: `R_blk`, `K_top`, `2·256·128256 < 2^31`) | — |
| beyond `SETUP.md`: hidden 128256, rank 256, 128 adapters, a token on adapter 65 (`max_loras·rank·hidden ≥ 2^31`) | `setup_hidden_max` | `setup_hidden_max_ok` (`Inst.ok` is wider than `SETUP.md` there, per adapter id) |

Every launch `SETUP.md` asks for is inside `Inst.ok`: ranks 16 to 256, multiples of 16
(`R_blk`, `R_pos`); hidden 2 to 128256 (`K_pos`, `K_top`, multiples of `bk·sk` or not); `max_loras·R·K <
2^31` gives `R·K < 2^31` (`RK_lt`, as `max_loras ≥ 1`) and, for every adapter id `l < max_loras`,
`l·R·K < 2^31` (`lora_wrap`); `max_loras` 1 to 128 gives `G ≤ 129 < 2^16`; `-1` tokens; both values of
`num_active_loras` (`max_loras + 1` and the distinct count); one to three slices. The bounds left that
`SETUP.md` does not state are `(M − 1)·R < 2^31` and `(S − 1)·M·R < 2^31` (Exclusions 3, 4), with the
grid bounds they imply. `satisfiable` uses `testInst` (one token, hidden 512, rank 16, one slice) with
a memory holding the host's metadata.

## Tolerances

From the library's facts, for one element `(s, t, j)` with finite inputs, `bk`, `sk` the host's
`BLOCK_K`, `SPLIT_K`:

- **One program's part.** The tile's program of part `p < sk` runs `T_p = stepsOf p` `tl.dot` steps
  of `K`-extent `bk`, one per block `b ≡ p (mod sk)` of `bk` consecutive `k` below `K` (a block from
  `K` on is skipped, lines 155–157; the masked lanes of the last load are `+0`, of sum and lifted
  magnitude `0`). Its accumulator starts at `+0.0` and each step is `tl.dot_add`, so two bounds hold
  on its distance to its exact part `partSum p`:
  - **chunk by chunk** (`DotBF16Chunk`): each `tl.dot` is `cpb` chunks of `cw = min(64, bk)`
    consecutive `k` (four of 64 at `bk = 256`, one of 32 at `bk = 32`), taken in increasing order
    (the part's chunks `q < nChunks p`, `chunkK p q`; `pSum p g` the exact partial sum of the first
    `g`). Each chunk's step is within `2^-13·Mj + 2^-20·|s j + c_j| + 2^-143` (the first arm,
    `DotChunk16Loss.step_left`), `Mj` any bound on `|s j|` and on the chunk's lifted products; from a
    partial value within `e` of the exact partial sum `P`, the next is within
    `cStep e P m P' = (1 + 2^-13 + 2^-20)·e + 2^-13·max(|P|, m) + 2^-20·|P'| + 2^-143` of the next,
    `P'` (`m = cLift`, the chunk's largest lifted product; the recursion of `Dot8Chunk.errB`), and
    `cChain` takes `n` such steps. Line 152's `accumulator += tl.dot(...)` is `tl.dot_add`, which the
    library gives two results, so step `t` (from the error `e` after the step before, `P = pSum p
    (t·cpb)`, `D` the step's exact sum) is charged the larger of:
    - the fused form `tl.dot(a, b, acc)`: the step's chunks continue the accumulator's chain,
      `cChain` from `e` against the part's exact partial sums;
    - the unfused form, `tl.dot(a, b)` from `+0` then the `float32` addition (`AccAdd`, fusable): the
      dot's own chain from `0` against its own partial sums `pSum p g − P`, within `L` of `D`, then
      `FloatV.AddRel.err` of the addition, `3u·(|d| + |acc|) + 4η` with `|d| ≤ |D| + L` and
      `|acc| ≤ |P| + e`: in all `e + L + 3u·(|D| + L + |P| + e) + 4η`.

    This is `tripErr p t e`; `accErr p` iterates it from the `+0.0` of `tl.zeros` (line 128), and
    `errChunk p = accErr p T_p`;
  - **step by step** (`DotBF16`): each step within `relDot16 bk·(|acc| + L) + absDot16 bk`
    (`DotLift.dotAdd_step`, `L` the step's lifted magnitude), composed by `DotLoop.accum_step` into
    `errStep p = DotLoop.bound ρ α T_p (partLift p)`, `ρ = stepRel relDot16 bk`,
    `α = stepAbs absDot16 bk`;
  - `errPart p = min (errChunk p) (errStep p)`. The room each `tl.dot` needs (`DotLiftAcc` on
    `|acc| + L`, `DotChunk16Loss` on `|acc| + Σ|a·b|`) holds with `dotMag ≤ 2^120`. The chunk bound
    is relative to the running partial sums, so it is the smaller one on signed data; the step bound
    is relative to `|acc| + L`, about `(bk + 2)·2^-22` of it per step, and is the smaller one when all
    products have one sign (`P q` then grows to the part's whole sum, charged `2^-13` per chunk).
- **The parts together.** The parts' blocks are disjoint and cover `0 … K−1`, so
  `Σ_p partSum p = S`; the accumulators' errors add up to at most `accTol = Σ_p errPart p`, their
  magnitudes to at most `partMag = Σ_p (|partSum p| + errPart p)`.
- **The scale.** Each accumulator is multiplied by `s32 = scaling32` with one `float32` rounding
  (`FloatV.mul_err`, `|acc_p·s32| ≤ |s32|·partMag ≤ vMag ≤ maxVal`): the `sk` roundings add up to at
  most `mulTol = 2^-24·|s32|·partMag + sk·2^-150`, and the scaled parts' magnitudes to at most
  `vMag = |s32|·partMag + mulTol`.
- **The atomic additions.** The element receives exactly `sk` `atomic_add`s (one per part; the tile's
  wrapped rows are masked, line 293), into the `+0.0` of `Pre`, in any order: `AtomicAddF32` per step
  (`SharedAcc.accStep_atomicAdd`) and `SharedAcc.replay_err_shared` put the result within
  `atomTol = DotLoop.bound 2^-23 2^-124 sk vMag` of the sum of the scaled parts, under its room
  hypothesis, which `vMag + atomTol ≤ maxVal` gives.
- **The host's double.** `|v − scaling·S| ≤ |v − s32·S| + |s32 − scaling|·|S|`: `0` for the `0.5` and
  `1.0` the host passes, at most `(2^-24·|scaling| + 2^-150)·|S|` for a double in `float32`'s range.

So `tol = |s32|·accTol + mulTol + atomTol + |s32 − scaling|·|S|`. The index conditions (Conditions)
only decide which `int32` index values occur, each below `2^31`; they do not change the arithmetic of
an element.

Against the output (`check_gpu.py`, `scaling = 1`): on the host's test data (`torch.rand`, all
products positive) `tol / |out|` is `7·10^-5` (below 128 tokens, up to hidden 16384) to `4.7·10^-4`
(hidden 14336 from 128 tokens), from the step bound, and the measured error is 0.6–2 % of `tol`. On
signed data (`torch.randn`), from the chunk bound:

| hidden | below 128 tokens (`bk = 256`, `sk = 64`): `tol / |out|` median, p90 | from 128 tokens (`bk = 32`, `sk = 8`): median, p90 |
|---|---|---|
| 4096 | `1.5·10^-3`, `7.4·10^-3` | `4.1·10^-3`, `2.1·10^-2` |
| 14336 | `2.8·10^-3`, `1.5·10^-2` | `1.5·10^-2`, `7.9·10^-2` |
| 128256 | — | `0.14`, `0.69` |

The measured error is then `4·10^-4`–`1.3·10^-3` of `tol`. At hidden 14336 a dropped last block of
`bk` `k` passes on 1.1 % of the elements below 128 tokens and 17.8 % from 128 tokens (a block of 32
`k` against the bound of 448 chunks of 32), and fails `Post`; a dropped part of the `sk` parts, a
missing scale, another adapter, swapped slices and an all-zero output pass on at most 2.3 % of them.

## Exclusions

Launches `Inst.ok` leaves out. `SETUP.md` asks for none of those in 1 and 2 (both are limits of the
kernel); 3 and 4 are the only ones within `SETUP.md`'s stated ranges, and real use does not reach them.

1. **Ranks that are not a multiple of 16 (1, 4, 8, …): a limit of the kernel under the library's
   semantics.** Line 237, `rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)`,
   promises that each tile's 16 columns are consecutive and that 16 divides the first. When
   `R % 16 ≠ 0` the last tile's columns wrap around `N` (`R = 8`: `0…7, 0…7`), the promise is false,
   and a false hint is undefined behaviour (SEMANTICS.md, "Compiler hints": hints are checks), so these
   launches have no meaning in the library and no claim can cover them. The hint is wrong for these
   ranks. First launch: any rank from 1 to 15, e.g. the host test's case 3 (37 tokens, hidden
   2049, rank 1). These ranks are not part of the setup (`SETUP.md`). On the H200 the
   host's results are within `tol` at ranks 1, 4 and 8 (`check_gpu.py`, informational lines): Triton
   3.8.0 does not currently exploit the false hint here, but nothing guarantees that.
2. **An adapter id `l` with `l·R·K ≥ 2^31`: a limit of the kernel.** Line 249's
   `lora_d0_stride * lora_index` is an `int32` product, which wraps, and the program reads another
   adapter's (or unallocated) memory. First launch: hidden 128256, rank 256, `max_loras ≥ 67`, a token
   on adapter 66 (`66·256·128256 = 2 167 013 376`); the weight tensor is then at least 4.3 GB. It needs
   a weight tensor of at least `2^31` elements, which is not part of the setup (`SETUP.md`:
   `max_loras·rank·hidden < 2^31`, which keeps every `l·R·K` below `2^31`). Not run on the
   GPU (more than the 4 GB a test may use). The condition is per adapter id, so `Inst.ok` also covers
   larger weight tensors when the adapters the tokens use stay below the limit (`setup_hidden_max`,
   beyond `SETUP.md`).
3. **`(M − 1)·R ≥ 2^31`: a limit of the kernel.** Line 290's `ram[:, None] * output_d1_stride` is an
   `int32` product of a token index and `R`. First launch: `M = 2^23 + 1` tokens at rank 256 (an 8 GiB
   output). Real use does not reach it (a step's tokens are far fewer). Not run on the GPU.
4. **`S > 1` with `(S − 1)·M·R ≥ 2^31`: a limit of the kernel.** Line 286's
   `slice_id * output_d0_stride` is an `int32` product of the slice and `M·R`. First launch: three
   slices, `M = 2^22` tokens at rank 256 (a 12 GiB output). Real use does not reach it. Not run on the
   GPU.
5. **`R·K ≥ 2^31`**: outside `SETUP.md` (`max_loras·R·K < 2^31`); there
   `lora_d0_stride` is an `int64` argument, and line 250's `int32` `rbn * lora_d1_stride` would wrap too.
6. **`K > 2^17`**: outside `SETUP.md`'s ranges (hidden at most 128256).
7. **A grid axis or the number of programs beyond CUDA's limits** (`Launch.Valid`): they need
   `max_loras ≥ 65535`, `2^16` slices or more than `2^31` programs, outside real use.
8. **A non-finite `scaling`**: outside real use (`1.0` in vLLM's layers, `0.5` in its test).
9. **An output that shares bytes with a buffer the kernel loads**: outside real use (the host
   allocates it fresh with `torch.zeros`). There a plain load would touch bytes the relaxed atomic
   additions of line 305 write, which the library gives no meaning (a race), and with `SPLIT_K = 1`
   the plain stores of line 303 would overwrite an input.

Not exclusions: other configurations (`VLLM_TUNED_CONFIG_FOLDER`'s tuned files), `USE_GDC` and
`STORE_PARTIALS` (the batch-invariant path) are not part of the setup (`SETUP.md`); `M = 0` and an
all-`-1` mapping launch nothing.

## Library requests

None.

## Checks

1. **It builds**: `lake env lean ../claims/lora_shrink_bf16/Claim.lean` from `lean/`, no errors, no
   warnings, no `sorry`, no `axiom`; `consistent`, `satisfiable` and every `setup_*_ok` depend only on
   `propext`, `Classical.choice`, `Quot.sound`.
2. **Not vacuous**: `consistent` (on `Platform.sm90ref`) and `satisfiable` (`testInst`, a memory with
   the host's metadata) are proved.
3. **Coverage**: the eleven `setup_*_ok` above are proved.
4. **GPU** (`check_gpu.py`, H200, Triton 3.8.0): 29 covered launches (the seven covered `SETUP.md`
   cases with their slices; decode and prefill layer launches with `scaling = 1.0`, merged (two or
   three slices) and single, on `torch.rand` and signed `torch.randn` data; 127 and 128 tokens, the
   configuration's boundary; `scaling` `0.1` and `1/3` (not `float32` values), `2^-7` and `−3.75`; and
   edges: hidden 2, 255, 2049, 16384, 128256, `2^17` and `2^17 − 1`; 128 adapters with `G = 129`; one
   token at rank 256; `specialize_active_lora`), all pass: the largest error / tolerance is
   about `2·10^-2` (it varies with the order of the atomic additions); every `-1` element is the bit pattern `+0.0` in every slice; the byte
   frame holds (the host's output is allocated inside a buffer with 4096 sentinel elements before and
   after it, which keep their bytes, and every input buffer is unchanged); the room conditions hold.
   Negative tests on signed data at hidden 14336 and 4096, both configurations (all zero, the last
   block of `bk` `k` dropped, one of the `sk` parts dropped, the scale missing, another adapter,
   swapped slices): `Post` rejects each. The ranks 1, 4, 8, which are not part of the setup and
   `Inst.ok` excludes (informational): error / tolerance `≤ 6.4·10^-3`.
