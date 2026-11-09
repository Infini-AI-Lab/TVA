# `chunk_delta_h_bf16`: the prepared claim

Kernel: `chunk_delta_h_bf16` (fla's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, the state
recurrence across chunks of the chunked gated delta rule's forward, `bfloat16` inputs, `float32`
state). Claim: [`Claim.lean`](Claim.lean), namespace `Claims.ChunkDeltaH`; the statement to prove
is `Statement : ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json).
GPU check: [`check_gpu.py`](check_gpu.py). Setup: [`SETUP.md`](SETUP.md).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.ChunkDeltaH.main : Statement` (`Statement`:
`Claim.lean` line 1021). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 669), and every initial memory `init` that `Pre` accepts (line 754), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 977).

**What `Post` asks** (line 977). For every sequence `n < N`, value head `hv < HV` and column
`j < V`:
1. `v_new`: for every token `s < lenOf d n` (line 638), element
   `uIdx d (rowOf d n) (startOf d n + s) hv j` of `vn` reads in `fin` as a `bfloat16` `x`; only
   under `Vals init d` (line 939) and `StOk fin d n hv j t'` for every `t' ≤ s / 64` (line 956;
   the stored states of the token's chunk and of every earlier chunk) is more asked: `x` is
   finite and `|x.toReal − vrefS …| ≤ tolV …` (`vrefS` line 831, exact in the reals from `init` and
   the stored `h[t]` in `fin`; `tolV` line 892, `2^-8·vAbs + 2^-100` with `vAbs` line 886);
2. `h`: for every chunk `t < nch d n` (line 641) and row `c < K`, element
   `hIdx d (cbase d n + t) hv c j` (lines 645, 663) of `h` holds a `bfloat16` value;
3. the states: there is `Y : Nat → Nat → F32` with
   - under `Vals`, `Y 0 c` finite and equal to `h0v init d n hv j c` (line 824; `0` without `h0`);
   - for `t < nch d n` and `c < K`, a finite `Y t c` gives `h[t][c] = FloatV.cvt (Y t c)` exactly;
   - for `t < nch d n`, under `Vals` and `StOk … t'` for every `t' ≤ t`, `Y t`, `Y (t + 1)`
     finite and `colNorm K (Y (t + 1) − Step … (Y t)) ≤ errStep …` (`Step` line 839, exact;
     `colNorm` line 846; `errStep` line 911);
   - with `storeHt`, `ht[sIdx d n hv c j]` (line 665) reads as `Y (nch d n) c` exactly;
4. the frame: every byte outside the first `B·NTt·HV·K·V` elements of `h`, the first
   `B·T·HV·V` of `vn` and (with `storeHt`) the first `N·HV·K·V` of `ht` is unchanged.

Outside `Vals` and these `StOk` guards only the presence clauses, the rounding link and
`ht = Y (nch n)` are asked.

**What you are given** (`Assumptions`, line 1003), five fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotBF16`, a `DotLiftAcc`, `relDot16 64`, `absDot16 64`): every `tl.dot` of
  `chunk_step` (line 61) and `chunk_step_vf` (line 275), the `w` dots and the `k` dots;
- `dot_total` (`Rules.Num.DotTotal` at `cfgBF16`, line 41): they have a result;
- `exp2_acc` (`Rules.Num.Exp2F32`, `2^-22` relative, `2^-126` absolute): the gate's two `tl.exp2`
  (model lines 141, 144, and their `vf` twins); `exp2_total` (`Rules.Num.MathTotal`): a result.

**The kernel, as the model runs it** (`fwd_kernel_h` line 528, `model` line 735, `launch` line
740): a 1-D grid of `gridN d` (line 653) programs. Program `pid` splits it in `int64` into the
column tile `i_v` (32 columns), the sequence `i_n` and the head `i_h` (`tl.irem`, `tl.idiv`). On
variable length it reads `cu_seqlens[i_n]`, `cu_seqlens[i_n + 1]` and `chunk_offsets[i_n]`
(`tl.load1`); on fixed length `bos = i_n·T`, `boh = i_n·cdiv(T, 64)`. `body_rest` (line 185, or
`body_rest_vf` line 409) zeroes up to four `[64, 32]` state blocks, adds `h0` if `useH0`, runs the
chunk loop (`loopVar` line 491 over `int64`, `loopFix` line 505 over `int32`, each a
`tl.forRange 0 NT 1`) whose trip `chunk_step` stores the rounded state blocks to `h`, computes the
`w` dots, `u − w·h` (`tl.subc`), stores `v_new`, applies the gate (with `useG`), and adds the `k`
dots into the state; after the loop it stores the state to `ht` if `storeHt`. Programs write
disjoint tiles and use no atomics: the route is `TVA.meets_independent`; the chunk loop is
`safe_forRange` with an invariant and per-trip write footprints (chunk `t`'s `h` tile and `v_new`
rows), disjoint across trips (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (the `Launch` default; `Launch.Valid` from
  `grid_lt`; the block shapes are fixed sizes);
- each program is `Safe`: the divisors `cdiv(V, 32)`, `HV`, `HV / H`, `64` are nonzero (`V_pos`,
  `HV_pos`, `H_dvd`); the table reads are in bounds (`cu_len`, `co_len`) and give `cbase`
  (`Pre`); the `int32` values (`T + 63`, `last_idx·HV`, `o_k*`) and the `int64` offsets do not
  wrap (`T_lt`, `THV_lt`, `HVV_lt`, `K_le`; with `T_ctx` and `N_batch`, `T ≤ 2^22` and
  `N, B ≤ 8192` keep every token and sequence index below `2^35`); every enabled lane of every load and store is in
  bounds (`k_len` … `co_len`, `var_*`); `safe_dot` (with `dot_total`), `safe_math` (with
  `exp2_total`), `safe_subc`, `safe_sub`, `safe_mul`, `safe_add`, `safe_cvt`, `safe_convert`,
  `safe_load`, `safe_store`;
- the programs' write tiles are disjoint (distinct `(i_n, i_h, i_v)`; on variable length the
  sequences' chunk ranges `[cbase n, cbase n + nch n)` are disjoint), and no program writes what any
  program reads (`disj_h`, `disj_vn`, `disj_ht`);
- the loop invariant: the state blocks after `t` trips are `Y t`, the values the trip stored to
  `h[t]` are their rounding, and, under `Vals` and while `StOk t'` holds for every `t' < t`, the
  padded rows `K ≤ c < 64·cdiv(K, 64)` stay within `padB t` (line 929; `StOk t'` keeps chunk
  `t'`'s gated values finite, so the `k` dot's fact bounds what it adds to the padded rows); then
  the value parts under `Vals` and `StOk`: the fine bounds `fineV` and `fineStep` of README
  `## Tolerances` (steps 0–7) from `dot` chained over the `K` blocks (`DotLoop.bound`),
  `exp2_acc`, `FloatV.mul_err`, `FloatV.sub_err`, `FloatV.AddRel.err` and `FloatV.cvt_err`; then
  relax them to `tolV` and `errStep` by the inequality chain of step 8 (`fineV ≤ tolV`,
  `fineStep ≤ errStep` under `Vals` and `StOk`);
- the bytes no program writes give part 4.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `ok_of` and the
`setup_*_ok` theorems; `bytes_disj` (line 1097), `disj_far` (line 1106) and `valid_zero`
(line 1378) can be reused.

## Specification

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance, from every memory `Pre`
allows, every launch ends normally and leaves `Post`. `Post` is **step-wise** in the kernel's own
`float32` state. For every sequence `n < N`, value head `hv < HV` and value column `j < V`, with
`h[t]` the column `h[cbase n + t, hv, :, j]` (`t < nch n = cdiv(len n, 64)`):

* for each token `s < len n` (chunk `t = s / 64`, token `i = s % 64`), `v_new[token s, hv, j]` holds
  a `bfloat16` value; when `Vals` and `StOk t'` for every `t' ≤ t` (`h[t']` finite with
  `‖h[t']‖ ≤ 2^40`), finite and `|v_new − vrefS t i| ≤ tolV t i`;
* every element of every `h[t]` holds a `bfloat16` value;
* **there are `float32` states** `Y 0, Y 1, …, Y (nch n)` (the column's `K` rows; the kernel's
  `b_h` entering each chunk, and after the last) such that:
  - when `Vals`, `Y 0` is the initial state `h0[n, hv, :, j]` (`0` without one), finite;
  - **each stored state is the rounding of its state**: for `t < nch n` and each row, a finite
    `Y t c` gives `h[t][c] = bf16(Y t c)` exactly (`FloatV.cvt`, nearest even);
  - **each state is one step from the previous one**: for `t < nch n`, when `Vals` and `StOk t'`
    for every `t' ≤ t`, `Y t` and `Y (t + 1)` are finite and
    `‖Y (t + 1) − Step t (Y t)‖ ≤ errStep t` (Euclidean norm over the `K` rows);
  - with `STORE_FINAL_STATE`, `ht[n, hv, :, j]` is `Y (nch n)` exactly;
* every byte outside `h`'s, `v_new`'s and (with `STORE_FINAL_STATE`) `ht`'s elements is
  unchanged.

**The reference** is one exact step of the gated delta rule's chunked state recurrence, from the
host's inputs (`k`, `w`, `u`, `g`, `h0`, and the sequences' boundaries `cu_seqlens`, exact values of
the initial memory), the state `Y t` and the stored state `h[t]` (exact values of the final memory:
an output, so observable). With chunk `t` of sequence `n` holding the tokens `64t ≤ i < 64t + cl t`
(`cl t = min(64, len n − 64t)`) and `g_last` the gate at its last token,

* `vrefS t i = u[i, j] − Σ_c w[i, c] · h[t][c]`;
* `Step t y c = 2^g_last · y c + Σ_i k[i, c] · 2^(g_last − g_i) · vrefS t i`,

`k` taken at the key head `hv // (HV / H)`, `2^x` the real power, the arithmetic exact; without the
gate (`USE_G` false, `g = None`) `g_last = g_i = 0`, so both powers are `1` (the delta rule). The
new values are from `h[t]` because the kernel's `w` dots read exactly that rounding of the state
(the same conversion line 185 stores); the state's decay is of `Y t`, the value the kernel scales
(line 252). The outputs' places are the documented layouts: `h[c₀, hv, c, j]` with
`c₀ = cbase n + t` (`n·cdiv(T, 64) + t` for fixed length; the chunks of the sequences before `n`,
plus `t`, for variable length), `v_new` like `u`, `ht[n, hv, c, j]`, and with `STATE_V_FIRST` the
`[V, K]` layouts `h[c₀, hv, j, c]`, `h0`/`ht[n, hv, j, c]` (`hIdx`, `sIdx`). Nothing in the
reference is what the kernel reads or how it indexes: `cbase` is a sum of the sequences' chunk
counts from `cu_seqlens`, not the `chunk_offsets` the kernel loads, and no program id appears.

**What a reader may conclude**: `h[0]` is `bf16(h0)` exactly; the stored states are the exact
`bfloat16` roundings of one `float32` state sequence that starts at the initial state, ends at `ht`,
and follows the recurrence step by step, each step within `errStep` (about 0.4–0.7% of `‖h‖` on
real inputs, median; the kernel's own step error is about a fifth of it); each new value is
`u − w·h[t]` within `tolV`. The state carried between chunks is pinned to the stored `h[t]`: it
lies in the box of `float32` values that round to `h[t]` (half a `bfloat16` unit), so a deviation
in one chunk cannot hide in the store's rounding, and `ht` is the state itself. The
composition of the steps is not bounded: deviations up to `errStep` in every chunk are accepted and
add up to a drift from the exact recurrence started at `h0` (Tolerances, "What `Post` admits").

**The tolerances** are functions of the chunk's inputs and of `h[t]` alone (Claim.lean, "The
tolerances"; derivation under Tolerances); nothing is carried from chunk to chunk. With

`vAbs t i = |vrefS t i| + 2^-5·(|u[i, j]| + ‖w[i]‖·‖h[t]‖)`

(the new value's magnitude, with `2^-5` of its operands' magnitudes; the norms over the `K` rows,
of `liftMag` values, a `bfloat16` subnormal counted at `2^-126`) and `Gᵢ = 2^(g_last − g_i)·vAbs t i`
(the gated magnitudes; `Gᵢ = vAbs t i` without the gate),

`tolV t i = 2^-8·vAbs t i + 2^-100`,

`errStep t = 2^-8·‖kᵀ‖₂·‖G‖ + 2^-15·(‖k‖_F·‖G‖ + 2^g_last·‖h[t]‖) + 2^-36`

(`‖kᵀ‖₂` the spectral norm of the chunk's keys, `‖k‖_F` their Frobenius norm of `liftMag` values):
one `bfloat16` rounding of the new value; one `bfloat16` rounding of each gated value carried
through `kᵀ`, the `k` dot's error, and the decay's and the scaling's roundings of the state; and a
floor far below real states.

## Conditions

`Inst`: the flags `varlen`, `useG` (`USE_G`), `useH0`, `storeHt`, `vf` (`STATE_V_FIRST`); `B`, `T`,
`H`, `HV`, `K`, `V`, `N`; `cu_seqlens`' dtype and entries; the ten storages (`Buf`). Fixed by
`SETUP.md`: `SAVE_NEW_VALUE = True`, `USE_GK = False`, `BT = 64`, `BV = 32`, all tensors contiguous. `STATE_V_FIRST` selects the model's
`chunk_step_vf`/`body_rest_vf` (the `[BV, 64]` state blocks, the `tl.trans` of lines 214, 221, …,
the `b_h += tl.trans(tl.dot(b_k, b_v))` of lines 293, 300, …) or `chunk_step`/`body_rest`;
`USE_G` takes lines 247–258 (the gate's loads, `exp2`, the gating of `b_v` and the decay of the
state) or skips them. `launch` is the grid
`(cdiv(V, 32)·N·HV,)`, the source's block shapes, `enable_fp_fusion` at its default.

`Inst.ok`:
* heads: `H ≥ 1` divides `HV`, `1 ≤ HV < 2^16`; `1 ≤ K ≤ 256` (the host asserts `K ≤ 256`);
  `1 ≤ V < 2^16`; `HV·V < 2^31` (the compile-time `HV*V` of lines 238, 242 is an `int32` constant);
  `N ≥ 1`; fewer than `2^31` programs;
* `T + 64 < 2^31` (`T` is an `int32` argument; line 97's `T + 63`) and, with the gate,
  `T·HV < 2^31` (line 247's `int32` `last_idx·HV` on a fixed-length launch; without the gate
  line 247 does not run);
* real model shapes (Exclusions): `T ≤ 2^22` (`T_ctx`: a row's tokens, or a packed batch's, is a
  context length); `N ≤ 8192` and `B ≤ 8192` (`N_batch`: the sequences and the batch rows are batch
  sizes). `K` and `V` are per-head state widths (head dims, real values 32 to 512), not model
  dimensions: they keep the bounds above;
* fixed length: `N = B`; variable length: `B = 1`, `cu_seqlens` `int32` or `int64`, from `0`,
  nondecreasing, ending at `T`;
* the storages hold their tensors (`g` when given); `h`, `v_new`, `ht` share no byte with each other or the inputs.

`Pre`: the buffers the launch uses are valid (aligned, present; `g` when given); on a variable-length launch
`cu_seqlens` holds its entries and `chunk_offsets[n] = cbase n` for `n ≤ N` (what
`prepare_chunk_offsets` builds). No condition on float values.

`Vals` (a hypothesis of `Post`'s numeric clauses; termination, the presence of every output and
the frame hold on every input): every element of `k`, `w`, `u` (and `h0` when given) finite and at
most `2^16` in magnitude, of `g` (when given) finite and at most `2^24`; in every chunk `g_last ≤ 8` and
`g_last − g_i ≤ 8` for each token; and `padB t ≤ 2^100` for every chunk (the padded state rows,
below; a bound from the inputs alone). Real inputs are inside: `k` is L2-normalized (`|k| ≤ 1`),
`w = A·(β·k·2^g)` and `u = A·(β·v)` (`2^g = 1` without the gate) of order `1` to `10^2`, the gate is a chunk-local cumulative
sum of `log2` decays (`g ≤ 0`, nonincreasing within a chunk, so `g_last − g_i ≤ 0` up to the
cumulative sum's rounding), and `padB` stays below `2^-79` for up to `2^20` chunks without decay.

`StOk t` (on the observed stored state; chunk `t`'s clauses assume `StOk t'` for every `t' ≤ t`):
every element of `h[t]` is a finite `bfloat16` value and `‖h[t]‖ ≤ 2^40`. The kernel's `Y t` is
then finite (the rounding of a NaN or an infinity is not finite) and `h[t] = bf16(Y t)`. With
`Vals` it keeps every intermediate of the step far from the `float32` and `bfloat16` overflow
(`|Y t| ≤ 2^41`, `|v| ≤ 2^61`, the gated values below `2^70`, the dots' lifted magnitudes below
`2^93`), so the library's rounding facts apply. The earlier chunks are needed because, when
`64 ∤ K`, the padded state rows carry hidden state from chunk to chunk: they stay within `padB`
only while every earlier chunk's gated values are finite (the `k` dot's fact says nothing about an
output whose operand column is not finite), and chunk `t`'s `w` dot reads them. `h[0]` meets
`StOk` whenever `Vals` holds (`|h0| ≤ 2^16`); real states are of order `1` to `10^3`.

The launches are within part (A) of `Compiler.Faithful`'s scope: the kernel is the `@triton.jit`
function of the source without its autotuner and heuristics (`SETUP.md`); the launch gives
`num_warps = 4`, `num_stages = 3` and no other option, no descriptors, no `CompiledKernel`, no host
`TensorDescriptor`. Part (B), the environment (Triton's knobs at their
defaults among it), is assumed and not checked.

## Assumptions

* `[rule] faithful`: `Rules.Compiler.Faithful P (model d) (launch d) (K d)` on every covered `d`;
* `[rule] dot`: `Rules.Num.DotBF16 P` (`tl.dot` of `bfloat16` operands into `float32`, lines 214–237,
  293–316);
* `[rule] dot_total`: `Rules.Num.DotTotal P cfgBF16`;
* `[rule] exp2_acc`: `Rules.Num.Exp2F32 P` (`tl.math.exp2` through the helper `exp2`, lines 250–251);
* `[rule] exp2_total`: `Rules.Num.MathTotal P .exp2 .binary32`.

`consistent` instantiates them at `Platform.sm90ref`.

## Coverage

`SETUP.md` asks for the host's launches with the flags above, with and without the gate, fixed or
variable length, any `int32`/`int64` `cu_seqlens`, the four combinations of initial and final state,
and the code base's sizes. The instances `setup_*` (built by `mkF` and `mkV`: the host's shapes,
contiguous storages at `i·2^40`, the gate given; `noG` of one: without the gate) are proved covered (`setup_*_ok`, through `ok_of`, by `decide`):
* fixed length: fla's `test_chunk` shapes `setup_f1 … setup_f10` (`tests/ops/test_gdn.py:149-169`:
  `(B, T, H, HV, D)` = `(4, 1024, 4, 4, 128)`, `(2, 75, 4, 4, 64)`, `(2, 500, 3, 3, 60)`,
  `(2, 1000, 3, 3, 64)`, `(3, 1024, 4, 4, 100)`, `(2, 1500, 4, 4, 128)`, `(4, 2048, 8, 8, 64)`,
  `(2, 256, 2, 4, 64)`, `(2, 512, 2, 8, 64)`, `(2, 1024, 4, 8, 128)`) and `test_chunk_with_chunk_size`
  at chunk size 64 (`setup_f11`, `(1, 64, 2, 4, 32)`); Qwen3-Next's heads on `2 × 4096` tokens with
  the four flag combinations (`setup_qf_00 … setup_qf_11`); fla's default GatedDeltaNet layer
  (`setup_gdn_f`, `K = 256`, `V = 512`);
* variable length: `test_chunk_varlen` and `test_chunk_varlen_prefill` (`test_gdn.py:752-771,
  848-863`; `setup_v1 … setup_v6`, `int64` `cu_seqlens`), one with `int32` `cu_seqlens`
  (`setup_v7`); Qwen3-Next's heads on a packed batch with empty and one-token sequences, the four
  flag combinations (`setup_qv_00 … setup_qv_11`); the default GatedDeltaNet layer (`setup_gdn_v`);
* `STATE_V_FIRST` (the `[V, K]` layout fla's layers use): `test_chunk_state_v_first`'s shapes
  (`setup_vf1 … setup_vf4`, `test_gdn.py:414-425`: `(B, T, H, D)` = `(1, 63, 1, 64)`,
  `(2, 500, 3, 60)`, `(3, 1024, 4, 128)`, `(4, 2048, 8, 64)`); the GatedDeltaNet layer
  (`gated_deltanet.py:302-334`) on a packed batch with `int32` `cu_seqlens` and the four flag
  combinations (`setup_vf_gdn_00 … setup_vf_gdn_11`) and fixed-length (`setup_vf_gdn_f`); Qwen3-Next's
  heads on a packed batch (`setup_vf_qv`);
* without the gate (`USE_G` false, `g = None`: the delta rule's forward,
  `fla/ops/delta_rule/chunk.py:39-49`, and its backward's recomputation, `:87-96`): fla's
  delta-rule `test_chunk` shapes `setup_d1 … setup_d6` (`tests/ops/test_delta.py:21-26`:
  `(B, T, H, D)` = `(1, 63, 1, 64)`, `(2, 100, 4, 60)`, `(2, 1000, 3, 128)`, `(2, 1024, 4, 128)`,
  `(3, 2000, 4, 128)`, `(4, 2048, 8, 64)`) and `test_chunk_with_chunk_size` at chunk size 64
  (`setup_d7`, `(1, 64, 2, 32)`); `test_chunk_varlen`'s sequences (`setup_dv1 … setup_dv4`,
  `test_delta.py:167-170`, `int32` `cu_seqlens`); fla's default DeltaNet layer (16 heads,
  `K = V = 128`, `fla/models/delta_net/configuration_delta_net.py:26-35`) fixed (`setup_dn_f`) and
  packed with the four flag combinations (`setup_dn_00 … setup_dn_11`); grouped value heads
  (`setup_d_gv`) and the `[V, K]` layout (`setup_d_vf`).

`satisfiable` uses `setup_min` (one token, one head, `K = V = 1`) in a memory of zeros. The sizes
beyond these instances are inside `Inst.ok`: any number of rows or sequences within the grid limit,
`T` below `2^31 / HV` tokens (`2^26`, about 67 million, at Qwen3-Next's 32 value heads); `h`'s
offsets are `int64` (lines 118–119, 176–179), so its size is not bounded.

## Tolerances

Per program, the kernel's arithmetic on one value column `j` is independent of the other columns,
so the derivation follows one column. Write `Y = Y t` for the kernel's `float32` state entering
chunk `t` (the witness of `Post`'s states: the kernel's `b_h`), `H = h[t]` for its stored rounding,
and `‖·‖` the Euclidean norm over the `K` rows. Every bound below is a function of the chunk's
inputs and of `H`. The padded rows `c ≥ K` of a block
(when `64 ∤ K`) are masked out of every load and store, and their `w` and `k` are `0`
(`liftMag 0 = 0`), so they add nothing to the real rows; but they are not exactly `0`: each chunk
scales them by `ẽ ≈ 2^g_last` and adds a `k` dot whose facts allow an absolute error `aD` on its
true `0` (with `STATE_V_FIRST` the `+`'s rounding too). They start at `0` (the zero blocks and
the masked `h0` load), and while `StOk t'` holds for every `t' < t` (with `Vals` it keeps chunk
`t'`'s gated values `vgb` finite and its operands in range, which the `k` dot's fact needs for the
padded rows' outputs too), induction gives `|pad| ≤ padB t`,
`padB (t+1) = (1 + rD + 3·2^-24)·(eB·(1 + 2^-24)·padB t + 2^-150 + aD) + 4·2^-150`, a bound from
the inputs alone. The `w` dot's fact needs them finite and in range: `Vals` keeps
`padB ≤ 2^100`. Without that bound the statement would be false: `K = 60`, `g ≡ 8` over 33 chunks
(`k_i = e_i`, `w_i = 256·e_i`, `u = 0`) lets the padded rows reach `2^128`. The value columns
`j ≥ V` of a block are independent of the columns `j < V` (the dot facts are per output element)
and are never stored. The library facts used are `FloatV.cvt_err` (`bfloat16`:
`2^-8·|x| + 2^-134`), `FloatV.mul_err`, `FloatV.sub_err` (`float32`: `2^-24·|x| + 2^-150`),
`FloatV.AddRel.err` (the fusable `-` and `+`: `3·2^-24·(|x| + |y|) + 4·2^-150`), `Num.Exp2F32`
(`2^-22·2^x + 2^-126` on every input) and `Num.DotBF16` at `K = 64` (`rD = relDot16 64 = 66·2^-22`,
`aD = absDot16 64 = 66·2^-126`, relative to `|acc| + Σ liftMag a·liftMag b`,
`liftMag q ≤ |q| + 2^-126`).

0. **The states** (line 185): the store writes `tl.cvt` of the `float32` block, `H = bf16(Y)`
   elementwise (exactly `FloatV.cvt` on a finite `Y`: `Post`'s rounding clause); a finite `H`
   (`StOk t`) has a finite `Y`. `|H − Y| ≤ 2^-8·|Y| + 2^-134` (`cvt_err`), so
   `|Y| ≤ (|H| + 2^-134)/(1 − 2^-8)` and `‖Y − H‖ ≤ rh = (2^-8·‖H‖ + √K·2^-134)/(1 − 2^-8)`,
   `‖Y‖ ≤ YB = ‖H‖ + rh`. `Y 0` is the initial state exactly (the masked `float32` load added to
   `+0`); `ht` is the last state stored without conversion (lines 318–349), so `ht = Y (nch n)`; for
   a sequence without tokens that is `Y 0`.
1. **The `w` dots** (lines 211–237): `nb = cdiv(K, 64)` dots, the first from `+0`, each from the
   previous result, on `bf16(Y) = H` (the same conversion line 185 stores; with `STATE_V_FIRST`,
   `tl.trans(b_h).to(bfloat16)`, the same values). Each step's error is `rD·(|acc| + A_b) + aD` with
   `A_b` its block's `Σ liftMag w·liftMag H`; composed as `DotLoop.accum` composes them, the chain's
   error is `dW i = chainErr nb (‖liftMag wᵢ‖·‖liftMag H‖)`, `chainErr nb L = ((1+rD)^nb − 1)·L +
   nb·(1+rD)^nb·aD` (Cauchy–Schwarz for `L`). So `|bv − wᵢ·H| ≤ dW i`, `|bv| ≤ bvB i`.
2. **The new value** (lines 239, 243): `vn = fl(u − bv)` by `AddRel.err`, error
   `sB i = 3·2^-24·(|u| + bvB i) + 4·2^-150`. So `|vn − vrefS t i| ≤ dW i + sB i`,
   `|vn| ≤ vnB i = |vrefS t i| + dW i + sB i`, and the stored `bf16(vn)` adds `2^-8·|vn| + 2^-134`:
   `fineV t i = dW i + sB i + 2^-8·vnB i + 2^-134`.
3. **The gate** (lines 247–251): `dg̃ = fl(g_last − g_i)` within `dgE = 2^-24·|dg| + 2^-150`;
   `f = exp2(dg̃)` within `2^-22·2^dg̃ + 2^-126`; then `|f − 2^dg| ≤ errF = (1 + 2^-22)·2^(dg+dgE) −
   2^dg + 2^-126` (for `dg̃ ≥ dg` directly; for `dg̃ < dg` the error is at most
   `2^dg − (1 − 2^-22)·2^(dg−δ) + 2^-126`, below `errF` because
   `2 ≤ (1 + 2^-22)·2^δ + (1 − 2^-22)·2^-δ`). Masked lanes get the exact `0` of `tl.where`. Likewise
   `ẽ = exp2(g_last)` within `errE = 2^-22·2^g_last + 2^-126`, `|ẽ| ≤ eB`. Without the gate
   (`USE_G` false) lines 247–258 do not run: `errF = errE = 0` (`Gam = gam = 1`, `fB = eB = 1`).
4. **The gated value** (lines 250, 288): `vg = fl(vn·f)`, `vgb = bf16(vg)`. Against
   `Gamᵢ·vrefS t i` (`Gamᵢ = 2^dg`): `vgb − Gamᵢ·vrefS = (vgb − vg) + (vg − f·vn) + (f − Gamᵢ)·vn +
   Gamᵢ·(vn − vrefS)`, bounded by `dB i = 2^-8·vgB + 2^-134 + 2^-24·fB·vnB + 2^-150 + errF·vnB +
   Gamᵢ·(dW + sB)`, with `vgB = fB·vnB·(1 + 2^-24) + 2^-150 ≥ |vg|`. Without the gate `vg = vn`:
   `vgB = vnB`, `dB i = 2^-8·vnB + 2^-134 + dW + sB`.
5. **The decay** (line 252): `Y' = fl(Y·ẽ)`, and `Y' − 2^g_last·Y = (Y' − Y·ẽ) + Y·(ẽ − 2^g_last)`:
   `‖Y' − 2^g_last·Y‖ ≤ dec = errE·YB + 2^-24·eB·YB + √K·2^-150`,
   `‖Y'‖ ≤ YpB = eB·(1 + 2^-24)·YB + √K·2^-150`. The reference decays `Y`, as the kernel does, so
   the state's `bfloat16` rounding enters only through the `w` dots, which read `H` (steps 1–2,
   `vrefS` from `h[t]`). Without the gate `Y' = Y`: `dec = 0`, `YpB = YB`.
6. **The update** (lines 293–316): the `k` dot from `Y'` errs by at most
   `rD·(|Y'_c| + Σᵢ liftMag k·liftMag vgb) + aD` per row, in norm `dotH = rD·(YpB +
   ‖liftMag k‖_F·‖vgbB + 2^-126‖) + √K·aD` (Cauchy–Schwarz). With `STATE_V_FIRST` (line 293,
   `b_h += tl.trans(tl.dot(b_k, b_v))`) the dot's accumulator is `+0` and `Y'` enters through the
   fusable `+` instead: the dot errs by `rD·‖liftMag k‖_F·‖vgbB + 2^-126‖ + √K·aD`, and the `+`
   (`AddRel.err`) by `3·2^-24·(‖Y'‖ + ‖dot‖) + 4·√K·2^-150`: `dotHvf`. The `+`'s other operand is
   the product `Y·ẽ` of line 252, and the backend contracts the two into an `fma` (16 `fma.rn.f32`
   per K block in the kernel's PTX); `AddRel` allows that result (the rounding of `p + q` with `p`
   the exact product), so `dotHvf` covers it. The combine pass does not fold the `+` into the dot
   (its operand is the `tt.trans`).
7. **Composition**: `Y (t + 1) − Step t (Y t) = (Y' − 2^g_last·Y) + kᵀ·δ + dotErr` with
   `|δᵢ| ≤ dB i`, so `‖Y (t + 1) − Step t (Y t)‖ ≤ dec + sKn·‖dB‖ + dotH = fineStep t`
   (`sKn = ‖kᵀ‖₂`). `h[t + 1] = bf16(Y (t + 1))` is the next trip's store (step 0) and `ht` the last
   `Y`: neither adds a tolerance.

8. **The relaxation `fineV ≤ tolV`, `fineStep ≤ errStep`.** Write `r = |vrefS t i|`, `m = |u| +
   ‖liftMag w[i]‖·‖liftMag h[t]‖`, so `vAbs = r + 2^-5·m` and `r ≤ m`; `Γ = Gamᵢ`, `γ = 2^g_last`,
   `u = 2^-24`. Under `Vals` and `StOk`: `|u|, |w|, |k| ≤ 2^16`, `K ≤ 256` (`nb ≤ 4`), `‖h[t]‖ ≤ 2^40`,
   `g_last, g_last − g ≤ 8`; so `‖w‖ ≤ 2^20`, `m ≤ 2^60.01`, `sKn ≤ ‖k‖_F ≤ 2^23`, `γ, Γ ≤ 2^8`.
   * **`v_new`.** `(1 + rD)^nb − 1 ≤ 6.30·10^-5` and `nb·(1 + rD)^nb·aD ≤ 2^-117`, so `dW + sB ≤
     6.32·10^-5·m + 2^-116.9` and `fineV = 2^-8·r + (1 + 2^-8)·(dW + sB) + 2^-134 ≤ 2^-8·r +
     6.35·10^-5·m + 2^-116 ≤ 2^-8·(r + 2^-5·m) + 2^-100 = tolV` (`2^-13 = 1.22·10^-4`).
   * **The gated values.** `errF = Γ·((1 + 2^-22)·2^dgE − 1) + 2^-126`: for `dg ≥ −126` the factor
     is at most `(1 + 2^-22)·2^{126·u + 2^-150} − 1 ≤ 5.5·10^-6 ≤ 2^-17`; for `dg < −126`,
     `Γ·(1 + 2^-22)·2^dgE ≤ (1 + 2^-22)·2^{−126·(1 − u)} ≤ 2^-125.9`; so `errF ≤ 2^-17·Γ + 2^-124.9`
     and `fB ≤ (1 + 2^-17)·Γ + 2^-124.9`. Then `dB ≤ c₁·Γ·vnB + Γ·(dW + sB) + 2^-124.8·vnB +
     2^-133.9`, `c₁ = 2^-8·(1 + u)·(1 + 2^-17) + u·(1 + 2^-17) + 2^-17 ≤ 2^-8 + 7.72·10^-6`, with
     `vnB ≤ r + 6.32·10^-5·m + 2^-116.9`; charging `7.72·10^-6·Γ·r` to `Γ·m` (`r ≤ m`),
     `dB ≤ 2^-8·Γ·r + 7.12·10^-5·Γ·m + A ≤ 2^-8·Γ·vAbs + A`, `A ≤ 2^-124.8·m + Γ·2^-116.8 +
     2^-133.8 ≤ 2^-64.7`. Without the gate (`Γ = 1`) `dB ≤ 2^-8·r + 6.35·10^-5·m + 2^-116 ≤
     2^-8·vAbs + 2^-116`. Over at most 64 tokens, `sKn·‖dB‖ ≤ 2^-8·sKn·‖G‖ + 2^23·8·2^-64.7`.
   * **The `k` dot.** `vgbB + 2^-126 ≤ 1.004·Γ·vnB + 2^-124.8·vnB + 2^-125.8 ≤ 1.004·Gᵢ + 2^-64.6`
     (`6.32·10^-5 ≤ 2^-5`), so `‖vgbB + 2^-126‖ ≤ 1.004·‖G‖ + 2^-61.6`; its factor on `kl = ‖k‖_F`
     is `rD` in `dotH` and `rD + 3u·(1 + rD) ≤ 1.5915·10^-5` in `dotHvf`, and
     `1.004·1.5915·10^-5 ≤ 2^-15 = 3.05·10^-5`; the rest is below `2^-54.5`.
   * **The decay and the scaling.** `YB = (1 + 2^-8/(1 − 2^-8))·‖h[t]‖ + √K·2^-134/(1 − 2^-8) ≤
     1.00392·‖h[t]‖ + 2^-129.9`, `eB ≤ (1 + 2^-22)·γ + 2^-126`; `dec + rD·YpB ≤ 1.00393·(2^-22 +
     u·(1 + 2^-22) + rD)·γ·‖h[t]‖ + 2^-85 ≤ 1.61·10^-5·γ·‖h[t]‖ + 2^-85` (with `STATE_V_FIRST`
     `dec + 3u·YpB`, smaller; without the gate `dec = 0`, `YpB = YB`, `γ = 1`), at most
     `2^-15·γ·‖h[t]‖ + 2^-85`; and `√K·aD ≤ 2^-116`.

   Summing, `fineStep ≤ 2^-8·sKn·‖G‖ + 2^-15·(kl·‖G‖ + γ·‖h[t]‖) + 2^-38.6 ≤ errStep` (floor
   `2^-36`). On `check_gpu.py`'s launches the median of `tolV / |v_new|` is `1.03–1.24×` that of
   `fineV` and the median of `errStep / ‖h[t + 1]‖` `1.04–1.19×` that of `fineStep`: the gated
   values' rounding, `fineStep`'s main term, is just above `2^-8` of `‖kᵀ‖₂·‖Γ·r‖`, and `2^-8·‖G‖`
   covers it with the `2^-5·m` share, which is larger where `u` and `w·h[t]` cancel (the delta rule
   without a gate); the `2^-15·kl·‖G‖` term doubles the `k` dot's share.

`StOk t` and `Vals` give each step room: `|Y| ≤ 2^41`, `|vn| ≤ 2^61`, `|vg| ≤ 2^70`, the dots'
lifted magnitudes below `2^93`, so no intermediate is near `2^128` and every fact's room condition
holds; `Y (t + 1)` is then below `2^94`, so it and `h[t + 1]` are finite.

The gated values' `bfloat16` roundings through `kᵀ` (`sKn·‖dB‖`, about `2^-8·‖kᵀ·vg‖`, the size of
one chunk's update) dominate; the dots' terms (`rD = 66·2^-22` relative) and the decay's follow. No
term depends on an earlier chunk, so the tolerance does not grow with the sequence's length.

**Meaningful at every length and decay.** `errStep t / ‖h[t + 1]‖`, over every chunk, value head
and column, with Mamba-style gates (`g = −A·softplus(a + dt_bias)`, `A ∈ [1, 16]`, `dt`
log-uniform per head), `k` L2-normalized, `w`, `u` the WY representation, an initial state of
order `0.5`, 16,384 tokens (256 chunks); the kernel's own step error is `‖Y (t + 1) − Step t (Y t)‖`
with its `float32` states:

| heads | `dt` | median | max | `fineStep`: median | the kernel's step error / `‖h[t + 1]‖` (median) |
|---|---|---|---|---|---|
| Qwen3-Next (`K = V = 128`, 32 value heads) | none | 0.62% | 1.00% | 0.54% | 0.11% |
| | `[10^-5, 10^-4]` | 0.63% | 1.00% | 0.55% | 0.12% |
| | `[10^-4, 10^-3]` (`[V, K]`) | 0.69% | 1.02% | 0.62% | 0.14% |
| | `[10^-3, 10^-1]` | 0.70% | 0.98% | 0.67% | 0.15% |
| GatedDeltaNet (`K = 256`, `V = 512`, 6 heads) | none | 0.42% | 0.62% | 0.36% | 0.08% |
| | `[10^-5, 10^-4]` | 0.44% | 0.63% | 0.38% | 0.09% |
| | `[10^-4, 10^-3]` (`[V, K]`) | 0.55% | 0.76% | 0.50% | 0.13% |
| | `[10^-3, 10^-1]` | 0.62% | 0.78% | 0.59% | 0.14% |

(`dt` drawn log-uniformly per token and head, `A` uniformly per head; the kernel's column is from
another draw of the same construction.) The per-chunk median is the same at the first and the last chunk (e.g. 0.53% and 0.48%,
Qwen3-Next without decay, for `fineStep`). The tolerance is about 4–5 times the kernel's measured step error: the
gated values' roundings it adds up through `kᵀ` are worst-case, and they partly cancel in the
kernel. On `check_gpu.py`'s launches the worst error / tolerance is up to 0.885 for the step (short
chunks, where one token's rounding is attained) and up to 0.965 for `v_new` (its own store's
rounding, attained up to the binade).

**What `Post` admits.** A kernel deviating by up to `errStep` in every chunk meets `Post`. The
deviation that drifts farthest pushes each state away from the exact recurrence
(`Y (t + 1) = Step t (Y t) + 0.98·errStep t·d/‖d‖`, `d = Step t (Y t) − X (t + 1)`, `X` the exact
recurrence started at `h0`; stored as `bf16(Y (t + 1))`). Its drift (median of
`‖h[t] − X t‖/‖X t‖` over 32,768 tokens, 512 chunks, `K = 128`, 4 value heads, `V = 64`;
`make_inputs`' gates by `gate_scale`):

| gates | admitted drift | the kernel's drift |
|---|---|---|
| none (`USE_G = False`, DeltaNet; `gate_scale = 0`) | 3.15% | 0.30% |
| `gate_scale = 0.003` (about `0.15%` decay per token) | 2.46% | 0.28% |
| `gate_scale = 0.05` | 0.91% | 0.24% |

(With `fineStep` in place of `errStep` the admitted drift is 2.73%, 2.16% and 0.86%.) Without the
gate the claim is the most permissive, about `10×` the kernel's drift. The drift is
flat, not compounding: it reaches its level within about 16 chunks and stays there (the
delta rule's update pulls the state back toward the keys' targets, and the gates' decay shrinks
earlier deviations).

A bias of `1 ± 2^-7` per chunk on every state, a dropped decay, gate or chunk, a wrong key head and
a corrupted element are refuted by `check_gpu.py` (no states `Y` exist: the nearest points of two
consecutive rounding boxes are farther apart than one step's tolerance). A chunk's `g_last` read
one token early is refuted on `make_inputs`' gates from `gate_scale = 0.02` (largest lower bound /
`errStep` `1.86` over 16,384 tokens); at `gate_scale = 0.01` (`0.89`) and below its error is below
`errStep` and those launches meet `Post`. Such a kernel
still does not meet `Statement`, which quantifies over every covered input: on inputs with strong
decay (larger gates) its error exceeds `errStep`, as `check_gpu.py` shows.

## Exclusions

What `SETUP.md` allows and the claim leaves out:

* **Out of scope**: dimensions that are not multiples of 128 or exceed 2^17, contexts longer than
  2^22 tokens, and batches of more than 8192 requests: the claim covers real model shapes only.
  Here: `T > 2^22` (`T_ctx`; a row's tokens on fixed length, the packed batch's on variable length),
  and more than `8192` sequences `N` or batch rows `B` (`N_batch`). The claim has no model dimension
  (`K`, `V` are head dims).
* **`T + 64 ≥ 2^31`, `T·HV ≥ 2^31` with the gate, `HV ≥ 2^16`, `V ≥ 2^16`, `HV·V ≥ 2^31`, `2^31` programs or
  more**: `int32` arguments, index products and compile-time constants, CUDA's grid; outside real
  use (`T·HV ≥ 2^31` is 67 million tokens at 32 value heads; real heads have `HV·V ≤ 2^12`).
* **Values** (`Vals`, a hypothesis of `Post`'s numeric clauses): non-finite inputs, inputs beyond
  `2^16` (`g` beyond `2^24`), a chunk whose gate rises by more than `8` (`g_last − g_i > 8` or
  `g_last > 8`: outside real use, where `g` is a nonincreasing sum of log decays), and padded state
  rows whose bound `padB` passes `2^100` (only when `64 ∤ K` and the decays grow the state; with
  `g_last ≤ 0`, `padB t ≈ t·66·2^-126·(1 + rD)^t`: about `2^-103` at `2^16` chunks and `2^-80` at
  `2^20` chunks, the most `T·HV < 2^31` allows at 32 value heads with the gate); a limit of the kernel's
  `float32`/`bfloat16` arithmetic toward overflow, far outside real use.
* **A stored state out of range** (`StOk t`): a non-finite `h[t]` or `‖h[t]‖ > 2^40`; the clauses
  of chunk `t` and of every later chunk are not claimed (when `64 ∤ K` the padded rows, never
  stored, carry state across chunks, and a chunk out of range leaves them unconstrained: at
  `K = 60`, once an input drives the state out of range, the dot facts allow a later `h[t]` in
  range while the padded rows its `w` dot reads are not finite). Real states are of order `1` to
  `10^3`.
* **The drift over many chunks**: `Post` bounds each step of the state, not the distance of `h[t]`
  from the exact recurrence started at `h0`; the steps' errors are not composed (Tolerances, "What
  `Post` admits": a deviation of up to `errStep` in every chunk gives a drift of about 0.9–3.2% of
  `‖h‖`, against 0.2–0.3% for the kernel).
* `gk`, `save_new_value=False`, other chunk sizes, other dtypes: not part of the setup
  (`SETUP.md`).

## Checks

* **Build**: `Claim.lean` builds with no error, no `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (at `Platform.sm90ref`) and `satisfiable` (`setup_min` in a zero
  memory) are proved.
* **Coverage**: `setup_*_ok` for the 56 instances above and `setup_min_ok` are proved.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): all 77 launches hold `Post` as stated (`SETUP.md`,
  "Checked on the H200": fla's test shapes, Qwen3-Next's and GatedDeltaNet's heads with every flag
  combination, `int32` and `int64` `cu_seqlens`, edges (initial states with `-0` entries among them), 16 with `STATE_V_FIRST`, 21 without the
  gate). `Post`'s states are witnessed by the kernel's own: a copy of the kernel with one more
  store per state block (the `float32` `b_h` beside each `bfloat16` store of lines 185–209) gives
  them, and its `h`, `v_new` and `ht` equal the host's bit for bit. `Y 0` is the initial state (as a
  value), `h[t] = bf16(Y t)` and `ht = Y (nch n)` bit for bit, and the worst error / tolerance is `0.885`
  for the step and `0.965` for `v_new`. `StOk` holds at every chunk, every output is finite, `Vals`
  holds, the frame holds (inputs unchanged, guard regions of 4096 elements around `h`, `v_new`,
  `ht` unchanged, every output element written), and the host's `chunk_offsets` equal `cbase`.
  Forty-seven corrupted outputs are refuted (no states `Y` can meet `Post`: the nearest points of
  two consecutive boxes of `float32` values rounding to the stored states, or to `h0` and `ht`, are
  farther than `errStep` from one step; or `h[0] ≠ bf16(h0)`; or a `v_new` element is out of its
  tolerance), on four launches (grouped value heads in both layouts, without the gate, and slow
  decay over 16,384 tokens, `gate_scale = 0.02`): an `h`, `v_new` or `ht` element moved by 4 times its tolerance, two
  sequences' `h` chunks exchanged, and the exact recurrence with one bug (no decay of the state, a
  chunk's update dropped, the tokens' gate `2^(g_last − g)` dropped, `k` read at the key head
  `hv % H`, every token gated from its chunk's first token, `g_last` read one token early, `ht` the
  state before the last chunk, every state times `1 + 2^-7` or `1 − 2^-7` each chunk); and a
  changed guard element is rejected. The exact recurrence with its state carried in `bfloat16`
  between chunks is accepted (states found as the nearest points of the boxes); with the state
  carried in `float64` it is refuted (its `v_new` is from the `float64` state, while `Post`
  specifies it from the stored `bfloat16` state, which the kernel's `w` dots read). `Vals` fails on
  the padded-row input above (and holds on the same input with `g = 0`, `w = k`).

## Library requests

None needed. (A bound relative to the real tree of a `tl.dot`'s chunks would not change the
dominant term here, the `bfloat16` rounding of the gated values.)
