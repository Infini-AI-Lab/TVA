# `extend_attn_bf16`: the prepared claim

Kernel: `source/extend_attention.py`, `_fwd_kernel` (lines 99–638; the kernel lines are 89–90,
93–96 and 99–638, the host lines 87, 641–673, 676–684, 687–857 and 860–977). Claim:
[`Claim.lean`](Claim.lean), namespace `Claims.ExtendAttn`,
`Statement := ∀ K P, Assumptions K P → Spec K`. Model map: [`model-map.json`](model-map.json). GPU
check: [`check_gpu.py`](check_gpu.py).

The covered paths are `SETUP.md`'s, the launches of SGLang's Triton backend for an extend batch
(`int64` `qo_indptr`, causal) and for a target verification (`int32` `qo_indptr`, the custom tree
mask, `USE_CUSTOM_MASK`), each on a layer with or without a sliding window
(`SLIDING_WINDOW_SIZE > 0`); the other flags as the backend passes them (causal, no logit cap, no
sinks, no xai temperature, no LSE, page size 1, no `score_mod`, `k_scale = v_scale = 1.0`,
`SKIP_PREFIX_CUSTOM_MASK = True`); `bfloat16` `q_extend`, `k_extend`, `v_extend`, KV cache and
output, `int64` `kv_indices`, `mask_indptr`, `window_kv_offsets`, `int32` `kv_indptr`, a `bool`
`custom_mask`; head dims `D = Lq = Lv` up to 256, on an H200 (`BLOCK_M = 128` for `D ≤ 128`, else
`64`; `BLOCK_N = BLOCK_N_PREFIX = 64`).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.ExtendAttn.main : Statement` (`Statement`:
`Claim.lean` line 987). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 605), and every initial memory `init` that `Pre` accepts (line 870), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 949).

**What `Post` asks** (line 949), two parts:
1. for every request `b < batch`, new token `i < extOf d init b`, head `h < H` and component
   `e < D`: element `oIdx d (qsOf d init b + i) h e` of `o` (line 588) reads in `fin` as a finite
   `bfloat16` `c` (`rdB`, line 707), and
   `|c.toReal - attnRef d init b h i e| ≤ tolRel d.BD (tripsPre d init b) (tripsExt d init b i)
   (Xrow d init b h i) * attnAbs d init b h i e + tolAbs (vSum d init b h i e)`.
   The reference `attnRef` (line 880) is the exact softmax in the reals over `keys` (line 812: the
   positions `attends`, line 806, allows), from the initial `q`, `keyK`, `keyV` (lines 777–786; the
   prefix through `kv_indices`, the new tokens from `k`, `v`) and `score` (line 792). `attnAbs` (line
   885) is the same mean of `|v|`, `vSum` (line 890) the sum of `|v|`, `Xrow` (line 817) the row's
   largest `scoreMag` (line 798), `tripsPre`, `tripsExt` (lines 894, 898) the row's trip counts. The
   tolerance is `tolRel` (line 933), built from `dlt`, `rho`, `gam`, `errE` (lines 909–926), and
   `tolAbs` (line 937). Every element in part 1 must be finite and within the bound; there is no
   guard beyond the ranges of `b`, `i`, `h`, `e`;
2. the frame: every byte outside `d.o.elems (outElems d init)` (line 941) is unchanged.

**What you are given** (`Assumptions`, line 962):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, for a valid launch with
  `debug = false` that is well behaved from `init`;
- `dot` (`Num.DotBF16`): each `bfloat16` `tl.dot` into `float32` is within `relDot16 K` of
  `|acc| + Σ liftMag a·liftMag b` plus `absDot16 K` (source lines 344, 437, 520, 593);
- `dot_total` (`Num.DotTotal P cfgBF16`): each `tl.dot` has a result;
- `exp_acc` (`Num.ExpF32`): `tl.exp` on `[-X, X]` within `(2^-22 + 2^-23·X)·e^x + 2^-140`
  (source lines 405, 406, 571, 572);
- `exp_ninf` (`Num.ExpF32NegInf`): `tl.exp(-∞) = +0` (masked lanes, the first trip's `re_scale`);
- `exp_total` (`Num.MathTotal P .exp .binary32`): `tl.exp` has a result;
- `div_full` (`Num.DivFull`): `/` on `float32` within `2^-22` relative, `2^-148` absolute (source line 636);
- `div_total` (`Num.DivTotal P .binary32`): `/` has a result.

**The kernel, as the model runs it** (`fwd_kernel` line 130, `model` line 684, `launch` line 701):
a grid `(batch, H, nBlocks d)`; program `(b, h, blk)` reads `qo_indptr`, `kv_indptr` (and, with
the custom mask, `mask_indptr`, `window_kv_offsets`) by scalar loads, and returns at once when
`ext_b ≤ blk·BM`. Otherwise it loads its `[BM, BD]` block of `q` (rows `blk·BM + [0, BM)` of
request `b`, masked by `mask_m`, `mask_d`), runs two `tl.forRange` loops with the online softmax
(`acc`, `deno`, `e_max`): the prefix loop (`int32`, `0` to `pre_b` by `64`) loads `kv_indices`,
then `k_buffer`, `v_buffer` at those slots; the new-token loop (`qo_indptr`'s type, `extend_start`
to `min(ext_b, (blk+1)·BM)` by `64`) loads `k`, `v` (and the custom mask's block). Each trip is a
runtime `if` on `SKIP_TILE`, two `tl.dot`s (the second as `tl.dot_add` in the new-token loop),
`tl.max1`, `tl.maximum`, `tl.exp`, `tl.sum1`, `tl.mul_add`, `tl.addc`. Then it divides `acc` by
`deno`, converts to `bfloat16` and stores its rows of `o`. The loops write nothing; programs
write disjoint parts of `o` and share only read-only inputs: the route is `TVA.meets_independent`,
with `safe_forRange` (body footprint `fun _ => ∅`, `iterW_empty`) and an invariant per loop
(PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.Valid`: `batch_lt`, `H_lt`, `nb_lt`,
  `grid_lt`; the block shapes from `bd_pow2`, `bd_le`, `bm`; `launch` keeps `debug` at `false`);
- each program is `Safe`: every enabled lane of every load is in bounds (`q_len`, `k_len`, `v_len`,
  `kb_len`, `vb_len`, `qo_len`, `kvp_len`, `kvi_len`, `cmask_len`, `mip_len`, `wko_len`, with
  `ReqOk`'s ranges and slots and `MaskOk`'s block bound); the `int32` products and offsets do not
  wrap (`strides_lt`, `heads_i32`, `cm_i32`, `maxExt_le`, `cm_maxExt`, `sw_le`; `TVA.Lemmas.Int`);
  `tl.idiv` by `kv_group_num` is defined (`kv_pos`, `kv_dvd`); each loop's `tl.RangeOk`; the
  store's lanes are distinct (`o_layout`) and it writes only the program's rows;
- the programs' write footprints are disjoint (requests' new tokens are consecutive ranges of `o`
  by `ReqOk`, heads and components distinct by `o_layout`), and no program reads `o`'s written
  bytes (`disj`);
- the value part of `Post` through the loop invariants: each score within `dlt` (`dot`, the
  products by `sm_scale`), the coefficients and `deno` within `rho` (`exp_acc`, `exp_ninf`,
  `sum_err_covers`, `safe_mul_add`), the second `tl.dot` and the accumulator within `gam` (`dot`,
  `DotLift.dotAdd_step`, `FloatV.AddRel.err`), the division (`div_full`) and the store's rounding
  (`FloatV.cvt_err`); the trips the program runs are at most `tripsPre`, `tripsExt`, and the bound
  grows with them; the values stay finite (`ValsOk`'s `2^100` and `2^12` bounds). The derivation is
  README's `## Tolerances`;
- the frame: programs that return early and the masked lanes write nothing, so every byte outside
  `outElems` keeps its initial value.

**Already in `Claim.lean`** (do not prove again): `consistent` (line 991), `satisfiable` (line
1415), `canon_ok` (line 1069) and the `setup_*_ok` theorems; `bytes_disj` (line 1047) can be reused.

## Specification

Request `b` has the new tokens `qs_b … qs_b + ext_b − 1` (`qs_b = qo_indptr[b]`,
`ext_b = qo_indptr[b+1] − qo_indptr[b]`) of `q_extend`, `k_extend`, `v_extend`, `o_extend`, and a
prefix of `pre_b = kv_indptr[b+1] − kv_indptr[b]` cached tokens at the slots
`slot_b(j) = kv_indices[kv_indptr[b] + j]` of `k_buffer`, `v_buffer`, all read from the initial
memory. Its sequence position `j` is the prefix token `j` (`j < pre_b`: `keyK`, `keyV` read
`k_buffer[slot_b(j)]`, `v_buffer[slot_b(j)]`) or the new token `j − pre_b` (`k_extend`, `v_extend`).
Every tensor is read by its own strides (`qIdx`, …, `kbIdx`, `vbIdx`), and query head `h` reads
key/value head `h // (num_heads // num_kv_heads)`. For the new token `i < ext_b` of request `b`, head
`h`, with `σ = sm_scale` (the double), the positions it attends are `keys` (`attends`): among
`j < pre_b + ext_b`, every prefix position `j < pre_b`, and the new token `j − pre_b` if
`j ≤ pre_b + i` (causal) or, with the custom mask, if its row of the mask allows it
(`maskBit`: element `mask_indptr[b] + i·(pre_b + ext_b + w_b) + w_b + pre_b + (j − pre_b)` of
`custom_mask`, `w_b = window_kv_offsets[b]` with a window, else `0`: request `b`'s row-major block
of `ext_b` rows; the kernel does not read the prefix columns, `SKIP_PREFIX_CUSTOM_MASK = True`, and
the reference attends the whole prefix); with a window `W = sw > 0`, only those with
`pre_b + i ≤ j + W` (the last `W + 1` positions up to its own). For a sliding-window layer the
prefix is the one the host passes (the backend's `window_kv_indptr`, `window_kv_indices`: the last
`min(seq, W)` cached tokens). Then:

* `score_j = σ·Σ_{e<D} q[qs_b+i, h, e]·keyK_j[e]`;
* `attnRef[e] = Σ_{j∈keys} e^{score_j}·keyV_j[e] / Σ_{j∈keys} e^{score_j}` (exact softmax);
* `attnAbs[e]` the same weighted mean of `|keyV|` (`≥ |attnRef|`), `vSum[e] = Σ_{j∈keys} |keyV_j[e]|`;
* `Xrow = max_{j∈keys ∪ {pre_b+i}} σ·Σ_e liftMag(q_e)·liftMag(keyK_je)` (`Rules.Num.liftMag .bf16`,
  a subnormal counted at `2^-126`; `Xrow ≥ max|score_j|`; every covered row attends its own
  position `pre_b + i`), and
  the loop trips of its program, `Tp = cdiv(pre_b, 64)` over the prefix (`tripsPre`) and
  `Te = cdiv(min(ext_b, (i // BM + 1)·BM), 64)` over the new tokens (`tripsExt`), `T = Tp + Te`.

`Post`: every `o_extend[qs_b + i, h, e]` (`b < batch`, `i < ext_b`, `h < num_heads`, `e < D`) is a
finite `bfloat16` `c` with

`|c − attnRef| ≤ tolRel(BLOCK_DMODEL, Tp, Te, Xrow)·attnAbs + 2^-80·(1 + vSum)`,

and every byte outside those elements is unchanged. With `B = BLOCK_DMODEL`:

* `δ = (B + 8)·2^-22·X + 2^-40` (each computed score's error),
* `ρ = exp(2^-8 + (T + 8)·2^-20 + (X + δ)·2^-17) − 1` (each position's coefficient and the
  normaliser `deno`),
* `γ = exp((66 + 2·Tp + 72·Te)·2^-22) − 1` (the second `tl.dot` and the accumulator's roundings:
  the prefix stage's dots have zero accumulators, the new-token stage's accumulate),
* `E = e^{2δ}·(ρ + γ·(1 + ρ))/(1 − ρ) + (e^{2δ} − 1)`, `tolRel = 2^-8 + (1 + 2^-8)·E`.

| `B` | `Tp` | `Te` | the request | `X` | `tolRel` |
|---|---|---|---|---|---|
| 128 | 0 | 1 | no prefix, ≤ 64 new tokens | 8 | `8.5·10^-3` |
| 128 | 16 | 32 | 1K prefix, 2K new tokens | 8 | `9.1·10^-3` |
| 128 | 94 | 97 | the unit test's longest (6K + 6K) | 8 | `1.04·10^-2` |
| 128 | 1920 | 128 | 120K prefix, 8K new tokens (128K) | 16 | `1.42·10^-2` |
| 128 | 4064 | 32 | 254K prefix, 2K new tokens (2^18) | 16 | `1.56·10^-2` |
| 128 | 0 | 2048 | 128K new tokens, no prefix | 16 | `4.7·10^-2` |
| 128 | 0 | 1 | short | 256 | `2.7·10^-2` |
| 128 | 0 | 1 | short | 4096 (the cap) | `0.36` |
| 256 | 4064 | 32 | 2^18 | 4096 (the cap) | `0.76` |

On `randn` inputs `Xrow` is about `0.64·D·σ` (`≈ 7` at head dim 128). The floor `≈ 2·2^-8` is the two
`bfloat16` roundings the kernel makes (`p.to(bfloat16)` and the store), each up to `2^-8` relative;
an input on which both push the same way (`check_gpu.py`'s `pre-adversarial`: every `p` just above a
`bfloat16` midpoint, `v` constant) reaches `0.96` of the bound, and with the smallest score bound
and one trip (`pre-adversarial-small`: head dim 16, scores `±0.345`, a gap of `ln 2`, the top key's
`v = 0` so every weighted value rounds up) `0.98` (the two roundings, `2^-7`, against a floor of
`2·2^-8` plus the per-trip terms; no input exceeds about `0.987`), and random inputs `0.72`. At the score cap
(`X = 2^12`) the bound is `0.36`–`0.88` of `attnAbs`, close to vacuous; real scores are far below it
(`X ≈ 7`).

## Conditions

`Inst`: `cm` (`USE_CUSTOM_MASK`: a target verification, whose `qo_indptr` is `int32`; else an
extend batch, `int64`), `sw` (`SLIDING_WINDOW_SIZE` when positive, `0` for the host's `-1`),
`batch`, `T` (`q_extend.shape[0]`), `S` (the cache's slots), `NI` (`kv_indices.shape[0]`), `NM`
(`custom_mask.shape[0]`), `H`, `HKV`, `D`, `BD` (`BLOCK_DMODEL = BLOCK_DV`), `BM` (`BLOCK_M`),
`maxExt` (`max_len_extend`), `sm_scale` (the double), each tensor's two outer strides, and the
twelve storages (`qo_indptr`'s of dtype `qoD cm`). `Inst.ok`:

* the grid `(batch, H, cdiv(maxExt, BM))` within CUDA's limits, fewer than `2^31` programs;
* `1 ≤ HKV`, `HKV ∣ H` (SETUP.md); `1 ≤ D ≤ BD ≤ 256`, `D ≠ 192` (Exclusions), `BD` a power of
  two; `BM = 128` for `D ≤ 128`, else `64` (the H200's choice);
* `maxExt ≤ 2^18` (see Exclusions), and `≤ 2^12` with the custom mask (the draft tokens of a
  request; the mask's row offsets are then `int32` products, below `2^12·2^18`);
* `T ≤ 2^22`: the batch's new tokens (`q_extend.shape[0]`) are a context length, at most `2^22`
  tokens (real model shapes, see Exclusions);
* `batch ≤ 8192`: the number of requests (`qo_indptr.shape[0] - 1`) is a batch size, at most `8192`
  (real model shapes, see Exclusions);
* `sw ≤ 2^18`;
* `sm_scale` finite in `[2^-64, 2^64]` (the default is `1/sqrt(D) ∈ [1/16, 1]`);
* the twelve strides are `int32` arguments (`< 2^31`);
* **the `int32` products `cur_head * stride_qh`, `cur_kv_head * stride_kh`, … do not wrap**
  (`heads_i32`): a limit of the kernel, see Exclusions;
* **with the custom mask (`int32` `qo_indptr`), the element offsets of `q_extend`, `k_extend`,
  `v_extend`, `o_extend` do not wrap** (`cm_i32`: `(T−1)·stride_t + (heads−1)·stride_h + D ≤ 2^31`):
  a limit of the kernel, see Exclusions;
* each storage holds its tensor; `qo_indptr`, `kv_indptr` hold `batch + 1` elements, `kv_indices`
  `NI`; with the custom mask, `custom_mask` `NM`, `mask_indptr` `batch + 1` and, with a window,
  `window_kv_offsets` `batch`;
* `o_extend` row-major in one order or the other (`empty_like(q_extend)`), and disjoint from the
  inputs the launch reads.

The kernel's element offsets of an extend batch are `int64` (`qo_indptr` and `kv_indices` are
`int64`, and every offset is built from them), so they need no bound beyond the storages'; with an
`int32` `qo_indptr` those of `q_extend`, `k_extend`, `v_extend`, `o_extend` are `int32` (`cm_i32`),
the cache's still `int64` (through `kv_indices`).

`Pre`: the nine buffers `Valid` (and, with the custom mask, `custom_mask`, `mask_indptr` and, with
a window, `window_kv_offsets`); `ReqOk`: for each request, `0 ≤ qo_indptr[b] ≤ qo_indptr[b+1] ≤ T`
with `ext_b ≤ maxExt`, `0 ≤ kv_indptr[b] ≤ kv_indptr[b+1] ≤ NI`, `pre_b + ext_b ≤ 2^18`, and every
prefix slot in `[0, S)` (the requests' new tokens are then disjoint: consecutive ranges of `q_extend`);
`MaskOk` (with the custom mask): `mask_indptr[b] ≥ 0`, with a window `0 ≤ window_kv_offsets[b] ≤
2^40`, request `b`'s block within `custom_mask` (`mask_indptr[b] + ext_b·(pre_b + ext_b + w_b) ≤
NM`), and a tree mask over the new tokens: every new token attends itself and no later new token
(speculative decoding's tree: a draft token sees its ancestors, which precede it);
`ValsOk`: in every request, every component of its queries, of the keys and values of its new tokens
and of its prefix slots finite, `|v| ≤ 2^100` (new tokens and prefix), and every attended pair with
`σ·Σ liftMag q·liftMag k ≤ 2^12` (pairs the masks remove are replaced by `−∞` with a
`tl.where`, so their scores need no bound; their values must be finite, since the second `tl.dot`
reads them with a zero weight).

The launch: grid `(batch, H, cdiv(maxExt, BM))`, the source's block shapes, `fusion = true`,
`debug = false`, `num_ctas = 1`, no `launch_pdl`; only `num_warps`/`num_stages` among the other
options, no heuristics or autotuning, a `JITFunction` launch, no descriptors:
within part (A) of `Compiler.Faithful`'s scope. The claim relies on part (B) (the environment) as the
rule states it.

## Assumptions

`[rule]` fields (`docs/rules/num.md`'s table): `faithful` (`Compiler.Faithful`), `dot` (`DotBF16`),
`dot_total` (`DotTotal P cfgBF16`), `exp_acc` (`ExpF32`), `exp_ninf` (`ExpF32NegInf`), `exp_total`
(`MathTotal P .exp .binary32`), `div_full` (`DivFull`), `div_total` (`DivTotal P .binary32`).
`consistent` uses `Platform.sm90ref`.

## Model notes

* `SKIP_TILE = False` (lines 296, 505) makes `SKIP_TILE` an `int1` tensor (a name assigned a Python
  `bool`), so `if not SKIP_TILE:` (lines 300, 509) is a runtime `if`; the model writes it as one
  (`if (Tensor.not SKIP_TILE).item then … else pure (acc, deno, e_max)`). With a window (both
  stages) or the custom mask (stage 2) `SKIP_TILE` is `tl.max(tl.max(final_mask.to(tl.int32),
  axis=1), axis=0) == 0` (lines 298, 507): `tl.convert .i1 .i32`, the exact folds
  `Tensor.foldRows`/`Tensor.foldAll` with `IntV.max` from the `int32` minimum `imin32`; a tile no
  row of the block attends is skipped. Without them every program takes the branch.
* `qo_indptr`'s element type is `qoD qo32` (`int32` for a target verification, `int64` for an
  extend batch; Triton compiles the kernel for each): `cur_seq_extend_start_idx`,
  `cur_seq_len_extend`, `cur_seq_len`, the offsets of `q_extend`, `k_extend`, `v_extend`,
  `o_extend`, `cur_block_m_end` and the second loop's induction variable have it (`QI qo32`);
  `loadQo` is the scalar `tl.load1` of `qo_indptr`, `toQ` the promotion of an `int32` operand
  beside them (none for `int32`, `tl.convert .i32 .i64` for `int64`), `qToI64` theirs beside an
  `int64` (`window_kv_offset`); `kv_indptr` is `int32`, so `cur_seq_len_prefix` and the first
  loop's induction variable are `int32`.
* The custom mask's names (`cur_seq_mask_start_idx`, line 214; `window_kv_offset`, lines 217–219:
  the `int32` `0`, or with a window the loaded `int64`) are read only under `USE_CUSTOM_MASK`; the
  model keeps them in `cm : Option (I64 × Option I64)`, `none` without the custom mask, and the
  mask's address (lines 465–475) is built with the type `window_kv_offset` has. The extend
  stage's `if USE_CUSTOM_MASK: … elif IS_CAUSAL: …` (lines 464–485) is a `match` on `cm`.
* With a window, stage 2 starts at `extend_start = (max(cur_block_m·BLOCK_M − W, 0) // BLOCK_N)·
  BLOCK_N` (lines 454–458), and both stages `&` the window mask (lines 288–294, 490–503;
  `IS_CAUSAL = True`, so the stage-2 mask is the one-sided `q ≤ kv + W`).
* `deno * re_scale + tl.sum(p, 1)` is `tl.mul_add` with the IEEE `tl.sum1`; the prefix stage's
  `acc * re_scale[:, None] + dot * v_scale` is `tl.addc` of two products (its `dot` feeds a
  product, not the addition: not the combine pass's shape); the extend stage's
  `acc * re_scale[:, None] + dot` is `tl.dot_add` (the dot's default zero accumulator, no other
  use). `qk - n_e_max[:, None]` is the IEEE `tl.sub`: `qk` is a `tl.where` value.
* `q.to(k.dtype)` (line 344) converts `bfloat16` to `bfloat16`: `tl.cvt`, exact.
* `row_max_fixed` (`-1e20` for a row with no attended position in the trip) keeps `n_e_max` finite,
  so `e_max − n_e_max` is never `−∞ − (−∞)`: rows past the request's new tokens (masked) run on
  finite numbers or `exp(−∞) = +0` and are never stored; a valid row whose first trips attend
  nothing (a window, or a tree mask) keeps `deno = 0` and `e_max = −1e20` until its first attended
  trip.

## Coverage

`canon cm sw batch T S NI NM H HKV D BD BM maxExt sm` is the host's launch (contiguous `q_extend`,
`k_extend`, `v_extend`, cache, `o_extend = empty_like(q_extend)`). `canon_ok` proves `Inst.ok` for
an extend batch or a target verification (`cm`), with or without a window `sw ≤ 2^18`, every grid
of fewer than `2^31` programs with up to `8192` requests (so also batches of many short extends),
`T ≤ 2^20`, `S ≤ 2^22`, any `NI` and `NM`, `1 ≤ H ≤ 128`, `HKV ∣ H`, `1 ≤ D ≤ BD ≤ 256`,
`D ≠ 192` (`BD` a power of two) with the H200's `BM`, `maxExt ≤ 2^18` (with the custom mask
`maxExt ≤ 2^12` and `T·H·D ≤ 2^31`) and every `sm_scale` in range. The code base's test at its
largest draw (19 requests, 12/4 heads, lengths up to 6164: `T = 117116`, `S = 234232`) is
`setup_test256`, `setup_test128`, `setup_test96`, `setup_test80`, `setup_test13`, each `_ok` with
the host's `1/sqrt(D)` (the doubles `sm256`, …, `sm13`); its sliding-window test (window 127, 64/8
heads, head dim 128, the same lengths, the whole prefix passed) is `setup_test_swa`. SETUP.md's
model shapes are `setup_llama` (32/8, 128, 8 requests, 8192 new tokens, a cache of `2^20` slots),
`setup_qwen` (28/4, 128, 16384 new tokens), `setup_h64` (14/2, 64, 64 requests), `setup_gemma`
(16/8, 256, `BM = 64`), `setup_mqa` (8/1, 128, 128 requests), `setup_long` (one request, a prefix
of `2^17 − 8192` and 8192 new tokens), `setup_long256` (32/4 heads, a prefix of `2^18 − 8192` and
8192 new tokens), `setup_gemma_local` (Gemma 3's local layers: window 1023, 16/8, 256, 8 requests,
8192 new tokens, window prefixes up to 1023), and the target verifications `setup_verify`
(Llama-3-8B's heads, 64 draft tokens per request, 32 requests, `int32` `qo_indptr`, a tree mask of
`2^29` elements), `setup_verify_chain` (4 tokens per request, 256 requests, 28/4) and
`setup_verify_swa` (Gemma 3's local layers, 8 tokens per request, window 1023). Prefix length 0 and
extend length 1, the masks' contents and the window offsets are values of `Pre`, not of `Inst`;
`satisfiable` uses one request with prefix 0 and one new token.

## Tolerances

Derived from the library's facts (`u = 2^-24`, `docs/rules/num.md`). Fix a row (`b`, `h`, `i`),
`X = Xrow`, `N = 64`, `B = BLOCK_DMODEL`; the row's trips are `t = 0 … T−1` (prefix trips first),
`M_t` the running maximum after trip `t` (`n_e_max`), `M = M_{T−1}`. `tripsPre` and `tripsExt`
bound the trips the row's program runs: a window starts stage 2 later (`extend_start`), and a tile
no row of the block attends is skipped (`SKIP_TILE`); every bound below grows with the trips, so
fewer trips only lower it.

1. **Scores.** `qk = tl.dot(q, k)` with a zero accumulator: `DotBF16` (`relDot16 B = (B+2)·2^-22`
   of `L = Σ liftMag q·liftMag k = scoreMag/σ`, plus `(B+2)·2^-126`; room since `L ≤ 2^76`).
   `σ32 = hostF32(σ)`: `u`; `σ32·k_scale` is `σ32` exactly (`k_scale = 1.0`); `qk·σ32`: `u`. So every
   attended `qk_j = score_j + ε_j`, `|ε_j| ≤ (B+3)·2^-22·X + 2^-52 ≤ δ`. Masked positions are `−∞`
   exactly (`tl.where`).
2. **Effect of the scores.** With `ŵ_j ∝ e^{score_j}` and `ŵ'_j ∝ e^{qk_j}` over `keys`,
   `ŵ'_j/ŵ_j ∈ [e^{-2δ}, e^{2δ}]`, so `|Σŵ'v − attnRef| ≤ (e^{2δ}−1)·attnAbs` and
   `A' = Σŵ'|v| ≤ e^{2δ}·attnAbs`.
3. **The online softmax, exactly.** Position `j` of trip `t` reaches `acc` with the coefficient
   `c_j = p̃_j·Π_{t'>t} re_{t'}` (`p̃_j = bfloat16(p_j)`, `p_j = exp(qk_j − M_t)`,
   `re_{t'} = exp(M_{t'−1} − M_{t'})`) and `deno` with `p_j·Π_{t'>t} re_{t'}`; in exact arithmetic both
   are `e^{qk_j − M}`, so `c_j/deno = ŵ'_j` without rounding. The first trip's `re = exp(−∞) = +0`
   (`ExpF32NegInf`; `e_max = −∞`, `n_e_max` finite) multiplies the zero `acc` and `deno`. A trip with
   no attended position for the row (its last trips over the new tokens) has `row_max = −∞`, so
   `n_e_max = maximum(−1e20, e_max) = e_max`, `re = exp(+0)` and `p = exp(−∞) = +0`. Without a
   window or the custom mask every valid row's first trip attends a position (the prefix's first, or
   new token 0), so `M_t` is finite from `t = 0`. With them a row's first trips may attend nothing
   while other rows of the block do: then `n_e_max = maximum(−1e20, −∞) = −1e20` and later
   `maximum(−1e20, −1e20)`, `p = exp(−∞) = +0` (`ExpF32NegInf`), so `deno` stays exactly `0` and
   `acc` holds only the second `tl.dot`'s absolute terms (`absDot16 64` per trip, step 7); at its
   first attended trip `re = exp(−1e20 − M_t) ≤ 2^-140` (`ExpF32`) multiplies them, and from there
   on the argument above holds with that trip as the first (the running maximum's increments after
   it sum to at most `2(X+δ)`). Every row attends its own position `pre_b + i` (`MaskOk`'s diagonal;
   `q = kv` passes the window), so some trip is attended and `deno ≥ 1/2` at the end.
4. **The coefficients' errors** (`1+κ_j` for `c_j/e^{qk_j−M}`, `1+λ_j` for position `j`'s share of
   `deno`), each a product of factors `1 ± r`:
   * `tl.exp` (`ExpF32` at `X' = |x|`): `2^-22 + 2^-23·|x|` relative, `2^-140` absolute; the
     argument's IEEE `−`: `e^{±u|x|}`. For `p_j`, `|x| ≤ 2(X + δ)`; for the `re`'s, the arguments
     `M_{t'} − M_{t'−1} ≥ 0` sum to at most `2(X+δ)` (the running maximum only grows), so all `re`'s
     together give `T·2^-22 + 1.5·2^-23·2(X+δ)`; the same `p_j` and `re` values enter `acc` and
     `deno`.
   * `p.to(bfloat16)` (`acc` only): `2^-8` relative, `2^-134` absolute.
   * `tl.sum1` of `p` (64 nonnegative lanes, any tree; `sum_err_covers`): `63u`; `tl.mul_add`
     (`deno·re + Σp`, nonnegative; `FusedErr.mulAdd_err`: `3u` of `|deno·re| + |Σp|`, plus `4η`):
     `3u` per trip.
   * The final `DivFull` (`acc/deno`, a quotient `≤ 2^101`, since `deno ≥ 1/2`: the position that
     attains `M` has `p = exp(+0)` and every later `re = exp(+0)`): `2^-22` and `2^-148`.

   Summed: per trip at most `11u` (the `re`'s `tl.exp`, `4u` in each of `κ` and `λ`, and `deno`'s
   `tl.mul_add`, `3u`; the product `acc·re` is charged in step 5), otherwise `2^-8 + 75u` (`p`'s
   `tl.exp`, `4u` in each, `tl.sum1`'s `63u`, the division's `4u`), and `6·2^-22·(X+δ)` from the
   growing `tl.exp` error, so `K + Λ ≤ ρ` (`K = max|κ_j|`, `Λ = max|λ_j|`), with margins of `1.45` on
   the per-trip term (`2^-20 = 16u`), `1.7` on the fixed term (`8·2^-20 = 128u`) and 5 on the `X` term
   (`2^-17 ≥ 5·6·2^-22`); `|Π(1±r) − 1| ≤ e^{Σr} − 1`. Then
   `|c_j/deno / ŵ'_j − 1| ≤ (K+Λ)/(1−Λ) ≤ ρ/(1−ρ)`, and `Σ (c_j/deno)|v_j| ≤ (1+ρ)/(1−ρ)·A'`.
5. **The second `tl.dot` and the accumulator.** `C_t` is the current `Σ c|v|` after trip `t`
   (room: everything is below `2^119`), `C = Σ_j c_j|v_j|` at the end; the later `re`'s are at most
   `1 + 2^-21`.
   * A prefix trip computes `tl.dot(p̃, v)` with a zero accumulator (`DotBF16`, `K = N = 64`):
     `66·2^-22` of that trip's own `A_t = Σ liftMag p̃·liftMag v`, not of the accumulator; then
     `dot·1.0` (exact), the IEEE `acc·re` (`u`) and `tl.addc` of the two (`AddRel`, fused operands:
     `3u` of the operands' magnitudes). The dots' errors sum, after the later scalings, to at most
     `66·2^-22·C`, and the roundings to `4u = 2^-22` of `C_t` per trip.
   * A new-token trip is `tl.dot_add(p̃, v, acc·re)`: either `tl.dot` with `acc·re` as accumulator,
     `66·2^-22` of `|acc·re| + A_t` (the library's fact charges the accumulator), or the dot then the
     fusable addition (`3u` more); with the IEEE `acc·re` (`u`), at most `67·2^-22` of `C_t`.

   By induction over the trips (prefix first) the total is at most
   `(e^{(66 + Tp + 67·Te)·2^-22} − 1)·C ≤ γ·C` (`2·Tp` and `72·Te` for the margin), and
   `C/deno ≤ (1+ρ)/(1−ρ)·A'`.
6. **Assembly.** `|acc/deno − attnRef| ≤ γ·C/deno + ρ/(1−ρ)·A' + (e^{2δ}−1)·attnAbs + abs ≤
   E·attnAbs + abs` (the division's `2^-22` is in `Λ`). The store: `|c − o| ≤ 2^-8·|o| + 2^-134`,
   `|o| ≤ attnAbs·(1+E) + abs`, so `|c − attnRef| ≤ (2^-8 + (1+2^-8)·E)·attnAbs + (1+2^-8)·abs +
   2^-134`, and `c` is finite (`|o| ≤ 2^101`).
7. **Absolute terms.** Before the division (by `deno ≥ 1/2`): the second dot's `absDot16 64 =
   66·2^-126` per trip; the lifts of subnormal `p̃` and `v` (`2^-126` each, times `relDot16 64`);
   `p̃`'s `2^-134·|v_j|` and `tl.exp`'s `2^-140·|v_j|` per position; the `re`'s `2^-140` times
   `|acc| ≤ deno·max|v|` per trip; the IEEE products' `2^-150` and `deno`'s
   `tl.mul_add`'s `4η = 2^-147` per trip. In all, after the division and the store, `abs ≤
   2^-106 + 2^-109·vSum ≤ 2^-80·(1 + vSum)` for `T ≤ 2^12 + 2`.

The bound is provable from `DotBF16`, `ExpF32`, `ExpF32NegInf`, `DivFull` and the library's IEEE
rounding lemmas (`FloatV.mul_err`, `add_err`, `sub_err`, `cvt_err`, `AddRel.err`, `mulAdd_err`,
`sum_err_covers`);
`ρ < 1/2` on every covered row (`T ≤ 2^12 + 2`, `X ≤ 2^12`: `ρ ≤ 0.044`). Meaningful: `tolRel ≈
0.85·10^-2` of the weighted mean of `|v|` at short lengths, `1.02×` the kernel's worst error on the
adversarial inputs (analytic ceiling: a margin of about `1.013`) (which reaches the sum of the two `bfloat16` roundings, `2^-7`), `1.4×` on random
inputs. It grows with the trips: by `2^-20` per trip in `ρ` (the derivation's count of the per-trip
`float32` roundings, with margin `1.45`), by `2·2^-22` per prefix trip and, through the library's
`tl.dot` fact (`66·2^-22` of the accumulator per `tl.dot` that accumulates), by `72·2^-22` per
new-token trip in `γ`: `tolRel ≈ 0.014` at a 128K context with an 8K chunk, `0.016` at `2^18`, but
`0.047` for 128K new tokens with no prefix (`0.087` for `2^18`): that growth is `DotBF16` charging
`66·2^-22` of the accumulator on every new-token trip (see Library requests). On random inputs at 128K the all-zero output and the
output without a quarter of the prefix are rejected (`2.6`, `1.5` times the bound); at `2^18` the all-zero output passes on all but the tail (`524,125` of the `524,288` elements are within the bound) and is rejected only by it (`1.3`); the real check there is the shifted-prefix test (`6.0`–`6.6`), and the output without a quarter of the prefix is within the
bound (`0.83`) on random values, where dropping a quarter of `2^18` independent positions changes the
exact attention by about `attnAbs/sqrt(n)`, and rejected (`6.0`) when the dropped quarter's values
differ (shifted by `0.25`).

## Exclusions

* **`int32` head offsets (a limit of the kernel).** `cur_head * stride_qh`,
  `cur_kv_head * stride_kh`, `stride_vh`, `stride_buf_kh`, `stride_buf_vh` and `cur_head * stride_oh`
  are `int32` products (lines 241, 317, 412, 513, 577, 623); beyond `2^31` they wrap and the kernel
  reads and writes the wrong elements. `Inst.ok` keeps `(H − 1)·stride < 2^31` (`heads_i32`). The
  first host launch that hits it has a head stride times the head count at `2^31` (contiguous tensors:
  `num_heads·head_dim ≥ 2^31`, far beyond any model); real use does not reach it.
* **More than `2^18` positions in one request** (`pre_b + ext_b > 2^18`) and `max_len_extend > 2^18`:
  a limit of the derivation and of the library's `tl.dot` fact, not of the kernel (correct there) nor
  of real use (contexts above 256K exist): the bound grows with the trips (`2^-20` per trip in `ρ`,
  and `72·2^-22` per new-token trip from the `tl.dot` fact, which charges the accumulator), and the
  absolute floor's derivation assumes `T ≤ 2^12 + 2`. At `2^18` it is `0.016` with a prefix and an 8K
  chunk, `0.087` for `2^18` new tokens with no prefix.
* **Contexts longer than `2^22` tokens, batches of more than `8192` (out of scope).** Dimensions
  that are not multiples of 128 or exceed `2^17`, contexts longer than `2^22` tokens, and batches of
  more than `8192` requests: the claim covers real model shapes only. The fields are `T ≤ 2^22`
  (`T_le`, the batch's new tokens) and `batch ≤ 8192` (`batch_le`, the requests); each request is
  already within `2^18` positions (`ReqOk`, `maxExt_le`). The kernel has no model dimension (`D` is
  a head dimension; `H`, `HKV` head counts; `S`, `NI` table sizes). Every launch of `SETUP.md`
  meets them (`T ≤ 2^17`, at most `256` requests).
* **Scores near the cap.** At `X = 2^12` the bound is `0.36` (head dim 128) to `0.88` (256) of
  `attnAbs`: close to vacuous. Real scores are around `X ≈ 7`; inputs that reach the cap are covered
  only in this weak sense.
* **Head dims of 8 or less** (`BLOCK_DMODEL < 16`) are rejected on the host (`tl.dot` needs
  `K ≥ 16`; `check_gpu.py` confirms it at `D = 1` and `8`): no outcome. They stay in `Inst.ok`.
  Head dims above 256, the MLA head dims 192/288/576 (`BLOCK_DPE > 0`, another path of the kernel)
  and `Lq ≠ Lv`: not part of SETUP.md's launches. At `D = 192` the host runs the DPE split
  (`BLOCK_DMODEL = 128`, `BLOCK_DPE = 64`, source lines 653–655), which is not this claim's
  configuration: `Inst.ok` requires `D ≠ 192`.
* **Input values.** Non-finite `q`, `k`, `v` or cache values in a request (a NaN or `∞` reaches a row
  through `0·NaN` in the second `tl.dot`), `|v| > 2^100` (the unnormalised `float32` accumulator,
  up to `deno·max|v|` with `deno` up to `2^18`, can overflow near `2^128`: a limit of the kernel at
  that magnitude), and attended pairs with `σ·Σ liftMag q·liftMag k > 2^12` (real scores are far
  below; beyond it the bound would be vacuous): outside real use. Values below about `2^-80` (subnormal
  `bfloat16`) meet the absolute floor trivially: the library's absolute `tl.dot` term alone is about
  `2^-108`.
* **`sm_scale` outside `[2^-64, 2^64]`**, negative or zero: outside real use.
* **Malformed requests**: `qo_indptr` decreasing or past `T` (two requests sharing a row would make two
  programs write one element), a prefix slot outside the cache, `ext_b > max_len_extend` (the grid
  would miss rows), `o_extend` overlapping the inputs: the backend's metadata never builds them.
* **The grid** beyond CUDA's limits (`num_heads ≥ 2^16`, `2^31` programs): no launch.
* **`int32` element offsets with an `int32` `qo_indptr` (a limit of the kernel).** In a target
  verification the offsets of `q_extend`, `k_extend`, `v_extend`, `o_extend` (lines 238–243,
  511–515, 575–579, 620–625) are built from the `int32` `qo_indptr` and wrap beyond `2^31`;
  `Inst.ok` keeps them below (`cm_i32`). The first host launch that hits it has
  `T·num_heads·head_dim > 2^31` tokens' elements in one verification (contiguous `q_extend`: e.g.
  `2^17` draft tokens at 32 heads of 128 at once); real verifications have `T` of a few thousand.
* **Custom masks that are not tree masks**: a mask with an entry above the diagonal, or a new
  token not attending itself (`MaskOk`). The kernel reads a row's new tokens only up to the end of
  its block (`IS_CAUSAL = True`), so it ignores entries beyond it; speculative decoding's tree masks
  have neither: outside real use. More than `2^12` draft tokens per request (`cm_maxExt`, where the
  mask's `int32` row offsets stay below `2^30`): outside real use (EAGLE uses up to 64 or so).
* **The other paths and dtypes** (logit cap, sinks, xai temperature, LSE, DCP's skip flags, pages
  larger than 1, `identity_kv_indices`, `score_mod`, non-causal layers and the two-sided window of
  bidirectional ones, `SKIP_PREFIX_CUSTOM_MASK = False`, `k_scale`/`v_scale ≠ 1`, an `int32`
  `qo_indptr` without the custom mask or an `int64` one with it, `int32` `kv_indices` as in the code
  base's unit tests): not part of SETUP.md's launches.
* **An fp8 KV cache** (`--kv-cache-dtype fp8_e4m3`), a common H200 setting: `k_buffer`, `v_buffer` are
  fp8, so the prefix stage runs another computation (`q.to(k.dtype)` converts `q` to fp8, and both
  prefix `tl.dot`s are fp8 dots, with `k_scale`, `v_scale` the cache's scales): another configuration
  of the kernel, not this claim's.

## Library requests

* **A tighter fact for an accumulated `tl.dot` of `bfloat16` operands.** `DotBF16` charges
  `(K+2)·2^-22` of `|acc|` on every `tl.dot` that accumulates, so over the new-token stage's trips the
  bound grows by `72·2^-22` per 64 new tokens (`tolRel ≈ 0.047` for 128K new tokens with no prefix,
  `0.087` for `2^18`), while the device's per-chunk truncation of the accumulator is about `2^-23` of
  it. `DotBF16Chunk` does not lower it: each `tl.dot` here is one chunk (`K = 64`), charged
  `min(2^-13·M, 2·A + 2^-22·M) + 2^-20·|result|`, where `2·A` is twice the chunk's own `Σ|p·v|` and
  `2^-13·M` is `1024·2^-23` of the accumulator. A fact bounding the accumulator's share per chunk
  near `2^-22·|acc|` without the `2·A` term would keep the bound near `10^-2` there. Real chunks are 8K–16K new tokens, where the bound is
  `0.009`–`0.011`.

## Checks

* **Builds**: `Claim.lean` builds with no errors or warnings, no `sorry`, no `axiom`
  (`#print axioms`: `propext`, `Classical.choice`, `Quot.sound`).
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (one request with no prefix and
  one new token, head dim 16, zero inputs, `qo_indptr = [0, 1]`, `kv_indptr = [0, 0]`).
* **Covers the setup**: `canon_ok` and the seventeen `setup_*_ok` above.
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0): the code base's test at full size for head dims 256,
  128, 96, 80 and 13 (worst `0.461`; on its `N(0.1, 0.2)` values the bound is a median `1.6%` of
  `|attnRef|`, at most `2.5%`); SETUP.md's model shapes with scattered prefix slots, prefix 0 and
  extend 1 (worst `0.720`, Llama-3-8B's heads); a strided `q_extend`; long requests (a 128K
  context, `2^17 − 8192` + 8192, and the cap, `2^18 − 2048` + 2048, on random values and with the
  prefix's first quarter shifted); sliding-window layers (the code base's window 127 at 64/8 heads,
  the whole prefix passed, worst `0.685`, median tol/|attnRef| `1.5%`; Gemma 3's local layers,
  window 1023, 16/8 heads, head dim 256, an 8K chunk over window prefixes, worst `0.600`; windows
  64, 1 and 4096 and head dim 64, worst `0.802`); target verifications (`int32` `qo_indptr`, random
  tree masks: 64 draft tokens at Llama-3-8B's heads, worst `0.784`; 4 tokens at 28/4; 8 tokens at
  head dim 256; Gemma 3's local layers with window 1023 and the windows' mask offsets, worst
  `0.615`; window 60 across 64-token blocks, `0.702`); `Inst.ok` edges (head dims 9 and 200,
  `sm_scale = 0.5`, worst `0.577`; requests with no new tokens, a grid longer than the longest
  request); `Pre` edges (scores scaled 36×, `|v| ≈ 10^30`, `|v| ≈ 10^-20`, subnormal `v`, uniform
  attention, and the adversarial rounding inputs, worst `0.980`). Worst error / tolerance `0.980`
  (margin `1.02×`); largest `tolRel` met `3.9·10^-2`; every written element finite, every input
  unchanged, every element of `o_extend` outside the requests' new tokens keeps its initial bits.
  Head dims 1 and 8 are rejected on the host as expected. On zero-mean `randn` values, where the
  softmax average cancels, the bound is a median `10%`–`22%` of `|attnRef|` (it is `~1%` of
  `attnAbs`). Negative tests: the outputs without the prefix, without the causal mask, with two
  heads swapped, with one element off by `0.25`, with a NaN, and with a write outside the requests
  are all rejected; with a window, the output without it (`7`–`67` times the bound), while a window off by one
  at `W = 1023` passes on the unit test's `N(0.1, 0.2)` values (`0.86` of the bound); with the
  custom mask, the causal output in place of the tree's (`1.9·10^3`–`6.6·10^3`), the output without
  the prefix (`1.6·10^3`), and with a window too the output without the window (`7.1`–`2.6·10^3`)
  or with the mask read without the window offset (`2.8·10^3`); at 128K the all-zero output and the
  output without the prefix's first quarter are rejected; at `2^18` the all-zero output passes on
  all but the tail (`524,125` of the `524,288` elements are within the bound) and is rejected only
  by it (`1.3`); the real check there is the shifted-prefix test (`6.0`–`6.6`), and the output
  without the prefix's first quarter is rejected on the shifted input (on random values at `2^18`
  it is within the bound, `0.83`: see Tolerances).
