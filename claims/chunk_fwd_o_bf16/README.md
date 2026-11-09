# `chunk_fwd_o_bf16`: the prepared claim

Kernel: `source/chunk_o.py`, `chunk_fwd_kernel_o` (lines 83–181), with the helpers
`unflatten_program_id` (lines 69–76) and `exp2` (lines 79–80); the kernel lines are 65–66, 69–76,
79–80 and 83–181, the host lines 184–355. Claim: [`Claim.lean`](Claim.lean), namespace
`Claims.ChunkFwdO`, `Statement := ∀ K P, Assumptions K P → Spec K`. Model map:
[`model-map.json`](model-map.json). GPU check: [`check_gpu.py`](check_gpu.py).

The covered path is `SETUP.md`'s: `bfloat16` `q`, `k`, `v`, `h` and output, `float32` gates `g`,
or `float32` per-head decays `g_gamma`, or neither; `state_v_first` either way, `BT = 64`, the
configurations `BK = BV = 32`, `64` and `128` of the source's list (the host's argument `BK`), fixed-length or
packed variable-length batches.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.ChunkFwdO.main : Statement` (`Statement`:
`Claim.lean` line 581). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 324), and every initial memory `init` that `Pre` accepts (line 396), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 553).

**What `Post` asks** (line 553), two parts:
1. for every chunk `r < nRows d` (line 317), token `t` with `rowLo d r ≤ t < rowHi d r` (lines
   426–431), value head `j < HV` and column `e < V`: element `oIdx d t j e` (line 449) of `o` reads
   in `fin` as a `bfloat16` `c`, for every such element. Only when `InRange d r t j e` holds (line
   532: the values the element reads finite, the gates bounded, the magnitudes below `2^100`) is
   more asked: `c` is finite and `|c.toReal − ref d r t j e| ≤ tol d r t j e`. `ref` (line 488) is
   exact, in the reals, from `Inst`'s contents (`qv`, `kv`, `vv`, `hv`, `gv`, `ggv`, tied to memory
   by `Pre`'s `Buf.Holds`): `sig·(gate·QS + Σ_{s ∈ upTo} Aqk·wgt·v)` (`QS`, `Aqk`, `gate`, `wgt`,
   `upTo`, lines 455, 461, 473, 476, 484). `tol` is defined at line 520 (with `Mabs`, `Mag`, `Flo`,
   `gamErr`, lines 494–512). Outside `InRange` only "holds a `bfloat16` value" is asked;
2. the frame: every byte outside `o.elems (written d)` (line 546) is unchanged.

**What you are given** (`Assumptions`, line 563), five fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotBF16`, a `DotLiftAcc`, `relDot16 K`, `absDot16 K` relative to the lifted
  magnitude): the four `tl.dot`s (model lines 175 or 185, 187, 233);
- `dot_total` (`Rules.Num.DotTotal` at `cfgBF16`, line 57): they have a result;
- `exp2_acc` (`Rules.Num.Exp2F32`, `2^-22` relative, `2^-126` absolute): the gates' `tl.exp2`
  (model lines 198, 200, 212, 215); `exp2_total` (`Rules.Num.MathTotal`): a result.

**The kernel, as the model runs it** (`chunk_fwd_kernel_o` line 74, `model` line 405, `launch`
line 416): a 2-D grid `(grid0 d, B·HV, 1)` (line 320). Program `(pid, i_bh)` splits `pid` into the
column tile `i_v` and the chunk `i_t`, and `i_bh` into `i_b`, `i_h` (`int64` `tl.irem`,
`tl.idiv`). On variable length it reads row `i_t` of `chunk_indices` and `cu_seqlens[i_n]`,
`cu_seqlens[i_n + 1]` (`tl.load1`); on fixed length `i_tg = i_b·NT + i_t`, `bos = i_b·T`. A
`tl.forRange` over `cdiv(K, BK)` trips loads masked tiles of `q`, `k` and the state `h[i_tg]` and
accumulates `b_o = q·h` and `b_A = q·k` (`tl.dot` with accumulator); it writes nothing. Then the
gates (with `g`: `tl.exp2`, `tl.sub`, `tl.mul`; with `g_gamma`: `tl.load1`, `tl.convert`,
`tl.subc`), the causal mask (`tl.where`, `+0`), the value tile, the output
`tl.addc (b_o·scale) (tl.dot(b_A.to(bf16), v)·scale)`, the rounding (`tl.cvt`) and the masked
store of the `[64, BV]` tile of `o`. Programs write disjoint tiles and use no atomics: the route is
`TVA.meets_independent`; the loop is `safe_forRange` with an invariant and `fun _ => ∅` as its
write footprints (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (the `Launch` default; `Launch.Valid` from
  `grid0_lt`, `grid1_lt`, `size_lt`, and the block shapes from `bk_ok`);
- each program is `Safe`: the divisors `cdiv(V, BV)`, `HV`, `HV / H`, `64` are nonzero (`V_pos`,
  `HV_pos`, `hv_dvd`); the table reads are in bounds (`ci_len`, `cu_len`, `N_lt`, `TablesOk`); the
  `int32` values (`T + 63` with `T_wrap`, `o_k`, `o_k·V`) and the `int64` offsets do not wrap
  (`T_lt`, `K_le`, `V_le`); every enabled lane of every load and store is in bounds (`q_len` …
  `gg_len`; on variable length `cu_seqlens` nondecreasing and `≤ T`); `safe_dot` (with
  `dot_total`), `safe_math` (with `exp2_total`), `safe_mul`, `safe_sub`, `safe_subc`, `safe_addc`,
  `safe_convert`, `safe_cvt`, `safe_load`, `safe_load1`, `safe_store`;
- the programs' write tiles are disjoint (fixed length: distinct `(i_t, i_b, i_h, i_v)`; variable
  length: distinct rows, `TablesOk`), no program writes what any program reads (`o_q`, `o_k`,
  `o_v`, `o_h`, `o_g`, `o_gg`, `o_cu`, `o_ci`), and every element of `written d` is written (every
  chunk has a row on variable length);
- the value part of part 1 under `InRange`, steps 1–7 of README `## Tolerances`: the `K` loop's
  dots (`dot`), the gates (`exp2_acc`, `FloatV.sub_err`, `FloatV.mul_err`, `FloatV.AddRel.err`),
  the scores' rounding (`FloatV.cvt_err`), the last dot, the scale and `tl.addc`, the store;
- the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok`,
`satInst_ok`, the `sc*_ok` facts and the `setup_*_ok` theorems; `bytes_disj` (line 637),
`zero_valid` (line 729) and `zero_holds_bf16` (line 733) can be reused.

## Specification

Tokens are numbered on the flat axis `b·T + t` of `[B, T]`. The chunks of the batch are its rows
`r`: on a fixed-length batch `r = b·NT + c` (`NT = cdiv(T, 64)`) is chunk `c` of row `b`, tokens
`[b·T + 64c, min(b·T + 64c + 64, b·T + T))`; on a variable-length batch row `r` of
`chunk_indices` names chunk `c = chunk_indices[r, 1]` of sequence `n = chunk_indices[r, 0]`,
tokens `[cu[n] + 64c, min(cu[n] + 64c + 64, cu[n+1]))` (`rowLo r`, `rowHi r`). Chunk `r`'s state
is `S_r = h[r]` (all of `h`'s leading `[B, NT]` or `[1, NT]` flattened), `S_r[j, x, e]` the element
`h[r, j, x, e]`, or `h[r, j, e, x]` with `state_v_first` (`hIdx`). Value head `j` reads query/key
head `qh j = j // (HV // H)`. For a token `t` of chunk `r`, value head `j`, column `e`, with
`σ = scale` (the double), the gates `G = 2^g[t, j]`, `w_s = 2^(g[t, j] − g[s, j])` with `g`;
with `g_gamma`, `G = 2^(γ_j·(i+1))`, `w_s = 2^(γ_j·(t − s))`, `γ_j = g_gamma[j]` and `i = t − rowLo r`
the local position (`gate`, `wgt`); both `1` with neither; and `s` ranging over the chunk's tokens
up to `t` (`upTo r t = [rowLo r, t]`):

* `QS = Σ_{x<K} q[t, qh j, x]·S_r[j, x, e]`, `A_s = Σ_{x<K} q[t, qh j, x]·k[s, qh j, x]`;
* `ref = σ·(G·QS + Σ_s A_s·w_s·v[s, j, e])`, the exact output;
* `PQS = Σ_x L(q)·L(S_r)`, `PA_s = Σ_x L(q)·L(k_s)` with `L = Rules.Num.liftMag .bf16` (a
  subnormal counted at `2^-126`);
* `Mabs = Σ_s |A_s|·w_s·|v_s|`, `Mag = G·PQS + Σ_s w_s·PA_s·L(v_s)`,
  `Flo = 1 + G + PQS + Σ_s (1 + w_s + PA_s)·L(v_s)`;
* `tol = 2^-8·|ref| + (1 + 2^-8)·σ·(2^-8·Mabs + (2^-13 + Γ)·Mag + 2^-100·Flo) + 2^-130`, with
  `Γ = 2^-15·|γ_j|` with `g_gamma` and `0` otherwise (`gamErr`).

`Post`: for every chunk `r`, token `t` in it, `j < HV`, `e < V`, `o[t, j, e]` holds a `bfloat16`
value `c`; when the element is `InRange`, `c` is finite and `|c − ref| ≤ tol`. Every byte outside
those elements is unchanged (the inputs, `o`'s other elements: on a variable-length batch, the
tokens after `cu[N]`).

`InRange` (per element): the query row `q[t, qh j, :]` and the state column `S_r[j, :, e]` are
finite, the keys `k[s, qh j, :]` for `s ≤ t` are finite, the value column `v[s, j, e]` is finite
over the whole chunk (the kernel multiplies the masked scores, zeros, by every `v[s]` of the
chunk), and with gates `g[s, j]` is finite with `|g[s, j]| ≤ 2^64` for `s ≤ t`, `g[t, j] ≤ 64` and
`g[t, j] − g[s, j] ≤ 64`, and with `g_gamma` `γ_j` is finite with `|γ_j| ≤ 64` and `γ_j ≤ 1`; and `PQS`, `G·PQS`, each `PA_s`, `w_s·PA_s`, `Mag + Flo` and
`σ·(Mag + Flo)` are at most `2^100`. Gated DeltaNet's gates (`g[t] ≤ g[s] ≤ 0` within a chunk:
cumulative sums of nonpositive log decays), fla's tests' (`0.1·randn`), and RetNet's and lightning
attention's decays (`γ` in `[−0.05, 0]` and `[−11.6, 0]`) are inside.

The tolerance is meaningful: its main terms are the kernel's two `bfloat16` roundings. On the GPU
check its median over the checked elements is `0.008` to `0.030` of `|ref|` in each configuration (2 to 8 times the
store's `2^-8`), and the error reaches `0.906` of it; the store's rounding alone can reach
`2^-8·|ref|` (see Tolerances), so the leading term cannot be smaller.

## Conditions

`Inst`: `varlen`, `B`, `T`, `H`, `HV`, `K`, `V`, `useG`, `useGamma`, `svf`, the configuration `bk`
(`BK = BV = bk`), the double `scale`, the
storages of `q`, `k`, `v`, `h`, `g`, `g_gamma`, `o` and their contents, and on a variable-length batch the table dtype
`idxD`, `N`, `cu_seqlens`, `NT = len(chunk_indices)`, `chunk_indices` and their entries.
`Inst.ok`:

* `bk ∈ {32, 64, 128}`: the source's list on Hopper (with 2, 4, 8 warps), whichever the autotuner
  picks (SETUP.md);
* `B, H, HV ≥ 1`, `H ∣ HV`; `1 ≤ K ≤ 256`, `1 ≤ V ≤ 512` (SETUP.md);
* `T < 2^31`: `T` is an `int32` argument; on a fixed-length batch of two rows or more also
  `T + 63 < 2^31`, so line 119's `int32` `tl.cdiv(T, 64)` does not wrap (a limit of the kernel,
  Exclusions; with one row `i_b = 0` makes `NT` unused, and a variable-length launch does not run
  line 119);
* at most one of `g`, `g_gamma` (`one_gate`);
* variable length: `B = 1`;
* the grid `(cdiv(V, BV)·NT, B·HV)` within CUDA's limits (`B·HV < 2^16`: a limit of the kernel on
  fixed-length batches, Exclusions) and fewer than `2^31` programs;
* each storage holds its tensor (`g_gamma` at least `HV` entries); `o` shares no byte with `q`,
  `k`, `v`, `h`, `g`, `g_gamma`, the tables;
* variable length: `int32` or `int64` tables, `N < 2^31`, the tables' lengths;
* `scale` a finite double in `[2^-64, 2^64]` (by default `K^-0.5 ∈ [1/16, 1]`).

No bound on the number of tokens is needed beyond `T`'s: every offset the kernel computes from a
token, a head or a chunk is `int64` (`i_bh` and `pid` converted to `int64`, `bos`, `eos` and the chunk
index loaded as `int64`, `o_t` and `o_v` `int64`); only `o_k` (at most 255) and `o_k·V` are `int32`.

`Pre`: the buffers `Valid` and holding the instance's contents (`g`, `g_gamma` when given); on a
variable-length batch `TablesOk`: `cu_seqlens` nonnegative, nondecreasing and at most `T`, each
row of `chunk_indices` a chunk that starts inside its sequence, no two rows equal, every chunk of
every sequence a row (the table `prepare_chunk_indices` builds, in any row order). The values are
covered element by element (`InRange`).

The launch: grid `(cdiv(V, BV)·NT, B·HV)`, the block shapes `[64]`, `[bk]`, `[64, 64]`, `[64, bk]`,
`[bk, 64]`, `[bk, bk]`, `[64, 1]`, `[1, 64]`, `[bk, 1]`, `[1, bk]`, `fusion = true`, `debug = false`,
`num_ctas = 1`, no `launch_pdl`; `num_warps` 2, 4 or 8 (by `bk`), `num_stages = 3` and no other options; no heuristics or autotuning, a `JITFunction` launch with a
grid function, no descriptors: within part (A) of `Compiler.Faithful`'s
scope. The claim relies on part (B) (the environment) as the rule states it.

## Assumptions

`[rule]` fields (`docs/rules/num.md`'s table): `faithful` (`Compiler.Faithful`), `dot`
(`DotBF16`), `dot_total` (`DotTotal P cfgBF16`), `exp2_acc` (`Exp2F32`), `exp2_total`
(`MathTotal P .exp2 .binary32`). The gates are finite on every in-range element, so
`Exp2F32NegInf` is not needed; an out-of-range `exp2` (a NaN gate, an overflow in the masked upper
triangle) needs only a result (`exp2_total`), which `tl.where` discards or which reaches only
out-of-range elements. `consistent` uses `Platform.sm90ref`.

## Model notes

- `unflatten_program_id` inlined (`Y = None`: the 2-D branch), `exp2` inlined (`.to(tl.float32)`
  of a `float32` tensor is no conversion).
- `if IS_VARLEN:` is a compile-time `if`; the model's Lean `if` returns `(i_tg, i_t, bos, T)` from
  each branch, `T` the `int64` value line 133 compares with (`eos − bos`, or the argument promoted
  to `int64`); line 117's `NT` (varlen) and line 121's `eos` (fixed length) are dead.
- The two compile-time `if STATE_V_FIRST:` (lines 140–145 and 153–156) are one Lean `if`, which
  computes the state's pointers and mask (pure) after the loads of `q` and `k` and before the
  state's load, as the source orders the loads.
- `b_o * scale + tl.dot(...) * scale` is `tl.addc` of two IEEE products (fusable); the dot's zero
  accumulator is the default (no `dot_add`: its result is multiplied, not added).
- `b_o * exp2(b_g)[:, None]` and `b_A * exp2(...)` are IEEE products (they feed products and
  `tl.where`, not additions); with `g`, `b_g[:, None] − b_g[None, :]` is the IEEE `−` (loaded
  values); with `g_gamma`, `b_g = b_gamma * (tl.arange(0, BT) + 1)` is the IEEE product of the
  scalar and the `int32` positions converted to `float32` (exact), and the `−` it feeds is fusable
  (`tl.subc`).
- `USE_G` and `USE_G_GAMMA` are two compile-time `if`s in sequence; `Inst.ok` takes at most one
  true.
- `BK` and `BV` are the model's parameters, both `d.bk`; `BT = 64` is a constant. Every tile's
  shape follows the source (`[BT, BK]`, `[BK, BT]`, `[BK, BV]` or `[BV, BK]`, `[BT, BV]`, `[BT, BT]`).

## Coverage

`mk vl useG useGm svf B T H HV K V sbits dt N NT bk` is the host's launch (contiguous tensors,
`o = empty_like(v)`, `bk = 64` by default); `mk_ok` proves `Inst.ok` for every `bk ∈ {32, 64, 128}`,
`B, H, HV ≥ 1` with `H ∣ HV`,
`1 ≤ K ≤ 256`, `1 ≤ V ≤ 512`, `T + 63 < 2^31`, at most one gate, `B = 1` on variable length, the grid within its
limits, each tensor at most `2^40` bytes, `int32` or `int64` tables and a scale in range.
SETUP.md's launches:

| Launch | `Inst` |
|---|---|
| fla's `test_chunk_fwd_o`, `bfloat16`: `(2, 128, 2, 2, 64)` gated, `(2, 128, 2, 4, 64)` gated, `(2, 128, 2, 2, 64)` ungated | `setup_test_g`, `setup_test_gva`, `setup_test_nog` |
| fla's deterministic test: packed `[0, 5000, 8192]`, `int32`, 8 heads, 128 | `setup_det_varlen` |
| Qwen3-Next's heads (16/32, 128), `state_v_first`, packed prefill of 8 × 4096, `int32` | `setup_qwen3next` |
| the same heads, `2^20` tokens in 256 sequences | `setup_qwen3next_1m` |
| fla's gated DeltaNet defaults (6 heads, `K = 256`, `V = 512`), `state_v_first`, packed, `int64` | `setup_gdn_default` |
| benchmarks: `B1_T8192_H96_D128` gated, `B4_T4096_H64_D128` ungated (the delta rule), `B8_T2048_H32_D256` gated | `setup_bench_h96`, `setup_bench_h64_nog`, `setup_bench_d256` |
| head dims 60 and 100 (`scale = 60^-0.5`) | `setup_dims_60_100` |
| RetNet's `g_gamma` at fla's retention defaults (8 heads, `K = 128`, `V = 256`): 2 × 4096 tokens; packed `[0, 1000, 3000, 8000, 8192]`, `int64` | `setup_retnet`, `setup_retnet_varlen` |
| lightning attention's `g_gamma` (`layer_idx = 0`, 12 layers): `B1_T8192_H96_D128`, `B4_T2048_H16_D128` | `setup_lightning_h96`, `setup_lightning_h16` |
| the largest fixed-length batch below the grid limit at 32 value heads: `B = 2047` | `setup_grid_edge` |
| `BK = BV = 32` (2 warps): fla's test `(2, 128, 2, 2, 64)` gated, the deterministic test, head dims 60/100 | `setup_test_g_bk32`, `setup_det_varlen_bk32`, `setup_dims_60_100_bk32` |
| `BK = BV = 128` (8 warps): the benchmarks `B1_T8192_H96_D128`, `B4_T4096_H64_D128` ungated, `B8_T2048_H32_D256`; Qwen3-Next's heads at 8 × 4096 and `2^20` tokens; the gated DeltaNet defaults; head dims 60/100; RetNet; lightning attention at `B1_T8192_H96_D128` | `setup_bench_h96_bk128`, `setup_bench_h64_nog_bk128`, `setup_bench_d256_bk128`, `setup_qwen3next_bk128`, `setup_qwen3next_1m_bk128`, `setup_gdn_default_bk128`, `setup_dims_60_100_bk128`, `setup_retnet_bk128`, `setup_lightning_h96_bk128` |

The contents (and so `TablesOk`, `InRange`) are not part of `Inst.ok`; `check_gpu.py` checks
them.

## Tolerances

Derived from the library's facts (`docs/rules/num.md`; `u = 2^-24`, `η = 2^-150` for `float32`;
`2^-8`, `2^-134` for `bfloat16`, `Format.bfloat16_u`, `Format.bfloat16_eta`). Fix an in-range
element `(r, t, j, e)` and the configuration `BK = BV = bk`; `n_K = cdiv(K, bk) ≤ 256/bk`
(`K ≤ 256`); `ρ = relDot16 bk = (bk + 2)·2^-22`, `α = absDot16 bk = (bk + 2)·2^-126` for the two
accumulating dots (`K`-dimension `BK`), `ρ_64 = relDot16 64 = 66·2^-22`, `α_64 = 66·2^-126` for the
last one (`K`-dimension `BT = 64`); a masked lane contributes `0`, `liftMag 0 = 0`.

1. **The two accumulating dots** (lines 154/156 and 158, `n_K` steps, `DotBF16`): each step adds
   at most `ρ·(|acc| + Σ L·L) + α`; by induction the accumulated error after `n_K` steps is at most
   `((1+ρ)^{n_K} − 1)·(P + α/ρ)` (`P = PQS` or `PA_s`; room: everything is below `2^101`;
   `α/ρ = 2^-104`). Per configuration, at the largest `n_K`:

   | `bk` | `ρ` | `n_K ≤` | `(1+ρ)^{n_K} − 1` |
   |---|---|---|---|
   | 32 | `34·2^-22 = 8.11·10^-6` | 8 | `6.49·10^-5` |
   | 64 | `66·2^-22 = 1.57·10^-5` | 4 | `6.30·10^-5` |
   | 128 | `130·2^-22 = 3.10·10^-5` | 2 | `6.20·10^-5` |

   So, with `δ = 6.49·10^-5` for every configuration, `|ô − QS| ≤ δ·PQS + 2^-117`,
   `|Â_s − A_s| ≤ δ·PA_s + 2^-117`.
2. **The gates** (lines 164–165 with `g`, `Exp2F32`: `2^-22` relative, `2^-126` absolute, on every finite
   input with room). `Ĝ = exp2(g[t])`: `|Ĝ − G| ≤ 2^-22·G + 2^-126` (`g[t] ≤ 64`).
   `ŵ_s = exp2(fl(g[t] − g[s]))`, the IEEE `−` (`|fl(Δ) − Δ| ≤ u·|Δ|`): for `Δ ∈ [−126, 64]`,
   `2^{fl(Δ)} = w_s·2^{εΔ}` with `|εΔ| ≤ 126u`, so `|ŵ_s − w_s| ≤ (2^{126u} − 1 + 2^-22·(1+…))·w_s
   + 2^-126 ≤ 2^-17·w_s + 2^-126`; for `Δ < −126`, `fl(Δ) ≤ −126` (rounding is monotone) and both
   `ŵ_s`, `w_s` are below `2^-124`. So `|ŵ_s − w_s| ≤ 2^-17·w_s + 2^-124` (`|g| ≤ 2^64` keeps
   `fl(Δ)` finite).

   With `g_gamma` (lines 167–170): `b_i = fl(γ·(i+1))`, the IEEE product of `γ = γ_j` and the
   exact `float32` `i + 1 ≤ 64`, so `|b_i − γ(i+1)| ≤ 64u·|γ| + η = 2^-18·|γ| + η`, and
   `Ĝ = exp2(b_i)` is within `(2^{2^-18|γ| + η}(1 + 2^-22) − 1)·G + 2^-126` of `G`. The score's
   exponent is `tl.subc(b_i, b_i')` (`AddRel`: the rounding of `p − q` for some `p`, `q` that round
   to `b_i`, `b_i'`; the IEEE `−` is one such): `|p − γ(i+1)| ≤ |p − b_i| + |b_i − γ(i+1)| ≤
   2·64u·|γ|(1 + u) + 2η`, the same for `q`, and the final rounding adds `u·|p − q| ≤ 64u·|γ|(1 + …)`,
   so the exponent is within `x = 2.5·2^-17·|γ| + 5η` of `γ·(i − i') = log2 w_s`, and
   `|ŵ_s − w_s| ≤ (2^x(1 + 2^-22) − 1)·w_s + 2^-126`. With `|γ| ≤ 64`, `x ≤ 2^-9.6` and
   `2^x − 1 ≤ 0.70·x`: the gates' relative errors are at most `0.70·2^-18·|γ|` (inter) and
   `0.70·2.5·2^-17·|γ| = 1.75·2^-17·|γ|` (intra) beyond the `2^-22` and floors of the `g` case,
   and the `2^-17·w_s` of the `g` case's `−` does not occur. Carried through steps 3–6 (each such
   relative error is multiplied by at most `(1 + 2^-8)(1 + 10^-3)`), they add at most
   `1.76·2^-17·|γ| ≤ 2^-15·|γ| = Γ` to `c` (margin 2.3). `γ ≤ 1` keeps the exponents below 64,
   and `Δ < −126` is handled as in the `g` case (the exponent's error is below `2^-9`).
3. **The gated values.** `õ = fl(ô·Ĝ)`: `|õ − G·QS| ≤ (δ + 2^-22 + u + …)·G·PQS + floors
   ≤ 6.52·10^-5·G·PQS + floors`. `Ã_s = fl(Â_s·ŵ_s)`: `|Ã_s − A_s·w_s| ≤ (δ·(1+2^-17) +
   2^-17 + u·(1+…))·w_s·PA_s + floors ≤ 7.26·10^-5·w_s·PA_s + floors`. Without gates, `õ = ô`,
   `Ã_s = Â_s`.
4. **The scores' rounding** (`b_A.to(bfloat16)`, nearest even): `|Ã16_s − Ã_s| ≤ 2^-8·|Ã_s| +
   2^-134 ≤ 2^-8·(|A_s|·w_s + 7.26·10^-5·w_s·PA_s) + floors`; masked scores are `+0`.
5. **The last dot** (`DotBF16`, `K = BT = 64` in every configuration):
   `|D − Σ_s Ã16_s·v_s| ≤ ρ_64·Σ_s L(Ã16_s)·L(v_s) + α_64`
   with `L(Ã16_s) ≤ |Ã16_s| + 2^-126 ≤ (1 + 2^-8)(1 + 7.3·10^-5)·w_s·PA_s + …`. So
   `|D − Σ_s A_s·w_s·v_s| ≤ 2^-8·Mabs + (7.26·10^-5·(1 + 2^-8) + ρ_64·(1 + 2^-8)(1 + 7.3·10^-5))
   ·Σ_s w_s·PA_s·L(v_s) + floors ≤ 2^-8·Mabs + 8.87·10^-5·Σ_s w_s·PA_s·L(v_s) + floors`.
6. **The scale and the addition.** `σ32 = hostF32(σ)`: `|σ32 − σ| ≤ u·σ`. `tl.addc` of
   `fl(õ·σ32)` and `fl(D·σ32)` (`AddRel`, fused or not: three roundings at most on either side):
   within `3u·(1+…)·σ32·(|õ| + |D|) + 2η` of `σ32·(õ + D)`, and `|õ| + |D| ≤ (1 + 2^-7)·Mag +
   floors`. With step 5 and `|σ32 − σ|·|G·QS + Σ_s A_s·w_s·v_s| ≤ u·σ·Mag`, the value `x` before
   the store has
   `|x − ref| ≤ E = σ·(2^-8·Mabs + c·Mag + f) + 2^-147` with `c ≤ 8.90·10^-5 ≤ 2^-13 = 1.22·10^-4`
   in every configuration
   (with `g_gamma`, `c ≤ 2^-13 + Γ`; inter- and intra-chunk parts each below it; `Mabs ≤ Mag` absorbs `2^-8·u·Mabs`) and the floors
   `f ≤ 2^-110·Flo`: the dots' `α` and `2^-117` terms, `exp2`'s `2^-126`/`2^-124` times `PQS`,
   `PA_s·L(v_s)` or the gates, the roundings' `η`s and `2^-134`, each a constant below `2^-110`
   times a term of `Flo = 1 + G + PQS + Σ_s (1 + w_s + PA_s)·L(v_s)`.
7. **The store** (nearest even into `bfloat16`): `|c − x| ≤ 2^-8·|x| + 2^-134 ≤ 2^-8·(|ref| + E) +
   2^-134`, so `|c − ref| ≤ 2^-8·|ref| + (1 + 2^-8)·E + 2^-134 ≤ tol` (`2^-100` for the floors'
   `2^-110`, `2^-130` for the absolute terms). `c` is finite: `|x| ≤ 2^101` (room).

Room: `InRange` bounds `PQS`, `G·PQS`, `PA_s`, `w_s·PA_s`, `Mag + Flo` and `σ·(Mag + Flo)` by
`2^100`, so every dot's `L + ρL + α`, every product and the result stay below `float32`'s and
`bfloat16`'s `2^128`, and every `MathAcc` input has room.

The store's rounding alone reaches `2^-8/(1 + 2^-8)·|x|` (at the midpoint `2^k(1 + 2^-8)`), so
the leading `2^-8·|ref|` cannot be smaller; the GPU check's worst element is at `0.906` of `tol`.
The `2^-8·Mabs` term (the scores' roundings, all aligned in the worst case) and `(2^-13 + Γ)·Mag`
make the rest; with them the median tolerance is `0.008` to `0.030` of `|ref|` on the checked
launches.

## Exclusions

- **Fixed-length batches with `B·HV ≥ 65536`** — a limit of the kernel: the host puts `B·HV` on
  grid axis 1, which CUDA limits to `65535`, and the launch fails on the host
  (`Triton Error [CUDA]: invalid argument`). The first failing launch: `B = 2048` rows at 32 value
  heads (Qwen3-Next's), any `T`; `check_gpu.py` shows `B = 2048` failing and `B = 2047` running.
  Real use reaches it with padded batches of 2048 or more sequences at 32 value heads (1M tokens
  at `T = 512`); packed (variable-length) batches have `B = 1` and do not.
- **`2^31 − 63 ≤ T < 2^31` on a fixed-length batch of two rows or more** — a limit of the kernel:
  line 119's `int32` `T + 63` wraps, so `NT` is negative and rows `b ≥ 1` read the wrong chunk of
  `h`. The first failing launch: `B = 2`, `T = 2^31 − 63`. With one row (`i_b = 0`) or on a
  variable-length launch the wrap does not matter, and these are covered. Real use does not reach
  it (rows of `2^31` tokens). **`T ≥ 2^31`** — the argument is an `int64` (another compiled
  kernel): outside real use.
- **A grid of `2^31` programs or more** (`cdiv(V, BV)·NT·B·HV ≥ 2^31`) — a limit of the launch
  (Triton's launcher skips it): e.g. `2^22` chunks (`2^28` tokens) at 64 value heads,
  `V = 512` and `BV = 64` (`2^21` chunks at `BV = 32`). Real use does not reach it
  (`setup_qwen3next_1m` has `2^20` programs, `2^21` at `BV = 32`).
- **Elements not `InRange`** — per element: non-finite inputs among those it reads, gates beyond
  the bounds, or magnitudes above `2^100` (`float32` overflows near `2^128`). Such an element is
  still a `bfloat16` value; real use (normalised `q`, `k`, nonpositive cumulative gates) is far
  inside.
- **Variable-length tables not `TablesOk`** or `B > 1` with `cu_seqlens` — outside real use (the
  callers build the table with `prepare_chunk_indices`; the gated delta rule and simple GLA assert
  `B = 1`).
- **Both `g` and `g_gamma`** — outside real use: simple GLA's interface says only one is given,
  and no caller passes both.
- **`g_gamma` with fewer than `HV` entries** — outside real use: `g_gamma[i_h]` is read for every
  value head (RetNet and lightning attention build `[H]` with `H = HV`).
- **`K > 256` or `V > 512`** — outside real use (head dims 32 to 256, value dims up to 512).
- **`chunk_size` 16 or 32** — not part of the setup: an opt-in keyword of the gated delta rule
  and simple GLA that no layer or model config passes (the default is 64); fla tests it only in
  `float32` and `float16` (`tests/ops/test_gdn.py:245-250`, `tests/ops/test_retention.py:70-79`).
- **`float16`/`float32` inputs** — not part of the setup.

## Library requests

None: the claim cites only facts the library states.

## Checks

- `Claim.lean` builds with no output (`lake env lean ../claims/chunk_fwd_o_bf16/Claim.lean`); no
  `sorry`, no `axiom`. `consistent`, `satisfiable` (one token, one head, `K = V = 1`, every byte
  `0`) and the twenty-eight `setup_*_ok` are proved.
- `check_gpu.py` (H200 NVL, Triton 3.8.0, peak 2.28 GiB): 52 launches pass (SETUP.md, "Checked on
  the H200"), 29 at `BK = BV = 64` and 23 at `32` or `128`; worst error / tolerance `0.906`; host
  and direct launch agree bit for bit; frame and inputs intact; nine wrong outputs rejected in each
  configuration; the grid limit fails at `B·HV = 65536`.
