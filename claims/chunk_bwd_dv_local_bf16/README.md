# `chunk_bwd_dv_local_bf16`: the prepared claim

fla's `chunk_bwd_kernel_dv_local` (`source/chunk_o.py`, kernel lines 74–149 with the helper `exp2`
70–71; host `chunk_bwd_dv_local` 191–241). Claim: [`Claim.lean`](Claim.lean), namespace
`Claims.ChunkBwdDvLocal`, `Statement := ∀ K P, Assumptions K P → Spec K`. Model map:
[`model-map.json`](model-map.json). GPU check: [`check_gpu.py`](check_gpu.py).

The covered path is `SETUP.md`'s: `bfloat16` `q`, `k`, `do`, `dv`, an optional `float32` gate `g`
(`USE_G` either way), `USE_G_GAMMA = USE_A = False`, fixed-length rows or a packed
variable-length batch (`IS_VARLEN` either way, `int32` or `int64` tables), on an H200
(`BK`, `BV` the host's `min(max(next_power_of_2(·), 16), 128)`).

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.ChunkBwdDvLocal.main : Statement` (`Statement`:
`Claim.lean` line 519). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 228), and every initial memory `init` that `Pre` accepts (line 471), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 489).

**What `Post` asks** (line 489), two parts:
1. for every sequence `n < nSeq d` (line 348), token `p` with `lo d n ≤ p < hi d n` (lines
   351–352), value head `hv < HV` and column `e < V`: element `oIdx d p hv e` (line 338) of `dv`
   reads in `fin` as a `bfloat16` `c`, `c` is finite, and
   `|c.toReal − ref d init n p hv e| ≤ tol d init n p hv e`. This holds for every such element (no
   guard: `Pre`'s `ValsOk`, line 455, covers the values). `ref` (line 409) is exact, in the reals:
   `Σ_{j ∈ [p, chunkEnd)} Aex·dov`, with `Aex` (line 398) `= sig·kq·wt` (lines 375, 378, 394) from
   `q`, `k`, `g`, `do` in `init`, and `chunkEnd` (line 357) from `cu_seqlens` or the rows. `tol` (line
   441) is `2^-7·Σ_j |Aex|·|dov| + 2^-13·Σ_j aMag·doMag + 2^-100` over the same `j`, with `aMag`
   (line 432) `= sig·(kqMag + 2^-94)·(wt + 2^-112) + 2^-86`;
2. the frame: every byte outside `dv.elems (written d)` (line 482) is unchanged.

**What you are given** (`Assumptions`, line 499), six fields:
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotBF16`, a `DotLiftAcc`): each `tl.dot` (model lines 130, 156) is within
  `relDot16 K` times the lifted magnitude plus `absDot16 K` of the exact sum, with room;
- `dot_total` (`Rules.Num.DotTotal` at `cfgBF16`, line 47): every `tl.dot` has a result;
- `exp2` (`Rules.Num.Exp2F32`, `2^-22` relative, `2^-126` absolute): the gate, model line 137;
- `exp2_ninf` (`Rules.Num.Exp2F32NegInf`): `exp2(−∞) = +0`, for a gate difference that overflows;
- `exp2_total` (`Rules.Num.MathTotal`): `exp2` has a result.

**The kernel, as the model runs it** (`chunk_bwd_kernel_dv_local` line 62, `model` line 291,
`launch` line 302): a 2-D grid `(nChunks d, nRows d, 1)` (lines 222, 225). Program `(i_t, i_bh)`
splits `i_bh` into `i_b`, `i_h` (`int64` `tl.idiv`, `tl.irem` by `HV`). On variable length it reads
row `i_t` of `chunk_indices` and `cu_seqlens[i_n]`, `cu_seqlens[i_n + 1]` (`tl.load1`); on fixed
length `bos = i_b·T`. Tokens `o_t = i_t·BT + [0, BT)` are masked by `o_t < T`. With a gate it loads
`b_g`. A first `tl.forRange` over `cdiv(K, BK)` trips loads masked `[BT, BK]` and `[BK, BT]` tiles
of `k`, `q` and accumulates `b_A` by `tl.dot` then `tl.mul_add` with `scale`; it writes nothing.
Then the gate (`tl.sub`, `tl.exp2`, `tl.mul`), the causal mask `m_A` (`tl.where`, `+0`) and the
rounding to `bfloat16` (`tl.cvt`). A second `tl.forRange` over `cdiv(V, BV)` trips loads a masked
`[BT, BV]` tile of `do`, computes `tl.dot(b_A, b_do)`, rounds it and stores the masked tile of
`dv`. Programs write disjoint tiles and use no atomics: the route is `TVA.meets_independent`;
both loops are `safe_forRange` with an invariant, the second with per-trip write footprints (the
column tiles), disjoint across trips (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (the `Launch` default; `Launch.Valid` from
  `grid0_lt`, `grid1_lt`, `grid_lt`, and the block shapes from `bt_ok`, `bk_*`, `bv_*`);
- each program is `Safe`: the divisors `HV` and `HV / H` are nonzero (`HV_pos`, `hv_dvd`); the
  table reads are in bounds (`ci_len`, `cu_len`, `N_lt`, `TablesOk`); the `int64` offsets and the
  `int32` `o_k`, `o_v` do not wrap (`T_lt`, `T_ctx`, `K_le`, `V_le`, `hvv_lt`; with `T ≤ 2^22`,
  every token index and chunk count is far below `2^31`); every enabled lane of every
  load and store is in bounds (`q_len`, `k_len`, `dO_len`, `dv_len`, `g_len`, and on variable
  length `TablesOk`'s `cu_seqlens` nondecreasing and `≤ T`); `safe_dot` (with `dot_total`),
  `safe_mul_add`, `safe_sub`, `safe_math` (with `exp2_total`), `safe_mul`, `safe_cvt`,
  `safe_convert`, `safe_load`, `safe_store`;
- the programs' write tiles are disjoint: on fixed length from `(i_t, i_bh)`; on variable length
  from `TablesOk` (distinct rows); no program writes what any program reads (`dv_q`, `dv_k`,
  `dv_dO`, `dv_g`, `dv_cu`, `dv_ci`); and every element of `written d` is written (every chunk has
  a row, `TablesOk`; `nChunks` covers `T` on fixed length);
- the value part of part 1: derive the fine bound of README `## Tolerances`, steps 1–6 (the `K`
  loop's dots and multiply-adds, `dot`, `FloatV.fma_err` or `FloatV.mul_err` and
  `FloatV.add_err`; the gate, `FloatV.sub_err`, `exp2`, `exp2_ninf`; the entry within `eA`; the two
  `bfloat16` roundings, `FloatV.cvt_err`; the second dot, `dot`; with room from `ValsOk`), then
  relax it to `tol` by step 7 (`|ref| ≤ B1`, `wErr ≤ 2^-14·wt + 2^-125`, `eA ≤ 8·10^-5·aMag' + 2^-140`);
- the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `satInst_ok`,
`canon_ok`, `setup_var3_tables`, the `s*_ok` facts and the `setup_*_ok` theorems; `bytes_disj`
(line 565) can be reused.

## Specification

Sequences (`nSeq`, `lo`, `hi`) are the rows `[b·T, b·T + T)` of a fixed-length batch, or
`[cu_seqlens[n], cu_seqlens[n+1])` of a variable-length one, in the flat token axis; token `p`'s
chunk ends at `chunkEnd = min(chunkStart + BT, hi)`, chunks of `BT` tokens counted from the
sequence's start. For value head `hv` the query/key head is `h = hv // (HV // H)` (`hk`). With
`σ` the host's double `scale` (`sig`), exactly:

* `kq(p, j) = Σ_{c<K} k[p, h, c]·q[j, h, c]`, and `kqMag(p, j) = Σ_c liftMag k·liftMag q`
  (`Rules.Num.liftMag .bf16`: a subnormal counted at `2^-126`; `≥ |kq|`);
* `wt(p, j) = 2^{g[j, hv] − g[p, hv]}` with a gate, `1` without (`dg` the exponent);
* `Aex(p, j) = σ·kq·wt`, the entry of the causal-masked matrix;
* `ref(p, hv, e) = Σ_{j=p}^{chunkEnd−1} Aex(p, j)·do[j, hv, e]`.

The reference is computed from `cu_seqlens` (or the rows), `q`, `k`, `g` and `do` alone, never
from `chunk_indices` or the kernel's indexing.

`Post`: every `dv[p, hv, e]` (`p` a token of a sequence, `hv < HV`, `e < V`) is a finite
`bfloat16` `c` with `|c − ref| ≤ tol`, and every byte outside those elements is unchanged. Over
`j ∈ [p, chunkEnd)`:

* `aMag(p, j) = σ·(kqMag + 2^-94)·(wt + 2^-112) + 2^-86`: the magnitude of the products
  `σ·k·q·2^{dg}` that make entry `(p, j)`, with floors for the absolute errors of the `tl.dot`s
  (`2^-94`), of `exp2` (`2^-112`) and of the second `tl.dot` (`2^-86`), far below real values;
* `tol = 2^-7·Σ_j |Aex|·|do_j| + 2^-13·Σ_j aMag·liftMag do_j + 2^-100`.

The first term is the kernel's two `bfloat16` roundings, of each entry of `A` (line 140) and of the
stored result (line 149), relative to the magnitudes of `ref`'s terms; the second the `float32`
errors of the entries (the `K` loop's `tl.dot`s and multiply-adds, the gate) and the second
`tl.dot`, relative to the products' magnitudes (`kqMag`, not `|kq|`: a dot product's error does not
cancel with its terms). One magnitude for both would not do: on the host's `randn` inputs
`Σ_j σ·kqMag·wt·|do|` is about `sqrt(2K/π)` times `Σ_j |Aex·do|` (`kq` cancels: 6 to 13 times at
head dims 64 to 256), so charging the roundings to it would make `tol` that much looser, and
charging the `float32` errors to `Σ_j |Aex·do|` fails where `kq` cancels. On those inputs `tol` is
`2^-7·Σ_j |Aex·do|` plus a tenth to a fifth of that; the worst error on the H200 is `0.934` of
it.

## Conditions

`Inst`: `varlen`, `useG`, `B`, `T`, `H`, `HV`, `K`, `V`, `BT`, `BK`, `BV`, `scale` (double), the
storages `q`, `k`, `dO` (the source's `do`), `dv`, `g`, and for variable length the tables' dtype,
`N`, `cu_seqlens` (`cuVal`), `NT`, `chunk_indices` (`ciVal`).

`Inst.ok`:
* `BT ∈ {16, 32, 64}`; `BK`, `BV` powers of two in `[16, 128]`;
* `1 ≤ K ≤ 256`, `1 ≤ V ≤ 2^16`, `HV·V < 2^31` (every constexpr of the kernel an `int32`
  constant, see Exclusions); `H ≥ 1` divides `HV ≥ 1`; `B ≥ 1`, and `B = 1` on a variable-length
  launch;
* `T < 2^31` (the `int32` argument); `T` is a context length (the tokens of a row, or of the
  packed batch): **`T ≤ 2^22`** (`T_ctx`);
* `B`, the batch rows, is a batch size: **`B ≤ 8192`** (`B_le`); on variable length the number of
  sequences too: **`N ≤ 8192`** (`N_le`);
* the grid: `NT < 2^31`, **`B·HV < 2^16`** (a limit of the kernel, see Exclusions), fewer than
  `2^31` programs;
* `scale` a finite double in `[2^-32, 2^32]`;
* the buffers hold the contiguous tensors; `dv` (a fresh `empty_like(do)`) shares no byte with
  `q`, `k`, `do`, `g` and the tables;
* variable length: `int32` or `int64` tables, `N < 2^31`, the tables' lengths.

`Pre`: the buffers valid; on a variable-length launch the tables hold the instance's contents and
are `TablesOk` (`cu_seqlens` nonnegative, nondecreasing and at most `T`; each row of
`chunk_indices` a distinct (sequence, chunk) whose chunk starts inside its sequence; every such
chunk has a row: what `prepare_chunk_indices` builds, in any row order); `ValsOk`: in every
sequence, every `q`, `k`, `do` component finite, `|do| ≤ 2^100`, every `g` finite (with a gate);
every pair `p ≤ j` of one chunk with `g[j] − g[p] ≤ 64` (with a gate), `kqMag ≤ 2^100` and
`σ·kqMag·(1 + wUp) ≤ 2^100`; every output with `Σ_j σ·kqMag·wUp·liftMag do ≤ 2^120`.

The launch: grid `(NT, B·HV, 1)`, the source's block shapes, `fusion = true`, `debug = false`,
`num_ctas = 1`, no `launch_pdl`; `num_warps=4`, `num_stages=3` only, no heuristics or autotuning
(removed from the source), a `JITFunction` launch, no descriptors: within
part (A) of `Compiler.Faithful`'s scope. The claim relies on part (B) (the environment) as the rule
states it.

## Assumptions

`[rule]` fields (`docs/rules/num.md`'s table): `faithful` (`Compiler.Faithful`), `dot`
(`DotBF16`), `dot_total` (`DotTotal P cfgBF16`), `exp2` (`Exp2F32`), `exp2_ninf`
(`Exp2F32NegInf`), `exp2_total` (`MathTotal P .exp2 .binary32`). `consistent` uses
`Platform.sm90ref`.

## Model notes

`b_A += tl.dot(b_k, b_q) * scale` is `tl.dot` with a zero accumulator then `tl.mul_add` (the
product by `scale` feeds the addition; the dot itself feeds a `*`, so it is not the combine pass's
`dot_add`); `b_g[None, :] - b_g[:, None]` is an IEEE `tl.sub` (no product feeds it) and the gate's
`*` an IEEE `tl.mul`; every element offset is `int64` (`i_t`, `i_bh` converted to `int64` at line 99, `bos`,
`o_t`), except the `int32` `o_k` (below `K + BK ≤ 2^9`), `o_v` (below `V + BV ≤ 2^16 + 2^7`) and `i_n`; the varlen branch's two loads at
line 102 both read the program's own `i_t`.

## Coverage

`canon vl G B T H HV K V BT BK BV sc dt N NT` is the host's launch for those sizes (contiguous
tensors, `dv = empty_like(do)`); `canon_ok` proves `Inst.ok` for every size within the bounds
above (`T ≤ 2^22` among them) with tensors of at most `2^38` elements. Every SETUP.md launch meets
`T ≤ 2^22` (the longest has `T = 2^16`); none is dropped. The `setup_*_ok` theorems, each for both values of
the gate flag `G`:
* the host's test cases (source lines 278–290): `setup_t63`, `setup_t500`, `setup_t1000`,
  `setup_t1024`, `setup_bt16`, `setup_bt32`, `setup_gdn`, `setup_qwen` (fixed length) and
  `setup_var1`, `setup_var3`, `setup_var5`, `setup_var3g`, `setup_var6` (variable length, `int64`
  tables, `NT` as `prepare_chunk_indices` gives it: 1, 16, 35, 16, 68);
* the benchmark shapes `setup_b1` … `setup_b6`;
* gated DeltaNet's default layer, `setup_gdn_default`;
* the `2^16`-token packed batch, `setup_train` (`int32` and `int64` tables, 1027 chunks);
* the largest grid `y`, `setup_rows` (`B·HV = 65535`).

`setup_var3` carries its real tables (`cu_seqlens = [0, 256, 500, 1000]` and the 16 rows
`prepare_chunk_indices` builds), and `setup_var3_tables` proves them `TablesOk`. The other
instances leave the tables' contents open; `TablesOk` and `ValsOk` are about contents, so
`check_gpu.py` checks them on the host's inputs (they hold on every one).

## Tolerances

From the library's facts (`docs/rules/num.md`): `DotBF16` (`relDot16 K = (K+2)·2^-22`,
`absDot16 K = (K+2)·2^-126`, relative to `|acc| + Σ liftMag a·liftMag b`), `Exp2F32` (`2^-22`
relative, `2^-126` absolute), `Exp2F32NegInf`, and the IEEE rounding lemmas (`float32`: `u = 2^-24`,
`η = 2^-150`; `bfloat16`: `2^-8`, `2^-134`). Fix a valid pair `p ≤ j` of one chunk, value head
`hv`, `L = kqMag(p, j)`, `S = kq(p, j)`. Notation (with `wErr`, `wUp` of `Claim.lean`, the gate's
error and upper bound): `nK = cdiv(K, BK)`, `εA = (BK + 8 + 2·nK)·2^-22`, the entry's error bound
`eA = σ·L·(εA·wUp + wErr) + wUp·(σ·2^-110 + 2^-140) + 2^-140`, and over `j ∈ [p, chunkEnd)`
`B1 = Σ_j |Aex|·|do_j|`, `Be = Σ_j eA·|do_j|`, `Bl = Σ_j (|Aex| + eA)·liftMag do_j`,
`Dl = Σ_j liftMag do_j`. Steps 1–6 give the fine bound
`tol₀ = 2^-8·|ref| + (1 + 2^-8)·(2^-8·B1 + (1 + 2^-8)·Be + (BT + 3)·2^-22·Bl) + 2^-100·(1 + Dl)`;
step 7 relaxes it to `tol`.

1. **The `K` loop** (lines 128–136, `nK ≤ 16` trips, each a fresh `tl.dot` of `BK` columns then
   `tl.mul_add` with `σ₃₂ = hostF32(σ)`, `|σ₃₂ − σ| ≤ u·σ` since `σ ∈ [2^-32, 2^32]`). Trip `t`'s
   dot is within `(BK+2)·2^-22·L_t + (BK+2)·2^-126` of its exact sum (`L_t` its columns' share of
   `L`; masked `K` tails load `+0`, of `liftMag 0`). The multiply-add, fused or not, adds at most
   `u·|D_t·σ₃₂| + u·|partial sum| + 2η`; summed over the trips, the partial sums are at most
   `σ₃₂·L·(1 + ε)`, so the computed `P` has
   `|P − σ₃₂·S| ≤ σ₃₂·L·((BK+2)·2^-22 + (nK+1)·u)·(1 + 2^-20) + nK·(σ₃₂·(BK+2)·2^-126 + 2^-148)`.
   Room: `L ≤ 2^100` and `σ·L ≤ 2^100` (`ValsOk`).
2. **The gate** (line 138). The IEEE `fl(g[j] − g[p])` is within `u·|dg| + 2^-149` of `dg` (an
   overflow to `−∞` gives `exp2 = +0`, `Exp2F32NegInf`, and then `2^{dg} < 2^-126`); `Exp2F32` on
   it: the computed `G` has `|G − 2^{dg}| ≤ 2^{dg}·(2^{u·|dg| + 2^-149}·(1 + 2^-22) − 1) + 2^-126 =
   wErr` and `G ≤ wUp`; finite since `dg ≤ 64`. Without a gate `G` is absent (`wt = 1`,
   `wErr = 0`).
3. **The entry before rounding.** `Ā = fl(P·G)` (or `P`): `|Ā − σ·S·wt| ≤ |P − σ₃₂S|·G +
   |σ₃₂ − σ|·|S|·G + σ·|S|·|G − wt| + u·|P·G| + η`. The coefficient of `σ·L·wUp` is at most
   `(BK + 2)·2^-22 + (nK + 3)·u + 2^-34 ≤ εA`: the cross terms (`εK·u`, `(1 + 2^-20)` factors) are
   at most `130·2^-42 < 2^-34` at `BK = 128`, and `εA`'s `(6 + 2nK)·2^-22` beyond `(BK+2)·2^-22`
   is more than six times `(nK + 3)·u + 2^-34`, the `σ·|S|·wErr` term is at most `σ·L·wErr`, and the absolute terms
   at most `wUp·(σ·2^-115 + 2^-144) + 2^-149`: in all `|Ā − Aex| ≤ eA`. Room: `σ·L·wUp ≤ 2^100`.
4. **Line 140.** Masked lanes (`p > j`, or a token past the sequence) are `+0` exactly (`tl.where`
   discards any `NaN` or `∞` there). A valid entry rounds to `bfloat16`:
   `|Â − Ā| ≤ 2^-8·|Ā| + 2^-134`, so `|Â − Aex| ≤ 2^-8·|Aex| + (1 + 2^-8)·eA + 2^-134`.
5. **Line 148** (`tl.dot`, `K = BT`, zero accumulator): within `(BT+2)·2^-22·Σ_j liftMag Â·liftMag do
   + (BT+2)·2^-126` of `Σ_j Â·do`; `liftMag Â ≤ |Â| + 2^-126 ≤ (1 + 2^-8)·(|Aex| + eA) + 2^-125`,
   and `(BT + 2)·(1 + 2^-8) ≤ BT + 3`. Room: `Σ_j liftMag Â·liftMag do ≤ 3·2^120 + 2^94 < 2^127`
   (`ValsOk`'s per-output bound, `|do| ≤ 2^100`, `eA ≤ (1 + εA)·σ·L·wUp + 2^-13`).
6. **Assembly.** With `x` the `float32` result,
   `|x − ref| ≤ E = 2^-8·B1 + (1 + 2^-8)·Be + (BT+3)·2^-22·Bl + abs`, `abs ≤ 2^-134·Dl +
   (BT+2)·2^-147·Dl + (BT+2)·2^-126`. The store: `|c − x| ≤ 2^-8·|x| + 2^-134` and
   `|x| ≤ |ref| + E`, so `|c − ref| ≤ 2^-8·|ref| + (1 + 2^-8)·E + 2^-134 ≤ tol₀`
   (`(1 + 2^-8)·abs + 2^-134 ≤ 2^-100·(1 + Dl)`); `c` is finite (`|x| < 2^126`).
7. **The relaxation `tol₀ ≤ tol`**, term by term in `j` (all terms are nonnegative; `d = |do_j|
   ≤ liftMag do_j = d'`). *The gate:* for `|dg| ≤ 2^10`, `2^{u·|dg| + 2^-149}·(1 + 2^-22) − 1 ≤
   2^{2^-14}·(1 + 2^-22) − 1 ≤ 4.26·10^-5`; for `dg < −2^10`, `2^{dg}·2^{u·|dg| + 2^-149}·(1 + 2^-22)
   ≤ 2^-1000`; so `wErr ≤ ρw·wt + 2^-126·(1 + 2^-800)`, `ρw = 4.26·10^-5 ≤ 2^-14`, and
   `wUp ≤ (1 + ρw)·wt + 2^-125`. *The entry:* with `εA ≤ 140·2^-22 = 3.34·10^-5` (`BK ≤ 128`,
   `nK ≤ 2` at `BK = 128` since `K ≤ 256`; `56·2^-22` at `BK = 16`), `σ ≥ 2^-32` and
   `aMag' = σ·(L + 2^-94)·(wt + 2^-112)` (so `aMag = aMag' + 2^-86`):
   `σ·L·(εA·wUp + wErr) ≤ (εA·(1 + ρw) + ρw)·σ·L·wt + 1.0001·2^-126·σ·L ≤ 7.6·10^-5·σ·L·wt +
   2^-13.99·σ·L·2^-112`; `wUp·σ·2^-110 ≤ 2^-16·1.0001·σ·2^-94·(wt + 2^-112)`; `wUp·2^-140 ≤
   wUp·σ·2^-108 ≤ 2^-14·1.0001·σ·2^-94·(wt + 2^-112)`; so `eA ≤ 8.0·10^-5·aMag' + 2^-140`.
   *The sums:* `|ref| ≤ B1`; `B1 ≤ Σ_j σ·L·wt·d ≤ Σ_j aMag'·d'`; `Be ≤ 8.0·10^-5·Σ_j aMag'·d' +
   2^-140·Dl`; `Bl ≤ (1 + 8.0·10^-5)·Σ_j aMag'·d' + 2^-140·Dl`. So, with `BT + 3 ≤ 67`,
   `tol₀ ≤ 2^-7·B1 + (2^-16 + 1.008·8.0·10^-5 + 1.004·67·2^-22·1.0001)·Σ_j aMag'·d' +
   1.0001·2^-100·Dl + 2^-100 ≤ 2^-7·B1 + 1.12·10^-4·Σ_j aMag'·d' + 2^-99·Dl + 2^-100`, and
   `1.12·10^-4 ≤ 2^-13 = 1.22·10^-4`, `2^-99 = 2^-13·2^-86`: `tol₀ ≤ 2^-7·B1 + 2^-13·Σ_j aMag·d' +
   2^-100 = tol`. (`2^-8·|ref| + (1 + 2^-8)·2^-8·B1 = 2^-7·B1 + 2^-16·B1`, the excess charged to
   the second term.) A randomized check of the per-`j` inequality over `σ ∈ [2^-32, 2^32]`, `L`
   from `0` to `2^100`, `kq ∈ [−L, L]`, `dg` from `−2^128` to `64`, `|do|` from `0` to `2^100`
   (subnormal included), `BK`, `K`, `BT` over their ranges (2·10^6 samples) finds the `tol₀` term at
   most `0.9934` of the `tol` term.

The bound is provable from `DotBF16`, `Exp2F32`, `Exp2F32NegInf` and the library's IEEE rounding
lemmas (`FloatV.mul_err`, `add_err`, `sub_err`, `cvt_err`, `AddRel.err`; `tl.mul_add`'s two
alternatives, the fused `FloatV.fma` or the two IEEE operations). It is meaningful: its leading
term is the two `bfloat16` roundings themselves, so on random inputs it is about `1.07×` the
kernel's worst error (`check_gpu.py`: worst error / tolerance `0.934`); over SETUP.md's launches
`tol / |ref|` has median `2.4·10^-2` (`1.5·10^-2` for `tol₀`), and a single wrong element, a
missing gate or a wrong chunk size is rejected.

## Exclusions

* **`B·HV ≥ 65536` on fixed-length batches (a limit of the kernel).** The wrapper puts `B·HV` on the
  grid's `y` axis, which CUDA limits to `65535`; the first failing launch has `B·HV = 65536`
  (`check_gpu.py`: `B = 4096`, `HV = 16`, `Triton Error [CUDA]: invalid argument`; also, for
  instance, 2048 rows at 32 value heads). `Inst.ok.grid1_lt` keeps `B·HV < 2^16`. SETUP.md's model
  and benchmark launches have at most `4·64 = 256` (its edge launch has `65535`); variable-length
  batches (`B = 1`) never reach it.
* **More than `2^31` programs or chunks** (`NT·B·HV ≥ 2^31`): CUDA's and the launcher's limit
  (`Launch.Valid`); real use is far below (the `2^16`-token batch has `1027·32` programs).
* **`int32` offsets: none.** Every element offset is `int64` (lines 99, 106, 108–113, 120–121,
  131–132, 145–146), so the kernel addresses tensors of more than `2^31` elements correctly; only
  `o_k`, `o_v` (below `2^9`) and the sequence index `i_n` (`N < 2^31`) are `int32`. `check_gpu.py`
  does not run a tensor of `2^31` elements (4 GiB in `bfloat16`, above the check's memory budget);
  `T ≥ 2^31` makes `T` an `int64` argument, another specialisation: outside real use.
* **Gate values.** With a gate rising by more than about 128 within a chunk the kernel's
  `exp2(g[j] − g[p])` overflows to `∞` and a valid entry becomes `∞` or `NaN` (`0·∞`): a limit of
  the kernel at those values; `ValsOk` keeps `g[j] − g[p] ≤ 64`. Real use never reaches it: fla's
  `g` is the chunk-local cumulative sum of nonpositive log decays, nonincreasing within a chunk up
  to `tl.cumsum`'s `float32` rounding (any rise is about `u·|g|`, far below 64).
* **Input magnitudes.** Non-finite inputs in a sequence, `|do| > 2^100`, `kqMag > 2^100`, and
  products beyond the room bounds (`2^100`, `2^120`): `float32` overflows near `2^128` (a limit of
  the kernel at those magnitudes); real gradients and normalised queries and keys are far below.
  Outside real use.
* **`scale` outside `[2^-32, 2^32]`**, zero or negative: outside real use (the default is
  `K^-0.5 ∈ [1/16, 1]`). An `int` `scale` (`1`, as some of fla's tests pass it) is a compile-time
  constant, another specialisation: not part of the setup.
* **`V > 2^16` or `HV·V ≥ 2^31`**: outside real use (SETUP.md: `V ≤ 1000`, `HV·V ≤ 12288`). At
  `V ≥ 2^31` the kernel is not correct (a limit of the kernel): Triton builds the constexpr `V` as a
  `uint32` constant, so line 144's `o_v < V` is an unsigned comparison the model's signed one does
  not match (at `V = 2^31` every lane is masked off and nothing is stored), and beyond `2^31` the
  `int32` `i_v * BV` (line 143) wraps.
* **`K > 256`**: outside real use (the chunked forward asserts `K ≤ 256`). Chunk sizes other than
  16, 32, 64: not part of the setup.
* **`USE_G_GAMMA`, `USE_A`** (`g_gamma` or `A` given), `float16`/`float32` inputs: not part of
  SETUP.md's launches (no caller in fla v0.6.0 passes `g_gamma` or `A` to this host).
* **Out of scope: dimensions that are not multiples of 128 or exceed `2^17`, contexts longer than
  `2^22` tokens, and batches of more than `8192` requests: the claim covers real model shapes
  only.** The context length: `T` (`T_ctx`, the tokens of a row or of the packed batch). The batch
  sizes: `B` (`B_le`, the batch rows) and, on variable length, `N` (`N_le`, the sequences). No
  field is a model dimension: `K`, `V` are head dims and `H`, `HV` head counts; they keep their
  bounds above. Every launch of `SETUP.md` meets them (at most `4369` rows, `6` sequences).
* **Tables other than `TablesOk`** (overlapping or out-of-range rows would make two programs write
  one element, or a program read outside its sequence), `dv` overlapping the inputs: outside the
  host's use.

## Library requests

None.

## Checks

* **Builds**: `Claim.lean` builds with no errors or warnings, no `sorry`, no `axiom`.
* **Not vacuous**: `consistent` (`Platform.sm90ref`); `satisfiable` (one row of one token, one
  head, head dims 16, chunk size 16, no gate, every byte `0`).
* **Covers the setup**: `canon_ok` and the `setup_*_ok` above (each for both gate flags; the
  training batch for both table dtypes); `setup_var3_tables` (`TablesOk` on real tables).
* **GPU** (`check_gpu.py`, H200, Triton 3.8.0, peak device memory 2.20 GiB): 46 checks pass: every
  SETUP.md launch without and with a gate, `Inst.ok` and `Pre` edges (SETUP.md, "Checked on the
  H200"); worst error / tolerance `0.934` (margin `1.07×`); `tol / |ref|` over SETUP.md's launches
  has median `2.4·10^-2`; every written element finite, `ValsOk`
  and `TablesOk` on every input, the inputs unchanged, every element of `dv` outside the sequences
  unchanged; four negative tests rejected; `B·HV = 65536` does not launch.
