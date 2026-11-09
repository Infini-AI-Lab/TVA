# `streamk_matmul_f16`: prepared claim

The Stream-K GEMM `streamk_matmul_kernel` (`source/streamk_matmul.py`, kernel lines 26–64 with the
imports of lines 22–23; host `streamk_matmul` 70–93). `Claim.lean`, namespace
`Claims.StreamKMatmul`; the claim to prove is `Statement`. The launches to cover are in `SETUP.md`.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.StreamKMatmul.main : Statement` (`Statement`:
`Claim.lean` line 398). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 191), and every initial memory `init` that `Pre` accepts (line 294: the buffers
valid, every element of the matrix `c` a finite zero), every outcome of the launch is a normal end
(no fault, no hang) whose final memory `fin` satisfies `Post` (line 365).

**What `Post` asks** (line 365), two parts:
1. for every `i < M`, `j < N`, `OutSpec d init fin i j` (line 354): element
   `i * stride_cm + j * stride_cn` of `c` reads in `fin` as some `float32` `c` (on every input);
   and for all `va vb : Nat → F16` such that, for every `k < K`, `init` reads `a`'s element
   `i * stride_am + k * stride_ak` as a finite `va k` and `b`'s element
   `k * stride_bk + j * stride_bn` as a finite `vb k`: `c` is finite and
   `|c.toRat - dotSum va vb d.K| ≤ tol va vb d.K`.
   The reference `dotSum` (line 307) is the exact sum `Σ_{k<K} va k · vb k` in the rationals.
   `tol` (line 348) is `2^-15 · tolMag va vb K + 2^-100`, with `tolMag` (line 342)
   `Σ_{t ≤ T} dev t + Σ_{t < T} chunkMag t` over the chunks `t < T = steps K` (`steps`, line 168;
   `chunk`, line 316: `k ∈ [64t, min(64t + 64, K))`). The helpers are `chunkSum`, `chunkMag`,
   `mag16`, `pre` and `dev` (lines 319, 324, 313, 328, 334). The bound holds whatever order the
   atomic additions ran in. With a non-finite input, only "holds a value" is asked;
2. the frame: every byte outside the matrix `c` (`c.elems (cIdx d)`, `cIdx` line 185) is
   unchanged: `a`, `b`, and the elements of `c`'s storage outside the matrix.

**What you are given** (`Assumptions`, line 373):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, when the launch is
  valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `dot` (`Rules.Num.DotF16`, a `DotLiftAcc`): the `tl.dot` of source line 57 is within
  `relDot16 K · (|acc| + Σ liftMag a · liftMag b) + absDot16 K` of `acc + Σ a·b`, with room below
  the `float32` range;
- `dot_total` (`Rules.Num.DotTotal` for the model's configuration): that `tl.dot` has a result;
- `atomic` (`Rules.Num.AtomicAddF32`): the `float32` `tl.atomic_add` of source line 63 is within
  `2^-23·|o + v| + 2^-124` of `o + v` when `o`, `v` are finite and the sum is in range;
- `atomic_total` (`Rules.Num.AtomicAddTotal P Format.binary32`): that atomic has a result.

**The kernel, as the model runs it** (`streamk_matmul_kernel`, line 49; `model` line 276, `launch`
line 286): a 1-D grid of `num_programs` programs with `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 64`.
Program `pid` owns the iterations `[pid · iters_per_program, end)`, `end = min(…, total_iters)`, and
runs a `while it < end` loop. Each trip takes one tile (`tile = it / iters_per_tile`), runs a
`for kk in [k_first, k_last)` loop of two masked loads and one `tl.dot` per step into a zero
accumulator, then either stores the whole tile (`k_first = 0` and `k_last = iters_per_tile`: the
program owns the tile) or `tl.atomic_add`s its partial sum into it (masked by `c_mask`); then
`it += k_last − k_first`. Programs share the cells of split tiles through atomics, so the route is
`TVA.meets_of_faithful` with `TVA.shared` (`TVA/Lemmas/Shared.lean`). The programs' footprints are:
private writes `W i` = the tiles program `i` owns entirely; shared cells `C` = the elements of the
split tiles; reads `R i` = the matrices `a` and `b`. Each program is proved with `SafeA`: the inner
`for` loop writes nothing and is a `Safe` segment (`safe_forRange` with an invariant and
`fun _ => ∅`), lifted by `SafeA.of_safe`; the store goes through the `SafeA.store` constructor, the
atomic through `SafeA.env`, then `SafeA.atomic`. No `SafeA` rule exists for `while`: unfold it
with the `SafeA.loop` constructor, by induction on `end − it`, which each trip lowers by
`k_last − k_first ≥ 1`.

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: `num_programs < 2^31` from
  `progs_lt`, with `iters_per_program ≥ 1` from `cover`, `total` and `M_pos`, `N_pos`, `K_pos`;
  the nine listed block shapes are powers of two);
- each program is `SafeA`: no `int32` value wraps (`M_le`, `N_le`, `K_le`, `progs_lt`; the
  products by `am_fit` … `cm_fit`, `c_rows`); the enabled lanes of the loads are in bounds
  (`a_len`, `b_len`) and the store's and atomic's lanes are in `c` (`c_len`), distinct
  (`cn`, `c_rows`); `safe_idiv`, `safe_irem`, `safe_convert`, `safe_dot` (`dot_total`) for the
  steps; the `while` loop ends; every column tile is full (`N_mul`: `128 ∣ N`, and `BLOCK_N = 128`)
  and every `K` chunk is a full 64 (`K_mul`: `128 ∣ K`, and `BLOCK_K = 64`), so the column and `k`
  masks are all true; only the row tiles may be partial (`M_le22` bounds `M`, not its remainder);
- the footprints: the owned tiles of different programs are disjoint, a split tile is owned by no
  program, the cells of `C` are equal or disjoint and present, and `c`'s matrix is disjoint from
  the reads (`c_a`, `c_b`); the postcondition `Q i log m` is `PermInv`, pairing the log with the
  program's partial sums (`Paired`, `Paired.lanes`);
- the value part of `Post`, `|c − S| ≤ 2^-15 · tolMag + 2^-100`: derive the fine bound
  `tolDot + DotLoop.bound 2^-23 2^-124 T (C + tolDot)` of README Tolerances (one program's part of
  a tile is within the `dev`-based recursion of `tolDot`: `dot`, one step by
  `DotLift.dotAdd_step`, its own induction; the parts of a tile are disjoint chunk ranges covering
  `0 … steps K − 1` (`cover`, `total`); a split element's atomic additions are bounded by
  `replay_err_shared` with `accStep_atomicAdd` (`atomic`) and `accStep_other`, under its room
  hypothesis), then relax it by the inequality chain there (`T ≤ 128`; `C` bounded by the `dev` terms);
- the frame: the bytes outside `W i` and `C` keep their initial value (`shared`'s last part).

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `idx_lt` (line 429) and `cdiv_cover` (line 434) can be reused.

## Specification

Element `(i, k)` of the matrix `a` is element `i·stride_am + k·stride_ak` of its buffer, likewise
for `b` and `c`. `Post d init fin`:
- for every `i < M`, `j < N` (`OutSpec`): element `(i, j)` of `c` holds a `float32` value `c`; when,
  moreover, row `i` of `a` and column `j` of `b` hold finite `float16` values `va k`, `vb k`
  (`k < K`) in `init`, `c` is finite and `|c − S| ≤ tol va vb K`, with `S = Σ_k va k·vb k` exact;
- this holds **whatever order the atomic additions ran in**, and for every value each rounding may
  take within the library's facts;
- frame: every byte outside the elements of the matrix `c` is unchanged (`a`, `b`, the padding of
  `c`'s storage).

The tolerance is stated per element, chunk by chunk of the kernel's `K` loop. Chunk `t < T =
cdiv(K, 64)` holds the products `k ∈ [64t, min(64t + 64, K))` (`chunk`), one `tl.dot` of `K = 64`:
- `chunkSum t = Σ_{k ∈ chunk t} va k·vb k` (exact), `chunkMag t = Σ_{k ∈ chunk t} mag16(va k)·mag16(vb k)`,
  with `mag16 q = |q|`, except that a nonzero `q` below `2^-14` (a subnormal) counts as `2^-14`, as
  the tensor core aligns it (`Rules.Num.liftMag .f16`, by which `DotF16` counts);
- `pre s = Σ_{u<s} chunkSum u`, and `dev t = max_{s ≤ t} |pre t − pre s|`: the largest exact value
  the accumulator of a program can hold at chunk boundary `t`, whichever chunk `s` its range starts
  at;
- `tolMag = Σ_{t ≤ T} dev t + Σ_{t < T} chunkMag t`: the accumulators' magnitudes at the `T + 1`
  boundaries plus the products' lifted magnitudes;
- **`|c − S| ≤ 2^-15 · tolMag + 2^-100`**.

Its size follows the element's own partial sums, not `T·Σ|a·b|`: on `torch.randn` inputs, where
`|pre t|` grows like `√(64t)` and `Σ_t chunkMag t ≈ (2/π)·K`, `tolMag` is about `Σ_t dev t + Σ|a·b|`.
What it rejects (`check_gpu.py`, at every tested `K` from 64 to 8192, the test's `4096`, `4097`
included): on `randn` inputs, the result rounded to `bfloat16` (worst error / tolerance `1.7`–`41`),
one 64-chunk left out (`96`–`13000`), and a `float16` accumulation chunk by chunk (`1.7`–`5.6`);
with the mass in the last chunk, the result rounded to `float16` (`7.7`–`8.0`). It does not reject the
result rounded to `float16` on `randn` inputs from about `K = 1024` (ratio `0.93` at `K = 1024`,
`0.37` at `K = 4096`): the `float16` rounding of `|S| ≈ √K` is then below `2^-15·Σ|a·b|`, about the
library's `DotF16` constant times the products' magnitude (Library requests). The median of
`tol / |S|` over the test's host launches is `2.6·10^-4` to `3.9·10^-3`. The device's error is
`0.0004`–`0.0051` of the tolerance on the test's launches. On every input `tol` is at most `0.4%` of
`A = Σ_k mag16(va k)·mag16(vb k)`, plus `2^-100`, at `K ≤ 2^13` (`dev t ≤ A`, so
`tolMag ≤ (T + 2)·A ≤ 130·A`).

With a NaN or an infinity in row `i` of `a` or column `j` of `b`, `Post` says only that element
`(i, j)` holds a value.

`Spec K := Claim.Meets Inst.ok Pre Post K`.

## Conditions

`Inst`: `M`, `N`, `K`, the six strides (elements), `num_programs`, the arguments
`iters_per_program` and `total_iters`, the `float16` buffers `a`, `b` and the `float32` buffer `c`.

`Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `M_pos`, `N_pos`, `K_pos` | `M, N, K > 0` | the host asserts it (line 75) |
| `M_le` | `M + 127 ≤ 2^31` | `M` is an `int32`; the tile rows `pid_m·128 + i` (line 48) stay below `cdiv(M, 128)·128 ≤ M + 127` |
| `N_le` | `N + 128 ≤ 2^31` | `N` is an `int32`; `tl.cdiv(N, 128)` computes `N + 127` in `int32` (line 37), and the tile columns stay below `2^31` |
| `K_le` | `K ≤ 2^13` | at most `128` chunks, which the relaxation of Tolerances uses; keeps the tolerance below `0.4%` of `A = Σ mag16·mag16` on every input; also `K + 63 < 2^31` (line 38) |
| `N_mul`, `N_le17` | `N % 128 = 0`, `N ≤ 2^17` | `N` is a model dimension (the GEMM's output features) |
| `K_mul`, `K_le17` | `K % 128 = 0`, `K ≤ 2^17` | `K` is a model dimension (the GEMM's inner features) |
| `M_le22` | `M ≤ 2^22` | `M` counts tokens, a context length |
| `am`, `ak`, `bk`, `bn`, `cm` | strides `< 2^31` | `int32` arguments (zero allowed) |
| `cn` | `0 < stride_cn < 2^31` | with `c_rows`, distinct elements of `c` are distinct storage elements |
| `am_fit` … `cm_fit` | `(M−1)·stride_am`, `(K−1)·stride_ak`, `(K−1)·stride_bk`, `(N−1)·stride_bn`, `(M−1)·stride_cm` `< 2^31` | the `int32` products of the lanes inside the matrices (lines 53, 55, 58) do not wrap |
| `c_rows` | `N·stride_cn ≤ stride_cm` | `c` row-major, possibly padded: rows do not overlap, and `offs_n·stride_cn` stays below `2^31` |
| `total` | `total_iters = cdiv(M,128)·cdiv(N,128)·cdiv(K,64)` | the host's formula (lines 79–80) |
| `cover` | `total_iters ≤ num_programs·iters_per_program` | the ranges cover every iteration, as `cdiv(total_iters, num_programs)` (line 81) gives |
| `progs_lt` | `num_programs·iters_per_program < 2^31` | `pid·iters_per_program`, `it + iters_per_program` (lines 39–40) do not wrap; fewer than `2^31` programs |
| `a_len`, `b_len`, `c_len` | every matrix element inside its buffer | |
| `c_a`, `c_b` | the matrix `c` shares no byte with the matrix `a`, nor `b` | programs reading bytes others write would race; the host's `c` is fresh |

The integers are runtime `int32` arguments, or the compile-time `1` when equal to `1`, which gives the
same model at each use (`model`'s docstring). Covered beyond the host's use: any strides within the
bounds (transposed or broadcast `a`, `b`, a padded `c`), any `num_programs` and `iters_per_program`
that cover the iterations, and `a`, `b` overlapping each other.

`Pre`: the three buffers are valid (`Buf.Valid`), and every element of the matrix `c` holds a
finite zero (the host's `torch.zeros`, line 78). Every `float16` input value is covered.

The launch: `Launch.grid1 num_programs` with the source's nine block shapes, `enable_fp_fusion =
true`, `debug = false`. Part (A) of `Compiler.Faithful`'s scope (`docs/rules/compiler-faithful.md`):
(4) the launch gives the constexprs and `num_warps=8`, `num_stages=3` only; no `warmup`,
`triton.compile`, `@triton.autotune`, `@triton.heuristics`; a bare `@triton.jit`; (5) no tensor
descriptor; (6) no `CompiledKernel`; (8) no host `TensorDescriptor`. The
claim relies on part (B), the environment, which is assumed and not checked.

The kernel is inside the supported subset (`docs/SEMANTICS.md`): a top-level `def` under a bare
`@triton.jit`; one `while` with a runtime condition carrying `it`, one `for kk in range(lo, hi)`
carrying `acc`, one runtime `if`/`else` assigning no name; it reads only `triton` and `tl`. The
source's name `end` is written `«end»` in the model.

## Assumptions

Five `[rule]` fields, each the library's fact as `docs/rules/num.md` states it: `faithful`
(`Compiler.Faithful`), `dot` (`Num.DotF16`: `DotLiftAcc` with `relDot16 K = (K+2)·2^-22`,
`absDot16 K = (K+2)·2^-126`, relative to `|acc| + Σ liftMag a·liftMag b`), `dot_total`
(`Num.DotTotal` for the model's configuration), `atomic` (`Num.AtomicAddF32`: `2^-23·|o + v| +
2^-124`), `atomic_total` (`Num.AtomicAddTotal P Format.binary32`). The `other=0.0` conversion is an
IEEE conversion the library defines. No `[kernel]` field.

## Coverage

`setup_512_512_512_p132`, `setup_3072_4096_1024_p132`, `setup_8192_8192_4096_p132`, with
`setup_*_ok`: the test launches of `SETUP.md` whose `N` and `K` are multiples of 128 (contiguous `a`,
`b`, a fresh `c`, the host's `total_iters` and `iters_per_program`; `mk`). `mk_ok` covers the host's
launch for every shape and program count within the bounds. The test's `1000³` (132 programs),
`257 × 129 × 4097` (7 programs) and `64³` (3 programs) are not covered: their `N` and `K` are not
multiples of 128 (Exclusions).

## Tolerances

**The fine bound**, from the library's facts, for one element `(i, j)` with finite inputs:
`tolDot + DotLoop.bound 2^-23 2^-124 T (C + tolDot)`, with
`tolDot = Σ_{t<T} (1 + ρ)^(T−1−t)·(ρ·(dev t + chunkMag t) + α)`, `ρ = relDot16 64 = 66·2^-22`,
`α = absDot16 64 = 66·2^-126` (`Rules.Num.DotF16`), `C = Σ_{t<T} |chunkSum t|`, and
`DotLoop.bound r a T B = ((1+r)^T − 1)·B + T·(1+r)^T·a` (`2^-23`, `2^-124`: `Rules.Num.AtomicAddF32`).
- **One `tl.dot` step.** Chunk `t` is one `tl.dot` of `K = 64` into the accumulator `acc` (the lanes
  past `K` load `+0`, of lifted magnitude `0`). `DotF16` (`DotLiftAcc`, one step by
  `DotLift.dotAdd_step`) gives a result within `ρ·(|acc| + chunkMag t) + α` of `acc + chunkSum t`,
  with room below the `float32` range (finite `float16` inputs and `K ≤ 2^13` give magnitudes below
  `2^46`).
- **One program's part.** A program that touches the tile runs the chunks `s ≤ t < e` of it from
  `acc = +0` (exact). Its exact partial value before chunk `t` is `pre t − pre s`, of magnitude at
  most `dev t` (as `s ≤ t`). With `E_t` the error before chunk `t`, `|acc| ≤ dev t + E_t`, so
  `E_{t+1} ≤ (1 + ρ)·E_t + ρ·(dev t + chunkMag t) + α`, and the part ends within
  `Σ_{s≤t<e} (1+ρ)^(e−1−t)·(ρ·(dev t + chunkMag t) + α) ≤ Σ_{s≤t<e} (1+ρ)^(T−1−t)·(…)`
  (`DotLoop.accum_step` does not give this directly: its bound uses the sum of the steps'
  magnitudes, not `dev`, so the proof needs its own induction on this recursion). The parts of a tile are disjoint ranges of chunks
  covering `0 … T−1` (from `cover`), so their errors add up to at most `tolDot`, whatever the split.
  A tile one program owns is stored: within `tolDot`.
- **The atomic additions.** A split tile's element receives one `atomic_add` per touching program
  (at most `T`) into the zero `c`, in any order; `AtomicAddF32` per step and `replay_err_shared`
  bound the result against `0 + Σ partials` by `DotLoop.bound 2^-23 2^-124 n (Σ|partial|)`, `n ≤ T`,
  under its room hypothesis (`hroom`: the running bound at most `float32`'s `maxVal`), which holds
  since every partial and running sum is below `2^47`.
  Each partial is within its part's error of `pre e − pre s`, of magnitude at most
  `Σ_{s≤t<e} |chunkSum t|`, so `Σ|partial| ≤ C + tolDot`. Their exact values add up to `S`, their
  errors to at most `tolDot`: `|c − S| ≤ tolDot + DotLoop.bound 2^-23 2^-124 T (C + tolDot)`.

**Relaxation to `Post`'s form.** `K ≤ 2^13` gives `T ≤ 128`. Let `M_0 = Σ_{t<T}(dev t + chunkMag t)`,
so `tolMag = M_0 + dev T`.
- `(1 + ρ)^127 < 1.0021`, so `tolDot ≤ 1.0021·(66/64)·2^-16·M_0 + 1.0021·T·α ≤ 1.0335·2^-16·M_0 + 2^-112`.
- `f = (1 + 2^-23)^T − 1 ≤ 128·2^-23·(1 + 2^-23)^128 ≤ 1.00002·2^-16`, and
  `T·(1 + 2^-23)^T·2^-124 ≤ 2^-116`.
- `C` against `tolMag`: `dev t ≥ |pre t − pre (t−1)| = |chunkSum (t−1)|` for
  `1 ≤ t ≤ T`, so `Σ_{t<T} dev t ≥ C − |chunkSum (T−1)|`, and `Σ_{t<T} chunkMag t ≥ C`
  (`|chunkSum t| ≤ chunkMag t`, as `mag16 q ≥ |q|`); hence `M_0 ≥ 2C − |chunkSum (T−1)|` and
  `dev T ≥ |chunkSum (T−1)|`.
- The fine bound is `(1 + f)·tolDot + f·C + T(1+2^-23)^T·2^-124 ≤ 1.0336·2^-16·M_0 + 1.00002·2^-16·C + 2^-111`.
  With the previous step, `1.00002·C ≤ 0.9664·M_0 + 2·dev T` (as `0.9664·(2C − x) + 2x ≥ 1.93·C`
  for `x = |chunkSum (T−1)|`), so the fine bound is at most
  `2^-16·(2·M_0 + 2·dev T) + 2^-111 ≤ 2^-15·tolMag + 2^-100 = tol`.

At `T ≤ 64` (every setup launch) the atomic term is at most half of this; the `T + 1`-th `dev` term
covers the worst case `T = 128` with all the mass in the last chunk.

Meaningful (Specification): it rejects a `bfloat16` result, a lost chunk and a `float16`
accumulation on the test's kind of inputs, and is `200`–`2500` times the measured error on the test's
launches (worst error / tolerance `0.0004`–`0.0051`; `0.018` over every case). The fine bound is
`1.9`–`2.2` times tighter on the test's launches (median `tol / |S|` `1.2·10^-4` to `2.1·10^-3`) and
up to `3.8` times with the mass in the last chunk. The remaining gap is the library's `DotF16`
constant, `66·2^-22` per 64-long `tl.dot` relative to `|acc| + chunkMag`.

## Exclusions

- **Dimensions that are not multiples of 128 or exceed 2^17, and contexts longer than 2^22
  tokens** — out of scope: the claim covers real model shapes only. The fields: `N` and `K` (model
  dimensions: `N_mul`, `N_le17`, `K_mul`, `K_le17`) and `M` (a number of tokens: `M_le22`). This
  leaves out three of the test's launches: `1000³`, `257 × 129 × 4097` and `64³`.
- **`K > 2^13`** — outside the file's launches (largest `4097`); the tolerance grows with `K`.
- **`int32` bounds** (`M_le`, `N_le`, the fits, `progs_lt`, the strides) — a kernel limit: Triton
  passes an integer `≥ 2^31` as an `int64`, and with `int32` arguments the offsets of lines 53, 55
  and 58 wrap once a matrix has about `2^31` elements (first launch: contiguous `a` with
  `(M − 1)·K ≥ 2^31`, e.g. `M = 2^18 + 1`, `K = 2^13`), so lanes would access memory outside the
  matrices; `tl.cdiv(N, 128)` wraps from `N = 2^31 − 127`. Real use (`8192 × 8192`) is far inside.
- **`M`, `N` or `K` `= 0`** — outside the file's launches (the host asserts them positive).
- **`total_iters` or the ranges not as the host computes them** — outside the file's launches; a
  launch with fewer iterations would leave elements of `c` untouched.
- **A `c` that is not zero** — outside the file's launches (`torch.zeros`); the atomic additions
  accumulate into it.
- **Column-major or overlapping layouts of `c`** (`c_rows`) — outside the file's launches (`c` is
  allocated row-major).
- **`c` overlapping `a` or `b`** — outside the file's launches (`c` is fresh); programs would race.
- **Block sizes other than `128, 128, 64`, other dtypes** — outside the file's launches.

## Library requests

None needed for this claim. The claim cites `DotF16`; the library also states `DotF16Chunk` (64 consecutive `k` per chunk), with which the tolerance could tell a `float16`-rounded result from a `float32` one on `randn` inputs at the test's `K = 4096`.

## Checks

- `Claim.lean` builds with no errors or warnings, no `sorry`/`axiom`; `consistent`, `satisfiable`
  and every `setup_*_ok` use only `propext`, `Classical.choice`, `Quot.sound`.
- `check_gpu.py` (H200): the six test launches (host), plus direct launches with `1`, `132`, `1000`
  programs and a padded `c` in a sentinel buffer on the smaller shapes; edges: `1 × 1 × 1`,
  `129 × 257 × 4100`, `K = 2^13`, operands of magnitude `~2000`, subnormal operands, NaN and
  infinities, all-positive inputs, transposed `a` and `b`, `1` to `2048` programs. Worst error /
  tolerance over every element with finite inputs: `0.0004`–`0.0051` on the test, `0.018` overall;
  every such element finite; `a`, `b` unchanged; frame holds. Negative tests at `K = 64 … 8192`
  (Specification) all fail `Post` as they must. 44 of 44 cases pass. Outside `Inst.ok` (`N` or
  `K` not a multiple of 128) and passing as well: the test's `1000³`, `257 × 129 × 4097` and `64³`,
  and the edges `1 × 1 × 1`, `129 × 257 × 4100`, `K = 2^13` (`N = 130`), magnitude `~2000`
  (`K = 700`), subnormal operands (`K = 2000`) and transposed `a`, `b` (`K = 1000`, `N = 200`).
