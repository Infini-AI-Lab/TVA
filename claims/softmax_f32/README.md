# `softmax_f32`: prepared claim

Triton tutorial 02's `softmax_kernel` (`source/02-fused-softmax.py`, kernel lines 84–109 with the
imports of lines 26–27; host `softmax` 124–175). `Claim.lean`, namespace `Claims.Softmax`; the claim
to prove is `Statement`. The launches to cover are in `SETUP.md`.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.Softmax.main : Statement` (`Statement`: `Claim.lean`
line 260). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 112), and every initial memory `init` that `Pre` accepts (line 167), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 221).

**What `Post` asks** (line 221). For every row `r < n_rows` there are values `xs` with:
1. for every `k < n_cols`, input element `x[r, k]` (`inIdx d r k`, line 100) reads in `init` as
   `xs k`;
2. for every `j < n_cols`, output element `y[r, j]` (`outIdx d r j`, line 103) reads in `fin` as some
   `float32` value. This holds for every row, whatever its inputs;
3. only when `RowOk xs n_cols` holds (line 174: every input finite or `-∞`, and at least one finite
   when the row is not empty): there is one real factor `sc` for the whole row with
   `|sc - 1| ≤ sumTol BLOCK_SIZE`, and for every `j < n_cols` the output `c` of `y[r, j]` is finite
   and

   $$|c - sc\cdot\sigma_j| \le (\mathtt{elemTol} + \mathtt{gapTol}\cdot\mathtt{softmaxGap}_j)\cdot\sigma_j + \mathtt{floorTol}\cdot\mathtt{softmaxMax},$$

   with `σ_j = softmaxRef xs n_cols j` (line 183): the exact softmax in the reals,
   `expIn (xs j) / Σ_{k<n_cols} expIn (xs k)` (`expIn`, line 179, `e^a` with `e^(-∞) = 0`).
   `softmaxMax` (line 188) is the row's largest `σ`, `softmaxGap` (line 193) is
   `log(softmaxMax / σ_j)`. The tolerances: `sumTol` (line 200, `(⌈log₂ B⌉ + 56)·2^-24`),
   `elemTol` (line 204), `gapTol` (line 208), `floorTol` (line 212). Outside `RowOk` only part 2 is
   asked;
4. the frame: every byte outside the output elements `outSet d` (line 109) is unchanged.

**What you are given** (`Assumptions`, line 237):
- `faithful` (`Compiler.Faithful`): the device's outcomes `K d` are outcomes of the model's launch,
  when the launch is valid, has `debug = false`, and the model's launch is well behaved from `init`;
- `exp_total` (`Num.MathTotal P .exp .binary32`): `tl.exp` has a result on every input (source
  line 103);
- `exp_acc` (`Num.ExpF32`): for finite `x` in `[-X, X]`, `tl.exp` is finite and within
  `(2^-22 + 2^-23·X)·e^x + 2^-140` of `e^x` (source line 103);
- `exp_ninf` (`Num.ExpF32NegInf`): `tl.exp(-∞) = +0` (source line 103);
- `div_total` (`Num.DivTotal P .binary32`): `/` on `float32` has a result for every pair of
  inputs (source line 105);
- `div_full` (`Num.DivFull`): `/` on `float32` within relative `2^-22`, absolute `2^-148`, for
  quotients up to `2^126` (source line 105);
- `sum_depth` (`Num.SumDepth`): the tree of the float `tl.sum` (source line 104) over `BLOCK_SIZE`
  positions has depth at most `⌈log₂ BLOCK_SIZE⌉`.

**The kernel, as the model runs it** (`softmax_kernel`, line 44; `model`, line 152; `launch`,
line 161): a 1-D grid of `num_programs` programs. Program `p` runs the `int32` loop
`tl.forRange p n_rows num_programs` over rows `p, p + num_programs, …` below `n_rows`. Each trip loads
row `row_idx` (one block of `BLOCK_SIZE` lanes, masked by `n_cols`, other lanes `-∞`), takes the max
(`tl.maxAll`), subtracts it (`tl.sub`), applies `tl.exp`, sums (`tl.sum`), divides every lane by the
sum (`tl.div`), and stores the row of `y`. Each row is handled by exactly one program; programs write
disjoint rows and share nothing: the route is `TVA.meets_independent`, with `safe_forRange` and an
invariant over the trips (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1` from `grid_lt`);
- each program is `Safe`: the loop is `tl.RangeOk` and does not overflow (`rows_grid`); each trip's
  loads and store are in bounds (`x_len`, `y_len`, `cols_le`), with `int32` row offsets that do not
  wrap (`in_stride_lt`, `out_stride_lt`, `in_off`, `out_off`); the trips' write footprints (the
  rows' output elements) are disjoint (`out_rows`); the rules are `safe_load`, `safe_maxAll`,
  `safe_sub`, `safe_math` (with `exp_total`), `safe_sum`, `safe_div` (with `div_total`),
  `safe_store`;
- the programs' write footprints are disjoint (different rows, `out_rows`), and no program reads
  what another writes (`y_x`);
- every row has at least `128` columns (`cols_dim`), so a `RowOk` row has a finite value and the
  row's max is finite; the mask still matters: `BLOCK_SIZE` may exceed `n_cols` (`384` columns in a
  block of `512`);
- parts 1 and 2 from the loads and stores of each row's trip (`grid_pos`: every row has a program);
- part 3 on a `RowOk` row: the max is a finite input; `exp_acc` and `exp_ninf` bound each lane;
  `sum_depth` with `SumDepth.depth_le` and `sum_err_depth` bound the sum and give `sc`; `div_full`
  bounds each quotient (README, Tolerances, gives the derivation, including lanes where `x − max`
  overflows);
- the bytes no program writes give the frame (part 4).

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `idx_lt` (line 284) can be reused.

## Specification

Element `j` of row `r` is `inIdx d r j = r·input_row_stride + j` of `x` and
`outIdx d r j = r·output_row_stride + j` of `y`. `Post d init fin`, for every row `r < n_rows`, with
`xs k` the row's input values read in `init` (`k < n_cols`):
- every output element `y[r, j]`, `j < n_cols`, holds a `float32` value;
- if the row is `RowOk` (every value a finite `float32` or `−∞`, and one value finite when
  `n_cols > 0`), there is **one factor `sc` for the whole row**, `|sc − 1| ≤ sumTol(BLOCK_SIZE)`,
  such that every output `c_j` is finite and

  `|c_j − sc·σ_j| ≤ (elemTol + gapTol·gap_j)·σ_j + floorTol·σ_max`,

  where `σ_j = softmaxRef xs n_cols j = e^(x_j) / Σ_k e^(x_k)` is the exact softmax in the reals
  (`e^(−∞) = 0`: `expIn`), `σ_max = softmaxMax` the row's largest `σ_k`, and
  `gap_j = softmaxGap = log(σ_max/σ_j)`, which is `x_max − x_j` for a finite `x_j` (`0` for `−∞`);
- frame: every byte outside the output elements (`outSet`) is unchanged: the input, the padding of
  `y`'s rows, every other byte.

| Name | Value | Size |
|---|---|---|
| `sumTol B` | `(⌈log₂ B⌉ + 56)·2^-24` | `3.8·10^-6` at `B = 256`, `3.9·10^-6` at `1024`, `4.2·10^-6` at `2^14` |
| `elemTol` | `2^-21 + 2^-30` | `4.8·10^-7` |
| `gapTol` | `3·2^-24 + 2^-30` | `1.8·10^-7` per unit of `x_max − x_j` |
| `floorTol` | `2^-133` | relative to `σ_max ≥ 1/n_cols` |

Only the row sum's error grows with `BLOCK_SIZE` (as `log₂ BLOCK_SIZE`), and it is one number shared
by the row (the outputs are `e_j / S`), so `Post` states it once, as `sc`, and holds each output to a
per-element relative bound that does not grow with `BLOCK_SIZE`. Consequences: `|c_j − σ_j| ≤ (sumTol + elemTol +
gapTol·gap_j)·σ_j + floorTol·σ_max`, and the ratio of two outputs of a row is within about
`2·elemTol + gapTol·(gap_i + gap_j)` of `σ_i/σ_j`. A result rounded to `float16` or `bfloat16` does
not meet `Post` at any covered `BLOCK_SIZE ≥ 2`: on the row `[0, −1]`, `Post` needs the two outputs'
relative errors to differ by at most `≈ 1.1·10^-6`, and their `float16` roundings differ by
`5.2·10^-4` (`bfloat16`: `3·10^-3`). Small outputs are held relatively: in a row `[20, 0, …, 0]` the
outputs `σ ≈ 2·10^-9` must be within about `4·10^-6` of `sc·σ`, relatively.

A row that is not `RowOk` (a NaN or `+∞`, or every value `−∞`) has no defined softmax; `Post` asks
only that its outputs hold values.

`Spec K := Claim.Meets Inst.ok Pre Post K`.

## Conditions

`Inst`: `n_rows`, `n_cols`, `input_row_stride`, `output_row_stride` (elements), the constexprs
`BLOCK_SIZE` and `num_stages`, `num_programs` (the grid `(num_programs, 1, 1)`), the `float32`
buffers `x`, `y`.

`Inst.ok`:

| Field | Condition | Why |
|---|---|---|
| `block_pow2`, `block_le` | `BLOCK_SIZE` a power of two, `≤ 2^14` | `tl.arange`; `2^14` is the file's largest block (`N = 12672`) |
| `cols_le` | `n_cols ≤ BLOCK_SIZE` | a row fits in one block (any such `BLOCK_SIZE`, not only `next_power_of_2`) |
| `cols_dim` | `0 < n_cols`, `n_cols % 128 = 0`, `n_cols ≤ 2^17` | `n_cols` is a model dimension (the softmax's width): real model shapes only (Exclusions) |
| `grid_pos` | `n_rows ≠ 0 → 0 < num_programs` | with no program nothing is written; the host's `min(NUM_SM·occupancy, n_rows)` is positive |
| `grid_lt`, `rows_grid` | `num_programs < 2^31`, `n_rows + num_programs ≤ 2^31` | `Launch.Valid`; `n_rows` is an `int32`, and the `int32` loop `tl.range(row_start, n_rows, row_step)` does not overflow its last increment (`tl.RangeOk`) |
| `in_stride_lt`, `out_stride_lt` | strides `< 2^31` | `int32` arguments |
| `in_off`, `out_off` | `(n_rows − 1)·stride < 2^31` | the `int32` products `row_idx * stride` do not wrap |
| `x_len`, `y_len` | every element read (written) is in `x` (`y`) | |
| `out_rows` | `n_rows ≤ 1 ∨ n_cols ≤ output_row_stride` | output rows do not overlap (else programs race) |
| `y_x` | the output elements share no byte with the input elements | a program reading bytes it or another writes is not well behaved; the host's `y` is fresh |

The four integers are runtime `int32` arguments, or the compile-time constant `1` when equal to `1`,
which each use makes an `int32` tensor (the same model). Covered: every shape with these bounds, any
strides (rows may be slices of larger tensors), `n_rows = 0`, and any number of
programs from `1` (programs beyond `n_rows` run no trip). For a contiguous input, `in_off` covers
`16384`-column rows up to `131072` rows and `12672`-column rows up to `169467` rows.

`Pre`: both buffers are valid (`Buf.Valid`). Every `float32` input value is covered; `Post` treats
rows that are not `RowOk` separately.

The launch: `Launch.grid1 num_programs [[BLOCK_SIZE]]`, `enable_fp_fusion = true`, `debug = false`.
Part (A) of `Compiler.Faithful`'s scope (`docs/rules/compiler-faithful.md`): (4) `warmup` gives the
constexprs and the options `num_stages`, `num_warps=8` only; no `triton.compile`,
`@triton.autotune`, `@triton.heuristics`; a bare `@triton.jit`; (5) no tensor descriptor; (6) the
`CompiledKernel` is `warmup`'s, from the kernel's own source, launched with exactly the warm-up's
arguments (same dtypes, pointers and integers), so in its specialisation; (8) no host
`TensorDescriptor`. The claim relies on part (B), the environment, which is assumed and
not checked; in particular scope point 1: the benchmark's first launch per call runs on a new stream
not ordered after the `torch.randn` that writes `x`, and the claim covers it only once `x` is
written.

The kernel is inside the supported subset (`docs/SEMANTICS.md`): a top-level `def` under a bare
`@triton.jit`, two pointers, four unannotated integers, two `tl.constexpr`s; one
`for row_idx in tl.range(…)` with nothing after it; it reads only `triton` and `tl`.

## Assumptions

Seven `[rule]` fields, each the library's fact as `docs/rules/num.md` states it:
`faithful` (`Compiler.Faithful`), `exp_total` (`Num.MathTotal P .exp .binary32`), `exp_acc`
(`Num.ExpF32`: relative `2^-22 + 2^-23·X` on `[−X, X]`, absolute `2^-140`), `exp_ninf`
(`Num.ExpF32NegInf`: `tl.exp(−∞) = +0`), `div_total` (`Num.DivTotal P .binary32`), `div_full`
(`Num.DivFull`: relative `2^-22`, absolute `2^-148` for quotients up to `2^126`), `sum_depth`
(`Num.SumDepth`: the `tl.sum` of line 104 along a tree of depth at most `⌈log₂ BLOCK_SIZE⌉`; its
result is stored, and no loop updates a carried value with a reduction, the loop form the library
excludes). The subtraction, the max and the sum are IEEE operations. No
`[kernel]` field.

## Coverage

`setup_bench_4096x256` … `setup_bench_4096x12672` (98), with `setup_*_ok`: each benchmark launch
of `SETUP.md` with its `BLOCK_SIZE`, `num_stages = 4` and the H200's `num_programs`. The test's
launch (`1823 × 781`) is not covered: `781` columns is not a multiple of `128` (`cols_dim`;
Exclusions). `mk_ok` covers a contiguous `R × C` launch, `C` a positive multiple of `128`, for every
`num_programs` from `1` to `R`,
every `BLOCK_SIZE = 2^e` from `C` to `2^14`, and any `num_stages`.

## Tolerances

From the library's facts, per `RowOk` row with `B = BLOCK_SIZE`, `u = 2^-24`:
- the max `m` is a finite input (`tl.maxAll` along a tree of `maximum`; masked lanes are `−∞`), and
  every lane is `≤ m`;
- each lane's `s = fl(x − m) ≤ 0` is one IEEE rounding (`u·|x − m|`), or `−∞` when `x − m` overflows
  or `x = −∞`; `tl.exp(−∞) = +0` (`ExpF32NegInf`); otherwise `ExpF32` at `X = |s|` gives `e^s` within
  `(2^-22 + 2^-23·|s|)·e^s + 2^-140`; with the rounding of `s` (a factor `e^(t−s)`, `|t − s| ≤ u·|s|`)
  this is within `(2^-22 + 2^-33)·e^s + (3u + 2^-33)·|s|·e^s + 2^-139` of `e^(x−m)` for
  `|s| ≤ 1024`, and below `2^-139` beyond;
- the lane at `m` gives `e^0 = 1` within `2^-22`, so the exact sum `A ≥ 1`; with
  `|s|·e^s ≤ 16·e^s + 2^-17`, the lanes' errors add up to at most `53u·A`;
- the sum `S` (`tl.sum`, IEEE additions along a tree of depth at most `D = ⌈log₂ B⌉`: `SumDepth`,
  `SumDepth.depth_le`, `sum_err_depth`) is within `53u·A + ((1+u)^D − 1)·(1 + 53u)·A + 2^-150·(B − 1)·(1+u)^D`
  of `A`, below `(D + 53.001)·u·A` (`D ≤ 14`, `A ≥ 1`); so `sc = A/S` is within
  `(D + 53.001)·u/(1 − 68u) ≤ (D + 54)·u` of `1`, inside `sumTol = (D + 56)·u`;
- `DivFull` on `e_j / S` (quotient `≤ 2`, far inside `2^126`): relative `2^-22`, absolute `2^-148`.
  Together: `|c_j − sc·σ_j| ≤ (2^-21 + 2^-30 + (3u + 2^-30)·|s_j|)·σ_j + 2^-133/A`, with
  `|s_j| ≤ (1+u)·gap_j` absorbed in `gapTol` and `1/A = σ_max`: `elemTol`, `gapTol`, `floorTol`;
- a lane whose `x − m` overflows to `−∞` (inputs near `±3.4·10^38` of opposite signs) has
  `gap_j ≥ 2^127`: its output is within `2^-148 ≤ floorTol·σ_max` of `0` (`σ_max ≥ 2^-14`), and
  `sc·σ_j ≤ gapTol·gap_j·σ_j`, so the same bound holds; no bound on the input magnitudes is needed.

Meaningful: the per-element part does not grow with `B` and is a few `float32` roundings
(measured worst error / tolerance `0.57`); a half-precision result fails it (Specification). The
common factor's bound `sumTol` is about `66` `float32` roundings at every covered `B`: a
denominator that drops one column (outputs `e_j/(A − e_last)`, exactly `sc·σ_j` with
`sc = A/(A − e_last)`) fails `Post` on every row of the test's shape and on about `99%` of the rows at
`4096 × 12672` (`check_gpu.py`).

## Exclusions

- **Dimensions that are not multiples of 128 or exceed 2^17** — out of scope: the claim covers real
  model shapes only. The field: `n_cols` (`cols_dim`: a positive multiple of `128`, at most `2^17`).
  Among the file's launches this leaves out the test (`1823 × 781`); the benchmark's `N = 128·i` are
  covered. `n_rows` is not bounded as a context length: a softmax's rows are not only tokens (an
  attention softmax has a row per head and query).
- **`BLOCK_SIZE > 2^14`** (rows of more than `16384` columns) — outside the file's launches
  (largest `12672` columns); the constants of the derivation (Tolerances) use `B ≤ 2^14`.
- **Rows that are not `RowOk`** — a limit of the specification, stated per row: with a NaN, a `+∞`,
  or only `−∞`, the exact softmax is undefined (the kernel gives NaN); such a row's outputs need only
  hold values. Real use (`torch.randn`) never produces them.
- **`n_rows + num_programs > 2^31`, `(n_rows − 1)·stride ≥ 2^31` with a stride below `2^31`** — a
  kernel limit (`int32` arguments and products): beyond them `row_idx * input_row_stride` wraps
  (first launch: a contiguous `131073 × 16384` input) or the loop's last increment overflows. The
  file's largest is `4096 × 12672` (`4095·12672 < 2^26`).
- **Strides `≥ 2^31`** — outside real use (rows `2^31` elements apart; the file's rows are
  contiguous). Not a kernel limit: Triton passes such a stride as an `int64` argument, and the
  product `row_idx * stride` is then an `int64` that does not wrap; the claim's model takes the
  strides as `int32` arguments.
- **`num_programs = 0` with rows** — outside the file's launches; nothing would be written.
- **Overlapping output rows, or output overlapping the input** — outside the file's launches (`y` is
  fresh); programs would race, or a program would conflict with itself.
- **Dtypes other than `float32`** — outside the file's launches.

## Library requests

None.

## Checks

- `Claim.lean` builds with no errors or warnings, no `sorry`/`axiom`; `consistent`, `satisfiable`
  and every `setup_*_ok` use only `propext`, `Classical.choice`, `Quot.sound`.
- `check_gpu.py` (H200): the test and all 98 benchmark shapes through the host; edge cases (rows
  with `−∞`, near-overflow rows where `x − m` overflows, gaps up to `1000`, subnormal inputs, rows
  that are not `RowOk`, `1`, `2`, `100`, `16384` columns, larger `BLOCK_SIZE`, `1` to `n_rows`
  programs, strided rows in a sentinel buffer). The test's shape and the edges with `1`, `2`, `8`,
  `100` or `1000` columns are outside `cols_dim`: they check `Post` beyond the claim. Worst error /
  tolerance over every `RowOk` row: `0.40` on the test, `0.39`–`0.46` on the benchmark, `0.57`
  overall; every `RowOk` output finite; inputs unchanged; frame holds. The host's output rounded to
  `float16` / `bfloat16`, and a denominator that drops the row's last column, fail `Post`. 112 of
  112 cases pass.
