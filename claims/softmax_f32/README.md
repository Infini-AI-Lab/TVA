# `softmax_f32`

**The kernel.** `softmax_kernel` of Triton 3.8.0's tutorial 02, "Fused Softmax"
(`source/02-fused-softmax.py`, kernel lines 84–109; host `softmax` lines 124–175). The host launches
the `CompiledKernel` of `softmax_kernel.warmup(…)`; the claim covers the same launches made directly
(see Exclusions). A 1-D grid of `num_programs` programs; program
`p` loops over rows `p, p + num_programs, …` below `n_rows`, and for each row loads one block of
`BLOCK_SIZE` lanes (masked by `n_cols`, other lanes `-∞`), subtracts the row's max, applies
`tl.exp`, sums, divides every lane by the sum and stores the row of `y`. The claim is
`Claims.Softmax.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
row `r < n_rows`, with `xs` the row's initial inputs:
- every output element `y[r, j]`, `j < n_cols`, holds a `float32` value;
- if the row is `RowOk` (every value a finite `float32` or `-∞`, and one value finite), there is one
  factor `sc` for the whole row with `|sc − 1| ≤ sumTol = 2^-17` such that every output `c_j` is
  finite and `|c_j − sc·σ_j| ≤ elemTol·(gap_j + 4)·σ_j + floorTol`, with `elemTol = 2^-22`,
  `floorTol = 2^-133`, `σ_j = e^(x_j) / Σ_k e^(x_k)` the exact softmax in the reals (`e^(−∞) = 0`)
  and `gap_j = log(σ_max / σ_j)`, which is `x_max − x_j` for a finite `x_j`;
- every byte outside the output elements is unchanged.

A row that is not `RowOk` (a NaN, a `+∞`, or only `−∞`) has no defined softmax; its outputs need
only hold values.

**The conditions** (`Inst.ok`, `Pre`):
- `BLOCK_SIZE` is a power of two, at most `2^14`, and `n_cols ≤ BLOCK_SIZE` (a row fits in one
  block);
- `n_cols` is a positive multiple of `128`, at most `2^17` (see Exclusions);
- with a row to process there is a program; fewer than `2^31` programs, and
  `n_rows + num_programs ≤ 2^31` (the `int32` loop does not overflow its last increment);
- the strides are below `2^31`, and the `int32` products `(n_rows − 1)·stride` do not wrap;
- `x` holds every element read and `y` every element written, both valid in the initial memory;
- the output rows do not overlap (one row, or `n_cols ≤ output_row_stride`), and the output
  elements share no byte with the input elements (the host's `y` is fresh).

Any strides, `n_rows = 0`, and any number of programs from `1` are covered. Every `float32` input is
covered.

**The launches covered** (`SETUP.md`): the 98 benchmark launches `4096 × N`, `N = 256, 384, …,
12672`, each with its `BLOCK_SIZE`, `num_stages = 4` and the H200's `num_programs`
(`setup_*_ok`). The test's launch (`1823 × 781`) is not covered: `781` is not a multiple of `128`.
`mk_ok` covers a contiguous `R × C` launch, `C` a positive multiple of `128`, for every
`num_programs` from `1` to `R`, every `BLOCK_SIZE = 2^e` from `C` to `2^14`, and any
`num_stages`.

**Device facts** the tolerances rest on (all in `Device P`): `Device.ExpF32` (`tl.exp` on `float32`:
relative `2^-22 + 2^-23·X` on `[−X, X]`, absolute `2^-140`), `Device.ExpF32NegInf`
(`tl.exp(−∞) = +0`), `Device.DivFull` (`/` on `float32`: relative `2^-22`, absolute `2^-148`, for
quotients up to `2^126`), `Device.SumDepth` (the tree of a float `tl.sum` over `B` positions has
depth at most `⌈log₂ B⌉`), and the totality facts for `exp` and `/`. The subtraction, the max and
the sum's additions are IEEE operations.

## Tolerances

Per `RowOk` row, with `B = BLOCK_SIZE`, `D = ⌈log₂ B⌉ ≤ 14` and `u = 2^-24`:
- the max `m` is a finite input, and each lane's `s = fl(x − m) ≤ 0` is one IEEE rounding (or `−∞`
  when `x − m` overflows or `x = −∞`, where `tl.exp` gives `+0`);
- `ExpF32` at `X = |s|`, with the rounding of `s`, gives `e^(x−m)` within
  `(2^-22 + 2^-33)·e^s + (3u + 2^-33)·|s|·e^s + 2^-139` for `|s| ≤ 1024`, below `2^-139` beyond; the
  lane at `m` gives `1` within `2^-22`, so the exact sum `A ≥ 1`, and the lanes' errors add up to at
  most `53u·A`;
- the sum `S` (IEEE additions along a tree of depth at most `D`) is within `(D + 53.001)·u·A` of `A`,
  so `sc = A/S` is within `(D + 54)·u` of `1`;
- `DivFull` on `e_j / S` (quotient `≤ 2`): relative `2^-22`, absolute `2^-148`. Together:
  `|sc − 1| ≤ (D + 56)·u` and `|c_j − sc·σ_j| ≤ (2^-21 + 2^-30 + (3u + 2^-30)·gap_j)·σ_j + 2^-133·σ_max`;
  a lane where `x − m` overflows has `gap_j ≥ 2^127` and meets the same bound.

Relaxed to `Post`'s constants: `(D + 56)·u ≤ 2^-17 = sumTol`;
`2^-21 + 2^-30 + (3u + 2^-30)·gap_j ≤ 2^-22·(gap_j + 4)` since `gap_j ≥ 0`; `σ_max ≤ 1`, so
`2^-133·σ_max ≤ floorTol`. A result rounded to `float16` or `bfloat16`, or a denominator that drops
a column, fails `Post`.

## Exclusions

- **The host's launch path.** `softmax` launches the `CompiledKernel` that `softmax_kernel.warmup(…)`
  returns, which TVA does not cover (`docs/EXCLUSIONS.md`, H6: direct launches only). The claim
  covers the same launches made directly, `softmax_kernel[(num_programs, 1, 1)](y, x, x.stride(0),
  y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages, num_warps=8)`: the same kernel, grid,
  arguments and compile-time constants, so the same model.
- **`n_cols` not a multiple of `128`, or above `2^17`**: the claim covers real model widths only.
  Among the file's launches this leaves out the test (`1823 × 781`).
- **`BLOCK_SIZE > 2^14`**: outside the file's launches; the tolerances use `B ≤ 2^14`.
- **Rows that are not `RowOk`**: the exact softmax is undefined; only that outputs hold values is
  claimed.
- **`n_rows + num_programs > 2^31`, or `(n_rows − 1)·stride ≥ 2^31`**: a kernel limit (`int32`
  products wrap, or the loop's last increment overflows).
- **Strides `≥ 2^31`**: outside real use (Triton would pass an `int64`; the model takes `int32`).
- **`num_programs = 0` with rows**, **overlapping output rows or output overlapping input**,
  **dtypes other than `float32`**: outside the file's launches.

`check_gpu.py` (one H200) checks `Post` on the test, all 98 benchmark shapes and edge cases.
