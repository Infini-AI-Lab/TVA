# `matmul_e5m2`

**The kernel.** `matmul_kernel` of Triton 3.8.0's tutorial 03, "Matrix Multiplication"
(`source/03-matrix-multiplication.py`, kernel lines 228–316; host `matmul` lines 330–350), in the
script's `float8e5` run (`ACTIVATION = ""`, the config of `SETUP.md`: blocks `128 × 256 × 64`,
`GROUP_SIZE_M = 8`). A 1-D grid of `cdiv(M, 128)·cdiv(N, 256)` programs, grouped along `M`; each
loops over `cdiv(K, 64)` steps of `tl.dot` of `float8e5` tiles into a `float32` accumulator,
rounds it to `float16` and stores its tile of `c`. The claim is `Claims.MatmulE5m2.claim` in
`Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- every element `(i, j)` of `c` holds a `float16` value `c`;
- when row `i` of `a` and column `j` of `b` hold finite `float8e5` values, `c` is the `float16`
  rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` with
  `|v − S| ≤ E`, where `S = Σ_k a[i, k]·b[k, j]` is the exact dot product and `E = tol` (see
  Specification); and when `|S| + E ≤ 65504`, `c` is finite and
  `|c − S| ≤ E + (|S| + E)·2^-11 + 2^-25`;
- every byte outside the matrix `c` is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31`, and fewer than `2^31` programs;
- `K ≤ 2048` (see Exclusions); `M ≤ 2^22`; `N` and `K` positive multiples of `128`, at most `2^17`;
- each stride positive (the kernel's `tl.assume`s) and below `2^31`, and the `int32` offsets of
  lines 285, 286, 302–303 and 314 do not wrap;
- distinct elements of the matrix `c` are distinct elements of its storage; each buffer holds its
  matrix; the matrix `c` shares no byte with the matrices `a` and `b`; the buffers are valid.

Any positive strides and layouts are covered, and `M = 0`. Every `float8e5` input is covered, NaN
and infinities included.

**The launches covered** (`SETUP.md`): the test (`512 × 512 × 512`) and the benchmark's `fp8`
line at `M = N = K = 256, 384, …, 2048` (`setup_*_ok`); `hostInst_ok` covers the host's layout at
every size the conditions allow.

**Device facts** the tolerance rests on (all in `Device P`): `Device.DotE5m2ChunkLoss` (`tl.dot`
of `float8e5` operands into `float32`, chunk by chunk of `32` products), `Device.CvtF32E5m2` (the
`other=0.0` of the masked loads converted to `float8e5`), and the totality facts for that dot and
that conversion.

## Specification

`tol va vb K = errAt (prodV va vb K) (liftV va vb K) (2·steps K)`: the device fact applied chunk by
chunk against the exact partial sums. The loop's `cdiv(K, 64)` `tl.dot`s add `n = 2·cdiv(K, 64)`
chunks of `32` products (the masked lanes are `0`). With `S_j` the exact partial sum before chunk
`j`, `m_j` the chunk's largest lifted product (`liftMag a·liftMag b`, a subnormal operand counting
as `2^-14`), `A_j` the chunk's exact `Σ|a·b|` and `X_j = max(|S_j|, m_j)`:
`E_0 = 0`, `E_{j+1} = (1 + σ)·E_j + min(ρ·(X_j + E_j), κ·A_j + ρ'·(X_j + E_j)) + σ·|S_{j+1}| + α`,
with the fact's constants `ρ = 2^-7`, `σ = 2^-12`, `α = 2^-126`, `κ = 2`, `ρ' = 2^-12`. `tol` is
never above `tolA K L = ((1 + ρ + σ)^n − 1)·L + n·(1 + ρ + σ)^n·α`, the first arm alone
(`L` the sum of the lifted products); where the partial sums stay large next to the chunks'
magnitudes the second arm makes `tol` far smaller.

## Tolerances

The relative part of `tolA`, `(1 + 2^-7 + 2^-12)^n − 1`, is `0.137` at `K = 512` and `0.671` at
`K = 2048`. On the file's data (`randn` cast to `float8e5`) the median `tol / |S|` over the outputs
is `0.095` at `K = 512`, `0.20` at `K = 1024` and `0.44` at `K = 2048`: at the larger sizes `Post`
pins each output much less, but typical wrong results (all zero, a step left out, another column's
result) still fail it.

## Exclusions

- **The benchmark's `fp8` sizes with `2048 < K ≤ 4096`** (`K_le`): the fp8 `tl.dot` fact cannot give
  a meaningful tolerance there (its median over the outputs is `0.74` of the output at `K = 3072`,
  and a result `0.9` times the exact one passes `Post`). The kernel is believed correct there.
- **Shapes that are not real model shapes**: `N` or `K` not a positive multiple of `128` or above
  `2^17`, `M` above `2^22`.
- **Strides of `2^31` or more, `ACTIVATION = "leaky_relu"`, other block configurations, `float16`
  operands**: outside the setup.
- **Kernel limits** (not reached by real use): a zero or negative stride (`tl.assume` fails); the
  `int32` offsets wrapping once a matrix spans `2^31` elements; overlapping output elements or an
  output overlapping an input; more than `2^31 − 1` programs.

`check_gpu.py` (one H200) checks `Post` element by element on the setup's launches and on edge
launches.
