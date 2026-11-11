# `matmul_f16`

**The kernel.** `matmul_kernel` of Triton 3.8.0's tutorial 03, "Matrix Multiplication"
(`source/03-matrix-multiplication.py`, kernel lines 228–316; host `matmul` lines 330–350), on
`float16` matrices with the config of `SETUP.md` (`BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`,
`BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`). A 1-D grid of `cdiv(M, 128)·cdiv(N, 256)` programs; each
maps its id to a tile (grouped by `GROUP_SIZE_M`), loops over `K` in steps of `64` accumulating
`tl.dot`s into a `float32` accumulator, rounds to `float16` and stores its `128 × 256` tile of `c`
(masked by `M`, `N`). The claim is `Claims.Matmul.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
`i < M`, `j < N`:
- element `(i, j)` of `c` holds a `float16` value `c`;
- if row `i` of `a` and column `j` of `b` hold finite values `va k`, `vb k` (`k < K`), then `c` is
  the `float16` rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` with
  `|v − S| ≤ tol = 2^-15·pathMag + 2^-100`, where `S = Σ_k va k·vb k` is the exact dot product
  (Tolerances); and when `|S| + tol ≤ 65504`, `c` is finite and
  `|c − S| ≤ 2^-11·|S| + 2^-15·pathMag + 2^-24`;
- every byte outside the elements of the matrix `c` is unchanged.

With a NaN or an infinity in row `i` of `a` or column `j` of `b`, only that element `(i, j)` holds
some `float16` value is claimed.

## Conditions

`Inst.ok` and `Pre`:
- `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31`, fewer than `2^31` programs;
- `K ≤ 2^13`; `N` and `K` are positive multiples of `128`, at most `2^17`; `M ≤ 2^22`;
- every stride is positive (the `tl.assume`s of lines 268–273) and below `2^31`, and the `int32`
  offsets of lines 285, 286 and 314, with the loop's steps, do not wrap (`a_fit`, `b_fit`,
  `cm_fit`, `cn_fit`);
- distinct elements of the matrix `c` are distinct elements of its storage; each buffer holds its
  matrix; the matrix `c` shares no byte with `a` or `b` (the host's `c` is fresh);
- the three buffers are valid in the initial memory (aligned, every byte present).

Any positive strides within `int32` (a transposed `b`, a padded or column-major `c`) and `M = 0` are
covered; `a` and `b` may overlap. Every `float16` input is covered, NaN and infinities included.

**The launches covered** (`SETUP.md`): the test (`512 × 512 × 512`) and the benchmark's 31 square
sizes `256, 384, …, 4096`, with the host's strides (`setup_*_ok`, through `hostInst_ok`).

**Device facts** the tolerance rests on (in `Device P`): `Device.DotF16` (a `float16` `tl.dot`
into `float32`: each finite output within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with
`L = |acc| + Σ liftMag a·liftMag b`, a subnormal operand counted at `2^-14`) and the totality of
`tl.dot`. The conversions to `float16` are IEEE roundings.

## Tolerances

With `p e = va e·vb e` for `e < K` and `0` beyond (`prodV`), `l e = liftMag(va e)·liftMag(vb e)`
(`liftV`), `T = cdiv(K, 64)` steps, `S_t` the exact partial sum before step `t` (`partSum`) and
`L_t` the lifted magnitude of step `t` (`stepMag`): `pathMag = Σ_{t<T} (|S_t| + L_t)`.

The fine bound: step `t` of the loop (line 300) is one `tl.dot` of `64` products into the
accumulator, so by `DotF16` the error follows `E_0 = 0`,
`E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + L_t) + α`, with `ρ = relDot16 64 = 66·2^-22` and
`α = absDot16 64 = 66·2^-126`. `K ≤ 2^13` keeps every step far inside the `float32` range. The
final `.to(tl.float16)` (line 308) is one rounding (`u = 2^-11`, `η = 2^-25`).

The relaxation: `T ≤ 128` and `(1 + ρ)^128 < 1.0021`, so `E_T ≤ 2^-15·pathMag + 2^-100 = tol`, and
`E_T + (|S| + E_T)·2^-11 + 2^-25 ≤ 2^-11·|S| + 2^-15·pathMag + 2^-24`. A result rounded to
`bfloat16`, a `float16` accumulator, or a missing step breaks `Post`.

## Exclusions

None of `SETUP.md`'s launches is left out.

- **`N`, `K` not positive multiples of `128` or above `2^17`, `M > 2^22`**: the claim covers real
  model shapes only.
- **`K > 2^13`**: outside the setup; the tolerance is stated for `K ≤ 2^13`.
- **Strides `≥ 2^31`, `int64` arguments**: outside real use.
- **A zero or negative stride** (the `tl.assume`s are false), **`int32` wrap of the offsets**,
  **overlapping elements of `c`**, **`c` overlapping `a` or `b`**, **more than `2^31 − 1`
  programs**: kernel limits, not reached by the host.
- **`ACTIVATION = "leaky_relu"`, other configs, `float8e5` operands**: not part of the setup.

`check_gpu.py` (one H200) checks `Post` on the setup's launches, edge cases and wrong kernels.
