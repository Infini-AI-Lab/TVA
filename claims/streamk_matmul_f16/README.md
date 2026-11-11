# `streamk_matmul_f16`

**The kernel.** `streamk_matmul_kernel` of `source/streamk_matmul.py` (kernel lines 26–64; host
`streamk_matmul` lines 70–93). The host multiplies `float16` matrices `a` (`M × K`) and `b`
(`K × N`) into a fresh `float32` `c = torch.zeros(M, N)` on a 1-D grid of `num_programs` programs
(the SM count unless the caller passes one), with `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 64`. The
`num_tiles · cdiv(K, 64)` iterations of all tiles' `K` loops are split into contiguous ranges of
`iters_per_program`, one per program; a tile one program owns entirely is stored, a tile split
across programs is accumulated into `c` with `tl.atomic_add` by each program that owns part of it.
The claim is `Claims.StreamKMatmul.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
`i < M`, `j < N`:
- element `(i, j)` of `c` holds some `float32` value;
- when row `i` of `a` and column `j` of `b` hold finite values, it is finite and within
  `tol = 2^-15·tolMag + 2^-100` of the exact `S = Σ_k a[i,k]·b[k,j]`, whatever the split of the tile
  between programs and the order of the atomic additions. With chunk `t` the products
  `k ∈ [64t, min(64t + 64, K))` (one `tl.dot`), `pre s` the exact sum of the first `s` chunks,
  `dev t = max_{s ≤ t} |pre t − pre s|` and `chunkMag t = Σ_{k ∈ chunk t} mag16(a)·mag16(b)`
  (`mag16 q = |q|`, or `2^-14` for a subnormal), `tolMag = Σ_{t ≤ T} dev t + Σ_{t < T} chunkMag t`,
  `T = cdiv(K, 64)`.

Every byte outside the elements of the matrix `c` is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `M`, `N`, `K ≥ 1`; `N` and `K` multiples of `128`, at most `2^17`; `M ≤ 2^22`; `K ≤ 2^13`;
- `M + 127 ≤ 2^31`, `N + 128 ≤ 2^31`; every stride below `2^31`, `stride_cn > 0`, and the `int32`
  products `(M − 1)·stride_am`, `(K − 1)·stride_ak`, `(K − 1)·stride_bk`, `(N − 1)·stride_bn`,
  `(M − 1)·stride_cm` below `2^31`;
- `N·stride_cn ≤ stride_cm` (the rows of `c` do not overlap);
- `total_iters = cdiv(M, 128)·cdiv(N, 128)·cdiv(K, 64)` (the host's), the ranges cover every
  iteration (`total_iters ≤ num_programs·iters_per_program`), and
  `num_programs·iters_per_program < 2^31`;
- each buffer holds its matrix and is valid in the initial memory; every element of the matrix `c`
  is zero; the matrix `c` shares no byte with the matrices `a` and `b`.

Every `float16` input is covered, NaN and infinities included (for an element whose row or column
holds one, only "holds a value" is claimed). Any strides and program counts within the bounds are
covered.

## Coverage

The test launches of `SETUP.md` whose `N` and `K` are multiples of `128`, with the H200's `132`
programs: `512³`, `3072 × 4096 × 1024`, `8192 × 8192 × 4096` (`setup_*_ok`; contiguous `a`, `b`, a
fresh `c`, the host's `total_iters` and `iters_per_program`). `mk_ok` covers the host's launch for
every shape and program count within the bounds. The test's `1000³`, `257 × 129 × 4097` and `64³`
are not covered (Exclusions).

**Device facts** the tolerance rests on (in `Device P`): `Device.DotF16` (`tl.dot` of `float16`
operands into `float32`: within `relDot16 K·(|acc| + Σ liftMag a·liftMag b) + absDot16 K` of
`acc + Σ a·b`), `Device.AtomicAddF32` (`tl.atomic_add` on `float32`: within `2^-23·|o + v| + 2^-124`
of the exact sum), and the totality of both.

## Tolerances

With `ρ = relDot16 64 = 66·2^-22`, `α = absDot16 64 = 66·2^-126`, `C = Σ_{t<T} |chunkSum t|`:
- a program's part of a tile runs chunks `s ≤ t < e` from `+0`; its exact value before chunk `t` is
  `pre t − pre s`, at most `dev t` in magnitude, so its error obeys
  `E_{t+1} ≤ (1 + ρ)·E_t + ρ·(dev t + chunkMag t) + α`; the parts of a tile cover `0 … T−1`, so
  their errors add up to at most `tolDot = Σ_{t<T} (1 + ρ)^(T−1−t)·(ρ·(dev t + chunkMag t) + α)`;
- a split tile's element receives at most `T` atomic additions into the zero `c`, in any order; with
  `AtomicAddF32` per step, against `Σ partials` (magnitude at most `C + tolDot`), the error is at most
  `((1 + 2^-23)^T − 1)·(C + tolDot) + T·(1 + 2^-23)^T·2^-124`;
- with `K ≤ 2^13` (`T ≤ 128`), `(1 + ρ)^127 < 1.0021`, and `Σ_{t<T} dev t + chunkMag t ≥ 2C −
  |chunkSum (T−1)|`, `dev T ≥ |chunkSum (T−1)|`, the sum of the two is at most
  `2^-15·tolMag + 2^-100 = tol`.

On the test's launches the worst error / tolerance is `0.0004`–`0.0051`; a `bfloat16` result, a
lost chunk, or a `float16` accumulation fails `Post`.

## Exclusions

- **`N` or `K` not a multiple of `128` or above `2^17`, `M > 2^22`**: the claim covers real model
  shapes only. This leaves out the test's `1000³`, `257 × 129 × 4097` and `64³`.
- **`K > 2^13`**: outside the file's launches (largest `4097`); the tolerance grows with `K`.
- **`int32` bounds** (`M_le`, `N_le`, the stride products, `progs_lt`): a kernel limit; the offsets
  wrap once a matrix has about `2^31` elements. Real use is far inside.
- **Zero sizes**, **`total_iters` or ranges not the host's**, **a nonzero `c`**, **column-major or
  overlapping layouts of `c`**, **`c` overlapping `a` or `b`**, **other block sizes or dtypes**:
  outside the file's launches.

`check_gpu.py` (one H200) checks `Post` on the test's launches and edge cases, and that wrong results
break it.
