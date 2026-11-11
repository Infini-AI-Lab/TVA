# `tma_persistent_matmul_f16`

**The kernel.** `matmul_kernel_tma_persistent` of Triton 3.8.0's tutorial 09, "Persistent Matmul"
(`source/09-persistent-matmul.py`, kernel lines 407–463, helper `_compute_pid` lines 291–298; host
`matmul_tma_persistent(a, b, False)` lines 466–503), on `float16` matrices with the config of
`SETUP.md` (`BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`,
`EPILOGUE_SUBTILE = True`, `num_stages = 4`, `num_warps = 8`, `NUM_SMS = 132` on the H200). The
host builds three `TensorDescriptor`s and launches `min(NUM_SMS, tiles)` programs; each loops over
the `128 × 256` tiles `start_pid, start_pid + NUM_SMS, …` of `c = a · bᵀ`, accumulates the tile
with `tl.dot` over `64`-column blocks of `K`, and stores it as two `128 × 128` `float16` halves. The
claim is `Claims.TmaPersistentMatmul.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
`i < M`, `j < N`:
- element `(i, j)` of `c` holds some `float16` value;
- when row `i` of `a` and row `j` of `b` hold finite values, that value is the `float16` rounding
  (to nearest even, up to NaN payload) of a finite `float32` value `v` with
  `|v − S| ≤ tol = 2^-15·pathMag + 2^-112`, where `S = Σ_k a[i, k]·b[j, k]` is the exact dot
  product and `pathMag = Σ_{t < cdiv(K, 64)} (|S_t| + L_t)` sums, over the inner loop's steps, the
  exact partial sum `S_t` before the step and the step's lifted magnitude `L_t`
  (`Σ liftMag(a)·liftMag(b)` over its 64 columns);
- when `|S| + tol ≤ 65504`, the value is finite.

Every byte outside the elements of the matrix `c` is unchanged.

**The conditions** (`Inst.ok`, `Pre`):
- `M`, `N`, `K ≥ 1`; `N` and `K` multiples of `128`, at most `2^17`; `M ≤ 2^22`; `K ≤ 2^13`
  (see Exclusions);
- `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31`, `0 < NUM_SMS < 2^31`, `tiles + NUM_SMS ≤ 2^31`: the `int32`
  `tl.cdiv`s, `num_tiles` and the tile loop's bounds do not wrap;
- row strides multiples of `8` elements and bases 16-byte aligned (the host's `TensorDescriptor`
  asserts both); `N ≤ ldc`;
- each buffer holds its matrix and is valid in the initial memory; the matrix `c` shares no byte
  with the matrices `a` and `b` (the host's `c` is fresh). `a` and `b` may overlap.

Every `float16` input is covered, NaN and infinities included (for an element whose rows hold one,
only "holds a value" is claimed). Partial row tiles (any `M`), padded rows and any `NUM_SMS` within
the bounds are covered.

**The launches covered** (`SETUP.md`): `M = N = 8192`, `K = 128, 256, …, 1024`, `NUM_SMS = 132`
(`setup_k*_ok`). The test's `validate(32, 32, 32)` is not covered (`N`, `K` not multiples of `128`).

**Device facts** the tolerance rests on (in `Device P`): `Device.DotF16` (`tl.dot` of `float16`
operands into `float32`: within `relDot16 K·(|acc| + Σ liftMag a·liftMag b) + absDot16 K` of
`acc + Σ a·b`), and the totality of `tl.dot`. The `float16` conversions are IEEE roundings; the
reshape, permute and split of the epilogue are exact.

## Tolerances

Per element, with `ρ = relDot16 64 = 66·2^-22` and `α = absDot16 64 = 66·2^-126`, the inner loop's
`T = cdiv(K, 64)` steps give the bound `E_0 = 0`, `E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + L_t) + α`
against the exact partial sums (each step is one `float16` `tl.dot` of `64` products with the
accumulator; the lanes past `K` load `+0`). So `E_T ≤ (1 + ρ)^T·(ρ·pathMag + T·α)`. With
`K ≤ 2^13`, `T ≤ 128` and `(1 + ρ)^128 ≤ 1.0021`, so `(1 + ρ)^T·ρ ≤ 2^-15` and
`(1 + ρ)^T·T·α ≤ 2^-112`: `E_T ≤ tol`. On the setup's `torch.randn` inputs `tol/|S|` has median
`8.3·10^-4`; a result rounded to `bfloat16`, a missing `K` block, or a wrong tile mapping fails
`Post`.

## Exclusions

- **`N` or `K` not a multiple of `128` or above `2^17`, `M > 2^22`**: the claim covers real model
  shapes only. Among the file's launches this leaves out the test `validate(32, 32, 32)`.
- **`K > 2^13`**: the tolerance argument is stated for `K ≤ 2^13` (the setup's largest is `1024`).
- **`int32` wrap** of `tl.cdiv`, `num_tiles` or the tile loop's bound (`M_le`, `N_le`,
  `tiles_le`): a kernel limit, far beyond real use.
- **Zero sizes, unaligned bases or strides**: the host's `TensorDescriptor` rejects them.
- **`NUM_SMS` outside `[1, 2^31)`**, **`warp_specialize = True`**, **`float8` operands**, other
  configs, **an output whose rows overlap or that overlaps `a` or `b`**: outside the file's
  launches.

`check_gpu.py` (one H200) checks `Post` on the launches of `SETUP.md` and on edge cases (largest
error / tolerance `0.0042`), and that wrong results break it.
