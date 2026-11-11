# `persistent_matmul_f16`

**The kernel.** `matmul_kernel_persistent` of Triton 3.8.0's tutorial 09, "Persistent Matmul"
(`source/09-persistent-matmul.py`, kernel lines 301–357, with its `@triton.jit` helper
`_compute_pid`, lines 287–294; host `matmul_persistent` lines 360–380), on `float16` matrices with
the config of `SETUP.md` (`BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`,
`GROUP_SIZE_M = 8`, `NUM_SMS = 132` on the H200). A 1-D grid of `min(NUM_SMS, tiles)` programs;
program `start_pid` loops over the tiles `start_pid, start_pid + NUM_SMS, …`, and for each tile
accumulates `tl.dot`s of `[128, 64]` and `[64, 256]` blocks into a `float32` accumulator, rounds it
to `float16` and stores the `[128, 256]` tile of `c`. The claim is `Claims.PersistentMatmul.claim`
in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
`i < M`, `j < N`:
- element `(i, j)` of `c` holds a `float16` value `c`;
- if row `i` of `a` and column `j` of `b` hold finite values `va k`, `vb k` (`k < K`), then `c` is
  the `float16` rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` with
  `|v − S| ≤ tol`, where `S = Σ_k va k·vb k` is the exact dot product and
  `tol = 2^-15·accMag + 2^-112` (Tolerances); and when `|S| + tol ≤ 65504`, `c` is finite and
  `|c − S| ≤ tol + (|S| + tol)·2^-11 + 2^-25`;
- every byte outside the elements of the matrix `c` is unchanged.

With a NaN or an infinity in row `i` of `a` or column `j` of `b`, only that element `(i, j)` holds
some `float16` value is claimed.

## Conditions

`Inst.ok` and `Pre`:
- `M % 128 = 0` and `N % 256 = 0` (whole tiles; see Exclusions);
- `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31`, `0 < NUM_SMS < 2^31`, `tiles + NUM_SMS ≤ 2^31` (the `int32`
  tile arithmetic and the tile loop's bounds do not wrap);
- `K ≤ 2^13`; `N` and `K` are positive multiples of `128`, at most `2^17`; `M ≤ 2^22`;
- every stride is below `2^31`, and the `int32` offsets of the enabled lanes of lines 340, 341 and
  351 do not wrap (`a_fit`, `b_fit`, `cm_fit`, `cn_fit`);
- distinct elements of the matrix `c` are distinct elements of its storage; each buffer holds its
  matrix; the matrix `c` shares no byte with `a` or `b` (the host's `c` is fresh);
- the three buffers are valid in the initial memory (aligned, every byte present).

Any strides within `int32` (a transposed `b`, a padded or column-major `c`), any `NUM_SMS` in the
bounds and `M = 0` are covered; `a` and `b` may overlap. Every `float16` input is covered, NaN and
infinities included.

**The launches covered** (`SETUP.md`): `M = N = 8192`, `K = 128, 256, …, 1024`, `NUM_SMS = 132`,
the host's strides (`setup_*_ok`, through `hostInst_ok`).

**Device facts** the tolerance rests on (in `Device P`): `Device.DotF16` (a `float16` `tl.dot`
into `float32`: each finite output within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with
`L = |acc| + Σ liftMag a·liftMag b`, a subnormal operand counted at `2^-14`) and the totality of
`tl.dot`. The conversions to `float16` are IEEE roundings.

## Tolerances

With `p e = va e·vb e` for `e < K` and `0` beyond (`prodV`), `l e = liftMag(va e)·liftMag(vb e)`
(`liftV`), `T = cdiv(K, 64)` steps, `S_t` the exact partial sum before step `t` (`partSum`) and
`L_t` the lifted magnitude of step `t` (`stepMag`): `accMag = Σ_{t<T} (|S_t| + L_t)`.

The fine bound: the inner loop's step `t` is one `tl.dot` of `64` products into the accumulator, so
by `DotF16` the accumulator's error follows `E_0 = 0`,
`E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + L_t) + α`, with `ρ = relDot16 64 = 66·2^-22` and
`α = absDot16 64 = 66·2^-126` (`rho`, `alpha`). `K ≤ 2^13` keeps every step far inside the
`float32` range. The final `.to(tl.float16)` is one rounding (`u = 2^-11`, `η = 2^-25`).

The relaxation to `tol`: `(1 + ρ)^T < 1.0021` (`tol_rel_lt`, `T ≤ 128`), so
`E_T ≤ 1.0021·ρ·accMag + 1.0021·T·α ≤ 2^-15·accMag + 2^-112`. A result rounded to `bfloat16`, a
`float16` accumulator, or a missing step breaks `Post`.

## Exclusions

- **Partial tiles** (`M % 128 ≠ 0` or `N % 256 ≠ 0`): a kernel limit. The `tl.where` of lines
  332–333 makes `offs_am` (`offs_bn`) non-contiguous on a partial tile, so the hint pair of lines
  334–335 is false. This leaves out the file's first test, `validate(32, 32, 32)`.
- **`N`, `K` not positive multiples of `128` or above `2^17`, `M > 2^22`**: the claim covers real
  model shapes only.
- **`K > 2^13`**: outside the file's launches; the tolerance is stated for `K ≤ 2^13`.
- **`NUM_SMS = 0`, strides `≥ 2^31`, `int64` arguments**: outside real use.
- **`int32` wrap of the offsets or of the tile count**, **overlapping elements of `c`**, **`c`
  overlapping `a` or `b`**: kernel limits, not reached by the host (`c` is fresh, the setup's
  matrices have at most `2^26` elements).
- **`float8e4nv` operands, other configs, the file's other kernels**: not part of the setup.

`check_gpu.py` (one H200) checks `Post` on the setup's launches, edge cases and wrong kernels.
