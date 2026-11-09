# Prepared claim: `matmul_e5m2`

The kernel: `source/03-matrix-multiplication.py`, Triton 3.8.0's tutorial 03, `matmul_kernel`
(lines 228–316), on `float8e5` operands with a `float16` result, with the config of `SETUP.md` (`BLOCK_SIZE_M = 128`,
`BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`, `num_stages = 3`, `num_warps = 8`),
launched by the host `matmul(a, b)` (lines 330–350). The claim is `Claims.MatmulE5m2.Statement` in
`Claim.lean`:

```lean
def Statement : Prop := ∀ K P, Assumptions K P → Spec K
```

`Claim.lean` proves `consistent`, `satisfiable`, `tolA_rel_lt`, `tolA_le` and the coverage facts `setup_*_ok`; it
does not prove `Statement`. The model map is `model-map.json` (every code line of the kernel lines
`[154, 155], [228, 316]`).

The kernel reads two module-level names, `triton` (its decorator) and `tl`, bound by the imports of
lines 154–155, which are kernel lines and are mapped (as `import` lines that add no step). Every other
module-level line is a host line: `import torch` (152), `DEVICE = …` (157), `is_cuda` (160–161), the
config functions' headers (164, 202, 216), `matmul` (330–350), the test (359–388), the benchmark's
configs and `benchmark` (400–437) and its run (440). The functions `get_cuda_autotune_config`,
`get_hip_autotune_config`, `get_autotune_config` and the `@triton.jit` helper `leaky_relu` are unused
(`source.unused`): nothing calls the first three, and the kernel's only call of `leaky_relu`
(line 307) is under `ACTIVATION == "leaky_relu"`, which no launch takes.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.MatmulE5m2.main : Statement` (`Statement`:
`Claim.lean` line 549). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 180), and every initial memory `init` that `Pre` accepts (line 270), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 503).

**What `Post` asks** (line 503), two parts:
1. for every `i < M`, `j < N`: `OutSpec d init fin i j` (line 486). Element
   `i·stride_cm + j·stride_cn` of `c` reads in `fin` as some `float16` `c`. This holds for every
   element. Only when row `i` of `a` and column `j` of `b` read in `init` as finite `float8e5`
   values `va k`, `vb k` (`k < K`) it also asks for a finite `float32` `v` with
   - `|v.toRat - dotSum va vb d.K| ≤ tol va vb d.K` and `FloatV.NaNEq (FloatV.cvt v) c`;
   - if `|dotSum va vb d.K| + tol va vb d.K ≤ Format.binary16.maxVal`: `c` is finite and
     `|c.toRat - dotSum …| ≤ tol … + (|dotSum …| + tol …) * Format.binary16.u + Format.binary16.eta`.

   `dotSum` (line 470) is the exact sum `Σ_k va k·vb k` in the rationals. `tol` (line 328) is
   `errAt (prodV …) (liftV …) (2 * steps K)`: the recursion `errAt` (line 314) over the
   `2·cdiv(K, 64)` chunks of 32 products (`steps`, line 278), with each step the smaller of the two
   arms of `Rules.Num.DotE5m2ChunkLoss`, the exact partial sums `partSum` (line 292), each chunk's
   largest lifted product `chunkMax` (line 295) of `liftV` (line 288) and its exact magnitude
   `chunkAbs` (line 300);
2. the frame: every byte outside `d.c.elems (cIdx d)` (`cIdx`, line 174) is unchanged.

**What you are given** (`Assumptions`, line 511):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, for a valid launch with
  `debug = false` that is well behaved from `init`;
- `dot` (`Rules.Num.DotE5m2ChunkLoss`, a `DotChunkLoss`): the `tl.dot` of source line 300 on finite
  inputs, with room in `float32`, gives a finite result reached through one partial value per chunk
  of 32, each within `min(2^-7·M, 2·A + 2^-12·M) + 2^-12·|s + c| + 2^-126` of `s + c`;
- `dot_total` (`Rules.Num.DotTotal`): that `tl.dot` always has a result;
- `cvt` (`Rules.Num.CvtF32E5m2`, a `CvtRn`): `float32` to `float8e5` rounds to nearest even in
  range, so `other=0.0` (source lines 297–298) is `+0`;
- `cvt_total` (`Rules.Num.CvtTotal`): that conversion always has a result.

**The kernel, as the model runs it** (`matmul_kernel` line 38, `model` line 254, `launch`
line 263): a 1-D grid of `cdiv(M, 128)·cdiv(N, 256)` programs. Program `pid` computes the grouped
tile `(pid_m, pid_n)` (`GROUP_SIZE_M = 8`, `tl.idiv`, `tl.irem`, `tl.iminimum`), checks eight
`tl.assume`s, and forms `a_ptrs` and `b_ptrs` with rows and columns taken `% M` and `% N`. A
`tl.forRange` loop of `steps K` trips loads the `[128, 64]` tile of `a` and the `[64, 256]` tile of
`b` masked on `k < K - 64·t` (`other` converted to `float8e5` by `tl.convert`), does
`acc ← tl.dot(a, b, acc)`, and advances both pointers. After the loop it rounds to `float16` and
stores the `[128, 256]` tile of `c` masked by `M` and `N`. No reduction, no atomic. Programs write
disjoint tiles of `c` and only read `a` and `b`: the route is `TVA.meets_independent`; the loop is
`safe_forRange` with an invariant on the pointers and the accumulator (each element within
`errAt … (2t)` of `partSum … (2t)`) and `Wt := fun _ => ∅` (`iterW_empty`) (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: fewer than `2^31` programs
  from `grid_lt`; the ten block shapes are powers of two);
- each program is `Safe`: the grouped `pid_m < cdiv(M, 128)`, `pid_n < cdiv(N, 256)` and the
  `tl.assume`s hold (`safe_assume`; the strides from `am` … `cn`); `safe_idiv`, `safe_irem` have no
  bad lane; `safe_convert` for `other` (`cvt_total`, `cvt`); the `int32` values do not wrap
  (`M_le`, `N_le`, `K_le`, `a_fit`, `b_fit`, `cm_fit`, `cn_fit`); every enabled load lane is inside
  `a`, `b` (`a_len`, `b_len`) and every enabled store lane inside `c` (`c_len`), with distinct
  lanes (`c_inj`);
- the map `pid ↦ (pid_m, pid_n)` is one-to-one on the grid, so the programs' write tiles are
  disjoint (`c_inj`), and no program reads what another writes (`c_a`, `c_b`);
- the value part of part 1: per trip, `safe_dot` with `dot` (and `dot_total`) and
  `Dot8Chunk.DotF8ChunkLoss.err` move the invariant over the trip's two chunks, from
  `errAt … (2t)` to `errAt … (2t + 2)`; `K_le` keeps the room condition; the final rounding is
  `FloatV.cvt_err`. `## Tolerances` gives the derivation;
- the bytes no program writes give part 2.

The real-shape conditions simplify the proof: `128 ∣ K` (`K_dim`) makes every trip of the loop a full
step of `64` (`steps K = K / 64`, no `k` lane masked off, no `other` value used in a product), and
`128 ∣ N` (`N_dim`) with `BLOCK_SIZE_N = 256` makes the last column tile full or half full; `M` stays
any token count up to `2^22` (`M_ctx`), so the last row tile may be partial.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `hostInst_ok`
(line 613) and the `setup_*_ok` theorems; reusable: `errAt_succ` (line 353), `errAt_nonneg`
(line 363), `errAt_succ_le_acc` (line 379), `errAt_succ_le_loss` (line 391), `partSum_succ`
(line 401), `le_chunkMax`, `chunkMax_le` (lines 407, 410), `errAt_le_bound` (line 416),
`tolA_rel_lt`, `tolA_le` (lines 440, 449), the constants `rho_eq` … `rhoL_eq`, `q_eq`
(lines 339–350), `row_lt` (line 582), `rm_inj` (line 588), `bytes_disjoint` (line 603).

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `i < M` and `j < N`, element `(i, j)` of `c` (element
  `i·stride_cm + j·stride_cn` of `c`'s storage) can be read in the final memory, with some
  `float16` value `c` (`OutSpec`). When, moreover, row `i` of `a` and column `j` of `b` hold finite
  `float8e5` values `va k` and `vb k` (`k < K`) in the initial memory, then, with
  `S = Σ_k va k · vb k` the exact dot product and `E = tol va vb K` (Tolerances):
  - `c` is the `float16` rounding (to nearest even, `FloatV.cvt`, up to NaN payload) of a finite
    `float32` value `v` with `|v − S| ≤ E`;
  - when `|S| + E ≤ 65504` (the largest `float16`), `c` is finite and
    `|c − S| ≤ E + (|S| + E)·2^-11 + 2^-25` (one more `float16` rounding:
    `Format.binary16.u = 2^-11`, `Format.binary16.eta = 2^-25`).

  With a NaN or an infinity in row `i` of `a` or column `j` of `b`, the claim says only that
  element `(i, j)` holds some `float16` value.
- **Nothing else changes.** Every byte outside the elements of the matrix `c` is the same in `fin`
  as in `init`: `a`, `b`, and the elements of `c`'s storage that are not in the matrix.

The reference `S` is the exact product of the host's matrices, row `i` of `a` by column `j` of `b`,
as `matmul(a, b)` means it.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `M`, `N`, `K`, the six strides (in elements), and two
`float8e5` buffers `a`, `b` (`Buf .f8e5`) and a `float16` buffer `c` (`Buf .f16`), each a base address and a length in elements; element
`(i, k)` of the matrix `a` is element `i·stride_am + k·stride_ak` of the buffer `a`, and likewise
for `b` and `c` (`aIdx`, `bIdx`, `cIdx`).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `M_le`, `N_le` | `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31` | `M` and `N` are `int32` arguments, and `tl.cdiv` computes `M + 127` and `N + 255` in `int32` (lines 253–254); the tile rows and columns `pid_m·128 + i`, `pid_n·256 + j` (lines 282–283, 312–313) then stay below `2^31` |
| `grid_lt` | `cdiv(M, 128)·cdiv(N, 256) < 2^31` | the grid's size (`Launch.Valid`: CUDA's limit on a 1-D grid; `pid` is an `int32`) |
| `K_le` | `K ≤ 2^11` | the setup's largest `K` (`2048`); the tolerance grows with `K`, and beyond `2048` `Post` stops rejecting typical wrong results (Tolerances, Exclusions) |
| `M_ctx` | `M ≤ 2^22` | `M` counts tokens: a context length, at most `2^22` (real model shapes, Exclusions) |
| `N_dim` | `0 < N`, `128 ∣ N`, `N ≤ 2^17` | `N` is a model dimension (the output features): a positive multiple of `128`, at most `2^17` (real model shapes, Exclusions) |
| `K_dim` | `0 < K`, `128 ∣ K`, `K ≤ 2^17` | `K` is a model dimension (the inner features): a positive multiple of `128`, at most `2^17` (real model shapes, Exclusions) |
| `am`, `ak`, `bk`, `bn`, `cm`, `cn` | each stride is positive and below `2^31` | the kernel's `tl.assume(stride_… > 0)` (lines 268–273) is undefined behaviour otherwise; each stride is an `int32` argument |
| `a_fit` | `(M − 1)·stride_am + 64·stride_ak < 2^31` | the `int32` offsets of `a_ptrs`, `offs_am·stride_am + offs_k·stride_ak` with `offs_am < M` (taken `% M`, line 282) and `offs_k < 64` (line 285), and the step `64·stride_ak` (line 302), do not wrap |
| `b_fit` | `64·stride_bk + (N − 1)·stride_bn < 2^31` | the same for `b_ptrs` (`offs_bn < N`, line 286) and its step (line 303) |
| `cm_fit`, `cn_fit` | `(M − 1)·stride_cm < 2^31`, `(N − 1)·stride_cn < 2^31` | the `int32` products `stride_cm·offs_cm` and `stride_cn·offs_cn` (line 314) of the enabled lanes do not wrap (each is added to the pointer on its own; the disabled lanes' products may wrap, and are not accessed) |
| `c_inj` | distinct elements `(i, j)` of the matrix `c` are distinct elements of its storage | otherwise two lanes or two programs store to one element; any layout in which the elements do not overlap is covered (row-major, padded rows, column-major) |
| `a_len`, `b_len`, `c_len` | every element of each matrix is inside its buffer | the kernel accesses them |
| `c_a`, `c_b` | the matrix `c` shares no byte with the matrix `a`, nor with the matrix `b` | a program that reads bytes another program writes races; `matmul` always allocates a fresh `c` |

`M`, `N`, `K` and the strides are the kernel's runtime arguments (or, at the value `1`, compile-time
constants, which give the same model: `model`'s docstring). Their values are part of the instance,
and the conditions above constrain them; `Compiler.Faithful`'s point 11 assumes that the launch
runs the code Triton compiles for these values. The conditions allow any positive strides (a
row-major `b`, a non-contiguous `a`, any layout of `c` whose elements do not overlap), any `M`, `N`,
`K` within the bounds, and `M = 0` (an empty grid: memory unchanged); `a` and `b` may overlap each
other in any way (both are only read).

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to its element size, the buffer inside `[0, 2^64)`, every byte present).
Every `float8e5` value is allowed in the inputs, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch of `SETUP.md` is a direct launch of the `@triton.jit` function,
the JIT launch `matmul_kernel[grid](…)` of `matmul` (lines 340–349), called at line 381 and, in the
benchmark, through `triton.testing.do_bench(lambda: matmul(a, b), …)` (line 435). Point by point:

- **4 (options).** The launch gives the constexprs `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`,
  `GROUP_SIZE_M`, `ACTIVATION` and two options, `num_stages=3` and `num_warps=8`, two of the four
  that do not change the meaning; no `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or any
  other option. Nothing calls `matmul_kernel.warmup(…)` or `triton.compile`; the kernel has no
  `@triton.autotune` (so no `triton.Config`; the config lists `get_…_autotune_config` build are
  never used) and no `@triton.heuristics`, and the decorator is a bare `@triton.jit`, so its `debug`
  is not set. So `launch` keeps every default: `enable_fp_fusion = true`, `debug = false`,
  `num_ctas = 1`, `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor, so no allocator is involved.
- **6 (precompiled kernel).** No `CompiledKernel` is launched: every launch is the JIT launch above.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are three
  tensors, nine integers and the constexprs.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`. Its parameters are three pointers (tensor
arguments), nine unannotated integers (`int` arguments) and five `tl.constexpr`s, four with `int`
arguments and `ACTIVATION` with the `str` argument `""`, none with a default. Its statements are a
docstring, assignments `name = e`, two `name += e` (lines 302–303), expression statements (the calls
`tl.assume` and `tl.store`), one `for k in range(0, …)` with two positional arguments, and one `if`
on a compile-time condition. Its expressions are `tl.program_id(axis=0)`, `tl.cdiv`, `tl.arange`,
`tl.zeros(…, dtype=tl.float32)`, `tl.load(…, mask=…, other=0.0)`, `tl.dot(a, b, acc)`,
`x.to(tl.float16)`, `tl.store(…, mask=…)`, `tl.assume`, Python's builtins `min` and `range`, the
operators `+ - * // % &`, the comparisons `>= > < ==` (`==` only between compile-time constants),
and the subscripts `[:, None]` and `[None, :]`. The only module-level names it reads are `triton`
and `tl`, Triton's modules, each bound only by its own `import` (lines 154 and 155); line 307 names
the module's `@triton.jit` function `leaky_relu`, bound only by its `def`, but the front end does not
visit that line. No other statement of the module binds `triton`, `tl`, `matmul_kernel` or
`leaky_relu`.

## Assumptions

Five assumptions, all of kind `[rule]`, each cited as the library states it
(`docs/PROVER_GUIDE.md` section 3, [num.md](../../docs/rules/num.md)):

- `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`;
- `dot : Rules.Num.DotE5m2ChunkLoss P` — `tl.dot` of `float8e5` operands into `float32` with
  `maxImpreciseAcc = 2^30`, the default on sm_90 (line 300), chunk by chunk, each chunk within the
  smaller of two bounds (`DotF8ChunkLoss P .f8e5 .f8e5`, that is `DotChunkLoss` with
  `rhoChunk8 = 2^-7`, `sigmaChunk8 = 2^-12`, `alphaChunk8 = 2^-126`, `kappaChunk8 = 2`,
  `rhoLossChunk8 = 2^-12`): for every `input_precision` and each output element whose row of `a`,
  column of `b` and accumulator are finite, with room below the `float32` range, the result is
  finite, and there are partial values `s 0 = acc, …, s ⌈K/32⌉ = result`, one per chunk of `32`
  consecutive `k` in increasing order, with `s (j+1)` within
  `min(2^-7·M, 2·A_j + 2^-12·M) + 2^-12·|s j + c_j| + 2^-126` of `s j + c_j` (`c_j` the chunk's
  exact sum, `A_j` its exact `Σ|a·b|`) for every `M` at least `|s j|` and each lifted product
  `liftMag a_k · liftMag b_k` of the chunk (a subnormal `float8e5` operand counted as `2^-14`);
- `dot_total : Rules.Num.DotTotal P {a := .f8e5, b := .f8e5, acc := .f32, precision := .tf32,
  maxImpreciseAcc := 2 ^ 30}` — the `tl.dot` of line 300 always has a result;
- `cvt : Rules.Num.CvtF32E5m2 P` — the conversion of a finite `float32` of magnitude at most `57344`
  into `float8e5` with the default rounding is its rounding to nearest even, up to NaN payload. The
  model's `other=0.0` (lines 297–298) is the `float32` `+0.0` converted to the pointers' dtype on the
  device (`tl.convert`, for an fp8 target the platform's `P.cvt`), so each disabled lane holds `+0`;
- `cvt_total : Rules.Num.CvtTotal P Format.binary32 Format.e5m2 .rn` — that conversion always has a
  result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversion of line 308 is an IEEE
rounding the library defines (`tl.cvt`) and needs no platform fact.

## Coverage

Every launch of `SETUP.md` ("The launches"): `a` contiguous `[M, K]` (strides `(K, 1)`), `b`
`[K, N]` the transpose of a contiguous matrix (strides `(1, K)`), the fresh `c` (strides `(N, 1)`),
`M = N = K ≤ 2048`, multiples of `128`. `Inst.ok` covers them: `M_ctx`, `N_dim` and `K_dim` hold
(`128·i ≤ 2048`), `a_fit` reads `(M − 1)·K + 64 < 2^31`, `b_fit`
`64 + (N − 1)·K < 2^31`, `cm_fit` `(M − 1)·N < 2^31`, `c_inj` holds for a row-major `c`, and
`K ≤ 2048 = 2^11`; at `2048` the binding bound is below `2^22 + 64`.

Proved instances (`hostInst M N K`: the host's strides, `b` with strides `(1, K)`, the three buffers at fixed aligned
addresses; `hostInst_ok`):

| Instance | `M = N = K` | Source |
|---|---|---|
| `setup_test_512_ok` | 512 | the test, lines 372–388 |
| `setup_bench_256_ok`, `setup_bench_384_ok`, …, `setup_bench_2048_ok` (15) | `128·i`, `i = 2 … 16` | the benchmark's `fp8` line, lines 402–440 |

`satisfiable` uses the test's instance with the all-zero memory.

## Tolerances

`tol va vb K = errAt (prodV va vb K) (liftV va vb K) (2·steps K)` (`Claim.lean`): the library's fp8
`tl.dot` fact (`Rules.Num.DotE5m2ChunkLoss`) applied chunk by chunk against the exact partial sums.
Each of the loop's `steps K = cdiv(K, 64)` `tl.dot`s of `K = 64` adds two chunks of `32` products, so
the loop adds `n = 2·cdiv(K, 64)` chunks, chunk `j` holding the products `32·j ≤ e < 32·j + 32`. With
`p e = va e · vb e` for `e < K` and `0` from `K` on (`prodV`, the masked lanes), `l e` its lifted
magnitude (`liftV`), `S_j = Σ_{e < 32j} p e` (`partSum`), `m_j = max` of `l e` on chunk `j`
(`chunkMax`), `A_j = Σ |p e|` on chunk `j` (`chunkAbs`) and `X_j = max(|S_j|, m_j)`:
`E_0 = 0`, `E_{j+1} = (1 + σ)·E_j + min(ρ·(X_j + E_j), κ·A_j + ρ'·(X_j + E_j)) + σ·|S_{j+1}| + α`,
with the fact's constants `ρ = 2^-7`, `σ = 2^-12`, `α = 2^-126`, `κ = 2`, `ρ' = 2^-12`; `tol = E_n`.

Why it is provable: each chunk is what the fact gives. It adds its exact sum `c_j` to an accumulator
`s` within `E_j` of `S_j`, up to `min(ρ·M, κ·A_j + ρ'·M) + σ·|s + c_j| + α` for every `M` at least `|s|`
and the chunk's lifted products; with `M = max(|S_j| + E_j, m_j) ≤ X_j + E_j` and
`|s + c_j| ≤ |S_{j+1}| + E_j`, the new accumulator is within `E_{j+1}` of `S_{j+1}`
(`TVA.Lemmas.Dot8Chunk`: `DotF8ChunkLoss.err`, `chain_errM` compose the same recursion). The
accumulator of each `tl.dot` is the loop's own (no separate addition), so the chunks chain across
the loop's steps. The final `.to(tl.float16)` is one rounding (`FloatV.cvt_err`). The room condition:
`float8e5` products are below `2^32` and `K ≤ 2^11`, so every partial value stays far inside `float32`
(`tolA_le`: `tol ≤ 0.7·L + 10^-35`, `L` the lifted magnitudes).

How meaningful it is. `tol` is never above `tolA K L = ((1 + ρ + σ)^n − 1)·L + n·(1 + ρ + σ)^n·α`
(the first arm composed alone; `(1 + ρ + σ)^n − 1` is `0.137` at `K = 512` and `0.671` at
`K = 2048`, `tolA_rel_lt`), and the second arm makes it far smaller where the running sum is large
next to each chunk's own products. On an input with one product `X` it is `1–4%` of `|X|` (`0.8%`
when `X` is in the last chunk, `4.0%` at `K = 2048` when it is in the first). On the file's own data
(`torch.randn` cast to `float8e5`) each chunk's magnitude (about `20`) stays above `2^-8` of the
running sum, the first arm is the smaller on every chunk, and the bound compounds by about
`1 + 2^-7` per chunk. `check_gpu.py` measures, over the outputs, a median `E / |S|` of `0.095` at
`K = 512` (the test; `0.375` at the 90th percentile), `0.20` at `K = 1024` and `0.44` at
`K = 2048` (`1.73` at the 90th percentile).

**Where the claim says little.** At `K = 512` `Post` pins each output to within about a tenth of its
size. At the setup's larger sizes it pins it much less: at `K = 2048` the tolerance is about half a
typical output and larger than the output for `18%` of the elements. Typical wrong results are
still rejected at every covered `K` (`Post` holds for all elements or fails), but by fewer and fewer
elements (negative tests of `check_gpu.py`, the share of elements that break `Post`):

| Wrong result | `K = 512` | `K = 1024` | `K = 2048` |
|---|---|---|---|
| all zero | 96% | 92% | 82% |
| the first step's 64 products left out | 86% | 58% | 14% |
| the last product left out | 14% | 2.2% | 0.02% |
| `0.9` times the exact result | 54% | 7.3% | 0.01% |
| another column's result, or `b` shifted by one along `K` | 97% | 93% | 84% |

So at `K = 2048` a result off by `10%` everywhere, or missing one product, breaks `Post` on only
about `0.01–0.02%` of the `4·10^6` elements (`400–800` elements). From `K = 3072` on such results
pass (Exclusions).

**Where the slack comes from.** Most of it is the device's real fp8 accumulation loss, not the
fact's form. The `wgmma` accumulator keeps a `13`-bit window, and products below it are truncated:
on coherent data (one large product first, then `K − 1` products of one sign below the window,
`check_gpu.py`'s `coherent_case`) the device's own error `|c − S| / |S|` is `5.2%` at `K = 512`,
`9.9%` at `1024` and `17.9%` at `2048` (`30%` at `K = 4096`), and the tolerance there is
`0.112`, `0.214`, `0.394` of the output: within a factor of about `2.2` of what the device really
loses (err/tol `0.45`). On the file's random data the truncations of different chunks largely cancel
and the device's error is about `1–2%` of the tolerance (`0.023` at `K = 512`, `0.010` at `2048`),
but the library's facts give one bound for both kinds of data: each chunk relative to
`M ≥ |s j|` with `2^-7`, about twice the window's worst chunk. A sharper fact (Library requests)
could remove at most that factor of about `2`; no tighter tolerance follows from the library's
facts.

## Exclusions

Launches of the file the claim leaves out:

- **The benchmark's `fp8` sizes with `2048 < K ≤ 4096` (`M = N = K = 2176, …, 4096`; `K_le`): a
  library gap, where the library's fp8 `tl.dot` fact cannot give a meaningful tolerance.** The tolerance keeps growing with `K`: on the file's data its median over the
  outputs is `0.74` at `K = 3072` and `1.12` at `K = 4096` (larger than the output for `57%` of the
  elements), and typical wrong results pass `Post` there: `0.9` times the exact result from
  `K = 3072` on, a result missing its last product at `K = 4096`. Most of this is the device's own
  loss on coherent data (`30%` of the output at `K = 4096`, Tolerances; on the file's own data the
  device's error is about `0.06%`), and one bound must hold for both, so a sharper fact would shrink
  it by at most a factor of about `2`. The kernel is believed correct there.

Conditions that exclude launches outside `SETUP.md`'s launches:

- **Out of scope**: dimensions that are not multiples of `128` or exceed `2^17`, contexts longer
  than `2^22` tokens, and batches of more than `8192` requests: the claim covers real model shapes
  only (`N_dim` and `K_dim` for `N` and `K`, `M_ctx` for `M`; the kernel has no batch field). The
  setup's launches are all inside.

- `K > 2^11` in general (outside the setup but for the sizes above): the tolerance only grows
  there.
- Strides of `2^31` or more, or sizes that make Triton pass an argument as `int64`: outside the
  setup; a different model (the arguments' integer class), not a kernel fault.
- `ACTIVATION = "leaky_relu"`, other block configurations, `float16` operands: not part of the setup.

Kernel limits (the kernel is wrong there under the library's semantics; real use does not reach
them):

- A zero or negative stride: `tl.assume(stride > 0)` (lines 268–273) is false, undefined behaviour.
  Not reached: the host's tensors have positive strides.
- `int32` wrap of the offsets (`a_fit`, `b_fit`, `cm_fit`, `cn_fit`, `M_le`, `N_le`): with `int32`
  arguments, the offsets of lines 285, 286 and 314 wrap once a matrix spans `2^31` elements of its
  storage; the first launch that hits it through `matmul` is `M = 2^31 / K + 1` rows (for `K = 2048`,
  `M = 1048577`), or, for the transposed `b`, `N = 2^31 / K + 1` columns. The setup's largest matrices
  have `2^24` elements.
- An output whose elements overlap in storage (`c_inj`), or that overlaps `a` or `b` (`c_a`, `c_b`).
  Not reached: `matmul` allocates a fresh `c`.
- More than `2^31 − 1` programs (`grid_lt`): CUDA's grid limit. Not reached.

## Library requests

- A sharper fp8 `tl.dot` fact for a fed-back accumulator: that a partial value never grows past
  `|acc| + Σ|a·b|` (the device truncates every term and the sum toward zero), or a per-chunk relative
  constant nearer the window's `33·2^-13`. With the present facts the tolerance is about `0.44` of a
  typical output at `K = 2048` on the file's data, and about `2.2` times the device's own loss on
  coherent data (Tolerances); such a fact would shrink it by at most that factor.

## Checks

1. **Builds**: `lake env lean ../claims/matmul_e5m2/Claim.lean`: no errors or warnings, no `sorry`, no
   `axiom`; `consistent`, `satisfiable`, `tolA_rel_lt`, `tolA_le` and every `setup_*_ok` depend only
   on `propext`, `Classical.choice`, `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (the test's instance, the
   all-zero memory).
3. **Coverage**: 16 `setup_*_ok` (the test and the 15 benchmark sizes up to `2048`).
4. **GPU** (`check_gpu.py`, H200 NVL, Triton 3.8.0): `matmul` on the test and the 15 benchmark sizes
   `256 … 2048` with the file's draws; two edges inside `Inst.ok` (`M = 4096`, `N = 384`,
   `K = 2048`; inputs scaled by 16) and four launches outside its real-shape conditions, where `Post`
   holds as well (`K = 2047` with `M = 1000`, `N = 777`; `K = 2000` with `M = 300`, `N = 1000`, whose
   last step's second chunk holds only disabled lanes; `M = N = 1` with `K = 65`; `K = 1`); two
   direct launches with a padded and a column-major `c` in a guarded buffer and a row-major `b`
   (outside the real-shape conditions too: `N = 1000`, `K = 777`; `N = 259`, `K = 1000`); coherent data at `K = 512, 1024, 2048` (`M = N = 256`). On the file's draws the
   largest error / tolerance is `0.045` (`K = 256`), `0.023` on the test and `0.010` at `K = 2048`,
   with a median `E / |S|` from `0.046` (`K = 256`) to `0.44` (`K = 2048`); on coherent data it is
   `0.458` (margin `2.2×`; the device's error `5.2%`, `9.9%`, `17.9%` of the output, and `30%` at
   `K = 4096`, printed only). Negative tests at `K = 512, 1024, 2048` (the table in Tolerances): every
   wrong result breaks `Post`. Inputs unchanged; no byte outside the matrix `c` written. PASS.
