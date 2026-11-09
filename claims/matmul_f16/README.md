# Prepared claim: `matmul_f16`

The kernel: `source/03-matrix-multiplication.py`, Triton 3.8.0's tutorial 03, `matmul_kernel`
(lines 228–316), on `float16` matrices with the config of `SETUP.md` (`BLOCK_SIZE_M = 128`,
`BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`, `num_stages = 3`, `num_warps = 8`),
launched by the host `matmul(a, b)` (lines 330–350). The claim is `Claims.Matmul.Statement` in
`Claim.lean`:

```lean
def Statement : Prop := ∀ K P, Assumptions K P → Spec K
```

`Claim.lean` proves `consistent`, `satisfiable`, `tol_rel_lt` and the coverage facts `setup_*_ok`; it
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

**The goal.** In `Proof.lean`, `theorem Claims.Matmul.main : Statement` (`Statement`:
`Claim.lean` line 370). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 176), and every initial memory `init` that `Pre` accepts (line 254), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 340).

**What `Post` asks** (line 340), two parts:
1. for every `i < M`, `j < N`: `OutSpec d init fin i j` (line 323). Element
   `i·stride_cm + j·stride_cn` of `c` reads in `fin` as some `float16` `c`. This holds for every
   element. Only when row `i` of `a` and column `j` of `b` read in `init` as finite values `va k`,
   `vb k` (`k < K`) it also asks for a finite `float32` `v` with
   - `|v.toRat - dotSum va vb d.K| ≤ tol va vb d.K` and `FloatV.NaNEq (FloatV.cvt v) c`;
   - if `|dotSum va vb d.K| + tol va vb d.K ≤ Format.binary16.maxVal`: `c` is finite and
     `|c.toRat - dotSum …| ≤ tol … + (|dotSum …| + tol …) * Format.binary16.u + Format.binary16.eta`.

   `dotSum` (line 282) is the exact sum `Σ_k va k·vb k` in the rationals. `tol` (line 315) is
   `errAt (prodV …) (liftV …) (steps K)`: the recursion `errAt` (line 308) over the
   `steps K = cdiv(K, 64)` loop steps (line 260), with `rho = relDot16 64`, `alpha = absDot16 64`
   (lines 264, 267), the exact partial sums `partSum` (line 296) and the steps' lifted magnitudes
   `stepMag` of `liftV` (lines 299, 292);
2. the frame: every byte outside `d.c.elems (cIdx d)` (`cIdx`, line 170) is unchanged.

**What you are given** (`Assumptions`, line 348):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, for a valid launch with
  `debug = false` that is well behaved from `init`;
- `dot` (`Rules.Num.DotF16`, a `DotLiftAcc`): the `tl.dot` of source line 300, on a finite row, column and
  accumulator, gives a finite element within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`,
  `L = |acc| + Σ liftMag a·liftMag b`, when that leaves room in `float32`;
- `dot_total` (`Rules.Num.DotTotal`): that `tl.dot` always has a result.

**The kernel, as the model runs it** (`matmul_kernel` line 36, `model` line 238, `launch`
line 247): a 1-D grid of `cdiv(M, 128)·cdiv(N, 256)` programs. Program `pid` computes the grouped
tile `(pid_m, pid_n)` (`GROUP_SIZE_M = 8`, `tl.idiv`, `tl.irem`, `tl.iminimum`), checks eight
`tl.assume`s, and forms `a_ptrs` and `b_ptrs` with rows and columns taken `% M` and `% N`. A
`tl.forRange` loop of `steps K` trips loads the `[128, 64]` tile of `a` and the `[64, 256]` tile of
`b` masked on `k < K - 64·t` (`other` `+0`), does `acc ← tl.dot(a, b, acc)`, and advances both
pointers by `64` columns or rows. After the loop it rounds to `float16` and stores the
`[128, 256]` tile of `c` masked by `M` and `N`. No reduction, no atomic. Programs write disjoint
tiles of `c` and only read `a` and `b`: the route is `TVA.meets_independent`; the loop is
`safe_forRange` with an invariant on the pointers and the accumulator (each element within
`errAt` of the partial sum) and `Wt := fun _ => ∅` (`iterW_empty`) (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: fewer than `2^31` programs
  from `grid_lt`; the ten block shapes are powers of two);
- each program is `Safe`: the grouped `pid_m < cdiv(M, 128)`, `pid_n < cdiv(N, 256)` and the
  `tl.assume`s hold (`safe_assume`; the strides from `am` … `cn`); `safe_idiv`, `safe_irem` have no
  bad lane; the `int32` values do not wrap (`M_le`, `N_le`, `K_le`, `a_fit`, `b_fit`, `cm_fit`,
  `cn_fit`); every enabled load lane is inside `a`, `b` (`a_len`, `b_len`) and every enabled store
  lane inside `c` (`c_len`), with distinct lanes (`c_inj`);
- the map `pid ↦ (pid_m, pid_n)` is one-to-one on the grid, so the programs' write tiles are
  disjoint (`c_inj`), and no program reads what another writes (`c_a`, `c_b`);
- the value part of part 1: per trip, `safe_dot` with `dot` (and `dot_total`) moves the invariant
  from `errAt … t` to `errAt … (t + 1)`; the room condition holds because `K_le` keeps every value
  far inside `float32`; the final rounding is `FloatV.cvt_err`. `## Tolerances` gives the
  derivation;
- the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `hostInst_ok`
(line 432) and the `setup_*_ok` theorems; reusable: `tol_rel_lt` (line 272), `row_lt` (line 401),
`rm_inj` (line 407), `bytes_disjoint` (line 422).

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `i < M` and `j < N`, element `(i, j)` of `c` (element
  `i·stride_cm + j·stride_cn` of `c`'s storage) can be read in the final memory, with some value
  `c` (`OutSpec`). When, moreover, row `i` of `a` and column `j` of `b` hold finite `float16`
  values `va k` and `vb k` (`k < K`) in the initial memory:
  - `c` is the `float16` rounding (to nearest even, `FloatV.cvt`, up to NaN payload) of a finite
    `float32` value `v` with `|v − S| ≤ E`, where `S = Σ_k va k · vb k` is the exact dot product
    and `E = tol va vb K` the tolerance of Tolerances below, computed from the inputs;
  - when `|S| + E ≤ 65504` (the largest `float16`), `c` is finite and
    `|c − S| ≤ E + (|S| + E)·2^-11 + 2^-25` (one more `float16` rounding:
    `Format.binary16.u = 2^-11`, `Format.binary16.eta = 2^-25`).

  With a NaN or an infinity in row `i` of `a` or column `j` of `b`, the claim says only that
  element `(i, j)` holds some `float16` value.
- **Nothing else changes.** Every byte outside the elements of the matrix `c` is the same in `fin`
  as in `init`: `a`, `b`, and the elements of `c`'s storage that are not in the matrix (padding of a
  row, say).

The reference `S` is the exact product of the host's matrices, row `i` of `a` by column `j` of `b`,
as `matmul(a, b)` means it.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `M`, `N`, `K`, the six strides (in elements), and three
`float16` buffers `a`, `b`, `c` (`Buf .f16`: a base address and a length in elements); element
`(i, k)` of the matrix `a` is element `i·stride_am + k·stride_ak` of the buffer `a`, and likewise
for `b` and `c` (`aIdx`, `bIdx`, `cIdx`).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `M_le`, `N_le` | `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31` | `M` and `N` are `int32` arguments, and `tl.cdiv` computes `M + 127` and `N + 255` in `int32` (lines 253–254); the tile rows and columns `pid_m·128 + i`, `pid_n·256 + j` (lines 282–283, 312–313) then stay below `2^31` |
| `grid_lt` | `cdiv(M, 128)·cdiv(N, 256) < 2^31` | the grid's size (`Launch.Valid`: CUDA's limit on a 1-D grid; `pid` is an `int32`) |
| `K_le` | `K ≤ 2^13` | keeps the tolerance meaningful (Tolerances); the host's largest `K` is `4096` |
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
transposed `b`, a non-contiguous `a`, any layout of `c` whose elements do not overlap), any `M`, `N`,
`K` within the bounds, `M = 0` or `N = 0` (an empty grid: memory unchanged) and `K = 0` (no loop
trip: `c = 0`); `a` and `b` may overlap each other in any way (both are only read).

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to 2 bytes, the buffer inside `[0, 2^64)`, every byte present). Every
`float16` value is allowed in the inputs, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch of `SETUP.md` is a direct launch of the `@triton.jit` function,
the JIT launch `matmul_kernel[grid](…)` of `matmul` (lines 340–349), called at line 362 and, in the
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

Three assumptions, all of kind `[rule]`, each cited as the library states it
(`docs/PROVER_GUIDE.md` section 3, [num.md](../../docs/rules/num.md)):

- `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`;
- `dot : Rules.Num.DotF16 P` — `tl.dot` of `float16` operands into `float32` (line 300), for every
  `input_precision`, with `maxImpreciseAcc = 0` (`Rules.Num.DotLiftAcc`): each output element whose
  row of `a`, column of `b` and accumulator are finite is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at `2^-14`), when that leaves room below the `float32` range;
- `dot_total : Rules.Num.DotTotal P {a := .f16, b := .f16, acc := .f32, precision := .tf32,
  maxImpreciseAcc := 0}` — the `tl.dot` of line 300 always has a result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversion of line 308 and the
`other=0.0` conversion of lines 297–298 are IEEE roundings the library defines (`tl.cvt`,
`tl.convert`), and need no platform fact.

## Coverage

Every launch of `SETUP.md` ("The launches"): `a` contiguous `[M, K]` (strides `(K, 1)`), `b`
contiguous `[K, N]` (strides `(N, 1)`), the fresh `c` (strides `(N, 1)`), `M = N = K ≤ 4096`.
`Inst.ok` covers them: `a_fit` reads `(M − 1)·K + 64 < 2^31`, `b_fit` `65·N − 1 < 2^31`, `cm_fit`
`(M − 1)·N < 2^31`, `c_inj` holds for a row-major `c`, and `K ≤ 4096 ≤ 2^13`; at `4096` the binding
bound is below `2^24 + 64`.

Proved instances (`hostInst M N K`: the host's strides, the three buffers at fixed aligned
addresses; `hostInst_ok`):

| Instance | `M = N = K` | Source |
|---|---|---|
| `setup_test_512_ok` | 512 | the test, lines 359–370 |
| `setup_bench_256_ok`, `setup_bench_384_ok`, …, `setup_bench_4096_ok` (31) | `128·i`, `i = 2 … 32` | the benchmark's `fp16` line, lines 402–440 |

`satisfiable` uses the test's instance with the all-zero memory.

## Tolerances

`tol va vb K = errAt (prodV va vb K) (liftV va vb K) (steps K)` (`Claim.lean`), the error bound of
the loop's `T = steps K = cdiv(K, 64)` steps against the exact partial sums. With `p e = va e · vb e`
for `e < K` and `0` from `K` on (`prodV`: the lanes the masks disable load `+0`), `l e` its lifted
magnitude `liftMag(va e)·liftMag(vb e)` (`liftV`; `liftMag q` is `|q|` for zero and normal values,
`2^-14` for a nonzero subnormal one), `S_t = Σ_{e < 64t} p e` (`partSum`) and
`L_t = Σ_{64t ≤ e < 64t+64} l e` (`stepMag`):

`E_0 = 0`, `E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + L_t) + α`, `tol = E_T`,

that is `E_T = Σ_{t<T} (1 + ρ)^(T−1−t)·(ρ·(|S_t| + L_t) + α)`, with the library's constants for one
fp16 `tl.dot` of `K = 64`: `ρ = relDot16 64 = 66·2^-22`, `α = absDot16 64 = 66·2^-126`.

**Derivation** (library facts only). Step `t` of the loop (line 300) is `acc ← tl.dot(a, b, acc)`, a
`float16` dot of `64` products with the accumulator as its own (no separate addition). By
`Rules.Num.DotF16` (`DotLiftAcc` with `K = 64`) each output element of it is within
`relDot16 64·(|acc| + L_t) + absDot16 64` of `acc + c_t`, `c_t = S_{t+1} − S_t` the step's exact sum
(the disabled lanes hold `+0`, adding nothing to `c_t` or `L_t`). If `acc` is within `E_t` of `S_t`
then `|acc| ≤ |S_t| + E_t`, so the new accumulator is within
`E_t + ρ·(|S_t| + E_t + L_t) + α = E_{t+1}` of `S_{t+1}`; by induction the loop leaves a `float32`
value within `E_T` of `S`. The fact's room condition holds at every step: finite `float16` products
are below `2^32`, so with `K ≤ 2^13` every `|acc| + L_t` and its bound stay far inside the `float32`
range. The final `.to(tl.float16)` (line 308) is one rounding (`FloatV.cvt_err`, `u = 2^-11`,
`η = 2^-25`), which gives the second clause. Since `|S_t| + L_t ≤ A` (`A` the sum of all lifted
magnitudes), `tol ≤ ((1 + ρ)^T − 1)·A + T·(1 + ρ)^T·α`, whose relative factor is below `0.21%` at every
covered `K` (`tol_rel_lt`).

**Meaningful.** The bound follows the exact partial sums, so on zero-mean data it is far below
`Σ|a·b|`: on the benchmark's `torch.randn` inputs its median is about `0.07` at `K = 4096` (at most `0.23`)
against a median `|S|` of `43`. `check_gpu.py` measures the device's error at `0.48–0.83` of the
tolerance on the setup's launches (`0.96` at `K = 1` and `0.98` with subnormal operands, where the output's
rounding is the whole error), and it checks that three wrong kernels break `Post` on the test's and
the benchmark's data (`K = 512 … 4096`): the exact result rounded to `bfloat16` (largest ratio `7` to
`32`), a `float16` accumulator rounded after each step (`6` to `8`), and a result missing the last
step's products (`890` and more).

## Exclusions

Nothing `SETUP.md` asks for is left out. Conditions that exclude launches outside `SETUP.md`'s
launches:

- `K > 2^13` (outside the setup, whose largest `K` is `4096`): the tolerance grows with `K` and the
  room argument of Tolerances is stated for `K ≤ 2^13`. The kernel is believed correct there.
- Strides of `2^31` or more, or sizes that make Triton pass an argument as `int64`: outside the
  setup; a different model (the arguments' integer class), not a kernel fault.
- `ACTIVATION = "leaky_relu"`, other block configurations, `float8e5` operands: not part of the setup.

Kernel limits (the kernel is wrong there under the library's semantics; real use does not reach
them):

- A zero or negative stride: `tl.assume(stride > 0)` (lines 268–273) is false, undefined behaviour.
  Not reached: `matmul`'s tensors have positive strides (a broadcast `b` from `expand` would have
  stride `0`).
- `int32` wrap of the offsets (`a_fit`, `b_fit`, `cm_fit`, `cn_fit`, `M_le`, `N_le`): with `int32`
  arguments, the offsets of lines 285, 286 and 314 wrap once a matrix spans `2^31` elements of its
  storage; the first launch that hits it through `matmul` is `M = 2^31 / K + 1` rows (for `K = 4096`,
  `M = 524289`: a `4 GiB` `a`). The setup's largest matrices have `2^24` elements.
- An output whose elements overlap in storage (`c_inj`), or that overlaps `a` or `b` (`c_a`, `c_b`):
  two stores to one element, or a read of what another program writes. Not reached: `matmul`
  allocates a fresh `c`.
- More than `2^31 − 1` programs (`grid_lt`): CUDA's grid limit. Not reached (`M·N` would be beyond
  `2^46`).

## Library requests

None.

## Checks

1. **Builds**: `lake env lean ../claims/matmul_f16/Claim.lean`: no errors or warnings, no `sorry`, no
   `axiom`; `consistent`, `satisfiable`, `tol_rel_lt` and every `setup_*_ok` depend only on
   `propext`, `Classical.choice`, `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (the test's instance, the
   all-zero memory).
3. **Coverage**: 32 `setup_*_ok` (the test and the 31 benchmark sizes).
4. **GPU** (`check_gpu.py`, H200 NVL, Triton 3.8.0): `matmul` on the test and the 31 benchmark sizes
   with the file's draws; edges (`K = 8192`; `K = 4095` with `M = 1000`, `N = 777`; `M = N = 1` with
   `K = 65`; `K = 1`; `K = 0`, where `c = 0` exactly; `M = 0`; inputs scaled by 8 and by `2^-12`, the
   last with subnormal operands; NaN and infinities in `a` and `b`, the other elements checked); two
   direct launches with a padded and a column-major `c` in a guarded buffer and `b` of strides
   `(1, K)`. Both clauses of `OutSpec` checked (the first as: some real `v` within `E` of `S` rounds
   to `c` in `float16`; the check does not snap `v` to the `float32` grid, whose spacing is far below
   every `E` here): largest error / tolerance **0.977** (margin 0.023,
   subnormal operands); `0.75` on the test, `0.83` down to `0.48` on the benchmark. Negative tests on
   the test and at `K = 512, 1024, 2048, 4096`: a `bfloat16`-rounded result, a `float16` accumulator
   and a missing step each break `Post` (Tolerances). Inputs unchanged; no byte outside the matrix
   `c` written. PASS.
