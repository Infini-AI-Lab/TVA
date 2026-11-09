# Prepared claim: `persistent_matmul_f16`

The kernel: `source/09-persistent-matmul.py`, Triton 3.8.0's tutorial 09, the pointer kernel
`matmul_kernel_persistent` (lines 301–357) with its helper `_compute_pid` (lines 287–294), on
`float16` matrices with the config of `SETUP.md` (`BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`,
`BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`, `num_stages = 3`, `num_warps = 8`, and `NUM_SMS`, `132` on
the H200), launched by the host `matmul_persistent(a, b)` (lines 360–380) on a grid of
`min(NUM_SMS, tiles)` programs, each looping over the tiles `start_pid, start_pid + NUM_SMS, …`.
The claim is `Claims.PersistentMatmul.Statement` in `Claim.lean`:

```lean
def Statement : Prop := ∀ K P, Assumptions K P → Spec K
```

`Claim.lean` proves `consistent`, `satisfiable`, `tol_rel_lt` and the coverage facts `setup_*_ok`;
it does not prove `Statement`. The model map is `model-map.json` (every code line of the kernel
lines `[40, 41], [287, 294], [301, 357]`).

The kernel reads three module-level names: `triton` (its decorators) and `tl`, bound by the imports
of lines 40–41, which are kernel lines and are mapped (as `import` lines that add no step), and the
`@triton.jit` helper `_compute_pid`, bound only by its `def` (line 288). Every other module-level
statement is a host line (`claim.json`'s `source.host`): the other imports, the device helpers,
`_matmul_launch_metadata` (lines 85–96, the host function the decorator's `launch_metadata=` names),
the config lists, the file's other kernels and their hosts, `matmul_persistent`, the test, the
benchmark and `__main__`.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.PersistentMatmul.main : Statement` (`Statement`:
`Claim.lean` line 417). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 209), and every initial memory `init` that `Pre` accepts (line 301), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 387).

**What `Post` asks** (line 387), two parts:
1. for every `i < M`, `j < N`: `OutSpec d init fin i j` (line 370). Element
   `i·stride_cm + j·stride_cn` of `c` reads in `fin` as some `float16` `c`. This holds for every
   element. Only when row `i` of `a` and column `j` of `b` read in `init` as finite values `va k`,
   `vb k` (`k < K`) it also asks for a finite `float32` `v` with
   - `|v.toRat - dotSum va vb d.K| ≤ tol va vb d.K` and `FloatV.NaNEq (FloatV.cvt v) c`;
   - if `|dotSum va vb d.K| + tol va vb d.K ≤ Format.binary16.maxVal`: `c` is finite and
     `|c.toRat - dotSum …| ≤ tol … + (|dotSum …| + tol …) * Format.binary16.u + Format.binary16.eta`.

   `dotSum` (line 329) is the exact sum `Σ_k va k·vb k` in the rationals. `tol` (line 362) is
   `errAt (prodV …) (liftV …) (steps K)`: the recursion `errAt` (line 355) over the
   `steps K = cdiv(K, 64)` inner steps (line 307), with `rho = relDot16 64`, `alpha = absDot16 64`
   (lines 311, 314), the exact partial sums `partSum` (line 343) and the steps' lifted magnitudes
   `stepMag` of `liftV` (lines 346, 339);
2. the frame: every byte outside `d.c.elems (cIdx d)` (`cIdx`, line 202) is unchanged.

**What you are given** (`Assumptions`, line 395):
- `faithful` (`Compiler.Faithful`): the device's outcomes are the model's, for a valid launch with
  `debug = false` that is well behaved from `init`;
- `dot` (`Rules.Num.DotF16`, a `DotLiftAcc`): the `tl.dot` of source line 345, on a finite row, column and
  accumulator, gives a finite element within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`,
  `L = |acc| + Σ liftMag a·liftMag b`, when that leaves room in `float32`;
- `dot_total` (`Rules.Num.DotTotal`): that `tl.dot` always has a result.

**The kernel, as the model runs it** (`matmul_kernel_persistent` line 57, `compute_pid` line 31,
`model` line 283, `launch` line 294): a 1-D grid of `min(NUM_SMS, tiles d)` programs (`tiles`,
line 191). Program `start_pid` runs an outer `tl.forRange` over the tiles
`tile_id = start_pid, start_pid + NUM_SMS, … < num_tiles`. Each trip maps `tile_id` to
`(pid_m, pid_n)` by `compute_pid`, forms `offs_am`, `offs_bn` (a `where` to `0` outside `M`, `N`),
checks the hint pair `tl.max_contiguous_multiple_of`, and runs an inner `tl.forRange` of
`steps K` trips: masked loads of the `[128, 64]` tile of `a` and the `[64, 256]` tile of `b`
(`other` `+0`), then `acc ← tl.dot(a, b, acc)`. It then advances `tile_id_c` by `NUM_SMS`, maps it
again by `compute_pid`, rounds to `float16` and stores the `[128, 256]` tile of `c` masked by `M`,
`N`. Each program writes several tiles, one per outer trip; no atomic. Programs write disjoint sets
of tiles and only read `a` and `b`: the route is `TVA.meets_independent`. The outer loop is
`safe_forRange` with per-trip footprints `Wt t` (the `c` tile of `start_pid + t·NUM_SMS`),
disjoint across trips; the inner loop is `safe_forRange` with an invariant on the accumulator
(each element within `errAt … t` of the partial sum) and `Wt := fun _ => ∅` (`iterW_empty`)
(PROVER_GUIDE §4).

**What the proof must establish.**
- each launch is valid and has `debug = false` (`Launch.valid_grid1`: fewer than `2^31` programs
  from `sms`; the ten block shapes are powers of two);
- each program is `Safe`: the outer loop's bounds are valid (`tl.RangeOk`, from `sms`,
  `tiles_le`); `safe_idiv`, `safe_irem` have no bad lane; `compute_pid` on `tile_id_c` gives the
  same tile as on `tile_id`, with `pid_m < cdiv(M, 128)`, `pid_n < cdiv(N, 256)`; the hints hold
  (`safe_max_contiguous_multiple_of`, from `M_tile`, `N_tile`); the `int32` values do not wrap
  (`M_le`, `N_le`, `K_le`, `tiles_le`, `am` … `cn`, `a_fit`, `b_fit`, `cm_fit`, `cn_fit`); every
  enabled load lane is inside `a`, `b` (`a_len`, `b_len`) and every enabled store lane inside `c`
  (`c_len`), with distinct lanes (`c_inj`);
- the tiles of different trips and of different programs are distinct (the residues of
  `tile_id` modulo `NUM_SMS`, and the tile map one-to-one), so the write footprints are disjoint
  (`c_inj`), and no program reads what another writes (`c_a`, `c_b`);
- every `K` tile is full: `128 ∣ K` (`K_dim`) and `BLOCK_SIZE_K = 64 ∣ 128`, so the masks of
  source lines 343–344 enable every lane on every step (`K - ki·64 ≥ 64`), `steps K = K/64`, and
  no lane loads `other`; with `M_tile`, `N_tile`, every tile is whole in all three dimensions;
- the value part of part 1: per inner trip, `safe_dot` with `dot` (and `dot_total`) moves the
  invariant from `errAt … t` to `errAt … (t + 1)`; `K_le` keeps the room condition; the final
  rounding is `FloatV.cvt_err`. `## Tolerances` gives the derivation;
- the bytes no program writes give part 2.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `hostInst_ok`
(line 481) and the `setup_*_ok` theorems; reusable: `tol_rel_lt` (line 319), `row_lt` (line 450),
`rm_inj` (line 456), `bytes_disjoint` (line 471).

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
as `matmul_persistent(a, b)` means it; it does not depend on which program computes which tile,
nor on `NUM_SMS`.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `M`, `N`, `K`, `NUM_SMS`, the six strides (in elements),
and three `float16` buffers `a`, `b`, `c` (`Buf .f16`: a base address and a length in elements);
element `(i, k)` of the matrix `a` is element `i·stride_am + k·stride_ak` of the buffer `a`, and
likewise for `b` and `c` (`aIdx`, `bIdx`, `cIdx`). `tiles d = cdiv(M, 128)·cdiv(N, 256)`.

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `M_tile`, `N_tile` | `M % 128 = 0`, `N % 256 = 0` | whole tiles: on a partial tile the `tl.where` of lines 332–333 makes `offs_am` (`offs_bn`) non-contiguous, and the hint pair `tl.max_contiguous(tl.multiple_of(…))` of lines 334–335 is false (Exclusions: a kernel limit) |
| `M_le`, `N_le` | `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31` | `M` and `N` are `int32` arguments, and `tl.cdiv` computes `M + 127` and `N + 255` in `int32` (lines 314–315); the tile rows and columns `pid_m·128 + i`, `pid_n·256 + j` then stay below `2^31` |
| `sms` | `0 < NUM_SMS < 2^31` | the tile loop's step (line 326) is positive (`tl.RangeOk`), and the constexpr is built as an `int32` (lines 321, 326, 347) |
| `tiles_le` | `tiles + NUM_SMS ≤ 2^31` | the `int32` `num_tiles` (line 317) does not wrap, and `num_tiles + NUM_SMS − 1`, the tile loop's bound (`tl.RangeOk`), is an `int32` |
| `K_le` | `K ≤ 2^13` | keeps the tolerance meaningful (Tolerances); the setup's largest `K` is `1024` |
| `N_dim`, `K_dim` | `0 < N`, `0 < K`, each a multiple of `128`, at most `2^17` | `N` and `K` are model dimensions (the output and inner features): real model shapes only (Exclusions) |
| `M_ctx` | `M ≤ 2^22` | `M` counts tokens: real model shapes only (Exclusions) |
| `am`, `ak`, `bk`, `bn`, `cm`, `cn` | each stride is below `2^31` | each stride is an `int32` argument (any value from `0`; the kernel has no `tl.assume` on them) |
| `a_fit` | `(M − 1)·stride_am + (K − 1)·stride_ak < 2^31` | the `int32` offsets of `a_ptrs` (line 340) do not wrap on the lanes the mask of line 343 enables (`offs_am < M` on a whole tile, `offs_k < K`); a disabled lane's offset may wrap and is not accessed |
| `b_fit` | `(K − 1)·stride_bk + (N − 1)·stride_bn < 2^31` | the same for `b_ptrs` (line 341) and the mask of line 344 |
| `cm_fit`, `cn_fit` | `(M − 1)·stride_cm < 2^31`, `(N − 1)·stride_cn < 2^31` | the `int32` products `stride_cm·offs_cm` and `stride_cn·offs_cn` (line 351) do not wrap (each is added to the pointer on its own) |
| `c_inj` | distinct elements `(i, j)` of the matrix `c` are distinct elements of its storage | otherwise two lanes or two programs store to one element; any layout in which the elements do not overlap is covered (row-major, padded rows, column-major) |
| `a_len`, `b_len`, `c_len` | every element of each matrix is inside its buffer | the kernel accesses them |
| `c_a`, `c_b` | the matrix `c` shares no byte with the matrix `a`, nor with the matrix `b` | a program that reads bytes another program writes races; `matmul_persistent` always allocates a fresh `c` |

`M`, `N`, `K` and the strides are the kernel's runtime arguments (or, at the value `1`, compile-time
constants, which give the same model: `model`'s docstring; `M` and `N` are never `1`, being
multiples of the blocks or `0`). Their values are part of the instance, and the conditions above
constrain them; `Compiler.Faithful`'s point 11 assumes that the launch runs the code Triton
compiles for these values. The conditions allow any strides within `int32` (a transposed `b`, as
the setup's, a non-contiguous `a`, any layout of `c` whose elements do not overlap), any `NUM_SMS`
within the bounds (the grid is the host's `min(NUM_SMS, tiles)`), any `M`, `N`, `K` within the
bounds, and `M = 0` (no tile: an empty grid, memory unchanged); `a` and `b` may overlap each other
in any way (both are only read).

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to 2 bytes, the buffer inside `[0, 2^64)`, every byte present). Every
`float16` value is allowed in the inputs, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch of `SETUP.md` is a direct launch of the `@triton.jit` function, the
JIT launch `matmul_kernel_persistent[grid](…)` of `matmul_persistent` (lines 372–379), called by
`run_test` from `validate` (line 715) and, in the benchmark, by `bench_fn` (line 682). Point by
point:

- **4 (options).** The launch gives the constexprs `NUM_SMS`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
  `BLOCK_SIZE_K`, `GROUP_SIZE_M` and two options, `num_stages=3` and `num_warps=8`, two of the four
  that do not change the meaning; no `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or any
  other option. Nothing calls `matmul_kernel_persistent.warmup(…)` or `triton.compile`; the kernel
  has no `@triton.autotune` (so no `triton.Config`) and no `@triton.heuristics`, and its decorator
  `@triton.jit(launch_metadata=_matmul_launch_metadata)` sets no `debug` (`launch_metadata=` only
  names a host function, `_matmul_launch_metadata`, lines 85–96). So `launch` keeps every default:
  `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`, `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor, so no allocator is involved.
- **6 (precompiled kernel).** No `CompiledKernel` is launched: every launch is the JIT launch above.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are three
  tensors, nine integers and the constexprs.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under `@triton.jit(launch_metadata=…)`; its helper `_compute_pid` a
top-level `def` under a bare `@triton.jit`, called by name from the kernel. The kernel's parameters
are three pointers (tensor arguments), nine unannotated integers (`int` arguments) and five
`tl.constexpr`s with `int` arguments, none with a default; the helper's five parameters are
unannotated, with no default, and both calls pass three `int32` scalars and the same two
constexprs. The kernel's statements are assignments `name = e`, two tuple assignments
`pid_m, pid_n = _compute_pid(…)` (one name per element of the helper's tuple result), one
`name += e` (line 347), expression statements (`tl.store`), a `for tile_id in tl.range(…)` with
three positional arguments and the keyword `flatten`, a nested `for ki in range(k_tiles)` with one
positional argument, and an `if`/`else` on a compile-time condition; the helper's are assignments
and one `return pid_m, pid_n`. The expressions are `tl.program_id(axis=0)`, `tl.cdiv`, `tl.arange`,
`tl.where`, `tl.max_contiguous`, `tl.multiple_of`, `tl.zeros(…, dtype=tl.float32)`,
`tl.load(…, mask=…, other=0.0)`, `tl.dot(a, b, acc)`, `x.to(tl.float16)`, `tl.store(…, mask=…)`,
Python's builtins `min` and `range`, the operators `+ - * // % &`, the comparisons `<` and `==`
(`==` only between compile-time values: `c_ptr.dtype.element_ty == tl.float8e4nv`), the
attributes `.dtype` and `.element_ty`, and the subscripts `[:, None]` and `[None, :]`. After each
loop no name is used: the tile loop carries `tile_id_c`, the inner loop `accumulator`, and neither
loop's variable nor a name first defined in a body is read after it. The module-level names the
kernel reads are `triton` and `tl`, Triton's modules, each bound only by its own `import`
(lines 40 and 41), and `_compute_pid`, bound only by its `def`; no other statement of the module
binds `triton`, `tl`, `_compute_pid` or `matmul_kernel_persistent`.

## Assumptions

Three assumptions, all of kind `[rule]`, each cited as the library states it
(`docs/PROVER_GUIDE.md` section 3, [num.md](../../docs/rules/num.md)):

- `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`;
- `dot : Rules.Num.DotF16 P` — `tl.dot` of `float16` operands into `float32` (line 345), for every
  `input_precision`, with `maxImpreciseAcc = 0` (`Rules.Num.DotLiftAcc`): each output element whose
  row of `a`, column of `b` and accumulator are finite is finite and within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a
  subnormal operand counted at `2^-14`), when that leaves room below the `float32` range;
- `dot_total : Rules.Num.DotTotal P {a := .f16, b := .f16, acc := .f32, precision := .tf32,
  maxImpreciseAcc := 0}` — the `tl.dot` of line 345 always has a result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversion of line 356 and the
`other=0.0` conversion of lines 343–344 are IEEE roundings the library defines (`tl.cvt`,
`tl.convert`), and need no platform fact. The tile loop's `flatten=True` and the launch's
`num_stages=3` are scheduling choices with no meaning for a well-behaved model (SEMANTICS.md,
"Performance-only arguments", "Loops"): each program writes each of its tiles once and reads only
`a` and `b`, so no program accesses bytes it wrote.

## Coverage

Every launch of `SETUP.md` ("The launches"): `a` contiguous `[8192, K]` (strides `(K, 1)`), `b.T`
of strides `(1, K)`, the fresh `c` (strides `(8192, 1)`), `M = N = 8192`, `K = 128, 256, …, 1024`,
`NUM_SMS = 132`. `Inst.ok` covers them: `8192` is a multiple of `128` and of `256`, `tiles = 2048`
and `2048 + 132 ≤ 2^31`, `a_fit` reads `8191·K + K − 1 < 2^31`, `b_fit` `K − 1 + 8191·K < 2^31`,
`cm_fit` `8191·8192 < 2^31`, `c_inj` holds for a row-major `c`, `K ≤ 1024 ≤ 2^13`, `K` and
`N = 8192` are positive multiples of `128` below `2^17`, and `M = 8192 ≤ 2^22`.

Proved instances (`hostInst M N K S`: the host's strides, `NUM_SMS = S`, the three buffers at fixed
aligned addresses; `hostInst_ok`):

| Instance | `M = N` | `K` | Source |
|---|---|---|---|
| `setup_k128_ok` | 8192 | 128 | the test `validate(8192, 8192, 128)` (line 765) with `--K_range 128 1024 --K_step 128` (line 31), and `bench(128)` (lines 668–692) |
| `setup_k512_ok` | 8192 | 512 | the default `-K 512` (line 747): `validate(8192, 8192, 512)` and `bench(512)` |
| `setup_k256_ok`, `setup_k384_ok`, `setup_k640_ok`, `setup_k768_ok`, `setup_k896_ok`, `setup_k1024_ok` | 8192 | 256, 384, 640, 768, 896, 1024 | `bench(K)` with `--K_range 128 1024 --K_step 128` |

`satisfiable` uses the `K = 512` instance with the all-zero memory.

## Tolerances

`tol va vb K = errAt (prodV va vb K) (liftV va vb K) (steps K)` (`Claim.lean`), the error bound of
the inner loop's `T = steps K = cdiv(K, 64)` steps against the exact partial sums. With
`p e = va e · vb e` for `e < K` and `0` from `K` on (`prodV`: the lanes the masks disable load
`+0`), `l e` its lifted magnitude `liftMag(va e)·liftMag(vb e)` (`liftV`; `liftMag q` is `|q|` for
zero and normal values, `2^-14` for a nonzero subnormal one), `S_t = Σ_{e < 64t} p e` (`partSum`)
and `L_t = Σ_{64t ≤ e < 64t+64} l e` (`stepMag`):

`E_0 = 0`, `E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + L_t) + α`, `tol = E_T`,

that is `E_T = Σ_{t<T} (1 + ρ)^(T−1−t)·(ρ·(|S_t| + L_t) + α)`, with the library's constants for one
fp16 `tl.dot` of `K = 64`: `ρ = relDot16 64 = 66·2^-22`, `α = absDot16 64 = 66·2^-126`.

**Derivation** (library facts only). For a tile, the inner loop (lines 338–345) starts from
`accumulator = 0` (line 337) and its step `t` (`ki = t`) is `acc ← tl.dot(a, b, acc)`, a `float16`
dot of `64` products with the accumulator as its own (no separate addition): row `i` of `a`'s
columns `64t … 64t + 63` and column `j` of `b`'s rows `64t … 64t + 63`, the lanes `≥ K` loaded as
`+0`. By `Rules.Num.DotF16` (`DotLiftAcc` with `K = 64`) each output element of it is within
`relDot16 64·(|acc| + L_t) + absDot16 64` of `acc + c_t`, `c_t = S_{t+1} − S_t` the step's exact
sum (the disabled lanes hold `+0`, adding nothing to `c_t` or `L_t`). If `acc` is within `E_t` of
`S_t` then `|acc| ≤ |S_t| + E_t`, so the new accumulator is within
`E_t + ρ·(|S_t| + E_t + L_t) + α = E_{t+1}` of `S_{t+1}`; by induction the loop leaves a `float32`
value within `E_T` of `S`. The fact's room condition holds at every step: finite `float16` products
are below `2^32`, so with `K ≤ 2^13` every `|acc| + L_t` and its bound stay far inside the `float32`
range. The final `.to(tl.float16)` (line 356) is one rounding (`FloatV.cvt_err`, `u = 2^-11`,
`η = 2^-25`), which gives the second clause. The tile loop only chooses which program computes
which tile, and each tile's computation is the same, so the bound holds for every element whatever
`NUM_SMS`. Since `|S_t| + L_t ≤ A` (`A` the sum of all lifted magnitudes),
`tol ≤ ((1 + ρ)^T − 1)·A + T·(1 + ρ)^T·α`, whose relative factor is below `0.21%` at every covered
`K` (`tol_rel_lt`).

**Meaningful.** The bound follows the exact partial sums, so on zero-mean data it is far below
`Σ|a·b|`: on the setup's `torch.randn` inputs its median is `0.0014` at `K = 128` (at most `0.0025`)
against a median `|S|` of `7.6`, and `0.014` at `K = 1024` (at most `0.035`) against `21.6`.
`check_gpu.py` measures the device's error at `0.74–0.92` of the tolerance on the setup's launches
(`0.97` at `K = 1` and `0.98` with subnormal operands, where the output's rounding is the whole
error), and it checks that three wrong kernels break `Post` at `K = 128, 512, 1024`: the exact
result rounded to `bfloat16` (largest ratio `24` to `93`), a `float16` accumulator rounded after
each step (`8` to `10`), and a result missing the last step's products (`3970` and more).

## Exclusions

Launches `SETUP.md` names that the claim leaves out:

- **Partial tiles: `M` not a multiple of `128` or `N` not a multiple of `256` (`M_tile`,
  `N_tile`) — a kernel limit.** On a partial tile, `tl.where(offs_am < M, offs_am, 0)` (line 332)
  replaces the rows past `M` by `0`, so `offs_am` is not `128` consecutive values, and the kernel's
  own hint `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)` (line 334;
  likewise line 335) is false: under the library's semantics a false hint is undefined behaviour,
  and the compiled loads read other elements than the source names there (the outputs are not
  affected, since the store mask of line 352 drops those lanes). The file's own first test,
  `validate(32, 32, 32)` (line 764), reaches it, and so does any `matmul_persistent` call whose `M`
  or `N` is not a multiple of the blocks; such launches are out of scope (a kernel limit: the hints are
  false there).
Conditions that exclude launches outside `SETUP.md`'s launches:

- **Dimensions that are not multiples of 128 or exceed 2^17, and contexts longer than 2^22
  tokens** — out of scope: the claim covers real model shapes only. The fields: `N` and `K`
  (`N_dim`, `K_dim`: positive multiples of `128`, at most `2^17`) and `M`, a token count (`M_ctx`:
  at most `2^22`). Every launch of `SETUP.md` is covered (`M = N = 8192`, `K = 128·i`); this also
  leaves out `K = 0` and `N = 0`.
- `K > 2^13` (outside real use: the setup's largest `K` is `1024`): the tolerance grows with `K`
  and the room argument of Tolerances is stated for `K ≤ 2^13`. The kernel is believed correct
  there.
- `NUM_SMS = 0` or `NUM_SMS ≥ 2^31` (`sms`; outside real use: the host passes the device's
  multiprocessor count, `132`). With `NUM_SMS = 0` the grid is empty and `c` is not written.
- Strides of `2^31` or more, or sizes that make Triton pass an argument as `int64`: outside real
  use; a different model (the arguments' integer class), not a kernel fault.
- `float8e4nv` operands (`--prec fp8`), other block configurations, the file's other kernels: not
  part of the setup.

Kernel limits (the kernel is wrong there under the library's semantics; real use does not reach
them):

- `int32` wrap of the offsets (`a_fit`, `b_fit`, `cm_fit`, `cn_fit`, `M_le`, `N_le`): with `int32`
  arguments, the offsets of lines 340, 341 and 351 wrap once a matrix spans `2^31` elements of its
  storage (`a`: `M·K`, `b`: `N·K`, `c`: `M·N` elements through `matmul_persistent`); with the
  setup's `N = 8192` and `K = 1024`, the first launch that hits it is `M = 2^18 + 128` rows
  (`cm_fit`: a `4 GiB` `c`). The setup's largest matrices have `2^26` elements (`c`).
- `tiles + NUM_SMS > 2^31` (`tiles_le`): `num_tiles` (line 317) or the tile loop's bound wraps;
  this needs `M·N` beyond `2^46`. Not reached.
- An output whose elements overlap in storage (`c_inj`), or that overlaps `a` or `b` (`c_a`, `c_b`):
  two stores to one element, or a read of what another program writes. Not reached:
  `matmul_persistent` allocates a fresh `c`.

## Library requests

None.

## Checks

1. **Builds**: `lake env lean ../claims/persistent_matmul_f16/Claim.lean`: no errors or warnings,
   no `sorry`, no `axiom`; `consistent`, `satisfiable`, `tol_rel_lt` and every `setup_*_ok` depend
   only on `propext`, `Classical.choice`, `Quot.sound`.
2. **Not vacuous**: `consistent` (`Platform.sm90ref`), `satisfiable` (the `K = 512` instance, the
   all-zero memory).
3. **Coverage**: 8 `setup_*_ok` (`M = N = 8192`, `K = 128, 256, …, 1024`, `NUM_SMS = 132`).
4. **GPU** (`check_gpu.py`, H200 NVL, Triton 3.8.0, `NUM_SMS = 132`): `matmul_persistent` on the
   eight setup sizes with the file's draws and layout; edges (`K = 8192`; `K = 4095` with
   `M = 1408`, `N = 768`; one tile with `K = 65`; `K = 1`; `K = 0`, where `c = 0` exactly; `M = 0`;
   inputs scaled by 8 and by `2^-12`, the last with subnormal operands; NaN and infinities in `a`
   and `b`, the other elements checked); four direct launches with `NUM_SMS = 1, 7, 132, 1000`
   (grids `1`, `7`, `132`, `40`), a padded or column-major `c` in a guarded buffer and `a` of
   strides `(1, M)`. Both clauses of `OutSpec` checked (the first as: some real `v` within `E` of
   `S` rounds to `c` in `float16`; the check does not snap `v` to the `float32` grid, whose spacing
   is far below every `E` here): largest error / tolerance **0.977** (margin 0.023, subnormal
   operands); `0.92` down to `0.74` on the setup's sizes. Negative tests at `K = 128, 512, 1024`: a
   `bfloat16`-rounded result, a `float16` accumulator and a missing step each break `Post`
   (Tolerances). Inputs unchanged; no byte outside the matrix `c` written. PASS. The launches with
   `K = 4095, 1000, 777, 700, 300, 65, 1, 0` are outside `K_dim`: they check `Post` beyond the
   claim.
