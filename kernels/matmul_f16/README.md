# matmul_f16

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/03-matrix-multiplication.py`](source/03-matrix-multiplication.py). Lean: [`lean/`](lean/),
the module `Matmul` (namespace `Kernels.Matmul`) and its parts `Matmul.Defs` (model, conditions,
specification, assumptions), `Matmul.Arith` (the program order), `Matmul.Basic` (arithmetic and
float lemmas) and `Matmul.Program` (one program).

The kernel is `matmul_kernel` (source lines 228–316). It reads two module-level names, `triton`
(its decorator) and `tl`, bound by the imports of lines 154–155, which are kernel lines too and
are mapped (dropped: they add no step). The host function `matmul(a, b, activation="")` (lines
330–350) checks `a.shape[1] == b.shape[0]` and that `a` is contiguous, allocates
`c = torch.empty((M, N), dtype=torch.float16)` (row-major, strides `(N, 1)`), and launches a 1-D
grid of `triton.cdiv(M, 128) * triton.cdiv(N, 256)` programs with `BLOCK_SIZE_M=128`,
`BLOCK_SIZE_N=256`, `BLOCK_SIZE_K=64`, `GROUP_SIZE_M=8`, `num_stages=3`, `num_warps=8` and
`ACTIVATION=activation`. The script calls it on `float16` matrices in the test (`512 × 512 × 512`,
`torch.rand(…) - 0.5`, lines 359–370) and in the benchmark (`M = N = K = 256, 384, …, 4096`,
`torch.randn`, lines 400–440, the `fp16` line), never with an `activation`. Its `float8e5` run
(lines 372–388, and the benchmark's `fp8` line) is not this folder's (Narrowings).

Every other module-level line is a host line: `import torch` (152), `DEVICE = …` (157), `is_cuda`
(160–161, called by the host at lines 373, 400, 404), the test, the benchmark's configs and
`benchmark` (423–437) with its run (440). The functions `get_cuda_autotune_config`,
`get_hip_autotune_config`, `get_autotune_config` and the `@triton.jit` helper `leaky_relu` are
unused (`source.unused`): nothing calls the first three, and the kernel's only call of
`leaky_relu`, line 307, is in the branch `ACTIVATION == "leaky_relu"`, which the front end does
not visit on covered launches. Their headers (lines 164, 202, 216, 320–321) are host lines: they
bind `get_cuda_autotune_config`, `get_hip_autotune_config`, `get_autotune_config` and
`leaky_relu`, which the launch does not read.

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `i < M` and `j < N`, element `(i, j)` of `c` (element
  `i·stride_cm + j·stride_cn` of `c`'s storage) can be read in the final memory, with some value
  `c` (`OutSpec`). When, moreover, row `i` of `a` and column `j` of `b` hold finite `float16`
  values `va k` and `vb k` (`k < K`) in the initial memory:
  - `c` is the `float16` rounding (to nearest even, `FloatV.cvt`, up to NaN payload) of a finite
    `float32` value `v` with `|v − S| ≤ tol K A`, where `S = Σ_k va k · vb k` is the exact dot
    product and `A = Σ_k mag16(va k)·mag16(vb k)` (`dotMag`) the sum of the products' lifted
    magnitudes: `mag16 q` is `|q|`, except that a nonzero `q` below `2^-14` (a subnormal `float16`)
    counts as `2^-14`, as the tensor core aligns it (`Rules.Num.liftMag .f16`, which `DotF16`
    counts by). With no subnormal operand, `A = Σ_k |va k · vb k|`; a subnormal `va k` adds at most
    `2^-14·|vb k| + 2^-28` to its term, and likewise for `vb k` (`DotLift.liftMag_mul_le`);
  - when `|S| + tol K A ≤ 65504` (the largest `float16`), `c` is finite and
    `|c − S| ≤ tol K A + (|S| + tol K A)·2^-11 + 2^-25` (one more `float16` rounding:
    `Format.binary16.u = 2^-11`, `Format.binary16.eta = 2^-25`).

  The tolerance is `tol K A = DotLoop.bound ρ α T A = ((1 + ρ)^T − 1)·A + T·(1 + ρ)^T·α`, with
  `T = cdiv(K, 64)` `tl.dot` steps and the library's constants for one fp16 `tl.dot` of
  `K = 64` (`Rules.Num.DotF16`): `ρ = relDot16 64 = 66·2^-22`, `α = absDot16 64 = 66·2^-126`. It
  is explicit and computable: `(1 + ρ)^T − 1` is about `1.26·10^-4` at `K = 512` (the test),
  `1.01·10^-3` at `K = 4096` (the benchmark's largest), and `2.02·10^-3` at `K = 2^13` (the
  largest covered; below `0.21%` at every covered `K`, `tol_rel_lt`), and the `α` term is below
  `10^-34`. So at the largest covered size the accumulation is within `0.21%` of `A`: about
  half the largest relative error of a rounding to `bfloat16` (`2^-8/(1 + 2^-8)`, about `0.39%`),
  so a kernel that rounds its result to `bfloat16` breaks `Post` on some inputs (normal operands
  whose products have one sign, so `A = |S|`, and a sum where the `bfloat16` rounding error is near
  its largest), and so does one that drops or misplaces a product. The bound is relative to `A`
  (`Σ|a·b|` without subnormal operands), not to `|S|`: for zero-mean inputs such as the benchmark's `torch.randn`, `Σ|a·b|` is about
  `(2/π)·K` while `|S|` is typically `√K`, so the bound is about `4%` of a typical `|S|` at
  `K = 4096` and `12%` at `K = 2^13`. (`K_le` stops at `2^13` for this reason: from about
  `K = 15745` the bound would no longer reject a `bfloat16` rounding, and from about `K = 34368` it
  would exceed a typical `|S|` of zero-mean inputs.)

  With a NaN or an infinity in row `i` of `a` or column `j` of `b`, the claim says only that
  element `(i, j)` holds some `float16` value.
- **Nothing else changes.** Every byte outside the elements of the matrix `c` is the same in `fin`
  as in `init`: `a`, `b`, and the elements of `c`'s storage that are not in the matrix (padding of a
  row, say).

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
| `K_le` | `K ≤ 2^13` | keeps the tolerance meaningful: below `0.21%` of `A` (`Σ\|a·b\|` without subnormal operands; `tol_rel_lt`), about half a `bfloat16` rounding (Specification, Narrowings); the host's largest `K` is `4096` |
| `am`, `ak`, `bk`, `bn`, `cm`, `cn` | each stride is positive and below `2^31` | the kernel's `tl.assume(stride_… > 0)` (lines 268–273) is undefined behaviour otherwise; each stride is an `int32` argument |
| `a_fit` | `M·stride_am + 64·stride_ak ≤ 2^31` | the `int32` offsets of `a_ptrs`, `offs_am·stride_am + offs_k·stride_ak` (line 285), and the step `64·stride_ak` (line 302) do not wrap |
| `b_fit` | `64·stride_bk + N·stride_bn ≤ 2^31` | the same for `b_ptrs` (line 286) and its step (line 303) |
| `cm_fit` | `M·stride_cm < 2^31` | the `int32` products `stride_cm·offs_cm` (line 314) of the rows below `M` do not wrap (and the grid has fewer than `2^31` programs, `grid_lt`) |
| `c_rows` | `N·stride_cn ≤ stride_cm` | each row of `c` lies within `stride_cm` (row-major, possibly padded), so the rows do not overlap (`c_inj`) and no two lanes or programs store to one element; with `stride_cn > 0` the products `stride_cn·offs_cn` (line 314) of the columns below `N` stay below `stride_cm < 2^31`. It also excludes column-major and transposed layouts of `c`, whose rows do not overlap (Narrowings) |
| `a_len`, `b_len`, `c_len` | every element of each matrix is inside its buffer | the kernel accesses them |
| `c_a`, `c_b` | the matrix `c` shares no byte with the matrix `a`, nor with the matrix `b` | a program that reads bytes another program writes races; `matmul` always allocates a fresh `c` |

`M`, `N`, `K` and the strides are the kernel's runtime arguments (or, at the value `1`, compile-time
constants, which give the same model: `model`'s docstring). Their values are part of the instance,
and the conditions above constrain them; `Compiler.Faithful`'s point 11 assumes that the launch
runs the code Triton compiles for these values.

**The host's use is covered.** The test (`M = N = K = 512`) and every benchmark size
(`M = N = K ≤ 4096`) meet `Inst.ok`: `a` is contiguous (`matmul` asserts it: strides `(K, 1)`),
`b` comes from `torch.rand`/`torch.randn` (strides `(N, 1)`) and `c` from `torch.empty` (strides
`(N, 1)`), so `a_fit` reads `M·K + 64 ≤ 2^31`, `b_fit` reads `65·N ≤ 2^31`, `cm_fit` reads
`M·N < 2^31` and `c_rows` reads `N ≤ N` (at `4096`: `2^24 + 64`, `266240`, `2^24`); `K ≤ 4096 ≤ 2^13`;
the fresh `c` overlaps neither input. The conditions are more general than that use: any positive
strides (a transposed `b`, a non-contiguous `a`), any `M`, `N`, `K` within the bounds, `M = 0` or
`N = 0` (an empty grid: memory unchanged) and `K = 0` (no loop trip: `c = 0`). `a` and `b` may
overlap each other in any way (both are only read).

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to 2 bytes, the buffer inside `[0, 2^64)`, every byte present). Every
`float16` value is allowed in the inputs, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch the script makes is a direct launch of the `@triton.jit` function,
the JIT launch `matmul_kernel[grid](…)` of `matmul` (lines 340–349), called at line 362 (and at line
381, in the `float8e5` run, which this folder does not cover) and, in the benchmark, through
`triton.testing.do_bench(lambda: matmul(a, b), …)` (line 435). Point by point:

- **4 (options).** The launch gives the constexprs `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`,
  `GROUP_SIZE_M`, `ACTIVATION` and two options, `num_stages=3` and `num_warps=8`, two of the four
  that do not change the meaning; no `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or any
  other option. Nothing calls `matmul_kernel.warmup(…)` or `triton.compile`; the kernel has no
  `@triton.autotune` (so no `triton.Config`; the config lists `get_…_autotune_config` build are
  never used) and no `@triton.heuristics`, and the decorator is a bare `@triton.jit`, so its `debug`
  is not set. So `launch` keeps every default: `enable_fp_fusion = true`, `debug = false`,
  `num_ctas = 1`, `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor (no
  `tl.make_tensor_descriptor`), so no allocator is involved.
- **6 (precompiled kernel).** No `CompiledKernel` is launched: every launch is the JIT launch above.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are three
  tensors, nine integers and the constexprs.
- **9 (CUDA graphs).** No call is made during a CUDA stream capture, and nothing is replayed: the
  script calls `matmul` directly (line 362) and through `triton.testing.do_bench` (line 435), which
  times plain calls of its function with CUDA events, not `triton.testing.do_bench_cudagraph`; the
  script opens no `torch.cuda.graph` and makes no capture of its own.

So a call of `matmul` makes this one direct launch, which the claim covers, from the memory at that
launch.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`, not under `@triton.autotune`. Its parameters
are three pointers (tensor arguments), nine unannotated integers (`int` arguments) and five
`tl.constexpr`s, four with `int` arguments and `ACTIVATION` with the `str` argument `""`, none with a
default. Its statements are a docstring, assignments `name = e`, two `name += e` (lines 302–303),
expression statements (the calls `tl.assume` and `tl.store`), one `for k in range(0, …)` with two
positional arguments, and one `if` on a compile-time condition. Its expressions are
`tl.program_id(axis=0)`, `tl.cdiv`, `tl.arange`, `tl.zeros(…, dtype=tl.float32)`,
`tl.load(…, mask=…, other=0.0)`, `tl.dot(a, b, acc)`, `x.to(tl.float16)`,
`tl.store(…, mask=…)`, `tl.assume`, Python's builtins `min` and `range`, the operators
`+ - * // % &`, the comparisons `>= > < ==` (one per expression; `==` only between compile-time
constants), and the subscripts `[:, None]` and `[None, :]`. The only module-level names it reads
are `triton` (its decorator) and `tl`, Triton's modules, each bound only by its own `import` (lines
154 and 155); line 307 names the module's `@triton.jit` function `leaky_relu`, bound only by its
`def`, but the front end does not visit that line. No other statement of the module binds `triton`,
`tl`, `matmul_kernel` or `leaky_relu` (the others bind `torch`, `DEVICE`, `is_cuda`, the three
`get_…_autotune_config`, `matmul`, `a`, `b`, `triton_output`, `torch_output`, `TORCH_HAS_FP8`,
`ref_lib`, `configs`, `fp8_inputs` and `benchmark`; no decorator, default or annotation binds a
name).

## Assumptions

Three assumptions, all of kind `[rule]`, each cited as the library states it:

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid (`Launch.Valid`), then from every initial memory from
  which the model's launch is well behaved (no undefined behaviour, no race between programs, no
  conflict within a program), every outcome of the compiled kernel launched as `launch d` is one
  of the model's: a normal end with a memory some interleaving of the model leaves, never a fault,
  and a hang only if some interleaving runs forever. The rule covers a launch only within its
  scope, which its text states: part (A), which the launches meet (Conditions), and part (B), the
  environment, which the claim relies on
  ([rule page](../../docs/rules/compiler-faithful.md)).
- `dot` — `Rules.Num.DotF16 P`, the library's fact for `tl.dot` of `float16` operands into
  `float32` (line 300; [num.md](../../docs/rules/num.md)), as restated in library round 25
  (`Rules.Num.DotLiftAcc`): for every `input_precision`, with `maxImpreciseAcc = 0`, each output
  element whose row of `a` and column of `b` are finite and whose accumulator is finite is finite
  and within `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, with
  `L = |acc| + Σ liftMag a·liftMag b` (a subnormal operand counted at `2^-14`), when that leaves
  room below the `float32` range.
- `dot_total` — `Rules.Num.DotTotal P cfg` for the model's configuration (`float16` operands,
  `float32` accumulator, `tf32`, `maxImpreciseAcc = 0`): the `tl.dot` of line 300 always has a
  result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversion of line 308 and the
`other=0.0` conversion of lines 297–298 are IEEE roundings the library defines (`tl.cvt`,
`ConvRel`), and need no platform fact.

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false` (Triton's default) by definition of `launch`.
- `hvalid` (`launch_valid`, by `Launch.valid_grid1`): the grid `cdiv(M, 128)·cdiv(N, 256)` has
  fewer than `2^31` programs (`grid_lt`: at most `M·N ≤ M·stride_cm < 2^31`, by `c_rows` and
  `cm_fit`), and each of the ten block shapes is a list of powers of two with at most `2^20`
  elements.
- Footprints: every program reads within the matrices `a` and `b` (`R`), and program `p` writes
  the elements of its tile of `c`, rows `pm·128 + i < M` and columns `pn·256 + j < N` (`W`, with
  `pm`, `pn` its tile).
- `hsafe` (`safe_program`, in `Matmul.Program`): one rule per line, from `program_id = p`
  (`launch_pid`).
  - Lines 252–260 (`safe_idiv1`, `safe_irem1`): every `int32` operation stays in range and computes
    the natural-number formulas of `Matmul.Arith` (`fpmOf`, `gsOf`, `pmOf`, `pnOf`), so `pid_m` and
    `pid_n` are `pm d p` and `pn d p`, with `pm < cdiv(M, 128)` and `pn < cdiv(N, 256)`.
  - Lines 266–273 (`safe_assume`): `pid_m, pid_n ≥ 0` and the strides are positive.
  - Lines 282–286 (`safe_irem`): `offs_am` and `offs_bn` are `(pm·128 + i) mod M` and
    `(pn·256 + j) mod N` (`rowA`, `colB`); `a_ptrs` and `b_ptrs` are `aPtrs d p 0` and
    `bPtrs d p 0` (no `int32` offset wraps: `a_fit`, `b_fit`).
  - Line 294 (`safe_idiv1`, `safe_forRange`): the loop runs `T = cdiv(K, 64)` trips, with the
    invariant: memory unchanged, `a_ptrs = aPtrs d p t`, `b_ptrs = bPtrs d p t`, and `AccInv`: every
    element of the tile whose row of `a` and column of `b` are finite is finite and within
    `DotLoop.bound ρ α t (partA t)` of the exact partial sum `partS t` (the products of
    `k < 64·t`). It holds at `t = 0` (`acc0`: `+0.0`, exact).
  - Lines 297–298 (`safe_convert`, `safe_load_buf`): `other=0.0` converts to a finite `float16`
    zero (`zero_conv`); each enabled lane (`64·t + k < K`) loads its element of `a` (resp. `b`),
    inside the matrix, each disabled lane the zero.
  - Line 300 (`safe_dot`, with `dot_total`), then `acc_step`: by `dot` (`Rules.Num.DotF16`, at
    `K = 64`, relative to the chunk's lifted magnitudes, `liftMag_f16`) and `DotLoop.accum_step`,
    the accumulator keeps `AccInv` at `t + 1`, whose magnitude `partA` sums the lifted products
    `liftV` (the disabled lanes add exact zeros, of lifted magnitude `0`; `|S_t| ≤ partA` by
    `abs_mul_le_mag16`; the values stay far inside the `float32` range because `K ≤ 2^13`;
    `acc_step` needs only `K ≤ 2^16`).
  - Lines 302–303 (`safe_ret`): the pointers advance to `aPtrs d p (t + 1)` and `bPtrs d p (t + 1)`.
  - Line 308 (`safe_cvt`): `c` is the rounding of the accumulator to `float16`, up to NaN payload.
  - Lines 312–316 (`safe_last`, `safe_store_buf`): enabled lanes (`c_mask`) store to element
    `(pm·128 + i, pn·256 + j)` of `c`, distinct lanes to distinct elements (`c_inj`, from `c_rows`).
    After `T` trips the partial sums are the whole sums (`partS_end`), so `OutSpec` follows from
    `AccInv` and the last rounding (`out_f16`, by `FloatV.cvt_err`).
- `hdisj`: program `i`'s tile of `c` is disjoint from `a` and `b` (`W_sub`, `c_a`, `c_b`) and,
  for `i ≠ j`, from program `j`'s tile (`W_disj`: two lanes storing to one element have the same
  row and column, `c_inj`, hence the same tile, and the program order is injective, `pid_inj`).
- `hRW`: program `p`'s tile of `c` is disjoint from `a` and `b` (as above).
- `hpost`: element `(i, j)` of `c` is in the tile `(i / 128, j / 256)`, which some program of the
  grid has (`pid_surj`: the program order is a bijection onto the tiles); the final memory agrees
  with that program's memory on its footprint, so `Mem.read_congr` gives the value. A byte outside
  the matrix `c` is in no write footprint, so it keeps its initial value.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref` gives a total behaviour that
satisfies `faithful`; `Platform.sm90ref.dotF16` gives `dot` and `Platform.ref.dotTotal` gives
`dot_total`.

`satisfiable`: the source's own test instance (lines 359–362), `M = N = K = 512` with row-major
`a`, `b` and `c` (strides `512` and `1`), each of `512·512` elements, at the 16-byte-aligned
addresses `2^21`, `2·2^21` and `3·2^21` (`testInst`), in a memory where every byte is present
(`testMem`), meets `Inst.ok` and `Pre`.

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## What changed (library round 25)

Library round 25 (issue L126) restated `Rules.Num.DotF16`: the tensor core reads a subnormal
`float16` operand's exponent as the smallest normal's, so a product with a subnormal operand can set
a chunk's alignment window while far smaller, and the old bound, relative to `|acc| + Σ|a·b|`, failed
on such inputs. The fact is now `DotLiftAcc`, relative to `|acc| + Σ liftMag a·liftMag b`, with the
same constants. The claim follows it:

- **The specification.** The tolerance's magnitude `A` is now `dotMag`, the sum of the products'
  lifted magnitudes (`mag16`, a subnormal operand counted at `2^-14`), in place of
  `Σ|va k · vb k|` (`dotAbs`, removed). The formula `tol K A` and its constants are unchanged.
  Subnormal inputs stay covered, with the larger `A` they need.
- **The figures.** With no subnormal operand `A = Σ|a·b|`, so every figure is unchanged: the
  relative part of `tol` is `1.26·10^-4` at `K = 512`, `1.01·10^-3` at `K = 4096` and below `0.21%`
  up to `K = 2^13`, and the `bfloat16` rejection holds on normal inputs as before. A subnormal
  operand `x` with partner `y` raises its term from `|x·y|` to at most `|x·y| + 2^-14·(|x| + |y|) +
  2^-28`.
- **The proof.** `acc_step` cites the restated fact: `partA` now sums the lifted products `liftV`
  (`mag16` is `Rules.Num.liftMag .f16` by definition, `liftMag_f16`), and `|partS| ≤ partA` follows
  from `|x·y| ≤ mag16 x·mag16 y`. Nothing else in the proof changed.

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float16` only.** This folder is the tutorial's `float16` run: `a`, `b` and `c` are
  `float16`. The script's `float8e5` run (lines 372–388: `a` and a transposed `b` cast to
  `torch.float8_e5m2`, and the benchmark's `fp8` line, lines 403–405 and 427–430) uses the same
  source with `float8e5` operands; it is not covered here (it would be a folder of its own, with an
  fp8 `tl.dot` and its own fact).
- **`ACTIVATION = ""`.** `matmul`'s default; the script never passes another. With
  `ACTIVATION = "leaky_relu"` line 307 would run (`tl.where` and a product with `0.01`), which the
  model leaves out.
- **The launch's configuration.** The block sizes `128, 256, 64` and `GROUP_SIZE_M = 8` are the
  launch's (`model` passes them); other configurations are not covered.
- **`int32` bounds** (`M_le`, `N_le`, `a_fit`, `b_fit`, `cm_fit`, and the strides below `2^31`).
  The host's largest matrices (`4096 × 4096`, `2^24` elements) are far inside. Beyond them the
  kernel itself goes wrong: Triton passes an integer of `2^31` or more as an `int64`, and with
  `int32` arguments the offsets of lines 285, 286 and 314 wrap once a matrix has `2^31` elements or
  more, so lanes would access memory outside the matrices.
- **`K ≤ 2^13`.** The tolerance grows with `K`. Up to `2^13` it is below `0.21%` of `A`
  (`tol_rel_lt`), about half a `bfloat16` rounding, so `Post` still tells a `float16` result from
  a `bfloat16`-rounded one. From about `K = 15745` it would not (`1.62%` at `K = 2^16`), and from
  about `K = 34368` it would exceed the typical `|S|` of zero-mean inputs such as the benchmark's
  `torch.randn`, so it would not even fix the sign of a typical output. The host's largest `K` is
  `4096`, covered twice over. The proof also uses the bound to keep the partial sums inside the
  `float32` range.
- **Positive strides.** The kernel itself assumes them (`tl.assume`, lines 268–273); a launch with
  a zero or negative stride is undefined behaviour.
- **Rows of `c` within `stride_cm`** (`c_rows`: `N·stride_cn ≤ stride_cm`). Each row's `N`
  elements, `stride_cn` apart, must lie within `stride_cm` (row-major, possibly padded); `matmul`
  allocates `c` row-major. Column-major and other transposed layouts of `c` are not covered, even
  though their rows do not overlap, and neither are overlapping layouts.
- **No overlap of `c` with `a` or `b`.** `matmul` always allocates a fresh `c`. An in-place call
  would make programs race.
- **Accuracy for finite inputs only.** With a NaN or an infinity in a row of `a` or a column of
  `b`, the claim says only that the element of `c` holds some `float16` value.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions): among it, point 7's target and host (an
  H200, driven from an x86-64 host running CPython 3.13 without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment). The script's `DEVICE` is
  whatever device is active.

Modelling decisions:

- `M`, `N`, `K` and the strides are `int32` (`IntV.ofNat`): Triton passes an integer argument in
  `[-2^31, 2^31)` other than `1` as an `int32`, and at `1` the constant `1`, which gives the same
  model at each use (`model`'s docstring; contiguous tensors pass `stride_ak = stride_bn =
  stride_cn = 1`). An integer that is a multiple of 16 is also marked divisible by 16 for the
  compiler, which is true of it (SEMANTICS.md, "Scalar arguments").
- `ACTIVATION` is no parameter of the model: it is a compile-time constant, `""` on every covered
  launch, and only line 306 reads it.
- Every constexpr operand of an `int32` operation (`BLOCK_SIZE_M - 1`, `GROUP_SIZE_M`, …) is built in
  `int32` (`IntV.ofNat … : I32`); `tl.cdiv(x, B)` is `(x + (B − 1)) // B`; Python's `min` with a
  tensor argument is `tl.minimum`, on `int32` the exact `tl.iminimum`.
- `tl.assume(c)` is a check: undefined behaviour when `c` is false (it never is on covered
  instances).
- `other=0.0` is the `float32` constant `+0.0` converted to `float16` (`tl.convert`); `tl.dot` has
  the default `input_precision` (`tf32`) and `max_num_imprecise_acc` (`0`), and its accumulator is
  the loop-carried value (no `tl.dot_add`); `.to(tl.float16)` is one rounding to nearest even
  (`tl.cvt`).
- The launch lists the source's ten block shapes (`[128]`, `[256]`, `[64]`, `[128, 1]`, `[1, 64]`,
  `[128, 64]`, `[64, 1]`, `[1, 256]`, `[64, 256]`, `[128, 256]`); `enable_fp_fusion` is written as
  Triton's default `true` (no operation of the model depends on it); `num_warps=8` and
  `num_stages=3` change no result and are not recorded.
