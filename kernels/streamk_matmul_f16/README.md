# streamk_matmul_f16

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/streamk_matmul.py`](source/streamk_matmul.py), written for this folder
([SETUP.md](SETUP.md)). Lean: [`lean/`](lean/), the module `StreamKMatmul` (namespace
`Kernels.StreamKMatmul`) and its parts `StreamKMatmul.Defs` (model, conditions, specification,
assumptions), `StreamKMatmul.Facts` (general lemmas from the library's definitions),
`StreamKMatmul.Sched` (the schedule of iterations, tiles and footprints), `StreamKMatmul.Basic`
(arithmetic and float lemmas, the tolerance's size), `StreamKMatmul.Parts` (the model in parts and
the invariants), `StreamKMatmul.Chunk`, `StreamKMatmul.Tail` and `StreamKMatmul.Program` (one
program).

The kernel is `streamk_matmul_kernel` (source lines 26–64). It reads two module-level names,
`triton` (its decorator) and `tl`, bound by the imports of lines 22–23, which are kernel lines too
and are mapped (dropped: they add no step). The host function `streamk_matmul(a, b, num_programs)`
(lines 70–93) asserts `float16` inputs, `a.shape[1] == b.shape[0]` and `M, N, K > 0`, takes
`num_programs` from the device's SM count unless the caller passes one, allocates
`c = torch.zeros((M, N), dtype=torch.float32)` (row-major, strides `(N, 1)`), computes
`total_iters = cdiv(M, 128) · cdiv(N, 128) · cdiv(K, 64)` and `iters_per_program =
cdiv(total_iters, num_programs)`, and launches a 1-D grid of `num_programs` programs with
`BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 64` (line 67), `num_warps=8` and `num_stages=3`. The test
(lines 96–114) calls it on `torch.randn` `float16` matrices of the shapes `512³`, `1000³`,
`3072 × 4096 × 1024` and `8192 × 8192 × 4096` with the SM count (132 on the H200),
`257 × 129 × 4097` with 7 programs and `64³` with 3 programs. Every other module-level line is a
host line: `import torch` (20), the block sizes (67, which the kernel's constexpr parameters of the
same names shadow), `streamk_matmul` and the test. No function is unused.

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `i < M` and `j < N`, element `(i, j)` of `c` (element
  `i·stride_cm + j·stride_cn` of `c`'s storage) can be read in the final memory, with some
  `float32` value `c` (`OutSpec`). When, moreover, row `i` of `a` and column `j` of `b` hold finite
  `float16` values `va k` and `vb k` (`k < K`) in the initial memory, `c` is finite and
  `|c − S| ≤ tol K A`, where `S = Σ_k va k · vb k` is the exact dot product and
  `A = Σ_k mag16(va k)·mag16(vb k)` (`dotMag`) the sum of the products' lifted magnitudes: `mag16 q`
  is `|q|`, except that a nonzero `q` below `2^-14` (a subnormal `float16`) counts as `2^-14`, as the
  tensor core aligns it (`Rules.Num.liftMag .f16`, which `DotF16` counts by). With no subnormal
  operand, `A = Σ_k |va k · vb k|`, the test's normaliser `(|A| @ |B|)[i, j]`.

  This holds **whatever order the atomic additions ran in**, and for every value each rounding may
  take within the library's facts: `Post` is a property of every outcome of every interleaving of
  the programs.

  The tolerance is explicit and computable:

  `tol K A = tolDot K A + DotLoop.bound 2^-23 2^-124 T (A + tolDot K A)`, with
  `tolDot K A = DotLoop.bound ρ α T A` and `DotLoop.bound ρ α T B = ((1 + ρ)^T − 1)·B + T·(1 + ρ)^T·α`,

  where `T = cdiv(K, 64)` is the number of `tl.dot` steps of a whole tile, `ρ = relDot16 64 =
  66·2^-22` and `α = absDot16 64 = 66·2^-126` are the library's constants for one fp16 `tl.dot` of
  `K = 64` (`Rules.Num.DotF16`), and `2^-23`, `2^-124` those of a `float32` `tl.atomic_add`
  (`Rules.Num.AtomicAddF32`). The first term bounds the `tl.dot` accumulation of all the tile's
  steps, wherever the programs split them; the second the at most `T` atomic additions of the
  programs' partial sums into the zero `c` (one per program that touches the tile; at most `T` do),
  whose magnitudes add up to at most `A + tolDot K A`. An element of a tile one program does whole
  is within `tolDot K A`, less.

  Its size, as a multiple of `A` (the absolute part is below `10^-33` at every covered `K`):

  | `K` (test shapes) | `T` | `tol K A / A` |
  |---|---|---|
  | `64` (`64³`) | 1 | `1.59·10^-5` |
  | `512` (`512³`) | 8 | `1.27·10^-4` |
  | `1000`, `1024` (`1000³`, `3072 × 4096 × 1024`) | 16 | `2.54·10^-4` |
  | `4096` (`8192 × 8192 × 4096`) | 64 | `1.015·10^-3` |
  | `4097` (`257 × 129 × 4097`) | 65 | `1.031·10^-3` |
  | `8192 = 2^13` (the largest covered) | 128 | `2.03·10^-3` |

  `tol_le` proves `tol K A ≤ 0.0021·A + 10^-33` for every covered `K`. On the test's data, `A` is
  the test's normaliser `(|A| @ |B|)[i, j]` up to a negligible excess: `torch.randn` in `float16`
  gives a few subnormal operands (the reviewers' sample, `torch.manual_seed(0)` on the CPU, found
  0 to 102 per matrix), each counted at `2^-14` in `A` in place of its magnitude, which raised `A`
  above the normaliser by at most `1.9·10^-7` of it on the sampled shapes. So the claim bounds the
  test's error measure by `1.6·10^-5` to `1.03·10^-3` per shape, to the digits given; the
  coordinator's H200 run measured `6.2·10^-8` to
  `9.3·10^-7` (SETUP.md), and the test asserts `< 10^-5`. The claim's bound is a worst case over
  every rounding the facts allow (each `tl.dot` step up to `66·2^-22` of its magnitude, each atomic
  addition up to `2^-23`), so it is looser than the observed errors and than the test's threshold
  at the larger `K`; it is still about half the largest relative error of one rounding to
  `bfloat16` (`2^-8/(1 + 2^-8)`, about `0.39%`) at `K = 2^13`, so a kernel that rounds its result to
  `bfloat16`, or drops or duplicates a partial sum (an error of a whole part's `|S|`), breaks `Post`
  on some inputs. What `Post` does not reject: from `K = 1921` (`T = 31`) on, the tolerance exceeds
  `4.88·10^-4·A` (`2^-11/(1 + 2^-11)`, the largest relative error of one rounding to `float16`;
  `5.07·10^-4·A` at `K = 2048`, `1.015·10^-3·A` at `K = 4096`, a test shape), so at those `K` it
  cannot tell a result rounded to `float16` from a `float32` one. This follows from the library's
  `DotF16` constant (`relDot16 64 = 66·2^-22` per `tl.dot` step, about `T·ρ·A` over the worst
  inputs), so the facts give no tighter bound in terms of `A`. The bound is relative to `A`
  (`Σ|a·b|` without subnormal operands), not to `|S|`: for zero-mean inputs such as the test's
  `torch.randn`, `Σ|a·b|` is about `(2/π)·K` while `|S|` is typically `√K`.

  With a NaN or an infinity in row `i` of `a` or column `j` of `b`, the claim says only that element
  `(i, j)` holds some `float32` value.
- **Nothing else changes.** Every byte outside the elements of the matrix `c` is the same in `fin`
  as in `init`: `a`, `b`, and the elements of `c`'s storage that are not in the matrix.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `M`, `N`, `K`, the six strides (in elements),
`num_programs` (the grid), the arguments `iters_per_program` and `total_iters`, two `float16`
buffers `a`, `b` and a `float32` buffer `c` (`Buf`: a base address and a length in elements);
element `(i, k)` of the matrix `a` is element `i·stride_am + k·stride_ak` of the buffer `a`, and
likewise for `b` and `c` (`aIdx`, `bIdx`, `cIdx`).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `M_pos`, `N_pos`, `K_pos` | `M, N, K > 0` | `streamk_matmul` asserts it (line 75) |
| `M_le`, `N_le` | `M + 128 ≤ 2^31`, `N + 128 ≤ 2^31` | `M`, `N` are `int32` arguments; `tl.cdiv(N, 128)` computes `N + 127` in `int32` (line 37), and the tile rows and columns `pid_m·128 + i`, `pid_n·128 + j` (lines 48–49) stay below `2^31` |
| `K_le` | `K ≤ 2^13` | keeps the tolerance meaningful: below `0.21%` of `A` (`tol_le`), about half a `bfloat16` rounding (Specification, Narrowings); the test's largest `K` is `4097`; it also gives `K + 63 < 2^31` (line 38) |
| `am`, `ak`, `bk`, `bn`, `cm` | each stride is below `2^31` | each stride is an `int32` argument (zero allowed: the kernel assumes nothing of them) |
| `cn` | `0 < stride_cn < 2^31` | with `c_rows`, distinct elements of the matrix `c` are distinct elements of its storage |
| `am_fit`, `ak_fit`, `bk_fit`, `bn_fit`, `cm_fit` | `(M − 1)·stride_am`, `(K − 1)·stride_ak`, `(K − 1)·stride_bk`, `(N − 1)·stride_bn`, `(M − 1)·stride_cm` below `2^31` | the `int32` products of the lanes inside the matrices (lines 53, 55, 58) do not wrap |
| `c_rows` | `N·stride_cn ≤ stride_cm` | each row of `c` lies within `stride_cm` (row-major, possibly padded), so the rows do not overlap (`c_inj`, `lane_inj`) and no two lanes, tiles or programs address one element unless they are the same lane of the same tile; with `cm` the products `offs_n·stride_cn` of the columns below `N` (line 58) stay below `2^31`. It also excludes column-major and transposed layouts of `c` (Narrowings) |
| `total` | `total_iters = cdiv(M, 128)·cdiv(N, 128)·cdiv(K, 64)` | the host's formula (lines 79–80) |
| `cover` | `total_iters ≤ num_programs·iters_per_program` | the programs' ranges cover every iteration, as `iters_per_program = cdiv(total_iters, num_programs)` (line 81) gives |
| `progs_lt` | `num_programs·iters_per_program < 2^31` | the `int32` values `pid·iters_per_program` and `it + iters_per_program` (lines 39–40) do not wrap, and the grid has fewer than `2^31` programs |
| `a_len`, `b_len`, `c_len` | every element of each matrix is inside its buffer | the kernel accesses them |
| `c_a`, `c_b` | the matrix `c` shares no byte with the matrix `a`, nor with the matrix `b` | a program that reads bytes another program writes races; `streamk_matmul` always allocates a fresh `c` |

`M`, `N`, `K`, the strides, `iters_per_program` and `total_iters` are the kernel's runtime
arguments (or, at the value `1`, compile-time constants, which give the same model: `model`'s
docstring). Their values are part of the instance, and the conditions above constrain them;
`Compiler.Faithful`'s point 11 assumes that the launch runs the code Triton compiles for these
values.

**The host's use is covered.** Every test shape meets `Inst.ok`: `a` and `b` come from
`torch.randn` (strides `(K, 1)` and `(N, 1)`) and `c` from `torch.zeros` (strides `(N, 1)`), so
the fits read `(M − 1)·K`, `K − 1`, `(K − 1)·N`, `N − 1`, `(M − 1)·N` below `2^31` (at the largest,
`8192 × 8192 × 4096`: `8191·4096`, `4095·8192`, `8191·8192`, all below `2^26`) and `c_rows` reads
`N ≤ N`; `K ≤ 4097 ≤ 2^13`; `total` and `cover` are the host's formulas; `progs_lt` holds as
`num_programs·cdiv(total_iters, num_programs) < total_iters + num_programs` (at most
`262144 + 132`); the fresh `c` overlaps neither input. The program counts are variables: the SM
count, 7, and 3 for `64³`, where `total_iters = 1`, `iters_per_program = 1`, and programs 1 and 2
have no iteration (more programs than iterations; `satisfiable`'s instance has 132 programs for 128
iterations). Both paths are covered: a tile whose iterations lie in one program's range is stored,
and a tile split across programs (every tile at `1000³` with 132 programs) is accumulated with
atomics. The conditions are more general than that use: any strides within the bounds (a transposed
or broadcast `a` or `b`, a padded `c`), any `num_programs` and `iters_per_program` that cover the
iterations, and `a` and `b` may overlap each other in any way (both are only read).

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to the element size, the buffer inside `[0, 2^64)`, every byte present),
and every element of the matrix `c` holds a finite `float32` zero (`streamk_matmul` allocates `c`
with `torch.zeros`, line 78). Every `float16` value is allowed in the inputs, NaN and infinities
included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch the script makes is a direct launch of the `@triton.jit` function,
the JIT launch `streamk_matmul_kernel[grid](…)` of `streamk_matmul` (lines 83–92), called by the
test at line 109. Point by point:

- **4 (options).** The launch gives the constexprs `BLOCK_M`, `BLOCK_N`, `BLOCK_K` and two options,
  `num_warps=8` and `num_stages=3`, two of the four that do not change the meaning; no
  `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or any other option. Nothing calls
  `streamk_matmul_kernel.warmup(…)` or `triton.compile`; the kernel has no `@triton.autotune` and no
  `@triton.heuristics`, and the decorator is a bare `@triton.jit`, so its `debug` is not set. So
  `launch` keeps every default: `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`,
  `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor (no
  `tl.make_tensor_descriptor`), so no allocator is involved.
- **6 (precompiled kernel).** No `CompiledKernel` is launched: every launch is the JIT launch above.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are three
  tensors, eleven integers and the constexprs.
- **9 (CUDA graphs).** No call is made during a CUDA stream capture, and nothing is replayed: the
  test calls `streamk_matmul` directly (line 109); the script opens no `torch.cuda.graph`, uses no
  `triton.testing.do_bench_cudagraph` and makes no capture of its own.

So a call of `streamk_matmul` makes this one direct launch, which the claim covers, from the memory
at that launch.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`, not under `@triton.autotune`. Its parameters
are three pointers (tensor arguments), eleven unannotated integers (`int` arguments) and three
`tl.constexpr`s with `int` arguments, none with a default. Its statements are assignments
`name = e`, one `name += e` (line 64), expression statements (the calls `tl.store` and
`tl.atomic_add`), one `while c:` with a runtime condition, one `for kk in range(lo, hi)` with two
positional arguments, and one runtime `if` (its condition an `int1` scalar) with an `else`; line 43
is a comment. Its expressions are `tl.program_id(0)`, `tl.cdiv`, `tl.minimum` (on `int32`
scalars), `tl.arange`, `tl.zeros(…, dtype=tl.float32)`, `tl.load(…, mask=…, other=0.0)`,
`tl.dot(a, b, acc)`, `tl.store(…, mask=…)`, `tl.atomic_add(…, mask=…)`, Python's `range`, the
operators `+ - * // % &`, the comparisons `< ==` (one per expression), and the subscripts
`[:, None]` and `[None, :]`. Across the runtime `if`, neither branch assigns a name; the `while`
loop carries `it`, the one name defined before it that its body assigns, and the `for` loop carries
`acc`; no name first defined in a loop body is used after the loop. The only module-level names it
reads are `triton` (its decorator) and `tl`, Triton's modules, each bound only by its own `import`
(lines 22 and 23). No other statement of the module binds `triton`, `tl` or
`streamk_matmul_kernel` (the others bind `torch`, `BLOCK_M`, `BLOCK_N`, `BLOCK_K`,
`streamk_matmul`, and, under `if __name__ == "__main__":`, `DEVICE`, `M`, `N`, `K`,
`num_programs`, `a`, `b`, `c`, `ref` and `err`; no decorator, default or annotation binds a name).

## Assumptions

Five assumptions, all of kind `[rule]`, each cited as the library states it:

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid (`Launch.Valid`), then from every initial memory from
  which the model's launch is well behaved (no undefined behaviour, no race between programs, no
  conflict within a program), every outcome of the compiled kernel launched as `launch d` is one
  of the model's: a normal end with a memory some interleaving of the model leaves, never a fault,
  and a hang only if some interleaving runs forever. The rule covers a launch only within its
  scope, which its text states: part (A), which the launches meet (Conditions), and part (B), the
  environment, which the claim relies on ([rule page](../../docs/rules/compiler-faithful.md)).
- `dot` — `Rules.Num.DotF16 P`, the library's fact for `tl.dot` of `float16` operands into
  `float32` (line 57; [num.md](../../docs/rules/num.md), `Rules.Num.DotLiftAcc`): for every
  `input_precision`, with `maxImpreciseAcc = 0`, each output element whose row of `a`, column of
  `b` and accumulator are finite is finite and within `relDot16 K·L + absDot16 K` of
  `acc + Σ a·b`, with `L = |acc| + Σ liftMag a·liftMag b` (a subnormal operand counted at `2^-14`),
  when that leaves room below the `float32` range.
- `dot_total` — `Rules.Num.DotTotal P cfg` for the model's configuration (`float16` operands,
  `float32` accumulator, `tf32`, `maxImpreciseAcc = 0`): the `tl.dot` of line 57 always has a
  result.
- `atomic` — `Rules.Num.AtomicAddF32 P`, the library's fact for a `float32` `tl.atomic_add`
  (line 63; `Rules.Num.AtomicAddAcc` with `2^-23` and `2^-124`): when the old value and the operand
  are finite and their exact sum is within the `float32` range, the new value is finite and within
  `2^-23·|o + v| + 2^-124` of `o + v`.
- `atomic_total` — `Rules.Num.AtomicAddTotal P Format.binary32`: a `float32` `tl.atomic_add`
  always has a result.

There is no `[kernel]` assumption. The `other=0.0` conversion of lines 54 and 56 is an IEEE
rounding the library defines (`tl.convert`, `ConvRel`), and needs no platform fact; the `float32`
store of line 61 converts nothing.

## Proof

`main` applies `TVA.meets_of_faithful`, with the theorem `TVA.shared` (programs that share memory
only through `acq_rel` atomics on a set of cells) for the launch:

- `hdebug`: `debug = false` (Triton's default) by definition of `launch`.
- `hvalid` (`launch_valid`, by `Launch.valid_grid1`): `num_programs < 2^31` (`np_lt`, from
  `progs_lt` and `iters_per_program ≥ 1`), and each of the nine block shapes is a list of powers of
  two with at most `2^20` elements.
- The schedule (`StreamKMatmul.Sched`): program `p`'s range of global iterations is
  `[pstart, pend) = [p·ipp, min(p·ipp + ipp, total))`; its part of tile `t` is
  `[tlo, thi) = [max(pstart, t·T), min(pend, t·T + T))`, nonempty when it `touches` the tile, with
  tile-local steps `[kfirst, klast)`. Program `p` `owns` tile `t` when the whole tile is in its
  range; a tile is `whole` when some program of the grid owns it, and otherwise split. The parts of
  a tile partition its `T` steps (`sum_parts`, `sum_part_len`; from `cover`), and at most `T`
  programs touch it (`card_touch_le`).
- Footprints for `shared`: every program reads within the matrices `a` and `b` (`R`); program `p`
  writes privately the elements of the tiles it owns (`W`); the shared cells are the elements of the
  split tiles (`cells`). `hdisj`: `W p` is disjoint from `R` (`c_a`, `c_b`) and, for `p ≠ q`, from
  `W q` (a tile has one owner, `owns_unique`, and two enabled lanes that address one element are the
  same lane of the same tile, `lane_inj`, from `c_rows`); `hRW` likewise; `hA`: the cells are
  disjoint from `R` and from every `W p` (an owned tile is whole); `hcells`: two cells whose bytes
  meet are the same element (`cells_eq`); `hpres`: `c` is valid in `init`.
- `hsafe` (`safe_program`, in `StreamKMatmul.Program`): one program, alone, with its atomic steps
  logged (`SafeA`).
  - Lines 36–40 (`safeA_ro` with `safe_idiv1`): `pid = p` (`launch_pid`), `num_pid_n =
    cdiv(N, 128)`, `iters_per_tile = T`, `it = pstart`, `end = pend`, no `int32` operation wrapping.
  - Line 41 (`safeA_while`, a `SafeA` form of the library's `safe_while`, proved in
    `StreamKMatmul.Facts` from `SafeA`'s rules): the invariant `LoopInv` — `it = n` with
    `pstart ≤ n`, `n` at a tile's start or at `pstart`, `a`, `b` unchanged and the buffers valid, the
    owned tiles whose part is done stored (`MemInv`: each element of such a tile holds a value within
    `tolDot K A` of its exact sum, `TileOut`), and the log of atomic steps holding, for each element
    of a split tile, one operand per part done (`LogInv`: the steps paired with their operands,
    `Paired`, each step on the element an `AccStep`, each other step leaving it alone, the operand
    within `DotLoop.bound ρ α (klast − kfirst)` of the part's exact sum `partS`); the variant
    `pend − it`; the iteration from `n` writes privately at most the tile it stores (`Wv`,
    footprints of different variant values disjoint, `Wv_disj`, and inside `W p`, `Wv_sub`). It
    holds at the start (no part done, empty log). At the exit `it ≥ end`, every touched part is done,
    which is `Q`.
  - One iteration (`safe_iter`): lines 42–47 compute `tile = n / T`, `k_first = n mod T = kfirst`,
    `k_last = min(T, n mod T + (end − n)) = klast`, `pid_m = tile / cdiv(N, 128)`,
    `pid_n = tile mod cdiv(N, 128)` (`iter_kfirst`, `iter_klast`); lines 48–49 the lanes
    `offs_m = pid_m·128 + i`, `offs_n = pid_n·128 + j` (`lane_add`, no wrap: the tile row is below
    `cdiv(M, 128)`); line 51 (`safe_forRange`, `tl.RangeOk`) runs `klast − kfirst` trips with the
    invariant `AccInv` (memory unchanged; every enabled element whose row of `a` and column of `b`
    are finite is finite and within `DotLoop.bound ρ α c (partA)` of the exact sum `partS` of the
    `c` chunks done), from `acc0` (`+0.0`, exact).
  - One trip (`safe_loadDot`): `offs_k = 64·kk + k`; the loads (`safe_convert`, `safe_load_buf`) read
    each enabled lane's element of `a` (resp. `b`), inside the matrix and unchanged since `init`, and
    give each disabled lane the converted `other=0.0`, a finite zero (`zero_conv`); `tl.dot`
    (`safe_dot` with `dot_total`), then `acc_step`: by `dot` at `K = 64` and `DotLoop.accum_step`,
    `AccInv` holds with one more chunk (the values stay far inside the `float32` range as
    `K ≤ 2^13`).
  - Lines 58–64 (`safe_storeAdd`): the lanes of `c_ptrs` address the tile's elements, `c_mask` is
    the tile's lanes inside the matrix. Line 60's condition holds exactly when the program owns the
    tile (`iter_owns`). Then line 61 (`safe_store_buf`) stores the tile, inside `Wv`; with
    `kfirst = 0` and `klast = T` the part is the whole sum (`chunks_end`), so `TileOut` holds.
    Otherwise the tile is split (`iter_split`) and line 63 (`SafeA.env`, `SafeA.atomic`) adds the
    accumulator into the tile's elements, which are cells: in any order of the lanes and for any old
    values, each enabled lane logs one step; for an element of the tile, its lane's step is paired
    with the operand `acc` (`accStep_atomicAdd`, by `atomic`, with the finite operand of `AccInv`)
    and every other new step with none (`accStep_other`: different elements have disjoint bytes).
    Line 64: `it` becomes `thi`, the end of the part (`iter_it`), so the variant decreases and
    `LoopInv` holds with this part done (`iter_done`).
  - `Q` does not depend on the order of the log (`Q_perm`, by `Paired.perm`).
- `Post` (`post`), from `shared`'s outcome: an interleaved log `G`, each program's memory agreeing
  with `fin` on its footprint, a replay of `G` from `init` agreeing with `fin` on the cells, and
  `fin = init` outside every footprint and cell.
  - Element `(i, j)` lies in tile `t = (i / 128)·cdiv(N, 128) + j / 128`, lane
    `(i mod 128, j mod 128)` (`tileOf_lane`, `tileOf_en`).
  - A whole tile: its owner's `MemInv` gives `TileOut`, and `fin` agrees with the owner's memory on
    `W p` (`Mem.read_congr`); `tolDot K A ≤ tol K A`.
  - A split tile: the element is a cell, read from the replay, readable since `init` holds the zero
    of `Pre` (`replay_read_some`). Each program's `LogInv` gives its operands (one per touching
    program); `replay_err_shared` (with `Paired.merge`) bounds the replay against
    `0 + Σ operands` by `DotLoop.bound 2^-23 2^-124 n (Σ|operand|)`, given room below the `float32`
    range (`split_room`: `A ≤ 2^46` for finite `float16` inputs and `K ≤ 2^13`). `split_sums`: the
    operands add up to within `tolDot K A` of `S` (the parts' exact sums add up to `S`, their bounds
    to at most `tolDot K A`, `sum_bound_le`), their magnitudes to at most `A + tolDot K A`, and there
    are at most `T`; `split_err` gives `tol K A`.
  - A byte outside the matrix `c` is in no write footprint and no cell, so it keeps its initial
    value.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref` gives a total behaviour that
satisfies `faithful`; `Platform.sm90ref.dotF16`, `Platform.ref.dotTotal`,
`Platform.sm90ref.atomicAddF32` and `Platform.ref.atomicAddTotal` give the four `Num` facts.

`satisfiable`: the source's first test instance (line 100), `M = N = K = 512` with row-major `a`,
`b` and `c` (strides `512` and `1`), each of `512·512` elements, at the aligned addresses `2^21`,
`2·2^21` and `3·2^21`, with 132 programs (the H200's SM count), `total_iters = 128` and
`iters_per_program = 1` (`testInst`), in a memory where every byte is present and zero
(`testMem`, so `c` reads `+0.0`), meets `Inst.ok` and `Pre`.

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

Three general lemmas are proved in the kernel's files from the library's definitions: `safeA_ro`
(a segment proved with `Safe` and an empty write footprint, then the rest, for `SafeA`),
`safeA_while` (the `while` rule for `SafeA`) and `replay_read_some` (a replay keeps a readable
element readable). They could move to the library; the proof does not need any library change.

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **The launch's configuration.** The block sizes `128, 128, 64` are the launch's (`model` passes
  them); other block sizes are not covered.
- **`int32` bounds** (`M_le`, `N_le`, the fits, `progs_lt`, and the strides below `2^31`). The
  test's largest matrices (`8192 × 8192`) are far inside. Beyond them the kernel itself goes wrong:
  Triton passes an integer of `2^31` or more as an `int64`, and with `int32` arguments the offsets
  of lines 53, 55 and 58 wrap once a matrix has about `2^31` elements, so lanes would access memory
  outside the matrices.
- **`K ≤ 2^13`.** The tolerance grows with `K`. Up to `2^13` it is below `0.21%` of `A` (`tol_le`),
  about half a `bfloat16` rounding, so `Post` still tells a `float32` accumulation from a
  `bfloat16`-rounded one; at `K = 2^16` it would be about `1.6%`. The test's largest `K` is `4097`,
  covered twice over. The proof also uses the bound to keep the partial sums inside the `float32`
  range.
- **`M, N, K > 0`.** `streamk_matmul` asserts it (line 75).
- **`total_iters` and the programs' ranges.** `total_iters` is the host's formula and the ranges
  `num_programs · iters_per_program` cover it (`total`, `cover`), as `streamk_matmul` computes
  them; a launch with fewer iterations than tiles need would leave elements of `c` untouched.
- **A zero `c`.** `Pre` requires every element of the matrix `c` to hold a finite zero, as
  `streamk_matmul`'s `torch.zeros` gives; the atomic additions accumulate into it.
- **Rows of `c` within `stride_cm`** (`c_rows`: `N·stride_cn ≤ stride_cm`, with `stride_cn > 0`).
  `streamk_matmul` allocates `c` row-major. Column-major and other transposed layouts of `c` are not
  covered, even though their rows do not overlap, and neither are overlapping layouts.
- **No overlap of `c` with `a` or `b`.** `streamk_matmul` always allocates a fresh `c`. An in-place
  call would make programs race.
- **Accuracy for finite inputs only.** With a NaN or an infinity in a row of `a` or a column of `b`,
  the claim says only that the element of `c` holds some `float32` value.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions): among it, point 7's target and host (an
  H200, driven from an x86-64 host running CPython 3.13 without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment). The script's `DEVICE` is
  whatever device is active.

Modelling decisions:

- `M`, `N`, `K`, the strides, `iters_per_program` and `total_iters` are `int32` (`IntV.ofNat`):
  Triton passes an integer argument in `[-2^31, 2^31)` other than `1` as an `int32`, and at `1` the
  constant `1`, which gives the same model at each use (`model`'s docstring; contiguous tensors pass
  `stride_ak = stride_bn = stride_cn = 1`, and `iters_per_program = 1` whenever
  `num_programs ≥ total_iters`, as at `512³` and `64³`). An integer that is a multiple of 16 is also
  marked divisible by 16 for the compiler, which is true of it (SEMANTICS.md, "Scalar arguments").
- The source's name `end` is a Lean keyword; the model writes it `«end»`, the same name.
- Every constexpr operand of an `int32` operation (`BLOCK_N - 1`, `BLOCK_M`, …) is built in `int32`
  (`IntV.ofNat … : I32`); `tl.cdiv(x, B)` is `(x + (B − 1)) // B`; `tl.minimum` on `int32` scalars
  is the exact `tl.iminimum`.
- `while it < end:` is `tl.whileLoop` with the condition `it < end` and the carried `it`;
  `range(k_first, k_last)` is `tl.forRange` with step `1` and the carried `acc`; the runtime `if`
  is a Lean `if` on the condition's value, each branch's operations in its branch.
- `other=0.0` is the `float32` constant `+0.0` converted to `float16` (`tl.convert`); `tl.dot` has
  the default `input_precision` (`tf32`) and `max_num_imprecise_acc` (`0`), and its accumulator is
  the loop-carried value; `tl.atomic_add` has the defaults `sem="acq_rel"`, `scope="gpu"`, and its
  result is unused.
- The launch lists the source's nine block shapes (`[128]`, `[64]`, `[128, 1]`, `[1, 64]`,
  `[128, 64]`, `[64, 1]`, `[1, 128]`, `[64, 128]`, `[128, 128]`); `enable_fp_fusion` is written as
  Triton's default `true` (no operation of the model depends on it); `num_warps=8` and
  `num_stages=3` change no result and are not recorded.
