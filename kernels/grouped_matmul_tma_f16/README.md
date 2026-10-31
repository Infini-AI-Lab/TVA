# grouped_matmul_tma_f16

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/08-grouped-gemm.py`](source/08-grouped-gemm.py). Lean: [`lean/`](lean/), the
module `GroupedMatmulTma` (namespace `Kernels.GroupedMatmulTma`) and its parts
`GroupedMatmulTma.Defs` (model, conditions, specification, assumptions), `GroupedMatmulTma.Sched`
(the schedule, the footprints, the program invariant `ProgInv`), `GroupedMatmulTma.Basic`
(arithmetic, descriptor lanes, float lemmas), `GroupedMatmulTma.Tile` (one tile) and
`GroupedMatmulTma.Program` (one program).

The kernel is `grouped_matmul_tma_kernel` (source lines 186–278), as the folder's source has it
(SETUP.md: the tutorial's `@triton.autotune` removed, the launches passing the config
`BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, num_stages=3, num_warps=4`). It reads two
module-level names, `triton` (its decorator) and `tl`, bound by the imports of lines 32–33, which
are kernel lines too and are mapped (dropped: they add no step). It is a persistent kernel: a 1-D
grid of `NUM_SM` programs; the tiles (`128 × 128`) of all groups are numbered one after another,
tile `t` of group `g` (row of tiles `t / cdiv(N g, 128)`, column `t % cdiv(N g, 128)`) having the
global index `start g + t`, and program `p` computes the tiles whose global index is `p` modulo
`NUM_SM` (lines 210, 220, 251, 275, 278). Each tile is a `float16` GEMM through TMA descriptors
(`tl.make_tensor_descriptor`, `desc.load`, `desc.store`) with a `float32` accumulator.

Two host functions launch it, both with `FP8=False` on `float16` matrices: `group_gemm_tma_fn`
(lines 281–326), called by the test (lines 354–357, four groups, `M = N = K = 1024, 512, 256,
128`, `group_B_T` the transposes, made contiguous), and `triton_tma_perf_fn` (lines 374–378),
called by the two benchmarks through `triton.testing.do_bench` (lines 445–448: four groups of
`N × N × N`, `N = 128, 256, 512, 1024`; lines 516–519: four groups of `M × 8192 × 8192`,
`M = 128, 256, 512, 1024`). Each builds `int64` tensors of the matrices' addresses and `int32`
tensors of the sizes and row strides, and launches `grouped_matmul_tma_kernel[grid]` with
`grid = (NUM_SM,)`, `NUM_SM=num_sms()` (the device's multiprocessor count).

Every other function and module-level line is a host line: the imports of `typing` and `torch`
(29–30), `DEVICE` (35), `is_cuda`, `supports_tma`, `num_sms` (38–49), `group_gemm_fn` (131–173,
which launches the other kernel), `tma_configs` (176–183, unused), `group_gemm_tma_fn`, the test
(329–357), `triton_perf_fn`, `triton_tma_perf_fn`, `torch_perf_fn` (361–383), the two benchmarks
(386–520) and their runs (523–524). The file's other `@triton.jit` function,
`grouped_matmul_kernel`, is out of scope ([exclusions.md](../../docs/exclusions.md)): it is in
`source.unused` (the claimed launch never runs it; its launches, lines 163–171 and 363–371, are
host code for this claim), and its header (lines 52–72) is a host line: it binds
`grouped_matmul_kernel`, which the launch does not read.

## Specification

`Post d init fin` says two things.

- **The outputs.** For every group `g < group_size`, `i < M g` and `j < N g`, element `(i, j)` of
  `C g` (element `i·ldc g + j` of `C g`'s storage) can be read in the final memory, with some value
  `c` (`OutSpec`). When, moreover, row `i` of `A g` (elements `i·lda g + k`) and row `j` of `B g`
  (elements `j·ldb g + k`, `B g` being the `N g × K g` matrix the host passes, `B.T.contiguous()`)
  hold finite `float16` values `va k` and `vb k` (`k < K g`) in the initial memory:
  - `c` is the `float16` rounding (to nearest even, `FloatV.cvt`, up to NaN payload) of a finite
    `float32` value `v` with `|v − S| ≤ tol (K g) A`, where `S = Σ_k va k · vb k` is the exact dot
    product (`dotSum`) and `A = Σ_k |va k · vb k|` (`dotAbs`);
  - when `|S| + tol (K g) A ≤ 65504` (the largest `float16`), `c` is finite and
    `|c − S| ≤ tol (K g) A + (|S| + tol (K g) A)·2^-11 + 2^-25` (one more `float16` rounding:
    `Format.binary16.u = 2^-11`, `Format.binary16.eta = 2^-25`).

  The tolerance is `tol K A = DotLoop.bound ρ α T A = ((1 + ρ)^T − 1)·A + T·(1 + ρ)^T·α`, with
  `T = cdiv(K, 64)` steps `acc += tl.dot(a, b.T)` (`steps`), and per step the library's constants
  for one fp16 `tl.dot` of `K = 64` (`Rules.Num.DotF16`: `relDot16 64 = 66·2^-22`,
  `absDot16 64 = 66·2^-126`) plus one `float32` addition (the combine pass may leave the addition
  separate, `tl.dot_add`): `ρ = DotLoop.stepRel relDot16 64 = relDot16 64 + 3·2^-24·(1 + relDot16 64)`
  (about `1.59·10^-5`) and `α = DotLoop.stepAbs absDot16 64 = absDot16 64·(1 + 3·2^-24) + 4·2^-150`
  (below `10^-36`). It is explicit and computable: `(1 + ρ)^T − 1` is about `3.2·10^-5` at
  `K = 128`, `2.55·10^-4` at `K = 1024` (the test's and the square benchmark's largest), and
  `2.04·10^-3` at `K = 8192 = 2^13` (the batch benchmark's `K`, the largest covered; below `0.21%`
  at every covered `K`, `tol_rel_lt`), and the `α` term is below `10^-34`. `Post`'s whole bound on
  the stored `c` adds the last `float16` rounding: with products of one sign (`Σ|a·b| = |S|`) it
  allows a relative error of `(1 + ρ)^T − 1 + (1 + ((1 + ρ)^T − 1))·2^-11` (plus the absolute terms
  `T·(1 + ρ)^T·α` and `2^-25`), about `5.20·10^-4` at `K = 128`, `7.43·10^-4` at `K = 1024` and
  `2.528·10^-3` (`0.253%`) at `K = 2^13`. So at the largest covered size `Post` allows `0.253%`, below
  the largest relative error of a rounding to `bfloat16` (`2^-8/(1 + 2^-8)`, about `0.389%`), so a
  kernel that rounds its accumulator to `bfloat16` breaks `Post` on some inputs (products of one
  sign, a sum where the `bfloat16` rounding error is near its largest), and so does one that drops
  or misplaces a product, or computes another group's or another tile's element. The host's inputs
  come from `torch.rand` (values in `[0, 1)`), so every product is non-negative and `Σ|a·b| = S`:
  the bound is then relative to the result itself. (`K_le` stops at `2^13`: `Post`'s whole relative
  bound reaches `2^-8/(1 + 2^-8)` at `T = 214` steps, so from `K = 13633` it would no longer reject a
  `bfloat16` rounding; the accumulation part `(1 + ρ)^T − 1` alone reaches it at `T = 245`, from
  `K = 15617`.)

  With a NaN or an infinity in row `i` of `A g` or row `j` of `B g`, the claim says only that
  element `(i, j)` of `C g` holds some `float16` value.
- **Nothing else changes.** Every byte outside the matrices `C g` (`cEl d g`, `g < group_size`) is
  the same in `fin` as in `init`: the tables, the matrices `A g` and `B g`, and the elements of each
  `C g`'s storage that are not in the matrix (padding of a row, say).

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `group_size`, `NUM_SM`, and for each group `g` the sizes
`M g`, `N g`, `K g`, the row strides `lda g`, `ldb g`, `ldc g` (in elements) and three `float16`
buffers `A g`, `B g`, `C g` (`Buf .f16`: a base address and a length in elements); and the five
tables: `group_a_ptrs`, `group_b_ptrs`, `group_c_ptrs` (`Buf .i64`) and `group_gemm_sizes`,
`g_lds` (`Buf .i32`). Element `(i, k)` of the matrix `A g` is element `i·lda g + k` of the buffer
`A g`, element `(j, k)` of `B g` is element `j·ldb g + k` of `B g`, element `(i, j)` of `C g` is
element `i·ldc g + j` of `C g` (`aIdx`, `bIdx`, `cIdx`): row-major with unit column stride, as the
descriptors' strides `[ld, 1]` say. `tiles g = cdiv(M g, 128)·cdiv(N g, 128)` and
`start g = Σ_{h < g} tiles h` (the value of `last_problem_end` at trip `g`).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `G_le` | `3·group_size + 3 ≤ 2^31` | `group_size` is an `int32` argument; the `int32` offsets `g * 3 + 2` (lines 214–224) do not wrap |
| `SM_pos` | `0 < NUM_SM` | at least one program |
| `tiles_lt` | `start group_size + NUM_SM < 2^31` | the `int32` values `num_m_tiles * num_n_tiles` (line 219), `last_problem_end + num_tiles` (lines 220, 251, 278) and `tile_idx + NUM_SM` (line 275) do not wrap; the grid has fewer than `2^31` programs |
| `M_le`, `N_le` | `M g + 128 ≤ 2^31`, `N g + 128 ≤ 2^31` | `M g`, `N g` are `int32` table entries; `tl.cdiv` computes `gm + 127`, `gn + 127` in `int32` (lines 217–218); the descriptors' shape entries are `int32` |
| `K_le` | `K g ≤ 2^13` | keeps the tolerance meaningful (Specification, Narrowings); the host's largest `K` is `8192 = 2^13` |
| `lda_ok`, `ldb_ok` | `lda g`, `ldb g` below `2^31` and multiples of `8` | each is an `int32` table entry; a descriptor's row stride is a multiple of `16` bytes (`TensorDesc.Valid`) |
| `ldc_ok` | as `lda_ok`, for `ldc g`, and `N g ≤ ldc g` | each row of `C g` lies within `ldc g`, so the rows do not overlap and no two lanes or programs store to one element (`c_inj`) |
| `base_al` | each `A g`, `B g`, `C g` starts at a 16-byte-aligned address | a descriptor's base is 16-byte aligned (`TensorDesc.Valid`) |
| `A_len`, `B_len`, `C_len` | every element of each matrix is inside its buffer | the kernel accesses them |
| `tab_len` | the pointer tables have at least `group_size` elements, the size tables at least `3·group_size` | the kernel reads them |
| `c_c` | the matrices `C g` of two groups share no byte | two programs storing to one byte race |
| `c_ab` | no matrix `C g` shares a byte with a matrix `A h` or `B h` | a program that reads bytes another writes races |
| `c_tab` | no matrix `C g` shares a byte with a table | the same |
| `ab_tab` | no matrix `A h` or `B h` shares a byte with a table | the kernel reads the tables through pointers and the matrices through descriptors, and pointer and descriptor accesses to the same bytes are out of scope ([exclusions.md](../../docs/exclusions.md)) |

`group_size` is the kernel's runtime argument (an `int32`; at the value `1`, a compile-time
constant, which gives the same model: `model`'s docstring); the sizes, strides and addresses are
the tables' values; `NUM_SM` is a constexpr. All are part of the instance, the conditions above
constrain them, and `Compiler.Faithful`'s point 11 assumes that the launch runs the code Triton
compiles for these values.

**The host's use is covered.** Every launch the script makes has `group_size = 4` and `NUM_SM` the
device's multiprocessor count (`132` on the H200, point 7), and meets `Inst.ok`:
- the test: `M = N = K = 1024, 512, 256, 128`, `lda = K` (`A` from `torch.rand`, row-major),
  `ldb = K` (`B.T.contiguous()`, `N × K`), `ldc = N` (`C = torch.empty((M, N))`); `start 4 = 64 +
  16 + 4 + 1 = 85`;
- the square benchmark: four groups of `N × N × N`, `N ≤ 1024`, `g_lds = [N, N, N]` (the strides
  of `A`, `B_T` and `C`, all `N × N` row-major); `start 4 ≤ 256`;
- the batch benchmark: four groups of `M × 8192 × 8192`, `M ≤ 1024`, `g_T_lds = [K, K, N] = [8192,
  8192, 8192]`; `start 4 ≤ 4·8·64 = 2048`; `K = 8192 = 2^13`.

Every stride is a multiple of `8` below `2^31`, `ldc = N`, every matrix and table is a fresh PyTorch
allocation used from its start (`data_ptr()`), so the matrices are 16-byte aligned (PyTorch's CUDA
allocations are), and no two of them share a byte. The conditions are more general than that use:
any number of groups and of programs within the bounds, any sizes (`M g = 0`, `N g = 0`: a group
with no tile; `K g = 0`: no loop trip, `C g = 0`), padded rows, and `A g`, `B h` overlapping each
other in any way (both are only read).

`Pre d init`: the five tables and every `A g`, `B g`, `C g` (`g < group_size`) are valid in the
initial memory (`Buf.Valid`: base address non-negative and aligned to the element size, the buffer
inside `[0, 2^64)`, every byte present); for each `g < group_size`, `group_gemm_sizes` holds
`M g, N g, K g` at elements `3g, 3g + 1, 3g + 2`, `g_lds` holds `lda g, ldb g, ldc g` there, and
`group_a_ptrs`, `group_b_ptrs`, `group_c_ptrs` hold the addresses of `A g`, `B g`, `C g` at element
`g` (as `int64`). Every `float16` value is allowed in the matrices, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch the script makes of this kernel is a direct launch of the
`@triton.jit` function: `grouped_matmul_tma_kernel[grid](…)` in `group_gemm_tma_fn` (lines
323–325), called at line 355, and in `triton_tma_perf_fn` (lines 376–378), called through
`triton.testing.do_bench(lambda: triton_tma_perf_fn(…), …)` (lines 446–448, 517–519). Point by
point:

- **4 (options).** The launches give the constexprs `FP8`, `NUM_SM`, `BLOCK_SIZE_M`,
  `BLOCK_SIZE_N`, `BLOCK_SIZE_K` and two options, `num_stages=3` and `num_warps=4`, two of the four
  that do not change the meaning; no `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` or any
  other option. Nothing calls `grouped_matmul_tma_kernel.warmup(…)` or `triton.compile`; the kernel
  has no `@triton.autotune` (removed, SETUP.md; `tma_configs` is never used) and no
  `@triton.heuristics`, and the decorator is a bare `@triton.jit`, so its `debug` is not set. So
  `launch` keeps every default: `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`,
  `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds three descriptors, so it needs an allocator:
  `group_gemm_tma_fn` sets one (`triton.set_allocator(alloc_fn)`, line 320) before its launch, and
  the benchmarks' launches come later in the same script and thread, after the test's call (line
  355, under the same `supports_tma()` as the benchmarks' `triton-tma` line), with no other
  `set_allocator`. `alloc_fn` returns a fresh `torch.empty(size, device="cuda",
  dtype=torch.int8)`: a new allocation, made while the tables and matrices the launch is given
  are alive, so it holds none of their bytes, and only this launch writes it (the script does
  nothing else with it).
- **6 (precompiled kernel).** No `CompiledKernel` is launched: every launch is a JIT launch above.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are five
  tensors, one integer and the constexprs.
- **9 (CUDA graphs).** No call is made during a CUDA stream capture, and nothing is replayed: the
  script calls `group_gemm_tma_fn` directly (line 355) and `triton_tma_perf_fn` through
  `triton.testing.do_bench`, which times plain calls of its function with CUDA events, not
  `triton.testing.do_bench_cudagraph`; the script opens no `torch.cuda.graph` and makes no capture
  of its own.

So each call makes this one direct launch, which the claim covers, from the memory at that launch.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`, not under `@triton.autotune`. Its parameters
are five pointers (tensor arguments), one unannotated integer (`group_size`, an `int` argument) and
five `tl.constexpr`s, four with `int` arguments and `FP8` with the `bool` argument `False`, none
with a default. Its statements are assignments `name = e`, two `name += e` (lines 266, 275), an
expression statement (`c_desc.store(…)`), one `for g in range(group_size)` (one argument), one
`for kk in range(0, …)` (two positional arguments), one runtime `if` without `else` (line 220) and
one `while` (line 251). Its expressions are a conditional expression on the compile-time `FP8`
(line 209), `tl.program_id(0)`, `tl.load(p)` of scalar pointers, `.to(tl.pointer_type(dtype))`,
`tl.cdiv`, `tl.make_tensor_descriptor(p, shape=…, strides=…, block_shape=…)`, `desc.load`,
`desc.store`, `tl.zeros(…, dtype=tl.float32)`, `tl.dot(a, b)`, `.T`, `x.to(dtype)`, the dtypes
`tl.float8e4nv` and `tl.float16`, Python's `range`, the operators `+ - * // %`, the comparisons
`>=` and `<` (one per operand of `and`), and `and` on two runtime `int1` scalars. The only
module-level names it reads are `triton` (its decorator) and `tl`, Triton's modules, each bound only
by its own `import` (lines 32 and 33). No other statement of the module binds `triton`, `tl` or
`grouped_matmul_tma_kernel` (the others bind `Optional`, `torch`, `DEVICE`, `is_cuda`,
`supports_tma`, `num_sms`, `grouped_matmul_kernel`, `group_gemm_fn`, `tma_configs`,
`group_gemm_tma_fn`, the test's names, `triton_perf_fn`, `triton_tma_perf_fn`, `torch_perf_fn`,
`benchmark_square_matrices` and `benchmark_batches`; no decorator, default or annotation binds a
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
  `float32` (line 266; [num.md](../../docs/rules/num.md)): for every `input_precision`, with
  `maxImpreciseAcc = 0`, each output element whose row of `a` and column of `b` are finite and whose
  accumulator is finite is finite and within `relDot16 K·(|acc| + Σ|a·b|) + absDot16 K` of
  `acc + Σ a·b`, when that leaves room below the `float32` range.
- `dot_total` — `Rules.Num.DotTotal P cfg` for the model's configuration (`float16` operands,
  `float32` accumulator, `tf32`, `maxImpreciseAcc = 0`): the `tl.dot` of line 266 always has a
  result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversion of line 271, the
`int32`-to-`int64` conversions of the strides (lines 233, 240, 246) and the separate addition of
`tl.dot_add` are operations the library defines (`tl.cvt`, `ConvRel`, `AccAdd`), and need no
platform fact beyond `dot`.

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false` (Triton's default) by definition of `launch`.
- `hvalid` (`launch_valid`, by `Launch.valid_grid1`): the grid has `NUM_SM < 2^31` programs
  (`tiles_lt`), and each of the three block shapes is a list of powers of two with at most `2^20`
  elements.
- Footprints (`GroupedMatmulTma.Sched`): every program reads within the tables' first elements and
  the matrices `A g`, `B g` (`R`); program `p` writes the elements of the tiles it owns, tile `t`
  of group `g` with `(start g + t) % NUM_SM = p`: rows `tm·128 + i < M g`, columns
  `tn·128 + j < N g` with `tm = t / cdiv(N g, 128)`, `tn = t % cdiv(N g, 128)` (`tileIdx`,
  `tileBytes`, `Wt`, `W`). Different tiles share no byte (`tileBytes_disj`, from `N g ≤ ldc g` and
  `c_c`), and a global index names one tile (`start_inj`).
- `hsafe` (`safe_program`, in `GroupedMatmulTma.Program`): one rule per line, from
  `program_id = p` (`launch_pid`).
  - Line 212 (`safe_forRange`, `group_size` trips) with the invariant `OI g`: `last_problem_end`
    is `start g`, and `tile_idx` is some `y ≡ p` modulo `NUM_SM` with `start g ≤ y < start g + NUM_SM`;
    the memory satisfies `ProgInv p y`: it agrees with `init` outside the tiles `p` owns of global
    index below `y`, keeps every byte present, and holds in each of those tiles what `OutSpec` says.
    It holds at the start (`ProgInv_start`: `y = p`, no tile written).
  - Lines 214–216, 222–228 (`safe_load1`, `ptr_tab3`, `ptr_tab1`): the table elements `3g + c` and
    `g`; the memory agrees with `init` on `R` (`ProgInv_R`), so they hold `Pre`'s values.
  - Lines 217–219 (`safe_idiv1`): `num_m_tiles`, `num_n_tiles`, `num_tiles` are `cdiv(M g, 128)`,
    `cdiv(N g, 128)`, `tiles g` (no `int32` wrap: `M_le`, `N_le`, `tiles_lt`).
  - Line 220: since `start g ≤ y`, the condition is `y < start g + tiles g`. When it fails,
    `tile_idx` is unchanged and `start (g + 1) ≤ y < start (g + 1) + NUM_SM`.
  - Lines 226–248 (`safe_convert`, `conv64`, `safe_make_tensor_descriptor`, `intToPtr_ofInt`): the
    descriptors are `⟨A g's base, [M g, K g], [lda g, 1], [128, 64], zero⟩` and the like (the
    address read back as the matrix's base; for `A g`, `B g` only when `K g > 0`, when the buffer
    is not empty).
  - Line 251 (`safe_while`) with the invariant `WI` (`tile_idx = y ≡ p`, `start g ≤ y <
    start (g + 1) + NUM_SM`, `ProgInv p y`), the variant `mu = start g + tiles g − tile_idx` and the
    footprint `Wv v`, the tile `tiles g − v` when `p` owns it (`Wv_disj`, `Wv_sub`). When the loop
    ends, `start (g + 1) ≤ y`, and line 278 gives `OI (g + 1)`.
  - Lines 252–275 (`safe_tile`, in `GroupedMatmulTma.Tile`), at `y = start g + t`, `t < tiles g`:
    `tile_m_idx = tm`, `tile_n_idx = tn` (`safe_idiv1`, `safe_irem1`); the loop of line 263
    (`safe_forRange`, `cdiv(K g, 64)` trips) with the invariant: memory unchanged, and `AccInv`:
    every element of the tile whose rows of `A g` and `B g` are finite is finite and within
    `DotLoop.bound ρ α s (partA s)` of the exact partial sum `partS s` (the products of
    `k < 64·s`); it holds at `s = 0` (`acc0`: `+0.0`, exact).
  - Lines 264–265 (`safe_desc_load`, `access2`, `lane2`): each access meets the descriptor's
    requirements (positive shape entries below `2^31`, `16`-byte-aligned base, row stride and inner
    offset of multiples of `16` bytes); a lane `(i, k)` inside the tensor reads element
    `(tm·128 + i)·lda g + 64·s + k` of `A g` (inside the matrix, in `R`), which the memory holds
    as `init` does; a lane outside reads the padding `+0.0` (`f16_zero`).
  - Line 266 (`safe_dot_add`, with `dot_total`), then `acc_step`: by `dot` (`Rules.Num.DotF16`, at
    `K = 64`), `DotLoop.dotAdd_step` (either form of `tl.dot_add`) and `DotLoop.accum_step`, the
    accumulator keeps `AccInv` at `s + 1` (the padded lanes add exact zeros; the values stay far
    inside the `float32` range because `K ≤ 2^13`; `acc_step` needs only `K ≤ 2^16`).
  - Line 271 (`safe_cvt`): `c` is the rounding of the accumulator to `float16`, up to NaN payload.
  - Line 272 (`safe_desc_store`): lanes inside `M g × N g` store to element
    `(tm·128 + i)·ldc g + tn·128 + j` of `C g`, inside the tile `t` (the write footprint `Wv`), distinct
    lanes to distinct elements (`c_inj`, from `N g ≤ ldc g`). After `cdiv(K g, 64)` trips the
    partial sums are the whole sums (`partS_end`), so `OutSpec` follows from `AccInv` and the last
    rounding (`out_f16`, by `FloatV.cvt_err`), and `ProgInv_step` gives `ProgInv p (y + NUM_SM)`.
  - Line 275: `tile_idx + NUM_SM` does not wrap (`tiles_lt`), and the variant decreases.
  - After the loop of line 212, `start group_size ≤ y`, so every tile `p` owns is written
    (`ProgInv_final`): `Q`.
- `hdisj`: program `i`'s tiles are disjoint from `R` (`R_W`: `c_tab`, `c_ab`) and, for `i ≠ j`,
  from program `j`'s tiles (`W_disj`: a tile has one owner, its global index modulo `NUM_SM`).
- `hRW`: `R_W`.
- `hpost`: element `(i, j)` of `C g` is in tile `t = (i / 128)·cdiv(N g, 128) + j / 128` at lane
  `(i % 128, j % 128)` (`tile_of`), which program `(start g + t) % NUM_SM` owns; its `Q` gives
  `OutSpec` in its memory, and the final memory agrees with it on its footprint, so
  `Mem.read_congr` gives the value. A byte outside every `C g` is in no write footprint (`W_sub`),
  so it keeps its initial value.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref` gives a total behaviour that
satisfies `faithful`; `Platform.sm90ref.dotF16` gives `dot` and `Platform.ref.dotTotal` gives
`dot_total`.

`satisfiable`: the source's own test instance (lines 329–357): four groups of
`M = N = K = 1024, 512, 256, 128` (`testSz`), `lda = ldb = ldc` the size, matrices of `2^20`
elements at the 16-byte-aligned addresses `2^24·(g + 1)`, `2^24·(g + 5)`, `2^24·(g + 9)`, the tables
at `4096`, `8192`, `12288`, `16384`, `20480`, and `NUM_SM = 132` (`testInst`), in a memory where
every byte is present and the tables hold their values (`testMem`, built with `fillBuf`, read back
with `fillBuf_read`), meets `Inst.ok` (`test_ok`) and `Pre` (`test_pre`).

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float16` only, `FP8 = False`.** This folder is the tutorial's `float16` runs: `A g`, `B g`
  and `C g` are `float16`, and `FP8` is `False` on every launch the script makes
  (`torch.float8_e4m3fn == group_A[0].dtype`, line 324, and `torch.float8_e4m3fn == dtype` with
  `dtype=torch.float16`, line 376). The `FP8 = True` path (`dtype = tl.float8e4nv`, an fp8 result
  and fp8 pointers) is never launched by the tutorial and is not covered (SETUP.md).
- **The launch's configuration.** `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 128`, `BLOCK_SIZE_K = 64`
  (`num_stages=3`, `num_warps=4`), the config SETUP.md picks; other configurations are not covered.
  `NUM_SM` is any positive number within `tiles_lt`.
- **`int32` bounds** (`G_le`, `tiles_lt`, `M_le`, `N_le`, the strides below `2^31`). The tables
  are `int32` tensors, so their entries are `int32` anyway; the bounds keep the kernel's `int32`
  arithmetic (`g * 3 + 2`, `gm + 127`, `num_m_tiles * num_n_tiles`, `last_problem_end + num_tiles`,
  `tile_idx + NUM_SM`) from wrapping. The host's largest launch (`start 4 ≤ 2048` tiles) is far
  inside.
- **`K ≤ 2^13`.** The tolerance grows with `K`. Up to `2^13` it is below `0.21%` of `Σ|a·b|`
  (`tol_rel_lt`), and `Post`'s whole bound on `c`, with the last `float16` rounding, below `0.253%`
  (products of one sign), under a `bfloat16` rounding's `0.389%`, so `Post` still tells a `float16`
  result from a `bfloat16`-rounded one; from `K = 13633` (`T = 214` steps) it would not. The host's
  largest `K` is exactly
  `8192 = 2^13`. The proof also uses the bound to keep the partial sums inside the `float32`
  range.
- **Row strides multiples of `8` and 16-byte-aligned matrices** (`lda_ok`, `ldb_ok`, `ldc_ok`,
  `base_al`): the descriptors' requirements (`TensorDesc.Valid`); the host's strides
  (`128` to `8192`) and fresh allocations meet them.
- **Rows of `C g` within `ldc g`** (`N g ≤ ldc g`): row-major, possibly padded; overlapping rows
  are not covered. Every layout is row-major with unit column stride, as the descriptors' strides
  `[ld, 1]` fix.
- **No overlap** of a `C g` with another `C h`, with any `A h` or `B h`, or with a table, and of an
  `A h` or `B h` with a table (the last because pointer and descriptor accesses to the same bytes are
  out of scope, [exclusions.md](../../docs/exclusions.md)). The host allocates every buffer fresh.
- **Accuracy for finite inputs only.** With a NaN or an infinity in a row of `A g` or `B g`, the
  claim says only that the element of `C g` holds some `float16` value.
- **`grouped_matmul_kernel`** (lines 52–128) is excluded ([exclusions.md](../../docs/exclusions.md));
  its launches are host code here.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions): among it, point 7's target and host (an
  H200, driven from an x86-64 host running CPython 3.13 without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment). The script's `DEVICE` is
  whatever device is active.

Modelling decisions:

- The pointer arguments are tensors of their dtypes: `group_*_ptrs` `*int64` (`torch.tensor` of
  Python integers), `group_gemm_sizes` and `g_lds` `*int32`; `group_size` is an `int32`
  (`IntV.ofNat`): Triton passes an integer argument in `[-2^31, 2^31)` other than `1` as an `int32`,
  and at `1` the constant `1`, whose only use, `range(group_size)`, then runs the same single trip
  with an `int32` induction variable (`model`'s docstring).
- `FP8` is no parameter of the model: it is a compile-time constant, `False` on every covered
  launch, and only line 209 reads it, whose conditional expression on it evaluates only its `else`
  side, giving `dtype` the compile-time dtype `tl.float16` (dropped in the map, `config`); the
  model writes `float16` at its uses (lines 226–228 and 271).
- `last_problem_end = 0` before the loop that carries it is an `int32` scalar tensor
  (`IntV.ofInt 0 : I32`); every constexpr operand of an `int32` operation (`BLOCK_SIZE_M - 1`,
  `BLOCK_SIZE_M`, `BLOCK_SIZE_K`, `NUM_SM`) and every Python integer beside an `int32` (`3`, `1`,
  `2`) is built in `int32`; `tl.cdiv(x, B)` is `(x + (B − 1)) // B` (`tl.idiv`, truncating).
- The `and` of lines 220 and 251 evaluates both comparisons (`&&` of the two `int1` values); the
  `if` of line 220 has no `else`, so the model's `else pure tile_idx` keeps `tile_idx`.
- `tl.load(p).to(tl.pointer_type(tl.float16))` is `tl.intToPtr .f16` of the loaded `int64`.
- Each descriptor's strides are converted to `int64` (`tl.convert .i32 .i64`), the Python `1`
  as an `int32` tensor first; the padding is the default `"zero"`.
- `accumulator += tl.dot(a, b.T)` is `tl.dot_add` (the combine pass's rewrite: the dot's
  accumulator is the default zero and the product has no other use), with `b.T` as `b.trans`, the
  default `input_precision` (`tf32`) and `max_num_imprecise_acc` (`0`); `.to(tl.float16)` is one
  rounding to nearest even (`tl.cvt`).
- The launch lists the source's three block shapes (`[128, 64]`, the blocks of `a` and `b`;
  `[64, 128]`, `b.T`; `[128, 128]`, the accumulator, `c` and its block); `enable_fp_fusion` is
  written as Triton's default `true`; `num_warps=4` and `num_stages=3` change no result and are not
  recorded.
