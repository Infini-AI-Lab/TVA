# matmul_e5m2

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/03-matrix-multiplication.py`](source/03-matrix-multiplication.py). Lean: [`lean/`](lean/),
the module `MatmulE5m2` (namespace `Kernels.MatmulE5m2`) and its parts `MatmulE5m2.Defs` (model,
conditions, specification, assumptions), `MatmulE5m2.Arith` (the program order),
`MatmulE5m2.Basic` (arithmetic and float lemmas), `MatmulE5m2.Program` (one program) and
`MatmulE5m2.Reject` (what the tolerance rejects).

The kernel is `matmul_kernel` (source lines 228–316). It reads two module-level names, `triton`
(its decorator) and `tl`, bound by the imports of lines 154–155, which are kernel lines too and
are mapped (dropped: they add no step). The host function `matmul(a, b, activation="")` (lines
330–350) checks `a.shape[1] == b.shape[0]` and that `a` is contiguous, allocates
`c = torch.empty((M, N), dtype=torch.float16)` (row-major, strides `(N, 1)`), and launches a 1-D
grid of `triton.cdiv(M, 128) * triton.cdiv(N, 256)` programs with `BLOCK_SIZE_M=128`,
`BLOCK_SIZE_N=256`, `BLOCK_SIZE_K=64`, `GROUP_SIZE_M=8`, `num_stages=3`, `num_warps=8` and
`ACTIVATION=activation`.

This folder is the script's **`float8e5` run**: the test of lines 372–388 (`M = N = K = 512`) and
the benchmark's `fp8` line (lines 400–440 with `fp8_inputs=True`: `M = N = K = 256, 384, …, 4096`).
Both draw `float16` matrices with `torch.randn`, cast `a` to `torch.float8_e5m2`, and cast the
transposed view `b.T` to `torch.float8_e5m2` (lines 377–380, 428–430), which keeps its strides: on
these square matrices `b` has strides `(1, K)`. So `a` and `b` are `float8e5` (`a` contiguous,
strides `(K, 1)`) and `c` is `float16`. The script never passes an `activation`. Its `float16`
run (lines 359–370 and the benchmark's `fp16` line) is not this folder's: it is
[`kernels/matmul_f16/`](../matmul_f16/README.md), a claim on the same source file (Narrowings).

This claim starts from `matmul_f16`'s Lean, model map and write-up, and changes what the `float8e5`
operands change: the pointer dtypes of `a_ptr` and `b_ptr` (`*fp8e5`), the loads' `other=0.0`
(converted to `float8e5` on the device, with the library's fact `CvtF32E5m2`), the `tl.dot`
configuration (`float8e5` operands, `max_num_imprecise_acc = 2^30`) and its library fact
(`DotE5m2`), and the tolerance (the fp8 fact applied step by step against the exact partial
sums, `K ≤ 4096`). The program order, the index arithmetic, the store and the last rounding to
`float16` are unchanged.

Every other module-level line is a host line: `import torch` (152), `DEVICE = …` (157), `is_cuda`
(160–161, called by the host at lines 373, 400, 404), the tests, the benchmark's configs and
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
  `i·stride_cm + j·stride_cn` of `c`'s storage) can be read in the final memory, with some
  `float16` value `c` (`OutSpec`). When, moreover, row `i` of `a` and column `j` of `b` hold finite
  `float8e5` values `va k` and `vb k` (`k < K`) in the initial memory, then, with
  `S = Σ_k va k · vb k` the exact dot product and `E = tol va vb K` the tolerance below:
  - `c` is the `float16` rounding (to nearest even, `FloatV.cvt`, up to NaN payload) of a finite
    `float32` value `v` with `|v − S| ≤ E`;
  - when `|S| + E ≤ 65504` (the largest `float16`), `c` is finite and
    `|c − S| ≤ E + (|S| + E)·2^-11 + 2^-25` (one more `float16` rounding:
    `Format.binary16.u = 2^-11`, `Format.binary16.eta = 2^-25`).

  **The tolerance** (`tol`, `errAt` in `MatmulE5m2.Defs`) is the library's fp8 `tl.dot` fact
  (`Rules.Num.DotE5m2` at `K = 64`, the loop's step) applied step by step against the exact
  partial sums. Let `p e = va e · vb e` for `e < K` and `0` from `K` on (`prodV`), `T = cdiv(K, 64)`
  the number of steps, `S_t = Σ_{e < 64t} p e` the exact partial sum before step `t` (`partSum`),
  and `B_t = Σ_{64t ≤ e < 64t+64} |p e|` the magnitudes of step `t`'s products (`stepMag`). Then
  `E_0 = 0`, `E_{t+1} = (1 + ρ)·E_t + ρ·(|S_t| + B_t) + α`, and `tol va vb K = E_T`, with the
  library's constants for one fp8 `tl.dot` of `K = 64`: `ρ = relDot8 64 = (1 + 2^-7)^2 − 1 =
  257·2^-14` (about `1.57%`, `relDot8_64`) and `α = absDot8 64 = 16641·2^-139` (below `2.4·10^-38`,
  `absDot8_64`). Each step is what the fact gives: step `t` adds its products to an accumulator
  `a` within `E_t` of `S_t`, up to `ρ·(|a| + B_t) + α`, and `|a| ≤ |S_t| + E_t` (`err_step`). The
  tolerance is explicit and computable from the inputs, and depends on where the products are
  (through the partial sums of the steps of `64`). It is never above `tolA K Σ|a·b|`, the
  claim's first tolerance, `((1 + ρ)^T − 1)·Σ|a·b| + T·(1 + ρ)^T·α` (`tol_le_tolA`), and it equals
  it, up to the `α` terms, on an input whose only product is in the first step.

  **What it rejects, at every covered size** (`MatmulE5m2.Reject`). `Post` holds for every input,
  so it rejects a wrong kernel as soon as one input exposes it.
  - With one product `X = x·y` at index `K − 1` (the last step), `tol ≤ ρ·|X| + 200·α`, about
    `1.57%` of `|X|` plus less than `10^-35`, at every covered `K` (`tol_spike`).
  - So a zero result (a kernel that stores `0`, or skips the loop) breaks `Post` at every covered
    `K ≥ 1` (`zero_rejected`: with `X = 1`, `0` is outside the `float16` bound, and there is room).
    So do a result `S/2`, `−S`, or any result more than about `1.6%` from `S` on that input, and a
    float8e5 rounding of the result where `S` is no float8e5 value: with `x = y = 1.25`,
    `S = 1.5625`, its float8e5 rounding `1.5` is `0.0625` from `S`, above the bound
    `0.0157·1.5625 + (1.5625 + 0.0245)·2^-11 + 2^-25 < 0.026`.
  - A kernel that skips any one step `s` of the loop (and is otherwise exact), returning
    `S − X_s` with `X_s` the exact sum of step `s`'s products, breaks `Post` at every covered `K`
    (`drop_rejected`): for the last step on the input above; for an earlier step on the row `1`
    at `64·s`, `-1` at `64·s + 64` against a column of `1`s, where `S = 0`, `X_s = 1` and the
    tolerance is below `1/5` (about `0.124` for `s = 0` at `K = 4096`).

  **What it does not reject.** A result within the tolerance of `S` on the input at hand, which
  is the device's own allowed error, and can be large:
  - on an input whose weight sits in the first step and stays in the partial sums (one product
    `X` in step `0`), the tolerance is `((1 + ρ)^T − 1)·|X|`: `13.3%` of `|X|` at the test's
    `K = 512`, above `|X|` from `K = 2817`, and `170.8%` at `K = 4096`; there a zero result passes
    on that input (and is rejected on the inputs above);
  - on products of one sign (all-ones matrices, say) it is `7.3%` of `S` at `K = 512` and
    `72.8%` at `K = 4096`;
  - on the script's own data (`torch.randn` cast to `float8e5`; exact evaluation over 20 draws,
    median), it is about `7.0` against a typical `|S|` of `22.6` at `K = 512` (`0.30·|S|`; the
    first tolerance gave `41.7`), and about `114` against a typical `|S|` of `55` at `K = 4096`
    (`2.3·|S|`; the first gave `4415`). So on such data a float8e5 rounding of the result (at most
    `1/9` relative) passes at both sizes, and at `K = 4096` the bound does not fix the sign of a
    typical output.

  The test's own check (`torch.allclose(…, atol=0.125)`, line 385) is far tighter than anything
  the library's fact promises on that data, and `Post` does not imply it.

  With a NaN or an infinity in row `i` of `a` or column `j` of `b`, the claim says only that
  element `(i, j)` holds some `float16` value.
- **Nothing else changes.** Every byte outside the elements of the matrix `c` is the same in `fin`
  as in `init`: `a`, `b`, and the elements of `c`'s storage that are not in the matrix (padding of a
  row, say).

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `M`, `N`, `K`, the six strides (in elements), two
`float8e5` buffers `a`, `b` (`Buf .f8e5`) and a `float16` buffer `c` (`Buf .f16`), each a base
address and a length in elements; element `(i, k)` of the matrix `a` is element
`i·stride_am + k·stride_ak` of the buffer `a`, and likewise for `b` and `c` (`aIdx`, `bIdx`,
`cIdx`).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `M_le`, `N_le` | `M + 128 ≤ 2^31`, `N + 256 ≤ 2^31` | `M` and `N` are `int32` arguments, and `tl.cdiv` computes `M + 127` and `N + 255` in `int32` (lines 253–254); the tile rows and columns `pid_m·128 + i`, `pid_n·256 + j` (lines 282–283, 312–313) then stay below `2^31` |
| `K_le` | `K ≤ 2^12 = 4096` | the host's largest `K` (the benchmark's `4096`). At every covered `K` the tolerance rejects a zero result and a skipped step (`zero_rejected`, `drop_rejected`; Specification); beyond the host's use it would only grow. The proof also uses it to keep the partial sums far inside the `float32` range |
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
(`M = N = K ≤ 4096`) of the `float8e5` run meet `Inst.ok`: `a` is contiguous (`matmul` asserts it:
strides `(K, 1)`), `b` is the transposed view cast to `float8e5` (strides `(1, K)`, so
`stride_bk = 1`, `stride_bn = K`), and `c` comes from `torch.empty` (strides `(N, 1)`), so `a_fit`
reads `M·K + 64 ≤ 2^31`, `b_fit` reads `64 + N·K ≤ 2^31`, `cm_fit` reads `M·N < 2^31` and `c_rows`
reads `N ≤ N` (at `4096`: `2^24 + 64`, `2^24 + 64`, `2^24`); `K ≤ 4096`; `b`'s storage holds the
`K·N` elements `k + j·K` (`b_len`); the fresh `c` overlaps neither input. The conditions are more
general than that use: any positive strides (a contiguous `b`, a non-contiguous `a`), any `M`, `N`
within the bounds and any `K ≤ 4096`, `M = 0` or `N = 0` (an empty grid: memory unchanged) and
`K = 0` (no loop trip: `c = 0`). `a` and `b` may overlap each other in any way (both are only
read). The lemma `fp8_run_ok` (in `MatmulE5m2`) proves this coverage: every `n × n × n` instance
with `1 ≤ n ≤ 4096`, `a` of strides `(n, 1)`, `b` of strides `(1, n)`, `c` of strides `(n, 1)`,
buffers of at least `n·n` elements and a `c` sharing no byte with `a` or `b`, meets `Inst.ok`;
`satisfiable` checks the test's own instance.

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to the element size (1 byte for `a` and `b`, 2 for `c`), the buffer
inside `[0, 2^64)`, every byte present). Every `float8e5` value is allowed in the inputs, NaN and
infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch of the `float8e5` run is a direct launch of the `@triton.jit`
function, the JIT launch `matmul_kernel[grid](…)` of `matmul` (lines 340–349), called at line 381
and, in the benchmark, through `triton.testing.do_bench(lambda: matmul(a, b), …)` (line 435).
Point by point:

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
  script calls `matmul` directly (line 381) and through `triton.testing.do_bench` (line 435), which
  times plain calls of its function with CUDA events, not `triton.testing.do_bench_cudagraph`; the
  script opens no `torch.cuda.graph` and makes no capture of its own.

So a call of `matmul` makes this one direct launch, which the claim covers, from the memory at that
launch.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it. Among it,
point 7's target, an H200 (sm_90): the model's default `max_num_imprecise_acc = 2^30` for the fp8
dot and the facts `DotE5m2` and `CvtF32E5m2` are the H200's.

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
`+ - * // % &` (on integers and pointers only: no operator computes in an fp8 dtype, and no unary
`-` is applied to an fp8 tensor; the `float8e5` tiles go only to `tl.dot`), the comparisons
`>= > < ==` (one per expression; `==` only between compile-time constants), and the subscripts
`[:, None]` and `[None, :]`. The only module-level names it reads are `triton` (its decorator) and
`tl`, Triton's modules, each bound only by its own `import` (lines 154 and 155); line 307 names the
module's `@triton.jit` function `leaky_relu`, bound only by its `def`, but the front end does not
visit that line. No other statement of the module binds `triton`, `tl`, `matmul_kernel` or
`leaky_relu` (the others bind `torch`, `DEVICE`, `is_cuda`, the three `get_…_autotune_config`,
`matmul`, `a`, `b`, `triton_output`, `torch_output`, `TORCH_HAS_FP8`, `ref_lib`, `configs`,
`fp8_inputs` and `benchmark`; no decorator, default or annotation binds a name).

## Assumptions

Five assumptions, all of kind `[rule]`, each cited as the library states it
([num.md](../../docs/rules/num.md), "Facts the library states"):

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid (`Launch.Valid`), then from every initial memory from
  which the model's launch is well behaved (no undefined behaviour, no race between programs, no
  conflict within a program), every outcome of the compiled kernel launched as `launch d` is one
  of the model's: a normal end with a memory some interleaving of the model leaves, never a fault,
  and a hang only if some interleaving runs forever. The rule covers a launch only within its
  scope, which its text states: part (A), which the launches meet (Conditions), and part (B), the
  environment, which the claim relies on
  ([rule page](../../docs/rules/compiler-faithful.md)).
- `dot` — `Rules.Num.DotE5m2 P`, the library's fact for `tl.dot` of `float8e5` operands into
  `float32` with `maxImpreciseAcc = 2^30`, the default on sm_90 (line 300): for every
  `input_precision`, each output element whose row of `a`, column of `b` and accumulator are finite
  is finite and within `relDot8 K·(|acc| + Σ|a·b|) + absDot8 K` of `acc + Σ a·b`, when that leaves
  room below the `float32` range.
- `dot_total` — `Rules.Num.DotTotal P cfg` for the model's configuration (`float8e5` operands,
  `float32` accumulator, `tf32`, `maxImpreciseAcc = 2^30`): the `tl.dot` of line 300 always has a
  result.
- `cvt` — `Rules.Num.CvtF32E5m2 P`, the library's fact for converting `float32` into `float8e5`
  with the default rounding, in range: a finite value of magnitude at most `57344` converts to its
  rounding to nearest even, up to NaN payload. The model's `other=0.0` (lines 297–298) is the
  `float32` `+0.0` converted to the pointers' dtype on the device (`tl.convert`, which for an fp8
  target is the platform's `P.cvt`), so each disabled lane holds `+0` (`zero_conv`).
- `cvt_total` — `Rules.Num.CvtTotal P Format.binary32 Format.e5m2 .rn`: that conversion always has
  a result.

There is no `[kernel]` assumption. The `float32`-to-`float16` conversion of line 308 is an IEEE
rounding the library defines (`tl.cvt`) and needs no platform fact.

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
- `hsafe` (`safe_program`, in `MatmulE5m2.Program`): one rule per line, from `program_id = p`
  (`launch_pid`).
  - Lines 252–260 (`safe_idiv1`, `safe_irem1`): every `int32` operation stays in range and computes
    the natural-number formulas of `MatmulE5m2.Arith` (`fpmOf`, `gsOf`, `pmOf`, `pnOf`), so `pid_m`
    and `pid_n` are `pm d p` and `pn d p`, with `pm < cdiv(M, 128)` and `pn < cdiv(N, 256)`.
  - Lines 266–273 (`safe_assume`): `pid_m, pid_n ≥ 0` and the strides are positive.
  - Lines 282–286 (`safe_irem`): `offs_am` and `offs_bn` are `(pm·128 + i) mod M` and
    `(pn·256 + j) mod N` (`rowA`, `colB`); `a_ptrs` and `b_ptrs` are `aPtrs d p 0` and
    `bPtrs d p 0` (no `int32` offset wraps: `a_fit`, `b_fit`).
  - Line 294 (`safe_idiv1`, `safe_forRange`): the loop runs `T = cdiv(K, 64)` trips, with the
    invariant: memory unchanged, `a_ptrs = aPtrs d p t`, `b_ptrs = bPtrs d p t`, and `AccInv`: every
    element of the tile whose row of `a` and column of `b` are finite is finite and within
    `errAt (prodV va vb K) t` of the exact partial sum `partS t` (the products of `k < 64·t`). It
    holds at `t = 0` (`acc0`: `+0.0`, exact).
  - Lines 297–298 (`safe_convert` with `zero_conv_ex`, from `cvt_total`; `safe_load_buf`):
    `other=0.0` converts to a finite `float8e5` zero (`zero_conv`, from `cvt`: the rounding to
    nearest even of the `float32` `+0`, `e5m2_cvt_zero`); each enabled lane (`64·t + k < K`) loads
    its element of `a` (resp. `b`), inside the matrix, each disabled lane the zero.
  - Line 300 (`safe_dot`, with `dot_total`), then `acc_step`: by `dot` (`Rules.Num.DotE5m2`, at
    `K = 64`) and `err_step`, the accumulator keeps `AccInv` at `t + 1`: the step adds the products
    `X` of step `t` up to `ρ·(|acc| + B_t) + α`, and `|acc| ≤ |S_t| + E_t`, which is `E_{t+1}`. The
    disabled lanes add exact zeros. The values stay far inside the `float32` range: each product is
    below `57344² < 2^32`, the partial sums of magnitudes below `2^44`, `E_t` is at most
    `DotLoop.bound ρ α t (partA t)` (`errAt_le_bound`) and `(1 + ρ)^t ≤ 3` for `t ≤ 64`
    (`pow_rho_le`).
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
satisfies `faithful`; `Platform.sm90ref.dotE5m2` gives `dot`, `Platform.ref.dotTotal` gives
`dot_total`, `Platform.sm90ref.cvtF32E5m2` gives `cvt` and `Platform.ref.cvtTotal` gives
`cvt_total`.

`satisfiable`: the source's own `float8e5` test instance (lines 374–381), `M = N = K = 512` with
row-major `a` and `c` (strides `512` and `1`) and the transposed `b` (strides `1` and `512`), each
of `512·512` elements, at the 16-byte-aligned addresses `2^21`, `2·2^21` and `3·2^21`
(`testInst`), in a memory where every byte is present (`testMem`), meets `Inst.ok` and `Pre`.

The lemmas about the tolerance, `tol_le_tolA` (`MatmulE5m2.Basic`), `tolA_rel_lt`, `tolA_le`
(`MatmulE5m2.Defs`), and `tol_spike`, `zero_rejected`, `drop_rejected` (`MatmulE5m2.Reject`:
the error bound on steps with no products, `errAt_quiet`, and on inputs of `1`, `-1` and `0`), and
`fp8_run_ok` (the host's `float8e5` run meets `Inst.ok`, Conditions), are proved by exact rational
and integer arithmetic; `main` does not need them.

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float8e5` only.** This folder is the script's `float8e5` run: `a` and `b` are `float8e5`,
  `c` is `float16`. The script's `float16` run (lines 359–370, and the benchmark's `fp16` line) uses
  the same source with `float16` operands; it is not covered here, but by
  [`kernels/matmul_f16/`](../matmul_f16/README.md). No other operand dtype is covered.
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
- **`K ≤ 4096`.** `K_le` stops at the host's largest `K` (the benchmark's `4096`), and covers
  every benchmark size. The tolerance needs no narrowing: at every covered `K` it rejects a zero
  result and a skipped step on some input (`zero_rejected`, `drop_rejected`). It is still the
  device's allowed error, as the library's fp8 fact states it, and can be large on some inputs
  (Specification, "What it does not reject"). Beyond `4096` it would only grow.
- **Positive strides.** The kernel itself assumes them (`tl.assume`, lines 268–273); a launch with
  a zero or negative stride is undefined behaviour.
- **Rows of `c` within `stride_cm`** (`c_rows`: `N·stride_cn ≤ stride_cm`). Each row's `N`
  elements, `stride_cn` apart, must lie within `stride_cm` (row-major, possibly padded); `matmul`
  allocates `c` row-major. Column-major and other transposed layouts of `c` are not covered, even
  though their rows do not overlap, and neither are overlapping layouts.
- **No overlap of `c` with `a` or `b`.** `matmul` always allocates a fresh `c`. An in-place call
  would make programs race.
- **Accuracy for finite inputs only.** With a NaN or an infinity in a row of `a` or a column of
  `b`, the claim says only that the element of `c` holds some `float16` value. The library states
  no `tl.dot` fact for non-finite operands. The script's inputs are `torch.randn` values, far
  below `57344` in magnitude, cast to `float8e5`, so they are finite; `Pre` allows every value
  anyway.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions): among it, point 7's target and host (an
  H200, driven from an x86-64 host running CPython 3.13 without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment). The script's `DEVICE` is
  whatever device is active; its `float8e5` run is guarded by `is_cuda()` (line 373).

Modelling decisions:

- `M`, `N`, `K` and the strides are `int32` (`IntV.ofNat`): Triton passes an integer argument in
  `[-2^31, 2^31)` other than `1` as an `int32`, and at `1` the constant `1`, which gives the same
  model at each use (`model`'s docstring; every launch of the `float8e5` run passes
  `stride_ak = stride_bk = stride_cn = 1`). An integer that is a multiple of 16 is also marked
  divisible by 16 for the compiler, which is true of it (SEMANTICS.md, "Scalar arguments").
- `ACTIVATION` is no parameter of the model: it is a compile-time constant, `""` on every covered
  launch, and only line 306 reads it.
- Every constexpr operand of an `int32` operation (`BLOCK_SIZE_M - 1`, `GROUP_SIZE_M`, …) is built in
  `int32` (`IntV.ofNat … : I32`); `tl.cdiv(x, B)` is `(x + (B − 1)) // B`; Python's `min` with a
  tensor argument is `tl.minimum`, on `int32` the exact `tl.iminimum`.
- `tl.assume(c)` is a check: undefined behaviour when `c` is false (it never is on covered
  instances).
- `other=0.0` is the `float32` constant `+0.0` converted to `float8e5` on the device
  (`tl.convert .f32 .f8e5`, SEMANTICS.md "Constants": an `other=` with an fp8 pointer is converted
  at run time, not built as an fp8 constant, and a constant zero's conversion keeps its sign), which
  `CvtF32E5m2` makes `+0`; `tl.dot` has the default `input_precision` (`tf32`) and
  `max_num_imprecise_acc` (`2^30` for an fp8 × fp8 dot on sm_90), and its accumulator is the
  loop-carried value (no `tl.dot_add`); `.to(tl.float16)` is one rounding to nearest even
  (`tl.cvt`).
- The launch lists the source's ten block shapes (`[128]`, `[256]`, `[64]`, `[128, 1]`, `[1, 64]`,
  `[128, 64]`, `[64, 1]`, `[1, 256]`, `[64, 256]`, `[128, 256]`); `enable_fp_fusion` is written as
  Triton's default `true` (no operation of the model depends on it); `num_warps=8` and
  `num_stages=3` change no result and are not recorded.
