# softmax_f32

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/02-fused-softmax.py`](source/02-fused-softmax.py). Lean: [`lean/`](lean/), the
module `Softmax` (namespace `Kernels.Softmax`) and its parts `Softmax.Model` (the model, the
conditions, the specification, the assumptions), `Softmax.Err` (the error of one row) and
`Softmax.Proof` (the proof, `main`, `consistent`, `satisfiable`).

The kernel is `softmax_kernel` (source lines 84–109). It reads two module-level names, `triton`
(its decorator) and `tl`, bound by the imports of lines 26–27, which are kernel lines too and are
mapped (dropped: they add no step). The host function `softmax` (lines 124–175) takes a 2-D tensor
`x`, sets `BLOCK_SIZE = triton.next_power_of_2(n_cols)`, `num_warps = 8` and
`num_stages = 4 if SIZE_SMEM > 200000 else 2`, allocates `y = torch.empty_like(x)`, compiles the
kernel with `softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols,
BLOCK_SIZE=…, num_stages=…, num_warps=8, grid=(1,))`, computes
`num_programs = min(NUM_SM * occupancy, n_rows)`, and launches the compiled kernel on the grid
`(num_programs, 1, 1)` with the warm-up's own arguments (line 174). The script calls it on a
`float32` matrix of shape `1823 × 781` (the test, lines 186–190) and, in the benchmark, on
`float32` matrices `4096 × N` for `N = 256, 384, …, 12672` (lines 203–226, run at line 229). The
other lines outside the kernel (the imports of `torch` and `driver`, `DEVICE`, `is_hip`, `is_cdna`,
`naive_softmax`, the device properties of lines 115–121, the test and the benchmark) are host
lines: the kernel names none of what they bind.

## Specification

`Post d init fin` says two things.

- **The outputs.** For every row `r < n_rows` there are values `xs 0, …, xs (n_cols − 1)`, the
  row's input elements `x[r, k]` (element `r · input_row_stride + k` of `x`) as read in the initial
  memory, and **one factor `sc` for the whole row**, with

  `|sc − 1| ≤ sumTol(BLOCK_SIZE)`,

  such that for every `j < n_cols` the output element `y[r, j]` (element
  `r · output_row_stride + j` of `y`) can be read in the final memory, is a finite `float32` `c`,
  and

  `|c − sc · σ_j| ≤ (elemTol + gapTol · gap_j) · σ_j + floorTol · σ_max`.

  Here `σ_j = softmaxRef xs n_cols j = e^(xs j) / Σ_{k<n_cols} e^(xs k)` is the exact softmax in
  the reals (an input `−∞` counts as `e^(−∞) = 0`: `expIn`); `σ_max = softmaxMax xs n_cols` is the
  row's largest `σ_k` (the softmax at the row's largest input); and
  `gap_j = softmaxGap xs n_cols j = log(σ_max / σ_j)`, which is `x_max − x_j` for a finite input
  (`0` for an input `−∞`, whose `σ_j` is `0`). The tolerances are rational constants
  (computable):

  | Name | Value | Where it comes from |
  |---|---|---|
  | `sumTol B` | `(B + 86) · 2^-24` | the row sum: a tree of depth up to `B − 1` (`(B − 1)·2^-24`, plus `≈ 16·2^-24` of second order at `B = 2^14`), and the lanes' `tl.exp` errors summed over the row (`53·2^-24` of the exact sum, since `|s|·e^s ≤ 16·e^s + 2^-17`) |
  | `elemTol` | `2^-21 + 2^-30` | `tl.exp`'s relative `2^-22` and the division's `2^-22` |
  | `gapTol` | `3 · 2^-24 + 2^-30` | per unit of `x_max − x_j`: the rounding of `x_j − x_max` (`2^-24`) and the part of `tl.exp`'s error that grows with its argument (`2^-23·X` in `ExpF32`) |
  | `floorTol` | `2^-133` | `tl.exp`'s absolute `2^-140` and the division's `2^-148` (relative to `σ_max ≥ 2^-14`) |

- **Nothing else changes.** Every byte outside the output elements `y[r, j]` (`r < n_rows`,
  `j < n_cols`) is the same in `fin` as in `init`: the input, the padding of `y`'s rows beyond
  `n_cols`, and every other byte the launch is given.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

**Why a common factor.** Only the row sum's error grows with `BLOCK_SIZE`, and it is one number,
the computed sum `S`, shared by every output of the row: the outputs are `e_j / S`. So `Post`
states it once, as `sc` (in the proof, `sc = A / S` with `A` the exact sum, scaled), and holds each
output to a per-element tolerance that does not grow with `BLOCK_SIZE`. It follows that
`|c − σ_j| ≤ (sumTol + elemTol + gapTol · gap_j) · σ_j + floorTol · σ_max`, and that the ratio of
two outputs of a row, `c_i / c_j`, is within about `2·elemTol + gapTol·(gap_i + gap_j)` of
`σ_i / σ_j`, whatever `BLOCK_SIZE`.

**What it means at the covered sizes.**
- `sumTol` is `2.0·10^-5` at `BLOCK_SIZE = 256`, `6.6·10^-5` at `1024` (the test) and `9.8·10^-4`
  at `2^14` (the benchmark's largest rows); it is within `87·2^-24` of the library's bound for a
  sum along any tree of `B` leaves, `(B − 1)·2^-24` to first order (`sum_err_covers`), which is
  what the library states of Triton's reduction tree.
- Per element, the tolerance is relative: `elemTol ≈ 4.8·10^-7`, plus `gapTol ≈ 1.8·10^-7` per unit
  of `x_max − x_j` (at `x_max − x_j = 20`, `≈ 4.1·10^-6`). The term `floorTol · σ_max ≈ 9·10^-41 ·
  σ_max` matters only for outputs near `float32`'s subnormal range. So small outputs are held to a
  relative bound: in a row `[20, 0, …, 0]`, the outputs `σ ≈ 2.06·10^-9` must be within about
  `4·10^-6` of `sc·σ`, relatively, and `0` does not meet it.
- **A half-precision result does not meet `Post`**, at any covered `BLOCK_SIZE ≥ 2`. Take a row
  whose inputs are `0` and `−1` (`n_cols = 2`, or any `n_cols ≤ BLOCK_SIZE` with the other inputs
  `−∞`): `σ_0 = 0.7310585786…`, `σ_1 = 0.2689414213…`. If the outputs are `c_j = σ_j·(1 + δ_j)`,
  `Post` needs one `sc` with `|1 + δ_j − sc| ≤ elemTol + gapTol·gap_j + floorTol·σ_max/σ_j` for
  both, so `|δ_0 − δ_1| ≤ 2·elemTol + gapTol·1 + floorTol·(1 + e) ≈ 1.1·10^-6`. The `float16`
  roundings of `σ_0` and `σ_1` have `δ_0 ≈ −1.39·10^-4` and `δ_1 ≈ +3.78·10^-4`
  (`|δ_0 − δ_1| ≈ 5.2·10^-4`), and the `bfloat16` ones `δ_0 ≈ −8.1·10^-4`, `δ_1 ≈ +2.2·10^-3`, so
  neither meets `Post`. (At `BLOCK_SIZE = 1` every output is `σ = 1`, which every format holds.)
- The common factor's bound at the test's `BLOCK_SIZE = 1024`, `6.6·10^-5`, is above the test's
  `torch.allclose` `rtol = 10^-5`: it is the library's bound for a sum along any tree of depth up
  to `1023`, which the claim cannot sharpen without a sharper library statement about the tree.

## Conditions

An instance `Inst` is the host's choice: `n_rows`, `n_cols`, `input_row_stride`,
`output_row_stride` (in elements), the constexprs `BLOCK_SIZE` and `num_stages`, the number of
programs `num_programs` (the grid `(num_programs, 1, 1)`), and the two `float32` tensors `x`
(`input_ptr`) and `y` (`output_ptr`) as buffers (`Buf .f32`: a base address and a length in
elements). Element `j` of row `r` is `inIdx d r j = r · input_row_stride + j` of `x` and
`outIdx d r j = r · output_row_stride + j` of `y`; `inSet d` and `outSet d` are these elements for
`r < n_rows`, `j < n_cols`.

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `block_pow2` | `BLOCK_SIZE` is a power of two | Triton requires it of `tl.arange`; the host's `next_power_of_2(n_cols)` is one |
| `block_le` | `BLOCK_SIZE ≤ 2^14` | the host's largest block (`N = 12672`); the tolerance is stated up to it (see Narrowings) |
| `cols_le` | `n_cols ≤ BLOCK_SIZE` | each row fits in one block, as the host's `next_power_of_2(n_cols)` gives |
| `grid_pos` | `n_rows ≠ 0 → 0 < num_programs` | with a row to process there is a program; the host's `min(NUM_SM · occupancy, n_rows)` is positive then (Narrowings) |
| `grid_lt` | `num_programs < 2^31` | `Launch.Valid`; the host's grid is at most `n_rows` |
| `rows_grid` | `n_rows + num_programs ≤ 2^31` | `n_rows` is an `int32`, and the `int32` loop `tl.range(row_start, n_rows, row_step)` does not overflow its last increment (`tl.RangeOk`) |
| `in_stride_lt`, `out_stride_lt` | each stride is below `2^31` | the strides are `int32` arguments |
| `in_off`, `out_off` | `(n_rows − 1) · stride < 2^31` | the `int32` products `row_idx * input_row_stride` and `row_idx * output_row_stride` do not wrap for any row |
| `x_len`, `y_len` | every element the launch reads (writes) lies in `x` (`y`) | the kernel accesses `x[r, j]` and `y[r, j]` for `r < n_rows`, `j < n_cols` |
| `out_rows` | `n_rows ≤ 1` or `n_cols ≤ output_row_stride` | the output rows do not overlap; two programs writing the same element would race |
| `y_x` | the output elements share no byte with the input elements | a program that reads and writes the same bytes, or two programs that touch bytes one of them writes, are not well behaved (Narrowings) |

The four integer arguments are runtime `int32` arguments, except one equal to `1`, which Triton
passes as the compile-time constant `1`. Each use makes it an `int32` tensor first (`row_idx * 1`
builds the constant in `int32`, the dtype of the tensor operand; `col_offsets < 1` and the bound
`1` of `tl.range` likewise), so the model is the same (`model`'s docstring). Their values are part
of the instance, and the conditions constrain them, as the user's decision of 2026-10-29 on library
issue L117 asks (Modelling decisions).

Covered in particular: every shape the host uses (`1823 × 781`, `4096 × N` for every `N` of the
benchmark), and every other shape the conditions allow: `n_cols ≤ BLOCK_SIZE ≤ 2^14`,
`n_rows + num_programs ≤ 2^31`, strides below `2^31` with `(n_rows − 1) · stride < 2^31` for both
strides, and buffers holding every element accessed. For a contiguous input of `n_cols` columns,
`in_off` (`(n_rows − 1) · input_row_stride < 2^31`) limits `n_rows`: an input is covered while its
last row starts below element `2^31`. So `16384`-column rows are covered up to `131072` rows
(`131072 × 16384`, exactly `2^31` elements, is covered), and `12672`-column rows up to `169467`
rows (`2^31 + 2176` elements). The limit is the last row's start, not the element count, and it is
the kernel's own limit, not only the proof's: beyond it the `int32` product
`row_idx * input_row_stride` wraps (the column offsets are added to the 64-bit pointer, so they do
not). Also covered: `n_rows = 0` (the loop runs no trip); `n_cols = 0` (every lane is
masked off: nothing is read or written); general strides (contiguous or not, the input and output
strides different), so `x` and `y` may be row slices of larger tensors; any number of programs
from `1` (one program runs every row) up (programs beyond `n_rows` run no trip).

`Pre d init`: both buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to 4 bytes, inside `[0, 2^64)`, every byte present); every input element
the launch reads, `x[r, j]` for `r < n_rows`, `j < n_cols`, is a finite `float32` of magnitude at
most `2^126` or `−∞` (`InputOk`); and every row with at least one column has a finite element.
The host's inputs are `torch.randn` samples (magnitude below 10). See Narrowings for why `+∞`, NaN
and rows of only `−∞` are left out.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch the script makes is the launch of line 174,
`kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE,
num_stages)`, where `kernel` is the `CompiledKernel` that `softmax_kernel.warmup(…)` returns
(lines 143–144), in `softmax`, called at line 188 and, in the benchmark, through
`triton.testing.do_bench(lambda: softmax(x))` (line 222). Point by point:

- **4 (options).** The call that compiles the kernel, `warmup`, gives the constexprs
  `BLOCK_SIZE` and `num_stages`, the warm-up keyword `grid=(1,)`, and the options `num_stages`
  (the kernel's parameter has the option's name, so the keyword is both) and `num_warps=8`: both
  among the four options that do not change the meaning. It gives no `enable_fp_fusion`, `debug`,
  `num_ctas`, `launch_pdl` or any other option; the launch of the `CompiledKernel` takes none.
  There is no `triton.compile`, no `@triton.autotune` (so no `triton.Config`) and no
  `@triton.heuristics`, and the decorator is a bare `@triton.jit`, so its `debug` is not set. So
  `launch` keeps every default: `enable_fp_fusion = true`, `debug = false`, `num_ctas = 1`,
  `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor (no
  `tl.make_tensor_descriptor`), so no allocator is involved.
- **6 (precompiled kernel).** The `CompiledKernel` comes from `warmup`, so from the kernel's own
  source (an `ASTSource`, not an IR file), and the launch passes exactly the warm-up's arguments
  (`y`, `x`, `x.stride(0)`, `y.stride(0)`, `n_rows`, `n_cols`, and the constexprs `BLOCK_SIZE`
  and `num_stages`, the same values): the same dtypes, the same pointers (so the same alignment),
  the same integers (so the same class), and no host descriptor. So the launch is in the warm-up's
  specialisation, and it runs the code compiled for its own arguments.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are two
  tensors, four integers and two constexprs.
- **9 (CUDA graphs).** No call is made during a CUDA stream capture, and nothing is replayed: the
  script calls `softmax` directly (line 188) and through `triton.testing.do_bench` (line 222),
  which times plain calls of its function with CUDA events, not through
  `triton.testing.do_bench_cudagraph`; the benchmark sets a new current stream
  (`set_stream(stream)`, lines 217–218), which captures nothing; the script opens no
  `torch.cuda.graph` and makes no capture of its own.

The kernel has no `@triton.autotune`, so a call of `softmax` makes the one launch above (after the
warm-up, which compiles and runs no program), which the claim covers, from the memory at that
launch.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it. One place
where the source's own host code makes it worth saying: in `benchmark` (lines 215–226),
`x = torch.randn(…)` (line 216) runs on the stream that is current when the call starts, and lines
217–218 then make a new stream current, on which `softmax(x)` launches the kernel (line 222,
through `do_bench`). Nothing in lines 216–222 orders the new stream after the old one, so the
first launch of a call may start while `randn` is still writing `x`, unless `do_bench`
synchronises before it first calls its function. That is scope point 1 (nothing else writes the
launch's bytes from its start until it ends), in part (B): host code outside the claim, which the
claim assumes and nobody checks. For these launches the claim covers only a launch that starts
after `randn` has finished writing `x`.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`; its parameters are two pointers (tensor
arguments), four unannotated integers (`int` arguments) and two `tl.constexpr` with `int`
arguments, none with a default; its statements are assignments `name = e`, one
`for row_idx in tl.range(…)` (with the scheduling keyword `num_stages`), and one call
(`tl.store`). After the loop nothing follows, so no name first defined in its body, nor `row_idx`,
is used after it. Its expressions are `tl.program_id(0)`, `tl.num_programs(0)`, `tl.arange`,
`tl.load(…, mask=…, other=…)`, `tl.max(…, axis=0)`, `tl.exp`, `tl.sum(…, axis=0)`,
`tl.store(…, mask=…)`, the builtin call `float('inf')` and the unary `-` on its compile-time
value, and the operators: `*` of two `int32` scalars (lines 92, 107); `+` of a `float32` pointer
and an `int32` scalar (lines 92, 107) or an `int32` block (lines 96, 108), which advances the
pointer by that many elements ([SEMANTICS.md](../../docs/SEMANTICS.md), "Values"); `<` of the
`int32` block `col_offsets` and `n_cols` (line 98); `-` of a `float32` block and a `float32` scalar
(line 101); and `/` of a `float32` block and a `float32` scalar (line 105). The only module-level
names it reads are `triton` (its decorator) and `tl`, Triton's modules, each bound only by its own
`import` (lines 26 and 27): no other statement of the module binds `triton`, `tl` or
`softmax_kernel` (the others bind `torch`, `driver`, `DEVICE`, `is_hip`, `is_cdna`,
`naive_softmax`, `properties`, `NUM_SM`, `NUM_REGS`, `SIZE_SMEM`, `WARP_SIZE`, `target`,
`kernels`, `softmax`, `x`, `y_triton`, `y_torch` and `benchmark`, and `benchmark`'s decorator binds
nothing).

## Assumptions

Six assumptions, all of kind `[rule]`, each cited as the library states it:

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid (`Launch.Valid`), then from every initial memory from
  which the model's launch is well behaved (no undefined behaviour, no race between programs, no
  conflict within a program), every outcome of the compiled kernel launched as `launch d` is one of
  the model's. The rule covers a launch only within its scope: part (A), which the launches meet
  (Conditions), and part (B), the environment, which the claim relies on
  ([rule page](../../docs/rules/compiler-faithful.md)).
- `exp_total` — `Rules.Num.MathTotal P .exp .binary32`: `tl.exp` on `float32` has a result for
  every input, so the model's launch has no undefined behaviour at line 103.
- `exp_acc` — `Rules.Num.ExpF32 P`: the accuracy of `tl.exp` on `float32`, the sm_90 fact of
  [docs/rules/num.md](../../docs/rules/num.md) (relative `2^-22 + 2^-23·X` on `[−X, X]`, absolute
  `2^-140`).
- `exp_ninf` — `Rules.Num.ExpF32NegInf P`: `tl.exp(−∞) = +0` on `float32`, for the masked lanes
  (loaded as `−∞`) and the `−∞` inputs.
- `div_total` — `Rules.Num.DivTotal P .binary32`: `/` on `float32` has a result for every pair, so
  no undefined behaviour at line 105.
- `div_full` — `Rules.Num.DivFull P`: `/` on `float32` is `div.full.f32` (relative `2^-22`,
  absolute `2^-148`, for quotients up to `2^126`).

There is no `[kernel]` assumption. The subtraction, the max and the sum are IEEE operations whose
meaning is the library's, and need no fact about `P`.

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false` (Triton's default) by definition of `launch`.
- `hvalid` (`launch_valid`, by `Launch.valid_grid1`): the grid has fewer than `2^31` programs
  (`grid_lt`), and the one block shape `[BLOCK_SIZE]` is a power of two of at most
  `2^14 ≤ 2^20` elements.
- Footprints: program `p` reads the input rows `r ≡ p (mod num_programs)` (`progIn`) and writes
  those output rows (`progOut`), each restricted to `r < n_rows` and `j < n_cols`.
- `hsafe` (`safe_program`): `body_eq` (by `rfl`) names the loop body `rowProg`; program `p` has
  `program_id = p` and `num_programs` the grid size (`launch_ctx`). The loop is `tl.forRange` from
  `p` to `n_rows` by `num_programs`, within `tl.RangeOk` (`rows_grid`, `grid_lt`), and runs trip
  `t` exactly when `p + t · num_programs < n_rows` (`trip_lt_iff`). `safe_forRange` with the
  invariant "every byte present; nothing changed outside the rows of the earlier trips; the rows of
  the earlier trips are done (`RowDone`)", trip `t` writing row `p + t · num_programs`'s outputs
  (rows of different trips are disjoint, `rowOut_disjoint` from `out_rows`), and each trip by
  `safe_row`:
  - line 99: `safe_load_buf` with `Buf.addrs_of_get` (`mask_get`, `ptr_get`: lane `i` reads
    element `r · input_row_stride + i`, enabled exactly when `i < n_cols`; the `int32` arithmetic
    does not wrap by `in_stride_lt`, `in_off`); the enabled lanes hold the row's inputs, read in a
    memory that agrees with `init` on them (no output element meets an input element, `y_x`), and
    the disabled ones `−∞` (`negInf_lit`);
  - lines 101–105: `safe_maxAll`, `safe_sub`, `safe_math` (with `exp_total`), `safe_sum`,
    `safe_div` (with `div_total`), each giving what the result may be;
  - line 109: `safe_last` and `safe_store_buf` (lane `i` writes element
    `r · output_row_stride + i`, distinct lanes distinct elements);
  - the row is done by `row_bound` (`Err.lean`). The max is a finite input element `m` with
    every lane `≤ m` (`max_facts`, `Tree.eval_maxRel`). Each lane's `s = x − m ≤ 0` is rounded
    once (IEEE), and `tl.exp` of it is within `(2^-22 + 2^-33)·e^s + (3·2^-24 + 2^-33)·|s|·e^s +
    2^-139` of `e^s` (`exp_lane`, from `exp_acc`: for `|s| ≤ 1024` the rounding moves `e^s` by a
    factor `e^(t−s)` with `|t − s| ≤ 2^-13`; beyond, every term is below `2^-139`), or is `+0`
    for a masked lane or a `−∞` input (`exp_ninf`). With `|s|·e^s ≤ 16·e^s + 2^-17`
    (`abs_mul_exp_le16`), the lanes' errors add up to at most `53·2^-24` of the exact sum `A`
    (`A ≥ 1`, from the max's own lane). The sum `S` is then within `(B + 69)·2^-24·A` of `A`
    (`sum_common`, from `sum_err_covers` with depth at most `B − 1`, `B ≤ 2^14`), so the common
    factor `sc = A / S` is within `(B + 86)·2^-24` of `1`. With `div_full`, each output is finite
    and within `(2^-21 + 2^-30 + (3·2^-24 + 2^-30)·|s|)·σ + 2^-133/A` of `sc·σ` (`final_common`).
    `|s| = x_max − x_j` is the gap `log(σ_max/σ_j)` (`σ_max ≥ 1/A`, the max's own `σ`), and
    `1/A ≤ σ_max`, which gives `Post`'s form;
- `hdisj`: for programs `p ≠ q`, `p`'s output rows are disjoint from `q`'s (`progOut_disjoint`,
  by `out_rows`: distinct `(row, column)` pairs are distinct elements, `outIdx_inj`) and from `q`'s
  input rows (`y_x`).
- `hRW`: program `p`'s output rows are disjoint from its input rows (`y_x`).
- `hpost`: row `r < n_rows` belongs to program `r mod num_programs` (positive by `grid_pos`),
  whose memory has it done; the final memory agrees with that program's memory on its footprint,
  so `rowDone_of_agree` gives the row in `fin`. A byte outside the output elements is in no write
  footprint, so it keeps its initial value.

`consistent`: `TVA.exists_total_faithful` with `Platform.sm90ref` gives a total behaviour
satisfying `faithful`, and `Platform.sm90ref` satisfies the other five
(`Platform.ref.mathTotal`, `Platform.sm90ref.expF32`, `Platform.sm90ref.expF32NegInf`,
`Platform.ref.divTotal`, `Platform.ref.divFull`).

`satisfiable`: the source's own test instance (lines 186–188), `1823 × 781`, contiguous
(strides `781`), `BLOCK_SIZE = 1024`, `num_stages = 4`, `132` programs, with `x` and `y` of
`1823 · 781` elements at the addresses `2^24` and `2^25` (`testInst`), in a memory where every byte
is present and `0` (`testMem`: every input reads `+0`, finite), meets `Inst.ok` and `Pre`.

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float32` only.** The kernel's source is dtype-generic, but its host code uses `float32`
  (`torch.randn` without a dtype, and `dtype=torch.float32`). Other dtypes would need other
  facts (and conversions) for `tl.exp`, the reductions and the division.
- **`BLOCK_SIZE ≤ 2^14`.** The host's largest row has `12672` columns (`BLOCK_SIZE = 16384`). The
  common factor's tolerance grows with `BLOCK_SIZE` (the sum's tree depth), and the error analysis
  is carried out up to `2^14`; larger rows are not covered. The tolerance is stated at
  `BLOCK_SIZE`, not `n_cols`: the masked lanes add `+0` to the sum, which is exact, so a bound in
  `n_cols` may well hold, but the library's tree bound counts every leaf; the host's
  `next_power_of_2(n_cols)` keeps the two within a factor `2`.
- **Inputs finite of magnitude at most `2^126`, or `−∞`, with a finite element in each row.** The
  host's inputs are `torch.randn` samples. A `+∞` or NaN input makes the row's exact softmax
  undefined (the kernel gives NaN there); a row of only `−∞` gives `0/0`. The bound `2^126` keeps
  `x − max` from overflowing in the error analysis (finite inputs above it are not covered). `−∞`
  is covered: it contributes `0`, as the padding lanes do.
- **`n_rows + num_programs ≤ 2^31`, strides and row offsets in `int32`.** Triton passes the
  integers as `int32` only below `2^31` (as `int64` from `2^31` on, another model); the bound on
  `n_rows + num_programs` keeps the `int32` loop's last increment from overflowing (outside
  `tl.RangeOk` the loop is undefined behaviour), and the bounds on `(n_rows − 1) · stride` keep
  `row_idx * stride` from wrapping. The host's largest is `4096` rows of `12672` columns
  (`4095 · 12672 < 2^26`).
- **A program when there is a row (`grid_pos`).** The host's `num_programs` is
  `min(NUM_SM · occupancy, n_rows)`; it is positive for `n_rows > 0` whenever `occupancy ≥ 1`,
  that is, whenever the compiled kernel's registers and shared memory fit on one SM (with
  `num_warps = 8`, at most 255 registers per thread and `NUM_REGS = 65536`,
  `NUM_REGS // (n_regs · 32 · 8) ≥ 1`). With no program and a row, nothing would be written, so the
  condition is needed.
- **Output rows do not overlap (`out_rows`).** The host's `y = torch.empty_like(x)` has
  `stride(0) = n_cols`. Overlapping output rows would make programs (or trips) write the same
  bytes.
- **No overlap of the output with the input (`y_x`).** The host always allocates a fresh `y`. An
  in-place call (`y` the same tensor as `x`) is out of scope: each program would load a row and then
  store to the same bytes, a conflict within one program, which the library does not order and
  which no claim covers ([SEMANTICS.md](../../docs/SEMANTICS.md), "Programs";
  [exclusions.md](../../docs/exclusions.md); library issue L15). Overlap outside the elements the
  launch accesses (padding between rows) is allowed: the conditions are stated on those elements.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions): among it, point 7's target and host
  (an H200, driven from an x86-64 host running CPython 3.13 without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment). The script's `DEVICE` is
  whatever device is active, and its `is_hip()` branch is for AMD devices, outside the target.

Modelling decisions:

- `BLOCK_SIZE` ranges over every power of two up to `2^14` with `n_cols ≤ BLOCK_SIZE`, not only
  `next_power_of_2(n_cols)`; `num_stages` over every natural number (it does not affect the model),
  and `num_programs` over every grid size, not only the host's formula.
- The four integers are `int32` (`IntV.ofNat`): Triton passes an integer argument in
  `[-2^31, 2^31)` other than `1` as an `int32`, and at `1` the constant `1`, which gives the same
  model (Conditions). An integer that is a multiple of 16 (`0` included) is also marked divisible
  by 16 for the compiler, which is true of it (SEMANTICS.md, "Scalar arguments"). The code a launch
  runs is the code Triton 3.8.0 compiles for that launch's own argument values (`Compiler.Faithful`'s
  point 11, in part (B), assumed), and the launch through the `CompiledKernel` is in the warm-up's
  specialisation (point 6, above).
- `tl.range(row_start, n_rows, row_step, num_stages=num_stages)` is `tl.forRange` with no carried
  state (`()`); the scheduling keyword `num_stages` is dropped: pipelining changes no outcome of a
  well-behaved model (SEMANTICS.md, "Loops"), and the proof shows that the model has no conflict
  within a program (each trip reads only input rows and writes only its own output row).
- `row_idx * input_row_stride` (and the output one) multiplies two `int32` values, wrapping, which
  does not happen on covered instances; the pointer `+` advances by elements.
- The load has the source's `mask` and `other=-float('inf')`, the `float32` `−∞`
  (`FloatV.litF32` of the double `0xFFF0000000000000`, SEMANTICS.md, "Python's builtins"), so the
  masked lanes are `−∞`; the store has the same `mask`.
- `tl.max(row, axis=0)` of the 1-D block is `tl.maxAll true` (along a tree Triton chooses, with
  `maximum`); `row - max` broadcasts the scalar and is the IEEE `tl.sub`; `tl.exp` is the
  platform's `exp`; `tl.sum(numerator, axis=0)` is `tl.sum` (IEEE additions along a tree Triton
  chooses: its input is `tl.exp`'s result, not a product, so no contraction applies); `/` is the
  platform's `tl.div` with the scalar broadcast.
- The launch lists one block shape, `[BLOCK_SIZE]`, the shape of every block value of the source
  (`col_offsets`, `input_ptrs`, `mask`, `row`, `row_minus_max`, `numerator`, `softmax_output`,
  `output_ptrs`); `enable_fp_fusion` is Triton's default `true`, which `warmup` does not change.
