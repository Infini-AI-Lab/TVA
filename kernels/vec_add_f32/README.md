# vec_add_f32

The prover's write-up. The reviewers judge [claim.md](claim.md) (generated); the review is
[REVIEW.md](REVIEW.md) (generated).

Source: [`source/01-vector-add.py`](source/01-vector-add.py). Lean: [`lean/`](lean/), one module
`VecAdd` (namespace `Kernels.VecAdd`).

The kernel is `add_kernel` (source lines 29–54). It reads two module-level names, `triton` (its
decorator) and `tl`, bound by the imports of lines 23–24, which are kernel lines too and are mapped
(dropped: they add no step). The host function `add` (lines 62–78) allocates
`output = torch.empty_like(x)`, sets `n_elements = output.numel()`, and launches a 1-D grid of
`triton.cdiv(n_elements, BLOCK_SIZE)` programs with `BLOCK_SIZE=1024`. The script calls it on
`float32` vectors of 98432 elements (lines 84–93) and, in the benchmark, of `2^12 … 2^27`
elements (lines 107–129, run at line 135). The other module-level lines, `import torch` (21) and
`DEVICE = …` (26), are host lines: the kernel names neither `torch` nor `DEVICE`.

## Specification

`Post d init fin` says two things.

- **The outputs.** For every `j < n`, element `j` of `x` and of `y` can be read in the initial
  memory (values `a`, `b`), element `j` of `output` can be read in the final memory (value `c`),
  and `c` is the IEEE `float32` sum `a + b`, rounded to nearest even (`FloatV.add`), up to NaN
  payload (`FloatV.NaNEq`: equal, or both NaN). There is no tolerance: the result is exact.
- **Nothing else changes.** Every byte outside the first `n` elements of `output` is the same
  in `fin` as in `init`. That covers the inputs, the elements of `output` from `n` on, and every
  other byte the launch is given.

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every launch
that runs ends normally, without fault or hang, with a final memory satisfying `Post`.

## Conditions

An instance `Inst` is the host's choice: `n` (`n_elements`), `BLOCK_SIZE`, and three
contiguous `float32` buffers `x`, `y`, `output` (`Buf .f32`: a base address and a length in
elements).

`Inst.ok d` (a structure, one field per condition):

| Field | Condition | Why |
|---|---|---|
| `block_pow2`, `block_le` | `BLOCK_SIZE` is a power of two, at most `2^20` | the block sizes Triton accepts for `tl.arange`; `Launch.Valid` needs them. The host's `1024` is one |
| `n_lt` | `n < 2^31` | Triton passes `n_elements` as an `int32` only below `2^31` (as an `int64` from `2^31` on); below it the `int32` offsets `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` of every program of the grid do not wrap (`Block.fits_of_grid_pow2`). See Narrowings |
| `x_len`, `y_len`, `output_len` | each buffer has at least `n` elements | the kernel accesses elements `0 … n−1` of each |
| `output_x`, `output_y` | the first `n` elements of `output` share no byte with the first `n` elements of `x` (resp. `y`) | a program that reads and writes the same bytes, or two programs that touch bytes one of them writes, are not well behaved (see Narrowings) |

`n` is the argument `n_elements`, a runtime argument at every `n` but `n = 1`, where Triton passes
it as the compile-time constant `1` (below). Its value is part of the instance, and the conditions
constrain it: `n < 2^31` (`n_lt`) and `n ≤ x.len`, `n ≤ y.len`, `n ≤ output.len` (`x_len`,
`y_len`, `output_len`), as the user's decision of 2026-10-29 on library issue L117 asks ("for the
viriable n_elements, its value should be merged into the assumptions of the prover. e.g., x.len >=
n_elements, etc."). `Compiler.Faithful`'s point 11 assumes that the launch runs the code Triton
compiles for these values (Modelling decisions), so these conditions, not Triton's caches, decide
which values of `n` the claim covers.

`x` and `y` may overlap each other in any way (both are only read), and may be the same tensor
(`add(x, x)`). `n = 0` is covered: the grid is empty and memory is unchanged. `n = 1` is covered:
Triton then passes `n_elements` as the compile-time constant `1`, which the comparison of line 47
makes an `int32` tensor, so the model is the same (`model`'s docstring).

`Pre d init`: the three buffers are valid in the initial memory (`Buf.Valid`: base address
non-negative and aligned to 4 bytes, the buffer inside `[0, 2^64)`, every byte present). Every
`float32` value is allowed in the inputs, NaN and infinities included.

**The kernel and its launches are within part (A) of `Compiler.Faithful`'s scope**
([rule page](../../docs/rules/compiler-faithful.md), "Scope" (A) and the checklist of "How a
bundle cites it"). Every launch the script makes is a direct launch of the `@triton.jit` function,
the JIT launch `add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)` of `add` (line 75), called at line 89
and, in the benchmark, through `triton.testing.do_bench(lambda: add(x, y), …)` (line 127). Point by
point:

- **4 (options).** The launch gives no option, only the constexpr `BLOCK_SIZE`: no
  `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl`, `num_warps`, `num_stages` or any other
  keyword. Nothing calls `add_kernel.warmup(…)` or `triton.compile`, there is no `@triton.autotune`
  (so no `triton.Config`) and no `@triton.heuristics`, and the decorator is a bare `@triton.jit`, so
  its `debug` is not set. So `launch` keeps every default: `enable_fp_fusion = true`,
  `debug = false`, `num_ctas = 1`, `launch_pdl = False`.
- **5 (descriptor scratch).** The kernel builds no tensor descriptor (no
  `tl.make_tensor_descriptor`), so no allocator is involved.
- **6 (precompiled kernel).** No `CompiledKernel` is launched: every launch is the JIT launch above.
- **8 (host descriptors).** The launch is given no host `TensorDescriptor`: its arguments are
  three tensors, an integer and the constexpr.
- **9 (CUDA graphs).** No call is made during a CUDA stream capture, and nothing is replayed: the
  script calls `add` directly (line 89) and through `triton.testing.do_bench` (line 127), which
  times plain calls of its function with CUDA events, not through `triton.testing.do_bench_cudagraph`;
  the script opens no `torch.cuda.graph` and makes no capture of its own.

The kernel has no `@triton.autotune` (a source with it is outside the supported subset, and the
rule covers no call of an autotuned kernel: rule page, "Which launch"), so a call of `add` makes
this one direct launch, which the claim covers, from the memory at that launch.

**Part (B) of the scope, the environment** (points 1, 2, 3, the environment's part of 4, 7, 10 and
11), is an assumption of the rule: the claim relies on it, and TVA does not check it.

**The supported subset** ([SEMANTICS.md](../../docs/SEMANTICS.md), "The supported subset"). The
kernel is a top-level `def` under a bare `@triton.jit`; its parameters are three pointers (tensor
arguments), one unannotated integer (an `int` argument) and one `tl.constexpr` with an `int`
argument, none with a default; its statements are assignments `name = e` and one call; its
expressions are `tl.program_id(axis=0)`, `tl.arange`, `tl.load(…, mask=…)`, `tl.store(…, mask=…)`,
`*`, `+` and one `<`. The `*` of line 44 multiplies the `int32` program id by the constexpr
`BLOCK_SIZE`; the `+` of line 45 adds that `int32` scalar to the `int32` block `tl.arange(…)`; the
`<` of line 47 compares the `int32` block `offsets` with `n_elements`; the `+` of lines 50, 51 and
54 adds the `int32` block `offsets` to a `float32` pointer (`*fp32`), which advances the pointer by
each lane's offset, in elements ([SEMANTICS.md](../../docs/SEMANTICS.md), "Values": `p + i`
advances by `i * sizeof(d)`); and the `+` of line 52 adds two `float32` blocks. The only module-level names it reads are
`triton` (its decorator) and `tl`, Triton's modules, each bound only by its own `import` (lines 23
and 24): no other statement of the module binds `triton`, `tl` or `add_kernel` (the others bind
`torch`, `DEVICE`, `add`, `size`, `x`, `y`, `output_torch`, `output_triton` and `benchmark`, and
`add`'s annotations and `benchmark`'s decorator bind nothing).

## Assumptions

One assumption, of kind `[rule]`:

- `faithful` — `Rules.Compiler.Faithful P (model d) (launch d) (K d)` for every covered `d`: if
  `launch d` has `debug = false` and is valid (`Launch.Valid`), then from every initial memory from
  which the model's launch is well behaved (no undefined behaviour, no race between programs, no
  conflict within a program), every outcome of the compiled kernel launched as `launch d` is one
  of the model's: a normal end with a memory some interleaving of the model leaves, never a fault,
  and a hang only if some interleaving runs forever. The rule covers a launch only within its
  scope, which its text states: part (A), which the launches meet (Conditions), and part (B), the
  environment, which the claim relies on. It is cited as the library states it
  ([rule page](../../docs/rules/compiler-faithful.md)).

The kernel uses no approximate operation (its only float operation is an IEEE `+`), so it needs
no fact about the platform `P`, and there is no `[kernel]` assumption.

## Proof

`main` applies `TVA.meets_independent`:

- `hdebug`: `debug = false` (Triton's default) by definition of `launch`.
- `hvalid` (`launch_valid`, by `Launch.valid_grid1`): the grid `cdiv(n, BLOCK_SIZE)` has fewer
  than `2^31` programs (`grid_lt`: it is at most `n`), and the one block shape `[BLOCK_SIZE]` is a
  power of two of at most `2^20` elements; `grid1` keeps `num_ctas = 1` and `launch_pdl = False`.
- Footprints: program `p` reads its tiles of `x` and `y` (`R`) and writes its tile of `output`
  (`W`); a tile is the elements `p·BLOCK_SIZE + l` below `n` (`Block.Tile`, `Buf.tile`).
- `hsafe` (`safe_program`): `body_eq` (by `rfl`) puts program `p`'s body in the form of the
  library's 1-D block pattern, with `program_id` equal to `p` (`launch_pid`). Then one rule per
  line: `safe_load_block` for the loads of lines 50 and 51 (each enabled lane holds its element of
  the initial memory), `safe_add` for line 52 (the IEEE sum up to NaN payload), `safe_last` and
  `safe_store_block` for the store of line 54 (the tile holds the sums), and `safe_ret`. `fits`
  (by `Block.fits_of_grid_pow2`, from `n < 2^31` and the power-of-two `BLOCK_SIZE`) gives the
  `Block.Fits` each block rule needs.
- `hdisj`: for programs `i ≠ j`, `output`'s tile `i` is disjoint from tile `j` of `x` and `y`
  (`tile_disjoint_in`, from `output_x` and `output_y`: tiles lie in the first `n` elements) and of
  `output` (`Buf.tile_disjoint`).
- `hRW`: program `p`'s tile of `output` is disjoint from its tiles of `x` and `y`
  (`tile_disjoint_in` again).
- `hpost`: element `j < n` is in the tile of program `j / BLOCK_SIZE` (`Block.tile_div`), which
  is in the grid (`div_lt_grid`); the final memory agrees with that program's memory on its
  footprint, so `Mem.read_congr` gives the value. A byte outside `output`'s first `n` elements
  is in no write footprint, so it keeps its initial value.

`consistent`: `TVA.exists_total_faithful` with `Platform.any` gives a total behaviour that
satisfies `faithful`.

`satisfiable`: the source's own test instance (lines 85–89), `n = 98432` and `BLOCK_SIZE =
1024`, with three buffers of `98432` elements at the 16-byte-aligned addresses `2^21`, `2·2^21`
and `3·2^21` (`testInst`), in a memory where every byte is present (`testMem`), meets `Inst.ok`
and `Pre`.

`main`, `consistent` and `satisfiable` use only the axioms `propext`, `Classical.choice` and
`Quot.sound` (the gate checks it).

## Narrowings and modelling decisions

Narrowings, each against the kernel's real use:

- **`float32` only.** The kernel's source is dtype-generic, but its host code and benchmark use
  `float32` (`torch.rand` without a dtype, and `dtype=torch.float32`). Other dtypes, and mixed
  dtypes (which would add promotions), are not covered.
- **`n < 2^31`.** The script's largest size is `2^27`. From `2^31` on, Triton passes
  `n_elements` as an `int64`, which would need another model (an `int64` `n_elements` and the
  `int64` comparison `offsets < n_elements`). Beyond `2^31` the kernel itself is wrong: every
  covered `BLOCK_SIZE` divides `2^31`, so the last program has `pid ≥ 2^31 / BLOCK_SIZE`, and its
  `int32` product `pid * BLOCK_SIZE` wraps; its enabled lanes then access memory before the
  buffers. At `n = 2^31` exactly nothing wraps, but `n_elements` is an `int64`, so it is excluded
  with the rest. Below `2^31` every size is covered.
- **Contiguous buffers.** Each tensor is a contiguous `Buf`, as `torch.rand` and
  `torch.empty_like` of a contiguous tensor give. The kernel ignores strides, so it is not meant
  for strided tensors.
- **`y` has at least `n` elements.** The host takes `n` from `output` (that is, from `x`) and
  does not check `y`'s size; a shorter `y` would be read out of bounds, so the condition is
  needed.
- **No overlap of `output` with an input** (within the first `n` elements). The host always
  allocates a fresh `output`, so its use is covered. An in-place call (`output` the same tensor as
  `x` or `y`) is out of scope: each program would load its tile and then store to the same bytes,
  a conflict within one program, which the library does not order and which no claim covers
  ([SEMANTICS.md](../../docs/SEMANTICS.md), "Programs"; [exclusions.md](../../docs/exclusions.md);
  library issue L15). A partial overlap would also make programs race. An overlap only in elements
  from `n` on is allowed: the conditions are stated on the first `n` elements.
- **The environment.** The claim covers only launches within part (B) of `Compiler.Faithful`'s
  scope, which it relies on and nobody checks (Conditions): among it, point 7's target and host (an
  H200, driven from an x86-64 host running CPython 3.13 without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment). The script's `DEVICE` is whatever device is active.

Modelling decisions:

- `BLOCK_SIZE` ranges over every block size Triton accepts, not only the host's `1024`.
- `n_elements` is an `int32` (`IntV.ofNat n`): Triton passes an integer argument in
  `[-2^31, 2^31)` other than `1` as an `int32`, and at `n = 1` the constant `1`, which gives the
  same model (above). An `n` that is a multiple of 16 (`0` included) is also marked divisible by 16
  for the compiler, which is true of it (SEMANTICS.md, "Scalar arguments"). The code a launch runs
  is the code Triton 3.8.0 compiles for that launch, from the kernel's own source, the launch's own
  argument values, compile-time (`BLOCK_SIZE`, and `n_elements` at `n = 1`) and runtime (the three
  pointers, and `n_elements` at every other `n`), as the instance gives them, and the settings of the scope's other
  points (`Compiler.Faithful`'s point 11, in part (B), assumed); code a cache returns counts when it
  is that code, for example for a repeated launch of the same instance. Under point 11, only a
  launch of a `CompiledKernel` runs code that is not the code compiled for its own arguments
  (point 6), and there is none; this is the assumption, not a statement about Triton's caches (the
  script's later launches run code the JIT compiled at its first launch).
- `pid * BLOCK_SIZE` multiplies two `int32` values: the constexpr, an operand of `*` with an
  `int32` tensor, is built in `int32`. `block_start + tl.arange(…)` adds the scalar to every lane.
- The loads have the source's `mask` and no `other`, so disabled lanes are unspecified; the
  proof uses only the enabled lanes, and the store has the same `mask`.
- `x + y` is `tl.add` (IEEE): its operands are loads, not products, so no contraction applies.
- The launch lists one block shape, `[BLOCK_SIZE]`, the shape of every block value of the
  source (`tl.arange`, `offsets`, `mask`, `x`, `y`, `output`); `enable_fp_fusion` is written as
  Triton's default `true` (it does not matter here).
