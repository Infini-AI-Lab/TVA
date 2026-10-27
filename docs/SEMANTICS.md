# Semantics

This page says what each construct means and points to the Lean definition. Reviewers judge
the semantics once, at library level. A bundle's reviewers judge only whether the model uses
the constructs faithfully.

## Values (`TVA/Lang/Scalar.lean`, `Tensor.lean`)

| Triton | Lean | Meaning |
|---|---|---|
| `tl.int8 … tl.int64`, `tl.uint8 … tl.uint64` | `IntV sg w` (`I32`, …) | `w`-bit two's complement; `+ - *` wrap; `//`, `%` and shifts return `Option` (`none` = undefined) |
| `tl.int1` | `Bool` | |
| `tl.float16/bfloat16/float32/float64` | `FloatV f` (`F16`, `BF16`, `F32`, `F64`) | IEEE bit patterns; `+ - * fma div_rn sqrt_rn` round to nearest even |
| `tl.float8e5`, `tl.float8e4nv` | `F8E5`, `F8E4` | conversions into them are platform relations (`tl.cvt8`); out of them exact on finite values (`tl.cvtFrom8`) |
| a block `tensor[s]` | `Tensor s α` | a function from indices `Idx s` to values |
| a pointer `*T` | `Ptr d` | a byte address; `p + i` advances by `i * sizeof(d)` |

`tl.constexpr` values, including shapes, are Lean values: the model is a Lean function of
them.

## Programs (`TVA/Lang/Prog.lean`, `TVA/Sem/Step.lean`)

A model is a `Prog`, written in `do` notation. Its constructors, and their steps:

| Constructor | Step |
|---|---|
| `load d s addrs k` | Each enabled lane (`addrs i = some a`) reads its element; disabled lanes get any value. If an enabled address is unreadable (unaligned, or a byte is absent): undefined behaviour. Event: reads the enabled bytes. |
| `store d s addrs vals k` | The enabled lanes are written in some order, so on a duplicate address the later lane wins. An unwritable address is undefined behaviour. Event: writes the enabled bytes. |
| `rmw d a upd sem scope k` | One atomic read-modify-write: the old value `o`, any new `n` with `upd o n`, returns `o`. If no `n` exists, or the address is unreadable: undefined behaviour. |
| `pick P k` | Continue with any `b` satisfying `P b`; if there is none, undefined behaviour. |
| `ub` | Undefined behaviour. |
| `loop init body k` | Unfold one iteration (`Prog.loopStep`). |
| `env k` | Read the environment (platform `P`, `fusion`). |

A program is **sequential**: each step sees the memory left by the program's earlier steps. The
threads that implement a program are not part of the language (see "No barrier semantics"
below).

## The `tl.*` operations (`TVA/Lang/Ops.lean`)

Each definition's docstring states its reading of the documentation. The non-obvious ones:

- **Float results are exact up to NaN payload.** The device returns some NaN, not a particular
  one: PTX `neg`/`abs` give "an unspecified NaN", and MLIR folds such as `x * 1.0 → x` keep the
  input's payload. So every exact float operation (`tl.add`, `sub`, `mul`, `fma`, `div_rn`,
  `sqrt_rn`, `neg`, `abs`, `cvt`) is a `Prog` whose result is `FloatV.NaNEq` to the IEEE
  result. Specifications compare outputs up to NaN payload.
- **`tl.load(ptrs, mask, other)`.** Disabled lanes are not accessed. They hold `other` when it
  is given, and an unspecified value otherwise (S6: 3.8.0 happens to give `0`; the library
  does not promise it). Cache, eviction and `volatile` hints do not change results in a
  race-free launch.
- **Constants.** A literal equal to zero, `-0.0` included, compiles to `+0.0`
  (`scalar_constant`), in `tl.full` and as a Python scalar operand. The model writes `+0.0`.
- **`tl.store(ptrs, vals, mask)`.** The model writes the implicit conversion to the pointer's
  dtype explicitly.
- **Atomics.** A tensor atomic is one `rmw` per enabled lane, in an order the program does not
  control. Each lane returns its element's old value; disabled lanes return unspecified values.
  - Integer `atomic_add` wraps. Float `atomic_add` is `P.atomicAdd`.
  - `atomic_max`/`atomic_min` compare integers by signedness. For floats, the result is one of
    the two values and, if neither is NaN, not below (above) either.
  - `atomic_and/or/xor` are integer-only. `atomic_cas` compares bit patterns.
  - The defaults are `sem = acq_rel` and `scope = gpu`.
- **Float `+` and `-`.**
  - `tl.add` is IEEE, for additions no product feeds.
  - `tl.mul_add a b c` is either `a*b + c` (two roundings) or `fma`.
  - `tl.addc` is any result `AddRel` allows: the IEEE sum, or, with fusion, the rounding of
    `p + q` for some exact `p`, `q` rounding to the operands.
  - Contraction is the only float rewrite the backend may apply: no reassociation, no
    reciprocal division, no flushing of subnormals ([review-questions.md](review-questions.md)
    S5). The combine pass's rewrites are below (`dot_add`, `sum_bcast_mul`).
- **`/` on floats** is `P.div` (the default `tl.fdiv` is not IEEE). `tl.div_rn` is IEEE. Its
  accuracy is a rule: `Rules.Num.DivFull` for fp32 (`div.full.f32`, 2 ulp) and
  `Rules.Num.DivRn` for fp64 (IEEE) ([rules/num.md](rules/num.md)).
- **`tl.exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `sqrt`, `rsqrt`, `erf`** are
  `P.math fn`. `tl.sqrt_rn` is IEEE. Their accuracy is `Rules.Num.MathAcc` (finite inputs in
  the function's natural domain, with room in the format) and their special values
  `Rules.Num.MathAt` (`exp(-∞) = +0`).
- **`tl.maximum`, `tl.minimum`** follow `FloatV.MaxRel`: the reference `max`/`min` up to NaN
  payload, or either operand when both are zeros (LLVM `maxnum`). Triton converts bf16
  operands to fp32 first, and the model writes that conversion.
- **Float conversions.**
  - Between the standard formats (`tl.cvt x rnd`): one rounding, to nearest even by default,
    or toward zero with `fp_downcast_rounding="rtz"`, which compiles only from fp32 to
    fp16/bf16. Widening is exact.
  - Into fp8 (`tl.cvt8 x rnd`, from fp16/bf16/fp32): the platform's `P.cvt f g rnd`.
  - Out of fp8 (`tl.cvtFrom8`, to fp16/bf16/fp32): exact on finite values. An infinity or NaN
    may come out as anything, since `float8e5 → bfloat16` is a bit trick that turns them into
    finite numbers. There is no fp8 → fp64 conversion.
  - `fp_downcast_rounding="rtz"` into fp8 compiles only from fp32 to `float8e5`.
  - Any other pair is undefined behaviour in the model.
- Triton's dtype promotions are written in the model:
  - `/` and `%` on fp16/bf16 run in fp32;
  - `tl.max`/`tl.min` reductions (without `return_indices`) widen floats narrower than 32 bits to
    fp32, and integers narrower than 32 bits to `int32` whatever their signedness: `int8`/`int16`
    sign-extended, `uint8`/`uint16` zero-extended, `int1` as `0`/`1`. So `tl.max` of an `int8`
    tensor is an `int32`, and `tl.max(x) + 100` does not wrap at 8 bits;
  - `tl.sum` widens integers narrower than 32 bits to `int32` (signed) or `uint32` (unsigned,
    `int1` included), and floats not at all;
  - `tl.max`/`tl.min` with `return_indices`, `tl.argmax` and `tl.argmin` widen only `bfloat16`
    (to fp32).
  ([evidence/int-reductions](../evidence/int-reductions/README.md))
- **`x.to(int)`** truncates. A NaN, infinite or out-of-range element is undefined behaviour
  (LLVM poison, read conservatively).
- **`x.to(float)`** from an int (`tl.itof`) rounds to nearest even once. For fp16/bf16
  targets it may instead round twice, through fp32 or fp64 (`tl.ItofRel`). Before sm_90 there
  is no direct integer-to-bf16 conversion. Any narrowing of an int-to-float result is one
  `tl.itof` to the final format, since MLIR folds `truncf(sitofp x)` into `sitofp x` with no
  one-use condition. That covers explicit chains (`x.to(f64).to(f32)`) and the implicit
  conversions of `tl.store`, the atomics and `desc.store` (storing `offs.to(f32)` into a bf16
  buffer). `tl.itof` and `x.to(int)` exist only for the standard float formats.
- **`tl.sum`, `tl.max`, `tl.min` (reductions)** combine along a `Tree` Triton chooses.
  - `tl.sum` (`sum1`, `sum0`) is for inputs that are not products, and every node is an IEEE
    addition.
  - `tl.sumc` (`sum1c`, `sum0c`) is for a sum of products, and its nodes are `AddRel` with the
    launch's fusion flag.
  - `tl.max1 isMax` (`axis=1`), `tl.max0 isMax` (`axis=0`) and `tl.maxAll isMax` (a whole 1-D
    tensor) combine with `maximum` (`isMax = true`) or `minimum`.
  - Integer reductions do not depend on the order: `Tensor.foldAll`, `foldRows`, `foldCols` with
    wrapping `+`, `IntV.max`, `IntV.min`, on the widened values (above).
- **Index reductions.** `tl.max(x, axis, return_indices=True)` and `tl.min(…)` give the values and
  their `int32` indices; `tl.argmax`/`tl.argmin` give the indices. They are `tl.argmaxAll`,
  `tl.argmax1`, `tl.argmax0` (`isMax`, `tieLeft`): the pairs `(x[j], j)` combined along a `Tree`
  Triton chooses by `tl.argCombine`, Triton's own combine function. A pair wins when its value is
  greater (`isMax`; less for `min`), or, with `tie_break_left` (`tieLeft`, the default), when the
  values are `==` and its index is smaller. The values compare with Triton's `<` and `==` on the
  dtype (`CmpKey`): signed or unsigned integers, ordered floats. So without NaN and with
  `tie_break_left`, the result is the first index of the maximum; with `tie_break_left=False` any
  maximal index may come out; a NaN compares false, and then the result depends on the tree
  ([evidence/argmax](../evidence/argmax/README.md)).
- **`tl.dot(a, b, acc)`** is `P.dot cfg`. A source `tl.dot(a, b, acc=z) + c` (either
  order), with `z` a constant zero and no other use of the product, is `tl.dot_add cfg a b c z`.
  Triton's combine pass may turn it into `tl.dot(a, b, c)`, so either form is allowed (S7).
  `tl.dot_scaled_add` is the same for `tl.dot_scaled`.
- **`tl.sum(x[:, :, None] * y[None, :, :], axis=1)`** of floats is `tl.sum_bcast_mul`: the
  products summed along a tree with fusable additions, or, when the additions are at most 32
  bits and `M, N ≥ 16`, Triton's rewrite to an IEEE-precision `tl.dot` (S7).
- **Tensor descriptors (TMA)** (`TVA.Lang.Desc`): `TensorDesc` holds the base, shape
  (`int32`), strides, the block shape fixed when it is made, and padding (`"zero"` or `"nan"`).
  Every access has that block. Lane `i` of a block at `offsets` is element `offsets + i`.
  - `tl.desc_load D offsets`: lanes inside the tensor read memory; the others hold the padding.
  - `tl.desc_store`: writes the in-bounds lanes and drops the others.
  - `tl.desc_atomic op` (`atomic_add/min/max/and/or/xor`): one relaxed, `gpu`-scope reduction
    per in-bounds lane, no result. Two programs reducing into the same element race.
  - `tl.desc_gather D xs y` / `tl.desc_scatter`: rows `xs`, columns `y + j` of a 2-D tensor
    with block `[1, BN]`.
    Out-of-bounds gather lanes are unspecified; an out-of-bounds scatter lane is undefined
    behaviour (neither is documented).
  - Undefined behaviour unless Triton's requirements hold (`TensorDesc.Access`): rank 1–5; a
    16-byte-aligned base; leading strides that are multiples of 16 bytes and a last stride of
    1; an innermost offset that is a multiple of 16 bytes (documented for load and store,
    required of every access here, a conservative reading); dtypes Triton allows for
    reductions (`RedOp.Allowed`).

  TMA's asynchrony is the compiler's, and we trust it. Pointer and descriptor accesses to
  the same bytes are out of scope ([review-questions.md](review-questions.md) Q11).
- **Block pointers** (`TVA.Lang.BlockPtr`), as Triton 3.8.0's frontend lowers them:
  - lane `i` has coordinate `offsets[k] + i_k` (`int64`, wrapping) and pointer
    `base + Σₖ coordₖ · strides[k]`;
  - `tl.load_block p bc pad` is a masked `tl.load`: a lane outside `[0, shape[k])` on a
    checked dimension `k ∈ bc` holds the padding (`undef`, `zero` or `nan`); unchecked
    dimensions are not checked;
  - `tl.store_block` is the masked `tl.store`; `tl.advance` adds to the offsets.
- **`tl.dot_scaled`** (`TVA.Lang.Scaled`) is `P.dotScaled cfg`, applied to the operands'
  element bits (unpacked exactly from fp4/fp8/bf16/fp16 storage) and each element's scale bits.
  Each side's scales are `e8m0` (`uint8`/`int8` tensors) or a float dtype (`cfg.lhsScale`,
  `cfg.rhsScale`, e.g. NVFP4's `float8e4nv`), one per group of `16` or `32` elements along `K`
  (S9). The storage dtypes are Triton's (`ScaledCfg.TypesOk`), and only `e2m1` may be packed
  along a dimension other than `K`. `scaledValue` gives the exact values for accuracy rules.
- **Shape changes** are exact: `Tensor.join`, `split`, `interleave` (proved equal to Triton's
  `join` + `reshape`), `reshape` (row-major, `can_reorder=False`), `trans` (rank 2) and
  `permute` (any rank: dimension `k` of the result is dimension `dims[k]` of the input, as
  `tl.permute(x, dims)` and `tl.trans(x, dims)`; [evidence/permute](../evidence/permute/README.md)).
  `tl.reshape_any` is `can_reorder=True`: some order of the same elements. A reshape to rank 1
  whose only users are reductions or histograms is also `tl.reshape_any`, since the combine pass
  allows it to reorder (S7). An index reduction (`tl.argmax`) of such a reshape then pairs
  reordered values with unreordered indices, as on the device.
- **Scans** (`TVA.Lang.Scan`): `tl.associative_scan x axis op reverse` gives position `j`
  as `op` evaluated along a tree whose leaves are `0, …, j` in order (`reverse`: `n-1, …, j`
  in that order, as Triton lowers it as `flip(scan(flip(x)))`; S8). `op` is the model's
  reading of `combine_fn`. `tl.cumsum`/`cumsumc` (float `+`, IEEE or fusable), `tl.cumprod`,
  `tl.icumsum`/`icumprod` (wrapping integers). Triton's promotions are written in the model:
  `tl.cumsum` widens `bfloat16` to `float32` and integers narrower than 32 bits to 32 bits
  (keeping signedness); `tl.cumprod` widens only `bfloat16`.
- **`tl.sort`** (along the last dimension): each row without NaN comes out as a permutation of
  itself up to `==` (`-0`/`+0` may be copied), sorted ascending or descending. A row with a NaN
  may come out as anything (the bitonic network duplicates on unordered pairs).
- **`tl.histogram(x, bins, mask)`**: bin `b` counts the enabled elements equal to `b`. An
  enabled element outside `[0, bins)` is undefined behaviour (undocumented).
- **Integer `//`, `%`, `<<`, `>>`** on tensors (`tl.idiv`, `irem`, `shl`, `ashr`, `lshr`)
  truncate (C semantics). Both shift operands have the promoted type. `>>` is `ashr` when the
  tensor operand Python dispatches on is signed before promotion, `lshr` otherwise. That
  operand is the left one if it is a tensor, otherwise the right (`__rrshift__`): `0xF0 >> s`
  with `s : uint8` is logical. They are undefined
  behaviour if any lane divides by zero, overflows, or shifts by an amount that, read as
  unsigned, is at least the width, even in a lane a later `tl.where` discards.
- **Compiler hints** are kept as checks (undefined behaviour if false). A hint is an attribute
  of the value and holds at every use of it, so the check sits where the value is defined,
  before any use. The exception is a hint on a block argument (a loop's induction variable or
  a loop-carried variable): Triton stores it on the loop operation as `<name>_arg<N>`
  (`triton-3.8.0-ir.cc:411-424`), which no pass reads, so it has no effect and the model drops
  it. They apply to integer and pointer tensors (`HintVal`: a pointer's
  divisibility is in bytes, its contiguity in elements):
  - `tl.assume`;
  - `tl.max_contiguous` (runs of `C` consecutive values);
  - `tl.max_constancy`;
  - `tl.multiple_of x v contig`: `v` divides the first element of every group of `contig`
    positions. `contig = 1` always gives a sound reading; a larger value must divide Triton's
    contiguity for `x`, and the reviewer checks it.
  - For rank ≥ 2, `tl.max_contiguousN`, `max_constancyN`, `multiple_ofN` take one value per
    dimension, as the source does. Along dimension `k`, a group is `C[k]` positions that
    agree on every other coordinate (`Idx.SameGroup`).
- **Loops.**
  - `for i in range(lo, hi, step)` and `tl.range` are `tl.forRange`. The induction variable
    has Triton's promoted bound type (`int32`, or `int64` when a bound is). It is undefined
    behaviour unless `tl.RangeOk`: `step > 0`, and `lo`, `step` and `hi + step − 1` within the
    *signed* range. `scf.for` compares signed even for unsigned bounds.
  - `tl.static_range` is unrolled by the frontend with a Python integer as the variable; the
    model writes the unrolled body.
  - Fixed counts are `tl.forN`.
  - `while` is `tl.whileLoop`.

- **Random numbers** (`TVA.Lang.Random`): `tl.randint`/`randint4x` are Triton's Philox 4x32-10,
  exactly (checked against the Random123 known-answer vectors). `tl.rand` is
  `uint_to_uniform_float` of it: an `tl.itof` conversion and an IEEE product with the `float32`
  constant `4.6566127342e-10`. The result is a product, and the function is inlined, so an
  addition that consumes it (`tl.rand(…) - 0.5`) may fuse: model it with `tl.addc`/`tl.subc`.

## Launches (`TVA/Sem/Launch.lean`)

- `Launch` = grid `(x, y, z)`, `enable_fp_fusion`, `debug` (default `false`), `num_ctas`
  (default `1`), `pdl` (`launch_pdl`, default `false`) and `blocks`, the source's block shapes on
  this instance.
  - `Compiler.Faithful` covers only launches compiled and run with Triton's environment at its
    defaults: none of the environment variables that change what Triton compiles or how it
    launches is set ([rules/compiler-faithful.md](rules/compiler-faithful.md) lists them). For
    example `TRITON_F32_DEFAULT` changes the `input_precision` of a `tl.dot` that does not give
    one, and `TRITON_DEFAULT_FP_FUSION` the default of `enable_fp_fusion`
    ([evidence/env-knobs](../evidence/env-knobs/README.md)).
  - `Compiler.Faithful` covers only `debug = false`, since with `debug` asserts and integer
    overflow checks trap.
  - It also covers only `Launch.Valid` launches: grid `x < 2^31` and `y, z < 2^16`, fewer than
    `2^31` programs (`x·y·z < 2^31`), `num_ctas = 1`, no programmatic dependent launch
    (`pdl = false`), and every block shape a power of two in each dimension with at most `2^20`
    elements. With `launch_pdl`, a kernel may start before the previous kernel on the stream has
    finished writing, so its programs may read memory other than the launch's `init`. Without it,
    `tl.extra.cuda.gdc_wait()` and `gdc_launch_dependents()` change nothing, and the model drops
    them ([evidence/pdl](../evidence/pdl/README.md)). Triton 3.8.0's launcher tests `x·y·z > 0` in C `int`
    and returns without an error, running no program, when the grid has `2^31` to `2^32`
    programs (`triton-3.8.0-nvidia-driver.c:968-972`;
    [evidence/launch-outcomes](../evidence/launch-outcomes/README.md)).
  - A launch reported to the host with an error before any program runs (shared memory,
    registers, threads, `tl.dot` sizes, dtypes, grid limits) has no outcome
    (`TVA.Claim.Behaviour`). A launch that returns without an error has one, even if no program
    ran: with a zero grid dimension it is `ok` with memory unchanged.

  Program `i` has `program_id = (i % gx, i / gx % gy, i / (gx·gy))`.
- A configuration is one memory plus each program's remaining `Prog`. A launch step is a step of
  any one program: an interleaving with sequentially consistent memory.
- **Race.** Two events of different programs, one writing a byte the other accesses. The only
  exception is a pair of `Event.Strong` atomics: both `acq_rel` (Triton's default), at
  `gpu`/`sys` scope, on the same bytes.
  - Relaxed, acquire-only and release-only atomics race, since an SC interleaving would hide
    their reorderings (store buffering).
  - Atomics on partly overlapping bytes race too.
  - There is no synchronises-with between programs. A scalar atomic is issued by one thread
    of the program with no barrier before it, so its release does not cover the other threads'
    writes. So handing data from one program to another through plain memory, behind a flag
    or counter, is a race; such kernels are excluded. Happens-before is program order.
- **`WellBehaved`.** No reachable configuration has a program whose next step is undefined
  behaviour, and no trace has a race.
- **`Outcomes`.**
  - `ok fin`: all programs finished with memory `fin`.
  - `ub`: some interleaving reaches undefined behaviour.
  - `hang`: some interleaving runs forever.

## No barrier semantics

The documentation gives `tl.debug_barrier()` only as "Insert a barrier to synchronize all
threads in a block". The language has no threads, and the user decided (2026-10-26): "we do not
need debug semantics here". A model never contains a barrier. A source `tl.debug_barrier()` is a
no-op in the model, and the reviewer sees it in the correspondence. Programs are sequential, so
a program that writes and then reads back its own data sees the write. Trusting Triton covers
that.

## Granularity

A tensor load or store is one step of the interleaving. Finer interleavings, such as per lane or
per thread, differ only in launches with a race between programs, and those launches have no
meaning (`WellBehaved` fails). The rule then makes no claim about them.
