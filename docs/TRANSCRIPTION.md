# Transcription rules

How a Triton source becomes a TVA model. This page is part of the library:
- the **kernel prover** follows it;
- the **kernel reviewer** checks the model against the source by it, and trusts that each rule is
  what Triton 3.8.0 does;
- **library review** checks that ([LIBRARY_REVIEW.md](LIBRARY_REVIEW.md)).

What each `tl.*` operation means is in [SEMANTICS.md](SEMANTICS.md).

A construct this page does not cover cannot be transcribed yet. The prover asks the library for
a rule, and a kernel reviewer who finds one files `UNSUPPORTED`. A rule marked **(open: L<n>)** is
under a library issue ([library-review/ISSUES.md](library-review/ISSUES.md)): a kernel that needs
it waits for the issue's resolution.

**Rule identities.** Each rule below starts with its identity in brackets, such as `[mask]`.
Every entry of a kernel's model map names the rules it applies (`"rules": ["mask", "call"]`), and a
dropped line names its drop reason, which is a rule too ([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md)
§7). A rule's text may name other rules, and then they apply as well. The gate folds the text of
every rule a map names into the kernel's `map` and `model` digests, so a change to a rule re-opens
exactly the kernels that use it. The text outside the rules (this introduction, §1, the last
paragraph of §3, §9) and [exclusions.md](exclusions.md) apply to every model, and a change to
them re-opens every kernel's `map` and `model`. The kernel reviewer checks that each entry names
every rule it applies.

## 1. The model's shape

The kernel is a `Kernel` whose `body : Ctx → Prog Unit` is a `do` block. It keeps the source's
line structure and names, one model line per source line where possible. A `tl.constexpr`
argument is a Lean parameter (`BLOCK_SIZE : Nat`). Every operation is a `TVA.tl.*` or another
`TVA.*` definition. The model declares nothing in the `TVA` or `tl` namespaces and no instance of
a library class (`SortKey`, `CmpKey`, `HintVal`, …): such a declaration would change an
operation's meaning without changing the model's text. An operation missing from `TVA.Lang.Ops`
is not defined in the kernel: the library adds it.

- **[helper]** An inlined `@triton.jit` helper is transcribed where it is called, or as a Lean
  function the call site applies, with the call's arguments.

## 2. Values and types

- **[call]** A call of a `tl.*` function or a tensor method (`tl.load(…)`, `tl.where(…)`,
  `x.to(…)`) is the library definition [SEMANTICS.md](SEMANTICS.md) gives for it, with the
  source's arguments, each the model's value of that argument. An operator on tensors (`+`, `*`,
  `<`, `&`, …) is the definition for that operator and the operands' dtype: wrapping integer
  arithmetic and comparisons on `Tensor` (`+`, `lt`, `ltS`, …), the float operations of `tl`
  (`tl.add`, `tl.mul`, …).
- **[implicit]** Implicit conversions are written. Dtype promotion, `.to(...)`, scalar–tensor
  arithmetic and broadcasting are written out (`tl.cvt`, `bc`, `.col`/`.row`).
- **[convert]** Conversions use the operation for their formats:
  - `tl.cvt` between the standard float formats, with the source's `fp_downcast_rounding`;
  - `tl.cvt8` into fp8;
  - `tl.cvtFrom8` out of fp8;
  - `tl.itof` from integers;
  - `tl.ftoi` from floats to integers.
- **[itof-narrow]** A narrowing of an integer-to-float value is one `tl.itof` to the final format.
  That covers an explicit chain `x.to(f32).to(bf16)`, and also the implicit conversion when such
  a value is stored into a narrower buffer, atomically added, or stored through a descriptor.
- **[shift]** `>>` is `tl.ashr` when the operand Python dispatches on is signed *before*
  promotion, and `tl.lshr` otherwise. That operand is the left one when it is a tensor, and the
  right one when the left is a Python scalar. Shift amounts have the promoted type.
- **[literal]** A float zero literal is `+0.0`, even when the source writes `-0.0`. A Python float
  constant that fits `float32` is the nearest `float32` (`FloatV.ofPat`, with the bit pattern in a
  comment).
- **[index]** Index arithmetic is `int32` where the source's is (`program_id`, `arange`, scalar
  arguments below `2^31`), and `int64` where the source casts.
- **[pointer]** Element types match the buffers. A pointer built from a loaded integer
  (`.to(tl.pointer_type(…))`) is `tl.intToPtr`.
- **[float-prog]** Float operations are `Prog`s (`let y ← tl.neg x`), because a NaN result may be
  any NaN.
- **[permute]** `tl.permute(x, dims)` and `tl.trans(x, dims)` (and the method forms) are
  `x.permute dims`, with `dims` the source's list, written as a literal list (the proof that it is
  a permutation is found by itself). `tl.trans(x)` without `dims` and `x.T` on a rank-2 tensor are
  `x.trans`. `tl.trans(x)` without `dims` on a tensor of rank `r ≥ 3` swaps the last two axes:
  `x.permute [0, …, r - 3, r - 1, r - 2]`, such as `x.permute [0, 2, 1]` for rank 3.

## 3. Memory

- **[mask]** `mask` and `other` are the source's. A masked load with no `other` leaves the
  disabled lanes unspecified, and the model never uses them as `0`.
- **[broadcast]** `[:, None]`, `[None, :]` and scalar–tensor operations keep their axes.
- **[block-ptr]** `tl.make_block_ptr` takes each `shape`/`strides`/`offsets` entry cast to
  `int64`, with the source's `block_shape`, `boundary_check` and `padding_option` (the default
  padding is undefined).
- **[desc]** A `TensorDesc` has the shape, strides, `block_shape` and padding the host's
  `TensorDescriptor` gives. Building a descriptor inside the kernel (`tl.make_tensor_descriptor`)
  is **(open: L5)**.
- **[atomic]** Atomics keep the source's `sem` and `scope`. The result of a masked-off atomic lane
  never decides control flow or an address.

No construct of [exclusions.md](exclusions.md) is used. In particular: no
`inline_asm_elementwise` replaced by "equivalent" code, no pointers to pointers, no access to the
same bytes through both pointers and descriptors, and no aggregate class (a class made with
Triton's `_aggregate`).

## 4. Float addition: IEEE or fusable

- **[fusion]** An addition, subtraction, reduction or scan that a product may feed is written in
  its fusable form. A product is an operand computed by `*`, `tl.dot` or `tl.rand`, through any
  dataflow, inlined helpers and library functions included. Otherwise the IEEE form is used.

  | IEEE form | fusable form |
  |---|---|
  | `tl.add` | `tl.mul_add a b c` for `a * b + c`; `tl.addc` otherwise |
  | `tl.sub` | `tl.subc` |
  | `tl.sum`, `sum1`, `sum0` | `tl.sumc`, `sum1c`, `sum0c` |
  | `tl.cumsum` | `tl.cumsumc` |
  | an `AddRel false` scan relation | `AddRel E.fusion` |

  The launch's `enable_fp_fusion` is the source's (Triton's default: `true`).

## 5. Shapes the compiler rewrites

These are transcribed in their rewritten form, whatever `enable_fp_fusion` says.

- **[dot-add]** `tl.dot(a, b) + c` or `c + tl.dot(a, b)`, where the dot's accumulator is a
  constant zero and the product has no other use, is `tl.dot_add` (with the source's zero as `z`).
- **[dot-scaled-add]** The same for `tl.dot_scaled`: `tl.dot_scaled_add`.
- **[sum-bcast-mul]** A float `tl.sum(x[:, :, None] * y[None, :, :], axis=1)` is
  `tl.sum_bcast_mul`.
- **[reshape-reorder]** `tl.reshape(x, s, can_reorder=True)` and `tl.view` are `tl.reshape_any`.
  So is a reshape to rank 1 whose only users are reductions or histograms. No index (`tl.argmax`,
  a `(value, index)` reduction) is paired with such a reshape as if the order were kept.

## 6. Reductions, `tl.dot`, sorting, scans

- **[reduce]** Reductions keep the source's axis. Triton's conversions before a reduction are
  written (`tl.cvt`, `tl.cvtFrom8`, `Tensor.castInt`):
  - `tl.max`, `tl.min` (without `return_indices`): a float narrower than 32 bits becomes
    `float32`; an integer narrower than 32 bits becomes `int32`, *whatever its signedness*:
    `int8` and `int16` sign-extended, `uint8` and `uint16` zero-extended (`x.castInt` to `I32`),
    and a `tl.int1` (`Bool`) tensor `0`/`1`. So `tl.max` of an `int8` tensor is an `int32`.
    Integer `tl.max`/`tl.min` are `Tensor.foldAll`/`foldRows`/`foldCols` with
    `IntV.max`/`IntV.min`; float ones are `tl.maxAll`/`max1`/`max0`.
  - `tl.sum`: an integer narrower than 32 bits becomes `int32` if signed and `uint32` if unsigned
    (`tl.int1` is unsigned); with `dtype=d` the input becomes `d`; floats are not widened. An
    integer `tl.sum` is a fold with wrapping `+`.
  - With `return_indices=True`, and in `tl.argmax`/`tl.argmin`, only `bfloat16` is converted (to
    `float32`) ([argmax]).
- **[argmax]** `tl.max(x, axis, return_indices=True, return_indices_tie_break_left=t)` and
  `tl.min(…)` are `tl.argmax1 isMax t x` (`axis=1` of a rank-2 tensor), `tl.argmax0 isMax t x`
  (`axis=0`) and `tl.argmaxAll isMax t x` (a rank-1 tensor), with `isMax` true for `tl.max`. They
  give the values and the `int32` indices. `tl.argmax(x, axis, tie_break_left=t)` and
  `tl.argmin` are the index part. The input is reduced in its own dtype, except that `bfloat16`
  becomes `float32` (`tl.cvt`).
- **[dot]** `tl.dot` keeps the source's `out_dtype` and `input_precision`, with the resolved
  `max_num_imprecise_acc` in its `DotCfg`: the source's value, or `2^30` for an fp8 × fp8 dot on
  sm_90 and `0` otherwise. An omitted `input_precision` is `tf32` (and `ieee` with
  `allow_tf32=False`); §9 says why.
- **[dot-scaled]** `tl.dot_scaled`'s `ScaledCfg` has the source's formats, `k_pack` flags and
  `fast_math`:
  - `lhsScale`/`rhsScale` are `e8m0` for `uint8`/`int8` scale tensors, and `float d` for float
    ones;
  - `group` is `K` divided by the scale tensors' last dimension (`16` or `32`);
  - the operands are the storage tensors as they are (packed fp4 as `uint8`).
- **[sort]** `tl.sort`: `descending` and the dimension (the last) match.
- **[histogram]** `tl.histogram`: `bins` and `mask` match.
- **[scan]** Scans:
  - `combine_fn` is written as a relation that covers every result its body can give. For float
    bodies, use the library's relations: `AddRel`, `NaNEq`, `MaxRel`. For integers, write
    `fun a b c => c = a + b`.
  - The axis and `reverse` match.
  - With `reverse=True` the left operand of `combine_fn` covers the higher indices, so a
    non-commutative relation is stated that way.
  - The promotions `tl.cumsum`/`tl.cumprod` do are written.

## 7. Control flow

- **[range]** `range(lo, hi, step)` is `tl.forRange`, and a fixed count is `tl.forN`. A loop's
  carried variables form a tuple state, and the induction variable has the promotion of the
  bounds' types. A `range` with unsigned bounds must satisfy `tl.RangeOk` (a signed range).
- **[static-range]** `tl.static_range` is unrolled with the same count.
- **[while]** A `while` loop is `tl.whileLoop`, with the source's condition.
- **[bool-ops]** `and`/`or` of scalar tensors evaluate both sides.

## 8. Compiler hints

- **[hint]** `tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy` and `tl.assume` are
  checks, not no-ops:
  - **Where.** The check sits where the hinted value is defined, before any use of it. Triton
    applies a hint to the value everywhere, also before the call and on paths that never reach
    it.
  - **Rank ≥ 2.** One value per dimension, in the source's order. What a per-dimension value
    promises, and the `contig` argument of `multiple_of`, are **(open: L2)**.
  - **Inside a loop body.** A hint on a name the loop carries, or on its induction variable, is
    **(open: L1)**. The earlier rule that dropped such a hint as a no-op is withdrawn: Triton
    applies it to the value the name has before the loop.

## 9. The launch

`launch d` gives the host's grid formula, and `program_id` reads the right axis.
`Launch.Valid` asks for fewer than `2^31` programs: Triton's launcher skips larger grids without
an error ([SEMANTICS.md](SEMANTICS.md), "Launches"). `blocks` lists every block shape of the
source on the instance. `Launch.Valid` checks only the shapes listed, so an empty or partial list
passes it. `num_ctas` is the source's (`1`), `debug` is `false`, and `launch_pdl` is not set
(`pdl = false`): a kernel launched with programmatic dependent launch may start before the
previous kernel has finished writing, and `Launch.Valid` excludes it. With `@triton.autotune`,
`Inst` ranges over the configs the autotuner may pick.

Triton's environment is at its defaults: `Rules.Compiler.Faithful` covers only kernels compiled
and launched without the environment variables it names ([rules/compiler-faithful.md](rules/compiler-faithful.md)).
So an omitted `input_precision` of `tl.dot` is `tf32`, whatever `TRITON_F32_DEFAULT` might say,
and an omitted `enable_fp_fusion` is `true`.

## 10. What may be dropped

The model map's drop reasons ([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §7). Each is a rule, and a
dropped line names it as its `drop`. Anything else that is dropped is a mismatch.

- **[no-op:debug_barrier]** `tl.debug_barrier()`.
- **[no-op:print]** `tl.static_print`, `tl.device_print`, Python's `print(...)`,
  `tl.device_assert` and the Python `assert` statement, when every argument is pure ([pure]).
  - `tl.static_print` prints at compile time.
  - `tl.device_print` and `print` emit a device `printf`, which writes no global memory. Python's
    `print` *is* `tl.device_print` in a kernel.
  - Without `debug`, `tl.device_assert` emits nothing, and `Compiler.Faithful` covers only
    `debug = false`. A Python `assert test, msg` is `tl.device_assert(test, msg)`.
  - The arguments that must be pure are all of them: the values printed; `cond`, `msg` and
    `mask` of `tl.device_assert`; `test` and `msg` of `assert`.

  Evidence: [evidence/debug-calls](../evidence/debug-calls/README.md) P1, P2, E3–E7, E9.
- **[no-op:cache]** Cache, eviction and `volatile` modifiers.
- **[no-op:range-kw]** `tl.range` scheduling keywords (`num_stages`, `loop_unroll_factor`,
  `warp_specialize`, …).
- **[no-op:gdc]** `tl.extra.cuda.gdc_wait()` and `tl.extra.cuda.gdc_launch_dependents()`. Every
  valid launch runs without programmatic dependent launch (§9), and Triton documents both calls
  as safe to execute then: the previous kernel on the stream has finished before the launch
  starts, so there is nothing to wait for, and no dependent kernel is waiting for the signal
  ([evidence/pdl](../evidence/pdl/README.md)).
- **[decl]** A decorator or signature line, covered by the model's definition.
- **[config]** A branch not taken in the covered configuration (a branch on constexprs). The
  note names the condition.
- **[compile-check]** `tl.static_assert(cond)` or `tl.static_assert(cond, msg)` whose `cond` is
  pure ([pure]) and holds on every covered instance wherever the front end evaluates it. The note
  names the condition of `Inst.ok` that makes it hold.
  - The front end evaluates `cond` at compile time, and a condition that is not a compile-time
    `bool` stops compilation. `x is None` and `x is not None` are compile-time `bool`s for any
    `x`, so the purity of `cond` matters
    ([evidence/debug-calls](../evidence/debug-calls/README.md) E1, E2).
  - `msg` is evaluated only when the assertion fails, and then compilation stops (E8). A failed
    compile is a launch rejected with an error on the host, which has no outcome. So `msg` never
    matters to a launch that runs, and need not be pure.
  - **Where it is evaluated.** At every place the front end visits it: in both branches of a
    runtime `if`, whether or not a program takes them; in each unrolled iteration of
    `tl.static_range`; at each call of the inlined helper that holds it; and in a `for` or
    `while` body, even one that runs zero times (V1–V3). Not in a constexpr branch the
    configuration does not take (V4). The note names the condition of `Inst.ok` that makes the
    assertion hold at every visit.
  - A `tl.static_assert` whose `cond` is not pure is transcribed as [pure] says: the operations
    of `cond` become model lines.
- **[dead]** A value no covered path uses, computed by a pure expression ([pure]). The note says
  why.
- **[comment]** A comment or blank line inside a range.

- **[pure]** Triton evaluates the arguments of every call as ordinary expressions, before the
  call. So an operation inside an argument runs even when the call itself does nothing: an atomic
  inside the condition of a `tl.static_assert`, or inside an argument of `tl.static_print`,
  `tl.device_print`, `print`, `tl.device_assert` or `assert`, runs in every program
  ([evidence/debug-calls](../evidence/debug-calls/README.md) E1–E7, E9). A drop is sound only for
  expressions that have no effect and are never undefined.

  An expression is **pure** when it is built only from:
  - names of values the model already has, literals, `tl.constexpr` values, and dtypes
    (`tl.float32`, …);
  - the attributes `.dtype`, `.shape` and `.numel`, the attributes of a dtype (such as
    `.element_ty`), and subscripts of shapes and tuples;
  - subscripts that only add axes (`x[:, None]`, `x[None, :]`);
  - string literals, and f-strings of the above;
  - the operators `+`, `-` (binary and unary), `*`, `~`, `&`, `|`, `^`, the comparisons, `not`,
    `and`, `or`, `is` and `is not`;
  - the operators `/`, `//`, `%`, `**`, `<<` and `>>` when both operands are `tl.constexpr`
    values (Python computes them at compile time). On tensors and runtime scalars they are not
    pure: integer `//` and `%` by zero, or a shift by the width or more, is undefined, and float
    `/` is the platform's `P.div`.

  **Operand types.** Every operand of an operator, and every value that the expression subscripts
  or reads an attribute of, is one of:
  - a tensor: a block or a scalar of a Triton dtype, pointers and `tl.int1` included;
  - a `tl.constexpr`, or a Python `int`, `float`, `bool`, `str` or `None`;
  - a dtype;
  - a tuple (a shape, or a tuple the source builds) whose elements are all of these kinds.

  Nothing else: in particular no instance of an aggregate class (a class made with Triton's
  `_aggregate`), and no tuple that holds one. On such a value Triton applies an operator,
  comparison or subscript by calling the class's own method, a `@triton.jit` function it inlines,
  and that method may run any operation
  ([evidence/pure-operands](../evidence/pure-operands/README.md)). The model has the type of
  every value, so the reviewer checks the operand types there.

  A pure expression contains **no call** of any kind: no `tl.*` function, no `@triton.jit`
  function, no method (`x.to(…)`), no Python function. A reviewer checks this by reading the
  expression.

  **When an argument is not pure,** the line is not dropped. Its argument expressions are
  transcribed as model lines, in Python's order (left to right), as if each stood alone, and the
  call itself adds nothing. For example `tl.device_assert(tl.atomic_add(c, 1) >= 0, "m")`
  becomes `let _ ← tl.atomic_add …` (the comparison of the result is pure, so it may be left
  out). The map entry is then a `model` entry that names [pure] among its rules, and its note
  says that the call adds nothing.
