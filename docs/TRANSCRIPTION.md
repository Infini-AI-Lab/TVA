# Transcription guidance

How to write a kernel's model, and what to look for when checking one: worked patterns and the
pitfalls that most often make a model differ from its source. This page is guidance. It states no
fact of its own:
- what each construct means is in [SEMANTICS.md](SEMANTICS.md): every `tl.*` operation and
  operator, how Triton's front end evaluates the source (which parts of an expression run, which
  branches it visits), which constants it builds, which shapes its compiler rewrites, and what a
  launch means. Kernel review trusts that page, and library review checks it;
- what is out of scope is in [exclusions.md](exclusions.md).

The model is the prover's transcription, and the kernel reviewer judges whether it means what the
kernel does: every line, every drop, every name. Neither follows this page mechanically (user,
2026-10-28: "our transcription is not based on a fixed rule, we provide the guidance and the
reviewer needs to check whether the model matches the kernel"). A pattern below that does not fit
a kernel is not to be applied anyway, and a construct missing here is not forbidden. The question
is always the same: on every covered instance, does the model do what the source does, by
SEMANTICS.md?

When the answer needs a fact about Triton that SEMANTICS.md does not state, the model cannot be
judged yet: the prover asks the library for the fact (a library request), and a reviewer who finds
a model relying on an unstated fact files `UNSUPPORTED`. A fact that is stated but doubtful is a
`LIBRARY_REPORT`. This page is in no digest, so a change to it re-opens nothing (library issue L36);
a change to SEMANTICS.md re-opens every item that reaches the library.

## 1. The model's shape

The kernel is a `Kernel` whose `body : Ctx → Prog Unit` is a `do` block. Keep the source's line
structure and names, one model line per source line where possible, so that the reviewer can read
the two side by side. A `tl.constexpr` argument is a Lean parameter (`BLOCK_SIZE : Nat`). An
inlined `@triton.jit` helper is written where it is called, or as a Lean function the call site
applies to the call's arguments.

The source must stay inside the supported subset of Python (SEMANTICS.md, "The supported subset";
library issue L88): a kernel that uses another form (a `break`, a `@triton.heuristics`, a
`@triton.autotune`, a bare
annotation, a `tl.constexpr(…)` global, a `for` loop's variable after the loop, …) has no claim yet,
whatever its model says. Check that first.

Every operation is a `TVA.tl.*` or another `TVA.*` definition, and every name in a model line must
mean what the library says. That is why the kernel's Lean declares nothing in the `TVA` namespace,
nothing named like a library name, no instance, notation or `export`, and no binder named like a
library namespace (a local `tl : Ops` makes `tl.add x y` a field of the binder; an instance on a
core type such as `HDiv Nat Nat Nat` changes the grid formula `(n + B - 1) / B`). Such a
declaration changes what a model line means without changing its text. The kernel reviewer reads
all of the kernel's Lean for this; the gate's `hygiene` and `namespace` checks are a backstop for
the common routes ([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §8). An operation missing from
`TVA.Lang.Ops` is not defined in the kernel: the library adds it.

## 2. Worked patterns

Each row is a common source construct, the model text it usually becomes, and the section of
SEMANTICS.md that says why. Adapt them; do not copy them blindly.

| Source | Model | SEMANTICS.md |
|---|---|---|
| `pid = tl.program_id(axis=0)` | `let pid := tl.program_id ctx 0` | Values, "Index arithmetic"; Launches |
| `offs = pid * BLOCK + tl.arange(0, BLOCK)` | `let offs := pid * (IntV.ofNat BLOCK : I32) + tl.arange 0 BLOCK` | Values, "Index arithmetic" |
| `mask = offs < n` | `let mask := offs.ltS n` | Values, "Operators on tensors" |
| `x = tl.load(x_ptr + offs, mask=mask)` | `let x ← tl.load (x_ptr + offs) (some mask)` (no `other`: masked-off lanes are unspecified) | `tl.load` |
| `x + y` on floats that no product feeds | `let z ← tl.add x y` | "Float `+` and `-`" |
| `a * b + c`, `acc + tl.dot(…)`, `tl.rand(…) - 0.5` | `tl.mul_add a b c`, `tl.addc`, `tl.subc` (a product feeds them) | "Float `+` and `-`" |
| `tl.store(out_ptr + offs, y, mask=mask)` | `tl.store (out_ptr + offs) y (some mask)`, after the conversion to the pointer's dtype | `tl.store`; "Float conversions" |
| `x.to(tl.float16)`, `x.to(tl.float16, fp_downcast_rounding="rtz")` | `← tl.cvt x`, `← tl.cvt x .rz` (`tl.cvt8` into fp8, `tl.cvtFrom8` out of it, `tl.itof` from integers, `tl.ftoi` to them) | "Float conversions"; `x.to(float)` |
| `offs.to(tl.float32).to(tl.bfloat16)` | one `tl.itof` to `bfloat16` (default rounding); with `rtz`, `tl.itof` to `float32`, then `tl.cvt … .rz` | `x.to(float)` |
| `x * 0.1` with `x` a `bfloat16` tensor | the constant `FloatV.litBF16 (FloatV.ofPat …)` (the `float64` pattern of `0.1`, with `0.1` in a comment) | "Constants" |
| `e = 0.1` then `x * e`, with `x` a `bfloat16` tensor | `e` is a `float32` scalar tensor (`FloatV.litF32 …`); `x` converted to `float32`, then the `float32` product | "How the front end evaluates the source": a name assigned a Python number |
| `i = 0` before a loop that carries `i`, `lo, hi = 0, N` | `i` and `lo` are `int32` scalar tensors in the model (`IntV.ofInt 0 : I32`), and `//`, `%` on them are tensor operations; `hi` is `N` itself when `N` is a tensor (a runtime scalar keeps its type), and a scalar tensor of `N`'s own type when `N` is a Python integer (`int32` below `2^31`; `uint32`, `int64` or `uint64` beyond) | the same |
| `c: tl.constexpr = -7` (a name not defined before), then `c // 2` | `c` is the Lean integer `-7` (a compile-time value), `c // 2` is Python's `-4`; after a later `c += 0`, `c` is an `int32` tensor | "Annotated names" |
| `h(x)` with `def h(x, e=1e-7)` | the body with `e` the Python number `1e-7` (a compile-time value, as if passed directly) | "Helpers are inlined" |
| `tl.load(p, mask=m, other=-float('inf'))` | `other` is the `float32` `−∞` (`float('inf')` is Python's, at compile time) | "Python's builtins"; "Constants" |
| `if isinstance(d, tl.tensor_descriptor):` | a compile-time `if`: the branch the argument's kind picks, the other dropped (`config`) | "Python's builtins" |
| `tl.maximum(x, 0.5)` with `x` a `float16` tensor | `0.5` as a `float32` tensor (`FloatV.litF32`), `x` converted to `float32`, then `tl.maximum` | "Constants"; `tl.maximum` |
| `x * w + b` with `x` a `float32` and `w`, `b` `float16` tensors | `← tl.convert .f16 .f32 w` and `← tl.convert .f16 .f32 b` (the dtype `BinOp.typing? .mul (.tensor .f32 true) (.tensor .f16 true)` gives), then the `float32` product and, for the addition it feeds, `tl.mul_add` | "Operators and promotion"; "Float `+` and `-`" |
| `x / n` with `x`, `n` `int32` | both `tl.convert .i32 .f32`, then `tl.div` | "Operators and promotion" |
| `x < 0.1` with `x` a `float64` tensor | the constant `FloatV.litF32 …` converted to `float64`, then the comparison | "Constants" (everywhere else) |
| `tl.where(keep, x, 0.0)` with `keep` an `int32` tensor | `← tl.convert .i32 .i1 keep`, then `Tensor.where_` in the dtype `whereTyping?` gives: for a float `x`, `x`'s dtype, with the constant built in it (`FloatV.litF32 …`, `litBF16 …`, …); for an integer `x`, `float32`: `x` converted (`tl.convert .i32 .f32 x`, `tl.itof`'s rounding: `16777217` becomes `16777216`) and the constant `+0.0` in `float32` | "Operators and promotion" (`tl.where`); "Constants" |
| `tl.where(offs < M, offs, 0)` with `offs` an `int32` tensor (tutorial 09) | `Tensor.where_` in `int32`, the constant `IntV.ofInt 0` | the same |
| `tl.where(mask, 0, -1.0e6)` (two Python numbers, tutorial 06) | `Tensor.where_` in `float32`, the constants `+0.0` and `FloatV.litF32 …` of `-1.0e6` | the same |
| `min(a, b)` (Python's builtin) on `int32` scalars or tensors | `tl.iminimum a b` | "Operators and promotion" |
| `if c:` with `c` a runtime scalar | a Lean `if` on `c`'s value (after `c != 0` for a non-`int1` `c`), each branch's operations in its branch | "How the front end evaluates the source": a runtime `if` |
| `if pid >= n: return` | `if … then return ()` at that point of the `do` block | "How the front end evaluates the source": `return` |
| `tl.sum(x, axis=1)`, `tl.max(x, axis=0)` | `tl.sum1 x` (or `tl.sum1c` for a sum of products), `tl.max0 true x` | reductions |
| `tl.max(x)` with `x` an `int8` tensor | `x.castInt` to `I32`, then a `Tensor.foldAll` with `IntV.max` | reductions (widening) |
| `tl.dot(a, b, acc)` | `← tl.dot cfg a b acc` (`cfg`: the operand dtypes, `out_dtype`, `input_precision`, the resolved `max_num_imprecise_acc`) | `tl.dot` |
| `tl.dot(a, b) + c` with a zero accumulator | `← tl.dot_add cfg a b c` | combine-pass rewrites |
| `tl.sum(x[:, :, None] * y[None, :, :], axis=1)` and its other spellings | `← tl.sum_bcast_mul d x y` | combine-pass rewrites |
| `tl.reshape(x, s, can_reorder=True)`, `tl.view`, a rank-1 reshape feeding only reductions | `tl.reshape_any` | Shape changes |
| `tl.trans(x)` on rank 3 | `x.permute [0, 2, 1]` | Shape changes |
| `a >> b` | `tl.ashr` when the operand Python dispatches on is signed before promotion, `tl.lshr` otherwise | Integer `//`, `%`, `<<`, `>>` |
| `for i in range(lo, hi, step)` | `tl.forRange` with the carried variables as a tuple state; a fixed count `tl.forN` | Loops |
| `for i in range(n)` (one argument) | `tl.forRange` from `0` with step `1`; the operations of `n`, if any, written twice before the loop, the bound being the second value (a plain name: nothing to write) | "Loops and unrolling": the iterator's arguments |
| `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)` with `M` a runtime `int32` and `BLOCK_SIZE_M` a `tl.constexpr` | the `int32` expression `(M + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M`: a wrapping addition, then `tl.idiv` (truncating, undefined for a zero divisor), so `Inst.ok` bounds `M` where the model needs the ceiling (`0 ≤ M` and `M + BLOCK_SIZE_M - 1 < 2^31`); with two `tl.constexpr`s, the call is Python's `(M + (B - 1)) // B`, a compile-time value, and the name assigned it then holds a scalar tensor of its own type | `tl.cdiv`; a name assigned a Python number |
| `for i in tl.static_range(N)` | the body written `N` times | Loops |
| `while c:` | `tl.whileLoop` | Loops |
| `x = tl.multiple_of(x, 16)` | the check `tl.multiple_of …` where `x` is defined, before any use (`contig` the lower bound SEMANTICS states for how `x` is computed, or `1`) | Compiler hints: the contiguity the analysis infers |
| `offs = tl.multiple_of(pid * B + tl.arange(0, B), 4)` | `tl.multiple_of offs 4 (contig := B)`: `4` divides `pid * B` | Compiler hints: the contiguity the analysis infers |
| `tl.multiple_of(p, 16)` in a loop body, on a name the loop carries | the check on `p`'s value before the loop, once, where it is defined (none after the loop starts) | Compiler hints: which value |
| `tl.max_contiguous(tl.multiple_of(offs, B), B)` | `tl.max_contiguous_multiple_of offs B B`, where `offs` is defined | Compiler hints: what it promises |
| `a, b = tl.split(z)` then `a = tl.multiple_of(a, 4)` | the check on `a` and on `b`, where the split defines them | Compiler hints: an operation with several results |
| `desc = tl.make_tensor_descriptor(ptr, shape=[M, K], strides=[K, 1], block_shape=[…])` | `let D ← tl.make_tensor_descriptor ptr [M, K] [K, 1] […]` (the shape `int32`, the strides `int64`) | Tensor descriptors |
| `tl.make_block_ptr(…)`, `tl.load(bp, boundary_check=…)` | `tl.make_block_ptr` with each entry cast to `int64`, `tl.load_block` | Block pointers |
| `desc.load([m, n])` on a host `TensorDescriptor` | `tl.desc_load D offsets` | Tensor descriptors |
| `tl.associative_scan(x, 0, combine)` | `tl.associative_scan x 0 op reverse`, `op` a relation covering every result of `combine` | Scans |
| `tl.cumsum(x, 0, dtype=d)` | `bfloat16` converted to `float32` first, then the conversion to `d` (`tl.convert`), then `tl.cumsum` (or `cumsumc`, `icumsum`) in `d` | Scans |
| `tl.atomic_add(p, v, mask=m)` | the atomic with the source's `sem` and `scope` | Atomics |

## 3. Pitfalls

What most often makes a model differ from its source, and what a reviewer should look for:
- **An IEEE `+` that a product feeds**, through any dataflow: a `*`, a `tl.dot`, a `tl.rand`, an
  inlined helper. It may be fused on the device, and the IEEE form does not allow that.
- **A missing conversion**: a dtype promotion (evaluate `BinOp.typing?` on the operands: `float16`
  with `bfloat16` is `float16`, an `int32` with an `int64` sign-extends, two integers under `/` are
  `float32`), the conversion of a stored value, `bfloat16` widened for `tl.maximum`, the widening
  of an integer `tl.max` or `tl.sum`, a `float16` `/` computed in `float32`, the `c != 0` of a
  `tl.where` condition that is not `int1`.
- **A constant in the wrong dtype**: a `bfloat16` constant is built from six decimals (`1e-7` is
  `0`); an operand of a comparison, an argument of `tl.maximum`, `other=` or a stored constant is a
  tensor of its own type first (`x < 0.1` on `float64` compares with `float32(0.1)`), and `other=`
  is then converted to the pointer's dtype (a `float16` load's `other` is rounded twice, through
  `float32`); `tl.where(m, x, 0.0)` with an integer `x` computes in `float32`.
- **A name that holds an assigned number, read as a Python number**: after `e = 1e-7`, `a = -7`,
  `F = False` or `lo, hi = 0, N`, each name holds a scalar tensor of the number's own type, unless the
  statement is `name: tl.constexpr = e` on a name not defined before. So `x * e` promotes a `bfloat16`
  `x` to `float32`, `a // b` truncates, `F and …` evaluates both sides, `if F:` is a runtime `if`
  whose branches the front end both visits, and `tl.arange(0, B)` after `B = BLOCK` is rejected. A
  number passed directly to a helper, or a helper's default, stays a compile-time value there
  (SEMANTICS.md, "How the front end evaluates the source").
- **A conditional expression on a runtime condition read as Python's**: `a if c else b` with a
  runtime `c` is a tensor, each Python-number side a scalar tensor of its own type, as an assigned
  number: `(-1 if c > 0 else -3) // 2` truncates, `x * (0.1 if c > 0 else 0.2)` on `bfloat16`
  computes in `float32`, and two sides of different types are rejected. Beware the scalar argument
  equal to `1`: it is a compile-time constant, so `c > 0` with `c = 1` is a compile-time condition
  and only one side is visited (SEMANTICS.md, "Order", "Scalar arguments").
- **A helper's value read as Python's**: a helper whose visited `return`s give different kinds of
  values (two different numbers, `1` and `1.0`, a number and a tensor), compile-time values that
  Python finds equal but that are not the same value of one Python type, element by element for a
  tuple (`(1, 2)` and `(1.0, 2)`, `0.0` and `-0.0`), or that can fall off its end while another
  `return` gives a value, is outside the subset: the front end types the call from all its `return`s,
  and equal values give the first one's on every path (SEMANTICS.md, "`return`", "A helper's
  value"). Likewise two calls of one helper
  whose compile-time arguments differ but name the same copy (two NaNs, `"x[y"` and `"x_y"`)
  (SEMANTICS.md, "Compile-time values Triton keys apart").
- **An fp8 constant read as a conversion**: `tl.full(s, c, tl.float8e5)` and a Python number
  beside an fp8 tensor in `tl.where` are the exact value of `c`'s `float32` (`FloatV.litF8`:
  `float('-inf')` stays `−∞`, an inexact `c` is rejected), not `tl.cvt8`; a stored number,
  `other=` and `.to(…)` convert on the device and saturate (SEMANTICS.md, "Constants").
- **An annotation read as lasting**: `name: tl.constexpr = e` makes only that statement keep the
  Python value. A later `c += 0` or `c = c * 1` on that name, or an assignment to a `tl.constexpr`
  parameter, makes a tensor (`c: tl.constexpr = -7; c += 0; c // 2` is `-3`). A name bound so in a
  branch of a runtime `if` belongs to that branch, even when both branches bind it. A bare
  `e: tl.constexpr` and an annotation other than `tl.constexpr` (`a: tl.int64 = …` is an `int32`, and
  its value is evaluated twice) are outside the subset (SEMANTICS.md, "Annotated names", "A runtime
  `if`").
- **Python's builtins**: `float`, `int`, `isinstance`, `len`, `list`, `getattr` and `hasattr` run at
  compile time and add no step (their arguments are still evaluated); `min`, `max` and `print` are
  Triton's; `range` is the loop's iterator; any other builtin is rejected on the host (SEMANTICS.md,
  "Python's builtins").
- **A scalar argument's type**: an unannotated integer argument equal to `1` is a compile-time
  constant (Python's arithmetic), another one in `[-2^31, 2^31)` an `int32`, one outside it an
  `int64` (in `[-2^63, 2^63)`) or a `uint64` (from `2^63`); a Python `float` is `float32`; a
  `bfloat16` parameter truncates. `Inst.ok` must fix the class the model assumes.
- **Masked-off lanes**: a masked load without `other` gives unspecified values, not necessarily
  `0`; the result of a masked-off atomic lane is unspecified and must decide nothing.
- **Index arithmetic** that wraps in `int32`, or a missing `int64` cast.
- **A loop's iterator and names**: `range(n)` with one argument evaluates `n` twice (an operation in
  it runs twice); Python's `range` ignores a keyword (`step=2`) and a fourth argument, which are
  outside the subset (`tl.range` takes `step=` as its step); a name first defined in a loop body,
  and a `for` loop's variable, are outside the subset after the loop, whatever the iterator
  (`tl.static_range` included) (SEMANTICS.md, "Loops and unrolling").
- **Code after an `if` that returns**: after a compile-time `if` whose taken branch returns, or a
  runtime `if` whose two branches return, the front end still compiles the rest of the function
  (its hints and `tl.static_assert`s act, though nothing in it runs), so such code, and a `return` in
  a `tl.static_range` body, are outside the subset; the statements after a `return` in its own
  statement list are not compiled, and a map may drop them (SEMANTICS.md, "`return`").
- **`tl.cdiv`** read as the ceiling: it is `(x + (div - 1)) // div`, which on an `int32` wraps and
  truncates (SEMANTICS.md, "`tl.cdiv`").
- **A hint** left out, or written at its call rather than where the value is defined; a hint in a
  loop body on a carried name checked on every iteration (it checks the value before the loop); a
  `multiple_of` alone read with a `contig` larger than the lower bound SEMANTICS.md states for how
  the value is computed ("The contiguity the analysis infers": a value computed in any other way than
  the list's has the bound `1`, every element); a `max_contiguous` alone below the tensor's own
  contiguity (with a `multiple_of` on the same value, the pair allows it); a hint on one result of a
  `tl.split`, a tuple reduction or scan, or a loop's or an `if`'s results, checked on that result only
  (it is checked on each, and a hint on such a value with a result of another shape, or a float
  result under `multiple_of` or `max_contiguous`, is outside the subset); a `contig` read from the list
  for a value built from one that carries a `max_contiguous` (its bound is the smaller of the hinted
  `C` and the list's bound without the hint); and the hinted `C` alone taken as that bound (the
  compiler may not read the mark: on a loop's or a runtime `if`'s result, the value an inlined call
  returns, a result of `tl.split` or of a tuple reduction or scan, or an operation it folds away such
  as `r + 0`, and then the list's bound without the hint is what holds). Tutorial 08's
  `grouped_matmul_kernel` is excluded, and tutorial 09's pointer kernels are covered on full tiles
  only (exclusions.md).
- **An operation inside something that does nothing at run time**: an atomic in the condition of a
  `tl.static_assert`, in an argument of a print or a `device_assert`, or in the condition of a
  branch the configuration decides, runs in every program, as far as the front end evaluates it
  (SEMANTICS.md, "How the front end evaluates the source").
- **A rewritten shape written as its parts**: `tl.dot(a, b) + c` with a zero accumulator, a
  broadcast-multiply-sum, a rank-1 reshape feeding a reduction. Their library operations allow
  the rewritten results.
- **Reverse scans**: with `reverse=True`, the left operand of `combine_fn` covers the higher
  indices, which matters for a combine that does not commute.
- **Names**: a kernel declaration or binder that makes a model line mean something else (§1).
- **The host**: nothing else may write the claim's buffers while the launch runs, with Triton's
  knobs at their defaults, no hook, only the options `Launch` records (also at `warmup`, in
  `triton.compile`'s options and in a config), no `@triton.heuristics`, a descriptor scratch that is
  none of the claim's buffers, a precompiled kernel compiled from the kernel's source (not an IR
  file) with arguments in its specialisation (host descriptors with the warm-up's dtype and block
  shape), no host descriptor with `round_f32_to_tf32=True`, an H200 as the device and an x86-64 host
  with CPython 3.13 run without `-O`/`-OO` or `PYTHONOPTIMIZE` (which remove Triton's `assert`
  rejections) and in the default floating-point environment (no `fesetround`, `_mm_setcsr` or
  `-ffast-math` library, which change the constants Triton builds), no CUDA graph capture (no call
  while a stream captures, on any stream, `torch.cuda.graph` or `do_bench_cudagraph`, and no replay),
  no `preload`, Triton and the kernel's module as installed and written (the installed `triton`; no
  other code changes `tl.*`, Triton's classes and objects, a `JITFunction` or a name of the module,
  other than through Triton's API as the tutorials use it: `set_allocator`, a hook's `block_shape`),
  and the kernel new (the code the launch runs is the code Triton compiles for its own argument
  values and settings; code a cache returns counts when it is that code)
  ([rules/compiler-faithful.md](rules/compiler-faithful.md), "Scope"); `init` is the memory when
  the direct launch starts, not the caller's memory before the call ("Which launch"). Of these, the
  kernel and its launches (the options, `@triton.heuristics`, the
  descriptor scratch, a precompiled kernel's arguments, host descriptors, graph capture) are part
  (A) of the scope, which the kernel reviewer checks; the rest is part (B), the environment, which
  TVA assumes and nobody checks.

## 4. Dropping a line

A source line may be left out of the model when the model means the same without it: on every
covered instance it has no effect on global memory, is never undefined behaviour, and decides
nothing the model computes. The map entry gives the reason as its `drop`, and a `note` with the
condition that makes it hold, such as the condition of `Inst.ok` that fixes a configuration. The
reviewer judges each drop by SEMANTICS.md. These reasons are common; any clear reason will do:

| Reason | What it drops | Why it may (SEMANTICS.md) |
|---|---|---|
| `decl` | a decorator (`launch_metadata=` included) or a signature line | covered by the model's definition; `launch_metadata=`: Launches |
| `comment` | a comment or a blank line inside a range | nothing to run |
| `no-op:debug_barrier` | `tl.debug_barrier()` | "No barrier semantics" |
| `no-op:print` | `tl.static_print`, `tl.device_print`, `print`, `tl.device_assert`, `assert`, when what the front end evaluates in their arguments has no effect | "How the front end evaluates the source": debugging calls, expressions with no effect |
| `no-op:cache` | cache and eviction modifiers of loads and stores, `volatile`, and the other performance-only arguments (`num_warps`, `maxnreg`, …) | "Performance-only arguments" |
| `no-op:hint` | a hint on a kernel's or a helper's parameter, on a loop's induction variable, or on the value an inlined call returns (a helper, `tl.sum`, `tl.max`, …) | Compiler hints: which value |
| `no-op:range-kw` | `tl.range` scheduling keywords (`num_stages`, …) | Loops |
| `no-op:gdc` | `gdc_wait()`, and `gdc_launch_dependents()` when the host launches nothing after it with `launch_pdl=True` | Launches |
| `config` | the branch a compile-time condition does not take (a `tl.constexpr`, an `isinstance`, …), with the later `elif` lines | "How the front end evaluates the source": `if`, `elif`, `else`; Python's builtins |
| `compile-check` | a `tl.static_assert` that holds on every covered instance, wherever the front end visits it | "How the front end evaluates the source": `tl.static_assert` |
| `dead` | a value no covered path uses, computed with no effect | "Expressions with no effect" |

When something the front end evaluates in a dropped construct does have an effect, that part is
not dropped: it becomes model lines, in the order the front end evaluates it, and the construct
itself adds nothing. Examples, with `c` a pointer and `FLAG` a `tl.constexpr` that is `False` on
every covered instance:
- `tl.device_assert(tl.atomic_add(c, 1) >= 0, "m")` becomes `let _ ← tl.atomic_add …`; the
  comparison of its result has no effect.
- `tl.static_print(FLAG and tl.atomic_add(c, 1))` may be dropped: `FLAG` ends the `and`, and the
  atomic is never evaluated.
- `if tl.atomic_add(c, 1) is None and tl.atomic_add(c, 1) is None:` evaluates the first atomic
  only (`is None` gives the compile-time `False`), so it is one model line, and the branch, never
  taken, is dropped.
- In `if FLAG: … elif tl.atomic_add(c, 1) is None: …`, the `elif` condition runs, since `FLAG` is
  false; after a taken `if`, it would not.

## 5. The model map

`model-map.json` maps every code line of the kernel to model text or a drop
([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §7). The gate checks only that every code line is covered,
that every function of the source is classed (in the kernel lines, the host lines or
`source.unused`; library issue L50), and that each piece of model text is in the kernel's Lean. A
`note` says what a reviewer needs to see the correspondence: why this operation (the IEEE `+`
because no product feeds it), which conversion, which condition of `Inst.ok`. An entry may carry `rules`, free labels such as the
SEMANTICS.md sections it relies on; nothing checks them.
