# Exclusions

This page lists what TVA does not cover: Triton features, source forms and launch situations
outside every claim. The trusted compiler statement (`TVA.Claim`, item 3 of "What a reader of a
proved claim trusts") is only true together with this list: it covers a launch only when the
launch is in none of the situations below.

An exclusion is one of three kinds:

- **A known miscompile.** Triton 3.8.0 compiles some source forms to code that computes something
  the source does not say. The trusted compiler statement does not cover them.
- **A feature without a meaning in the library.** `TVA.Triton` gives it no meaning, or gives it a
  meaning only in launches that avoid it, so no model can transcribe it.
- **A launch or host situation assumed not to happen.** Something outside the kernel's source (the
  host process, Triton's settings, other code) that would change what the launch computes. TVA
  verifies the kernel, not the Triton installation or the rest of the program, and nobody checks
  these.

A claim's model must not use an excluded feature, and a claim's launches must not be in an
excluded situation. The claim's author checks the kernel's source and the host code that launches
it against this list, and each claim's `README.md` lists, in its "Exclusions" section, the
exclusions that touch its kernel (for example a constexpr path that builds descriptors in the
kernel, F10, or a `launch_pdl` path, H2). The identifiers below (M1, F10, …) are for citing them.

## How each exclusion is kept out

Some exclusions are enforced by the model itself: a claim that covered such a launch could not be
proved, because `Claim.Holds` asks for `Correct`, which asks for a valid, well-behaved launch with
no undefined behaviour. These are marked *enforced* below:

- **`Launch.Valid`** (`TVA.Launch.Run`) requires `debug = false` (H1), `pdl = false` (no
  programmatic dependent launch, H2) and `numCtas = 1` (H3), besides CUDA's grid limits and block
  shapes Triton accepts. The claim's author transcribes these options from the launch.
- **`WellBehaved`** (`TVA.Launch.WellBehaved`) rules out conflicts within a program (`IntraRace`),
  races between programs (`InterRace`) and an atomic's result used on a shared cell
  (`UsesSharedAtomicResult`): every item of group C.
- **Undefined behaviour in a `tl.*` definition**: `tl.forRange` outside `tl.RangeOk` (F12, F13),
  `tl.histogram` out of range (F14), poison (F15), bitwise atomics on floats (F7), and a false
  compiler hint (D1, D2).

Every other exclusion is kept out by the claim's author, who reads the source and the host code
(*author checks*), or is an assumption about the host that nobody checks (*assumed*).

Contents: [M. Miscompiles](#m-miscompiles) ·
[F. Features without a meaning](#f-features-without-a-meaning-in-the-library) ·
[C. Concurrency without a meaning](#c-concurrency-without-a-meaning) ·
[P. Python outside the supported subset](#p-python-outside-the-supported-subset) ·
[H. Launch and host situations](#h-launch-and-host-situations-assumed-not-to-happen) ·
[D. Kernels excluded by name](#d-kernels-excluded-by-name).

## M. Miscompiles

**M1. A float reduction rewritten across the trips of a loop.** Out of scope is a launch whose
kernel has a float reduction (`tl.sum`, `tl.max`, `tl.min`, or any `tl.reduce` with a float
operand) along the last axis of a tile of rank 2 or more,

- at least one of whose operands equals, element for element up to NaN payload, the tile of one
  `tl.load` (of a pointer tensor or of a block pointer, with or without a mask, `other` or a cache
  modifier, in the kernel or in a helper; a descriptor's `load` is not one), on the launch's values
  of its compile-time constants and of the integer arguments Triton specialises (those equal to
  `1`), whatever operations stand between them and whether or not Triton folds them;
- anywhere in the body of a `for` loop over `range` or `tl.range` (with any options,
  `loop_unroll_factor` included): directly, in a helper the body calls, under an `if` (on a runtime
  or a compile-time condition), or in a `tl.static_range` loop inside it;
- whose result is used, through any chain of operations on values (not through memory), in a value
  that loop carries (a variable the body assigns that has a value before the loop),

whatever the carried value's initial value, the operation that updates it, the other uses of the
reduction's result or of the updated value, the width of the float dtype and the layout. A `while`
loop is not such a loop.

*Why.* Triton's `OptimizeThreadLocality` pass (on for NVIDIA, `nvidia/backend/compiler.py:276`)
matches (`OptimizeThreadLocality.cpp:252-372`) a `tt.reduce` directly in the body of an `scf.for`,
whose combine is one float `addf`, `mulf`, `maximumf`, `maxnumf`, `minimumf` or `minnumf`, whose
operand, of rank 2 or more in a blocked layout with more than one element per thread along the
last axis, is defined by a `tt.load`, whose result's one use is an operation `U` whose one use is
the loop's yield, and whose carried value starts from a constant `c`. It folds each thread's
partial values across the trips with `U` from the combine's neutral element and reduces across the
threads once after the loop (`OptimizeThreadLocality.cpp:393-520`), so the device computes
`U(c, R(v))`: a reassociation of the trips' elements when `U` is the combine, and no association
of the source at all otherwise (`acc - tl.sum(x, 1)` gives `+s` for `-s`). The pass reads the IR
after inlining, folding, specialisation, dead-code removal and unrolling, so which source forms
reach it depends on folding and the layout; the exclusion is therefore stated on the source and is
broader than the pass's pattern. [evidence/reduce-tree](../evidence/reduce-tree/README.md), part L:
`acc = acc + tl.sum(x, axis=1)` over two trips stored `2.0` in every row, where every association
of the source gives `1.0`. *Author checks.*

## F. Features without a meaning in the library

| | Excluded | Why | Kept out by |
|---|---|---|---|
| F1 | `tl.inline_asm_elementwise` | arbitrary PTX | author checks |
| F2 | `tl.float8e4b15` and the AMD fp8 formats (`float8e4b8`, `float8e5b16`) | not used on the H200; `DType` lacks them | author checks |
| F3 | pointers to pointers (`tl.pointer_type` of a pointer type as an element type) | needs a typed pointer memory; an integer converted to a pointer (`tl.intToPtr`) is covered | author checks |
| F4 | 3-D (batched) `tl.dot_scaled` | only the 2-D form is in `TVA.Triton.Dot` | author checks |
| F5 | the 64-bit Philox (`uint64` offsets as counters), `tl.randn` | `TVA.Triton.Random` defines the 32-bit generator only (`tl.randint`, `tl.randint4x`, `tl.rand`) | author checks |
| F6 | `%` on floats (`frem`) | not modelled | author checks |
| F7 | `tl.atomic_and`, `tl.atomic_or`, `tl.atomic_xor` on `float32` or `float64` elements | Triton 3.8.0 accepts them and performs the integer operation on the bit patterns (`and.b32`, …, `xor.b64`), but the library gives them no meaning: `bitUpd` has no new value for a float. Element types Triton rejects on the host need no exclusion: such a launch has no outcome | enforced: the model's atomic is undefined behaviour |
| F8 | aggregate classes a kernel or a library other than Triton's language defines (classes made with Triton's `_aggregate`, `triton.language.core`); an operator or a subscript applied to a block pointer | their operators, subscripts and methods are `@triton.jit` functions Triton inlines, so an operator on them may run any operation. Triton's own block pointer (`tl.make_block_ptr`) is covered through its operations (`TVA.Triton.BlockPtr`) | author checks |
| F9 | `tl.debug_barrier()` as synchronisation | the language has no threads. A barrier would only order accesses that already conflict within a program (C1), or two atomics on the same bytes, whose every order the semantics already allows (`Step.defer`) outside C7; inside C7 the launch stays excluded with or without a barrier | a model writes nothing for the call |
| F10 | a descriptor built in the kernel: a launch on which the front end visits a call of `tl.make_tensor_descriptor`, in the kernel or a helper (a call in a compile-time branch the instance does not take, such as tutorial 06's `_maybe_make_tensor_desc` given a host `TensorDescriptor`, is not visited) | each program writes such a descriptor into a global scratch buffer that the allocator set by `triton.set_allocator` returns, outside the claim's memory, and what the launch would need of that buffer (none of the launch's bytes, the alignment asked for, no rewrite of a descriptor rebuilt in a loop while an access through it is in flight) is outside the trusted compiler statement. `tl.make_tensor_descriptor` in the library is the call's meaning in the language only. Host `TensorDescriptor` arguments are covered | author checks |
| F11 | pointer and descriptor accesses to the same bytes | a pair with a write is already a conflict within the program (C1) or a race between programs (C2); the remaining cases are two reads, and a pointer atomic with a descriptor reduction (`desc.atomic_*`), whose atomicity across the generic and the TMA (async) proxy is not established | author checks |
| F12 | `range` or `tl.range` with a non-positive step | not modelled (constant negative steps follow Python's semantics, `code_generator.py:1246-1251`) | enforced: `tl.forRange` is undefined behaviour (`tl.RangeOk`) |
| F13 | `range` or `tl.range` with unsigned bounds at or above `2^(w−1)` | `scf.for` compares signed, so such a loop runs a different number of times | enforced: `tl.forRange` is undefined behaviour (`tl.RangeOk`) |
| F14 | `tl.histogram` with an enabled value outside `[0, bins)` | undocumented | enforced: undefined behaviour in `tl.histogram` |
| F15 | lanes that hold LLVM poison (`x.to(int)` out of range or from a NaN or an infinity, a shift by the width or more) but whose values are never used | poison is read conservatively as undefined behaviour; a kernel that converts unspecified masked-off lanes needs `other` or a `tl.where` before the conversion | enforced: undefined behaviour in `tl.ftoi`, `tl.shl`, `tl.ashr`, `tl.lshr` |
| F16 | a scalar kernel parameter annotated with a type other than `tl.int1`, `tl.int8`, `tl.int32` (or `int`), `tl.int64`, `tl.uint32`, `tl.float16`, `tl.bfloat16`, `tl.float32` and `tl.float64` (for example `tl.int16`, `tl.uint8`, `tl.uint64`) | the values the launcher passes for them are not established | author checks |
| F17 | an operator whose computation dtype is an fp8 dtype (`computationType?`: `x * 1.0` with an fp8 `x`, `x + y` or `x < y` with `x` and `y` of one fp8 dtype), and unary `-` on an fp8 tensor | the backend rejects most of them on the host, but MLIR's folders remove some first, which then run (`x * 1`, `1 * x`, `x / 1` and `x - 0` give `x`; an operation on two fp8 constants is computed at compile time), so a model that gave every such form no outcome would be wrong where a fold applies. Convert an fp8 value first (`tl.cvtFrom8`) | author checks |
| F18 | a compiler hint with a value that is not a power of two, or a `tl.max_contiguous` or `tl.max_constancy` value above the dimension's size | Triton documents powers of two, and the lowering asserts it for constancy (`ElementwiseOpToLLVMBase.h:92`); the library's readings of the hints are given for such values only | author checks |
| F19 | a compiler hint on one result of an operation with several results (`tl.split`, a tuple `tl.reduce` or `tl.associative_scan`, a loop, a runtime `if`) when another result has another shape, or, for `tl.multiple_of` and `tl.max_contiguous`, is a float tensor | the hint may apply to every result, and the library's readings (`HintVal`, `ContigIs`, `FirstsDiv`) are about integer and pointer tensors of the hinted value's shape (`ConstAlong`, `tl.max_constancy`'s, of the hinted value's shape) | author checks |

## C. Concurrency without a meaning

A launch runs its programs as an interleaving of whole steps over one memory
([SEMANTICS.md](SEMANTICS.md#launches)). That is the launch's meaning only when it is well behaved;
each pattern below makes it not well behaved, so every item here is *enforced* by `WellBehaved`.

| | Excluded | Why | Enforced by |
|---|---|---|---|
| C1 | a program that writes and reads back its own data, or updates an input in place: two global accesses of one program to the same bytes, at least one writing them (a store or an atomic), unless both are atomics on the same bytes, across loop iterations included | one program is not sequential for global memory: its threads access memory with no barrier, and Triton's pipelining reorders loop iterations | `IntraRace` |
| C2 | atomics on partly overlapping bytes (different widths at overlapping addresses), in one program or in two | not two atomics on the same bytes | `IntraRace`, `InterRace` |
| C3 | `relaxed`, `acquire`-only or `release`-only atomics on memory other programs touch, when the program uses the value they return | an interleaving would hide their reorderings. Those whose returned value the source never uses, at `gpu` or `sys` scope, are covered: their model is the `acq_rel` atomic (`MemSem.unused`, `tl.atomic_add_unused` and the others; [SEMANTICS.md](SEMANTICS.md#atomics)) | `InterRace` |
| C4 | `cta`-scope atomics used across programs | atomic only within a CTA | `InterRace` (`Event.Strong` asks for `gpu` or `sys` scope) |
| C5 | descriptor reductions (`desc.atomic_*`) into elements other programs also reduce into | TMA reductions are relaxed (`tl.desc_atomic`), and an interleaving would hide their reorderings. Covered when one program owns the elements | `InterRace` |
| C6 | data passed between programs through plain memory behind a flag or counter ("the last program reduces the partial tiles", semaphore split-K, a lock) | a scalar atomic is issued by one thread with no barrier before it, so its release does not publish the other threads' writes; the semantics has no synchronises-with, so such launches race | `InterRace` |
| C7 | a program that uses the result of a `tl.atomic_*` call (`add`, `xchg`, `cas`, `max`, `min`, `and`, `or`, `xor`; in a store, an address, a mask, a branch or any other operation that runs, of any lane, with any `sem`) when a cell that call reaches is also accessed by another program of the launch, or by another atomic operation of the same program (another `tl.atomic_*` call, or the same call run again in a loop; the lanes of one call are not another operation). This includes programs that wait on each other (locks, spin loops, tickets) and a counter that hands out work (`tile = tl.atomic_add(cnt, 1)`) | the semantics orders a program's steps: a tensor step that uses one lane of an atomic's result holds back every later atomic of the program (`Step.defer`), while on the device each thread orders only its own accesses, so the old value can show an order the semantics lacks | `UsesSharedAtomicResult` |

## P. Python outside the supported subset

TVA gives meaning to a subset of the Python that Triton's front end accepts: the forms listed
here, with the meanings [SEMANTICS.md](SEMANTICS.md#how-the-front-end-reads-the-source) gives them.
Every other form is excluded (P1), and a kernel that uses one is outside every claim until the
library adds the form. The claim's author checks every line the launch runs on the device or at
compile time. A form that Triton rejects on the host needs no meaning, since such a launch has no
outcome. *Author checks*, for every item of this group.

**Functions.**
- *The kernel*: a function defined by a `def` statement at the top level of the kernel's source
  file, under `@triton.jit`, or `@triton.jit(…)` with no arguments but `launch_metadata=` and
  `do_not_specialize=`.
- *Helpers*: functions defined by `def` at the top level of the kernel's source file under a bare
  `@triton.jit`, called by name from the kernel or from another helper.
- No other function runs at compile time (P2).

**Parameters and arguments.**
- A kernel parameter is a pointer (a `torch.Tensor` argument, `Ptr d`), a scalar (an `int`,
  `float`, `bool` or `None` argument, or a parameter annotated with a type F16 admits), a host
  `TensorDescriptor`, or a parameter annotated `tl.constexpr` whose argument is a Python `int`,
  `float`, `bool`, `str` or `None`, or a dtype of `tl`. It has no default value.
- A helper's parameter holds what the call passes, by position or by keyword, or its default; an
  annotation on it has no effect. A default is a literal (a number, `True`, `False`, `None` or a
  string), a tuple of literals, or a dtype of `tl`.

**Statements.**
- An expression statement: a call, or a string literal (a docstring).
- `name = e`; `a, b, … = e` with names only, exactly one per element of `e`'s tuple (a tuple
  expression, a helper's tuple result, `tl.split`); `name op= e` with `op` one of the binary
  operators below; and `name: tl.constexpr = e` on a name not yet defined (P3).
- `if`, `elif`, `else`, on a compile-time condition (a `bool`, an `int` or `None`, as a
  `tl.constexpr` holds it) or on a runtime one (P4).
- `for name in range(…)` with Python's `range` given one, two or three positional arguments and no
  keyword (P7), `for name in tl.range(…)`, `for name in tl.static_range(…)`, and `while c:` with a
  runtime condition (P5).
- `return`: in a helper, `return`, `return e` or `return a, b, …`; in the kernel, `return` with no
  value (P6, P9).
- `pass`; `assert test, msg`.

**Names.**
- The function's parameters and the names its statements assign (P4, P5).
- At the top level of the module: Triton's modules as the source imports them (`triton`, and
  `triton.language` as `tl`), and the module's `@triton.jit` functions, each bound only by its own
  `import` or `def` statement (not by an assignment, a `:=` in a decorator, a default or an
  annotation of another function, or a second `def` or `import` of the name).
- Python's builtins `float`, `int`, `isinstance`, `len`, `list`, `getattr`, `hasattr`, `min`,
  `max`, `print` and `range`.

**Expressions.**
- Literals: an `int`, `float`, `bool`, `str` or `None`; a tuple `(…)` or a list `[…]` of these
  expressions; an f-string of compile-time values.
- Attributes: a tensor's `.dtype`, `.type` and `.shape`, `.T` of a rank-2 tensor, `.element_ty`
  of a pointer type or of a block's type, a dtype's predicates (`.is_fp8()`, …), the methods
  below, and the functions, dtypes and classes of `tl` (`tl.float32`, `tl.pointer_type(d)`,
  `tl.math.exp2`, `tl.tensor_descriptor` in `isinstance`).
- Subscripts: `None` and `:` in each position of a tensor's subscript (`x[:, None]`,
  `x[None, :]`); a compile-time integer subscript of a shape or a tuple (`acc.shape[0]`).
- Operators: `+ - * / // % & | ^ << >>`, the comparisons `< <= > >= == !=`, one per expression,
  `is` and `is not`, `and`, `or` and `not`, unary `-` and `~` (not those of F17). With only
  compile-time operands, the operator is Python's on their values, computed at compile time
  (`BLOCK_SIZE_N // 2`, `ACTIVATION == "leaky_relu"`), and `**` is supported only so.
- `a if c else b`.
- Calls: of the `tl.*` functions, and of the methods of tensors, descriptors and block pointers,
  that `TVA.Triton` gives a meaning, with arguments of the kinds they state (an argument they do
  not state is excluded); of the kernel's helpers; of the builtins above.

**On the host**, the kernel is launched directly, as `kernel[grid](…)`, the grid a tuple or a
function of the meta-parameters; every other way of launching it is excluded (H6).

The forms the subset leaves out that the front end handles in a way a reader would not expect:

| | Excluded | Why |
|---|---|---|
| P1 | any form the list above does not give, for instance `break`, `continue`, `with`, `del`, `global`, `nonlocal`, `try`, `raise`, `import` in a function, a nested `def` or `class`, `lambda`, comprehensions, `:=`, `for … else` and `while … else`, an assignment to a subscript or an attribute, `*` and `**` in a call or a signature, a slice with bounds or an integer index of a tensor, chained comparisons, a tuple assignment whose names and elements differ in number, `@triton.jit(noinline=True)`, a kernel parameter's default, an enclosing function's names, a helper defined in another file, `@triton.heuristics` and `@triton.autotune` (H10, H12) | TVA gives meaning to the forms its claims' kernels use and the generalisations SEMANTICS.md states. A tuple assignment with too many elements drops the extra ones silently |
| P2 | a `@triton.constexpr_function`, or another Python function a kernel or helper calls (of its module, of another module, a method of a Python object), other than the builtins above and Triton's `tl.*` functions; a value at the top level of the module other than Triton's modules and the module's `@triton.jit` functions (a `tl.constexpr(…)` global, a dtype, a number, a namedtuple) | Python runs such a function at compile time on the front end's objects, and the library gives its result no meaning |
| P3 | an annotated assignment other than `name: tl.constexpr = e` on a name not yet defined: a bare annotation `name: T` (it changes the next assignment to `name`), and an annotation other than `tl.constexpr` (its value is evaluated twice, and the annotation is otherwise ignored: `a: tl.int64 = 2**31 - 1` is an `int32`) | the front end keeps pending bare annotations, and for another annotation visits the value and then visits it again as a plain assignment (`code_generator.py:694-760`): `a: tl.int32 = tl.atomic_add(c, 1)` runs two atomics in each program |
| P4 | in a branch of a runtime `if`, an assignment to a name that holds, before the `if`, a value other than a tensor (`None`, a dtype, a tuple, a string; a `tl.constexpr` there is rejected on the host) | the front end does not merge such a name as it merges a tensor: after `y = None; if n > 0: y = tl.load(p)`, `y` is still `None` (`code_generator.py:858-861`) |
| P5 | after a loop (`for` over `range`, `tl.range` or `tl.static_range`, or `while`), a use of a `for` loop's variable, or of a name first defined in the loop's body | after a loop over `range` or `tl.range` the front end leaves the variable a placeholder with no defined value, not the last value, and a name first defined in the body is not defined (a use is rejected on the host); after `tl.static_range`, whose copies are statements in sequence, both hold the last copy's values, which the subset does not admit either, to keep one rule for every loop ([SEMANTICS.md](SEMANTICS.md#loops)) |
| P6 | a `return` inside a `tl.static_range` body, and a statement that comes, in the same statement list, after an `if` that returns on the instance (a compile-time `if` whose taken branch has a `return` or an `if` that returns among its statements, or a runtime `if` whose two branches each have one) | the front end stops visiting only the statements after a `return` in its own statement list, and visits what follows the statement that holds it into a block no program reaches (`code_generator.py:508-517, 544-554, 893-929`): that code never runs, but its hints mark operations defined before the `return` (an earlier load vectorised and read wrong values or faulted), its `tl.static_assert`s are evaluated, and in a helper its `return`s take part in the return type. The statements after a `return` in its own statement list are not compiled and may stay; a runtime `if` that returns in one branch only is covered |
| P7 | Python's `range` with a keyword argument (`range(0, n, step=2)`) or more than three arguments | the front end evaluates the keywords and the extra arguments and ignores them (`code_generator.py:1207, 1237-1243`): with `n = 8`, `range(0, n, step=2)` runs 8 trips and `range(0, n, 2, 99)` 4, where Python rejects both. `tl.range(0, n, step=2)` is in the subset |
| P8 | two calls of one helper, in one compilation, whose compile-time arguments differ but give the same `mangle_fn` name (the `repr` of each, with `[`, `]` written `_` and `'` written `_sq_`: two NaNs of other patterns, `"x[y"` and `"x_y"`) | the front end compiles one copy per name, and the later call uses the copy made for the earlier call (`code_generator.py:32-41`): a helper called with `-float('nan')` after `float('nan')` stores `0x7fc00000`. This is the front end's naming within one compile, not a cache |
| P9 | a helper whose `return`s, among those the front end visits on the instance, are neither one `return`, nor all tensors of one type, nor all the same compile-time value (the same `None`, `int`, `float` bit for bit, `bool`, string or dtype, of one Python type, element by element for a tuple: not `(1, 2)` with `(1.0, 2)`, not `0.0` with `-0.0`, not two NaNs); and a helper that can reach the end of its body without a `return` while a visited `return` gives a value | the front end gives the call one type from every visited `return` (`decide_return_type`, `code_generator.py:556-626`): `==`-equal compile-time values give the first `return`'s value on every path, other Python numbers become scalar tensors of their own type, and a path that falls off the end gives the compile-time value or an unspecified `poison` value, so the call is not the value of the `return` its program takes as Python reads it. A conditional expression on a runtime condition, whose sides become tensors, is in the subset |

## H. Launch and host situations assumed not to happen

The trusted compiler statement (`TVA.Claim`, item 3) covers a launch of the kernel's own
`@triton.jit` function launched directly, `kernel[grid](…)`, in the setting it names. These items say precisely what falls outside that setting.

| | Excluded | Why | Kept out by |
|---|---|---|---|
| H1 | `debug=True`, however it is set: the launch's or a config's `debug`, `@triton.jit(debug=True)`, the kernel's `JITFunction` attribute set from Python (`kernel.debug = True`), `TRITON_DEBUG`, or the knob `triton.knobs.runtime.debug` | asserts and integer overflow checks trap (with the attribute set, an `int32` overflow trapped) | enforced for the launch's own options: `Launch.Valid` requires `debug = false`; the attribute set from outside the kernel's source, the environment variable and the knob are assumed |
| H2 | programmatic dependent launch (`launch_pdl=True`) | the kernel may start before the previous kernel on the stream has finished writing, so its programs may read memory other than the claim's `init`. Without it, `tl.extra.cuda.gdc_wait()` changes nothing and a model drops it | enforced: `Launch.Valid` requires `pdl = false` |
| H3 | `num_ctas > 1` | a program is a cluster of CTAs; `cta`-scope atomics are atomic only within one CTA | enforced: `Launch.Valid` requires `numCtas = 1` |
| H4 | a kernel that calls `tl.extra.cuda.gdc_launch_dependents()` followed on its stream by a launch with `launch_pdl=True` that writes a byte the kernel is given | that launch may start while the kernel still runs, and write memory the kernel reads. Without such a next launch, `gdc_launch_dependents` changes nothing and a model drops it | assumed (H5) |
| H5 | anything else that writes a byte the launch is given (its `init`, the buffers of `Pre`) from the launch's start to its end: the host, a kernel on another stream or device, a later kernel that programmatic dependent launch starts early | the claim is about the launch's own outcome from its `init`. Writes to other bytes, concurrent or not, do not matter: the device's stores are byte-granular | assumed (the trusted compiler statement states it) |
| H6 | any launch other than a direct `kernel[grid](…)`: a compiled kernel launched by hand, whether `kernel.warmup(…)` returned it or `triton.compile` built it (from the kernel's source or from a file: an IR file, `.ttir`, `.ttgir` or `.ptx`), and its `run` or launcher called directly | a compiled kernel's launcher checks nothing against the specialisation it was compiled for (another dtype, alignment or integer class: such launches fault, write past their bounds, read a stride as `1` or wrap integers), and `triton.compile` of a file compiles that file instead of the source (`compiler.py:87-112, 233-240`); TVA keeps one launch path, the direct one | author checks |
| H7 | a Triton knob not at its default, set by an environment variable (`TRITON_F32_DEFAULT`, `TRITON_DEFAULT_FP_FUSION`, `TRITON_INTERPRET`, `TRITON_KERNEL_OVERRIDE`, `TRITON_OVERRIDE_ARCH`, `TRITON_LIBDEVICE_PATH`, `TRITON_PTXAS_PATH`, `PTXAS_OPTIONS`, `DISABLE_PTXAS_OPT`, `DISABLE_LLVM_OPT`, `TRITON_CACHE_MANAGER`, `TRITON_REMOTE_CACHE_BACKEND`, …) or from Python (`triton.knobs.language.fp32_default = …`, `triton.knobs.compilation.instrumentation_mode`, …); every knob of `triton.knobs` (`knobs.py:320-513`) and every environment variable Triton's C++ reads (`GetEnv.h:15-57`) counts, except those that only name Triton's home, cache and dump directories, only dump, print or time, or only change debug information: `TRITON_HOME`, `TRITON_CACHE_DIR`, `TRITON_DUMP_DIR`, `TRITON_KERNEL_DUMP`, `TRITON_ALWAYS_COMPILE`, `TRITON_PRINT_AUTOTUNING`, `TRITON_DUMP_PTXAS_LOG`, `MLIR_ENABLE_DUMP`, `MLIR_DUMP_PATH`, `LLVM_IR_ENABLE_DUMP`, `NVPTX_ENABLE_DUMP`, `MLIR_ENABLE_TIMING`, `LLVM_ENABLE_TIMING`, `TRITON_DISABLE_LINE_INFO`, `USE_IR_LOC`, and the knobs they set | a knob changes defaults the model fixes (`fp32_default` is the `input_precision` of a `tl.dot` that gives none, `tf32` in `DotCfg`; `default_fp_fusion` the default of `enable_fp_fusion`), or replaces the compiler's output | assumed |
| H8 | a hook: `triton.knobs.runtime.launch_enter_hook`, `launch_exit_hook`, `kernel_load_start_hook`, `kernel_load_end_hook`, `kernel_unload_hook`, `jit_cache_hook`, `jit_post_compile_hook`, `add_stages_inspection_hook`, `knobs.compilation.listener`, `knobs.autotuning.listener` (`knobs.py:470-492`), and a pre-run hook of the kernel's `JITFunction` (`jit.py:735-736`) | a hook can rewrite any compiler stage (a stages hook turned `x + 1` into `x - 1`), or make the call return with nothing launched (`jit.py:757-759, 880-882`). `launch_metadata=` is harmless without a hook | assumed |
| H9 | a compile or launch option that `Launch` does not record (`ir_override`, `allowed_dot_input_precisions`, `default_dot_input_precision`, `enable_reflect_ftz`, `extern_libs`, `ptx_options`, `ptx_version`, `arch`, `warp_size`, `supported_fp8_dtypes`, `deprecated_fp8_dot_operand_dtypes`, `max_num_imprecise_acc_default`, `sanitize_overflow`, `instrumentation_mode`, `backend_name`), given to the call that compiles the kernel (a launch, `kernel.warmup(…)`) or in a `triton.Config`; and `JITFunction.preload` called on the kernel in the process. The options allowed are `enable_fp_fusion`, `debug`, `num_ctas` and `launch_pdl` (`Launch.fusion`, `debug`, `numCtas`, `pdl`) and the four that do not change the meaning: `num_warps`, `num_stages`, `maxnreg` and `launch_cooperative_grid` | it replaces the code or changes what an operation computes (`ir_override` gave `x + 2` for `x + 1`; `allowed_dot_input_precisions` switched a `tl.dot` to `ieee`; `enable_reflect_ftz=False` changed `tl.math.exp2(-130)` from `0` to `2^-130`); an option given at `warmup` reaches every launch of the compiled kernel it returns. `preload` compiles the kernel with the options its data names and stores it under the cache key its data names, so a later launch with no option runs that code (`jit.py:823-875`) | author checks the options; `preload` is assumed |
| H10 | `@triton.heuristics` | a heuristic sets launch keywords before the JIT reads them, compile options included, and replaces a launch keyword or a config's value of the same name (`autotuner.py:473-483`): a heuristic `enable_fp_fusion: True` fused `x*y+z` under a launch's `enable_fp_fusion=False` | author checks |
| H11 | a host `TensorDescriptor` with `round_f32_to_tf32=True` | the launcher builds a TMA descriptor whose loads round `float32` to tf32 (`nvidia/backend/driver.py:231-232`: `1 + 2^-23` is loaded as `1.0`), where `tl.desc_load` returns the stored values | author checks |
| H12 | the autotuner: `@triton.autotune` on a kernel, and a call of an autotuned kernel with its benchmark runs, key, pruning, `reset_to_zero` and `restore_value`, hooks, `do_bench` and choice of a config | TVA verifies the kernel, not the autotuner: such a call runs the kernel many times on the caller's buffers, and its hooks write them (`autotuner.py:133-173, 217-282`). For a kernel its source autotunes, a claim fixes one config: its `source/` is the original without the decorator, and its launch passes the config's values and does what the config's `pre_hook` does | author checks (outside the subset, P1) |
| H13 | a target other than the H200 (NVIDIA sm_90): AMD GPUs and Blackwell (sm_100, sm_110); code Triton compiled for another target than the device it runs on (`triton.compile`'s `target`, the `arch` option, `TRITON_OVERRIDE_ARCH`); a host other than an x86-64 machine running CPython 3.13; and a host process run with `-O` or `-OO` or with `PYTHONOPTIMIZE` set, or in another floating-point environment than the default (another rounding mode by `fesetround`, flush-to-zero or denormals-are-zero set by `_mm_setcsr` or by a loaded library, such as one built with `-ffast-math`) | the device facts are the H200's, and some values SEMANTICS.md states are the host's (a NaN passed as a `tl.float16` argument, by CPython 3.13's `PyFloat_Pack2`; the sign of Python's `float('inf') - float('inf')`, x86-64's default NaN). `-O` removes the Python `assert`s by which Triton's front end rejects some kernels on the host (`code_generator.py:497, 872, 886, 1020, 1117-1121`), and the floating-point environment decides the constants Triton builds (`FloatV.litF32`, `litF16`, `litBF16`) and the `float32` of a `float` argument (`FloatV.hostF32`): after `fesetround(FE_TOWARDZERO)` the constant `0.1` is `0x3dcccccc`, and with flush-to-zero `1e-40` is `0` | assumed |
| H14 | Triton or the kernel's module changed by code of the process: Triton's installed files edited, another `triton` imported (earlier on `sys.path`, a substitute in `sys.modules`); a change, by any code but Triton's own, to an attribute of a Triton module, of a class Triton defines or of an object of one, or of the kernel's or a helper's `JITFunction`, by assignment, `setattr`/`delattr` or a call made to change it (`tl.exp = tl.log`, `tl.arange = …`, a `JITFunction`'s `fn`, `src` or `noinline`, its `_unsafe_update_src`, `triton.runtime.driver.set_active`, `importlib.reload` of a Triton module), other than through Triton's API as launches use it (launches, `kernel.warmup(…)`, `triton.set_allocator`, the knobs H7 allows, the fields of a host `TensorDescriptor` the launch is given, such as the `block_shape` a config's `pre_hook` sets); a name of the kernel's module bound by anything but the module's own statements (`kmod.double = kmod.triple` in another module, an assignment into `kernel.fn.__globals__`) | the front end compiles what these objects and names hold, so the launch computes something other than its source (`3·x` for `2·x`, `log x` for `exp x`). A kernel source that itself rebinds such a name is outside the subset (P1) | assumed |
| H15 | a launch that runs code other than the code Triton 3.8.0 compiles for that launch, from its source, its own argument values (compile-time and runtime, as the claim's instance gives them) and the settings of the other items. Code taken from Triton's caches (the `JITFunction`'s cache in the process, or the cache directory) may be such code when it was compiled earlier for other compile-time values that share the launch's key (in the process, values Python's `==` equates and every NaN; on disk, the `str`s of the values joined with `-`), or under other knobs, `TRITON_KERNEL_OVERRIDE`, a hook, another state of Triton or another state of the host process; code a cache returns counts when it is that code, for example for a repeated launch of the same instance | the caches' keys equate such values and record none of those settings (`jit.py:592-614`, `compiler.py:71-76, 244-270`): `C = 1.0` after `C = 1` runs the integer kernel, `MODE = "1"` after `MODE = 1` runs `MODE = 1`'s code. TVA does not verify Triton's caches: the trusted compiler statement assumes the kernel compiled afresh | assumed |

## D. Kernels excluded by name

Kernels of Triton's tutorials that the items above exclude, named so that nobody writes a claim for
them.

| | Kernel | Excluded by |
|---|---|---|
| D1 | tutorial 08's `grouped_matmul_kernel` | its `tl.multiple_of(a_ptrs, [16, 16])` and `tl.multiple_of(b_ptrs, [16, 16])` promise that every pointer of the block is 16-byte aligned, which no launch meets (`core.py:3307-3330`: along a dimension of contiguity 1, every element), so the kernel has no defined meaning on any launch. *Enforced*: a false hint is undefined behaviour in the model |
| D2 | tutorial 09's pointer kernels (`matmul_kernel`, `matmul_kernel_persistent`) on partial tiles (`M` or `N` not a multiple of the config's `BLOCK_SIZE_M`, `BLOCK_SIZE_N`) | their `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)` (and `offs_bn`) is false on a partial tile, where the compiled loads read other elements than the source names. *Enforced*: a false hint is undefined behaviour; covered on full tiles |
| D3 | tutorial 05's `_layer_norm_bwd_dx_fused` | it hands partial sums between programs under a spin lock and updates them in place within a program (C1, C6, C7); tutorial 05's `_layer_norm_bwd_dwdb` is not excluded |
| D4 | tutorial 11 (programmatic dependent launch) | H2 |
| D5 | tutorial 08's `grouped_matmul_tma_kernel`, tutorial 09's `matmul_kernel_descriptor_persistent`, and tutorial 06's `_attn_fwd` on the launches where its host passes tensors rather than host `TensorDescriptor`s (the warp-specialised path) | F10: they build descriptors in the kernel. Tutorial 09's `matmul_kernel_tma_persistent`, tutorial 06's `_attn_fwd` with host descriptors and tutorial 10 are not excluded |
| D6 | tutorial 10's `block_scaled_matmul_kernel_cdna4`, and tutorial 10's `block_scaled_matmul_kernel` on Blackwell | H13. On the H200, `tl.dot_scaled` runs through Triton's emulation and is covered |
