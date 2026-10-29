# Semantics

This page says what each construct of a Triton 3.8.0 kernel means and points to the Lean
definition. It states every fact about Triton a model relies on: what each `tl.*` operation and
operator computes, how Triton's front end evaluates the source (which parts of an expression run,
which branches it visits), which constants it builds, which shapes its compiler rewrites, and what
launches, memory and outcomes mean. Library reviewers judge these facts once, against Triton's
sources and the device. A kernel's reviewers trust them, and judge whether the model means what
the source does.

The facts are about TVA's target (library issues L74, L107): Triton 3.8.0 on an NVIDIA H200
(sm_90), driven from an x86-64 host running CPython 3.13, as TVA's evidence ran
(`Compiler.Faithful`'s scope 7). A few of them are the host's rather than Triton's, and may differ on
another host: the bits of a NaN passed as a `tl.float16` argument (CPython's `PyFloat_Pack2`,
"Scalar arguments"), and the sign of the NaN Python computes for `float('inf') - float('inf')`
(x86-64's default NaN, "Constants"). The host process is CPython 3.13 run without `-O` or `-OO` and
with `PYTHONOPTIMIZE` unset, in the default floating-point environment (round to nearest even;
flush-to-zero and denormals-are-zero off: no `fesetround`, no `_mm_setcsr`, no loaded library that
changes them), as TVA's evidence ran (library issue L110): `-O` removes the Python `assert`s by which
Triton's front end rejects some kernels this page states as rejected on the host, and the
floating-point environment decides the constants Triton builds, the `float32` of a `float` argument
and compile-time float arithmetic, all computed on the host
([evidence/host-round21](../evidence/host-round21/README.md) H1–H3). The target and the host are
part (B) of `Compiler.Faithful`'s scope, the environment, which TVA assumes and nobody checks
("Launches").

[TRANSCRIPTION.md](TRANSCRIPTION.md) is guidance on writing and checking a model (worked
patterns, pitfalls, common reasons to drop a line); it states no fact of its own, and whatever
it says about Triton rests on a fact of this page (library issue L36). What is out of scope is in
[exclusions.md](exclusions.md). The page gives meaning only to the forms of Python listed in "The
supported subset" (below; library issue L88): a kernel that uses any other form is out of scope.

This page and exclusions.md are fact pages: their whole text is in the digest of every kernel
item whose cone reaches the library, so a change here re-opens those items
([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §4).

## Values (`TVA/Lang/Scalar.lean`, `Tensor.lean`)

| Triton | Lean | Meaning |
|---|---|---|
| `tl.int8 … tl.int64`, `tl.uint8 … tl.uint64` | `IntV sg w` (`I32`, …) | `w`-bit two's complement; `+ - *` wrap; `//`, `%` and shifts return `Option` (`none` = undefined) |
| `tl.int1` | `Bool` | |
| `tl.float16/bfloat16/float32/float64` | `FloatV f` (`F16`, `BF16`, `F32`, `F64`) | IEEE bit patterns; `+ - * fma div_rn sqrt_rn` round to nearest even |
| `tl.float8e5`, `tl.float8e4nv` | `F8E5`, `F8E4` | conversions into them at run time are platform relations (`tl.cvt8`); out of them exact on finite values (`tl.cvtFrom8`); a Python number built in them is the exact value of its `float32` constant (`FloatV.litF32`), or the kernel is rejected (`FloatV.litF8`, "Constants") |
| a block `tensor[s]` | `Tensor s α` | a function from indices `Idx s` to values |
| a pointer `*T` | `Ptr d` | a byte address; `p + i` advances by `i * sizeof(d)` |

`tl.constexpr` values, including shapes, are Lean values: the model is a Lean function of
them.

- **Operators on tensors** mean the library's operation for the operator and the operands'
  dtype: wrapping integer arithmetic and comparisons on `Tensor` (`+`, `*`, `lt`, `ltS`, …), and
  the float operations of `tl` (`tl.add`, `tl.mul`, …, below). The library's operations take
  operands of one dtype and one shape, so Triton's implicit steps are explicit operations of the
  model: dtype promotion (`tl.convert` to the dtype `TVA.Lang.Promote` gives, "Operators and
  promotion" below), `.to(…)` and the conversion of a stored value (`tl.cvt`, `tl.itof`, …,
  "Float conversions" below), scalar–tensor arithmetic and broadcasting (`bc`, `.col`/`.row`;
  `x[:, None]` and `x[None, :]` add an axis and keep the others).
- **Index arithmetic.** `tl.program_id` and `tl.arange` give `int32` values, and arithmetic on
  them stays `int32` (wrapping) unless an operand promotes it ("Operators and promotion"): a cast
  (`.to(tl.int64)`), or an operand of a wider type, or of an unsigned type at least 32 bits wide,
  such as a runtime `int64`, `uint32` or `uint64` scalar argument (a `uint8` or `uint16` operand
  does not promote `int32`). A scalar argument's type depends on its value: an integer `1` is a
  compile-time constant, another integer in `[-2^31, 2^31)` an `int32` ("Launches", "Scalar
  arguments").
- **Pointers.** A pointer's element type is its buffer's (`Ptr d`). A pointer built from a loaded
  integer (`.to(tl.pointer_type(…))`) is `tl.intToPtr`. Pointers to pointers are excluded.
- **Float operations are `Prog`s** (`let y ← tl.neg x`), because a NaN result may be any NaN
  (below, "Float results are exact up to NaN payload").

## Programs (`TVA/Lang/Prog.lean`, `TVA/Sem/Step.lean`)

A model is a `Prog`, written in `do` notation. Its constructors, and their steps:

| Constructor | Step |
|---|---|
| `load d s addrs k` | Each enabled lane (`addrs i = some a`) reads its element; disabled lanes get any value. If an enabled address is unreadable (unaligned, or a byte is absent): undefined behaviour. Event: reads the enabled bytes. |
| `store d s addrs vals k` | The enabled lanes are written in some order, so on a duplicate address the later lane wins. An unwritable address is undefined behaviour. Event: writes the enabled bytes. |
| `rmw d a upd sem scope k` | One atomic read-modify-write: the old value `o`, any new `n` with `upd o n`, returns `o`. If no `n` exists, or the address is unreadable: undefined behaviour. Event: reads and writes the element's bytes, names its cell. It may also stay pending while a later step goes first (`Step.defer`, below). |
| `pick P k` | Continue with any `b` satisfying `P b`; if there is none, undefined behaviour. |
| `ub` | Undefined behaviour. |
| `loop init body k` | Unfold one iteration (`Prog.loopStep`). |
| `env k` | Read the environment (platform `P`, `fusion`). |

A program is **not sequential for global memory** (library issue L15, user decision
2026-10-27). The threads that implement a program access memory with no barrier between its
operations. On the H200, a program that stores a block and loads it back reversed reads stale
data in 23–41% of the elements, and a loop that loads the row the previous iteration stored reads
stale data in 97% of the reads when Triton pipelines it (`tl.range(num_stages=3)`, or a `tl.dot`
in the body); a barrier, the same offsets, or no pipelining gives the sequential result
([evidence/intra-program](../evidence/intra-program/README.md); the first reproduction is
`issue/stale_readback.py`). So the semantics does not order a program's accesses to the same
bytes, and a model that has them is not well behaved:

- **Conflict within a program** (`IntraRace`). Two events of one program that touch the same
  bytes, at least one writing them (a store or an atomic), conflict, unless both are atomics on
  the same bytes (`Event.SameAtomic`). Program order does not excuse the pair, and a conflict
  across loop iterations counts. A plain load and an atomic on the same bytes conflict too
  (tutorial 05 loads `Count` and then `atomic_xchg`s it). A model with a conflict anywhere is
  not well behaved, like a race between programs, and no claim covers it. Kernels that write and
  read back their own data, or update an input in place, are out of scope
  ([exclusions.md](exclusions.md)).
- **Atomics of a program in any order** (`Step.defer`). Two atomics of one program on the same
  bytes do not conflict, but the device may perform them in either order: two atomics of one
  program on one address ran in the reverse of source order on 0.33–0.76% of the addresses, each
  still atomic ([evidence/intra-program](../evidence/intra-program/README.md)). An `rmw` may
  stay pending while the continuation takes a step that does not depend on its old value (for
  every old value, the same step, the same memory and the same event); it is performed later. So
  the semantics performs a program's atomics in every order their data and control dependences
  allow, and the old values they return, and what `xchg` and `cas` leave, reflect that order.

The steps are otherwise those of the source, in order: a well-behaved model never depends on the
order of two accesses to the same bytes. A program's other accesses (loads, stores, atomics on
different bytes) may also reach memory in another order on the device; in a well-behaved launch
no program can observe that, except through atomics, whose every order the semantics allows.

## The supported subset

TVA gives meaning to a subset of the Python that Triton's front end accepts (library issue L88;
user, 2026-10-28): the forms listed here, which are those the in-scope tutorial kernels use
(`tutorials/inventory.json`, without the kernels [exclusions.md](exclusions.md) excludes), and a
few generalisations this page states. Every other form is excluded ([exclusions.md](exclusions.md),
"Python outside the supported subset"): a kernel that uses one has no claim until the library adds
the form. The kernel reviewer checks the kernel against this list, every line the launch runs on
the device or at compile time ([REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) §1). A form that Triton
rejects on the host needs no meaning, since such a launch has no outcome ("Launches"); the list
names some only to say so. Each item points to the section that gives its meaning.

**Functions.**
- *The kernel*: a function defined by a `def` statement at the top level of the kernel's source
  file, under `@triton.jit`, or `@triton.jit(…)` with no arguments but `launch_metadata=`
  ("Launches") and `do_not_specialize=` ("Scalar arguments"), with or without one
  `@triton.autotune(…)` above it: its `configs` (`triton.Config(kwargs, …)` with only the options
  `Compiler.Faithful` admits), its `key`, and its other arguments, which run on the host and choose
  the config ("One launch").
- *Helpers*: functions defined by `def` at the top level of the kernel's source file under a bare
  `@triton.jit`, called by name from the kernel or from another helper ("Helpers are inlined").
- No other function runs at compile time: no `@triton.constexpr_function`, and no Python function
  that a kernel or a helper calls (of its module, of another module, a method of a Python object)
  other than Python's builtins and Triton's functions below.

**Parameters and arguments.**
- A kernel parameter is a pointer (a `torch.Tensor` argument, `Ptr d`), a scalar ("Scalar
  arguments": an `int`, `float`, `bool` or `None` argument, or a parameter annotated with a type
  listed there), a host `TensorDescriptor` ("Tensor descriptors"), or a parameter annotated
  `tl.constexpr` whose argument is a Python `int`, `float`, `bool`, `str` or `None` (a Lean
  parameter of the model). It has no default value. That a launch runs code compiled for its own
  compile-time values, not code from Triton's caches, `Compiler.Faithful`'s scope 11 assumes (the
  kernel is new; "Compile-time values Triton keys apart").
- A helper's parameter holds what the call passes, by position or by keyword, or its default; an
  annotation on it has no effect ("Helpers are inlined"). A default is a literal (a number, `True`,
  `False`, `None` or a string), a tuple of literals, or a dtype of `tl`. Two calls of one helper
  whose compile-time arguments differ but share the helper copy's name are outside the subset
  ("Compile-time values Triton keys apart"; library issue L103).

**Statements.**
- An expression statement: a call, or a string literal (a docstring); its value is discarded.
- `name = e`; `a, b, … = e` with names only, exactly one per element of `e`'s tuple (a tuple
  expression, a helper's tuple result, `tl.split`); `name op= e` with `op` one of the binary
  operators below ("A name assigned a Python number holds a tensor"); and
  `name: tl.constexpr = e` ("Annotated names").
- `if`, `elif`, `else`, on a compile-time condition ("`if`, `elif`, `else` on a compile-time
  condition": a `bool`, an `int` or `None`, as a `tl.constexpr` holds it; another compile-time
  value is rejected on the host, `code_generator.py:986-993`) or a runtime one ("A runtime `if`").
- `for name in range(…)` with Python's `range` given one, two or three positional arguments and no
  keyword (the front end ignores a keyword and a fourth argument; library issue L90),
  `for name in tl.range(…)`, `for name in tl.static_range(…)`, and `while c:` with a runtime
  condition ("Loops and unrolling", "Loops").
- `return` ("`return`"): in a helper, `return`, `return e` or `return a, b, …`; in the kernel,
  `return` with no value. Not inside a `tl.static_range` body; and no statement comes, in the same
  statement list, after an `if` that *returns* on the instance: a compile-time `if` whose taken
  branch has, among its own statements, a `return` or an `if` that returns, or a runtime `if` whose
  two branches each have one. The front end compiles what follows such an `if`, though no program
  runs it ("`return`"); the statements after a `return` in its own statement list it does not
  visit, and they may stay (library issue L89). In a helper, the `return`s the front end visits on
  the instance are one of these (library issue L102): one `return`; or `return`s that all give
  tensors (blocks, scalars of a Triton dtype, pointers, descriptors, or tuples of them) of one type;
  or `return`s that all give the same compile-time value: the same `None`, `int`, `float` (bit for
  bit: not a NaN, and not `0.0` with `-0.0`), `bool`, string or dtype, of one Python type, and for a
  tuple the same element by element (`(1, 2)` with `(1, 2)`, not with `(1.0, 2)`). And when a visited `return` gives a value
  other than `None`, no path of the body on the instance reaches its end without a `return` (a
  runtime `if` with a `return` in one branch only, and nothing after it, is such a path). Any other
  helper is outside the subset ("`return`" says what the front end makes of it).
- `pass`; `assert test, msg` ("Debugging calls").

**Names.**
- The function's parameters and the names its statements assign. Across a runtime `if`, each name a
  branch assigns is new to the `if` or holds a tensor before it (a block, a scalar of a Triton
  dtype or a pointer; "A runtime `if`"). After a loop, a name is used only if it was defined before
  the loop (a `for` loop over `range` or `tl.range`, or a `while` loop, carries it; after
  `tl.static_range` it holds what the last copy left): a name first defined in the body, and a `for`
  loop's variable, are not used after the loop, whatever the loop ("Loops and unrolling"; library
  issue L92).
- At the top level of the module: Triton's modules as the source imports them (`triton`, and
  `triton.language` as `tl`), and the module's `@triton.jit` functions. Each such name is admitted
  only as its own `import` or `def` statement binds it, and no other statement of the module may
  bind it: not an assignment, not a `:=` in a decorator, a default or an annotation of another
  function (Python evaluates the decorators and defaults when it runs the `def`, at import, whether
  or not anything calls the function, and Python 3.13 the annotations too), not a second `def` or
  `import` of the name. Nor may any code outside the module bind it (another module's
  `kmod.double = kmod.triple`, an assignment into a JIT function's `fn.__globals__`), and the
  process imports Triton 3.8.0 as installed, whose modules, classes, objects and `JITFunction`s no
  code but Triton's own changes (`tl.exp = tl.log`, `setattr`, `_unsafe_update_src`), other than
  through Triton's API as the in-scope tutorials use it: that is `Compiler.Faithful`'s scope 10, an
  assumption of the rule's part (B), which nobody checks (library issue L104); a kernel source that
  itself rebinds such a name is outside the subset.
- Python's builtins `float`, `int`, `isinstance`, `len`, `list`, `getattr`, `hasattr`, `min`,
  `max`, `print` and `range` ("Python's builtins").

**Expressions.**
- Literals: an `int`, `float`, `bool`, `str` or `None`; a tuple `(…)` or a list `[…]` of these
  expressions (both are tuples of the front end, as for shapes, offsets and block shapes); an
  f-string of compile-time values.
- Attributes: a tensor's `.dtype` and `.shape`, `.T` of a rank-2 tensor, a dtype's `.element_ty`,
  the methods below, and the functions, dtypes and classes of `tl` (`tl.float32`,
  `tl.pointer_type(d)`, `tl.math.exp2`, `tl.tensor_descriptor` in `isinstance`).
- Subscripts: `None` and `:` in each position of a tensor's subscript, which add and keep axes
  (`x[:, None]`, `x[None, :]`; "Values"); a compile-time integer subscript of a shape or a tuple
  (`acc.shape[0]`).
- Operators ("Operators and promotion"): `+ - * / // % & | ^ << >>`, the comparisons
  `< <= > >= == !=`, one per expression, `is` and `is not` ("Order"), `and`, `or` and `not`, unary
  `-` and `~`. With only compile-time operands, the operator is Python's on their values, computed
  at compile time (`BLOCK_SIZE_N // 2`, `STAGE & 1`, `ACTIVATION == "leaky_relu"`,
  `c_ptr.dtype.element_ty == tl.float8e4nv`; `code_generator.py:795-808, 1076-1110`), and `**` is
  supported only so. An operator whose computation dtype is an fp8 dtype (`computationType?`, as
  "Operators and promotion" gives it: `x * 1.0` with an fp8 `x`, and `x + y` or `x < y` with `x` and
  `y` of one fp8 dtype), and unary `-` on an fp8 tensor, are outside the subset (library issue L111;
  [exclusions.md](exclusions.md)).
- `a if c else b` ("Order").
- Calls: of the `tl.*` functions, and of the methods of tensors, descriptors and block pointers,
  that this page and `TVA.Lang` give meaning to (`tl.cdiv` and `x.cdiv` included: "`tl.cdiv`";
  library issue L88), with arguments of the kinds they state (an argument they do not state is
  excluded); of the kernel's helpers; of the builtins above.

**On the host**, the kernel is launched as `kernel[grid](…)`, the grid a tuple or a function of the
meta-parameters, or through the `CompiledKernel` of `kernel.warmup(…)` or `triton.compile`, within
`Compiler.Faithful`'s scope ("Launches").

Outside the subset, for instance (all excluded, [exclusions.md](exclusions.md)): `break` and
`continue`, `with`, `del`, `global`, `nonlocal`, `try`, `raise`, `import` in a function, a nested
`def` or `class`, `lambda`, comprehensions, `:=`, `for … else` and `while … else`, a bare annotation
`name: T`, an annotation other than `tl.constexpr`, an assignment to a subscript or an attribute,
`*` and `**` in a call or a signature, a slice with bounds or an integer index of a tensor, chained
comparisons, `@triton.heuristics`, `@triton.jit(noinline=True)`, a kernel parameter's default, a
value at the top level of the module other than those above (a `tl.constexpr(…)`, a dtype, a
number), an enclosing function's names, Python's `range` with a keyword argument or a fourth
argument (L90), a `return` in a `tl.static_range` body and a statement after an `if` that returns
(L89), a helper whose visited `return`s give values of other kinds than one admits (compile-time
values that are not the same value of one Python type, element by element: `1` and `1.0`, `(1, 2)`
and `(1.0, 2)`, `0.0` and `-0.0`), or that can reach its end without a `return` while another
`return` gives a value (L102), two calls of one helper whose compile-time arguments differ but share
the copy's name (L103), an operator computed in an fp8 dtype and a unary `-` on an fp8 tensor
(L111), and a use after a loop of a name first defined in its body or of a `for` loop's variable.

## How the front end evaluates the source

Triton's front end (the code generator) visits the kernel's Python source and emits one operation
for each operation it evaluates. What runs on the device is what it evaluates, with two exceptions:
the first visit of a loop body (before it emits a `for` or `while` loop, the front end visits the
body once to find the names the loop carries, and discards what that visit emits; only the marks of
the hints it visits stay: "Loops and unrolling", "Compiler hints"), and the code it visits after an
`if` that returns on the instance, into a block no program reaches (outside the subset: "`return`",
library issues L89, L108). So these facts say which parts of the source
become steps of the program (library issues L6, L10, L14, L25, L59, L75, L80, L84-L86;
[evidence/config-cond](../evidence/config-cond/README.md),
[evidence/debug-calls](../evidence/debug-calls/README.md),
[evidence/pure-operands](../evidence/pure-operands/README.md),
[evidence/python-numbers](../evidence/python-numbers/README.md),
[evidence/py-builtins](../evidence/py-builtins/README.md),
[evidence/front-end-subset](../evidence/front-end-subset/README.md)).

- **Helpers are inlined.** A call of a `@triton.jit` function from the kernel means its body with
  the call's arguments, bound to its parameters as Python binds a call (by position and by
  keyword, with the `def`'s defaults for the parameters the call does not give). The model may
  write it in place or as a Lean function applied at the call. A Python number passed directly as
  an argument (a literal, a `tl.constexpr`), or given by a parameter's default, stays a
  compile-time value in the body, where the parameter holds it; a name that holds a tensor passes
  the tensor (`code_generator.py:1358-1365`; evidence/python-numbers A8; evidence/front-end-subset
  D1-D4: with `def h(x, e=1e-7)`, `h(x)` is `h(x, 1e-7)`, a `bfloat16` product of `0`). An
  annotation on a helper's parameter changes nothing: the parameter holds what the call passes, so
  `def g(n: tl.constexpr)` called with a name holding a tensor gets the tensor (P1, P2; tutorial 06
  passes the tensors `offs_m` and `offs_n` to `_attn_fwd_inner`'s `tl.constexpr` parameters).
  Before the body, the front end also visits each default once, as an assignment of it to its
  parameter (`name = default`, or `name: tl.constexpr = default` when the parameter is so
  annotated; `code_generator.py:634-649`), which the binding of the call then replaces. So the
  default of an unannotated parameter that the assignment cannot make a tensor (a string, an integer
  outside the 64-bit range: "A name assigned a Python number holds a tensor", below) makes the helper
  rejected on the host even when the call passes the argument (library issue L92;
  `docs/library-review/round-15/lr-15b/work/fe_edges.py` H1: `def h(x, mode="a")` called as
  `h(x, "b")` was rejected), and a default `None` is accepted (H2). The call's value is the value
  of the `return` its program takes, for the helpers the subset admits ("`return`" below; library
  issue L102), and each call means its own body only when its copy is its own ("Compile-time values
  Triton keys apart", below; library issue L103).
- **Compile-time values Triton keys apart** (library issues L98, L103;
  [evidence/compile-keys](../evidence/compile-keys/README.md)). Triton tells compiled code apart by a
  key it builds from the compile-time values, not by the values, and two different values with one
  key share the code compiled for the first:
  - **A helper's copies.** The front end compiles one copy of a helper per name `mangle_fn` gives the
    call: the helper's name and each argument's type, a compile-time argument written as `c` and its
    `repr` (a dtype by its own name), joined with `_`, then `'` written `_sq_` and `[`, `]` written
    `_` (`code_generator.py:32-41`, `core.py:196-201`, `code_generator.py:1358-1396`). A later call
    whose name is the same uses the copy made for the first call the front end visited, whatever its
    own compile-time values: `store_c(o, float('nan')); store_c(o + 1, -float('nan'))` stored
    `0x7fc00000` twice (every NaN's `repr` is `nan`), and `pick(o, "x[y"); pick(o + 1, "x_y")` ran
    `"x[y"`'s body twice (K5; `1` and `1.0`, `1` and `True`, `inf` and `-inf` have names of their
    own). So a call means its body with its own arguments when no call visited before it in the same
    compilation names the same copy with other compile-time values; two calls of one helper whose
    compile-time arguments differ but whose names are the same are outside the subset ("The supported
    subset"). A NaN passed to a helper is covered, with its own pattern, unless an earlier call names
    the same copy with another NaN.
  - **A launch's compiled kernel.** Triton's caches key a launch's compiled kernel by its
    compile-time values in a like way (in the process, values Python's `==` equates and every NaN; on
    disk, the `str`s of the values joined with `-`), so a launch that takes code from them may run code
    compiled for other values (K1–K4; [evidence/host-round21](../evidence/host-round21/README.md)
    K1–K4). TVA does not verify Triton's caches: `Compiler.Faithful`'s scope 11 assumes the kernel is
    new, compiled for the launch's own values, and nobody checks it ("Launches").
- **A name assigned a Python number holds a tensor** (library issue L59). When a statement assigns
  a value to a name (`=`, `+=` and the other augmented assignments, and each target of a tuple
  assignment), a Python `int`, `float` or `bool` (a literal, the value of a `tl.constexpr`,
  the result of a compile-time expression such as `BLOCK * 2`) becomes a scalar tensor of its own
  type (`Operand.ownType?`, as in "Operators and promotion": an `int` is `int32`, `uint32`, `int64`
  or `uint64`; a `float` is `float32` or `float64`; a `bool` is `int1`), holding the constant Triton
  builds for it in that type (`Operand.constIn?`, as "everywhere else" in "Constants"; `+0` for a
  zero, `-0.0` included), unless the statement is `name: tl.constexpr = e` ("Annotated names",
  below; `code_generator.py:727-760`). The name then holds a tensor, not a Python number: every rule of
  this page about Python numbers or compile-time values applies to it as to a tensor
  (evidence/python-numbers A1-A12):
  - after `e = 1e-7`, `x * e` with a `bfloat16` `x` computes in `float32` (`float32(1e-7)·x`),
    where `x * 1e-7` computes in `bfloat16` and gives `0`; after `e = 1e-300`, `x * e` with a
    `float32` `x` computes in `float64`;
  - after `a = -7; b = 2` (or `lo, hi = -7, 2`, or `k = 0; k -= 7`), `a // b` is the tensor `//`
    (`-3`, truncating), where `-7 // 2` is Python's (`-4`);
  - after `F = False`, `if F and tl.atomic_add(c, 1) > 0:` is a runtime `if` whose condition runs
    the atomic in every program, and `if F: tl.static_assert(False)` is visited and rejects the
    kernel on the host;
  - after `B = BLOCK`, `tl.arange(0, B)` is rejected on the host;
  - a loop variable initialised from a literal (`i = 0`, `lo, hi = 0, N`) is a tensor from the
    start;
  - a name assigned from the Python integer of `tl.static_range` is a tensor (A6).

  `None` and dtypes are kept as they are (A13), a tuple is kept with each element converted as
  above, and another Python value (a string, a Python `list` as `list(…)` returns; a list display
  `[…]` is a tuple of the front end) is rejected on the host (A14).
- **Annotated names** (library issue L84; evidence/front-end-subset C1-C9). The statement
  `name: tl.constexpr = e`, on a name the function has not defined before it (not a parameter, not a
  name an earlier statement assigned), binds `name` to `e`'s value as it is: a Python number stays a
  compile-time value, and a tensor stays a tensor (`code_generator.py:694-712`). So after
  `c: tl.constexpr = -7`, `c // 2` is Python's `-4` (C1; evidence/python-numbers A7), and
  `B: tl.constexpr = BLOCK` (A11), `BM: tl.constexpr = acc.shape[0]` (C7) and
  `P: tl.constexpr = A == 1 and B == 2` (C8; tutorials 06 and 10) are compile-time values. On a name
  already defined, the statement is rejected on the host (C2), so also in the second copy of an
  unrolled `tl.static_range` body (C9). The annotation belongs to that statement only: a later
  assignment to the name follows the rule above, as does an assignment to a `tl.constexpr`
  parameter. After `c: tl.constexpr = -7; c += 0`, `c` is an `int32` tensor and `c // 2` is `-3`
  (C3); after `BLOCK = BLOCK * 1`, `tl.arange(0, BLOCK)` is rejected on the host (C4). A bare
  annotation (`e: tl.constexpr` with no value, which changes the next assignment to `e`: C5) and an
  annotation other than `tl.constexpr` are outside the subset ([exclusions.md](exclusions.md)). The
  latter evaluates its value twice (`code_generator.py:698`, then `visit_Assign`, `:712`, `:757`), and
  is otherwise ignored: `a: tl.int32 = tl.atomic_add(c, 1)` ran two atomics in each program
  (library issue L92; `docs/library-review/round-15/lr-15b/work/fe_edges.py` A1), and after
  `a: tl.int64 = 2**31 - 1`, `a` is an `int32` and `a + 1` wraps (C6).
- **Python's builtins** (library issue L75, evidence/py-builtins). A name that neither the function
  nor its module defines is looked up among `len`, `list`, `range`, `float`, `int`, `isinstance`,
  `getattr` and `hasattr`, which are Python's own, and `print`, `min` and `max`, which are Triton's
  (`tl.device_print`, below, and the `min` and `max` of "Operators and promotion")
  (`code_generator.py:357-365, 381-427`). No other builtin is visible: a call of `abs`, `round`,
  `sum` or another is rejected on the host (F9). A call of `float`, `int`, `isinstance`, `len`,
  `list`, `getattr` or `hasattr` is evaluated by Python at compile time, on its arguments' values
  (a `tl.constexpr` unwrapped to its value; a tensor, a pointer or a descriptor as the front end's
  object), after the arguments are evaluated as usual, so an operation inside them runs (F7: the
  atomic in `isinstance(tl.atomic_add(c, 1), tl.tensor)` ran in every program)
  (`code_generator.py:1444-1448, 1638-1659`). The call emits no operation, and its result is a
  compile-time value (a number from it is a Python number, `normalize_value`). Python's own rules
  give the result:
  - `float('inf')` is `+∞`, and `-float('inf')` is the Python `float` `−∞`; then "Constants"
    applies: `other=-float('inf')` is the `float32` `−∞` (`0xFF800000`), and
    `tl.zeros([B], tl.float32) - float("inf")` builds `+∞` in `float32` (F1, F2: tutorials 02 and
    06);
  - `int(2.7)` is `2`; `len` of a tuple or a shape is its length (F3, F6);
  - `isinstance(v, tl.tensor)` is true of a block, a scalar and a pointer argument;
    `isinstance(v, tl.tensor_descriptor)` is true of a host `TensorDescriptor` argument, which is not
    a `tl.tensor` (F4: tutorial 06's `_maybe_make_tensor_desc`), and of what
    `tl.make_tensor_descriptor` returns (`semantic.py:1888-1891`); in a compile-time `if`, only the
    branch it picks is visited (F4);
  - `getattr(x, "dtype")` and `hasattr(x, "dtype")` read the front end's object (F5).

  `float` and `int` of a runtime value (a loaded scalar, a scalar argument) are rejected on the host
  (F8, F10).
- **Every argument is evaluated before the call**, as an ordinary expression, whatever the call
  does at run time, with one exception: the `msg` of `tl.static_assert`, which is evaluated only
  when the assertion fails, and then the kernel is rejected on the host ("`tl.static_assert`",
  below; `code_generator.py:1617-1637`; library issue L108). So an operation inside an argument runs
  even when the call itself does nothing: an atomic inside the condition of `tl.static_assert`, or
  inside an argument of `tl.static_print`, `tl.device_print`, `print`, `tl.device_assert` or
  `assert`, runs in every program (evidence/debug-calls E1–E7, E9).
- **Order.** Every part of an expression is evaluated, except where a *compile-time value*
  decides (below). Operands are evaluated left to right, except in a call: there the called
  function's expression comes first (for a method, the object), then the keyword arguments in
  order, then the positional arguments in order, so in `tl.store(p, tl.load(a), mask=tl.load(m) > 0)`
  the load of `m` comes first ([evidence/front-end-order](../evidence/front-end-order/README.md) K).
  Each part is evaluated once, except the argument of a one-argument `range`, evaluated twice
  ("Loops and unrolling"). The order of the evaluated operations does not matter to a model, only
  which ones are evaluated:
  in a well-behaved model two accesses of a program to the same bytes with a write among them are
  a conflict unless both are atomics, and the semantics already performs a program's atomics in
  every order ("Programs", `Step.defer`). Where a compile-time value decides:
  - `a and b and …`: an operand whose value is a compile-time value that is false ends it, and no
    operand after it is evaluated. A compile-time value that is true, or a tensor, does not end
    it; with tensor operands both sides are evaluated;
  - `a or b or …`: the same, with a compile-time value that is true ending it;
  - `x if c else y`: when `c` is a compile-time value, only the side it picks is evaluated. When
    `c` is a tensor (a runtime scalar), each side is emitted inside its own branch of a runtime
    `if`, and only the taken side runs, in each program (evidence/front-end-order C,
    [evidence/runtime-if](../evidence/runtime-if/README.md) R8): it means the runtime `if` below,
    whose value is a tensor. Each side, once evaluated, becomes a tensor (`to_tensor`,
    `code_generator.py:998-1036`; library issue L102): a Python `int`, `float` or `bool` a scalar
    tensor of its own type, holding the constant Triton builds for it there, as an assigned number
    does ("A name assigned a Python number holds a tensor", above; `Operand.ownType?`,
    `Operand.constIn?`), and another compile-time value is rejected on the host. Both sides must then
    have the same type, or the kernel is rejected on the host, and a block `c` is rejected too. So
    `(-1 if c > 0 else -3) // 2` is the `int32` tensor `//` (`0` for `c = 5`, where Python's floor
    division gives `-1`), `x * (0.1 if c > 0 else 0.2)` with a `bfloat16` `x` computes in `float32`
    (`float32(0.1)·x`), and `1 if c > 0 else 1.0` (`int32` against `float32`) is rejected
    ([evidence/front-end-round20](../evidence/front-end-round20/README.md) E1a–E1d).

  A value is a compile-time value when it is not a tensor: a `tl.constexpr`, a Python `int`,
  `float`, `bool`, `str` or `None`, a dtype, a tuple of these, the result of `is` or `is not`,
  whatever its operands (`tl.atomic_add(c, 1) is None` is the compile-time `False`, after the atomic
  has run), and the result of a call of Python's `float`, `int`, `isinstance`, `len`, `list`,
  `getattr` or `hasattr` ("Python's builtins", above). A name that a statement assigned a Python
  number is not one: it holds a tensor (above); nor is a conditional expression on a runtime
  condition (above). Parts that are not evaluated never run
  (evidence/config-cond S1–S16: `FLAG and <atomic>` runs the atomic in every program with
  `FLAG = True` and in none with `False`).
- **`if`, `elif`, `else` on a compile-time condition.** The front end evaluates the condition (by
  the order above, so an operation inside it runs in every program), then visits only the branch
  it picks. An `elif` condition is evaluated only when every condition before it in the chain is
  false; after a taken `if` or `elif`, the rest of the chain is never visited, conditions included
  (evidence/config-cond C1–C3, S8). The branch not taken has no meaning in the program.
- **A runtime `if`** (its condition a tensor, a runtime scalar; library issue L28,
  [evidence/runtime-if](../evidence/runtime-if/README.md)). The front end evaluates the condition
  (by the order above), converts it to `int1` as `c != 0` (for a float, unordered: NaN is true and
  `-0.0` false), and emits both branches into one runtime branch (`scf.if`). Each program runs the
  operations of the branch its condition takes, and only those: an atomic, or a store far out of
  bounds, in the other branch does not run (R1–R3). An `elif` is an `if` in the `else` branch, so
  its condition runs only in the programs that reach it (R9). After the `if`, the names it merges
  have the taken branch's value: a name defined before the `if` that a branch assigns, and a name
  new to the `if` that both branches define (other than with `name: tl.constexpr = e`, below). A
  name defined before the `if` must keep its dtype and
  shape in each branch that assigns it, also when both do, and a new name both branches define must
  have one dtype and shape in both; otherwise the kernel is rejected on the host (R5;
  [evidence/if-names](../evidence/if-names/README.md) N3-N6). A name new to the `if` that only one
  branch defines belongs to that branch: it may be used there (N1), and a use after the `if` is
  rejected on the host (N2). So does a name new to the `if` that a branch binds with
  `name: tl.constexpr = e` ("Annotated names"), also when both branches do: that statement does not
  record the name among the branch's definitions, so the `if` does not merge it
  (`code_generator.py:704-710`; library issue L92;
  `docs/library-review/round-15/lr-15a/work/probes.py` A2: a use after the `if` was rejected, "c is
  not defined"; A3: a use inside each branch ran). A block condition with more than one element is
  rejected on the host, and a one-element block is read as its element (R4). These rules are for
  the names the subset admits ("The supported subset"): a name new to the `if`, and a name that
  holds a tensor before it (a block, a scalar of a Triton dtype or a pointer). A branch that assigns a name holding another
  value before the `if` is outside the subset (library issue L61): a `tl.constexpr` there is
  rejected on the host, but `None` and a dtype are not merged like a tensor (after
  `y = None; if n > 0: y = tl.load(p)`, `y` is still `None`; after
  `d = tl.float16; if n > 0: d = tl.float32`, `d` is still `float16`;
  `code_generator.py:858-861`; `docs/library-review/round-11/lr-11a/work/if_compile_time_livein.py`
  I2, I6), and the library excludes such a kernel
  ([exclusions.md](exclusions.md)).
  The model is a Lean `if` on the condition's value, with each branch's operations in its branch.
  `and` and `or` in such a condition evaluate every operand (no short circuit at run time, R9), as
  "Order" says.
- **`return`.** A `return` inside a runtime `if` ends the program, or the inlined helper, for the
  programs that take it: the front end splits the function there (`cf.cond_br`), and those programs
  run nothing after it (R6, R7). In a `for` or `while` body of the same function, at any depth of
  `if`s, a `return` is rejected on the host; `tl.static_range` is unrolled and does not count
  (R7). A kernel that returns a value is rejected. The model returns from its `do` block at the same
  point (`return ()` in the branch, or the rest of the body in the other branch).

  **A helper's value** (library issue L102). The front end gives a helper's call one type, from
  every `return` it visits in the body (`decide_return_type`, `cast_to`, `handle_returns`,
  `code_generator.py:556-626`), not the value of the `return` a program takes as Python reads it:
  when the visited `return`s all give compile-time values that Python's `==` finds equal (`5` and
  `5`, but also `1`, `1.0` and `True`), the call is the first visited `return`'s value on every path;
  otherwise each returned Python number becomes a scalar tensor of its own type, and all must then
  have one type (else the kernel is rejected on the host); and a path that reaches the end of the
  body without a `return` gives the compile-time value, or an unspecified value (LLVM `poison`) of
  the tensor type. So with a runtime `c`, `if c > 0: return -7` then `return -9` makes `h(c) // 2`
  the `int32` tensor `//` (`-3` for `c = 5`, where Python gives `-4`), `return 1` then `return 1.0`
  makes `x * h(c)` the integer product on both paths, and `if c > 0: return 5` with nothing after
  makes `h(c)` `5` also where the `if` is not taken ([evidence/front-end-round20](../evidence/front-end-round20/README.md)
  E2a–E2c; `docs/library-review/round-17/lr-17a/work/helper_returns.py` R4: a block on one path,
  `poison` on the other). Equal tuples are compared element by element with `==`, and `0.0` equals
  `-0.0`: with a runtime `c`, `if c > 0: return (1, 2)` then `return (1.0, 2)` makes `x * th(c)[0]`
  the integer product on both paths, and `if c > 0: return 0.0` then `return -0.0` makes
  `f"{h(c)}"` `"0.0"` on both paths ([evidence/front-end-round21](../evidence/front-end-round21/README.md)
  T1–T4). The subset admits only the helpers for which this is the value of the `return` the
  program takes ("The supported subset", `return`): one visited `return`, `return`s of tensors of
  one type (E2e), or `return`s of the same compile-time value, of one Python type and, for a float,
  bit for bit (not a NaN, not `0.0` with `-0.0`), element by element for a tuple (E2d), and no path
  to the end of the body without a `return` when a `return` gives a value. The model gives the
  call that value.

  The statements after a `return` in its own statement list are not compiled
  (`code_generator.py:508-517`). What follows an `if` whose taken branch returns (a compile-time
  `if`, or a runtime `if` whose two branches return), or a `tl.static_range` copy that returns, is
  outside the subset (library issue L89; user, 2026-10-28): the front end still visits it, into a
  block no program reaches (`:544-554, 893-929`), so it never runs, but its hints still mark the
  values they name, those defined before the `return` included, its `tl.static_assert`s are still
  evaluated, and in a helper its `return`s still count towards the return type (`:556-585`)
  (`docs/library-review/round-15/lr-15b/work/dead_after_return.py` D1, D5, D7, D9;
  [evidence/dead-code-round17](../evidence/dead-code-round17/README.md) E1, E4, E7). A `return` in a
  `tl.static_range` body is outside the subset too ("The supported subset";
  [exclusions.md](exclusions.md)).
- **Loops and unrolling.** `tl.static_range` is unrolled: the body is visited once per iteration,
  with a Python integer as the variable. Before a `for` or `while` loop, the front end visits the
  body once with every name bound to its value before the loop, to find the names the loop carries,
  and then emits the loop (library issue L1, "Compiler hints" below). What that first visit emits
  does not run (`code_generator.py:465-503`; evidence/front-end-subset Z1, Z2: an atomic in a body
  that runs zero times ran in no program, and in a body that runs once, once per program).
  - **The iterator's arguments** are evaluated before the loop, each once, except that `range(e)`
    with one argument evaluates `e` twice, one after the other, and the second value is the bound
    (library issue L85; `code_generator.py:1206, 1241-1242`; evidence/front-end-subset R1, R4:
    `range(tl.atomic_add(c, 1) * 0)` ran two atomics in each program). The model writes the
    operations of `e` twice. `range(lo, hi)`, `range(lo, hi, step)`, `tl.range` with any number of
    arguments, and `tl.static_range` evaluate each argument once (R2, R3). Python's `range` uses only
    its first three positional arguments: a keyword argument and a fourth argument are evaluated and
    ignored (`code_generator.py:1207, 1237-1243`; library issue L90;
    `docs/library-review/round-15/lr-15b/work/fe_edges.py` K1, K2, K4: with `n = 8`,
    `range(0, n, step=2)` and `range(n, step=2)` ran 8 trips and `range(0, n, 2, 99)` 4), and both
    are outside the subset. `tl.range` and `tl.static_range` take their keywords as their signatures
    say (K3: `tl.range(0, n, step=2)` ran 4 trips).
  - **Names after a loop.** A `for` loop over `range` or `tl.range`, or a `while` loop, carries the
    names defined before it that its body assigns ("Loops"), and after the loop they hold what the
    last iteration left (their values before the loop if it ran zero times). Each holds a Triton
    value before the loop (a tensor, a block pointer, a descriptor) and keeps its type: a body that
    assigns a name holding a compile-time value (a `tl.constexpr`, `None`, a dtype), or gives a
    carried name another type, is rejected on the host (`code_generator.py:482-497, 1116-1124`). A
    name the body defines that was not defined before the loop is not defined after it: a use there
    is rejected on the host (L1, L4). The loop's variable, after a `for` loop over `range` or
    `tl.range`, does not hold the last value: the front end leaves it a placeholder with no defined
    value (`code_generator.py:1277-1278, 1320-1327`; L2 stored `16776272` after `range(0, 4)`).
    `tl.static_range` is unrolled instead: its copies of the body are statements in sequence, so after
    the loop a name holds what the last copy left, a name first defined in the body is defined, and
    the loop's variable holds its last value, a compile-time integer (`code_generator.py:1208-1216`;
    library issue L92; `docs/library-review/round-15/lr-15b/work/fe_edges.py` S1: `2` after
    `tl.static_range(3)`; S2: a name the body defines held the last copy's `20`). The subset admits
    neither of the last two uses after any loop: a kernel that uses, after a loop, a name first
    defined in its body or a `for` loop's variable is outside the subset
    ([exclusions.md](exclusions.md)).
- **`tl.static_assert(cond, msg)`** is evaluated at compile time wherever the front end visits it:
  in both branches of a runtime `if`, whether or not a program takes them; in each unrolled
  iteration of `tl.static_range`; at each call of the inlined helper that holds it; and in a `for`
  or `while` body, even one that runs zero times; not in a branch the configuration does not take
  (evidence/debug-calls V1–V4). `cond` must evaluate to a compile-time `bool` (`x is None` and
  `x is not None` are, for any `x`), or compilation stops. `msg` is evaluated only when the
  assertion fails, and then compilation stops (E8): the launch is rejected with an error on the
  host and has no outcome. So an assertion that holds adds nothing to the program but the
  operations its `cond` evaluates.
- **Debugging calls.** `tl.static_print` prints at compile time. `tl.device_print` emits a device
  `printf`, which writes no global memory, and Python's `print` in a kernel *is*
  `tl.device_print`. Without `debug`, `tl.device_assert` emits nothing, and a Python
  `assert test, msg` is `tl.device_assert(test, msg)`; `Compiler.Faithful` covers only
  `debug = false` ("Launches"). Their arguments are evaluated as above (evidence/debug-calls P1, P2,
  E3–E7, E9).
- **Operators on aggregates.** On an instance of an aggregate class (made with Triton's
  `_aggregate`), an operator, comparison or subscript calls the class's own method, a
  `@triton.jit` function the front end inlines, which may run any operation
  (evidence/pure-operands: an atomic ran in 4 of 4 programs). Aggregates a kernel or another
  library defines are excluded ([exclusions.md](exclusions.md)). A block pointer
  (`tl.make_block_ptr`) is Triton's own aggregate (`_block_ptr`, `core.py:1828-1936`): its methods
  are Triton's code and it defines no operator, so the library covers it through its operations
  ("Block pointers" below; library issue L30), and an operator on it has no meaning.
- **Expressions with no effect.** The following have no effect on memory and are never undefined,
  so a part of the source built only from them adds no step to the program and cannot make it
  undefined:
  - names of values, literals, `tl.constexpr` values and dtypes (`tl.float32`, …);
  - the attributes `.dtype`, `.shape` and `.numel`, the attributes of a dtype (such as
    `.element_ty`), and subscripts of shapes and tuples;
  - subscripts that only add axes (`x[:, None]`, `x[None, :]`);
  - string literals, and f-strings of the above;
  - calls of Python's `float`, `int`, `isinstance`, `len`, `list`, `getattr` and `hasattr`
    (evaluated at compile time, "Python's builtins"), when their arguments are built from the
    expressions of this list;
  - the operators `+`, `-` (binary and unary), `*`, `~`, `&`, `|`, `^`, the comparisons, `not`,
    `and`, `or`, `is` and `is not`, on operands that are tensors (blocks or scalars of a Triton
    dtype, pointers and `tl.int1` included), compile-time values, or tuples of these;
  - the operators `/`, `//`, `%`, `**`, `<<` and `>>` when both operands are `tl.constexpr`
    values: Python computes them at compile time.

  Everything else may have an effect or be undefined: any other call (a `tl.*` function, a
  `@triton.jit` function, a method such as `x.to(…)`, a Python function); integer `//` and `%` on
  tensors or runtime scalars (by zero, undefined) and shifts (by the width or more, undefined),
  float `/` (the platform's `P.div`), and any operator on an aggregate.

## The `tl.*` operations (`TVA/Lang/Ops.lean`)

Each definition's docstring states its reading of the documentation. The non-obvious ones:

- **Float results are exact up to NaN payload.** The device returns some NaN, not a particular
  one: PTX `neg`/`abs` give "an unspecified NaN", and MLIR folds such as `x * 1.0 → x` keep the
  input's payload. So every exact float operation (`tl.add`, `sub`, `mul`, `fma`, `div_rn`,
  `sqrt_rn`, `neg`, `abs`, `cvt`) is a `Prog` whose result is `FloatV.NaNEq` to the IEEE
  result. Specifications compare outputs up to NaN payload.
- **`tl.load(ptrs, mask, other)`.** Disabled lanes are not accessed. They hold `other` when it
  is given, and an unspecified value otherwise (S6: 3.8.0 happens to give `0`; the library
  does not promise it). `other` is converted to the pointer's dtype, as `tl.convert`
  (`semantic.cast`, `semantic.py:1043-1045`; library issue L86), after a Python number has become a
  tensor of its own type ("Constants", everywhere else): with a `float16` pointer,
  `other=1 + 2**-11 + 2**-30` is the `float32` `1 + 2^-11` rounded to `float16`, `1.0`, where the
  `float16` nearest to the number is `1 + 2^-10` (evidence/front-end-subset O1). Its
  `cache_modifier`, `eviction_policy` and `volatile` are performance-only (below).
- **Constants** (library issues L23, L35, L59). A Python number `c` (an `int` or `float` literal
  written in place, a `tl.constexpr` holding one, or the result of a builtin such as `float('inf')`)
  takes the value Triton builds for it, which depends on where it is used
  ([evidence/literals](../evidence/literals/README.md)). A name that a statement assigned a Python
  number is not one: it holds a scalar tensor of the number's own type, built as "everywhere else"
  below, and is a tensor operand wherever it is used ("How the front end evaluates the source": a
  name assigned a Python number holds a tensor):
  - **With an arithmetic or bitwise operator, as a value of `tl.where`, and in `tl.full`.** Where
    `c` is an operand of `+ - * / // % & | ^ << >>` on a tensor (`x * 0.1`, `0.5 - x`, …) or a
    value of `tl.where` (`tl.where(m, x, 0.0)`), it is built directly in the dtype the operation
    computes in, after Triton's promotion ("Operators and promotion" below): a Python number does
    not widen a tensor of its own kind, so `x * 0.1` with `x` a `bfloat16` tensor computes in
    `bfloat16`, and a `float` with an integer tensor computes in the number's own type (below).
    `tl.full(shape, c, d)` builds it in `d`. The value is the library's function of `c`'s double
    (`FloatV.ofPat` of the `float64` bit pattern of `float(c)`):
    - `float64`: `FloatV.litF64`, the double itself;
    - `float32`: `FloatV.litF32`, the nearest `float32`;
    - `float16`: `FloatV.litF16`, the nearest `float16` to that `float32` (two roundings);
    - `bfloat16`: `FloatV.litBF16`, the `bfloat16` nearest to the six-decimal text (C's `%f`) of
      that `float32`: `1e-7` is `+0`, `-1e-7` is `-0`, `1.23456e-5` is the `bfloat16` of `1.2e-5`,
      and every constant below `5·10⁻⁷` in magnitude is `±0`;
    - an fp8 dtype: the exact value of the `float32` constant `FloatV.litF32` in that dtype,
      `FloatV.litF8` (library issue L64). Triton builds the `float32` constant and truncates it with
      `arith.truncf`, which MLIR folds at compile time only when the conversion loses nothing
      (`semantic.py:601-613`, `llvm-ArithOps.cpp:1588-1611, 1788-1825`), and an unfolded one is
      rejected on the host: so the constant is never the device's saturating conversion (`tl.cvt8`),
      and `c` is rounded once, to its `float32` constant, and no more. A `c` whose `float32` constant
      (`litF32 c`) the dtype holds is that value, also when `c` itself is not (`448.0` in `float8e4nv`
      is `0x7e`; `1 + 2^-30` and `1 + 2^-40` are `1.0`, and `-1e-50` is `-0`,
      `docs/library-review/round-20/lr-20a/work/probes.out` P1,
      `docs/library-review/round-20/lr-20b/work/probes2.out`); `±∞` is `±∞` in `float8e5` (`0x7c`,
      `0xfc`; also a `c` whose `float32` overflows, such as `1e39`); a NaN in `float8e5` keeps its
      sign, its quiet bit and the next fraction bit of its `float32` (`float('nan')` is `0x7e`,
      `-float('nan')` `0xfe`); and every other `c` (`0.1`, `460.0`, a value beyond `maxVal`, `±∞` and
      NaN in `float8e4nv`, a NaN whose `float32` has a lower fraction bit set) is rejected on the
      host, where `litF8` is `none` ([evidence/fp8-constants](../evidence/fp8-constants/README.md)
      F1–F3: every value of both formats, 160 values that one of them does not hold, the special
      values, bit for bit). An arithmetic operator that computes in an fp8 dtype is outside the
      subset ("Operators and promotion", library issue L111), so this is the constant of `tl.full`
      and of a value of `tl.where`;
    - an integer dtype: `c` modulo `2^w` (`IntV.ofInt c`). `tl.full` does not check the range of a
      dtype narrower than 64 bits (`tl.full([1], 300, tl.int8)` is `44`); a negative `c` in an
      unsigned dtype, a `c` outside the 64-bit range, and an operator's out-of-range `c` are
      rejected on the host.

    Each of them gives `+0` for a zero, `-0.0` included. A NaN keeps its sign and is quiet:
    `float('nan')` (the double `0x7ff8000000000000`) is `0x7fc00000` in `float32`, `0x7e00` in
    `float16`, `0x7fc0` in `bfloat16` and the double itself in `float64`, and `-float('nan')` (also
    Python's `float('inf') - float('inf')`, `0xfff8000000000000`) is the same with the sign bit set;
    the functions also give the top bits of a payload in `float32` and `float16` (library issue
    L98; [evidence/literals](../evidence/literals/README.md), "NaN constants", bit for bit; the sign
    of `float('inf') - float('inf')` is the host's, x86-64's default NaN). A NaN that reaches the
    kernel as a compile-time value (a `tl.constexpr` argument, a helper's argument) has this pattern
    when the helper copy the call uses is its own and the kernel is new: Triton's keys write every NaN
    as `nan` ("Compile-time values Triton keys apart"; `Compiler.Faithful`'s scope 11, an assumption;
    library issue L103). `Operand.constIn?` (`TVA.Lang.Promote`) is this value for a dtype, which a
    prover evaluates.
  - **Everywhere else, as a tensor of its own.** As an operand of a comparison (`x < 1e-3`), as the
    condition of `tl.where`, as an argument of any other `tl.*` function (`tl.maximum`,
    `tl.minimum`, `tl.clamp`, `tl.fma`, the math functions, the atomics), as a stored value, as
    `other=` of `tl.load`, or as a side of a conditional expression on a runtime condition ("Order"),
    `c` first becomes a tensor of its own type: an `int`
    is `int32` (`uint32`, `int64`, `uint64` when larger), and a `float` is `float32`
    (`FloatV.litF32`) when it is zero, infinite, NaN or of magnitude in `[2^-126, 2^128 - 2^104]`,
    and `float64` (`FloatV.litF64`) otherwise. Then the operation's own promotions and conversions
    apply: `tl.maximum(x, 0.5)` with `x` a `float16` tensor compares in `float32`, and `x < 0.1`
    with `x` a `float64` tensor compares against `float32(0.1)`, widened (true for `x = 0.1`;
    [evidence/promotion](../evidence/promotion/README.md) K1, K4). A stored number or an `other=`
    with an fp8 pointer is so converted on the device, by the saturating conversion `tl.cvt8`
    (`tl.store(p8, 1000.0)` into `float8e4nv` stores `448`, `0x7e`; a conversion of a constant zero
    is folded and keeps its sign), not built as an fp8 constant
    ([evidence/fp8-constants](../evidence/fp8-constants/README.md) F4).
- **`tl.store(ptrs, vals, mask)`.** The value is converted to the pointer's dtype (an explicit
  `tl.cvt`, `tl.itof`, … in the model, "Float conversions" below). Its `cache_modifier` and
  `eviction_policy` are performance-only (below).
- **Performance-only arguments** (library issue L44; user, 2026-10-28: "we do not need to provide
  models for performance-only primitives"). These change how fast a kernel runs, not what it
  computes, per Triton's documentation, which `Compiler.Faithful` trusts; a model omits them, and
  the library keeps no evidence of its own for each: the `cache_modifier` (`".ca"`, `".cg"`,
  `".wt"`, …) and `eviction_policy` of `tl.load`, `tl.store` and the descriptor and block-pointer
  accesses, `volatile` of `tl.load`, the `num_warps`, `maxnreg` and `launch_cooperative_grid` of a
  launch or a `triton.Config`, `launch_metadata=` ("Launches"), and the scheduling keywords of
  `tl.range` other than `num_stages` and `warp_specialize` (`loop_unroll_factor`,
  `disallow_acc_multi_buffer`, `flatten`, `disable_licm`). Not performance-only, and given by a model:
  `enable_fp_fusion` (`Launch.fusion`), `input_precision` and `max_num_imprecise_acc` of
  `tl.dot` (`DotCfg`), which change results; `num_stages` (of a launch, a `triton.Config` or
  `tl.range`) and `warp_specialize`, which change the order of a program's accesses, harmless only
  because a well-behaved model has no conflict within a program ("Programs", "Loops").
- **Atomics.** A tensor atomic is one `rmw` per enabled lane, in an order the program does not
  control. Each lane returns its element's old value; disabled lanes return unspecified values.
  Lanes of different atomics of the program may also be performed in any order their
  dependences allow (`Step.defer`, above).
  - Integer `atomic_add` wraps. Float `atomic_add` is `P.atomicAdd`.
  - `atomic_max`/`atomic_min` compare integers by signedness. For floats, the result is one of
    the two values and, if neither is NaN, not below (above) either.
  - `atomic_and/or/xor` are integer-only. `atomic_cas` compares bit patterns.
  - The defaults are `sem = acq_rel` and `scope = gpu`.
- **Float `+` and `-`: IEEE or fusable.** With `enable_fp_fusion` (Triton's default: `true`, the
  launch's `fusion`), the backend may contract a product and an addition into one `fma`. An
  addition, subtraction, reduction or scan that a *product* may feed may therefore be computed
  with fusion. A product is an operand computed by `*`, `tl.dot` or `tl.rand`, through any
  dataflow, inlined helpers and library functions included. Such an operation is the library's
  fusable form; an operation no product feeds is the IEEE form:

  | IEEE form | fusable form |
  |---|---|
  | `tl.add` | `tl.mul_add a b c` for `a * b + c`; `tl.addc` otherwise |
  | `tl.sub` | `tl.subc` |
  | `tl.sum`, `sum1`, `sum0` | `tl.sumc`, `sum1c`, `sum0c` |
  | `tl.cumsum` | `tl.cumsumc` |
  | an `AddRel false` scan relation | `AddRel E.fusion` |

  - `tl.add` is IEEE.
  - `tl.mul_add a b c` is either `a*b + c` (two roundings) or `fma`.
  - `tl.addc` is any result `AddRel` allows: the IEEE sum, or, with fusion, the rounding of
    `p + q` for some exact `p`, `q` rounding to the operands. The fusable forms allow every result
    the IEEE forms allow.
  - Contraction is the only float rewrite the backend may apply: no reassociation, no
    reciprocal division, no flushing of subnormals ([review-questions.md](review-questions.md)
    S5). The combine pass's rewrites are below (`dot_add`, `sum_bcast_mul`); they apply whatever
    `enable_fp_fusion` says.
- **`/` on floats** is `P.div` (the default `tl.fdiv` is not IEEE). `tl.div_rn` is IEEE. Its
  accuracy is a rule: `Rules.Num.DivFull` for fp32 (`div.full.f32`: within
  `2^-22·|x/y| + 2^-148` of `x/y` for `|x/y| ≤ 2^126`) and `Rules.Num.DivRn` for fp64 (IEEE)
  ([rules/num.md](rules/num.md)).
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
- **Operators and promotion** (`TVA.Lang.Promote`; library issue L45,
  [evidence/promotion](../evidence/promotion/README.md)). For an operator with a tensor operand,
  the library's definitions give what Triton's front end does: the dtype both operands are
  converted to, the result's dtype, whether the result is a block, or that the front end rejects
  the operator on the host (no outcome). They agree with the front end on all 37896 cases of the
  evidence's table that have a tensor operand. A model writes each implicit conversion as
  `tl.convert d d' x` (`semantic.cast`: an integer wraps, or extends by the source's signedness;
  `int1` is `x != 0`; integer to float is `tl.itof`'s rounding, float to integer `tl.ftoi`'s
  truncation, float to float `tl.cvt`, `tl.cvt8` or `tl.cvtFrom8`) to the dtype the definitions
  give, which the prover evaluates on the kernel's dtypes.
  - **Python numbers** (`Operand.ownType?`), as "Constants" counts them (not a name that a
    statement assigned one, which is a tensor of the type given here): a `bool` is `int1`; an `int`
    is `int32`, `uint32`, `int64` or `uint64`, the first whose range holds it (rejected beyond); a
    `float` is `float32` when it is zero, infinite, NaN or of magnitude in
    `[2^-126, (2-2^-23)·2^127]`, and `float64` otherwise.
  - **The computation dtype** (`computationType?`), the first that applies: a Python number of a
    kind (`int1` < the other integers < floats) no higher than the tensor's takes the tensor's
    dtype (`float32` for `float16`/`bfloat16` under `/`, `//`, `%`); either operand `float64`
    gives `float64`; either `float32`, `float32`; either `float16`, `float16` (`float32` under
    `/ // %`), so `float16` with `bfloat16` is `float16`; both `bfloat16`, `bfloat16` (`float32`
    under `/ // %`), and one `bfloat16`, `float32`; two fp8, the same one, or `float16` for two
    different ones; an fp8 with an integer is rejected; two integers give the wider when their
    signedness is the same, and otherwise the unsigned one if it is at least as wide, else the
    signed one (`int1` counts as unsigned, of width 1); mixed signedness is rejected under
    `/ // %`. A negative Python number with an unsigned result, and a Python integer outside an
    integer result's range, are rejected.
  - **Per operator** (`BinOp.typing?`). `+ - *` compute in that dtype. So does `/`, except that two
    integers (`int64` included) are both converted to `float32`: `int32` `16777217 / 1` is
    `16777216`. `//` takes integers only. `%` on floats is `frem`, which the library does not give
    ([exclusions.md](exclusions.md)). `& | ^ << >>` take integers only. A comparison first makes
    each Python number a tensor of its own dtype and then promotes, so `x < 2**31` with an `int32`
    `x` compares in `uint32` and is false for `x = -1` (evidence/promotion K3); its result is
    `int1`. `tl.minimum` and `tl.maximum` do the same and also convert `bfloat16` to `float32`; on
    integers they are `tl.iminimum`/`tl.imaximum`, signed or unsigned by the dtype, exact.
    Python's `min` and `max` in a kernel fold `tl.minimum`/`tl.maximum` over their arguments, left
    to right, when one of them is a tensor (a name assigned a number included: `a = 3; min(a, 5)` is
    an `int32` `tl.minimum`), and are Python's own when all are compile-time values; then none may
    be NaN or `-0.0`, or the kernel is rejected on the host (`core.py:3882-3929`;
    evidence/python-numbers B1-B4).
  - **`and`, `or`, `not`** (`boolOp?`, `UnOp.typing?`). Two tensor operands must both be `int1`
    (`Tensor.and`, `Tensor.or`), and `not` needs an `int1` tensor (`Tensor.not`); another dtype is
    rejected on the host. A Python value that decides the result (false for `and`, true for `or`)
    is the result, and another Python value is dropped, so `x and True` is `x` itself.
  - **Unary operators.** `-x` keeps the dtype (`tl.neg` on floats, wrapping `0 - x` on integers);
    `~x` takes integers only (`int1` included); `+x` on a tensor is rejected on the host.
  - **`tl.where(c, x, y)`** (`whereTyping?`): `c` becomes a tensor and is converted to `int1` as
    `c != 0` (so an `int32` mask `m` is `m != 0`, and a float condition is unordered: NaN is true;
    an fp8 condition is rejected); `x` and `y` are typed as an arithmetic operator's operands,
    Python numbers included; the result is `Tensor.where_`.
  - Arithmetic whose computation dtype is an fp8 dtype type-checks, and the backend rejects it on
    the host, except where MLIR's folders remove the operation first: `x * 1`, `1 * x`, `x / 1` and
    `x - 0` give `x`, and an operation on two fp8 constants is computed at compile time
    ([evidence/front-end-round21](../evidence/front-end-round21/README.md) F: those forms compiled and
    ran on every fp8 pattern of both dtypes, and `x + 0.0`, `x * 2.0`, `x + x` and `-x` were rejected,
    as was the comparison `x < x`). None of these forms is in the subset: every operator whose computation dtype is an
    fp8 dtype, and unary `-` on an fp8 tensor, are outside it ("The supported subset"; library issue
    L111).
- The promotions of reductions and scans are written in the model:
  - `tl.max`/`tl.min` reductions (without `return_indices`) widen floats narrower than 32 bits to
    fp32, and integers narrower than 32 bits to `int32` whatever their signedness: `int8`/`int16`
    sign-extended, `uint8`/`uint16` zero-extended, `int1` as `0`/`1`. So `tl.max` of an `int8`
    tensor is an `int32`, and `tl.max(x) + 100` does not wrap at 8 bits;
  - `tl.sum` widens integers narrower than 32 bits to `int32` (signed) or `uint32` (unsigned,
    `int1` included), and floats not at all; with `dtype=d` the input is first converted to `d`;
  - `tl.max`/`tl.min` with `return_indices`, `tl.argmax` and `tl.argmin` widen only `bfloat16`
    (to fp32).
  ([evidence/int-reductions](../evidence/int-reductions/README.md))
- **`x.to(int)`** of a float (`tl.ftoi`) truncates. A NaN, infinite or out-of-range element is
  undefined behaviour (LLVM poison, read conservatively).
- **`x.to(float)`** from an int (`tl.itof`) rounds to nearest even once. For fp16/bf16
  targets it may instead round twice, through fp32 or fp64 (`tl.ItofRel`). Before sm_90 there
  is no direct integer-to-bf16 conversion. A narrowing with the default rounding (no
  `fp_downcast_rounding`, or `"rtne"`) of an int-to-float result is one `tl.itof` to the final
  format, since MLIR folds `truncf(sitofp x)` into `sitofp x` with no one-use condition when the
  `truncf` has the default rounding. That covers explicit chains (`x.to(f64).to(f32)`,
  `x.to(f32).to(bf16)`) and the implicit conversions of `tl.store`, the atomics and `desc.store`
  (storing `offs.to(f32)` into a bf16 buffer). A narrowing with `fp_downcast_rounding="rtz"` is
  not folded: it is `tl.itof` to the intermediate format, then a conversion toward zero
  (`tl.cvt … .rz`, as the source writes them; library issue L24,
  [evidence/itof-rtz](../evidence/itof-rtz/README.md)).
  `tl.itof` and `x.to(int)` exist only for the standard float formats.
- **`tl.sum`, `tl.max`, `tl.min` (reductions)** combine along a `Tree` Triton chooses.
  - `tl.sum` (`sum1`, `sum0`) is for inputs that are not products, and every node is an IEEE
    addition.
  - `tl.sumc` (`sum1c`, `sum0c`) is for a sum of products, and its nodes are `AddRel` with the
    launch's fusion flag.
  - `tl.max1 isMax` (`axis=1`), `tl.max0 isMax` (`axis=0`) and `tl.maxAll isMax` (a whole 1-D
    tensor) combine with `maximum` (`isMax = true`) or `minimum`.
  - Integer reductions do not depend on the order: `Tensor.foldAll`, `foldRows`, `foldCols` with
    wrapping `+`, `IntV.max`, `IntV.min`, on the widened values (above; `x.castInt` to `I32`).
  - A reduction keeps the source's axis; with `keep_dims=True` the reduced axis stays, of size 1.
- **Index reductions.** `tl.max(x, axis, return_indices=True)` and `tl.min(…)` give the values and
  their `int32` indices; `tl.argmax`/`tl.argmin` give the indices. They are `tl.argmaxAll`,
  `tl.argmax1`, `tl.argmax0` (`isMax`, `tieLeft`): the pairs `(x[j], j)` combined along a `Tree`
  Triton chooses by `tl.argCombine`, Triton's own combine function. A pair wins when its value is
  greater (`isMax`; less for `min`), or, with `tie_break_left` (`tieLeft`, the default), when the
  values are `==` and its index is smaller. The values compare with Triton's `<` and `==` on the
  dtype (`CmpKey`): signed or unsigned integers, ordered floats. So without NaN and with
  `tie_break_left`, the result is the first index of the maximum; with `tie_break_left=False` any
  maximal index may come out. A NaN compares false, and then the result depends on the tree.
  Moreover, with a NaN in the reduced slice, the device's `float16` and `bfloat16` results (the
  latter reduced in `float32`) take the value and the index from different positions in up to
  19% of the slices (a NaN value with a number's index, or the reverse;
  [evidence/argmax](../evidence/argmax/README.md)). So with a NaN in the slice the library assumes
  only that the value is one of the slice's elements (up to NaN payload) and the index is in range
  (`tl.ArgRes`), for every float type; without NaN, the pair is one that some tree gives.
- **`tl.dot(a, b, acc)`** is `P.dot cfg`, with `cfg` (`DotCfg`) the operand dtypes, the source's
  `out_dtype`, and:
  - `precision`: the source's `input_precision`; when it is omitted, `tf32` (and `ieee` with
    `allow_tf32=False`), since `TRITON_F32_DEFAULT` and the knob `fp32_default` are at their
    defaults ("Launches");
  - `maxImpreciseAcc`: the source's `max_num_imprecise_acc`; when it is omitted, `2^30` for an
    fp8 × fp8 dot on sm_90 and `0` otherwise.
- **The combine pass's rewrites** ([review-questions.md](review-questions.md) S7). Triton rewrites
  these shapes whatever `enable_fp_fusion` says, and the library's operation for each allows both
  the rewritten and the unrewritten result:
  - **`tl.dot_add`.** A source `tl.dot(a, b) + c` or `c + tl.dot(a, b)`, where the dot's
    accumulator is `±0` after constants are folded (the default accumulator, `tl.zeros(…)`,
    `tl.full(…, 0.0)`, or any constant expression that folds to a zero, such as
    `tl.zeros(…) * 2.0`) and the product has no other use, is `tl.dot_add cfg a b c z`, with that
    zero as `z`: either `tl.dot(a, b, c)` or the product followed by the (fusable) addition.
    `tl.dot_scaled_add` is the same for `tl.dot_scaled`.
  - **`tl.sum_bcast_mul`.** A float reduction that sums, over `k`, the products
    `x[i, k] * y[k, j]` of a rank-2 `x` broadcast along a new last axis and a rank-2 `y` broadcast
    along a new first axis is `tl.sum_bcast_mul d x y`, with `d` the dtype of both factors (after
    their conversions): the products summed along a tree with fusable additions, or, when the
    additions are at most 32 bits wide and `M, N ≥ 16`, Triton's rewrite to an IEEE-precision
    `tl.dot`. The combine pass matches the IR, not the spelling, so this is the computation,
    whatever its spelling (library issue L26, [evidence/bcast-mul](../evidence/bcast-mul/README.md)):
    - the factors expanded by `x[:, :, None]` or `tl.expand_dims(x, 2)`, `y[None, :, :]` or
      `tl.expand_dims(y, 0)`, or `tl.reshape`, each with or without `tl.broadcast_to`, in either
      order;
    - the sum by `tl.sum(…, axis=1)` or the method `.sum(axis=1)` (with or without
      `keep_dims=True`, which keeps the axis: `[:, None, :]` of the result), or by
      `tl.reduce(…, 1, f)` whose `f` returns the sum of its two arguments.

    The pass rewrites some spellings and leaves others, and `tl.sum_bcast_mul` allows both results.
    A sum with `dtype=` that converts the products first is not this computation: it is a
    reduction of the converted products. The pass does not check which values the addition adds,
    so a `tl.reduce` whose combine returns `a + a` or `b + b` is rewritten into a dot as well at
    `M, N ≥ 16`; that is not this computation, and the library gives it no meaning yet.
  - **`tl.reshape_any`.** A reshape to rank 1 whose only users are reductions or histograms may
    reorder its elements (below, "Shape changes").
- **Tensor descriptors (TMA)** (`TVA.Lang.Desc`): `TensorDesc` holds the base, shape
  (`int32`), strides, the block shape fixed when it is made, and padding (`"zero"` or `"nan"`): for
  a descriptor the host builds (`TensorDescriptor`), the host's values; for one built in the
  kernel, `tl.make_tensor_descriptor base shape strides block padding`, its arguments' values
  (shape entries `int32`, strides `int64`), with no memory access in the model. On the device each
  program writes such a descriptor (128 bytes) into the launch's scratch, a buffer the allocator
  set by `triton.set_allocator` returns; without an allocator the launch is rejected on the host.
  The scratch is not part of the claim's memory: `Compiler.Faithful` requires that it holds none of
  the bytes the launch is given and that nothing else writes it during the launch ("Launches";
  library issue L5, [evidence/desc-scratch](../evidence/desc-scratch/README.md)).
  Every access has that block. Lane `i` of a block at `offsets` is element `offsets + i`. A
  descriptor that an access goes through has positive shape entries (`TensorDesc.Valid`): on the
  H200 a load, store or reduction through an in-kernel descriptor with a zero shape entry faults,
  whatever its offsets and rank, so it is undefined behaviour in the model; building one without
  accessing it is harmless, and a host-built one is rejected on the host (library issue L5,
  [evidence/desc-zero-shape](../evidence/desc-zero-shape/README.md) Z2–Z13).
  - `tl.desc_load D offsets`: lanes inside the tensor read memory; the others hold the padding.
  - `tl.desc_store`: writes the in-bounds lanes and drops the others.
  - `tl.desc_atomic op` (`atomic_add/min/max/and/or/xor`): one relaxed, `gpu`-scope reduction
    per in-bounds lane, no result. Two programs reducing into the same element race.
  - `tl.desc_gather D xs y` / `tl.desc_scatter`: rows `xs`, columns `y + j` of a 2-D tensor
    with block `[1, BN]`.
    Out-of-bounds gather lanes are unspecified; an out-of-bounds scatter lane is undefined
    behaviour (neither is documented). On the H200 neither compiles (ptxas: `.tile::gather4` needs
    sm_100, `.tile::scatter4` is not on sm_90a), through an in-kernel or a host-built descriptor: a
    launch that uses one is rejected on the host and has no outcome (evidence/desc-zero-shape
    Z14–Z16).
  - Undefined behaviour unless Triton's requirements hold (`TensorDesc.Access`): rank 1–5; positive
    shape entries below `2^31` (above); a 16-byte-aligned base; leading strides that are multiples of 16 bytes and a last stride of
    1; an innermost offset that is a multiple of 16 bytes (documented for load and store,
    required of every access here, a conservative reading); dtypes Triton allows for
    reductions (`RedOp.Allowed`).

  TMA's asynchrony is the compiler's, and we trust it. A program's descriptor accesses are not
  ordered with its other accesses, as for pointers ("Programs"), so a pair with a write on the same
  bytes is a conflict unless both are atomics. Pointer and descriptor accesses to the same bytes
  are out of scope ([exclusions.md](exclusions.md): two atomics, one through each, whose atomicity
  across the proxies is not confirmed, and two reads, until [review-questions.md](review-questions.md)
  Q11 is settled).
- **Block pointers** (`TVA.Lang.BlockPtr`), as Triton 3.8.0's frontend lowers them:
  - `tl.make_block_ptr` casts each `shape`, `strides` and `offsets` entry to `int64`, and keeps the
    source's `block_shape`, `boundary_check` and `padding_option`; without `padding_option` the
    padding is undefined (`undef`);
  - lane `i` has coordinate `offsets[k] + i_k` (`int64`, wrapping) and pointer
    `base + Σₖ coordₖ · strides[k]`;
  - `tl.load_block p bc pad` is a masked `tl.load`: a lane outside `[0, shape[k])` on a
    checked dimension `k ∈ bc` holds the padding (`undef`, `zero` or `nan`); unchecked
    dimensions are not checked;
  - `tl.store_block` is the masked `tl.store`; `tl.advance` adds to the offsets.

  A block pointer is Triton's own aggregate (`_block_ptr`), and these operations are its whole
  meaning (library issue L30): `tl.load`, `tl.store` and `tl.advance` call its methods, which are
  the code above.
- **`tl.dot_scaled`** (`TVA.Lang.Scaled`) is `P.dotScaled cfg`, applied to the operands'
  element bits (unpacked exactly from fp4/fp8/bf16/fp16 storage) and each element's scale bits.
  Each side's scales are `e8m0` (`uint8`/`int8` tensors) or a float dtype (`cfg.lhsScale`,
  `cfg.rhsScale`, e.g. NVFP4's `float8e4nv`), one per group of `16` or `32` elements along `K`
  (S9). So `ScaledCfg` has the source's formats, `k_pack` flags and `fast_math`; `lhsScale` and
  `rhsScale` are `e8m0` for `uint8`/`int8` scale tensors and `float d` for float ones; `group` is
  `K` divided by the scale tensors' last dimension; the operands are the storage tensors as they
  are (packed fp4 as `uint8`). The storage dtypes are Triton's (`ScaledCfg.TypesOk`), and only
  `e2m1` may be packed along a dimension other than `K`. `scaledValue` gives the exact values for
  accuracy rules.
- **Shape changes** are exact: `Tensor.join`, `split`, `interleave` (proved equal to Triton's
  `join` + `reshape`), `reshape` (row-major, `can_reorder=False`), `trans` (rank 2) and
  `permute` (any rank: dimension `k` of the result is dimension `dims[k]` of the input, as
  `tl.permute(x, dims)` and `tl.trans(x, dims)` and their method forms, with `dims` the source's
  list; [evidence/permute](../evidence/permute/README.md)). `tl.trans(x)` without `dims`, and `x.T`,
  on a rank-2 tensor is `x.trans`; `tl.trans(x)` without `dims` on a tensor of rank `r ≥ 3` swaps
  the last two axes, `x.permute [0, …, r - 3, r - 1, r - 2]`.
  `tl.reshape_any` is `tl.reshape(x, s, can_reorder=True)` and `tl.view`: some order of the same
  elements. A reshape to rank 1
  whose only users are reductions or histograms is also `tl.reshape_any`, since the combine pass
  allows it to reorder (S7). An index reduction (`tl.argmax`) of such a reshape then pairs
  reordered values with unreordered indices, as on the device: no index is paired with such a
  reshape as if the order were kept.
- **Scans** (`TVA.Lang.Scan`): `tl.associative_scan x axis op reverse` gives position `j`
  as `op` evaluated along a tree whose leaves are `0, …, j` in order (`reverse`: `n-1, …, j`
  in that order, as Triton lowers it as `flip(scan(flip(x)))`; S8). So with `reverse=True` the left
  operand of `combine_fn` covers the higher indices. `op` is a relation that covers every result
  the body of `combine_fn` can give: the library's `AddRel`, `NaNEq`, `MaxRel` for float bodies, and
  `fun a b c => c = a + b` for an integer addition. The scan keeps the source's axis.
  `tl.cumsum`/`cumsumc` (float `+`, IEEE or fusable), `tl.cumprod`, `tl.icumsum`/`icumprod`
  (wrapping integers). Triton's promotions are written in the model:
  `tl.cumsum` widens `bfloat16` to `float32` and integers narrower than 32 bits to 32 bits
  (keeping signedness); with `dtype=d` it widens `bfloat16` to `float32`, then converts the input
  to `d` (`tl.convert`), with no other widening, and scans in `d` (library issue L87;
  `standard.py:270-281, 334-343`; evidence/front-end-subset S1-S3: `dtype=tl.int8` sums wrap at 8
  bits, `dtype=tl.bfloat16` rounds each sum to `bfloat16`); `tl.cumprod` widens only `bfloat16`.
- **`tl.sort`** (along the last dimension, with the source's `descending`): each row without NaN
  comes out as a permutation of itself up to `==` (`-0`/`+0` may be copied), sorted ascending or
  descending. A row with a NaN
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
- **`tl.cdiv(x, div)`** (and the method `x.cdiv(div)`) is not ceiling division. It is Triton's
  `@triton.jit` function whose body is `(x + (div - 1)) // div` (`triton-3.8.0-standard.py:32-43`),
  which the front end inlines like a helper ("Helpers are inlined"; library issue L88), so it means
  that expression, by the rules of this page
  ([evidence/cdiv-round17](../evidence/cdiv-round17/README.md)):
  - on two compile-time values (`tl.constexpr`s, Python numbers passed directly), Python's
    `(x + (div - 1)) // div`, computed at compile time, and the call's result is a compile-time value
    (`code_generator.py:556-585, 1396`): `tl.cdiv(-5, 4)` is `-1`, `tl.cdiv(-7, -2)` is `5`, and
    `tl.arange(0, tl.cdiv(13, 4))` is accepted (C6). Python's floor division makes it the ceiling of
    `x / div` for a positive `div`;
  - with a tensor operand (a runtime scalar argument, a block, a name assigned a number), the tensor
    operations in the dtype "Operators and promotion" gives: `div - 1` is Python's, computed at
    compile time, when `div` is a compile-time value (`tl.cdiv(M, BLOCK_SIZE_M)`), and a wrapping
    tensor `-` otherwise (library issue L108); the addition wraps, and `//` truncates (`tl.idiv`),
    undefined if `div` is `0` or the quotient overflows. So
    `tl.cdiv(n, 128)` with the `int32` `n = 2^31 - 1` is `-16777215` (the addition wraps) and
    `tl.cdiv(-5, 4)` with an `int32` `-5` is `0` (C1-C5: every `int32` `x` for eight constant
    divisors, `2^24` random runtime pairs and their edges, `int64` and the method form, all as the
    expression gives).
- **Compiler hints** (`tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy`, `tl.assume`)
  are checks (undefined behaviour if false), not no-ops. Triton applies a hint to a value: it is an
  attribute of the operation that defined the value (`set_attr`, `ir.cc:411-425`), and holds at
  every use of the value, also before the call and on paths that never reach it, so its check
  belongs where the value is defined, before any use. Hints apply to integer and pointer tensors
  (`HintVal`: a pointer's divisibility is in bytes, its contiguity in elements).
  - **Which value** (library issues L1 and L60, [evidence/hint-loops](../evidence/hint-loops/README.md),
    [evidence/hint-multi](../evidence/hint-multi/README.md)). A hint marks the operation that defined
    its argument's value when the front end visits the call. A value no operation defines takes no
    mark: a kernel's or a helper's parameter, a loop's induction variable, and, inside a loop's body,
    the value the loop carries. Before it emits a `for` or `while` loop, the front end visits the
    body once with every name bound to its value before the loop and the induction variable bound to
    a placeholder no code uses ("Loops and unrolling"). So:
    - a hint on a value the body defines checks each iteration's value;
    - a hint on a name the loop carries, where the name still holds the carried value, checks the
      value before the loop, once, where an operation defined it, and promises nothing about the
      values of later iterations (C1–C4, C12, C13). With nested loops that all carry the name, it
      is the value before the outermost of them (C8, C9); a value an enclosing loop's body defines
      is checked on each iteration of that body (C10, C11);
    - a hint on a parameter or an induction variable has no effect, and the model drops it (C5–C7);
    - a hint on the value a call of a `@triton.jit` function returns, where Triton inlines the call,
      marks the call, and inlining removes it: the hint has no effect, and the model drops it
      (evidence/hint-multi M12, M13). Every call is inlined except that of a helper declared
      `@triton.jit(noinline=True)`. The calls include the `tl.*` functions Triton writes in Triton,
      and their method forms: `tl.argmax`, `argmin`, `bitonic_merge`, `cdiv`, `cumprod`, `cumsum`,
      `flip`, `interleave`, `max`, `min`, `ravel`, `reduce_or`, `sigmoid`, `softmax`, `sort`,
      `squeeze`, `sum`, `swizzle2d`, `topk`, `unsqueeze`, `xor_sum`, `zeros`, `zeros_like`
      (`triton-3.8.0-standard.py`);
    - **an operation with several results.** When the operation that defined the hinted value has
      several results, the hint may apply to every one of them: the compiler reads it off the
      operation for each result (`AxisInfo.cpp:1239-1242, 1318-1342`), and a true hint on one result
      made the loads through another vectorise, and they faulted (M4–M6, M14). These operations are
      `tl.split` (`a, b = tl.split(z)`), `tl.reduce` and `tl.associative_scan` on a tuple, a call of
      a helper declared `noinline=True` that returns a tuple (outside the subset), and a `for` or
      `while` loop or a runtime `if`, whose results are the names it carries or merges (a hint after
      it on one of them). Such a hint promises the same about each result of that operation, and the
      model checks each, where the operation defines them. The library states that promise when
      every result has the hinted value's shape and, for `tl.multiple_of` and `tl.max_contiguous`, is
      an integer or pointer tensor (`HintVal`; `ConstAlong`, `tl.max_constancy`'s promise, holds of
      any element type). A hint on one result of an operation with a result of another shape (a
      loop carrying a scalar beside a block), or with a float result under `tl.multiple_of` or
      `tl.max_contiguous` (a tuple reduction of an integer and a float tensor, a loop carrying a
      float accumulator), is outside the subset (library issue L86; [exclusions.md](exclusions.md)):
      the readings below are not defined for such a result. A hint after a loop or a runtime `if`
      marks the loop or the `if`, whose mark the compiler does not read (M7–M10), and in some other
      cases the mark is lost (M1); checking each result is then stronger than the device needs, and
      sound (and "The contiguity the analysis infers", below, takes the smaller bound for them);
    - with `tl.static_range` the body is visited once per iteration, and each copy's hint checks
      that copy's value.

    When two hints of one kind mark one value, the later one replaces the earlier; the model checks
    both, which is stronger.
  - **What it promises** (library issues L2 and L58, [evidence/hint-rank2](../evidence/hint-rank2/README.md),
    [evidence/hint-contig](../evidence/hint-contig/README.md)), along each dimension `k` of the hinted
    tensor, for the source's value there (one value for rank 1). Positions are in one group of `C`
    along `k` when they agree on the other coordinates and their `k`-th coordinates have the same
    quotient by `C` (`Idx.SameGroup`). The compiler's analysis believes three things of each value
    per dimension: a contiguity, the divisibility of the first element of each contiguous group, and
    a constancy; a hint replaces one of them (`AxisInfo.cpp:1244-1259`).
    - `tl.max_contiguous(x, C)` alone: `C` is `x`'s contiguity along `k` (`ContigIs`): each group of
      `C` holds consecutive values, and the groups of `2·C` do not (unless `2·C` exceeds the
      dimension). "Each group of `C` holds consecutive values" alone is not sound here: with a `C`
      below the tensor's own contiguity, the compiler keeps the divisibility its analysis inferred
      for the longer groups and applies it to the hinted ones (C1: a hint that holds in that weaker
      sense, groups of 1, made the launch fault).
    - `tl.max_constancy(x, C)`: each group of `C` holds equal values (`ConstAlong`). A false one
      gives wrong values: the compiler computes one value per group (K1).
    - `tl.multiple_of(x, v)` alone: `v` divides the first element of each group of `C`
      (`FirstsDiv`), where `C` is the contiguity the compiler's analysis infers for `x` along `k`,
      with which it pairs `v`. The model takes for `C` a lower bound of that contiguity stated below,
      and `1`, so every element, where none is stated (`AxisInfo.h:31-40, 92-101`). A smaller `C`
      makes a stronger promise, so a lower bound is sound. So `tl.multiple_of(s + tl.arange(0, B), v)`
      promises that `v` divides `s` (`C = B`; H1–H12), and `tl.multiple_of(offs, [1, 4])` with
      `offs = r[:, None] + tl.arange(0, BN)[None, :]` promises that `4` divides each `r[i]`
      (H13–H16).
    - `tl.max_contiguous(tl.multiple_of(x, v), C)`, both on the same value in either order: the pair
      replaces the compiler's contiguity and divisibility, and promises that each group of `C` holds
      consecutive values (`ContigAlong`) and that `v` divides the first element of each group
      (`FirstsDiv`), also with a `C` below the tensor's own contiguity
      (`tl.max_contiguous_multiple_of`; H17–H19).

    **The contiguity the analysis infers**, a lower bound along each dimension `k` of the value the
    hinted operation defines (`AxisInfo.cpp:138-147, 169-183, 242-279, 553-575, 607-668`; H1–H16):
    - a value that carries a `max_contiguous` hint, alone or in the pair (its defining operation is
      marked, "Which value" above): the smaller of the hinted `C` along `k` (with two such hints, the
      smallest) and the bound the items below give the value as if it carried no hint. When the
      compiler reads the mark, the hint replaces what the analysis infers (`AxisInfo.cpp:1232-1259`),
      and its contiguity is `C`. But it reads no mark on the result of a `for` or `while` loop or of a
      runtime `if` (a hint after it marks the loop or the `if`), inlining removes the mark from the
      value a call returns, the mark of an operation with several results is read in some cases and
      lost in others ("Which value" above), and the canonicalizer drops it with an operation it
      folds away (`r + 0`, `r * 1`); then the analysis keeps its own contiguity, which the items below
      bound from below. So the smaller of the two is a lower bound in every case. Values built from
      the hinted value start from that bound: after
      `a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4)`, `a + s`, `s + a` and
      `a.to(tl.int64) + s` have the bound `4`, not `512`
      ([evidence/hint-hinted-bound](../evidence/hint-hinted-bound/README.md) G1–G6: the loads through
      them narrowed from 16 to 4 bytes), and after the pair `C = v = 16` on a loaded `r` (bound `1`),
      `r + s` has the bound `1` (library issue L58;
      [evidence/hint-unread-round17](../evidence/hint-unread-round17/README.md) N0-N10: the loads
      through `x + tl.multiple_of(r + s, 16)` were 16 bytes when an add defines `r`, and 1 byte, as
      without the hint, when `r` is a loop's, a runtime `if`'s, an inlined helper's or `tl.split`'s
      result, or `tl.load(…) + 0` or `* 1`);
    - for a value that carries no `max_contiguous` hint:
      - `tl.arange(a, b)`: `b - a`;
      - `x + y`, `y + x` and `x - y`, where `x` has the bound `C` along `k` (by this list) and `y` is
        constant along `k` (a scalar, a Python number, or a tensor of size 1 along `k` that the
        operation broadcasts): `C`; for pointers, `p + x` and `x + p` likewise (in elements);
      - `x[:, None]`, `x[None, :]` and `tl.expand_dims` keep the bound of the other dimensions and
        give the new one `1`; a broadcast (implicit, or `tl.broadcast_to`) keeps the bound of each
        dimension it does not broadcast;
      - a conversion between integer dtypes (`x.to(tl.int64)`) keeps `x`'s bound.

    Anything else has the bound `1`. A `tl.multiple_of` alone on a sub-expression does not change
    its bound (G4). So tutorial 08's `tl.multiple_of(a_ptrs, [16, 16])` promises
    that every pointer of the block is 16-byte aligned (dimension 0 has the bound `1`, and the
    compiler too applies its value to every element), which no launch meets (`a_ptrs[0, 1]` is 2
    bytes past `a_ptrs[0, 0]`; R1, R2; the kernel is excluded, [exclusions.md](exclusions.md)), and
    tutorial 09's `tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)` holds on a
    full tile and is false on a partial one, where the compiled loads read other elements than the
    source names (T2; covered on full tiles only, exclusions.md). A claim covers only instances on
    which its hints hold. The library gives these readings for values that are powers of two, at
    most the dimension's size for `max_contiguous` and `max_constancy`; other values are excluded
    ([exclusions.md](exclusions.md)).
  - The forms: `tl.assume`; for rank 1, `tl.max_contiguous`, `tl.max_constancy`,
    `tl.multiple_of x v contig` (`contig` a lower bound stated above, default `1`) and
    `tl.max_contiguous_multiple_of x C v` for the pair, and `tl.multiple_of1` for a scalar; for
    rank ≥ 2, `tl.max_contiguousN`, `max_constancyN`, `multiple_ofN` and
    `max_contiguous_multiple_ofN`, with one value per dimension, as the source gives them.
- **Loops.**
  - `for i in range(lo, hi, step)` and `tl.range` are `tl.forRange` (`range(hi)` with `lo = 0`,
    `range(lo, hi)` with `step = 1`; a one-argument `range` evaluates its argument twice, "Loops
    and unrolling"). The induction variable
    has Triton's promoted bound type (`int32`, or `int64` when a bound is). It is undefined
    behaviour unless `tl.RangeOk`: `step > 0`, and `lo`, `step` and `hi + step − 1` within the
    *signed* range. `scf.for` compares signed even for unsigned bounds.
  - A loop's carried variables are its state (a tuple).
  - `tl.static_range` is unrolled by the frontend with a Python integer as the variable, the same
    count of times; the model writes the unrolled body.
  - Fixed counts are `tl.forN`.
  - `while` is `tl.whileLoop`, with the source's condition.
  - The scheduling keywords of `tl.range` (`num_stages`, `loop_unroll_factor`,
    `warp_specialize`, …) change no outcome of a well-behaved model: pipelining (`num_stages`) can
    make an iteration miss what an earlier one stored, but that is a program reusing its own data,
    a conflict within the program, which no well-behaved model has ("Programs", library issue L15).

- **Random numbers** (`TVA.Lang.Random`): `tl.randint`/`randint4x` are Triton's Philox 4x32-10,
  exactly (checked against the Random123 known-answer vectors). `tl.rand` is
  `uint_to_uniform_float` of it: an `tl.itof` conversion and an IEEE product with the `float32`
  constant `4.6566127342e-10`. The result is a product, and the function is inlined, so an
  addition that consumes it (`tl.rand(…) - 0.5`) may fuse: model it with `tl.addc`/`tl.subc`.

## Launches (`TVA/Sem/Launch.lean`)

- `Launch` = grid `(x, y, z)`, `enable_fp_fusion` (Triton's default `true`), `debug` (default
  `false`), `num_ctas` (default `1`), `pdl` (`launch_pdl`, default `false`) and `blocks`, the
  source's block shapes on this instance. The grid is the host's grid formula, and
  `tl.program_id(a)` reads axis `a` of it. `Launch.Valid` checks only the shapes `blocks` lists, so
  an empty or partial list passes it.
  - **Memory.** `init` gives the launch its bytes: the bytes present in it (`Mem` is a partial
    map), the buffers of the claim's `Pre`. The outcome `ok fin` gives the same bytes, with their
    contents when the launch ends; other bytes are not part of the outcome, and `Post` cannot speak
    of them (library issue L49). The model accesses no other byte, and `Compiler.Faithful` states
    that a valid, well-behaved launch within its scope writes none of them, except the scratch of
    the descriptors the kernel builds ("Tensor descriptors"; library issue L73,
    [evidence/launch-frame](../evidence/launch-frame/README.md): kernels with masked tiles,
    reductions, a pipelined `tl.dot`, atomics, sorts and scans changed no byte outside their output
    buffers, and a descriptor-building one none outside its output and its scratch). So a claim
    that cites the rule implies that the launch leaves the host's other memory as it was
    (`TVA.Claim`, "Other memory").
  - **One launch** (library issue L21, [evidence/autotune-launch](../evidence/autotune-launch/README.md)).
    A claim is about one launch. A call of a `@triton.autotune`d kernel is host code: on a key it
    has not timed, it runs the kernel many times on the caller's buffers (thousands of runs of an
    in-place `+1`), and runs `reset_to_zero`, `restore_value`, its `pre_hook` and `post_hook` and
    each config's `pre_hook`, which write those buffers; then it makes one launch with the config
    it chose, after that config's `pre_hook` (on a cached key too). The launch a claim covers is
    that last one, with one of the configs the autotuner may pick, from the memory after the last
    hook, its `init`, and with the arguments the last hook leaves: a hook may change them as well
    (tutorials 06 and 09 set a host `TensorDescriptor`'s `block_shape` in a config's `pre_hook`).
    What the caller's memory was before the call is the host's part, outside the claim. A kernel
    under `@triton.heuristics` is outside the subset: a heuristic sets launch keywords, compile
    options included, and replaces the launch's own (library issue L78; below). A call made while a
    CUDA stream capture is in progress (`torch.cuda.graph`, `triton.testing.do_bench_cudagraph`), on
    any stream, is not a launch a claim covers, and neither is a replay of the captured graph (library
    issue L91; below). A call on the capturing stream returns without an error, compiling the kernel
    first if needed, and runs no program, and its programs run at each `graph.replay()`, with the
    arguments the call captured, on the memory at the replay; a call on another stream during the
    capture runs at once, as any launch, and no replay runs it
    ([evidence/host-round20](../evidence/host-round20/README.md) G0–G3).
  - `Compiler.Faithful` covers a launch only within its scope
    ([rules/compiler-faithful.md](rules/compiler-faithful.md), "Scope"; library issues L17–L19, L72,
    L74, L76–L78, L91, L103–L105, L107, L109, L110):
    in two parts (the user's decisions, 2026-10-29: "To make the verifier easy, we do not verify
    triton cache.", "we assume the kernel is new." and "So basically, we want to verify the kernel
    itself, rather than triton installation or other code programmed by users."):
    - **(A) the kernel and its launches, checked** by the kernel reviewer against the kernel's
      source and its launch call (scopes 4, 5, 6, 8 and 9):
      - **only the launch options `Launch` records**, and `num_warps`, `num_stages`, `maxnreg` and
        `launch_cooperative_grid`, which do not change the meaning: no other compile option
        (`ir_override`, `allowed_dot_input_precisions`, `enable_reflect_ftz`, `extern_libs`, …) given
        to the call that compiles the kernel (a JIT launch, `kernel.warmup(…)`, `triton.compile`'s
        `options`) or in an autotune `triton.Config` (library issue L72,
        [evidence/compiled-options](../evidence/compiled-options/README.md) O1–O4: `enable_reflect_ftz=False`
        given at `warmup` reached a launch that gave no keyword), and no `@triton.heuristics`, which
        sets launch keywords, options included, before the JIT reads them (`autotuner.py:473-483`;
        library issue L78);
      - **descriptor scratch**: when the kernel builds descriptors (`tl.make_tensor_descriptor`), the
        allocator's buffer holds none of the bytes `init` gives the launch, and nothing else writes
        it during the launch ("Tensor descriptors");
      - **a precompiled kernel's specialisation** (library issue L47,
        [evidence/compiled-launch](../evidence/compiled-launch/README.md)): a launch through a
        `CompiledKernel` (`kernel.warmup(…)` or `triton.compile`, then `compiled[grid](…)`, as
        tutorial 02 does) runs the code compiled for the warm-up's arguments, and nothing checks the
        new ones against it: outside
        it, a launch faulted (a misaligned pointer), wrote past `n`, or wrapped an integer. Such a
        launch is covered, as the launch of the warm-up's specialisation, only when each argument is
        in it: the same dtypes, each pointer 16-byte aligned exactly when the warm-up's was, each
        integer in the same class ("Scalar arguments": equal to `1`; otherwise a multiple of 16 or
        not, and `int32`, `int64` or `uint64` range), and each host `TensorDescriptor` with the
        warm-up's dtype and block shape (its specialisation; the launcher builds the TMA descriptor
        with the compiled block shape: a `[16, 16]` descriptor given to a kernel compiled for
        `[32, 32]` had 1024 elements copied, evidence/compiled-options O6). Its constexpr positions are
        ignored: the warm-up's values hold. A kernel `triton.compile` built has the specialisation of
        its signature and attributes, and is covered only when `triton.compile`'s source is an
        `ASTSource` of the kernel's `@triton.jit` function: given a file path, `triton.compile`
        compiles that IR file (a `.ttir`, `.ttgir` or `.ptx` file; `compiler.py:87-112, 233-240`),
        which replaces the source, as `ir_override` does (library issue L77);
      - **host descriptors that load what memory holds**: no host `TensorDescriptor` the launch is
        given has `round_f32_to_tf32=True`. With it, the launcher builds a TMA descriptor whose loads
        round `float32` to tf32 (`nvidia-driver.py:231-232`; library issue L76,
        [evidence/desc-tf32](../evidence/desc-tf32/README.md): `1 + 2^-23` loaded as `1.0`), which
        `tl.desc_load` does not model;
      - **no CUDA graph capture**: the call is not made while a CUDA stream capture is in progress, on
        any stream (`torch.cuda.graph(…)`, `torch.cuda.CUDAGraph.capture_begin`,
        `triton.testing.do_bench_cudagraph`, a `cudaStreamBeginCapture`), and the launch is not a replay
        of a captured graph. A call on the capturing stream returns without an error and runs no
        program, and each replay runs the programs with the captured arguments on the memory at the
        replay (`docs/library-review/round-16/lr-16b/work/graph_capture.py` G1-G3: the output kept its
        sentinel after the captured call, held `x + 1` after the first replay and `11` after the host
        set `x = 10` and replayed, and a captured `n = 16` wrote 16 elements whatever the host's `n` was
        later); a call on another stream during the capture runs at once, and a replay does not run it
        ([evidence/host-round20](../evidence/host-round20/README.md) G0–G3). The library states none of
        these, and excludes all of them ([exclusions.md](exclusions.md); library issue L91; user,
        2026-10-28).
    - **(B) the environment, assumed, not verified**: TVA verifies the kernel, not the Triton
      installation or other code users write, and the gate, the prover and the kernel reviewer do
      not check these (scopes 1, 2, 3, the environment's part of 4, 7, 10 and 11); a claim's own
      launch or kernel source that visibly breaks one is still a finding:
      - **its bytes**: from its start to its end, nothing else writes a byte `init` gives the launch:
        not the host, another kernel on any stream or device, or the next launch on its stream
        started early by programmatic dependent launch after this kernel's `gdc_launch_dependents()`
        ([evidence/pdl](../evidence/pdl/README.md) part D). Writes to other bytes do not matter,
        concurrent or not: the device's stores are byte-granular, and concurrent writes to the
        neighbouring bytes, in the same 4-byte words, changed nothing
        ([evidence/other-writers](../evidence/other-writers/README.md); library issue L49);
      - **Triton's knobs at their defaults**, however they would be set, by the environment or from
        Python: for example `TRITON_F32_DEFAULT` or `triton.knobs.language.fp32_default` changes
        the `input_precision` of a `tl.dot` that does not give one, and `TRITON_DEFAULT_FP_FUSION`
        or `knobs.language.default_fp_fusion` the default of `enable_fp_fusion`
        ([evidence/env-knobs](../evidence/env-knobs/README.md),
        [evidence/launch-options](../evidence/launch-options/README.md));
      - **no hook installed** (`triton.knobs.runtime.*_hook`, the listeners, a `JITFunction`'s
        pre-run hooks): a hook can rewrite the compiled code or skip the launch;
      - **no `debug` attribute set from outside and no `preload`** (library issue L105): no code but
        the kernel's source sets the kernel's `JITFunction` attribute `debug` (`kernel.debug = True`
        gives `Launch.debug`, below), and no `JITFunction.preload` of the kernel is made, which installs
        code compiled with the options its data names under the key a plain launch looks up
        ([evidence/host-round20](../evidence/host-round20/README.md) D1, P1: a launch with no option ran
        code compiled with `enable_reflect_ftz=False`);
      - **the target**: the launch runs on an NVIDIA H200 (sm_90), with code Triton compiles for the
        device it runs on (no `arch` option, no `TRITON_OVERRIDE_ARCH`, no `triton.compile` target
        other than the device's: one for sm_80 was rejected on the host, O5). The model's defaults
        (`DotCfg`'s `tf32`, the fp8 `maxImpreciseAcc = 2^30`) and the device facts (`Rules.Num.Sm90`)
        are the H200's (library issue L74). The host is an x86-64 machine running CPython 3.13, where
        TVA's evidence ran (library issue L107; the page's introduction names the values that are the
        host's), run without `-O` or `-OO` and with `PYTHONOPTIMIZE` unset, in the default
        floating-point environment (round to nearest even; flush-to-zero and denormals-are-zero off: no
        `fesetround`, no `_mm_setcsr`, no loaded library that changes them; library issue L110): under
        `-O` a kernel the front end rejects through a Python `assert` ran
        ([evidence/host-round21](../evidence/host-round21/README.md) H1: "Names after a loop"), and after
        `fesetround(FE_TOWARDZERO)` the constant `tl.full([1], 0.1, tl.float32)` and the argument `0.1`
        were `0x3dcccccc`, not `litF32`'s and `hostF32`'s `0x3dcccccd`, and with flush-to-zero set
        `tl.full([1], 1e-40, tl.float32)` was `0` (H2, H3);
      - **Triton and the kernel's module as installed and written** (library issue L104): the process
        imports Triton 3.8.0's installed package, unmodified (no other `triton` earlier on `sys.path`,
        no substitute in `sys.modules`); no code but Triton's own, at any time before the launch ends,
        changes an attribute of a Triton module, of a class Triton defines or of an object of one, or of
        the kernel's or a helper's `JITFunction`, by assignment, by `setattr` or `delattr`, or by a call
        made to change it (`tl.exp = tl.log`, a `JITFunction`'s `fn`, `src` or `noinline`, its
        `_unsafe_update_src`, `triton.runtime.driver.set_active`, `importlib.reload`), other than through
        Triton's API as the in-scope tutorials use it (launches, `kernel.warmup(…)`, `triton.autotune`
        with its `triton.Config`s and hooks, the `JITFunction`'s `debug`, above, `triton.set_allocator`,
        the knobs the scope allows, and the fields of a host `TensorDescriptor` the launch is given, such
        as the `block_shape` a config's `pre_hook` sets, "One launch"); and no code binds a name of the
        kernel's module other than as the module's own statements do (another module's
        `kmod.double = kmod.triple`, an assignment into `kernel.fn.__globals__`): the front end compiles
        what these hold, and such a change gave `3·x` for the source's `2·x`, `log x` for `exp x` and
        `5·x` after `_unsafe_update_src` for `2·x`
        ([evidence/host-round20](../evidence/host-round20/README.md) M1–M4);
      - **the kernel is new** (library issues L98, L103, L109; the user's decisions, 2026-10-29: "To
        make the verifier easy, we do not verify triton cache." and "we assume the kernel is new."): the
        code the launch runs was compiled for that launch, from its own source, its compile-time values
        (its `tl.constexpr` arguments, a config's values, an integer `1` or `None` passed to a parameter
        without annotation) and the settings the conditions above give; it is not code compiled
        earlier, for other values or under other settings, and taken from Triton's caches (the
        `JITFunction`'s cache in the process, or the cache directory). Code compiled earlier for the
        same source, values and settings (an autotuner's benchmark run of the chosen config) counts as
        compiled for it. TVA assumes this and does not verify Triton's caches: neither the gate, nor the
        prover, nor the kernel reviewer checks it. The caches' keys equate some different compile-time values
        and record no knob set from Python, no `TRITON_KERNEL_OVERRIDE`, no hook, no change to Triton in
        memory and no state of the host process, so code taken from them may differ from the launch's
        own ([evidence/compile-keys](../evidence/compile-keys/README.md) K1–K4,
        [evidence/host-round21](../evidence/host-round21/README.md) K1–K4 and X1–X5; "Compile-time
        values Triton keys apart").
  - **`launch_metadata=`** (an argument of `@triton.jit`, tutorial 09) changes no result, and a model
    omits it. It names a host function that Triton calls only to hand metadata (a name, a count of
    operations) to a launch-enter hook, and only when such a hook is installed, which the scope
    above excludes; without a hook the function is never called, and the compiled kernel is the same
    with it or without it ([evidence/front-end-order](../evidence/front-end-order/README.md)).
  - `Compiler.Faithful` covers only `debug = false`, since with `debug` asserts and integer
    overflow checks trap. `debug` is on when the call that compiles the kernel gives `debug=True`
    (the launch, `warmup`, `triton.compile`'s options, the chosen `triton.Config`), when that call
    gives none and the kernel's `JITFunction` attribute `debug` is true (the decorator's
    `@triton.jit(debug=True)`, or `kernel.debug = True` from Python, `jit.py:727`: with it an `int32`
    overflow trapped, [evidence/host-round20](../evidence/host-round20/README.md) D1), and whenever
    `TRITON_DEBUG=1` or the knob `triton.knobs.runtime.debug = True` asks for it (`Launch.debug`;
    library issue L105).
  - It also covers only `Launch.Valid` launches: grid `x < 2^31` and `y, z < 2^16`, fewer than
    `2^31` programs (`x·y·z < 2^31`), `num_ctas = 1`, no programmatic dependent launch
    (`pdl = false`), and every block shape a power of two in each dimension with at most `2^20`
    elements. With `launch_pdl`, a kernel may start before the previous kernel on the stream has
    finished writing, so its programs may read memory other than the launch's `init`. Without it,
    `tl.extra.cuda.gdc_wait()` changes nothing, and `gdc_launch_dependents()` changes nothing as
    long as the next launch on the stream does not use `launch_pdl` (or writes none of the
    launch's bytes); the model drops them then ([evidence/pdl](../evidence/pdl/README.md)). Triton 3.8.0's launcher tests `x·y·z > 0` in C `int`
    and returns without an error, running no program, when the grid has `2^31` to `2^32`
    programs (`triton-3.8.0-nvidia-driver.c:968-972`;
    [evidence/launch-outcomes](../evidence/launch-outcomes/README.md)).
  - A launch reported to the host with an error before any program runs (shared memory,
    registers, threads, `tl.dot` sizes, dtypes, grid limits) has no outcome
    (`TVA.Claim.Behaviour`). A launch that returns without an error has one, even if no program
    ran: with a zero grid dimension it is `ok` with memory unchanged. (A call on a stream that is
    capturing into a CUDA graph also returns without an error and runs no program, but every call
    made during a capture is outside the scope above.)

  - **Scalar arguments** (library issue L29, [evidence/scalar-args](../evidence/scalar-args/README.md);
    `specialize.cc:264-306, 377-388`, `jit.py:396-458`, the launcher's extractors
    `nvidia-driver.c:1090-1163`). The type of a scalar parameter without annotation depends on the
    argument's value at each launch:
    - an `int` equal to `1` is a compile-time constant (`tl.constexpr`: Python's unbounded integers
      and floor division), unless the parameter is listed in `do_not_specialize`; so `(n - 2) // 2`
      is `-1` there, and `0` for an `int32` `n = 1`;
    - another `int` is a runtime `int32` in `[-2^31, 2^31)`, `int64` in `[-2^63, 2^63)` and `uint64`
      in `[2^63, 2^64)`, and is rejected on the host beyond; a multiple of 16 (also `0`) is also
      marked divisible by 16 for the compiler, which is true of it;
    - a `bool` is a runtime `int1`; a `float` (or a `numpy.float64`) is a runtime `float32`, the
      double rounded to nearest even (`FloatV.hostF32` of the double, C's `(float)`: an overflow is
      `±inf`, `-0.0` stays, a NaN keeps its sign and the top bits of its payload; library issue
      L98, evidence/literals N4); other numpy scalars are rejected on the host;
    - `None` is a compile-time constant, and a 0-d tensor is passed as a pointer to its data, not as
      its value.

    With an annotation the parameter has the annotated type, and an integer is never a constant:
    `tl.int8`, `tl.int32` (and `int`) and `tl.uint32` take the value modulo `2^w` (a negative value
    for `tl.uint32` is rejected), `tl.int64` the exact value (out of range: rejected); `tl.float64`
    the double, `tl.float32` the double rounded to nearest even (`FloatV.hostF32`), `tl.float16` the
    double rounded to nearest even (a finite value that rounds to `±inf` is rejected on the host; a
    NaN is `0x7e00` or `0xfe00`, its payload dropped, by CPython 3.13's `PyFloat_Pack2`, the host's:
    another CPython may keep payload bits), `tl.bfloat16` the top 16 bits of
    `FloatV.hostF32` of the double (truncation, not rounding); `tl.int1` the lowest bit. Other
    annotated scalar types are excluded ([exclusions.md](exclusions.md)). A model takes each scalar
    argument as a value of its type, and `Inst.ok` fixes its class where the model's meaning
    depends on it (for instance `n ≠ 1`, or `n < 2^31`).

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
- **Conflict within a program** (`IntraRace`): two events of the same program on the same
  bytes, one writing, unless both are atomics on the same bytes (see "Programs").
- **`WellBehaved`.** No reachable configuration has a program whose next step is undefined
  behaviour, and no trace has a race between programs or a conflict within a program.
- **`Outcomes`.**
  - `ok fin`: all programs finished with memory `fin`.
  - `ub`: some interleaving reaches undefined behaviour.
  - `hang`: some interleaving runs forever.

## No barrier semantics

The documentation gives `tl.debug_barrier()` only as "Insert a barrier to synchronize all
threads in a block". The language has no threads, and the user decided (2026-10-26): "we do not
need debug semantics here". A model never contains a barrier. A source `tl.debug_barrier()` is a
no-op in the model, and the reviewer sees it in the correspondence.

It keeps no meaning after L15 either. A barrier would order a program's accesses, but in a
well-behaved model the only accesses of a program to the same bytes whose order matters are two
atomics (their old values, `xchg`, `cas`), and the semantics already performs them in every order
(`Step.defer`), so a barrier could only remove behaviours the model has. A model that writes and
then reads back its own data has a conflict within the program (see "Programs") whether or not the
source has a barrier, and no claim covers it. `num_stages` stays dropped for the same reason: the reuse
of a program's own data that pipelining could reorder is excluded. A model of phases separated by
barriers (L15's option 2) may be added later without breaking claims made under this rule.

## Granularity

A tensor load or store is one step of the interleaving. Finer interleavings, such as per lane or
per thread, differ only in launches with a race between programs or a conflict within a program,
and those launches have no meaning (`WellBehaved` fails). The rule then makes no claim about
them. Atomics are per lane already, and `Step.defer` gives them every order a program's
dependences allow.
