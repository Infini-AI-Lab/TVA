# What a Triton program means in TVA

This page says what each construct of a Triton 3.8.0 kernel means as TVA models it, and where in
`lean/TVA/` that meaning is stated. It is for two readers: someone who checks a claim's model
against the kernel's source, and someone who writes a claim. The Lean definitions are the meaning;
this page points to them and explains them, and does not repeat them. Where the page and a
docstring disagree, the Lean is right.

What a reader of a proved claim trusts is listed in `TVA.Claim` (`lean/TVA/Claim.lean`): Lean and
Mathlib, that the model is the kernel, that Triton compiles the kernel as the model says, and the
device facts. Everything below is about the second item: how to read a model, and how to write one
that means what the source does.

Contents: [the shape of a model](#the-shape-of-a-model) ·
[values](#values) · [how the front end reads the source](#how-the-front-end-reads-the-source) ·
[Python numbers and promotion](#python-numbers-and-promotion) · [integers](#integer-arithmetic) ·
[floats](#float-arithmetic) · [conversions](#conversions) · [pointers, loads and
stores](#pointers-loads-and-stores) · [block pointers](#block-pointers) ·
[descriptors](#tensor-descriptors) · [`tl.dot`](#tldot-and-tldot_scaled) ·
[reductions](#reductions) · [scans and sorting](#scans-and-sorting) · [atomics](#atomics) ·
[random numbers](#random-numbers) · [loops](#loops) · [hints](#compiler-hints-and-assertions) ·
[undefined behaviour](#undefined-behaviour) · [launches](#launches) · [device
facts](#the-device-facts) · [exclusions](#exclusions).

## The shape of a model

A claim's model is a `Kernel` (`TVA.Launch.Run`): for each program's context `Ctx`
(`tl.program_id`, `tl.num_programs`), the program it runs, a `Prog Unit` (`TVA.Program`). A
`Prog` is written in `do` notation and follows the source line by line, keeping its names:

```lean
def add_kernel (x_ptr y_ptr output_ptr : Ptr .f32) (n_elements : I32) (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    let pid := tl.program_id ctx 0
    let block_start := pid * (IntV.ofNat BLOCK_SIZE : I32)
    let offsets := block_start + tl.arange 0 BLOCK_SIZE
    let mask := offsets.ltS n_elements
    let x ← tl.load (x_ptr + offsets) (some mask)
    let y ← tl.load (y_ptr + offsets) (some mask)
    let output ← tl.add x y
    tl.store (output_ptr + offsets) output (some mask)
```

The conventions, stated in `TVA.Triton.Core`'s module docstring:

- **Exact operations are pure Lean; the others are programs.** An operation whose result Triton
  defines exactly (integer arithmetic, comparisons, `tl.where`, shape changes, integer
  reductions) is a function on tensors, bound with `let x := …`. An operation that rounds, that the
  device computes only up to an accuracy, that may be undefined, or that touches memory returns a
  `Prog`, bound with `let x ← …`.
- **Implicit steps are explicit.** Triton broadcasts operands and converts mixed dtypes without
  saying so; a model writes each broadcast (`bc`, `.col`, `.row`, `tl.broadcast_to`) and each
  conversion (`tl.convert`, `tl.cvt`, `tl.itof`, …).
- **Compile-time values are Lean values.** A `tl.constexpr` parameter (a block size, a flag, a
  dtype) is a Lean parameter of the model, and Python arithmetic on such values is Lean arithmetic
  (`tl.py.floordiv`, `tl.py.mod`, `tl.py.cdiv` for Python's floor division). A shape is a
  `List Nat`. A compile-time `if` is a Lean `if` on the parameter.
- **What the device chooses is a `pick`.** When Triton leaves a result open (a masked-out lane
  without `other`, an approximate `tl.exp`, a `tl.dot`, the tree of a float reduction, the order
  of an atomic's lanes), the operation is `Prog.pick P k`: the program continues with *some* value
  satisfying `P`, and a proof must hold for each. When nothing satisfies `P`, the `pick` is
  undefined behaviour; that is how a model states a precondition (`tl.assume`, a division by
  zero).
- **NaN payloads are unspecified.** Every exact float operation returns its IEEE result up to NaN
  payload (`tl.upToNaN`, `FloatV.NaNEq`), so a specification compares floats up to NaN payload.

The effects a program can have are the constructors of `Prog`: `load`, `store`, `rmw` (one
element of an atomic), `pick`, `ub`, `loop` and `env` (reading the platform and the
`enable_fp_fusion` flag). Their meaning is one step each, in `TVA.Launch.Step`.

**A program is not sequential for global memory.** A program is many GPU threads with no barrier
between its operations, so a load need not see a store the same program made before it, and two of
its stores to the same bytes may land in either order. The semantics does not order them; instead
a launch in which one program accesses the same bytes twice, at least once writing, is not well
behaved, and no claim covers it ([launches](#launches)). Two atomics on the same element are the
exception. `tl.debug_barrier()` orders nothing here, and a model writes nothing for it.

## Values

| Triton | Lean | Module |
|---|---|---|
| `tl.int8` … `tl.int64`, `tl.uint8` … `tl.uint64` | `IntV signed w` (`I8` … `I64`, `U8` … `U64`) | `TVA.Base.Int` |
| `tl.int1` | `Bool` | `TVA.Base.DType` |
| `tl.float16`, `tl.bfloat16`, `tl.float32`, `tl.float64` | `FloatV f` (`F16`, `BF16`, `F32`, `F64`) | `TVA.Base.Float.Value` |
| `tl.float8e5`, `tl.float8e4nv` | `F8E5`, `F8E4` | `TVA.Base.Float.Value` |
| a dtype | `DType` (`.i32`, `.f16`, …); `DType.Val d` is the Lean type of its values | `TVA.Base.DType` |
| a block of shape `s` | `Tensor s α`, a value at each index `Idx s` | `TVA.Tensor.Basic` |
| a pointer to `d` | `Ptr d`, a byte address | `TVA.Memory` |
| a host tensor argument | `Buf d`: its base address and length | `TVA.Memory` |

An integer is held as its bit pattern; signedness only says how the pattern is read (`toInt`) and
which LLVM operation an operator lowers to. A float is held as its bit pattern, so NaN payloads,
signed zeros and infinities are kept; `toRat`, `toReal` give the exact value of a finite pattern.
An element of `d` takes `d.bytes` bytes, little-endian (`TVA.Base.Bytes`); `tl.int1` takes one
byte and reads as `true` when the byte is not `0`.

Not covered: `tl.float8e4b15`, the AMD fp8 formats, pointers to pointers, 64-bit Philox, `tl.randn`
and 3-D (batched) `tl.dot_scaled`.

**Scalar arguments.** A runtime integer argument without annotation is an `int32` when its value
is in `[-2^31, 2^31)`, an `int64` up to `2^63`, a `uint64` beyond. An integer argument equal to
`1` is special: Triton compiles the kernel with the Python constant `1` in its place (unless the
parameter is in `do_not_specialize`). A model may take it as an `int32` all the same when every
use converts it to a tensor first (an arithmetic operation or a comparison with a tensor does);
otherwise the claim's model says how it treats `1`, and `Inst.ok` fixes the class of each
argument it relies on (for example `n < 2^31`). A Python `float` argument is a `float32`, the
double rounded to nearest (`FloatV.hostF32`). `None` is a compile-time value.

## How the front end reads the source

Triton's front end walks the kernel's Python source and emits one operation for each operation it
evaluates; what runs on the device is what it evaluated. The facts a transcriber most needs:

- **Helpers are inlined.** A call of a `@triton.jit` function means its body, with the call's
  arguments bound as Python binds them (by position, by keyword, then the defaults). A Python
  number passed directly, or a default, stays a compile-time value in the body. The model writes
  the body in place, or as a Lean function applied at the call. `tl.cdiv`, `tl.sigmoid` and
  `tl.softmax` are such functions of Triton's own (see [integers](#integer-arithmetic) and
  [floats](#float-arithmetic)).
- **A name assigned a Python number holds a tensor.** After `e = 1e-7`, `a = -7` or `i = 0`, the
  name holds a scalar tensor of the number's own dtype (`Operand.ownType?`), not a Python number:
  `x * e` with a `bfloat16` `x` computes in `float32`, and `a // 2` truncates. The exception is
  `name: tl.constexpr = e` on a name not defined before, which keeps the Python value
  (`TVA.Triton.Promote`'s docstring).
- **A compile-time condition picks one branch.** An `if` on a compile-time value (a
  `tl.constexpr`, `x is None`, `isinstance(…)`) visits only the branch it picks; the other has no
  meaning. In `a and b` and `a or b`, a compile-time operand that decides the result ends the
  evaluation; tensor operands are all evaluated. `a if c else b` with a compile-time `c` evaluates
  one side.
- **A runtime `if`** (its condition a tensor) runs, in each program, the branch its condition
  takes, and only that one: the model is a Lean `if` on the condition's value (after `c != 0` for a
  condition that is not `int1`). A `return` in a runtime `if` ends the program (or the inlined
  helper) for the programs that take it: the model returns at the same point. `a if c else b` with
  a runtime `c` is such an `if`, whose sides become tensors of their own dtype.
- **Every argument of a call is evaluated**, even when the call does nothing at run time: an
  operation inside `tl.static_assert`'s condition, a print or `tl.device_assert` runs. Python's
  `float`, `int`, `len`, `isinstance`, `getattr`, `hasattr` and `list` run at compile time and add
  no step; `min`, `max` on tensors are `tl.minimum`/`tl.maximum` (`tl.iminimum`/`tl.imaximum` on
  integers).
- **Debugging calls.** `tl.static_print` prints at compile time; `tl.device_print` and `print`
  write no global memory; without `debug` (which every covered launch has), `tl.device_assert` and
  `assert` emit nothing. A model drops them, keeping any operation their arguments evaluate.
- **Loops.** `tl.static_range` is unrolled: the body once per value, with a Python integer as the
  variable. Before a `for` or `while` loop the front end visits the body once to find the names it
  carries, and discards what that visit emits. `range(n)` with one argument evaluates `n` twice.
  A loop carries the names defined before it that its body assigns ([loops](#loops)).
- **Performance-only arguments** change no result of a well-behaved launch and are omitted:
  `cache_modifier`, `eviction_policy`, `volatile`, `num_warps`, `num_stages`, `maxnreg`, the
  scheduling keywords of `tl.range`, `launch_metadata=`. Not omitted: `enable_fp_fusion`
  (`Launch.fusion`), and `input_precision`, `max_num_imprecise_acc` of `tl.dot` (`DotCfg`).

## Python numbers and promotion

`TVA.Triton.Promote` is the one statement of Triton's dtype promotion, and
`TVA.Base.Float.Literal` of the constants Triton builds for Python numbers. A model writes each
implicit conversion as `tl.convert d d' x` to the dtype these definitions give; a prover
evaluates them on the kernel's dtypes with `decide` or `rfl`.

- **Which dtype an operator computes in.** `computationType?`, `BinOp.typing?`, `UnOp.typing?`,
  `whereTyping?` and `boolOp?` give, per operator, the dtype both operands are converted to, the
  result's dtype, and whether the result is a block, or `none` when the front end rejects it on
  the host. Some consequences: `float16` with `bfloat16` computes in `float16`; two `bfloat16`
  under `/`, `//`, `%` compute in `float32`; two integers under `/` are both converted to
  `float32`; an `int32` with an `int64` sign-extends; mixed signedness takes the unsigned type if
  it is at least as wide.
- **A Python number beside a tensor.** As an operand of `+ - * / // % & | ^ << >>`, as a value of
  `tl.where`, or in `tl.full`, a Python number `c` is built directly in the dtype the operation
  computes in (`Operand.constIn?`), and does not widen the tensor: `x * 0.1` with a `bfloat16` `x`
  computes in `bfloat16`. Everywhere else (a comparison, the condition of `tl.where`, an argument of
  another `tl.*` function, a stored value, `other=`, `tl.cast`) it first becomes a tensor of its own
  dtype (`Operand.ownType?`: an `int` is `int32`, `uint32`, `int64` or `uint64`, the first that
  holds it; a `float` is `float32` when zero, infinite, NaN or in `float32`'s normal range, else
  `float64`), and then the operation's own promotion applies: `x < 0.1` with a `float64` `x`
  compares against `float32(0.1)`, widened.
- **The constant built for a Python float** (`TVA.Base.Float.Literal`): in `float64`, the double
  (`litF64`); in `float32`, the nearest `float32` (`litF32`); in `float16`, the nearest `float16` to
  that `float32` (`litF16`, two roundings); in `bfloat16`, the `bfloat16` nearest to the six-decimal
  text of that `float32` (`litBF16`: `1e-7` is `+0`); in fp8, the `float32` constant when the fp8
  format holds it exactly, else the kernel is rejected on the host (`litF8`). Every zero, `-0.0`
  included, becomes `+0`. Each function takes the Python float as an `F64` pattern
  (`FloatV.ofPat` of its `float64` bits). A Python `int` in an integer dtype is `IntV.ofInt c`.
- **`tl.full(shape, c, dtype)`** is `tl.full` of that constant; `tl.zeros` is the zero bit pattern.

## Integer arithmetic

`TVA.Base.Int` gives each integer operation its LLVM meaning; `TVA.Triton.Arith` lifts the ones
that can be undefined to tensors.

- `+`, `-`, `*` and unary `-` wrap modulo `2^w` (pure: `x + y`, `x * y` on tensors). Comparisons
  are `Tensor.lt`, `ltS` (against a scalar), `le`, `eq`; `x > y` is `Tensor.lt y x` and `x != y` is
  `(Tensor.eq x y).not`. They compare as signed or unsigned by the dtype.
- `//` and `%` truncate toward zero (not Python's floor division): `tl.idiv`, `tl.irem`. A zero
  divisor, or `min // -1`, is undefined behaviour, in any lane, even one a later `tl.where`
  discards.
- `<<` and `>>` are undefined when the amount, read unsigned, is at least the width: `tl.shl`,
  `tl.ashr`, `tl.lshr`. `>>` is arithmetic when the tensor operand Python dispatches on (the left
  one, or the right one when the left is a Python number) is signed before promotion, logical
  otherwise.
- `tl.maximum`, `tl.minimum` and Python's `max`, `min` on integer tensors are `tl.imaximum`,
  `tl.iminimum`.
- `tl.cdiv(x, d)` is not ceiling division: it is Triton's body `(x + (d - 1)) // d`. On two
  compile-time values it is Python's (`tl.py.cdiv`); with a tensor operand it is `tl.cdiv`, whose
  addition wraps and whose `//` truncates.
- Between integer dtypes, `x.to(…)` wraps or extends by the source's signedness (`IntV.cast`,
  `Tensor.castInt`); to `int1` it is `x != 0`.

Index arithmetic is `int32` unless an operand promotes it (an `int64` argument, `.to(tl.int64)`), so
a claim bounds its sizes to keep the offsets from wrapping, or the model wraps exactly as the
kernel does.

## Float arithmetic

`TVA.Triton.Arith` holds the float operations; their exact reference results are those of
`TVA.Base.Float` (IEEE 754 rounding to nearest even, `FloatV.add`, `mul`, `fma`, `divRn`, …).

- **Exact operations, up to NaN payload:** `tl.add`, `tl.sub`, `tl.mul`, `tl.fma`, `tl.div_rn`,
  `tl.sqrt_rn`, `tl.neg`, `tl.abs`. Each returns the IEEE result, any NaN in place of a NaN.
- **Division.** `/` and `tl.fdiv(x, y)` are `tl.div`, the platform's division (`Platform.div`:
  `div.full.f32` on `float32`, not the IEEE quotient; IEEE on `float64`). `tl.fdiv(…,
  ieee_rounding=True)` is `tl.div_rn` (`tl.fdiv` has the flag). Triton divides `float16` and
  `bfloat16` operands in `float32`: the model converts first.
- **Approximate functions:** `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`, `tl.sin`, `tl.cos`,
  `tl.sqrt`, `tl.rsqrt`, `tl.erf` are `tl.math fn`, the platform's `Platform.math`. Triton accepts
  them on `float32` and `float64` only: the model converts a narrower tensor first.
- **`tl.maximum`, `tl.minimum`** on floats (default `propagate_nan`) return a result
  `FloatV.MaxRel` allows: the IEEE `maxNum`/`minNum`, up to NaN payload, either operand when both
  are zeros. Triton converts `bfloat16` operands to `float32` first.
- **`tl.sigmoid(x)`** is Triton's body `1 / (1 + exp(-x))`: `tl.sigmoid` writes it with `tl.neg`,
  `tl.exp`, an IEEE addition of `1` and the default division.
- **Float comparisons** are pure: `Tensor.flt`, `fle`, `feq`, `fne` (ordered: a comparison with a
  NaN is false, except `!=`).
- `%` on floats is not modelled.

**Fusion.** With `enable_fp_fusion` (Triton's default, `Env.fusion`), the backend may contract a
product and a following addition into one `fma`. A *product* is an operand computed by `*`,
`tl.dot`, `tl.rand`, or a float `/` whose quotient is `float32` (`div.full.f32` ends in a multiply
that `ptxas` may fuse), through any dataflow, inlined helpers and library functions included:
`tl.sigmoid` of a `float32` tensor is one. An addition that a product may feed is written in its
fusable form, any other in its IEEE form:

| IEEE form | fusable form |
|---|---|
| `tl.add` | `tl.mul_add a b c` for `a * b + c`; `tl.addc` otherwise |
| `tl.sub` | `tl.subc` |
| `tl.sum`, `tl.sum1`, `tl.sum0` | `tl.sumc`, `tl.sum1c`, `tl.sum0c` |
| `tl.cumsum` | `tl.cumsumc` |

`tl.addc` returns what `AddRel` allows: the IEEE sum, or, with fusion, the rounding of `p + q` for
exact values `p`, `q` that round to the operands. The fusable forms allow every result the IEEE
forms allow, so writing an IEEE addition in fusable form is sound but loose; the reverse is a
wrong model. Contraction is the only float rewrite the backend applies: no reassociation, no
reciprocal division, no flushing of subnormals by arithmetic. `tl.div_rn` and `float64` `/` are
not products.

## Conversions

`TVA.Triton.Convert`. `x.to(d, fp_downcast_rounding=r, bitcast=b)`, `tl.cast(x, d, r, b)` and
`x.cast(d, r, b)` are one operation; `tl.cast` first makes a Python number a tensor of its own
dtype.

- To `x`'s own dtype it is `x` itself.
- Without `bitcast`: `tl.convert d d' x` (each element as `ConvRel` allows, the default rounding),
  which covers every pair of dtypes, or one of its parts when the model wants a narrower operation:
  - between the standard float formats, `tl.cvt x rnd`: one rounding, to nearest even, or toward
    zero with `rnd = .rz` (`"rtz"`, from `float32` to `float16` or `bfloat16` only); widening is
    exact;
  - into fp8, `tl.cvt8 x rnd` (from `float16`, `bfloat16` or `float32`): the platform's
    `Platform.cvt`, which the device facts say saturates;
  - out of fp8, `tl.cvtFrom8` (to `float16`, `bfloat16`, `float32`): exact on finite values;
  - integer to float, `tl.itof`: one rounding to nearest even, or for a 16- or 32-bit target two
    roundings through `float32` or `float64` (`ItofRel`). A chain such as
    `x.to(tl.float32).to(tl.bfloat16)` with the default rounding is one `tl.itof` to the final
    format (MLIR folds it);
  - float to integer, `tl.ftoi`: truncation; a NaN, infinite or out-of-range element is undefined
    behaviour (`FtoiRel`).
- With `bitcast=True`: `tl.bitcast d d' x`, each element's bits read in the target dtype (same
  width required).
- `x.to(tl.pointer_type(d))` of an integer is `tl.intToPtr`.

There is no fp8 ↔ `float64` and no fp8 ↔ fp8 conversion. Every implicit conversion of the source
(promotion, the value of `tl.store` and of an atomic, `other=`) is one of these with the default
rounding: into an fp8 pointer that is `tl.cvt8 x .rn`, the device's saturating conversion, not
the exact constant of `FloatV.litF8`.

## Pointers, loads and stores

`TVA.Memory` and `TVA.Triton.Memory`.

- **Pointer arithmetic** is written as in the source. `p + offs` (scalar or tensor) moves by whole
  elements, each offset taken at its value: sign-extended for a signed dtype, zero-extended for an
  unsigned one (`Ptr.offset`, `Tensor.ptrAdd`). `p - k` is `p + (-k)`, the negation in `k`'s own
  dtype, wrapping: for a signed `k` above its minimum it is `k` elements back
  (`Ptr.sub_int_of_signed`), for an unsigned `k ≠ 0` it is `2^w − k` forward. Addresses are
  mathematical integers.
- **A valid access** is aligned to the element's size, lies in `[0, 2^64)`, and touches only bytes
  the launch was given (`Mem.Aligned`, `Mem.read`, `Mem.Writable`). Anything else is undefined
  behaviour.
- **`tl.load(ptrs, mask, other)`** is `tl.load ptrs (some mask) (some other)`: each enabled lane
  reads its element; a disabled lane accesses nothing and holds `other` when given, an unspecified
  value otherwise (`tl.addrs` gives each lane `some a` or `none`). `other` is converted to the
  pointer's dtype by the model, after a Python number has become a tensor of its own dtype.
  `tl.load1` loads a scalar pointer.
- **`tl.store(ptrs, vals, mask)`** writes each enabled lane; when two lanes have the same address,
  the later one in some order wins. The model converts `vals` to the pointer's dtype. `tl.store1`
  stores to a scalar pointer.
- A tensor load or store is one step: all its enabled lanes at once.

## Block pointers

`TVA.Triton.BlockPtr`. Triton lowers a block pointer to pointer arithmetic and a masked load or
store, and so does TVA; a block pointer adds no semantics of its own.

- `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` casts each entry of
  `shape`, `strides` and `offsets` to `int64` (the model writes the casts); `order` changes no
  value.
- Lane `i` has coordinate `offsets[k] + i_k` along dimension `k` and pointer
  `base + Σₖ coordₖ · strides[k]`, all in wrapping `int64`.
- `tl.load(p, boundary_check, padding_option)` is `tl.load_block p bc pad`: a lane outside
  `[0, shape[k])` along a checked dimension is masked off and holds the padding (`Padding.undef`
  when `padding_option` is not given, `.zero`, `.nan`). An unchecked dimension is not checked: such
  a lane accesses its address.
- `tl.store(p, v, boundary_check)` is `tl.store_block`; `tl.advance` adds to the offsets.
- An operator or a subscript applied to a block pointer has no meaning here.

## Tensor descriptors

`TVA.Triton.Desc`. A host-built `TensorDescriptor` argument (`TensorDesc d`) describes a tensor of
rank 1 to 5 with the host's shape, strides, block shape and padding. Its accesses are TMA copies;
their asynchrony and barriers are the compiler's, which TVA trusts.

- Lane `i` of a block at `offsets` is element `offsets + i`, at `base + Σₖ (offsetsₖ + iₖ) ·
  stridesₖ`.
- `desc.load(offsets)` is `tl.desc_load`: lanes inside the tensor read memory, the others hold the
  padding (`"zero"`, the default, or `"nan"`). `desc.store` is `tl.desc_store`: lanes outside are
  dropped.
- `desc.atomic_add/min/max/and/or/xor` is `tl.desc_atomic`: one *relaxed* `gpu`-scope reduction
  per lane inside the tensor, so two programs reducing into one element race.
- `desc.gather` and `desc.scatter` (`tl.desc_gather`, `tl.desc_scatter`) do not compile for the
  H200: a launch that uses one is rejected on the host.
- Undefined behaviour unless Triton's requirements hold (`TensorDesc.Access`): positive shape
  entries below `2^31`, a 16-byte-aligned base, leading strides that are multiples of 16 bytes and
  a last stride of `1`, an innermost offset that is a multiple of 16 bytes.
- Descriptor accesses are not ordered with the program's other accesses, like pointer accesses. A
  model does not reach the same bytes both through a descriptor and through pointers.
- A descriptor built in the kernel (`tl.make_tensor_descriptor`) is outside every claim: on the
  device each program writes it into a scratch buffer outside the claim's memory. The library's
  `tl.make_tensor_descriptor` is the call's meaning in the language only.

## `tl.dot` and `tl.dot_scaled`

`TVA.Triton.Dot`.

- **`tl.dot(a, b, acc)`** is `tl.dot cfg a b acc`: the platform's `Platform.dot cfg`, since the
  tensor cores' accumulation order and precision are the hardware's. `DotCfg` holds the operand
  dtypes, the accumulator dtype (`out_dtype`), `input_precision` (`tf32` when omitted, as Triton's
  default knobs give it) and the resolved `max_num_imprecise_acc` (`2^30` for an fp8 × fp8 dot when
  omitted, `0` otherwise). The device facts bound the result.
- **The integer `tl.dot`** is `int8 × int8` into an `int32` result (`K ≥ 32`); with a `uint32`
  accumulator the model bit-casts it to `int32` (`tl.bitcast`).
- **`tl.dot(a, b) + c`**, where the dot's accumulator folds to a zero and the product has no other
  use, is `tl.dot_add cfg a b c`: Triton's combine pass may fold `c` into the dot, so the result is
  either `tl.dot(a, b, c)` or the product followed by a fusable addition. This holds whatever
  `enable_fp_fusion` says. `acc += tl.dot(a, b)` in a GEMM loop is this.
- **`tl.sum(x[:, :, None] * y[None, :, :], axis=1)`**, however spelled (`tl.expand_dims`,
  `tl.broadcast_to`, `tl.reshape`, a `tl.reduce` whose combine adds), is `tl.sum_bcast_mul d x y`:
  the products summed along a tree of fusable additions, or Triton's rewrite to an IEEE-precision
  `tl.dot`.
- **`tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc)`** is
  `tl.dot_scaled cfg …`, the platform's `Platform.dotScaled` on the operands' element bits and
  their scales. `ScaledCfg` holds the formats, the scale format (`e8m0` for `uint8`/`int8` scale
  tensors, a float dtype for NVFP4's), the group size (`16` or `32` along `K`), the `k_pack` flags
  and `fast_math`. The operands are written as stored (`e2m1` packed two per `uint8`, low nibble
  first); TVA unpacks them (`ScaledCfg.lhsBits`, `rhsBits`). `tl.dot_scaled_add` is the
  `+ c` form; `scaledValue` gives the exact values the device facts speak about.

## Reductions

`TVA.Triton.Reduce`, with the trees of `TVA.Tensor.Tree`.

- **Float sums** combine along a binary tree Triton chooses (`Tree`, `Tree.Covers`, `Tree.Eval`),
  of depth at most `tl.sumBound P n`; the device facts say that depth is `⌈log₂ n⌉` on the H200.
  `tl.sum` (`tl.sum1` for `axis=1`, `tl.sum0` for `axis=0`) has IEEE additions; `tl.sumc`
  (`sum1c`, `sum0c`) is for a sum of products and has fusable additions.
- **`tl.max`, `tl.min`** are `tl.maxAll isMax`, `tl.max1`, `tl.max0`: `FloatV.MaxRel` along some
  covering tree.
- **Integer reductions** do not depend on the order: they are folds of `TVA.Tensor`
  (`Tensor.foldAll`, `foldRows`, `foldCols`).
- **Index reductions** (`tl.argmax`, `tl.argmin`, `tl.max(…, return_indices=True)`) are
  `tl.argmaxAll`, `tl.argmax1`, `tl.argmax0` (`isMax`, `tieLeft`): pairs `(x[j], j)` combined along
  a tree by `tl.argCombine`, Triton's combine function. Without NaN and with `tie_break_left`, the
  result is the first index of the maximum. With a NaN in the slice the library promises only that
  the value is an element and the index is in range (`tl.ArgRes`).
- **`tl.histogram(x, bins, mask)`** counts, per bin `b`, the enabled elements equal to `b`; an
  enabled element outside `[0, bins)` is undefined behaviour.
- **Promotions the model writes:** `tl.max`/`tl.min` (without indices) widen floats narrower than
  32 bits to `float32`, and integers narrower than 32 bits to `int32`; `tl.sum` widens integers
  narrower than 32 bits to `int32` (`uint32` if unsigned) and no float; the index reductions widen
  only `bfloat16`, to `float32`. With `keep_dims=True` the reduced axis stays, of size 1 (the model
  reshapes).

## Scans and sorting

`TVA.Triton.Scan`.

- **`tl.associative_scan(x, axis, combine_fn, reverse)`** is `tl.associative_scan x axis op
  reverse`: output position `j` is some binary tree over the leaves `0, …, j` in order
  (`ScanLeaves`); with `reverse=True` the leaves are `n-1, …, j`, so the left operand covers the
  higher indices. The combine function is a relation, the model's reading of `combine_fn`:
  `fun a b c => c = a + b` for an integer addition, `AddRel` for a float one, `FloatV.MaxRel` for
  `tl.maximum`.
- `tl.cumsum` (IEEE additions), `tl.cumsumc` (fusable), `tl.cumprod`, `tl.icumsum`, `tl.icumprod`
  (wrapping integers). `tl.cumsum` converts `bfloat16` to `float32` and integers narrower than 32
  bits to 32 bits; with `dtype=d` it converts `bfloat16` to `float32`, then the input to `d`, and
  scans in `d`. `tl.cumprod` converts `bfloat16` to `float32`.
- **`tl.sort`** along the last dimension: each row without NaN comes out as a sorted permutation
  of itself up to `==` (`-0` and `+0` may be copied); a row with a NaN may come out as anything.

## Atomics

`TVA.Triton.Memory`, with the step semantics of `TVA.Launch.Step`.

- A tensor atomic is one atomic read-modify-write per enabled lane (`Prog.rmw`), in an order the
  program does not control; each lane returns its element's old value, a disabled lane an
  unspecified one. The defaults are `sem="acq_rel"`, `scope="gpu"` (`MemSem`, `Scope`).
- An atomic may be performed later than written: a pending atomic may be passed by later steps of
  its program that do not depend on its result (`Step.defer`).
- Integer `atomic_add` wraps; a float `atomic_add` is the platform's `Platform.atomicAdd`.
  `atomic_max`/`atomic_min` compare integers by signedness; on floats the new value is one of the
  two values, not below (above) either when neither is NaN (`fBound`). `atomic_xchg` writes the
  operand; `atomic_cas` compares bit patterns. `atomic_and/or/xor` have a meaning on integers only:
  on floats the model's atomic is undefined behaviour (`bitUpd`).
- Element types: Triton accepts `atomic_add` on 32- and 64-bit integers, `float16`, `bfloat16`,
  `float32`, `float64`; the other read-modify-writes on 32- and 64-bit integers and `float32`,
  `float64`; `atomic_cas` on every 16-, 32- and 64-bit element. It rejects the others on the host.
- The operand of every atomic but `atomic_cas` is converted to the cell's dtype by the model, as a
  stored value is; `atomic_cas`'s operands must already have the cell's dtype.
- **An atomic whose result is unused** (an expression statement, or a value nothing that runs
  reads) is written `tl.atomic_add_unused` (and `atomic_xchg_unused`, …): the same atomic with
  order `MemSem.unused sem scope`, which is `acq_rel` at `gpu` or `sys` scope whatever the source's
  `sem`. Two such atomics on the same bytes from different programs do not race. `MemSem.unused`'s
  docstring says why this is sound.
- **An atomic's result used on a shared cell** is outside every claim: a launch in which a program
  uses the old value an atomic returns, when the atomic's element is also reached by another
  program or by another access of the same program, is not well behaved
  (`UsesSharedAtomicResult`). The semantics performs a tensor step that uses one lane's result as
  one step, while on the device each thread orders only its own accesses.

## Random numbers

`TVA.Triton.Random`. Triton's generator is Philox 4x32 with 10 rounds, written in Triton itself;
TVA defines it exactly (`Philox.impl`, `Philox.randint4x`). `tl.randint(seed, offset)` and
`tl.randint4x` are exact functions. `tl.rand(seed, offset)` is `tl.uint_to_uniform_float` of
`tl.randint`: the bits read as `int32`, `~x` when negative, converted to `float32` (`ItofRel`),
times the `float32` constant `Philox.scale32`. Its result is a product, so an addition that
consumes it (`tl.rand(…) - 0.5`) is `tl.subc`.

## Loops

`TVA.Triton.Loop`. A loop's carried variables (the names defined before the loop that its body
assigns) are its state, a tuple; after the loop they hold what the last iteration left.

- `for i in range(lo, hi, step)` and `tl.range(…)` are `tl.forRange lo hi step init body`. The
  induction variable has the promoted type of the bounds (`int32`, or `int64` when a bound is).
  Bounds outside `tl.RangeOk` (a positive step, and `lo`, `step`, `hi + step − 1` within the
  signed range) are undefined behaviour here; the trip count is `tl.tripCount`.
- `tl.static_range` is unrolled: the model writes the copies, or `tl.forN` over a Lean count.
- `while c:` is `tl.whileLoop init cond body`, built on `Prog.loop`.
- A name first defined in the body, and a `for` loop's variable, have no defined value after the
  loop; a model does not use them there.
- The scheduling keywords of `tl.range` (`num_stages`, `loop_unroll_factor`, `warp_specialize`,
  `flatten`, …) change no outcome of a well-behaved model: pipelining can make an iteration miss
  what an earlier one stored, but that is a conflict within the program, which a well-behaved
  launch never has.

## Compiler hints and assertions

`TVA.Triton.Core`'s section on assertions and hints.

- `tl.assume(cond)` is a check: a false `assume` is undefined behaviour (the compiler may have
  optimised on it).
- `tl.static_assert(cond)` is checked at compile time; a false one rejects the kernel on the host,
  and the model makes it undefined behaviour, so no claim covering such an instance can be proved.
- `tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy` (and `tl.max_contiguous_multiple_of`
  for the pair, `tl.multiple_of1` for a scalar, the `…N` forms for rank 2 and more) are checks,
  not no-ops: a false hint makes the compiled code read other elements, or fault. A hint marks the
  IR operation that defined its argument's value, so the model checks it where the value is
  defined, before any use. A hint on a parameter, an induction variable or the value an inlined
  call returns marks nothing and is dropped; a hint inside a loop on a name the loop carries checks
  the value before the loop, once.
- What each promises, along each dimension: `max_contiguous(x, C)` alone, that `C` is `x`'s
  contiguity (`ContigIs`); `max_constancy(x, C)`, that groups of `C` are equal (`ConstAlong`);
  `multiple_of(x, v)` alone, that `v` divides the first element of each group of `C`, `C` the
  contiguity the compiler infers (`FirstsDiv`; the model takes a lower bound of it, stated in the
  docstring); both on one value, `ContigAlong` and `FirstsDiv`. A pointer's divisibility is in
  bytes, its contiguity in elements (`HintVal`).

## Undefined behaviour

A program reaches undefined behaviour (`Prog.ub`, or a `pick` that nothing satisfies) when:

- an enabled lane of a load, a store or an atomic accesses a misaligned address, an address
  outside `[0, 2^64)`, or a byte the launch was not given;
- an integer `//` or `%` has a zero divisor or overflows, or a shift amount is at least the width,
  in any lane;
- a float is converted to an integer out of range, or from a NaN or an infinity;
- `tl.assume` or a compiler hint is false; a `tl.static_assert` is false;
- `tl.forRange`'s bounds are outside `tl.RangeOk`;
- a block pointer or a descriptor access breaks Triton's requirements (`BlockPtr.Valid`,
  `TensorDesc.Access`);
- `tl.histogram` gets an enabled element outside its bins;
- a conversion or atomic the library gives no meaning (`atomic_and` on floats, a conversion pair
  Triton lacks) is used;
- an operation of the platform has no result (the device facts say every one has).

On the device undefined behaviour may fault, return garbage or corrupt memory, so a claim covers
no launch that reaches it: a proof shows it is unreachable.

## Launches

`TVA.Launch` explains this in its module docstring; the files are `Launch/Step`, `Launch/Run`,
`Launch/WellBehaved` and `Launch/Progress`.

**A launch** (`Launch`) is the grid `(x, y, z)`, the options that change the meaning
(`fusion` for `enable_fp_fusion`, and `debug`, `numCtas`, `pdl` for `launch_pdl`) and the
source's block shapes (`blocks`). `Launch.grid1 n blocks` is a 1-D grid with the other options at
Triton's defaults. Program number `i` gets `tl.program_id = (i % x, i / x % y, i / (x·y))` and
`tl.num_programs` the grid (`Launch.ctx`).

**`Launch.Valid`** is when the semantics describes the launch at all: `x < 2^31`, `y, z < 2^16`,
fewer than `2^31` programs, `num_ctas = 1`, no programmatic dependent launch, no `debug`, and every
listed block shape a power of two in each dimension with at most `2^20` elements
(`Shape.TritonOk`). A launch Triton or the driver rejects before running (too much shared memory, a
block too small for `tl.dot`, a grid beyond CUDA's limits) has no outcome, and a claim says nothing
about it.

**Memory.** The launch starts from `init : Mem`, a partial map from addresses to bytes: the bytes
present are the ones the launch is given (the claim's buffers), and accessing any other byte is
undefined behaviour. So the final memory has the same bytes present, and a claim's `Post` speaks
only of them. `Buf.Valid` says a buffer's elements are present and aligned.

**Steps** (`Step`, in `Launch/Step`). One step of one program performs its next effect as one
indivisible action and records an `Event` (the bytes read and written, and for an atomic its
`sem`, `scope` and element). A load reads all its enabled lanes at once; a store writes them at once;
an atomic is one step per element; a `pick` chooses; a `loop` unfolds once. A pending atomic may
be passed by a later step that does not depend on its result (`Step.defer`).

**Interleaving** (`Launch/Run`). A launch step (`LStep`) is a step of any one program on the
shared memory; a trace (`Reach`) is any sequence of them. Every interleaving is allowed and each
step sees what all earlier steps left: sequential consistency between programs. Happens-before
(`HB`) is each program's own order; programs do not synchronise.

**Well behaved** (`WellBehaved`, in `Launch/WellBehaved`). The interleaving is the launch's meaning
only when:

1. no reachable state has a program whose next step is undefined behaviour;
2. no trace has a race between programs (`InterRace`): two steps of different programs touching a
   common byte, at least one writing it, unless both are `acq_rel` atomics at `gpu` or `sys` scope
   on exactly the same bytes (`Event.Strong`);
3. no trace has a conflict within a program (`IntraRace`): two steps of one program touching a
   common byte, at least one writing it, unless both are atomics on exactly the same bytes
   (`Event.SameAtomic`); loop iterations count;
4. no program uses an atomic's result on a cell another access also reaches
   (`UsesSharedAtomicResult`).

Why each is there: undefined behaviour lets the device do anything; a race between programs can be
observed piece by piece on the device, which an interleaving of whole steps does not show; a
program's threads do not order its own accesses, so a conflict within a program can have results
the text order does not give; and an atomic's result used on a shared cell can reveal an order of
the device's threads that the semantics lacks. A relaxed, acquire or release atomic whose result is
used races with another program's atomic on the same bytes, and handing data from one program to
another behind a flag is a race.

**Outcomes** (`Outcomes`, `Outcome`): every program finishes (`ok fin`), some interleaving reaches
undefined behaviour (`ub`), or some interleaving runs forever (`hang`). A program never gets stuck
in any other way (`Step.progress`), so a well-behaved launch always has an outcome
(`WellBehaved.exists_outcome`).

**What a claim states** is `Correct E k L init Q`: the launch is valid, well behaved, and every
outcome is `ok fin` with `Q fin`. `Claim.Holds` asks it of every covered instance, every covered
initial memory, and every platform satisfying the device facts.

## The device facts

Some operations have no exact meaning: the compiler lowers them to instructions whose results are
documented only up to an accuracy. TVA leaves each to a platform `P` (`TVA.Platform`), a relation
per operation: `Platform.math` (`tl.exp`, …), `Platform.div` (the default `/`), `Platform.dot`,
`Platform.cvt` (into fp8), `Platform.atomicAdd` (float), `Platform.dotScaled`, and
`Platform.sumDepth` (how deep a float reduction's tree may be). A program reads the platform
through `Env`, with the launch's fusion flag. A relation rather than a function lets two
occurrences of one operation on the same operands give different results, as the device may.

Every claim quantifies over every `P` with `Device P` (`TVA.Device`): the conjunction of all the
facts TVA states about an H200 running Triton 3.8.0's code. Each fact is one shape of
`TVA.Device.Shapes` (totality, accuracy on a domain, an exact value) at one operation and format,
with its constants written in (`TVA.Device.Sm90`, `TVA.Device.Dot8Chunk`, `TVA.Device.Dot16Chunk`).
Each docstring names the instructions Triton emits, the bound and the folder of `evidence/` where
TVA measured it; each constant keeps slack over what was observed.

| Operation | Fact | Bound |
|---|---|---|
| every approximate operation | `MathTotal`, `DivTotal`, `CvtTotal`, `DotTotal`, `AtomicAddTotal`, `DotScaledTotal` | has a result |
| `tl.exp` (`float32`) | `Device.ExpF32` | `(2^-22 + 2^-23·X)·eˣ + 2^-140` on `[-X, X]` |
| `tl.exp(-∞)`, `tl.math.exp2(-∞)` | `Device.ExpF32NegInf`, `Device.Exp2F32NegInf` | `+0` |
| `tl.exp` past `88.7228…` | `Device.ExpF32Overflow` | `+∞` |
| `tl.math.exp2`, `tl.log`, `tl.math.log2`, `tl.sqrt`, `tl.math.rsqrt` (`float32`) | `Device.Exp2F32`, `LogF32`, `Log2F32`, `SqrtF32`, `RsqrtF32` | relative `2^-22`, small absolute terms |
| `/` on `float32` | `Device.DivFull` | `2^-22·|x/y| + 2^-148` for `|x/y| ≤ 2^126` |
| `x / ±∞` on `float32` | `Device.DivF32InfDivisor` | the IEEE signed zero |
| `/` on `float64` | `Device.DivF64` | IEEE |
| conversions into fp8 | `Device.CvtSatF32E4m3`, … (six) | saturating round to nearest even |
| `tl.dot` of `float16`, `bfloat16` | `Device.DotF16`, `DotBF16` | `relDot16 K` relative to lifted magnitudes, `absDot16 K` |
| the same, chunk by chunk | `Device.DotF16Chunk`, `DotBF16Chunk` | per chunk of 64 `k` |
| `tl.dot` of fp8 | `Device.DotE4m3ChunkLoss`, … (four) | per chunk of 32 `k` |
| `tl.dot` of `float32`, `ieee` | `Device.DotF32Ieee` | `(K+1)·2^-23` relative |
| `tl.dot` of `int8` | `Device.DotI8` | exact while `|acc| + Σ|a·b| < 2^31` |
| `tl.dot_scaled` (MXFP8, MXFP8 × MXFP4) | `Device.DotScaledE4m3`, `DotScaledE4m3E2m1` | `relDot16`, `absDot16` |
| float `tl.atomic_add` (`float32`) | `Device.AtomicAddF32` | `2^-23·|o + v| + 2^-124` |
| depth of a float reduction | `Device.SumDepth` | `⌈log₂ n⌉` |

Facts implied by others are theorems: the round-to-nearest conversions in range
(`Device.cvtF32E4m3`, …) and the first arm of the fp8 chunk facts (`Device.dotE4m3Chunk`, …).
`TVA.Device.Weaken` derives weaker forms (larger constants, smaller domains).

**Why the facts cannot make a claim vacuous.** If the facts contradicted each other, every claim
would hold whatever its kernel does. `Device.ref_ok` proves that one platform, `Platform.ref`
(IEEE arithmetic rounded once, `TVA.Device.Ref.Platform`), satisfies them all. So a proof can draw
nothing from `Device P` that a platform as accurate as `Platform.ref` would not give.

**Tolerances in specifications.** Some claims bound an error by a formula that follows how the
kernel accumulates; those formulas are in `TVA.Device.Tolerance` (`DotLoop.bound`,
`DotLoop.stepRel`, `DotLoop.stepAbs`, `Dot8Chunk.psum`, `Dot8Chunk.errB`, `Dot8Chunk.errM`), next to
the facts' constants, because a claim's reader reads them as part of its specification.

## Exclusions

What TVA does not cover is listed in [EXCLUSIONS.md](EXCLUSIONS.md): known miscompiles (a float
reduction in a loop that Triton's `OptimizeThreadLocality` pass rewrites), features the library
gives no meaning (`tl.inline_asm_elementwise`, float `%`, aggregates, descriptors built in the
kernel, Python outside the supported subset, …), concurrency the semantics gives no meaning
(conflicts within a program, races, an atomic's result used on a shared cell), and launch and host
situations assumed not to happen (knobs, hooks, the autotuner, other writers, `debug=True`,
programmatic dependent launch). A model uses no excluded feature, and a claim covers no excluded
launch; the trusted compiler statement below holds only together with that list. Some exclusions
are enforced by the model itself (`Launch.Valid`, `WellBehaved`, undefined behaviour in a `tl.*`
definition); EXCLUSIONS.md says which.

## What the compiler is trusted for

`TVA.Claim` lists it in full. In short: Triton 3.8.0 compiles the kernel so that every execution
of a valid, well-behaved launch on the H200 ends in one of the outcomes of the model's launch, and
touches no global memory beyond the bytes the launch is given. This covers a launch of the
kernel's own `@triton.jit` function, directly or through `warmup`, with any performance options;
Triton's knobs at their defaults, no autotuner or hook, the kernel compiled afresh from its source;
an x86-64 host running CPython 3.13 without `-O`, in the default floating-point environment; and
nothing else writing the launch's bytes until it ends.
