# Which operands keep an expression free of effects

Library evidence (not a kernel's), for library issue L14. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27, compiled without `debug`
(`TRITON_DEBUG` unset). Script: [run.py](run.py); output: [out.txt](out.txt).

**Since round 6** (library issue L36), TRANSCRIPTION.md is guidance and states no rule; the facts this
evidence establishes are stated in [SEMANTICS.md](../../docs/SEMANTICS.md), "How the front end evaluates the source" (operators on
aggregates, expressions with no effect).

**Question.** The drops `compile-check`, `no-op:print` and `dead` need a *pure* expression
([TRANSCRIPTION.md](../../docs/TRANSCRIPTION.md) §10), one that has no effect and is never
undefined. Round 2 listed the operators (`+`, `==`, `<`, `is not`, subscripts, …) as pure. Is that
true for every operand type?

**What the front end does** (`primitives/triton-3.8.0-code-generator.py`):
- A binary operator or a comparison on a tensor calls the tensor's builtin method. On a value that
  is neither a tensor nor a `tl.constexpr` (nor a tuple with a `constexpr` right operand), it looks
  the operator's method up on the value and calls it (`_apply_binary_method`, `:795-808`;
  `visit_BinOp`, `:810-817`; `visit_Compare`, `:1076-1091`, where only `is` and `is not` are
  answered without a call).
- `get_Attribute` (`:1535-1544`) wraps a `@triton.jit` method of a Triton value (an aggregate) as
  a bound method, and calling it inlines the method's body.
- A subscript of a Triton value calls its `__getitem__` (`visit_Subscript_Load`, `:1187-1193`).
- A unary operator on a non-tensor calls the method directly from Python (`visit_UnaryOp`,
  `:1097-1111`).
- `tl.tuple`'s `__eq__` and `__add__` are Python methods: list equality and concatenation
  (`primitives/triton-3.8.0-core.py:1311-1322`).

**Result** (part A: a grid of 4 programs; each `@triton.jit` operator method of the aggregate adds 1
to a counter).

| Case | Result |
|---|---|
| A0 control: the aggregate is made, nothing else | counter 0 |
| A1 `tl.static_assert((a + a) is not None)` (`__add__`) | compiles; counter 4: the method runs |
| A2 `tl.device_assert(a == a, ...)` (`__eq__`) | compiles; counter 4 |
| A3 `assert a < a, ...` (`__lt__`) | compiles; counter 4 |
| A4 `unused = a + a` (a `dead` value) | compiles; counter 4 |
| A5 `tl.static_assert(a[0] is not None)` (`__getitem__`) | compiles; counter 4 |
| A6 `tl.static_assert((-a) is not None)` (`__neg__`) | rejected on the host: the jitted method is called from Python |
| A7 `tl.static_assert((a,) == (a,))` | compiles; counter 0 (a Python list comparison) |
| A8 `tl.static_assert(((a,) + (a,)) is not None)` | compiles; counter 0 (a concatenation) |
| B1 every operator the purity rule lists, on tensors of `int32`, `float32`, pointers and booleans, on `tl.constexpr` values, dtypes and tuples of these, inside `tl.static_assert`, `tl.device_assert`, `assert`, `tl.static_print` and as unused values | PTX identical to the kernel without these lines; TTIR identical up to value names and the order of its lines; no atomic, print, assert or call; outputs correct |

A comparison of two pointer tensors (`p == p`) does not compile in 3.8.0, so part B uses only `+`
on pointers.

**Conclusions.**
1. An operator, comparison or subscript whose operand is an aggregate runs the aggregate's
   `@triton.jit` method, with its effects (A1–A5), even inside a call that is dropped or as a value
   nobody uses. So purity must restrict the operand types: an expression with an aggregate operand
   is not pure.
2. On the allowed types (tensors, `tl.constexpr` and plain Python values, dtypes, and tuples of
   these), the listed operators leave nothing in the compiled kernel (B1).
3. A tuple's own `==` and `+` call nothing (A7, A8). A tuple that holds an aggregate is still not
   an allowed operand, since an operator applied to its elements would dispatch.
