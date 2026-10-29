# Calls of Python's builtins in a kernel

**Question.** Tutorials 02 and 06 call Python's builtins in their kernels (`-float('inf')`,
`tl.zeros(…) - float("inf")`, `isinstance(desc_or_ptr, tl.tensor_descriptor)`). Which builtins can a
kernel call, what does a call give, and does it emit any operation? Library issue L75 (round 11;
raised by the round-10 coverage audit, `docs/library-review/round-10/coverage/work/probes.py` P2).

**Sources** (`primitives/triton-3.8.0-code-generator.py`).
- Lines 357-365: `builtin_namespace` holds Python's `len`, `list`, `range`, `float`, `int`,
  `isinstance`, `getattr` and `hasattr`, and maps `print`, `min` and `max` to
  `tl.device_print`, `builtin_min` and `builtin_max`.
- Lines 381-427 (`_define_name_lookup`): a name is looked up in the function's locals, then its
  module's globals, then `builtin_namespace`; nothing else (no other builtin) is visible, and an
  unknown name raises `NameError`.
- Lines 1444-1448 (`call_Function`): for a function of `builtin_namespace`, the arguments are
  unwrapped from `tl.constexpr` and the Python function is called; the result is wrapped by
  `normalize_value` (lines 95-100), a compile-time value unless it is a Triton value.
- Lines 1638-1659: `int` and `len` run through `static_executor`, which wraps the result as a
  `tl.constexpr`.

**Method.** `run.py` compiles and runs each case on GPU 0 in its own process, and reports what it
stores (with the bits), what the front end emits, or its error. The cases (F1-F10) are listed in the
script's docstring.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/py-builtins/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | source | result |
|---|---|---|
| F1 | `tl.store(out, -float('inf'))`; `tl.load(p, mask=i < 2, other=-float('inf'))` | `-inf` (`0xff800000`); `1.0, 2.0, -inf, -inf` |
| F2 | `tl.zeros([4], tl.float32) - float("inf")` | four `-inf` (`0xff800000`) |
| F3 | `tl.static_assert(float("inf") > 1e308)`, `int(2.7) == 2`, `len((1, 2, 3)) == 3` | compiled and ran |
| F4 | `isinstance(src, tl.tensor_descriptor)`, `isinstance(src, tl.tensor)`, each deciding an `if` | pointer: not a descriptor, a tensor; host `TensorDescriptor`: a descriptor, not a tensor; two of the four stores emitted |
| F5 | `getattr(x, "dtype") == tl.float32`, `hasattr(x, "dtype")` | compiled and ran |
| F6 | `len(x.shape) == 1`, `list(x.shape) == [4]` | compiled and ran |
| F7 | `tl.static_assert(isinstance(tl.atomic_add(c, 1), tl.tensor))`, 8 programs | 8 atomics: the argument ran |
| F8 | `float(tl.load(x_ptr))` | rejected on the host (`float() argument must be … a real number, not 'tensor'`) |
| F9 | `abs(-3)` | rejected on the host (`NameError('abs is not defined')`) |
| F10 | `int(n)`, `n` a runtime `int32` argument | rejected on the host |

**Conclusions**, used by `docs/SEMANTICS.md` ("How the front end evaluates the source": Python's
builtins; "Order"; "Expressions with no effect") and `docs/TRANSCRIPTION.md`:

1. A kernel sees only these builtins: `float`, `int`, `isinstance`, `len`, `list`, `getattr`,
   `hasattr` and `range` are Python's, and `print`, `min` and `max` Triton's; a call of any other
   is rejected on the host (F9).
2. A call of `float`, `int`, `isinstance`, `len`, `list`, `getattr` or `hasattr` is evaluated by
   Python at compile time on its arguments' values and gives a compile-time value; it emits no
   operation (F3-F6), but its arguments are evaluated as any expression's, so an operation in them
   runs (F7). Python's own results apply: `float('inf')` is `+∞` (F1, F2), `isinstance` tells a
   host descriptor from a pointer and decides a compile-time `if` (F4).
3. `float` or `int` of a runtime value is rejected on the host (F8, F10).
