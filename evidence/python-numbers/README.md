# Python numbers assigned to names, and Python's `min` and `max` of compile-time values

**Question.** `docs/SEMANTICS.md` gives Python numbers ("Constants", "Operators and promotion") and
compile-time values ("Order", the compile-time `if`) their own rules. Does a name that a statement
assigns a Python number still hold one? And what do Python's `min` and `max` do with NaN and `-0.0`
when all their arguments are compile-time values? Library issues L59 and L61 (round 11; raised by
lr-8b, `docs/library-review/round-8/lr-8b/work/t_assign_literal.py`, and lr-8a).

**Sources** (Triton 3.8.0).
- `primitives/triton-3.8.0-code-generator.py` lines 727-760 (`visit_Assign`, `_sanitize_value`): the
  value assigned to a name is unwrapped from `tl.constexpr`, and a value that is not a Triton value,
  `None`, a dtype or a tuple becomes `semantic.to_tensor(value)`, unless the target is annotated
  `: tl.constexpr` (`normalize_value`); a tuple is converted element by element. Augmented
  assignments and `:=` go through `visit_Assign` (lines 762-778).
- `primitives/triton-3.8.0-semantic.py` lines 118-158 (`to_tensor`, `to_tensor_type`) and 601-613
  (`scalar_constant`): an `int`, `float` or `bool` becomes a scalar constant of its own type
  (`int32`, …; `float32` or `float64`; `int1`), `0` as the null value; anything else raises.
- `code_generator.py` lines 95-100, 1358-1365: the arguments of a call of a `@triton.jit` helper are
  wrapped with `normalize_value`, so a Python number passed directly stays a `tl.constexpr`.
- `primitives/triton-3.8.0-core.py` lines 3882-3888, 3906-3912 (`builtin_max`, `builtin_min`): with
  only compile-time arguments, an assertion refuses NaN and negative zero.

**Method.** `run.py` compiles and runs each case on GPU 0 in its own process, and reports what it
stores, or the front end's error. "in place" is the same expression with the number written where
it is used. Cases A1-A14 (L59) and B1-B4 (L61) are listed in the script's docstring.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/python-numbers/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | source | result |
|---|---|---|
| A1 | `e = 1e-7; x * e`, `x` `bfloat16` `[1, 2, 3, 1000]` | `[1.0000000116860974e-07, …, 0.00010000000474974513]`: `float32(1e-7)·x` in `float32` |
| A1p | `x * 1e-7` in place | `[0, 0, 0, 0]`: the `bfloat16` constant of `1e-7` is `0` |
| A2, A2p | `e = 1e-300; x * e` / `x * 1e-300`, `x` `float32` | the product is `float64` / `float32` (`tl.static_assert` on its dtype held) |
| A3, A3p | `a = -7; b = 2; a // b` / `-7 // 2` | `-3` (tensor `//`, truncating) / `-4` (Python's) |
| A4, A5, A6 | `lo, hi = -7, 2`; `k = 0; k -= 7`; `m = j - 7` in `tl.static_range` | `-3` each |
| A7 | `c: tl.constexpr = -7; c // 2` | `-4` |
| A8, A8a | `half(-7)` / `a = -7; half(a)`, `half(n) = n // 2` | `-4` / `-3` |
| A9, A9p | `F = False; if F and tl.atomic_add(c, 1) > 0:` / `False and …` in place, 8 programs | 8 atomics / 0 |
| A10, A10p | `F = False; if F: tl.static_assert(False)` / `if False: …` in place | rejected on the host / compiled and ran |
| A11, A11c | `B = BLOCK; tl.arange(0, B)` / `B: tl.constexpr = BLOCK` | rejected on the host ("arange's arguments must be of type tl.constexpr") / ran |
| A12 | `z = -0.0`, stored | `0x00000000` (`+0`) |
| A13 | `n = None; d = tl.float16; if n is None: tl.full([1], 1, d)` | ran: `None` and dtypes are kept |
| A14 | `s = "abc"` | rejected on the host (`cannot convert abc … to tensor`) |
| B1, B2 | `max(1.0, F)` with the constexpr `F = nan`; `min(1.0, -0.0)` | rejected on the host ("constexpr max does not support NaN values", "… negative zero") |
| B3 | `max(1.0, 2.0)` | ran: `2.0` |
| B4 | `a = 3; min(a, 5)` | ran: an `int32` tensor (`tl.static_assert` held), `3` |

**Conclusions**, used by `docs/SEMANTICS.md` ("How the front end evaluates the source",
"Constants", "Operators and promotion"), the `Operand` docstring of `TVA.Lang.Promote`, and
`docs/TRANSCRIPTION.md`:

1. A name that a statement assigns a Python number holds a scalar tensor of the number's own type,
   with the constant Triton builds for it there (A1-A6, A12): every rule about Python numbers and
   compile-time values treats it as a tensor. It promotes as a tensor (A1, A2), divides as one
   (A3-A6), is no compile-time value in `and` or `if` (A9, A10), and is no `tl.constexpr` where one
   is required (A11).
2. A target annotated `: tl.constexpr` keeps the Python value (A7, A11c), and a number passed
   directly to a helper stays one there (A8). `None` and dtypes are kept (A13); another Python
   value is rejected on the host (A14).
3. Python's `min` and `max` of compile-time values refuse NaN and `-0.0`: the launch is rejected on
   the host (B1, B2). With a tensor argument, a name assigned a number included, they are
   `tl.minimum`/`tl.maximum` (B4).
