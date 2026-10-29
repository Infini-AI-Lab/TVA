# A runtime conditional expression and a helper's `return`s: typed by the front end

Library evidence (not a kernel's), for library issue L102. Round 20.

**Question.** SEMANTICS said a conditional expression with a runtime condition "means the runtime
`if`" and a helper call "means its body", which reads as keeping a Python number a Python number.
lr-17a and lr-17b found that the front end types both: the sides of `a if c else b`, and the values
of a helper's `return`s, become tensors of one type. What exactly do these forms give, and which
helper forms keep the value of the `return` a program takes?

**Sources** (pinned in `primitives/`). `triton-3.8.0-code-generator.py:998-1036` (`visit_IfExp`):
with a runtime condition, each side is evaluated in its own branch and `to_tensor`ed, and both must
then have one type. Lines 556-626 (`decide_return_type`, `cast_to`, `handle_returns`): a helper's
call has one type, from every `return` the front end visits; `return`s of compile-time values that
`==` equates give the first one's value; otherwise each value becomes a tensor, and all must have
one type.

**Method** ([run.py](run.py), output [out.txt](out.txt)). `c` is an `int` argument, `5` or `-5`
(never `1`, which Triton passes as a compile-time constant, SEMANTICS "Scalar arguments"), so every
condition is a runtime one. The kernels are in the script's docstring:
- E1a `(-1 if c > 0 else -3) // 2`; E1b `x * (0.1 if c > 0 else 0.2)` on every finite `bfloat16` `x`
  (65,280), against the `float32` product rounded to `bfloat16` and against the `bfloat16` product;
  E1c `1 if c > 0 else 1.0`, and `x if c > 0 else 1.0` with an `int32` `x`; E1d `x + 1 if c > 0 else
  x - 1` with an `int32` `x`;
- E2a a helper `if c > 0: return -7` then `return -9`, `h(c) // 2`; E2b `return 1` then `return
  1.0`, `x * h(c)` with `x = 2^24 + 1`; E2c `if c > 0: return 5` and nothing after; E2d `return 5` on
  both paths; E2e `return x + 1` / `return x - 1`.

**Triton version.** 3.8.0 (torch 2.11.0+cu130, Python 3.13.9).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`out.txt`).

| case | `c = 5` | `c = -5` | Python's value |
|---|---|---|---|
| E1a `(-1 if c > 0 else -3) // 2` | `0` | `-1` | `-1`, `-2` |
| E1b `x * (0.1 if c > 0 else 0.2)`, `bfloat16` `x` | 0 of 65,280 differ from the `float32` product (13,076 from the `bfloat16` one) | 0 differ (13,102 from the `bfloat16` one) | |
| E1c `1 if c > 0 else 1.0`; `x if c > 0 else 1.0` | rejected on the host (`CompilationError`), both | | |
| E1d `x + 1 if c > 0 else x - 1`, `x = 7` | `8` | `6` | `8`, `6` |
| E2a `return -7` / `return -9`, `h(c) // 2` | `-3` | `-4` | `-4`, `-5` |
| E2b `return 1` / `return 1.0`, `x * h(c)` | `16777217` | `16777217` | `16777217`, `16777216` |
| E2c `return 5` in the `if` only | `5` | `5` | `5`, `None` |
| E2d `return 5` on both paths (admitted) | `5` | `5` | `5`, `5` |
| E2e `return x + 1` / `return x - 1` (admitted) | `8` | `6` | `8`, `6` |

**Conclusions.**
1. A conditional expression with a runtime condition is the runtime `if` whose sides are tensors:
   a Python number becomes a scalar tensor of its own type, and both sides must have one type, or
   the host rejects the kernel. SEMANTICS ("How the front end evaluates the source", "Order")
   states this, and TRANSCRIPTION has it as a pitfall.
2. A helper's call has the value of the `return` the program takes only when the visited
   `return`s are one, or all tensors of one type (E2e), or all one compile-time value of one type
   (E2d), with no path falling off the end when a value is returned. SEMANTICS states the rule
   ("A helper's value", under "`return`") and the subset admits only these forms; the others
   (E2a–E2c) are outside it ([exclusions.md](../../docs/exclusions.md); library issue L102).
