# The supported subset of the front end: the facts round 15 states, and the forms it excludes

**Question.** Library issue L88 (user, 2026-10-28) makes TVA give meaning to a supported subset of
Triton's Python front end, listed in `docs/SEMANTICS.md` ("The supported subset"), and excludes the
rest. Round 15 settles L80, L83, L84, L85, L86 and L87 by that rule. For the forms it states, what
does Triton 3.8.0 do: `tl.where` with a Python number as a value (L83), `name: tl.constexpr = e` and
the assignments after it (L84), a one-argument `range` (L85), a helper's defaults and annotated
parameters (L80), the names a loop body defines, `tl.load`'s `other` (L86) and `tl.cumsum`'s
`dtype=` (L87)? For the forms it excludes, what makes them differ from the rules it states (a bare
annotation, another annotation, the induction variable after a loop)?

**Sources** (Triton 3.8.0, pinned in `primitives/`).
- `triton-3.8.0-semantic.py` lines 1638-1652 (`where`): `x` and `y` go through
  `binary_op_type_checking_impl` (175-210), which makes a Python number a tensor of its own type and
  promotes by `computation_type_impl` (68-116): a Python number of a kind no higher than the tensor's
  takes the tensor's dtype, two Python numbers or a `float` with an integer tensor follow the float
  rules. `TVA.Lang.Promote.whereTyping?` is this.
- `triton-3.8.0-code-generator.py` lines 694-712 (`visit_AnnAssign`): with the annotation
  `tl.constexpr` and a value, a name already in scope is refused ("already defined"), and otherwise
  the value is kept as it is (`normalize_value`, 95-100); a bare annotation is remembered in
  `pending_annotations` and applies to the next assignment of the name (740-750); another annotation
  goes to `visit_Assign` (712), which makes a Python number a tensor (727-760).
- lines 1204-1245 (`visit_For`): every argument of the iterator is visited (1206); for Python's
  `range` with one argument the bound is that argument visited again (1242), for `tl.range` and
  `tl.static_range` the arguments are used once (1208-1236). The induction variable is a placeholder
  (`create_poison`, 1277-1278) whose uses inside the body are replaced by the loop's variable
  (1320-1327); after the loop the name is the placeholder again (the scope is restored on leaving the
  region, 121-140). A name the body defines that was not defined before the loop is not carried
  (`_find_carries`, 465-503), so it is not defined after the loop; the same for `while` (1126-1185).
- lines 1358-1365 (`call_JitFunction`): the call is bound to the helper's Python signature
  (`bind`, `apply_defaults`), so a default is the value the `def` gives, and every argument is wrapped
  by `normalize_value`: a Python number stays a `tl.constexpr`, a tensor stays a tensor. No step reads
  the parameters' annotations (the prototype is built from the arguments' types, 1368-1373, and the
  parameters are bound to the arguments, 657-663).
- lines 714-725 (`assignTarget`): a tuple target takes element `i` of the value for its `i`-th name,
  so elements beyond the names are dropped.
- `triton-3.8.0-semantic.py` lines 1043-1045 (`load`): `other` is cast to the pointer's element type.
- `triton-3.8.0-standard.py` lines 270-281 and 334-343 (`_pick_sum_dtype`, `cumsum`): `bfloat16` is
  widened to `float32` first, then, when `dtype` is given, the input is converted to it.

**Method.** `run.py` compiles and runs each case on GPU 0 in its own process, with its own empty
Triton cache, and reports what it stores, or the front end's error ("rejected on the host"). Cases
W1-W5, C1-C9, R1-R4, D1-D4, P1-P2, L1-L4, Z1-Z2, U1, O1 and S1-S3 are listed in the script's docstring.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/front-end-subset/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | source | result |
|---|---|---|
| W1, W1i | `tl.where(keep, x, 0.0)`, `x` `int32` `[16777217, 3, -5, 7]` | a `float32` result (`tl.static_assert` held): `[16777216.0, 0.0, -5.0, 7.0]`, and `16777216` stored back into `int32` |
| W2 | `tl.where(keep, x, 0)`, `x` `int32` (tutorial 09's form) | an `int32` result: `[16777217, 0, -5, 7]` |
| W3 | `tl.where(m, 0, -1.0e6)` (tutorial 06's form) | a `float32` result: `[0.0, -1000000.0, 0.0, 0.0]` |
| W4, W5 | `tl.where(keep, x, 0.0)`, `x` `bfloat16`, `float16` | a `bfloat16`, a `float16` result |
| C1 | `c: tl.constexpr = -7; c // 2` | `-4` (Python's `//`) |
| C2 | `c = 1`, then `c: tl.constexpr = -7` | rejected on the host: "c is already defined. constexpr cannot be reassigned." |
| C3 | `c: tl.constexpr = -7; c += 0; c // 2` | `-3`: the plain assignment made `c` an `int32` tensor |
| C4 | `BLOCK = BLOCK * 1` (a `tl.constexpr` parameter), `tl.arange(0, BLOCK)` | rejected on the host ("arange's arguments must be of type tl.constexpr") |
| C5, C5c | a bare `e: tl.constexpr`, then `e = 1e-7`; `x * e` with a `bfloat16` `x` / without the bare line | `[0, 0, 0, 0]`: `e` stayed a Python number / `float32(1e-7)·x` in `float32` |
| C6 | `a: tl.int64 = 2**31 - 1; a + 1` | `-2147483648`: the annotation is ignored, `a` is an `int32` |
| C7 | `BM: tl.constexpr = x.shape[0]; tl.arange(0, BM)` (tutorial 06) | ran |
| C8 | `P: tl.constexpr = A == 1 and B == 2` (tutorial 10), `if P:` | ran: a compile-time `True`, the other branch not visited |
| C9 | `b: tl.constexpr = 4` in the body of `tl.static_range(2)` | rejected on the host: the second copy defines `b` again |
| R1 | `for i in range(tl.atomic_add(c, 1) * 0)`, 8 programs | 16 atomics: two per program |
| R2, R3 | `range(0, …)`, `tl.range(…)` with the same argument | 8 atomics each |
| R4 | `for i in range(tl.atomic_add(c, 1))`, one program, `c = 5` | 6 trips, `c = 7`: two atomics, and the bound is the second's value |
| D1, D2 | `h(x)` with `def h(x, e=1e-7): return x * e` / `h(x, 1e-7)`, `x` `bfloat16` | `[0, 0, 0, 0]` both: the default stays a Python number, like a literal passed directly |
| D3 | `e = 1e-7; h(x, e)` | `float32(1e-7)·x` (the tensor `e` is passed) |
| D4 | `def h(x, e: tl.constexpr = 1e-7)`, `h(x)` | `[0, 0, 0, 0]`, as D1 |
| P1, P2 | `def g(n: tl.constexpr): return n // 2`; `a = -7; g(a)` / `g(-7)` | `-3` / `-4`: the parameter holds the tensor `a`; the annotation changes nothing |
| L1 | `for i in range(0, n): t = i * 2`, then `t` after the loop | rejected on the host: "t is not defined" |
| L2 | `for i in range(0, n): s += i`, then `i` after the loop (`n = 4`) | ran and stored `16776272`, not `3`: the placeholder (poison) |
| L3 | the same, `s` after the loop (control) | `6` |
| L4 | `while j < n: u = j; j += 1`, then `u` after the loop | rejected on the host: "u is not defined" |
| Z1, Z2 | `for i in range(0, n): tl.atomic_add(c, 1)`, 8 programs, `n = 0` / `n = 1` | 0 / 8 atomics: the dry visit of the body emits nothing that runs |
| U1 | `a, b = 1, 2, 3` | ran and stored `1`, `2`: the third element is dropped silently |
| O1 | the masked-off lanes of a `float16` load with `other=1 + 2**-11 + 2**-30` | `0x3c00`: the `float32` constant (`1 + 2^-11`, a tie) rounded to `float16`, to even; the `float16` nearest to the double is `0x3c01` |
| S1, S1d | `tl.cumsum(x, 0, dtype=tl.bfloat16)`, `x` `bfloat16` `[1, 2^-8, 2^-8, 2^-8]` / default | `[1, 1, 1.0078125, 1.0078125]`: `bfloat16` sums / `[1, 1.00390625, 1.0078125, 1.01171875]`: `float32` sums |
| S2, S2d | `tl.cumsum(x, 0, dtype=tl.int8)`, `x` `int8` `[100, 100, 100, 100]` / default | `[100, -56, 44, -112]`: wrapping `int8` sums / `[100, 200, 300, 400]` in `int32` |
| S3, S3d | `tl.cumsum(x, 0, dtype=tl.float32)`, `x` `float16` `[2048, 1, 1, 1]` / default | `[2048, 2049, 2050, 2051]` in `float32` / `[2048, 2048, 2050, 2050]` in `float16` |

**Conclusions**, used by `docs/SEMANTICS.md` ("The supported subset", "How the front end evaluates
the source", `tl.load`, "Scans"), `docs/exclusions.md` and `docs/TRANSCRIPTION.md`:

1. `tl.where`'s values are typed as an arithmetic operator's operands (`whereTyping?`): with an
   integer `x` and a `float` constant the result is `float32`, `x` converted to it (W1); an `int`
   constant takes an integer `x`'s dtype (W2) and a `float` constant a float `x`'s (W4, W5); two
   Python numbers follow the float rules (W3).
2. `name: tl.constexpr = e` on a name not yet defined keeps `e`'s value as it is (C1, C7, C8); on a
   defined name it is rejected on the host (C2, and C9 in a second unrolled copy). Any later plain assignment to the name, or one to a
   `tl.constexpr` parameter, makes a Python number a tensor (C3, C4). A bare annotation changes the
   next assignment (C5) and another annotation is ignored (C6): both are excluded.
3. A one-argument `range(e)` evaluates `e` twice, and the bound is the second value (R1, R4); the
   other loop forms evaluate their arguments once (R2, R3).
4. A helper's default is a compile-time value in its body, like a Python number passed directly (D1,
   D2, D4), and an annotation on a helper's parameter has no effect (P1, P2).
5. A name a `for` or `while` body defines is not defined after the loop unless the loop carries it
   (L1, L3, L4), and a `for` loop's induction variable after the loop holds the placeholder, not the
   last value (L2): such a use is excluded.
6. Before a `for` loop the front end visits the body once to find the names the loop carries, and
   what that visit emits does not run: an atomic in a body that runs zero times ran in no program,
   and once in each program in a body that runs once (Z1, Z2). A tuple assignment with fewer names
   than elements drops the rest without an error (U1): the subset asks for one name per element.
7. `tl.load`'s `other` is converted to the pointer's dtype: a Python number first becomes a tensor of
   its own type, so a `float16` load's `other` is rounded twice, through `float32` (O1).
8. `tl.cumsum(x, dtype=d)` widens `bfloat16` to `float32`, then converts the input to `d`, and scans
   in `d` (S1-S3); without `dtype` it widens `bfloat16` and integers narrower than 32 bits.
