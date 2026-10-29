# What debugging and compile-time calls leave in a kernel

Library evidence (not a kernel's), for library issues L6 and L10. Triton 3.8.0
(`~/fpv/tva/.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27, compiled
without `debug` (`TRITON_DEBUG` unset). Script: [run.py](run.py); output: [out.txt](out.txt).

**Since round 6** (library issue L36), TRANSCRIPTION.md is guidance and states no rule; the facts this
evidence establishes are stated in [SEMANTICS.md](../../docs/SEMANTICS.md), "How the front end evaluates the source" (debugging
calls, `tl.static_assert`, expressions with no effect).

**Question.** The transcription rules drop `tl.static_assert` (`compile-check`) and
`tl.static_print`, `tl.device_print`, `tl.device_assert` (`no-op:print`). When is that sound?

**What the front end does** (`primitives/triton-3.8.0-code-generator.py`):
- A call's arguments are evaluated as ordinary expressions before the call, so every operation
  in them is emitted (`visit_Call`, `:1455-1476`; the static calls get their own evaluation, below).
- `tl.static_assert` evaluates its condition, and its message only when the condition is false
  (`execute_static_assert`, `:1617-1637`). A condition that is not a compile-time `bool` is an
  error. `x is None` and `x is not None` are compile-time `bool`s for any value `x`
  (`visit_Compare`, `:1083-1086`).
- `tl.static_print` evaluates its arguments and prints them at compile time (`static_executor`,
  `:1639-1659`).
- A Python `assert c, m` statement is `tl.device_assert(c, m)` (`visit_Assert`, `:1353-1356`), and
  Python's `print` is `tl.device_print` (`builtin_namespace`, `:357-365`).
- Without `debug`, `tl.device_assert` emits nothing (`triton-3.8.0-semantic.py:1810-1815`).
  `tl.device_print` emits `tt.print` (`:1796-1808`), a device `printf` that writes no global memory.

**Result.**

| Case | Result |
|---|---|
| P1 four `tl.static_assert`s, a `tl.static_print`, two `tl.device_assert`s and an `assert`, all on pure arguments | TTIR and PTX identical to the kernel without them; same output |
| P2 `tl.device_print` and `print` of pure values | a `tt.print` (`vprintf`) is added and nothing else; same output |
| E1 `tl.static_assert(tl.atomic_add(c, 1) is not None)` | compiles; the atomic runs in every program |
| E2 `tl.static_assert(bump(c) is None)`, `bump` a `@triton.jit` helper that adds | compiles; the helper's atomic runs |
| E3–E7 an atomic inside `tl.static_print`, `tl.device_print`, `tl.device_assert`, an `assert` (through a helper), `print` | each compiles; the atomic runs in every program |
| E8 an atomic in the *message* of a `tl.static_assert` that holds | compiles; the atomic does **not** run |
| E9 an atomic in the `mask` of a `tl.device_assert` | the atomic runs |
| V1 a false `tl.static_assert` inside a runtime `if` no program takes | rejected at compile time |
| V2 a `tl.static_assert` false in one unrolled `tl.static_range` iteration | rejected at compile time |
| V3 a false `tl.static_assert` in a `for` body that runs zero times | rejected at compile time |
| V4 a false `tl.static_assert` in a constexpr `if` branch not taken | compiles (the branch is not visited) |

**Conclusions.**
1. With pure arguments (P1, P2), these calls change nothing a launch does to global memory, so the
   model may drop them. Which expressions have no effect is stated in [SEMANTICS.md](../../docs/SEMANTICS.md) ("How the front end
   evaluates the source", since round 6).
2. An operation inside an argument runs, whatever the call (E1–E7, E9). Dropping such a line loses
   the operation, so the argument's operations must be transcribed.
3. A `tl.static_assert`'s message is evaluated only when the assertion fails (E8), and then the
   compile fails, so the message never matters to a launch that runs.
4. A `tl.static_assert` is evaluated wherever the front end visits it: in both branches of a
   runtime `if`, in each unrolled iteration, and in a loop body even if it runs zero times
   (V1–V3); not in a constexpr branch the configuration does not take (V4). A false one rejects
   the compile (no outcome, library issue L3).
