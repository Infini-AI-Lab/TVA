# The condition of a branch the configuration decides, and what the front end evaluates

Library evidence (not a kernel's), for library issue L25 (rounds 4 and 5). Triton 3.8.0
(`~/fpv/tva/.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-28 (round 5;
round 4's cases C1–C4 and P1–P3 rerun with the same results). Script: [run.py](run.py); output:
[out.txt](out.txt).

**Since round 6** (library issue L36), TRANSCRIPTION.md is guidance and states no rule; the facts this
evidence establishes are stated in [SEMANTICS.md](../../docs/SEMANTICS.md), "How the front end evaluates the source".

**Questions.**
1. Round 4. The drop reason `[config]` dropped the lines of a branch a constexpr condition does
   not take. Does an operation inside such a condition run?
2. Round 5. `[config]`, `[compile-check]` and `[no-op:print]` transcribe the operations of a
   condition or argument that is not pure. Which of them does the front end evaluate, when
   `and`, `or`, a conditional expression or an `elif` meets a compile-time value?

**What the library reads.** The front end evaluates the condition of every `if` and conditional
expression as an ordinary expression before it decides whether the branch is static, and then
visits only the branch it takes (`visit_If`, `visit_IfExp`,
`primitives/triton-3.8.0-code-generator.py:967-1049`); an `elif` is an `if` inside the `else`
branch. `visit_BoolOp` (`:1481-1531`) visits the operands of `and`/`or` in order and returns at
the first operand that is not a tensor and decides the result (false for `and`, true for `or`); a
tensor operand never stops it. `visit_Compare` (`:1076-1093`) makes `is` and `is not` compile-time
values whatever their operands.

**Method.** Kernels of 4 programs whose condition or call argument holds `tl.atomic_add(c, 1)`
(S15 also `tl.atomic_add(c, 2)`). The counter is the sum of the operands of the atomics that ran
over all programs (4 per atomic that runs in every program), the outputs show which branch was
taken, and the TTIR whether the atomic is compiled and whether the branch is static (no `scf.if`).
Each kernel compiles in a fresh cache, and what the compiler and the device print
(`tl.static_print`, `tl.device_print`) is captured and counted.

**Result** (`out.txt`).

| Case | Counter | Branch | TTIR |
|---|---|---|---|
| C1 `if tl.atomic_add(c, 1) is not None:` | 4 | then | 1 atomic, no `scf.if` |
| C2 `v = 1 if tl.atomic_add(c, 1) is None else 2` | 4 | else | 1 atomic, no `scf.if` |
| C3 `elif tl.atomic_add(c, 1) is not None:` after `if FLAG:` with `FLAG = False` | 4 | the `elif` | 1 atomic, no `scf.if` |
| C4 `if FLAG and tl.atomic_add(c, 1) is not None:` with `FLAG = False` | 0 | else | no atomic |
| P1 `if FLAG:`, P2 `if N > 4:`, P3 `1 if FLAG else 2` | 0 | as configured | no atomic, no `scf.if` |
| S1 C4's kernel with `FLAG = True` | 4 | then | 1 atomic |
| S2, S3 `if FLAG or tl.atomic_add(c, 1) is None:` with `FLAG = True`, `False` | 0, 4 | then, else | 0, 1 atomic |
| S4 `if (<atomic> is None) and (<atomic> is None):` | 4 | else | 1 atomic: the first, a compile-time `False`, ends the `and` |
| S5 `if (<atomic> is not None) or (<atomic> is None):` | 4 | then | 1 atomic: the first, a compile-time `True`, ends the `or` |
| S6, S7 `v = 1 if (<atomic> if FLAG else None) is None else 2` with `FLAG = False`, `True` | 0, 4 | then, else | 0, 1 atomic |
| S8 C3's kernel with `FLAG = True`: the `elif` follows a taken `if` | 0 | the `if` | no atomic |
| S9 `if (tl.load(x) > 0) and (tl.atomic_add(c, 1) > 0):` with `x = 0` | 4 | else | 1 atomic, 1 `scf.if`: a tensor operand does not end the `and` |
| S10 `tl.static_assert(FLAG or <atomic> is None)` with `FLAG = True` | 0 | — | no atomic |
| S11 `tl.static_print(FLAG and <atomic>)` with `FLAG = False` | 0 | — | no atomic; prints `False` |
| S12, S13 `tl.static_print(<atomic> if FLAG else 0)` with `FLAG = False`, `True` | 0, 4 | — | 0, 1 atomic; prints `0`, `int32[]` |
| S14 `tl.device_print("v", <atomic> if FLAG else 0)` with `FLAG = False` | 0 | — | no atomic; the device prints `v: 0` (512 lines) |
| S15 `if (FLAG and <atomic +1> is None) or <atomic +2> is not None:` with `FLAG = False` | 8 | then | 1 atomic, the `+2` one: the inner `and` ends at `FLAG`, the `or` goes on |
| S16 `if <atomic> is not None and FLAG:` with `FLAG = False` | 4 | else | 1 atomic: it comes before the deciding operand |

**Conclusions.**
1. An operation inside the condition of a static branch runs in every program (C1–C3), so the
   `if`/`elif` line is not a no-op. `[config]` may drop such a line only when its condition is pure
   (`[pure]`); otherwise the operations the front end evaluates in the condition are model lines.
2. The front end does not evaluate every operation of an expression (S1–S16): `and` stops at a
   compile-time false operand and `or` at a compile-time true one (a tensor operand stops
   neither), a conditional expression with a compile-time condition evaluates one side, and an
   `elif` after a taken compile-time branch is not visited at all. The same holds for the arguments
   of `tl.static_assert`, `tl.static_print` and `tl.device_print` (S10–S14). SEMANTICS.md ("How
   the front end evaluates the source", since round 6) states this:
   a part the front end does not evaluate never runs, so it neither makes an expression impure nor
   becomes a model line.
