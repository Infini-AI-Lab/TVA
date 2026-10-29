# Compiler hints inside loop bodies and on block arguments

**Question.** `docs/SEMANTICS.md` ("Compiler hints") reads `tl.multiple_of`, `tl.max_contiguous`
and `tl.max_constancy` as promises about the hinted value. Which value does a hint written inside
a `for` or `while` body, or on a parameter, actually promise something about? The front end visits
a loop body once with placeholder values before it emits the loop (`_find_carries`,
`primitives/code_generator.py` lines 465-503), and `set_attr` (`primitives/ir.cc` lines 411-425)
marks the *defining operation* of the hinted value. Library issue L1 (round 8).

**Method.** `run.py` compiles thirteen small kernels with Triton 3.8.0 and launches each once, in
its own process, on GPU 0. Each kernel loads 128 float32 with one warp, so each thread holds four
consecutive elements and a load is vectorised (`ld.global.v4.b32`, 16 bytes) exactly when the
compiler believes the address is 16-byte aligned. For each case the script prints the result
against a host reference (or the launch's error), the global loads in the PTX, and the hint
attributes in the front end's output (`asm['source']`, before any pass). A base 4 bytes past a
16-byte boundary (`x[1:]`) makes a 16-byte promise false.

- C1-C4: `for` loop carrying `p`; the hint `tl.multiple_of(p, 16)` is the first statement of the
  body. C1 `p` true before the loop, `p += 1` (later values 4-byte aligned); C2 `p += 4`, base
  false; C3 `p += 4`, base true; C4 as C2 without the hint.
- C5: `i = tl.multiple_of(i, 16)` on the induction variable of `range(1, n)`.
- C6: `x_ptr = tl.multiple_of(x_ptr, 16)` on a kernel parameter, given `x[1:]`.
- C7: the same on a `@triton.jit` helper's parameter, given `x + 1`.
- C8, C9: two nested `for` loops that both carry `p`; the hint is in the inner body.
- C10, C11: `p = x + o + offs` assigned in the outer body (`o` the outer induction variable), carried
  by the inner loop; the hint is in the inner body. C10 runs one outer iteration, C11 two.
- C12, C13: a `while` loop carrying `p`, as C1 and C2.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-loops/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`). The column "promise true for" reads each `tl.multiple_of(p, 16)` as the
library does since round 11 (library issue L58, `evidence/hint-contig`): `p = x_ptr + offs` with
`offs = tl.arange(0, 128)` has contiguity 128 for the compiler's analysis, so the hint promises that
the first pointer of the block, `p[0]`, is 16-byte aligned (`FirstsDiv p 0 128 16`). Under the
library's reading before round 11 (every element a multiple of 16, since no `max_contiguous` is on
the value) no case's promise is true: consecutive `float32` pointers are 4 bytes apart.

| case | hint on | promise true for | loads | result |
|---|---|---|---|---|
| C1 | value carried by the loop | value before the loop only | `ld.global.b32` | correct |
| C2 | value carried by the loop | none (value before the loop false) | `ld.global.v4.b32` | misaligned address |
| C3 | value carried by the loop | every value | `ld.global.v4.b32` | correct |
| C4 | (no hint) | - | `ld.global.b32` | correct |
| C5 | induction variable | - | `ld.global.b32` | correct |
| C6 | kernel parameter | - (false) | `ld.global.b32` | correct |
| C7 | helper parameter | - (false) | `ld.global.b32` | correct |
| C8 | carried by both loops | value before the outer loop only | `ld.global.b32` | correct |
| C9 | carried by both loops | none | `ld.global.v4.b32` | misaligned address |
| C10 | assigned in the outer body | the one outer iteration's value | `ld.global.v4.b32` | correct |
| C11 | assigned in the outer body | first outer iteration's value only | `ld.global.v4.b32` | misaligned address |
| C12 | value carried by a `while` | value before the loop only | `ld.global.b32` | correct |
| C13 | value carried by a `while` | none | `ld.global.v4.b32` | misaligned address |

What the front end's output shows: a hint on a carried name marks the operation that defined the
value *before* the loop (`%p_0 = tt.addptr %p, %offs {tt.divisibility = ...}`, outside the loop),
plus an attribute `tt.divisibility_arg<N>` on the loop operation that no pass reads. A hint on the
induction variable marks the dead placeholder `ub.poison` the dry visit used. A hint on a kernel
or helper parameter (a block argument of the function) leaves no attribute at all.

Conclusions, used by `docs/SEMANTICS.md` ("Compiler hints"):

1. A hint on a value carried by a loop is a promise about that value as it was **before the loop**
   (where an operation defined it), and promises nothing about the values of later iterations
   (C1 and C12 stay scalar; C2, C9 and C13 fault when the value before the loop is false). With
   nested loops that both carry the value, it is the value before the outermost carrying loop (C8,
   C9).
2. A value assigned in an enclosing loop's body and carried by the inner loop is checked once per
   iteration of the enclosing body: the promise is about each such value (C10 correct, C11 faults).
3. A hint on a loop's induction variable, a kernel parameter or a helper's parameter has no effect
   (C5, C6, C7: false promises, no fault, no vector load).
