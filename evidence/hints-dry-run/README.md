# Hints on loop-carried values reach the code before the loop

Library evidence (not a kernel's). Triton 3.8.0 (`~/fpv/tva/.venv`), NVIDIA H200 NVL, 2026-10-27.
Script: [run.py](run.py); output: [out.txt](out.txt).

**Question.** Tutorial 08 writes `tl.multiple_of(a_ptrs, [16, 16])` inside a `for` loop that
carries `a_ptrs`. Is that hint a no-op, as the library said?

**Why it might not be.** To find the loop-carried names, the front end visits the loop body
once before building the loop (`_find_carries`, `code_generator.py:465-503`), with every name
still bound to its value from before the loop, and then erases the block it built. During that
visit `tl.multiple_of(a_ptrs, …)` sets `tt.divisibility` on the operation that defines the
pre-loop `a_ptrs`. That operation is outside the erased block, so the attribute stays.

**What the run shows** (the kernel, verbatim from the tutorial, compiled as written and without
its two hint lines):

| | as written | without lines 148-149 |
|---|---|---|
| TTIR | `tt.divisibility = dense<16>` on the pre-loop `tt.addptr` of `a_ptrs` and `b_ptrs` | no such attribute |
| PTX global loads of `A`, `B` | `cp.async.cg.shared.global` (16-byte copies) | `ld.global.b16` (2-byte loads) |
| `lda = ldb = ldc = 64` | correct | correct |
| `lda = 36` (row 1 of `A` at byte 72, not 16-byte aligned) | **`CUDA error: misaligned address`** | correct |

**Conclusion.** A hint inside a `for` (or `while`) body applies to the value each hinted name
has *before* the loop, as well as being recorded, without effect, on the loop's block argument.
The library's meaning of hints has to say this; dropping such a hint is not sound.
