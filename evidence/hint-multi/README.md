# A hint on one result of an operation with several results, and on an inlined call's result

**Question.** A hint marks the operation that defines the hinted value (`set_attr`,
`primitives/triton-3.8.0-ir.cc` lines 411-425). When that operation has several results, does the
hint reach the others? And what does a hint on the result of a call of a `@triton.jit` function, a
loop or a runtime `if` do? Library issue L60 (round 11; raised by lr-8b, whose
`docs/library-review/round-8/lr-8b/work/t_hint_multiresult.py` showed it for `tl.split`).

**Sources** (Triton 3.8.0).
- `primitives/triton-3.8.0-AxisInfo.cpp` lines 1232-1259 and 1318-1342: an operation the analysis
  has no visitor for sets every result to `getPessimisticValueState`, which reads the hints off the
  defining operation, so a hint on one result is read for each. The operations with several results
  here (`tt.split`, `tt.reduce`, `tt.scan`, `tt.call`) have no visitor (lines 1172-1214 list the visitors).
- A `for` or `while` loop's results and a runtime `if`'s are the results of one operation
  (`scf.for`, `scf.while`, `scf.if`; `primitives/triton-3.8.0-code-generator.py` lines 932-966), whose
  values the data-flow analysis takes from what the body yields.
- A call of a `@triton.jit` function is a `tt.call` that the inliner removes, except for a helper
  declared `noinline=True` (`code_generator.py` lines 1358-1396); so are the `tl.*` functions written
  in Triton (`primitives/triton-3.8.0-standard.py`).

**Method.** `run.py` loads two index vectors of 64 `int32` from memory, `a = 4i` (every element a
multiple of 4) and `b = 4i + 2` (none), builds two results of one operation holding them, hints one
with `tl.multiple_of(…, 4)`, and loads `x[w[i] + c]`, `c` in 0..3, through one result `w`. With one
warp, the load is `ld.global.v4.b32` when the compiler believes `w[i]` is a multiple of 4, and that
faults on `b`'s rows. It also prints the operation the hint is on in the front end's output and
after the TTIR passes (inlining among them). Each case runs in its own process, on GPU 0.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-multi/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | operation | hint on | loads through | result |
|---|---|---|---|---|
| M1 | `tl.split` of a loaded `[64, 2]` block | `r0` (true) | `r1` | scalar loads, correct (the hint is gone after the conversion to TTGIR) |
| M14 | `tl.split` of `tl.reshape(tl.arange(0, 128), (64, 2)) * 2` | `r0` (true) | `r1` | `v4`, misaligned address |
| M2 | `tl.split`, no hint (control) | - | `r1` | scalar loads, correct |
| M3 | `tl.max(…, return_indices=True)` (a call, inlined) | the indices (true) | the values | scalar loads, correct; the mark was on the `tt.call`, gone after inlining |
| M4 | tuple `tl.reduce` | `r0` (true) | `r1` | misaligned address |
| M5 | tuple `tl.associative_scan` | `r0` (true) | `r1` | misaligned address |
| M6 | a `noinline=True` helper returning two scalars | `s0` (true) | `s1 + 4i` | misaligned address |
| M7, M8 | the results of a `for` loop | `r0` (true), `r1` (false) | `r1` | scalar loads, correct: the mark is on the `scf.for` and has no effect |
| M9, M10 | the results of a runtime `if` | `r0` (true), `r1` (false) | `r1` | scalar loads, correct: the mark is on the `scf.if` and has no effect |
| M11 | `b + d` (one operation, control) | `r1` (false) | `r1` | misaligned address |
| M12 | an inlined helper's result | `r1` (false) | `r1` | scalar loads, correct; the mark was on the `tt.call`, gone after inlining |
| M13 | `tl.max(…, axis=1)` (an inlined library function) | `r1` (false) | `r1` | scalar loads, correct; likewise |

**Conclusions**, used by `docs/SEMANTICS.md` and `TVA.Lang.Ops` ("Compiler hints": which value):

1. A hint on one result of an operation with several results can apply to every result: a true
   hint on one result made the loads through another vectorise, and they faulted, for a tuple
   `tl.reduce` (M4), a tuple `tl.associative_scan` (M5), a `noinline` helper (M6) and a `tl.split`
   (M14). Whether the mark survives depends on the passes (M1: lost). So the library reads such a
   hint as a promise about each result, and the model checks each; that is sound whether the mark
   survives or not.
2. A hint on the results of a loop or a runtime `if` had no effect, true or false (M7-M10), but the
   library treats them as above (each result checked), which is sound either way.
3. A hint on the result of an inlined call (a helper, `tl.max`, and so the other `tl.*` functions
   written in Triton) has no effect: the mark is on the call, and inlining removes it (M3, M12,
   M13: a false hint, no fault, no vector load). The model drops it, as for a parameter.
