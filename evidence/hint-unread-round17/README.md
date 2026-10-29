# The contiguity bound of a value whose contiguity hint the compiler may not read

**Question.** `docs/SEMANTICS.md` ("Compiler hints", "The contiguity the analysis infers") reads
`tl.multiple_of(x, v)` alone as a promise about the first element of each group of `C`, with `C` a
lower bound of the contiguity Triton's analysis infers for `x`. Round 15 gave a value that carries a
`max_contiguous` hint, alone or in the pair, the hinted `C` as that bound, "whatever operation
defines it". lr-15a reopened library issue L58: the library itself says that the compiler does not
read the mark on a loop's or a runtime `if`'s result, drops it on an inlined call and may lose it on
an operation with several results, and for those values the hinted `C` is above the compiler's
contiguity (`docs/library-review/round-15/lr-15a/work/probes.py` B0-B5). Which values does the
analysis give the hinted `C`, what is the bound of the others, and is there a bound that holds for
every value with a hint? Round 17.

**Sources** (Triton 3.8.0, pinned in `primitives/`).
- `triton-3.8.0-AxisInfo.cpp` lines 1232-1259 (`visitOperation`): for every operation the dataflow
  analysis visits, after a visitor computes its contiguity, divisibility and constancy, a
  `tt.contiguity`, `tt.divisibility` or `tt.constancy` attribute on the operation replaces the
  computed value; an operation no visitor handles gets the state of lines 1309-1341, which reads the
  same attributes. `tl.max_contiguous` and `tl.multiple_of` set them on the operation that defines
  the hinted value (`triton-3.8.0-ir.cc` lines 411-425).
- The results of an `scf.for` or `scf.if` are not computed by `visitOperation`: MLIR's dataflow
  framework joins into them the values the regions yield (its handling of region branches), so an
  attribute on the loop or the `if` is read by no visitor (evidence/hint-multi M7-M10; N2, N3 here). A
  call of a `@triton.jit` function is a `tt.call`, which the inliner replaces by the callee's body:
  an attribute on the call goes with it, and the call's result becomes the value the body returns
  (evidence/hint-multi M3, M12, M13; N4, N7 here).
- An operation the canonicalizer folds away (`arith.addi %r, 0` to `%r`, `arith.muli %r, 1` to `%r`)
  is erased with its attributes, and its uses get the folded value, which carries no mark (N9, N10
  here).

**Method.** `run.py` loads 512 `int8` elements through `x_ptr + y`, with
`y = tl.multiple_of(r + s, 16)`, `s = 16` and one warp, for eleven definitions of `r`. A load is at most
`min(contiguity, divisibility)` bytes wide, so the PTX shows the contiguity the analysis gave `y`: 16
bytes (`ld.global.v4.b32`), 4 bytes (`ld.global.b32`) or 1 byte (`ld.global.b8`). In N0-N5, N9 and
N10, `r` is loaded from memory (`idx`, or `z` in N5), so the PTX also has that load (`v4.b32`,
`v2.b32`); the load through `y` is the other one. Each case runs in its own process, with its own Triton cache, and
checks the loaded values.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/hint-unread-round17/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`; the load through `y`).

| case | `r` | load through `y` | values |
|---|---|---|---|
| N0 | `tl.load(idx + i)`, no hint (control) | 1 byte | correct |
| N1 | `tl.load(idx + i) + step`, the pair (16, 16) on it | 16 bytes | correct |
| N2 | the result of a `for` loop that carries `r`, the pair (16, 16) after the loop | 1 byte | correct |
| N3 | the result of a runtime `if` that merges `r`, the pair (16, 16) after the `if` | 1 byte | correct |
| N4 | an inlined helper's result (a load), the pair (16, 16) on it | 1 byte | correct |
| N5 | the first result of `tl.split` of a loaded `[512, 2]` block, the pair (16, 16) on it | 1 byte | correct |
| N6 | `tl.arange(0, 512)`, the pair (4, 4) on it (control) | 4 bytes | correct |
| N7 | an inlined helper's result (`tl.arange(0, 512)`), the pair (4, 4) on it | 16 bytes | correct |
| N8 | an inlined helper that returns the pair (4, 4) on `tl.arange(0, 512)` | 4 bytes | correct |
| N9 | `tl.load(idx + i) + 0`, the pair (16, 16) on it | 1 byte | correct |
| N10 | `tl.load(idx + i) * 1`, the pair (16, 16) on it | 1 byte | correct |

**Conclusions**, used by `docs/SEMANTICS.md` and `TVA.Lang.Ops` ("Compiler hints": the contiguity
the analysis infers):

1. A hinted `C` is the analysis' contiguity only when the compiler reads the mark: on an operation
   with one result that is not a loop, a runtime `if` or a call, and that the canonicalizer keeps (N1,
   N6, N8, and evidence/hint-hinted-bound G2-G6). On a loop's or a runtime `if`'s result (N2, N3), on
   the value an inlined call returns (N4, N7), on a result of `tl.split` (N5) and on an operation that
   folds away (N9: `+ 0`; N10: `* 1`) the hint gave no bound: the loads were as wide as without it
   (N0: 1 byte; N7: the arange's 16 bytes, not the hinted 4). With an operation with several results
   the mark is sometimes read (evidence/hint-multi M4-M6, M14) and sometimes lost (M1, N5).
2. Which operations fold depends on the canonicalizer's patterns, so the library does not list the
   operations whose mark survives. It takes, for a value with a `max_contiguous` hint, the smaller of
   the hinted `C` and the bound the list gives the value without the hint. That is a lower bound in
   both cases: when the mark is read the contiguity is `C`, and when it is not the analysis keeps its
   own contiguity, of which the list's bound is a lower bound (`1` for a loop's, an `if`'s, a
   several-result operation's and a load's result; for an inlined call's result, the bound of the
   value its body returns, N7: `512`, N8: the hinted `4`). No wrong value arose here: the vectoriser
   takes the smaller of contiguity and divisibility, so a bound above the compiler's would make the
   model's promise weaker than the compiler's belief, not the loads wider.
