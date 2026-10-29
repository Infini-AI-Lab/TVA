# Code the front end visits after a `return`: compiled, never run

**Question.** `docs/SEMANTICS.md` ("`return`") said that code after a `return` at the top level is
not compiled. lr-15b (library issue L89; `docs/library-review/round-15/lr-15b/work/dead_after_return.py`
D1-D9) showed that after a compile-time `if` whose taken branch returns, and after a
`tl.static_range` copy that returns, the rest of the function is still compiled: it never runs, but a
hint in it marks an operation defined before the `return` (D7, D9: an earlier load vectorised and read
wrong values; D1: it faulted), and a `tl.static_assert(False)` in it rejects the kernel (D5). Does
the same hold after a runtime `if` whose two branches return, and what else of the visited code stays?
Round 17.

**Sources** (Triton 3.8.0, `primitives/triton-3.8.0-code-generator.py`).
- lines 508-517 (`visit_compound_statement`): the visit of a statement list stops after a `return`
  statement of that list, and only there.
- lines 544-554 (`visit_Return`): the returned value is recorded, and a new block, which no branch
  enters, becomes the insertion point; whatever the front end visits next is emitted into it.
- lines 893-929 (`visit_if_top_level`, a runtime `if` holding a `return`): each branch ends with a
  branch to the block after the `if`, from the block it ended in; when both branches returned, both
  are blocks no program reaches, and so is the block after the `if`.
- lines 556-585 (`decide_return_type`): a function's return type is the common type of every
  recorded `return`, those visited into an unreachable block included: two equal compile-time values
  stay a compile-time value, two different ones become a tensor type, and other mismatches are
  rejected.

**Method.** `run.py`, each case in its own process with its own Triton cache, one warp:
- E1: `offs = tl.where(i < m, 16 + i, 0)` (`m = 100`), a load of 256 `int8` through `offs`, then a
  runtime `if n > 0: return` / `else: return`, then the pair `tl.max_contiguous(tl.multiple_of(offs,
  16), 16)`, false past lane 100; the PTX's loads and the values. E2: the same without the hint;
- E3: `if FLAG: return` (`FLAG` a `tl.constexpr`, true), then `tl.atomic_add(c, 1)`, 8 programs;
- E4-E6: a helper called with `FLAG` true: `if FLAG: return -7` then `return 0` (E4), then
  `return -7` (E5), and `return -7` alone (E6); the kernel stores `h() // 2` (Python's `-4` on a
  compile-time value, the tensor `//`'s `-3` on a tensor);
- E7: E1's runtime `if` followed by `tl.static_assert(False)`.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/dead-code-round17/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

| case | what | result |
|---|---|---|
| E1 | the pair hint after a runtime `if` whose branches both return | `ld.global.v2.b32`; values wrong in 137 of 256 lanes |
| E2 | the same without the hint | `ld.global.b8`; values correct |
| E3 | `tl.atomic_add` after a taken compile-time `return` | `c = 0` after 8 programs |
| E4 | `if FLAG: return -7` then `return 0` | `h() // 2` stored `-3`: a tensor |
| E5 | `if FLAG: return -7` then `return -7` | `-4`: a compile-time value |
| E6 | `return -7` alone | `-4` |
| E7 | `tl.static_assert(False)` after E1's `if` | rejected on the host (`CompileTimeAssertionFailure`) |

**Conclusions**, used by `docs/SEMANTICS.md` ("`return`", "The supported subset") and
`docs/exclusions.md`:

1. After a statement every path through which ends in a `return` (a compile-time `if` whose taken
   branch returns, a `tl.static_range` copy that returns, a runtime `if` whose two branches return),
   the front end still compiles the rest of the function, into a block no program reaches. Nothing
   in it runs (E3), but what visiting it does at compile time stays: its hints mark operations
   defined before the `return` (E1; lr-15b D1, D7, D9), its `tl.static_assert`s are evaluated (E7;
   D5), and in a helper its `return`s take part in the helper's return type (E4: a compile-time
   value became a tensor).
2. Such code is outside the supported subset (library issue L89): no in-scope tutorial kernel has a
   statement after one that returns, and a model that drops that code, as its never running suggests,
   misses these effects. After a `return` in the same statement list nothing is visited (lr-15b D4,
   D6), and that form stays in the subset.
