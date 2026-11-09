# Library batch, round 37: author's notes

Issues L165 (reopened), L167 and L168 ([ISSUES.md](../../ISSUES.md)). No kernel or claim folder was edited.
`lake build TVA` is green. The only Lean edits are three docstrings in `lean/TVA/Lang/Ops.lean` (the "Atomics" module
comment, `fBound`, `bitUpd`), so no definition or statement changed; the digests re-open every item whose cone reaches
the edited files (`lean/TVA/Lang/Ops.lean`, `docs/SEMANTICS.md`, `docs/exclusions.md`). No claim changes: none uses an
atomic's result, a float `atomic_and`/`or`/`xor`, or `tl.softmax` (every `softmax` in `claims/*/source` is the host's
wrapper), and the one `tl.debug_barrier` (`claims/layer_norm_fwd_f16`, source line 191) is in the backward kernel,
outside its claim. Last-round sweep: `docs/` (prompts included), every `evidence/*/README.md` and the Lean docstrings
were grepped for "every order", "any order", "either order", "one of the two values", "integer-only" and `softmax`;
each hit that described the semantics' order of a program's atomics without the exclusion, a float max/min's returned
value, or `tl.softmax`'s format is fixed below; the others describe orders the semantics does give (the `Safe`,
`Shared` and `SharedAcc` docstrings, `docs/LIBRARY.md`) or are already qualified (SEMANTICS.md "No barrier
semantics", "Granularity", `lean/TVA/Lang/Prog.lean`'s module comment).

## L165: Wording around the atomic-result exclusion

- **What changed.** Part (1) in its twin: the `tl.debug_barrier` row of `docs/exclusions.md` now says that the
  semantics allows every order of two atomics on the same bytes outside the exclusion of an atomic's result used on a
  shared cell (below; L161, L165), that inside it the device gives orders the semantics lacks and a barrier would
  remove some of them (`evidence/atomic-result-order`, `variants.py`), but that it keeps no meaning there either, so
  such a launch stays excluded. Parts (2) and (3) were accepted in round 36 and are unchanged.
- **Why.** lr-36a: the row still said the semantics allowed "every order" of those atomics, false inside the
  exclusion (launch B, and `variants.py` with and without a barrier).
- **Evidence.** `evidence/atomic-result-order/variants.py` re-run (GPU 2, H200 NVL, Triton 3.8.0, 2026-11-08): output
  identical to `variants.out.txt` and to lr-36a's re-run (barrier 0 of 100; loop, addr, unused2 100 of 100).
- **Constants whose meaning changed.** None (wording).

## L167: Unqualified "every order"

- **What changed.** `docs/SEMANTICS.md` "Programs": the semantics performs a program's atomics in every order that the
  dependences of its own steps allow; those are coarser than the device's (a tensor step that uses any lane of an
  atomic's result depends on every lane, a device thread orders only its own accesses), so it lacks orders the device
  gives where a program uses an atomic's result on a shared cell, and those launches are excluded ("Atomics"); the
  closing sentence ("except through atomics, whose every order the semantics allows") now adds "outside the exclusion
  of an atomic's result used on a shared cell". The same qualification in SEMANTICS.md "Order" (front-end evaluation
  order), in Conclusion 1 of `evidence/front-end-order/README.md` and Conclusion 3 of `evidence/intra-program/README.md`.
  `docs/LIBRARY_REVIEW.md`, the `gate` area: `tvr check` keeps the claim question; the release check's question is now
  whether `tvr release`, which checks no claim and no kernel's gate (L160) and reads of `kernels/` and `claims/` only
  the review records and, in the audit, the text, can pass a library that does not build or fails `tvr library-check`,
  its records, the lint or the audit.
- **Why.** lr-36a: the unqualified sentences are the premise L161 found false across threads; the release check has
  read no claim since round 35 (`cmd_release`, `tools/tvr.py`).
- **Evidence.** None needed (wording); the device side is `evidence/atomic-result-order` (re-run above).
- **Constants whose meaning changed.** None (wording).

## L168: Softmax formats and atomic wording

- **What changed.** (1) SEMANTICS.md "`/` on floats" states `tl.softmax` per input format: on `float16`, `bfloat16`
  and `float32` it computes and returns `float32` (`tl.max` widens), ending in `tl.div_rn` with `ieee_rounding=True`
  and in the default `float32` `/` otherwise; on `float64` it returns `float64`, ends in the IEEE `float64` `/` with
  the default, and fails to compile with `ieee_rounding=True`. `tl.fdiv(…, ieee_rounding=True)` fails on `float16`,
  `bfloat16`, `float64` and mixed operands. "Float `+` and `-`": the products are `tl.sigmoid` of `float32` and the
  default `tl.softmax` of 16- and 32-bit tensors; `float64` `/`, hence `tl.sigmoid` and the default `tl.softmax` of
  `float64`, are IEEE. The same in `evidence/fdiv-ieee` (new Part E) and `evidence/div-fusion`'s conclusion.
  (2) "Atomics": a float `atomic_max`/`atomic_min`'s *new value* is one of the old value and the operand (`fBound`),
  the returned value is the old one (also `fBound`'s docstring and review-questions.md S3); a new item lists the
  element types Triton 3.8.0 accepts per op (`semantic.py:1257-1290`), every other one rejected on the host.
  The probe found that `atomic_and`/`or`/`xor` on `float32`/`float64` compile (on the bit patterns), which the old
  "integer-only" denied: the library gives them no new value (`bitUpd`, undefined behaviour in the model), so
  SEMANTICS.md says so and `docs/exclusions.md` gains a row excluding them; `bitUpd`'s docstring and the "Atomics"
  module comment of `Ops.lean` say the same.
- **Why.** lr-36b: the softmax sentence could be read as a 16-bit compile failure; "the result" of a float max/min
  meant the new value; the accepted dtypes were not stated.
- **Evidence.** `evidence/fdiv-ieee/softmax_fmt.py` (lr-36b's `work/softmax_fmt.py`, adding a static check of the
  result's dtype), `softmax_fmt.out.txt` (GPU 0, H200 NVL, Triton 3.8.0, 2026-11-08): 16-bit and `float32` results
  `float32`, `div.rn.f32` 0 of 1024 off IEEE, `div.full.f32` by default; `float64` `div.rn.f64` 0 of 1024 off, rejected
  with `True`; same as lr-36b's output. New `evidence/atomic-dtypes` (`run.py`, `out.txt`, GPU 0, 2026-11-08): 8 ops ×
  15 element types; each rejection a front-end `CompilationError`; every accepted case returns the old bits and
  writes the expected value on 64 of 64 lanes. `evidence/relaxed-atomics/float_maxmin.py` re-run: identical.
- **Constants whose meaning changed.** None (docstrings of `fBound` and `bitUpd` only; no definition changed).
