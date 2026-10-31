# Library batch, round 23: author's notes

Issues: L117 and L118 ([ISSUES.md](../../ISSUES.md)), 2026-10-29, both from `vec_add_f32`'s
round-2 reviewers' library reports, and L117 by the user's decision of 2026-10-29 (under L36). No
review follows this batch (coordinator): it is checked by `lake build TVA`, `tvr library-check`,
`tvr lint library 23 author`, `tvr audit`, `tvr check vec_add_f32` and the release check. Worked in
TVA's tree itself.

**Build and checks.** `lake build TVA` is green (3192 jobs). Only docstrings changed, no
definition, statement or proof, so no axiom changed; no `sorry` or `axiom` in the library.
`tvr library-check` passes (hygiene, and `leanchecker TVA` re-checks every declaration). `tvr audit`
prints `independent`. `tvr check vec_add_f32`, run once after the last library change, passes: gate
PASS, digest `5f65eb1f7a961315`. Its items re-open, as the coordinator expected, because
`lean/TVA/Rules/Compiler.lean`, `lean/TVA/Claim.lean`, `lean/TVA/Lang/Scalar.lean`,
`docs/SEMANTICS.md`, `docs/exclusions.md` and `docs/rules/compiler-faithful.md` are hashed whole;
no kernel stops building.

**Release check.** Run once at the end, after `tvr check vec_add_f32` and these notes, in an empty
folder; its output is kept in `evidence/tool-round23/` (`release.out`, `release.json`, and a
README with the result).

## L117: condition 11 is stated for the launch's own argument values, with no reuse clause

- **What changed.** Condition 11 of `Compiler.Faithful`'s scope (part (B), assumed) now reads as the
  user decided (2026-10-29, under L36: "for the viriable n_elements, its value should be merged into
  the assumptions of the prover. e.g., x.len >= n_elements, etc."): the code the launch runs is the
  code Triton 3.8.0 compiles for that launch, from the kernel's own source, the launch's own
  argument values, compile-time and runtime, as the claim's instance gives them, and the settings
  the other conditions give. The clause "or earlier for the same source, compile-time values and
  settings (a repeated launch of the same instance)" is dropped, and so is "It is not code compiled
  earlier, for other values or under other settings, and taken from Triton's caches". In their
  place: code compiled earlier counts when it is that code, that is code a cache returns (the
  `JITFunction`'s cache in the process, the cache directory), for example for a repeated launch of
  the same instance, and the code a `CompiledKernel` runs within scope 6; TVA does not verify the
  caches. And a runtime argument's value is part of the instance, which the prover's conditions
  constrain (for example `n ≤ x.len` for a length `n_elements`). The places:
  - `lean/TVA/Rules/Compiler.lean`, the docstring of `Faithful`: condition 11 as above, with the
    decision quoted; its rationale (the caches' keys) ends "and may not be that code"; L117 joins
    the scope's list of issues.
  - `lean/TVA/Claim.lean`, the module docstring: part (B)'s "the kernel new" says the same (the code
    Triton 3.8.0 compiles for the launch's own argument values; code a cache returns counts when it
    is that code, for example for a repeated launch; TVA does not verify the caches; a runtime
    argument's value is part of the instance, and `Inst.ok` constrains it); L117 joins its list.
  - `docs/rules/compiler-faithful.md`: point 11 as in Lean, and a new paragraph on runtime
    arguments (the decision quoted; code compiled for another value of a runtime argument counts
    only when it is the code compiled for the launch's own, with point 6's record,
    evidence/compiled-launch K2); the summary in "What it says" ("the kernel new: the code Triton
    compiles for the launch's own argument values and settings"); "What it does not cover" (code
    other than that code, which Triton's caches may return); the checklist line on `d.ok` (a
    runtime argument's value is part of the instance, and `d.ok` constrains it; point 11); and the
    scope's list of issues.
  - `docs/SEMANTICS.md`: "Launches", the point "the kernel is new" says what condition 11 says, the
    decision quoted, and the list of issues gains L117. The one other sentence that paraphrased
    point 11 as "not code from Triton's caches" ("The supported subset", "Parameters and
    arguments") now says the code Triton compiles for the launch's own argument values, "and not
    other code from Triton's caches".
  - `docs/exclusions.md`: the cache row excludes a launch that runs code other than the code Triton
    3.8.0 compiles for that launch, from its source, its own argument values and the other
    conditions' settings; code from the caches may be such code in the cases the row listed
    before; code a cache returns counts when it is that code; and a runtime argument's value is part
    of the instance (the decision quoted). L117 joins the row's issues.
  - `docs/PROVER_GUIDE.md` (§2, after `Inst.ok`'s list, and part (B)'s summary) and
    `docs/REVIEWER_GUIDE.md` (the environment's point 11, and §2 "The conditions"): a runtime
    argument's value is part of the instance and the prover's conditions constrain it, for example
    `n ≤ x.len` for a length `n_elements` (the decision quoted); point 11 assumes the code Triton
    compiles for those values, so the conditions, not the caches, decide which values a claim
    covers.
  - `docs/TRANSCRIPTION.md` (guidance, in no digest): its one paraphrase of point 11, in "The host",
    says the same, so that it does not contradict the rule.
- **Why.** r-vecadd (F3) and r-vecadd2 (F4) found, independently, that the reuse clause admitted
  code compiled earlier "for the same source, compile-time values and settings" without the same
  specialisation of the runtime arguments: read literally, a launch with `n = 98433` was covered
  running code compiled for `n = 98432`, and point 6's own record (evidence/compiled-launch K2)
  shows code of another specialisation writing past `n`; its example, "a repeated launch of the same
  instance", could also be read as the only reuse allowed. The user decided that a runtime
  argument's value belongs to the instance, where the prover's conditions constrain it, so the
  condition is stated for the launch's own argument values (compile-time and runtime) and needs no
  clause about reusing other code: code a cache returns counts exactly when it is that code. The
  example is given as one case ("for example"). The code a `CompiledKernel` runs within scope 6 is
  named because the dropped clause was what covered it (code compiled earlier, at the warm-up, for
  the same source, compile-time values and settings); without it, scope 6's launches (tutorial 02)
  would read as outside condition 11, which part (A) covers. `vec_add_f32`'s claim is not affected:
  its `Inst.ok` has `n ≤ x.len`, `n ≤ y.len` and `n ≤ output.len`, and each of its launches is an
  instance with its own `n`.
- **Evidence.** No new evidence: condition 11 is part (B), an assumption TVA does not verify. The
  rule page cites the existing record of evidence/compiled-launch K2 (`n = 1001` on code compiled
  for `n = 1024` wrote three elements past `n`), and the caches' keys stay as recorded
  (evidence/compile-keys K1–K4, evidence/host-round21 K1–K4 and X1–X5). Why cached code compiled
  for another runtime value can be the launch's own code, from the pinned source: a JIT launch
  compiles from the signature, the constexprs and the attributes `specialize.cc` derives from the
  arguments, and looks code up by that specialisation and the options
  (`primitives/triton-3.8.0-jit.py:592-614, 702-724, 741-757`); the library states no fact from
  this, since the condition is assumed.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: its docstring, the scope a
  claim's `[rule]` assumption cites (condition 11, as above); its Lean definition and statement are
  unchanged. `TVA.Claim`'s module docstring restates the scope. No definition changed. Every kernel
  item whose cone reaches the library re-opens through the file digests (`vec_add_f32`'s, the only
  kernel).

## L118: the `int32` docstring moves to `I32`, and two pages say what condition 7 says

- **What changed.** `lean/TVA/Lang/Scalar.lean`: the docstring "`tl.int32` values, the type of
  `tl.program_id`, `tl.arange` and most index arithmetic." moved from `abbrev I8` to `abbrev I32`,
  and `I8` has a docstring of its own, "`tl.int8` values.". `docs/rules/compiler-faithful.md` ("What
  it says", and part (B) in "What it does not cover") and `lean/TVA/Claim.lean`'s module docstring
  now give the host process as condition 7 does: CPython 3.13 run without `-O` or `-OO` and with
  `PYTHONOPTIMIZE` unset (they said "without `-O`", and "one run with `-O`").
- **Why.** r-vecadd (F2) found the `int32` docstring on `I8`, which every claim reader shows under
  `TVA.I8`, and `I32` without one; r-vecadd2 remarked (its filing's notes) that the rule page's
  "What it says" and `TVA.Claim`'s docstring named only `-O`, where condition 7, SEMANTICS.md,
  exclusions.md and the guides name `-O`, `-OO` and `PYTHONOPTIMIZE`.
- **Evidence.** None needed: wording only. Condition 7's record is unchanged
  (evidence/host-round21 H1: a kernel rejected under `python -I` ran under `-O`, `-OO` and
  `PYTHONOPTIMIZE=1`).
- **Constants whose meaning changed.** None. `TVA.I8` and `TVA.I32` are unchanged (`IntV true 8`,
  `IntV true 32`); only their docstrings changed, now each describing its own type.
  `TVA.Rules.Compiler.Faithful`'s condition 7 did not change; the summaries in the rule page and
  `TVA.Claim`'s module docstring now match it.

