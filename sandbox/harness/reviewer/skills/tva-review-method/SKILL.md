---
name: tva-review-method
description: How to review a TVA kernel claim, part by part (model and map, conditions, specification, assumptions, write-up), with the question to ask of each and the finding kind for each defect. Use for every review round.
---

# Reviewing a claim (REVIEWER_GUIDE.md, condensed)

Lean has checked `main : ∀ K P, Assumptions K P → Spec K`. It cannot tell whether the model is the
kernel, whether `Post` says what the kernel is for, whether `Inst.ok` covers its real use, or whether an
assumption says what the write-up says. You judge those. Trust the library (SEMANTICS.md, the rules, the
lemmas); a doubt about it is a non-blocking `LIBRARY_REPORT`.

Work through the five parts in order. For each, write down (in `~/scratch/review-notes.md`) what you
checked and what you found; at the end turn it into filing entries.

## 1. Model and map → `MODEL_MISMATCH`, `MAP_INCOMPLETE`, `UNSUPPORTED`

`tva-kit review-brief` prints every map entry beside its source line (`--map A-B` for a part).
- For every entry: on every covered instance, does the model text do what the source line does, by
  SEMANTICS.md (`tva-kit doc <construct>`)? Check above all (TRANSCRIPTION.md §3): an IEEE `tl.add` that a
  product feeds (must be `mul_add`/`addc`); a missing conversion or broadcast; a constant in the wrong
  dtype; a different `mask`/`other`; loop bounds and carried variables; `int32` arithmetic that can wrap;
  a hint missing or misplaced.
- For every drop: does the line really change nothing, under the note's condition, on every covered
  instance? A drop that does something → `MAP_INCOMPLETE` on `map:<line>`.
- Names: read every Lean file; every operation in the model must be `TVA.*` or a kernel helper you have
  read (a kernel `def` named like a library op, an `instance`, a local `notation` → `MODEL_MISMATCH`). You
  can print with full names in a scratch file: `set_option pp.fullNames true` then `#print <decl>`.
- The launch (`launch`): grid formula, `program_id` axes, block shapes, `debug = false`, fusion.
- Classes (`tva-kit source` marks each line K/H/U): the kernel lines hold the launched kernel and every
  JIT function it runs; host lines bind nothing the kernel reads. A form outside SEMANTICS.md's
  "supported subset" (`@triton.autotune`, `break`, a module-level constant the kernel reads) →
  `UNSUPPORTED` on `src:<line>`.

## 2. Conditions → `CONDITION_TOO_NARROW`, `CONDITION_WRONG`

Do `Inst.ok` and `Pre` cover the real use (the host code's sizes, dtypes, configs, alignment; SETUP.md if
present; realistic values: NaN, infinities, large values)? Is every narrowing stated in README with a
reason (e.g. a bound `int32` indexing needs)? Are the conditions satisfiable and what README says? A
launch outside part (A) of `Compiler.Faithful`'s scope → `CONDITION_WRONG`.

## 3. Specification → `SPEC_TOO_WEAK`, `SPEC_WRONG`

Does `Post` determine or bound every output element? Is a tolerance meaningful at the **largest** sizes
`Inst.ok` allows (compute it: `python3 -I -c '…'`)? Would it reject a plausible wrong kernel (a lower
precision, a dropped term)? Does it say memory outside the outputs is unchanged?

## 4. Assumptions → `RULE_MISCITED` (on `[rule]`), `ASSUMPTION_FALSE/TOO_STRONG/UNCLEAR`, `EVIDENCE_REQUEST` (on `[kernel]`)

`tva-kit targets` lists each assumption with its kind. `[rule]`: is it the library's fact for the
operation the model uses (same function, format, target; `docs/rules/num.md`)? `Faithful` must be
`∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`. `[kernel]`: is it needed, clear, and
do the constants it constrains mean what README says?

## 5. Write-up → `DOC_WRONG` (blocking), `DOC` (non-blocking)

Does README say in words what the Lean says (specification, conditions with every narrowing,
assumptions, proof outline)? Compare every number and condition README states with its definition:
a constant, tolerance, condition, narrowing or assumption that differs (in README, or in the docstring
of a constant the claim uses) tells the reader a different claim, and is `DOC_WRONG`, which blocks
the certificate. A prover who changes a constant must change every sentence that states it. Wording
that misleads without stating a different claim is `DOC`.

## Deciding

- Accept an item only when you checked it and hold no blocking finding on it. Items: `model`, `map`,
  `conditions`, `spec`, and each `assumption:<field>`.
- A blocking finding must make the claim wrong or too weak for the kernel's real use, name the exact
  place, and say what would resolve it. Taste and wording are `DOC`; questions are `QUESTION`.
- From round 2: read `REVIEW.md` (the prover's responses). Release what is settled; re-raise only with a
  concrete reason (a second re-raise of the same finding ends the task without a certificate). Re-check
  the items whose digest changed (`tva-kit review-brief --short` marks them STALE).
