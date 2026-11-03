---
name: tva-gate
description: Run TVA's gate (tvr check) on the kernel, fix each failing check, and submit to the review service. Use at stage 10, before every submission, and whenever tva-kit gate prints FAIL.
---

# The gate, and submitting

`tva-kit gate` runs `tvr check` and prints each check with what to do when it fails. The review service
runs the same gate on your files; a submission whose gate fails is rejected (it costs no round, but the
rejections are limited).

| Check | What it wants | Typical fix |
|---|---|---|
| manifest | kernel.json's fixed fields untouched; ranges/host are `[first, last]` pairs | `tva-kit classify --write` |
| source | the source unchanged | never edit `source/` |
| hygiene | no `sorry`, `admit`, `#eval`, `#print`, `run_cmd`, `notation`, `macro`, `instance`, `axiom`, `set_option` beyond limits, `open … renaming`, strings with `{` | delete the construct named |
| independence | imports only `TVA`, `Mathlib.*`, your own modules; no other project named | fix the import |
| build | the Lean builds | `tva-kit build`, first error first (skill tva-proof) |
| replay | leanchecker re-checks | follows the build |
| shape | `main`, `consistent`, `satisfiable` with exactly the statements of the example; `Spec K := Claim.Meets Inst.ok Pre Post K`; `structure Assumptions (K : Claim.Behaviour Inst) (P : Platform)` | copy the example's statements verbatim |
| axioms | only propext, Classical.choice, Quot.sound | no `sorry`, `native_decide`, `axiom` |
| usage | every Assumptions field used by `main` | remove unused fields |
| namespace | everything in your namespace; no instance; no name a library constant also has; ASCII names | rename (`myAdd`), move into the namespace |
| kinds | each field `[rule]` (Faithful, or a listed `Rules.Num` fact applied to `P`) or `[kernel]` (closed, no `K`/`P`) | skill tva-claim |
| map | every code line of the ranges mapped; every model text found in the Lean; every function classed | `tva-kit map-check`, `tva-kit map-skeleton` |
| doc | README.md has Specification, Conditions, Assumptions, Proof | fill the sections |

## Submitting

1. `tva-kit gate` prints `gate PASS  digest …`.
2. `tva-review submit` (it sends lean/, README.md, model-map.json, kernel.json, and from round 2 your
   response folder).
3. `tva-review wait` (≤ 9 minutes per call); run it again while it says the round is under review. A
   round can take an hour.
4. Rejected? `tva-review status` and `tva-review log` show why (the service's gate output). Fix and
   submit again.
5. Findings? Skill tva-respond. Certificate GRANTED? You are done.

Before submitting, read your README against the Lean once: reviewers file `DOC` findings for a write-up
that says something the Lean does not.
