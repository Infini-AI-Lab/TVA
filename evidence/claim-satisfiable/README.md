# The claim reader's `satisfiable` check

**Question.** `Claim.Consistent` bounds only the assumptions, so a claim whose conditions cover no
instance (`Inst.ok _ := False`) passed every check of the gate with a real `consistent`
(`docs/library-review/round-10/lr-10a/work/kinds_probes.py` V0), while `Claim.Meets` holds of every
behaviour for it. Does the claim reader now refuse such a claim, and still accept a claim that has a
covered instance? Library issue L71 (round 11).

**Change.** `TVA.Claim.Satisfiable ok Pre := ∃ d init, ok d ∧ Pre d init` (`lean/TVA/Claim.lean`).
The claim reader (`tools/claim_reader.lean.in`) elaborates
`example : TVA.Claim.Satisfiable NS.Inst.ok NS.Pre := NS.satisfiable` beside the shape of `main` and
`consistent`, and reports the axioms of `satisfiable`, which the gate's `axioms` check then covers;
the gate's `shape` check fails when the reader does not elaborate (`tools/tvr.py`).

**Method.** `run.py` writes two small claims into a temporary folder, compiles each against the
built library with `lake env lean`, fills in the claim reader for it with `tools/tvr.py`'s own
`claim_reader_text`, and runs it. S1 has `Inst.ok d := d.n = 1` and a `satisfiable` proof; S2 has
`Inst.ok _ := False` and no `satisfiable` (none can be proved).

Run: `python3 -I evidence/claim-satisfiable/run.py` from TVA's root, after
`cd lean && ~/.elan/bin/lake build TVA`.

**Lean.** The toolchain of `lean/lean-toolchain`, with the library built from this tree.
**Date.** 2026-10-28. (No GPU.)

**Result** (`output.txt`).

- S1: the claim compiles; the reader elaborates and prints one result, with the axioms of `main`,
  `consistent` and `satisfiable` (none).
- S2: the claim compiles; the reader does not elaborate (`Unknown identifier RSat2.satisfiable`), so
  the gate's `shape` check fails.

**Conclusion.** A claim must now exhibit a covered instance and memory, which Lean checks; a claim
whose conditions cover nothing does not pass the gate. Whether the conditions cover the kernel's
real use remains the kernel reviewer's judgment.
