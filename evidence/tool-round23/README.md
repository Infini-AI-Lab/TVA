# The release check after round 23

Library evidence (not a kernel's), for library issues L117 and L118 (round 23). It records the round's release
check (`python3 -I tools/tvr.py release --dir <an empty folder outside TVA and the home directory>`), run once,
from TVA's root, after the round's last library change, `tvr check vec_add_f32` and the author's notes, finishing
at 2026-10-29T14:27:50Z: Lean 4.35.0 (`leanprover/lean4:v4.35.0`), the system's Python 3.12 in the sandbox; no
GPU and no Triton run. Round 23 changed no tool file, so there is no script of its own: the command above is the
method.

**Question.** With L117's and L118's changes (condition 11 of `Compiler.Faithful`'s scope stated for the
launch's own argument values, the `I8`/`I32` docstrings, condition 7's wording in two summaries;
`lean/TVA/Rules/Compiler.lean`, `lean/TVA/Claim.lean`, `lean/TVA/Lang/Scalar.lean`,
`docs/rules/compiler-faithful.md`, `docs/SEMANTICS.md`, `docs/exclusions.md`, the guides and
`docs/TRANSCRIPTION.md`), does the shipped tree still build from source in an empty sandbox, gate `vec_add_f32`,
hold to its ledgers, lint every record in progress and pass the audit?

**Result** ([release.out](release.out), [release.json](release.json); the output writes the sandbox's folder as
`<dir>` and TVA's copy as `<tva>`). `release check PASS (12 of 12 steps pass)`, `ok: true`, 1224 shipped files.
The steps, as `release.json` names them: shipped, packages (Mathlib and the packages it pins fetched afresh
from their upstream repositories), sandbox (bwrap, no network), build (`lake build TVA`, Mathlib built from its
pinned source), library (hygiene, `leanchecker TVA`), gate vec_add_f32 (PASS, digest `5f65eb1f7a961315`),
pages, records, lint library 23 author, lint vec_add_f32 reviewer r-vecadd, lint vec_add_f32 reviewer
r-vecadd2, and audit (independent). `release.out` is the command's text output as it was saved when it ran;
its last line, `exit 0`, is the command's exit status, which the shell that ran it appended. The sandbox was
removed by the check, and the `--dir` folder deleted after `release.json` was copied here.
