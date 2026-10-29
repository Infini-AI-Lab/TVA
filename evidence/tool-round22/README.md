# The release check after round 22

Library evidence (not a kernel's), for library issue L116 (round 22). It records the round's release check
(`python3 -I tools/tvr.py release --dir <an empty folder outside TVA and the home directory>`), run once, from
TVA's root, after the round's last library change, `tvr check vec_add_f32` and the author's notes, on
2026-10-29: Lean 4.35.0 (`leanprover/lean4:v4.35.0`), the system's Python 3.12 in the sandbox; no GPU and no
Triton run. Round 22 changed no tool file, so there is no script of its own: the command above is the method.

**Question.** With L116's changes (the autotuner not verified, `@triton.autotune` outside the supported
subset, a kernel a tutorial autotunes set up with one config; `lean/TVA/Rules/Compiler.lean`,
`lean/TVA/Claim.lean`, `docs/rules/compiler-faithful.md`, `docs/SEMANTICS.md`, `docs/exclusions.md`, the guides,
`docs/REVIEW_PROTOCOL.md` and two prompts), does the shipped tree still build from source in an empty sandbox,
gate `vec_add_f32`, hold to its ledgers, lint every record in progress and pass the audit?

**Result** ([release.out](release.out), [release.json](release.json); the output writes the sandbox's folder as
`<dir>` and TVA's copy as `<tva>`). `release check PASS (12 of 12 steps pass)`, 1211 shipped files: shipped,
packages (Mathlib and the packages it pins fetched afresh from their upstream repositories), sandbox (bwrap, no
network), build (`lake build TVA`, Mathlib built from its pinned source), library (hygiene, `leanchecker TVA`),
gate `vec_add_f32` (PASS), pages, records, `lint library 22 author`, the lint of the two `vec_add_f32` reviewers'
records, and audit (independent). The command exited with status 0. The sandbox was removed, and the `--dir`
folder deleted after `release.json` was copied here.
