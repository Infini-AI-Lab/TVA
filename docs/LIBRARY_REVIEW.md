# Library review

Kernel reviewers trust the library without checking it ([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md)
§1), so the library is checked here, once. The library author's prompt is
[prompts/library-author.md](prompts/library-author.md) and the library reviewer's is
[prompts/library-reviewer.md](prompts/library-reviewer.md).

## What the library promises

| Part | Where | What it promises |
|---|---|---|
| semantics | `lean/TVA/Lang`, `lean/TVA/Sem`, [SEMANTICS.md](SEMANTICS.md) | what each `tl.*` operation, a launch and an outcome mean, as Triton 3.8.0 documents and implements them |
| transcription rules | [TRANSCRIPTION.md](TRANSCRIPTION.md) | for each source construct, the model text it becomes, or why it may be dropped; each rule has an identity, a kernel's map entries name the rules they apply, and the gate folds those rules' text into the kernel's `map` and `model` digests |
| compiler rule | `TVA.Rules.Compiler`, [rules/compiler-faithful.md](rules/compiler-faithful.md) | a valid, well-behaved launch behaves as its model; a launch reported to the host with an error before any program runs has no outcome |
| device facts | `TVA.Rules.Num`, [rules/num.md](rules/num.md) | per operation, format and target: how Triton lowers it, its constants, TVA's evidence |
| lemmas | `lean/TVA/Lemmas` | proved; Lean checks them |
| exclusions | [exclusions.md](exclusions.md) | what is out of scope, and why |

## What a library reviewer checks

For each issue of a batch (`docs/library-review/ISSUES.md`):
1. **Truth.** Is the statement true of Triton 3.8.0 on the target? Check it against the
   documentation, the compiler (the pinned sources in `primitives/`, listed in
   `primitives/semantics-manifest.json`) and the device. Re-run the evidence, and add tests where it does
   not settle the question.
2. **Evidence.** Is it TVA's own, with Triton's version, the GPU and the date? Is it exhaustive
   where the input space allows (every `float32` input of a unary function), and adversarial
   elsewhere? Another project's measurement is not evidence.
3. **Usability.** Can a kernel reviewer apply the statement without compiler knowledge? A
   transcription rule names the model text; a device fact names its operation, format, target
   and constants.
4. **Consistency.** Every rule is satisfiable together with the others (`Platform.ref`, the
   consistency lemmas), and no stronger than its evidence.
5. **Reach.** The batch lists the constants whose meaning changed, so the kernel items that
   reach them re-open.

## Rounds

The library improves in rounds, each a batch of issues:
1. **The coordinator** sets the batch: every issue reopened or raised in the last round (except
   `wording`, which may wait), and any older open issues that are ready. Each gets the round's
   number in `ISSUES.md`.
2. **A fresh library author** fixes the whole batch at once and writes the notes
   ([prompts/library-author.md](prompts/library-author.md)).
3. **Two fresh library reviewers** judge every issue of the batch independently. Each also
   surveys one area of the library (below) for new defects. The areas rotate from round to round,
   so every area is surveyed again after the library has changed.
4. **The coordinator** records the verdicts: an issue is `ACCEPTED` when every reviewer accepts
   it, and is reopened otherwise. Each new issue goes into `ISSUES.md`.

The library is **settled** when a round accepts every issue of its batch and raises no new
`soundness` or `gap` issue. Then the kernels re-run their gates on it.

**Classes of new issues.**
- **`soundness`**: a library statement is false of Triton 3.8.0 on the target, or lets a wrong or
  vacuous kernel claim through (the gate, the digests, the transcription rules).
- **`gap`**: the library lacks a statement kernels need, or a statement is unusable by a kernel
  reviewer, inconsistent with the rest, or weaker than it should be.
- **`wording`**: the text is unclear or wrong, but what the library means is not.

## Survey areas

| Area | What it covers | Where to read |
|---|---|---|
| `semantics` | what each `tl.*` operation, a launch, memory and an outcome mean, against Triton 3.8.0's documentation and compiler | `lean/TVA/Lang`, `lean/TVA/Sem`, `docs/SEMANTICS.md` |
| `transcription` | the transcription rules and the drops: could a model that follows them differ from the source? | `docs/TRANSCRIPTION.md`, `docs/exclusions.md`, the drop list in `tools/tvr.py` |
| `rules` | `Compiler.Faithful`, the launch conditions and the claim's shape: true, satisfiable, and what kernels need? | `lean/TVA/Claim.lean`, `lean/TVA/Rules`, `docs/rules/compiler-faithful.md` |
| `device` | the device facts and their evidence: true on the target, with TVA's own evidence? | `lean/TVA/Rules/Num*`, `docs/rules/num.md`, `evidence/` |
| `gate` | the gate and the claim reader: can a wrong, vacuous or foreign claim pass `tvr check`, or a change escape the digests? | `tools/tvr.py`, `tools/claim_reader.lean.in`, `docs/REVIEW_PROTOCOL.md` §4–§8 |
| `lemmas` | the lemmas: does each statement say what its name and docstring claim, and what kernels need? | `lean/TVA/Lemmas`, `docs/LIBRARY.md` |

The coordinator gives each reviewer one area: `tvr prompt library reviewer <round> <id> --area <area>`.

## Records

```
docs/library-review/
  ISSUES.md                        the issues (coordinator)
  REPORTS.md                       what kernel review reported (generated by tvr)
  round-<N>/author/notes.md        a batch of fixes (library author)
  round-<N>/<reviewer>/report.md   a verdict per issue (ACCEPT or REOPEN) and new issues (library reviewer)
  round-<N>/<reviewer>/work/       the reviewer's scripts and output
evidence/<topic>/                  README.md, the script, its output (library author)
```

An issue is `OPEN`, `FIXED` (with its batch and the constants it changed), `ACCEPTED` (by a
library reviewer who did not write the fix) or `DECLINED` (with the reason; a change of scope
needs the user's decision).
