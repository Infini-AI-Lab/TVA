# Library review

Kernel reviewers trust the library without checking it ([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md)
§1), so the library is checked here, once. The library author's prompt is
[prompts/library-author.md](prompts/library-author.md) and the library reviewer's is
[prompts/library-reviewer.md](prompts/library-reviewer.md).

## What the library promises

| Part | Where | What it promises |
|---|---|---|
| semantics | `lean/TVA/Lang`, `lean/TVA/Sem`, [SEMANTICS.md](SEMANTICS.md) | what each `tl.*` operation, a launch and an outcome mean, as Triton 3.8.0 documents and implements them, and every other fact about Triton a model relies on: how the front end evaluates the source, the constants it builds, the shapes its compiler rewrites |
| transcription guidance | [TRANSCRIPTION.md](TRANSCRIPTION.md) | worked patterns, pitfalls and common reasons to drop a line, each pointing to the fact of SEMANTICS.md it rests on; it states no fact of its own and is in no digest (library issue L36) |
| digests | `tools/tvr.py`, `tools/claim_reader.lean.in`, [REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §4 | each kernel item's digest holds the sha256 of the canonical text of every constant its cone reaches (library issue L48), an assumption item's the canonical text of its statement too (L68), and, whole, every library file its cone reaches (the Lean modules, SEMANTICS.md and exclusions.md, the rule pages), so a library change re-opens exactly the items that reach it, and no two different meanings share a digest |
| compiler rule | `TVA.Rules.Compiler`, [rules/compiler-faithful.md](rules/compiler-faithful.md) | a valid, well-behaved launch behaves as its model; a launch reported to the host with an error before any program runs has no outcome |
| device facts | `TVA.Rules.Num`, [rules/num.md](rules/num.md) | per operation, format and target: how Triton lowers it, its constants, TVA's evidence; the table of rules/num.md's section "Facts the library states" lists the facts a kernel may cite, and nothing else of `TVA.Rules.Num` is citable (library issue L62) |
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
   semantic fact says what a source construct means in terms of the library's operations (the
   model text it implies); a device fact names its operation, format, target and constants. Is
   every fact about Triton that the guidance relies on stated in SEMANTICS.md?
4. **Consistency.** Every rule is satisfiable together with the others (`Platform.ref`, the
   consistency lemmas), and no stronger than its evidence.
5. **What kernels may cite.** A constant becomes citable by every kernel's `[rule]` field once
   [rules/num.md](rules/num.md)'s section "Facts the library states" lists it in its table (a name
   alone: with any arguments after `P`; a name applied, such as `DivRn P .binary64`: with exactly
   those), and the gate then accepts it with no further check (library issue L62). The gate reads
   only the section's first table, and `tvr audit` fails on a second one; it reads the section line
   by line, so a section holding anything but blank lines, table rows, `###` headings and text (a
   setext heading, a fence, an indented line, a quote, a list, HTML anywhere in a line, a `~`), a
   table whose header and delimiter rows differ in their cells (split as GFM splits them), or a page
   whose lines above the section could hide it as CommonMark reads them (a fence, an HTML block)
   lists no fact, and `tvr audit` names the page (library issue L95; protocol §2). Is
   every entry of that list
   true on the target as listed, for every argument a name alone admits, and no shape whose
   constants a field would choose?
6. **Reach.** The batch lists the constants whose meaning changed. The digests hash whole
   library files, so the items that reach a changed file re-open whatever the list says; the list
   tells reviewers what to re-read.

Above all, ask how a kernel claim that trusts the statement could be wrong.

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
4. **The coordinator** records each role folder of the round in the library's ledger
   (`tvr record library <round> <who>`, library issue L52), then the verdicts: an issue is
   `ACCEPTED` when every reviewer accepts it, and is reopened otherwise. Each new issue goes into
   `ISSUES.md`. Recorded folders are frozen, and the release check holds them to the ledger. A
   ledger shows that each recorded file is unchanged since its line, not that its lines are
   unchanged: the lines are trusted as the coordinator keeps them, with no commit at each
   recording (the user's decision, library issues L36, L82), and an edited or removed line shows
   only against a copy kept elsewhere (a commit, or a kept `release.json`, which records each
   ledger's number of lines and sha256), compared by hand; no step reads an earlier `release.json`
   back ([REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md) §3, §10).

The library is **settled** when a round accepts every issue of its batch and raises no new
`soundness` or `gap` issue. Then the kernels re-run their gates on it.

**Classes of new issues.**
- **`soundness`**: a library statement is false of Triton 3.8.0 on the target, or lets a wrong or
  vacuous kernel claim through (the gate, the digests, the release check, or guidance that leads a
  model astray: it relies on a fact about Triton that SEMANTICS.md does not state, or states
  otherwise).
- **`gap`**: the library lacks a statement kernels need, or a statement is unusable by a kernel
  reviewer, inconsistent with the rest, or weaker than it should be.
- **`wording`**: the text is unclear or wrong, but what the library means is not.

## Coverage

The library's target is every pinned tutorial kernel, except those the user excluded (issue L37).
After a batch is fixed, a coverage auditor ([prompts/library-coverage.md](prompts/library-coverage.md),
`tvr prompt library coverage <round>`) writes `round-<N>/coverage/matrix.md`. For each tutorial
function and each construct it uses, the matrix gives covered, gap (with its issue) or excluded
(with the decision). The coordinator files each new gap. The library is settled when a round
reopens nothing, raises no new soundness issue, and the matrix has no gap.

## Survey areas

| Area | What it covers | Where to read |
|---|---|---|
| `semantics` | what each `tl.*` operation, a launch, memory and an outcome mean, against Triton 3.8.0's documentation and compiler | `lean/TVA/Lang`, `lean/TVA/Sem`, `docs/SEMANTICS.md` |
| `transcription` | the transcription guidance and the facts it rests on: could a model that follows the guidance, or a drop it suggests, differ from the source? Is every fact about Triton it relies on stated in SEMANTICS.md? | `docs/TRANSCRIPTION.md`, `docs/SEMANTICS.md` ("How the front end evaluates the source"), `docs/exclusions.md` |
| `rules` | `Compiler.Faithful`, the launch conditions and the claim's shape: true, satisfiable, and what kernels need? | `lean/TVA/Claim.lean`, `lean/TVA/Rules`, `docs/rules/compiler-faithful.md` |
| `device` | the device facts and their evidence: true on the target, with TVA's own evidence? | `lean/TVA/Rules/Num*`, `docs/rules/num.md`, `evidence/` |
| `gate` | the gate, the claim reader and the release check: can a wrong, vacuous or foreign claim pass `tvr check`, or a change escape the digests? Can `tvr release`, which checks no claim and no kernel's gate (library issues L160, L167, L171; of `kernels/` and `claims/` it reads only the review records and, in the audit, the text and the kernels' Lean imports), pass a library that depends on anything it does not ship (its `shipped`, `packages` and `sandbox` steps), does not build (`build`), or fails `tvr library-check` (`library`), its records (`records`), the lint of the records not yet recorded (`lint`: the library's last round and each kernel's current round) or the audit (`audit`)? | `tools/tvr.py`, `tools/claim_reader.lean.in`, `docs/REVIEW_PROTOCOL.md` §2, §4–§8 |
| `lemmas` | the lemmas: does each statement say what its name and docstring claim, and what kernels need? | `lean/TVA/Lemmas`, `docs/LIBRARY.md` |

The coordinator gives each reviewer one area: `tvr prompt library reviewer <round> <id> --area <area>`.

## Records

```
docs/library-review/
  ISSUES.md                        the issues (coordinator)
  REPORTS.md                       what kernel review reported (generated by tvr)
  ledger.jsonl                     every recorded role folder, with each file's sha256 (tvr record)
  round-<N>/author/notes.md        a batch of fixes (library author)
  round-<N>/<reviewer>/report.md   a verdict per issue (ACCEPT or REOPEN) and new issues (library reviewer)
  round-<N>/<reviewer>/work/       the reviewer's scripts and output
evidence/<topic>/                  README.md, the script, its output (library author)
```

An issue is `OPEN`, `FIXED` (with its batch and the constants it changed), `ACCEPTED` (by a
library reviewer who did not write the fix) or `DECLINED` (with the reason; a change of scope
needs the user's decision).
