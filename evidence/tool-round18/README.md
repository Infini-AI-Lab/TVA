# The tool after round 18

Library evidence (not a kernel's), for library issues L79, L81, L94, L95 and L96 (round 18). It tests TVA's tool
(`tools/tvr.py`, `tools/claim_reader.lean.in`), not the device: Lean 4.35.0, Python 3.13, 2026-10-28; no GPU and
no Triton run. What an unused function's `def` does on the device is lr-16a's run
(`docs/library-review/round-16/lr-16a/work/unused_gpu.out`: Triton 3.8.0 on the H200 NVL, both of its kernels store
`3·x` once the unused function's default has rebound `SCALE` or `double` at import). Script: [run.py](run.py);
output: [out.txt](out.txt), made with `--semantics` the `docs/SEMANTICS.md` of the round's start (before L79's
sentence), whose sha256 `f58c4fdbc51b6a83…` is the one vec-add's accepted gate `401e1fa92db421d7` records (out.txt
prints both). The round's release check (run once, from TVA's root, after these tests) is kept as
[release.out](release.out) and [release.json](release.json): 11 of 12 steps pass, and `records` fails on vec-add's
four round-1 work files until the coordinator's `tvr freeze vec_add_f32` (round-18 author's notes).

**Questions.**
1. (L94) Does the claim reader refuse an `Assumptions` with parameters other than `(K : Claim.Behaviour Inst)
   (P : Platform)`, lr-16a's `[Inhabited Unit] [Inhabited Bool] (K …) (P …)` included, whose shifted flags let a
   `[kernel]` field that is the specification itself, and one that is a device fact, pass `kinds`?
2. (L95) Does the gate read only the first table of docs/rules/num.md's section "Facts the library states", so that
   a second table there (lr-16b's `### … (not citable)` subsection with `MathAcc` and `DivRn P .binary32`, or one
   after a line of text) adds no citable name, and does `tvr audit` name the page? Does a table that only comes
   after a `###` heading leave the gate with no fact (closed)?
3. (L79) Does the map check fail on the header of an unused function (lr-16a's `_unused_helper(f=(double :=
   triple))`, its decorators and `def` lines) that lies neither in the kernel lines nor in the host lines, pass
   when it is classed, and leave alone a function nested in another one? Does claim.md §2 list the headers?
4. (L81) Are the folders of vec-add's early `file` and `response` lines (no `dir`) frozen, so that its round-1 work
   files that no event records show? Does `tvr freeze` record them once, as a `freeze` event the records step holds
   to what the tool writes, and refuse a filing or report? Are lr-16a's routes closed: a filing whose `file` and
   `raise` lines are removed from such a folder (L1), and a note that records the filing and its report (N)? After
   `tvr check` and `tvr freeze`, as the coordinator will run them, is vec-add's certificate granted?
5. (L96) Does the release check's `pages` step compare the regenerated pages with those the copy shipped, read
   before the sandbox runs, rather than with the live tree?

**Method.** Part K compiles lr-16a's kernel with four `Assumptions` headers with `lean -o` into the scratch folder
and runs the tree's claim reader on each (all four `example`s, as the gate runs it), then the gate's axioms, usage,
namespace and kinds checks. Part L calls `stated_facts`, `facts_section_tables`, `rule_fact_problem` and the audit's
`fact_page_problems` on the tree's num.md and four scratch copies. Part U calls `def_headers`, `module_lines` and
`map_problems` on lr-16a's two sources (`round-16/lr-16a/work/probes_src.py`), on a source with a nested unused
function and a method, and on vec-add's, and renders claim.md §2 in a scratch copy. Part R pins a scratch copy's
vec-add ledger at its 14 events (as shipped, before the coordinator's freeze), runs `tvr freeze` and checks
hand-written `freeze` lines; then, at events 1–9 with a gate made by hand at r-vecadd's accepted digests (so the
certificate turns on the records alone), repeats lr-16a's L1 and N. Part P calls `shipped_records` and
`page_mismatch` on scratch copies standing for the live tree and the release check's copy: "old" is the comparison
with the live tree (round 16's), "new" with the pages the copy shipped. Part V runs the tool as the coordinator
will (`tvr check vec_add_f32`, `tvr status`, `tvr freeze vec_add_f32 --dry-run`, `tvr freeze vec_add_f32`) in a
scratch copy of the tree with vec-add's ledger at 14 events, its `lean/.lake` a copy of the tree's build and a link
to the tree's packages, once with the tree's `docs/SEMANTICS.md` and once with the given one.

**When out.txt was made** (round 20; library issue L108). out.txt was made in round 18, before the coordinator's
`tvr freeze vec_add_f32` (vec-add's event 15), on the tree as it was then. After that freeze the round-18 script no
longer re-ran: `pin_ledger` removed every file that only a later event records, the four frozen round-1 work files
too, so Part R crashed and Part V found nothing to freeze (lr-18a, lr-18b). Since round 20 `pin_ledger` keeps a file
a later `freeze` event records (it was there before the freeze), so a copy pinned at 14 events is the tree as it
was then; Parts R and V re-run after the freeze ([rerun-round20.txt](rerun-round20.txt), round 20: Part R as in
out.txt, Part V freezes the four work files again). A re-run on a later tree gives that tree's digests, not
out.txt's.

**Result** (`out.txt`).

| Case | Result |
|---|---|
| K (L94) | the control (`(K …) (P …)`) elaborates, its flags are right and `kinds` refuses both `[kernel]` fields; lr-16a's two instance parameters before `K` and `P`, and a third parameter with a default after them, now fail `shape` (the reader's error names L94); an instance parameter after them does not even elaborate `Claim.Consistent Assumptions` |
| L (L95) | the tree's num.md has one table, lists 38 facts, and the audit is silent; with lr-16b's `###` subsection, or a second table after a line of text, the list is the same 38 (`MathAcc` and `DivRn P .binary32` refused, `ExpF32 P` passing) and the audit names the page ("holds 2 tables"); a table split by a blank line lists its first part only (7 facts) and the audit names it; the only table under a `###` heading lists none (every fact refused) and the audit names the page |
| U (L79) | in lr-16a's two sources the unused functions' headers are lines 18–19 (`@triton.jit` and `def triple(x):`) and 23 (`def _unused_helper(…):`); lr-16a's manifest fails on them; it passes with them in the host lines or in the kernel lines; of a nested unused `inner` inside an unused `outer` only `outer`'s header counts, and a method's header does (a class body runs at import); vec-add's manifest (no unused function) passes; claim.md §2 lists the headers with the module lines and asks the reviewer to judge them against the supported subset |
| R (L81) | at 14 events, the records step and `kernel_record_problems` name vec-add's four round-1 work files (`work/Cover.lean`, `Print.lean`, `Spec.lean`, `offsets.py`; events 2, 5 and 7 name no folder); `tvr freeze --dry-run` lists them and writes nothing; `tvr freeze` appends one `freeze` event of `round-1/r-vecadd` with their sha256s, after which nothing is named, and a second run has nothing to freeze; hand-written lines are refused: a second freeze, a freeze of `round-2/r-vecadd` (event 11 names it as `dir`), one by `r-vecadd`, one of a folder no event records, a note naming a `dir`, a freeze recording `filing.json`. At events 1–9: the certificate waits for the four work files, and is granted after `tvr freeze`; L1: with filing-2's `file` and `raise` lines removed, `filing-2.json` and `report-2.md` are named and the certificate is not granted, and `tvr freeze` refuses them; N: `tvr note … --files` refuses them, and the same note written by hand accounts for neither |
| P (L96) | the same pages regenerated at another gate time: no mismatch either way; A, the live tree's claim.md rewritten during the run: old names it (a false failure, as in the coordinator's run of round 16), new does not; B, the copy shipped a stale claim.md that the live tree fixed during the run: old passes it, new names it |
| V (L81) | with the given SEMANTICS.md, `tvr check` passes at `401e1fa92db421d7`, the certificate is not granted (the four work files), `tvr freeze` records them, and the certificate is granted again: gate `401e1fa92db421d7`, the five items at the gate's digests, reviewers r-vecadd and r-vecadd2; with the tree's SEMANTICS.md (L79's sentence), the gate passes at `da2091f0e278a1d8`, `tvr freeze` makes the records pass, and the certificate waits for the reviewers to accept the items whose digests the changed page re-opened |

**Conclusions.**
1. `kinds` reads `K` and `P` where they are: an `Assumptions` with any other parameter fails `shape` (L94).
2. Only the first table of the facts section lists citable facts, and a page with a second fails the audit (L95).
3. The header of every unused module-level function is module code the prover classes, so what an unused `def`
   binds at import is in the reviewer's view (L79).
4. A kernel's role folder and every file in it follow from that folder's own filing or response events, never a
   note's; the work files early lines left out are recorded once by `tvr freeze`, never by editing a line, and that
   leaves the certificate as the digests and findings make it (L81).
5. The release check's pages are those the copy shipped (L96).
