# Library batch, round 18: author's notes

Issues: L79, L81, L94, L95, L96, L97 ([ISSUES.md](../../ISSUES.md)), 2026-10-28. The tool and gate batch, run beside
rounds 17 (semantics and transcription) and 19 (numerics) in a separate copy of the tree, on round 14's work (in
review). The changes are in `tools/tvr.py`, `tools/claim_reader.lean.in`, the protocol and role documents
(`docs/REVIEW_PROTOCOL.md`, `docs/LIBRARY_REVIEW.md`, `docs/DESIGN.md`, `docs/REVIEWER_GUIDE.md` and
`docs/PROVER_GUIDE.md` outside their host-condition lists, `docs/prompts/`), `evidence/tool-round16/` (L97) and a new
evidence folder, `evidence/tool-round18/`. No Lean file of `lean/` changed; `lake build TVA` is green with the
standard axioms only.

**The user's decision** (user, 2026-10-28, through the coordinator): "the reviewer can challenge malicious provers
and can access the entire proof." For holes only a malicious prover can exploit, L94's (parameters before `K` and
`P`) and L79's (an unused function's header rebinding a name the kernel reads), a stated limit and a reviewer check
are enough. This round had already made a cheap complete check for each; both are kept, not extended. The protocol
says exactly what the gate does not check (§7: which names a classed module line or header binds; §5 and §8: the
reader's parameters), and REVIEWER_GUIDE gives the reviewer's check (§1: an unused function's header binds or
changes no name the kernel lines read; §4: `Assumptions` has exactly the parameters `K` and `P`). For L81 the cheap
freeze is kept; ledger lines stay trusted as the coordinator keeps them (L82), and so is a freeze.

**Files of other rounds.** One: `docs/SEMANTICS.md` (round 17's), one sentence in "The supported subset", **Names**,
after the item "At the top level of the module" (L79, as the coordinator asked). Nothing else of round 17's files: in
REVIEWER_GUIDE §1 I changed "The ranges" and "The code outside every function", not the host paragraph (L93's); in
PROVER_GUIDE only §1's "Class every function" bullet, not §2's host list.

**Kernels.** `vec_add_f32` builds and its gate passes with this round's tool. With `docs/SEMANTICS.md` as it was, its
digest is unchanged, `401e1fa92db421d7` (the claim reader's L94 check and the other changes leave every item as it
was; [evidence/tool-round18](../../../../evidence/tool-round18/README.md) Part V). With L79's sentence in
`docs/SEMANTICS.md`, which every item's digest holds whole, `tvr check vec_add_f32` in this tree passes at
`da2091f0e278a1d8`: all five items change, and `REVIEW.md` names `file:docs/SEMANTICS.md` as the only change of each
(round 17's changes to the same page re-open them after the merge too), so its reviewers re-accept them. Its records
fail after L81's fix until the coordinator's step below (its four round-1 work files); I did not edit
`kernels/vec_add_f32` or its ledger.

**For the coordinator, in the merged tree, after the merge.** From TVA's root:

```
python3 -I tools/tvr.py check vec_add_f32
python3 -I tools/tvr.py freeze vec_add_f32 --dry-run
python3 -I tools/tvr.py freeze vec_add_f32
```

The first re-gates the kernel (the gate's code changed). The second prints what the third records: the four work
files of `kernels/vec_add_f32/review/round-1/r-vecadd/work/`, `Cover.lean` (sha256 `b08e0a33251b69d4…`),
`Print.lean` (`31635598eddc2399…`), `Spec.lean` (`e3b83f24eb913c8b…`) and `offsets.py` (`c23361fbcb8f51dc…`), which
r-vecadd's round-1 report and filing cite and which its `file` line (event 2, from before the tool named folders)
left out. Check that the dry run names exactly these, then run the third: it appends one `freeze` event (event 15)
and edits no line. Afterwards the records step and the certificate's ledger condition pass. The certificate is then
granted again exactly when the reviewers' acceptances match the gate's digests: on a scratch copy with the
`docs/SEMANTICS.md` they accepted, it is granted at `401e1fa92db421d7` with both reviewers (Part V); with the merged
`docs/SEMANTICS.md` it waits for them to re-accept the items that page re-opens. No recorded finding is changed by it.

**Release check.** Run once, at the end, on this tree, from TVA's root, with these notes regular (2026-10-28;
output kept as [evidence/tool-round18/release.out](../../../../evidence/tool-round18/release.out) and
[release.json](../../../../evidence/tool-round18/release.json)): 11 of 12 steps pass (shipped, packages, sandbox,
build, library, gate vec_add_f32, pages, the lints of these notes and of vec-add's round 2, audit). The one failure
is the expected one: `records`, on vec-add's four round-1 work files, which L81's fix shows and the coordinator's
`tvr freeze vec_add_f32` above records. The `pages` step ran with L96's comparison.

## L79: gap: module-level values a JIT function reads are in no class of the map (an unused function's header)

- **What changed.** The header of every function named in `source.unused` that no other function encloses (its
  decorators and its `def` lines, from the first decorator to the `:` before its body, found with `tokenize`) is
  module code:
  - `def_headers` and `module_lines(src, unused)` (`tools/tvr.py`) add these lines to the code lines outside every
    function, and the `map` check fails on one that lies neither in the kernel lines nor in the host lines, with its
    own message (put it in `source.host`, or in the kernel lines if it binds a name the launch reads). A function
    nested in another one is left out (its `def` runs only when the outer one does), and a method's header counts
    (a class body runs at import). A function in the kernel or the host lines has its header there already.
  - claim.md §2 lists the headers with the module lines, says that a `def` evaluates its decorators, defaults and
    annotations at import, and asks the reviewer to judge that every module-level name the launch reads lies in the
    kernel lines, mapped, and is one the supported subset admits (the imports of `triton` and `triton.language as
    tl`, the module's `@triton.jit` functions, each bound only by its own `import` or `def`; any other is
    `UNSUPPORTED`).
  - Protocol §7: in "The ranges" an unused function's body is run by nothing but its `def` statement runs; "Code
    outside every function" adds the headers, `def unused(f=(double := triple))` as the example, the subset's
    module-level names, and what the gate does not check (which names a classed line binds; whether a line in the
    host lines, an unused header included, binds a name the kernel lines read: the reviewer's, by the user's
    decision above). §8's `map` row. REVIEWER_GUIDE §1 ("The ranges", "The code outside every function", with the
    reviewer's check that an unused function's header binds or changes no name the kernel lines read),
    PROVER_GUIDE §1 ("Class every function"), the kernel reviewer's prompt (step 1) and the kernel prover's prompt
    (its scope): the headers, and lr-16b's wording point, that the module-level names a launch may read are the
    imports of `triton` and `triton.language as tl` and the module's JIT functions, and that a kernel reading any
    other (a `tl.constexpr(…)` global, a dtype) is `UNSUPPORTED` (L88).
  - `docs/SEMANTICS.md` (round 17's file), "The supported subset", **Names**: one sentence after "At the top level of
    the module": each such name is admitted only as its own `import` or `def` binds it, and no other statement of the
    module may bind it (an assignment, a `:=` in a decorator, a default or an annotation of another function, a
    second `def` or `import` of the name). It says that Python evaluates decorators and defaults when it runs the
    `def`, at import, and Python 3.13 the annotations too (TVA's Triton environment is Python 3.13.9, where `def
    f(a: (x := 5) = 1)` sets `x` at the `def`; Python 3.14 defers annotations, so the sentence names 3.13).
- **Why.** lr-16a (round 16): a function in `source.unused` was exempt from classing, but its `def` statement runs
  at import; in `round-16/lr-16a/work/probes_src.py` an unused `_unused_helper(f=(double := triple))` rebinds the JIT
  helper `double` the kernel calls, `map_problems` passed, and on the H200 the kernel stores `3·x`
  (`work/unused_gpu.out`). By the user's rule (L36) the complete cheap check is to class those lines, and the
  reviewer judges the class, as for every other module line. lr-16b's wording: the guidance gave as module-level
  values to map forms the supported subset excludes.
- **Evidence.** [evidence/tool-round18](../../../../evidence/tool-round18/README.md) Part U: in both of lr-16a's
  sources the headers are lines 18–19 and 23, lr-16a's manifest fails on them, and it passes with them in the host
  lines or in the kernel lines (mapped); of an unused `inner` nested in an unused `outer`, only `outer`'s header
  counts, and a method's does; vec-add's manifest passes; claim.md §2 lists the headers.
- **Constants whose meaning changed.** None in Lean. `docs/SEMANTICS.md` changed, and it is in the digest of every
  item whose cone reaches a library module, so every such item of every kernel re-opens (vec-add's five, above).

## L81: gap: a kernel ledger shows nothing of a reviewer whose lines are removed while their folder stays (folders without `dir`, and notes)

- **What changed.** In `tools/tvr.py`:
  - **A role folder follows from its own events only.** `own_records` gives, for each role folder
    `review/round-N/<who>/`, the files its own events record: the `file` events whose filing lies in it, the
    `response` events whose response does, then a `freeze` of it; never a `note`, nor another folder's event.
    `kernel_folder_problems` counts a folder as recorded only by those, and fails on every file of a recorded folder
    that none of them records, whether or not a line names the folder as `dir` (the folder of every `file` and
    `response` event's path is frozen). A kernel's ledger no longer uses `ledger_problems`' `dir` test
    (`folders=False`), which this replaces and which counted any event's files; the library's ledger keeps it.
    `kernel_event_problems` refuses a `dir` on any event but a `file`, a `response` (its own folder) or a `freeze`.
  - **`tvr note … --files`** refuses a file of a role folder.
  - **The supported way to bring an existing ledger to a passing state: `tvr freeze <kernel> [--dry-run]`.** For each
    role folder whose own events all come from before the tool named folders (no `dir`), and which holds files they
    leave out, it appends one `freeze` event (the coordinator's) recording those files with their sha256s. It records
    only work files (in a subfolder, such as `work/`): a filing, report, response or notes left unrecorded is refused,
    since only its role's `tvr file` or `tvr respond` records it, with the events it gives. `kernel_event_problems`
    (`freeze_problem`) holds a `freeze` line to what the tool writes: the coordinator's, of such a folder, after one
    of its own events, once, of work files only that they leave out. `REVIEW.md` shows the event. Work files give no
    event, so a freeze changes no finding, acceptance or digest.
  - Protocol §2 (`records`), §3 ("Recorded files are frozen", "Every role folder follows from the events", a new
    bullet "Folders recorded before the tool named them", which states the limit: nothing shows that the frozen
    files were in the folder when its early line was written, and a freeze is trusted as the coordinator keeps the
    ledger, L82, as the user's decision above allows; "What a ledger shows"), §9 (the coordinator's notes, the
    certificate's ledger condition); DESIGN.md's ledger row; the tool's usage.
- **Why.** lr-16a's reopen: (1) a folder counted as recorded by any event naming one of its files, and only folders
  named as `dir` were frozen, so in vec-add's folders of events 2, 5 and 7 (no `dir`) a filing added later, with its
  `file` and `raise` lines removed, left the certificate granted (lr-16a's Part R, L1); (2) on a frozen folder a
  coordinator's `tvr note … --files` of the filing and its report accounted for both (N). The fix lr-16a gave; the
  freeze is the supported route the issue asks for, for vec-add's four round-1 work files, which no line can record
  otherwise without an edit by hand (a note no longer accounts for them, by (2)).
- **Evidence.** [evidence/tool-round18](../../../../evidence/tool-round18/README.md) Part R: at vec-add's 14 events
  the records step names its four round-1 work files; `tvr freeze --dry-run` lists them and writes nothing; `tvr
  freeze` appends one event, after which nothing is named, and a second run finds nothing; six hand-written lines are
  refused (a second freeze, a freeze of a folder its events name as `dir`, a freeze by a reviewer, of a folder no
  event records, recording a `filing.json`, and a note naming a `dir`). At events 1–9 with a gate at r-vecadd's
  accepted digests, the certificate waits for the work files and is granted after the freeze; lr-16a's L1 (a filing
  whose `file` and `raise` lines are removed) names `filing-2.json` and `report-2.md` and the certificate is not
  granted, and `tvr freeze` refuses them; N: `tvr note … --files` refuses them, and the same note written by hand
  accounts for neither. Part V runs the coordinator's three commands above on a scratch copy of the tree (outside it):
  with the accepted `docs/SEMANTICS.md`, the gate passes at `401e1fa92db421d7`, the certificate fails on the records,
  and after the freeze it is granted, with the same gate, items and reviewers as the one vec-add ships; with this
  round's, the gate passes at `da2091f0e278a1d8` and the freeze makes the records pass.
- **Constants whose meaning changed.** None (the tool and the protocol).

## L94: soundness: the claim reader finds `K` and `P` by position

- **What changed.** The claim reader (`tools/claim_reader.lean.in`) fails, before it reads any field, unless
  `Assumptions` has exactly two parameters, both explicit, the first of a type defeq to `TVA.Claim.Behaviour Inst`
  and the second to `TVA.Platform` (`numParams = 2`, and its type has exactly those two binders before `Prop`), with
  an error that names L94; the gate's `shape` check then fails, so `kinds` never reads a shifted binder. Protocol §8
  (`shape`) and §5 (why a `[kernel]` assumption cannot be about the device or the compiler), REVIEWER_GUIDE §4
  (`[kernel]`, with the reviewer's own check of the structure's header, by the user's decision above), the reader's
  header and the `shape` check's text say so. The check is complete for what the reader reads by position.
- **Why.** lr-16a: with `structure Assumptions [Inhabited Unit] [Inhabited Bool] (K …) (P …)` the reader's
  `xs[0]`/`xs[1]` were the instances, so a `[kernel]` field that is the specification itself (`main := h.spec`) and
  one that is a device fact false on the H200 passed every check; protocol §5 and REVIEWER_GUIDE §4 promised what the
  gate did not check.
- **Evidence.** [evidence/tool-round18](../../../../evidence/tool-round18/README.md) Part K: the control elaborates
  with the right flags and `kinds` refuses both fields; lr-16a's header, and one with a third parameter with a default
  (which `Assumptions K P` fills in), fail `shape` with the reader's L94 error; an instance parameter after `K` and `P`
  does not elaborate `Claim.Consistent Assumptions`. vec-add's gate passes unchanged (Part V, `401e1fa92db421d7`).
- **Constants whose meaning changed.** None (the gate and the protocol).

## L95: gap: `stated_facts` reads every table of the facts section

- **What changed.** `facts_section_tables` (`tools/tvr.py`) gives the tables of docs/rules/num.md's section "Facts the
  library states" (to its next `#` or `##` heading, `###` subsections included), each a run of lines that start with
  `|`, and whether a heading comes before it. `stated_facts` reads only the first table, which ends at its first
  line that is not a table row, and only if no heading of the section comes before it (else no fact: closed).
  `fact_page_problems`, in `tvr audit`, names a section with more than one table. Protocol §2 ("Fact pages") and §5,
  LIBRARY_REVIEW.md check 5 and the library author's prompt say so.
- **Why.** lr-16a and lr-16b: every table row of the section was read, so a second table, even under a
  `### … (not citable)` subsection, made `MathAcc` citable with any arguments, while protocol §5 said "the first
  column of its table".
- **Evidence.** [evidence/tool-round18](../../../../evidence/tool-round18/README.md) Part L: the tree's page has one
  table and lists 38 facts, and the audit is silent; with lr-16b's subsection or a second table after a line of text
  the list stays 38, `MathAcc P …` and `DivRn P .binary32` are refused, `ExpF32 P` passes, and the audit names the
  page; a table split by a blank line lists its first part only, and the audit names it; the only table under a
  `###` heading lists nothing, and the audit names the page.
- **Constants whose meaning changed.** None: the list the gate reads from the tree's page is the same 38 facts.

## L96: gap: the release check's `pages` step reads the live tree

- **What changed.** `cmd_release` reads, right after copying what TVA ships and before the packages and the sandbox,
  the copy's generated pages, gates and certificates (`shipped_records`). The `pages` step compares the pages the
  sandbox regenerates with those (`page_mismatch(shipped, copy)`, the gate times masked with the shipped and the
  recomputed gates' times), and each kernel's gate and certificate step uses the gates and certificates read then.
  `page_mismatch` still accepts a root for its first argument (an older evidence script calls it so). Protocol §2's
  `pages` step says so.
- **Why.** lr-16a: the step compared with `ROOT`'s pages, so a page fixed in the tree during the run passed although
  the copy shipped it stale, and one regenerated in the tree during the run failed (the coordinator's run of round
  16, with vec-add's prover's `tvr check` meanwhile).
- **Evidence.** [evidence/tool-round18](../../../../evidence/tool-round18/README.md) Part P: the same pages at another
  gate time match both ways; with the live tree's claim.md rewritten during the run, the old comparison fails and the
  new one passes; with the copy shipping a stale claim.md the live tree fixed, the old one passes and the new one
  names it. This round's release check ran with it (below).
- **Constants whose meaning changed.** None (the release check).

## L97: wording: evidence/tool-round16's output predates the protocol's fix

- **What changed.** [evidence/tool-round16/out.txt](../../../../evidence/tool-round16/out.txt) is regenerated on the
  tree after rounds 12 to 16 were merged, with the round-16 tool (`tools/tvr.py` sha256 `2a390f9dca6f5ae1`, its first
  line), before this round changed the tool. Two changes to its `run.py` keep its cases as round 16 tested them: Part
  R pins the scratch copy's vec-add ledger at events 1–9 (`pin_ledger`; the kernel's review went on to 14 events, with
  a second filing of r-vecadd in round 2, so the old script overwrote a recorded `filing-2.json`), and Part M prints
  vec-add's ranges as shipped. Its README has a "Re-run" paragraph, and its table now says what the output does:
  num.md lists 38 facts (round 14 added the six `CvtSat…`), vec-add's manifest passes Part M (its prover classed the
  lines), and the tree cites no missing file.
- **Why.** lr-16a and lr-16b: `out.txt` line 139 named a `docs/DESIGN` citation the protocol no longer has, against a
  README that says the tree cites no missing file, and Part M's vec-add line was stale.
- **Evidence.** The new `out.txt` against the first: lines 43, 80 and 83 (38 facts), 104 (vec-add passes, with its
  ranges) and 139 ("every cited file exists") differ, and nothing else. A run with this round's tool (checked for
  Parts F, L, M, R and C) differs in its first line, in Part L's audit message for a page with no usable table (L95),
  in Part M's reworded messages and claim.md §2 paragraph (L79), and in Part R, which names vec-add's four round-1
  work files until the coordinator's freeze (L81); Parts F and C are the same. tool-round16's README says so.
- **Constants whose meaning changed.** None (evidence only).

