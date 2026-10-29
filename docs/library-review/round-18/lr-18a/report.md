# Library review, round 18, by lr-18a

## Verdicts

- **L79** ACCEPT — `def_headers` gives, for each function named in `source.unused` that no other function
  encloses, its lines from the first decorator to the `:` before its body (read with `tokenize`), and
  `module_lines` and `map_problems` require them in the kernel or the host lines, with their own message.
  evidence/tool-round18 Part U re-ran identical: lr-16a's two sources fail on lines 18–19 and 23 and pass with them
  classed, a nested `inner` is left out and a method is counted. `work/headers.py` adds six headers: a one-line
  `def`, comments and a blank line between decorators, an `async def`, a signature over three lines with a docstring
  body, a `def` inside the `__main__` block, and a decorator that calls the function it decorates. Each header is
  found, and each passes only once classed. The gate leaves two things to the reviewer: which names a classed line
  binds, and an unused function whose decorator runs its body. Protocol §7 states both, as the user's decision on a
  malicious prover allows. REVIEWER_GUIDE §1, claim.md §2 and both kernel prompts ask the reviewer's check: the
  body of an unused function is run by nothing, and its header binds or changes no name the kernel lines read.
  lr-16b's wording point is fixed in all four places. The new sentence in SEMANTICS is true on TVA's Python 3.13.9:
  `def f(a: (x := 5) = (y := 6))` binds both names when the `def` runs. One route is still open: another module can
  rebind the name before the first launch (New issues, gap).
- **L81** ACCEPT — `own_records` gives a role folder only its own events: the `file` and `response` events whose
  recorded filing or response lies in it, then a `freeze` of it. `kernel_folder_problems` fails on every file of
  such a folder that those events do not record, whether or not a line names it as `dir`, and a note never counts.
  Part R re-ran identical to out.txt once run.py's `pin_ledger` is replaced (`work/rerun18.py`): on the tree after
  the freeze the shipped script crashes (New issues, wording). On the coordinator's two questions:
  1. A freeze cannot account for a filing or a report. `cmd_freeze` and `freeze_problem` take only files in a
     subfolder of a folder that its events leave out, only in a folder whose own events all lack `dir`, and only
     where no freeze came before. A filing's findings come only from its own `file` event. So a filing whose lines
     are removed is still named by the folder check, whatever a freeze or a note records: out.txt L1 and N show it,
     and so does `work/freeze_paths.py`. There a hand-written freeze recording `work/../filing-2.json` passes the
     line check, yet `filing-2.json` and `report-2.md` are still named.
  2. The tree's records pass for the right reason. Event 15 records exactly the four files of
     `round-1/r-vecadd/work/` that event 2 left out, with the sha256s they hold now. Their file times (14:01–14:05
     local, 2026-10-27) come before event 2 (18:06:15Z), and r-vecadd's round-1 filing and report cite all four.
     `freeze_problem` finds nothing wrong with event 15, `legacy_folders` is empty, `kernel_record_problems` names
     nothing, and the coordinator's release check of the merged tree passes `records`. §3 states the limit:
     nothing shows that the files were there at event 2, and a freeze is trusted as the ledger is.
- **L94** ACCEPT — The reader now fails unless the type of `Assumptions` has exactly two binders before its sort,
  both explicit, with types defeq to `Claim.Behaviour Inst` and `Platform`, and `numParams = 2`. Every place the
  reader reads by position (`mentions 0` and `mentions 1`, the `K` and `platform` flags, the `npar + 1` telescopes)
  then reads `K`, `P` and the structure. Part K re-ran identical: lr-16a's instance parameters, and a third
  parameter with a default, fail `shape` with the L94 error. `work/params.py` adds two cases. With `K` and `P` bound
  by `variable`, the claim passes with the right flags and `kinds` refuses both fields. With a default value on
  `K`, the claim does not compile. Protocol §5 and §8 and REVIEWER_GUIDE §4 say what the reader checks. A separate
  hole sits next to it: the reader does not require `Assumptions` to be a structure (New issues, soundness).
- **L95** ACCEPT — `stated_facts` reads only the first run of `|` lines in the section, and none if a `#…` heading
  of the section comes before it. `fact_page_problems` names a section with more than one table. Part L re-ran
  identical:
  - the tree's page lists 38 facts;
  - with lr-16b's `###` subsection, or a second table, the list stays at 38 and the audit names the page;
  - a table split by a blank line lists only its first part, and the audit names it;
  - a table under a `###` heading lists nothing.

  Protocol §2 and §5, check 5 of LIBRARY_REVIEW.md and the library author's prompt all match the code. The tree's
  page has one table, so the list is unchanged. The line reader can still take as the facts table a table that the
  rendered page shows as no table, or under another heading (New issues, gap).
- **L96** ACCEPT — `cmd_release` calls `shipped_records(copy)` right after it copies the shipped files, before the
  packages and the sandbox. The `pages` step, and each kernel's gate and certificate comparison, use what it read
  then, with the times of both gates masked. Part P re-ran identical. In its case A, a page rewritten in the live
  tree during the run no longer fails; in case B, a stale page that the copy shipped now fails. The coordinator's
  release check of the merged tree passes all 11 steps, `pages` included.
- **L97** ACCEPT — evidence/tool-round16/out.txt now agrees with the table in its README:
  - 38 facts (lines 43, 80 and 83);
  - vec-add's manifest passes Part M (line 104);
  - every cited file exists (line 139).

  Parts F, L, M, R and C re-ran with the current tool. They differ from out.txt where the README's "Re-run"
  paragraph says: the first line, Part L's three audit messages, and Part M's two messages and claim.md §2
  paragraph. Part R's R3 message also differs, reworded by L81. Since the coordinator's freeze, Part R no longer
  names vec-add's four work files. Its `pin_ledger(9)` removes them, because only event 15 records them. The
  README's "until the coordinator's freeze" allows for this.

## New issues

- **NEW** soundness — The claim reader accepts an `Assumptions` that is not a structure, and then sees no
  assumption at all. The test kernel (`work/params.py`, case IndPrivate) has:
  - `inductive Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop | intro : Claim.Meets Inst.ok Pre Post
    K → Assumptions K P`, whose one constructor holds the specification;
  - a `main` that calls a `private` lemma, and the lemma matches on `h`.

  The reader's `getStructureFields` panics on stderr ("PANIC at Lean.getStructureInfo …: structure expected") and
  returns no field. The reader then exits 0 with one result line, which `tvr check` takes as a passing `shape`.
  `axioms`, `usage`, `namespace` and `kinds` all pass. `usage` passes because `proofCone` walks only constants
  whose names start with the kernel's namespace, and a private name does not, so the match is not seen as
  destructuring; with the match in `main` itself (case IndDirect), `usage` fails. The claim is vacuous:
  `Assumptions K P` is `Spec K`. claim.md lists no assumption, while protocol §5 and §8 and REVIEWER_GUIDE §4 say
  the gate checks the kind of every assumption. Where it is:
  - `tools/claim_reader.lean.in`: its `npar` match accepts any inductive type;
  - `check` in `tools/tvr.py`: a reader that panics still passes `shape`.

  Fix (cheap and complete): the reader fails unless `isStructure env (ns ++ `Assumptions)`; `tvr check` fails
  `shape` when the reader prints a `PANIC` line; and §8's `shape` row and REVIEWER_GUIDE §4 say that `Assumptions`
  is a `structure`. By the user's decision on a malicious prover, the minimum fix is that stated limit and the
  reviewer's check.
- **NEW** gap — `facts_section_tables` reads num.md line by line, so the gate can take as the facts table one that
  the rendered page does not show as that table. A run of `|` lines counts as a table even inside an HTML comment
  or a fenced code block, and a heading underlined with `---` or `===` (setext) does not count as a heading. In
  `work/facts.py`, three variants of the page put a table with `MathAcc` and `DivRn P .binary32` in place of the
  facts table, which moves elsewhere:
  - under a setext heading `Shapes, not citable`;
  - inside `<!-- … -->`;
  - inside a fenced code block.

  In each, the gate lists 2 facts, fields citing `MathAcc P …` and `DivRn P .binary32` pass, `ExpF32 P` is
  refused, and the audit is silent: it sees one table. The same commented table placed before the real one is
  caught, since the audit then sees 2 tables. Fix (cheap): in `facts_section_tables`, treat a setext underline as a
  heading and skip fenced and commented lines, or have `fact_page_problems` fail on any fence, HTML comment or
  setext underline inside the section. Then say so in protocol §5.
- **NEW** gap — A module-level name can be rebound from another module, and nothing in the library excludes it.
  SEMANTICS "The supported subset" (Names) admits such a name only as its own `import` or `def` binds it, "and no
  other statement of the module may bind it". `Compiler.Faithful`'s scope has no condition on other modules. In
  `work/rebind_ext.py`, the kernel module `work/rebind_mod.py` stays inside the subset (its `k` calls the JIT
  helper `double`, `x * 2`). A caller in another module runs `mod.double = mod.triple` before the first launch, and
  the launch stores `3·x`. The run used GPU 0, Triton 3.8.0, torch 2.11.0+cu130 on an NVIDIA H200 NVL,
  2026-10-28. Rebinding after a first launch changes nothing: the cached kernel still stores `2·x`, with no error.
  A model taken from the source gives `2·x`. Fix: say in that sentence of SEMANTICS (and in REVIEWER_GUIDE §1's
  host check) that no code binds the name before the launch compiles, whether in the module or in another one
  that imports it; or add this to Faithful's scope as scope 2 does for knobs ("however it would be set").
- **NEW** wording — evidence/tool-round18/run.py can no longer be re-run on the tree since the coordinator's `tvr
  freeze vec_add_f32`. Its `pin_ledger(14)` removes from the scratch copy every file that only a later event
  records, and that now includes the four round-1 work files event 15 froze, though they lay in the folder all
  along. As a result:
  - Part R no longer starts "as shipped before the coordinator's freeze";
  - Part R stops with `FileNotFoundError` at its hand-written freeze lines (`work/rerun18.out`, first run);
  - Part V would start from the wrong copy too.

  With the pin keeping the files a later `freeze` records (`work/rerun18.py`), Parts R and P reproduce out.txt
  exactly. Fix: change `pin_ledger` that way, and say in the README that out.txt was made before the freeze.
- **NEW** wording — `freeze_problem` accepts a path that `tvr freeze` never writes. Its work-file test
  (`is_work_file`) only compares path text, so a hand-written freeze line recording `<folder>/work/../filing-2.json`
  passes `kernel_event_problems` and `ledger_problems` (`work/freeze_paths.py`). Nothing is gained by it: the
  folder check compares normalised paths and still names the filing and its report. But protocol §3's "holds a
  `freeze` to what the tool writes" is then not quite true. Fix: refuse a recorded path that is not normalised
  (`..`, `.`, `//`) in every event.
- **NEW** wording — Several table rows hold a `|` inside a code span: docs/rules/num.md lines 42, 44, 45, 47, 50,
  51, 53 and 110 (the facts table's `CvtF32E4m3` row: "finite `|x| ≤ maxVal`"), and docs/LIBRARY.md line 40. In
  GitHub's Markdown an unescaped `|` ends a table cell even inside a code span, so these rows render with their
  cells shifted and their last cells dropped. In the `CvtF32E4m3` row, for instance, "What Triton runs" shows `x`.
  The gate reads only the first column, so it is not affected. Fix: write `\|`.

## Re-ran

- evidence/tool-round18/run.py, Parts K, L, U, R and P (not V), on the merged tree:
  - Parts K, L and U are identical to out.txt;
  - Part R stops on the tree as it now stands (New issues);
  - with `work/rerun18.py` (run.py loaded unchanged, `pin_ledger` keeping the files a later freeze records), Parts
    R and P are identical to out.txt.
- evidence/tool-round16/run.py, Parts F, L, M, R and C (not G), with the current tool: it differs only as written
  under L97.
- New tests in `work/`, each with its output:
  - `params.py`: five `Assumptions` headers, compiled with `lean -o` into a scratch folder and read with the
    tree's claim reader and the gate's Lean-side checks;
  - `facts.py`: five scratch copies of num.md;
  - `headers.py`: six unused-function headers;
  - `freeze_paths.py`: a hand-written freeze line, on a scratch copy of the tree;
  - `rebind_ext.py` with `rebind_mod.py`: run on GPU 0 with Triton 3.8.0, torch 2.11.0+cu130, NVIDIA H200 NVL,
    2026-10-28.
- On the tree, read only:
  - `records_problems` names nothing under kernels/, and the only folders it names are this cycle's unrecorded
    round 17 and 18 reviewer folders;
  - `kernel_record_problems` and `legacy_folders` for vec-add are empty, and `freeze_problem` finds nothing wrong
    with event 15;
  - I compared the sha256s and file times of the four frozen work files;
  - `tvr audit` prints `independent` with my work files.
- I read the coordinator's release check of the merged tree: 11 of 11 steps pass, `pages` and `records` included.
  I did not use the one re-run allowed to me. The only release-check change in the batch is the `pages`
  comparison, which Part P tests and that run exercises, and a run made now would fail `lint` on this round's
  reports, which are still in progress.

## Not checked

- Part V of evidence/tool-round18: it runs `tvr check`, and so `lake build`, in a copy whose packages are linked to
  the tree's.
- Part G of evidence/tool-round16, which needs the network.
- vec-add's gate and certificate in the tree, which I was not to touch.
- A full `tvr check` of the inductive claim. The Lean-side checks ran on its compiled module with the gate's own
  functions, and the `map` and `doc` checks do not depend on the claim.
- How GitHub actually renders the num.md rows: I judged from the GFM specification (a pipe in a table cell must be
  escaped, even inside a code span) and did not run a renderer.
- The device truth of num.md's facts, which belongs to the device area.
