# Library review, round 18, by lr-18b

## Verdicts

- **L79** ACCEPT — The header of every function named in `source.unused` that no other function encloses (its
  decorators and `def` lines up to the `:` before its body, found with `tokenize`) is now module code that the map
  check holds to the kernel or the host lines (`def_headers`, `module_lines`, `map_problems`). A function nested in
  another one is left out, and a method's header counts. Part U re-ran identical: lr-16a's two sources fail on lines
  18–19 and 23 and pass once those lines are classed; the nested and method cases behave as stated; vec-add passes.
  work/headers_probe.py adds forms Part U lacks: a decorator over three lines, a default on a continuation line, a
  return annotation, a one-line body, an `async def`, and `def`s inside a module-level `if` and `try`. Every header
  line is taken (10–16, 20, 23, 28, 32), and the `if` and `try` lines are module code. On TVA's Python 3.13.9, a
  `:=` in the decorator, a default, an annotation or the return annotation of a `def` that is never called rebinds
  the module-level name at import, and one in its body does not. That is what the new SEMANTICS sentence ("Names")
  says. Protocol §7 and §8, claim.md §2, REVIEWER_GUIDE §1, PROVER_GUIDE §1 and both kernel prompts give the stated
  limit (the gate checks a line's class, not what it binds) and the reviewer's check, as the user decided for a
  malicious prover. lr-16b's wording point is fixed too: the module-level names a launch may read are `triton`,
  `triton.language as tl` and the module's JIT functions, and reading any other is `UNSUPPORTED`. A rebinding from
  outside the module's own statements is a separate defect (New issues).
- **L81** ACCEPT — A role folder is now accounted for only by its own events (`own_records`): the `file` and
  `response` events whose path lies in it, then a `freeze` of it, and never a `note`. Every folder those events
  record is frozen, whether or not a line names it as `dir` (`kernel_folder_problems`). `kernel_event_problems`
  refuses a `dir` on any other event, and `tvr note --files` refuses a file of a role folder. A freeze cannot
  account for a filing or a report: `cmd_freeze` dies on any unrecorded file at a folder's top, and
  `freeze_problem` refuses a hand-written freeze that records one (`is_work_file`). It also refuses a freeze of a
  folder whose events name a `dir`, a second freeze, one by another actor, and one before the folder's own event.
  Part R no longer re-runs on the tree after event 15 (New issues, wording). On the shipped ledger,
  work/freeze_probe.py puts nine hand-written freezes to the tool's `freeze_problem`. Eight are refused: a second
  freeze, a freeze of a folder whose events name `dir`, one by a reviewer, one of a folder that no event records,
  and ones recording `notes.md`, a `response-2.json` or another folder's filing. Only a work file of the still
  legacy `round-1/prover` is accepted, and it gives no event. In the merged tree the records step passes for the
  right reason. Event 15, the coordinator's
  `tvr freeze`, comes after event 2 and records exactly `work/Cover.lean`, `Print.lean`, `Spec.lean` and
  `offsets.py` with their current sha256s, and `freeze_problem` accepts it. Every other role folder's files are
  recorded by that folder's own events. r-vecadd's round-1 report and filing cite all four files. Their times
  (14:01–14:05) come before the filing's (14:05:50) and event 2 (18:06:15Z). `kernel_record_problems`,
  `unrecorded_current` and `legacy_folders` are empty, and the coordinator's release check of the merged tree passes
  `records`. What remains is the limit protocol §3 states (L82): a freeze is trusted as the coordinator keeps the
  ledger, and it stays available for a legacy folder not frozen yet (`round-1/prover`) until the coordinator judges
  any file there.
- **L94** ACCEPT — Before it reads any field, the claim reader now fails unless `Assumptions` is an inductive type
  with `numParams = 2` whose type has exactly two binders, both explicit and defeq to `Claim.Behaviour Inst` and
  `TVA.Platform`, in this order. The gate's `shape` check then fails, so `kinds` never reads a shifted binder. A
  structure has no indices, so these are the only binders before each projection's `self`. `xs[0]` and `xs[1]` are
  therefore `K` and `P`, and the check is complete for what the reader reads by position. A `def` or `abbrev` is
  refused as not a structure. Part K re-ran identical: the control's flags are right and `kinds` refuses both
  `[kernel]` fields; lr-16a's header and a defaulted third parameter fail `shape` with the L94 error; a trailing
  instance parameter does not elaborate. Protocol §5 and §8 and REVIEWER_GUIDE §4 state the check and the
  reviewer's reading of the header. vec-add's gate passes with the new reader: at `721f1db52c61b7ef` in my Part V
  run and in the coordinator's release check.
- **L95** REOPEN — For `#` headings, the gate reads only the section's first table and the audit names a second;
  Part L re-ran identical. But the tool reads the section as lines, not as Markdown renders it, and Markdown's other
  heading form brings the issue's harm back. In work/facts_probe.py S2, a setext heading (the line `Shapes the facts
  use (not citable)` over a line of `-`, which renders as a `##` heading) sits over a table of `MathAcc` and `DivRn
  P .binary32`, followed by `## The facts (citable)` over the real table. The gate then lists 2 facts:
  `MathAcc P …` passes and `ExpF32 P` is refused. The audit is silent, and the rendered page shows the opposite of
  what the gate reads. Protocol §5 says the table "must come before any heading of the section", which is false
  for this heading. A table inside an HTML comment (C1, not rendered) or a fenced code block (B1, rendered as code)
  is likewise read as the facts, and the audit is silent. Fix (cheap): `facts_section_tables` takes a setext
  underline (a line of only `=` or `-` under a non-blank line that is not a table row) as a heading, both to end the
  section and for `under_heading`. `fact_page_problems` refuses an HTML comment and a fenced or indented code block
  in the section; or protocol §2 and §5 say that the gate reads `#` headings and `|` lines only, whatever the page
  renders.
- **L96** ACCEPT — `cmd_release` now reads `shipped_records(copy)` right after it copies the shipped files, before the
  packages and the sandbox. The gate, certificate and pages steps compare with what it read, and `page_mismatch`
  masks both the shipped and the recomputed gate's time. Part P re-ran identical. In case A, the old comparison
  fails falsely and the new one passes. In case B, the old one passes a stale shipped page and the new one names
  it. Protocol §2's `pages` bullet says what the code does. The coordinator's release check of the merged tree ran
  with it: 11 of 11 steps pass, `pages` included.
- **L97** ACCEPT — The regenerated out.txt differs from lr-16b's round-16 re-run (the first out.txt with line 139
  already fixed) at lines 43, 80, 83 (38 facts) and 104 (vec-add's ranges), and nowhere else. Line 139 reads
  "every cited file exists". The README's table now says what the output shows, and its "Re-run" paragraph gives
  the tool's sha256 and both changes to `run.py`. I re-ran Parts F, L, M, R and C with this round's tool on the
  merged tree. They differ from out.txt only where the README says: the first line, Part L's three audit messages,
  and Part M's messages and §2 paragraph. One more line differs: Part R's R3 message, reworded by L81 ("no filing or
  response of … records"), which the README does not name (trivial). After the freeze, Part R names none of
  vec-add's four work files, because `pin_ledger(9)` removes them (only event 15 records them). That fits "until the
  coordinator's `tvr freeze`".

## New issues

- **NEW** soundness — `Compiler.Faithful`'s scope, SEMANTICS "The supported subset" and protocol §7 assume that
  Triton's modules and objects, and the names of the kernel's module, are as Triton 3.8.0 and the kernel's source
  make them. Nothing states this, and host code outside the kernel's file changes them with no knob, hook, option,
  decorator or statement of the kernel's module. SEMANTICS "Names" says only that "no other statement of the
  module" may bind such a name, and protocol §7 says the mapped kernel lines cover "every module-level name it
  reads". In work/host_mutation.py (GPU 0, an H200 NVL, Triton 3.8.0, 2026-10-28), every line of the kernel module
  is in the supported subset, and the kernel `k` stores `double(x)` (the helper is `x * 2`). Each case below is run
  from another module:
  - R1, `kmod.double = kmod.triple`: the kernel stored `3·x`;
  - R2, `kmod.k.fn.__globals__["double"] = kmod.triple`: the kernel stored `3·x`;
  - T1, `tl.exp = tl.log`: a kernel's `tl.exp(x)` stored `log x`;
  - S1, Triton's own method `kmod.k._unsafe_update_src(…)` (`triton-3.8.0-jit.py:561-569`): the kernel stored `5·x`.

  A reviewer who models the source by SEMANTICS accepts `2·x`, and `Faithful` then promises what the device does
  not do. Fix (a stated limit and the reviewer's check, as the user decided for a malicious prover): add a scope
  point, "Triton and the kernel's module as installed and written". Triton's files are 3.8.0's, unmodified. No host
  code assigns or replaces an attribute of a Triton module, class or object (`tl.*`, a `JITFunction`'s `fn`,
  `debug`, `noinline` or `_unsafe_update_src`; `preload`, next item). Nothing outside the module's own statements
  binds a name of the kernel's module. Say "no statement of the module or of any other code" in SEMANTICS "Names",
  qualify protocol §7's "covers", and add the point to REVIEWER_GUIDE §1's host list and to compiler-faithful.md's
  scope and checklist.
- **NEW** soundness — Two routes set the compile options of a plain JIT launch, and neither `Launch.debug`'s
  docstring nor `Faithful`'s scope 4 names them.
  - D1: the kernel's `JITFunction` attribute `debug` (`kernel.debug = True`). `JITFunction.run` takes
    `kwargs.get("debug", self.debug) or knobs.runtime.debug` (`triton-3.8.0-jit.py:727`). With int32 `a + b`
    overflowing in lane 0, the launch with the attribute set trapped (device-side assert, "int32 overflow detected
    for operation add"). The control D0 wrapped to `-2^31` (work/host_mutation.py). `Launch.debug`'s docstring says
    `debug` is on when "the launch, the decorator, `TRITON_DEBUG=1` or the knob" asks for it, and "`False`
    otherwise". That is false here: a reviewer who follows it models `debug := false`, and `Faithful` promises no
    fault.
  - P: `JITFunction.preload(data)` compiles with the options its JSON names and stores the kernel at the key its
    JSON names (`triton-3.8.0-jit.py:824-875, 902-903`), and a later plain `k[grid](…)` with no option runs that
    kernel. With a specialization compiled under `enable_reflect_ftz=False` and preloaded under the plain launch's
    key, the plain launch stored `exp2(-130) = 2^-130`. The control stored `0`, the model's
    (work/preload_probe.py). Scope 4 names only the launch, `warmup`, `triton.compile`, a `triton.Config` and
    `@triton.heuristics`.

  Fix: name the attribute in `Launch.debug`'s docstring, along with `warmup`'s and `triton.compile`'s `debug`. Add
  `preload` to scope 4: no `preload`, or only data that a plain launch of the claimed configuration wrote. Add both to
  REVIEWER_GUIDE §1's and compiler-faithful.md's host checklists.
- **NEW** wording — evidence/tool-round18/run.py no longer re-runs on the tree after the coordinator's freeze (event
  15). `pin_ledger(14)` removes every file that only a later event records, so it deletes the four round-1 work
  files, which were in the folder at event 14. Part R then stops with a FileNotFoundError on `work/Cover.lean`
  (work/rerun-tool-round18-klur.err). Part V prints "nothing to freeze", with the gate at `721f1db52c61b7ef`
  (work/rerun-tool-round18-pv.out). So neither part tests `tvr freeze` any more, and the README does not say that
  out.txt needs the tree at 14 events. Fix: have `pin_ledger` keep the files a dropped `freeze` event records, or
  say in the README that Parts R and V re-run only on a tree before event 15.

## Re-ran

- evidence/tool-round18/run.py on the merged tree (`tools/tvr.py` sha256 `38cde42e17f4b038`, the claim reader
  `8e1888db2fbf5319`). Parts K, L, U and P are identical to out.txt. Part R is identical until the freeze cases, then
  differs and stops (New issues). Part V ran once, with the tree's SEMANTICS.md (no `--semantics`: the round's
  starting page is not in the tree). Output in work/rerun-tool-round18-klur.out, rerun-tool-round18-pv.out and the
  masked traceback rerun-tool-round18-klur.err.
- evidence/tool-round16/run.py, Parts F, L, M, R and C, with this round's tool (work/rerun-tool-round16-flmrc.out).
  Its out.txt was also compared with lr-16b's round-16 re-run (L97).
- vec-add's shipped ledger in the tree: `own_records`, `freeze_problem` on event 15, `kernel_record_problems`,
  `unrecorded_current`, `legacy_folders`, and the files' times against the filing's. New, work/freeze_probe.py:
  nine hand-written freezes after event 15, judged in memory by `freeze_problem`.
- New, work/headers_probe.py (Python 3.13.9): the L79 header forms, and what a `def` evaluates at import.
- New, work/facts_probe.py: num.md's facts section with a setext heading (S1, S2), an HTML comment (C1), a fenced
  table (B1) and a fenced `#` line (H1), through `facts_section_tables`, `stated_facts`, `rule_fact_problem` and
  `fact_page_problems`.
- New, work/host_mutation.py, GPU 0 (H200 NVL, Triton 3.8.0, torch 2.11.0+cu130, 2026-10-28): R1, R2, D0, D1, T1
  and S1, each in its own process.
- New, work/preload_probe.py, GPU 0, same versions: generator G, preload P and control C, each in a fresh process
  with a fresh Triton cache.
- Read, not re-run: the coordinator's release check of the merged tree (11 of 11 steps pass).
- Read against the code: `claim_reader.lean.in`'s L94 block and field flags; in `tools/tvr.py`, `check`,
  `def_headers`, `module_lines`, `map_problems`, `own_records`, `freeze_problem`, `kernel_folder_problems`,
  `kernel_event_problems`, `cmd_note`, `cmd_freeze`, `facts_section_tables`, `stated_facts`, `fact_page_problems`,
  `shipped_records`, `page_mismatch` and `cmd_release`. Also `TVA.Claim`, `Rules.Compiler.Faithful`,
  `Sem.Launch`'s `Launch` fields and `Valid`, and compiler-faithful.md, against `triton-3.8.0-jit.py`,
  `code-generator.py:370-429` and `compiler.py:493-502`.

## Not checked

- The release check itself (lr-18a's to repeat), and Part G of tool-round16 (it needs the network).
- Part V of tool-round18 with the round's starting SEMANTICS.md: the certificate granted again at
  `401e1fa92db421d7` rests on the author's out.txt.
- A compiled kernel whose `Assumptions` is an inductive type with two parameters but no structure: I judged it
  from the reader's code (no fields, then the `usage` check's `destructured`), without a run.
- A Triton compile cache whose files were changed by hand (`TRITON_CACHE_DIR` is one of scope 2's exceptions).
- The device truth of num.md's facts (the device area), and the in-scope tutorials' hosts against the proposed scope
  point.
