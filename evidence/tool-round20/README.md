# The tool after round 20

Library evidence (not a kernel's), for library issues L95, L106 and L108 (round 20). It tests TVA's tool
(`tools/tvr.py`, `tools/claim_reader.lean.in`), not the device: Lean 4.35.0, Python 3.13.9, 2026-10-28; no GPU
and no Triton run. Script: [run.py](run.py); output: [out.txt](out.txt) (its first line gives the sha256 of the
two tool files it ran). The round's release check (run once, from TVA's root, after the round's last library
change, `tvr check vec_add_f32` and the author's notes) is kept as [release.out](release.out) and
[release.json](release.json): 12 of 12 steps pass.

**Questions.**
1. (L95) The gate reads docs/rules/num.md's section "Facts the library states" line by line. lr-18a and lr-18b
   showed pages whose rendering differs from what such a reader takes as the facts table: a setext heading, an
   HTML comment, a fenced block. Does the gate now accept no fact, and does `tvr audit` name the page, whenever
   the section holds a form whose rendering a line reader does not know, while the tree's page, and text between
   the heading and the table, still give the 38 facts?
2. (L106) lr-18a's `inductive Assumptions … | mk (spec : …)` made the round's-start claim reader panic in
   `getStructureFields`, go on with no field, and pass the shape step, so a claim whose one assumption is its
   specification showed no assumption. Does the reader now refuse an `Assumptions` that is not a structure, and
   does the gate's shape step fail on a reader that panicked, whatever reader runs?
3. (L108) Does the tool refuse a recorded path that is not normalised (`<folder>/work/../filing.json`), in a
   `freeze` event and in any ledger event, and do the tree's ledgers pass?

**Method.**
- Part L calls `facts_section`, `stated_facts`, `rule_fact_problem` and the audit's `fact_page_problems` on the
  tree's num.md and on eleven scratch copies, each with one change: text between the heading and the table (the
  control); the real table inside an HTML comment with a fake one after it; a fenced fake table before the real
  one; the table indented by four spaces; in a block quote; in a list item; a setext heading (`Not citable` over
  `---`) before it; a rule (`***`) before it; a text line against the table that the page reads as its header; a
  second `## Facts the library states` heading; the section's heading inside a fenced block.
- Part S compiles three small kernels with `lean -o` into the scratch folder, against the library's built
  modules: `Assumptions` a structure whose `[kernel]` field is the specification (the control), lr-18a's
  one-constructor `inductive`, and a `def` whose body is the specification. On each it runs the tree's claim
  reader and the round's-start reader (the same template without the `isStructure` check, the one change round
  20 made to it), and gives each reader's output to `tvr.reader_result`, the gate's shape decision, and to the
  round's-start decision (exit 0 and one result line).
- Part F calls `is_work_file` and `freeze_problem` on vec-add's `freeze` event (event 15) with three
  non-normalised paths added (`work/../filing.json`, `work/./Cover.lean`, `//work/Cover.lean`), and
  `ledger_problems` on a scratch copy of vec-add's folder whose `freeze` also records the filing as
  `work/../filing.json` with the filing's own sha256, and on the tree's vec-add and library ledgers.

Run: `python3 -I evidence/tool-round20/run.py <an empty folder outside TVA>` from TVA's root.

**Result** (`out.txt`).

| part | case | result |
|---|---|---|
| L | the tree's num.md | 38 facts, no problem, the audit says nothing |
| L | text between the heading and the table (control) | 38 facts, `ExpF32 P` PASS |
| L | the other ten pages | no fact (`stated_facts` is `None`: a `[rule]` field citing `ExpF32 P` fails closed), and the audit names num.md, with the line and the form (HTML comment, fence, indented line, quote, list, setext heading, rule, text against the table, second heading, heading inside a fence) |
| S | a structure (control) | both readers pass; the claim's fields: `spec` |
| S | lr-18a's `inductive` | the tree's reader exits 1 (`TVR: Assumptions must be a structure (library issue L106)`), shape FAIL; the round's-start reader prints `PANIC at Lean.getStructureInfo … structure expected`, exits 0 with one result line and no field, which the round's-start decision passed and `tvr.reader_result` fails |
| S | a `def` | the tree's reader refuses it; the round's-start reader panics and exits 1; both FAIL |
| F | vec-add's freeze, as recorded | `freeze_problem`: none (the tool's own) |
| F | the three paths added | `is_work_file` false, and `freeze_problem` names each |
| F | a copy whose freeze records `work/../filing.json` | `ledger_problems` names event 15 and the path ("not a normalised path from TVA's root") |
| F | the tree's vec-add and library ledgers | no problem |

**Conclusions.**
1. The facts section may hold only blank lines, table rows, `###` headings and plain text (`facts_section`); any
   other form makes the gate accept no fact and the audit name the page, so what the gate reads as the facts
   table is what the rendered page shows (L95).
2. The reader refuses an `Assumptions` that is not a structure, and the gate's shape step fails on any reader
   output with a PANIC line (L106): with the round's-start reader too, the inductive claim fails.
3. A recorded path must be normalised; a freeze cannot record a filing through `work/..`, and the tree's ledgers
   pass (L108).
