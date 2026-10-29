# The tool after round 21

Library evidence (not a kernel's), for library issues L95 and L112 (round 21). It tests TVA's tool
(`tools/tvr.py`, `tools/claim_reader.lean.in`), not the device: Lean 4.35.0, Python 3.13.9, 2026-10-29; no GPU
and no Triton run. Script: [run.py](run.py); output: [out.txt](out.txt) (its first line gives the sha256 of the
two tool files it ran). The round's release check (run once, from TVA's root, after the round's last library
change, `tvr check vec_add_f32` and the author's notes) is kept as [release.out](release.out) and
[release.json](release.json).

**Questions.**
1. (L95) The gate reads docs/rules/num.md's section "Facts the library states" line by line. lr-20c showed pages
   on which the round's-start tool still read the 38 facts, with the audit silent, while a CommonMark or GFM
   renderer shows the section as code, or not at all, or the fact in another column
   (`docs/library-review/round-20/lr-20c/work/facts_probe.out`): a fence of four backticks with a three-backtick
   line inside, a fence with a ```` ```python ```` line inside, a line ```` ```a`b ```` before a real fence,
   `<script>` or `<pre>` before the heading closed after the next `##` heading, a row starting `||`, inline HTML
   (`<details>`, `<sub>`), and a struck-through first cell. Does the gate now accept no fact, and does `tvr audit`
   name the page, on each of these, and on a header row wider than its delimiter row, a list item's fence and a
   fence indented by one space; and does a row starting `||` give its first cell as GFM does (empty)?
2. (L112) Does the claim reader refuse a `private` field of `Assumptions` with an error that names it (it
   listed the field and read nothing of it, so `kinds` said "not tagged"), and does the shape step pass a claim
   whose field docstring holds the word PANIC (it took the result line for a panic) while still failing on a
   panic line of Lean's?

**Method.**
- Part L calls `facts_section`, `stated_facts`, `rule_fact_problem` and the audit's `fact_page_problems` on the
  tree's num.md and on fourteen scratch copies, each with one change: two controls (a line of text between the
  heading and the table; a closed `~~~` fence and a closed ```` ````text ```` fence before the section), lr-20c's
  nine forms, a header row with one cell more, a list item `- item` with a fence indented by two spaces followed
  by a ```` ``` ```` line at the margin opening a fence around the section, and a fence indented by one space
  around the section.
- Part R compiles three small kernels with `lean -o` into the scratch folder, against the library's built
  modules (a public `[kernel]` field, the control; lr-20c's `private` field; a field docstring holding PANIC),
  runs the tree's claim reader on each, and gives its output to `tvr.reader_result` (the gate's shape step),
  `tvr.kinds_problems` and the usage decision. It also gives `tvr.reader_result` three made-up outputs: a
  result line holding PANIC alone, and with a line `PANIC at …` of Lean's on stderr or on stdout.

Run: `python3 -I evidence/tool-round21/run.py <an empty folder outside TVA>` from TVA's root.

**Result** (`out.txt`).

| part | case | result |
|---|---|---|
| L | the tree's num.md | 38 facts, no problem, the audit says nothing |
| L | the two controls | 38 facts, `ExpF32 P` PASS, the audit says nothing |
| L | lr-20c's three fences | no fact; "the section's heading lies inside a fenced block"; the audit names num.md |
| L | `<script>`, `<pre>` before the heading | no fact; "a line before the section that starts with `<`"; the audit names num.md |
| L | a row starting `\|\|` | 38 facts, the row's first cell empty: `DivRn P .binary32` is refused, as the page shows it under no fact |
| L | `<details>` in a text line, `<sub>` in a cell | no fact; "HTML (a `<` before a letter, `/`, `!` or `?`)"; the audit names num.md |
| L | a struck-through first cell | no fact; "a `~`"; the audit names num.md |
| L | a header row wider than its delimiter row | no fact; "different numbers of cells, so GFM shows no table" |
| L | a list item's fence, then a fence at the margin | no fact; "a fence indented by one to three spaces", and the heading inside a fence |
| L | a fence indented by one space | no fact; "a fence indented by one to three spaces" |
| R | a public field (control) | the reader passes; `kinds` reads the field (a `[kernel]` field that mentions `K`: refused, as it should be); usage PASS |
| R | a `private` field | the reader exits 1: "TVR: the field spec of Assumptions has no projection Kernels.Demo.Assumptions.spec (a private field?); every field must be public (library issue L112)"; shape FAIL |
| R | PANIC in a field's docstring | shape PASS (round 20's tool: "the claim reader panicked") |
| R | made-up outputs | a result line holding PANIC alone: PASS; with a `PANIC at …` line on stderr or on stdout: FAIL, "the claim reader panicked" |

**Conclusions.**
1. Before the section, the gate reads fences as CommonMark does at the top level of the page (a closer of the
   opener's character, at least as long, with nothing after it; no opener whose backtick info string holds a
   backtick), refuses a line that starts with `<` (an HTML block) and a fence indented by one to three spaces
   (which may belong to a list item); in the section it refuses HTML anywhere in a line and a `~`, and a table
   whose header and delimiter rows differ in their cells; and it splits rows as GFM does. On every form lr-20c
   found, and the four added, the gate accepts no fact and the audit names the page, or the gate reads the row
   as the page shows it (library issue L95).
2. The claim reader refuses a `private` field by name, so no field is listed that the reader did not read; the
   shape step takes PANIC as a panic only on a line other than the reader's result line, which quotes the claim's
   names and docstrings (library issue L112).
