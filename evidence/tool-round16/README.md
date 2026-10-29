# The tool after round 16

Library evidence (not a kernel's), for library issues L7, L62, L79, L81 and L82 (round 16). It tests TVA's tool
(`tools/tvr.py`, `tools/claim_reader.lean.in`), not the device: Lean 4.35.0, git 2.52 (`/usr/bin/git`), Python
3.13, 2026-10-28; no GPU and no Triton run. What the front end reads at module level, on the device, is lr-13a's
run (`docs/library-review/round-13/lr-13a/work/globals.py`: Triton 3.8.0 on the H200 NVL stores `x·3` rounded to
`float16` from a module-level `SCALE = tl.constexpr(3)` and `OUT_T = tl.float16`). Part G needs the network (it
reads Mathlib's upstream refs and the history of its `master`). Script: [run.py](run.py); output:
[out.txt](out.txt). The round's release check (run once, from TVA's root, after these tests) is kept as
[release.out](release.out) and [release.json](release.json), and summed up in the round-16 author's notes.

**Re-run** (library issue L97, round 18, 2026-10-28). The first `out.txt` was written before protocol §2 stopped
citing `docs/DESIGN` without its `.md`, and vec-add's prover has since classed its module lines, so two of its lines
no longer matched this README. It was regenerated on the tree after rounds 12 to 16 were merged, with the same tool
(`tools/tvr.py` sha256 `2a390f9dca6f5ae1`, `tools/claim_reader.lean.in` `3fecc3f2be58ee48`: out.txt's first line),
before round 18 changed it. Two changes to `run.py` keep its cases as round 16 tested them: Part R pins the scratch
copy's vec-add ledger at its events 1–9 (`pin_ledger`; the kernel's review went on, to 14 events, with a second
filing of r-vecadd in round 2 and a second reviewer), and Part M's vec-add line prints the manifest's ranges as
shipped. A run with a later tool differs where a later round changed the tool. With round 18's (see
[evidence/tool-round18](../tool-round18/README.md)), checked on the same tree for Parts F, L, M, R and C: the first
line (the tool's sha256); Part L's audit message for a page with no usable table, which adds "before any heading
of the section" (L95); Part M's messages and claim.md §2 paragraph, reworded (L79); and Part R, where vec-add's
four round-1 work files, which no event records, are named in every case, in the shipped tree too until the
coordinator's `tvr freeze vec_add_f32` (L81). Parts F and C are the same.

**Questions.**
1. (L62) Does the gate's `kinds` check accept `Compiler.Faithful` only as `Faithful P (model x) (launch x) (K x)`
   for one variable `x` the statement binds, and refuse lr-13a's whole claim F1 (`model false` for every
   instance, whose launched model never ends) and F2 (`model e` or `launch e` for another bound `e`, `model
   (Classical.choose h)`), a behaviour of another instance, a fixed instance's model, a launch written out?
2. (L62) Does it accept a fact only when docs/rules/num.md's section "Facts the library states" lists it, as
   listed (a name alone with any arguments, `DivRn P .binary64` with exactly those), and refuse lr-13a's helpers
   (a `TVA.Rules.Num` definition of type `Platform → Prop` not listed, one named `…Total` not listed), the shapes,
   and `DivRn P .binary32`? Does it fail closed when the section, its table or the page is missing, and does the
   audit then name the page?
3. (L79) Does the map check fail on a code line outside every function that lies neither in the kernel lines nor
   in the host lines (lr-13a's `SCALE` and `OUT_T`, imports, a `__main__` block), pass when each is classed, and
   does claim.md §2 list the classes?
4. (L81) Do the certificate and the records step fail on lr-13a's Part R (a reviewer's `assign`, `file` and
   `raise` lines removed, the folder kept), on the same with the `assign` kept (the current round's folder), on an
   earlier round's folder no event records and on entries of review/ that are no role folder? Do `tvr lint` and
   `tvr start` refuse a reviewer not assigned? Does vec-add's shipped ledger pass?
5. (L82) Is a cited path component in capitals only (`docs/DESIGN`) checked, while `round-N`, `<topic>`, `*.md`
   and `...` stay placeholders? Does a change to a kind's default blocking make an older ledger fail, as protocol §3
   now says?
6. (L7) Does the tool's git, run from inside a repository whose `.git/config` sends github.com to another host
   (lr-13a's `work/gitcfg.py`), still reach Mathlib's repository, for a tag and for `master`'s history? Does a git
   given a folder inside a repository that is not one itself find no repository, and a git given a repository read
   that one's configuration?

**Method.** Part F compiles lr-13a's F1 claim with `lean -o` (not `lake build`) into the scratch folder and runs
the tree's claim reader on it with its `consistent` example, then the gate's axioms, usage, namespace and kinds
checks on the output; then one kernel with ten `faithful`-like fields, read without the `consistent` example (the
fields test `kinds`, not consistency), and `kinds_problems` per field. Part L lists the facts the tree's num.md
names (`stated_facts`), compiles a scratch library module `TVA.Rules.Num.Extra` with lr-13a's two helpers beside a
mirror of the library's build, and a kernel with twelve fields citing listed facts, unlisted shapes and the
helpers; then it calls `stated_facts`, `rule_fact_problem` and the audit's `fact_page_problems` on scratch copies of
num.md (the heading's parenthetical changed, a full name, the section renamed, the table removed, the page
removed). Part M calls `module_lines` and `map_problems` on lr-13a's source (with a class and a `__main__` block
added) and on vec-add's, and renders claim.md §2 in a scratch copy. Part R repeats lr-13a's Part R in a scratch
copy of the tree, with vec-add's ledger at its events 1–9 (a passing gate made by hand, as the kernel did not
build against this library then; `tvr assign`, `tvr file` twice), removes ledger lines, and adds folders. Part C calls `citation_problems` on a page in a scratch
root, and `kernel_event_problems` on vec-add's shipped ledger with `KINDS["DOC"]` made blocking. Part G makes a
repository whose `.git/config` holds `url.https://example.invalid/.insteadOf = https://github.com/`, and runs the
tool's `_git` and `upstream_problem` from outside it, inside it and in a subfolder.

**Result** (`out.txt`).

| Case | Result |
|---|---|
| F (L62) | F1 passes shape, axioms, usage and namespace, and `kinds` refuses it (`model false`); of the ten fields, the standard form, the form without `d.ok` and the form on another bound variable with an extra hypothesis pass; `model e`, `launch e`, `K e`, `model (Classical.choose h)`, `model ⟨0⟩` (also refused for `Inst.mk`), a launch written out and `K d` as a `fun` are refused, each naming the form it found |
| L (L62) | num.md lists 38 facts (6 `…Total` rules, the 30 of `TVA.Rules.Num.Sm90`, round 14's six `CvtSat…` among them, `DivFull`, and `DivRn` as `DivRn P .binary64`; 32 before round 14 added the `CvtSat…` facts); `DivFull P`, `∀ d, d.ok → Exp2F32 P`, `DotF16 P`, `MathTotal P .exp .binary32`, `DivRn P .binary64` (also with the format's full name) pass; `DivRn P .binary32` is refused as not the listed application; `MathAcc` (lr-9a's), `DotF8`, and lr-13a's helpers `DivRnF32` (`Platform → Prop`) and `DivAccTotal` are refused as not listed; with the heading's parenthetical changed or a full name in the table the list is the same; with the section renamed, the table removed or the page missing no fact passes and the audit names num.md |
| M (L79) | lr-13a's manifest now fails on lines 1–3, 5–6 (imports, `SCALE`, `OUT_T`) and 31–32 (the `__main__` block); a class's own lines are code outside every function, its docstring not; each classing passes; vec-add's code lines outside every function are 21, 23–24, 26, 84–93 and 135, and its shipped manifest, whose prover has classed them since (23–24 in the kernel lines, the others in the host lines), passes (before that, the check named 21, 23–24, 26, 90–93 and 135); claim.md §2 lists the classed lines and asks the reviewer to judge them |
| R (L81) | vec-add as shipped: no folder unaccounted for; `tvr start` and `tvr lint` refuse `r-two` before it is assigned; with r-two's three lines removed (lr-13a) the certificate fails and the records step names `review/round-2/r-two`, and `tvr lint` refuses r-two; with only the `file` and `raise` lines removed the folder is the current round's in progress: the records step passes and lints it, and the certificate waits for it; an earlier round's folder with nothing recorded, `review/round-02/`, `review/notes.md` and `review/round-2/stray.json` are each named |
| C (L82) | `docs/DESIGN` is named as missing; `docs/DESIGN.md`, `round-N`, `<topic>`, `*.md` and `...` pass; the tree cites no missing file; vec-add's shipped ledger passes, and fails at its round-1 `DOC` finding once `DOC` is made blocking by default |
| G (L7) | a plain git inside the repository reaches `example.invalid`; the tool's `_git ls-remote` and `upstream_problem` reach Mathlib's repository and accept the pin as the tag `v4.35.0` from outside the repository, inside it and in a subfolder, and as a commit of `master` (a clone of its history) from inside it; a git given a subfolder that is not a repository finds none; a git given the repository reads its `insteadOf`; TVA's own repository is found for `shipped_files`; the folder git runs in stays empty |

**Conclusions.**
1. `Compiler.Faithful` is cited only about the model of the instance launched (L62).
2. A constant of the library is citable only as num.md's list states it; a helper or a name with a suffix is not,
   and without the list nothing is (L62).
3. Every code line outside a function is classed by the prover; which ones the launch reads stays the reviewer's
   judgment (L79).
4. A kernel's role folders follow from its ledger: removing a reviewer's lines leaves a folder the records step
   or the certificate names (L81). A ledger's lines themselves are trusted as the coordinator keeps them.
5. A capitals-only cited name is checked (L82).
6. The tool's git reads no repository's configuration but the one it is given (L7).
