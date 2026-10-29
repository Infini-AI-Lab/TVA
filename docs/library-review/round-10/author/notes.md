# Library batch, round 10: author's notes

Issues: L7, L36, L40, L48, L50, L51, L52, L53, L54, L55, L56 ([ISSUES.md](../../ISSUES.md)), 2026-10-28. The
tool and gate batch from round 7's review, run beside rounds 8 (semantics) and 9 (numerics) in a separate
tree. The changes are in `tools/tvr.py`, `tools/claim_reader.lean.in` and the protocol and role documents;
no Lean file of the library changed, so `lake build TVA` is as before (green, standard axioms only).

**Kernel that does not build or pass the gate:** `kernels/vec_add_f32`, as before this round (expected; its
prover re-proves it after these rounds). L48 changes every kernel item's digest, and L53 makes every gate
recorded before this round stale (it kept no sha256 of the kernel's files and of the gate's code), so the
kernel needs `tvr check` and its reviewers re-accept. I did not edit the kernel folder or run `tvr check`.

**Files of other rounds.** None of rounds 8 and 9 (no Lean, SEMANTICS.md, TRANSCRIPTION.md, exclusions.md or
rule page changed). New files: `evidence/tool-round10/`, and `docs/library-review/ledger.jsonl` (L52; written
by `tvr record`, with rounds 1 to 7 recorded, see L52). `evidence/release-check/README.md` (round 7's, the
library author's) has a corrected conclusion (L36).

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed.** The release check's `packages` step (`make_packages` and helpers in `tools/tvr.py`;
  protocol §2 "Enforcement" and "Lake"), against both round-7 reopens:
  - **upstream only, and anchored to upstream refs** (lr-7a): each package is fetched from its manifest URL
    only, https only (`protocol.allow=never` but https); the working tree's checkout is never a source.
    Mathlib's pinned commit must be what its `inputRev` names in the upstream repository's own refs
    (`upstream_problem`, `git ls-remote`): the tag's commit (peeled if annotated), or a branch's head or a
    commit reachable from it (checked on a commits-only clone of the branch, `--filter=tree:0`). An `inputRev`
    that names no tag or branch (a hash) is refused. A commit fetched by its hash alone is not shown to be
    the repository's;
  - **inherited packages are Mathlib's** (lr-7a): every inherited package must equal, field for field but
    `inherited`, the entry of Mathlib's own `lake-manifest.json` at the verified commit (name, URL, rev,
    inputRev, scope, subDir, manifestFile, configFile, type), and the two sets of names must be equal
    (`inherited_problems`); `lean/lean-toolchain` must be the toolchain Mathlib's commit pins. Exactly one
    direct package, Mathlib;
  - **git free of the user's configuration** (lr-7b): every git the tool runs (`_git_env`, `GIT`) has
    `GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM`, `GIT_ATTR_NOSYSTEM`, an empty temporary `HOME` and
    `XDG_CONFIG_HOME`, none of the caller's `GIT_*` variables, and `-c` overrides for hooks, the attributes
    and excludes files, line-ending conversion, symbolic links and fsmonitor. `shipped_files` runs git the
    same way (lr-7b: the user's excludes file);
  - **every file is its blob** (lr-7b): after checkout, `checkout_problems` computes each file's git blob id
    from its bytes in Python and compares it with `git ls-tree -r`, refuses a missing or extra file (`.git`
    aside), a link where the commit has a file or the reverse, and a submodule;
  - the audit's lake check (`lake_violations`): `packagesDir` is `.lake/packages` and `lakeDir` `.lake`
    (lr-7a), Mathlib's `inputRev` is the lakefile's `rev`, every package is a git package at its root.

  The step needs the network now (about a minute for the nine packages here).
- **Why.** lr-7a: a local commit adding `Mathlib/Extra/Borrowed.lean`, pinned in the manifest, passed the
  lake check, the packages step and the import check (and the same for Batteries, and an inherited package
  from another URL). lr-7b: with a global attributes file and a smudge filter in the user's `.gitconfig`,
  every `.lean` file of a package differed from its blob and nothing noticed.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md): U9 (TVA's inherited
  packages equal Mathlib's manifest at the pinned commit; lr-7a's local Batteries commit and its other URL
  are refused; `packagesDir` elsewhere is refused), N1 (Mathlib's pin is the tag's commit; lr-7a's local
  commit and a hash `inputRev` are refused; Batteries' pin is reachable from `main`; a commit of another
  repository is not), N2 (`make_packages` on the tree's manifest, with lr-7b's smudge-filter home
  directory: no problem, the files are the commit's, and a changed and an added file are then named), N3
  (lr-7a's route end to end: Mathlib pinned at a local commit is refused before anything is fetched). The
  release check (below, L36) ran the new step on the tree.
- **Constants whose meaning changed.** None (the tool and the protocol).

## L36: Leverage judgment: mechanical checks only where complete; transcription as guidance; advisory scans

- **What changed.** lr-7a reopened L36 only for the release check, which L7's fix completes (above), and
  corrected round 7's reasoning on building Mathlib from source. That correction is now in
  [evidence/release-check](../../../../evidence/release-check/README.md), conclusion 5: the `.olean` files do
  not depend on the build path (lr-7a's fifth build, with TVA at its own path, gave R1–R3's digest
  `3f335c460304e5f1`), so 4 of 5 builds agree and R0 is an outlier of unknown cause. The release check keeps
  building Mathlib from source in each run, which both round-7 reviewers judged sound; a pinned build would
  first need the outlier explained, and its pin checked against upstream (L7). The round-7 notes' statement
  that the toolchain "can supply no module TVA imports" was wrong: it supplies `Init`, `Std` and `Lean`;
  protocol §2 now says what the toolchain's trust covers (L56).
- **Why.** lr-7a's verdict on L36, and its correction of the author's reasoning.
- **Evidence.** lr-7a's `work/build_digest_watch.out` (cited, not re-run: a build of Mathlib). The release
  check of this round, on the tree (2026-10-28, in an empty scratch folder outside TVA and the home
  directory, about 22 minutes):
  9 of 11 steps pass. `shipped` (565 files), `packages` (the nine packages made from their upstream
  repositories by L7's new step), `sandbox`, `build` (Mathlib built from source), `library` (hygiene, the
  import closure, `leanchecker TVA`), `records` (every kernel and library ledger as recorded, rounds 1–7
  recorded), `lint library 10 author`, `lint vec_add_f32 reviewer r-vecadd` and `audit` pass. The two
  failures are `vec_add_f32`'s, as expected: its gate fails at its own build (`VecAdd.lean:155`, against
  the current library), and its shipped `gate.json` and certificate are stale (L48, L53); and `pages`
  names its `REVIEW.md`, `claim.md` and `kernels/README.md`, which were generated before this round (they
  still say GRANTED). Its output holds no absolute path (`<dir>`, L56). The check ran before two small
  changes to `tools/tvr.py` that its steps do not use or that only relax a comparison: the refusal in
  `tvr file`/`tvr respond` (L52), and `page_mismatch` masking both gates' times (a page a failing gate
  does not rewrite keeps the old gate's time, which made vec-add's unrewritten `claim.md` count as
  different); evidence/tool-round10 ran on the final tool.
- **Constants whose meaning changed.** None.

## L40: gap: library modules are not replayed with `leanchecker`; any `leanOptions` key is accepted

- **What changed.** The two open parts:
  - **interpolated strings** (lr-7a, lr-7b): Lean reads `"…"` as a plain string or as an interpolated one
    (`s!`, `m!`, `f!`, `throwError`, …), whose `{…}` is code with literals and comments of its own, depending
    on the syntax before it, so no lexer of the text alone reads both right. The two readings differ only for
    a string holding `{`. The hygiene now refuses a string literal (raw ones included) that holds `{`, in the
    library's code and in a kernel's Lean (`brace_string_problems`, in the gate's `hygiene` and in `tvr
    library-check`). Wherever the hygiene passes, the comment reader `_uncommented` (used by the library
    hygiene, the header reader and `opened_names`) reads as Lean does. Neither the library nor the kernel has
    such a string;
  - **the import closure** (lr-7b): every Lean file under `lean/` must be a module that `lean/TVA.lean`
    reaches through its imports (`library_module_problems`, in `tvr audit` and `tvr library-check`), so
    `lake build TVA` builds it and `leanchecker TVA` replays it. Today every file is reached.

  Protocol §2 ("Imports") and §8 (`hygiene`).
- **Why.** lr-7a, lr-7b: `s!"{"--"}" #eval 7` runs the `#eval` while the hygiene's reading blanked it, and an
  `open Nat in` after such a string escaped the namespace check; a library module outside the closure would
  never be replayed.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) U1 (lr-7a's H1 and
  lr-7b's three strings are refused, a plain string and braces in comments are not; the library has nothing
  flagged), U2 (an orphan module is named; the tree has none).
- **Constants whose meaning changed.** None.

## L48: soundness: item digests rest on 64-bit non-cryptographic hashes; two different kernel values share one digest

- **What changed.** The claim reader (`tools/claim_reader.lean.in`) no longer hashes: `TVR.canonConst` writes
  each constant of the cones out in full, as a canonical text: its kind, name and universe parameters, its
  type, its value if a definition (with its safety), and the kind-specific data (an inductive's parameters,
  indices, flags, constructors; a constructor's index and fields; a recursor's rules), with every constant
  name, universe level, literal and binder kind spelled out, each atom with its length or a terminator
  (`canonName`, `canonLevel`, `canonExpr`), bound variables as de Bruijn indices, and a library constant's
  docstring. `tvr` hashes that text with sha256 (`const_table`), and the item digests are made of those
  hashes as before (`item_digests`). Binder names (hygienic ones are renumbered by unrelated edits, and carry
  the module's name) and `mdata` are left out: they change nothing a term means. Protocol §4, §13;
  LIBRARY_REVIEW.md (digests).
- **Why.** lr-7a found `fun x => x + 4536642005803190292` and `fun x => x + 3061831643823180776` with the same
  `Expr.hash`/`String.hash` line and so the same item digest: one model reviewed, the other shipped, and every
  check agreeing.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) Part L: with lr-7a's
  pair as two small kernels compiled with `lean`, the new reader gives different `model` and assumption
  digests (the `conditions` and `spec` items, which do not reach the model, stay equal); a third kernel, A
  with a proof edit and a renamed bound variable, has every digest equal to A's; a fourth, `a + 1`, differs.
- **Constants whose meaning changed.** None. Every kernel item's digest changes once (the hashes are now
  sha256 of the canonical text), so every acceptance goes STALE at the next `tvr check`, as the coordinator
  expected.

## L50: gap: the map's coverage rests on `source.ranges`, which nothing checks

- **What changed.** The map check (`map_problems`, protocol §7, §8) now requires the ranges to hold every
  line of every `@triton.jit` function of the source, from its first decorator to its last line
  (`jit_functions`: every function, at any depth, with a decorator naming `jit`, also under
  `@triton.autotune`), except those `source.unused` names (a new optional manifest key, a list of function
  names, checked by `manifest_problems`); a name there must be a jit function of the source. claim.md §2
  lists the jit functions and the unused ones, and §6 says "every code line of the kernel lines … is mapped,
  and the kernel lines hold every `@triton.jit` function of the source". The reviewer guide, the kernel
  reviewer's prompt and the prover guide say that the reviewer judges an unused function (never run by the
  claimed launch, as the kernel or as a helper).
- **Why.** lr-7a M1: with `output = x + y` left out of vec-add's map, ranges that skip line 52, or no ranges,
  passed.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) U3 (both of lr-7a's
  M1 cases now fail, naming line 52 and lines 29–54; a helper outside the ranges fails, and passes once
  named unused; an unused name that is no jit function fails; vec-add's own manifest passes).
- **Constants whose meaning changed.** None. `vec_add_f32`'s ranges `[[29, 54]]` already hold `add_kernel`.

## L51: gap: claim.md numbers lines with `splitlines()`, so a form feed or U+2028 shifts the quoted source and declarations

- **What changed.** Both fixes: claim.md splits the source and the Lean files at line feeds only
  (`render_claim`, carriage returns stripped), as Python's tokenizer, Triton and Lean's declaration ranges
  count; and the gate refuses a source (`map`) or a kernel Lean file (`hygiene`) holding another line break
  (`LINE_SEPARATORS`: a carriage return not before a line feed, `\v`, `\f`, `\x1c`–`\x1e`, `\x85`, U+2028,
  U+2029), which `str.splitlines` and some editors break at. CRLF line ends stay allowed. The source is now
  read with no newline translation, so its sha256 is the file's. Protocol §7, §8; prover guide.
- **Why.** lr-7a M2, lr-7b: one U+2028 in a comment shifted every later row of the map table and every later
  quoted declaration.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) U4 (lr-7a's M2 line by
  `splitlines` and by line feeds; the map check refuses U+2028, a form feed, a lone carriage return and
  U+2029 and accepts CRLF; the kernel hygiene refuses a form feed in `VecAdd.lean`). No pinned tutorial,
  kernel or library file holds such a character.
- **Constants whose meaning changed.** None.

## L52: gap: the release check does not compare certificates and review records with the ledger

- **What changed.**
  - **The certificate**: after each kernel's gate in the sandbox, the certificate the tool regenerates from
    the ledger and the recomputed gate must equal the shipped one (`certificate_mismatch`: both absent, or
    both present with the same kernel, gate, items and reviewers);
  - **the ledgers** (`ledger_problems`, the release check's new `records` step): every file a ledger records
    exists with the sha256 recorded last for it, the events are numbered 1, 2, …, and no file was added to a
    folder an event records. `tvr file` now records every file of the reviewer's round folder (work files
    included) and the folder, and `tvr respond` the prover's; both refuse to record again a file recorded
    before with another sha256 (a second filing in the round cannot re-freeze a changed first one);
  - **a ledger for library rounds**: `docs/library-review/ledger.jsonl`, written by the new `tvr record
    library <round> <who> [--no-lint]` (the coordinator's, when a round is done): it lints the folder as `tvr
    lint library` does, scans it, refuses a folder recorded before, and records the sha256 of every file. The
    `records` step also fails on a role folder of a round before the last that is not recorded, and
    `release_records` lints only the last round's folders not yet recorded (a recorded round is frozen, and its
    lint would depend on the issue list's current rounds). I recorded rounds 1 to 7 (19 folders) with
    `--no-lint`, since they ended before the ledger existed; the coordinator records rounds 8 to 10;
  - protocol §2, §3 (recorded files), §10 (recording), §13; LIBRARY_REVIEW.md (rounds, records).
- **Why.** lr-7a G1: a `certificate.json` naming the recomputed gate passed with no acceptance behind it; a
  recorded file changed after recording shipped; library records had no recorded hash at all.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) U6 (the shipped kernel
  ledger is consistent; a changed recorded filing, and a work file added to a recorded folder, are named;
  lr-7a's G1 certificate is refused; library: round 7 recorded in a copy, a second record refused, a changed
  and an added work file named, the unrecorded rounds named, the lint list).
- **Constants whose meaning changed.** None.

## L53: gap: gate staleness uses file times, leaves out the source and the write-up, and ignores changes to the gate

- **What changed.** `gate.json` records the sha256 of every kernel file the gate read (`files`: the manifest,
  the source, the model map, the write-up, every Lean file; `kernel_files`) and of the gate's own code
  (`tool`: `tools/tvr.py`, `tools/claim_reader.lean.in`). `gate_staleness` compares contents, not file times:
  a gate is stale when a recorded sha256 differs (or a Lean file was added or removed), when the gate's code
  changed, or, as before, when a library file or the packages of its snapshot differ; a gate without these
  records (before round 10) is stale. The release check compares the shipped `gate.json`'s `files` and `tool`
  too (`gate_mismatch`), and its new `pages` step regenerates and compares every generated page (each
  kernel's `claim.md` and `REVIEW.md`, `kernels/README.md`, `REPORTS.md`), the gate's time masked
  (`page_mismatch`); `REVIEW.md`'s header no longer carries the time it was rendered, so it is
  reproducible. Protocol §8, §9.
- **Why.** lr-7a S1, lr-7b: an edit of the source or the write-up, or of a Lean file with its old time put
  back, left the certificate GRANTED; a gate fix left old gates current; REVIEW.md and kernels/README.md said
  GRANTED for vec-add while its certificate had lapsed.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) U5 (a fresh gate is not
  stale; after a source comment, a write-up line, a Lean edit with its old modification time restored, an
  added Lean file, or a claim-reader change, it is, naming what changed; the shipped vec-add gate is stale),
  U6 (pages that differ only in the gate's time are the same; a shipped REVIEW.md that says otherwise is
  named).
- **Constants whose meaning changed.** None.

## L54: gap: `lean/lean-toolchain` is unchecked

- **What changed.** The audit's lake check requires `lean/lean-toolchain` to be one line
  `leanprover/lean4:v<major>.<minor>.<patch>` (optionally `-rc<n>`, `TOOLCHAIN`); the release check refuses a
  toolchain folder, elan `bin` or `settings.toml` that is a symbolic link or lies under one (`elan_parts`,
  as `elan toolchain link` makes them), binds the parts at their own paths, and requires the toolchain to be
  the one Mathlib's commit pins (L7). Protocol §2.
- **Why.** lr-7b: the audit passed `someone/lean4:v4.35.0-patched` and `leanprover/lean4:nightly`, and the
  release check bound whatever folder a link pointed to.
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) U7 (the two names and a
  two-line file are refused, the real one accepted; a linked toolchain folder is refused, a real one and the
  user's are accepted).
- **Constants whose meaning changed.** None.

## L55: gap: a `TVA.Rules.*` constant declared outside `lean/TVA/Rules` gets no rule page

- **What changed.** Both fixes lr-7a named: the gate's `kinds` check (`kinds_problems`) refuses any
  `TVA.Rules.*` constant of a cone declared in a module outside `lean/TVA/Rules` (an equation lemma Lean
  generates for one in the kernel's module aside), and `library_files` takes rule pages from the names of
  the cone's `TVA.Rules.*` constants as well as from their modules (`rule_page` on a name). Protocol §4, §8.
- **Why.** lr-7a P1: the page followed the module, the `[rule]` kind the name, so such a constant passed
  `kinds` with no rule page in its digest (latent: none exists).
- **Evidence.** [evidence/tool-round10](../../../../evidence/tool-round10/README.md) U8 (`TVA.Rules.Num.borrowed`
  in `TVA.Lemmas.Num` is refused, `Faithful` and its equation lemma are not; the cone's digest holds
  `docs/rules/num.md`).
- **Constants whose meaning changed.** None.

## L56: wording: the toolchain's trust, the release check mid-round, `[kernel]` assumptions, skipped grids, DESIGN.md

- **What changed.**
  - **the toolchain's trust** (protocol §2): it supplies Lean, Lake, `leanchecker` and the core modules
    `Init`, `Std` and `Lean`, which every module reaches; it sits in the user's writable home and its files
    are unchecked (trusted as installed); what is checked is which toolchain it is (L54);
  - **mid-round runs, and the folder in the output**: `tvr release` prints, and writes into `release.json`,
    its folder as `<dir>`, TVA's copy as `<tva>` and the home directory as `~` (step details included), so its
    output can be kept in `work/`; protocol §2 and the library author's and reviewer's prompts say that a
    mid-round run fails `lint` on the round's records in progress and that a scratch path in any work file
    fails the audit and every kernel's `independence`;
  - **`[kernel]` assumptions are proved by `consistent`** (protocol §2, §4, §5, §9 `EVIDENCE_REQUEST` and
    `ASSUMPTION_*`, REVIEWER_GUIDE §4 and its opening, PROVER_GUIDE, the kernel prover's and reviewer's
    prompts): a closed statement, which the reviewer judges for need, clarity and whether the constants it
    constrains mean what the write-up says;
  - **skipped grids** (protocol §6, DESIGN.md): `2^31` to `2^32` programs are skipped without an error, a
    larger grid may run (`2^32 + 1` wraps to one), and `Launch.Valid` excludes both;
  - **DESIGN.md's decision row** on the release check: a dependency fails however it is spelled or computed
    in the steps run in the sandbox; evidence and work scripts are checked by the lexical scan only.
- **Why.** lr-7a's and lr-7b's wording findings, each false or incomplete as written.
- **Evidence.** evidence/launch-outcomes C8 (the `2^32 + 1` grid runs every program); lr-7b's
  `work/KernelFieldProved.lean` (`consistent` proves a `[kernel]` field); this round's release check (L36):
  its printed output and `release.json` name the folder `<dir>` and hold no absolute path.
- **Constants whose meaning changed.** None.
