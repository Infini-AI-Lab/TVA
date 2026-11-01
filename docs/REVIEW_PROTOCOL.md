# Prover/reviewer protocol (`tvr-2`)

Who writes what, who checks what, and what each of them trusts. The tool is `tools/tvr.py` (run
it as `python3 -I tools/tvr.py …` from the TVA root). The agents' prompts are in
[docs/prompts/](prompts/): each role's prompt states its scope, and `tvr prompt` fills it in.

## 1. Two layers, four roles

A kernel's claim rests on two layers. Each has an author and reviewers, and each layer's
reviewers check only that layer.

| | **Library** (`lean/TVA`, `docs/`, `evidence/`) | **Kernel** (`kernels/<kernel>/`) |
|---|---|---|
| Author | **library author**: what each Triton construct means (`TVA.Lang`, `TVA.Sem`, [SEMANTICS.md](SEMANTICS.md)), guidance on transcribing source into a model ([TRANSCRIPTION.md](TRANSCRIPTION.md)), the compiler rule and the device facts with their constants (`TVA.Rules`), lemmas | **kernel prover** ([PROVER_GUIDE.md](PROVER_GUIDE.md)): the model, the model map, the conditions, the specification, the kernel's own assumptions, the proof, the write-up |
| Reviewer | **library reviewer** ([LIBRARY_REVIEW.md](LIBRARY_REVIEW.md)): checks the library against Triton 3.8.0 and the target GPU, using the compiler's sources, PTX and runs on the GPU | **kernel reviewer** ([REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)): checks the prover's choices against the kernel's source and its real use, **trusting the library** |

Two more roles:
- **The coordinator** assigns reviewers, keeps the ledgers, and triages library reports into
  library issues.
- **The user** decides escalated disputes and questions of scope.

### The trust boundary

Kernel review takes the library as given, and kernel reviewers do not need to understand how a
library fact is established. Take "on sm_90, `tl.math.exp2` on `float32` is within `E` of `2^x`".
Its lowering and its measurement are library matters, settled in library review with evidence in
`evidence/`. Everything in the library is trusted in kernel review:
- the meaning of every `tl.*` operation, of launches and of outcomes, and every other fact
  [SEMANTICS.md](SEMANTICS.md) states about Triton: how its front end evaluates the source, the
  constants it builds, the shapes its compiler rewrites;
- `Compiler.Faithful`;
- the device facts;
- the lemmas.

[TRANSCRIPTION.md](TRANSCRIPTION.md) is guidance (worked patterns, pitfalls, common reasons to
drop a line), not a trusted statement: it states no fact of its own, and whether a model means
what its source does is the kernel reviewer's judgment (library issue L36).

A kernel reviewer who doubts the library files a `LIBRARY_REPORT`. It does not block the kernel;
it goes to the library's issue list (§10). A library fix re-opens exactly the kernel items whose
meaning reaches the changed constants (§4).

## 2. Scopes

### Kernel prover

- **Writes:**
  - the model: the kernel's source in the library's vocabulary, meaning what the source does by
    [SEMANTICS.md](SEMANTICS.md) ([TRANSCRIPTION.md](TRANSCRIPTION.md) is guidance);
  - the model map;
  - `Inst`, `Inst.ok`, `Pre`, `Post`;
  - the `Assumptions` structure;
  - `main`, `consistent`, `satisfiable`;
  - the write-up.
- **Assumptions are of two kinds only** (§5):
  - `[rule]`: a library rule or device fact, cited as the library states it;
  - `[kernel]`: a fact about the kernel's own values. It mentions neither `K` nor `P`, so it is a
    closed statement, and Lean proves it through `consistent` (library issue L56).
- **No device or compiler facts of the kernel's own.** The prover never states a device or
  compiler fact or picks accuracy constants. If a kernel needs a fact the library lacks (say, the
  accuracy of an fp16 `tl.dot`), the prover asks the library author for it (a library issue).
  The kernel waits until the library has it.
- **Gaps in the semantics.** If the source uses a construct whose meaning the library does not
  state, or states as open, the prover asks the library for the fact.

### Kernel reviewer

- **Trusts the library** (§1). Does not read the compiler's sources or PTX, does not consult other
  projects' material, and runs nothing on a GPU.
- **Judges**, by asking how the claim could be wrong:
  - **the model against the source**: on every covered instance, does the model do what the
    source does, by [SEMANTICS.md](SEMANTICS.md)? Every line, every drop and its reason, and the
    model map. [TRANSCRIPTION.md](TRANSCRIPTION.md) lists patterns and pitfalls to help, not rules
    to tick off;
  - **what every name in the kernel's Lean means**: the reviewer reads every Lean file of the
    kernel in full, every `def`, `notation`, `open`, `namespace`, `instance`, `attribute`, `export`
    and `set_option`, every binder and field access, and anything that changes what a library name
    or a model line means. Lean
    checks the proofs; what the names mean is the reviewer's to judge, and a model line that reads
    as a library operation but means another is a `MODEL_MISMATCH` (user, 2026-10-27, library
    issue L20). The gate's `hygiene` and `namespace` checks (§8) are a backstop;
  - **the conditions**: they cover the kernel's real use (the host code), and they are what the
    write-up says;
  - **the specification**: it says what the kernel is for, and a tolerance is meaningful at the
    sizes the conditions allow;
  - **each `[rule]` assumption**: it is the library's fact for the operation the model uses
    (format, configuration, function);
  - **each `[kernel]` assumption**: Lean proves it through `consistent`, so the reviewer judges
    whether it is needed and clearly stated, and whether the kernel constants it constrains mean
    what the write-up says;
  - **the write-up**: it matches the Lean.
- **Not in scope:**
  - whether the kernel compiles. A launch that Triton or the driver reports to the host with
    an error, before any program runs, is outside every claim (§6);
  - how Triton lowers an operation;
  - whether a device fact's constants are right;
  - whether a `tl.*` definition matches Triton.

  These go to the library as `LIBRARY_REPORT`s.

### Library author

- Maintains the semantics (every fact about Triton a model relies on), the transcription
  guidance, `Compiler.Faithful`, and one device-fact page per target: per operation, its
  lowering, its constants and the evidence.
- Writes the evidence in `evidence/<topic>/`: a `README.md` (the question, the method, Triton's
  version, the GPU, the date, the result), the script, and its output.
- Triages library reports into issues, and fixes them in batches.

### Library reviewer

- Checks the library against Triton 3.8.0's documentation and compiler (the pinned sources in
  `primitives/`) and against the target GPU.
- May read any source, compile, inspect PTX and run measurements.
- A device fact is accepted on TVA's own evidence only.

### Independence

TVA stands on itself and on Mathlib (general mathematics), and on nothing else. This holds for
every role.
- **Results and lemmas.** A proof in TVA, in the library or in a kernel, uses only TVA's
  definitions and lemmas and Mathlib's. Lean enforces this through imports:
  - the library imports only `TVA` and `Mathlib`, and a kernel also its own modules;
  - the lake configuration requires only Mathlib.
- **Evidence.** A device fact rests on TVA's own evidence in `evidence/`, produced with Triton
  3.8.0 on the target. Another project's measurement, card or argument is not used, not even as
  a pointer.
- **Text.** No library file, kernel file or review record names another project or a path into
  one, and no role reads another project's material.
- **Enforcement.** TVA will be released on its own (user, 2026-10-28: "we need to keep the
  project independent. since tva will be released without others."), so every check is blocking:
  - **The release check** (`tvr release --dir <empty folder>`, library issues L36, L7) is complete
    by construction: its sandbox is built from an allowlist, so nothing exists in it but what TVA
    ships and an explicit, read-only list of what the toolchain needs (with `bwrap`; no network):
    - **TVA**: exactly the files TVA ships (what git tracks, and the untracked files git does not
      ignore, as git lists them with no user or system configuration), copied into an empty home
      directory at TVA's own path. No shipped file may lie in a folder the text scan skips (`.lake`,
      `.venv`, `.git`, `__pycache__`), and `.gitignore` ignores `.lake/` at any depth (L41);
    - **Mathlib and its dependencies**, made afresh in the copy from their upstream repositories
      (library issue L7; this step, outside the sandbox, needs the network):
      - Mathlib's `inputRev` must be the release tag of the toolchain's Lean (`v4.35.0` for
        `leanprover/lean4:v4.35.0`), and the pinned commit that tag's commit; or the branch
        `master`, and the pinned commit its head or a commit reachable from it; as the repository's
        own refs show. Another branch of the repository (it has dozens, personal and CI ones among
        them) is not reviewed Mathlib, and is refused (library issue L7). A commit fetched by its
        hash alone is not enough: a host may serve a commit that no branch or tag of the repository
        holds;
      - every inherited package must be, field for field, the one that Mathlib's own
        `lake-manifest.json` at that commit pins, and `lean/lean-toolchain` the toolchain that
        Mathlib's commit pins;
      - each package is a new git repository that fetches exactly its pinned commit from its URL
        (https only, never the working tree's checkout) and checks it out, with git run with no user
        or system configuration (no global attributes, filters or excludes, no hooks, no line-ending
        conversion); then every file of the checkout is compared with the commit's blob, byte for
        byte, and no other file may be there.

      So a package holds that commit's files and nothing else (none of the working tree's ignored or
      untracked files, none of its build outputs), and the sandbox builds Mathlib from that source
      (about 20 minutes on 96 cores; the folder needs about 16 GB, and the sandbox is removed after
      the run unless `--keep`, `release.json` kept);
    - **the system directories** `/usr` and `/etc` (and `/bin`, `/sbin`, `/lib`, `/lib64`, links
      into `/usr` here), read-only; the check fails if anything in them is writable by the user or
      owned by a regular user, so nothing can have been placed there for the check to find;
    - **the Lean toolchain** that `lean/lean-toolchain` pins (elan's `bin`, `settings.toml` and that
      toolchain), read-only. It supplies Lean, Lake and `leanchecker`, and the core modules `Init`,
      `Std` and `Lean`, which every module reaches (each imports `Init`; Mathlib imports `Lean` and
      `Std`). It sits in the user's home directory, where the user can write it, and nothing checks
      its files: it is trusted as installed, as Lean itself is. What is checked is which toolchain it
      is: `lean/lean-toolchain` must name a release of Lean (`leanprover/lean4:v<version>`, library
      issue L54), the one Mathlib's commit pins, and the check refuses a toolchain folder, `bin` or
      `settings.toml` that is a symbolic link or lies under one (such as a toolchain that
      `elan toolchain link` made);
    - **the system's Python** (a `/usr/bin/python3.N`, `N ≥ 11`), on which the tool runs with
      `-I -S`: its standard library is the system's, and no user's file is importable but the
      copy's;
    - `/dev`, `/proc` and an empty `/tmp`.

    The programs the check runs outside the sandbox are the system's (library issue L7):
    `/usr/bin/git` (the packages, the upstream refs they are checked against, the shipped files),
    `/usr/bin/bwrap` (the sandbox) and `/usr/bin/find` (the system directories' check), each a file
    under `/usr` that the user cannot write and that no regular user owns, never what the caller's
    `PATH` finds first; a missing one fails the `sandbox` step. They run with a `PATH` of the
    system's folders only (`/usr/bin:/bin`) and nothing else of the caller's environment (git keeps
    only the network settings, as above). Git also reads the configuration of the repository it
    finds from the folder it runs in (a `url.<base>.insteadOf` there sends a URL to another host, a
    proxy there decides what answers), so it reads no repository's configuration but that of the
    repository it is given: each new package repository's own (made with no template, so it holds
    only what `git init` writes) and, to list what ships, TVA's own. Every git runs in an empty
    folder of the tool's own; one given a repository (`git -C <repo>`) finds only that one
    (`GIT_CEILING_DIRECTORIES` is the folder above it), and every other one (the ref lookups, the
    clone of a branch's history, the `init`s) finds none, wherever the check is run from (library
    issue L7). The tool itself runs on the Python that calls it (`python3 -I tools/tvr.py`), and
    inside the sandbox on the system's.

    There it runs, in order:
    - `lake build TVA`;
    - `tvr library-check`: the library's hygiene, that `lean/TVA.lean` reaches every library module
      through its imports, and `leanchecker TVA`, which re-checks every library declaration with
      Lean's kernel (L40);
    - the gate on every kernel, each compared with what the kernel ships: its `gate.json` (the
      result, the digest, the item digests, the kernel files it read, the gate's code and the build's
      other inputs must be the same, L42, L53, L69), and its certificate, which the tool regenerates from the ledger and the
      recomputed gate (both absent, or both present with the same gate, items and reviewers, L52);
    - `pages`: every page `tvr` generates (each kernel's `claim.md` and `REVIEW.md`,
      `kernels/README.md`, `docs/library-review/REPORTS.md`) must be the one regenerated from the
      shipped files, the gate's time aside (L53). The check compares each regenerated page with the
      page the copy shipped, read from the copy before anything runs in the sandbox (as it reads each
      shipped gate and certificate), never with the live tree, which may change during the run
      (library issue L96);
    - `records`: every file a ledger records (each kernel's `review/ledger.jsonl`, and the library's
      `docs/library-review/ledger.jsonl`) must exist with the sha256 it was recorded with, and no file
      may be recorded with two; no file may have been added to a folder a ledger records; every event
      of a kernel's ledger must follow from the recorded file it belongs to, and every role folder of
      a kernel's `review/`, and every file in it, from its own events (§3, L81); every line of the library's ledger records one
      role folder, once, linted unless its round is one of the first seven; and
      every role folder of a library round before the last must be recorded (L52, §10). `release.json`
      also keeps each ledger's number of lines and sha256, which no step reads back: an edited line
      shows only against a copy kept elsewhere, compared by hand (§3, L82);
    - `tvr lint` on the records not yet recorded: the role folders of the library's last round that
      its ledger does not record, and each kernel's current round;
    - `tvr audit`.

    A dependency of these steps on anything but TVA, Mathlib's pinned source, the Lean toolchain and
    the system fails there, however it is spelled or computed, because nothing else exists.
    Evidence scripts and reviewers' work scripts are not run in it (they need Triton and the GPU):
    their text is checked by the scan (below), whose path tests are lexical. Recorded records are
    frozen: their format was checked when they were recorded, the `records` step compares them with
    their ledgers, and `tvr audit` reads their text. Without `bwrap` the check fails, since it cannot
    build the sandbox; with `--no-mask` only `HOME` changes, and the check fails its `sandbox` step
    (a run for testing, not a release check).

    The check's output (what it prints, and `release.json`) writes its folder as `<dir>`, TVA's
    copy as `<tva>` and the home directory as `~`, so the output can be kept in TVA (library issue
    L56). Run in the middle of a round, it fails `lint` on that round's records still in progress,
    and a scratch path that any work file holds fails the audit, and so the `independence` check of
    every kernel's gate, in every run.
  - `tvr audit` checks the whole project's imports, lake configuration and text.
  - The gate's `independence` check covers a kernel and the library.
  - `tvr file`, `tvr respond` and `tvr lint` refuse a filing, report, work file, response or
    notes that refers to another project.

  What the checks read:
  - **Imports.** The header of every Lean file under `lean/` and in a kernel's `lean/` folder,
    read as Lean reads it: `module`, `prelude`, and every `public`, `meta` and `all` form of
    `import`, with comments. A header the tool cannot read, or an `import` after the header, fails.
    Every Lean file under `lean/` must be a module that `lean/TVA.lean` reaches through its imports,
    so that `lake build TVA` builds it and `leanchecker TVA` replays it (library issue L40).
  - **Lake.** `lean/lakefile.toml`, parsed as TOML: the only requirement is Mathlib from
    `leanprover-community` (no `path`, no `git`); the libraries are `TVA` and one per kernel folder,
    exactly `srcDir = "../kernels/<kernel>/lean"` with the kernel's module as its root; no other
    key. `leanOptions`, which apply to every module, may set only resource limits, `autoImplicit`,
    `relaxedAutoImplicit`, `linter.*` and `pp.*` (library issue L40). There is no `lakefile.lean`.
    The lakefile's `rev` is the release tag of the toolchain's Lean or `master` (library issue L7).
    In `lake-manifest.json` the only direct package is Mathlib, from its own repository, with the
    lakefile's `rev` as its `inputRev`; every package is a git package at its repository's root;
    `packagesDir` is `.lake/packages` and `lakeDir` `.lake` (library issue L7). Which commits the
    packages are is checked by the release check (above), against the upstream repositories.
    `lean/lean-toolchain` is one line, `leanprover/lean4:v<major>.<minor>.<patch>` (optionally
    `-rc<n>`), a release of Lean (library issue L54).
  - **Text.** Every file outside `.lake`, `.venv`, `.git` and `__pycache__`, whatever its
    extension, must be UTF-8 text. It may not contain another project's name: as a word, as a
    piece of a word (split at `_`, case changes and digits), or at the start or end of a
    one-case piece, also after the escapes below are decoded. It may not contain a path that leads
    out of TVA:
    - a path component `fpv` followed by a component other than `tva` (after `..` is resolved),
      wherever it occurs, whatever precedes it: `fpv/<name>` names a sibling of TVA;
    - a home path other than TVA (`~/fpv/tva`, by whole path components), the Lean toolchain
      (`~/.elan`), Triton's compile cache (`~/.triton`) and the folders that hold them. A home path
      is `~/…`, `~<user>/…` or `/home/<user>/…`, with `/` or `\` after the home prefix, also when
      written with `$HOME`, `${HOME}` (with any expansion operator and nested braces, `${HOME:-}`,
      `${HOME%/}`, `${HOME:-${X}}`), `%HOME%`, `%USERPROFILE%`, `$env:HOME`, a command substitution
      `$(…)` (nested to any depth) or one in backquotes that mentions the home directory
      (`` `echo ~` ``), or as a `file:` URL; other URLs are not paths. It counts after any
      character other than a letter, a digit, `_`, `.`, `/`, `\` or `~`, so also after `>` (a
      shell redirect), a closing bracket or `-`;
    - another absolute path outside the system directories (`/usr`, `/etc`, …) and the
      conventional placeholder `/path/to/…`, counted after a character that cannot be part of a
      path name (a space, an opening quote, an opening bracket, `=`, `:`, `;`, `,`, …), but not
      after a closing bracket, `>`, `}` or `-`, which end a placeholder such as `round-<N>/` or
      `{kernel}/` inside a relative path;
    - a relative path whose `..` climbs out of TVA from the file's folder.

    In a git checkout the audit also lists what TVA ships, and fails on a shipped file inside a
    folder the scan skips, so the scan reads every file that ships (library issue L41). Every file
    under `kernels/` is scanned, in a folder that holds no `kernel.json` too (library issue L70).

    Before these tests the text is normalised (NFKC) and the escapes of path characters, letters
    and digits are decoded: `\/`, `\uXXXX`, `\UXXXXXXXX`, `\xXX`, Python's `\NNN` and `\N{…}`,
    `%XX`, `&#N;`, `&#xN;` and named HTML entities (`&sol;`); a line continuation (a backslash
    before a line break) joins its lines. The tests read the text as written, the text with its
    quotes removed (as a shell joins `~/"fpv"/<x>` into one word), and each text with every brace
    group without spaces (`{a,b}`) replaced by one of its alternatives (as a shell expands it). A
    symbolic link may not lead out of TVA. These tests apply to every file. Triton's own files,
    pinned by sha256 in `primitives/` and `tutorials/`, skip only the generic absolute-path and
    relative-path tests (Triton's text holds paths of its own build tree and placeholders). A
    kernel's source is the prover's file, though it pins it by sha256: every test applies to it.

    **The text scan is lexical.** It reads paths as they are written. A path a program or a shell
    computes at run time is not checked: `os.path.join`, an f-string, `Path.home()`, a shell
    variable other than the home directory's, and the shell's expansions other than braces and
    quotes, such as a glob (`fpv/*/…`, `fpv/[o]…`) or `cd ~/fpv` followed by a relative path,
    except that a computed path into a sibling of TVA that still shows `fpv/<name>` is found. The
    guarantee for results and lemmas does not rest on the text scan: it is Lean's imports and the
    lake configuration (above), which leave no other project's code to import.
  - **Citations** (library issue L70). A cited file of TVA must exist: in every Markdown and Lean
    file but the records (`docs/library-review/`, a kernel's `review/`, which cite files as they
    were and are frozen), the evidence folders (whose scripts and outputs name the scratch files of
    their tests) and Triton's pinned files, every path from TVA's root that starts `docs/` or
    `evidence/`, and in a Markdown file every link to a relative path (code spans aside). A path is
    read up to a placeholder (`round-<N>/`, `{kernel}`, `*.md`, `…`, `...`, a lower-case word with an
    upper-case one after its last hyphen, such as `round-N`), and a line number, an anchor and closing
    punctuation are not part of it. A component in capitals only (`DESIGN`, cited without its `.md`) is a
    name, and is checked (library issue L82).
  - **Fact pages.** `tvr audit` checks that `RULE_PAGES` in `tools/tvr.py` gives a page of
    `docs/rules/` for every module of `lean/TVA/Rules`, at any depth (a module in a subfolder, such
    as `TVA.Rules.Num.Exp2`, has the page of the nearest module above it that has one, library issue
    L43), that every entry names a module or a folder there, that every page belongs to an entry,
    that the general fact pages exist (§4), and that [rules/num.md](rules/num.md) has its section
    "Facts the library states" with a table of facts, without which the gate accepts no fact but
    `Compiler.Faithful` (§5, library issue L62), and with only that one table: the gate reads only the
    section's first table, so a second one, under a `###` subsection too, fails here (library issue
    L95). The gate reads the page line by line, not as Markdown renders it, so the audit also fails,
    and the gate accepts no fact, when:
    - the section holds anything but blank lines, table rows, `###` headings and lines of text, with
      a blank line around the table (a setext heading or rule, a fence, an indented line, a quote, a
      list, an HTML comment), or holds HTML anywhere in a line (a `<` before a letter, `/`, `!` or
      `?`, code spans included) or a `~` (GFM's strikethrough);
    - the table's header row and delimiter row have different numbers of cells (GFM then shows no
      table); the gate splits each row as GFM does (one leading and one trailing `|` removed, then a
      split at each `|` that no backslash escapes), so a row that starts `||` has an empty first cell;
    - before the section, read as CommonMark reads its top-level blocks, the section's heading lies
      inside a fence (one opens at a line that starts with three or more backticks or tildes, unless
      a backtick fence's info string holds a backtick, and closes at a line of the same character, at
      least as long, with nothing after it but spaces) or an HTML comment; a line outside a fence
      starts with `<` after up to three spaces (an HTML block, which can run on across blank lines
      and headings); or a fence line is indented by one to three spaces (it may belong to a list
      item, which a line reader cannot tell);
    - another heading of that name is on the page.

    Then what the gate reads as the facts table is what the page shows as it.

  The tool keeps the names of other projects only as sha256 hashes, so TVA does not name them.
  The text scan's path tests stay as they are, and are not extended spelling by spelling: a path
  spelling that names no project, and that the release check shows is no dependency, is a
  wording matter, not a soundness defect.
- **A finding that rests on another project** is not recorded. Restate it on TVA's own terms,
  or as a library request.

## 3. The kernel folder

Everything about one kernel is in one folder. Every file has one author, and `tvr` records a
reviewer's or the prover's file only from that author's place. `kernels/README.md` (generated)
lists the kernels and where each one stands.

```
kernels/<kernel>/
  README.md                  prover    the write-up: Specification, Conditions, Assumptions, Proof, Narrowings
  kernel.json                prover    manifest: title, Lean module and namespace, source file, sha256, line ranges
  SETUP.md                   setup     for a kernel a tutorial autotunes: the config chosen at setup, and from where (below)
  source/<file>.py           setup     the kernel source, pinned by its sha256 (copied at setup, below)
  lean/<Module>.lean, lean/<Module>/*.lean
                             prover    the Lean bundle (a lake library of its own: lean/lakefile.toml)
  model-map.json             prover    each source line → model text, or a drop and its reason (§7)
  claim.md                   tvr       the sheet the reviewers judge (tvr check)
  gate.json                  tvr       the gate's checks and the item digests (tvr check)
  REVIEW.md                  tvr       the review, readable: status, items, findings, every round in full
  review/
    round-1/
      <reviewer>/filing.json reviewer  findings, releases, re-raises, acceptances (§9)
      <reviewer>/report.md   reviewer  the review in words: verdict, what was checked, what was not
      <reviewer>/work/       reviewer  scratch files behind the findings (Lean checks, arithmetic)
      prover/response.json   prover    one response per open finding (§9)
      prover/notes.md        prover    what changed, in words
    round-2/ …
    ledger.jsonl             tvr       every event, in order, append only (REVIEW.md is made from it)
    snapshots/*.json         tvr       each passing gate's item cones (for "what changed")
    certificate.json         tvr       present exactly while the certificate is granted
```

A kernel folder holds no device evidence: device facts are the library's.

- **Rounds.** Round `N` opens when the prover has answered round `N − 1` (round 1 at the
  start). Each reviewer files in round `N` (a second filing in the same round is `filing-2.json`
  and `report-2.md`). Then the prover answers every open finding once, which closes the round.
- **Recorded files are frozen.** The ledger keeps the sha256 of every file it records: a filing
  records every file of the reviewer's round folder (the filing, the report and the work files),
  a response every file of the prover's, and both record the folder. `REVIEW.md` flags any
  recorded file that changed later, and the release check fails on one, on a file recorded with
  two sha256s (the tool never writes that), and on a file added to a recorded folder after its last
  recording (library issue L52). The folder of every filing and response a `file` or `response`
  event records is frozen, whether or not the line names it as `dir`: early lines did not (vec-add's
  events 2, 5 and 7), and recorded only the filing and the report, or the response and the notes
  (library issue L81, below).
- **Every event follows from a recorded file** (library issue L52). A `file` event is followed by
  exactly the events its filing gives at the gate it was filed at (a `raise` per finding, a
  `release` and a `reraise` per entry, an `escalate` after the last allowed re-raise, an `accept`
  of the items it names at that gate's digests, which the gate's snapshot holds), and it must be the
  filing of a reviewer assigned before it, at the round and the path the tool gives it; a
  `response` event the same for the prover's response. The tool writes these events with the same
  function that derives them again, so an event written by hand (an `accept` with no filing behind
  it) fails: the certificate needs a ledger that holds to its records (§9), and the release check's
  `records` step fails. `assign`, `note` and `freeze` (below) are the coordinator's and `decide` the
  user's (on an escalated finding only); no file stands behind them. The tool derives the events again with its
  current defaults (a kind's default `blocking`, the number of re-raises before an escalation and
  the escalation's memo), not with those of the day a filing was recorded: changing one of them
  makes every older ledger with an event it decided fail the `records` step and the certificate
  (library issue L82).
- **Every role folder follows from the events** (library issue L81). Conversely, the events must
  account for every folder of `review/`, so that a reviewer's lines cannot be removed while their
  filing stays. A role folder is accounted for only by its own events: the `file` events whose
  filing lies in it, or the `response` events whose response does, and then a `freeze` of it (below);
  never by a `note`, nor by another folder's event, whatever files they name. The `records` step and
  the certificate fail on a `round-N/<who>/` folder of a reviewer no `assign` event names, on a
  folder of a round other than the current one that none of its own events records (an earlier
  round's whose events are gone; no event can record a later round's), on a file of a folder its own
  events record that none of them records (a file added after the last recording, or one an early
  line left out), and on anything else in `review/` (the ledger, the certificate, the snapshots and
  the `round-<N>/` folders of role folders aside). The current round's folders that no event records
  yet are filings and responses in progress: the release check lints them, and the certificate waits
  until they are recorded. `tvr start` and `tvr lint` refuse a reviewer not assigned, as `tvr file`
  does, and `tvr note … --files` refuses a file of a role folder. Only a `file`, a `response` (its
  own folder) and a `freeze` event may name a folder as `dir`.
- **Folders recorded before the tool named them** (library issue L81). An early `file` or
  `response` line recorded its filing and report, or its response and notes, but not the folder,
  so a work file of that folder is recorded by no event and fails the `records` step and the
  certificate. `tvr freeze <kernel>` (the coordinator's; `--dry-run` shows what it would do) is the
  supported way to bring such a ledger to a passing state without editing a line: for each role
  folder whose own events all come from before the tool named folders, it appends one `freeze` event
  recording, with their sha256s, the work files (in a subfolder, such as `work/`) that those events
  leave out. The release check holds a `freeze` to what the tool writes: the coordinator's, of such a
  folder, once, after one of its own events, of work files only that they leave out. Every path a
  ledger records, in any event of either ledger, must be one the tool writes, normalised from TVA's
  root (no `.`, `..` or empty component, no leading `/`, no backslash), so a hand-written
  `<folder>/work/../filing-2.json` fails the `records` step (library issue L108). A filing,
  report, response or notes those events leave out is never frozen (the command refuses it): only
  its role's `tvr file` or `tvr respond` records it, with the events it gives. Work files give no
  event, so a freeze changes no finding, acceptance or digest; the coordinator runs it after judging
  that the files are the role's. Nothing shows that they were in the folder when its early line was
  written: as for every ledger line, the freeze is trusted as the coordinator keeps the ledger
  (library issue L82).
- **What a ledger shows.** That each recorded file is unchanged since its line, that the events
  follow from those files, and that every role folder, and every file in it, follows from its own
  events. Not that the lines
  themselves are unchanged: the ledgers are trusted as the coordinator keeps them, with no commit at
  each recording (user's decision, library issues L36, L82). An edited line, a ledger cut short, or
  a reviewer's folder removed with their lines, shows only against a copy kept elsewhere, compared
  by hand: a commit, or a kept `release.json`, which records each ledger's number of lines and
  sha256 (a later ledger whose first lines hash to that sha256 only grew). No step reads an earlier
  `release.json` back.
- **A new kernel**: `tvr new <kernel> --module <Module> --namespace <NS> --source <file>`.
- **Setting up a kernel folder** (library issue L116). The user and the coordinator ("setup" above:
  whoever creates `kernels/<kernel>/`) set up each kernel's folder, before any prover starts: `tvr
  new` copies the source into `source/` and pins its sha256. TVA verifies the kernel itself, not
  the autotuner (the user's decisions, 2026-10-29: "actually we do not neccessarily cover autotune.
  we do verification basically for the kernel." and "our principle is, we are verifying kernel
  itself. [...] actually not the prover picks a config, it is we pick a config. prover is not aware
  of the autotune."): `@triton.autotune` is outside the supported subset
  ([exclusions.md](exclusions.md)), and `Compiler.Faithful` covers a direct launch of the
  `@triton.jit` function, no call of an autotuned kernel ([rules/compiler-faithful.md](rules/compiler-faithful.md),
  "Which launch"). So, for a kernel a tutorial autotunes, the user and the coordinator pick one
  config of the tutorial's list at this step, and the source they give `tvr new` is the tutorial's
  with the `@triton.autotune` decorator removed, its launch passing that config's meta-parameters
  and launch options (`num_warps`, `num_stages`, …); where the config has a `pre_hook`, the host
  code does what the hook does before the launch (tutorials 06 and 09 set a host
  `TensorDescriptor`'s `block_shape`). They record in `SETUP.md`, in the kernel's folder, which
  config they chose and where it is in the tutorial (the file, the kernel, and the `triton.Config`
  or the list and position it comes from), and what they changed from the tutorial's source. The
  prover does not write `SETUP.md` or choose the config, is not told about autotuning, and proves
  the kernel as given in `source/`; a source that still has `@triton.autotune` is outside the
  subset, and the prover asks for the kernel to be set up again. The kernel reviewer judges the
  kernel as given in `source/`, not the choice of config, and files `UNSUPPORTED` on a source that
  still has `@triton.autotune`.

## 4. Claim items and digests

The claim splits into items. A reviewer accepts items, not files.

| Item | Roots | What the kernel reviewer judges |
|---|---|---|
| `model` | `model`, `launch` (and the source's sha256) | the transcription, the launch, the block shapes |
| `map` | `model-map.json`, the source, and the library files the model reaches | that each source line corresponds to its model text, and each drop is sound |
| `conditions` | `Inst`, `Inst.ok`, `Pre` | coverage of real use |
| `spec` | `Post`, `Spec` | what the outputs are; that a tolerance is meaningful |
| `assumption:<f>` | the statement of field `f` and its docstring | `[rule]`: the right library fact; `[kernel]` (proved through `consistent`): needed, clear, about what the write-up says |

Each item's digest covers its *cone*: every `TVA.*` and kernel constant its definitions reach,
by elaborated term (definition bodies included; theorems by statement only; a structure with every
field's projection), and the roots' docstrings. An assumption item's roots are the constants its
statement mentions, and its digest also holds the statement itself: the sha256 of the canonical
text (below) of the field's type, with `K` and `P` bound (library issue L68), so a change to a
literal, a binder or the statement's structure re-opens the item, not only a change to the
constants it mentions (`statement` in what changed). Each constant enters the digest as the sha256 of
its canonical text, which the claim reader writes out in full: its kind, name and universe
parameters, its type, its value if it is a definition, and the kind-specific data, with every
constant's name, universe level, literal and binder kind spelled out, each atom with its length or
a terminator, and bound variables as de Bruijn indices (library issue L48). So two constants that
differ in what a cone holds of them (a statement, a definition's value) in anything but the names
of bound variables differ in their digests, short of a sha256 collision (binder names, and the
metadata the kernel ignores, are left out: they change nothing a term means; a theorem's proof is
not held, as above). Digests are bookkeeping of what changed, so they
are complete by construction (library issue L36): an item's digest also holds, whole, every
library file its meaning rests on:
- the Lean file of every library module that holds a constant of the cone: its definitions,
  statements, docstrings, module docstrings and comments, all of it;
- the general fact pages, [SEMANTICS.md](SEMANTICS.md) and [exclusions.md](exclusions.md), when the
  cone reaches any library module;
- the rule page of every `TVA.Rules` module that holds a constant of the cone, and of every module
  a `TVA.Rules.*` name of the cone lies under ([rules/num.md](rules/num.md) for `TVA.Rules.Num`,
  with its list of the facts a kernel may cite;
  [rules/compiler-faithful.md](rules/compiler-faithful.md) for `TVA.Rules.Compiler`; a module in a
  subfolder, such as `TVA.Rules.Num.Exp2`, has the page of the nearest module above it; `RULE_PAGES`
  in `tools/tvr.py`, which `tvr audit` checks against the modules and pages). The gate's `kinds`
  check refuses a `TVA.Rules.*` constant of a cone declared outside `lean/TVA/Rules` (library issue
  L55);
- the versions of the packages Lean resolves (Mathlib's, from `lean/lake-manifest.json`) and
  Lean's version.

The `model` digest also covers the source's sha256, and the `map` digest is `model-map.json`, the
source's sha256 and the library files of the `model` item. [TRANSCRIPTION.md](TRANSCRIPTION.md) is
guidance and is in no digest. So:
- a kernel's proof edit, comment or reformatting changes no digest (its constants are hashed by
  elaborated term); any change to a library file an item reaches does, a proof edit or a comment
  included, since the file is hashed whole;
- a library fix re-opens every item whose cone reaches a changed file, and no other;
- `REVIEW.md` names what changed since a reviewer accepted: the constants (`TVA.tl.add`), the
  library files (`file:lean/TVA/Lang/Ops.lean`, `file:docs/SEMANTICS.md`) and the packages.

The claim reader (`tools/claim_reader.lean.in`) computes the cones from the elaborated Lean
claim. It never reads the kernel's source: there is no translator, and the model is the
prover's transcription.

## 5. Assumptions

Each field of `Assumptions K P` has a docstring that starts with its kind:

| Kind | What it is | Its statement may mention | Who stands behind it |
|---|---|---|---|
| `[rule]` | a fact the library states (`TVA.Rules.*`), applied to `P`, as the library states it | the library's constants, `P`, the kernel's `Inst`, `Inst.ok`, `model`, `launch`, and `K` in `Compiler.Faithful` only — no other kernel constant | the library (library review) |
| `[kernel]` | a fact about the kernel's own values, e.g. a property of an error constant | the kernel's constants; neither `K` nor `P` | Lean (through `consistent`), and the kernel reviewer for what it means |

A `[rule]` field's statement, under its `∀` and `→` binders, applies one of the facts the
library states to `P` (library issue L62):
- `Compiler.Faithful P (model x) (launch x) (K x)`, with the kernel's own `model` and `launch` and
  one variable `x` that the statement binds, the same in all three (`∀ d, d.ok → Faithful P
  (model d) (launch d) (K d)`). The rule relates the behaviour of a launch to the model of the
  instance launched: the model of another instance (`model false`, `model e`, `model
  (Classical.choose h)`), a launch or behaviour of another, or one written out, is refused, and so
  is a form the reader cannot match, such as `K d` written as a `fun`;
- a fact that [rules/num.md](rules/num.md)'s section "Facts the library states" lists, by a code
  span in the first column of its table (a name under `TVA.Rules.Num`), as it lists it. The gate
  reads only the section's first table, which ends at its first line that is not a table row and
  must come before any heading of the section, and `tvr audit` fails on a section with a second
  table, under a `###` subsection too, or with any form but blank lines, table rows, `###` headings
  and text, HTML or a `~` included, or on a page whose earlier lines could hide the section (a fence,
  an HTML block), which the gate then reads as listing no fact; the cells are split as GFM splits
  them (§2, library issue L95). A name alone
  (`ExpF32`, `DotF16`,
  `DivFull`, `MathTotal`) is cited with any arguments after `P` (a fact
  with its constants in its definition, of type `Platform → Prop`, takes none; a `…Total` rule
  holds for every argument); a name listed applied (`DivRn P .binary64`) is cited with exactly
  those arguments (`P` the platform, `.x` the constant `x` of the argument's type, as Lean's dot
  notation reads it).

So a constant becomes citable by every kernel only when the library lists it there, whatever its
type or name; without that section and its table no fact is (the gate fails closed, and `tvr audit`
names the page). The page is in the digest of every item that cites a fact of `TVA.Rules.Num`
(§4), so a change to the list re-opens those items. Never a shape whose constants the field would
choose (`MathAcc`, `MathAt`, `DivAcc`, `CvtAcc`, `CvtRn`, `DotAcc`, `DotScaledAcc`, `AtomicAddAcc`,
a family such as `DotF8Chunk`, `DivRn` on another format): such a field states a device fact of the
kernel's own, even with the library's constants written out. The kernel's own constants are every
constant of the statement that is not the library's (rooted at `TVA`) but `Inst`, `Inst.ok`,
`model` and `launch`: a `private` one, and one declared outside the kernel's namespace
(`_root_._x`), count too (library issue L63).

The gate checks the kinds and these limits (§8). So a kernel cannot slip its own constants into
a library rule, and a `[kernel]` assumption cannot be about the device or the compiler: the claim
reader finds `K` and `P` among a field's binders by their place, so the gate's `shape` check requires
`Assumptions` to be a `structure` (library issue L106) with exactly the two parameters
`(K : Claim.Behaviour Inst) (P : Platform)`, in this order, both explicit (library issue L94); the kernel reviewer, who reads the whole proof, checks the
structure's header too (REVIEWER_GUIDE §4). A
`[kernel]` assumption mentions neither `K` nor `P`, so it is a closed statement, and
`consistent` (`∃ K P, Total K ∧ Assumptions K P`) proves it: it is not unproved. The reviewer
judges whether it is needed and clearly stated, and whether the kernel constants it constrains
mean what the write-up says (library issue L56).

## 6. What a claim covers

`Spec K := Claim.Meets Inst.ok Pre Post K`: on every covered instance and input, every outcome
of a launch of the compiled kernel is a normal end whose memory satisfies `Post`. A launch is
*rejected* when Triton or the driver reports it to the host with an error before any program
runs, and a rejected launch has no outcome. Examples are too much shared memory, too many
registers, or a `tl.dot` below its minimum size. So a claim never promises that the kernel
compiles, and nobody asks for compile logs in kernel review (user, 2026-10-27). A launch that
returns without an error has an outcome, even if no program ran. The launcher skips a grid of
`2^31` to `2^32` programs without an error (its product of the grid's sizes wraps to a value
`≤ 0`), and a larger grid may run (`2^32 + 1` programs wrap to one, and every program runs);
`Launch.Valid`, which every claim needs for its covered instances, excludes both (fewer than
`2^31` programs). A call made on a stream that is capturing into a CUDA graph also returns without
an error and runs no program, its programs running at each replay of the graph, and a call made on
another stream during the capture runs at once: a claim covers no call made while a capture is in
progress, on any stream, and no replay (`Compiler.Faithful`'s scope 9; library issue L91).

## 7. The model map

`model-map.json` is `{"entries": [...]}`, one entry per source line or line range of the kernel:

```json
{"src": [153], "model": "let accumulator ← tl.dot_add cfg a b accumulator", "note": "a zero accumulator; the product has no other use"}
{"src": [189], "drop": "config", "note": "FP8_OUTPUT = False on every covered instance"}
```

- `model` is a string or a list of strings. Each must occur, up to whitespace, in the kernel's
  Lean files.
- `drop` is the prover's reason for leaving the line out, and `note` the condition that makes it
  hold. [TRANSCRIPTION.md](TRANSCRIPTION.md) §4 lists common reasons (`decl`, `comment`,
  `no-op:print`, `config`, `compile-check`, `dead`, …) as guidance; any clear reason will do, and
  the kernel reviewer judges whether the drop is sound by [SEMANTICS.md](SEMANTICS.md).
- `note` says what a reviewer needs to see the correspondence. `rules`, if given, is a list of
  free labels (such as the SEMANTICS.md sections an entry relies on); nothing checks them.
- The gate fails when a code line of the kernel's ranges is not covered, when model text is not
  found, or when an entry is not regular (no `src`, neither or both of `model` and `drop`, an
  empty reason). Whether each entry is right is the reviewer's judgment (library issue L36).
- **The ranges.** Every function the source defines, at any depth (every `def` and `lambda`, found
  with Python's `ast`, from its first decorator to its last line), must lie in the kernel lines
  (`source.ranges` in `kernel.json`), in the host lines (`source.host`), or, a `def`, be named in
  `source.unused`; a function under a decorator that names `jit` (`@triton.jit`, also under
  `@triton.autotune`) may not be in the host lines (library issue L50). The gate does not look for
  the spellings of `jit`: the launch runs only functions (a JIT function is made from a `def` of a
  Python file, however it is spelled: a decorator, an alias, a call `triton.jit(f)`; a
  `triton.constexpr_function` runs any Python function, a lambda included, at compile time, and the
  front end calls a plain function it reaches through a module), so
  every one is classed by the prover, and claim.md §2 lists the three classes. The reviewer judges
  the classes: a function the launch runs, on the device or at compile time, as the kernel or as
  anything it calls, lies in the kernel lines; host lines run only on the host; the body of an
  unused function is run by nothing; and the launch calls no function the source does not define
  but Triton's own. An unused function's `def` statement itself runs, though (next bullet).
- **Code outside every function** (library issue L79). Every code line that no function spans
  (an import, an assignment, a class's own lines, an `if __name__ == "__main__":` block) runs when
  Python executes the module, and so does the `def` statement of a function at the top level or in
  a class: Python evaluates its decorators and its parameters' defaults then (and Python 3.13 its
  annotations), whether or not anything calls the function, and they can bind a module-level name
  (`def unused(f=(double := triple))`). A JIT function reads what these bind: the front end resolves
  a name of the kernel to a module-level value (a module such as `tl` and its attributes, a JIT
  function, a `tl.constexpr(…)` global, a dtype, a namedtuple) at compile time, and the value is part
  of the compiled kernel. So each such line, and the header (its decorators and `def` lines, up to
  its body) of each function named in `source.unused` that no other function encloses, must lie in
  the kernel lines or the host lines (a function in either class has its header there already), and
  claim.md §2 lists both. The gate checks only that each such line is in one of the two classes: not
  which names a line binds, nor whether a line in the host lines (an unused function's header, a
  module-level assignment) binds or changes a name the kernel lines read. The reviewer judges the
  classes (user, 2026-10-28: the reviewer can challenge a malicious prover and reads the whole
  proof): every module-level name the launch reads lies in the kernel lines, where it is mapped, and
  the host lines bind nothing the launch reads, an unused function's header included. The
  supported subset admits only the imports of `triton` and of `triton.language as tl` and the
  module's `@triton.jit` functions as such names, each bound only by its own `import` or `def`
  ([SEMANTICS.md](SEMANTICS.md), "The supported subset"; library issue L88): a kernel that reads any
  other module-level value (a `tl.constexpr(…)` global, a dtype), or one that another statement
  rebinds, is `UNSUPPORTED`. So "every code line of the kernel lines is mapped" covers the launched
  kernel, every helper it calls and every module-level name it reads, as the module's own statements
  bind them. Code outside the kernel's file can still change them (another module's
  `kmod.double = kmod.triple`, an assignment to `tl.exp`): nothing in the gate sees it, and
  `Compiler.Faithful`'s scope 10 assumes it does not happen, in the scope's part (B), which nobody
  checks (library issues L104, L109; the user's decision, 2026-10-29: "So basically, we want to
  verify the kernel itself, rather than triton installation or other code programmed by users.").
- **Line numbers** count line feeds only, as Python's tokenizer, Triton and Lean's declaration
  ranges do, and claim.md quotes the source and the Lean declarations that way. A source or a
  kernel Lean file that holds another line break (a carriage return not before a line feed, a
  vertical tab, a form feed, `\x1c`–`\x1e`, `\x85`, U+2028, U+2029), which `str.splitlines` and
  some editors break at, fails the gate (library issue L51).
- A *code line* is one that holds a Python token, read with Python's own parser (`ast`,
  `tokenize`), other than a comment, a line break, indentation, or the string of a statement
  that is only a string literal (a docstring). A line with a triple-quoted string inside an
  expression is code, on every line the string spans (library issue L39). A source the parser
  cannot read fails the check.

## 8. The gate (`tvr check <kernel>`)

| Check | Passes when |
|---|---|
| `manifest` | `kernel.json`'s `lean.module` and `lean.namespace` are dotted names of ASCII letters, digits, `_` and `'`; `source.file` and `doc` are paths inside the kernel folder; the ranges are pairs of line numbers; `source.unused`, if given, is a list of function names (§7). The claim reader is Lean text built around the module and the namespace, so nothing else runs unless this passes (library issue L38); an `open` of a name that is not of that form fails `namespace`, and the claim reader must print exactly one result |
| `source` | the kernel source has the manifest's sha256 |
| `hygiene` | the library's code (comments and docstrings aside) has no proof escape, kernel or compiler bypass, meta code, meta type, handler attribute or option of those below (library issue L40); neither the library's code nor a kernel's Lean has a string literal holding `{` (Lean reads such a literal as a plain string or as an interpolated one, whose `{…}` is code with literals and comments of its own, depending on the syntax before it, so the comment reader could take a command after it for a comment; without one, the reader is exact, L40); a kernel's Lean has no line break but line feeds (§7, L51); and the kernel's Lean files, read whole (comments and strings included), contain no proof escape (`sorry`, `admit`, `native_decide`, `+native`, `ofReduceBool`, an `axiom` command), no kernel or compiler bypass (`implemented_by`, `extern`, `unsafe`, `set_option debug.*`, `addDecl…`, `doCheck`, `skipKernelTC`, `modifyEnv`, `setEnv`, `Environment`), no meta code (`#eval`, `#exit`, `run_cmd`, `run_elab`, `run_meta`, `run_tac`, `by_elab`, `eval%`, `elab`, `macro`, `syntax`, `initialize`, `register_…`, `include_str`, `simproc`, `delab`, …), no meta type (`IO`, `CoreM`, `MetaM`, `TermElabM`, `CommandElabM`, `TacticM`, `MacroM`, `Lean.Elab`, `Lean.Meta`, `Lean.Syntax`), no attribute that installs a handler (`tactic`, `init`, `norm_num`, `positivity`, `aesop`, `export`), and nothing that gives a name another meaning (L20): no notation (`notation`, `infix`, `infixl`, `infixr`, `prefix`, `postfix`, `binder_predicate`, `unif_hint`), no `open … renaming`, no `export`, no `local`/`scoped` instance, attribute or notation, no instance attribute or instance erasure (`attribute [-…]`), and no `set_option` other than resource limits, `autoImplicit`, `pp.*` (`pp.notation` included), `trace.*` and `linter.*`; the full list is `HYGIENE` in `tools/tvr.py` |
| `independence` | §2 |
| `build` | `lake build <module>` succeeds. It runs only after `hygiene` and `independence` pass, since building runs the kernel's Lean |
| `replay` | `leanchecker <root>` (Lean's own tool, in the toolchain) re-checks every declaration of the kernel's modules from the compiled files, with Lean's kernel, in the environment of their imports. A declaration that meta code added without the kernel's check fails here. The library's own modules are replayed the same way (`leanchecker TVA`) by the release check (§2, library issue L40) |
| `shape` | the claim reader elaborates: `main : ∀ K P, Assumptions K P → Spec K`, `consistent : Claim.Consistent Assumptions`, `satisfiable : Claim.Satisfiable Inst.ok Pre` (some covered instance and memory, library issue L71), `Spec K = Claim.Meets Inst.ok Pre Post K`; `Assumptions` is a `structure` (of another inductive type the reader would read no field, so the claim would show no assumption while a constructor held anything; the reader fails, and so does a reader that panics, on a line of its output other than its result line, which quotes the claim's names and docstrings, library issue L106, L112); every field of `Assumptions` is public (a `private` field's projection has another name than `<namespace>.Assumptions.<field>`, so the reader would list the field and read none of it: it refuses the claim with an error that names the field, library issue L112); and `Assumptions` has exactly the two parameters `(K : Claim.Behaviour Inst) (P : Platform)`, in this order, both explicit: the reader finds `K` and `P` among the binders of each field's projection by their place, so a parameter before them (an instance argument) would make `kinds` read other binders as `K` and `P`; any other parameter is refused too (library issue L94) |
| `axioms` | `main`, `consistent` and `satisfiable` depend only on `propext`, `Classical.choice`, `Quot.sound` |
| `usage` | `main` uses every assumption, through its projection (no destructuring of `Assumptions`) |
| `namespace` | every constant the kernel's modules declare is in the kernel's namespace and not in `TVA` (a name outside the namespace only if Lean made it, a name with a numeric component such as a macro scope's; a user-written `_x` or `_root_._x` is the kernel's, library issue L63); none has a name component `tl`, a component outside plain ASCII (letters, digits, `_`, `'`, `!`, `?`), or a name whose last components, as the kernel may write them, would otherwise mean another constant: through `TVA`, through any namespace whose name ends with one the kernel's files `open` (read over every line of the command), or at the root, an alias such as `min` included (`Kernels.K.tl.add`, `Kernels.K.Tensor.add`, `Kernels.K.Launch.mk`, `Kernels.K.min`); only the names Lean generates are exempt (constructors, recursors, matchers, internal names, and the auxiliary names of a structure or inductive type the kernel declares); the kernel declares no instance (but the `SizeOf` instance `<type>._sizeOf_inst` that Lean makes for a type the kernel declares; an instance named `_x` is the kernel's, L63); and no binder of a kernel definition or type is named like a library namespace that is not a definition (`tl`, `Tensor`, `Block`, …), or a look-alike of one (NFKC, and letters of other scripts read as Latin) (L20). The claim reader checks the elaborated environment, whatever the source syntax |
| `kinds` | every assumption is `[rule]` or `[kernel]` and keeps to its kind's limits (§5): a `[rule]` applies a fact the library states to `P`, `Compiler.Faithful` as `Faithful P (model x) (launch x) (K x)` for one bound `x`, or a fact [rules/num.md](rules/num.md)'s "Facts the library states" lists, as it lists it (L62), mentions no constant of the kernel's own but `Inst`, `Inst.ok`, `model` and `launch` (L63), and mentions `K` only in `Compiler.Faithful`; every `TVA.Rules.*` constant an item reaches is declared in a module under `lean/TVA/Rules`, where its rule page is (an equation lemma Lean generates for one aside, library issue L55) |
| `map` | §7: every code line of the ranges covered; every function of the source in the kernel lines, the host lines or `source.unused`, and none under a `jit` decorator in the host lines (L50); every code line outside every function, and the header of every function in `source.unused` that no other function encloses, in the kernel lines or the host lines (L79); the source's lines ended by line feeds only |
| `doc` | the write-up `README.md` has the sections Specification, Conditions, Assumptions, Proof |

`gate.json` also records what made it (library issue L53): the sha256 of every kernel file the gate
read (`kernel.json`, the source, `model-map.json`, the write-up, every Lean file), of the gate's
own code (`tools/tvr.py`, `tools/claim_reader.lean.in`), and of the build's other inputs, which
decide what the kernel's Lean elaborates to (`build`, library issue L69): `lean/lean-toolchain`,
`lean/lakefile.toml`, `lean/lake-manifest.json`, and every library file the kernel's modules
import, directly or through other library modules (one that holds no constant of any cone, such as
a module of notation, included). A gate is *stale* when any of these differs from the tree's now,
by contents (file times play no part), or when a library file or the packages its items' digests
hold differ (§9). A stale gate whose items' digests come out the same after `tvr check` re-opens
nothing.

**What a gate in the tree rests on** (library issue L82). `gate.json`, its snapshot and `claim.md`
are as the last `tvr check` wrote them, in its caller's environment (the `lake` that `TVR_LAKE`
names, the toolchain elan picks, the Mathlib of the local `lean/.lake/packages`), or as anyone
edited them since. Staleness compares the tree with what `gate.json` says it read, so a `gate.json`
written by hand, with the tree's sha256s, is current for `tvr` and for the certificate. Only the
release check recomputes them, in its clean sandbox from what TVA ships and Mathlib's upstream
source, and fails on a shipped gate, certificate or page that differs (§2). As for ledger lines
(§3), a gate in the tree is trusted as the coordinator keeps it until the release check runs.

The gate writes `gate.json` and `claim.md`:
1. the claim;
2. the source;
3. the conditions;
4. the specification;
5. the assumptions, each with its kind and the library fact it cites;
6. the model, with the rendered map;
7. the gate.

Declarations are quoted from their files by Lean's declaration ranges, with lines counted at line
feeds (§7).

## 9. Kernel review

### The flow

1. **The prover** makes the gate pass and asks for review.
2. **The coordinator** assigns reviewers: `tvr assign <kernel> <reviewer> [--focus model|claim|all]`.
   They start each agent with its role's prompt (`tvr prompt <kernel> reviewer <id>`, §12) and
   may record notes (`tvr note <kernel> <text> [--files …]`, never with a file of a role folder, §3).
3. **Round 1 (blind).** Each reviewer reads `claim.md`, the source, the Lean files, the write-up
   and the library pages the claim cites. They run `tvr start <kernel> reviewer <id>`, fill in
   `review/round-1/<reviewer>/filing.json` and `report.md` (and any `work/`), check them with
   `tvr lint <kernel> reviewer <id>` (§11), and record the filing with `tvr file <kernel> <reviewer>`. Until
   every assigned reviewer has filed, `REVIEW.md` and `tvr status` hide round 1's findings
   (`--as <reviewer>` shows one's own).
4. **The prover answers every open finding in one batch** (the tool refuses a partial batch):
   fix, re-run the gate, `tvr start <kernel> prover`, fill in `review/round-N/prover/response.json`
   and `notes.md`, check them with `tvr lint <kernel> prover` (§11), then run `tvr respond <kernel>`.
   This closes round `N`.
5. **Round N + 1.** Each reviewer re-reads what changed (`REVIEW.md` names the changed
   constants per item), then files in `review/round-<N+1>/<reviewer>/`: release settled
   findings, re-raise the others, raise new ones, accept items.
6. Steps 4 and 5 repeat until the certificate is granted or a finding is escalated.
7. **Escalation.** A finding re-raised twice goes to the user (`ESCALATED`). The coordinator
   writes a short memo with both positions (`tvr note`), and the user's decision is recorded
   with `tvr decide <kernel> <id> <text>`.

### Findings

| Kind | Target | Blocks | Meaning |
|---|---|---|---|
| `MODEL_MISMATCH` | `item:model`, `map:<line>`, `src:<line>`, `decl:<name>` | yes | the model does not mean what the source does, by SEMANTICS.md |
| `MAP_INCOMPLETE` | `item:map`, `map:<line>`, `src:<line>` | yes | a source line's map entry is missing, wrong, or drops what may not be dropped |
| `UNSUPPORTED` | `src:<line>`, `map:<line>` | yes | the source uses a construct whose meaning the library does not state, or states as open; resolved in the library |
| `CONDITION_TOO_NARROW` | `item:conditions`, `decl:<name>` | yes | `Inst.ok`/`Pre` exclude real use |
| `CONDITION_WRONG` | `item:conditions`, `decl:<name>` | yes | unsatisfiable, or not what the write-up says |
| `SPEC_TOO_WEAK` | `item:spec`, `decl:<name>` | yes | `Post` does not say what the kernel is for, or a tolerance is meaningless at covered sizes |
| `SPEC_WRONG` | `item:spec`, `decl:<name>` | yes | `Post` says something other than the kernel's purpose |
| `RULE_MISCITED` | `item:assumption:<f>`, `[rule]` only | yes | not the library's fact for the operation the model uses |
| `ASSUMPTION_FALSE`, `ASSUMPTION_TOO_STRONG`, `ASSUMPTION_UNCLEAR` | `item:assumption:<f>`, `[kernel]` only | yes | the kernel's own assumption is false of what the write-up says it is about (its statement, which `consistent` proves, says something else), stronger than needed, or unclear |
| `EVIDENCE_REQUEST` | `item:assumption:<f>`, `[kernel]` only | yes | an argument for the kernel's own assumption: that it is needed, or that the constants it constrains mean what the write-up says (Lean proves the statement itself, through `consistent`) |
| `LIBRARY_REPORT` | `decl:<TVA name>`, `lib:<path>`, `src:<line>` | no | a doubt about the library; listed in `docs/library-review/REPORTS.md` for triage |
| `DOC`, `QUESTION` | any, or `doc:<path>` | no | wording, or a question |

- `decl:<name>` must be a constant in some item's cone. `src:`/`map:` lines must be in the
  manifest's ranges. `lib:`/`doc:` paths must exist.
- A finding is *about* some items:
  - an `item:` target names one;
  - `map:`/`src:` lines are about `model` and `map`;
  - a `decl:` target is about the items whose cone contains the constant;
  - any finding may list the items it `affects` (`"affects": ["item:model"]`).
- A reviewer cannot accept an item while one of their own blocking findings about it is
  unresolved.

**States**:
- a raised finding is `OPEN`;
- the prover's response makes it `ANSWERED`, or `REFERRED` if the response refers it to the
  library;
- a re-raise by a raiser makes it `OPEN` again;
- it is `CLOSED` once every raiser has released it, or by the user's decision;
- after the second re-raise it is `ESCALATED`.

`REFERRED` does not block.

**Responses**:

| Response | Meaning | What the tool checks |
|---|---|---|
| `REVISED` | the kernel changed | each item in `changed` has a different digest than when the finding was last raised |
| `DISPUTED` | an argument; nothing changed | |
| `CLARIFIED` | wording or the write-up only | |
| `DEFERRED` | kept as a stated narrowing or limitation | the reviewer agrees by releasing |
| `REFERRED` | the finding is about the library: it becomes library issue `L<n>`, named in the text | the issue exists in `ISSUES.md`; the finding stops blocking; the reviewer may re-raise it if it is about the kernel after all |

### Filings and responses

A reviewer's filing, `review/round-N/<reviewer>/filing.json` (recorded by
`tvr file <kernel> <reviewer>`):

```json
{"reviewer": "r-gemm",
 "findings": [{"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "…", "detail": "…",
               "evidence": "…", "blocking": true}],
 "release": [{"id": "F2", "note": "the revision settles it"}],
 "reraise": [{"id": "F3", "note": "the response does not address …"}],
 "accept": ["item:conditions", "item:assumption:dotTotal"],
 "notes": "non-blocking remarks"}
```

- `accept` may be `"all"`. Acceptance records each item's current digest, and it goes `STALE`
  when the digest changes.
- Only an `ANSWERED` or `REFERRED` finding can be re-raised.

The prover's responses, `review/round-N/prover/response.json` (recorded by `tvr respond <kernel>`):

```json
{"responses": [{"id": "F1", "response": "REVISED", "text": "…", "changed": ["item:spec"]}]}
```

The tool refuses three things:
- a stale gate (§8): one whose recorded sha256 of a kernel file (the manifest, the source, the map,
  the write-up, a Lean file), of the gate's own code (library issue L53) or of the build's other
  inputs (library issue L69) differs from the tree's now, or one whose items' library files or
  packages (recorded in the gate's snapshot) differ from the tree's now, as after a library fix
  (library issue L42). A gate recorded before round 13 kept no sha256 of the build's other inputs,
  and is stale;
- a response before every reviewer has filed in the round;
- a batch that leaves an open finding unanswered.

### The certificate

The certificate is granted, and `review/certificate.json` written, when all of these hold:
- the ledger holds to its records (§3, library issues L52, L81): every recorded file as it was
  recorded, every event following from its recorded filing or response, and every role folder of
  `review/`, and every file in it, following from its own events, the current round's included (no
  filing or response left unrecorded);
- the gate passes, and is not stale (above): after a change to a kernel file, to the gate's code,
  to the build's inputs, or to a library file that reaches the kernel's items, the certificate
  lapses until `tvr check` runs again and the reviewers re-accept the items whose digests changed
  (after a change that leaves the digests as they were, such as one to the gate's code, `tvr check`
  alone restores the certificate); meanwhile `REVIEW.md` and `kernels/README.md` mark the gate
  stale;
- every assigned reviewer has accepted every item at its current digest;
- no blocking finding is unresolved.

It holds under the library as it stands. `REVIEW.md` lists the open library issues that touch the
kernel, and a library fix re-opens the affected items. In the tree, the certificate rests on the
gate as the last `tvr check` wrote it and on the ledger as the coordinator keeps it (§3, §8,
library issue L82); the release check recomputes the gate and regenerates the certificate in its
sandbox, and fails on a shipped one that differs.

## 10. Library issues and library review

```
docs/library-review/
  ISSUES.md               coordinator   the library issues: L<n>, state, source, summary, resolution, constants touched
  REPORTS.md              tvr           every LIBRARY_REPORT and REFERRED finding from kernel review (generated)
  ledger.jsonl            tvr           every recorded role folder of a round, with each file's sha256 (append only)
  round-<N>/author/notes.md
                          author        a batch of fixes: what changed, why, the evidence
  round-<N>/<reviewer>/report.md
                          reviewer      the library reviewer's verdict on each issue in the batch
evidence/<topic>/         author        README.md, the script, its output
```

- **Recording** (library issue L52). When a round is done, before `ISSUES.md` moves its issues
  on, the coordinator records each role folder of the round with `tvr record library <N> <who>`:
  it checks the folder as `tvr lint library <N> <who>` does, and appends to `ledger.jsonl` the
  sha256 of every file in it (the notes or the report, and every work file). A recorded folder
  is frozen: `tvr record` refuses it a second time, and the release check fails on a recorded file
  that changed or went missing, on a file recorded with two sha256s, on a file added to a recorded
  folder, on a folder recorded twice, on a line that records no role folder as `tvr record` writes
  it, and on a role folder of a round before the last that is not recorded. Rounds 1 to 7, which
  ended before the ledger existed, were recorded with `--no-lint` (their format was checked when
  they were written, and `ISSUES.md` no longer lists their issues under their rounds); `tvr record`
  refuses `--no-lint` for a later round, and the release check fails on a later round's line that
  says it was not linted. As for a kernel's ledger (§3), the ledger shows that each recorded file is
  unchanged since its line, not that its lines are: the lines are trusted as the coordinator keeps
  them, with no commit at each recording (user's decision, library issues L36, L82), and an edited
  or removed line shows only against a copy kept elsewhere (a commit, or a kept `release.json`'s
  `ledgers`), compared by hand.

- **Sources of issues.** An issue comes from a kernel finding (`LIBRARY_REPORT`, or a response
  `REFERRED`), from library review, or from the author. The coordinator records it in
  `ISSUES.md` with its source, for example `vec_add_f32#F2`.
- **States.** An issue is `OPEN`, `FIXED` (in a batch, with the constants it changed), `ACCEPTED`
  (a library reviewer accepted the fix), or `DECLINED` (with the reason, and the user's decision
  where the scope changes).
- **Fixes.** The author fixes issues in batches (one round each) with evidence, and a library
  reviewer who did not write the fix checks the batch.
- **Effect on kernels.** Each kernel item's digest covers the library constants it reaches, so
  a fix re-opens exactly the affected kernel items. Their reviewers re-accept them in the next
  kernel round.

## 11. Formats

Every output of a role is regular: the tool parses it and refuses it otherwise.
- `tvr start` writes the skeleton.
- `tvr lint` checks it with the same parser `tvr file` and `tvr respond` run. So each role checks
  its own output and fixes it until `tvr lint` prints `regular`, then records it.
- The parser also rejects placeholders (`TODO`) and references to other projects (§2).

**Kernel reviewer: `review/round-N/<id>/filing.json`.**

```json
{"reviewer": "<id>", "kernel": "<kernel>", "round": 1,
 "findings": [{"kind": "<KIND>", "target": "<target>", "summary": "<one line, at most 200 characters>",
               "detail": "<required for a blocking kind>", "evidence": "<optional>",
               "affects": ["item:<name>"]}],
 "release": [{"id": "F<n>", "note": "<why>"}],
 "reraise": [{"id": "F<n>", "note": "<why>"}],
 "accept": "all",
 "notes": "<optional>"}
```

- There are no other keys.
- `kind` is one of §9.
- Whether a finding blocks follows from its kind; `blocking`, if given, must agree.
- `accept` is `"all"` or a list of `item:<name>`.

**Kernel reviewer: `report.md`.**

```
# Review of <kernel>, round <N>, by <id>

## Verdict

accept | accept-partly | reject
<why, in a paragraph>

## Findings

- **<KIND>** `<target>` — <one line>        one per new finding, in the filing's order
- **release F<n>** — <why>
- **re-raise F<n>** — <why>
                                             or: None.
## Accepted

- `item:<name>`                              one per accepted item; or: All items. / None.

## Checked

- <what you checked>                         at least one

## Not checked

- <what you could not check>                 or: Nothing.
```

The verdict must agree with the filing:
- `accept`: every item accepted, no new blocking finding, no re-raise;
- `reject`: no item accepted;
- `accept-partly`: the rest.

**Kernel prover: `review/round-N/prover/response.json`.**

```json
{"kernel": "<kernel>", "round": 1,
 "responses": [{"id": "F<n>", "response": "REVISED", "text": "<…>", "changed": ["item:<name>"]},
               {"id": "F<m>", "response": "REFERRED", "text": "<…>", "issue": "L<k>"}]}
```

- Exactly one response per open finding.
- `changed` comes with `REVISED`, always, and with nothing else.
- `issue` comes with `REFERRED`, always, and with nothing else. It names an issue of `ISSUES.md`.

**Kernel prover: `notes.md`.**

```
# Prover's notes on <kernel>, round <N>

## Changes

- <what changed and why>                     or: None.

## Responses

- **F<n>** <RESPONSE> — <one line>           one per response, in order

## Library requests

- <what the library should state, and why>   or: None.

## Gate

PASS, digest `<the gate's 16-digit digest>`
```

**Library author: `docs/library-review/round-N/author/notes.md`.**

```
# Library batch, round <N>: author's notes

## L<n>: <title>                              one per issue of the round, in order

- **What changed.** …
- **Why.** …
- **Evidence.** …                             when there is any
- **Constants whose meaning changed.** …
```

**Library reviewer: `docs/library-review/round-N/<id>/report.md`.**

```
# Library review, round <N>, by <id>

## Verdicts

- **L<n>** ACCEPT — <reason>                 one per issue of the round
- **L<m>** REOPEN — <what is wrong, and what would fix it>

## New issues

- **NEW** <class> — <summary>                 or: None. <class>: soundness, gap or wording
                                              (docs/LIBRARY_REVIEW.md)

## Re-ran

- <evidence re-run, tests added>              or: Nothing.

## Not checked

- <…>                                         or: Nothing.
```

**Coverage auditor: `docs/library-review/round-N/coverage/matrix.md`** (issue L37).

```
# Tutorial coverage, round <N>

## <tutorial file>: <function>                one per function of tutorials/inventory.json, in order

| Construct | Status | Where |
|---|---|---|
| `tl.load` with a mask and `other` | covered | SEMANTICS "tl.load" |
| a runtime `if` with an early `return` | gap | L28 |
| <what is missing> | gap | NEW: <what a proof needs> |
| <construct> | excluded | L37 |
                                             or, for a whole function the user excluded:
Excluded: L37 (user decision).

## New gaps

- **NEW** gap — <summary>                     one per `NEW:` row; or: None.
```

- The status is `covered`, `gap` or `excluded`.
- A gap cites an open issue (`L<n>`) or starts with `NEW:`.
- An exclusion cites its decision (`L<n>` or the `exclusions.md` entry).

## 12. Prompts

Each agent starts from its role's prompt in [docs/prompts/](prompts/):

| Prompt | Role | Fills in |
|---|---|---|
| [kernel-prover.md](prompts/kernel-prover.md) | kernel prover | the kernel, the round, what to answer |
| [kernel-reviewer.md](prompts/kernel-reviewer.md) | kernel reviewer | the kernel, the reviewer id, the round |
| [library-author.md](prompts/library-author.md) | library author | the issues of the batch |
| [library-reviewer.md](prompts/library-reviewer.md) | library reviewer | the batch (round), the reviewer id |

`tvr prompt <kernel> prover|reviewer [<id>]` prints a kernel prompt with the kernel's name, the
current round and the reviewer's id filled in. The coordinator adds nothing about the content of
the claim: what the agent reads is in the kernel's folder and the library, and an agent handed
the coordinator's view of the claim reviews that view instead of the claim.

## 13. Why it is built this way

- **Scope.** Kernel reviewers trust a separately reviewed library, so device and compiler
  knowledge is checked once, in library review, not again for every kernel.
- **Per-item digests computed in Lean.** An acceptance survives proof edits and changes to other
  items, and a library change re-opens only what it reaches. The claim reader writes each constant,
  and each assumption's statement, out in full and `tvr` hashes it with sha256, so no two different
  meanings share a digest.
- **One ledger and one folder per kernel.** Findings have explicit states, the tool enforces
  the transitions, and `REVIEW.md` renders the whole record beside the kernel's files. Every
  recorded file is kept by its sha256, in the kernel's ledger or the library's, every event of a
  kernel's ledger follows from a recorded file, and the certificate and the release check hold the
  records to them.
- **Checked by Lean and by the gate.** The claim's shape is checked by Lean, not by text
  matching, and every declaration of the kernel is re-checked by Lean's kernel from the compiled
  files (`replay`), and every declaration of the library by the release check, so meta code cannot
  slip in an unchecked theorem. The model map is checked for coverage, and the assumption kinds for
  ownership.
- **Judgment where it is needed, mechanics where they are complete** (user, 2026-10-28: "assume
  the provers and reviewers have intelligence and properly leverage that"; library issue L36).
  The tool checks what it can check completely or cheaply, or what a reader cannot see: the
  build, `replay` and the axioms; the imports, the lake configuration and the release check; the
  claim's shape; regular output formats; digests of whole library files; and the name-resolution
  backstop of `namespace` and `hygiene`. Whether the model means what the kernel does, whether a
  drop is sound, and what every name means are judged by reviewers who read the source, the
  kernel's whole Lean and the library's facts. A denylist that stood in for that judgment would
  miss what it does not list, and give a false sense that the rest was checked.
- **Fixed prompts and regular outputs.** Every agent of a role starts from the same prompt,
  which carries the scope. Every output has a fixed format that the tool parses, and each role
  checks its own output with `tvr lint` before recording it.
- **Independence.** TVA's results come from TVA and Mathlib, and its evidence is its own. Lean's
  imports, the release check and the text scan enforce this.
