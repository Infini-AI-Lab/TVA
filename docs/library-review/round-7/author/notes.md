# Library batch, round 7: author's notes

Issues: L7, L31, L36, L38, L39, L40, L41, L42, L43, L46 ([ISSUES.md](../../ISSUES.md)), 2026-10-28.
Most of the batch is the tool (`tools/tvr.py`). L7 and L36 are one fix, the release check; L38–L43 are
the gaps round 6 found in the gate, the audit and the release check; L31 and L46 are wording.

The principle (user, through the coordinator): "assume the provers and reviewers have intelligence and
properly leverage that". Mechanical checks stay where they are complete, cheap, or invisible to a
reader, and the release check is such a place: its sandbox is now complete by construction, built from
an allowlist (what TVA ships, plus an explicit read-only list of what the toolchain needs), not by
hiding more folders. Independence stays blocking.

The library builds (`lake build TVA`); no Lean definition, theorem statement or proof changed: the Lean
changes are docstrings (L31, L46). `python3 -I tools/tvr.py audit` prints "independent".
The release check (`python3 -I tools/tvr.py release --dir <empty folder>`, rebuilt this round) on
the tree as it stands passes every step but one: the gate of `kernels/vec_add_f32`, which fails at its
own build, as in the tree, and whose shipped `gate.json` and certificate are stale (below; release
check R0 in [evidence/release-check](../../../../evidence/release-check/README.md)). With that kernel
brought up to date in a copy, every step passes (R3).

**Kernel that does not build or pass the gate:** `kernels/vec_add_f32`, as before this round (its
prover's to fix; I did not edit the kernel folder or run `tvr check` on it). Two things are new about
it. (1) Its shipped `gate.json` (passing, 2026-10-27) and `review/certificate.json` (granted) predate
whole-file digests (round 6) and the library it no longer builds against, and the tool now says so:
the gate is stale (its snapshot records no library files, L42), so `REVIEW.md` and
`kernels/README.md` will mark it, and `certificate.json` will be removed, at the kernel's next refresh
(any `tvr` command on it, or `tvr index`). I ran none, since that writes the kernel folder; the
coordinator may. (2) The release check fails its gate step on the mismatch as well as on the build. Brought up to date in a copy (the two updates of
[evidence/gate-digests](../../../../evidence/gate-digests/README.md)), it passes the round-7 gate
([evidence/gate-round7](../../../../evidence/gate-round7/README.md) C5) and the release check (below,
R3).

**Records of other roles.** I edited none, and no check flags text in one.

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed** (with L36, one fix): the release check's sandbox is built from an allowlist
  (`cmd_release`, `make_packages`, `system_python`, `system_problems`, `elan_parts`, `shipped_problems`
  in `tools/tvr.py`; protocol §2 "Enforcement"). With `bwrap`, the sandbox's root is empty, and it shows
  only:
  - **TVA as shipped**: exactly the files git tracks plus the untracked files it does not ignore,
    copied into an empty home directory at TVA's own path. A shipped file inside a folder the text scan
    skips fails the `shipped` step (L41);
  - **Mathlib and its eight dependencies, made afresh from the pinned commits** (lr-6a's second route):
    for each package of `lean/lake-manifest.json`, a new repository (no template, hooks off) fetches
    exactly the pinned commit, from the working tree's checkout of the package if it has it, else from
    the manifest's URL, and checks it out. Git checks every object it receives against its hash, so
    the package holds exactly that commit's files: none of the working tree's ignored files (Mathlib's
    `**/__pycache__/`, `.git/info/exclude`), untracked files or build outputs exist in the sandbox, and
    Mathlib is built there from its source. This answers lr-6a's whole route, ignored source files and
    borrowed `.olean` files alike, by construction rather than by inspecting the working tree's
    checkout. The cost is a build of Mathlib in each release check: 19 minutes on this machine's 96
    cores and 13 GB (measured; the check refuses a folder with less than 16 GB free);
  - **the system directories** `/usr` and `/etc`, and `/bin`, `/sbin`, `/lib`, `/lib64` as the links into
    `/usr` they are here, read-only. The check fails if anything in them (links aside) is writable by the
    user or owned by a regular user or group (`find … -uid +999 -o -gid +999 -o -writable`, 2 s here,
    nothing found), so nothing can have been placed in them for the check to find. `/var`, `/opt`,
    `/scratch`, `/data`, `/srv`, `/mnt`, `/afs`, `/media`, `/root` and the rest do not exist inside
    (lr-6a's first route);
  - **the Lean toolchain** that `lean/lean-toolchain` pins: elan's `bin`, its `settings.toml` and that one
    toolchain folder, read-only and trusted as installed, as Lean itself is. It can hold no module TVA
    may import: the import check allows only the roots `TVA` and `Mathlib` (and a kernel's own), whose
    modules come from TVA's copy and Mathlib's pinned source;
  - **the system's Python**: the tool runs inside on a `/usr/bin/python3.N` with `N ≥ 11` (here
    `/usr/bin/python3.12`) with `-I -S`, so its standard library is root's, and no user's file is
    importable but the copy's own. The base prefix of the Python that started the check (an install
    under the home directory, with its site-packages and environments) is no longer shown (lr-6a: "`-I
    -S` holds only until the code extends `sys.path`"). The venv and the Triton cache are not shown
    either: nothing the release check runs uses Triton;
  - `/dev`, `/proc` and an empty `/tmp`; no network.

  The sandbox must start, or the check fails: a run probes it with the system Python first, and
  without `bwrap` the check fails instead of falling back. `--no-mask` still runs a HOME-only check for
  testing, and it fails its `sandbox` step. The text scan's tests are unchanged (not extended spelling by
  spelling, as L36 decided).
- **Why.** lr-6a reopened L7 for the release check only: the sandbox hid `/home` and `/tmp` and showed
  the rest of the root, the Python prefix, and package trees whose ignored files the `packages` step
  could not see. ISSUES.md (round 7): "Make the sandbox show only what TVA ships plus an explicit list
  of system paths the toolchain needs (read-only), and check package trees including ignored files, or
  rebuild them." An allowlist does not depend on knowing where else a file could be; rebuilding the
  packages from their commits does not depend on inspecting the checkout.
- **Evidence.** [evidence/release-check](../../../../evidence/release-check/README.md): four runs, each building Mathlib from source in its own
  sandbox. R0, the tree, passes the shipped, packages, sandbox, build, library, lint and audit steps
  (`vec_add_f32`'s gate fails, as above). R1 (a), lr-6a's route (`tvr audit` reads a file in a fresh
  folder of the system's second temporary directory at a path computed at run time) and (b) (`tvr audit`
  puts the site-packages folder of the Python that ran the script on `sys.path` and imports `IPython`):
  outside the sandbox the copy's audit passes; inside, it fails with both probes. R2, lr-6a's package
  route (Mathlib's checkout holds `Mathlib/__pycache__/Lemmas.lean`, which its `.gitignore` ignores, and
  `lean/TVA.lean` imports it): outside, TVA's import check accepts it and the round-6 `packages` test sees
  nothing; inside, the build fails, "bad import 'Mathlib.__pycache__.Lemmas'". R3: with `vec_add_f32`
  up to date, every step passes. The sandbox shows the copy, `/usr`, `/etc` and the links into `/usr`,
  the toolchain, `/dev`, `/proc` and an empty `/tmp`, and nothing else (no `/var`, no other home folder).
  The text scan did not change, so
  [evidence/independence-audit](../../../../evidence/independence-audit/README.md) stands (its script takes
  the other project's name, so I did not rerun it).
- **Constants whose meaning changed.** None (the tool and the role documents).

## L31: Stale hint text, `[no-op:debug_barrier]`'s reason, `[dot-add]`'s "constant zero"

- **What changed.** lr-6b found the withdrawn hint reading still in Lean: `lean/TVA/Lang/Ops.lean`,
  the module docstring section "Compiler hints", said a hint on a loop's induction variable or carried
  variable has no effect and the model drops it; the docstrings of `tl.multiple_of` and
  `tl.multiple_ofN` defined `contig` as "the contiguity Triton assigns" and had "the reviewer checks
  it". Now:
  - the "Compiler hints" section keeps what holds (a hint is a check, an attribute of the value, placed
    where the value is defined), says that a hint inside a loop body on a carried name or the induction
    variable is open (L1), and names the earlier reading as withdrawn with the reason (the front end's
    dry run of the body marks the pre-loop value; evidence/hints-dry-run), as SEMANTICS.md does;
  - it says that what a per-dimension value promises at rank ≥ 2, and what a `contig` above 1 promises,
    are open (L2), and that `contig = 1` is the only reading the library gives in terms of the values;
  - `tl.multiple_of`'s docstring: `contig = 1` (the default) means every element is a multiple of `v`,
    the library's reading; a larger `contig` is open (L2), and no kernel relies on one until the library
    states it. Nothing asks a kernel reviewer to check Triton's contiguity any more;
  - `tl.multiple_ofN`'s docstring: the same, per dimension, describing the definition (the elements
    whose coordinate `k` is a multiple of `contig[k]`) and the open L2.

  `docs/review-questions.md` (a record, in no digest) marks its round-2 note "`contig` must divide
  Triton's contiguity" as withdrawn and ties Q9 to L2. The other two items of L31 (`debug_barrier`'s
  reason, `dot_add`'s `±0`) were accepted in round 6 and are unchanged.
- **Why.** The library stated two readings of the same hint; the Lean one could lead a prover to drop a
  hint the device relies on (evidence/hints-dry-run: the kernel faults).
- **Evidence.** [evidence/hints-dry-run](../../../../evidence/hints-dry-run/README.md) (unchanged); no new
  measurement, since nothing new is asserted: the change withdraws a reading and states two questions as
  open.
- **Constants whose meaning changed** (docstrings only): `TVA.tl.multiple_of`, `TVA.tl.multiple_ofN`, and
  the "Compiler hints" module docstring of `lean/TVA/Lang/Ops.lean`. No definition changed.

## L36: Leverage judgment: mechanical checks only where complete; transcription as guidance; advisory scans

- **What changed.** lr-6a reopened L36 only for the release check, and L36's release check is L7's fix
  (above): the sandbox is an allowlist, the packages are made from their pinned commits and built in it,
  the tool runs on the system's Python, and the check also re-checks the library (L40) and compares each
  recomputed gate with the shipped one (L42). Protocol §2 says what the sandbox shows and what it runs,
  step by step, and that evidence and work scripts are not run in it (L46). The rest of L36 (whole-file
  digests, guidance, the map check, prompts) was accepted by both reviewers in round 6 and is unchanged.
  **The cost, and the choice: Mathlib built from source in each run** (the coordinator asked me to
  weigh it against providing Mathlib's built files read-only, pinned by the manifest's revision and
  hashed). Measured ([evidence/release-check](../../../../evidence/release-check/README.md)): a run takes
  17–19 minutes on this machine's 96 cores when it runs alone (about 25–35 with two at once) and
  13 GB, which the check now removes after the run unless `--keep` (`release.json` stays). It runs once
  per role per round, one at a time. Builds from the pinned source gave the same 54186 `.olean` files
  bit for bit in three runs (R1, R2, R3: copies at paths of the same length), but another digest in a
  fourth (R0, TVA at its own path), and the build folders as a whole differed every time. I kept the
  build from source this round, because:
  - a hash pins a build only as well as its anchor. A pinned digest of Mathlib's build means "what the
    pinned source gives" only if someone built that source and recorded it, and a reviewer can check
    the record only by building it again and comparing. That needs a reproducible build, and the
    measurement shows it is not, in general: the `.olean` files changed with the copy's path (R0
    against R1–R3), and the files a run would borrow are more than the `.olean` files (Lean also runs
    Mathlib's compiled elaborator code, its notation, macros and tactics, while it elaborates TVA's
    statements). Borrowing an unverified build is the route L7 closes (lr-6a: "a built `.olean` that is
    not what Mathlib's source gives");
  - the toolchain is not hashed, for comparison: it is trusted as installed, as Lean itself is
    (protocol §2), and it can supply no module TVA imports. Mathlib's build outputs are what TVA
    imports, so "trusted as installed" would trust exactly the files of the route;
  - the cost falls on rounds, not on kernels, and is bounded: one 20-minute run per role per round,
    13 GB at a time, removed afterwards.

  A follow-up that would keep the check complete and cut its cost, if it can be made sound: build at a
  fixed path, establish which build files Lean and Lake read when TVA imports Mathlib and that those are
  reproducible there, pin their digest (computed by a from-source run) in a shipped file, and let a run
  borrow a cached build whose digest matches the pin, with `--rebuild` re-deriving it. That is one more
  mechanism to review and an open measurement, so it is the coordinator's and the user's call; I did
  not build it.
- **Why.** As L7. The release check is where a mechanical check is complete by construction, so it has
  to be complete: the user's "we need to keep the project independent. since tva will be released
  without others."
- **Evidence.** As L7.
- **Constants whose meaning changed.** None.

## L38: soundness: a kernel's `kernel.json` namespace is pasted unchecked into the claim reader

- **What changed.**
  - The gate's first check is now `manifest` (`manifest_problems`): `lean.module` and `lean.namespace`
    must be dotted names of ASCII letters, digits, `_` and `'`
    (`[A-Za-z_][A-Za-z0-9_']*(\.[A-Za-z_][A-Za-z0-9_']*)*`), `source.file` and `doc` paths inside the
    kernel folder, and the ranges pairs of line numbers. When it fails, nothing else runs: no build, no
    claim reader written, and `gate.json` records the failure. Such a name can be nothing but a name in
    the reader's Lean text (no space, bracket, `:=`, line break, `#` or `-`).
  - The other text spliced into the reader, the names the kernel's files `open`, is spliced only when it
    has the same form; an `open` of any other name fails the `namespace` check.
  - The claim reader must print exactly one `TVR_JSON` line, or `shape` fails.
  - `tvr new` refuses a module or namespace that is not of that form.
  - Protocol §8: the `manifest` row.
- **Why.** lr-6a: a crafted namespace closed the reader's first `example`, printed an earlier honest
  JSON line and ended the file with `#exit`, and `main : True` passed every check. Checking the form
  before splicing makes the substitution unable to change what the reader checks, whatever the
  manifest says; the reader still elaborates the claim and computes everything else itself.
- **Evidence.** [evidence/gate-round7](../../../../evidence/gate-round7/README.md): U2 (lr-6a's
  namespace, a module with a line break, a namespace with a space and a source file outside the kernel
  folder are refused; the real manifest passes) and C4 (lr-6a's injection end to end on a copy: the gate
  fails at `manifest`, and no claim reader is written).
- **Constants whose meaning changed.** None.

## L39: gap: the map coverage check skips lines with `"""` and the lines between two of them

- **What changed.** The map check's code lines come from Python's own parser (`code_lines`): a line is
  code when it holds a token (`tokenize`) other than a comment, a line break or indentation, except the
  string tokens of a statement that is only a string literal (`ast`: a docstring, or a bare string,
  which runs nothing). A triple-quoted string inside an expression is code on every line it spans. `ast`
  counts columns in UTF-8 bytes and `tokenize` in characters, and the function converts them. A source
  the parser cannot read fails the map check. Protocol §7 defines a code line this way.
- **Why.** lr-6a and lr-6b: a line holding `"""` anywhere, and every line between two such lines, was
  treated as a docstring, so `tl.device_print("""x""", tl.atomic_add(c, 1))`, or an atomic between two
  comments holding `"""`, needed no map entry.
- **Evidence.** [evidence/gate-round7](../../../../evidence/gate-round7/README.md) U1: on lr-6a's and
  lr-6b's cases and two more (a two-line string argument; a bare non-ASCII string followed by code), the
  round-6 loop missed every such line and `code_lines` counts each as code, while the docstring and a
  bare string statement stay exempt; a source that does not parse gives no code lines.
- **Constants whose meaning changed.** None.

## L40: gap: library modules are not replayed with `leanchecker`; any `leanOptions` key is accepted

- **What changed.**
  - **Replay.** `tvr library-check` (new) runs the library's hygiene and `leanchecker TVA`, which
    re-checks every declaration of every library module with Lean's kernel from the compiled files. The
    release check runs it after `lake build TVA`, inside the sandbox (step `library`), so every library
    declaration is re-checked on each release check, from a build made there. `replay` and the new
    command share `leanchecker(root, what)`.
  - **Library hygiene** (`LIBRARY_HYGIENE`, `library_hygiene_problems`): the kernel hygiene list's proof
    escapes, kernel and compiler bypasses, meta code, meta types, handler attributes and options, over
    the code of every library Lean file (comments and docstrings removed, since the library's prose names
    these things; notation and instances stay allowed, the library being where names get their meaning).
    The gate's `hygiene` check includes it, and so does `tvr library-check`. The comment stripper
    (`_uncommented`) now reads string literals (raw ones too), character literals and `«…»` names as
    Lean's lexer does, so a `--` or `/-` inside one cannot hide a command after it; the header reader's
    test for a late `import` and `opened_names` use it too. The library as it is has nothing flagged.
  - **Lake options.** `lake_violations` allows in `leanOptions` only resource limits, `autoImplicit`,
    `relaxedAutoImplicit`, `linter.*` and `pp.*` (the kernel's `set_option` allowlist without `trace.*`),
    with a boolean or number value; dotted keys written as nested TOML tables are read as dotted names.
    So `debug.skipKernelTC` cannot be set package-wide. Protocol §2 ("Lake") and §8 (`replay`, `hygiene`).
- **Why.** lr-6a: nothing replayed `TVA.*`, no hygiene list read `lean/TVA`, and `lake_violations`
  accepted any `leanOptions` key, so "lemmas: proved; Lean checks them" rested on the library staying
  clean.
- **Evidence.** [evidence/gate-round7](../../../../evidence/gate-round7/README.md): U3 (a meta command
  after `"--"`, `'"'`, `r#"a"--"#` or `«a--b»` is flagged; a nested comment, a docstring naming `syntax`
  and a line comment naming `run_cmd` are not; the library as it is: nothing), C2 (`debug.skipKernelTC`,
  `trace.*` and a string value fail the audit; `maxHeartbeats`, a nested `linter.unusedVariables` and
  `pp.all` pass). The replay of the library is the release check's `library` step
  ([evidence/release-check](../../../../evidence/release-check/README.md)): in R0, R1 and R3 it replays
  the library with `leanchecker TVA` inside the sandbox, after a build made there, and passes; in R2,
  where the build fails, it does not run.
- **Constants whose meaning changed.** None.

## L41: gap: `.lake` folders outside `lean/` ship, but the text scan skips them

- **What changed.** Both fixes lr-6a named:
  - `.gitignore` ignores `.lake/` at any depth (it ignored only `lean/.lake/`), so a kernel's
    `lean/.lake`, made whenever lake runs there, does not ship;
  - what ships must be what the scan reads: `shipped_problems` fails on a shipped file inside any folder
    the scan skips (`.lake`, `.venv`, `.git`, `__pycache__`, at any depth; a file can still be tracked
    there with `git add -f`), and on a nested repository git lists as a folder. `tvr audit` runs it in a
    git checkout, and the release check runs it as its `shipped` step. Inside the sandbox the copy holds
    only what ships, so its scan's skipped folders hold only build outputs.

  Protocol §2 ("Text" and "Enforcement") says so.
- **Why.** lr-6a: a note at `docs/.lake/note.md` or a kernel's `lean/.lake/note.md` naming a sibling of
  TVA shipped (it was untracked and not ignored) and passed the audit.
- **Evidence.** [evidence/gate-round7](../../../../evidence/gate-round7/README.md) C3 (an untracked note in
  `docs/.lake/` is now ignored, not shipped, and the audit passes; force-added, it ships and the audit
  fails, naming it; the same note in `docs/` fails the scan); the release check's `shipped` step on a
  force-added note: [evidence/release-check](../../../../evidence/release-check/README.md) R1 (c).
- **Constants whose meaning changed.** None.

## L42: gap: a library change leaves kernel gates stale and certificates GRANTED

- **What changed.**
  - `gate_staleness` (new): a passing gate is stale when a kernel file changed after it ran (as before),
    when a library file or the packages' digest recorded in its snapshot (each item's `file:` entries and
    `packages`) differs from the tree now, or when its snapshot predates whole-file digests (round 6) and
    so records no library files. `current_gate` (used by `tvr file` and `tvr respond`) refuses a stale
    gate, naming the changed files.
  - The certificate needs a gate that is not stale (`cert_state`): after a library change that reaches a
    kernel, `REVIEW.md` marks the gate stale and names the files, `kernels/README.md` shows "PASS
    (stale)", and `certificate.json` is removed at the next refresh. After `tvr check`, the items that
    reach the changed files have new digests and their acceptances are STALE, as before.
  - The release check compares, for each kernel, the gate it recomputes in the sandbox with the shipped
    `gate.json` (the result, the digest and the item digests) and the shipped `certificate.json` (its
    gate and items), and fails the kernel's gate step on a mismatch (`gate_mismatch`).
  - Protocol §9 (the stale gate the tool refuses; the certificate).
- **Why.** lr-6a: `current_gate` looked only at kernel files, so a certificate stayed GRANTED after a
  library change, and the release check did not compare its gates with the shipped ones.
- **Evidence.** [evidence/gate-round7](../../../../evidence/gate-round7/README.md): C0 (the tree's own
  `vec_add_f32` gate, passing with a granted certificate from 2026-10-27, is reported stale), C5 (on a
  copy: a certificate granted; a comment added to `lean/TVA/Lang/Ops.lean`; after `tvr index` the
  certificate is gone, `REVIEW.md` and `kernels/README.md` mark the gate stale naming the file, a further
  filing is refused; after `tvr check` the four items that reach `Ops.lean` are STALE and `conditions`
  stays accepted). In the release check
  ([evidence/release-check](../../../../evidence/release-check/README.md)), R0, the tree's `vec_add_f32`:
  "the shipped gate.json is stale (the result, the items … differ from the recomputed
  gate); the shipped certificate.json names a gate other than the recomputed one"; R1 (d): a kernel gated
  in the copy, then a comment added to `Ops.lean`: the recomputed gate passes, and the step fails on
  the four items whose digests changed; R3: the gate is current and the step passes.
- **Constants whose meaning changed.** None.

## L43: gap: a `TVA.Rules` module in a subfolder gets no rule page, and the audit is silent

- **What changed.** `rule_page(mod)` gives a `TVA.Rules` module its own `RULE_PAGES` entry or else that
  of the nearest module above it (`TVA.Rules.Num.Exp2` gets `docs/rules/num.md`), and `library_files`
  uses it, so a device-fact submodule's items hold its family's page. `fact_page_problems` walks
  `lean/TVA/Rules` and `docs/rules` at any depth: a module with no page above it is named, an entry must
  name a module or a folder of modules, every page must be some entry's. Protocol §2 ("Fact pages") and
  §4.
- **Why.** lr-6a: `fact_page_problems` listed only the top level and `library_files` matched module names
  exactly, so a module such as `TVA.Rules.Num.Exp2` (L4 adds device facts) would have left
  `rules/num.md` out of the digests of the items that cite it, silently.
- **Evidence.** [evidence/gate-round7](../../../../evidence/gate-round7/README.md) C1:
  `lean/TVA/Rules/Num/Exp2.lean` passes the audit and an item that reaches it holds `docs/rules/num.md`;
  `lean/TVA/Rules/Extra/Y.lean` fails the audit, named.
- **Constants whose meaning changed.** None (today's two rules modules keep their pages).

## L46: wording: eight items (release check's reach in §2, evaluation order of keywords, conditional expressions, Desc and L5, `launch_metadata=`)

- **What changed.**
  - **Protocol §2, the release check's reach** (lr-6a, lr-6b): §2 now lists what the sandbox shows
    (L7) and the steps it runs, says that a dependency of *these steps* on anything but TVA, Mathlib's
    pinned source, the Lean toolchain and the system fails there, and that evidence scripts and
    reviewers' work scripts are not run in it (they need Triton and the GPU), so their text is checked
    by the scan, whose path tests are lexical. "Nothing outside exists" is gone; the comment above the
    release check in `tools/tvr.py` says the same.
  - **Keyword arguments first** (lr-6b): SEMANTICS.md "Order" says that operands are evaluated left to
    right, but a call evaluates the called function's expression (for a method, the object), then its
    keyword arguments, then its positional ones (`visit_Call`,
    `primitives/triton-3.8.0-code-generator.py:1455-1476`; `static_executor`, `:1639-1650`), with lr-6b's
    example, and why the order does not matter to a model: only which operations are evaluated does,
    since accesses of a program to the same bytes with a write among them are a conflict unless both are
    atomics, which the semantics performs in every order.
  - **Conditional expressions** (lr-6b): with a runtime condition, "each side is emitted inside its own
    branch of a runtime `if`, and only the taken side runs" (`visit_IfExp`, `:998-1036`); the meaning in
    a model stays open (L28).
  - **`TVA.Lang.Desc`** (lr-6b): the module docstring no longer says the model "assumes" in-kernel
    descriptor scratch does not overlap (with "the reviewer checks the allocator"); it says the library
    covers host-built descriptors and that `tl.make_tensor_descriptor` is open (L5), as SEMANTICS.md does.
  - **`launch_metadata=`** (coverage, tutorial 09; the user: "we do not need to provide models for
    performance-only primitives"): SEMANTICS.md "Launches" says that it changes no result and a model
    omits it: Triton calls the function only to hand metadata to a launch-enter hook, lazily, and only
    when a hook is installed, which `Compiler.Faithful`'s scope excludes; the compiled kernel is the same
    with it or without it. From the sources: `CompiledKernel.launch_metadata` wraps the function in a
    `LazyDict` that calls it only on `.get()` (`triton-3.8.0-compiler.py:374-387, 493-502`, newly pinned in
    `primitives/`), and the launcher passes it to the enter and exit hooks, which are empty `HookChain`s
    unless one is installed (`triton-3.8.0-jit.py:778-779`, `triton-3.8.0-nvidia-driver.c:1406-1415, 1447`,
    `triton-3.8.0-knobs.py:415-433, 477-478`). The general statement for all performance-only
    arguments is L44 (round 8); I did not write it here.

  These are the items ISSUES.md lists under L46 (some raised by more than one report: the first bullet
  answers lr-6a's and lr-6b's, on the release check's reach and "nothing outside exists").
- **Why.** Each sentence was false as written or gave an open construct a covered reading.
- **Evidence.** [evidence/front-end-order](../../../../evidence/front-end-order/README.md) (new; Triton
  3.8.0, H200 NVL, GPU 0): K (the TTIR emits a call's keyword-argument atomics before its positional
  ones, in two calls), C (`x if flag > 0 else y` with a runtime `flag` runs only the taken side: counter
  4 or 400, never 404, with an `scf.if`), M (without a hook the metadata function runs 0 times; the PTX
  and the results are identical with and without `launch_metadata`; with a hook that reads the metadata
  it runs once).
- **Constants whose meaning changed.** The module docstring of `TVA.Lang.Desc` (docstring only); the fact
  page SEMANTICS.md ("Order", the conditional expression, "Launches").
