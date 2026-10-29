# Library review, round 7, by lr-7a

## Verdicts

- **L7** REOPEN — The allowlist sandbox is as protocol §2 describes it. My release check of the tree
  (work/release.out, work/release.json) gave the author's R0: shipped (510 files), packages, sandbox,
  build, library and audit pass; `vec_add_f32`'s gate fails at its own build and on its stale shipped
  gate.json and certificate; the only other failure is the lint of this report, still in progress
  when the check copied the tree. One route is still open: "Mathlib's pinned source" is whatever
  commit `lean/lake-manifest.json` names, and nothing checks that the commit is Mathlib's.
  `make_packages` fetches the pinned commit from the working tree's own checkout first, and
  `lake_violations` checks the URL of the direct package only: an inherited package's URL and every
  revision may be anything. work/packages_local_commit.py builds a scratch root whose Mathlib and
  Batteries checkouts each get one local commit that adds a module (`Mathlib/Extra/Borrowed.lean`, a
  stand-in for another project's lemma under a new name), with the manifest pinning those commits.
  `lake_violations` accepts it (P1). `make_packages` passes, and the package it makes holds the file
  (P2). The import check accepts `import Mathlib.Extra.Borrowed` (P3). An inherited package from
  another URL is accepted too (P4). Lake treats every `Mathlib.*` import as a module of Mathlib's
  library, by name, and builds it from the package's source (the toolchain's
  `src/lean/lake/Lake/Config/Module.lean:41-42` and `LeanLibConfig.lean:135-142`). So the build,
  `leanchecker`, the gate and the audit would all pass with the borrowed lemma in use, and the same
  holds for a lemma added to an existing Batteries file that Mathlib imports. This is lr-6a's route
  ("a lemma from a file that a fresh Mathlib clone lacks"), moved from an ignored file into a
  commit. **Fix:** make each package from its URL only (outside the sandbox). Require the pinned
  commit to be the upstream ref that the manifest's `inputRev` names (`v4.35.0` for Mathlib), or to
  be reachable from it (fetch that ref, then `git merge-base --is-ancestor`): a commit fetched by
  its hash alone is not shown to be on an upstream branch. Require each inherited package's name,
  URL and revision to equal Mathlib's own `lake-manifest.json` at the pinned commit. Check
  `packagesDir` too.
- **L31** ACCEPT — The "Compiler hints" section of `lean/TVA/Lang/Ops.lean` now says what
  SEMANTICS.md "Compiler hints" says. A hint is a check, placed where the value is defined. A hint
  inside a loop body on a carried name or on the induction variable is open (L1), and the withdrawn
  reading is named with its reason (evidence/hints-dry-run). Per-dimension values at rank ≥ 2, and
  a `contig` above 1, are open (L2). The `multiple_of` and `multiple_ofN` docstrings describe their
  definitions (`contig = 1`: every element) and no longer cite Triton's contiguity or ask the
  reviewer to check it: no "Triton assigns" or "contiguity Triton" is left in `lean/TVA/Lang`.
  Triton's docstring (`primitives/triton-3.8.0-core.py:3308-3322`) agrees that at contiguity 1
  every element is a multiple. A false check is undefined behaviour (`Step.pickUB`), so the stricter
  N-dimensional reading can make a claim harder to prove, but never vacuous. Only docstrings
  changed, and the notes list them.
- **L36** REOPEN — Only for the release check, as in round 6, and tracked under L7. Its guarantee,
  that a dependency on anything outside TVA fails there, does not hold for a Mathlib commit that is
  not Mathlib's (L7). The digests, the guidance, the map check and the prompts are as round 6
  accepted them. On the coordinator's question, building Mathlib from source in each run rather than
  shipping built files pinned by a hash: the choice holds. A hash pins a build only if a build from
  source reproduces it. Lean also loads Mathlib's compiled tactics and notation while it elaborates
  TVA, so a borrowed build is trusted code, which is the route L7 closes. The cost is bounded and
  falls on rounds, not kernels. Two parts of the reasoning need correcting. (1) The `.olean` files
  do not, after all, depend on the build path. My run, with TVA at its own path, gave 54186 `.olean`
  files with digest `3f335c460304e5f1` (evidence/release-check's own `build_digest`,
  work/build_digest_watch.out). That is R1–R3's digest, not R0's `3ae1aa797d96970a`. So 4 of 5
  builds agree bit for bit, and R0 is the outlier, for a cause the notes do not find. The pinned
  build the notes leave as a follow-up looks feasible, but only once the pin is checked against
  upstream (L7): a pinned build and a build from source are equally exposed to a local commit.
  (2) The toolchain is said to "supply no module TVA imports". In fact every module imports `Init`,
  and Mathlib imports `Lean` and `Std`, all from the toolchain. What carries the toolchain is
  §2's "trusted as installed" (a wording item below).
- **L38** ACCEPT — `manifest_problems` runs first, and nothing else runs on a manifest that fails
  it: no build, and no claim reader written. My rerun of evidence/gate-round7 U2 is identical to its
  out.txt, apart from the author's masked path. Names that are plain but special in Lean pass
  `manifest` (work/gate_probes.out N1), but they cannot change what the reader checks:
  - `_root_` makes ``ns ++ `Assumptions`` name no constant, so the reader's `#eval` throws and `shape`
    fails (work/RootNamespace.lean, exit 1);
  - a keyword such as `end` is read as a plain name component (Lean reads `end.trivial` as
    `«end».trivial`, work/KeywordNamespace.out), so it names a namespace like any other;
  - `TVA` puts the kernel's declarations in the library's namespace, which `namespace` refuses.

  Opened names are spliced in only when they are plain, and the reader must print exactly one
  `TVR_JSON` line.
- **L39** ACCEPT — `code_lines` finds code with `ast` and `tokenize`. My rerun of U1 (K1–K5, and a
  source that does not parse) is identical. Further cases (work/gate_probes.out M3):
  - code after a docstring's `;` counts as code;
  - a bare f-string statement counts as code, since it evaluates its fields;
  - a string statement continued with `\`, and a bytes statement, are exempt, since they run nothing.

  Protocol §7 defines a code line this way. Two other weaknesses of the map check, the ranges and
  claim.md's line numbers, are new issues below.
- **L40** ACCEPT — In my release check, the `library` step ran the library hygiene and
  `leanchecker TVA` inside the sandbox, after a build made there, and passed. `leanchecker`
  (`src/lean/LeanChecker.lean` in the toolchain) replays every `.olean` whose module name starts
  with `TVA` into the environment of its imports, so it covers every library module. `lake_violations`
  refuses `debug.skipKernelTC` written dotted, quoted or as a nested table, a string value, a float,
  and an unlisted `backward.*` option. It accepts `autoImplicit`, `maxHeartbeats = 0` and `pp.all`
  (work/tool_probes.out O1). My rerun of U3 is identical: the library has nothing flagged. The
  comment stripper misreads Lean's interpolated strings, a new gap below. That is not a reopen: the
  release check's `leanchecker` re-checks every library declaration whatever the hygiene misses,
  and the kernel hygiene reads whole files.
- **L41** ACCEPT — `.gitignore` ignores `.lake/` at any depth. `shipped_problems` refuses a shipped
  path inside `.lake`, `.venv`, `.git` or `__pycache__` at any depth, and a nested repository, while
  names that only look like these pass (work/tool_probes.out F1). `tvr audit` runs it in a checkout,
  and my release check's `shipped` step passed.
- **L42** ACCEPT — `gate_staleness` compares the packages digest and every `file:` entry of the
  gate's snapshot with the tree, and treats a snapshot older than round 6 as stale. `current_gate`
  refuses a stale gate, and `cert_state` needs a gate that is not stale. In the release check,
  `gate_mismatch` fails the gate step on a different result, digest or item digest, or on a
  certificate that names another gate. In my run it reported `vec_add_f32`'s shipped gate.json and
  certificate as stale. C5 was not re-run, since it runs `tvr check`, and so `lake build`. Three
  weaker points are new gaps below:
  - a certificate that no ledger supports passes the release check's comparison;
  - staleness reads kernel files by mtime, and leaves out the source and the write-up;
  - a change to the gate itself makes no gate stale.
- **L43** ACCEPT — `rule_page` gives a module the page of the nearest module above it, and
  `fact_page_problems` walks `lean/TVA/Rules` and `docs/rules` at any depth. On a scratch tree
  (work/tool_probes.out P1), modules at depths 2 and 3 get `num.md`. The audit is silent when every
  module has a page, and names both a module with no page above it and a page that no entry claims.
  A latent case, keyed on names rather than modules, is a new gap below.
- **L46** ACCEPT — Each item reads as asked:
  - §2 lists the release check's steps, and says that evidence and work scripts are not run in it;
  - "Order" puts keyword arguments first, as `visit_Call` and `static_executor` do
    (`primitives/triton-3.8.0-code-generator.py:1455-1476, 1639-1648`);
  - a runtime conditional expression runs only the taken side;
  - the `TVA.Lang.Desc` docstring calls `tl.make_tensor_descriptor` open (L5);
  - `launch_metadata=` changes no result.

  evidence/front-end-order re-ran on GPU 0 with output identical to its out.txt (K1, K2, C, M1–M4).
  The coordinator's extra items, the release-check paragraphs of the author's and the reviewer's
  prompts and DESIGN.md's row, match the tool: an empty folder outside TVA and the home directory,
  16 GB, about 25 minutes, run in the background, report the result. They inherit L7's unchecked
  "Mathlib's pinned source", and a wording item below covers their usability.

## New issues

- **NEW** soundness — **A change to a kernel definition can keep every item digest, so acceptances,
  the certificate and the release check's comparison all survive it.** Where:
  `tools/claim_reader.lean.in` (`constJson`: `th`, `vh`) and `tools/tvr.py` (`check`: the cone lines
  `name hash`). A kernel constant's hash comes only from 64-bit non-cryptographic values.
  `Expr.hash` is 32 bits (`Expr.Data.hash := c.toUInt32.toUInt64` in Lean), `String.hash` and
  `mixHash` are fast 64-bit hashes, and no kernel file is hashed whole (§4: "a kernel's proof edit,
  comment or reformatting changes no digest"). work/HashWidth.lean finds two values,
  `fun x => x + 30606` and `fun x => x + 33013`, with the same `Expr.hash` after 2^15 tries.
  work/DigestCollision.lean, a parallel collision search with distinguished points, found two
  numbers `a ≠ b` whose values `fun x => x + a` and `fun x => x + b` have the same `vh`:
  a = 4536642005803190292 and b = 3061831643823180776, `vh` 16081895171158148313. That took 1.0·10^10
  evaluations, 21 minutes on 64 threads of this machine (work/DigestCollision.out).
  work/digest_check.py declares each as `Kernels.Demo.model` in its own file and runs `constJson`'s
  code on it. Both get the identical hash line `9216286579760591762-16081895171158148313`, and so the
  same item digest, while `model 0` is `a` in one and `b` in the other (work/digest_check.out). A prover can therefore prepare two models with equal digests, have one
  reviewed and accepted, and ship the other: the gate passes, REVIEW.md shows nothing stale, the
  certificate stays granted, and the release check's recomputed digests agree. **Fix:** hash a
  canonical text of each kernel constant with sha256. For example, the reader prints `toString` of
  each type and value, which shows constants with their universe levels, literals and binder kinds,
  and `tvr` hashes that text. Alternatively, hash the kernel's Lean files whole into the items.
- **NEW** gap — **The map check's coverage rests on `source.ranges`, which nothing checks, while
  claim.md says "Every code line of the source is mapped".** Where: `tools/tvr.py` `check` (the
  `missing` expression) and `render_claim` (§6's sentence), and protocol §7.
  work/gate_probes.out M1, on `vec_add_f32`'s source with `output = x + y` (line 52) left out of the
  map: with ranges `[[29, 54]]` the map fails; with ranges `[[29, 51], [53, 54]]`, or with `[]`, it
  passes. `manifest_problems` accepts both. **Fix:** have the gate require the ranges to cover every
  line of each `@triton.jit` function in the source (found with `ast`), unless the manifest names a
  function as unused and the reviewers judge it. Have claim.md and §7 say "every code line of the
  kernel lines", and have the reviewer guide ask reviewers to check that the ranges cover the
  launched kernel and every helper it calls.
- **NEW** gap — **claim.md's map table shows other source text than the line each entry covers,
  when the source holds a form feed or a Unicode line separator.** Where: `tools/tvr.py` `check`
  (`src_lines = src_text.splitlines()`, used by `render_claim`). `str.splitlines` also splits at
  `\x0c`, `\x0b`, `\x1c`–`\x1e`, `\x85`, U+2028 and U+2029. Python's tokenizer, and so `code_lines`
  and the entries' `src`, counts only line feeds. work/gate_probes.out M2: with one such character in
  a comment, every later row of the table shows the text of the line before, for example
  `tl.atomic_add(c_ptr, 1)` beside the entry for line 6, which is `tl.store(x_ptr, x)`. **Fix:**
  number the lines as Python does (`src_text.split("\n")`), or have the gate refuse such characters
  in a source.
- **NEW** gap — **The library hygiene's comment stripper misreads Lean's interpolated strings, so a
  command can hide after one.** Where: `tools/tvr.py` `_uncommented`, used by
  `library_hygiene_problems`, the header reader and `opened_names`. In
  `def s : String := s!"{'"'} -- " #eval 7`, the stripper ends the string at the quote inside the
  `{…}` term, reads ` -- …` as a comment, and blanks `#eval 7`, so nothing is flagged
  (work/gate_probes.out H1). Lean reads the `{…}` as a term, ` -- ` as text of the string, and
  `#eval 7` as a command, which it runs (work/InterpStripper.lean and .out print `7`). The same goes
  for `m!`, `f!`, `throwError "…"` and other users of `interpolatedStr`. `leanchecker` in the
  release check re-checks the declarations, but meta code would run at every gate's build outside
  the sandbox. **Fix:** inside a string literal, read `{…}` as nested code with its own literals,
  as Lean's `interpolatedStr` lexer does, or fail closed on a `--` or `/-` that follows a string
  literal containing `{` on the same line.
- **NEW** gap — **The release check does not compare the review records with the ledger.** Where:
  `tools/tvr.py` `cmd_release`, `gate_mismatch` and `release_records`; protocol §2 and §3.
  - A `certificate.json` that names the recomputed gate and items passes `gate_mismatch` with no
    reviewer and no acceptance behind it (work/gate_probes.out G1), although the tool in the sandbox
    would remove it at the next refresh.
  - A recorded filing, report or response changed after recording ships and passes: REVIEW.md flags
    it ("changed after recording"), but no step reads that flag, and earlier rounds are not linted.
  - Library review records have no recorded hash at all, so §2's "earlier rounds' records are
    frozen" rests on nothing the tool checks.

  **Fix:** after each kernel's gate in the sandbox, require the regenerated certificate (its
  presence, gate, items and reviewers) to equal the shipped one, and fail when any recorded file
  differs from its sha256 in the ledger. Record the library rounds' files with their sha256 (a
  library ledger), and check them the same way.
- **NEW** gap — **Gate staleness reads kernel files by mtime, leaves out the source and the
  write-up, and ignores changes to the gate itself.** Where: `tools/tvr.py` `gate_staleness`; §9.
  work/tool_probes.out S1: after the source file and README.md change, the gate is "not stale". A
  changed Lean file whose old mtime is put back is "not stale" either. Nothing records which claim
  reader or gate code produced a gate, so after a gate fix such as L38's, a gate that passed under
  the old tool stays current and its certificate granted, until someone re-runs `tvr check`. The
  release check is the only backstop. **Fix:** record in gate.json the sha256 of every kernel file
  (source and write-up included), and of `tools/claim_reader.lean.in` with a gate-version constant
  that changes when a check changes, and compare contents rather than times.
- **NEW** gap — **A rule's page follows its module, while the `[rule]` kind follows its name, so a
  `TVA.Rules.*` constant declared outside `lean/TVA/Rules` gets no rule page, and the audit is
  silent.** Where: `tools/tvr.py` `library_files` (rule pages taken from `consts[n]["module"]`),
  `check`'s `kinds` (`c.startswith("TVA.Rules.")`) and `fact_page_problems` (which walks files). A
  `[rule]` assumption citing, say, `TVA.Rules.Num.foo` defined in `TVA.Lemmas.X`, or in a module
  `lean/TVA/Rules.lean`, passes `kinds`, and `num.md` is not in its digest (work/tool_probes.out P1:
  `rule_page` of `TVA.Rules` and of `TVA.Lemmas.Num` is none). No such constant exists today.
  **Fix:** have the audit (or the claim reader) require every constant named `TVA.Rules.*` to be
  declared in a module of `lean/TVA/Rules`, or take the rule page from the constant's name.
- **NEW** wording — **The release check's statements about the toolchain, and its use mid-round.**
  - Protocol §2 says the toolchain is "trusted as installed, as Lean itself is". It does not say that
    the toolchain supplies the core modules `Init`, `Std` and `Lean`, which every TVA module imports
    (`Init`) or reaches through Mathlib. It lives in the user's writable home directory, and nothing
    checks it. The author's notes say instead that it "can supply no module TVA imports". Say what
    the trust covers.
  - The release check prints, and writes into release.json, the absolute path of its folder. A
    reviewer who keeps its output in `work/` then fails the audit: when I ran the audit mid-round it
    flagged both reviewers' `release.out`, and I masked mine by hand. Run mid-round, the check also
    always fails `lint` on that round's records still in progress. Say both in the reviewer's
    prompt, or print the folder masked.

## Re-ran

- `tvr release --dir <empty scratch folder>` on the tree, once (work/release.out, work/release.json:
  8 of 10 steps pass; the failures are `vec_add_f32`'s gate, as expected, and the lint of this
  report in progress). The `.olean` digest of its Mathlib build is in work/build_digest_watch.out.
- evidence/gate-round7 Part U (U1–U3), through the script's own `part_u` (work/gate_probes.py):
  identical to out.txt, apart from the author's masked path.
- evidence/front-end-order on GPU 0 (Triton 3.8.0, H200 NVL, 2026-10-28): identical to out.txt
  (work/front-end-order.rerun.out).
- New tests, each with its output: work/packages_local_commit.py (L7), work/gate_probes.py (M1–M3,
  H1, N1, G1), work/tool_probes.py (S1, O1, P1, F1), work/RootNamespace.lean,
  work/KeywordNamespace.lean, work/InterpStripper.lean, work/HashWidth.lean,
  work/DigestCollision.lean and work/digest_check.py (with DigestCheckA.lean and DigestCheckB.lean).

## Not checked

- evidence/gate-round7 Part C (C0–C5) and evidence/release-check R1–R3: they run `tvr check`, and
  so `lake build`, on copies outside the sandbox, or a release check per run (I ran one). C1–C3
  are covered by direct calls of the same functions on scratch trees (P1, O1, F1); C4 and C5 by
  reading the code, with U2, S1 and G1.
- The local-commit route end to end in a real release sandbox: its parts were tested separately
  (manifest, packages, imports), and Lake's module resolution was read in its source.
- evidence/independence-audit: its script takes the other project's name, which I neither know nor
  sought.
- The semantics, transcription, rules, device and lemma areas beyond the batch's items.
