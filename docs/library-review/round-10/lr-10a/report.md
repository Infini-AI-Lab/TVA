# Library review, round 10, by lr-10a

## Verdicts

- **L7** ACCEPT — The packages step does what both round-7 reopens asked. Re-running
  evidence/tool-round10 gave N1–N3 and U9 identical to out.txt, apart from timings. In the code,
  each package is fetched from its manifest URL only (https; `protocol.allow=never` otherwise). Mathlib's
  pin is checked against the upstream refs (`ls-remote`; an annotated tag is peeled; a branch's
  ancestry is checked on a commits-only clone with `GIT_NO_LAZY_FETCH`). The inherited packages must equal
  Mathlib's manifest at the verified commit, field for field, and `lean-toolchain` must be Mathlib's.
  git runs with `GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM`, an empty HOME and none of the
  caller's `GIT_*` variables. Every checked-out file is compared with a blob id computed in Python, so a
  filter, an `eol` or `ident` attribute in the commit, or an extra file fails closed. The coordinator's
  release check of the merged tree passed `packages`. On the coordinator's question, the fetch lets in
  any commit that a tag or a branch of Mathlib's repository holds, and then whatever packages that
  commit's manifest pins. For a release tag that is enough; a feature branch also passes (new gap below).
- **L36** ACCEPT — The open parts were the release check (L7, above) and the round-7 reasoning.
  evidence/release-check conclusion 5 now matches lr-7a's fifth build: `3f335c460304e5f1`, the same as
  R1–R3, in round-7/lr-7a/work/build_digest_watch.out, so 4 of 5 builds agree. Protocol §2 says what the
  toolchain supplies (`Init`, `Std`, `Lean`) and that it is unchecked. The coordinator's release check
  ran with the final tool on the merged tree, and 8 of 10 steps pass. The two failures are
  `vec_add_f32`'s gate (its own build fails at `VecAdd.lean:155`; its shipped gate and certificate are
  stale) and `pages` (its REVIEW.md and kernels/README.md, generated before the round, still say
  GRANTED). Both are expected.
- **L40** ACCEPT — Re-running U1 and U2 gave output identical to out.txt. Refusing a string literal
  that holds `{` makes the comment reader exact, since a literal without `{` reads the same as a plain
  or an interpolated string. I compared the scanner's other branches with Lean's lexer and found no
  other route. The branches I checked: `'"'` after a non-identifier character, primes as in `x'`, raw
  strings, `«…»`, and nested `/- -/` (Lean nests doc comments too). `import all`, which could bring a
  private constant of another module into a cone without that module's file, is refused by Lean on
  TVA's files, which are not modules (work/ImportAll1.lean, ImportAll2.lean, lib_probes.out). The import
  closure check (`library_module_problems`) runs in both the audit and `library-check`.
- **L48** ACCEPT — Re-running Part L gave A and B different digests, C equal to A and D different, as
  in out.txt. Only the `conditions` and `spec` digests differ from out.txt, and they changed with the
  library files that rounds 8 and 9 edited. `canonConst`, `canonExpr`, `canonName` and `canonLevel`
  delimit every atom (length-prefixed strings, numbers ended by `;` or `)`, parenthesised nodes), so the
  text is injective up to binder names and mdata, neither of which the kernel reads. `const_table`
  hashes the text with sha256. Private and generated kernel constants are in the cone (`kernelModule`).
  Outside `TVA` the library holds only 299 `_private` constants (work/LibRoots.lean), and their
  modules' files are hashed whole. One hole is left: an assumption item's digest does not hold the
  assumption's statement. I filed it under New issues as a soundness issue.
- **L50** REOPEN — `jit_functions` finds a function only through a decorator whose expression names
  `jit`. Triton 3.8.0 also runs a JITFunction made by a call (`h = triton.jit(_h)`, or a kernel
  `k = triton.jit(_k)`) and one under an aliased decorator (`from triton import jit as J`). All three
  ran on GPU 0 (H200 NVL; work/jit_spellings.py, .out). When a decorated kernel calls `h`, ranges that
  leave out `_h` pass the map check (work/gate_probes.out G6). Yet protocol §7 says the mapped lines
  cover "the launched kernel and every helper it calls", and the reviewer guide asks reviewers only to
  check that no unused function is called. A `@triton.constexpr_function` helper, which runs as Python
  at compile time, is not required either (G5). **Fix:** make the rule complete, so that every function
  the source defines, at any depth, lies in the kernel ranges, in `source.host` or in `source.unused`.
  Alternatively, say exactly what the gate covers and ask reviewers to check helpers made any other way.
  Re-running U3 gave output identical to out.txt.
- **L51** ACCEPT — Re-running U4 gave output identical to out.txt. `LINE_SEPARATORS` holds exactly the
  breaks that `str.splitlines` adds to `\n`: a lone `\r`, `\v`, `\f`, `\x1c`–`\x1e`, `\x85`, U+2028 and
  U+2029. claim.md splits at `\n` and strips `\r`, and `tokenize` (through StringIO, with no
  translation) and Lean both count `\n` only. The source is read with `newline=""`, so its sha256 is the
  file's; vec-add's matches its manifest. No kernel file holds a `\r`.
- **L52** REOPEN — Two routes remain open.
  1. `tvr record library N WHO --no-lint` records a folder of any round without linting it. For round 10
     it even reports "a record from before the ledger". `release_records` then skips the folder because
     it is recorded, so its report is never linted (work/gate_probes.out G2).
  2. Nothing ties a ledger line to the tool. A hand-written `accept` line grants the certificate. No
     filing stands behind it: the reviewer's recorded filing accepted at the old gate. REVIEW.md says
     GRANTED and "all unchanged", and `ledger_problems` and `certificate_mismatch` both agree (G1). An
     edited sha256 in the library ledger hides a changed recorded report (G3). So lr-7a's G1, a
     certificate with no acceptance behind it, has moved from certificate.json into ledger.jsonl.

  On the coordinator's question, the library ledger is faithful today. The files of all 22 events
  match, and rounds 1–7 were recorded as they stood at 15:04Z on 2026-10-28. Before that recording,
  round 2's author notes had been edited (a path spelling became `…/tva-old`). Nothing anchors the
  ledger, though, and git holds only two commits, so history cannot either. The refusal in `tvr file`
  works (G9). **Fix:**
  - allow `--no-lint` only for rounds 1–7, and have the release check lint every recorded folder of a
    later round that has `linted: false`;
  - derive each kernel event from the recorded file it belongs to (accepts, raises, releases and
    re-raises from that file event's filing, responses from the response), and fail on any event that no
    recorded file supports;
  - check that each ledger is append-only across the git history, committing at each recording, or say
    in §3 and §10 that the ledgers are trusted as the coordinator keeps them.
- **L53** ACCEPT — Re-running U5 and U6 gave output identical to out.txt. gate.json records `files` and
  `tool`, `gate_staleness` compares contents, and `gate_mismatch` compares both. `pages` regenerates the
  four generated pages with the gate times masked, and the coordinator's run named vec-add's stale
  GRANTED pages. The REVIEW.md header no longer carries a time. Staleness still misses some of the
  gate's inputs (new gap below).
- **L54** ACCEPT — Re-running U7 gave output identical to out.txt. `TOOLCHAIN` must match the whole
  one-line file. `elan_parts` compares each part's realpath with its path, so a link at or above elan's
  `bin`, the toolchain or `settings.toml` is refused. `make_packages` requires Mathlib's toolchain, so a
  non-ASCII digit, which matches `\d`, still fails closed.
- **L55** ACCEPT — Re-running U8 gave output identical to out.txt. `kinds` refuses a `TVA.Rules.*`
  constant outside a `TVA.Rules.*` module, including a module named `TVA.Rules` itself, and
  `library_files` adds rule pages by name. The equation-lemma exemption also admits a hand-written
  `TVA.Rules.X.eq_1` whose prefix is not a definition. Even so, its rule page and its module file still
  reach the digest, which is what L55 needs.
- **L56** ACCEPT — Each item now reads as asked:
  - §2's toolchain paragraph;
  - `masked()`, which writes `<dir>`, `<tva>` and `~` (the coordinator's run printed `sandbox <dir>`);
  - the prompts' paragraphs on mid-round runs;
  - "proved through `consistent`" in protocol §2, §4, §5 and §9 (`EVIDENCE_REQUEST`, `ASSUMPTION_*`), in
    REVIEWER_GUIDE §4, in PROVER_GUIDE and in both kernel prompts;
  - the grids in §6 and DESIGN.md;
  - DESIGN.md's decision row.

## New issues

- **NEW** soundness — **An assumption item's digest does not hold the assumption's statement.**
  - **Where:** `tools/claim_reader.lean.in` (`fitems`, whose roots are the in-scope constants the field
    mentions plus `Inst`) and `item_digests`.
  - **Problem:** no item's cone reaches `Assumptions`, so only the set of TVA and kernel constants a
    field mentions, plus docstrings, enters the digest. work/digest_probes.py tests three edits, and each
    one keeps every item digest:
    - a `[rule]` `DivAcc P .binary32 (2^126) (2^(-22)) (2^(-148))` changed to `2^(-40)`, a bound 2^18
      times tighter than the library's;
    - the same field wrapped in `∀ _ : Nat`;
    - a `[kernel]` `model 0 = model 0` changed to `model 0 ≤ model 0 + 1`.
  - **Consequence:** `tvr check` alone restores the certificate, and REVISED cannot even name the
    changed item. So a `[rule]` that no reviewer saw ships certified, and the release check agrees.
    §4 ("the statement of field `f`") and LIBRARY_REVIEW.md ("no two different meanings share a
    digest") are false for assumption items.
  - **Fix:** have the reader write the canonical text of each field's type, with `K` and `P` bound, and
    hash it into that field's item.
- **NEW** soundness — **A `[rule]` can carry a constant of the kernel's own.**
  - **Where:** `kinds_problems` (`own` counts only names that start with `ns.`) and the reader's
    namespace check, which exempts every `isInternal` name.
  - **Test:** work/kinds_probes.py builds complete kernels with a real `consistent` and standard axioms.
    Each has a `[rule]` `DivAcc … tol …` whose `tol = 2^(-40)` the kernel defines.
  - **Result:** a public `def tol` fails `kinds` (the control). `private def tol`,
    `def _root_.Kernels._tol` and `def _root_._tol` all pass `axioms`, `usage`, `namespace` and `kinds`.
    The last two are declared outside the namespace, which §8 allows only for names Lean generates. So
    §5's "a kernel cannot slip its own constants into a library rule" is false.
  - **Fix:** count as the kernel's own every in-scope constant that is not rooted at `TVA`, other than
    `RULE_ARGS`, and exempt from the namespace check only the names Lean generates, not a user-written
    `_x`.
- **NEW** gap — **A gate does not go stale on every input that changes what the kernel's Lean
  elaborates to.**
  - **Where:** `gate_staleness` and the snapshot (protocol §8, §9).
  - **Test:** after `lean/lean-toolchain` changes to `v4.36.0`, which the audit accepts, the gate is
    not stale and the certificate's gate condition still holds (work/gate_probes.out G4), even though
    every item digest holds Lean's version. By the code, the same holds for `lean/lakefile.toml`
    (`leanOptions`) and for an imported library module that holds no constant of any cone (notation,
    macros).
  - **Consequence:** only the release check catches these changes.
  - **Fix:** record in gate.json the sha256 of these files and of every library file that the kernel's
    modules import.
- **NEW** gap — **A feature branch of Mathlib's repository passes the packages step.**
  - **Where:** `upstream_problem` and `lake_violations` (protocol §2).
  - **Problem:** the repository has 71 branches, some personal or for CI (`kim-em/p-core`,
    `ci-dev/marcelo/parsteps4-broken`). Any branch head is accepted as `inputRev` (G7). The audit only
    asks that `inputRev` equal the lakefile's `rev`, and the inherited packages are then whatever that
    commit's manifest pins, from any https host. Such a branch is not reviewed Mathlib.
  - **Fix:** require Mathlib's `inputRev` to be a release tag `v<version>` (the toolchain's version) or
    `master`.
- **NEW** gap — **`tvr audit` does not scan a folder of `kernels/` that has no `kernel.json`.**
  - **Where:** the audit in `main`. `library_independence` skips `kernels/`, `kernel_independence`
    covers only `kernels()`, and the top-level scan reads only files directly in `kernels/`.
  - **Test:** a file `kernels/notes/readme.md` that holds a path into a sibling of TVA is not named,
    and it would ship. The same file under `kernels/vec_add_f32/` is named (work/gate_probes_g8.out).
  - **Fix:** scan every file under `kernels/`, or refuse a folder there that is not a kernel.
- **NEW** gap — **The gate passes a claim whose conditions cover no instance.**
  - **Where:** protocol §8's checks, DESIGN.md "No vacuous claims", and `TVA.Claim`'s module docstring
    ("so the claim is not vacuous").
  - **Problem:** `Consistent` bounds only the assumptions. The kinds_probes V0 kernel has
    `Inst.ok _ := False` and a real `consistent`, and it passes `axioms`, `usage`, `namespace`, `kinds`
    and the reader's shape examples. The reviewer guide asks whether the conditions are satisfiable, but
    an unsatisfiable combination can be invisible to a reader.
  - **Fix:** require `nonempty : ∃ d init, Inst.ok d ∧ Pre d init`, or reword the docs to say that this
    is the reviewer's judgment.

## Re-ran

- evidence/tool-round10/run.py in full (U1–U9, N1–N3, L), output in work/tool-round10.rerun.out. It is
  identical to out.txt except for timings, U6's lint list (the round-10 reviewers' folders now exist),
  and Part L's `conditions` and `spec` digests (library files that rounds 8 and 9 changed).
- The coordinator's release check of the merged tree, with the final tool (read only, not re-run): 8 of
  10 steps pass, as described under L36.
- New tests, each with its output in `work/`:
  - `digest_probes.py`;
  - `kinds_probes.py`;
  - `gate_probes.py` (G1–G7; G8 and G9 in `gate_probes_g8.out` and `gate_probes_g9.out`);
  - `LibRoots.lean`, `ImportAll1.lean` and `ImportAll2.lean` (`lib_probes.out`);
  - `jit_spellings.py`, on GPU 0 (H200 NVL, Triton 3.8.0, 2026-10-28).
- Read-only calls of `records_problems` and `release_records` on the live tree. Rounds 8 and 9's
  reviewer folders still in progress are named as unrecorded, as expected in the middle of a round.

## Not checked

- My own release check. The coordinator's run covers the merged tree with the final tool, and no route
  needed one end to end: `vec_add_f32` does not build, so no gate can pass in a sandbox.
- `tvr check` and `lake build`. The probes call the gate's functions and the claim reader on small
  kernels compiled with `lean`.
- evidence/release-check R0–R3 and lr-7a's build digests (cited, not rebuilt). The toolchain's files
  are trusted as installed.
- The other survey areas (semantics, transcription, rules, device, lemmas) beyond what touches the gate.
