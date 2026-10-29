# Library batch, round 13: author's notes

Issues: L7, L50, L52, L62, L63, L68, L69, L70 ([ISSUES.md](../../ISSUES.md)), 2026-10-28. The tool and gate
batch from rounds 9 and 10's reviews, run beside rounds 11 (semantics and rules) and 12 (the float layer) in a
separate tree. The changes are in `tools/tvr.py`, `tools/claim_reader.lean.in`, the protocol and role documents,
and a new evidence folder, `evidence/tool-round13/`. No statement of the library changed: the only Lean edits
are two docstrings (L70, below), and `lake build TVA` is green with the standard axioms only.

**Kernels.** `kernels/vec_add_f32` still fails to build against the current library (expected; its prover
re-proves it). The gate's code changed and gates now record the build's inputs (L69), so every gate is stale
until `tvr check` runs; L68 adds each assumption's statement to its item, so every `assumption:<f>` digest
changes once (the `model`, `map`, `conditions` and `spec` digests do not change from this round). vec-add's one
field, `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`, keeps to the new `kinds`
rules, its manifest classes every function of its source (L50), and its shipped ledger passes the new event
checks (L52). I did not edit the kernel folder or run `tvr check`.

**Files of other rounds.** `lean/TVA/Math/Float/Convert.lean` and `lean/TVA/Math/Float/Format.lean` (round 12's,
which rewrites them): one docstring each, where a path cited a record TVA does not hold (L70); take round 12's
version of both files. `docs/review-questions.md` (no round's): one phrase (L70). In `docs/PROVER_GUIDE.md` and
`docs/REVIEWER_GUIDE.md` I changed only the ranges bullet and paragraph (L50), the prover guide's §3 and its
table's accuracy rows, and REVIEWER_GUIDE §4's `[rule]` bullet (L62), not round 11's passages (L57, L61, L72).
`tools/claim_reader.lean.in` keeps its structure: one new function (`TVR.leanMade`), the namespace check's
exemption and the instance flag (L63), and four more keys in each field's `field_uses` entry (L62, L68). A
reader check for round 11's L71 can go beside the shape examples without meeting these.

**For the coordinator (L52).** The existing ledger lines pass every new check unchanged (rounds 1 to 7 are
unlinted, before the ledger; rounds 8 to 10 linted, one line per folder). Three things only the coordinator can
do: commit TVA after each `tvr record`, and after each kernel's `tvr file` and `tvr respond`, so that the git
history holds the ledgers' lines; keep each release check's `release.json` outside the tree (its `ledgers` gives
every ledger's line count and sha256), so that a later ledger can be shown to have only grown; and note under L52
in `ISSUES.md` that lines 1 to 19 of the library ledger (rounds 1 to 7) were written by round 10's author with
`--no-lint`, as the files stood in round 10 (round 2's notes after a path was masked), under the tool's fixed
actor `coordinator`.

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed.** The two open routes of the release check (protocol §2, "Enforcement" and "Lake"):
  - **the system's programs** (lr-10b): the tool runs `/usr/bin/git`, `/usr/bin/bwrap` and `/usr/bin/find`,
    never what the caller's `PATH` finds first (`GIT[0]`; `system_tool`: a file under `/usr` that the user
    cannot write and that no regular user owns; a missing one fails the `sandbox` step). Git gets
    `PATH=/usr/bin:/bin` and, as before, no configuration and no `GIT_*` variable of the caller's; `find`,
    `bwrap` and the system Python's probe run with `PATH=/usr/bin:/bin` and nothing else of the caller's
    environment (`SYSTEM_ENV`, so no `LD_PRELOAD` either). `shipped_files`, which `tvr audit` also runs, uses the
    same git;
  - **Mathlib's revision** (lr-10a's G7): `inputRev` must be the release tag of the toolchain's Lean (`v4.35.0`
    for `leanprover/lean4:v4.35.0`) or `master` (`mathlib_ref`). The audit's lake check refuses another `rev`;
    the packages step refuses another `inputRev` before it fetches anything, and checks a tag only against the
    repository's tags and `master` only against its branch (`upstream_problem(…, kind)`).
- **Why.** lr-10b P3: wrappers first on `PATH` ran as git, bwrap and find, so a user's git decided the packages
  and the refs they are checked against, and a user's bwrap the sandbox. lr-10a G7: any of Mathlib's 71
  branches, personal and CI ones among them, passed as `inputRev`.
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) Part N: with git, bwrap and
  find wrappers first on `PATH`, the tool ran `/usr/bin/git` (2.52.0), `/usr/bin/bwrap` and `/usr/bin/find`, and
  no wrapper ran; git's `PATH` is `/usr/bin:/bin`; lr-10a's feature branch is refused by the audit and by the
  packages step before any fetch; Mathlib's pin is the tag `v4.35.0`'s commit and is reachable from `master`;
  `master` asked for as a tag is refused. The release check of this round (under L52) ran the new steps.
- **Constants whose meaning changed.** None (the tool and the protocol).

## L50: gap: the map's coverage rests on `source.ranges`, which nothing checks

- **What changed.** The map check (`map_problems`, protocol §7, §8) no longer depends on how `jit` is spelled.
  Every function the source defines, at any depth (`source_functions`: every `def`, `async def` and `lambda`,
  from its first decorator to its last line), must lie in the kernel lines (`source.ranges`), in the host lines
  (`source.host`), or, a `def`, be named in `source.unused`; a function under a decorator naming `jit` may not be
  in the host lines (it must be in the kernel lines or unused, as before). `source.unused` may name any `def`.
  claim.md §2 lists the three classes, lambdas included, and asks the reviewer to judge them; the gate's `map`
  detail says what it checked. REVIEWER_GUIDE §1 ("The ranges"), the kernel reviewer's prompt and the prover
  guide ask the reviewer to check that the kernel lines hold every function the launch runs, on the device or
  at compile time (a JIT function however it is made, a `triton.constexpr_function`, a plain function the
  front end calls), and that the launch calls no function the source does not define but Triton's own.
- **Why.** lr-10a G5, G6 and lr-10b P1: a helper made by `triton.jit(_h)`, under an aliased decorator or bound
  to `deco = triton.jit`, and a `@triton.constexpr_function`, could lie outside the ranges with the map check
  passing. Classing every function makes the rule complete without a list of spellings, since the launch runs
  only functions: a JIT function is made from a `def` of a Python file (`triton-3.8.0-jit.py:465-474, 546-551`);
  a constexpr function runs any Python function, a lambda included (`:1138-1177`); and the front end calls any
  other callable it reaches, such as a plain function through a module (`triton-3.8.0-code-generator.py:387-416,
  1444-1446`). Which class is right stays the reviewer's judgment (library issue L36).
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) U1: vec-add's manifest
  passes (its seven functions are in its kernel or host lines) and fails once the benchmark's host lines are
  removed; each of the five spellings above fails with the helper outside the classes; a lambda in the host
  lines and a helper nested in the kernel pass; a `@triton.jit` helper in the host lines fails.
- **Constants whose meaning changed.** None. `vec_add_f32`'s manifest already classes every function.

## L52: gap: the release check does not compare certificates and review records with the ledger

- **What changed.** Both of round 10's open parts:
  - **`--no-lint`** (lr-10a G2): `tvr record` refuses it after round 7 (`PRE_LEDGER_ROUNDS`), and the release
    check's `records` step fails on a library ledger line of a later round that says it was not linted (the
    tool never writes one, so it fails rather than being linted afresh);
  - **events tied to their files** (lr-10a G1, G3; lr-10b P2): `kernel_event_problems` folds a kernel's ledger
    and derives, at each `file` event, the events its recorded filing gives at the gate it was filed at, with
    the item digests from the gate's snapshot (`filing_events`), and at each `response` event those its
    response gives (`response_events`). A `file` event must be a filing of a reviewer assigned before it, at the
    round and the path the tool gives it, and record its filing and its report. `tvr file` and `tvr respond`
    build their events with the same two functions. Any other `raise`, `release`, `reraise`, `escalate`,
    `accept` or `respond` is an event no recorded file gives, and a `decide` must be on an escalated finding
    (`assign` and `note` are the coordinator's). The certificate needs a ledger that holds to its records
    (`kernel_record_problems`, in `cert_state`), and `REVIEW.md` says whether it does; the release check's
    `records` step runs the same check;
  - **two sha256s, a folder twice** (lr-10b P2): `ledger_problems` fails on a file recorded with two different
    sha256s (the tool never writes that), and `library_ledger_problems` on a library folder recorded twice, on
    a line that records no role folder as `tvr record` writes it, and on files outside the folder;
  - **what a ledger shows**: protocol §3 ("What a ledger shows"), §9 (the certificate), §10 (recording) and
    LIBRARY_REVIEW.md ("Rounds") say that a ledger shows each recorded file unchanged since its line, and the
    events following from those files, not that its lines are unchanged: those rest on the git history, or on
    a digest of the ledger kept outside the tree. `release.json` now keeps each ledger's line count and sha256
    (`ledger_digests`), so that a later run can show a ledger only grew.
- **Why.** lr-10a G1: a hand-written accept line granted a certificate that every check confirmed; G2: `tvr record
  … --no-lint` recorded a round-10 folder unlinted, and the release check then skipped it; G3 and lr-10b P2: an
  appended line, or an edited sha256, re-froze an edited record. The other route lr-10a named, committing at
  each recording, is the coordinator's (above): the release copy ships no git history to check.
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) U2 (vec-add's shipped ledger
  is consistent; lr-10a's G1 accept, an edited accept, a release with no filing and a decision on a closed finding
  are named, and the certificate's ledger condition fails; an edited report re-recorded by an appended line is
  named), U3 (`tvr file` and `tvr respond`, in a copy, write events the derivation accepts), U4 (the tree's library
  ledger passes; lr-10b's route 1 is named twice; route 2, an edited sha256, passes the records step, as the
  protocol now says, and the earlier ledger digest no longer matches the ledger's first lines; lr-10a's G2
  `--no-lint` is refused, and the same line written by hand is named). The release check of this round's tree
  (2026-10-28, in an empty scratch folder outside TVA and the home directory): 9 of 11 steps pass (shipped,
  packages, sandbox, build with Mathlib from source, library with leanchecker, records, both lints, audit). The two
  failures are the expected ones: vec_add_f32's gate (its proof does not build, and its shipped gate.json is stale
  under the new digests and `build` key) and the pages (kernels/README.md and vec_add_f32's REVIEW.md, regenerated
  with the new claim rendering, differ from the shipped ones). release.json now records the ledgers' digests
  (`ledgers`).
- **Constants whose meaning changed.** None.

## L62: soundness: a `[rule]` field can state a device fact with the kernel's own constants (`MathAcc` with literals)

- **What changed.** The gate's `kinds` check (`rule_fact_problem`, protocol §5, §8) accepts in a `[rule]` field
  only a fact the library states, applied to the platform: under the statement's `∀` and `→` binders,
  `Compiler.Faithful`; a `TVA.Rules` constant whose type is `Platform → Prop`, applied to `P` alone (every fact
  of `TVA.Rules.Num.Sm90` and `DivFull`: today exactly 25 constants, the facts of `docs/rules/num.md`'s table);
  a `…Total` rule of `TVA.Rules.Num`, with any arguments; or `DivRn` on the format `binary64`. A shape whose
  constants the field chooses (`MathAcc`, `MathAt`, `DivAcc`, `CvtAcc`, `CvtRn`, `DotAcc`, `DotScaledAcc`,
  `AtomicAddAcc`, a family such as `DotF8`, `DivRn` on another format), a conjunction of facts, and `K` outside
  `Faithful` are refused. The claim reader reports each field's shape (the constant it applies, whether that
  constant's type is `Platform → Prop`, and each argument: `P`, or a constant alone). The facts are accepted by
  what they are, not by a list of the float layer's definitions; the one name the rule reads is the last
  component `binary64` of `DivRn`'s format argument, a constant of the library's. The guides: the prover
  guide's §3 cites the facts by name and proves `consistent` with `Platform.sm90ref`, and its table names the
  facts for `tl.exp` …, `/`, `tl.dot` and `tl.dot_scaled`; REVIEWER_GUIDE §4 says what the gate ensures; the
  kernel prover's prompt says the same.
- **Why.** lr-9a: `MathAcc P .exp .binary32 Set.univ 2^-24 2^-150` passed `kinds` (it refused only the kernel's
  namespace), `Platform.ref` satisfies it, and it is false on the device (`tl.exp` reaches `2^-17.99`). The
  guides led there: the prover guide gave the shapes and said `Platform.ref`'s constants are the bundle's, and
  REVIEWER_GUIDE said the gate ensures no constant of the kernel's own.
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) Part K: the accepted forms
  pass (`Faithful`, `DivFull P`, `∀ d, d.ok → Exp2F32 P`, `MathTotal P .exp .binary32`, `DivRn P .binary64`);
  lr-9a's field, `DivAcc` with `DivFull`'s own constants, `DotF8 P .f8e4nv .f8e5`, `DivRn P .binary32`, `K`
  outside `Faithful` and a conjunction are refused. The `Platform → Prop` constants of `TVA.Rules`, listed with
  Lean on the built library, are the 24 facts of `TVA.Rules.Num.Sm90` and `DivFull`.
- **Constants whose meaning changed.** None (the gate and the guides). Should round 12 rename the format
  constant `binary64`, `rule_fact_problem` needs the new last component; the facts of `TVA.Rules.Num.Sm90` are
  found by their type and need nothing, as long as each is a `def … : Prop` over `variable (P : Platform)`.

## L63: soundness: a `private` or `_root_` kernel constant passes `kinds` in a `[rule]`

- **What changed.**
  - **`kinds`** (protocol §5, §8): a constant of the kernel's own is every constant of the statement that is not
    rooted at `TVA`, other than `Inst`, `Inst.ok`, `model` and `launch` in the kernel's namespace, so a private
    one and one declared outside the namespace count too;
  - **the namespace check** (`tools/claim_reader.lean.in`): a name declared outside the kernel's namespace is
    exempt only when Lean made it, a name with a numeric component (`TVR.leanMade`: a macro scope, a private
    prefix; a kernel's source cannot spell one), no longer whenever `Name.isInternal` holds (any component
    starting with `_`). The instance check exempts only the `SizeOf` instance `<type>._sizeOf_inst` that Lean
    makes for a type the kernel declares: an instance the kernel named `_x` was exempt by the same test.
- **Why.** lr-10a's kinds probes: `private def tol`, `def _root_.Kernels._tol` and `def _root_._tol` passed
  `axioms`, `usage`, `namespace` and `kinds` in a `[rule]` field, so protocol §5's "a kernel cannot slip its own
  constants into a library rule" was false. The instance route is the same exemption.
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) Part K: the three bounds are
  each named as a constant of the kernel's own; the namespace check names `Kernels._tol` and `_tol` as declared
  outside the namespace; the instance `_hdiv` is the kernel's and Lean's `Inst._sizeOf_inst` is not; a control
  kernel (structures with defaults, recursion, a private definition, `simp`, `omega`, `decide`, `norm_num`) has
  no namespace problem.
- **Constants whose meaning changed.** None.

## L68: soundness: an assumption item's digest omits the assumption's statement

- **What changed.** The claim reader writes each field's statement as canonical text (`TVR.canonExpr` of the
  field projection's type, so `K` and `P` are bound, as de Bruijn indices; the `canon` key of `field_uses`), and
  `item_digests` adds its sha256 to the field's item (`statement`, which `REVIEW.md` names when it changes).
  Protocol §4, §13; LIBRARY_REVIEW.md (digests); DESIGN.md (the digests row).
- **Why.** lr-10a's digest probes: a `[rule]` relative bound changed from `2^-22` to `2^-40`, the same field
  wrapped in `∀ _ : Nat`, and a changed `[kernel]` statement each kept every item digest, since the item held
  only the set of constants its statement mentions; so REVISED could not name the change, and `tvr check` alone
  restored the certificate.
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) Part D: lr-10a's three edits
  each change exactly the edited assumption's digest; a proof edit changes none.
- **Constants whose meaning changed.** None. Every `assumption:<f>` digest of every kernel changes once (it now
  holds the statement), so those acceptances go STALE at the next `tvr check`; the other items' digests do not
  change from this round.

## L69: gap: gates don't go stale on `lean-toolchain`, `lakefile.toml` or an imported module with no constant in a cone

- **What changed.** `gate.json` records `build`: the sha256 of `lean/lean-toolchain`, `lean/lakefile.toml`,
  `lean/lake-manifest.json` and every library file the kernel's Lean files import, directly or through other
  library modules (`build_inputs`, with the tool's import reader). `gate_staleness` compares it by contents (a
  gate without it, from before this round, is stale), and the release check's `gate_mismatch` compares it with
  the recomputed gate. Protocol §8, §9 (a stale gate whose digests come out the same after `tvr check` re-opens
  nothing).
- **Why.** lr-10a G4: `lean/lean-toolchain` changed to `v4.36.0` left the gate current and the certificate's gate
  condition holding; by the code, the same for `leanOptions` and for an imported module of notation or macros.
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) U5: a gate recorded now is not
  stale; lr-10a's toolchain change, a `leanOptions` key, a comment in `lean/TVA.lean` (which holds no constant)
  and one in `lean/TVA/Lemmas/Sm90.lean` each make it stale, naming the file (vec-add's Lean imports 69 library
  files); a gate without `build` is stale.
- **Constants whose meaning changed.** None.

## L70: gap: the audit skips `kernels/` folders without `kernel.json`, and cited paths that do not exist

- **What changed.**
  - **`kernels/`**: `tvr audit` scans every file under `kernels/` that is in no kernel's folder (each kernel's
    folder is scanned as before), so a folder with no `kernel.json` is read;
  - **citations** (`citation_problems`, in `tvr audit`; protocol §2): in every Markdown and Lean file but the
    records (`docs/library-review/`, a kernel's `review/`: frozen, they cite files as they were), the evidence
    folders (whose scripts and outputs name their tests' scratch files) and Triton's pinned files, every path
    from TVA's root that starts `docs/` or `evidence/`, and in a Markdown file every link to a relative path
    (code spans aside), must exist. A path is read up to a placeholder (`round-<N>/`, `{kernel}`, `*.md`, `…`,
    an upper-case word such as `round-N`); a line number, an anchor and closing punctuation are not part of it;
  - the tree's three citations of missing files: `lean/TVA/Math/Float/Convert.lean` and `Format.lean` cited
    `docs/evidence/conformance/cvt.md`, which TVA never held (now "records TVA does not hold, library issue
    L65"; round 12 replaces both files), and `docs/review-questions.md` named `docs/examples`, removed in the
    restart of 2026-10-27 (now "the examples (a folder since removed)").
- **Why.** lr-10a G8: `kernels/notes/readme.md` holding a path into a sibling of TVA was not read, though it would
  ship. lr-9a: L65's citations of a record TVA does not hold passed the audit.
- **Evidence.** [evidence/tool-round13](../../../../evidence/tool-round13/README.md) U6: lr-10a's file is named by
  the audit; of eight citations in a probe page, the two missing files are named, and the line number, the
  anchor, the placeholders and the code span are not; lr-9a's citation in a Lean docstring is named; the tree
  now cites no missing file.
- **Constants whose meaning changed.** None (two docstrings of `TVA.Math.Float`, no statement).
