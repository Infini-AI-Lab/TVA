# Library review, round 7, by lr-7b

## Verdicts

- **L7** REOPEN — The sandbox is the allowlist protocol §2 describes. My probe runs the tool's own
  bwrap command line (work/sandbox_probe.py, .out). Inside it sees only `bin`, `dev`, `etc`, `home`,
  `lib`, `lib64`, `proc`, `sbin`, an empty `tmp` and `usr`. The home holds only `.elan`, with the
  pinned toolchain. There is no `var`, `opt`, `run` or `sys`. Two PIDs are visible; the environment is
  HOME, LANG, PATH and PWD; `sys.path` is the system's standard library only. There is no network, and
  `usr`, `etc` and elan's settings are read-only. My release check on the tree (work/release.out,
  release.json) passed `shipped` (512 files), `packages`, `sandbox`, `build` (Mathlib built from
  source) and `library`. It failed what is expected mid-round: the `vec_add_f32` gate and its stale
  shipped gate and certificate, the lint of this round's reviews in progress, and the audit (on the
  other reviewer's in-progress output).
  One route remains. The `packages` step runs git outside the sandbox, with the user's HOME, so the
  user's git configuration decides what the checkout writes. work/git_config_probe.py runs
  `tvr.make_packages` on one package with a scratch HOME whose `.gitconfig` names a global attributes
  file giving `*.lean` a smudge filter (the mechanism git-lfs uses). The step reports no problem, but
  `git status` is not clean: 5 of the package's 12 files, every `.lean` file, differ from the commit's
  blobs, each with a line the commit lacks. So a file outside TVA, the user's git configuration, can
  put any text, a lemma included, into the Mathlib source that the sandbox builds and TVA imports, and
  every step passes. "Git checks every object it receives against its hash" holds for the objects,
  not for the files the checkout writes.
  **Fix:** run the packages' git with no global or system configuration (`GIT_CONFIG_GLOBAL=/dev/null`,
  `GIT_CONFIG_NOSYSTEM=1`, an empty HOME and XDG_CONFIG_HOME, `-c core.attributesFile=/dev/null`), or
  run it inside the sandbox. Then check every checked-out file against its blob
  (`git hash-object --no-filters` against `git ls-tree -r HEAD`), with no other file present.
  `shipped_files` reads the user's excludes file the same way, so run it without user configuration
  too.
- **L31** ACCEPT — The "Compiler hints" section of `lean/TVA/Lang/Ops.lean` now says what
  SEMANTICS.md says:
  - an in-loop hint on a carried name or on the induction variable is open (L1);
  - the reading that drops such a hint is named as withdrawn, with the reason;
  - per-dimension values, and a `contig` above 1, are open (L2).

  The `multiple_of` and `multiple_ofN` docstrings no longer define `contig` by Triton's contiguity,
  and no longer ask the reviewer to check it. Their `contig = 1` reading (every element a multiple of
  `v`) is stronger than Triton's, so it is a sound check. A search of `lean/TVA`, `docs/*.md` and
  `docs/rules` finds no other stale hint text: every hit names the withdrawn reading as withdrawn. The
  other two items were accepted in round 6 and are unchanged.
- **L36** ACCEPT — The parts accepted in round 6 (digests, guidance, map check, prompts) are
  unchanged. The release check, the part lr-6a reopened, is L7's fix, and I judge it under L7. On the
  coordinator's question, building Mathlib from source in each run holds up:
  - A pinned digest of a build means something only if a from-source build at the same path
    reproduces it. TVA's own measurement shows the `.olean` files agree bit for bit across three copies
    at paths of one length, and differ at a fourth path.
  - When TVA imports Mathlib, Lean reads more than the `.olean` files: each module also has `.ir`,
    `.olean.server` and `.olean.private` files in Mathlib's build here.
  - A cached build sits in a folder the user can write, so a run would have to verify it again, and
    that means building it again.

  The cost is bounded: my run took about 27 minutes beside the other reviewer's, needed 13 GB, and
  removed its sandbox afterwards.
- **L38** ACCEPT — `check` runs `manifest_problems` first. On a failure it writes `gate.json` and
  returns before anything is read, built or spliced. `LEAN_NAME.fullmatch` admits only dotted ASCII
  identifiers, so the substituted text can be nothing but a name: no space, line break, bracket, `#`,
  `-` or quote. Opened names are spliced only in that form, and the reader must print exactly one
  `TVR_JSON` line. evidence/gate-round7 Part U, U2 included, re-ran identical. A keyword such as `end`
  or `sorry` passes the form check (work/gate_probes.out), but at worst the reader then fails to
  parse, or the namespace check fails: the kernel would have to declare its constants in that
  namespace, and the kernel hygiene refuses `sorry`. So no claim gets through that way.
- **L39** ACCEPT — `code_lines` finds code lines with `ast` and `tokenize`. Part U's cases re-ran
  identical. I added five cases (work/gate_probes.py): a bare f-string holding an atomic,
  `"""…""".format(<atomic>)`, implicit concatenation over several lines, a line continuation, and
  `'x'; <atomic>`. In each, every line that holds code counts, and only bare string statements are
  exempt. A line-numbering mismatch between the tokenizer and claim.md is filed below; it is not a
  defect of the coverage check.
- **L40** REOPEN — Replay and options hold. My release check's `library` step ran the library
  hygiene and `leanchecker TVA` inside the sandbox, on a build made there, and passed. On a copy
  (work/audit_probes.out), `lake_violations` refuses `debug.skipKernelTC` (flat and as a nested
  table), `trace.profiler`, `Elab.async` and a string value, and accepts `maxHeartbeats` and
  `pp.all`. Two defects remain:
  1. The comment reader `_uncommented` misreads an interpolated string that holds a nested string.
     This round made it read string literals "as Lean's lexer does, so a `--` or `/-` inside one
     cannot hide a command after it". But Lean runs an `#eval` that follows `s!"{"--"}"` or
     `s!"a{"/-"}b"` on the same line (work/InterpStr.lean, interp.out), while the hygiene's view
     blanks it and flags nothing (gate_probes.out). The same reader feeds `opened_names`.
     `def s : String := s!"{"--"}"  open Nat in` opens `Nat` for Lean (InterpOpen.lean: `zero` is
     `Nat.zero`), but the namespace check never sees it, and to a reader it looks like a comment.
  2. `lake build TVA` builds, and `leanchecker TVA` replays, only the modules that `lean/TVA.lean`
     reaches through its imports. A module under `lean/TVA` outside that closure is built only when a
     kernel imports it, during that kernel's gate, which runs after the `library` step. The kernel's
     `replay` covers only the kernel's own modules, so that module's declarations are never replayed.
     Nothing requires the closure to be complete: an orphan module passes the audit
     (audit_probes.out). Today every module is in the closure.

  **Fix:**
  - In `_uncommented`, treat a `{` inside a string as opening a nested region, strings included, up to
    its `}`, and keep that region as code. Alternatively, refuse interpolated strings in library and
    kernel Lean; none uses one today.
  - Make the audit require that `lean/TVA.lean` imports every module under `lean/TVA`, or replay the
    library after the gates.
- **L41** ACCEPT — `.gitignore` ignores `.lake/` at any depth. `shipped_problems` refuses a shipped
  file under `.lake`, `.venv`, `.git` or `__pycache__`, and refuses a nested repository. On a copy
  (audit_probes.py), an untracked note at `kernels/vec_add_f32/lean/.lake/note.md` is not shipped and
  the audit passes; force-added, it fails the audit, which names it. My release check's `shipped` step
  passed on the tree.
- **L42** ACCEPT — `gate_staleness` compares every `file:` entry of the gate's snapshot, and the
  packages' digest, with the tree, and it calls a snapshot older than round 6 stale. `current_gate`
  refuses a stale gate, and `cert_state` needs a gate that is not stale. A read-only call on the tree
  reports the shipped `vec_add_f32` gate as stale. My release check failed that kernel's gate step on
  "the shipped gate.json is stale … the shipped certificate.json names a gate other than the
  recomputed one". The kernel-file side of staleness predates this round and is filed below.
- **L43** ACCEPT — `rule_page` gives a submodule the page of its nearest ancestor in `RULE_PAGES`:
  `TVA.Rules.Num.Exp2` gets `num.md`, in `library_files` as well. `fact_page_problems` walks
  `lean/TVA/Rules` at any depth. On a copy, `Rules/Num/Exp2.lean` passes the audit, and
  `Rules/Extra/Y.lean` fails it and is named.
- **L46** ACCEPT — Each item now matches the sources and the evidence:
  - Protocol §2 and the comment in `tools/tvr.py` name the steps the sandbox runs, and say that
    evidence and work scripts are not run there.
  - "Order" matches `visit_Call` (`primitives/triton-3.8.0-code-generator.py:1455-1476`), which
    visits the function, then `node.keywords`, then `node.args`.
  - The statements on the conditional expression and on `launch_metadata` match the sources:
    `CompiledKernel.launch_metadata` returns `None` when no enter hook is installed, and otherwise
    defers the call to `LazyDict.get` (`triton-3.8.0-compiler.py:374-387, 493-502`).
  - evidence/front-end-order re-ran on GPU 0, identical after its date line.
  - The `Desc` docstring calls in-kernel descriptors open (L5).

  DESIGN.md still repeats the release check's old reach; that is filed below as wording.

## New issues

- **NEW** gap — Rules survey: `Compiler.Faithful`'s first scope condition is stronger than claims
  need. "The launch runs alone" means that nothing else, on any stream, device or host, writes global
  memory while it runs. That excludes every launch that shares the GPU with other work, such as
  another stream or another process, and a kernel reviewer cannot check it from host code. Yet a write
  to bytes the launch is not given cannot change the launch's outcome in the semantics or on the
  device.
  - **Where:** `lean/TVA/Rules/Compiler.lean` ("Scope" 1), `docs/rules/compiler-faithful.md`,
    `lean/TVA/Claim.lean`, SEMANTICS.md "Launches", `exclusions.md`.
  - **Fix:** "from its start to its end, nothing else writes the bytes `init` gives the launch (the
    buffers of `Pre`)". Keep the PDL case as an instance.
- **NEW** gap — Gate staleness on the kernel side uses modification times only, and leaves out the
  source file and the write-up. `gate_staleness` compares the mtimes of the kernel's Lean files, the
  map and `kernel.json` with the gate's time. So an edit of `source/<file>.py` or `README.md`, or of a
  Lean file whose old mtime is preserved (`cp -p`, `touch`), leaves the certificate GRANTED and lets
  `tvr file` proceed. The release check catches this, but between releases nothing does. The release
  check also does not compare the shipped `REVIEW.md` and `kernels/README.md`, which can still say
  GRANTED. Today the tree's `vec_add_f32` shows GRANTED there, while its certificate has lapsed.
  - **Fix:** record the digests of the kernel's files (source and write-up included) in the snapshot
    and compare those digests. In the release check, also regenerate the generated pages and compare
    them.
- **NEW** gap — `lean/lean-toolchain` is not checked. The audit passes with
  `someone/lean4:v4.35.0-patched` and with `leanprover/lean4:nightly` (audit_probes.out). The release
  check binds whatever toolchain folder the name gives, and through `os.path.realpath` it binds the
  target of a folder that elan's `toolchain link` made. Lean and `leanchecker` are "trusted as
  installed", so that trust should rest on an official release.
  - **Fix:** make the audit require `leanprover/lean4:v<version>`, and make the release check refuse
    a toolchain folder that is a link. Today the file names `leanprover/lean4:v4.35.0`, a real folder.
- **NEW** gap — claim.md can show reviewers the wrong source line or the wrong Lean declaration. It
  numbers lines with `str.splitlines()`, which also breaks at U+000B, U+000C, U+001C–U+001E, U+0085,
  U+2028 and U+2029. Python's tokenizer (the map check), Triton and Lean's declaration ranges count
  only line breaks. One U+2028 or form feed in a comment therefore shifts every later row of claim.md's
  map table and every later quoted declaration (gate_probes.out: the atomic's line is shown as the line
  before it, and `model` is quoted as the declaration before it).
  - **Where:** `check` (`src_lines = src_text.splitlines()`) and `render_claim`
    (`open(p).read().splitlines()[a - 1:b]`).
  - **Fix:** split at line breaks only, as the tokenizer and Lean do, or refuse such characters in
    sources and kernel Lean.
- **NEW** wording — Rules survey: a `[kernel]` assumption is proved, not "unproved". The `kinds`
  check keeps it free of `K` and `P`, so it is a closed proposition, and `consistent` (`∃ K P, Total K
  ∧ Assumptions K P`) proves it (work/KernelFieldProved.lean checks this with Lean).
  - **Where:** protocol §2, §4, §5 and §9 (`EVIDENCE_REQUEST`), REVIEWER_GUIDE §4, PROVER_GUIDE.
  - **Fix:** say that Lean proves it through `consistent`, and that the reviewer judges whether it is
    needed and clear, and whether the kernel constants it constrains mean what the write-up says.
- **NEW** wording — Rules survey: protocol §6 says that "the launcher skips grids of `2^31` programs
  or more without an error". TVA's own evidence (launch-outcomes C8) runs every program of a grid of
  `2^32 + 1` programs, whose product wraps to 1. `Launch.Valid` excludes all such grids, so no claim
  changes. Claim.lean and the rule page already say "`2^31` to `2^32`".
  - **Fix:** say "2^31 to 2^32 programs are skipped, a larger grid may run, and `Launch.Valid`
    excludes both".
- **NEW** wording — The release check's text outside protocol §2 has two problems:
  - DESIGN.md's decision row still says "so a dependency fails whatever its spelling". L46 limited
    this in §2 to the steps run in the sandbox.
  - Neither the author's nor the reviewer's prompt says what a mid-round run shows. It fails `lint` on
    the round's records in progress. One absolute scratch path in any work file fails the
    `independence` step of every kernel's gate, and the audit, in everyone's run. The check's own
    output names its scratch folder. In my run, the other reviewer's output did this, and the gate then
    wrote the path into `gate.json`, which the audit also flagged.

  **Fix:** correct the DESIGN.md row. Say both points in the prompts, or have `tvr release` print the
  sandbox path relative.

## Re-ran

- `tvr release` on the tree, once, in an empty scratch folder outside the home directory: about
  27 minutes beside the other reviewer's run (work/release.out, release.json; results under L7). The
  sandbox was removed afterwards.
- On GPU 0 (H200 NVL, Triton 3.8.0, 2026-10-28):
  - evidence/front-end-order: identical after its date line;
  - evidence/launch-options and evidence/env-knobs: identical after line 1;
  - evidence/launch-outcomes: identical but for timings.
- evidence/gate-round7 Part U: identical (work/gate-round7-partU.rerun.out).
- New tests, each with its output in `work/`:
  - `sandbox_probe.py`: the sandbox's mounts;
  - `git_config_probe.py`: L7;
  - `gate_probes.py`, with `InterpStr.lean`, `InterpOpen.lean` and `interp.out`: L38, L39, L40 and
    line numbering;
  - `audit_probes.py`: the audit on a copy for L40, L41 and L43, the toolchain name and an orphan
    module;
  - `KernelFieldProved.lean`: `consistent` proves a `[kernel]` field;
  - `scratch_survey.py`: only `tl.make_tensor_descriptor` asked for global scratch, 256 bytes. Loads,
    reductions, atomics, sort, gather, histogram and host descriptors asked for none, and a
    `warp_specialize` loop over pointer loads was rejected at compile time;
  - an import-closure check: every module under `lean/TVA` is reached from `lean/TVA.lean`.

## Not checked

- evidence/release-check R1–R3: each one builds Mathlib, and the coordinator asked for one release
  check per reviewer. I probed the sandbox's mounts directly instead.
- evidence/gate-round7 Part C. C5 builds the kernel; I repeated C1–C3 with the audit on a copy.
- The `vec_add_f32` gate inside my sandbox stopped at `independence`, on the other reviewer's
  in-progress file, so I did not see its own build fail.
- evidence/independence-audit, which takes the other project's name.
- evidence/pdl and evidence/intra-program.
- The device facts and the lemmas, which are other survey areas.
