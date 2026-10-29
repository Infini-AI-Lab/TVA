# Library review, round 10, by lr-10b

## Verdicts

- **L7** REOPEN — The fetch is as protocol §2 describes it. Mathlib's URL is fixed by the audit.
  Its pin must be the upstream tag `v4.35.0`'s commit (`ls-remote`). The inherited packages must
  equal Mathlib's own manifest at that commit, field for field. Git runs over https only, with no
  user or system configuration and none of the caller's `GIT_*` variables, and with the system's
  CAs. Every checked-out file is compared with its blob. So the fetch lets in what upstream Mathlib
  `v4.35.0` pins and nothing else, and a moved tag fails closed. My rerun of evidence/tool-round10
  U9 and N1–N3 matches out.txt. The coordinator's release check of the merged tree passed
  `packages` and `sandbox`.

  One route of lr-7b's kind is left: the check takes the programs it runs from the caller's PATH.
  These are git (`GIT = ["git", …]` in `make_packages`, `upstream_problem`, the `ls-tree` that
  `checkout_problems` trusts, and `shipped_files`), bwrap (`shutil.which("bwrap")` in `cmd_release`)
  and find (`system_problems`). `_git_env` also keeps that PATH. In this session it lists four
  folders of the home directory before /usr/bin. In work/probes_batch.py P3, wrappers placed first
  on PATH run as all three. A git there decides both the packages' files and the `ls-remote` and
  `ls-tree` answers the checks compare them with. A bwrap there decides whether the sandbox isolates
  anything, and the `sandbox` step only checks that the probe starts. Every step could then pass. A
  git installed in a conda environment is enough to take the choice out of the system's hands.
  **Fix:** run /usr/bin/git, /usr/bin/bwrap and /usr/bin/find. The binaries must resolve under
  /usr, which the sandbox step already checks for user-placed files, as `system_python` does. Give
  git a PATH of system folders only, and say this in protocol §2.
- **L36** ACCEPT — lr-7a corrected the reasoning, and that correction is now in
  evidence/release-check, conclusion 5: 4 of 5 builds agree and R0 is an outlier. Protocol §2 says
  that the toolchain supplies `Init`, `Std` and `Lean` and that its files sit unchecked in the
  user's home. The coordinator's release check of the merged tree passed 8 of 10 steps: shipped,
  packages, sandbox, build, library, records, the lint and the audit. The two failures belong to
  `vec_add_f32`, as expected: its build, its stale gate and certificate, and its pages. The release
  check's remaining route is tracked under L7.
- **L40** ACCEPT — `brace_string_problems` refuses every string literal holding `{`, raw strings
  included, in the library's code and in a kernel's Lean. `library_module_problems` requires the
  closure from `lean/TVA.lean`. My reruns of U1 and U2 are identical to out.txt. I added cases in
  work/probes_batch.py P4: a guillemet name holding `"`, the character literals `'"'` and `'\''`, a
  primed name before a string, and `r#"a"b"#`. In each, `_uncommented` keeps the `#eval` or
  `open … in` after the token as code, so nothing hides there.
- **L48** ACCEPT — `canonConst` writes out each constant's kind, name, universe parameters, type,
  value and binder kinds, and the data for its kind. Every atom delimits itself: strings and name
  components carry their lengths, numeric components a terminator, and terms and levels are tagged
  and bracketed. So two different constants get different texts, and only binder names and `mdata`
  are dropped. In my rerun of Part L, A and B differ in `model` and `assumption:own`, C equals A, and
  D differs. The `conditions` and `spec` digests now differ from out.txt. These items hash
  SEMANTICS.md, exclusions.md and `TVA/Claim.lean` whole, and round 8 changed those files after the
  author's run, so this is expected. I found no other change to a digest's parts. The roots'
  docstrings, the whole library files, the source's sha256 and Lean's version are as protocol §4
  says. L55's rule pages by name add no file to today's cones. The source's sha256 is unchanged for
  LF files. So the only change is the per-constant hash, as the notes say.
- **L50** ACCEPT — The map check requires the ranges to hold every decorated `@triton.jit` function
  unless `source.unused` names it, and `manifest` checks `unused`. My rerun of U3 is identical:
  lr-7a's M1 ranges now fail, and a helper outside the ranges fails. JIT functions made by a call or
  under an aliased decorator still escape the check. That is a narrower route, filed below as a new
  gap.
- **L51** ACCEPT — Over all of Unicode, `LINE_SEPARATORS` matches exactly the characters other than
  `\n` at which `str.splitlines` breaks. CRLF is allowed. claim.md splits at line feeds. My rerun of
  U4 is identical.
- **L52** ACCEPT — `certificate_mismatch`, `ledger_problems`, the `records` step and the library
  ledger do what the issue asked. My rerun of U6 is identical. In the coordinator's release check,
  `records` passed, and every round before 10 is recorded (19 folders, plus the three author folders
  of rounds 8 to 10).

  The coordinator asked whether the ledger is faithful and what a hand-written line could let
  through. One appended line re-freezes an edited record, and so does an edited sha256 in the
  original line. Filed as a new gap below.
- **L53** ACCEPT — `gate.json` records the sha256 of every kernel file and of the gate's code.
  `gate_staleness` compares contents. The release check compares `files`, `tool` and the pages,
  with the gate's time masked. My reruns of U5 and U6 are identical. In the coordinator's run, the
  shipped `vec_add_f32` gate is stale on its kernel files and on the gate's code. `pages` names
  REVIEW.md and kernels/README.md, and no longer claim.md.
- **L54** ACCEPT — `TOOLCHAIN` must match the file's single line exactly, and a carriage return
  fails. `elan_parts` refuses links, and `make_packages` requires the toolchain that Mathlib's
  commit pins. My rerun of U7 is identical. The coordinator's sandbox shows `leanprover/lean4:v4.35.0`
  with no link. A toolchain link inside the folder would dangle in the sandbox, so it fails closed.
- **L55** ACCEPT — `kinds_problems` refuses any `TVA.Rules.*` constant whose module does not start
  with `TVA.Rules.`. A module named `TVA.Rules` itself is refused too. `library_files` also takes the
  page from the name. My rerun of U8 is identical.
- **L56** ACCEPT — Each item now reads as asked:
  - the toolchain's trust in protocol §2;
  - the masked output: the coordinator's output names only `<dir>`;
  - the mid-round runs, in the prompts;
  - `[kernel]` proved through `consistent`, in protocol §2, §4, §5 and §9, REVIEWER_GUIDE, PROVER_GUIDE
    and the prompts. I checked the claim: a field free of `K` and `P` cannot mention an earlier field
    either, since a projection carries `K` and `P`, so `consistent` proves it;
  - the skipped grids, in protocol §6 and DESIGN.md;
  - DESIGN.md's row.

  One sentence of the same kind is left in `Rules/Compiler.lean` (round 8's file), filed below as
  wording.

## New issues

- **NEW** gap — Scope 4 of `Compiler.Faithful` misses the options of a precompiled kernel. It
  excludes compile options given "as a launch keyword" or "in an autotune `triton.Config`" (in
  `lean/TVA/Rules/Compiler.lean`, docs/rules/compiler-faithful.md, exclusions.md row 39 and
  SEMANTICS.md "Launches"). A `CompiledKernel` (scope 6, tutorial 02) gets its options at
  `warmup(…)`, which takes the same keywords (`triton-3.8.0-jit.py:362-363`), or at
  `triton.compile(…, options=…)` (`triton-3.8.0-compiler.py:226`). Its launch then takes none.

  work/warmup_options.py, on GPU 0, ran `k.warmup(…, enable_reflect_ftz=False)` and then
  `compiled[(1, 1, 1)](…)` with no keyword. `tl.math.exp2(-130)` gave `2^-130` instead of the
  default's `0`. Scope 4 as written is satisfied there. **Fix:** say "given to the call that compiles
  the kernel (a JIT launch, `warmup`, `triton.compile`'s options) or in an autotune `triton.Config`",
  and have the reviewer's host check name `warmup`.
- **NEW** gap — A claim cannot say that the launch leaves the rest of memory alone. Round 8 (L49)
  added a "Memory" paragraph to Faithful, in `lean/TVA/Rules/Compiler.lean` and the rule page, and
  added the `init` bullet in `TVA/Claim.lean`. That paragraph puts "what the device or anything else
  does" to bytes `init` does not give outside the outcome, and `Outcome.ok fin` holds only `init`'s
  bytes. So no certificate excludes a launch that writes past its buffers into other tensors.

  A user who composes launches on the host needs that excluded. Trusting the compiler implies it
  anyway: the model touches no other byte, and only the descriptor scratch of scope 5 is written
  outside. **Fix:** state in Faithful that a valid, well-behaved launch in scope writes no byte the
  host allocated outside those `init` gives, its descriptor scratch aside. Say in `TVA.Claim` what a
  claim then implies about other memory. Or else say plainly there that a claim carries no such
  frame.
- **NEW** gap — The L50 check finds only some `@triton.jit` functions. In `tools/tvr.py`,
  `jit_functions` finds only functions under a decorator spelled `….jit` or `jit`. So a JIT function
  made by a call (`helper = triton.jit(_helper)`), or under a decorator bound to another name
  (`from triton import jit as tjit`, `deco = triton.jit`), need not be in the ranges.

  In work/probes_batch.py P1, with ranges holding only the kernel, the map check passes and the
  helper's atomic line is unmapped. claim.md §6 still says that the kernel lines hold every
  `@triton.jit` function, and REVIEWER_GUIDE §1 says that the gate checks this. **Fix:** also take
  functions passed to a call that names `jit`, and decorators bound to `jit` under another name.
  Otherwise fail on a source that names `jit` in any other way. Have the guide ask reviewers to
  check that the ranges hold every function the launch runs.
- **NEW** gap — The ledgers hold files to their lines, but nothing holds the lines to their history.
  `ledger_problems` (`tools/tvr.py`) checks each file against the sha256 recorded *last* for it.
  `tvr record`, `tvr file` and `tvr respond` all refuse to record a file again with another sha256.
  Even so, one hand-written line appended to `docs/library-review/ledger.jsonl` re-freezes an edited
  record, and an edited sha256 in the original line passes too. In work/probes_batch.py P2, round 7's
  lr-7a report was changed to "**L7** ACCEPT", and both routes passed the `records` step. In a
  kernel's ledger, appended `assign` and `accept` events would likewise make the certificate that the
  release check regenerates GRANTED.

  The rounds 1–7 entries also freeze the files as they stood in round 10. For example, round 2's
  author notes differ from the committed version by a masked path. Those entries name the
  coordinator as their actor, though the author ran them. So they record what ships, not what the
  reviewers read. **Fix:** have `ledger_problems` refuse a file recorded with two sha256s and a
  library folder recorded twice, as the tool itself does. Say in protocol §3 and §10 and in
  LIBRARY_REVIEW.md that a ledger shows a file is unchanged since its line, not that the line is
  unchanged. The git history, or a ledger digest kept outside the tree, shows that.
- **NEW** wording — Three gaps in the wording of Faithful's scope:
  - **The target.** Faithful's docstring, its rule page and `TVA.Claim` speak of "the device" but
    never say which. exclusions.md row 45 limits TVA to the H200 (sm_90), and the model's defaults
    are sm_90's: `DotCfg`'s `tf32` and the fp8 `maxImpreciseAcc = 2^30`. A claim that cites only
    `Compiler.Faithful` reads as one for any GPU. Name the target as a scope condition.
  - **Skipped grids.** In `lean/TVA/Rules/Compiler.lean`, Faithful's docstring still gives the
    reason for `2^31` as "since Triton's launcher skips larger grids without an error". It should say
    "`2^31` to `2^32` are skipped, and a larger grid may run", as L56 made protocol §6 say.
  - **Host descriptors.** Scope 6's list of argument classes does not name host tensor descriptors.
    A descriptor's specialisation is its dtype and block shape (`triton-3.8.0-specialize.cc:165-226`).

## Re-ran

- evidence/tool-round10 in full (U1–U9, N1–N3 with the network, L) on the merged tree,
  work/tool-round10.rerun.out. It is identical to out.txt except in three places:
  - the timings;
  - `release_records`, which now also lists this round's two reviewer folders, still in progress;
  - Part L's `conditions` and `spec` digests (see L48).
- evidence/other-writers (W1–W3), on GPU 0 (Triton 3.8.0, torch 2.11.0+cu130, H200 NVL,
  2026-10-28), work/other-writers.rerun.out. It is identical to output.txt but for timings: 20 of 20
  correct with a neighbour writer, in the same 4-byte words too, and the control corrupted.
- The coordinator's release check of the merged tree, which I read but did not re-run: 8 of 10
  steps pass, and the failures are `vec_add_f32`'s gate and pages.
- `tvr audit` on the tree. It names only another reviewer's work file, which is in progress.
- New tests, each with its output in `work/`:
  - probes_batch.py: P1 (L50), P2 (L52), P3 (L7) and P4 (L40);
  - warmup_options.py (scope 4, GPU 0).

## Not checked

- The release check itself, which only lr-10a may run again. Its `packages` step was checked through
  tool-round10 Part N, and its PATH lookups through P3.
- evidence/gate-round7, evidence/release-check R1–R3, and evidence/independence-audit (which takes
  the other project's name).
- The round 8 evidence behind the scope other than other-writers: pdl, launch-options,
  launch-outcomes, compiled-launch, autotune-launch and desc-scratch. I read it against the pinned
  sources (`knobs.py`, `GetEnv.h`, `CUDAOptions`, `jit.py`, `autotuner.py`, `nvidia-driver.py`,
  `specialize.cc`) but did not re-run it.
- The semantics, transcription, device and lemma areas, beyond what the rules rest on.
