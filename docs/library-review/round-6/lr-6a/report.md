# Library review, round 6, by lr-6a

## Verdicts

- **L7** REOPEN — The text scan is as round 5 left it, and my spot probes go as §2 says
  (work/scan_spot.py: 15 probes, 0 wrong). The round's fix is the release check, which §2 calls
  "complete by construction": "a dependency on anything outside TVA, however it is spelled or
  computed, fails there, because nothing outside exists". Two routes pass it.
  (1) **The sandbox hides only /home and /tmp.** `cmd_release` mounts the whole root read-only
  (`--ro-bind / /`) and puts a tmpfs on those two only, so var/tmp, scratch, data, opt, srv, mnt, afs,
  hf, media and root stay visible inside; it also binds the whole base prefix of the Python that runs
  the tool (here an anaconda3 folder, with its site-packages and environments), so `-I -S` holds only
  until the code extends `sys.path`. work/release_scope.py: a copy of TVA whose `tvr audit` first reads
  a file under var/tmp, at a path computed at run time (the author's R2, with the file moved out of
  /tmp), passes the audit outside and **passes it inside the sandbox** ("audit: independent"), where
  R2 fails. (2) **The `packages` step sees only what git does not ignore.** `package_problems` runs
  `git status --porcelain --untracked-files=all`, which omits every ignored file: Mathlib's own
  .gitignore ignores `**/__pycache__/` and `.lake` at any depth, and `.git/info/exclude` (or a global
  excludes file) can hide any path. work/packages_ignored.py: in TVA's Mathlib checkout,
  `Mathlib/__pycache__/Lemmas.lean` is ignored; in a scratch repository with Mathlib's rule, such a file
  and a file listed in `.git/info/exclude` (`Mathlib/Analysis/FloatErrLocal.lean`, a normal-looking
  name) leave the step's command silent (only `--ignored` shows them); TVA's import check accepts
  `import Mathlib.__pycache__.Lemmas` (root `Mathlib`); Lean resolves it to
  `Mathlib/__pycache__/Lemmas.olean`; Lake's `isLocalModule` makes every `Mathlib.*` name a module of
  Mathlib's library, built from the package folder the sandbox binds (and not rebuilt there when its
  build is already present). So a lemma from a file that a fresh Mathlib clone lacks, another project's
  under a new name, passes `packages`, `audit`, the sandbox build and every gate; so does a built
  `.olean` that is not what Mathlib's source gives (every build output is ignored). **Fix:** build the
  sandbox from an allowlist (the system directories, the toolchain, the interpreter and its standard
  library, the packages; nothing else, or a tmpfs on every other top-level directory); in `packages`,
  use `--ignored` and allow only build outputs under each package's `.lake/`, and require every
  `Mathlib.*` module that TVA or a kernel imports to be a file Mathlib's git tracks; for build outputs,
  replay the kernels' imports from scratch (`leanchecker --fresh`) or rebuild the imported part of
  Mathlib in the sandbox, or state in §2 that Mathlib's build is trusted as found.
- **L20** ACCEPT — Unchanged since round 5, as L36 decides (the full read is the kernel reviewer's;
  the gate is a backstop for the common route). My rerun of evidence/gate-digests gives its out.txt
  line for line: every route of rounds 3–5 fails `namespace` or `hygiene` with the same reason (L, M,
  N, O, Q, R, S, T, U, V, W), the printing-option control X passes, and the restored copy passes (AD).
  I read the claim reader's generated-name test, the suffix, root-alias, plain-ASCII and binder checks
  against protocol §8: they say what §8 says. The kernel's manifest can switch the claim reader off
  entirely, which is a new issue below, not a defect of this fix.
- **L22** ACCEPT — The module docstring of `TVA.Lang.Desc`, SEMANTICS.md "Tensor descriptors" and the
  exclusions.md row say that descriptor accesses are unordered with the program's other accesses
  (L15), and name the two cases the exclusion adds (a pointer atomic with a descriptor reduction; two
  reads, Q11). The text no longer reads as if a program were sequential, and L36 only moved the host
  descriptor sentence of `[desc]` into SEMANTICS.md.
- **L25** ACCEPT — SEMANTICS.md "How the front end evaluates the source" states what Triton 3.8.0's
  code generator does: `visit_BoolOp` visits operands in order and returns at the first non-tensor
  value that decides `and`/`or`; `visit_Compare` makes `is`/`is not` compile-time whatever their
  operands; `visit_If`/`visit_IfExp` evaluate the condition and visit only the branch taken, so an
  `elif` after a taken branch is never visited (`primitives/triton-3.8.0-code-generator.py:967-1049,
  1076-1093, 1481-1531`). "Compile-time value" is defined as the code tests it ("not a tensor").
  evidence/config-cond (C1–C4, P1–P3, S1–S16) and evidence/debug-calls rerun on GPU 0 give their
  out.txt line for line. TRANSCRIPTION.md §4's worked examples follow from these facts.
- **L31** ACCEPT — SEMANTICS.md "Compiler hints" names the withdrawn reading as withdrawn and leaves
  L1 and L2 open; "No barrier semantics" gives the reason with `Step.defer`; the `dot_add` bullet says
  `±0` after constants are folded, with examples. Moving these from the withdrawn rules to
  SEMANTICS.md changed no meaning.
- **L32** ACCEPT — Subsumed by L36's whole-file digests: a module docstring (Y), the docstring of an
  unprojected field (Z) and `rules/compiler-faithful.md` (AA) re-open the items that reach their files,
  and `rules/num.md` re-opens nothing for a kernel that cites no `Num` fact (AB), in my rerun as in the
  author's. No library constant lies outside the `TVA` namespace (work/NonTVANames.lean: 0 of the
  library's constants), so every cone constant's module file is hashed. A latent gap in `RULE_PAGES`
  for future submodules is filed below.
- **L35** ACCEPT — The seven items read as asked: `DivFull` as the relative bound (with the 2.11 ulp
  measured by library review), no `tl.maximum` in `TVA.Lang.Scalar`'s list, the integer wrap of
  `tl.full` in SEMANTICS.md "Constants", `tl.ItofRel` with the `rtz` model, `tl.sum_bcast_mul` giving
  `a + a` no meaning, "library review" in `Num.lean`, and protocol §2 naming the shell's run-time
  expansions.
- **L36** REOPEN — Most of it holds. Digests: my rerun of evidence/gate-digests reproduces every case
  (a docstring, whitespace or a comment in `Lang/Ops.lean` re-opens exactly the four items that reach
  it; a fact page re-opens every item; a module no item reaches, the kernel's proof and
  TRANSCRIPTION.md re-open nothing); with no library constant outside `TVA`, no cone constant escapes
  its file. Guidance: every Triton fact TRANSCRIPTION.md uses that I checked is in SEMANTICS.md, but
  one (the store's cache modifiers, a gap below). The map check, the kept mechanical checks and the
  prompts are as the decisions say. What fails is the release check, which L36 adds as the blocking
  guarantee "complete by construction rather than by spelling": the sandbox hides only /home and /tmp,
  and its `packages` step does not see what git ignores, so a real dependency outside TVA passes (both
  routes, with the evidence and the fix, under L7).

## New issues

- **NEW** soundness — **The kernel's manifest can replace what the claim reader reports, so a claim
  of any shape passes the gate.** `check` builds the claim reader by splicing `lean.namespace` from
  `kernel.json` (the prover's file) into `tools/claim_reader.lean.in` (`@NS@`) as Lean text, unchecked,
  and trusts the first line that starts with `TVR_JSON `; the hygiene list reads only `.lean` files.
  work/ns_injection.py, on a copy: with `main : True` and `consistent : True` (the honest proofs kept
  under other names) the gate fails `shape` (B); with the namespace set to a string that closes the
  reader's first `example`, prints an earlier honest JSON line and ends the file with `#exit`, **the gate
  passes every check** (C): shape, axioms, usage, namespace, kinds, map, and digests computed from the
  forged JSON. The release check runs the same gate, so it passes too. claim.md §1 then shows the
  injected text, so a reviewer who reads it closely can see that something is off, but the gate's
  shape, axioms and namespace checks, which L36 keeps because they are complete, are void. **Fix:**
  require `lean.namespace` (and `lean.module`) to be dotted plain identifiers
  (`[A-Za-z_][A-Za-z0-9_']*(\.[A-Za-z_][A-Za-z0-9_']*)*`) and fail the gate otherwise, before
  anything is spliced; better, splice nothing the prover writes into Lean text.
- **NEW** gap — **The map's coverage check skips code lines that hold triple quotes.** In `check`, a
  line of the ranges is skipped when it contains `"""` anywhere, and an odd count toggles a
  "docstring" state that skips every line up to the next such line. work/map_coverage.py runs the
  tool's own loop: `tl.device_print("""x""", tl.atomic_add(c, 1))` is not required (K1), and with `"""`
  in two comments the atomic line between them is not required either (K2); Python reads code on
  each of them. Such a line needs no map entry, so claim.md's map table does not show it, and the
  gate reports "entries cover every code line" (§7). The kernel reviewer still reads the source, so
  this is a gap in a mechanical check rather than a hidden route. **Fix:** decide which lines are code
  with Python's `tokenize` (a line holding a token other than a comment, a newline or an indent, a
  docstring statement aside).
- **NEW** gap — **The library's own modules are never replayed, and its lake options are
  unchecked.** `replay` runs `leanchecker <kernel root>`, which replays only the kernel's modules into
  the environment of their imports; nothing replays `TVA.*`, and no hygiene list reads `lean/TVA`. A
  library declaration added by meta code without the kernel's check (or built under a package-wide
  `leanOptions` such as `debug.skipKernelTC`, which `lake_violations` accepts for any key) would pass
  every gate: the axioms check walks proofs but finds no axiom in a forged one. Today the library is
  clean: `leanchecker TVA` replays every library module with no problem in 2 min 16 s on this machine
  (work/leanchecker_TVA.out), and a search of `lean/TVA` finds no meta code, `sorry`, `axiom` or
  `set_option` beyond `autoImplicit`. "Lemmas: proved; Lean checks them" (LIBRARY_REVIEW.md) rests on
  that staying so. **Fix:** the release check (or `tvr audit`) runs `leanchecker TVA` and the hygiene
  list's escape, bypass and meta-code patterns over `lean/TVA`; `lake_violations` allows only an
  allowlist of `leanOptions` keys.
- **NEW** gap — **Nested `.lake` folders ship but are never scanned.** `text_files` skips every folder
  named `.lake` at any depth, while .gitignore ignores only `lean/.lake/`. work/nested_lake.py, on a
  copy: a note holding a path into a sibling of TVA, at `docs/.lake/note.md` or
  `kernels/vec_add_f32/lean/.lake/note.md`, is listed by `git ls-files --others --exclude-standard`
  (so the release check copies it) and `tvr audit` passes; the same note at `docs/note.md` fails the
  audit. A kernel's `lean/` folder gets such a folder whenever lake runs there. **Fix:** ignore `.lake/`
  at any depth in .gitignore (so it never ships), or have the scan read exactly the files the release
  check ships.
- **NEW** gap — **A library change re-opens nothing until someone re-runs the gate, and nothing
  notices a stale gate.** `current_gate` calls a gate stale only when a kernel file is newer; the
  library files in its items' digests are not compared. So after a library change, `tvr file` records
  acceptances against digests of the old library, REVIEW.md, kernels/README.md and `certificate.json`
  keep "GRANTED", and the release check, which recomputes every gate in the sandbox, does not compare
  the result with the shipped `gate.json` or certificate, so a release can ship a certificate whose
  items no longer have its digests. **Fix:** treat a gate as stale when a `file:` entry of its snapshot
  (or the packages digest) differs from the file now; in the release check, fail when a kernel's
  recomputed item digests differ from its shipped `gate.json`, or a shipped certificate names another
  gate digest.
- **NEW** gap — **A `TVA.Rules` module in a subfolder would get no rule page, and the audit would not
  say so.** `fact_page_problems` lists only the top level of `lean/TVA/Rules`, and `library_files` adds
  a page only for a module named exactly as in `RULE_PAGES`. A device-fact module such as
  `TVA.Rules.Num.Exp2` (L4 adds such facts) would leave `rules/num.md` out of the digests of the items
  that cite it, with no complaint. **Fix:** walk `lean/TVA/Rules` recursively, and give a submodule its
  ancestor's page.
- **NEW** gap — **`tl.store`'s cache modifiers have no stated meaning.** TRANSCRIPTION.md §4 drops
  "cache, eviction and `volatile` modifiers" (`no-op:cache`) and cites SEMANTICS.md `tl.load`, but
  SEMANTICS.md and `TVA.tl.store`'s docstring say nothing about `tl.store`'s `cache_modifier` and
  `eviction_policy`; only the load's are stated. **Fix:** state in SEMANTICS.md (and the docstring)
  that the store's modifiers change no result in a well-behaved launch, or that they are open.
- **NEW** wording — **Protocol §2 overstates what the release check runs.** It says a dependency on
  anything outside TVA "fails there"; the check runs the build, the gates, the lints and the audit, so
  evidence scripts and reviewers' work scripts, which are never run there, are checked by the text
  scan only (a computed path in them passes). **Fix:** say which steps the guarantee covers.

## Re-ran

- evidence/gate-digests/run.py on a copy (TMPDIR outside the home directory): output identical to
  evidence/gate-digests/out.txt (work/gate-digests.rerun.out).
- evidence/config-cond/run.py and evidence/debug-calls/run.py on GPU 0 (Triton 3.8.0, H200 NVL):
  identical to their out.txt after the version line (work/config-cond.rerun.out,
  work/debug-calls.rerun.out).
- The release check on a copy with a probe outside /home and /tmp: work/release_scope.py and .out
  (the copy's other steps are as the author's R0: packages and build pass, `vec_add_f32`'s gate fails at
  its build, and the round-6 records still in progress fail lint, as expected mid-round).
- New tests: work/ns_injection.py (the gate bypassed through `kernel.json`), work/map_coverage.py (the
  coverage loop), work/packages_ignored.py (git-ignored files in the packages; Lean's resolution of the
  import), work/nested_lake.py (shipped folders the scan skips), work/scan_spot.py (15 text-scan
  probes), work/NonTVANames.lean (library constants outside `TVA`), `leanchecker TVA`
  (work/leanchecker_TVA.out).

## Not checked

- evidence/independence-audit (its script takes the other project's name, which I neither know nor
  sought); the scan was spot-checked with work/scan_spot.py instead.
- The packages route end to end inside a real release sandbox: I did not put a file into TVA's shared
  Mathlib checkout; its parts were tested separately (git, the import check, Lean's resolution, Lake's
  source).
- Whether `leanOptions = {debug.skipKernelTC = true}` takes effect through lake (no lake build).
- The transcription and semantics areas beyond the facts L25 and L36 moved (other reviewers' survey
  areas), and the device facts.
