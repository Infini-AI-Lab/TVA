# Library review, round 13, by lr-13a

## Verdicts

- **L7** REOPEN — Both parts asked for hold: with `git`, `bwrap` and `find` wrappers first on `PATH`
  the tool ran `/usr/bin/git` (2.52.0), `/usr/bin/bwrap` and `/usr/bin/find` and no wrapper, git got
  `PATH=/usr/bin:/bin`, and only `v4.35.0` (as a tag) and `master` (as a branch) pass as Mathlib's
  `inputRev`, in the audit and in the packages step before any fetch (evidence/tool-round13 Part N,
  re-run identical). Open: git still reads the configuration of the repository it is run from.
  `upstream_problem` runs `git ls-remote` (and, for `master`, `git clone`) without `-C`, in the
  caller's folder, which the protocol's opening paragraph makes TVA's root, a git repository; `_git_env` removes the
  user's and the system's configuration, not that repository's `.git/config`, which does not ship,
  is not scanned (`.git` is skipped) and the user can write. In `work/gitcfg.py` a repository whose
  `.git/config` holds `url.https://example.invalid/.insteadOf = https://github.com/` sends the
  tool's own `_git("ls-remote", Mathlib's URL, "refs/tags/v4.35.0")` and `upstream_problem` to that
  host when run from inside it; from a folder outside any repository both reach GitHub and accept the
  pin (`work/gitcfg.out`). So the host such a file names (or a proxy it sets, with `http.sslVerify`
  off) decides which commit is "the tag `v4.35.0`" or "`master`'s head", and the packages step then
  fetches that commit by its hash in a fresh repository, which protocol §2 itself says is not enough
  (a host may serve a commit no branch or tag holds). Fix: run every git that needs no repository
  (`ls-remote`, `clone`, the scratch `init`s) with `cwd` an empty scratch folder and
  `GIT_CEILING_DIRECTORIES` set to it, and say in §2 that no repository's configuration is read but
  the new package repository's own.
- **L50** ACCEPT — Every `def`, `async def` and `lambda`, at any depth, from its first decorator to
  its last line, must lie in the kernel lines, the host lines or (a `def`) `source.unused`, and one
  under a decorator naming `jit` not in the host lines (`map_problems`, `source_functions`). U1
  re-ran identical: vec-add passes and fails without its benchmark's host lines, each of the five
  spellings (`triton.jit(_h)`, an alias, `deco = triton.jit`, `@triton.constexpr_function`, a lambda
  made a constexpr function) fails with the helper unclassed, and the controls pass. The reasoning
  holds against the pinned sources (a JIT function is made from a `def`'s source,
  `triton-3.8.0-jit.py:465-474, 546-551`; a constexpr function runs any callable; the front end
  calls what it reaches, `triton-3.8.0-code-generator.py:387-416`), and claim.md §2, REVIEWER_GUIDE
  §1 and the reviewer's prompt ask the reviewer to judge the classes. The rule is complete for
  functions; module-level values a kernel reads are another route (New issues).
- **L52** ACCEPT — `tvr record` refuses `--no-lint` after round 7 and the records step names an
  unlinted line of a later round; `kernel_event_problems` derives the events after each `file` and
  `response` event with the functions `tvr file` and `tvr respond` write them with, and names lr-10a's
  hand-written accept, an edited accept, a release with no filing, a decision on a finding not
  escalated and a file recorded with two sha256s; the library ledger check names a folder recorded
  twice and a hand-written unlinted line; the certificate needs a ledger that holds to its records
  (`cert_state`). U2-U4 re-ran, identical but for U4, where round 11's reviewer folders, now in the
  merged tree, are named as unrecorded earlier records, and the ledger has one more line.
  The coordinator's release check of the merged tree passed `records`. The limit the user decided is
  stated in §3 and §10 ("trusted as the coordinator keeps them"); but the docs also say the lines
  "rest on the git history", which TVA cannot promise, and a reviewer's lines removed with their
  folder left in place go unseen (both under New issues).
- **L62** REOPEN — The `Platform → Prop` facts, the `…Total` rules, `DivRn P .binary64` and the
  refusals of shapes, families, `DivRn P .binary32`, `K` outside `Faithful` and conjunctions behave as
  stated (Part K re-ran identical; the 25 `Platform → Prop` constants of `TVA.Rules` are the 24 facts
  of `TVA.Rules.Num.Sm90` and `DivFull`, all in num.md's table), and the guides cite the facts by name
  with `Platform.sm90ref`. But `Compiler.Faithful` is accepted with any arguments after `P`:
  `rule_fact_problem` returns at `head == FAITHFUL`, and the own-constant check sees only TVA and
  kernel constants. In `work/probes.py` F1, a whole claim with `Inst := Bool`, a model whose instance
  `true` is a loop that never ends and whose `false` does nothing, `Post` "memory unchanged", and
  `faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model false) (launch d) (K d)` passes `shape`
  (with `consistent` from `exists_total_faithful`), `axioms`, `usage`, `namespace` and `kinds`:
  `main` proves every covered launch ends normally from the idle model, while the model of `true`,
  the one its launch runs, never ends. F2: `model e` or `launch e` for another bound `e`, and
  `model (Classical.choose h)`, pass as well. Protocol §5 gives the accepted form as
  `Compiler.Faithful P (model d) (launch d) (K d)`, REVIEWER_GUIDE §4 says the gate ensures the field
  applies the library's fact by its name, and nothing asks the reviewer to check Faithful's arguments.
  Fix (cheap and complete): have the reader report, for each argument, its head constant and the
  bound variable it is applied to, and accept `Faithful` only as `P (model x) (launch x) (K x)` for one
  variable `x` the statement binds; say so in §5 and §8. On the coordinator's question: the check
  depends on names (the last component `binary64`, the full names of `Faithful` and `DivRn`, the
  suffix `Total` under `TVA.Rules.Num`, and `TVA.Platform` in the reader). A rename of any of these
  refuses the fact (fails closed), except that a new `TVA.Rules.Num` constant whose name ends in
  `Total` is accepted with any arguments (New issues).
- **L63** ACCEPT — `kinds` counts as the kernel's own every in-scope constant of the statement not
  rooted at `TVA` but `Inst`, `Inst.ok`, `model` and `launch`, so `private def tol`, `Kernels._tol` and
  `_root_._tol` are each named; the namespace check exempts outside the namespace only names with a
  numeric component (`TVR.leanMade`, applied after `privateToUserName`, so a private `_root_` name is
  checked as written), and the instance check only the `SizeOf` instance `<type>._sizeOf_inst` of a
  type the kernel declares. Part K re-ran identical, and the control kernel (defaults, recursion, a
  private definition, `simp`, `omega`, `decide`, `norm_num`) has no namespace problem. A constant the
  kernel declares under `TVA` fails `namespace`, and one under `TVA.Rules` that a field cites also
  fails `kinds` (L55).
- **L68** ACCEPT — The reader writes `TVR.canonExpr` of each field projection's whole type (`K`, `P`
  and the structure bound as de Bruijn indices; literals, binder kinds and structure spelled out), and
  `item_digests` adds its sha256 to the field's item as `statement`, which the snapshot keeps, so
  `REVIEW.md` names it. Part D re-ran identical: lr-10a's `2^-40`, `∀ _ : Nat` and changed `[kernel]`
  statement each change exactly the edited item, a proof edit none. Protocol §4 and §13,
  LIBRARY_REVIEW.md and DESIGN.md say so.
- **L69** ACCEPT — gate.json records `build` (the toolchain file, the lakefile, the manifest and every
  `TVA` module the kernel's Lean files import, transitively, read with `lean_header`), `gate_staleness`
  compares it by contents and treats a gate without it as stale, and `gate_mismatch` compares it in the
  release check. U5 re-ran identical (69 imported files; the toolchain, a `leanOptions` key, a comment
  in `lean/TVA.lean` and one in `lean/TVA/Lemmas/Sm90.lean` each make the gate stale); the
  coordinator's release check names vec-add's shipped gate stale on the build's inputs among others.
- **L70** ACCEPT — `tvr audit` scans every file under `kernels/` outside the kernels' own folders, and
  `citation_problems` checks every `docs/…` and `evidence/…` path and every relative Markdown link in
  the Markdown and Lean files outside the records, the evidence and Triton's pinned files. U6 re-ran
  identical, and the tree cites no missing file. Skipping the evidence folders is right: the three
  missing paths their READMEs cite are test inputs (`work/probes.out`, Part C). One small miss, at
  wording level: a path component in capitals only (`docs/DESIGN`) is read as a placeholder and
  passes.

## New issues

- **NEW** gap — The map's coverage misses values a JIT function reads from module level. Triton
  3.8.0's front end resolves a name in a kernel to a module-level `tl.constexpr(…)`, a dtype, a
  namedtuple or a module's attribute at compile time (`triton-3.8.0-code-generator.py:387-420`), and
  the value is part of the compiled kernel (`triton-3.8.0-jit.py:532-536`, checked at every launch
  `:761-766`). Such a line is in no function, so nothing classes it: in `work/globals.py` a kernel that
  reads `SCALE = tl.constexpr(3)` and `OUT_T = tl.float16` from lines outside its kernel lines passes
  `map_problems` with every function classed and every code line of the ranges mapped, and on the H200
  it stores `(x·3)` rounded to `float16`. Protocol §7 ("covers the launched kernel and every helper it
  calls"), claim.md §2 and REVIEWER_GUIDE §1 speak of functions only. Fix (cheap and complete, as for
  functions): class every module-level code line outside a function as kernel lines or host lines,
  have the reviewer judge that every module-level value the launch reads is in the kernel lines, and
  state in SEMANTICS what such a name means in a kernel (it says only that a name the module does not
  define is looked up among the builtins).
- **NEW** gap — A kernel's ledger shows nothing of a reviewer whose lines are removed while their
  folder stays. Every event must follow from a recorded file, but no filed folder must have its events:
  in `work/probes.py` Part R (a scratch copy of vec-add with a passing gate made by hand), `r-two` is
  assigned and files a blocking `SPEC_TOO_WEAK`; with `r-two`'s `assign`, `file` and `raise` lines
  removed and the rest renumbered, `kernel_record_problems` and the records step name nothing,
  `tvr lint vec_add_f32 reviewer r-two` prints `regular` (the release check lints it as a filing to
  come), and the certificate is granted, with `review/round-2/r-two/filing.json` still in the tree.
  The library's ledger has the counterpart (an unrecorded role folder of an earlier round fails).
  Under the user's decision this is either a stated limit or a cheap check; I recommend the check: the
  records step and the certificate fail on a `review/round-N/<who>/` folder of a reviewer no `assign`
  names, or of an earlier round that no event records, and `tvr lint` refuses a reviewer not assigned.
- **NEW** gap — The `[rule]` check accepts facts by their type and name, not by the library's list:
  `rule_fact_problem` accepts any `TVA.Rules` constant of type `Platform → Prop` applied to `P` alone,
  and any `TVA.Rules.Num` constant whose name ends in `Total` with any arguments. Today these are
  exactly facts of num.md's table, but nothing tells a library author or reviewer that a definition of
  that type, or a name with that suffix, becomes citable by every kernel (the library author's prompt,
  LIBRARY_REVIEW.md "What a library reviewer checks"): a helper `def DivRnF32 : Prop := DivRn P
  .binary32` (false on the H200) or an accuracy shape named `…AccTotal` would pass `kinds` as a stated
  fact, and round 12 is rewriting the layer these facts are stated on. Fix (cheap and complete): accept
  a head only if num.md's "Facts the library states" table (protocol §4: "its list of the facts a
  kernel may cite") names it, and say in the author's prompt that a fact is citable once listed there.
- **NEW** wording — What TVA can promise about ledger lines. LIBRARY_REVIEW.md ("Rounds", step 4) says
  the lines "rest on the git history, or on the ledger's digest that the release check keeps in
  release.json", protocol §10 says the same, §3 that an edited line "is shown only by the git history
  or by a digest", and `ledger_digests` that "a later run can show a ledger only grew". The user
  decided that the lines are trusted as the coordinator keeps them, with no commit at each recording;
  TVA's history holds two commits (2026-10-27), the library ledger has never been committed (it is
  untracked), a release copy ships no history, and no step reads an earlier `release.json` back. Say
  that the lines are trusted as the coordinator keeps them, and that an edit shows only against a copy
  kept elsewhere (a commit, a kept `release.json`'s `ledgers`), compared by hand. The author's notes
  ask the coordinator to commit after each recording, which that decision declined. Also say in §3
  that `filing_events` re-derives old events with the tool's current `KINDS` defaults,
  `ESCALATE_AFTER` and escalation memo, so changing any of them makes every older ledger fail the
  records step (or derive those from what the event recorded).
- **NEW** wording — What a gate in the tree rests on. Protocol §8 and §9 say the certificate needs a
  passing gate that is not stale, but not that gate.json, its snapshot and claim.md are as the last
  `tvr check` wrote them in its caller's environment (`TVR_LAKE` picks the `lake` it runs,
  `tools/tvr.py:61`; elan the toolchain; the local `.lake/packages` the Mathlib it builds) or as
  anyone edited them since: `gate_staleness` compares the tree with what gate.json says it read, so a
  gate.json made by hand is current (evidence/tool-round13 U3 makes one, and Part R here grants a
  certificate on one). Only the release check recomputes them in a clean sandbox. Say so in §8 and §9, as §3 now does
  for ledger lines.

## Re-ran

- evidence/tool-round13/run.py in full (Parts K, D, U1-U6 and N, with the network), on the merged
  tree: identical to its out.txt except U4, where round 11's reviewer folders (in progress in the
  merged tree) are named as unrecorded records of an earlier round and the ledger's event numbers are
  one higher (round 11's author's notes recorded since).
- The coordinator's release check of the merged tree (read, not re-run; I did not run my own): 8 of 10
  steps pass (shipped, packages, sandbox, build, library, records, the lint of vec-add's round 2,
  audit); vec_add_f32's gate (its proof does not build; its shipped gate is stale on the result, the
  digests, the kernel files, the gate's code and the build's inputs) and the pages fail, as expected.
- `work/probes.py` (Part F: two Lean kernels through the tree's claim reader, the first with its
  `consistent` example; Part R on a scratch copy; Part C), `work/gitcfg.py` (network), and
  `work/globals.py` (Triton 3.8.0, NVIDIA H200 NVL, GPU 0); outputs in `work/*.out`.
- `tvr audit` on the tree, with my work files: `independent`.

## Not checked

- `lean/TVA/Math` and the numeric lemmas, which round 12 replaces, including L70's two docstrings
  there.
- Whether GitHub serves a commit of a fork of Mathlib by its hash from Mathlib's URL: protocol §2
  already assumes a host may; I showed only that the ref lookup goes where the repository's
  configuration says.
- vec_add_f32's gate, which is expected to fail (`tvr check` not run).
- A function made by `exec` or defined in another file: the reviewer's judgment by L50's design.
