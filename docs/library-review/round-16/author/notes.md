# Library batch, round 16: author's notes

Issues: L7, L62, L79, L81, L82 ([ISSUES.md](../../ISSUES.md)), 2026-10-28. The tool and gate batch from round
13's reviews, run beside rounds 12 (the float layer) and 15 (the supported subset) in a separate tree. The changes
are in `tools/tvr.py`, `tools/claim_reader.lean.in`, the protocol and role documents (`docs/REVIEW_PROTOCOL.md`,
`docs/LIBRARY_REVIEW.md`, `docs/REVIEWER_GUIDE.md`, `docs/PROVER_GUIDE.md`, `docs/DESIGN.md`, `docs/prompts/`), and
a new evidence folder, `evidence/tool-round16/`. No statement of the library changed: no Lean file of `lean/`
changed, and `lake build TVA` is green with the standard axioms only.

**Kernels.** `kernels/vec_add_f32` still fails to build against the current library (expected; its prover
re-proves it). Two more things for its prover, neither a fault of its proof: the gate's code changed, so its gate
is stale until `tvr check` runs; and its manifest now fails the `map` check (L79) on the code outside every
function that it leaves unclassed, lines 21, 23–24, 26 (the imports and `DEVICE`), 90–93 and 135 (the prints and
the benchmark's run). Each goes in the kernel lines, and the map, if the launch reads what it binds (the imports
of `triton` and of `triton.language as tl`, which the kernel lines name), or else in `source.host`. Its one field,
`faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)`, keeps to the new form (L62), and
its shipped ledger and role folders pass the new checks (L81). I did not edit the kernel folder or run
`tvr check`.

**Files of other rounds.** None. In `docs/REVIEWER_GUIDE.md` I changed only §1's ranges (a new paragraph after
"The ranges", L79) and §4's `[rule]` bullet (L62); in `docs/PROVER_GUIDE.md` only §1's "Class every function"
bullet (L79) and §3's `[rule]` bullet (L62), not round 15's passages. `docs/rules/num.md` (round 12's) is read,
not edited: the gate takes the citable facts from its section "Facts the library states" (L62).

**For the coordinator.** After merging round 12's `docs/rules/num.md`, run `tvr audit`: its fact-page check names
the page if the section's heading or its table no longer reads (without them the gate accepts no fact but
`Compiler.Faithful`, so every kernel citing a device fact would fail `kinds`). Round 13's notes asked for a commit
after each recording; the user's decision (L36) declines it, and the documents now say that the lines are trusted
as the coordinator keeps them (L82), so that request is withdrawn.

**Release check.** Run once on this tree, from TVA's root, with these notes regular (2026-10-28; output kept in
[evidence/tool-round16/release.out](../../../../evidence/tool-round16/release.out) and
[release.json](../../../../evidence/tool-round16/release.json)): 9 of 11 steps pass (shipped, packages, sandbox,
build, library, records, the lints of these notes and of vec-add's round 2, audit). The two failures are the
expected ones: vec_add_f32's gate (its proof does not build against this library, and its shipped gate is stale
on the result, the digests, the kernel files, the gate's code and the build's inputs), and the pages
(`kernels/README.md` and vec-add's `REVIEW.md`, which the recomputed, failing gate changes).

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed.** lr-13a's open route: git read the configuration of the repository around the folder it ran
  in, which for the release check and the audit is TVA's root. `_git` (`tools/tvr.py`) now runs every git in an
  empty folder of the tool's own (`cwd` inside `_git_home()`, the temporary folder that is already git's `HOME`
  and `XDG_CONFIG_HOME`), with `GIT_CEILING_DIRECTORIES` set:
  - a git given a repository (`-C <repo>`, always its first arguments: each new package repository, and TVA's own
    for `shipped_files`) gets the folder above `<repo>`, so it reads that repository and no other, not even one
    around a folder that is not a repository itself;
  - every other git (`ls-remote` in `upstream_problem`, the `clone` of a branch's history, the two `init`s,
    `--version`) gets the tool's folder, above `cwd`, so it finds no repository at all.

  Protocol §2 ("The programs the check runs") now says that git reads no repository's configuration but that of
  the repository it is given: each new package repository's own (made with no template) and, to list what ships,
  TVA's own. TVA's configuration can then decide only which of TVA's own files are copied into the sandbox (every
  listed file is copied from TVA's tree, and `tvr audit` reads every file of the tree whatever git lists). The
  release check's comments and `make_packages`' docstring say the same.
- **Why.** lr-13a `work/gitcfg.py`: with `url.https://example.invalid/.insteadOf = https://github.com/` in the
  `.git/config` of the repository the tool ran from, the ref lookup for Mathlib's tag went to that host, so that
  file decided which commit counted as Mathlib's. The ceiling is the folder above the one git runs in, not that
  folder itself: git stops only at a ceiling that is a proper ancestor of its folder (checked with git 2.52: with
  the ceiling equal to its folder, `git ls-remote` still read the repository around it; with the folder above, it
  reached GitHub).
- **Evidence.** [evidence/tool-round16](../../../../evidence/tool-round16/README.md) Part G: a plain git inside
  lr-13a's repository reaches `example.invalid`; the tool's `_git ls-remote` and `upstream_problem` reach
  Mathlib's repository and accept the pin as the tag `v4.35.0` from outside the repository, inside it and in a
  subfolder of it, and as a commit of `master` (a clone of its history) from inside it; `_git -C` a subfolder that
  is not a repository finds none; `_git -C` the repository reads its `insteadOf`; TVA's repository is found for
  `shipped_files`. The release check of this round (below) ran the packages step from TVA's root.
- **Constants whose meaning changed.** None (the tool and the protocol).

## L62: soundness: a `[rule]` field can state a device fact with the kernel's own constants

- **What changed.** Both open parts:
  - **`Compiler.Faithful`'s arguments** (lr-13a F1, F2). The claim reader now gives, for each argument of a
    `[rule]` statement's head (under its `∀` and `→` binders), the head constant of its type (`type`), the
    function it applies to one variable the statement binds (`fn`: a constant, or `K`) with that variable's
    place among the statement's binders (`var`, else -1), and the argument as Lean prints it (`pp`).
    `faithful_problem` accepts `Faithful` only as `Faithful P (model x) (launch x) (K x)`: the kernel's own
    `model` and `launch`, and `K`, each applied to the same bound variable. Any hypotheses may stand before it
    (they only weaken the field). Anything else is refused, with the form found, also a form the reader cannot
    match (`K d` written as a `fun`: it fails closed).
  - **The list of facts** (lr-13a's new gap). `stated_facts` reads the citable facts from
    `docs/rules/num.md`'s section "Facts the library states" (a `##` heading starting with those words, up to the
    next `#` or `##` heading): every code span of its table's first column, a name under `TVA.Rules.Num` (or a
    full name), alone or applied. `rule_fact_problem` accepts a head other than `Faithful` only if it is listed:
    a name alone with any arguments after `P` (so a `Platform → Prop` fact with none, a `…Total` rule with any),
    a name listed applied with exactly those arguments (`DivRn P .binary64`: `P` the platform, `.x` the constant
    `x` of the argument's type, as Lean's dot notation reads it). The type `Platform → Prop`, the suffix `Total`
    and the last component `binary64` no longer decide anything. Without the page, the section or its table no
    fact passes but `Faithful` (it fails closed), and `tvr audit`'s fact-page check names the page.
  - **Documents.** Protocol §2 (fact pages), §5 (the two forms, and that a constant becomes citable only once
    listed), §8 (`kinds`); REVIEWER_GUIDE §4; PROVER_GUIDE §3; the kernel prover's prompt; the library author's
    prompt (a fact is citable by every kernel once listed there; what to list alone and what applied; never a
    shape; keep the heading and the table); the library reviewer's prompt and LIBRARY_REVIEW.md (the promise
    table, and a new check 5, "What kernels may cite").

  On the coordinator's question in ISSUES.md, the check now depends on these names: `Faithful`'s full name and
  the kernel's `model` and `launch` (a rename refuses the field: closed); the page `docs/rules/num.md` and the
  heading's words "Facts the library states" (a rename refuses every fact, and the audit names it: closed); and
  the names the table writes, which round 12 keeps current. A format's renamed type keeps `.binary64` matching;
  a renamed `binary64` matches only once the table says so (closed until then).
- **Why.** lr-13a F1: a whole claim with `faithful : ∀ d, d.ok → Faithful P (model false) (launch d) (K d)`
  passed every check, so `main` was proved from the idle model while the launched instance's model never ends;
  F2: `model e`, `launch e` and `model (Classical.choose h)` passed too. lr-13a's new gap: any `TVA.Rules`
  constant of type `Platform → Prop`, and any `TVA.Rules.Num` name ending in `Total`, passed as a stated fact,
  so a helper such as `def DivRnF32 := DivRn P .binary32` (false on the H200) would have been citable.
- **Evidence.** [evidence/tool-round16](../../../../evidence/tool-round16/README.md) Part F: F1 still passes
  shape, axioms, usage and namespace, and `kinds` now refuses it; of ten `Faithful` fields, the standard form, the
  form without `d.ok` and the form on another bound variable pass, and `model e`, `launch e`, `K e`, the chosen
  instance, `model ⟨0⟩`, a launch written out and `K d` as a `fun` are refused. Part L: num.md lists 32 facts;
  `DivFull P`, `Exp2F32 P` under binders, `DotF16 P`, `MathTotal P .exp .binary32` and `DivRn P .binary64` pass;
  `DivRn P .binary32`, lr-9a's `MathAcc`, `DotF8` and lr-13a's two helpers, compiled as a scratch library module
  `TVA.Rules.Num.Extra`, are refused; a changed parenthetical or a full name in the table reads the same; a
  renamed section, a removed table or a removed page refuses every fact, and the audit names num.md.
- **Constants whose meaning changed.** None (the gate and the protocol). Which constants are citable is now what
  num.md's list says, which today is the set the old rules admitted (the 25 `Platform → Prop` facts are num.md's,
  as lr-13a found, and the six `…Total` rules and `DivRn P .binary64` are listed), so no kernel item changes.

## L79: gap: module-level values a JIT function reads are in no class of the map

- **What changed.** `module_lines` (`tools/tvr.py`) gives the code lines that no function spans (`code_lines`
  minus every `def`, `async def` and `lambda` from its first decorator to its last line: imports, module-level
  assignments, a class's own lines, a `__main__` block). The `map` check fails on one that lies neither in the
  kernel lines nor in the host lines, naming the lines and what to do. claim.md §2 lists the classed lines (in the
  kernel lines, in the host lines) and asks the reviewer to judge that every module-level value the launch reads
  lies in the kernel lines, mapped there, and that the host lines bind nothing it reads. Protocol §7 (a new
  bullet, "Code outside every function") and §8 (`map`), REVIEWER_GUIDE §1 (a paragraph after "The ranges",
  naming the import that binds `tl`: another module imported as `tl` would give `tl.load` another meaning),
  PROVER_GUIDE §1, and both kernel prompts say so. What such a name means in a kernel is SEMANTICS.md's (L80,
  round 15); the documents point there.
- **Why.** lr-13a `work/globals.py`: a kernel reading `SCALE = tl.constexpr(3)` and `OUT_T = tl.float16` from
  module-level lines outside its kernel lines passed `map_problems`, and on the H200 stores `x·3` rounded to
  `float16`; the front end resolves such names at compile time (`triton-3.8.0-code-generator.py:383-420`), and
  the value is part of the compiled kernel (`triton-3.8.0-jit.py:532-536`, checked at each launch, `:761-766`).
  By the user's rule (L36) the cheap complete check is that every such line is classed, and the reviewer judges
  the classes, as for functions (L50).
- **Evidence.** [evidence/tool-round16](../../../../evidence/tool-round16/README.md) Part M: lr-13a's manifest now
  fails on lines 1–3, 5–6 (the imports, `SCALE`, `OUT_T`) and 31–32 (an added `__main__` block); a class's own
  lines are code outside every function and its docstring is not; each full classing passes (the module's lines
  in the host lines pass too: which class is right is the reviewer's judgment); vec-add's shipped manifest fails
  on lines 21, 23–24, 26, 90–93 and 135; claim.md §2 lists the classes.
- **Constants whose meaning changed.** None (the gate and the protocol).

## L81: gap: a kernel ledger shows nothing of a reviewer whose lines are removed while their folder stays

- **What changed.** `kernel_folder_problems` (`tools/tvr.py`) holds a kernel's `review/` folder to its ledger,
  the converse of L52's check that every event follows from a recorded file. It names:
  - a `round-N/<who>/` folder of a reviewer no `assign` event names (`prover` is the prover's);
  - a folder of a round other than the current one that no event records (one of its files, or the folder as
    `dir`): an earlier round's filing or response whose events are gone, or a later round's, which no event can
    record;
  - an entry of `review/` that is no role folder (anything but the ledger, the certificate, the snapshots and
    `round-<N>/` folders of role folders), so that nothing filed can sit beside them.

  `kernel_record_problems` (the certificate's ledger condition, and REVIEW.md's "Recorded files") and the release
  check's `records` step include it. The current round's folders that no event records yet are filings and
  responses in progress: the release check lints them, as before, and the certificate now also waits for them
  (`unrecorded_current`, with its own line in the status), so removing a reviewer's `file` and `raise` lines while
  keeping their `assign` cannot leave the certificate granted beside their filing. `tvr lint <kernel> reviewer
  <id>` and `tvr start <kernel> reviewer <id>` refuse a reviewer not assigned, as `tvr file` does. Protocol §3 (a
  new bullet, "Every role folder follows from the events"), §2 (`records`) and §9 (the certificate) say so.
- **Why.** lr-13a Part R: with r-two's `assign`, `file` and `raise` lines removed, the records step named nothing,
  `tvr lint` printed `regular` and the certificate was granted, with r-two's blocking filing still in the tree.
- **Evidence.** [evidence/tool-round16](../../../../evidence/tool-round16/README.md) Part R: vec-add as shipped has
  no folder unaccounted for; `tvr start` and `tvr lint` refuse r-two before it is assigned; with lr-13a's three
  lines removed the certificate fails, the records step names `review/round-2/r-two`, and `tvr lint` refuses
  r-two; with only the `file` and `raise` lines removed, the records step passes and lints the folder, and the
  certificate waits for it; an earlier round's unrecorded folder, `review/round-02/`, `review/notes.md` and
  `review/round-2/stray.json` are each named.
- **Constants whose meaning changed.** None (the tool and the protocol).

## L82: wording of what the tool can promise

- **What changed.** The four points:
  - **Ledger lines.** LIBRARY_REVIEW.md ("Rounds", step 4), protocol §3 ("What a ledger shows"), §10
    ("Recording"), §2 (`records`) and `ledger_digests`' docstring now say that the lines are trusted as the
    coordinator keeps them, with no commit at each recording (the user's decision, L36), that an edited or
    removed line shows only against a copy kept elsewhere (a commit, or a kept `release.json`'s `ledgers`),
    compared by hand, and that no step reads an earlier `release.json` back. DESIGN.md has a decision row for it.
  - **Old events.** Protocol §3 says that the tool derives a filing's events again with its current defaults (a
    kind's default `blocking`, `ESCALATE_AFTER`, the escalation memo), so changing one makes every older ledger
    with an event it decided fail the `records` step and the certificate; the comment at `ESCALATE_AFTER` and
    `filing_events`' docstring say the same. I stated it rather than derive those values from the recorded
    events: a derivation that took `blocking` from the recorded `raise` line would accept an edited line, which
    is what the derivation is there to catch.
  - **What a gate in the tree rests on.** Protocol §8 has a paragraph saying that `gate.json`, its snapshot and
    claim.md are as the last `tvr check` wrote them in its caller's environment (`TVR_LAKE`, elan's toolchain,
    the local `lean/.lake/packages`) or as anyone edited them since, that staleness compares the tree with what
    `gate.json` says it read (so a hand-made `gate.json` is current), and that only the release check recomputes
    them, in its clean sandbox; §9 says the same of the certificate.
  - **Capitals.** `citation_problems` no longer reads a path component in capitals only as a placeholder: a
    placeholder is `round-<N>/`, `{…}`, `*`, `…`, `...`, or a lower-case word with an upper-case one after its
    last hyphen (`round-N`). Protocol §2 ("Citations") says so. The tree cites no such name that is missing
    (checked: the only placeholder components in the tree's cited paths are `round-N`).
- **Why.** lr-13a and lr-13b: the documents said ledger lines "rest on the git history", which TVA cannot promise
  under the user's decision; old events were re-derived with today's defaults without saying so; nothing said
  that a gate in the tree is as it was written; `docs/DESIGN` passed as a placeholder.
- **Evidence.** [evidence/tool-round16](../../../../evidence/tool-round16/README.md) Part C: `docs/DESIGN` is named
  as missing, while `docs/DESIGN.md`, `round-N`, `<topic>`, `*.md` and `...` pass, and the tree cites no missing
  file; vec-add's shipped ledger passes, and fails at its round-1 `DOC` finding once `DOC` is blocking by default.
- **Constants whose meaning changed.** None (the tool and the documents).
