# Library batch, round 6: author's notes

Issues: L7, L20, L22, L25, L31, L32, L35 (fixed in round 5, carried) and L36
([ISSUES.md](../../ISSUES.md)). 2026-10-28. L36 is the batch: it reshapes how the library is checked
(user, 2026-10-28: "assume the provers and reviewers have intelligence and properly leverage
that"), and with it L7, L20, L25 and L32. Each carried issue's section says what L36 changed for
it, or that it is unchanged since round 5 ([round-5 notes](../../round-5/author/notes.md)).

The library builds (`lake build TVA`), and no Lean definition, theorem statement or proof changed:
the Lean changes are three docstrings (under L35 and L36). `python3 -I tools/tvr.py audit` prints
"independent". The release check (`tvr release`, new under L36) passes every step on what TVA ships
except the gate of `kernels/vec_add_f32`, which fails at its build, as it does in the tree
(below).

**Kernel that does not build or pass the gate:** `kernels/vec_add_f32`, as before this round (its
prover's to fix; I did not edit the kernel folder or run `tvr check` on it). It now needs two of the
three updates round 3 asked for (`Launch.valid_grid1`, disjoint read and write footprints with
`hRW`): its map needs no `rules` any more. With those two made on a scratch copy, the round-6 gate
passes every check ([evidence/gate-digests](../../../../evidence/gate-digests/README.md), case A).
The digests are computed differently (whole library files), so every one of its items re-opens
once.

**Records of other roles.** I edited none, and no check flags text in one.

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed** (with L36):
  - **The release check** (`tvr release [--dir DIR]`, `shipped_files`, `release_records`,
    `cmd_release` in `tools/tvr.py`) makes the guarantee complete by construction. It copies exactly
    the files TVA ships (`git ls-files --cached --others --exclude-standard`: tracked files, and
    untracked files git does not ignore) into an empty sandbox, and there runs `lake build TVA`,
    `tvr check` on every kernel, `tvr lint` on the records of each process's current round, and
    `tvr audit`. With `bwrap` (present here), the sandbox hides `/home`, `/tmp` and the network; its
    home directory, mounted at the real home's path, holds only the copy, the Lean toolchain
    (`~/.elan`), the Python the venv is built on, the venv (`.venv`), an empty `~/.triton`, and
    Mathlib's built packages (`lean/.lake/packages`), read-only except the copy. What it borrows is
    checked too: each package is a clean git checkout at the revision `lean/lake-manifest.json`
    pins (`package_problems`), and the tool runs with `python3 -I -S` (the standard library only).
    So a dependency on anything outside TVA fails whatever its spelling, also one computed at run
    time, which the text scan cannot see. Records of earlier rounds are frozen (their format was
    checked when recorded; today's lint would apply today's issue list to them), and `tvr audit`
    reads their text.
  - **Names:** unchanged. The tool hashes the only other project's name; as the coordinator
    instructed, I added none and did not look in `~/fpv` outside TVA.
  - **Paths:** kept as round 5 built them (the quote and brace views, the escapes, the home
    spellings, every test on kernel sources and the `fpv/<name>` and home tests on pinned files),
    and not extended. Protocol §2 says that a path spelling that names no project, and that the
    release check shows is no dependency, is a wording matter, not a soundness reopen.
  - **Protocol §2** ("Enforcement") puts the release check first and says what it holds and hides;
    `docs/prompts/library-author.md` and `library-reviewer.md` ask for it; LIBRARY_REVIEW.md's `gate`
    survey area includes it.
- **Why.** L36 and the user (2026-10-28): "we need to keep the project independent. since tva will
  be released without others." Four rounds reopened a pattern-by-pattern scanner; a release check
  does not depend on spelling.
- **Evidence.** [evidence/release-check](../../../../evidence/release-check/README.md): R0, the tree as
  it is, passes the build, the lint of the current records and the audit in the sandbox, and fails
  only `vec_add_f32`'s gate (its build, as in the tree); R1, a copy whose library imports a module in
  a folder git ignores, builds and passes the audit outside but fails the build in the sandbox; R2, a
  copy whose `tvr audit` reads a file outside TVA at a path computed at run time, passes the audit
  outside and fails it in the sandbox. The text scan's code did not change this round, so
  [evidence/independence-audit](../../../../evidence/independence-audit/README.md) (96 probes, 0
  wrong) stands; I did not rerun it, since its script takes the other project's name on the command
  line.
- **Constants whose meaning changed.** None (the tool and the role documents).

## L20: Model text can mean another operation (a kernel's own `tl.add`, `local notation`)

- **What changed** (with L36): no check was added or removed. The user's "keep both" stands: the
  kernel reviewer reads the kernel's whole Lean, and the gate's name-resolution check is a backstop
  for the common route. Of round 5's additions I kept all, each for a reason L36 names:
  - generated names by kind (`def Launch.mk` is checked), `open` read over every line of the
    command, and root names and aliases (`min`): these make the name-resolution backstop correct
    for the route it covers, a kernel declaration a model line's name could resolve to;
  - no instance at all, and plain ASCII declaration names: complete and cheap (every instance and
    every non-ASCII declaration name is flagged; no kernel needs either);
  - binders named like a library namespace, after NFKC and the fold of Cyrillic and Greek
    look-alikes: a look-alike is what a reader cannot see, and for the exact names the check is
    complete. I did not extend the fold;
  - `set_option pp.notation` allowed (a printing option).

  The reviewer's documents now say why names matter instead of listing what to tick:
  REVIEWER_GUIDE.md §1 ("Names: read the kernel's whole Lean"), the kernel-reviewer prompt, and
  TRANSCRIPTION.md §1, which explains why the kernel's Lean declares no such names. The other routes
  are the reviewer's, as L36 says.
- **Why.** L36: keep mechanical only what is complete, cheap, or invisible to a reader, and the
  name-resolution check as a backstop (user: "keep both").
- **Evidence.** [evidence/gate-digests](../../../../evidence/gate-digests/README.md), the round-5 cases
  L–X rerun with the round-6 tool: each route still fails `namespace` or `hygiene` with the same
  reason, and the control X passes.
- **Constants whose meaning changed.** None.

## L22: Descriptor and TMA text written before L15

- **What changed.** Unchanged since round 5 ([round-5 notes](../../round-5/author/notes.md), L22).
  L36 moved one sentence of the withdrawn `[desc]` into SEMANTICS.md "Tensor descriptors": a host
  descriptor has the host's shape, strides, block shape and padding, and an in-kernel descriptor is
  open (L5). The L22 text itself (descriptor accesses unordered with the program's other accesses,
  the pointer/descriptor exclusion) is as round 5 left it.
- **Why.** L36 moves every Triton fact of TRANSCRIPTION.md into SEMANTICS.md.
- **Constants whose meaning changed.** None in Lean; the fact page SEMANTICS.md.

## L25: `[config]` drops branch conditions that may run operations

- **What changed** (with L36). The facts round 5 put in `[pure]` ("What the front end
  evaluates"), `[config]` (`elif`) and `[compile-check]` are now facts of SEMANTICS.md, "How the
  front end evaluates the source": every argument is evaluated before the call; the order, with
  `and`, `or` and a conditional expression stopping only at a compile-time value; what a compile-time
  value is (`is`/`is not` included); `if`/`elif`/`else` visiting only the branch taken, and an
  `elif` only when every earlier condition is false; where a `tl.static_assert` is evaluated and
  that its `msg` is evaluated only on failure; what the debugging calls emit; and which expressions
  have no effect. The drops (`config`, `compile-check`, `no-op:print`) are guidance in
  TRANSCRIPTION.md §4, with round 5's worked examples (an atomic in a `device_assert`,
  `FLAG and <atomic>`, `<atomic> is None and <atomic> is None`, an `elif` after a false `FLAG`). The
  substance is round 5's; only its place and status changed.
- **Why.** L36 names condition evaluation and short-circuiting among the facts to move.
- **Evidence.** Unchanged: [evidence/config-cond](../../../../evidence/config-cond/README.md) (C1–C4,
  P1–P3, S1–S16) and [evidence/debug-calls](../../../../evidence/debug-calls/README.md); their READMEs
  now point to the SEMANTICS.md section.
- **Constants whose meaning changed.** None in Lean; the fact page SEMANTICS.md.

## L31: Stale hint text, `[no-op:debug_barrier]`'s reason, `[dot-add]`'s "constant zero"

- **What changed.** Unchanged since round 5 ([round-5 notes](../../round-5/author/notes.md), L31), but
  for where two facts now live (L36): SEMANTICS.md "Compiler hints" also states that Triton applies a
  hint to the value everywhere (from `[hint]`), and "The combine pass's rewrites" states `dot_add`'s
  accumulator as round 5 did (`±0` after constants are folded, with the examples). The reason for
  dropping `tl.debug_barrier` is SEMANTICS.md "No barrier semantics", as before.
- **Why.** L36.
- **Constants whose meaning changed.** None in Lean; the fact page SEMANTICS.md.

## L32: Text kernels rely on is outside the digests

- **What changed** (with L36): subsumed. A kernel item's digest now holds the whole text of every
  library file its cone reaches (L36 below), so every module docstring, field docstring, comment
  and rule page of those files is in it by construction. The claim reader no longer hashes module
  docstrings separately, and `library_texts` is gone; `RULE_PAGES` stays (the rule page of each
  `TVA.Rules` module), with `tvr audit` checking it (`fact_page_problems`).
- **Why.** L36: digests complete by construction, not a curated list of docstrings.
- **Evidence.** [evidence/gate-digests](../../../../evidence/gate-digests/README.md): Y (module
  docstring of `TVA.Claim`), Z (docstring of the unprojected field `TVA.Launch.blocks`), AA
  (`rules/compiler-faithful.md`) re-open the items that reach them; AB (`rules/num.md`, which the
  kernel does not reach) re-opens nothing.
- **Constants whose meaning changed.** None.

## L35: Seven wording items

- **What changed.** Unchanged since round 5 ([round-5 notes](../../round-5/author/notes.md), L35),
  but for where the constants' facts live (L36): `[literal]`'s cases, the integer wrap included, are
  SEMANTICS.md "Constants"; the module docstring of `TVA.Lang.Scalar` and the docstring of
  `TVA.tl.ItofRel` point to SEMANTICS.md instead of the withdrawn rules; `TVA.tl.sum_bcast_mul`'s
  docstring says the library gives an `a + a` combine no meaning yet ("no rule covers it" before).
- **Why.** L36 withdraws the rule identities these texts cited.
- **Constants whose meaning changed** (docstrings only): the module docstring of `TVA.Lang.Scalar`,
  `TVA.tl.ItofRel`, `TVA.tl.sum_bcast_mul`.

## L36: Leverage judgment: mechanical checks only where complete; transcription as guidance; advisory scans

- **What changed.**
  - **Digests are complete by construction** (`library_files`, `packages_digest`, `check` in
    `tools/tvr.py`; `tools/claim_reader.lean.in`; protocol §4). An item's digest holds its cone's
    constants by elaborated term and the roots' docstrings, as before, and now the whole text
    (sha256) of: the Lean file of every library module that holds a constant of the cone; the fact
    pages SEMANTICS.md and exclusions.md when the cone reaches the library; the rule page of each
    `TVA.Rules` module it reaches; and the packages' versions (`lean/lake-manifest.json`), with
    Lean's. The `map` digest is the map, the source, and the `model` item's library files.
    `REVIEW.md` names a changed file as `file:<path>`, and claim.md lists each item's files.
    TRANSCRIPTION.md is in no digest. This subsumes L12, L27 and L32: the rule-text digests, the
    "general text" and the module-docstring hashing are gone.
  - **Transcription is guidance.** TRANSCRIPTION.md is rewritten: why the model is judged and not
    matched (the user's words), the model's shape, worked patterns (each with the SEMANTICS.md
    section behind it), pitfalls, dropping a line (the common reasons as guidance, round 5's worked
    examples), the map. Every Triton fact it stated is now in SEMANTICS.md: the literals per dtype
    (L23) in "Constants"; `rtz` narrowing (L24) in `x.to(float)`; condition evaluation and
    short-circuiting (L25), the debugging calls, `tl.static_assert` (L6, L10), the aggregate
    operators (L14) and the expressions with no effect in the new section "How the front end
    evaluates the source"; the dot rewrite and the broadcast-multiply-sum spellings (L26) in "The
    combine pass's rewrites"; the rest where it belongs (`[call]`, `[implicit]`, `[index]`,
    `[pointer]`, `[float-prog]` in "Values"; `[fusion]` in "Float `+` and `-`: IEEE or fusable";
    `[dot]` and `[dot-scaled]` in their bullets; `[permute]` and `[reshape-reorder]` in "Shape
    changes"; `[block-ptr]`, `[desc]`, `[scan]`, `[sort]`, `[hint]` in theirs; `[range]`,
    `[static-range]`, `[while]` and the `tl.range` keywords in "Loops"; `[bool-ops]` in "Order";
    §9's `blocks` and autotuning in "Launches").
  - **The map check** checks coverage, model text found in the kernel's Lean, and regular entries
    (`src`; exactly one of `model` and `drop`; a non-empty reason). `rules` is optional and
    unchecked; a drop reason is free text, judged by the reviewer. `DROPS`, `NOTE_REQUIRED`, the
    rule parser (`transcription_rules`, `rule_closure`, `rule_parts`) and the audit of the page's
    rule identities are removed. Protocol §7, §8 and §9 (`MODEL_MISMATCH`: the model does not mean
    what the source does; `UNSUPPORTED`: a construct the library gives no meaning, or an open one)
    say so.
  - **What stays mechanical**, because complete, cheap, or invisible to a reader: the build,
    `replay` and the axioms; the claim's shape, usage and assumption kinds; the imports, the lake
    configuration, the release check and the text scan (L7); regular output formats; digests; the
    `hygiene` list and the `namespace` backstop (L20). Of round 5's gate and scanner additions I
    kept all (L7 and L20 above say which and why) and added no denylist entry.
  - **Prompts and guides give rationale and stakes**: REVIEWER_GUIDE.md is rewritten around "how
    could this claim be wrong?"; the kernel-reviewer prompt has "Why your review matters"; the
    library-reviewer and library-author prompts state the stakes and ask the same question; the
    prover's prompt and guide treat TRANSCRIPTION.md as guidance and ask for notes that explain
    each entry. The protocol (§1 trust boundary, §2 scopes, §13), LIBRARY_REVIEW.md (what the
    library promises, the checks, the `transcription` and `gate` survey areas, the classes),
    DESIGN.md (principle 5 and three decisions), README.md, PLAN.md, exclusions.md and LIBRARY.md
    follow.
- **Why.** The user (2026-10-28): "our transcription is not based on a fixed rule, we provide the
  guidance and the reviewer needs to check whether the model matches the kernel"; "basically we need
  to assume the provers and reviewers have intelligence and properly leverage that". Every reopen of
  L7, L12 and L20 so far was a list that missed an entry; whole-file digests and a release check do
  not depend on a list, and correspondence is better judged by a reader than matched by name.
- **Evidence.** [evidence/gate-digests](../../../../evidence/gate-digests/README.md), rerun with the
  round-6 tool on a scratch copy: a change to TRANSCRIPTION.md re-opens nothing (B, C); a change to a
  fact page re-opens every item (D, J); a docstring, a whitespace change or a comment in a library
  module the model reaches re-opens the items that reach it (K, K2, K3), a comment in a module no
  item reaches re-opens nothing (K4), and a comment in the kernel's proof re-opens nothing (K5); a
  map with no `rules`, unknown `rules` or a free-text drop reason passes (A, E, E2), and an entry
  with neither model text nor a drop, an uncovered line, or an entry without `src` fails (E3–E5);
  the hygiene, replay and namespace cases fail as before (F, G, L–W). For the release check,
  [evidence/release-check](../../../../evidence/release-check/README.md) (under L7).
- **Constants whose meaning changed.** No Lean definition, statement or proof. Docstrings: the
  module docstring of `TVA.Lang.Scalar`, `TVA.tl.ItofRel`, `TVA.tl.sum_bcast_mul` (L35). The fact
  pages SEMANTICS.md (the moved facts, no fact withdrawn) and exclusions.md (references). Since the
  digests now hash whole files, every kernel item that reaches the library re-opens once; after
  that, exactly the items that reach a changed file. A change to TRANSCRIPTION.md re-opens nothing.
