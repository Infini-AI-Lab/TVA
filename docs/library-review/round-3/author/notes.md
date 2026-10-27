# Library batch, round 3: author's notes

Issues: L7, L12, L13, L14, L16, L17, L18 ([ISSUES.md](../../ISSUES.md)). 2026-10-27. L15 is not
in this batch, and nothing the library says about ordering within one program,
`tl.debug_barrier` or `num_stages` changed. The library builds (`lake build TVA`), and every new
or changed `TVA` constant depends only on `propext`, `Classical.choice` and `Quot.sound`.
`python3 -I tools/tvr.py audit` prints "independent".

**Kernel that no longer builds or passes the gate:** `kernels/vec_add_f32` (its prover's to fix;
I did not edit the kernel folder or run `tvr check` on it):
- its `launch_valid` destructures `Launch.Valid`, which now has two more conjuncts than the
  kernel's proof gives (`L.size < 2^31` from round 2, `L.pdl = false` from L17).
  `Launch.valid_grid1` proves it;
- its map entries name no transcription rules (L12), which the gate now asks of every `model`
  entry.

On a scratch copy of the tree, with those two changes made there, the kernel's gate passes every
check, `replay` included (L12, L13 below).

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed** in `tools/tvr.py` (`foreign_refs`, `_path_text`, `_home_ok`):
  - Before the path test, a text is normalised (NFKC, and `\/` read as `/`). A `file:` URL (with
    `//`, `///` or `//localhost`) is read as the path it names, and every other URL is removed.
    The home directory's spellings `$HOME`, `${HOME}`, `%HOME%`, `%USERPROFILE%` and `$env:HOME`
    are read as `~`.
  - An absolute path now starts after *any* character that cannot be part of a path name (a
    space, a quote, an opening bracket, `=`, `:`, `;`, `,`, `|`, …), not only after the old
    short list. So a path glued to `;` or inside `<…>` is found. A closing bracket, `>` or `}`
    before a `/` does not start a path, since they end a placeholder such as `<N>/` or
    `{kernel}/` inside a relative path.
  - `~<user>/…` is a home path, tested like `/home/<user>/…`.
  - A relative path is now any run of path characters between separators, `=` and `:`
    included, so `P=../../x` is found.
  - A folder that holds an allowed path (the home directory, `~/fpv`) is allowed too: it names
    no project.
  - [REVIEW_PROTOCOL.md](../../../REVIEW_PROTOCOL.md) §2 "Enforcement" lists the spellings.
- **Why.** Both library reviewers of round 2 found home paths that a shell or a browser
  resolves, written as `$HOME/…`, `${HOME}/…`, `file:///home/…` or `~<user>/…`, and a path glued
  to `;`, which passed the audit.
- **Evidence.** EVIDENCE_L7
- **Constants whose meaning changed.** None (the tool only).

## L12: No digest covers the transcription rules

- **What changed.**
  - **Identities.** [TRANSCRIPTION.md](../../../TRANSCRIPTION.md) is rewritten so that every rule
    is a list item that starts with its identity in bold brackets: 44 rules, such as `[mask]`,
    `[fusion]`, `[reduce]`, `[pure]`, and one per drop reason (`[no-op:print]`, `[config]`, …).
    A rule runs to the next rule or heading, and may name other rules as `[id]`. The text
    outside the rules (the introduction, §1, the last paragraph of §3, §9) and
    [exclusions.md](../../../exclusions.md) apply to every model. The page's meaning is unchanged
    except for the new rules of L14, L16, L17 and L18. I also moved the compiler citations out of
    the page, which kernel reviewers read; they are in the evidence pages and the Lean docstrings.
  - **Map entries name rules.** A `model` entry of `model-map.json` has `"rules": [...]`, at least
    one rule; a `drop` entry may name more besides its reason. The gate's `map` check fails on an
    entry without rules, on a name the page does not define, and on an inconsistent page (an
    identity defined twice, a name of no rule, a drop reason of the tool that the page does not
    define). `tvr audit` checks the page too.
  - **Digests.** The `map` and `model` digests fold in the general text and the text of every
    rule the map names, with the rules those name in turn, whitespace normalised. The gate's
    snapshot keeps a hash per rule (`rule:<id>`, `rule:(general)`), so `REVIEW.md` names the
    rules that changed since an acceptance. `claim.md` shows each entry's rules, and below the map
    the text of every rule it applies.
  - The drop list (`DROPS`) gains `no-op:gdc` (L17). Each drop reason is a rule of the page.
  - [REVIEW_PROTOCOL.md](../../../REVIEW_PROTOCOL.md) §4 (digests), §7 (the map), §3, §13;
    [REVIEWER_GUIDE.md](../../../REVIEWER_GUIDE.md) (check that an entry names every rule it
    applies), [PROVER_GUIDE.md](../../../PROVER_GUIDE.md),
    [LIBRARY_REVIEW.md](../../../LIBRARY_REVIEW.md), and the kernel prompts.
- **Why.** A kernel's acceptance of its `map` and `model` must go stale when a rule they rest on
  changes, as L6 and L10 changed rules last round. With identities, a change re-opens exactly the
  kernels whose maps name the rule, and a change to the general text re-opens every kernel.
- **Evidence.** EVIDENCE_L12
- **Constants whose meaning changed.** None in Lean. Every kernel's `map` and `model` digests
  change once (the new formula), so `vec_add_f32`'s `item:map` and `item:model` re-open.

## L13: A kernel's Lean can add `main` without the kernel checking it

- **What changed** in `tools/tvr.py` (`check`, `HYGIENE`, `hygiene_problems`, `replay`):
  - **New gate check `replay`.** After the build, `leanchecker <root>` (Lean 4.35.0's own tool,
    shipped in the toolchain) replays every declaration of every module whose name starts with the
    kernel's module root into the environment of the module's imports, and Lean's kernel checks
    each. It reads the compiled `.olean` files, so a theorem that meta code stored without the
    kernel's check fails, whatever the route.
  - **`hygiene` reads the whole file and runs first.** It no longer strips `--` comments line by
    line, which let anything after a `"--"` string hide. The list now covers proof escapes, kernel
    and compiler bypasses (`addDecl…`, `doCheck`, `skipKernelTC`, `modifyEnv`, `setEnv`,
    `Environment`, …), meta commands (`#eval`, `run_cmd`, `run_elab`, `run_meta`, `run_tac`,
    `by_elab`, `eval%`, `elab`, `macro`, `syntax`, `initialize`, `register_…`, `include_str`,
    `simproc`, `delab`, …), meta types (`IO`, `CoreM`, `MetaM`, `TermElabM`, `CommandElabM`,
    `TacticM`, `MacroM`, `Lean.Elab`, `Lean.Meta`, `Lean.Syntax`) and attributes that install
    handlers (`tactic`, `init`, `norm_num`, `positivity`, `aesop`, `export`). `hygiene` and
    `independence` run before the build, and the build runs only if both pass, since building
    runs the kernel's meta code.
  - [REVIEW_PROTOCOL.md](../../../REVIEW_PROTOCOL.md) §8 and §13 say so.
- **Why.** The library reviewer of round 2 built a bundle with `Post = False` whose `main` was
  added by `addDeclCore … (doCheck := false)` inside a `run_cmd`; it passed `build`, `hygiene`,
  `shape`, `axioms` and `usage`. Only a re-check of the compiled declarations closes this for
  every route; the token list stops meta code from running during the gate at all.
- **Evidence.** [evidence/gate-replay](../../../../evidence/gate-replay/README.md): `leanchecker`
  accepts a control module, and rejects the library reviewer's bundle ("declaration has
  metavariables"), a `run_cmd` + `addDeclCore` theorem `main : False := True.intro` and the same
  through a command defined with `elab` ("declaration type mismatch"). All three elaborate, and
  Lean then accepts `example : False := main`. The hygiene list flags all three and passes the
  control. It flags nothing in the library's Lean or in `vec_add_f32`'s. EVIDENCE_L13
- **Constants whose meaning changed.** None (the tool only).

## L14: "Pure" is unsound for aggregate operands

- **What changed.** EVIDENCE_L14_WHAT
- **Why.** EVIDENCE_L14_WHY
- **Evidence.** EVIDENCE_L14
- **Constants whose meaning changed.** None in Lean. The rule `[pure]` changed, and with it the
  rules that name it (`[no-op:print]`, `[compile-check]`, `[dead]`).

## L16: `tl.max`/`tl.min` widen small integers to `int32`

- **What changed.** EVIDENCE_L16_WHAT
- **Why.** EVIDENCE_L16_WHY
- **Evidence.** EVIDENCE_L16
- **Constants whose meaning changed.** EVIDENCE_L16_CONST

## L17: No `tl.permute` (rank ≥ 3), PDL or `tl.argmax`

- **What changed.** EVIDENCE_L17_WHAT
- **Why.** EVIDENCE_L17_WHY
- **Evidence.** EVIDENCE_L17
- **Constants whose meaning changed.** EVIDENCE_L17_CONST

## L18: `TRITON_F32_DEFAULT` changes `tl.dot`'s default precision

- **What changed.** EVIDENCE_L18_WHAT
- **Why.** EVIDENCE_L18_WHY
- **Evidence.** EVIDENCE_L18
- **Constants whose meaning changed.** EVIDENCE_L18_CONST
