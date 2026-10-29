# Library batch, round 3: author's notes

Issues: L7, L12, L13, L14, L15, L16, L17, L18 ([ISSUES.md](../../ISSUES.md)). 2026-10-27. L15
joined the batch after the user's decision (option 1, no conflicting accesses within a program).
The library builds (`lake build TVA`), and every new or changed `TVA` constant depends only on
`propext`, `Classical.choice` and `Quot.sound`. `python3 -I tools/tvr.py audit` prints
"independent".

**Kernel that no longer builds or passes the gate:** `kernels/vec_add_f32` (its prover's to fix;
I did not edit the kernel folder or run `tvr check` on it):
- its `launch_valid` destructures `Launch.Valid`, which now has two more conjuncts than the
  kernel's proof gives (`L.size < 2^31` from round 2, `L.pdl = false` from L17).
  `Launch.valid_grid1` proves it;
- its map entries name no transcription rules (L12), which the gate now asks of every `model`
  entry;
- `meets_independent` now asks that each program's read and write footprints be disjoint
  (`hRW`, L15). The kernel's `Inst.ok` allows the output to *be* `x` or `y` (in place), and then a
  program reads the bytes it writes: a conflict within the program, which L15 puts out of scope.
  The claim has to drop the in-place case (`output_x`, `output_y` disjoint only). The rest of its
  proof is unchanged (`safe_store_block` now hands the rest of the program `W` without the tile,
  and the rest is `safe_ret`).

On a scratch copy of the tree, with those three changes made there, the kernel's gate passes every
check, `replay` included ([evidence/gate-digests](../../../../evidence/gate-digests/README.md)).

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
- **Evidence.** [evidence/independence-audit](../../../../evidence/independence-audit/README.md)
  adds probes P10–P19 for these spellings (`$HOME/…`, `${HOME}/…`, `file:///…` bare and in angle
  brackets, `~<user>/…`, a path glued to `;`, `%USERPROFILE%/…`, a relative path after `=` that
  climbs out of TVA, JSON-escaped slashes in an evidence file) with a control: 63 probes, 0 wrong
  ([out.txt](../../../../evidence/independence-audit/out.txt)). The round-2 reviewers' own probes
  (`round-2/lr-2a/work/l7_path_forms.py`, `round-2/lr-2b/work/l7_more_probes.py`), rerun against
  the new tool, flag every case they flagged by hand. `tvr audit` on the tree prints
  "independent".
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
- **Evidence.** [evidence/gate-digests](../../../../evidence/gate-digests/README.md) runs `tvr check`
  on a copy of the tree, with `vec_add_f32` updated there: changing `[mask]`, which its map names,
  or a sentence of `exclusions.md` (general text) changes the `map` and `model` digests;
  changing `[sort]`, which it does not name, changes no digest; an entry naming `[no-such-rule]`
  fails the `map` check. `tvr audit` checks the page's consistency on the real tree
  (44 rules, no duplicate, every reference and drop reason defined).
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
  control. It flags nothing in the library's Lean or in `vec_add_f32`'s. End to end,
  [evidence/gate-digests](../../../../evidence/gate-digests/README.md): the updated `vec_add_f32` passes
  `replay`; with a `run_cmd` + `addDeclCore` theorem added, `hygiene` fails and nothing is
  built; with the copy's hygiene list switched off, the file builds and `replay` rejects the
  declaration ("while replaying declaration 'Kernels.VecAdd.extraFalse'").
- **Constants whose meaning changed.** None (the tool only).

## L14: "Pure" is unsound for aggregate operands

- **What changed.**
  - `[pure]` in [TRANSCRIPTION.md](../../../TRANSCRIPTION.md) restricts the operand types: every
    operand of the expression's operators and comparisons, and every value it subscripts or reads
    an attribute of, is a tensor, a `tl.constexpr` or plain Python value, a dtype, or a tuple of
    these. A value of an aggregate class (made with `_aggregate`), or a tuple holding one, makes
    the expression not pure. The rules that rely on purity (`[no-op:print]`, `[compile-check]`,
    `[dead]`) name `[pure]`.
  - Aggregate classes are out of scope ([exclusions.md](../../../exclusions.md)), and the page's
    general text says no model uses one.
  - [PROVER_GUIDE.md](../../../PROVER_GUIDE.md), [REVIEWER_GUIDE.md](../../../REVIEWER_GUIDE.md)
    and the kernel prompts say so.
- **Why.** An operator, comparison, subscript or attribute on an aggregate calls the class's
  `@triton.jit` method, which Triton inlines with its effects
  (`primitives/triton-3.8.0-code-generator.py:795-808` binary operators, `:1076-1091`
  comparisons, `:1097-1111` unary operators, `:1187-1193` subscripts, `:1535-1545` attributes;
  `_aggregate`, `primitives/triton-3.8.0-core.py:1625`). So a dropped line could run an atomic, as
  lr-2a's `work/l6_operator_dispatch.py` showed.
- **Evidence.** [evidence/pure-operands](../../../../evidence/pure-operands/README.md): with a
  method that adds to a counter, `tl.static_assert((a + a) is not None)`, `tl.device_assert(a == a)`,
  `assert a < a`, an unused `a + a` and `a[0]` each run the method in all 4 programs; `-a` is
  rejected on the host; a tuple's own `==` and `+` run nothing. On the allowed types, every
  operator the rule lists, in every dropped position, leaves the PTX identical to the kernel
  without those lines.
- **Constants whose meaning changed.** None in Lean. The rule `[pure]` changed, and with it the
  digests of every map that names it or a rule naming it (`[no-op:print]`, `[compile-check]`,
  `[dead]`).

## L15: One program is not sequential for global memory

- **What changed.**
  - **The rule** (`TVA/Sem/Launch.lean`). `IntraRace tr`: two events of one program, the first
    before the second, that conflict (one writes bytes the other reads or writes), unless both are
    atomics on the same bytes (`Event.SameAtomic`: both atomic, the same bytes read and written).
    `WellBehaved` now also asks `¬ IntraRace tr` of every reachable trace, so program order
    excuses nothing and a conflict across loop iterations counts. A plain load and an atomic on
    the same bytes conflict.
  - **Every order of a program's atomics** (`TVA/Sem/Step.lean`). A new step `Step.defer`: an
    `rmw` may stay pending while its continuation takes a step that is the same for every old
    value (the same memory, the same event); it is performed later. So the semantics performs a
    program's atomics in every order their data and control dependences allow, and the old values
    they return, and what `xchg` and `cas` leave, follow that order. `Event` gains `cell` (an
    atomic's dtype and address).
  - **Docstrings and pages.** `TVA.Lang.Prog` (module and the `rmw` row), `TVA.Sem.Launch` and
    `TVA.Sem.Step` (modules), `Rules.Compiler.Faithful` (docstring; its definition is unchanged,
    its `WellBehaved` is the new one), [rules/compiler-faithful.md](../../../rules/compiler-faithful.md),
    [SEMANTICS.md](../../../SEMANTICS.md) ("Programs", "Atomics", "Launches", "No barrier
    semantics", "Granularity"), [DESIGN.md](../../../DESIGN.md), [exclusions.md](../../../exclusions.md)
    (kernels that write and read back their own data or update an input in place; atomics of one
    program on partly overlapping bytes), TRANSCRIPTION `[no-op:debug_barrier]` and
    `[no-op:range-kw]` (why a barrier and `num_stages` still mean nothing),
    [review-questions.md](../../../review-questions.md) S1 (superseded, pointing here), Q7 and Q11
    (two writes or a write and a read through a pointer and a descriptor are now a conflict
    anyway).
  - **The proof kit**, so that its theorems prove the new `WellBehaved`:
    - `Safe` (`TVA/Lemmas/Safe.lean`) writes each byte at most once: its write footprint is an
      index, a store or atomic hands the rest of the program `W` without the bytes it wrote, and
      an atomic's bytes must be outside `R` (instead of inside it). `Safe.bind`/`seq` take two
      disjoint write footprints (`Safe.bind_ro` when the first part writes nothing); `mono` no
      longer enlarges `R`. `Safe.stepT` (a step from a memory that agrees on the footprint, a
      deferred atomic included) and `Safe.no_ub` replace `Safe.transfer`; `step` gives the
      smaller footprint; `acc` covers deferred atomics. `Step.frame` and the new `Step.transfer`,
      `Mem.mix`, `writeAll_mix`, `Mem.mix_write` are generic facts about one step.
    - Store and atomic rules (`safe_store`, `safe_store_any`, `safe_store_buf`,
      `safe_store_block`, `safe_store_block_ptr`, `safe_desc_store`, `safe_desc_scatter`;
      `safe_atomic_buf`, `safe_rmwSeq_buf`, `safe_desc_atomic`) give the rest the footprint without the written
      bytes; the atomic rules ask `Disjoint … R` instead of `… ⊆ R`. Loop rules take per-iteration
      write footprints, disjoint across iterations (`safe_forN`, `safe_forN_bind`,
      `safe_forRange`: `Wt` per trip; `safe_while`: `Wv` per value of the measure; `iterW`).
    - `independent` and `meets_independent` take `hRW : Disjoint (R i) (W i)` and prove no
      `IntraRace` (with `intraRace_snoc`).
    - `SafeA` (`TVA/Lemmas/Shared.lean`) changes the same way. Its log entries are
      `AtomicStep`s of a cell, an old and a new value (no update relation), which a step
      determines (`AtomicStep.unique`); `SafeA.perm` reorders a log; `SafeA.stepT`, `no_ub` and
      `acc` cover deferred atomics. `shared` takes `hRW` and `hQ : PermInv` (the postcondition
      ignores the log's order, since a program's atomics may run in any order); `addsTo_add` takes
      the step's values only.
- **Why.** The user's decision on L15 (2026-10-27): no conflicting accesses within a program; two
  atomics on the same bytes do not race, but the semantics must allow every order; `debug_barrier`
  keeps no meaning and `num_stages` stays dropped. The library said a program is sequential, and
  the device does not run one that way, so `Compiler.Faithful` was false for such models.
- **Evidence.** [evidence/intra-program](../../../../evidence/intra-program/README.md) (TVA's own
  reproduction, after `issue/stale_readback.py` and lr-2b's `work/seq_readback.py`,
  `loop_carried_mem.py`): reading a block back reversed gives stale data in 41.4% (1024 elements)
  and 23.0% (4096) of the elements, and none with a barrier or at the same offsets; an in-place
  reversed copy is wrong in 0.2% and 1.8%; a loop that loads the previous iteration's row is 96.9%
  stale under `tl.range(num_stages=3)` or through `tl.dot` at the default `num_stages`, and never
  with plain `range`. Two atomics of one program on one address ran in the reverse of source order
  on 0.33–0.76% of the addresses, each still atomic: the reason for `Step.defer`. In Lean,
  `independent` and `shared` prove the new `WellBehaved`, and `exists_total_faithful` still holds
  (no state is stuck: `Step.progress` is unchanged).
- **Choices to review.**
  - The exemption is for two atomics *on the same bytes*: atomics of one program on partly
    overlapping bytes conflict (listed in exclusions.md). The decision says "unless both are
    atomics" and "two atomics on the same bytes do not race"; overlapping atomics of different
    widths are not atomic with respect to each other in PTX.
  - `Step.defer` lets a later step overtake an atomic only when the step does not depend on the
    atomic's old value. That is every order the device can produce (a thread cannot use a value
    before the atomic returns it). It also lets a program's atomics on *different* bytes reach
    memory in any order, which other programs can observe; the semantics before allowed only
    source order there.
  - The kit covers programs that write each private byte once and read none of what they write.
    A model that is well behaved under the rule but performs two atomics on the same *private*
    bytes is not covered by `independent` (only `shared`'s cells take repeated atomics).
- **Constants whose meaning changed.** `TVA.WellBehaved` (and so `Rules.Compiler.Faithful`, whose
  docstring also changed), `TVA.Step` (new constructor `defer`), `TVA.Event` (new field `cell`),
  `TVA.Prog` (docstring), `TVA.Safe`, `TVA.SafeA`, `TVA.AtomicStep`, `TVA.Safe.Next`,
  `TVA.SafeA.Next`; the statements of `Safe.bind`, `Safe.seq`, `Safe.mono`, `Safe.step`,
  `SafeA.bind`, `SafeA.mono`, `SafeA.of_safe`, `SafeA.atomic1`, the store, atomic and loop rules
  listed above, `independent`, `meets_independent`, `shared`, `addsTo_add`. New: `IntraRace`,
  `Event.SameAtomic`, `Step.transfer`, `Mem.mix`, `Safe.stepT`, `Safe.no_ub`, `Safe.bind_ro`,
  `iterW`, `intraRace_snoc`, `PermInv`, `SafeA.perm`, `SafeA.stepT`, `SafeA.no_ub`,
  `AtomicStep.unique`. Removed: `Safe.transfer`, `SafeA.transfer`, and `AtomicStep`'s fields `upd` and `ok`.

## L16: `tl.max`/`tl.min` widen small integers to `int32`

- **What changed.** `[reduce]` in [TRANSCRIPTION.md](../../../TRANSCRIPTION.md): before `tl.max`
  and `tl.min` without `return_indices`, an integer narrower than 32 bits becomes `int32` whatever
  its signedness (`int8`, `int16` sign-extended; `uint8`, `uint16` and `int1` zero-extended), and
  the model writes that conversion; `tl.sum` reduces in `int32` (signed) or `uint32` (unsigned)
  and wraps there; the index reductions keep the input's type. [SEMANTICS.md](../../../SEMANTICS.md)
  (the promotions and "Integer reductions"), the docstrings of `Tensor.foldAll`, `foldRows`,
  `foldCols`, `tl.maxAll`, `max1`, `max0`, and [PROVER_GUIDE.md](../../../PROVER_GUIDE.md) say so.
- **Why.** `tl.max`/`tl.min` convert every integer narrower than 32 bits to `int32`
  (`primitives/triton-3.8.0-standard.py:185-190`, `:244-249`), and `tl.sum` picks `int32` or
  `uint32` by the input's sign (`_pick_sum_dtype`, `:270-281`). The library did not say so.
- **Evidence.** [evidence/int-reductions](../../../../evidence/int-reductions/README.md): all 35
  results of part A (each integer type, `max`, `min`, `sum`, `argmax`) and the 8 sums of part B
  agree with the statement; `uint8`/`uint16` are zero-extended (`255`, `65535`); the value of `tl.max`
  with `return_indices` keeps the input's type (`r + 1000` is rejected at compile time for `int8`
  and `uint8`, and wraps in 16 bits for `int16`).
- **Constants whose meaning changed.** None by definition (the folds and `tl.max*` are generic;
  the model now writes the conversion). Docstrings of `Tensor.foldAll`, `foldRows`, `foldCols`,
  `tl.maxAll`, `max1`, `max0` (not hashed into claims); the rule `[reduce]`.

## L17: No `tl.permute` (rank ≥ 3), PDL or `tl.argmax`

- **What changed.**
  - **`tl.permute`.** `Tensor.permute x dims` (output dimension `k` is input dimension
    `dims[k]`; `dims` a permutation, checked by `decide`), with `Tensor.get_permute`,
    `Tensor.permute_trans` (`[1, 0]` is `trans`), `Idx.ofCoords`, `Idx.coord_ofCoords`,
    `Shape.dim_eq_getD`. TRANSCRIPTION `[permute]`: `tl.permute(x, dims)` and `tl.trans(x, dims)`
    with a literal `dims`; `tl.trans(x)` of rank `r ≥ 3` swaps the last two axes.
  - **PDL.** `Launch.pdl` (`launch_pdl`, default `false`) and `Launch.Valid` asks `pdl = false`;
    `tl.extra.cuda.gdc_wait()` and `gdc_launch_dependents()` are dropped (`[no-op:gdc]`, a new
    drop reason of the tool); exclusions.md lists `launch_pdl=True`;
    `primitives/triton-3.8.0-gdc.py` is pinned.
  - **Index reductions.** `CmpKey` (Triton's `<` and `==` on the element type, and which elements
    are NaN), `tl.argCombine` (Triton's `_argmax_combine`/`_argmin_combine`), `tl.ArgRes` and
    `tl.argmaxAll`, `argmax1`, `argmax0` for `tl.max`/`tl.min` with `return_indices`,
    `tl.argmax` and `tl.argmin`; rules `safe_argmaxAll`, `safe_argmax1`, `safe_argmax0`, with
    `ArgRes.index_lt`, `ArgRes.value`, `ArgRes.tree_of_noNaN`, `Tree.eval_argCombine`,
    `Tree.eval_argCombine_first`, `IntV.cmpKey_lt`/`eq`/`nan`. TRANSCRIPTION `[argmax]`;
    SEMANTICS.md "Index reductions".
- **Why.** lr-2b found the three missing: `tl.permute` and `trans` of rank 3 or more had no
  model, `launch_pdl=True` passed `Launch.Valid`, and the docs named `tl.argmax` without defining
  it.
- **Evidence.**
  - [evidence/permute](../../../../evidence/permute/README.md): every permutation of ranks 3 and 4
    (6 and 24), through `tl.permute(x, dims)`, `tl.trans(x, dims)` and `tl.permute(x, *dims)`,
    equals `torch.permute`; `tl.trans(x)` swaps the last two axes at every rank from 2.
  - [evidence/pdl](../../../../evidence/pdl/README.md): the calls compile to `griddepcontrol`
    instructions and change no result without PDL (0 differing elements); `launch_pdl=True` is
    accepted, and a consumer without `gdc_wait` then read stale data in 19 of 20 runs (35% of the
    elements); with `gdc_wait`, or without PDL, none.
  - [evidence/argmax](../../../../evidence/argmax/README.md): without NaN in the slice the device
    agrees with the definition for every type tested (first index of the maximum with
    `tie_break_left`); with a NaN, `float32` still pairs the value with its index, but `float16`
    and `bfloat16` (the latter reduced in `float32`) take the value and the index from different
    positions in 19.4% of the `[16, 64]` axis-0 slices and 1.0–1.1% of the `[32, 256]` axis-1
    slices. So `tl.ArgRes` allows, when the slice holds a NaN, any element's value (up to NaN
    payload) with any index in range, for every float type; the value was always an element and
    the index always in range. fp8 operands are rejected on the host. (An earlier version of the
    page missed the `bfloat16` case through a script bug, now fixed and described there.)
- **Constants whose meaning changed.** `TVA.Launch` (new field `pdl`), `TVA.Launch.Valid` (new
  conjunct `pdl = false`), `Launch.valid_grid1` (proof). New: `Tensor.permute` and its lemmas,
  `Idx.ofCoords`, `CmpKey`, `tl.argCombine`, `tl.ArgRes`, `tl.argmaxAll`, `tl.argmax1`,
  `tl.argmax0` and their lemmas. Rules `[permute]`, `[argmax]`, `[no-op:gdc]`; the tool's `DROPS`.

## L18: `TRITON_F32_DEFAULT` changes `tl.dot`'s default precision

- **What changed.** `Rules.Compiler.Faithful` now covers only launches with Triton's environment
  at its defaults: none of the environment variables Triton 3.8.0 reads to compile or launch a
  kernel is set (`primitives/triton-3.8.0-knobs.py:343-513`, `primitives/triton-3.8.0-GetEnv.h:15-57`,
  both pinned), except those that only name directories, dump, print, time or change debug
  information, listed in the docstring and on [the rule page](../../../rules/compiler-faithful.md).
  `TRITON_F32_DEFAULT`, `TRITON_DEFAULT_FP_FUSION`, `TRITON_DEBUG`, `TRITON_INTERPRET`, … are
  unset. The definition is unchanged: this is part of the rule's stated scope, as `debug = false`
  and `Launch.Valid` are. exclusions.md lists a non-default environment; the docstrings of
  `DotCfg.precision`, `Launch.fusion` and `Launch.debug` and TRANSCRIPTION §9 and `[dot]` (an
  omitted `input_precision` is `tf32`, `ieee` with `allow_tf32=False`) say what the defaults are.
- **Why.** `TRITON_F32_DEFAULT` sets the `input_precision` of a `tl.dot` that gives none
  (`primitives/triton-3.8.0-knobs.py:496`), and `TRITON_DEFAULT_FP_FUSION` the default of
  `enable_fp_fusion` (`:497`), so the model's reading of a source depends on them; other
  variables replace the compiler or its output.
- **Evidence.** [evidence/env-knobs](../../../../evidence/env-knobs/README.md), each case in a
  subprocess with Triton's variables removed and a fresh cache: unset, a `tl.dot` without
  `input_precision` is `tf32` and fusion is on (one `fma`); `TRITON_F32_DEFAULT=ieee`/`tf32x3`
  change the precision and override even `allow_tf32=False`, `bogus` is rejected on the host;
  `TRITON_DEFAULT_FP_FUSION=0` gives a separate multiply and add (`y = 0` instead of `2^-26`). An
  explicit `input_precision`, or `enable_fp_fusion` at the launch, is not overridden.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (docstring, hashed into every
  claim's `faithful` assumption, so every kernel's `assumption:faithful` re-opens), and the
  docstrings of `TVA.DotCfg.precision`, `TVA.Launch.fusion`, `TVA.Launch.debug`.
