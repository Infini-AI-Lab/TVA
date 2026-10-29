# Library batch, round 21: author's notes

Issues: L95, L102, L103, L104, L109, L110, L111, L112, L113, L114
([ISSUES.md](../../ISSUES.md)), 2026-10-28 to 2026-10-29: the final batch, with no review after it
(coordinator). Each issue is fixed as its round-20 record asks; where the record offers two fixes,
the section says which was taken. Worked in TVA's tree itself.

**The user's decisions on `Compiler.Faithful`'s scope** (2026-10-29, relayed by the coordinator
while this batch ran, and applied as one change): "To make the verifier easy, we do not verify
triton cache.", "we assume the kernel is new." and "So basically, we want to verify the kernel
itself, rather than triton installation or other code programmed by users." The scope is now in
two named parts, in this order wherever it is stated (`lean/TVA/Rules/Compiler.lean`,
`docs/rules/compiler-faithful.md` with its checklist, `docs/REVIEWER_GUIDE.md`,
`docs/PROVER_GUIDE.md`, `docs/SEMANTICS.md` "Launches", `docs/exclusions.md`, `TVA.Claim`; also
TRANSCRIPTION's host pitfall, SEMANTICS "Names", REVIEW_PROTOCOL §7 and the conclusions of
`evidence/host-round20`, `evidence/host-round21` and `evidence/compile-keys`). Each condition keeps
its number, so that every citation of "scope `n`" (one is in `kernels/vec_add_f32/README.md`,
which this batch may not edit) still names it, and the items are bullets labelled with their
numbers.
- **(A) The kernel and its launches: checked**, as before, by the kernel reviewer: 4 (the options
  the launch call, `warmup`, `triton.compile` and the configs give, `@triton.heuristics`, the
  decorator's `debug`), 5 (the descriptor allocator), 6 (a precompiled kernel's source and
  arguments), 8 (host descriptors) and 9 (CUDA graph capture), and "Which launch" under
  `triton.autotune`.
- **(B) The environment: assumed, not verified**: 1 (nothing else writes the launch's bytes), 2
  (knobs and environment variables), 3 (hooks), the environment's part of 4 (no `kernel.debug` set
  by code other than the kernel's source, no `JITFunction.preload`: library issue L105), 7 (the H200
  and the host: x86-64, CPython 3.13 without `-O`, the default floating-point environment: library
  issues L107, L110), 10 (Triton as installed and no other code changing Triton or the kernel's
  module: L104) and 11 (the kernel is new: L103, L109). Their texts are unchanged but for being
  assumptions. The gate, the prover and the kernel reviewer do not check them, and the reviewer
  reads no host code for them. A claim's own launch or kernel source that visibly breaks one is
  still a finding. REVIEWER_GUIDE and the rule page's checklist list only (A), with one paragraph
  on (B); PROVER_GUIDE says the write-up only states that the claim relies on (B).
- **Choices where the part was unclear** (the coordinator's rule: (A) if the kernel's source or its
  launch call controls it, else (B)). Scope 1 is in (B): other writers, the next launch with
  `launch_pdl=True` included, are other code, not the kernel or its launch call. Scope 9 is in (A):
  whether the launch call is made inside a capture is a property of the launch. Scope 7 is wholly in
  (B), the device included; an `arch` option or a `triton.compile` target the launch's own call
  gives is still scope 4's or 6's, checked. Scope 4 is split: the options the launch's code passes
  are (A), the attribute and `preload` set from elsewhere are (B). Scopes 5 and 8 are (A), as the
  coordinator listed them, with their "nothing else writes the scratch" clause kept as it was.
- **Scope 11's reading.** As the coordinator's text puts it, the code the launch runs "was not taken
  from Triton's caches … as code compiled earlier". Read literally, that would cover no launch
  under `triton.autotune`: its benchmark run compiles the chosen config, and the final launch
  takes that code from the cache. It would also cover no repeated launch. So scope 11 says: not
  code compiled earlier *for other values or under other settings* and taken from a cache; code
  compiled earlier for the same source, values and settings counts as compiled for the launch.
- **L105 and L107** are not issues of this round, so they have no section here (the notes' format
  allows only the round's issues). What moved for them: L105's `kernel.debug` set from outside and
  `preload` now form scope 4's (B) part, an assumption, where round 20 had the reviewer check them.
  The decorator's `debug` stays `Launch.debug`, in (A). L107's host (x86-64, CPython 3.13) is in
  scope 7, now (B), with L110's additions.

**Build.** `lake build TVA` is green (3302 jobs). The new theorems (`TVA.safe_convert`,
`TVA.safe_sum0c`, `TVA.safe_store1`) depend only on `propext`, `Classical.choice` and
`Quot.sound`, and `TVA.tl.store1_eq` on none (`#print axioms` on each); no `sorry` or `axiom` in
the library. `tvr library-check` passes (hygiene, and `leanchecker TVA` re-checks every
declaration). `tvr lint library 21 author` prints `regular` and `tvr audit` `independent`.

**Kernel.** After the last library change, `python3 -I tools/tvr.py check vec_add_f32`: PASS
(every step), digest `1fcbc848632b79f5` (was `7db73696a58abeb2`; `5d9654e5e7845e76` before the
user's decisions on the scope were applied). All five items' digests change:
their cones reach `docs/SEMANTICS.md`, `docs/exclusions.md`, `docs/rules/compiler-faithful.md`
and `docs/rules/num.md`, which are hashed whole, and Lean files whose texts this batch changes
(`lean/TVA/Claim.lean`, `Lang/Scalar.lean`, `Rules/Compiler.lean`); the constants of the cones that
changed are `TVA.Rules.Compiler.Faithful`, `TVA.FloatV.hostF32` and `TVA.FloatV.litF8`, docstrings
only. No fix as good leaves those files alone. The run rewrote the files `tvr check` writes
(`kernels/vec_add_f32/gate.json`, `claim.md`, `REVIEW.md`, a snapshot, `kernels/README.md`); no
file of the kernel's own was edited. No kernel fails to build.

**Release check.** Run at the end, from TVA's root (after `tvr check vec_add_f32` and these
notes): `release check PASS (12 of 12 steps pass)`, 1197 shipped files, kept as
[evidence/tool-round21/release.out](../../../../evidence/tool-round21/release.out) and
`release.json` beside it.

## L95: the facts section read as CommonMark and GFM render it

- **What changed.**
  - `tools/tvr.py`, `facts_section`: before the section, fences are tracked as CommonMark tracks
    top-level ones (an opener of three or more backticks or tildes, none when a backtick opener's
    info string holds a backtick; a closer of the opener's character, at least as long, with nothing
    after it but spaces). A problem (the gate then accepts no fact, the audit names the line): a line
    before the section, outside a fence, that starts with `<` after up to three spaces (an HTML
    block); a fence line indented by one to three spaces (it may be a list item's, which a line reader
    cannot tell); in the section, a `<` before a letter, `/`, `!` or `?` anywhere in a line (code spans
    included), a `~` (GFM's strikethrough), and a table whose header and delimiter rows have different
    numbers of cells. The HTML-comment tracking and the round-20 allowlist stay.
  - `tools/tvr.py`: new `gfm_cells(row)`, a row split as GFM splits it (one leading and one trailing
    `|` removed, then a split at each `|` no backslash escapes); `stated_facts` and the cell-count
    check use it, so a row that starts `||` has an empty first cell (the record's first option).
  - `docs/REVIEW_PROTOCOL.md` §2 ("Fact pages") says exactly this, §5 and `docs/LIBRARY_REVIEW.md`
    check 5 and `docs/prompts/library-author.md` in short. `docs/rules/num.md` already met it but for
    one cell: the atomic fact's row wrote a scope as `<scope>`, which is HTML to GFM; it is now
    written with braces (the fact and its name are unchanged).
- **Why.** lr-20c: a page whose section renders as code, or not at all, or with a fact in another
  column, still gave 38 facts with the audit silent. The three added refusals (`~`, unequal cell
  counts, an indented fence) are of the same class, cheap, and fail closed.
- **Evidence.** [evidence/tool-round21](../../../../evidence/tool-round21/README.md) Part L
  (2026-10-29): the tree's page, 38 facts, no problem; two controls (text after the heading; closed
  `~~~` and ```` ````text ```` fences before it), 38 facts; lr-20c's nine forms and the four added:
  no fact (`ExpF32 P` refused) and the audit names num.md, but the `||` row, which gives an empty
  first cell, so `DivRn P .binary32` is refused, as the page shows it under no fact.
- **Constants whose meaning changed.** None (the tool, the protocol and a cell of num.md; the facts
  page lists the same 38 facts with the same statements).

## L102: a helper's equal `return`s: the same value, bit for bit and element by element

- **What changed.** The record's first option, made exact for floats: the admitted third form is
  now "`return`s that all give the same compile-time value: the same `None`, `int`, `float` (bit for
  bit: not a NaN, and not `0.0` with `-0.0`), `bool`, string or dtype, of one Python type, and for a
  tuple the same element by element (`(1, 2)` with `(1, 2)`, not with `(1.0, 2)`)", in
  `docs/SEMANTICS.md` "The supported subset" (`return`), with "A helper's value" saying that `==`
  compares tuples element by element and finds `0.0` equal to `-0.0`; `docs/exclusions.md`'s row,
  TRANSCRIPTION's pitfall "A helper's value read as Python's" and `docs/REVIEWER_GUIDE.md` §1's list
  say the same; new folder `evidence/front-end-round21`.
- **Why.** lr-20b: `(1, 2)` and `(1.0, 2)` are equal values of one Python type, and the front end
  gives the first visited `return` on every path. `0.0` and `-0.0` are the same case (equal, of one
  type, different values), so "not a NaN" alone was not enough; excluding tuples would have left it
  open too. Bit for bit is the smallest rule that is true.
- **Evidence.** [evidence/front-end-round21](../../../../evidence/front-end-round21/README.md)
  T1–T4 (Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28), with a runtime `c = 5` and `c = -5`: T1
  `return (1, 2)` then `(1.0, 2)`, `x * th(c)[0]` stored `16777217` on both paths (the second's
  value: `16777216`); T2 the other order, `16777216` on both; T3 through `a, b = th(c)`, as T1; T4
  `return 0.0` then `-0.0`, `0.0` on both paths.
- **Constants whose meaning changed.** None (no Lean constant models a helper's `return`s). Fact
  pages: `docs/SEMANTICS.md`, `docs/exclusions.md`.

## L103: the kernel is new: assumed, and Triton's caches not verified

- **What changed.** I followed the user's decisions (2026-10-29, relayed by the coordinator): "To
  make the verifier easy, we do not verify triton cache.", "we assume the kernel is new." and "So
  basically, we want to verify the kernel itself, rather than triton installation or other code
  programmed by users." So the record's fix (a corrected reviewer's check) is replaced. Scope 11 of
  `Faithful` is now one assumption, *The kernel is new*, in the scope's part (B): the code the
  claimed launch runs was compiled for that launch, from its own source, its compile-time values
  (an unannotated integer `1` and `None` included) and the settings the other scope conditions
  give; it is not code compiled earlier for other values or under other settings and taken from
  Triton's caches (the `JITFunction`'s cache in the process, or the cache directory). Code compiled
  earlier for the same source, values and settings counts as compiled for it (the intro, "Scope
  11's reading"). TVA does not verify Triton's caches. Nobody checks it (not the gate, the prover or
  the kernel reviewer), and no checklist item covers it. The text is in
  `lean/TVA/Rules/Compiler.lean` (scope 11, and the scope's lead-in), `docs/rules/compiler-faithful.md`
  (summary, "Scope" 11, "What the rule does not cover"), `docs/SEMANTICS.md` ("Parameters and
  arguments", "A launch's compiled kernel", the NaN constant, "Launches"), `docs/exclusions.md` and
  `TVA.Claim`'s module docstring; `FloatV`'s section docstring in `lean/TVA/Lang/Scalar.lean` points
  to it. Removed: the cache-key rules and the lists of colliding values from scope 11, the
  fresh-cache shortcut, and the checks. `docs/REVIEWER_GUIDE.md`'s host list and the rule page's
  checklist now hold only part (A), and `docs/PROVER_GUIDE.md` names scope 11 among the
  assumptions the write-up only says it relies on. TRANSCRIPTION's host pitfall says the same. The rule page cites `evidence/compile-keys`, `evidence/host-round21` and
  the round-20 reports as the record of why the assumption is needed. Kept: round 20's helper
  copy-name rule in the supported subset (SEMANTICS "Compile-time values Triton keys apart", its own
  row of `docs/exclusions.md`), which is the front end's naming within one compile, not a cache.
- **Why.** lr-20a, lr-20b, lr-20c, lr-20f showed the check the reviewer was given was false: the
  process keeps its own cache, the values are keyed as one tuple, every `==` pair collides, and an
  autotuner's benchmark runs are earlier launches. The user chose not to verify the caches at all,
  and to assume the kernel is new.
- **Evidence.** The record of why the assumption is needed:
  [evidence/compile-keys](../../../../evidence/compile-keys/README.md) K1–K4 (its conclusion 1 now
  says this), and [evidence/host-round21](../../../../evidence/host-round21/README.md) K1–K5
  (Triton 3.8.0, H200 NVL, GPU 0, 2026-10-29, one process each): K1, on a fresh directory,
  `C = 2.0` after `C = 2` stored `33554434` (alone: `33554432`), and the reverse; K2, `C = 1`, then
  a fresh directory set in the process, then `C = 1.0`: `16777217` (alone: `16777216`); K3,
  `("a", "b-c")` after `("a-b", "c")`: `31` (alone: `13`); K4, an autotuner's chosen `{"C": 1.0}`
  ran the benchmarked `{"C": 1}`'s code; K5, the controls. The README's new conclusion 3 says the
  output's headings name the first draft's scopes.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: scope 11 is now the
  assumption, in part (B), that the kernel is new (it was a condition the reviewer checked). It
  covers the same launches as round 20's scope 11 meant to: those that run code compiled for their
  own values and settings. The condition is now assumed, not checked. Its definition is unchanged. `TVA.Claim`: module docstring. Fact pages:
  `docs/rules/compiler-faithful.md`, `docs/SEMANTICS.md`, `docs/exclusions.md`.

## L104: scope 10 as (a) to (d): the installed Triton, changes by any means, the tutorials' API

- **What changed.** Scope 10 is the record's (a)–(d), the same text in
  `lean/TVA/Rules/Compiler.lean`, the rule page's scope 10, `docs/SEMANTICS.md` "Names" and
  "Launches", and `docs/exclusions.md`'s row. I followed the user's decisions (2026-10-29, relayed
  by the coordinator): "To make the verifier easy, we do not verify triton cache.", "we assume the
  kernel is new." and "So basically, we want to verify the kernel itself, rather than triton
  installation or other code programmed by users." So scope 10 moved to part (B) of the scope: it
  is an assumption, which nobody checks. Round 20's "stated limit with the reviewer's check" is
  gone, as are its item in REVIEWER_GUIDE's host list, in the rule page's checklist and in
  PROVER_GUIDE's list (each now names it among the (B) assumptions), and REVIEW_PROTOCOL §7's
  sentence that the reviewer checks it. The text of (a)–(d): (a) the process imports Triton 3.8.0's installed
  package, unmodified (no other `triton` earlier on `sys.path`, no substitute in `sys.modules`);
  (b) no code but Triton's own changes an attribute of a Triton module, of a class Triton defines or
  of an object of one, or of the kernel's or a helper's `JITFunction`, by assignment,
  `setattr`/`delattr` or a call made to change it (`_unsafe_update_src`, `driver.set_active`,
  `importlib.reload`); (c) other than through Triton's API as the in-scope tutorials use it:
  launches, `warmup`, `triton.autotune` with its `Config`s and hooks, the `JITFunction`'s `debug`
  (scope 4), `triton.set_allocator` (scope 5), the knobs scope 2 allows, and the fields of a host
  `TensorDescriptor` the launch is given (a hook may set its `block_shape`); every other scope still
  holds of these; (d) the clause on the kernel module's names, unchanged.
- **Why.** lr-20a, lr-20d: round 20's text forbade what in-scope hosts must do (a hook's
  `block_shape`, `set_allocator`, a knob) and missed changes made by a call and a substitute
  `triton`. To (b) "or of an object of one" was added beside the record's text: the knobs, the
  allocator and a host `TensorDescriptor` are objects of Triton's classes, so (c)'s exceptions need
  (b) to name such objects. As the coordinator asked under the first two decisions, the record's
  remark that nothing says the cache directory holds only what Triton wrote is left out of
  scope 10: the cache is scope 11's assumption (L103, L109).
- **Evidence.** Round 20's [evidence/host-round20](../../../../evidence/host-round20/README.md)
  M1–M4 stays cited (`3·x` for `2·x`, `log x` for `exp x`, `5·x` after `_unsafe_update_src`); the
  exceptions follow the tutorials' own code (`tutorials/06-fused-attention.py`,
  `08-grouped-gemm.py`, `09-persistent-matmul.py`, as
  `docs/library-review/round-20/lr-20a/work/scope10.out` lists) and
  `primitives/triton-3.8.0-allocation.py`. No new device test: (a)–(d) narrow and widen the scope
  only where lr-20a's probe showed the old text wrong.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: scope 10's docstring (the
  rule now covers the tutorials' `set_allocator`, hooks' `block_shape` and allowed knobs, and covers
  no launch after a change made by a call or an imported substitute; the condition is now assumed,
  part (B), not checked). Fact pages:
  `docs/rules/compiler-faithful.md`, `docs/SEMANTICS.md`, `docs/exclusions.md`.

## L109: code compiled earlier under other settings: covered by the same assumption

- **What changed.** I followed the user's decisions (2026-10-29, relayed by the coordinator): "To
  make the verifier easy, we do not verify triton cache.", "we assume the kernel is new." and "So
  basically, we want to verify the kernel itself, rather than triton installation or other code
  programmed by users." No new scope condition and no reviewer's check. This round's first draft
  had a scope 12 (scopes 2, 3, 7 and 10 at every compile whose code the launch may run, met by a
  fresh directory or `TRITON_ALWAYS_COMPILE=1`, with a host check). It is removed everywhere it
  was put: Lean, the rule page and its checklist, both guides, SEMANTICS "Launches", exclusions.md
  and `TVA.Claim`. Code compiled earlier under other knobs, `TRITON_KERNEL_OVERRIDE`, a hook, or
  another state of Triton or of the host process, and reused through either cache, is what scope
  11 assumes away (the kernel is new; L103), in part (B), the environment. Nobody checks it.
  `docs/exclusions.md` has one row for all such reuse, citing the user's decisions. The scopes are
  numbered 1 to 11, and each keeps its number in parts (A) and (B).
- **Why.** lr-20c, lr-20d: neither cache key holds a knob set from Python,
  `TRITON_KERNEL_OVERRIDE`, Triton's modules in memory, a hook or the host process's state, so a
  clean launch may run code compiled outside the scope. The user chose to assume the kernel new
  rather than verify the caches.
- **Evidence.** The record of why the assumption is needed:
  [evidence/host-round21](../../../../evidence/host-round21/README.md) X1–X5
  (2026-10-29): a process A outside the scope, then B (Triton at its defaults) on A's directory, C
  on a fresh one, D on A's with `TRITON_ALWAYS_COMPILE=1`. X1 `tl.exp = tl.log` in A: B stored
  `log 2`, C and D `exp 2`; X2 `fp32_default = "ieee"` set from Python: B the `ieee` `tl.dot`, C and
  D tf32; X3 `fesetround(FE_TOWARDZERO)`: B the constant `0x3dcccccc`, C and D `0x3dcccccd`; X4 A
  under `-O`: B ran a kernel C and D reject on the host; X5 in one process, the knob reset to its
  default left the `ieee` code in use, also with `TRITON_ALWAYS_COMPILE=1`. Round 20's
  `docs/library-review/round-20/lr-20d/work/cache_cross_process.py` X3 (`TRITON_KERNEL_OVERRIDE`)
  stays cited.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: no new scope; scope 11's
  assumption (L103) covers this case, and its definition is unchanged. `TVA.Claim`: module
  docstring. Fact pages: `docs/rules/compiler-faithful.md`, `docs/SEMANTICS.md`,
  `docs/exclusions.md`.

## L110: the host process: no `-O`, the default floating-point environment

- **What changed.** Scope 7 names the host process as CPython 3.13 run without `-O`/`-OO` and with
  `PYTHONOPTIMIZE` unset, in the default floating-point environment (round to nearest even;
  flush-to-zero and denormals-are-zero off: no `fesetround`, no `_mm_setcsr`, no loaded library that
  changes them), with the two reasons (`-O` removes the `assert`s by which the front end rejects
  some kernels, `triton-3.8.0-code-generator.py:497, 872, 886, 1020, 1117-1121`; the environment
  decides the constants Triton builds, the `float32` of a `float` argument and compile-time float
  arithmetic), in `lean/TVA/Rules/Compiler.lean`, the rule page's scope 7, both guides' list of
  part (B), `docs/SEMANTICS.md`'s introduction and "Launches", `docs/exclusions.md`'s targets
  row, TRANSCRIPTION's host pitfall, `FloatV.hostF32`'s docstring (`lean/TVA/Lang/Scalar.lean`),
  and `TVA.Claim`'s module docstring. The record asked for "the reviewer's check". I followed the
  user's decisions instead (2026-10-29, relayed by the coordinator): "To make the verifier easy, we
  do not verify triton cache.", "we assume the kernel is new." and "So basically, we want to
  verify the kernel itself, rather than triton installation or other code programmed by users."
  So scope 7 (the H200, and L107's and L110's host) moved to part (B) of the scope: it is an
  assumption, which nobody checks, and it left the reviewer's host list and the rule page's
  checklist. The guides name it among the (B) assumptions; `hostF32`'s docstring and
  exclusions.md's targets row say "assumed".
- **Why.** lr-20b, lr-20d: both host states change outcomes the library states (a host rejection
  becomes a run; `litF32`, `hostF32`, `litBF16` round otherwise).
- **Evidence.** [evidence/host-round21](../../../../evidence/host-round21/README.md) H1–H3
  (2026-10-29), each on a fresh directory: H1, a loop body giving a carried `tl.float16` name the
  dtype `tl.float32`, rejected under `python -I`, ran under `-I -O`, `-I -OO` and
  `PYTHONOPTIMIZE=1` and stored `1.0` (Python's value: `1.00048828125`); H2, after
  `fesetround(FE_TOWARDZERO)`, `tl.full([1], 0.1, tl.float32)` and the argument `0.1` stored
  `0x3dcccccc` (plain process: `0x3dcccccd`), and the constexprs `0.1 + 0.2` gave
  `0x3fd3333333333333`; H3, flush-to-zero and denormals-are-zero set by `_mm_setcsr` and by loading
  a library built with `-ffast-math` (MXCSR read back): `tl.full([1], 1e-40, tl.float32)` stored `0`.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: scope 7 (the rule covers no
  launch of a process run with `-O` or in another floating-point environment); `TVA.FloatV.hostF32`:
  docstring (its value is the default environment's, as before). Fact pages:
  `docs/rules/compiler-faithful.md`, `docs/SEMANTICS.md`, `docs/exclusions.md`.

## L111: arithmetic in an fp8 dtype is outside the subset

- **What changed.** `docs/SEMANTICS.md` "The supported subset": its Operators item puts every
  operator whose computation dtype (`computationType?`) is an fp8 dtype, and unary `-` on an fp8
  tensor, outside the subset, and its list of excluded forms names them; "Operators and promotion"'s
  last item and the fp8 bullet of "Constants" now say what the record asks (the backend rejects
  such arithmetic on the host, except where MLIR's folders remove the operation first: `x * 1`,
  `1 * x`, `x / 1` and `x - 0` give `x`, and an operation on two fp8 constants is computed at
  compile time; none of these forms is in the subset), in place of "rejected anyway" and "the
  backend rejects it on the host". New row of `docs/exclusions.md`, citing lr-20e's
  `work/fp8_arith.py` and the new `evidence/front-end-round21` F.
- **Why.** lr-20e: some such forms fold and run, so a model that gives them no outcome is wrong at
  `SCALE = 1.0`. No in-scope tutorial computes in fp8, so excluding the class costs nothing.
- **Evidence.** [evidence/front-end-round21](../../../../evidence/front-end-round21/README.md) F
  (2026-10-28), 20 forms on all 256 patterns of each fp8 dtype: `x * 1.0`, `1.0 * x`, `x * 1`,
  `1 * x`, `x / 1.0`, `x / 1`, `x - 0.0`, `x - 0` compiled and stored every pattern unchanged; three
  operations on two fp8 constants ran, computed at compile time (`1.5 * 2.0` stored `3.0`); `x + 0.0`,
  `x * 2.0`, `x * -1.0`, `x / 2.0`, `x + x`, `x - x`, `x * x`, `-x` and `x < x` were rejected on the
  host (22 of 40 ran).
- **Constants whose meaning changed.** None (`computationType?` and the operators' Lean models are
  unchanged; the forms are now outside the subset the claim's semantics covers). Fact pages:
  `docs/SEMANTICS.md`, `docs/exclusions.md`.

## L112: the claim reader: a private field refused by name, PANIC read only off the result line

- **What changed.**
  - `tools/claim_reader.lean.in`: after it lists the fields of `Assumptions`, the reader checks that
    each has its projection `<ns>.Assumptions.<f>` in the environment, and otherwise stops with
    "TVR: the field f of Assumptions has no projection … (a private field?); every field must be
    public (library issue L112)" (the record's second option for the field: smaller than reading
    through `projFn`, and a claim's fields are meant to be public anyway).
  - `tools/tvr.py`, `reader_result`: a line holding `PANIC` counts as a panic only when it is not the
    reader's `TVR_JSON ` result line (the record's first option), on stdout and stderr alike.
  - `docs/REVIEW_PROTOCOL.md` §8 (the `shape` row) says so.
- **Why.** lr-20c: a docstring or name holding PANIC failed `shape` as a panic, and a `private`
  field was listed but not read, so `kinds` said "not tagged" of a tagged field. Both failed closed;
  now each says what is wrong.
- **Evidence.** [evidence/tool-round21](../../../../evidence/tool-round21/README.md) Part R
  (2026-10-29), three small claims compiled against the library: the public field (control) passes
  the reader and `kinds` reads it; the `private` field stops the reader (exit 1) with the error
  naming `Kernels.Demo.Assumptions.spec`, and `shape` fails; PANIC in a field's docstring passes
  `shape` (round 20's tool: "the claim reader panicked"); made-up outputs: PANIC on the result line
  alone passes, and a `PANIC at …` line of Lean's on stderr or stdout fails.
- **Constants whose meaning changed.** None (the tool and the protocol).

## L113: rules for `tl.convert`, `tl.sum0c` and `tl.store1`

- **What changed.** `lean/TVA/Lemmas/Ops.lean`: new `safe_convert` (premise
  `∀ i, ∃ r, ConvRel E.P d d' (x.get i) r`; continuation for every `r` with
  `∀ i, ConvRel E.P d d' (x.get i) (r.get i)`, as the record states it), `safe_sum0c` (as
  `safe_sum1c`, along the other axis), and `safe_store1` with `tl.store1_eq` (`tl.store1 p v` is
  `tl.store` of the scalar tensors, `rfl`), so `tl.store1` has a rule of its own (the record's first
  option); the module imports `TVA.Lang.Promote` (for `ConvRel`), and its docstring lists them.
  `docs/LIBRARY.md`'s `Ops` row lists all four.
- **Why.** lr-20f: three model operations had no rule, against "one rule per operation", and every
  kernel using `tl.convert` had to build the result by choice itself.
- **Evidence.** None needed beyond the build: they are theorems about the model, checked by Lean
  (`#print axioms`: `propext`, `Classical.choice`, `Quot.sound`; `tl.store1_eq` none) and by
  `leanchecker TVA`.
- **Constants whose meaning changed.** None; new: `TVA.safe_convert`, `TVA.safe_sum0c`,
  `TVA.safe_store1`, `TVA.tl.store1_eq`. `TVA.Lemmas.Ops`: module docstring and one import.

## L114: wording from the final review of round 20

- **What changed.**
  - `docs/SEMANTICS.md` "Values" (the fp8 row) and "Constants" (the fp8 bullet), and `FloatV.litF8`'s
    docstring (`lean/TVA/Lang/Scalar.lean`): a Python number built in an fp8 dtype is the exact value
    of its `float32` constant (`litF32 c`), or the kernel is rejected; "is never rounded" is dropped
    (`c` is rounded once, to its `float32` constant), with `1 + 2^-30`, `1 + 2^-40` (`1.0`), `1e39`
    (`+∞` in `float8e5`) and `-1e-50` (`-0`) as examples, citing lr-20a's `probes.out` P1 and lr-20b's
    `probes2.out`.
  - `docs/PROVER_GUIDE.md`, "Write each Python number as the constant Triton builds": adds
    "`FloatV.litF8` in an fp8 dtype (not `tl.cvt8`; `Operand.constIn?` gives each)".
  - `docs/rules/num.md` under the facts table: "The conversion facts (`Cvt…`, `CvtSat…`) and
    `DivRn P .binary64` are exact (bit for bit, up to NaN payload); `DivFull` is a bound with margin,
    like the others (no error above 0.651 of it, num-div-f32)".
  - "half an ulp" for `u·|y| + η` is now "one rounding's bound" in the module docstrings of
    `lean/TVA/Lemmas/FusedErr.lean` and `Consistency.lean` and in the docstring of `base`
    (`lean/TVA/Lemmas/Sm90.lean`, a private theorem of `TVA.Platform.sm90ref`), and in
    `Consistency.lean`'s docstring and LIBRARY.md's `Consistency` row `FloatV.exists_nearest` is
    stated for a real `y` with `|y| ≤ maxVal` (was "from any real").
  - `FloatV.ofRat_err` (`lean/TVA/Lemmas/FloatErr.lean`): "The nearest float to an exact rational
    (`FloatV.ofRat`)".
  - `lean/TVA/Lemmas/Shared.lean`'s module docstring: an `AtomicStep` records the cell, its old value
    and its new value, not the operand (which `TVA.Lemmas.SharedAcc` pairs with each step).
- **Why.** lr-20a, lr-20b, lr-20e, lr-20f: each text said more, or other, than its constant or fact.
- **Evidence.** The fp8 examples are round 20's reviewers' probes (cited above) and
  [evidence/fp8-constants](../../../../evidence/fp8-constants/README.md) F1–F3; the `DivFull` margin
  is the facts page's own measurement ([evidence/num-div-f32](../../../../evidence/num-div-f32/README.md)).
- **Constants whose meaning changed.** None in meaning, docstrings only: `TVA.FloatV.litF8`,
  `TVA.FloatV.ofRat_err`, the private `base` of `TVA.Platform.sm90ref`; module docstrings of
  `TVA.Lemmas.FusedErr`, `TVA.Lemmas.Consistency` and `TVA.Lemmas.Shared`. Fact pages:
  `docs/SEMANTICS.md`, `docs/rules/num.md` (no fact's statement changes).

