# Library batch, round 20: author's notes

Issues: L64, L91, L95, L98, L101, L102, L103, L104, L105, L106, L107, L108
([ISSUES.md](../../ISSUES.md)), 2026-10-28: the one fix attempt for every issue still open
(coordinator). Worked in TVA's tree itself; every fix is meant to be final, with nothing left
"for later".

**Build.** `lake build TVA` is green (3302 jobs). The new theorems (`IntV.le_toInt`,
`Format.e4m3_u`, `Format.e4m3_eta`, `Format.e5m2_u`, `Format.e5m2_eta`) depend only on
`propext`, `Classical.choice` and `Quot.sound` (`#print axioms` on each); no `sorry` or `axiom`
in the library. `tvr library-check` passes (hygiene, and `leanchecker TVA` re-checks every
declaration). `tvr lint library 20 author` prints `regular` and `tvr audit` `independent`.

**Kernel.** After the last library change, `python3 -I tools/tvr.py check vec_add_f32`: PASS
(every step), digest `7db73696a58abeb2` (was `721f1db52c61b7ef`). All five items' digests change
(they were already stale for both reviewers, who accepted at `401e1fa92db421d7`): their cones
reach `docs/SEMANTICS.md`, `docs/exclusions.md` and `docs/rules/compiler-faithful.md`, which are
hashed whole, and Lean files whose texts this batch corrects (`lean/TVA/Claim.lean`,
`Lang/Ops.lean`, `Lang/Scalar.lean`, `Rules/Compiler.lean`, `Sem/Launch.lean`); the constants of
the cones that changed are `TVA.Launch.Valid`, `TVA.Launch.debug`, `TVA.Rules.Compiler.Faithful`
and `TVA.Claim.Behaviour`, docstrings only. No fix as good leaves those files alone. The run
rewrote the files `tvr check` writes (`kernels/vec_add_f32/gate.json`, `claim.md`, `REVIEW.md`, a
snapshot, `kernels/README.md`). No kernel fails to build.

**Release check.** Run once, at the end, from TVA's root (after `tvr check vec_add_f32` and these
notes): `release check PASS (12 of 12 steps pass)`, kept as
[evidence/tool-round20/release.out](../../../../evidence/tool-round20/release.out) and
`release.json` beside it.

## L64: an fp8 constant is the exact conversion of its `float32` constant, or a host error

- **What changed.**
  - `lean/TVA/Lang/Scalar.lean`: new `FloatV.litF8 c : Option (FloatV g)`, the constant Triton
    builds for the Python number `c` in an fp8 format: `+0` for a zero; for a finite
    `litF32 c`, `cvt` of it when that is finite and exact, else `none`; for `±∞`, the format's
    infinity in a format that has one (`float8e5`), else `none`; for a NaN, `nanNarrow` (its sign,
    quiet bit and next fraction bit) in a format with NaNs of that kind (`float8e5`) when the
    `float32`'s lower fraction bits are zero, else `none`. The section docstring "Python number
    constants" says so (it said: the `float32` constant, then `tl.cvt8`).
  - `lean/TVA/Lang/Promote.lean`: `Operand.constIn?` is `c.toF64?.bind FloatV.litF8` for `.f8e5`
    and `.f8e4nv` (was `none`, its docstring saying "the `float32` constant, then the platform's
    `tl.cvt8`").
  - `docs/SEMANTICS.md`: the fp8 row of "Values"; "Constants", the fp8 case (was "`FloatV.litF32`,
    then `tl.cvt8`"): `litF8`, with its sources (`semantic.py:601-613`,
    `llvm-ArithOps.cpp:1588-1611, 1788-1825`), its cases and the host rejection; and in
    "everywhere else", a stored number or an `other=` with an fp8 pointer is converted on the
    device by `tl.cvt8`, which saturates.
  - `docs/rules/num.md`, "Conversions into fp8": the `CvtSat…` facts are about conversions at run
    time; a constant is `litF8`, never the device's conversion. This replaces round 14's account
    ("folded by the compiler only when the conversion is exact … so the facts cover it too"),
    which its notes, a recorded record, keep.
  - `docs/TRANSCRIPTION.md`: the pitfall "An fp8 constant read as a conversion"; `docs/LIBRARY.md`,
    the `Scalar` row; new folder `evidence/fp8-constants`.
- **Why.** lr-14a, lr-14b: `tl.full([16], float('-inf'), tl.float8e5)` stores `0xfc`, while the
  model through `CvtSatF32E5m2` gave `0xfb`, so a wrong claim was provable. `scalar_constant`
  builds the `float32` constant and `arith.truncf`s it, and MLIR folds that only when it loses
  nothing; an unfolded one does not compile. A function with `none` where the host rejects is
  exactly that, needs no device fact, and leaves the `CvtSat` facts to the run-time conversions
  they were measured on.
- **Evidence.** [evidence/fp8-constants](../../../../evidence/fp8-constants/README.md) (`run.py`,
  `out.txt`; Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28), against `litF8` evaluated by Lean, bit
  for bit: F1, every value of both formats (248 of `float8e5`, 254 of `float8e4nv`) in `tl.full`
  and in `tl.where`: 0 differ; F2, 160 values one of the formats does not hold (`tl.full`; 41 in
  `tl.where`): every one rejected on the host, where `litF8` is `none`; F3, `±inf`, doubles whose
  `float32` overflows, `±nan` and NaN doubles with payloads, as `tl.constexpr` arguments, each in
  a process of its own: `float8e5` `0x7c`/`0xfc`, `0x7e`/`0xfe`, `0x7f`, and a NaN with lower
  `float32` bits rejected; `float8e4nv` all rejected: 0 differ; F4, the run-time paths (a store,
  `other=`, `.to`) against `FloatV.cvtSat .rn (litF32 c)`: 0 differ (`differences in all: 0`).
- **Constants whose meaning changed.** `TVA.FloatV.litF8` (new); `TVA.Operand.constIn?` (on the
  two fp8 dtypes: `litF8`, was `none`). Fact pages: `docs/SEMANTICS.md` ("Values", "Constants"),
  `docs/rules/num.md` ("Conversions into fp8"); no fact's statement changes.

## L91: CUDA graph capture: the exclusion's reason stated truly

- **What changed.** The exclusion is unchanged: no call made while a CUDA stream capture is in
  progress, on any stream, and no replay of a captured graph, is a launch a claim covers. Its
  account now says what happens: a call on the capturing stream is captured (it returns without
  an error, compiling the kernel first if needed, runs no program, and its programs run at each
  `graph.replay()` with the captured arguments, on the memory at the replay), and a call on
  another stream during the capture is not captured (it runs at once, as any launch, and no
  replay runs it). Corrected in every place that gave the old reason: `Rules.Compiler.Faithful`'s
  scope 9 (`lean/TVA/Rules/Compiler.lean`), `Launch.Valid`'s docstring
  (`lean/TVA/Sem/Launch.lean`), `lean/TVA/Claim.lean`'s module docstring,
  `docs/rules/compiler-faithful.md` (the summary, scope 9, checklist item 9), `docs/SEMANTICS.md`
  ("One launch", the scope's "no CUDA graph capture", the note on launch outcomes),
  `docs/exclusions.md` (the capture row), `docs/REVIEW_PROTOCOL.md` §6, `docs/REVIEWER_GUIDE.md`
  and `docs/PROVER_GUIDE.md` (item 9); new folder `evidence/host-round20`.
- **Why.** lr-17b: the reason given ("a call made while a stream capture is in progress … runs no
  program") is false for a call on another stream, which ran at once. Keep the exclusion and state
  the reason truly (the issue; user, 2026-10-28: L91 excluded).
- **Evidence.** [evidence/host-round20](../../../../evidence/host-round20/README.md) G0–G3
  (Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28): G0, the call on the capturing stream left the
  output at its sentinel after the capture, and the replay wrote `x + 1`; G1–G3, the call on
  another stream in the capture modes `global`, `thread_local` and `relaxed` wrote `x + 1` at
  once, and after a reset the replay left the sentinel. Round 16's
  `docs/library-review/round-16/lr-16b/work/graph_capture.py` (the replay's arguments and memory)
  stays cited.
- **Constants whose meaning changed.** None in meaning: `TVA.Rules.Compiler.Faithful` (scope 9)
  and `TVA.Launch.Valid`, docstrings only; the scope they exclude is the same. Fact pages:
  `docs/SEMANTICS.md`, `docs/exclusions.md`, `docs/rules/compiler-faithful.md`.

## L95: the facts section is read as the page renders it, or lists no fact

- **What changed.**
  - `tools/tvr.py`: new `facts_section(root)`, the section "Facts the library states" of
    `docs/rules/num.md` as `(lines, problems)`. The section may hold only the forms whose
    rendering a line reader knows: blank lines, table rows (lines that start with `|`), `###` to
    `######` headings, and lines of text that start with a letter, a digit that is not a list's
    number, `(`, `[`, a quote or one backtick, with a blank line (or the section's start or end)
    around each table. Anything else is a problem: a setext heading or a rule, a fence, an indented
    line, a block quote, a list, HTML (an HTML comment included), a text line against a table; so
    is a second heading of that name on the page, and a section heading inside a fence or an HTML
    comment. `facts_section_tables` reads the section through it; `stated_facts` returns `None`
    (the gate accepts no fact but `Compiler.Faithful`) when the section has a problem; the audit's
    `fact_page_problems` names each problem with its line.
  - `docs/REVIEW_PROTOCOL.md` §2 and §5, `docs/LIBRARY_REVIEW.md` check 5 and
    `docs/prompts/library-author.md` say so. `docs/rules/num.md` already holds only these forms
    (38 facts, no problem).
- **Why.** lr-18b (and lr-18a's `work/facts.py`): a setext heading made a "not citable" table the
  citable one with the audit silent, and a table inside an HTML comment or a fenced block was read
  as the facts table. Recognising each Markdown construct one by one would leave the next one
  open; an allowlist of the forms a line reader reads as Markdown renders them closes the class,
  fails closed, and costs the page nothing (the user's rule on gate holes: complete when cheap).
- **Evidence.** [evidence/tool-round20](../../../../evidence/tool-round20/README.md) Part L
  (2026-10-28): the tree's page gives 38 facts with no problem, and so does a page with text
  between the heading and the table; ten scratch pages, each with one form (the real table inside
  an HTML comment with a fake after it, a fenced fake table, the table indented, in a quote, in a
  list item, a setext heading before it, a rule, a text line against it, a second heading, the
  heading inside a fence), give no fact (`ExpF32 P` refused) and the audit names the page.
- **Constants whose meaning changed.** None (the tool and the protocol; the facts page lists the
  same 38 facts).

## L98: a NaN passed to a helper, settled under L103

- **What changed.** A NaN that reaches the kernel as a compile-time value has the pattern
  "Constants" gives when the helper copy the call uses, and the code the launch runs, are its own
  (L103's rule, below). `docs/SEMANTICS.md` "Constants" says so (it said a `tl.constexpr` NaN is
  excluded and a NaN passed otherwise is covered with its pattern); the section docstring "Python
  number constants" of `lean/TVA/Lang/Scalar.lean` points to the same rule; `docs/exclusions.md`'s
  round-19 row (a `tl.constexpr` NaN) is now a case of L103's row, which also names two helper
  calls whose NaNs share the copy's name. A NaN passed to a helper with no such earlier call is
  covered, with its own pattern.
- **Why.** lr-19a: `h(o, float('nan')); h(o + 1, -float('nan'))` stored `0x7fc00000` twice, since
  `mangle_fn` names each copy by the `repr` of its compile-time arguments and every NaN's is
  `nan`. ISSUES.md: one case of L103; settle it there.
- **Evidence.** [evidence/compile-keys](../../../../evidence/compile-keys/README.md) K5
  (`helper(nan, -nan)` and the reverse: the second call ran the first call's copy) and K2 (a
  launch with `-nan` after `nan` ran `nan`'s kernel), with round 19's
  `evidence/literals` "NaN constants" for the patterns themselves.
- **Constants whose meaning changed.** None (`FloatV.litF32` and the other builders are
  unchanged; their docstring section names the rule). Fact pages: `docs/SEMANTICS.md`,
  `docs/exclusions.md`.

## L101: round 14's wording: three evidence folders, fp8's `u` and `eta`, `replay_err`

- **What changed.**
  - `evidence/num-cvt8`: the README says which non-finite inputs `run.py` converts (`float32`'s
    `±inf` and its canonical NaN only), and a new `specials.py` (output `specials.out`) converts
    every non-finite pattern of each source format into both fp8 formats: `±inf` and all 16,777,214
    NaN patterns of `float32`, all 2,046 of `float16`, all 254 of `bfloat16`. Conclusion 2 says what
    each script covers; `Rules.Num.Sm90`'s section docstring on saturation names both scripts.
  - `evidence/num-dot/README.md`: the largest ratio "lies below `(K−1)/(2(K+1))` and within
    `3.1·10^-5` of it", with the exact values at `K = 8`, `512` and `1024`. The issue proposed
    "within `2·10^-5`", which holds up to `K = 512` (gap `1.5·10^-5`) but not at `K = 1024` (gap
    `3.05·10^-5`, `ieee_ratio.out`'s exact table), so the README gives the true bound.
  - `evidence/num-atomic-add`: new `sem_scope.py` (output `sem_scope.out`) runs the pointer atomic
    in all 12 combinations of `sem` and `scope`; the README has the section "Every `sem` and
    `scope`" and conclusion 1 says the fact covers them all. `AtomicAddF32`'s docstring
    (`lean/TVA/Rules/Num/Sm90.lean`) and its row and section in `docs/rules/num.md` name
    `atom.global.<scope>.<sem>.add.f32`, every `sem` and `scope` (they said
    `atom.global.gpu.….add.f32`, "any `sem`").
  - `lean/TVA/Math/Encoding.lean`: `Format.e4m3_u = 2^-4`, `Format.e4m3_eta = 2^-10`,
    `Format.e5m2_u = 2^-3`, `Format.e5m2_eta = 2^-17`, proved as `binary32_u` is; `docs/LIBRARY.md`'s
    `Encoding` row lists the constants each format has (`u` and `eta` of all six float formats,
    `maxVal` of five and `e2m1`, `binary32_emin`) instead of "the constants of each format".
  - `replay_err`'s docstring (`lean/TVA/Lemmas/SharedAcc.lean`) names its operands
    `L.filterMap Prod.snd` (it said "the operands `W`").
- **Why.** lr-14a and lr-14b's wording points: each statement was true but its folder did not show
  it, or the text named something that is not there. `FloatV.cvtSat_rn_err`'s `|c|·u + η` needs
  fp8's `u` and `eta` as numbers, which lr-14b derived by `rfl`; the library now states them.
- **Evidence.** Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28:
  [evidence/num-cvt8](../../../../evidence/num-cvt8/README.md) `specials.out`: every infinity gives
  the largest finite value of its sign (`0x7e`/`0xfe`, `0x7b`/`0xfb`) and every NaN `0x7f`, a NaN
  of both formats (`rows not as CvtSat says: 0`);
  [evidence/num-atomic-add](../../../../evidence/num-atomic-add/README.md) `sem_scope.out`: in all
  12 combinations, 74,991,578 pairs each, none outside the fact, the largest error `0.5000` of its
  bound, and every result bit-identical to `acq_rel` at `gpu`;
  [evidence/num-dot](../../../../evidence/num-dot/README.md) `ieee_ratio.out` (unchanged).
- **Constants whose meaning changed.** `TVA.Math.Float.Format.e4m3_u`, `e4m3_eta`, `e5m2_u`,
  `e5m2_eta` (new theorems); `TVA.Rules.Num.AtomicAddF32` and `TVA.replay_err`, docstrings only.
  Fact page: `docs/rules/num.md` (`AtomicAddF32`'s row and section; the fact's statement is the
  same).

## L102: a helper's `return`s and a runtime conditional expression, as the front end types them

- **What changed.**
  - A conditional expression on a runtime condition is stated (`docs/SEMANTICS.md`, "Order", the
    `x if c else y` bullet): each side becomes a tensor (`to_tensor`, `code_generator.py:998-1036`),
    a Python number a scalar tensor of its own type holding the constant Triton builds for it,
    another compile-time value is rejected on the host, and both sides must have one type or the
    kernel is rejected. "Order" no longer counts it as a compile-time value, and "Constants"'
    "everywhere else" lists its sides.
  - A helper's value is stated ("`return`", the paragraph "A helper's value":
    `decide_return_type`, `cast_to`, `handle_returns`, `code_generator.py:556-626`), and the
    subset admits only the helpers for which it is the value of the `return` the program takes
    ("The supported subset", `return`): one visited `return`; `return`s that all give tensors of one
    type; `return`s that all give one compile-time value of one Python type; and no path to the end
    of the body without a `return` when a visited `return` gives a value. Any other helper is
    outside the subset (L88: no in-scope helper has two visited `return`s), with a row in
    `docs/exclusions.md`. "Helpers are inlined" says the call's value is that `return`'s.
  - `docs/TRANSCRIPTION.md`: pitfalls "A runtime conditional expression" and "A helper's value";
    new folder `evidence/front-end-round20`.
- **Why.** lr-17a, lr-17b: with a runtime condition, `(-1 if c > 0 else -3) // 2` stored `0`,
  `x * _h_f(c)` on `bfloat16` was computed in `float32`, `return -7` / `return -9` gave
  `h(c) // 2 = -3`, and `if c > 0: return 5` returned `5` on every path, while SEMANTICS read as
  Python's values. The conditional expression is cheap to state truly; the helper forms beyond the
  admitted ones are not in use, so they are excluded (the issue's two options; the user's L88 rule).
- **Evidence.** [evidence/front-end-round20](../../../../evidence/front-end-round20/README.md)
  (Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28; `c = 5` and `-5`, never `1`, which Triton passes as a
  compile-time constant): E1a `0`, `-1` (Python: `-1`, `-2`); E1b, every finite `bfloat16` `x`
  (65,280): 0 differ from the `float32` product, 13,076 and 13,102 from the `bfloat16` one; E1c
  both mixed forms rejected on the host; E1d the taken side; E2a `-3`, `-4` (Python `-4`, `-5`);
  E2b `16777217` on both paths; E2c `5` on both paths; E2d and E2e, the admitted forms, the taken
  `return`'s value.
- **Constants whose meaning changed.** None in Lean (the rules are the front end's, in
  `docs/SEMANTICS.md`). Fact pages: `docs/SEMANTICS.md`, `docs/exclusions.md`.

## L103: compile-time values Triton's keys do not tell apart, as one family

- **What changed.** Two places where Triton keys code by its compile-time values, each settled for
  the whole family:
  - **A launch's compiled kernel**: new scope 11 of `Rules.Compiler.Faithful` (*The code compiled
    for the launch's own compile-time values*, `lean/TVA/Rules/Compiler.lean`): the rule covers a
    launch only when no earlier launch or warm-up of the kernel (an autotuner's benchmark runs
    included), in the process or through the same cache directory from any process, had
    compile-time values that differ from its own but share Triton's key: in the process, values
    `==` equates (`1`, `1.0`, `True`; `0`, `0.0`, `-0.0`, `False`) and then one `str` text (every
    NaN is `nan`; `compute_cache_key`, `jit.py:592-614`); on disk, each value's `str` joined with
    `-` (`1` and `"1"`, `None` and `"None"`, `2.5` and `"2.5"`, every NaN and `"nan"`,
    `("a-b", "c")` and `("a", "b-c")`; `ASTSource.hash`, `compiler.py:71-76`), which a launch also
    reaches within its own process. In `docs/rules/compiler-faithful.md` (scope 11 and checklist
    item 11), `docs/REVIEWER_GUIDE.md` and `docs/PROVER_GUIDE.md` (item 11: the reviewer checks
    the host), `docs/SEMANTICS.md` ("Launches"), `lean/TVA/Claim.lean` (module docstring,
    `Claim.Behaviour`).
  - **A helper's copies**: `docs/SEMANTICS.md`, the new section "Compile-time values Triton keys
    apart": the front end makes one copy of a helper per `mangle_fn` name (the `repr` of each
    compile-time argument, `'` written `_sq_`, `[` and `]` written `_`; `code_generator.py:32-41`,
    `core.py:196-201`), and a later call with the same name uses the first call's copy. Two calls
    of one helper whose compile-time arguments differ but share the name are outside the subset
    ("The supported subset", twice).
  - `docs/exclusions.md`: one row for the family (round 19's NaN row is a case of it); the section
    docstring "Python number constants" of `lean/TVA/Lang/Scalar.lean`; new folder
    `evidence/compile-keys`.
- **Why.** lr-17a and lr-19a: `C = 1.0` after `C = 1` ran the integer kernel, `"1"` after `1` on one
  cache folder ran `1`'s code, and two NaNs or `"x[y"` and `"x_y"` passed to a helper ran the first
  call's body. Narrowing the admitted values would have to forbid floats, booleans, strings and
  `None` wherever another value of the parameter collides, and in-scope kernels pass such values
  (tutorial 03's string `ACTIVATION`, round 10's coverage matrix); a scope
  condition with the reviewer's host check is the issue's other option and leaves every value in
  use covered. Helper collisions happen inside one compilation, in the kernel's own text, and no
  in-scope helper call has them, so the subset excludes them (L88).
- **Evidence.** [evidence/compile-keys](../../../../evidence/compile-keys/README.md) (Triton 3.8.0,
  H200 NVL, GPU 0, 2026-10-28; each sequence in processes of its own, against each value launched
  alone): K1, the 8 in-process pairs, both orders, all ran the earlier value's code (`C = 1.0`
  after `C = 1` stored `16777217`); K2, two NaNs, both orders; K3, 7 disk pairs over two processes
  and `1` then `"1"` in one process; K4, `("a-b", "c")` then `("a", "b-c")` stored `31`, not `13`;
  K5, the helper's NaNs and strings, both orders, ran the first copy, while `±inf`, `1`/`1.0`/`True`
  and `"a"`/`"x_y"` did not collide: 23 launches ran code compiled for other values than their own.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: its scope gains condition 11
  (the definition is unchanged; the rule's text, which a claim cites, covers fewer launches).
  `TVA.Claim.Behaviour`, docstring only. Fact pages: `docs/SEMANTICS.md`, `docs/exclusions.md`,
  `docs/rules/compiler-faithful.md`.

## L104: Triton and the kernel's module as installed and written: a stated limit and the reviewer's check

- **What changed.** New scope 10 of `Rules.Compiler.Faithful` (*Triton and the kernel's module as
  installed and written*, `lean/TVA/Rules/Compiler.lean`): Triton's files are 3.8.0's as installed,
  unmodified, and no code of the process (the kernel's module, another module, a library it
  imports), at any time before the launch ends, changes a module, class or object of Triton
  (`tl.exp = tl.log`, `tl.arange = …`, a `JITFunction`'s `fn`, `src` or `noinline`, a call of its
  `_unsafe_update_src`), other than the `JITFunction`'s `debug` (scope 4), or binds a name of the
  kernel's module other than as the module's own statements do (`kmod.double = kmod.triple` in
  another module, an assignment into `kernel.fn.__globals__`). The library cannot see code outside
  the kernel's file: the kernel reviewer checks the host code for it. Stated in
  `docs/rules/compiler-faithful.md` (scope 10, checklist item 10), `docs/REVIEWER_GUIDE.md` (host
  checks, item 10) and `docs/PROVER_GUIDE.md`, `docs/SEMANTICS.md` ("The supported subset", Names:
  no code outside the module binds a name or changes Triton; "Launches", the scope list),
  `docs/exclusions.md` (a row), `docs/REVIEW_PROTOCOL.md` §7 (what the map check does not see),
  `lean/TVA/Claim.lean` (module docstring, `Claim.Behaviour`).
- **Why.** lr-17a, lr-18a, lr-18b: host code in another module changed what a supported kernel
  computes (`3·x` for `2·x`, `tl.exp` computing a logarithm, `_unsafe_update_src` giving `5·x`),
  and neither SEMANTICS nor the scope excluded it. By the user's decision on a malicious prover
  (under L36), a stated limit and the reviewer's check suffice; no gate machinery is added.
- **Evidence.** [evidence/host-round20](../../../../evidence/host-round20/README.md) M0–M4
  (Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28; the kernel module imported from the script, every
  line of it in the subset): M0 `2·x`; M1 `kmod.double = kmod.triple` and M2 the same through
  `fn.__globals__`: `3·x`; M3 `tl.exp = tl.log`: `log x`; M4 `_unsafe_update_src`: `5·x`.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: its scope gains condition 10
  (the definition is unchanged). `TVA.Claim.Behaviour`, docstring only. Fact pages:
  `docs/SEMANTICS.md`, `docs/exclusions.md`, `docs/rules/compiler-faithful.md`.

## L105: `kernel.debug` gives `Launch.debug`; `JITFunction.preload` is outside the scope

- **What changed.**
  - `Launch.debug`'s docstring (`lean/TVA/Sem/Launch.lean`): on when the call that compiles the
    kernel gives `debug=True` (the launch, `warmup`, `triton.compile`'s options, the chosen
    `triton.Config`); when that call gives none, when the kernel's `JITFunction` attribute `debug`
    is true, which the decorator sets and Python code may assign (`kernel.debug = True`;
    `JITFunction.run` takes `kwargs.get("debug", self.debug)`, `jit.py:727`); and always when
    `TRITON_DEBUG=1` or the knob asks for it. `False` otherwise (the old text omitted the
    attribute set from Python, so its "False otherwise" was false).
  - Scope 4 of `Rules.Compiler.Faithful`: the attribute is one more route to `Launch.debug`, which
    the rule covers only when false; and the rule covers no launch of a kernel `JITFunction.preload`
    was called on in the process (`jit.py:823-875`: it installs code compiled with the options its
    data names, under the key its data names). The same in `docs/rules/compiler-faithful.md`
    (scope 4, checklist item 4), `docs/SEMANTICS.md` ("Launches": the options bullet and
    `debug`), `docs/exclusions.md` (the `debug` row; `preload` in the options row),
    `docs/REVIEWER_GUIDE.md` and `docs/PROVER_GUIDE.md` (host check, item 4), and
    `lean/TVA/Claim.lean`'s module docstring.
- **Why.** lr-18b: `kernel.debug = True` made an `int32` overflow trap, and after `preload` of
  code compiled with `enable_reflect_ftz=False` a plain launch ran that code. The attribute is a
  value `Launch.debug` can record, so it is stated; `preload` installs code with options nothing at
  the launch names, and no in-scope host calls it, so it is excluded (the issue's two options).
- **Evidence.** [evidence/host-round20](../../../../evidence/host-round20/README.md) (Triton 3.8.0,
  H200 NVL, GPU 0, 2026-10-28): D0, an `int32` `2^31 − 1 + 1` wraps to `-2147483648`; D1, after
  `kmod.add_i32.debug = True`, with no keyword, decorator argument, environment variable or knob,
  the same launch hit the device assert `int32 overflow detected for operation add`; P0, a plain
  `tl.math.exp2(-130)` gives `0` (the model, `enable_reflect_ftz=True`); P1, in a fresh process
  after `preload` of a first process's `enable_reflect_ftz=False` compile, a plain launch gave
  `2^-130`.
- **Constants whose meaning changed.** `TVA.Launch.debug` (docstring: which calls set it; the field
  is unchanged); `TVA.Rules.Compiler.Faithful` (scope 4 excludes launches after a `preload`; the
  definition is unchanged). Fact pages: `docs/SEMANTICS.md`, `docs/exclusions.md`,
  `docs/rules/compiler-faithful.md`.

## L106: an `Assumptions` that is not a structure fails `shape`

- **What changed.**
  - `tools/claim_reader.lean.in`: before it reads the fields, the reader requires
    `isStructure env (ns ++ `Assumptions)`, and otherwise stops with the error
    `TVR: Assumptions must be a structure (library issue L106)`.
  - `tools/tvr.py`: the gate's shape decision is the new function `reader_result(code, stdout,
    stderr)`, used by `tvr check`: the reader passes only when it exits with `0`, prints one result
    line, and no line of its output says `PANIC`; a panic is named ("the claim reader panicked
    (library issue L106)"). The shape step's text says "Assumptions is a structure with exactly the
    parameters …".
  - `docs/REVIEW_PROTOCOL.md` §5 (what the reader requires) and §8 (the `shape` row: a structure;
    a reader that panics or exits non-zero fails), `docs/REVIEWER_GUIDE.md` §4.
- **Why.** lr-18a: an `inductive Assumptions` whose constructor holds the specification passed
  `shape`, `axioms`, `usage`, `namespace` and `kinds`: the reader panicked in
  `getStructureFields`, returned no field and still exited `0`, so the gate showed a claim with no
  assumption. Both checks are cheap and complete (L36): the reader refuses the declaration, and the
  gate refuses any reader output that went on past an internal error.
- **Evidence.** [evidence/tool-round20](../../../../evidence/tool-round20/README.md) Part S
  (2026-10-28; kernels compiled with `lean -o` into a scratch folder): a structure (control) passes
  with its field `spec`; lr-18a's one-constructor `inductive` is refused by the tree's reader
  (exit 1, the L106 error), while the round's-start reader (the template without the check) prints
  `PANIC at Lean.getStructureInfo … structure expected`, exits `0` with no field, which the
  round's-start decision passed and `reader_result` fails; a `def Assumptions` fails with both.
- **Constants whose meaning changed.** None (the tool and the protocol).

## L107: the host is part of the target: x86-64, CPython 3.13

- **What changed.** The host is named wherever the target is: `docs/SEMANTICS.md` (a new
  paragraph at the top: the facts are about Triton 3.8.0 on an H200 driven from an x86-64 host
  running CPython 3.13, and which values are the host's, a NaN passed as a `tl.float16` argument
  and the sign of `float('inf') - float('inf')`; "Constants" and "Scalar arguments" say which of
  their values are the host's; "Launches", the target bullet); `docs/exclusions.md` (the targets
  row: another host); `Rules.Compiler.Faithful`'s scope 7 (`lean/TVA/Rules/Compiler.lean`) and
  `docs/rules/compiler-faithful.md` (scope 7, checklist item 7); `docs/REVIEWER_GUIDE.md` and
  `docs/PROVER_GUIDE.md` (item 7); `docs/TRANSCRIPTION.md` (the host pitfall);
  `lean/TVA/Claim.lean` (module docstring). `pyproject.toml` pins `requires-python =
  ">=3.13,<3.14"` (was `">=3.10"`), with a comment naming the host.
- **Why.** lr-19b: under CPython 3.14, which `pyproject.toml` allowed, a `tl.float16` NaN argument
  keeps payload bits that 3.13 drops (`0x7ff4…` became `0x7d00`, not `0x7e00`), and the sign of
  `inf - inf` is x86-64's default NaN; a statement that holds only on one host must name it.
- **Evidence.** TVA's evidence ran on x86-64 under CPython 3.13.9 (the outputs of this round's
  folders print the version); the 3.14 difference is lr-19b's
  `docs/library-review/round-19/lr-19b/work/host_pack2.py`, cited in scope 7. No new device run:
  the fix states the host the existing evidence ran on.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful`: scope 7 names the host (the
  definition is unchanged; a launch from another host is outside the rule's text). Fact pages:
  `docs/SEMANTICS.md`, `docs/exclusions.md`, `docs/rules/compiler-faithful.md`.

## L108: wording from the reviews of rounds 17 to 19

- **What changed.** Each point of the issue:
  - `tl.cdiv` (`docs/SEMANTICS.md`): `div - 1` is Python's, at compile time, for a compile-time
    `div`, and a wrapping tensor `-` otherwise (lr-17a).
  - The two general sentences of SEMANTICS name their exceptions (lr-17b): "what runs is what it
    evaluates" except the first visit of a loop body and the code visited after an `if` that
    returns on the instance (L89); "every argument is evaluated" except `tl.static_assert`'s
    `msg`, evaluated only when the assertion fails (`code_generator.py:1617-1637`).
  - `evidence/tool-round18/run.py`: `pin_ledger` keeps a file a later `freeze` event records (it
    was there before the freeze), so Parts R and V re-run after vec-add's freeze; the README says
    `out.txt` was made in round 18, before the freeze, and the re-run is kept as
    `rerun-round20.txt` (lr-18a, lr-18b).
  - Non-normalised paths (lr-18a): `tools/tvr.py`'s new `normal_path`; `is_work_file` requires it,
    so `freeze_problem` refuses `<folder>/work/../filing-2.json`, and `ledger_problems` refuses a
    recorded path or `dir` that is not normalised, in any event of either ledger
    (`docs/REVIEW_PROTOCOL.md` §3). Every path the tree's ledgers record is normalised.
  - Unescaped `|` in code spans: 9 rows of `docs/rules/num.md`'s tables (lr-18a's 8 and the
    in-range conversions' row of the facts table, 28 bars in all) and 2 rows of `docs/LIBRARY.md`
    (line 40 and one more) now write `\|`; the gate's reader splits a row at an unescaped `|` only,
    and lists the same 38 facts.
  - "Within half an ulp": `docs/rules/num.md` and `Format.sqrtMag_rn_spec` (lr-19b), and every other
    place that called `u·|y| + η` half an ulp, now say "one rounding's bound" or "at least half an
    ulp, and up to about a whole one": `Math/Rounding.lean` (`roundRat_rn_err_of_finite`,
    `sqrtMag_rn_spec`), `Lang/Ops.lean` (`AddRel`), `Lemmas/Consistency.lean` (`FloatV.exists_nearest`,
    `DType.val?_ofRat_float`), `Lemmas/FusedErr.lean` (`RoundsTo.err`), `Lemmas/Sm90.lean` (module
    docstring, `of_const`), `Rules/Num.lean` (module docstring, `CvtAcc`, `DivRn.divAcc`), and in
    `docs/LIBRARY.md`.
  - `FloatV.nan`'s docstring (lr-19b): `tl.div_rn` and `tl.sqrt_rn` are for `float32` only, the one
    format Triton accepts them on, and `float64` `tl.maximum`/`tl.minimum` with
    `propagate_nan=ALL` give `0x7ff8000000000000`.
  - `lean/TVA/Lemmas/Int.lean` (lr-19b): new `IntV.le_toInt` (in `Lemmas/SharedAcc.lean`, beside
    `IntV.lt_toInt`): `x.le y = decide (x.toInt ≤ y.toInt)` for either signedness, the lemma the
    docstring promised; the docstring names both. `docs/LIBRARY.md`'s `Block` row names
    `addrs_block` (and `accBytes_block`, `img_block`), the `Int` and `SharedAcc` rows
    `IntV.le_toInt`.
  - `evidence/float-layer/README.md` conclusion 4 (round 19's author, lr-19a, lr-19b): "The library
    states the first (the `CvtSat…` facts …), not the second" (it said "neither yet").
- **Why.** The reviews' wording points: each text was false or incomplete as written.
- **Evidence.** None needed for the wording; for the tool points,
  [evidence/tool-round20](../../../../evidence/tool-round20/README.md) Part F (a freeze recording
  `work/../filing.json`, `work/./Cover.lean` or `//work/Cover.lean` is refused, a ledger recording
  `work/../filing.json` is named, the tree's vec-add and library ledgers pass) and
  `evidence/tool-round18/rerun-round20.txt` (Part R as in `out.txt`; Part V freezes the four
  round-1 work files again).
- **Constants whose meaning changed.** `TVA.IntV.le_toInt` (new). Docstrings only:
  `TVA.FloatV.nan`, `TVA.Math.Float.Format.roundRat_rn_err_of_finite`, `Format.sqrtMag_rn_spec`,
  `TVA.AddRel`, `TVA.FloatV.exists_nearest`, `TVA.DType.val?_ofRat_float`, `TVA.RoundsTo.err`,
  `TVA.Rules.Num.CvtAcc`, `TVA.Rules.Num.DivRn.divAcc`. Fact pages: `docs/SEMANTICS.md`,
  `docs/rules/num.md` (no fact's statement changes).

