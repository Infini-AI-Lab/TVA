# Library review, round 13, by lr-13b

## Verdicts

- **L7** ACCEPT — The release check now runs `/usr/bin/git`, `/usr/bin/bwrap` and `/usr/bin/find`.
  `system_tool` accepts each only when it resolves to an executable under /usr that the user cannot
  write and no regular user or group owns. Git runs with `PATH=/usr/bin:/bin` and no configuration.
  `inputRev` must be the toolchain's release tag (checked as a tag) or `master` (as a branch). The
  audit checks this, and so does the packages step, before it fetches anything. My re-run of
  evidence/tool-round13 Part N matches out.txt line for line:
  - with wrappers first on PATH, no wrapper ran;
  - git is 2.52.0, and lr-10a's feature branch is refused;
  - Mathlib's pin is `v4.35.0`'s commit, reachable from `master`;
  - `master` asked for as a tag is refused.

  The coordinator's release check of the merged tree passed `packages` and `sandbox`.
- **L50** ACCEPT — `source_functions` uses Python's parser to list every `def`, `async def` and
  `lambda`, at any depth. `map_problems` requires each to lie in the kernel lines, in the host
  lines or, for a `def`, in `source.unused`. A function under a `jit` decorator may not be in the
  host lines. So no spelling of `jit` escapes. claim.md §2 lists every function with its lines and
  class, so a reviewer also sees two functions of one name that are both named unused. My re-run
  of U1 matches.

  REVIEWER_GUIDE §1, claim.md §2 and the kernel reviewer's prompt ask the reviewer which functions
  the launch runs at compile time: a `triton.constexpr_function`, or a plain Python function the
  front end calls. SEMANTICS.md states no such fact, so the reviewer cannot judge such a function
  by it. I filed this below as a gap; it is not a defect of the map check.
- **L52** ACCEPT — What the fix does:
  - `kernel_event_problems` derives every `file` and `response` event from its recorded file, with
    the same functions `tvr file` and `tvr respond` use;
  - a file with two sha256s, or a library folder recorded twice, fails;
  - `tvr record` refuses `--no-lint` after round 7 (`PRE_LEDGER_ROUNDS`).

  My re-run of U2–U4 matches out.txt except where the merged tree differs. Round 11's two reviewer
  folders, in progress beside this round, are named as unrecorded. The library ledger has one more
  line, so the event numbers and the earlier ledger digest shift by one line. The `records` step of
  the coordinator's release check passed. On the coordinator's
  question: protocol §3 states the limit the user decided ("trusted as the coordinator keeps
  them"). LIBRARY_REVIEW.md "Rounds" and protocol §10 say the lines "rest on the git history". TVA
  cannot promise that, since the coordinator does not commit at each recording. Filed below as
  wording.
- **L62** ACCEPT — Under `∀`/`→` binders, `rule_fact_problem` accepts only four forms:
  `Compiler.Faithful`; a `TVA.Rules` constant of type `Platform → Prop` applied to `P` alone; a
  `…Total` rule of `TVA.Rules.Num`; or `DivRn` whose format argument is a `TVA` constant whose name
  ends in `binary64`. Part K re-ran identical: 25 `Platform → Prop` facts, and lr-9a's `MathAcc`,
  `DivAcc`, `DotF8` and `DivRn P .binary32` are refused.

  The `binary64` test carries weight. `Platform.sm90ref` satisfies `DivRn` for every format
  (`lean/TVA/Lemmas/Sm90.lean:183`), so without the test, `/` on `float32` could be cited as IEEE.
  The test depends on the name. If round 12 renames that format constant, `DivRn P .<new>` is
  refused until `rule_fact_problem` is updated: it fails closed. The `Sm90` facts are found by
  their type. PROVER_GUIDE §3 and its table, and REVIEWER_GUIDE §4, cite facts that exist, by name,
  and `Platform.sm90ref`.
- **L63** ACCEPT — The kernel's own constants are now every constant not rooted at `TVA`, other
  than `RULE_ARGS`. The namespace check exempts only names with a numeric component
  (`TVR.leanMade`). Part K re-ran identical:
  - `private def tol`, `Kernels._tol` and `_root_._tol` are each named;
  - the instance `_hdiv` is the kernel's, and Lean's `_sizeOf_inst` is not;
  - the control kernel has no namespace problem.
- **L68** ACCEPT — The reader writes `TVR.canonExpr` of each field's type, with `K` and `P` bound,
  and `item_digests` adds its sha256 to that field's item. Part D re-ran identical. Each of
  lr-10a's three edits changes exactly the edited assumption's digest, and a proof edit changes
  none.
- **L69** ACCEPT — `gate.json` records `build`: the sha256 of `lean-toolchain`, `lakefile.toml`,
  `lake-manifest.json` and every library file the kernel's Lean imports, transitively.
  `gate_staleness` and the release check's `gate_mismatch` compare it. U5 re-ran identical.
- **L70** ACCEPT — The audit now reads every file under `kernels/` outside a kernel's folder.
  `citation_problems` refuses a missing `docs/…` or `evidence/…` path, and a broken relative
  Markdown link, everywhere but the records, the evidence and the pinned files. Protocol §2 states
  that scope. U6 re-ran identical, and the tree cites no missing file. The `audit` step of the
  coordinator's release check passed.

## New issues

- **NEW** soundness — TRANSCRIPTION.md §2, row "`tl.where(keep, x, 0.0)` with `keep` an `int32`
  tensor", says "`Tensor.where_` with the constant in `x`'s dtype". That holds only for a float
  `x`, as in tutorial 04. For an integer `x`, SEMANTICS.md ("Constants": a `float` with an integer
  tensor computes in the number's own type) and `whereTyping?` give `float32`. On the device the
  result is `fp32`, and stored back into an `int32` buffer, 16777217 became 16777216
  (`work/fe_probes.py` W1; W2 is the float control). A model that follows the row keeps the
  integer values. **Fix:** limit the row to a float `x`, and give the integer case (`x` converted
  to `float32`), or say "in the dtype `whereTyping?` gives".
- **NEW** soundness — SEMANTICS.md, "How the front end evaluates the source", "A name assigned a
  Python number holds a tensor" ends with "unless the target is annotated `: tl.constexpr`". Under
  either reading of "annotated", some case is false (`code_generator.py:694-760`, which keeps
  `pending_annotations`):
  - **Bare annotation.** After a bare `e: tl.constexpr` line, the next assignment keeps the Python
    value, though its own statement has no annotation. `e = 1e-7` then `x * e` with a `bfloat16`
    `x` computes in `bfloat16` and gives 0 (C4). Without the bare line, the product is in
    `float32` (C5).
  - **Annotated earlier.** A later assignment to a name annotated earlier, or to a `tl.constexpr`
    parameter, makes a tensor. `c: tl.constexpr = -7; c += 0; c // 2` gives `-3` (C2, against `-4`
    in C3). After `BLOCK = BLOCK * 1`, `tl.arange(0, BLOCK)` is rejected on the host (C1).
  - **Other annotations.** An annotation other than `tl.constexpr` is ignored.
    `a: tl.int64 = 2**31 - 1` is an `int32` tensor, and `a + 1` wraps to `-2^31` (C6; C7 is the
    control).

  **Fix:** state the rule exactly. An assignment keeps the Python value only when its own
  statement annotates the target `: tl.constexpr`, or when it is the first assignment to the name
  after a bare `name: tl.constexpr`. Every other assignment makes a tensor, including one to a
  `tl.constexpr` parameter or to a name annotated before. Another annotation does not change the
  type. Add the pitfall to TRANSCRIPTION.md §3.
- **NEW** soundness — `for i in range(e)` with a single argument evaluates `e` twice.
  `code_generator.py:1206` visits every argument, and `:1242` visits the single argument again for
  the upper bound. So an operation in `e` runs twice in every program:
  - `range(tl.atomic_add(c, 1) * 0)` ran 16 atomics in 8 programs (L4);
  - `range(0, …)` ran 8 (L5), and so did `tl.range(…)` with one argument (L6).

  SEMANTICS.md ("Order": every part of an expression is evaluated; "Loops") and the TRANSCRIPTION.md
  row "`for i in range(lo, hi, step)`" imply one evaluation. **Fix:** state the double evaluation
  in "Loops" and "Order", and in that row: the model writes an operation of `range`'s single
  argument twice, before the loop.
- **NEW** gap — SEMANTICS.md gives no meaning to three kinds of compile-time evaluation that
  kernels and the guidance rely on:
  - **Constexpr and plain Python functions.** Python runs a `@triton.constexpr_function` at compile
    time (`triton-3.8.0-jit.py:1153-1169`), and also any plain Python function reached through a
    module (`code_generator.py:1444-1446`). Their arguments are evaluated first: a constexpr
    function unwraps constexprs, and a plain function gets the front end's objects. The result is a
    compile-time value: `tl.arange(0, twice(HALF))` ran (F1). An
    arithmetic operation on a tensor argument inside one is rejected on the host (F2). L50's
    REVIEWER_GUIDE §1, claim.md §2 and the kernel reviewer's prompt rely on this.
  - **Helper defaults.** A helper parameter's default value is bound and kept as a compile-time
    value (`call_JitFunction`, `:1358-1365`). With `e=1e-7`, `h(x)` computes as `h(x, 1e-7)`: a
    `bfloat16` product of 0 (D1, D2). SEMANTICS.md speaks only of "a Python number passed directly".
  - **Module globals.** Only some globals are visible (`:383-418`): a `tl.constexpr(…)` global is a
    compile-time value (G1), and a plain number is rejected on the host (G2).

  **Fix:** state these in "How the front end evaluates the source".
- **NEW** wording — LIBRARY_REVIEW.md "Rounds" step 4 and protocol §10 say that a ledger's lines
  "rest on the git history, or on the ledger's digest that `release.json` keeps". The user decided
  on no commit at each recording (ISSUES.md, under L36 and L52), so TVA cannot promise this. The
  git history holds a ledger only as of the commits the coordinator makes, and `release.json` only
  if someone keeps it outside the tree. Protocol §3 says it as decided ("trusted as the coordinator
  keeps them"), and LIBRARY_REVIEW.md omits that. The round's notes also ask the coordinator to
  commit after each recording, against the decision. **Fix:** say in both places that the lines are
  trusted as the coordinator keeps them, and that a commit or a kept `release.json` shows only
  changes made after it.
- **NEW** wording — Seven small points in the transcription pages:
  - (a) TRANSCRIPTION.md §3 "Python's builtins" leaves out `range`, then says that any other builtin
    is rejected. SEMANTICS.md lists `range` among the visible builtins (`code_generator.py:357-360`).
  - (b) §3 "A scalar argument's type" says "at or above `2^31` an `int64`". An argument below
    `-2^31` is also `int64`, and one from `2^63` is `uint64`.
  - (c) The §2 row "`lo, hi = 0, N`" says `hi` is an `int32` scalar tensor. It holds `N`'s own type,
    which is `int64`, `uint32` or `uint64` for a large `N`.
  - (d) The §2 row "`x * w + b`" ends with "then the `float32` operations". The addition the product
    feeds is `tl.mul_add`.
  - (e) §5 says the gate checks only line coverage and the model text. Since L50 it also checks
    that every function is classed.
  - (f) SEMANTICS.md's opening says "What runs on the device is what it evaluates". The loop body's
    dry visit is evaluated and its operations are discarded: an atomic in a zero-trip `for` or
    `while` body ran in no program (L1, L3), and in a one-trip body it ran once (L2). Only the
    marks of its hints stay.
  - (g) SEMANTICS.md's `tl.load` does not say that `other` is converted to the pointer's dtype
    (`semantic.py:1043-1045`), as `tl.store` says of its value. With a `float16` pointer,
    `other=1 + 2^-11 + 2^-30` came out `0x3c00`: two roundings, through `float32`, as "Constants"
    implies (O1).

## Re-ran

- `evidence/tool-round13/run.py` in full, in an empty scratch folder, with the network (Parts K,
  D, U1–U6 and N): `work/tool-round13.out`. It is identical to out.txt except U4. There, round 11's
  two reviewer folders (in progress) are named as unrecorded, and the event numbers and the
  earlier ledger digest shift by the one ledger line added since.
- New: `work/fe_probes.py`, 23 cases (W1–W2, C1–C7, D1–D2, G1–G2, R1, L1–L6, O1, F1–F2), each in its
  own process with a fresh Triton cache, on GPU 0: Triton 3.8.0, torch 2.11.0+cu130, NVIDIA H200 NVL,
  2026-10-28. Output: `work/fe_probes.out`. R1 (`if FLAG: return` with a constexpr `FLAG`
  ends the program) agrees with SEMANTICS.md.
- Read against `primitives/`:
  - code_generator.py: name lookup and builtins, assignments, `visit_For`, `visit_While`,
    `_find_carries`, `call_JitFunction`, `call_Function`, `visit_BoolOp` and the static executors;
  - semantic.py: `to_tensor`, `binary_op_type_checking_impl`, `cast` and `load`;
  - jit.py: `ConstexprFunction`;
  - Promote.lean: `whereTyping?` and `computationType?`.

  I compared every row and pitfall of TRANSCRIPTION.md §2–§4 with SEMANTICS.md and these sources.
- The coordinator's release check of the merged tree: I read its output and did not re-run it.
  8 of 10 steps passed: shipped, packages, sandbox, build, library, records, the lint and the
  audit. The two failures are the expected ones: vec_add_f32's gate (its proof does not build) and
  the pages.

## Not checked

- I did not re-run the release check (lr-13a may), and I did not run `tvr check`.
- I did not review `lean/TVA/Math` or the numeric lemmas: round 12 replaces them.
- I did not survey the gate's code beyond the parts each fix changed: that area is lr-13a's.
