# Library review, round 15, by lr-15b

## Verdicts

- **L58** ACCEPT — SEMANTICS "The contiguity the analysis infers" and the hint section of
  `TVA.Lang.Ops` now give a value that carries a `max_contiguous` (alone or in the pair) the hinted
  `C`, whatever operation defines it, and start the list's forms from it. That is what the analysis
  does. `AxisInfo.cpp:1244-1259` replaces the computed contiguity by the hint on any operation, and
  the add visitor (`:265-279`) gives `x + y` with `y` constant `gcd(contig(x), const(y))`, which is
  `C`. evidence/hint-hinted-bound G1-G6 re-ran identical: 16, 4, 8, 16, 4 and 4 bytes, values
  correct.

  I also checked that the list is still a lower bound when other hints are present. A `max_constancy`
  hint cannot lower the constancy of the constant operands the list names: a splat, a Python number
  and an implicit broadcast are built by the front end after any hint the user writes. A
  `multiple_of` alone changes only divisibility (G4). TRANSCRIPTION's hint pitfall says the same as
  SEMANTICS.
- **L61** ACCEPT — Item 2 is now settled by exclusion, and the text is true:
  - SEMANTICS "A runtime `if`" limits its merge rules to names new to the `if` and names holding a
    tensor before it;
  - it says a `tl.constexpr` there is rejected on the host. A `tl.constexpr` is a `base_value`, so
    it reaches the type check and fails it;
  - it says `None` and a dtype are not merged. `code_generator.py:858-861` skips every live-in that
    is not a `base_value`.

  The subset's "Names" item and exclusions row 53 exclude the rest. That is conservative for tuples
  and descriptors, which the front end does merge. lr-11a's `if_compile_time_livein.py` I1-I6 re-ran
  identical. In my scan (`work/subset_scan2.py`), no in-scope kernel assigns a name holding a
  non-tensor in a runtime `if`: tutorial 10's `output_dtype` chain is on a `tl.constexpr`.
- **L76** ACCEPT — With `round_f32_to_tf32` set, `nvidia-driver.py:231-232` builds the TMA
  descriptor with `TMA_TF32`. The exclusion is named in:
  - scope 8 of `Faithful`'s docstring and of its rule page;
  - SEMANTICS "Launches" and exclusions row 50;
  - REVIEWER_GUIDE's host check, TRANSCRIPTION's host pitfall and `TVA.Claim`'s scope summary.

  evidence/desc-tf32 T1-T2 re-ran identical. Its README gives the version, the GPU and the date.
- **L77** ACCEPT — In `compiler.py:233-240`, any source that is not an `ASTSource` must be a path,
  and it becomes an `IRSource` (`:87-112`), which replaces the kernel's source. Scope 6 (docstring
  and rule page), SEMANTICS "Launches", exclusions row 39, REVIEWER_GUIDE and TRANSCRIPTION now cover
  only an `ASTSource` of the kernel's `@triton.jit` function. There is no device question.
- **L78** ACCEPT — `Heuristics.run` (`autotuner.py:480-483`) writes each heuristic into the keywords
  before the JIT's `run` reads them. The kernel is now excluded in:
  - scope 4 and the `Launch.fusion` docstring;
  - SEMANTICS "One launch" and "Launches", and the subset's excluded forms;
  - exclusions rows 39 and 49;
  - REVIEWER_GUIDE, TRANSCRIPTION and `TVA.Claim`.

  lr-11b's `heur_options.py` H1-H5 re-ran identical. No in-scope tutorial uses `@triton.heuristics`.
- **L80** ACCEPT — "Helpers are inlined" matches `call_JitFunction`. It binds the call with
  `signature.bind`, applies the defaults, wraps each argument with `normalize_value`
  (`:1358-1365`), and binds the parameters from the prototype (`:657-663`). No step reads a
  parameter's annotation. front-end-subset D1-D4 and P1-P2 re-ran identical.

  Constexpr functions, plain Python calls and other module-level values are excluded (the subset's
  "Functions" and "Names" items, exclusions row 51). My scan found none in the in-scope kernels:
  their only free names are `tl`, the module's helpers, `float`, `isinstance`, `min` and `range`.
  One wording point below: a string default makes the helper rejected.
- **L83** ACCEPT — The three `tl.where` rows of TRANSCRIPTION §2 give `whereTyping?`'s dtype.
  `semantic.where` (`:1638-1652`) types `x` and `y` with `binary_op_type_checking_impl`
  (`:175-210`), which gives:
  - an integer `x` with `0.0`: `float32`, by rule 2 of `computation_type_impl`, with `x` cast;
  - an integer `x` with `0`: `x`'s dtype;
  - two Python numbers: `float32`.

  I evaluated `whereTyping?`/`common?` by hand on these cases, and on `int64` and `int1` values of
  `x`; they agree. W1-W5 re-ran identical.
- **L84** ACCEPT — "Annotated names" states `visit_AnnAssign` (`:694-712`) and
  `_sanitize_target_value` (`:740-750`) exactly for the one form the tutorials use. In my scan, the
  only annotated assignments in the in-scope kernels are `BM`, `BN`, `LN2`, `MASK_BLOCK_M1`,
  `MASK_BLOCK_N2` and `MIXED_PREC`, each on a new name. C1-C9 re-ran identical. A bare annotation
  and an annotation other than `tl.constexpr` are excluded (row 52). The reason row 52 gives,
  "ignored", is incomplete (wording below).
- **L85** ACCEPT — `visit_For` visits every argument once (`:1206`) and the single argument of
  `range` a second time (`:1241-1243`), as stated. R1-R4 re-ran identical, and my R1
  (`range(atomic * 0, 4)`) evaluates its first argument once. A related gap, the keyword and extra
  arguments of `range`, is filed below.
- **L86** ACCEPT — Each point is fixed and true:
  - the several-result promise is limited to results of the hinted value's shape (integer or pointer
    for `multiple_of`/`max_contiguous`); the other cases are excluded (row 55);
  - index promotion: only an unsigned operand at least 32 bits wide promotes (`integer_promote_impl`);
  - TRANSCRIPTION §3 now lists `range`;
  - the scalar argument ranges;
  - the `lo, hi = 0, N` row;
  - the `x * w + b` row converts `b` and uses `tl.mul_add`;
  - §5 says the gate checks that every function is classed;
  - the dry visit in SEMANTICS' opening: `_find_carries` erases its block (`:465-503`), and Z1-Z2
    re-ran identical;
  - `other`'s cast (`semantic.py:1043-1045`): O1 re-ran identical.
- **L87** ACCEPT — `cumsum` (`standard.py:334-343`) promotes `bfloat16` first, takes `dtype` from
  `_pick_sum_dtype` when given, converts with `.to(out_dtype)`, and scans with `_sum_combine`. That
  is what SEMANTICS "Scans", the `TVA.Lang.Scan` docstring and TRANSCRIPTION's row say. S1-S3
  re-ran identical.
- **L88** ACCEPT — The list covers every form the 23 in-scope tutorial functions use. My own AST
  scans (`work/subset_scan.py`, `work/subset_scan2.py`) found only listed forms:
  - statements: assignments to names, and to tuples of 2 or 3 names with as many elements;
    augmented assignments with `+ - *`; `name: tl.constexpr = e` on new names; expression statements
    that are calls, or one docstring; `for` over `range` or `tl.range`, `while`, `if` and `return`;
  - listed operators; conditional expressions on compile-time conditions only; the subscripts
    `[:, None]`, `[None, :]` and integer subscripts of a shape;
  - listed builtins, with `tl` and the module's helpers as the only free names;
  - no kernel parameter default, no `for` variable or body-only name used after its loop.

  A reviewer can check a kernel against the list. Each form L61, L76, L78, L80, L84, L85 and L87
  state is stated truly, and each form they exclude has its exclusions.md row. Two forms the list
  admits differ from what the library says; they are filed below as new issues (code after a
  `return`, and `range`'s arguments).

## New issues

- **NEW** soundness — **Code after a `return` reached through a compile-time `if` is still compiled,
  and its hints apply.**
  - **The text.** SEMANTICS "`return`" says "Code after a `return` at the top level is not
    compiled". Nothing says what happens after a compile-time `if` whose taken branch returns, or
    after a `tl.static_range` copy that returns. The natural reading is that the taken branch
    replaces the `if`, so the rest of the function is not compiled either. A model that drops that
    code (TRANSCRIPTION §4) then drops its hints.
  - **The front end.** It stops visiting only at the statements after a `return` in the `return`'s
    own statement list (`code_generator.py:508-517`). `visit_Return` opens an unreachable block
    (`:544-554`). Everything after the enclosing `if`, and every later `tl.static_range` copy, is
    still visited into that block. That code never runs, but its compile-time effects stay: a hint
    marks the operation that defined its argument, including one defined before the `return`, and a
    `tl.static_assert` is evaluated.
  - **The device** (`work/dead_after_return.py`):
    - D7: `offs = tl.where(i < m, 16 + i, 0)`, a load through `offs`, then `if FLAG: return` with
      `FLAG` a `tl.constexpr` that is true, then `offs = tl.max_contiguous(tl.multiple_of(offs, 16),
      16)`. The load became `ld.global.v2.b32` and read wrong values in 137 of 256 lanes. Without the
      hint (D8) it was `b8` and correct;
    - D1: with `offs = base + tl.arange(0, 256)` and `base = 4`, the launch faulted (misaligned
      address);
    - D9: the same as D7 through `tl.static_range` copies;
    - D5: a `tl.static_assert(False)` after the `if` rejects the kernel;
    - D4, D6: after a plain `return`, nothing is visited.

    A model without the check is well behaved and ends normally with the right values, so
    `Faithful` is false for it.
  - **Fix.** State it in SEMANTICS "`return`" and in TRANSCRIPTION §3/§4. After a compile-time `if`
    whose taken branch returns, and after a `tl.static_range` copy that returns, the rest of the
    function is visited but does not run. Its hints are checks where their values are defined, and
    its `tl.static_assert`s are evaluated. Alternatively, exclude statements after such an `if`. No
    in-scope tutorial has one: `_maybe_make_tensor_desc` returns in both branches, with nothing
    after.
- **NEW** gap — **The subset admits `range(…)` with any arguments, but the front end ignores
  keywords and a fourth argument.**
  - **The text.** The subset's "Statements" item admits `for name in range(…)` with any arguments.
    SEMANTICS gives meaning only to one, two or three positional arguments ("Loops"), and calls
    `range` one of "Python's own" builtins, and Python rejects keywords and a fourth argument.
  - **The front end.** `visit_For` evaluates the keyword arguments (`:1207`), but for Python's
    `range` it uses only `iter_args[0..2]` (`:1237-1243`). A keyword is ignored, and so is a fourth
    positional argument.
  - **The device** (`work/fe_edges.py`, `n = 8`):
    - K1: `for i in range(0, n, step=2)` ran 8 trips, not 4;
    - K2: `range(n, step=2)` ran 8 trips;
    - K4: `range(0, n, 2, 99)` ran 4 trips, dropping `99`;
    - K3, the control: `tl.range(0, n, step=2)` ran 4.

    A model that reads `step=` as the step, by analogy with `tl.range`, differs from the device.
  - **Fix.** Make the item "Python's `range` with one, two or three positional arguments". Add a
    keyword argument and a fourth argument to the subset's excluded forms and to exclusions row 48.
- **NEW** wording — Six points in the transcription area:
  - (a) Exclusions row 52 and SEMANTICS "Annotated names" say an annotation other than
    `tl.constexpr` is "ignored". Its value is also evaluated twice: `visit_AnnAssign` visits
    `node.value` (`:698`), then `visit_Assign` visits it again (`:712`, `:757`). In
    `work/fe_edges.py` A1, `a: tl.int32 = tl.atomic_add(c, 1)` ran 16 atomics in 8 programs. Say
    "its value is evaluated twice, and the annotation is ignored".
  - (b) Exclusions row 54 and SEMANTICS "Names after a loop" say that after the loop, a `for`
    loop's variable holds a placeholder with no defined value. After `tl.static_range` it holds the
    last value (S1: `2`), and a name the body defines is defined after the loop (S2: `20`). The
    subset's "Names" item excludes both uses, which is conservative. "Names after a loop" calls the
    unrolled copies "statements in sequence", which suggests the uses are fine. Limit the
    placeholder sentence to `range` and `tl.range`, and say whether these uses are in the subset.
  - (c) SEMANTICS "A runtime `if`" merges "a name new to the `if` that both branches define". A
    name both branches bind with `name: tl.constexpr = e` is not merged: `visit_AnnAssign` writes
    `lscope` only, not `local_defs` (`:704-710`). A use after the `if` is then rejected on the host
    (A2: "c is not defined"). The divergence is conservative. Say it.
  - (d) SEMANTICS "Helpers are inlined" binds a call as Python does, with the `def`'s defaults. But
    the front end also visits each default as an assignment (`:634-649`). An unannotated default it
    cannot make a tensor (a string; an integer beyond 64 bits) therefore rejects the helper, even
    when the call passes the argument. H1: `def h(x, mode="a")`, called as `h(x, "b")`, was rejected
    with "cannot convert a of type <class 'str'> to tensor". The divergence is conservative. Say it.
  - (e) TRANSCRIPTION §2, row "`x = tl.multiple_of(x, 16)`", takes `contig` from "the lower bound
    SEMANTICS states for `x`'s shape". The §3 hint pitfall says "for the value's shape" and "the
    listed shapes". SEMANTICS' bound depends on how `x` is computed, not on its tensor shape, so a
    reader could take `B` for any `[B]` tensor. Say "for how `x` is computed (SEMANTICS 'The
    contiguity the analysis infers')".
  - (f) `docs/rules/compiler-faithful.md`, the reviewer's checklist, third item, ends with "and the
    arguments those the last hook leaves". The sentence is garbled.

## Re-ran

- In full, on GPU 0 (NVIDIA H200 NVL, Triton 3.8.0, torch 2.11.0+cu130, 2026-10-28), identical to the
  recorded output apart from the first line: evidence/front-end-subset, evidence/hint-hinted-bound
  and evidence/desc-tf32 (`work/rerun-*.out`); lr-11a's
  `docs/library-review/round-11/lr-11a/work/if_compile_time_livein.py` (I1-I6, L61) and lr-11b's
  `docs/library-review/round-11/lr-11b/work/heur_options.py` (H1-H5, L78), in
  `work/rerun-r11-*.out`.
- New tests, each case in its own process with its own Triton cache:
  - `work/fe_edges.py` (K1-K4, A1-A3, S1-S2, H1-H2, W1, R1). W1: a `return` inside a compile-time
    `if` in a `while` body is rejected, as SEMANTICS says;
  - `work/inline_probe.py` (I0-I3, I5). A helper with a `return` inside a runtime `if` is inlined at
    the top level, in a `for` body and in a runtime `if` (no `tt.call` left), and a hint pair on its
    result has no effect: `b8` loads, against `v2.b32` for the same hint on the value directly (I5).
    This agrees with SEMANTICS' "every call is inlined", on which `no-op:hint` relies;
  - `work/dead_after_return.py` (D1-D9);
  - the AST scans `work/subset_scan.py` and `work/subset_scan2.py` of the 23 in-scope functions.
- Read against `primitives/`:
  - `triton-3.8.0-code-generator.py`, whole;
  - `semantic.py`: promotion, `to_tensor`, `scalar_constant`, `where` and `load`;
  - `core.py`: `_unwrap_if_constexpr`, `builtin_min`/`builtin_max`;
  - `standard.py`: `cumsum`, `_pick_sum_dtype`;
  - `AxisInfo.cpp`: `236-282`, `1225-1259`;
  - `compiler.py`: `80-115`, `225-245`; `nvidia-driver.py`: `220-235`; `autotuner.py`: `455-490`.

  Read in TVA: `lean/TVA/Lang/Promote.lean`; the hint section of `lean/TVA/Lang/Ops.lean`;
  `Faithful`'s docstring; `TVA.Claim`; `Launch.fusion`. I compared every row and pitfall of
  TRANSCRIPTION §2-§4 with SEMANTICS and these sources.
- The coordinator's release check of the merged tree: I read its output and did not re-run it.
  8 of 10 steps pass; the two failures are vec_add_f32's gate (being re-proved) and the pages.

## Not checked

- A runtime `if` whose two branches both return, followed by code. By the same source, that code
  is visited into an unreachable block. I did not run it.
- I did not run the release check or `tvr check`, and I did not touch `kernels/vec_add_f32`.
- Outside my area: the device facts and the float layer (round 12), and the tool and gate
  (round 16).
