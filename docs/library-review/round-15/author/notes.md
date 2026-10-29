# Library batch, round 15: author's notes

Issues: L58, L61, L76, L77, L78, L80, L83, L84, L85, L86, L87, L88 ([ISSUES.md](../../ISSUES.md)),
2026-10-28. The semantics batch framed by L88 (user, 2026-10-28: TVA gives meaning to a supported
subset of the Python front end), run beside rounds 12 (the float layer) and 16 (tool and gate) in a
separate tree. `lake build TVA` is green; no definition, statement or proof changed (only
docstrings: `TVA.Rules.Compiler.Faithful`, `TVA.Launch`'s `fusion` field, `TVA.tl.load`, the module
docstrings of `TVA.Claim`, `TVA.Lang.Ops` (its hint section) and `TVA.Lang.Scan`, and the `Operand`
docstring of `TVA.Lang.Promote`), so every theorem keeps the standard axioms. No float definition,
numeric lemma or line of `lean/TVA/Lang/Scalar.lean` changed.

**How L88 settles the batch.** SEMANTICS.md has a new section, "The supported subset": a list of
the functions, parameters, statements, names, expressions and calls TVA gives meaning to, each
pointing to the section that states its meaning. It is what the in-scope tutorial kernels use (I
checked every in-scope function of `tutorials/inventory.json` against it, by reading them and by an
AST scan of their node, call, attribute, keyword and annotation kinds), plus generalisations the page
already stated or that cost a sentence (all operators, helper defaults, `tl.cumsum`'s `dtype=`, a
one-argument `range` with any argument). exclusions.md gets one general row ("Python outside the
supported subset") and one row per form this batch excludes. REVIEWER_GUIDE §1 gets the reviewer's
check (file `UNSUPPORTED` for a form outside the list), PROVER_GUIDE §1 one bullet, TRANSCRIPTION §1
a paragraph. For L61, L76, L78, L80 (in part), L84 (in part) and L86 (in part) I chose exclusion;
for L80 (helper defaults), L83, L85 and L87 I stated the form, as each is in the subset or cheap.

**Kernels.** `kernels/vec_add_f32` still fails to build against the current library (expected; its
prover re-proves it after round 12). My changes are docstrings and fact pages, so no other kernel
build can break; SEMANTICS.md, exclusions.md, the rule page and `Faithful`'s docstring changed, so
every item of every kernel that reaches the library re-opens. I did not edit any kernel folder or
run `tvr check`.

**Files of other rounds** (small changes, for the coordinator's merge): `docs/REVIEWER_GUIDE.md` (§1:
a new paragraph "The supported subset" before "The ranges", one clause in "The ranges", and the host
check's sentence: L88, L80, L76-L78), `docs/PROVER_GUIDE.md` (§1: one new first bullet, L88). New
evidence folders, named for this round: `evidence/front-end-subset`, `evidence/hint-hinted-bound`,
`evidence/desc-tf32`.

## L58: Two hint readings are stronger than Triton relies on

- **What changed.** SEMANTICS "Compiler hints" ("The contiguity the analysis infers") and the hint
  section of `TVA.Lang.Ops`: a value that carries a `max_contiguous` hint, alone or in the pair (its
  defining operation is marked, by "Which value"), has the hinted `C` as its bound along each
  dimension, whatever operation defines it, and values built from it start from that `C`; the list
  (`tl.arange`, `x ± y` with `y` constant, expands, broadcasts, integer conversions) applies to values
  that carry none. A `multiple_of` alone on a sub-expression does not change its bound. TRANSCRIPTION's
  hint pitfall says the same.
- **Why.** lr-11a's reopen: the analysis replaces its contiguity by the hinted one on any operation
  (`AxisInfo.cpp:1244-1259`), and `a + s` inherits it (`:242-279`), so for
  `a = tl.max_contiguous(tl.multiple_of(tl.arange(0, B), v), C)` with `C < B` the list's bound `B`
  for `a + s` was above the compiler's belief, and `FirstsDiv` with it promised less than the
  compiler assumes.
- **Evidence.** [evidence/hint-hinted-bound](../../../../evidence/hint-hinted-bound/README.md)
  G1-G6: `y = tl.multiple_of(a + s, 16)` loaded 16 bytes per access with a plain arange (G1) and with
  a `multiple_of` alone on it (G4), and 4 or 8 bytes with the pair `C = 4` or `8` on `a` (G2, G3),
  also through a conversion (G5) and as `s + a` (G6); the values were correct in every case.
- **Constants whose meaning changed.** The reading of the `contig` argument of `TVA.tl.multiple_of`
  and `TVA.tl.multiple_ofN` (the hint section of `TVA.Lang.Ops`); their definitions are unchanged.

## L61: wording: index arithmetic, one-branch names, builtin min/max, hooks and arguments, masked lanes, PROVER_GUIDE's hints

- **What changed.** Item 2, reopened by lr-11a at soundness level, settled by exclusion under L88.
  SEMANTICS "A runtime `if`": its merge rules are for the names the subset admits, a name new to the
  `if` and one that holds a tensor (a block, a scalar of a Triton dtype or a pointer) before it; a
  branch that assigns a name holding another value before the `if` is outside the subset (a
  `tl.constexpr` is rejected on the host; `None` and dtypes are not merged like a tensor). The subset's
  "Names" item and a new exclusions.md row say so.
- **Why.** `visit_then_else_blocks` skips a live-in that is not a Triton value
  (`code_generator.py:858-861`), so after `y = None; if n > 0: y = tl.load(p)` the name is still
  `None`, and a pre-`if` `None` that both branches assign is merged after all; the text said the
  kernel is rejected. No in-scope tutorial assigns such a name in a runtime `if` (tutorial 08's
  `dtype` is only read inside one), so under L88 the form is excluded, not modelled. Items 1 and 3-6
  stay as round 11 fixed them.
- **Evidence.** lr-11a's `docs/library-review/round-11/lr-11a/work/if_compile_time_livein.py` I1-I6,
  cited by the SEMANTICS text and the exclusions row; the source above.
- **Constants whose meaning changed.** None (SEMANTICS.md and exclusions.md).

## L76: soundness: a host `TensorDescriptor` built with `round_f32_to_tf32=True` makes its loads round `float32` to tf32, which `tl.desc_load` does not model

- **What changed.** Excluded, as the coordinator proposed. `Rules.Compiler.Faithful` has a scope 8
  ("Host descriptors load what memory holds": no host `TensorDescriptor` the launch is given has
  `round_f32_to_tf32=True`), in its docstring and on its rule page (now eight conditions, the
  checklist and "What it does not cover" included); SEMANTICS "Launches" lists it; an exclusions.md
  row; REVIEWER_GUIDE's host check, TRANSCRIPTION's host pitfall and `TVA.Claim`'s scope summary name
  it.
- **Why.** The launcher builds the TMA descriptor with the element type `TMA_TF32` when the field is
  set (`primitives/triton-3.8.0-nvidia-driver.py:231-232`), so the device's loads differ from
  memory, while the model's `tl.desc_load` returns the stored values. No pinned tutorial sets it.
- **Evidence.** [evidence/desc-tf32](../../../../evidence/desc-tf32/README.md): with the field
  `False` every stored value came back (T1); with `True` the six values with low mantissa bits came
  back rounded to 10 mantissa bits, to nearest (`1 + 2^-23` as `1.0`, `0x3dcccccd` as `0x3dccc000`;
  T2).
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (docstring: scope 8); the module
  docstring of `TVA.Claim`.

## L77: gap: `Faithful` covers a `CompiledKernel` from `triton.compile` of an IR file, which replaces the source

- **What changed.** `Faithful`'s scope 6 (docstring and rule page) and SEMANTICS "Launches": a kernel
  `triton.compile` built is covered only when its source is an `ASTSource` of the kernel's
  `@triton.jit` function. exclusions.md's options row now names `triton.compile` of a file path, and
  REVIEWER_GUIDE's host check, TRANSCRIPTION's host pitfall and `TVA.Claim`'s scope summary say it.
- **Why.** Given a file path, `triton.compile` compiles that IR file (`IRSource`,
  `primitives/triton-3.8.0-compiler.py:87-112, 233-240`), as `ir_override` does, and the scope
  presented every `CompiledKernel` as the source's code.
- **Evidence.** The pinned compiler source above (no device question).
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (docstring: scope 6); the module
  docstring of `TVA.Claim`.

## L78: soundness: `@triton.heuristics` sets compile options, a route the library never names

- **What changed.** Excluded under L88 (no in-scope tutorial uses it). `Faithful`'s scope 4
  (docstring and rule page) covers a launch only when the kernel is not under `@triton.heuristics`;
  the docstring of `Launch`'s `fusion` field, SEMANTICS ("One launch", "Launches"), the subset's
  exclusions, an exclusions.md row (and the options row), REVIEWER_GUIDE's host check, TRANSCRIPTION's
  host pitfall and `TVA.Claim`'s scope summary say so.
- **Why.** `Heuristics.run` sets each heuristic's keyword before the JIT parses options out of the
  keywords, replacing a launch keyword or a config's of the same name
  (`primitives/triton-3.8.0-autotuner.py:473-483`), so `Launch.fusion` ("the source's or the launch's
  value") could differ from what the compiler used. Stating "the value the JIT receives" instead
  would bring heuristics into the subset with no tutorial needing them.
- **Evidence.** lr-11b's `docs/library-review/round-11/lr-11b/work/heur_options.py` H1-H5 (cited in
  the rule page and the exclusions row); the pinned source above.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (docstring: scope 4),
  `TVA.Launch.fusion` (docstring); the module docstring of `TVA.Claim`.

## L80: gap: SEMANTICS gives no meaning to constexpr functions, plain Python calls, helper defaults or module globals

- **What changed.** Stated: SEMANTICS "Helpers are inlined" binds a call as Python binds it (position,
  keyword, the `def`'s defaults), a default that is a Python number stays a compile-time value in the
  body like one passed directly, and an annotation on a helper's parameter changes nothing (the
  parameter holds what the call passes, a tensor included, as tutorial 06 does). Excluded: a
  `@triton.constexpr_function`, any other Python function a kernel or helper calls (other than
  Python's listed builtins and Triton's `tl.*` functions), and any module-level value other than
  Triton's modules and the module's `@triton.jit` functions (a `tl.constexpr(…)` global, a dtype, a
  number, a namedtuple): the subset's "Functions" and "Names" items and an exclusions.md row. A
  kernel parameter's default is excluded too (general row). REVIEWER_GUIDE's ranges paragraph notes
  that constexpr functions and plain calls are outside the subset; the `TVA.Lang.Promote` `Operand`
  docstring names helper defaults.
- **Why.** Helper defaults are one sentence on top of the existing "passed directly" rule
  (`call_JitFunction` binds and applies defaults, then `normalize_value`,
  `code_generator.py:1358-1365`; nothing reads a helper parameter's annotation). Compile-time Python
  functions run arbitrary Python on the front end's objects, and no in-scope tutorial calls one or
  reads a module-level value other than `tl` and its `@triton.jit` functions; the map does not class
  module-level lines yet (L79, round 16).
- **Evidence.** [evidence/front-end-subset](../../../../evidence/front-end-subset/README.md) D1-D4
  (`def h(x, e=1e-7)`: `h(x)` gave the `bfloat16` product `0`, as `h(x, 1e-7)`; `e = 1e-7; h(x, e)`
  gave `float32(1e-7)·x`; an annotated default the same as D1) and P1-P2 (`def g(n: tl.constexpr)`:
  `a = -7; g(a)` gave `-3`, `g(-7)` gave `-4`).
- **Constants whose meaning changed.** `TVA.Operand` (docstring).

## L83: soundness: TRANSCRIPTION's `tl.where(keep, x, 0.0)` row puts the constant in `x`'s dtype, false for an integer `x`

- **What changed.** Fixed, not excluded (user, 2026-10-28). TRANSCRIPTION §2's row now gives the
  dtype `whereTyping?` gives: for a float `x`, `x`'s dtype with the constant built in it; for an
  integer `x`, `float32`, with `x` converted (`tl.convert .i32 .f32`, `tl.itof`'s rounding) and the
  constant `+0.0`. Two new rows give tutorial 09's `tl.where(offs < M, offs, 0)` (`int32`, the constant
  `IntV.ofInt 0`) and tutorial 06's `tl.where(mask, 0, -1.0e6)` (two Python numbers: `float32`), and
  the "constant in the wrong dtype" pitfall names the integer case. SEMANTICS "Operators and
  promotion" (`tl.where`) and "Constants" already stated the facts; `whereTyping?` is unchanged.
- **Why.** lr-13b: the row held only for a float `x`; for an integer `x` the device computes in
  `float32`, so a model that kept the integers differed from it.
- **Evidence.** [evidence/front-end-subset](../../../../evidence/front-end-subset/README.md) W1-W5:
  `tl.where(keep, x, 0.0)` with `x` `int32` `[16777217, …]` is `float32` and stores `16777216`; with
  `0` it is `int32` and exact; `tl.where(m, 0, -1.0e6)` is `float32`; a `bfloat16` or `float16` `x`
  keeps its dtype (each dtype checked by `tl.static_assert`).
- **Constants whose meaning changed.** None (TRANSCRIPTION.md).

## L84: soundness: the rule "unless the target is annotated `: tl.constexpr`" is false under either reading

- **What changed.** SEMANTICS has a new bullet "Annotated names", and "A name assigned a Python
  number holds a tensor" now ends "unless the statement is `name: tl.constexpr = e`". The rule: that
  statement, on a name not defined before it, binds the name to `e`'s value as it is (a Python number
  stays a compile-time value); on a defined name (a parameter, an earlier assignment, the second copy
  of an unrolled body) it is rejected on the host; the annotation belongs to that statement only, so
  a later plain or augmented assignment to the name, or any assignment to a `tl.constexpr` parameter,
  makes a Python number a tensor. A bare annotation and an annotation other than `tl.constexpr` are
  outside the subset (exclusions.md row). TRANSCRIPTION has a new pitfall ("An annotation read as
  lasting") and a worked row; the `TVA.Lang.Promote` `Operand` docstring says the same.
- **Why.** lr-13b: `visit_AnnAssign` and the pending annotations (`code_generator.py:694-760`) make
  the old rule false for a bare annotation (it changes the next assignment), for a later assignment to
  an annotated name (a tensor) and for other annotations (ignored). The in-scope tutorials use only
  `name: tl.constexpr = e` on new names (tutorials 06 and 10: `BM`, `BN`, `LN2`, `MASK_BLOCK_M1`,
  `MASK_BLOCK_N2`, `MIXED_PREC`), so that form is stated exactly and the others excluded.
- **Evidence.** [evidence/front-end-subset](../../../../evidence/front-end-subset/README.md) C1-C9:
  `-4` for `c: tl.constexpr = -7; c // 2` (C1); "already defined" for a defined name (C2) and in a
  second `tl.static_range` copy (C9); `-3` after `c += 0` (C3); `tl.arange(0, BLOCK)` rejected after
  `BLOCK = BLOCK * 1` (C4); a bare annotation kept `1e-7` a Python number, `0` in `bfloat16` (C5,
  against `float32` without it, C5c); `a: tl.int64 = 2**31 - 1; a + 1` wrapped to `-2^31` (C6);
  tutorial 06's and 10's forms ran (C7, C8).
- **Constants whose meaning changed.** `TVA.Operand` (docstring).

## L85: soundness: `for i in range(e)` with one argument evaluates `e` twice

- **What changed.** Stated (the form is in the subset): SEMANTICS "Loops and unrolling" has a bullet
  "The iterator's arguments": each is evaluated once before the loop, except that `range(e)` with one
  argument evaluates `e` twice, one after the other, the second value being the bound, and the model
  writes the operations of `e` twice; `range(lo, hi)`, `range(lo, hi, step)`, `tl.range` with any
  number of arguments and `tl.static_range` evaluate each once. "Order" and "Loops" point to it, and
  TRANSCRIPTION has a row for `for i in range(n)` and a pitfall.
- **Why.** `visit_For` visits every argument (`code_generator.py:1206`) and, for Python's `range` with
  one argument, the argument again for the bound (`:1241-1242`). The in-scope tutorials pass a plain
  name, for which the double evaluation is harmless, so stating it costs one sentence and keeps the
  form whole.
- **Evidence.** [evidence/front-end-subset](../../../../evidence/front-end-subset/README.md) R1-R4:
  `range(tl.atomic_add(c, 1) * 0)` ran 16 atomics in 8 programs, `range(0, …)` and `tl.range(…)` 8
  each; `range(tl.atomic_add(c, 1))` from `c = 5` ran 6 trips and left `c = 7`.
- **Constants whose meaning changed.** None (SEMANTICS.md, TRANSCRIPTION.md).

## L86: wording: several-result hints, index promotion by narrow unsigned types, and seven transcription points

- **What changed.**
  - Several results (SEMANTICS "Compiler hints" and the hint section of `TVA.Lang.Ops`): the promise
    "the same about each result" is stated when every result has the hinted value's shape and, for
    `multiple_of` and `max_contiguous`, is an integer or pointer tensor (`ConstAlong` holds of any
    element type); a hint on one result of an operation with a result of another shape, or with a
    float result under `multiple_of`/`max_contiguous`, is outside the subset (exclusions.md row); a
    hint after a loop or a runtime `if` marks the loop or the `if`, whose mark the compiler does not
    read (M7-M10), so checking each result is stronger than needed, and sound.
  - "Index arithmetic": promotion by an operand of a wider type, or of an unsigned type at least 32
    bits wide (`uint8`, `uint16` do not promote `int32`).
  - TRANSCRIPTION §3 "Python's builtins" names `range`; "A scalar argument's type" gives the `int64`
    range on both sides and `uint64` from `2^63`; §2's `lo, hi = 0, N` row says `hi` holds `N`'s own
    type; the `x * w + b` row converts `b` too and names `tl.mul_add`; §5 says the gate also checks
    that every function is classed (L50).
  - SEMANTICS' opening of "How the front end evaluates the source": what runs is what the front end
    evaluates, except the first visit of a loop body, whose operations are discarded (only its hint
    marks stay); "Loops and unrolling" says the same.
  - SEMANTICS `tl.load` and the `TVA.tl.load` docstring: `other` is converted to the pointer's dtype
    (`semantic.cast`, `semantic.py:1043-1045`), after a Python number has become a tensor of its own
    type, so a `float16` load's `other` is rounded twice.
- **Why.** lr-11a's and lr-13b's wording points: each statement was imprecise or incomplete as
  quoted in ISSUES.md. For the several-results point I chose exclusion for the undefined cases
  rather than lr-11a's "integer or pointer results of the hinted rank" alone, because a
  `max_constancy` hint also applies to a float result, which that wording would leave unchecked.
- **Evidence.** [evidence/front-end-subset](../../../../evidence/front-end-subset/README.md) Z1, Z2
  (an atomic in a zero-trip body ran in no program, in a one-trip body once per program) and O1 (the
  masked-off lanes of a `float16` load with `other=1 + 2**-11 + 2**-30` hold `0x3c00`, the `float32`
  tie rounded to even; the `float16` nearest to the double is `0x3c01`); evidence/hint-multi M7-M10
  (unchanged); `computationType?` for the promotion point.
- **Constants whose meaning changed.** `TVA.tl.load` (docstring); the reading of
  `TVA.tl.multiple_of`, `TVA.tl.max_contiguous`, `TVA.tl.max_constancy`,
  `TVA.tl.max_contiguous_multiple_of` and their `N` forms on a result of an operation with several
  results (the hint section of `TVA.Lang.Ops`); definitions unchanged.

## L87: gap: `tl.cumsum`'s `dtype=` argument is not stated

- **What changed.** Stated (cheap): SEMANTICS "Scans" and the `TVA.Lang.Scan` module docstring: with
  `dtype=d`, `tl.cumsum` widens `bfloat16` to `float32`, then converts the input to `d`
  (`tl.convert`), with no other widening, and scans in `d`. TRANSCRIPTION has a row.
- **Why.** `cumsum` promotes `bfloat16` first and then converts to `_pick_sum_dtype`'s choice, which
  is `dtype` when given (`primitives/triton-3.8.0-standard.py:270-281, 334-343`). No tutorial uses
  it, but it costs one sentence and completes `tl.cumsum`.
- **Evidence.** [evidence/front-end-subset](../../../../evidence/front-end-subset/README.md) S1-S3:
  `dtype=tl.bfloat16` on `bfloat16` `[1, 2^-8, 2^-8, 2^-8]` gave `bfloat16` sums
  `[1, 1, 1.0078125, 1.0078125]` (default: `float32` sums); `dtype=tl.int8` on `[100, 100, 100, 100]`
  wrapped (`[100, -56, 44, -112]`; default `int32`); `dtype=tl.float32` on `float16`
  `[2048, 1, 1, 1]` gave `[2048, 2049, 2050, 2051]` (default `float16`: `[2048, 2048, 2050, 2050]`);
  each dtype checked by `tl.static_assert`.
- **Constants whose meaning changed.** The reading of `TVA.tl.cumsum`, `TVA.tl.cumsumc` and
  `TVA.tl.icumsum` for a source `tl.cumsum(…, dtype=…)` (module docstring of `TVA.Lang.Scan`);
  definitions unchanged.

## L88: TVA gives meaning to a supported subset of Triton's Python front end; every other form is excluded

- **What changed.** SEMANTICS.md: a sentence in its opening, and a new section "The supported subset"
  before "How the front end evaluates the source", listing functions and decorators, parameters,
  statements, names, expressions and calls, and the host's launch forms, then the excluded forms a
  reviewer should look for. New facts the list needed, each stated in its section: a compile-time
  `if` takes a `bool`, `int` or `None` (another value is rejected); an operator on two compile-time
  values is Python's; a tuple assignment needs one name per element; names after a loop ("Names
  after a loop": a name new to the body is not defined after it, a carried name must hold a tensor
  and keep its type, and a `for` loop's variable after the loop holds no defined value, so its use is
  excluded). exclusions.md: the general row "Python outside the supported subset" and rows for
  `@triton.heuristics` (L78), host descriptors with `round_f32_to_tf32=True` (L76), compile-time
  Python functions and module-level values (L80), annotations other than `name: tl.constexpr = e`
  (L84), a runtime `if` assigning a name that holds a compile-time value (L61), a `for` loop's variable
  after the loop, and several-result hints with results of another shape or a float result (L86).
  REVIEWER_GUIDE §1: the reviewer checks every line the launch runs against the list (`UNSUPPORTED`
  otherwise). PROVER_GUIDE §1 and TRANSCRIPTION §1: stay inside the list. The rule page's checklist
  asks that the source stay inside the subset. A syntactic check in the gate is left to a later tool
  batch, as L88 says.
- **Why.** The user's decision (L88). The list is what the in-scope tutorial kernels use: I read each
  in-scope function of `tutorials/inventory.json` and scanned their syntax (statements: assignments,
  augmented assignments, `name: tl.constexpr = e`, expression statements, `for`, `while`, `if`,
  `return`; no `break`, `with`, comprehension, slice with bounds, default value, global other than
  `tl` and JIT functions), and every construct they use is in the list. The generalisations are those
  SEMANTICS already stated (all operators, `assert`, `print`, `tl.static_range`, `do_not_specialize`)
  or that this batch states in a sentence.
- **Evidence.** [evidence/front-end-subset](../../../../evidence/front-end-subset/README.md) L1-L4
  (`t` defined in a `for` body and `u` in a `while` body: "not defined" after the loop; the `for`
  variable after `range(0, 4)` stored `16776272`, not `3`; a carried `s` stored `6`) and U1
  (`a, b = 1, 2, 3` stored `1`, `2`, dropping the third element without an error); the sources each
  fact cites.
- **Constants whose meaning changed.** None (SEMANTICS.md, exclusions.md and the guides); the
  `Faithful` rule page's checklist names the subset.
