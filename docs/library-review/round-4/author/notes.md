# Library batch, round 4: author's notes

Issues: L7, L12, L17, L18, L19, L20, L23, L24, L25, L26, L27 ([ISSUES.md](../../ISSUES.md)).
2026-10-27. The library builds (`lake build TVA`); the new constants (`FloatV.round6`, `litF64`,
`litF32`, `litF16`, `litBF16`) and the rules' lemmas (`exists_total_faithful`, `independent`,
`shared`, `Launch.valid_grid1`) depend only on `propext`, `Classical.choice` and `Quot.sound`, and
no theorem's statement changed.
`python3 -I tools/tvr.py audit` prints "independent". I followed the coordinator's notes for L7
and for the class L17/L18/L19.

**Kernel that does not build or pass the gate:** `kernels/vec_add_f32`, as before this round (its
prover's to fix; I did not edit the kernel folder or run `tvr check` on it). This round asks it
for nothing new: on a scratch copy of the tree with round 3's three updates made there
(`Launch.valid_grid1`, map rules, disjoint output and inputs), the round-4 gate passes every check,
`namespace` and `hygiene` included ([evidence/gate-digests](../../../../evidence/gate-digests/README.md)
case A). Every one of its item digests changes once (L27 hashes every library docstring, and the
general text now holds SEMANTICS.md), so every item re-opens.

## L7: TVA cites no other project's results, lemmas or measurements, and the tool enforces it

- **What changed** in `tools/tvr.py` (`_unescape`, `_path_text`, `_HOME_VARS`, `_HOME_START`,
  `_FPV`, `foreign_refs`):
  - **The coordinator's rule.** A path component `fpv` followed by a component other than `tva`
    (after `..` is resolved) is a path to a sibling of TVA, wherever it occurs and whatever
    precedes it (`_FPV`). This one rule catches every round-3 spelling that points into
    `~/fpv/<other>`, whatever comes before the path: a redirect, a bracket, `-`, `$(…)`, an
    f-string.
  - **Home paths after any punctuation.** A home path (`~/`, `~<user>/`, `/home/`) now starts
    after any character other than a letter, digit, `_`, `.`, `/`, `\` or `~` (`_HOME_START`),
    so also after `>`, `)`, `]`, `}` and `-`. The placeholder exception (`round-<N>/author`,
    `{kernel}/review`) stays only for other absolute paths, as lr-3b proposed.
  - **Spellings.** `${HOME…}` with any expansion operator (`${HOME:-}`, `${HOME%/}`) and every
    command substitution `$(…)` are read as `~`, as lr-3a proposed. The escapes `\uXXXX`,
    `\UXXXXXXXX`, `\xXX`, `%XX`, `&#N;` and `&#xN;` of path characters, letters and digits are
    decoded before the path test, and the name check also runs on the decoded text.
  - **Protocol §2** ("Enforcement", "Text") lists the rules, and states that the text scan is
    lexical: a path a program computes at run time is not checked (except that a computed path into
    a sibling still shows `fpv/<name>`), and the guarantee for results and lemmas is Lean's imports
    and the lake configuration, not the text scan.
  - **An earlier record changed, because the stricter audit flagged it.** In
    `docs/library-review/round-2/author/notes.md` lines 105–106, two example paths (a sibling
    folder whose name starts with `tva`, and a path into TVA that climbs out with `..`) were
    written with the folder `fpv` before them; now they read `…/tva-old` and `…/tva/../x`, with
    `…` in place of `fpv`. The new `fpv/<name>` rule reads the old spelling as a path into a
    sibling of TVA, so the audit failed on that file. Nothing else in it changed, and no other
    earlier record needed a change.
- **Why.** Both round-3 reviewers found home paths that a shell or a JSON reader resolves and the
  round-3 scanner passed: `${HOME:-}/…`, `${HOME%/}/…`, `$(echo ~)/…`, `/` escapes, and a home
  path right after `>`, `)`, `]`, `}` or `-`. This was the third reopen of a pattern-by-pattern
  scanner, so the main fix is a rule that does not depend on what precedes the path.
- **Evidence.** [evidence/independence-audit](../../../../evidence/independence-audit/README.md)
  adds probes P20–P34 and N9 (the reviewers' spellings, `fpv/<sibling>` without a home prefix,
  `%2F` escapes, a control with TVA's own paths, placeholders, `$(date)` and `100%`): 79 probes,
  0 wrong, baseline independent ([out.txt](../../../../evidence/independence-audit/out.txt)). The
  reviewers' own probes rerun against the new tool: lr-3a's `work/l7_more_spellings.py` flags all
  seven spellings, lr-3b's `work/l7_probes.py` gives "17 probes, 0 not as the rule asks".
- **Constants whose meaning changed.** None (the tool only).

## L12: No digest covers the transcription rules

- **What changed.**
  - **Where a rule ends** (`transcription_rules` in `tools/tvr.py`). A rule is its list item: its
    first line and the lines after it that are blank or indented. The first unindented line ends
    it. So the closing paragraph of TRANSCRIPTION §3 ("No construct of exclusions.md is used …")
    is now general text, not part of `[atomic]`.
  - **The page says so.** TRANSCRIPTION.md's introduction has a paragraph "Where a rule ends",
    which states the rule and gives the general parts as examples (the introduction, the
    paragraphs that open §1, §5 and §10, the closing paragraph of §3, and §9). It no longer calls
    §1 general as a whole: §1 holds `[helper]`.
  - **The tool checks the page.** An unindented line that directly continues a rule's paragraph
    (Markdown would show it as part of the rule, while the digests would treat it as general text)
    is a page error. `tvr audit` and the gate's `map` check report it. The page has none.
  - I did not add a check that each part the introduction names is general, as lr-3b suggested.
    With the rule above, each of those parts is general by construction, and the continuation
    check keeps the page's look and the digests' split the same.
- **Why.** The page said the closing paragraph of §3 was general text, while the parser folded
  it into `[atomic]`, so a change to it re-opened only maps naming `[atomic]` (lr-3a
  `work/l12_general_text.py`, lr-3b `work/l12_digests.py` X1/X2).
- **Evidence.** [evidence/gate-digests](../../../../evidence/gate-digests/README.md) case I: on a
  copy, changing that paragraph changes the `map` and `model` digests of `vec_add_f32`, whose map
  does not name `[atomic]`. Cases A–H (round 3) give the same results as before.
- **Constants whose meaning changed.** None in Lean. The rule texts changed: `[atomic]` (shorter),
  and the general text (it gains that paragraph, and SEMANTICS.md, L27). So every kernel's `map`
  and `model` re-open.

## L17: No `tl.permute` (rank ≥ 3), PDL (`gdc_*`, `launch_pdl`) or `tl.argmax`

- **What changed.** The coordinator's proposal, one scope statement for the class L17–L19, in the
  docstring of `Rules.Compiler.Faithful`, section "Scope", point 1: *the launch runs alone*. From
  its start until it ends, nothing else writes global memory: not the host, not another kernel on
  any stream or device, and not a later kernel that programmatic dependent launch starts early.
  - `[no-op:gdc]` is split. `gdc_wait()` is dropped because a valid launch has `pdl = false`.
    `gdc_launch_dependents()` is dropped only when no next launch on the stream uses
    `launch_pdl=True`, which point 1 asks and the kernel reviewer checks in the host code.
  - exclusions.md lists a kernel that calls `gdc_launch_dependents()` followed by a launch with
    `launch_pdl=True`, and any other writer during the launch.
  - Also stated in the `TVA.Claim` module docstring and the docstrings of `Claim.Behaviour` and
    `Launch.pdl`, in [rules/compiler-faithful.md](../../../rules/compiler-faithful.md) ("Scope",
    "What it does not cover", what the reviewer checks), SEMANTICS.md "Launches", TRANSCRIPTION §9
    (four points the kernel reviewer checks), PROVER_GUIDE.md and REVIEWER_GUIDE.md.
  - The scope is stronger than "nothing writes the bytes the launch accesses". A claim's `Post`
    usually says that every other byte is unchanged, which another writer anywhere would falsify.
- **Why.** The claim is about one launch, from the memory it starts with, to the memory it ends
  with. lr-3a and lr-3b showed that a kernel launched without PDL, which calls
  `gdc_launch_dependents()`, lets the next kernel run while it still runs. So the dropped call
  was not a no-op, and `Compiler.Faithful` was false for a valid, well-behaved launch.
- **Evidence.** [evidence/pdl](../../../../evidence/pdl/README.md) part D (rerun with A–C):
  - the claimed kernel calls the function, and the next launch uses `launch_pdl=True`: `y` is
    wrong in 20 of 20 runs (35225600 of 83886080 elements);
  - without the call, or with a next launch without PDL: right in all 20 runs of each variant.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (docstring, hashed: every
  kernel's `assumption:faithful` re-opens) and the docstrings of `TVA.Claim.Behaviour` and
  `TVA.Launch.pdl`. No definition changed. Rule `[no-op:gdc]`, and the general text
  (exclusions.md, SEMANTICS.md, TRANSCRIPTION §9).

## L18: `TRITON_F32_DEFAULT` changes `tl.dot`'s default precision

- **What changed.** `Faithful`'s scope, points 2 and 3 (the coordinator's proposal and lr-3a's
  fix).
  - **Point 2: Triton's knobs at their defaults, however they would be set.** Every knob of
    `triton.knobs` (`primitives/triton-3.8.0-knobs.py:320-513`) and every variable of
    `GetEnv.h` keeps its default, whether it would be set by the environment variable or by
    assigning the knob from Python, with or without `propagate_env`. The exceptions are the
    directory, dump, print, timing and debug-information ones, listed as before, and the knobs
    they set.
  - **Point 3: no hook.** `knobs.runtime.launch_enter_hook`, `launch_exit_hook`,
    `kernel_load_start_hook`, `kernel_load_end_hook`, `kernel_unload_hook`, `jit_cache_hook`,
    `jit_post_compile_hook`, `add_stages_inspection_hook`, `knobs.compilation.listener`,
    `knobs.autotuning.listener`, and a `JITFunction`'s pre-run hooks.
  - `Launch.debug` says that `knobs.runtime.debug = True` (and the decorator's `debug`) turns
    `debug` on. `Launch.fusion` and `DotCfg.precision` speak of the knobs.
  - The same text is on the rule page, SEMANTICS.md "Launches", TRANSCRIPTION §9, exclusions.md
    (a knob not at its default; a hook) and PROVER_GUIDE.md.
  - I pinned `primitives/triton-3.8.0-jit.py` (equal to the installed file, sha256 in
    `semantics-manifest.json`) to cite `run` (`:727`, `:735-736`, `:757-759`) and
    `_do_compile` (`:880-882`).
- **Why.** lr-3a showed that the same switches are set from Python with no environment variable
  set, and that a hook rewrites the PTX or skips the launch (`work/options_knobs.py`). Round 3's
  scope named environment variables only.
- **Evidence.** [evidence/launch-options](../../../../evidence/launch-options/README.md), each case
  in a fresh process with no `TRITON_*` variable and an empty cache:
  - `knobs.language.fp32_default = "ieee"` with `propagate_env = False` makes a `tl.dot` `ieee`
    with no variable set (D3); with the default `propagate_env` the assignment sets
    `TRITON_F32_DEFAULT` (D4);
  - `knobs.language.default_fp_fusion = False` removes the `fma` (F1);
  - `knobs.runtime.debug = True` makes a dropped `device_assert` trap (S1);
  - a stages hook turns `x + 1` into `x - 1` (A2), and a `jit_cache_hook` returning true makes the
    call return with nothing run (A3).
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (docstring, hashed), and the
  docstrings of `TVA.Launch.debug`, `TVA.Launch.fusion` and `TVA.DotCfg.precision`. No definition
  changed. The general text (exclusions.md, SEMANTICS.md, TRANSCRIPTION §9).

## L19: Launch keywords outside `Launch` change the compiled code

- **What changed.** `Faithful`'s scope, point 4: *only the launch options `Launch` records*.
  - No field of `CUDAOptions` (`primitives/triton-3.8.0-nvidia-compiler.py:108-135`) is given at
    the launch or in an autotune `triton.Config`, except `enable_fp_fusion`, `debug`, `num_ctas`
    and `launch_pdl` (`Launch.fusion`, `debug`, `numCtas`, `pdl`).
  - Four options may vary because they do not change the meaning: `num_warps`, `num_stages`,
    `maxnreg` and `launch_cooperative_grid`.
  - The excluded fields are named: `ir_override`, `allowed_dot_input_precisions`,
    `default_dot_input_precision`, `enable_reflect_ftz`, `extern_libs`, `ptx_options`,
    `ptx_version`, `arch`, `warp_size`, `supported_fp8_dtypes`,
    `deprecated_fp8_dot_operand_dtypes`, `max_num_imprecise_acc_default`, `sanitize_overflow`,
    `instrumentation_mode` and `backend_name`.
  - The `TVA.Launch` docstring says so, and so do the rule page, TRANSCRIPTION §9, SEMANTICS.md,
    exclusions.md and REVIEWER_GUIDE.md (the host code within the scope).
- **Why.** lr-3a: `ir_override` gave `x + 2` for `x + 1`, `allowed_dot_input_precisions` switched a
  `tl.dot` to `ieee`, and `enable_reflect_ftz=False` turned FTZ off. `jit.py:713-715` accepts every
  option field as a keyword, and `Faithful` excluded none.
- **Evidence.** [evidence/launch-options](../../../../evidence/launch-options/README.md):
  - A1, D1 and E1 reproduce the three changes;
  - D2 (`default_dot_input_precision`) changed nothing here, but the rule excludes it anyway, as it
    excludes every field it does not list;
  - B: 9 variants of `num_warps`, `num_stages`, `maxnreg` and `launch_cooperative_grid` on a kernel
    with exact results (an `int8` `tl.dot`, an integer `tl.sum`, a fused `x * 2 + 1` with NaNs) give
    0 differing elements.

  The other excluded fields (`extern_libs`, `ptx_options`, `arch`, …) were read in the compiler
  source, not run.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (docstring, hashed) and the
  docstring of `TVA.Launch` (the structure). No definition changed.

## L20: Model text can mean another operation

- **What changed.**
  - **The user's addition: kernel reviewers read the whole Lean.** In
    [prompts/kernel-reviewer.md](../../../prompts/kernel-reviewer.md), the `kernels/<kernel>/lean/`
    row no longer says the reviewer need not read the Lean files. It says to read every Lean file in
    full, and what-you-judge item 1 gains "What the names mean". The same is in
    [REVIEWER_GUIDE.md](../../../REVIEWER_GUIDE.md) §1 ("Names: read the kernel's whole Lean") and
    REVIEW_PROTOCOL §2 (the kernel reviewer judges what every name means). A model line that reads
    as a library operation but means another is a `MODEL_MISMATCH`, and the gate is a backstop.
  - **Gate, `namespace` check** (`tools/claim_reader.lean.in`, `check` in `tools/tvr.py`). The
    claim reader now reads the constants of the kernel's modules from the module data. Before,
    it read `env.constants.map₂`, which holds no imported constant, so the old check never fired.
    It flags a kernel constant:
    - declared in `TVA`, or outside the kernel's namespace;
    - with a name component `tl`;
    - whose last components, as the kernel may write them, are also a library constant through
      `TVA` or through a library namespace the kernel's files `open` (`tvr` reads the `open`
      lines and passes them in), such as `Kernels.K.lane_tile` under `open Block`;
    - an instance of a library class, or one whose type mentions a library constant (an `HAdd`
      on `F32`).

    Private names are checked under the name the kernel wrote. Equation lemmas that Lean generates
    when a proof unfolds a library definition (`TVA.Launch.ctx.eq_1`, a reserved name) are exempt.
    The reader's cones now also follow kernel-module constants named outside the kernel's root.
  - **Gate, `hygiene`.** It also forbids notation (`notation`, `infix`, `infixl`, `infixr`,
    `prefix`, `postfix`, `binder_predicate`, `unif_hint`), `open … renaming`, `export`, `local`
    and `scoped` instances (which the claim reader cannot see), instance attributes and instance
    erasure, and `set_option` other than resource limits, `autoImplicit`, `pp.*`, `trace.*` and
    `linter.*`.
  - TRANSCRIPTION §1, REVIEW_PROTOCOL §8 and PROVER_GUIDE.md state the rule.
- **Why.** lr-3a's `ShadowRoute.lean` declared `tl.add := TVA.tl.addc` in the kernel's
  namespace, and `NotationRoute.lean` a `local notation "tl.add"`. Both pass round 3's gate, while
  the model text reads as IEEE `tl.add`. The user: give the reviewer the full Lean; such cheating
  is then easily caught.
- **Evidence.** [evidence/gate-digests](../../../../evidence/gate-digests/README.md) cases L–O, on
  a copy of the tree with `vec_add_f32` updated there:
  - L: a `def tl.add` in the kernel's namespace fails `namespace`;
  - M: a `local notation "tl.add"` fails `hygiene`, and nothing is built;
  - N: `instance : HAdd F32 F32 F32 := ⟨FloatV.sub⟩` fails `namespace`;
  - O: a `theorem lane_tile`, a name of the opened `TVA.Block`, fails `namespace`;
  - P: with all restored, the gate passes again.

  The real kernel's Lean passes both checks (case A).
- **Constants whose meaning changed.** None (the tool, the claim reader and the role documents).
  The `namespace` and `hygiene` checks are stricter, so a kernel that passed them may fail now.
  The only kernel, `vec_add_f32`, passes them.

## L23: `[literal]` is wrong for fp64 and bf16 constants

- **What changed.**
  - **Lean** (`TVA/Lang/Scalar.lean`). New constants, each a function of the Python number's
    double, with `+0` for a zero:
    - `FloatV.litF64`, the double itself;
    - `FloatV.litF32`, the nearest `float32`;
    - `FloatV.litF16`, the nearest `float16` to that;
    - `FloatV.litBF16`, the `bfloat16` nearest to the six-decimal text of that `float32`, where
      `FloatV.round6` is C's `%f`, ties to even;
    - the module text cites `scalar_constant` and the builder functions.

    `decide` evaluates them (`litBF16 (ofPat 0x3E7AD7F29ABCAF48) = ofPat 0`, `1e-7`).
  - **`[literal]`** is restated by where the number is used:
    - *with an operator or `tl.where`, and in `tl.full`*: built in the dtype the operation
      computes in, written with these functions (an fp8 dtype: `litF32`, then `tl.cvt8`; an
      integer: the number);
    - *everywhere else* (`other=`, a stored value, any other `tl.*` function such as `tl.maximum`):
      a tensor of its own type (`float32` in float32's normal range, `float64` otherwise), then
      the conversions.
  - SEMANTICS.md "Constants", LIBRARY.md and PROVER_GUIDE.md say the same.
- **Why.** lr-3b showed that a Python scalar becomes `scalar_constant(c, dtype of the operation)`.
  In fp64 that is the exact double, and in bf16 it is built from `std::to_string` (six decimals),
  so `x * 1e-7` on bf16 multiplies by `+0`. The rule (nearest fp32, then converted) held only for
  fp32, fp16 and `other=`. From the source: `semantic.py:118-158` (a number's own type),
  `:205-206`, `:601-613`, `:622-623`; `ir.cc:952-972`; `core.py:2534-2535`, `:2947-2948` (`load`
  and `maximum` make the number a tensor first).
- **Evidence.** [evidence/literals](../../../../evidence/literals/README.md):
  - the literals: 9364 values, each with both signs, adversarial (around the six-decimal cut, its
    ties, next to `bfloat16` rounding midpoints, outside float32's range) and random;
  - for each of four dtypes, `tl.full`, `x * c` and `other=` on the device;
  - compared bit for bit with the Lean definitions evaluated by `#eval`: 0 differences for
    `tl.full` and `x * c` against `FloatV.lit…`, and 0 for `other=` against the own type then one
    rounding;
  - the operation's constant and `other=`'s differ for 3202 `bfloat16` and 9294 `float64`
    literals;
  - C's `%f` and Python's agree on every value.

  fp8 constants were read in the source, not run.
- **Constants whose meaning changed.** New: `TVA.FloatV.round6`, `litF64`, `litF32`, `litF16`,
  `litBF16`. No existing constant changed. Rule `[literal]`, and the general text (SEMANTICS.md).

## L24: `[itof-narrow]` is wrong under `fp_downcast_rounding="rtz"`

- **What changed.**
  - `[itof-narrow]` is restricted to the default rounding: no `fp_downcast_rounding`, or `"rtne"`,
    which `cast` treats as the default (`semantic.py:822-845`).
  - With `"rtz"`, the model is `tl.itof` to the intermediate format, then `tl.cvt … .rz`.
  - SEMANTICS.md "`x.to(float)`" and PROVER_GUIDE.md say so.
- **Why.** The fold `truncf(sitofp x) → sitofp x` needs the default rounding mode
  (`llvm-ArithCanonicalization.td:453-463`), and an rtz narrowing is `tt.fp_to_fp`, which no fold
  touches (lr-3b `work/itof_rtz.py`).
- **Evidence.** [evidence/itof-rtz](../../../../evidence/itof-rtz/README.md), exhaustive over all
  `2^32` `int32` values:
  - `x.to(tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtz")` and the same into `float16`
    differ from the new rule in 0 elements;
  - they fall outside `tl.ItofRel` (the old rule) in 2147431680 and 4294894593 elements;
  - the default-rounding chains are inside `tl.ItofRel` everywhere;
  - the TTIR has two `tt.fp_to_fp` and no `arith.truncf`.
- **Constants whose meaning changed.** None in Lean. Rule `[itof-narrow]`; the general text
  (SEMANTICS.md).

## L25: `[config]` drops branch conditions that may run operations

- **What changed.** `[config]` drops the untaken branch of an `if`, `elif` or conditional
  expression that the configuration decides, together with the `if`, `elif` and `else` lines,
  only when the condition is pure (`[pure]`). When it is not, the condition's operations are
  model lines (as `[pure]` says), and only the untaken branch is dropped.
- **Why.** The front end evaluates every condition before it picks the branch
  (`code-generator.py:967-1049`). lr-3b's `work/config_cond.py` ran an atomic inside a constexpr
  condition in every program.
- **Evidence.** [evidence/config-cond](../../../../evidence/config-cond/README.md), 4 programs each:
  - an atomic inside an `if`, a conditional expression and an `elif` condition runs in all 4
    programs, with no `scf.if` in the TTIR;
  - after a constexpr `False and`, it does not run: `and` short-circuits at compile time;
  - pure conditions run nothing.
- **Constants whose meaning changed.** None in Lean. Rule `[config]` (and so every map that drops
  a line with it).

## L26: `[sum-bcast-mul]` names one spelling of what the compiler rewrites

- **What changed.**
  - `[sum-bcast-mul]` is stated by the computation: a float reduction over `k` of
    `x[i, k] * y[k, j]`, with `x` broadcast along a new last axis and `y` along a new first axis,
    in any spelling, is `tl.sum_bcast_mul`. The spellings include `[:, :, None]`/`[None, :, :]` or
    `tl.expand_dims`, `tl.reshape`, `tl.broadcast_to`, either factor order, `tl.sum`, the
    `.sum` method, `keep_dims`, and `tl.reduce` with one addition.
  - This is sound for every spelling, because `tl.sum_bcast_mul` allows both the plain reduction
    (fusable additions of the IEEE products) and the IEEE `tl.dot` the combine pass makes. So the
    reviewer need not know which spellings the pass matches.
  - A sum with a `dtype=` that converts the products first is not this computation.
  - The docstring of `tl.sum_bcast_mul` and SEMANTICS.md say so.
- **Why.** The combine pass matches the IR (`Combine.cpp:112-198`), so `tl.expand_dims` and a
  `tl.reduce` with `+` are rewritten too, even with fusion off, and the old rule sent a reviewer to
  `[reduce]`/`[fusion]` there (lr-3b `work/bcast_mul.py`: 58 of 256).
- **Evidence.** [evidence/bcast-mul](../../../../evidence/bcast-mul/README.md), `float32`,
  `M = N = 16`, `K = 2`:
  - eight spellings are rewritten (`tt.dot` in the TTIR), with fusion on or off, and give 49 of 256
    outputs that no IEEE tree gives: the index form, `tl.expand_dims`, both or one factor through
    `tl.broadcast_to`, `tl.reduce` with `a + b`, the method form, `keep_dims=True`, and
    `dtype=tl.float32` on `float32` inputs;
  - three are not rewritten (swapped factors, `tl.reshape`, a combine with `+ 0.0`), and give 0 such
    outputs with fusion off;
  - on `float16`/`bfloat16` with `M, N ≥ 16`, the rewrite fails to compile (a rejected launch);
  - with `M = 8` or `N = 8`, nothing is rewritten.
- **Constants whose meaning changed.** The docstring of `TVA.tl.sum_bcast_mul` (hashed since L27).
  The definition is unchanged. Rule `[sum-bcast-mul]`; the general text (SEMANTICS.md).

## L27: SEMANTICS.md, which `[call]` relies on, is outside every digest

- **What changed.**
  - **SEMANTICS.md is general text.** `transcription_rules` folds it, with exclusions.md, into the
    general text of every `map` and `model` digest. SEMANTICS.md's introduction, TRANSCRIPTION.md
    "Where a rule ends" and REVIEW_PROTOCOL §4 say so.
  - **Every library docstring is hashed.** The claim reader hashes, with whitespace normalised,
    the docstring of every `TVA` constant in an item's cone, not only those under `TVA.Claim`,
    `TVA.Rules` and `TVA.Platform`. SEMANTICS.md says each `tl.*` definition's docstring states its
    reading of Triton (and what the model writes), so those docstrings are obligations too.
- **Why.** lr-3b: changing SEMANTICS.md's "Constants" bullet, or its `tl.maximum` conversion,
  re-opened nothing (`work/l12_digests.py` X3, X4). I folded the page in, rather than moving each
  "the model writes …" obligation into a rule. The page is short, `[call]` defers to all of it, and
  folding leaves no obligation outside a digest.
- **Evidence.** [evidence/gate-digests](../../../../evidence/gate-digests/README.md):
  - J: a sentence of SEMANTICS.md (the `tl.maximum` bf16 conversion) changes the `map` and `model`
    digests;
  - K: a change to the docstring of `TVA.tl.add`, which the model uses, changes `model`;
  - K2: a whitespace-only change to it changes nothing.
- **Constants whose meaning changed.** None by definition. The digest of every item now covers
  the docstrings of the library constants in its cone, so every kernel item re-opens once, and
  from now on a library docstring change re-opens exactly the items that reach the constant.
