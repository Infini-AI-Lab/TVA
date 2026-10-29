# Library review, round 17, by lr-17b

## Verdicts

- **L58** ACCEPT — A value that carries a `max_contiguous` hint now has as its bound the smaller of
  the hinted `C` and the list's bound without the hint (SEMANTICS "The contiguity the analysis
  infers", the hint section of `TVA.Lang.Ops`, TRANSCRIPTION §2 and §3). That holds in every case the
  subset admits:
  - when the mark survives, `visitOperation` replaces the visitor's contiguity by the
    `tt.contiguity` attribute (`AxisInfo.cpp:1232-1259`), and `getPessimisticValueState` (1309-1341)
    reads the same attribute for an operation no visitor handles. So the contiguity is exactly `C`,
    which is at least the smaller bound;
  - when the mark does not survive (a loop's or runtime `if`'s result, a removed `tt.call`, a folded
    operation, a several-result mark that is lost), the analysis keeps its own contiguity, which the
    list bounds from below;
  - with two hints, the later replaces the earlier (`set_attr`, `ir.cc:411-425`), and the smallest is
    at most the later.

  Every bound involved is a power of two (the hinted values by exclusion, `tl.arange`'s size by
  Triton), so the smaller one divides the compiler's contiguity, and `FirstsDiv x contig v` implies
  the compiler's `FirstsDiv`. The add, broadcast, `expand_dims` and integer-cast visitors are monotone
  in their operands' contiguity, so values built from the hinted value stay bounded. So
  `TVA.tl.multiple_of`'s `contig` is a divisor of the analysis' contiguity, which is what a kernel
  needs. Its cost is a stronger check where a hint the compiler reads lifts the contiguity above the
  list's, and it touches no in-scope kernel: tutorial 06 hints a scalar and an induction variable,
  and tutorial 09 uses the pair. evidence/hint-unread-round17 N0-N10 and lr-15a's B0-B5 re-ran
  identical.
- **L88** ACCEPT — The `tl.cdiv` bullet states Triton's body (`standard.py:32-43`; the method form
  comes from `_tensor_member_fn`, `core.py:51-92`). The front end inlines it as a helper
  (`call_JitFunction`, `code_generator.py:1358-1396`), and `decide_return_type` (556-585) keeps a
  compile-time return type when the body returns one compile-time value. So two compile-time values
  give Python's floor expression, and otherwise the result is the wrapping `+`/`-` and the truncating
  `//` of "Operators and promotion". evidence/cdiv-round17 re-ran identical (C1: every `int32` for
  eight divisors). My own AST scan of the 23 in-scope functions (`work/call_scan.py`) finds the same
  calls as evidence/subset-calls-round17, which also re-ran identical: the `tl` functions, the
  methods `.to`, `.reshape`, `.permute`, `.split`, `.trans`, `.load` and `.store`, the builtins
  `float`, `isinstance`, `min` and `range`, and the six helpers, with the same keywords. Each has a
  meaning in SEMANTICS or `TVA.Lang` (`tl.num_programs` and `tl.zeros` by their `Ops.lean`
  definitions). The other names (`filter`, `list`, the config builders) are in decorators, which run
  on the host. The five conditional expressions are all on compile-time conditions.
- **L89** ACCEPT — The form is excluded clearly, in five places: the subset's `return` item, which
  defines "returns on the instance" recursively, the subset's excluded forms, exclusions.md row 56,
  SEMANTICS "`return`" and TRANSCRIPTION §3. The reason given is true of `code_generator.py`:
  - `visit_compound_statement` stops only after a `return` in its own list (508-517);
  - `visit_Return` opens a block with no predecessor (544-554);
  - `visit_if_top_level` joins two branches that return in a block reached only from such blocks
    (893-929);
  - `decide_return_type` takes every `return` the front end visited.

  The recursive definition catches every way the front end reaches a later statement after a
  `return`: nested compile-time `if`s, `elif` chains (an `if` in `orelse`), runtime `if`s whose
  branches return through inner `if`s, and a helper's own statement lists. A `return` in a `for` or
  `while` body is rejected: through a runtime `if` by the `scf_stack` check (980-982), and through
  a compile-time `if` as lr-15b's W1 found. evidence/dead-code-round17 E1-E7 and lr-15b's
  `dead_after_return.py` D1-D9 re-ran identical.
- **L90** ACCEPT — `visit_For` evaluates the iterator's positional and keyword arguments
  (1205-1207). For Python's `range` it uses only `iter_args[0..2]` (1237-1243), and `tl.range` and
  `tl.static_range` get their arguments through their signatures. Four places say so: the subset's
  `for` item ("one, two or three positional arguments and no keyword") and its excluded forms,
  SEMANTICS "The iterator's arguments", exclusions.md row 57 and TRANSCRIPTION's loop pitfall.
  lr-15b's `fe_edges.py` K1-K4 re-ran identical.
- **L91** REOPEN — The exclusion itself is clear and conservative. It is stated in scope 9 of
  `Faithful`'s docstring and page, `TVA.Claim`, `Launch.Valid`, SEMANTICS, exclusions.md row 58,
  protocol §6, both guides and TRANSCRIPTION, and lr-16b's G1-G3 re-ran identical. The fault is the
  sentence that says why. It reads "a call made while a CUDA stream capture is in progress also
  returns without an error and runs no program", and it is false as written. Where it appears:
  `TVA.Claim`'s module docstring, `Launch.Valid`'s docstring, compiler-faithful.md "What it says" and
  scope 9, `Compiler.lean`'s scope 9 ("Under capture the call … runs no program"), SEMANTICS "One
  launch", its scope list and its outcome bullet, exclusions.md row 58's reason, and protocol §6.

  Only a call on the stream that is capturing is recorded and runs nothing. A call on another stream
  while a capture is in progress runs its programs at once. In `work/probes.py` G1-G3,
  `torch.cuda.graph` captured in modes `"global"`, `"thread_local"` and `"relaxed"`, and the call was
  made under `torch.cuda.stream(s2)`. It returned without an error and wrote `x + 1` before any
  replay, and a replay of the graph wrote nothing. In G0, the call on the capture stream kept the
  sentinel. These are fact pages that reviewers trust: a reviewer checking scope 1 could take a
  writer launched on a second stream during someone's capture as running nothing.

  Fix: say that a call on a capturing stream (the current stream inside `torch.cuda.graph` or
  `do_bench_cudagraph`) returns without an error and runs no program, and that its programs run at
  each replay with the captured arguments. Say that a call on another stream during a capture runs
  at once, as any launch. Say that scope 9 excludes both, and every replay.
- **L92** ACCEPT — Each point now follows the pinned source:
  - (a) `visit_AnnAssign` binds `lscope` only (704-710), so `visit_then_else_blocks` (834-891) does
    not merge the name. lr-15a's A2 and A3 re-ran;
  - (b) `Autotuner.run` passes `**kwargs, **config.all_kwargs()` to the JIT (`autotuner.py:276-281`),
    and a name given by both is rejected (`_bench`'s check, 142-145, or Python's duplicate keyword),
    so `fusion` is the launch's or the chosen config's;
  - (c) the value of another annotation is visited at 698 and again in `visit_Assign` (712, 757),
    and the annotation is ignored (lr-15b's A1);
  - (d) `tl.static_range` sets the variable to each `constexpr` and restores no scope (1208-1216; S1,
    S2), while `range` and `tl.range` leave the placeholder (`enter_sub_region` restores the
    live-ins);
  - (e) `visit_FunctionDef` visits each default as an `Assign` or `AnnAssign` before `set_value`
    binds the arguments (634-663; H1, H2);
  - (f) TRANSCRIPTION's row and pitfall point to "The contiguity the analysis infers";
  - (g) the checklist numbers the nine points, and the autotune clause reads correctly.
- **L93** ACCEPT — REVIEWER_GUIDE §1 and PROVER_GUIDE §2 list the nine scope points in the order of
  compiler-faithful.md's "Scope" and checklist: 4's heuristics, 5's allocator, 6's `ASTSource` and
  specialisation with host descriptors, 7's H200, 8's tf32 and 9's capture. Both add the autotune
  clause (the last launch, from the memory and with the arguments the last hook leaves) and point to
  the page.

## New issues

- **NEW** soundness — **A Python number that a helper returns, or that is a side of a conditional
  expression on a runtime condition, becomes a scalar tensor of its own type, and SEMANTICS never
  says so.**
  - **The text.** "Helpers are inlined" says a call "means its body", and "Order" says `x if c else y`
    with a runtime `c` "means the runtime `if`". Both read as keeping the Python number. The
    "`return`" bullet says that `return`s "count towards the return type", but gives no rule.
  - **The front end.** `decide_return_type` (`code_generator.py:556-585`) gives a call the common
    type of every `return` the front end visited. That is a compile-time value only when all of them
    are the same value. Otherwise each becomes a tensor of its own type (`to_tensor_type`, then
    `cast_to` and `scalar_constant`, 587-604), and they must all have one type or the kernel is
    rejected. `visit_IfExp` makes each side `to_tensor` (1008, 1015).
  - **Both forms are in the subset**: a helper `if c > 0: return -1` followed by `return -3`, and
    `a if c else b` with a runtime `c`. No in-scope kernel uses either: their conditional expressions
    are on compile-time conditions, and no in-scope helper has two reachable `return`s.
  - **The device** (`work/probes.py`):
    - H1, H2: `_h(c) // 2` stored `0` for `c = 5` and `-1` for `c = -5`. That is the tensor's
      truncating `//`; Python's would give `-1` and `-2`;
    - H6: `(-1 if c > 0 else -3) // 2` stored `0`;
    - H4: `tl.static_assert(_h(c) == -1)` was rejected as not compile-time;
    - H5, H7: with `x` every finite `bfloat16`, `x * _h_f(c)` and `x * (0.1 if c > 0 else 0.2)` gave
      the `float32` product on all 65280 inputs. They differed on 13076 inputs from the `bfloat16`
      product that a model built on "the Python number `0.1`" (SEMANTICS "Constants") gives;
    - H3: two `return`s of the same `-1` stay compile-time, as do evidence/dead-code-round17 E5 and
      E6.
  - **Fix.** State both in SEMANTICS. In "Helpers are inlined" or "`return`", give the rule for the
    call's value. In "Order", say that each side of a runtime conditional expression becomes a
    tensor of its own type (`Operand.ownType?`, `constIn?`), as an assigned number does, and that
    both sides must then have one type. Add both to the "everywhere else" list of "Constants", and
    add a TRANSCRIPTION pitfall. Or exclude both forms under L88: a runtime conditional expression
    with a side that is not a tensor, and a helper whose visited `return`s are not all tensors or
    all the same compile-time value.
- **NEW** wording — Two general sentences of "How the front end evaluates the source" have
  exceptions they do not name:
  - the opening, "What runs on the device is what it evaluates, except the first visit of a loop
    body". The code visited after an `if` that returns also runs nowhere (L89; outside the subset);
  - "Every argument is evaluated before the call". `tl.static_assert`'s `msg` is evaluated only
    when the assertion fails (`execute_static_assert`, `code_generator.py:1617-1637`), as the
    `tl.static_assert` bullet says.

  Fix: name both exceptions where the sentences are.

## Re-ran

- In full, on GPU 0 (NVIDIA H200 NVL, Triton 3.8.0, torch 2.11.0+cu130, 2026-10-28), through
  `work/rerun.sh`, each output identical to the recorded one, the header line included
  (`work/rerun-*.out`):
  - evidence/hint-unread-round17 (N0-N10);
  - evidence/dead-code-round17 (E1-E7);
  - evidence/cdiv-round17 (C1-C6; C1 runs every `int32` for each of the eight divisors);
  - lr-15a's `docs/library-review/round-15/lr-15a/work/probes.py` (B0-B5, A1-A3, K1, K2);
  - lr-15b's `docs/library-review/round-15/lr-15b/work/fe_edges.py` and `dead_after_return.py`
    (D1-D9);
  - lr-16b's `docs/library-review/round-16/lr-16b/work/graph_capture.py` (G1-G3), with an empty
    Triton cache, so that G3 compiles under capture.
- evidence/subset-calls-round17 (no GPU): identical.
- Added `work/probes.py` (`work/probes.out`): H1-H7, the value of a helper with two `return`s and of
  a runtime conditional expression; G0-G3, a call on another stream during a capture. Each case ran
  in its own process with its own Triton cache.
- Added `work/call_scan.py` (`work/call_scan.out`): every node kind, call and keyword of the 23
  in-scope functions.
- Read against `primitives/`:
  - `triton-3.8.0-code-generator.py`: 144-230, 340-430, 440-620, 627-780, 815-1060, 1195-1330,
    1358-1400, 1455-1475 and 1610-1660;
  - `AxisInfo.cpp`: 1220-1345;
  - `ir.cc`: 409-425;
  - `semantic.py`: 118-160, 601-634 and 1775-1791;
  - `standard.py`: 25-45;
  - `core.py`: 51-92, 1228 and 3875-3930;
  - `autotuner.py`: 130-175, 215-285 and 355-375;
  - `jit.py`: 600-615.

  I compared each row and pitfall of TRANSCRIPTION §2-§4 with SEMANTICS and these sources.
- Read the coordinator's release check of the merged tree (11 of 11 steps pass), and did not re-run
  it.

## Not checked

- I did not re-run the release check or `tvr check`, and did not touch `kernels/vec_add_f32`.
- Rounds 18 and 19's changes, beyond reading SEMANTICS "Constants", "Scalar arguments" and
  exclusions.md row 59 for consistency with my area.
- The capture finding with autotuning, descriptors built in the kernel, or a legacy default stream.
- Whether a Triton pass other than the axis analysis reads the `tt.*_arg<N>` attribute that a hint
  on a block argument leaves on an `scf.for` or `scf.while`. The library says no pass reads it, and
  I searched only the pinned `AxisInfo.cpp`.
