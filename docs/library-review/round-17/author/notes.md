# Library batch, round 17: author's notes

Issues: L58, L88, L89, L90, L91, L92, L93 ([ISSUES.md](../../ISSUES.md)), 2026-10-28. The semantics
and transcription batch, run beside rounds 18 (tool and gate) and 19 (numerics) in a separate copy of
TVA, on top of rounds 12 to 16 and round 14 (in review). `lake build TVA` is green. No definition,
statement or proof changed: only docstrings (`TVA.Rules.Compiler.Faithful`, `TVA.Launch.Valid`,
`TVA.Launch`'s `fusion` field, `TVA.Claim.Behaviour`, and the module docstrings of `TVA.Claim` and of
`TVA.Lang.Ops`'s hint section), so every theorem keeps the standard axioms. No line of
`lean/TVA/Lang/Scalar.lean` changed.

**How the batch settles.** Under L88 (a form outside the supported subset closes by exclusion), I
stated `tl.cdiv` (in scope: the in-scope kernels call it 23 times) and the wording points of L92, and
excluded the forms of L89 (code after an `if` that returns, a `return` in a `tl.static_range` body),
L90 (`range`'s keyword and fourth arguments) and L91 (CUDA graph capture). L89 and L91 are excluded by
the user's decision (user, 2026-10-28: "L89, L91 can be excluded", relayed by the coordinator); no
in-scope tutorial kernel uses any of the three forms, and stating L89 would need a complete account of
what a visit of code that never runs leaves behind (this round's evidence found a third effect, on a
helper's return type, beside lr-15b's two). The user plans to end the round-after-round cycle after
round 20, so I aimed each fix at the general case rather than at the reported instances: L58's bound
holds whether or not the compiler reads a hint's mark, L89's exclusion covers every way the front end
visits code after a `return` (compile-time and runtime `if`s, `tl.static_range` copies), and L91's
covers a call under any stream's capture.

**Kernels.** `kernels/vec_add_f32` still passes: `python3 -I tools/tvr.py check vec_add_f32` in this
tree printed `gate PASS digest a9d42d70d2935986` (certified: `401e1fa92db421d7`). Every one of its five
item digests changed (`model`, `map`, `conditions`, `spec`, `assumption:faithful`), so its reviewers
re-open all of them: SEMANTICS.md and exclusions.md, which every issue of the batch but L93 had to
change, are in the digest of every item whose cone reaches the library, and so are
`lean/TVA/Sem/Launch.lean`, `lean/TVA/Claim.lean`, `lean/TVA/Rules/Compiler.lean` (with
`docs/rules/compiler-faithful.md`) and `lean/TVA/Lang/Ops.lean`. No fix as good leaves them alone. I
did not edit the kernel folder (the check rewrote its gate files in this tree only). No other kernel
exists.

**Files of other rounds** (small changes, for the coordinator's merge): `docs/REVIEW_PROTOCOL.md` §6
(one sentence: a call under CUDA graph capture, and a replay, are not launches a claim covers; L91);
`docs/REVIEWER_GUIDE.md` §1 and `docs/PROVER_GUIDE.md` §2, only their lists of the host conditions
(L93, with L91's ninth point). New evidence folders, named for this round:
`evidence/hint-unread-round17`, `evidence/cdiv-round17`, `evidence/dead-code-round17`,
`evidence/subset-calls-round17`. I also re-ran, identical to their recorded outputs apart from the
header line, the reviewers' work the batch cites: `docs/library-review/round-15/lr-15a/work/probes.py`,
`docs/library-review/round-15/lr-15b/work/fe_edges.py` and
`docs/library-review/round-16/lr-16b/work/graph_capture.py` (GPU 0, H200 NVL, Triton 3.8.0).

## L58: Two hint readings are stronger than Triton relies on

- **What changed.** SEMANTICS "Compiler hints", "The contiguity the analysis infers", and the hint
  section of `TVA.Lang.Ops`: a value that carries a `max_contiguous` hint, alone or in the pair, has
  as its bound the smaller of the hinted `C` (of the smallest, with two such hints) and the bound the
  list gives the value as if it carried no hint, and values built from it start from that bound. The
  text says why: the compiler's contiguity is `C` when it reads the mark, and the analysis' own when
  it does not, which happens on a `for` or `while` loop's or a runtime `if`'s result, on the value an
  inlined call returns, on a result of an operation with several results in some cases, and on an
  operation the canonicalizer folds away. The several-result bullet of "Which value" points to it.
  TRANSCRIPTION §2's `tl.multiple_of` row and §3's hint pitfall say the same, and take the bound "for
  how `x` is computed" (with L92). I took the smaller of the two rather than lr-15a's "limit the
  bullet to values whose defining operation's mark the compiler reads": the evidence shows that an
  ordinary one-result operation also loses its mark when it folds (N9, N10), and which ones fold is
  the canonicalizer's choice, so no list of such operations would be safe; the smaller bound is a
  lower bound in every case, and equals lr-15a's for the cases it names (the list gives `1` for a
  loop's, an `if`'s and `tl.split`'s result), at the cost of a smaller bound where a hint raises the
  contiguity above the list's.
- **Why.** lr-15a's reopen: round 15 gave the hinted `C` to a value "whatever operation defines it",
  but the analysis reads a hint only off an operation it visits (`AxisInfo.cpp:1232-1259`, and
  1309-1341 for an operation no visitor handles); the results of `scf.for` and `scf.if` are joins of
  what their regions yield, inlining removes a `tt.call` with its mark, and a fold erases the marked
  operation. For those values the stated bound was above the compiler's, and `FirstsDiv` with it
  promised less than the compiler assumes.
- **Evidence.** [evidence/hint-unread-round17](../../../../evidence/hint-unread-round17/README.md)
  N0-N10 (512 `int8` loads through `x + tl.multiple_of(r + s, 16)`, one warp): with the pair
  `C = v = 16` on `r`, 16-byte loads when an add defines `r` (N1) and 1-byte loads, as with no hint
  (N0), when `r` is a loop's (N2), a runtime `if`'s (N3), an inlined helper's (N4) or `tl.split`'s
  (N5) result, or `tl.load(…) + 0` (N9) or `* 1` (N10); with the pair `C = 4` on an inlined helper's
  `tl.arange(0, 512)`, 16 bytes, the arange's own bound (N7), against 4 bytes on the arange directly
  (N6) and when the helper's body hints it (N8). Values correct in all eleven. This repeats lr-15a's
  B0-B5 (re-run identical) and adds the inlined call's bound and the folds.
- **Constants whose meaning changed.** The reading of the `contig` argument of `TVA.tl.multiple_of`
  and `TVA.tl.multiple_ofN` (the hint section of `TVA.Lang.Ops`); no definition changed.

## L88: A supported subset of the Python front end

- **What changed.** SEMANTICS has a new bullet, "`tl.cdiv(x, div)`" (after "Integer `//`, `%`, `<<`,
  `>>`"): `tl.cdiv` and `x.cdiv` are Triton's `@jit` body `(x + (div - 1)) // div`, inlined as a
  helper, so on two compile-time values Python's expression at compile time, a compile-time value,
  and with a tensor operand the wrapping `+`/`-` and the truncating `//` (`tl.idiv`, undefined for a
  zero divisor or an overflowing quotient) in the promoted dtype. The subset's "Calls" item names it.
  TRANSCRIPTION §2 has a `tl.cdiv` row (with the `Inst.ok` bound a model needs for the ceiling) and §3
  a pitfall. The list was checked once more against every call of the in-scope kernels (below):
  nothing else lacks a meaning.
- **Why.** lr-15a's reopen: the in-scope kernels of tutorials 03, 06, 08, 09 and 10 call `tl.cdiv`,
  which neither SEMANTICS nor `TVA.Lang` gave a meaning, and which is not the ceiling
  (`standard.py:32-43`). I stated it in SEMANTICS rather than add a `TVA.tl.cdiv`: its meaning is the
  page's existing operators, which a model writes with `tl.idiv` and the integer operations.
- **Evidence.** [evidence/cdiv-round17](../../../../evidence/cdiv-round17/README.md): for every
  `int32` `x` and eight constant divisors (`1, 2, 3, 4, 7, 128, 1024, -4`; `2^32` inputs each), 16.8
  million runtime pairs with their edges, runtime scalars, `int64` and the method form, the device's
  result was the expression's in every case (`tl.cdiv(2^31 - 1, 128)` is `-16777215`, `tl.cdiv(-5, 4)`
  on an `int32` is `0`); on `tl.constexpr`s it was Python's (`tl.cdiv(-7, -2) = 5`), accepted by a
  `tl.static_assert` and as `tl.arange`'s bound.
  [evidence/subset-calls-round17](../../../../evidence/subset-calls-round17/README.md): an AST scan of
  the 23 in-scope functions lists 37 `tl` names, 11 methods and attributes, 4 builtins, 6 helpers and
  the keywords given, and finds each given a meaning by SEMANTICS or `TVA.Lang` (`tl.num_programs` and
  `tl.zeros` by their `TVA.Lang.Ops` definitions).
- **Constants whose meaning changed.** None (no Lean constant gives `tl.cdiv` a meaning; the model
  writes its operations).

## L89: Code after a `return` taken through a compile-time `if` is still compiled

- **What changed.** Excluded under L88, by the user's decision (user, 2026-10-28: "L89, L91 can be
  excluded"). SEMANTICS "`return`" says what the front end does: it stops
  only at the statements after a `return` in the same statement list (not compiled), and visits what
  follows the statement holding the `return` into a block no program reaches, after a compile-time
  `if` whose taken branch returns, after a `tl.static_range` copy that returns, and after a runtime
  `if` whose two branches return; that code never runs, but its hints mark values defined before the
  `return`, its `tl.static_assert`s are evaluated, and in a helper its `return`s take part in the
  return type. The subset's `return` item excludes a `return` in a `tl.static_range` body and any
  statement that follows, in the same statement list, an `if` that returns on the instance (defined
  recursively: a compile-time `if` whose taken branch has a `return` or such an `if` among its own
  statements, a runtime `if` whose two branches each have one); the subset's list of excluded forms,
  a new exclusions.md row and a TRANSCRIPTION §3 pitfall say the same. The old sentence "Code after a
  `return` at the top level is not compiled" is replaced by the exact one.
- **Why.** lr-15b: the natural reading of the old sentence drops that code, while its hints apply
  (`code_generator.py:508-517, 544-554`), so `Faithful` was false for a model that drops it. I
  excluded rather than stated it: no in-scope kernel has such code (`_maybe_make_tensor_desc` returns
  in both branches with nothing after), and the evidence found one more compile-time effect than the
  issue lists, so a statement would need to be complete about everything such a visit can leave.
- **Evidence.** `docs/library-review/round-15/lr-15b/work/dead_after_return.py` D1-D9 (compile-time
  `if` and `tl.static_range`), and
  [evidence/dead-code-round17](../../../../evidence/dead-code-round17/README.md) E1-E7: after a
  runtime `if` whose two branches return, a pair hint made an earlier load `ld.global.v2.b32`, which
  read wrong values in 137 of 256 lanes (E1; E2 without it correct), and a `tl.static_assert(False)`
  rejected the kernel (E7); an atomic after a taken compile-time `return` ran in no program (E3); a
  helper `if FLAG: return -7` then `return 0` returned an `int32` tensor (`// 2` gave `-3`, E4), where
  with `return -7` twice or once it stayed a compile-time value (`-4`, E5, E6).
- **Constants whose meaning changed.** None.

## L90: `range`'s keyword and fourth arguments

- **What changed.** The subset's `for` item is now Python's `range` "given one, two or three
  positional arguments and no keyword"; the subset's excluded forms, SEMANTICS "Loops and unrolling"
  ("The iterator's arguments": evaluated and ignored), a new exclusions.md row and TRANSCRIPTION §3's
  loop pitfall name the keyword and fourth-argument forms; `tl.range` and `tl.static_range` keep their
  keywords.
- **Why.** lr-15b: `visit_For` evaluates the keywords (`code_generator.py:1207`) but uses only
  `iter_args[0..2]` for Python's `range` (`:1237-1243`), so a model reading `step=` as the step differs
  from the device.
- **Evidence.** `docs/library-review/round-15/lr-15b/work/fe_edges.py` K1-K4, re-run identical: with
  `n = 8`, `range(0, n, step=2)` and `range(n, step=2)` ran 8 trips, `range(0, n, 2, 99)` 4, and
  `tl.range(0, n, step=2)` 4.
- **Constants whose meaning changed.** None.

## L91: CUDA graph capture

- **What changed.** Excluded, by the user's decision (user, 2026-10-28: "L89, L91 can be
  excluded"). `Compiler.Faithful` has a ninth scope point, "No CUDA graph capture" (its docstring and its page, with "What it says", "Which launch",
  "What it does not cover" and the checklist): the call is not made while a CUDA stream capture is in
  progress (`torch.cuda.graph`, `CUDAGraph.capture_begin`, `triton.testing.do_bench_cudagraph`, a
  `cudaStreamBeginCapture`), and the launch is not a replay; the rule covers neither. The exclusion
  names a capture in progress on any stream, not only the launch's own: that is wider than "its
  stream is capturing", so conservative, and spares the reviewer the capture mode. The same is in
  `TVA.Claim` (module docstring and `Behaviour`), `Launch.Valid`'s docstring (a valid launch runs its
  whole grid when it returns without an error, outside capture), SEMANTICS "One launch", its scope
  list and its outcome bullet, exclusions.md (new row), protocol §6 (one sentence), the guides' host
  lists (L93) and TRANSCRIPTION's host pitfall.
- **Why.** lr-16b: under capture `k[grid](…)` returns without an error and runs no program, so if
  that call were the launch, `Faithful` would make "ok, memory unchanged" a model outcome, false for
  any kernel that writes; the replays run with the captured arguments on the memory of the replay,
  which no text stated. Stating replays would need TVA's own evidence (descriptors, autotuning and
  PDL under replay included); no in-scope tutorial's host captures (tutorial 11 is excluded).
- **Evidence.** `docs/library-review/round-16/lr-16b/work/graph_capture.py` G1-G3, re-run identical:
  the sentinel kept after the captured call, `x + 1` after the first replay, `11` after the host set
  `x = 10` and replayed; a captured `n = 16` wrote 16 elements whatever the host's `n` later was; a
  capture with no warm-up compiled and ran nothing.
- **Constants whose meaning changed.** `TVA.Rules.Compiler.Faithful` (its docstring: scope 9 narrows
  what the rule covers); `TVA.Launch.Valid` and `TVA.Claim.Behaviour` (docstrings; definitions
  unchanged).

## L92: Wording in the front end and transcription

- **What changed.** (a) SEMANTICS "A runtime `if`": a name new to the `if` that a branch binds with
  `name: tl.constexpr = e` belongs to its branch, also when both do (not merged); TRANSCRIPTION's
  annotation pitfall too. (b) `TVA.Launch`'s `fusion` docstring: the value given to the call that
  compiles the kernel, the launch's keyword, `warmup`'s or `triton.compile`'s option, or the chosen
  `triton.Config`'s (the old "the source's" is gone: `@triton.jit` takes no such option).
  (c) exclusions.md's annotation row and SEMANTICS "Annotated names": another annotation evaluates its
  value twice and is otherwise ignored. (d) exclusions.md's loop-variable row, SEMANTICS "Names after a
  loop" and the subset's "Names" item: the placeholder sentence is for `range` and `tl.range`; after
  `tl.static_range` a name holds the last copy's value, a name first defined in the body is defined
  and the variable holds its last value, and the subset admits neither of the last two uses after any
  loop. (e) SEMANTICS "Helpers are inlined": the front end visits each default as an assignment
  before the body, so an unannotated default it cannot make a tensor (a string, an integer beyond 64
  bits) rejects the helper even when the call passes the argument; and the subset's "Parameters and
  arguments" item now admits as a default only a literal, a tuple of literals or a dtype of `tl`
  (no in-scope helper has a default), so that the expression the front end visits and the value
  Python bound at the `def` are the same. (f) TRANSCRIPTION §2's
  `tl.multiple_of` row and §3's hint pitfall take the bound for how the value is computed, citing
  "The contiguity the analysis infers". (g) compiler-faithful.md's checklist: the host item is a
  numbered list of the nine scope points, and the autotune clause reads "its arguments those the last
  hook leaves".
- **Why.** lr-15a's and lr-15b's wording points; each statement follows the pinned source:
  `visit_AnnAssign` writes `lscope`, not `local_defs` (`code_generator.py:704-710`), and visits the
  value before `visit_Assign` visits it again (`:698, 712, 757`); `Autotuner.run` passes
  `config.all_kwargs()` to the JIT (`autotuner.py:276-281, 365-375`); `visit_For` sets a
  `tl.static_range` variable to each `constexpr` in turn (`:1208-1216`); `visit_FunctionDef` visits
  the defaults (`:634-649`).
- **Evidence.** `docs/library-review/round-15/lr-15a/work/probes.py` A2, A3 and
  `docs/library-review/round-15/lr-15b/work/fe_edges.py` A1, S1, S2, H1, H2, both re-run identical
  (A2: "c is not defined"; A1: 16 atomics in 8 programs; S1: `2`; S2: `20`; H1 rejected, H2 ran).
- **Constants whose meaning changed.** `TVA.Launch` (the `fusion` field's docstring: which value it
  is under autotuning); no definition changed.

## L93: The guides' lists of the host conditions are incomplete

- **What changed.** REVIEWER_GUIDE §1 ("The host code") and PROVER_GUIDE §2 ("The host stays within
  `Compiler.Faithful`'s scope") give the full list, numbered as the rule's scope (nine points, with
  L91's capture), and the autotune clause (the last launch, from the memory and with the arguments
  the last hook leaves), and point to the rule page's "Scope" and its checklist, which list the same.
- **Why.** lr-16b: REVIEWER_GUIDE left out scope 5, scope 6's arguments of the specialisation and the
  autotune clause, and PROVER_GUIDE also `@triton.heuristics`, the `ASTSource` of `triton.compile`, the
  H200 target and `round_f32_to_tf32`.
- **Evidence.** None needed (wording; the list is compiler-faithful.md's "Scope").
- **Constants whose meaning changed.** None.

