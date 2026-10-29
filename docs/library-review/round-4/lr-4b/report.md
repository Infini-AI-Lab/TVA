# Library review, round 4, by lr-4b

## Verdicts

- **L7** REOPEN — The new `fpv/<name>` rule (`_FPV`) and the home-path rule catch every round-3
  spelling: `evidence/independence-audit` re-runs as recorded (79 probes, 0 wrong), lr-3a's
  `l7_more_spellings.py` flags all 7, lr-3b's `l7_probes.py` gives 17 probes, 0 wrong, and 12
  further `fpv` spellings of mine are flagged (doubled slash, `./`, `tva/../`, `\`, `%2f`, `&#47;`,
  `/`, `\x2f`, fullwidth solidus, upper case; `work/l7_probes4.py`, `.out`). But the
  coordinator's rule ("flag any path segment `fpv/<name>` … wherever it occurs") is not applied to
  the files `upstream_files()` lists: `scan_files` calls `foreign_refs(…, paths=False)` on them, and
  protocol §2 says they are "checked for names only". That list includes each kernel's source,
  which the kernel prover writes and pins in `kernel.json`. On a clean copy, `tvr audit` passes a
  kernel source holding the comment `# from ~/fpv/<sibling>/vec_add.py` with its sha re-pinned
  (E6), and a new file in `primitives/` holding a sibling path once its sha is added to
  `semantics-manifest.json` (E5); the gate's `independence` check reads the same code. A kernel
  file can thus point into another project and pass. Smaller misses, also end to end on the copy:
  shell quoting splits the literal segment (`~/"fpv"/<sibling>`, `~/fpv/"<sibling>"`, E1, E2;
  also `fpv''/`, `'<sibling>'`, `{<sibling>,x}`, `[o]ther…`), the entity `&sol;` (E3), and home
  spellings §2 promises but the tool misses (`${HOME:-${X}}/…`, `` `echo ~`/… ``, a `$(…)` nested
  three deep, `%USERPROFILE%\…`, PowerShell `$HOME\…`). Fix: run the `fpv/<name>` and home-path
  tests on every file, pinned ones included (exempt pinned files only from the generic
  absolute-path test; none of the 45 pinned files trips the `fpv` or home test), and say so in §2;
  strip shell quotes and decode named entities before the path test, and accept `\` after a home
  prefix.
- **L12** ACCEPT — `transcription_rules` now ends a rule at the first unindented line, which is what
  the page's new paragraph "Where a rule ends" says. A dump of the parser's split
  (`work/gate_rule_split.py`, `.out`) gives 44 rules, with the closing paragraph of §3 and the whole
  of §9 in the general text and no page problem. The continuation check reports a lazy
  continuation line, which Markdown would show inside the rule (`work/gate_rule_cases.py`, `.out`).
  Re-ran `evidence/gate-digests` on a temporary copy: identical to `out.txt`, case I included (a
  change to that paragraph changes `map` and `model` of a kernel whose map does not name
  `[atomic]`). Two splits still differ from CommonMark without a report: a line indented by one
  space after a blank line, and an unindented `2.` right after a rule. Neither occurs on the page
  today.
- **L17** ACCEPT — Scope point 1 ("the launch runs alone") in the hashed `Faithful` docstring, the
  rule page, `Claim`'s module docstring, `Claim.Behaviour`, `Launch.pdl`, SEMANTICS "Launches", §9
  and exclusions.md, and the split `[no-op:gdc]`, close the hole lr-3a and lr-3b found: the drop of
  `gdc_launch_dependents()` now rests on a condition the kernel reviewer checks in the host code
  (no next launch with `launch_pdl=True`). `gdc_wait()` stays dropped because a valid launch has
  `pdl = false`, so stream order finishes the previous kernel first. The scope is stronger than the
  hazard (any other writer anywhere), which is sound and matches a `Post` that says every other
  byte is unchanged. Re-ran `evidence/pdl` A–D: part D gives `y` wrong in 20 of 20 runs only with
  the call and a PDL successor, 0 of 20 in the three other variants; A–C as recorded (part C's
  stale count differs, as a race does). The changed docstrings are listed.
- **L18** ACCEPT — Scope point 2 covers every knob of `knobs.py` (`build`…`nvidia`, read in the pinned
  file) however it is set, with or without `propagate_env`, and point 3 names every hook of
  `runtime_knobs` (`knobs.py:470-492`), both listeners and the `JITFunction` pre-run hooks, which
  `jit.py:727-736` runs before the launch. The exceptions only name directories, dump, print or
  time, or change debug information. `Launch.debug` names `knobs.runtime.debug` (`jit.py:727` ORs
  it into `debug`). Re-ran `evidence/launch-options`: D0–D4, F0/F1, S0/S1, A2, A3 identical to
  `out.txt` (D3: `fp32_default` with `propagate_env = False` makes the dot `ieee` with no variable
  set; A3: a `jit_cache_hook` returns with nothing launched).
- **L19** ACCEPT — Point 4 allows only `enable_fp_fusion`, `debug`, `num_ctas`, `launch_pdl` and the
  four options that do not change the meaning, and excludes every other `CUDAOptions` field
  (`nvidia-compiler.py:108-135`, all 23 fields checked against the list: 8 allowed, 15 named as excluded); `_pack_args`
  (`jit.py:713-715`) rejects any other keyword, so the list is complete. Re-ran A1 (`ir_override`
  gives `x + 2`), D1 (`allowed_dot_input_precisions` makes the dot `ieee`), E1
  (`enable_reflect_ftz=False` gives `exp2(-130) = 2^-130`) and B (9 variants, 0 differing
  elements): identical. `num_warps` and `num_stages` change only what the semantics already leaves
  open (reduction trees, `P.dot`, the order of a program's accesses, which a well-behaved model
  cannot observe). One consequence for L4 (open): `num_warps` selects the `tl.dot` instruction
  (`wgmma` needs 4 warps), so a future `DotAcc` constant must hold for every `num_warps` a config
  may give.
- **L20** REOPEN — The user's half is done: the kernel-reviewer prompt, REVIEWER_GUIDE §1,
  protocol §2 and TRANSCRIPTION §1 tell the reviewer to read the whole Lean and judge what every name
  means. The gate's half works on the author's cases: `evidence/gate-digests` L–P re-run
  identically (the shadowing `def tl.add`, a `local notation`, an `HAdd F32` instance and a
  `theorem lane_tile` under `open Block` fail). But the check the issue asks for and protocol §8
  states ("the claim reader checks the elaborated environment, whatever the source syntax") misses
  two shadowing routes. On a copy with `vec_add_f32` updated and the probes appended, `tvr check`
  flags only the control (an `abbrev F16`), none of the probes (`work/gate_probes.py`, `.out`; that
  each probe changes what model text means is checked in Lean, `work/gate_resolve.lean`, `.out`):
  - P1, `def Launch.mk …` in the kernel's namespace: the claim reader treats any declaration whose
    last component is on its list of generated names (`mk`, `ext`, `inj`, `rec`, …,
    `claim_reader.lean.in:173-175`) as generated, so it is never compared with library names. Model
    text `Launch.mk … false …` then means the kernel's definition, for example one that sets
    `debug := true`.
  - P5, `open TVA` with `Block` on the next line: `opened_names` reads an `open` command to the end
    of its line only (`_OPEN`, `tvr.py:684`), while Lean reads names on later lines too. So
    `lane_tile`, case O, is not flagged.

  The gate also passes routes TRANSCRIPTION §1 does not name, which the reviewer must catch: an
  instance on non-library types that changes model arithmetic (`instance : HDiv Nat Nat Nat`, which
  changes the grid formula `(n + B - 1) / B`), a binder named `tl` in `model`
  (`(tl : Ops := ⟨TVA.tl.addc⟩)`), and the look-alike `tℓ.add`. Fix: exempt only names Lean
  generated (constructors, recursors, equation lemmas: check the constant's kind, as is done for
  `ctorInfo`/`recInfo`), not a `def` the kernel wrote; read the opened namespaces from the
  elaborated environment, or treat every library namespace as opened; and, as a backstop, flag every
  instance the kernel declares (generated `_sizeOf_inst` aside), binders of `model`/`launch` named
  like a library namespace, and non-ASCII names. TRANSCRIPTION §1 and REVIEWER_GUIDE should then say
  "any instance", binders and look-alikes. Minor: `set_option pp.notation …` is refused by the
  notation pattern, although §8 allows `pp.*` (`work/gate_hygiene.py`, `.out`).
- **L23** ACCEPT — `FloatV.litF64`, `litF32`, `litF16`, `litBF16` and `round6` follow
  `scalar_constant` (`semantic.py:601-613`) and the builders (`ir.cc:952-972`): `round6` is C's
  `%f` with ties to even (floor, then the doubled remainder against the denominator; symmetric for
  negative values), and a zero text keeps its sign. `[literal]` splits by where the number is used,
  which matches the source: binary operators and `tl.where` pass the Python number to
  `binary_op_type_checking_impl` (`semantic.py:180-206`, `core.py:2837-2856`), `tl.full` to
  `make_scalar`, while `tl.maximum` and the math functions call `to_tensor` first
  (`core.py:2947-2950`, `math.py:198-208`). Re-ran `evidence/literals`: 0 differences in every column
  for 9364 literals, identical to `out.txt`. I added the fp8 case the evidence read only in the source
  (`work/l23_fp8_const.py`, `.out`): a constant not exactly representable in `float8e5` or
  `float8e4nv` (50 and 56 values, all rejected: ties, between neighbours, beyond the largest finite value, below
  the smallest subnormal, `±inf` for `float8e4nv`) makes the compile fail on the host (`llvm.fptrunc`
  to `i8`), a rejected launch with no outcome; an exact one is folded to itself and equals the
  run-time `.to(fp8)` (24 and 18 values), except `±inf` in `float8e5`, which the constant keeps as
  `±inf` while the run-time conversion saturates to `±57344`. `CvtAcc` says nothing about infinite
  inputs, so `litF32` then `tl.cvt8` still allows the constant. The rule is sound; one stale
  sentence in `TVA.Lang.Scalar` is filed below as wording.
- **L24** ACCEPT — `[itof-narrow]` is limited to the default rounding (none, or `"rtne"`, which
  `cast` turns into no rounding attribute, `semantic.py:822-826`), and `"rtz"` goes through
  `create_fp_to_fp` (`:835-840`), which no MLIR fold touches; the model `tl.itof` then `tl.cvt … .rz`
  is what runs. Re-ran `evidence/itof-rtz` over all `2^32` `int32` inputs: 0 differences for both
  rtz chains against the new rule, the old rule's counts and the default-rounding controls as
  recorded, the TTIR with 2 `tt.fp_to_fp` and no `arith.truncf`.
- **L25** REOPEN — Dropping a static branch's `if`/`elif` line only for a `[pure]` condition is
  right, and `evidence/config-cond` re-runs identically. But the new text for an impure condition,
  "its operations are model lines, as [pure] says", with `[pure]`'s "transcribed as model lines, in
  Python's order (left to right), as if each stood alone", gives a model with operations the device
  never runs. The front end short-circuits `and`/`or` at compile time when a non-tensor operand
  decides (`visit_BoolOp`, `code-generator.py:1481-1501`), and a conditional expression with a
  compile-time condition evaluates one side only (`visit_IfExp`). The author's own C4 shows it (the
  atomic after `FLAG and` does not run), and the README concludes the reviewer "does not need to
  know which parts run", which holds only for the pure case. `work/l25_shortcircuit.py` (H200,
  4 programs, counter = atomics that ran, against what the rules transcribe): K1
  `if FLAG and tl.atomic_add(c, 1) is not None:` (`FLAG = False`): 0, the rule's model 4; K2
  `if (tl.atomic_add(c, 1) is None) and (tl.atomic_add(c, 1) is None):` 4, model 8; K5
  `if (tl.atomic_add(c, 1) if FLAG else None) is None:` 0, model 4. The same flaw reaches
  `[compile-check]` (K3, `tl.static_assert(FLAG or tl.atomic_add(c, 1) is None)`: 0, model 4) and
  `[no-op:print]` (K4, `tl.static_print(tl.atomic_add(c, 1) if FLAG else 0)`: 0, model 4). A model
  with an atomic the device skips makes `Faithful` false and lets a wrong `Post` through. Fix: in
  `[pure]`, transcribe only the operations the front end evaluates: the right operand of `and`
  (`or`) is skipped when the left one is a compile-time value that is false (true), and a
  conditional expression with a compile-time condition evaluates only the side it picks (or forbid
  impure operands after a compile-time `and`/`or` operand and inside a compile-time conditional
  expression); add these cases to `evidence/config-cond`.
- **L26** ACCEPT — `[sum-bcast-mul]` is now stated by the computation, and it can be, because
  `tl.sum_bcast_mul` allows both results: a tree of `AccAdd E.fusion` over the IEEE products
  (`MulRel`), or `P.dot` with `precision := .ieee`, a `+0` accumulator and `maxImpreciseAcc = 0`,
  which is the `tt.dot` the pattern builds (`Combine.cpp:111-195`: rank-3 reduce on axis 1 whose
  combine region is one `addf` of at most 32 bits, a `mulf` of two broadcasts of `expand_dims` on
  axes 2 and 0, `M, N ≥ 16`). Spellings the pass does not match (swapped factors, `tl.reshape`,
  `a - (-b)` as the combine) are IEEE or fusable trees, which the first disjunct covers; the
  `float16`/`bfloat16` rewrite builds an `f32` splat into a 16-bit accumulator and is rejected.
  `safe_sum_bcast_mul` states exactly the definition's two disjuncts. Re-ran `evidence/bcast-mul`:
  identical to `out.txt`.
- **L27** ACCEPT — SEMANTICS.md and exclusions.md are in the general text of every `map` and `model`
  digest, and the claim reader hashes the whitespace-normalised docstring of every `TVA` constant
  in a cone. Re-ran `evidence/gate-digests` J, K and K2 on a copy: identical (a SEMANTICS sentence
  changes `map` and `model`; `tl.add`'s docstring changes `model`; a whitespace-only change changes
  nothing). A constructor's docstring is covered too: changing `TVA.Step.defer`'s re-opens
  `assumption:faithful` (`work/gate_docs.py`, `.out`, D2). The issue is fixed. What remains outside
  every digest, module docstrings and the docstrings of fields whose projection no term uses, is
  filed below as a new issue.

## New issues

- **NEW** soundness — Library text that states meaning is still outside every digest, so a change
  to it re-opens no kernel item (the class of L27). (1) Module docstrings (`/-! … -/`) are hashed
  nowhere, and some state readings found nowhere else: `TVA.Lang.BlockPtr`'s says `order` "does not
  change values" (`[block-ptr]` and SEMANTICS.md do not mention `order`); `TVA.Claim`'s states the
  claim's scope. On a copy, a change to `TVA.Claim`'s module docstring re-opens nothing
  (`work/gate_docs.py`, `.out`, D3). (2) A structure field's docstring is hashed only when some term
  in the cone uses its projection. A model that builds the structure reaches only its constructor:
  a change to `TVA.Launch.blocks`'s docstring re-opens `assumption:faithful` (through `Launch.Valid`)
  but not `model` (D1). Nothing in the library projects `DotCfg.precision`, whose docstring L18
  changed and the notes list as changed, so that change re-opens no item at all. Today `[dot]` and
  §9 repeat the field's meaning, so no obligation is wrong yet. Fix: when a cone reaches a structure
  or its constructor, add every field's projection (and its docstring); hash the module docstrings
  of the modules a cone reaches, or move their readings into constant docstrings or SEMANTICS.md.
- **NEW** gap — No error bound for a reduction with fusable additions. `[fusion]` sends every
  reduction or scan a product may feed to `tl.sumc`/`sum1c`/`sum0c`/`cumsumc` (nodes
  `AddRel E.fusion`), and `tl.sum_bcast_mul`'s first disjunct is such a tree too (`AccAdd`). But `sum_err`, `sum_err_covers` and `sum_err_list`
  (`Lemmas/Sum.lean`, `Lemmas/Scan.lean`) take `AddRel false` nodes only, and `Tree.eval_err` asks of
  each node a bound relative to `|a + b|`, which `AddRel true` cannot meet: with `a = 1`, `b = -1`
  the fused result may be `≈ 2^-24` while `a + b = 0`. `FloatV.AddRel.err` bounds one node by
  `3u·(|x| + |y|) + 4η`, relative to `|x| + |y|`, and no tree lemma takes such a bound. So every
  kernel that sums products (a row's `tl.sum(x * x)` in a norm, `tl.sum(a * b)` in a dot product, a
  `tl.cumsum` of products) must prove its own tree lemma. Fix: a `Tree` lemma for per-node bounds
  `δ·(|a| + |b|) + η` (the bound grows with the absolute sum of the leaves and the depth), and
  versions of `sum_err`, `sum_err_covers` and `sum_err_list` for `AddRel E.fusion` (and `AccAdd`).
- **NEW** gap — The shared-atomics kit (`Lemmas/Shared.lean`) covers only a scalar, unmasked atomic
  (`SafeA.atomic1`) and only `int32` addition (`AddsTo`, `replay_add`), though exclusions.md says
  "accumulation through atomics is covered by `TVA.shared`". A tensor atomic on the shared region
  with a mask, `tl.atomic_add(c_ptrs, acc, mask=m)` (split-K into a shared output tile), unfolds to
  `atomicWith`'s choice of junk and lane order and an `rmwSeq` (`Lang/Ops.lean:205-219`), for which
  `SafeA` has no rule (the private case has `safe_atomic_buf`). Nor is there an accumulation lemma
  for a float `atomic_add` (`P.atomicAdd`, bounded per step by `Rules.Num.AtomicAddAcc`) replayed in
  any order, nor for other widths or `atomic_max`/`atomic_min`. Fix: a `SafeA` rule for a masked
  tensor atomic on cells of `C` (one log entry per enabled lane, any order), and replay lemmas for
  float addition (a bound on the final cell from `AtomicAddAcc`, for example through
  `DotLoop.accum` along the log) and for the other integer widths and `max`/`min`.
- **NEW** wording — The module docstring of `TVA.Lang.Scalar` ("Python number constants") says a
  Python number meeting a tensor "in an operation (an operator, `tl.where`, `tl.maximum`, …)" is
  built "directly in the operation's dtype". For `tl.maximum` that is false and contradicts
  `[literal]`: `tl.maximum` calls `to_tensor` first (`core.py:2947-2950`), so
  `tl.maximum(x_bf16, 1e-7)` compares with the `float32` `1e-7` after promotion, not with
  `litBF16 1e-7 = +0`. Fix: drop `tl.maximum` from that list and refer to `[literal]`'s two cases.

## Re-ran

- On GPU 0 (NVIDIA H200 NVL, Triton 3.8.0 from `.venv`, torch 2.11.0+cu130, 2026-10-27), with
  `work/rerun_gpu.sh`: `evidence/config-cond`, `literals`, `launch-options`, `pdl`, `bcast-mul` and
  `itof-rtz`. Each `work/rerun_<topic>.out` is identical to its `out.txt`, except the stale-element
  count of `pdl` part C (a race; still 19 of 20 runs).
- On temporary copies of the tree: `evidence/gate-digests` (A–P identical to `out.txt`,
  `work/gate_rerun_gate-digests.out`) and `evidence/independence-audit` (79 probes, 0 wrong,
  `work/l7_rerun_independence-audit.out`); round 3's `lr-3a/work/l7_more_spellings.py` (7 of 7
  flagged) and `lr-3b/work/l7_probes.py` (17 probes, 0 wrong).
- Tests added in `work/`, each with its `.out`:
  - `l25_shortcircuit.py` (L25: five impure conditions and arguments with compile-time
    short-circuits, on the device);
  - `l23_fp8_const.py` (L23: fp8 constants, 106 rejected inexact ones, 42 exact ones against the
    run-time conversion);
  - `l7_probes4.py` (L7: 41 probes of `foreign_refs`, 12 wrong; 7 end-to-end `tvr audit` cases on
    copies, 6 wrong; controls pass);
  - `gate_probes.py`, `gate_resolve.lean` (L20), `gate_rule_split.py`, `gate_rule_cases.py` (L12),
    `gate_docs.py`, `gate_hygiene.py` (L27, L20), on copies of the tree;
  - `lean/Axioms.lean`: all 2210 theorems declared in `TVA` modules depend only on `propext`,
    `Classical.choice` and `Quot.sound`.
- Survey (`lemmas`): read the statement and docstring of every theorem and definition in the 25
  files of `lean/TVA/Lemmas` against `docs/LIBRARY.md`; every name LIBRARY.md's proof-kit table gives
  exists. Checked `Block` (`Fits`, `fits_of_grid*`), `Claim`, `Safe`, `Independent`, `Shared`, the
  `Ops` rules (each against its `tl.*` definition: loads, stores, IEEE operations, conversions,
  reductions, index reductions, `dot_add`, `sum_bcast_mul`, loops), `Num`, `Tree`, `Sum`, `Fused`,
  `DotLoop` (`dotAdd_step`'s two forms against `tl.dot_add`), `Softmax`/`SoftmaxMax`, `Scan`, `Desc`,
  `BlockPtr`, `Access`, `Scalar`, `Int` and `Consistency` (`Platform.ref` and each satisfiability
  lemma against its `Rules.Num` statement).
- Read in `primitives/`: `semantic.py` (promotion, `to_tensor`, `scalar_constant`, `cast`, `where`),
  `core.py` (`where`, `maximum`, `full`, operators), `math.py`, `ir.cc` (constant builders),
  `llvm-ArithOps.cpp` (`TruncFOp` fold, `convertFloatValue`), `Combine.cpp`, `code-generator.py`
  (`visit_BoolOp`), `nvidia-compiler.py` (`CUDAOptions`), `jit.py` (`run`, `_pack_args`), `knobs.py`,
  `nvidia-ElementwiseOpToLLVM.cpp`; and `tools/tvr.py` (`upstream_files`, `scan_files`,
  `foreign_refs`, `opened_names`, `transcription_rules`) and `tools/claim_reader.lean.in`.

## Not checked

- The Lean proofs of the lemmas (Lean checks them); I judged statements, docstrings and axioms.
- The excluded compile options other than `ir_override`, `allowed_dot_input_precisions`,
  `default_dot_input_precision` and `enable_reflect_ftz` (`extern_libs`, `ptx_options`, `arch`, …):
  read in the source, not run.
- Whether `num_warps` changes `tl.dot`'s accuracy for floating-point operands (it matters for L4's
  future `DotAcc` constants, not for this batch).
- PDL, the literals and the other device results on GPUs other than the H200.
- `tvr check` on the real tree (coordinator's instruction); the gate ran only on copies.
