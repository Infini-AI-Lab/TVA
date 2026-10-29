# Library batch, round 8: author's notes

Issues: L1, L2, L5, L21, L28, L29, L30, L37, L44, L45, L47, L49 ([ISSUES.md](../../ISSUES.md)),
2026-10-28. The batch is semantics coverage: each issue gets a stated meaning, on TVA's own evidence
(Triton 3.8.0, torch 2.11.0+cu130, NVIDIA H200 NVL, GPU 0), in `docs/SEMANTICS.md`, the Lean
definitions and docstrings, `docs/rules/compiler-faithful.md` and `docs/exclusions.md`, with
guidance in `docs/TRANSCRIPTION.md`.

The library builds (`lake build TVA`). New Lean: the module `TVA.Lang.Promote` (L45), the hint
predicates `ContigAlong`, `ContigIs`, `FirstsDiv`, `ConstAlong` (L2), `tl.imaximum`/`tl.iminimum`
(L45), `tl.make_tensor_descriptor` and its lemma (L5). One definition is stronger than before:
`tl.max_contiguous` (and `…N`), L2. `Rules.Compiler.Faithful`'s definition is unchanged; its
docstring, which is the claim's scope, changed (L5, L21, L47, L49).

New evidence folders (all with `run.py` or the scripts named, `README.md`, outputs):
`evidence/hint-loops`, `hint-rank2`, `runtime-if`, `scalar-args`, `promotion`, `desc-scratch`,
`autotune-launch`, `compiled-launch`, `other-writers`. Newly pinned Triton 3.8.0 sources
(`primitives/semantics-manifest.json`): `AxisInfo.h`, `ElementwiseOpToLLVMBase.h`, `specialize.cc`,
`autotuner.py`, `allocation.py` (`runtime/_allocation.py`), `nvidia-driver.py`, `TMALowering.cpp`,
`TMAUtilities.h`, `GlobalScratchMemoryAllocation.cpp`, `TritonGPUToLLVM-Utility.cpp`,
`backends-compiler.py`.

**Files of other rounds or unassigned, changed a little:** `lean/TVA/Lemmas/Ops.lean` (the seven
hint lemmas restated with the new predicates, L2), `lean/TVA/Lemmas/Desc.lean` (one new lemma, L5),
`lean/TVA.lean` (one import), `docs/LIBRARY.md` (the module map rows of `Ops`, `Promote`, `Desc`),
`primitives/` (the pins above). No record of another role is edited.

**Kernels.** No kernel uses a changed definition. `kernels/vec_add_f32` still does not build, as
before this round; I did not edit it or run `tvr check`.

**Checks.** `python3 -I tools/tvr.py lint library 8 author`: regular. `python3 -I tools/tvr.py audit`:
independent. The release check (`python3 -I tools/tvr.py release --dir <empty folder>`), once, on the
tree at the end of this round: 8 of 9 steps pass (shipped, packages, sandbox, build, library replay,
both lints, audit); the gate of `kernels/vec_add_f32` fails at its own build and its shipped
`gate.json` and certificate are stale, as before this round.

**Open, for the coordinator or the user.**
- Tutorial 08's `tl.multiple_of(a_ptrs, [16, 16])` is false under the library's reading on every
  launch (L2 below), so tutorial 08 has no claim as written. Covering it needs a decision: a
  kernel-level assumption that reviewers judge, or a narrower library reading of that pattern with
  evidence of how 3.8.0 uses it.
- Tutorial 09's claims cover full tiles only (its hints are false on partial tiles, L2).
- The machine's temporary volume was full during this round; I ran every script with `TMPDIR`
  set to a folder on the root volume.

## L1: A hint inside a loop body on a loop-carried name is not a no-op

- **What changed.** SEMANTICS.md "Compiler hints", new part "Which value"; the section docstring
  of the hints in `lean/TVA/Lang/Ops.lean`; TRANSCRIPTION.md (a pattern, the hint pitfall, the drop
  `no-op:hint`). A hint marks the operation that defined its argument's value when the front end
  visits the call (`set_attr`, `primitives/triton-3.8.0-ir.cc:411-425`); a block argument (a kernel's
  or helper's parameter, an induction variable, the value a loop carries inside its body) takes no
  mark. With the dry visit of `_find_carries` (`code-generator.py:465-503`): a hint on a value the
  body defines checks each iteration's value; a hint on a carried name checks the value before the
  loop, once, and promises nothing about later iterations; with nested loops that carry it, the
  value before the outermost; a value an enclosing body defines is checked per iteration of that
  body; a hint on a parameter or an induction variable has no effect (dropped, `no-op:hint`);
  `tl.static_range` copies check their own values. The withdrawn `no-op:block-arg-hint` stays
  withdrawn except for those two cases.
- **Why.** The issue's to-do: the meaning, the transcription rule, the induction variable, loops in
  loops.
- **Evidence.** [evidence/hint-loops](../../../../evidence/hint-loops/README.md), 13 cases, each
  in its own process: vector loads and faults follow exactly the reading above (C2, C9, C11, C13
  fault with a false value before the loop or in the enclosing body; C1, C8, C12 stay scalar with
  false later values; C5–C7 false hints on block arguments have no effect). The front end's output
  shows the mark on the pre-loop operation and the unread `tt.divisibility_arg<N>` on the loop.
- **Constants whose meaning changed.** None by definition; the docstrings of the section and of
  `TVA.tl.multiple_of`, `max_contiguous`, `max_constancy` and their `…N` forms (which the item
  digests cover) changed.

## L2: What a hint on a rank ≥ 2 tensor promises, in terms a kernel can check

- **What changed.** SEMANTICS.md "Compiler hints", new part "What it promises"; `lean/TVA/Lang/Ops.lean`:
  new predicates `ContigAlong x k C`, `ContigIs x k C` (groups of `C` consecutive along `k`, and
  not groups of `2·C`), `FirstsDiv x k C v`, `ConstAlong x k C`; `tl.max_contiguous` and
  `tl.max_contiguousN` now promise `ContigIs`, the tensor's own contiguity; `tl.multiple_of(N)`'s
  `contig` is defined as the value of a `tl.max_contiguous` on the same value, `1` (every element)
  without one; `tl.max_constancy(N)` unchanged in meaning. Exclusions: values that are not powers of
  two, or above the dimension. N4 (tutorial 09's partial tiles): a false hint stays undefined
  behaviour, and a claim covers the instances where the hints hold.
- **Why.** The documented reading ("groups of `C` consecutive values"; "the first element of each
  contiguous group") is not sound, because the compiler combines a hinted value with what its
  analysis infers for the others (`AxisInfo.cpp:1244-1259`) and applies a dimension's divisibility
  to every element when its contiguity is 1 (`AxisInfo.h:31-40, 92-101`). `multiple_of` alone is
  paired with an inferred contiguity a reviewer cannot compute, which may be 1, so only "every
  element" is sound; `max_contiguous` below the tensor's own contiguity keeps a divisibility
  inferred for longer groups. With `max_contiguous` on the same value the pair is the compiler's
  whole belief. The issue's "fastest dimension only" is not what 3.8.0 does either (the global
  rule).
- **Evidence.** [evidence/hint-rank2](../../../../evidence/hint-rank2/README.md): C1, a true
  "groups of 1" `max_contiguous`, faults (misaligned vector loads), C2 with the tensor's own
  contiguity is correct; R1/R2 tutorial 08's `[16, 16]` (false: `a_ptrs[0, 1]` is 2 bytes in) is
  correct with 128-byte rows and faults with 72-byte rows; R4/R5 the sound spelling
  `max_contiguous(multiple_of(p, [1, 16]), [1, 64])`; T1–T3 tutorial 09 (a partial tile reads other
  elements: odd lanes past `N` hold `x[1]`); K1–K3 a false `max_constancy` gives wrong values.
- **Constants whose meaning changed.** `TVA.tl.max_contiguous`, `TVA.tl.max_contiguousN` (stronger:
  `ContigIs`); `TVA.tl.multiple_of`, `multiple_ofN`, `max_constancy`, `max_constancyN` (the same
  propositions, written with `FirstsDiv` and `ConstAlong`); new `TVA.ContigAlong`, `ContigIs`,
  `FirstsDiv`, `ConstAlong`. Lemmas restated: `safe_max_contiguous(N)` (hypothesis `ContigIs`),
  `safe_multiple_of(N)`, `safe_max_constancy(N)`.

## L5: A descriptor built inside the kernel writes launch scratch memory

- **What changed.** New `tl.make_tensor_descriptor base shape strides block padding`
  (`lean/TVA/Lang/Desc.lean`): the descriptor of its arguments' values, no memory access, undefined
  for a negative shape entry or stride; `safe_make_tensor_descriptor` (`Lemmas/Desc.lean`). The
  module docstring, SEMANTICS.md "Tensor descriptors" and "Launches", `Rules.Compiler.Faithful`
  scope point 5 and the rule page, and an exclusion row: the launch's scratch is not part of the
  claim's memory, and the rule covers the launch only when the allocator's buffer holds none of the
  bytes `init` gives and nothing else writes it; without an allocator the launch is rejected.
- **Why.** With L49 (below) the claim's memory is the bytes `init` gives, so the scratch, which the
  launch writes and the model never reads, needs no model of its own; only its overlap with the
  claim's bytes and other writers can change an outcome.
- **Evidence.** [evidence/desc-scratch](../../../../evidence/desc-scratch/README.md): D1 the host
  error without an allocator; D2 one request per launch, `programs × 128 × descriptors` bytes on the
  launch's stream; D3 each program writes only its own scratch (guards checked); D4 an allocator
  returning an input's bytes overwrote 1024 elements; D5 host descriptors need none. Sources pinned:
  `TMALowering.cpp:166-184`, `nvidia-driver.py:297-333`, `allocation.py:17-21`.
- **Constants whose meaning changed.** New `TVA.tl.make_tensor_descriptor`,
  `TVA.safe_make_tensor_descriptor`; docstrings of `TVA.Rules.Compiler.Faithful` and of the module
  `TVA.Lang.Desc`.

## L21: Autotuning and hooks run the kernel and host code around the claimed launch

- **What changed.** `Rules.Compiler.Faithful` docstring, new paragraph "Which launch"; the rule page,
  new section "Which launch"; SEMANTICS.md "Launches", "One launch"; the `TVA.Claim` module
  docstring; TRANSCRIPTION.md's host pitfall. A claim covers one launch: for an autotuned call, the
  final launch with the chosen config, from the memory after the last hook (`reset_to_zero`,
  `restore_value`, the autotuner's and the configs' hooks); the benchmark runs and hooks are host
  code, and the caller's memory before the call is the host's part.
- **Why.** The issue's to-do.
- **Evidence.** [evidence/autotune-launch](../../../../evidence/autotune-launch/README.md): about
  3900–4900 runs of an in-place `+1` on a new key; `reset_to_zero` makes the final launch start from
  zeros; a config's `pre_hook` runs before every final launch, cached keys included; no benchmark on
  a cached key, with one config, after pruning to one, or from the disk cache. Source pinned:
  `autotuner.py:57-82, 133-173, 217-282`.
- **Constants whose meaning changed.** Docstrings of `TVA.Rules.Compiler.Faithful` and of the module
  `TVA.Claim`.

## L28: No rule for a runtime `if`/`else` or an early `return`

- **What changed.** SEMANTICS.md "How the front end evaluates the source": new bullets "A runtime
  `if`" and "`return`", and the conditional expression with a runtime condition; TRANSCRIPTION.md
  patterns. A runtime `if` converts its condition by `c != 0`, runs only the taken branch's
  operations per program, makes a name live after it only when defined before or in both branches
  with one type, and rejects a block condition; an `elif` runs its condition only where reached;
  `and`/`or` evaluate every runtime operand; a `return` in a runtime `if` ends the program (or the
  helper) for the programs that take it, and is rejected inside a loop. The model is a Lean `if`
  and an early `return ()`.
- **Why.** Tutorial 08 needs both.
- **Evidence.** [evidence/runtime-if](../../../../evidence/runtime-if/README.md), R1–R9: per-program
  counters and sentinel buffers on the device, the front end's `scf.if`/`cf.cond_br`, and every
  rejection raised on the host before any program runs (`code-generator.py:144-229, 834-1049`).
- **Constants whose meaning changed.** None (SEMANTICS.md only).

## L29: No rule for the types of scalar arguments

- **What changed.** SEMANTICS.md "Launches", new "Scalar arguments", and "Index arithmetic";
  TRANSCRIPTION.md pitfall; an exclusion row for unmeasured annotations. An unannotated `int` equal
  to `1` is a compile-time constant (Python's arithmetic) unless `do_not_specialize`; other ints are
  `int32`, `int64`, `uint64` by range, rejected beyond, and multiples of 16 get a divisibility mark;
  `bool` is `int1`; `float` is `float32` rounded to nearest; annotated parameters take the
  annotation (integers wrap, `bfloat16` truncates, `float16` rounds and rejects overflow); `None` is
  a constant; a 0-d tensor is a pointer.
- **Why.** The issue's examples, and the coverage matrix's L29 rows (`n_elements = 1`, `2^31`).
- **Evidence.** [evidence/scalar-args](../../../../evidence/scalar-args/README.md), S1–S6, device
  readback against host conversions (4000 random doubles per float kind, all edge values;
  `(n - 2) // 2` is `-1` for the constexpr `1`, `0` for an `int32`). Sources pinned:
  `specialize.cc:264-306, 377-388`, `backends-compiler.py:75-80`; `jit.py:396-458`;
  `nvidia-driver.c:1090-1163`.
- **Constants whose meaning changed.** None (SEMANTICS.md only).

## L30: Block pointers are aggregates, which exclusions.md excludes

- **What changed.** The aggregate exclusion is narrowed to aggregates a kernel or another library
  defines; SEMANTICS.md "Operators on aggregates" and "Block pointers", and the `TVA.Lang.BlockPtr`
  module docstring, say that `_block_ptr` is Triton's own aggregate, covered through
  `tl.make_block_ptr`, `tl.advance`, `tl.load`, `tl.store`, and that an operator on it has no
  meaning.
- **Why.** `_block_ptr`'s methods are Triton's code (`core.py:1828-1936`), not `@triton.jit`
  functions of the kernel, and it defines no operator, so the reason for the exclusion (an operator
  may run any operation) does not apply to it.
- **Evidence.** The pinned source; no device fact is involved.
- **Constants whose meaning changed.** The module docstring of `TVA.Lang.BlockPtr`.

## L37: Cover every pinned tutorial kernel, except tutorial 05's `_layer_norm_bwd_dx_fused` and tutorial 11

- **What changed.** exclusions.md: the user's exclusions (tutorial 05's `_layer_norm_bwd_dx_fused`,
  tutorial 11) and target (the H200: tutorial 10's AMD kernel excluded, its `block_scaled_matmul_kernel`
  covered on the H200, Blackwell later), each citing the decision. The round-8 gaps of the coverage
  matrix ([round 6](../../round-6/coverage/matrix.md)) now have a stated meaning: N1 (mixed-dtype
  promotion), N2 (`tl.where` on `int32`), N3 (builtin `min`, integer `tl.minimum`) by L45; N4
  (partial tiles) by L2; N6 by L47; N7 by L44; N8, N9 by the target decision; the L1, L2, L5, L21,
  L28, L29 rows by those issues. Tutorial 06's `start_n = tl.multiple_of(start_n, BLOCK_N)` on the
  induction variable is a hint with no effect (L1).
- **Why.** The user's decisions of 2026-10-28, recorded where the scope is stated.
- **Evidence.** The issues' evidence above. What remains for full coverage: tutorial 08's false
  hint (L2, open above), and the round-9 device facts (L4, L33).
- **Constants whose meaning changed.** None (exclusions.md).

## L44: `tl.store` cache modifiers have no stated meaning

- **What changed.** SEMANTICS.md, new "Performance-only arguments" (and the `tl.load`/`tl.store`
  bullets): cache and eviction modifiers of loads, stores and descriptor or block-pointer accesses,
  `volatile`, `num_warps`, `maxnreg`, `launch_cooperative_grid`, `launch_metadata=` and the `tl.range`
  scheduling keywords other than `num_stages` and `warp_specialize` change no result, per Triton's
  documentation, which `Compiler.Faithful` trusts; a model omits them. Out of the list:
  `enable_fp_fusion`, `input_precision`, `max_num_imprecise_acc` (results) and `num_stages`,
  `warp_specialize` (ordering, L15's rule). TRANSCRIPTION.md's `no-op:cache` cites it.
- **Why.** The user's decision (2026-10-28): no model and no per-primitive evidence for
  performance-only primitives.
- **Evidence.** None, by that decision.
- **Constants whose meaning changed.** None (SEMANTICS.md only).

## L45: Operator and promotion semantics

- **What changed.** New module `lean/TVA/Lang/Promote.lean`: `Operand` (a tensor, or a Python
  number), `Operand.ownType?` (`to_tensor_type`), `intPromote`, `computationType?`, `common?`
  (with the Python-number range checks), `BinOp.typing?` (`+ - * / // % & | ^ << >>`, comparisons,
  `tl.minimum`/`tl.maximum` and builtin `min`/`max`), `UnOp.typing?`, `whereTyping?`, `boolOp?`;
  `Operand.constIn?`; `semantic.cast` as `ConvRel` and `tl.convert`, which every implicit conversion
  of a mixed-dtype operator is. `tl.imaximum`/`tl.iminimum` in `Ops.lean`. SEMANTICS.md, new
  "Operators and promotion"; "Constants" corrected (below); TRANSCRIPTION.md patterns and pitfalls;
  an exclusion for float `%`. The Python-number constants' docstring in `Lang/Scalar.lean` is
  corrected the same way.
- **Why.** The user (2026-10-28): "L45 should be built in the library", as one definition every
  mixed-dtype operator uses, evaluated at proof time. **Soundness fix found on the way:**
  SEMANTICS.md "Constants" said a Python number in a comparison is built in the comparison's
  dtype; a comparison first makes it a tensor of its own type, so `x < 0.1` with a `float64` `x`
  compares against `float32(0.1)` widened (true for `x = 0.1`), and a `bfloat16` compared with
  `1e-7` compares in `float32` with `float32(1e-7)`.
- **Evidence.** [evidence/promotion](../../../../evidence/promotion/README.md): `table.py`, the front
  end's own typing of 40240 operator/operand cases (15 dtypes as blocks and scalars, Python ints,
  floats, bools); `rules.py` (0 differences); `lean_compare.py` evaluates the Lean definitions on the
  table: 37896 cases with a tensor operand compared, **0 differ**; `device.py` P1–P11 (exhaustive
  over `int8`/`uint8` pairs for `//`, `%`, mixed signedness; all `float16` values for `fp16 + bf16`,
  which is `float16`, not `float32` as the issue's coverage text guessed); `constants.py` K1–K4 (the
  comparison constants above).
- **Constants whose meaning changed.** New: everything in `TVA.Lang.Promote` (`TVA.DType.tname`,
  `all`, `isTritonInt`, `isSigned`, `isFp8`, `intWidth`, `kind`, `intMin`, `intMax`, `TVA.Operand`
  and its functions, `intPromote`, `computationType?`, `Typing`, `common?`, `BinOp`, `bf16ToF32`,
  `BinOp.typing?`, `UnOp`, `UnOp.typing?`, `whereTyping?`, `BoolOpRes`, `boolOp?`, `ofIntRel`,
  `FtoiRel`, `FloatRel`, `ofFloatRel`, `ConvRel`, `tl.convert`), `TVA.tl.imaximum`,
  `TVA.tl.iminimum`. Docstring: the "Python number constants" section of `TVA.Lang.Scalar` (no
  definition changed).

## L47: Launching a precompiled `CompiledKernel` skips the JIT's argument checks

- **What changed.** `Rules.Compiler.Faithful` scope point 6 and the rule page, SEMANTICS.md
  "Launches", an exclusion row, the `TVA.Claim` docstring: a `CompiledKernel` launch runs the
  warm-up's specialisation with its constexprs (a constexpr position is ignored) and is covered only
  when each argument is in that specialisation (dtypes, the alignment class of pointers, the class
  of integers).
- **Why.** The launcher checks nothing against the compiled specialisation.
- **Evidence.** [evidence/compiled-launch](../../../../evidence/compiled-launch/README.md): K1
  tutorial 02's pattern is correct, constexpr values ignored; K2 outside the specialisation a
  misaligned pointer faulted, `n = 1001` wrote past `n`, a stride `3` was read as `1`, `2^31` wrapped,
  a `float16` tensor was read as `float32` bytes, too few arguments failed on the host; K3
  `launch_metadata=` runs only when a hook reads the metadata (`compiler.py:407-515`).
- **Constants whose meaning changed.** Docstrings of `TVA.Rules.Compiler.Faithful` and of the module
  `TVA.Claim`.

## L49: `Compiler.Faithful`'s "runs alone" forbids every other writer of global memory

- **What changed.** As the issue asks: `Rules.Compiler.Faithful` (new "Memory" paragraph, scope point
  1), the rule page (new "Memory" section, scope 1, the reviewer's host check), `TVA.Claim` (the
  `init` bullet, `Behaviour`, `Outcome.ok`), `Launch.pdl`'s docstring, SEMANTICS.md "Launches", and
  exclusions.md: from its start to its end nothing else writes a byte `init` gives the launch (the
  buffers of `Pre`); writes to other bytes do not matter; `init` and `fin` give the same bytes. The
  PDL case is an instance (a next launch that writes those bytes).
- **Why.** A write to bytes the launch is not given cannot change its outcome, and the condition on
  the claim's buffers is one a kernel reviewer can check from the host code.
- **Evidence.** [evidence/other-writers](../../../../evidence/other-writers/README.md): another
  stream's kernel writing the bytes between the claimed ones, also inside the same 4-byte words,
  overlapping in time in 19 of 20 trials, changed none of the claimed bytes (20 of 20 correct); the
  control writer to the claimed bytes corrupted up to all 4096 elements.
- **Constants whose meaning changed.** Docstrings of `TVA.Rules.Compiler.Faithful`,
  `TVA.Claim.Behaviour`, `TVA.Claim.Outcome.ok`, `TVA.Launch.pdl` and of the module `TVA.Claim`.
