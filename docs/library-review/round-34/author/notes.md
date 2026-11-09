# Library batch, round 34: author's notes

Issues L149 (reopened), L155, L156, L157, L158 and L159 ([ISSUES.md](../../ISSUES.md)). No kernel or claim folder
was edited. `lake build TVA` is green, and the new theorems use only the standard axioms (`propext`,
`Classical.choice`, `Quot.sound`; `MemSem.unused_gpu` and `tl.atomic_add_unused` only `propext`). The kernel files
that mention `AddRel`, the fusable forms, atomics or races were checked one by one against the new library with
`lean` (the kernels' `.olean`s are not rebuilt by `lake build TVA`).

## L149: The depth of Triton's reduction and scan trees

- **What changed.** The `OptimizeThreadLocality` exclusion is now stated on the source, more broadly than the
  pass's own IR pattern (`docs/SEMANTICS.md`, "Float `+` and `-`", in three parts; `docs/exclusions.md`, its row,
  "library issue L149 (rounds 32, 33, 34)"). Out of scope: a float reduction (`tl.sum`, `tl.max`, `tl.min`, any
  `tl.reduce` with a float operand) along the last axis of a tile of rank 2 or more, one of whose operands equals,
  element for element up to NaN payload, the tile of one `tl.load` (pointer tensor or block pointer, in the kernel
  or a helper; not a descriptor's `load`), judged on the launch's compile-time constants and on the integer
  arguments Triton specialises (those equal to `1`), whatever operations stand between them, folded or not;
  anywhere in the body of a `for` over `range` or `tl.range` (any options, `loop_unroll_factor` included; in
  helpers, under runtime or compile-time `if`s, in `tl.static_range` loops); whose result reaches, through value
  operations, a variable the loop carries; whatever the initial value, the update, the other uses, the float width
  and the layout. The pass's own pattern (after inlining, folding, specialisation, dead-code removal and
  unrolling, `OptimizeThreadLocality.cpp:252-372`, `nvidia-compiler.py:244-276`) and what the device then computes,
  `U(c, R(v))`, stay as an informative part; the "not rewritten" list says which of those cases the exclusion
  still covers. `docs/rules/num.md` (the depth section) and `evidence/reduce-tree/README.md` point to the source
  statement. The old "not rewritten" item "a `float16` `tl.sum` (converted to `float32` first)" was wrong:
  `tl.sum` converts only integers narrower than 32 bits (`standard.py:268-293`); it is corrected.
- **Why.** lr-33a: the IR wording sent to "not rewritten" loops that the pass rewrites (identities after the
  reduction or the update, dead uses, compile-time `if`s, `tl.static_range(1)`, helpers, a runtime int argument
  equal to `1`). What reaches the pass's IR depends on folding, specialisation and layout, which a kernel reviewer
  should not have to follow, so the exclusion is stated on the source and covers more than the pass rewrites.
- **Evidence.** [evidence/thread-locality](../../../../evidence/thread-locality/README.md) Part W (H200 NVL),
  every lr-33a form re-run: all rewritten, `2.0` stored for the source's `1.0` (the initial value `iv = 1`: `3.0`
  for `2.0`); also rewritten: `x * k` with a runtime `k = 1`, `.to(tl.float64).to(tl.float32)`, a `float16`
  accumulation, a `float16` `tl.reduce` with a maximum (`−inf` for `4.0`). Not rewritten: `iv = 0`, `k = 2`,
  `* 2.0`, a runtime `if`, `static_range(2)` (its `0.5` is the source's own `float32` value: `2^23 + 0.5` rounds to
  `2^23`; lr-33a's "1.0" label was wrong), `loop_unroll_factor=2` with a compile-time trip count; with a runtime
  trip count the pass's assertion fails and the compile ends with an error (no outcome). Parts U, F, L re-ran
  identical. `claims_scan.py` (follows helpers): no claim's covered path has the broader pattern (four candidates
  read: `extend_attn_bf16` 397, 563, `moe_router_bf16` 145, `paged_attention_f16` 68, none a load reduced into a
  carried value).
- **Constants whose meaning changed.** None (docs and evidence only).

## L155: `AddRel` with one operand unrounded

- **What changed.** `TVA.AddRel` (`lean/TVA/Lang/Ops.lean`) is narrowed: with fusion, the rounding of `p + q` for
  exact `p`, `q` rounding to the operands now requires `p = x.toRat ∨ q = y.toRat` (one operand enters as it
  is); the IEEE and infinite cases are unchanged. The previous relation is kept as `TVA.AddRelWide`, and
  `TVA.AddRel.wide` derives it. `AddRel.ieee` and `AddRel.false_iff` are unchanged in statement;
  `FloatV.AddRel.err` (`Lemmas/FusedErr.lean`) now takes the relation apart through `AddRel.wide` (same
  statement). Every other lemma (`SumFused`, `Sum`, `Ops`, `Scan`, `Scaled`, `DotLoop`) uses only these, so all
  hold of the narrow relation unchanged. `docs/SEMANTICS.md` (`tl.addc`) states the narrowing and its evidence.
- **Why.** Round 33 filed it (split from L154 (2)): on sm_90 contraction gives an `fma`, which takes one product
  unrounded and the other operand as it is, so the wide relation allowed results no fusion gives; the narrow one
  is what lets `l2norm_fwd_bf16` and `layer_norm_fwd_bf16` cover `eps ∈ [2^-126, 2^-125)`.
- **Evidence.** Source: Triton enables fusion as LLVM's `AllowFPOpFusion = Fast` (`triton-3.8.0-llvm.cc:136-140`)
  and `ptxas` FMA formation (`nvidia-compiler.py:532`); every fused NVPTX instruction is `fma(a, b, c)`, one
  product (`llvm-NVPTXInstrInfo.td:1278-1315`, packed forms per lane); `dot_add` makes a `tl.dot` accumulate, not
  a two-product addition. Device: [evidence/fma-one-product](../../../../evidence/fma-one-product/README.md) (H200
  NVL), `float32`, `float64`, `float16`, `bfloat16`, five forms (`a*b + c*d`, `a*b - c*d`, a `tl.sum` of two
  products, an accumulation in a loop, a three-product sum) on nearly cancelling products: 0 of 163840 results
  outside the narrow relation; on 156264 inputs the rounding of both exact products differed from every value it
  allows, and the device never gave it. SASS: `FFMA`, `DFMA`, `HFMA2`, `HFMA2.BF16_V2` with `FMUL`/`DMUL`/`HMUL2`.
- **Constants whose meaning changed.** `TVA.AddRel` (narrowed) and so everything reaching it: `tl.addc`,
  `tl.subc`, `tl.sumc`, `sum1c`, `sum0c`, `sumWith`, `tl.cumsumc`, `AccAdd`, `tl.dot_add`, `tl.sum_bcast_mul`,
  the scaled dot's addition (`Lang/Scaled.lean`); `FloatV.AddRel.err`
  (proof and docstring). New: `TVA.AddRelWide`, `TVA.AddRel.wide`. The kernel files that use the relation
  (`attn_fwd_f16`, `attn_fwd_e5m2`, `attn_bwd_f16`, `attn_bwd_pre_f16`, `paged_attention_f16`, `softmax_f32`) only
  pass it to `AddRel.err` or build it with `AddRel.ieee`, and check against the new library.

## L156: num-special-f32's conclusion is out of date

- **What changed.** `evidence/num-special-f32/README.md`, conclusion 2, now names the fact round 33 stated on this
  evidence, `Rules.Num.ExpF32Overflow` (`+∞` from `1453635/16384 = 0x42b17218` up, and at `+∞`), and the platform
  change it needed: `Platform.ref` gives its table's value where no float is within the bound
  (`Platform.ref.mathAt_far`), `sm90Special` gives `exp x = +∞` there, `Platform.sm90ref.expF32Overflow` proves it.
- **Why.** lr-33a: the conclusion still said the library states no fact from this evidence.
- **Evidence.** None new.
- **Constants whose meaning changed.** None.

## L157: Relaxed atomics that only accumulate

- **What changed.** A rule, in Lean at the model's level: `TVA.MemSem.unused` (`lean/TVA/Lang/Prog.lean`) is the
  order of an atomic whose result the source never uses: `acq_rel` at `gpu` or `sys` scope whatever the source's
  `sem` (`relaxed`, `acquire`, `release`), the source's `sem` at `cta`. `tl.atomic_add_unused`,
  `atomic_xchg_unused`, `atomic_cas_unused`, `atomic_max_unused`, `atomic_min_unused`, `atomic_and_unused`,
  `atomic_or_unused`, `atomic_xor_unused` (`Lang/Ops.lean`) are the source's `tl.atomic_*` with that order and no
  result, so the model cannot use the old values; `MemSem.unused_gpu`, `unused_sys`, `unused_cta` (`rfl`). Two such
  atomics (or one and an `acq_rel` atomic) on the same bytes of different programs are then `Event.Strong`; the
  race exemption itself (`Event.Strong`, `InterRace`) is unchanged in definition, and its docstring and
  `Sem/Launch.lean`'s module docstring state the rule. "Unused" on the source: an expression statement, or a value
  bound only to names nothing that runs reads (code the front end drops does not read); on the IR, a
  `tt.atomic_rmw`/`tt.atomic_cas` result with no use. A plain load or store of those bytes still races (or
  conflicts within its program), so such cells are reached by atomics only. Descriptor reductions keep `relaxed`
  and still race. Docs: `docs/SEMANTICS.md` ("Atomics", "Race"), `docs/exclusions.md` (the relaxed row),
  `docs/review-questions.md` Q10 (settled for pointer atomics; left open: used results, descriptor reductions),
  `docs/TRANSCRIPTION.md` (a row), `docs/LIBRARY.md`, `Lemmas/Shared.lean`'s module docstring.
- **Why.** User decision 2026-11-08 (a library rule rather than narrowing the claim). It is stated on the model,
  not on events: an event does not show whether its program uses the old value, and `Step.defer` re-chooses a
  deferred atomic's continuation at each later step, so an exemption keyed on "the continuation ignores the value"
  would not survive deferral. Soundness (`MemSem.unused`'s docstring): the semantics has no synchronises-with and
  already performs atomics after independent later steps; a relaxed atomic can further be performed before its
  thread's earlier accesses, which no other program can observe when its result is unused, since its bytes are
  reached by atomics only and PTX performs them with single-copy atomicity (PTX ISA 9.5 §8.4 strong operations,
  §8.7 morally strong operations, §8.10.3 atomicity), and other bytes it accessed would race.
- **Evidence.** [evidence/relaxed-atomics](../../../../evidence/relaxed-atomics/README.md) (H200 NVL): Part P, every
  op, dtype, `sem` and scope compiles to one `atom` with the source's order and scope, no barrier, used or not
  (except a scalar `atomic_add` of constant `0` with `relaxed`/`acquire`: a strong `ld`, memory unchanged); Part A,
  120 cases (12 ops and dtypes, scattered and contiguous lanes, vectorised `.v2`, five `sem`/scope pairs), 16384
  updates per cell, 50 launches each: no cell wrong of 384000; Part M, plain stores beside the atomics: nothing
  wrong; Part L, vLLM's split-K LoRA shrink pattern at both configurations: exact on integer data (428800
  outputs), relaxed and `acq_rel` alike on random data. `AtomicAddF32` already covers every `sem`.
- **Constants whose meaning changed.** New: `TVA.MemSem.unused`, `MemSem.unused_gpu`, `unused_sys`, `unused_cta`,
  `TVA.tl.atomic_add_unused` and the seven others. Docstring changed: `TVA.Event.Strong` (every item whose cone
  reaches `WellBehaved` re-opens, that is every claim citing `Compiler.Faithful`). Claim that would change:
  `claims/lora_shrink_bf16` (source line 305; `Claim.lean` line 242, `let _ ← tl.atomic_add c_ptr accumulator (some c_mask)
  .relaxed`) should be `tl.atomic_add_unused c_ptr accumulator (some c_mask) .relaxed`; its launches are then well
  behaved (the shared cells are reached by `acq_rel` atomics only) and `TVA.shared` applies.

## L158: `torch.bool` arguments

- **What changed.** `docs/SEMANTICS.md` ("Parameters and arguments") states that a `torch.bool` tensor argument is
  `Ptr .i1` (signature `*u1`, `!tt.ptr<i1>`, `element_ty` `tl.int1`), one byte per element; `tl.load` reads the
  byte and gives `≠ 0` (`DType.ofBits .i1`), a masked-off lane `other` converted to `int8` and then tested; and
  `tl.store` converts the value to `int8`, not to `int1` (an `int1` stores `0`/`1`; an `int32` `256` stores `0`),
  which the model writes.
- **Why.** The library stated it only in the docstrings of `DType.bytes` and `DType.ofBits`.
- **Evidence.** `semantic.py:874-877, 1036-1040, 1061-1062, 1236-1243`; [evidence/bool-args](../../../../evidence/bool-args/README.md)
  (H200 NVL): bytes `0, 1, 2, 127, 128, 255` load as `false, true, true, true, true, true` (`ld.global.b8`);
  `other = 5` gives `true`; comparisons store `0`/`1`; `int32` `0, 1, 2, 255, 256, 257, -1, 512` store their low
  bytes `0, 1, 2, 255, 0, 1, 255, 0`.
- **Constants whose meaning changed.** None.

## L159: Depth form of the broadcast-product sum lemma

- **What changed.** `lean/TVA/Lemmas/SumFused.lean`: `bcastDepthBound D K B` (the bound with depth `D`),
  `bcastTreeBound_eq` (`bcastTreeBound K B = bcastDepthBound (K − 1) K B`), and `sum_bcast_mul_depth_err`: rounded
  `float32` products summed along a tree over `0, …, K−1` of depth at most `D` with `AccAdd` nodes are within
  `bcastDepthBound D K B` of `Σ a·b`; with `Rules.Num.SumDepth` and `SumDepth.depth_le`, `D = Nat.clog 2 K`. Its
  docstring says it is for a model that writes the products and a reduction (`tl.mul`, then `tl.sumc`), since
  `tl.sum_bcast_mul` states no depth for its tree form. `docs/LIBRARY.md`.
- **Why.** A prover had to rebuild the composition from `sumc_err_depth` and the products' roundings
  (`claims/attn_bwd_pre_f16`, whose `tol D B` is `bcastDepthBound (Nat.clog 2 D) D B`).
- **Evidence.** None (a lemma).
- **Constants whose meaning changed.** New: `TVA.bcastDepthBound`, `TVA.bcastTreeBound_eq`,
  `TVA.sum_bcast_mul_depth_err`.
