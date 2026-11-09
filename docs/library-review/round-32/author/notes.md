# Library batch, round 32: author's notes

Issues L141 (reopened) and L144 and L146 to L154 ([ISSUES.md](../../ISSUES.md)). The claim sources each issue
names were read, not edited, and no kernel folder was edited. `lake build TVA` is green, and the new theorems use
only the standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Triton 3.8.0's C++ sources the batch
cites beyond `primitives/` were read from the v3.8.0 release tree and pinned into `primitives/`
(`triton-3.8.0-PipeliningUtility.cpp`, and the reduction sources of L149).

## L141: The integer `tl.dot`: the `uint32` accumulator

- **What changed.** `docs/SEMANTICS.md` (`tl.dot`): an accumulator is `int32` or `uint32`, with `out_dtype`, when
  given, the accumulator's dtype; every other accumulator dtype, and `out_dtype=tl.int32` with a `uint32`
  accumulator, is rejected on the host; with a `uint32` accumulator the dot computes as with its bits read as
  `int32`, so the model writes `tl.dot (tl.bitcast .u32 .i32 acc h) …` with `cfg ⟨.i8, .i8, .i32, p, 0⟩`, an
  `int32` result, and `DotI8`'s room is on that reading. `docs/rules/num.md` ("The integer `tl.dot`", the
  `DotI8` rows of both tables): the same. `evidence/num-dot-int8/README.md`: a row of I5 and conclusion 1
  corrected ("as corrected in round 32"). `docs/TRANSCRIPTION.md`: a pitfall.
- **Why.** Round 31's two sentences said that an accumulator other than `int32` is rejected; lr-31b found a
  `uint32` one accepted, whose result reads its bits as `int32` (`3000000000 + 9` gave `−1294967287`).
  `semantic.dot` takes `out_dtype` from the accumulator and passes its handle on; the IR's integers are signless,
  so a `uint32` accumulator has the `int32` result's IR type and the verifier accepts it.
- **Evidence.** [evidence/num-dot-int8-round32](../../../../evidence/num-dot-int8-round32/README.md): A1, ten
  accumulator dtypes by three `out_dtype`s (only `int32`, and `uint32` with `out_dtype` omitted or `tl.uint32`,
  accepted); A2, a `uint32` accumulator on `mma.sync` and `wgmma` at `K = 32, 64, 256`, half the bits at or above
  `2^31`, the edge of the room on two rows: 4096 of 4096 and 65536 of 65536 results at each `K` equal the bits
  read as `int32` plus the exact sum; A3, the result is an `int32` in the kernel (`c < 0`). Sources:
  `semantic.py:1429-1520`.
- **Constants whose meaning changed.** None: `DotI8` and `tl.bitcast` are unchanged; the page now names the case.

## L144: `Compiler.Faithful` about an execution of the kernel; no CUDA graphs

- **What changed.** `lean/TVA/Rules/Compiler.lean` (the docstring of `Rules.Compiler.Faithful`): a new paragraph
  "A launch is an execution of the kernel": the rule, and a claim's `K`, are about one execution of the
  kernel's grid of programs, running the code Triton compiles for the launch's arguments, compile-time values and
  options, from `init` until its last program ends; how a caller makes it happen is no part of it, and a host call
  that does not run the grid it is given is not an execution of it and has no outcome, while the run of its
  programs, when they run, is an execution like any other. Scope 9 is removed (the other points keep their
  numbers, as the docstring says), and "Which launch" and part (A) speak of an execution. The same in
  `lean/TVA/Claim.lean` (the module docstring and `Claim.Behaviour`), `lean/TVA/Sem/Launch.lean`
  (`Launch.Valid`'s docstring), `docs/rules/compiler-faithful.md` ("What it says", "Scope", "Which launch", "What
  it does not cover", the checklist), `docs/SEMANTICS.md` ("Launches": "One launch: an execution of the
  kernel", part (A)), `docs/exclusions.md` (the CUDA-graph row removed), `docs/PROVER_GUIDE.md`,
  `docs/REVIEWER_GUIDE.md`, `docs/REVIEW_PROTOCOL.md` (§6) and `docs/TRANSCRIPTION.md`. No framework page now
  names CUDA graphs; `evidence/host-round20` (G0–G3) and round 16's work file keep the record, cited by none.
- **Why.** The user (2026-11-08): "we verify the kernel itself"; "I do not think we even need to mention CUDA
  Graph in our claims or in our framework". The rule stays true: a captured call runs no program, so it is not an
  execution and the rule says nothing of it; a replay runs the captured code with the captured arguments on the
  memory it runs on, and the rule's conditions (nothing else writes the bytes, the code is the code Triton
  compiles for those argument values) are about that execution as they are about any other.
- **Evidence.** None new: the facts the rule rests on are unchanged (evidence/launch-outcomes for the grids the
  launcher skips; evidence/host-round20 G0–G3 showed what a capture does).
- **Constants whose meaning changed.** None by value. The docstrings of `TVA.Rules.Compiler.Faithful` (the
  claim's text of the rule), `TVA.Claim.Behaviour` and `TVA.Launch.Valid` changed, so every item whose cone
  reaches them is re-opened (every claim that cites `Compiler.Faithful`).

## L146: Descriptors built inside a kernel are excluded

- **What changed.** Scope 5 of `Compiler.Faithful` (`lean/TVA/Rules/Compiler.lean`,
  `docs/rules/compiler-faithful.md`) is now "No descriptor built in the kernel": the front end visits no call of
  `tl.make_tensor_descriptor` on the instance, in the kernel or a helper (a call in a compile-time branch not
  taken, as tutorial 06's `_maybe_make_tensor_desc` given a host `TensorDescriptor`, is not visited); host
  descriptors (scope 8) stay covered. The frame ("Memory") has no scratch exception left, and part (A) no
  allocator. `docs/SEMANTICS.md` ("Tensor descriptors", "Launches"), `lean/TVA/Lang/Desc.lean` (module docstring
  and `tl.make_tensor_descriptor`'s: kept as the call's meaning, which no covered launch reaches),
  `lean/TVA/Lemmas/Desc.lean`, `docs/TRANSCRIPTION.md` (the pattern row), `docs/LIBRARY.md`, the guides, and
  `docs/exclusions.md`: the allocator row replaced by "a descriptor built in the kernel", which names the tutorial
  kernels it removes (tutorial 08's `grouped_matmul_tma_kernel`, tutorial 09's
  `matmul_kernel_descriptor_persistent`, tutorial 06's `_attn_fwd` on the launches where its host passes tensors,
  the Hopper warp-specialised path); the row of tutorial 08's `grouped_matmul_kernel` no longer says the TMA
  kernel stays in scope. `ISSUES.md`: L37's list of user exclusions gains the 2026-11-08 decision and these
  kernels; L120 and L121 are `CLOSED`, settled by the exclusion.
- **Why.** The user (2026-11-08): "let's exclude this case. we only cover descriptor outside of triton kernels".
  A `tl.make_tensor_descriptor` is what makes Triton 3.8.0 give a launch a global scratch buffer (it lowers to a
  `GlobalScratchAllocOp`, `TMALowering.cpp:166-184`, multi-buffered by the pipeliner in a loop,
  `PipeliningUtility.cpp:596-612`; the other users are instrumentation, scopes 2 and 4), so within the new scope
  the launcher allocates no scratch and calls no allocator (`nvidia-driver.py:301-307`), and what L120 (a
  rebuild in a loop) and L121 (the allocator's alignment) asked of that scratch does not arise.
- **Evidence.** Sources: `primitives/triton-3.8.0-TMALowering.cpp`, `-PipeliningUtility.cpp` (pinned this round),
  `-GlobalScratchMemoryAllocation.cpp`, `-nvidia-driver.py`. evidence/launch-frame (the global scratch size was
  `0` for every kernel but the one that built descriptors).
- **Constants whose meaning changed.** None by value; the docstrings of `TVA.Rules.Compiler.Faithful` (scope 5)
  and `TVA.tl.make_tensor_descriptor` changed (items reaching them are re-opened). Claims: `attn_fwd_f16`'s model
  builds descriptors for pointer arguments (`claims/attn_fwd_f16/Claim.lean:50`), so its warp-specialised path is
  outside the rule and the claim narrows to host descriptors; the READMEs' "5 (descriptor scratch)" lines
  (`matmul_e5m2`, `persistent_matmul_f16`, `ssd_bmm_chunk_bf16`) stay true under the new wording. The kernels
  `grouped_matmul_tma_f16` (whole) and `attn_fwd_f16` (that path) are outside the scope.

## L147: `tl.cast`: what a float narrowing is

- **What changed.** `docs/SEMANTICS.md` ("`x.to(d)`, `tl.cast(x, d)` and `x.cast(d)`"): a `fp_downcast_rounding`,
  either value, on a conversion that is not from a float to a float of fewer bits is rejected on the host, with
  `bfloat16` to `float16` and back (the same width) among the examples.
- **Why.** Triton's test is a float target of fewer bits (`semantic.py:824-832`), so the 16-bit pair, a
  conversion that rounds, takes no rounding mode.
- **Evidence.** [evidence/front-end-round32](../../../../evidence/front-end-round32/README.md) R1: `"rtne"` and
  `"rtz"` rejected on `bfloat16 → float16` and `float16 → bfloat16`, accepted on `float32 → float16` and
  `float32 → bfloat16`.
- **Constants whose meaning changed.** None.

## L148: A block's `.type` is not a dtype

- **What changed.** `docs/SEMANTICS.md` ("A tensor's `.type`"): where the page admits a dtype, write a dtype, a
  scalar's `.type` or a block type's `.element_ty`; a block's `.type` as `tl.zeros`'s or `tl.full`'s dtype, and as
  a scalar's `.to` target of another dtype, is rejected on the host; as a block's `.to` target it means its
  element dtype, whatever its shape.
- **Why.** lr-31a: `block.to(x.type)` works but `scalar.to(x.type)` and `tl.zeros(…, dtype=x.type)` are rejected;
  `semantic.cast` keeps a block input's shape and takes the target's scalar type.
- **Evidence.** evidence/front-end-round32 Y1 (nine forms: `x.to(y.type)` converts to `y`'s element dtype with
  `y` of another shape; `v.to(y.type)`, `v.to(z.type)`, `tl.zeros(…, dtype=x.type)`, `tl.full(…, x.type)`
  rejected; `tl.zeros(…, dtype=v.type)` and the same-dtype forms accepted).
- **Constants whose meaning changed.** None.

## L149: The depth of Triton's reduction and scan trees

- **What changed.** `lean/TVA/Lang/Platform.lean`: a field `Platform.sumDepth : Nat → Nat` (default `n - 1`).
  `lean/TVA/Lang/Ops.lean`: `tl.sumBound P n = max (P.sumDepth n) (Nat.clog 2 n)`, and `tl.sumWith`,
  `sum1With`, `sum0With` read the platform (`.env`) and pick a tree with `t.Covers n ∧ t.depth ≤ tl.sumBound e.P n`
  (a balanced tree always qualifies, so the pick is never empty and no kernel needs a new rule).
  `Tree.depth` moved from `Lemmas/Tree.lean` to `Lang/Tensor.lean` (same name). `lean/TVA/Lemmas/Ops.lean`:
  `Tree.bal`, `bal_leaves`, `bal_depth`, `bal_covers`, `Tree.exists_sumBound`, and the rules `safe_sum_depth`,
  `safe_sum1_depth`, `safe_sum0_depth`, `safe_sumc_depth`, `safe_sum1c_depth`, `safe_sum0c_depth` (and the
  `…With_depth` forms); the existing `safe_sum*` statements are unchanged (re-proved). `sum_err_depth`
  (`Lemmas/Sum.lean`), `sumc_err_depth` (`Lemmas/SumFused.lean`): the error with depth `D` and `n − 1` nodes.
  `Rules/Num/Sm90.lean`: the fact `Rules.Num.SumDepth P : ∀ n, P.sumDepth n ≤ Nat.clog 2 n` (listed alone),
  `SumDepth.depth_le`, `clog_two_pow`; `Platform.ref` sets `sumDepth n := Nat.clog 2 n`
  (`Lemmas/Consistency.lean`), and `Platform.sm90ref.sumDepth` (`Lemmas/Sm90.lean`). `docs/rules/num.md`: rows in
  both tables, a section "The depth of a float reduction", "Satisfiability". `docs/SEMANTICS.md` (reductions;
  "Float `+` and `-`"): the depth bound, and the one reassociating rewrite, which `docs/exclusions.md` now
  excludes (a new row). `docs/LIBRARY.md`. Scans: no depth fact (below).
- **Why.** `sum_err_covers` and `sumc_err_covers` allow any tree (depth `n − 1`), so claims composing them
  (`layer_norm_fwd_f16` at `N ≥ 8192`, `moe_router_bf16`, and looser `softmax_f32`, `attn_bwd_pre_f16`,
  `layer_norm_fwd_bf16`, `l2norm_fwd_bf16`) accepted wrong results. Triton 3.8.0's `ReduceOpToLLVM.cpp` reduces
  each thread's elements of the axis along a balanced binary tree (`treeReduce`, arity 2 unless a target
  overrides `getReductionTreeArity`, which only AMD does, `TargetInfoBase.h`), then along a `shuffleXor`
  butterfly over the lanes and, after a conversion through shared memory, over the warps: a perfect tree of depth
  `log₂ n` over the `n = 2^k` positions, whatever the layout. A scan (`ScanOpToLLVM.cpp`) is sequential within a
  thread and over the warps, so its depth depends on the layout the compiler picks, which a claim cannot name
  without compiling: no scan fact is stated, and `chunk_cumsum_f32` is unchanged. Reading the passes found a
  soundness defect: `OptimizeThreadLocality.cpp` (on for NVIDIA, `compiler.py:276`) rewrites
  `acc = acc + tl.sum(tl.load(2-D), axis=-1)` in a `for` loop, from a constant `acc`, into per-thread
  accumulation across the trips and one reduction after the loop, a reassociation that SEMANTICS.md's "no
  reassociation" denied; the pattern is now excluded (no claim's or kernel's source has it: their reduced
  operands come from a dot, an `exp` or a product, not directly from a load).
- **Evidence.** [evidence/reduce-tree](../../../../evidence/reduce-tree/README.md) (H200 NVL): 7960
  configurations, each checked twice (a tuple `tl.reduce` returning its tree's depth and leaf count: `log₂ n`
  over exactly `n` leaves; float sums of leaves `1 + 2^-p` and `1 − 2^-(p+1)`, exact only along a perfect tree),
  0 failures: 1800 1-D cases (`n` from 1 to 16384, 1 to 16 warps, contiguous and strided loads, `float32`,
  `float16`, `bfloat16`, `float64`, products), 5616 2-D cases (both axes, row- and column-major, `tl.trans`,
  `keep_dims`), 256 on `tl.dot` results (`mma.sync` and `wgmma`), 288 reductions of reductions; 259 distinct
  operand layouts (228 blocked, 31 `nvidia_mma`). Part L: the loop pattern stored `2.0` on every row where every
  association of the source gives `1.0`, its TTGIR showing the rewrite; with a loaded initial `acc`, `1.0`.
  Sources pinned: `triton-3.8.0-ReduceOpToLLVM.cpp`, `-TargetInfoBase.h`, `-OptimizeThreadLocality.cpp`.
- **Constants whose meaning changed.** `TVA.tl.sumWith`, `TVA.tl.sum1With`, `TVA.tl.sum0With`, and so
  `TVA.tl.sum`, `sum1`, `sum0`, `sumc`, `sum1c`, `sum0c` (a depth bound read from the platform; under the default
  `sumDepth` every tree as before); `TVA.Platform` (a new field) and `TVA.Platform.ref` (its value);
  `TVA.Tree.depth` moved (same definition). New: `TVA.tl.sumBound`, `TVA.Rules.Num.SumDepth`, the lemmas above.
  Kernel libraries with proofs (Softmax, AttnBwdPre, PagedAttention, AttnFwd, AttnBwd, AttnFwdE5m2, Matmul,
  MatmulE5m2, StreamKMatmul) still build; their items reaching these sums re-open.

## L150: Sharper per-chunk 16-bit `tl.dot` facts

- **What changed.** New `lean/TVA/Rules/Num/Dot16Chunk.lean`: the chunks of 16 (`chunks16`, `chunk16`,
  `chunkSum16`, `chunkAbs16`), the constants (`rhoChunk16 = 2^-20`, `sigmaChunk16 = 2^-22`,
  `alphaChunk16 = 2^-145`, `kappaChunk16 = 2`, `rhoLossChunk16 = 2^-24`), the room (`relChunk16`, `absChunk16`),
  the shape `DotChunk16Loss` (`.mono`, `.step_left`), `Dot16BitChunk`, and the facts `DotF16Chunk` and
  `DotBF16Chunk`: partial values one step per chunk of 16 consecutive `k` in increasing order, each within
  `min(2^-20·Mj, 2·A_j + 2^-24·Mj) + 2^-22·|s j + c_j| + 2^-145`, for every `precision`, with `maxImpreciseAcc = 0`.
  New `lean/TVA/Lemmas/Dot16Chunk.lean`: `DotChunk16Loss.err`, `.errB`, `.liftErr`, `Dot16BitChunk.err` (the
  result within the composed `errM` from the exact inputs), `Dot16BitChunk.liftAcc` (`DotLiftAcc` with
  `relLift16`, `absLift16`), and satisfiability (`Platform.ref.dotChunk16Loss`,
  `Platform.sm90ref.dotF16Chunk`, `.dotBF16Chunk`). `docs/rules/num.md`: a row in both tables (listed alone: both
  are of type `Platform → Prop`), a paragraph "Chunk by chunk" under the 16-bit operands (the statement, the
  constants and their derivation, the use, the evidence), "Satisfiability". `docs/LIBRARY.md`; `lean/TVA.lean`
  imports the two modules. `DotF16` and `DotBF16` stay.
- **Why.** `relDot16 K = (K+2)·2^-22` per `tl.dot` is about 6× the per-chunk derivation and is charged on the
  whole `|acc|` each trip; the claims named (`lora_shrink_bf16`, `extend_attn_bf16`, `streamk_matmul_f16`,
  `fused_moe_bf16`, `ssd_bmm_chunk_bf16`, `attn_fwd_f16`) need the accumulator's share bounded per chunk.
- **Evidence.** [evidence/num-dot-chunk16](../../../../evidence/num-dot-chunk16/README.md) (H200 NVL, both paths,
  both formats, about `1.2·10^8` element results, each equal to evidence/num-dot-sub16-round25's per-instruction
  model, none outside the bound): single chunks at `K = 16` against the device's own partial values (largest ratio
  `0.4996`, accumulator-dominated cases `0.40` of the second arm), chains to the largest `K` that compiles (each
  step at most `0.4996`, the composed `errM` at most `0.4972`), and tutorials 03's and 06's fed-back loops to
  `K = 16384` (at most `0.3998`; on those loops the composed bound is `0.21` to `0.30` of `DotF16`'s composed step
  by step).
- **Constants whose meaning changed.** None; new constants only (those above).

## L151: Pointer minus integer

- **What changed.** `lean/TVA/Lang/Mem.lean`: three `HSub` instances (`ptr - k`, `ptr - offsets`,
  `ptrs - offsets`), each `p + (−k)` with the negation wrapping in `k`'s type, and the lemmas `Ptr.sub_int`
  (`p - k = p.offset (-k).toInt`) and `Ptr.sub_int_of_signed` (a signed `k` above its type's minimum: `k` elements
  back). `docs/SEMANTICS.md` ("Pointers"): `p + k` and `p - k` (the `int32` `−2^31` its own negation, an unsigned
  `k ≠ 0` `2^w − k` elements forward, a Python number first a tensor of its own type, the rejected forms).
  `docs/LIBRARY.md`.
- **Why.** vLLM's `cu_num_draft_tokens_ptr + req_idx - 1` (`claims/rejection_greedy_i32/source`, line 91) had no
  meaning: `semantic.sub` builds `ptr - offset` as `add(ptr, minus(offset))`, `minus` is `0 - x` in `x`'s type
  (`semantic.py:261-267, 478-485`), and `add` zero-extends an unsigned offset narrower than 64 bits.
- **Evidence.** evidence/front-end-round32 S1–S3: every address equal to `p + (−i)·size` on `float32`, `float16`
  and `float64` buffers for a runtime, loaded `int32`, `int64` and `uint32` and literal `i`, edge values included,
  and lane by lane on blocks; the TTIR (`arith.subi` from `0`, `tt.addptr`); `i - p`, `p - q`, `p - 1.0`, `1 - p`
  rejected; `cu + req - 1` reads and writes element `req − 1`.
- **Constants whose meaning changed.** None; new constants only (the three `HSub` instances, `TVA.Ptr.sub_int`,
  `TVA.Ptr.sub_int_of_signed`). `lean/TVA/Lang/Mem.lean` changed, so every item whose cone reaches it is
  re-opened.

## L152: `dtype.is_fp8()` and the other dtype predicates

- **What changed.** `docs/SEMANTICS.md`: a new item "A dtype's predicates" in "Values" (23 predicates, each a
  compile-time `bool` on the dtype's name: `is_fp8()` true of `float8e4nv` and `float8e5`, `is_int_unsigned()`
  true of `int1`, …; they decide a compile-time `if` and a conditional expression); the subset's attributes and
  "Expressions with no effect" list them; "`tl.store`" states that a value stored to an fp8 pointer is converted by
  `tl.cvt8 x .rn`, the saturating rounding to nearest even (it was stated only for a Python number).
- **Why.** vLLM's `reshape_and_cache_kernel_flash` branches on `key_load.dtype.is_fp8()`
  (`claims/kv_cache_write_bf16/source`, lines 167, 176), and stores the bf16 quotient to an fp8 cache.
- **Evidence.** evidence/front-end-round32 T1 (23 predicates by 13 dtypes, 299 kernels, in `tl.static_assert`, a
  compile-time `if` and a conditional expression: none differs from the name), T2 (vLLM's form stored to
  `float8e4nv` and `float8e5`: 0 of 1024 bits differ from the saturating rounding to nearest even, `10^6` stored
  as `448` and `57344`). Sources: `core.py:391-394, 449-524`, `semantic.py:811-916, 1205-1243`.
- **Constants whose meaning changed.** None.

## L153: Tutorial 10's `mixed` format runs

- **What changed.** `docs/exclusions.md` (the targets row): tutorial 10's `mxfp8` and `mixed` formats run on the
  H200 and are covered (`DotScaledE4m3`, `DotScaledE4m3E2m1`); its `mxfp4` and `nvfp4` launches at the tutorial's
  config are rejected on the host, out of shared memory; an `e2m1` by `e2m1` `tl.dot_scaled` that runs at a
  smaller config has no device fact. `ISSUES.md` (L37): a note that the decision's "fp4 launches are rejected"
  holds of `mxfp4` and `nvfp4` only.
- **Why.** The row said that tutorial 10's fp4 launches are rejected on the host, but `--format mixed` (`e4m3` by
  `e2m1`, the `MIXED_PREC` branch) runs, so a prover trusting it would call those launches rejected.
- **Evidence.** [evidence/tut10-formats](../../../../evidence/tut10-formats/README.md): the tutorial's own inputs
  and launch at `1024³` for each format and `8192³` for `mixed`: `mxfp4` and `nvfp4` `OutOfResources` (287864 and
  297080 bytes of shared memory, 232448 available); `mxfp8` and `mixed` run, largest error `0.0625` (`mixed` at
  `8192³`: `0.2495`), the tutorial's check passing.
- **Constants whose meaning changed.** None.

## L154: Small precision items

- **What changed.**
  - (1) `lean/TVA/Lemmas/Random.lean`: `maxVal64_ge`, `ofRat_int_f64` (an integer below `2^53` in magnitude is
    exact in `float64`) and `tl.itofRel_binary32` (below `2^53`, `tl.ItofRel .binary32 z r ↔ r = FloatV.ofRat z`:
    the alternatives coincide), and the module docstring.
  - (3) `lean/TVA/Rules/Num/Sm90.lean`: a new fact `Rules.Num.DivF32InfDivisor` (on `float32`, a finite `x` by
    an infinite `y` gives the zero of sign `sign x xor sign y`); `Platform.sm90ref.divF32InfDivisor`
    (`lean/TVA/Lemmas/Sm90.lean`: IEEE division gives it); `docs/rules/num.md`, both tables (listed alone) and
    "Division". `tl.exp` past its overflow is measured (below) but no fact is stated: `Platform.sm90ref` leaves an
    approximate function's result free where no finite float is within its bound, so a fact fixing `+∞` there
    needs `Platform.ref` changed first, which is left to a later round.
  - (4) `lean/TVA/Lang/Desc.lean` (`TensorDesc.Valid`'s docstring) and `docs/SEMANTICS.md` ("Tensor
    descriptors"): a leading stride of `0`, or one shorter than the row, is admitted, with the evidence.
  - (2) `AddRel` is not narrowed: a narrower fusable addition (only a product's operand unrounded) changes the
    meaning of every fused addition and of the lemmas on it, and is left to a later round with its own evidence.
- **Why.** (1) `claims/seeded_dropout_f32`'s `uRef` writes `FloatV.ofRat` for `tl.ItofRel .binary32`; (3)
  `gdn_gating_bf16` divides by a value that may be `+∞`; (4) the H200's behaviour had no evidence folder.
- **Evidence.** (3) [evidence/num-special-f32](../../../../evidence/num-special-f32/README.md) D1: every finite
  `float32` dividend by `+∞` and by `-∞` (`2 × 4278190080` pairs), none other than the signed zero; E1: `tl.exp` is
  `+∞` exactly from `0x42b17218` (`88.72283935546875`) to `+∞` among the inputs from `+64`, the largest finite
  result `0x7f7fff4f`. (4) [evidence/desc-strides](../../../../evidence/desc-strides/README.md): host descriptors
  with strides `[0, 1]` and rows that overlap, in `float32`, `float16` and `int8`, 2-D and 3-D, loads at every block
  offset and `atomic_add`: no lane or element differs from the formula.
- **Constants whose meaning changed.** None; new constants only (`TVA.maxVal64_ge`, `TVA.ofRat_int_f64`,
  `TVA.tl.itofRel_binary32`, `TVA.Rules.Num.DivF32InfDivisor`, `TVA.Platform.sm90ref.divF32InfDivisor`).
  `TVA.TensorDesc.Valid`'s docstring changed, not its value.
