# Library batch, round 33: author's notes

Issues L144, L146, L149, L150 and L154, each reopened by round 32's library review ([ISSUES.md](../../ISSUES.md)).
No kernel or claim folder was edited. `lake build TVA` is green, and the new theorems use only the standard
axioms (`propext`, `Classical.choice`, `Quot.sound`). Triton 3.8.0's sources cited beyond `primitives/` were taken
from the v3.8.0 tag and pinned (`triton-3.8.0-AccelerateMatmul.cpp`, `triton-3.8.0-TritonGPUAttrDefs.td`).

## L144: The framework should not mention CUDA graphs

- **What changed.** The sentence "a host call that does not run the grid it is given (… later, or never) … has no
  outcome" is replaced, in `lean/TVA/Rules/Compiler.lean` (the docstring of `Rules.Compiler.Faithful`, "A launch is
  an execution of the kernel"), `lean/TVA/Claim.lean` (the module docstring and `Claim.Behaviour`),
  `docs/rules/compiler-faithful.md` ("What it says"), `docs/SEMANTICS.md` ("Launches", twice) and
  `docs/REVIEW_PROTOCOL.md` §6, by: a host call that only records the grid, for a run that a later call makes, is
  not an execution and has no outcome; that later run is the execution. A call that returns without an error and
  whose grid never runs (the launcher's skip of `2^31` to `2^32` programs) is not such a call: it ends `ok` with
  memory unchanged, which `Launch.Valid` excludes. `lean/TVA/Sem/Launch.lean` (`Launch.Valid`) says why its bound
  is needed. No page names CUDA graphs.
- **Why.** lr-32a: "or never" contradicted the kept "a launch that returns without an error has an outcome, even
  if no program ran" on the skipped grid, and made its `Meets` vacuous (the L9 hazard); "returns while its
  programs run later" described every asynchronous launch.
- **Evidence.** None new: evidence/launch-outcomes (the skipped grids end with memory unchanged).
- **Constants whose meaning changed.** None by value; the docstrings of `TVA.Rules.Compiler.Faithful`,
  `TVA.Claim.Behaviour` and `TVA.Launch.Valid` changed, so every item whose cone reaches them re-opens (every claim
  citing `Compiler.Faithful`).

## L146: Exclude descriptors built inside a kernel

- **What changed.** Scope 10(c) of `Compiler.Faithful` (`lean/TVA/Rules/Compiler.lean`,
  `docs/rules/compiler-faithful.md`) lists `triton.set_allocator` again among the allowed uses of Triton's API,
  saying why: within scope 5 the launcher never calls the allocator it sets. `docs/exclusions.md` (the row of code
  that changes Triton) says the same, so the two agree.
- **Why.** lr-32a: round 32 dropped it from 10(c) but kept it in exclusions.md, and tutorial 06's in-scope host path
  calls `triton.set_allocator` (`claims/attn_fwd_f16`, source line 548), so reviewers could read 10(b) either way.
  `set_allocator` only sets a context variable (`triton-3.8.0-allocation.py:26-34`); the launcher calls it only for
  a positive `global_scratch_size` (`triton-3.8.0-nvidia-driver.py:301-321`), which only a
  `tl.make_tensor_descriptor` gives.
- **Evidence.** [evidence/set-allocator](../../../../evidence/set-allocator/README.md) (H200 NVL): lr-32a's seven
  in-scope kernels (host descriptors pipelined and warp-specialised, pointer loads with `tl.dot`, block pointers,
  gather/sort/cumsum/histogram/atomics/inline asm, `tl.dot_scaled`, a 3-D dot), each with scratch sizes `0`, ran
  with the null allocator, then called a counting allocator set by `triton.set_allocator` `0` times and wrote the
  same bytes; the control (a descriptor built in the kernel) called it once.
- **Constants whose meaning changed.** None by value; the docstring of `TVA.Rules.Compiler.Faithful` (scope 10)
  changed.

## L149: The depth of Triton's reduction and scan trees

- **What changed.** The `OptimizeThreadLocality` exclusion is restated on the IR the pass sees, and called a
  miscompile: `docs/SEMANTICS.md` ("Float `+` and `-`": the pattern in three points, what the device computes,
  `U(c, R(v))`, and what is not rewritten), `docs/exclusions.md` (its row), `docs/rules/num.md` (the depth
  section), `evidence/reduce-tree/README.md` (a note on its source reading and conclusion 3). An operand counts as a
  load when it is a `tl.load` of a pointer tensor or block pointer, masked or not, in the kernel or a helper, up to
  operations that give back every element unchanged (a same-dtype `.to`, `* 1.0`, `/ 1.0`, `- 0.0`,
  `tl.maximum(x, x)`, a double `tl.trans`, a reshape to its own shape, a compile-time `tl.where`), whether or not
  Triton folds them; the update is any operation `U`; the loop is a `for` (`range`, `tl.range`), the reduction
  directly in its body; the exclusion does not depend on the layout. `SumDepth` and its lemmas are unchanged.
- **Why.** lr-32a, lr-32b: the pass clones the update (`OptimizeThreadLocality.cpp:393-520`), so with an update
  other than the combine the result is no association of the source; and it reads the IR after folding, so
  "produced directly by `tl.load`" missed forms it rewrites.
- **Evidence.** [evidence/thread-locality](../../../../evidence/thread-locality/README.md) (H200 NVL), every value
  exact: Part U, 12 updates, all rewritten, the stored value `U(c, R(v))` (`acc - sum` `+260` for `-260`, `sum - acc`
  `260` for `0`, `acc + max` `-inf` for `8`, `acc + min` `+inf` for `4`, `acc * sum` `0` for `17850625`,
  `acc / sum` NaN for `0`, `acc * max` `-inf` for `16`, `maximum(acc, sum)` `2` for `1`); Part F, 16 operand forms
  rewritten (lr-32a's and lr-32b's included: `.to(float32)`, `* 1.0`, `* SCALE`, `/ 1.0`, `- 0.0`, block pointer,
  helper; and `tl.maximum(x, x)`, double transpose, reshape, compile-time `where`, masks, cache modifier), 9
  controls not; Part L, `tl.range` rewritten, `static_range`, `while`, axis 0, 1-D and one element per thread not.
  Round 32's library review read the reductions of all 40 claims: none has the pattern on its covered path.
- **Constants whose meaning changed.** None.

## L150: Sharper 16-bit `tl.dot` facts

- **What changed.** `DotF16Chunk` and `DotBF16Chunk` are restated per chunk of 64 consecutive `k`
  (`lean/TVA/Rules/Num/Dot16Chunk.lean`: `chunks16 K = ⌈K/64⌉`, `chunk16 K j` the `k` with `k / 64 = j`;
  `rhoChunk16 = 2^-13`, `sigmaChunk16 = 2^-20`, `alphaChunk16 = 2^-143`, `rhoLossChunk16 = 2^-22`, `kappaChunk16 = 2`;
  the module docstring states which `k` each instruction adds for each `kWidth` and the derivation).
  `lean/TVA/Lemmas/Dot16Chunk.lean`: the same proofs on 64 (docstrings). `docs/rules/num.md`: both rows, the
  sentence on `DotF16`'s chunks (any 16 per instruction), "Chunk by chunk" (why 64, the constants, the evidence).
  `docs/LIBRARY.md`. `DotF16` and `DotBF16` stay (their derivation needs only 16 products per instruction).
- **Why.** lr-32b: on `mma.sync` with an operand converted from 8 bits (`kWidth = 4`) an instruction adds
  `k ≡ 0, 1 (mod 4)` of 32, and a `K = 32` launch refutes the 16-chunk facts. `kWidth = 8` also occurs for a 16-bit
  dot (an 8-bit load unpacked by `tl.join`), with instructions spanning 58 `k` of a block of 64, so chunks of 32 would
  be false too; every partition lies in aligned blocks of 64 (Triton sets `kWidth` from the narrowest load,
  `AccelerateMatmul.cpp:385-427, 561-568`, `TritonGPUAttrDefs.td:1478-1488`).
- **Evidence.** [evidence/num-dot-chunk16-round33](../../../../evidence/num-dot-chunk16-round33/README.md) (H200
  NVL): Part W, `kWidth` 2, 4, 8 by origin (no other); Part P, 104 configurations probed on the device, each
  instruction's `k` in one aligned block of 64, in order; Part X, lr-32b's counterexample (no 16-chunk partial
  values; the new step `0.008` of its bound); Part M, the per-instruction model equal to the device on all 2605056
  element results (`a` or `b` converted), 62976 elements checked exactly, none outside, largest ratio `0.381`.
- **Constants whose meaning changed.** `TVA.Rules.Num.chunks16`, `chunk16` (and so `chunkSum16`, `chunkAbs16`,
  `relChunk16`, `absChunk16`), `rhoChunk16`, `sigmaChunk16`, `alphaChunk16`, `rhoLossChunk16`, `DotChunk16Loss`'s
  docstring, `Dot16BitChunk`, `DotF16Chunk`, `DotBF16Chunk`; `TVA.Dot16Chunk.relLift16`, `absLift16`, `chunkLift16`,
  `chunkLiftSum16`. No kernel or claim cites them yet.

## L154: Small precision items

- **What changed.** (3) A new fact `Rules.Num.ExpF32Overflow` (`lean/TVA/Rules/Num/Sm90.lean`, with
  `posInfF32`): on `float32`, every input from `1453635/16384` (`0x42b17218`) up, and `+∞`, gives `+∞`; listed
  alone in rules/num.md's tables, with "Past the overflow" and "Satisfiability"; `docs/SEMANTICS.md`,
  `docs/LIBRARY.md`. `Platform.ref` (`lean/TVA/Lemmas/Consistency.lean`) now gives the table `sp`'s value also at
  a finite input where no float is within the requested bound (new lemma `Platform.ref.mathAt_far`; `mathAcc`,
  `mathTotal` re-proved); `sm90Special` gives `exp x = +∞` from `1453635/16384` and at `+∞`, and
  `Platform.sm90ref.expF32Overflow` proves the fact (no float is within one rounding of `e^x` there, from
  `Real.log_two_lt_d9`). (2) `AddRel`'s narrowing is filed as its own issue, L155 (round 34).
- **Why.** lr-32b: both were left undone with nothing tracking them. The narrowing changes the meaning of every
  fusable addition, the lemmas and six kernels' proofs that take `AddRel` apart and three claims, and needs its own
  evidence on Triton's lowering (L155 says how).
- **Evidence.** [evidence/num-special-f32](../../../../evidence/num-special-f32/README.md) E1 (every pattern from
  `+64` to `+∞`: `+∞` exactly from `0x42b17218`), re-run this round: identical.
- **Constants whose meaning changed.** `TVA.Platform.ref` (its `math`), `TVA.sm90Special`, `TVA.Platform.sm90ref`;
  new: `TVA.Rules.Num.posInfF32`, `ExpF32Overflow`, `TVA.Platform.ref.mathAt_far`,
  `TVA.Platform.sm90ref.expF32Overflow`. Items whose cone reaches `Platform.sm90ref` (every `Claim.Consistent`
  proof) re-open; their lemma names and statements are unchanged, so they still build.
