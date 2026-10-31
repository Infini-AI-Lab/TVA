# Review of attn_bwd_pre_f16, round 1, by r-bwdpre

## Verdict

accept
The model follows lines 248-262 line by line under SEMANTICS.md. The pointer sums are evaluated left to right, each `int32` offset is added to the pointer separately, and every constexpr is built in `int32`. The `float16` × `float32` product is promoted to `float32` (`BinOp.typing?`), and `tl.sum` of a product is `tl.sum1c` with the launch's fusion flag. It is not the bcast-mul shape, because the two operands are same-shape blocks. The launch's grid, defaults and five block shapes are those of lines 600-607, and the launch stays within part (A) of `Compiler.Faithful`'s scope. `Inst.ok` covers every launch the script makes. Its narrowings (`BLOCK_M | N_CTX`, `HEAD_DIM ≤ 256`, the `int32` no-wrap bounds `row_fit` and `head_fit`, the grid limits, and no overlap with the freshly allocated `delta`) are the host's asserts, CUDA's limits or the kernel's own limits, and the README states each one. `Post` fixes every `Delta[r]` to within `tol(HEAD_DIM, Σ|o·do|)` of the exact row dot product. That tolerance depends only on `HEAD_DIM`, and at the largest covered `HEAD_DIM` (256) it is `766·2^-24·B`, well below a 16-bit rounding. `Post` also keeps every other byte unchanged. The only assumption is `Compiler.Faithful` in the standard form. My one finding is a non-blocking wording fix in the README.

## Findings

- **DOC** `doc:kernels/attn_bwd_pre_f16/README.md` — README Narrowings, "The grid": "the host's largest has Z·H = 192 and 128 · 128 programs" joins two maxima from different launches; no launch of the script has both.

## Accepted

All items.

## Checked

- Every Lean file of the kernel in full (`AttnBwdPre.lean`, `Model.lean`, `Err.lean`, `Proof.lean`). There are no notations, instances, attributes, renamings or local binders named like library namespaces. Printed with full names (`work/print_model.lean`), the model reaches only `TVA.*` operations and instances and the kernel's own definitions. The launch's omitted fields are `Launch`'s defaults. `#print axioms` shows only `propext`, `Classical.choice` and `Quot.sound`.
- The model map, entry by entry, against SEMANTICS.md. This covers the drops of lines 20, 21 and 248, the parameter classes (`Z` and `H` are never read, `N_CTX` is an `int32` and is the constant `1` only when `BLOCK_M = 1`, with the same model), the `int32` constants, the broadcasts written on the operands, the loads with no mask, the exact widening `.to(tl.float32)`, the promotion of `o * do`, `tl.sum1c` rather than `tl.sum_bcast_mul`, and the store with no conversion.
- The source's function and line classes. The launch runs only `_attn_bwd_preprocess`. It reads only `triton` and `tl`, bound only at lines 20-21. Its `HEAD_DIM` is its parameter, not the benchmark loop's module-level `HEAD_DIM`. The unused functions' headers have no defaults, and their `tl.constexpr` annotations bind no name.
- The launch: the grid `(N_CTX // 128, BATCH * N_HEAD, 1)`, the program ids, the five block shapes, and no options. On scope points 4, 5, 6, 8 and 9: no option or heuristics, no descriptor, no `CompiledKernel`, no host descriptor, and no graph capture (the benchmark uses `do_bench`).
- The conditions against the host. `backward` asserts `do.is_contiguous()` and equal strides, so `o` and `do` have the contiguous layout the kernel indexes. `HEAD_DIM ∈ {16..256}` (line 512), `N_CTX % 128 == 0` (line 599), and `delta` is freshly allocated. Every test and benchmark size meets every bound, checked with exact arithmetic. `satisfiable`'s witness is a test instance.
- The specification. The tolerance values in the README's table and its example are confirmed with exact rationals. Finite `float16` inputs give a finite result. `Post` rejects a dropped column, a wrong row, a missing store and 16-bit rounding. The frame clause covers every byte outside the written `Delta` elements.
- The assumption `faithful`: `Faithful P (model d) (launch d) (K d)` under `∀ d, d.ok`, as the rule page states. `Assumptions` is a structure with exactly `(K, P)`.
- The README against the Lean: the conditions table, `Pre`, the tolerance, the narrowings, the proof outline and the line numbers.

## Not checked

- The library's own facts (the `tl.*` meanings, `AddRel`, `sum_bcast_mul_tree_err`, `meets_independent`, `Compiler.Faithful`), which kernel review trusts.
- Part (B) of `Compiler.Faithful`'s scope (the environment), which is assumed. The script's own code visibly breaks none of it.
- The proofs' internals beyond what Lean checks. Whether the kernel compiles is not part of any claim.
