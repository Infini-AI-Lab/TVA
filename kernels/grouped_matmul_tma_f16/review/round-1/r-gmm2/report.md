# Review of grouped_matmul_tma_f16, round 1, by r-gmm2

## Verdict

accept
The model follows the source line by line by SEMANTICS.md on every covered instance (FP8 = False, the config of SETUP.md), the conditions cover every launch the script makes and state each narrowing with its reason, Post fixes every element of every C g within a tolerance that still means something at the largest covered K (0.253% of the sum of |a·b| at K = 8192, below a bfloat16 rounding's 0.389%) and keeps every other byte, and the three [rule] fields are the library's facts for the model's tl.dot and the launch. I found no blocking problem. One wording slip in the README (a threshold beyond the covered K) and one doubt about the library's evidence for descriptors rebuilt in a loop are non-blocking.

## Findings

- **DOC** `doc:kernels/grouped_matmul_tma_f16/README.md` — the "from about K = 15745" bfloat16 threshold is off (15617 for tol alone, 13633 for Post's bound on c); the covered claim is right
- **LIBRARY_REPORT** `decl:TVA.tl.make_tensor_descriptor` — does the evidence cover descriptors rebuilt in a runtime loop (one set per group), rewriting the scratch slot while a TMA store through the old one may be in flight?

## Accepted

All items.

## Checked

- Every Lean file of the kernel in full (Defs, Sched, Basic, Tile, Program, GroupedMatmulTma): no notation, instance, attribute, export, set_option, renaming or look-alike; the model printed with `pp.fullNames` (work/Names.lean) uses only `TVA.*` operations and Lean's own `if`/`&&`; `#print axioms` of `main`, `consistent`, `satisfiable` gives only `propext`, `Classical.choice`, `Quot.sound`.
- The model against lines 186–278 and every map entry: `int32` index arithmetic and Python constants built in `int32`; `tl.cdiv` as `(x + (div - 1)) // div` with `tl.idiv`; the runtime `if` of line 220 (both comparisons evaluated, signed, `tile_idx` merged, `else` keeps it); `tl.whileLoop` with the source's condition; `range(group_size)` and `range(0, cdiv(k, 64))` as `tl.forRange` with `int32` induction variables and the carried names; the scalar loads, `tl.intToPtr .f16` of the loaded `int64`; the three descriptors (shape `int32`, strides converted to `int64`, block shapes, default zero padding); `desc.load`/`desc.store` blocks and offsets; `accumulator += tl.dot(a, b.T)` as `tl.dot_add` with `b.trans`, default `tf32` and `maxImpreciseAcc = 0`; `.to(float16)` as `tl.cvt`; `tile_idx += NUM_SM`; the drops of lines 32, 33, 186 and 209 (the conditional expression on the compile-time `FP8` evaluates only `tl.float16`).
- The `group_size = 1` case (a compile-time `1`): its only use is `range`, whose `int32` induction variable and single trip the model gives.
- The supported subset: statements, names, calls and parameters of the kernel; no hint, helper, `break`, global or constexpr global; no name used after its loop; no dtype or `None` assigned in the runtime `if`; module-level names read are only `triton` and `tl` (lines 32–33); host lines (module `group_size`, `M`, `N`, `K`, `A`, `B`, `tma_configs`, the unused kernel's header) bind nothing the kernel reads (its parameter `group_size` shadows the module's).
- The launch: `Launch.grid1 NUM_SM` for `grid = (META['NUM_SM'],)`, the block shapes `[128, 64]`, `[64, 128]`, `[128, 128]` (all of the source's), defaults for fusion, debug, `num_ctas`, PDL; part (A) points 4, 5, 6, 8, 9 against both launch sites, the allocator set at line 320 before the test and the benchmarks, and `do_bench` (no graph capture).
- The conditions against the host's launches: test (`M = N = K` in 128…1024, `lda = ldb = K`, `ldc = N`), square benchmark (`N` in 128…1024, `g_lds = [N, N, N]` with `B_T`), batch benchmark (`M` in 128…1024, `N = K = 8192`, `g_T_lds = [K, K, N]`), `group_size = 4`, `NUM_SM = 132`; each narrowing (`int32` bounds, `K ≤ 2^13`, strides multiples of 8, 16-byte bases, `N ≤ ldc`, the disjointness fields, `FP8 = False`, the config) is in the README with its reason; `Pre` holds of the host's tables (`int64` addresses, `int32` sizes and strides).
- The specification: `OutSpec` pins element `(i, j)` of `C g` to the `float16` rounding of a finite `float32` within `tol` of `Σ A[i,k]·B_T[j,k]` (the host's `B.T.contiguous()`), with a finite bound when there is room; the frame keeps every byte outside the matrices `C g`; the numbers of `rho`, `alpha`, `tol` and `tol_rel_lt` (work/tol.py); it rejects a bfloat16-rounded result, a dropped 64-step, a wrong tile or group, and an `f16` accumulator.
- The `[rule]` fields: `faithful` is `Faithful P (model d) (launch d) (K d)` under `d.ok`; `dot` is `DotF16 P` (fp16 into fp32, `maxImpreciseAcc = 0`, any precision), the fact num.md lists for the model's `tl.dot`; `dot_total` is `DotTotal P` at the model's `DotCfg`. No other model operation depends on the platform beyond the library's definitions (`ConvRel` for `int32 → int64` is exact).
- The write-up against the Lean: specification, conditions table, assumptions and proof outline agree, but for the DOC finding.

## Not checked

- Whether Triton 3.8.0 orders the rewrite of a loop-rebuilt descriptor after the asynchronous TMA store that used the previous one (the library's matter; the LIBRARY_REPORT).
- The proofs themselves beyond reading them: Lean and the gate check them.
