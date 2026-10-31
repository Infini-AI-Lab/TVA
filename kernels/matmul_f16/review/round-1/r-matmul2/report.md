# Review of matmul_f16, round 1, by r-matmul2

## Verdict

accept
I found no blocking problem. Read by SEMANTICS.md, the model does what `matmul_kernel` does, line by line, on every covered instance. That includes an integer argument equal to `1` (a compile-time constant), which the host passes for `stride_ak`, `stride_bn` and `stride_cn`. Both drops are sound: the imports, and the unvisited `ACTIVATION == "leaky_relu"` branch. The launch is the host's, and it stays within part (A) of `Compiler.Faithful`'s scope. `Inst.ok` covers every launch the script makes on `float16` (the test, and the benchmark from 256 to 4096; proved in my work file), and every narrowing is stated with its reason. `Post` fixes each output as the `float16` rounding of a finite `float32` value within an explicit tolerance of the exact dot product, and leaves every other byte unchanged. The three `[rule]` fields are the library's facts for the model's `tl.dot` and its launch. My three findings are non-blocking wording points.

## Findings

- **DOC** `decl:Kernels.Matmul.model` — the docstring's argument for arguments equal to 1 leaves out the loop bound of line 294 at `K = 1` (`range(0, 1)`); the conclusion still holds
- **DOC** `doc:kernels/matmul_f16/README.md` — `c_rows` also excludes non-overlapping layouts of `c` (column-major, transposed); the README says only overlapping ones
- **DOC** `item:spec` — at `K = 2^16`, `tol` is 1.62% of `Σ|a·b|`, which for sign-cancelling (randn-like) data is about 2.6 times the typical `|S|`; worth one sentence in the README

## Accepted

All items.

## Checked

- The model against the source, lines 228–316, by SEMANTICS.md. `tl.cdiv` is the inlined `(x + (div - 1)) // div`, built in `int32` with the wrapping `+` and the truncating `tl.idiv`. Python's `min` with a tensor is `tl.iminimum`. Every constexpr operand is built in `int32`. Left-to-right `+` at line 314 gives two pointer additions. The masks and `other=0.0` are right (a `float32` `+0.0` converted to `float16`). `tl.dot` has the default `tf32` and `maxImpreciseAcc = 0`, and its loop-carried accumulator makes it no `tl.dot_add`. `.to(tl.float16)` rounds to nearest even. The loop's bound is evaluated once; it carries `a_ptrs`, `b_ptrs` and `accumulator`, and neither `k` nor a name first defined in the body is used after it.
- Arguments equal to `1` (compile-time constants) at each use: `M`, `N`, `K` and each stride. `tl.assume(True)` and `64 * 1` added to a pointer give the same model, and so does `range(0, 1)` at `K = 1` (DOC above).
- Every map entry and both drops. The imports bind only `triton` and `tl`. The `config` drop of lines 306–307 is a compile-time `False`. The docstring at 245–247 is not a code line.
- `kernel.json`'s classes. The launch runs no function outside the kernel lines, apart from Triton's `tl.cdiv` and `min`. With Python's `ast` I checked that `triton`, `tl`, `matmul_kernel` and `leaky_relu` are each bound only by their own `import` or `def`, and that no host line, unused header, decorator or default binds a name the launch reads.
- The supported subset: statements, a `str` constexpr, a compile-time `if`, a two-argument `range`, the subscripts, and no `@triton.autotune` or `@triton.heuristics`.
- Every Lean file of the kernel, read in full. The model printed with full names and explicit instances uses only `TVA.*` operations and instances, plus Lean's own auxiliaries (work/Names.lean, work/Explicit.lean). There is no instance, notation, attribute, export, set_option or renaming. `#print axioms` on `main`, `consistent` and `satisfiable` shows only `propext`, `Classical.choice` and `Quot.sound`.
- `launch`: the grid `cdiv(M, 128)·cdiv(N, 256)` as the host computes it, a 1-D grid on axis 0, `debug`, `num_ctas` and `pdl` at their defaults, `fusion = true`, and the ten block shapes the source builds (`[128, 1]`, `[1, 64]`, `[64, 1]` and `[1, 256]` included).
- Part (A) of `Compiler.Faithful`'s scope: point 4 (only `num_stages` and `num_warps`, a bare `@triton.jit`); 5, 6 and 8 (no descriptors, no `CompiledKernel`); 9 (direct calls and `triton.testing.do_bench`, no capture).
- The conditions against the host. work/Cover.lean proves `Inst.ok` for every row-major square launch with `n ≤ 4096`. The `int32` bounds (`M_le`, `N_le`, `a_fit`, `b_fit`, `cm_fit`) and positive strides are what the kernel needs. `c_rows`, `c_a` and `c_b` are stated narrowings that the host meets.
- The specification: `OutSpec` fixes `c = cvt(v)` with `v` within `tol` of the exact sum. It rejects dropped or mis-indexed products and a wrong rounding. The frame `∀ z ∉ c.elems (cIdx d), fin z = init z` holds. I evaluated the tolerance at K = 512, 4096 and 2^16 (norm_num for the first two, exact fractions for the third).
- The assumptions. `faithful` is in the standard form. `dot` is `DotF16 P`, the num.md fact for an fp16 `tl.dot` into `float32` with `maxImpreciseAcc = 0` and any precision, which is the model's operation at K = 64. `dot_total` is `DotTotal P` at the model's `DotCfg`. `Assumptions` is a structure with exactly `(K : Claim.Behaviour Inst) (P : Platform)`.
- The write-up against the Lean: the conditions table, the narrowings, the proof outline, the numbers (tolerances, the `satisfiable` instance at `2^21`, `2·2^21`, `3·2^21`).

## Not checked

- The library itself (the `tl.*` definitions, `Compiler.Faithful`, `DotF16`'s constants), which kernel review trusts, and part (B) of `Compiler.Faithful`'s scope, which is assumed.
- The proof steps beyond what Lean checks. I relied on the gate's build and replay, and did not re-run `lake build`.
