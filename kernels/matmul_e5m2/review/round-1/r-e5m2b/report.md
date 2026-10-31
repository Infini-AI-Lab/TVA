# Review of matmul_e5m2, round 1, by r-e5m2b

## Verdict

accept-partly
The model, the map and the five `[rule]` assumptions are right: every line of `matmul_kernel` means what the source does by SEMANTICS.md on every covered instance (the scalar arguments equal to `1` included), every name of the model is a `TVA` constant or a kernel definition, the drops are sound, and each rule is the library's fact for the model's operation. The conditions cover the host's `float8e5` run. The specification is not meaningful at all covered sizes: from `K = 2817` to the covered `K = 4096` the tolerance exceeds `Σ|a·b|`, so `Post` holds for a kernel that writes `+0` to every element of `c` (F1, blocking). I hold back `spec` and `conditions` until that is settled, since narrowing `K_le` is one way to settle it.

## Findings

- **SPEC_TOO_WEAK** `item:spec` — for `2817 ≤ K ≤ 4096` (covered; ten of the benchmark's thirty-one fp8 sizes) `tol K A > Σ|a·b|`, so an all-zero `c` meets `Post`: no accuracy claim there (`work/ZeroPasses.lean`, `zero_meets_tol`); narrow `K_le` to `2816`, get a tighter library fact, or make `Post` claim no tolerance above `2816`.
- **DOC** `doc:kernels/matmul_e5m2/README.md` — what the tolerance rejects: below `2817` it fixes the result only within `1 ± γ` (a result `S/2` passes from `K = 1665`, `half_meets_tol`); from `2817` it still rejects `−S` and a read of another row or column, which the README says it no longer does.

## Accepted

- `item:model`
- `item:map`
- `item:assumption:faithful`
- `item:assumption:dot`
- `item:assumption:dot_total`
- `item:assumption:cvt`
- `item:assumption:cvt_total`

## Checked

- Every Lean file of the kernel in full (`MatmulE5m2.lean`, `Defs.lean`, `Arith.lean`, `Basic.lean`, `Program.lean`): no notation, instance, `open … renaming`, `export`, option or look-alike name; the model printed with full names (`work/Names.lean`) uses only `TVA.*` constants and the kernel's own definitions; `Assumptions` is a structure with exactly `(K : Claim.Behaviour Inst) (P : Platform)`; `#print axioms` of `main`, `consistent` and `satisfiable` shows only `propext`, `Classical.choice`, `Quot.sound`.
- The model against lines 228–316, line by line, by SEMANTICS.md: `tl.cdiv` as `(x + (div − 1)) // div` with the wrapping `int32` `+` and the truncating `tl.idiv`; the grouped order of lines 255–260; Python's `min` as the signed `tl.iminimum` with `GROUP_SIZE_M` as an `int32` tensor; the `tl.assume` lines as signed `int32` comparisons with `0`; `% M`, `% N` as `tl.irem`; `x[:, None]`, `x[None, :]` as `col`/`row` (broadcasting before an elementwise integer product changes no value); the masks; `other=0.0` as the `float32` `+0` converted to `float8e5` on the device (`tl.convert`, then `CvtF32E5m2`); `tl.dot` with the defaults `tf32` and `maxImpreciseAcc = 2^30` and the loop-carried accumulator (no `dot_add`); the loop's bound, carried names and `int32` induction variable; `.to(tl.float16)` as `tl.cvt` with `.rn`; the two pointer additions of line 314; the store without conversion.
- Every use of a scalar argument equal to `1` (`M`, `N`, `K` and the six strides; `stride_ak = stride_bk = stride_cn = 1` on every launch of the fp8 run): each gives the same model, as `model`'s docstring says (SEMANTICS.md, "Scalar arguments", "A name assigned a Python number holds a tensor", `tl.cdiv`, "Loops and unrolling").
- The map's 40 entries and the drops of the imports, `@triton.jit` and lines 306–307 (`ACTIVATION == "leaky_relu"` compares two compile-time constants, so the branch is not visited); `kernel.json`'s classes: the kernel lines hold `matmul_kernel` and the imports that bind `triton` and `tl`; the host lines and the unused functions' headers bind nothing the launch reads (no `:=`, no rebinding of `triton`, `tl` or `matmul_kernel`); `leaky_relu` is named only in the branch not visited.
- The supported subset: statements, names (neither a name first defined in the loop body nor the loop variable is used after the loop), the `str` constexpr `ACTIVATION`, the builtins `min` and `range`, no operator computing in an fp8 dtype; the three `tl.cdiv` calls name distinct copies (divisors `128`, `256`, `64`).
- The launch: the grid `cdiv(M, 128)·cdiv(N, 256)` on axis 0, all ten block shapes of the source, `fusion = true`, `debug = false`, `num_ctas = 1`; part (A) of `Compiler.Faithful`'s scope: only `num_stages` and `num_warps`, no descriptors, no `CompiledKernel`, no host `TensorDescriptor`, `do_bench` (not `do_bench_cudagraph`), no capture.
- The conditions against the host: the test (`512³`) and the benchmark's fp8 line (`n = 256 … 4096`, `b` the transposed view with strides `(1, n)`, `c` fresh and row-major) meet `Inst.ok` (`fp8_run_ok`); each `int32` bound is what the kernel's offsets need; the narrowings (`c_rows`, positive strides, `K ≤ 4096`, accuracy for finite inputs only) are stated; `satisfiable` uses the test's instance.
- The specification: `OutSpec` and `Post` read as the README says; memory outside the matrix `c` is unchanged; the tolerance evaluated at the covered sizes (`work/ZeroPasses.lean`; relative part `0.1326` at `512`, `0.9834` at `2816`, `1.0145` at `2817`, `1.7077` at `4096`).
- Each `[rule]`: `DotE5m2` is `DotF8 P .f8e5 .f8e5`, every `precision` with `maxImpreciseAcc = 2^30` into `float32`, the model's `tl.dot`; `DotTotal` with the model's own `DotCfg`; `CvtF32E5m2` (`CvtRn binary32 e5m2`) and `CvtTotal P binary32 e5m2 .rn` for the conversion of `other=0.0`; `faithful` in the form `∀ d, d.ok → Faithful P (model d) (launch d) (K d)`.
- The write-up against the Lean: the specification, the conditions table, the narrowings, the assumptions and the proof outline agree with the definitions and lemmas.

## Not checked

- Whether the library's facts and definitions are true of Triton 3.8.0 and the H200 (trusted, by the protocol); in particular whether a tighter fp8 `tl.dot` fact than `DotE5m2` holds, which option (b) of F1 would need.
- Part (B) of `Compiler.Faithful`'s scope (the environment), by the protocol.
- The proofs beyond reading them: Lean and the gate check them; I did not re-run the gate or `lake build`.
- How `torch` lays out `b.T.to(torch.float8_e5m2)` beyond what the README states; `Inst.ok` holds for either stride order, so the claim does not depend on it.
