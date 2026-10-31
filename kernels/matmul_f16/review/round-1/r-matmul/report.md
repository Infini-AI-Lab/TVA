# Review of matmul_f16, round 1, by r-matmul

## Verdict

accept-partly
The model is a faithful transcription of `matmul_kernel` with `ACTIVATION = ""`, line by line by SEMANTICS.md, including the `int32` arithmetic, the integer arguments equal to `1`, and every mask, broadcast and conversion. The launch, the map, the supported-subset check and the three `[rule]` assumptions are right. The conditions cover the script's use (the test at 512, the benchmark's square sizes 256..4096) and state every narrowing. One blocking finding is about the specification at the top of the covered range. At `K = 2^16` the tolerance is 1.62% of `Σ|a·b|`. That no longer rejects a result rounded to `bfloat16` (from `K ≈ 15745`, or about `18177` allowing for the device's measured accumulation error), and for the benchmark's own `randn` inputs it exceeds the typical `|S|` (from `K ≈ 34368` on). Narrowing `K_le` to about `2^13`, or stating the limitation, settles it. I hold back `spec` and `conditions` until then.

## Findings

- **SPEC_TOO_WEAK** `item:spec` — at the largest covered `K = 2^16` the tolerance (1.62% of `Σ|a·b|`) no longer rejects a `bfloat16`-rounded result, and exceeds the typical `|S|` of `randn` inputs; narrow `K_le` (about `2^13`) or state it
- **DOC** `decl:Kernels.Matmul.tol` — the docstring's figures 1.0e-3·A and 1.6e-2·A are `tol` itself, not the stated bound `2T·(ρ·A + α)`, which is twice them
- **DOC** `decl:Kernels.Matmul.model` — the argument that an argument equal to `1` gives the same model leaves out `K = 1` as the loop bound of line 294 (true, but unstated)

## Accepted

- `item:model`
- `item:map`
- `item:assumption:faithful`
- `item:assumption:dot`
- `item:assumption:dot_total`

## Checked

- Every Lean file of the kernel (`Matmul.lean`, `Defs.lean`, `Arith.lean`, `Basic.lean`, `Program.lean`), read in full. There is one `namespace Kernels.Matmul`, one `open TVA` per file, no instance, notation, attribute, `export` or `set_option`, and no binder named like a library namespace. `#print` with `pp.fullNames` (`work/Names.lean`) and with `pp.notation false` (`work/Instances.lean`) shows every operation of `matmul_kernel`, `model` and `launch` as `TVA.*` or core `HAdd`/`HMul`/`HSub`/`Bind`. `DotCfg`'s defaults are `precision := tf32` and `maxImpreciseAcc := 0`, as the model's `tl.dot` needs. `#print axioms` for `main`, `consistent` and `satisfiable` shows only `propext`, `Classical.choice` and `Quot.sound`.
- The model against lines 252–316, by SEMANTICS.md:
  - `tl.cdiv` is `(x + (B - 1)) // B` in `int32`, with `B - 1` computed by Python.
  - The constexprs are built in `int32`.
  - Python's `min` is `tl.iminimum`.
  - The two `%` of line 259 come in Python's order, and line 260 computes `pid % num_pid_in_group` again.
  - The eight `tl.assume` checks keep the source's order and signed comparisons with the `int32` `0`.
  - `offs_am`/`offs_bn` take `% M`/`% N`.
  - `col`/`row` give `x[:, None]`/`x[None, :]` (definitions printed).
  - Broadcasting before an elementwise product gives the same values as Triton's `[m,1]`·scalar then `[m,n]` broadcast.
  - The loop is `range(0, cdiv(K, 64))` with the `int32` variable and the carried `(a_ptrs, b_ptrs, accumulator)`. No body name or loop variable is used after the loop.
  - The masks are `offs_k < K - k*64`, signed, broadcast to the pointer shape.
  - `other=0.0` is the `float32` `+0` converted to `float16`.
  - `tl.dot` is `f16×f16→f32` with the defaults and the carried accumulator (no `dot_add` shape).
  - `+= 64*stride` is the `int32` product, broadcast.
  - `.to(tl.float16)` is `tl.cvt` `.rn`.
  - Line 314 is the two pointer additions left to right, and line 315 is the two signed comparisons and `&`.
  - The store needs no conversion.
- Integer arguments equal to `1`, one by one: `M`, `N`, `K` (incl. `tl.cdiv(1, B)` and `range(0, 1)`), `stride_ak = stride_bn = stride_cn = 1` (every host launch: `tl.assume(True)`, `x * 1`, `a_ptrs + 64`) and `stride_cm = 1` (`N ≤ 1` by `c_rows`). Each gives the same model.
- The drop of lines 306–307 (`ACTIVATION == "leaky_relu"`). Both sides are compile-time values, `""` on every launch (`matmul`'s default; no caller passes another), so the branch is never visited and `leaky_relu` is never read.
- The launch:
  - The grid `cdiv(M,128)·cdiv(N,256)` is the host lambda.
  - The ten block shapes are every block value of the source, checked shape by shape.
  - `grid1` gives `fusion = true`, `debug = false`, `numCtas = 1`, `pdl = false`.
  - `num_warps = 8` and `num_stages = 3` are meaning-preserving options.
- Part (A) of `Compiler.Faithful`'s scope, points 4, 5, 6, 8 and 9. The launch is a direct JIT launch with only `num_stages`/`num_warps`: no `@triton.autotune`/`heuristics`, no descriptors, no `CompiledKernel`, no host `TensorDescriptor`, and `do_bench` (not `do_bench_cudagraph`) in the benchmark.
- The supported subset, line by line, for 228–316. The module-level names read are `triton` and `tl`, bound only at lines 154–155. The host lines and the unused headers (164, 202, 216, 320–321) bind none of `triton`, `tl`, `matmul_kernel`, `leaky_relu`. The host's module-level `a`, `b` are not what the kernel's locals `a`, `b` read. `kernel.json`'s classing of every function and module-level line is right.
- The conditions:
  - Each field against the kernel's needs: no `int32` wrap on enabled lanes, positive strides required by `tl.assume`, row separation for distinct stores, and lengths.
  - The host's instances: `a` contiguous `(K,1)`, `b` `(N,1)`, `c` `(N,1)`, sizes ≤ 4096.
  - Edge sizes `M`/`N`/`K = 0`.
  - The README's table and Narrowings match `Inst.ok` field for field.
- The specification:
  - The indices in `OutSpec` match the kernel's pointer arithmetic, and `c` is pinned as `cvt` of a finite `float32` within `tol`.
  - The room clause is right (`binary16.u = 2^-11`, `eta = 2^-25`).
  - The frame `∀ z ∉ c.elems cIdx, fin z = init z`.
  - `tol` evaluated at 512, 4096, 2^13–2^16 (`work/Tol.lean`, python).
  - Rejection of dropped terms and of `float16` accumulation at every covered `K`, and of a `bfloat16` result up to `K ≈ 15700`.
- `dot` is `DotF16` (`f16` operands into `f32`, `maxImpreciseAcc = 0`, any precision: num.md's table), and `dot_total` is `DotTotal` with the model's own `DotCfg`. Each docstring matches `DotAcc`'s statement. There is no `[kernel]` assumption.
- The README against the Lean: the arithmetic of the coverage paragraph (`2^24 + 64`, `266240`, `2^24`), the tolerance figures, the proof outline, `consistent` and `satisfiable`.

## Not checked

- The proofs' internals beyond what Lean checks (I relied on Lean for `main`, `consistent` and `satisfiable`), and the gate itself (not re-run).
- Part (B) of `Compiler.Faithful`'s scope (the environment), by the protocol.
- Whether Triton accepts `tl.assume` of a compile-time `True`, which happens when a stride is `1`. If it rejects it, the launch has no outcome. If it accepts it, the model's `tl.assume true` is a no-op either way.
- How `triton.testing.do_bench` times its calls, beyond its being the plain (not CUDA-graph) benchmark function.
