# Review of matmul_e5m2, round 1, by r-e5m2

## Verdict

accept-partly
The model, the map, the conditions and the five `[rule]` assumptions are right: the model is the source line by line on every covered instance (including the compile-time `1` of `stride_ak`, `stride_bk` and `stride_cn` in the script's fp8 run), the drops are sound, the conditions cover the test and every benchmark size of the fp8 run with each narrowing stated, and each rule is the library's fact for the model's operation and configuration. The specification is where the claim is weak. Its tolerance, built from `DotLoop.bound`, exceeds the sum of magnitudes from `K = 2817` on, so on the benchmark's ten largest fp8 sizes `Post` accepts an all-zero `c`; and from `K = 448`, the test's `K = 512` included, it accepts outputs rounded to float8e5 precision. The per-step library fact the proof already uses gives a tighter, order-aware tolerance (at most `tol`) that rejects both at every covered `K`. So I do not accept `spec` yet.

## Findings

- **SPEC_TOO_WEAK** `item:spec` — from `K = 2817` to `4096` `Post` accepts an all-zero `c`, and from `K = 448` (the test's `512` too) a `c` rounded to float8e5; a tighter, order-aware tolerance follows from `DotE5m2`.
- **DOC** `doc:kernels/matmul_e5m2/README.md` — "no bound tighter than the fact follows" is inaccurate, and `Post` still rejects a wrong sign and a read of another row or column for `K > 2816`.
- **LIBRARY_REPORT** `decl:TVA.Rules.Num.relDot8` — the per-chunk `2^-7` is about twice the `33·2^-13` plus a rounding that the docstring derives; at `K = 4096` that is `1.708` instead of about `0.673`.

## Accepted

- `item:model`
- `item:map`
- `item:conditions`
- `item:assumption:faithful`
- `item:assumption:dot`
- `item:assumption:dot_total`
- `item:assumption:cvt`
- `item:assumption:cvt_total`

## Checked

- Every Lean file of the kernel in full (`MatmulE5m2.lean`, `Defs`, `Arith`, `Basic`, `Program`): no instance, notation, `open … renaming`, local binder or kernel definition that changes a library name; the model, `model`, `launch` and `Assumptions` printed with `pp.fullNames` (`work/names.lean`): every operation is `TVA.*`; `Assumptions` is a structure with exactly `(K : Claim.Behaviour Inst) (P : Platform)`; `#print axioms` of `main`, `consistent` and `satisfiable` gives only `propext`, `Classical.choice` and `Quot.sound`.
- The model against lines 252–316 by SEMANTICS.md: the constexprs built in `int32`; `tl.cdiv` as `(x + (B − 1)) // B` with the wrapping `+` and `tl.idiv`; Python's `min` as `tl.iminimum`; the two `%` of line 259 and the repeated one of line 260; the eight `tl.assume` checks; `% M`, `% N` on lines 282–283; the broadcasts `[:, None]`, `[None, :]` (`col`, `row`); the masks and `other=0.0` (a `float32` `+0` converted on the device, `tl.convert .f32 .f8e5`); `tl.dot` with the defaults `tf32` and `max_num_imprecise_acc = 2^30`; the loop bound evaluated once and the carried names; the `+=` steps; `.to(tl.float16)` as one rounding; the two pointer additions of line 314; the mask and store of lines 315–316. Also at the compile-time value `1` of an integer argument (`K`, `M`, `N`, a stride).
- The map's 40 entries and the drops of lines 154, 155, 228 and 306–307 (`ACTIVATION = ""` on every launch the script makes, so line 307 is never visited).
- The source's classing in `kernel.json`: the launch runs only `matmul_kernel`; `leaky_relu` is not visited; the three `get_…_autotune_config` and their headers bind nothing the kernel reads; the host's module-level `a`, `b` are never read by the kernel (its `a`, `b` are assigned in the loop body before use); `triton` and `tl` are bound only by lines 154–155; the supported subset (statements, builtins, operators, the `str` constexpr, no fp8 arithmetic).
- The launch: grid `cdiv(M, 128)·cdiv(N, 256)`, the ten block shapes, `num_stages=3` and `num_warps=8` only, a bare `@triton.jit`, no descriptors, no `CompiledKernel`, no graph capture (`do_bench`, not `do_bench_cudagraph`).
- The conditions against the host: the test (`512³`) and every benchmark size `256, 384, …, 4096`, with `a` contiguous, `b = b.T.to(float8_e5m2)` of strides `(1, K)` and `c` from `torch.empty`; `fp8_run_ok`; each `int32` bound against the offsets of lines 253–254, 282–286, 294–303 and 312–315; `c_rows`, `c_a`, `c_b`; `satisfiable`.
- The specification at every covered size: the tolerance's table in the write-up (recomputed exactly), what it accepts at `K = 448`, `512`, `2817` and `4096` (`work/tolerance.lean`: `zero_meets`, `round8_meets`), and the order-aware recurrence (`ord_step`), evaluated on adversarial inputs and on `torch.randn` data.
- Each `[rule]` field against docs/rules/num.md: `DotE5m2` for `float8e5 × float8e5 → float32` at `2^30`, any precision; `DotTotal` for the model's `DotCfg`; `CvtF32E5m2` and `CvtTotal … .rn` for the `other=` conversion into `float8e5`; `Compiler.Faithful` in the cited form.
- The write-up against the Lean (Specification, Conditions, Assumptions, Proof, Narrowings).

## Not checked

- Part (B) of `Compiler.Faithful`'s scope (the environment), as the protocol says.
- Whether the library's facts and constants are true (`DotE5m2`, `CvtF32E5m2`), beyond the LIBRARY_REPORT on how loose `relDot8` is.
- The proofs themselves beyond what Lean checks: I read them for what the names mean, not to re-derive each step.
