# Review of grouped_matmul_tma_f16, round 1, by r-gmm

## Verdict

accept
I looked for a line where the model and the source part ways, a covered use the conditions leave out, a size where the tolerance means nothing, and a cited fact that is not the operation's, and found none that blocks. The model follows `grouped_matmul_tma_kernel` line by line, by SEMANTICS.md: the compile-time conditional of line 209, `int32` index arithmetic that wraps, `tl.cdiv` as `(x + (div - 1)) // div`, the runtime `if` and the `while` with both comparisons evaluated, three in-kernel descriptors (`int32` shape, `int64` strides, zero padding), `tl.desc_load`/`tl.desc_store`, `tl.dot_add` with `b.trans`, and one rounding to `float16`. Every name in the kernel's Lean is the library's or a helper I read. `Inst.ok` covers every launch the script makes: the test, both benchmarks, `group_size = 4`, `NUM_SM = 132`, `K` up to `2^13`, strides multiples of 8, fresh buffers. Each narrowing is stated with its reason. `Post` fixes every output element within a tolerance that is 0.253% of `Σ|a·b|` at the largest covered `K`, and still rejects a `bfloat16`-rounded result there. It also says that every other byte is unchanged. The three `[rule]` fields are the library's facts for the model's `tl.dot`. I raise one non-blocking `DOC` finding on the stated `bfloat16` threshold, and one `LIBRARY_REPORT`.

## Findings

- **DOC** `doc:kernels/grouped_matmul_tma_f16/README.md` — The bfloat16 margin leaves out Post's last float16 rounding: the whole bound on c is 0.253% at K = 2^13, and it reaches bfloat16's 0.389% from about K = 13700, not 15745
- **LIBRARY_REPORT** `lib:docs/rules/compiler-faithful.md` — Scope 5 (descriptor scratch) says nothing of the scratch's alignment, though the launcher passes the allocator an alignment that the tutorial's alloc_fn ignores

## Accepted

All items.

## Checked

- Every Lean file of the kernel, in full (`GroupedMatmulTma.lean`, `Defs`, `Sched`, `Basic`, `Tile`, `Program`). Each has `open TVA` and the namespace `Kernels.GroupedMatmulTma`, and none has a notation, instance, `set_option`, `export` or renaming. The kernel's `def`s and `abbrev cfg16` are proof helpers or the test instance. `#print` with `pp.fullNames` (work/names.lean) shows every model operation as `TVA.*`. `#synth` (work/instances.lean) shows that pointer `+` is `TVA.instHAddPtrIntV` (`Ptr.offset` by the signed `toInt`) and that `+ - *` on `I32` are the library's wrapping operations. `tl.dot_add`'s default `z` is `+0`, `tl.cvt`'s default rounding is `.rn`, and `Launch.grid1` keeps `debug = false`, `num_ctas = 1` and no PDL. The axioms are `propext`, `Classical.choice` and `Quot.sound`.
- The model against source lines 186-278, entry by entry in the map, by SEMANTICS.md:
  - the parameter dtypes (`*int64` address tables, `*int32` size tables, `group_size` an `int32`, constant `1` giving the same one-trip loop);
  - the drops: line 209 (only the `else` side is evaluated; the dtype is kept as it is), the imports of lines 32-33 and the decorator of line 186;
  - `last_problem_end = 0` as an `int32` tensor; the pointer offsets `g * 3 + c`; the scalar loads; `tl.intToPtr`;
  - `tl.cdiv` at lines 217, 218 and 263, with `div - 1` computed by Python; the signed `>=` and `<`, and `and` evaluating both operands;
  - the `if` merging only `tile_idx`; the names new to each branch or loop body, none used after it;
  - `make_tensor_descriptor`: the shape in `int32`, the strides `lda` and `1` as `int64` (the same values), block shapes `[128,64]`, `[128,64]` and `[128,128]`, padding `"zero"`;
  - `desc.load` and `desc.store` offsets in `int32`; `acc += tl.dot(a, b.T)` as `tl.dot_add` (zero default accumulator, no other use of the product; `tf32` and `0` defaults); `.to(tl.float16)` as `tl.cvt`.
- `kernel.json`'s classes:
  - the claimed launch runs only the kernel and Triton's own `tl.cdiv`;
  - the grid lambdas and `alloc_fn` run on the host;
  - `grouped_matmul_kernel` is never launched by this claim, and its header (lines 52-72) binds only its own name;
  - the module-level code (`DEVICE`, `tma_configs`, the test's `group_size`, `M`, `N`, `K`, `A`, `B`, `i`, the `perf_report` decorators) binds nothing the kernel reads: the kernel's `group_size` is a parameter, and it reads no other global but `tl`;
  - lines 32-33 are kernel lines and mapped;
  - no form outside the supported subset.
- The launch: grid `(NUM_SM,)` from `META['NUM_SM']`; options `num_stages=3, num_warps=4` only; no autotune, heuristics, warmup, host descriptor or graph capture (`do_bench` is not `do_bench_cudagraph`). The allocator is set at line 320 before every TMA launch (the benchmarks' `triton-tma` lines run only under the same `supports_tma()` as line 355), and it returns a fresh allocation: scope (A) points 4, 5, 6, 8 and 9 hold.
- The conditions against the host's sizes:
  - test `M = N = K` from 128 to 1024; square benchmark `N` from 128 to 1024 with `g_lds = [N, N, N]`; batch benchmark `M` from 128 to 1024 with `N = K = 8192` and `g_T_lds`;
  - `start 4` at most 2048; every stride a multiple of 8; `ldc = N`; fresh buffers, aligned and disjoint;
  - `K_le` exactly at the host's largest `K`.
  
  `Pre` matches what the host writes into the tables. `satisfiable` is the test instance. The zero edge cases follow from the library's rules: a group with `M = 0` or `N = 0` has no tile, and with `K = 0` a descriptor is built but never accessed, and `C` is `0`.
- The specification, in exact rationals (work/tol.py):
  - the tolerance at `K` = 128, 1024 and 8192, and the README's numbers for `rho`, `alpha` and `(1 + rho)^T - 1`;
  - a witness at `K = 2^13` (`S = 8224`) where Post allows 20.79 and a `bfloat16` rounding is 32 off;
  - Post rejects dropped tiles or products, wrong tile mapping, a wrong reduction length and writes outside the matrices, through its frame;
  - the `K = 0` and non-finite cases are as the write-up says.
- The assumptions: `DotF16 P` and `DotTotal P cfg` are num.md's facts for an fp16×fp16→fp32 `tl.dot` with `maxImpreciseAcc = 0` and the model's `cfg`, and `faithful` has the standard form. There is no `[kernel]` field. The write-up's Specification, Conditions, Assumptions and Proof match the Lean, apart from the DOC finding.

## Not checked

- Whether the library's facts are true (DotF16's constants, the descriptor and `tl.cdiv` semantics, Compiler.Faithful), as the protocol asks: trusted.
- Part (B) of Compiler.Faithful's scope (the environment): assumed, not checked.
- The Lean proofs beyond what Lean checks: I read them for what their names mean, not step by step.
