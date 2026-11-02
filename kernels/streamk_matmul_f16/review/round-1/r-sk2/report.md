# Review of streamk_matmul_f16, round 1, by r-sk2

## Verdict

accept
The model is the source line by line, by SEMANTICS.md, on every covered instance, the integer arguments equal to `1` included. Every name in the kernel's Lean resolves to `TVA.*` or to a kernel helper I read. The schedule, the split of `c` into owned tiles (stored by their one owner) and split tiles (atomically added by every program that touches them, stored by none), and the `int32` arithmetic hold on every covered instance. I checked them by reading the Lean, by simulating the test's instances and 3000 random covered ones, and at the largest covered sizes. The conditions cover the wrapper's launches, and each narrowing is stated with its reason. `Post` bounds every element of `c` whatever the order of the atomic additions, and fixes every byte outside the matrix `c`. Its tolerance is the best the library's constants allow (about `T·ρ·A`), and at `K = 2^13` it still rejects a `bfloat16`-rounded result and a dropped or duplicated partial sum. The five `[rule]` assumptions are the library's facts for the model's operations. My two findings are non-blocking wording slips about subnormal operands.

## Findings

- **DOC** `doc:kernels/streamk_matmul_f16/README.md` — README (Specification) says the test's data has no subnormal operand; torch.randn float16 gives subnormal operands at most test shapes, where A exceeds (|A| @ |B|)[i, j]
- **DOC** `decl:Kernels.StreamKMatmul.Inst.ok.K_le` — K_le's docstring says tol is below 0.21% of sum|a*b| (tol_le); tol_le bounds it by 0.21% of A = dotMag, which exceeds sum|a*b| when an operand is subnormal

## Accepted

All items.

## Checked

- Every Lean file of the kernel, read in full (`StreamKMatmul.lean`, `Defs`, `Facts`, `Sched`, `Basic`, `Parts`, `Chunk`, `Tail`, `Program`). They have no `instance`, `notation`, `export`, `attribute` or `set_option`, and no `open` but `open TVA`. No binder or kernel `def` takes the name of a library operation the model uses. Printed with `pp.fullNames` (`work/print_model.lean`, `.out`), every operation of `streamk_matmul_kernel` is `TVA.*`, and the `+`, `*`, `-` and comparisons are the library's `IntV`, `Tensor` and `Ptr` instances (wrapping `int32`, signed `IntV.lt`, `Ptr.offset`). The cone's `streamk_matmul_kernel._proof_1` is `NeZero 3`, for the `Fin 3` literal of `tl.program_id` (`work/print_aux.lean`). `#print axioms` of `main`, `consistent` and `satisfiable` gives `propext`, `Classical.choice` and `Quot.sound`.
- The model against the source, line by line, and all 30 map entries with their 3 drops (lines 22, 23, 26), by SEMANTICS.md:
  - `tl.cdiv(N, 128)` is `(N + 127) // 128`, with `127` computed by Python;
  - `tl.minimum` of `int32` scalars is `tl.iminimum`;
  - the `while` is `tl.whileLoop` with the signed condition and the carried `it`;
  - `range(k_first, k_last)` is `tl.forRange` with step `1` and the carried `acc`;
  - `other=0.0` is the `float32` `+0` converted to `float16`;
  - `tl.dot` has the defaults `tf32` and `maxImpreciseAcc = 0`;
  - the runtime `if` is on the `int1` `&` of two `==`;
  - the store converts nothing;
  - `tl.atomic_add` has the defaults `acq_rel` and `gpu`, and its result is unused;
  - the pointer sums are taken lane by lane, as Triton broadcasts them.

  An integer argument equal to `1`, a compile-time constant, gives the same value at each use (lines 37-40, 53-59), as `model`'s docstring argues. `tl.cdiv`'s two calls have different compile-time divisors, so their helper copies have different names.
- The supported subset. The source's forms are: a `while` with a runtime condition, a runtime `if`/`else` that assigns no name, `+=`, and no name first defined in a loop used after it. The only module-level names the kernel reads are `triton` and `tl`, from lines 22-23, which are kernel lines. The `BLOCK_M`, `BLOCK_N`, `BLOCK_K` of line 67 are shadowed by the constexpr parameters. The `__main__` block's `M`, `N`, `K`, `a`, `b` are parameters or locals of the kernel. No unused function. The only function the launch calls is Triton's own `tl.cdiv`. `kernel.json` classes the lines correctly.
- The launch. The grid is `grid1 num_programs`. The nine block shapes are every block shape of the source. `enable_fp_fusion` is `true`, `debug` is `false` and `num_ctas` is `1`. `num_warps=8` and `num_stages=3` are allowed. Scope (A) points 4, 5, 6, 8 and 9 hold for the wrapper's one direct launch.
- The schedule (`work/schedule_sim.py`, `.out`). I simulated it with `int32` checks, truncating `//` and `%` and every value bounded, on the six test instances and on 3000 random covered instances, including `iters_per_program` above the host's `cdiv`. In all of them:
  - each tile's `T` iterations are done exactly once;
  - an owned tile is stored by one program and touched by no other;
  - a split tile is atomically added by at least two programs and stored by none;
  - at most `T` programs touch a tile;
  - every loop ends.

  The stored and split tile counts match SETUP and the README: 1000^3 has all 64 tiles split, 512^3 all 16 split, 8192x8192x4096 has 3969 stored and 127 split, and 64^3 has its one tile stored with two idle programs. This agrees with the Lean (`owns_unique`, `iter_owns`, `iter_split`, `sum_parts`, `card_touch_le`, `lane_inj`).
- The `int32` arithmetic at the largest covered sizes (`work/int32_extremes.py`, `.out`). I took `M = 2^31 - 128`, `N = 1`, `K = 2^13` with 1 and 132 programs, `N` at its bound with `M = 1`, and `2^24 - 1` programs with one iteration each. `pid * iters_per_program`, `it + iters_per_program`, `k_first + (end - it)`, `tile * T`, `pid_m * 128 + 127`, `pid_n * 128 + 127` and `kk * 64 + 63` all stay in `int32`. The lanes inside the matrices stay in range by the fits.
- The conditions cover the wrapper's launches as the test makes them:
  - `float16` `a` and `b` with any non-negative strides within the fits;
  - a fresh, zero, row-major `float32` `c` (`c_rows` holds as `N <= N`);
  - the host's `total_iters` and `iters_per_program`;
  - any program count (`progs_lt` at most `262152` at the largest test shape).

  Each narrowing is stated in the README with its reason: the block configuration, the `int32` bounds, `K <= 2^13`, a zero `c`, the row layout of `c`, no overlap of `c` with `a` or `b`, and accuracy for finite inputs only. `satisfiable` holds of the 512^3 instance.
- The specification. `Post` holds of every outcome, so in every order of the atomic additions, with `Pre` giving the zero `c` the additions start from. The tolerance table, recomputed exactly (`work/tol_table.py`, `.out`), matches the README (`1.59e-5·A` at `K = 64` up to `2.03e-3·A` at `K = 2^13`, absolute part below `1.1e-34`). At `K = 2^13` the bound is below the `0.39%` of one `bfloat16` rounding, and below the `1/128` share of `A` that a dropped one-step part can carry.
- The assumptions. `faithful` is `∀ d, d.ok → Faithful P (model d) (launch d) (K d)`. `DotF16` is the fact for `float16` operands into `float32` with `maxImpreciseAcc = 0` at any precision. `DotTotal` is cited for the model's exact `DotCfg`. `AtomicAddF32` covers every `sem` and `scope`, and `AtomicAddTotal` is cited at `binary32`. These are num.md's listed facts for the model's operations. There is no `[kernel]` assumption, and `Assumptions` is a structure with exactly `(K, P)`.
- The write-up against the Lean: the conditions table, the narrowings, the assumptions and the outline of the proof match the definitions and theorem names, apart from the two DOC findings.

## Not checked

- The H200 error measurements that the README quotes from SETUP.md (no GPU).
- The subnormal counts under the H200's random generator: I counted them on the CPU generator only, and the finding rests on the normal law and on those counts.
- The proofs' internals, beyond what each name means (Lean checks them).
