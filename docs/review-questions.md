# Open review questions about the semantics

These are readings of the documentation that a library reviewer should confirm or overturn. A
change here changes the library definitions, not bundle proofs. Settled questions keep their
answer.

## Settled

- **S1. Barrier and intra-program ordering.** *Superseded 2026-10-27 by library issue L15*: one
  program is not sequential for global memory on the device, and the user chose to exclude
  conflicting accesses within a program. Round 3 implements it: `IntraRace` in `WellBehaved`,
  `Step.defer` for the order of a program's atomics, `Safe` writing each byte once, and
  [SEMANTICS.md](SEMANTICS.md) "Programs" ([evidence/intra-program](../evidence/intra-program/README.md)).
  The original entry follows. Settled by the user, 2026-10-26: "we do not need
  debug semantics here". A program is sequential; `tl.debug_barrier` has no meaning and is
  dropped from models. The intra-program hazard rule of an earlier draft is gone.
- **S2. Atomic defaults.** When the source gives none, `sem = acq_rel` and `scope = gpu`.
  `primitives/triton-3.8.0-core.py:2730-2734`: "If not provided, the function defaults to using
  "acq_rel" semantics", and "gpu" (default).

- **S3. Float `atomic_max`/`atomic_min`** (formerly Q3). Library review, 2026-10-26: Triton
  implements them on the bit pattern, `atomic_max` as signed max or unsigned min and `atomic_min`
  as signed min or unsigned max, chosen by the sign bit of `val` (`semantic.py:1299-1374`; library
  issue L174). The stored (new) value is always one of the two values, the old one and the operand,
  so `fBound` is sound; the returned value is the old one (library issue L168). fp16/bf16 are
  rejected (`semantic.py:1271-1275`).
- **S4. Library review 2026-10-26: findings fixed.**
  - NaN payloads are free (`FloatV.NaNEq`).
  - `maximum`/`minimum` may return either zero (`MaxRel`).
  - `AddRel` covers products that overflow.
  - Only `acq_rel` gpu/sys atomics on the same bytes are exempt from races (`Event.Strong`).
  - Compiler hints are checks.
  - `forRange` is generic in its integer type, with an overflow condition.
  - `tl.dot_add` covers the combine pass.
  - `debug` is excluded from `Faithful`.
  - The `AtomicAddAcc` guidance is `abs ≥ 2^(emin+1)`.
  - Tensor `//`, `%` and shifts are undefined on bad lanes.

- **S5. Float rewrites** (formerly Q1). Settled from the 3.8.0 sources, 2026-10-26. The
  sources are pinned in `primitives/` and listed in `primitives/semantics-manifest.json`.
  - The only float-math option of the NVIDIA backend is contraction. `enable_fp_fusion` sets
    LLVM's `AllowFPOpFusion = Fast` (`triton-3.8.0-llvm.cc:139-140`) and, when off, passes
    `--fmad=false` to `ptxas` (`triton-3.8.0-nvidia-compiler.py:532`). No fast-math flags
    (`reassoc`, `arcp`, `nnan`, `nsz`, `afn`) are set. `+`, `-`, `*` lower to plain LLVM
    `fadd`/`fsub`/`fmul` (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:895-897`).
  - Subnormals are kept: no denormal attribute is set. `enable_reflect_ftz` only selects the
    flush-to-zero paths inside libdevice (`triton-3.8.0-triton_nvidia.cc:301-313`), whose
    functions are platform relations here.
  - `AddRel` stays looser than the exact contraction `fma(a, b, c)`, because the model does not
    track which operands are products. The cost is half an ulp per fused operand in error
    bounds. `tl.mul_add` is the exact form when the product is visible.
  - The combine pass rewrites some float code regardless of `enable_fp_fusion`. See S7. The
    library covers this with `tl.dot_add`, `tl.dot_scaled_add` and `tl.sum_bcast_mul`.
- **S6. Masked-off lanes of `tl.load` without `other`** (formerly Q2). They hold a defined but
  unspecified value, not undefined behaviour, as the library already had it.
  - The IR canonicaliser leaves an all-false-mask load without `other` alone
    (`triton-3.8.0-Ops.cpp:64-101`).
  - The NVIDIA lowering initialises the destination registers to `0` before the predicated
    `ld` (`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:237-238`,
    `triton-3.8.0-PTXAsmFormat.cpp:22-51`).
  - The `cp.async` path zero-fills masked lanes (`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:1075-1085`).

  So in 3.8.0 on NVIDIA such lanes read `0`, but the library keeps "unspecified", which is
  what the documentation promises.

  One caveat for atomics: a masked-off lane of a bf16 atomic before sm_90 returns LLVM
  `undef` (`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:820-821`). The reviewer therefore
  checks that results of masked-off atomic lanes never decide control flow or an address.
- **S7. The combine pass** (formerly Q8). `triton-3.8.0-Combine.cpp`, run on every kernel in
  `make_ttir` (`triton-3.8.0-nvidia-compiler.py:252`). The rewrites that change float or
  integer results:
  - `dot(a, b, z) + c` or `c + dot(a, b, z)` becomes `dot(a, b, c)`
    (`CombineDotAddFPattern`/`AddIPattern`, `Combine.cpp:255-298`). It fires when `z` is a
    constant zero (`m_Zero` or `m_AnyZeroFloat`, so `±0.0`), the dot has one use, and, for
    float `tl.dot`, `max_num_imprecise_acc` is `0`. `tl.dot_add` (with the source's zero
    accumulator) allows both forms.
  - `dot_scaled(…, z) + c` becomes `dot_scaled(…, c)` (`CombineDotScaledAddFPattern`), under
    the same conditions. Covered by the new `tl.dot_scaled_add`.
  - `sum(x[:, :, None] * y[None, :, :], axis=1)` becomes `dot(x, y)` with
    `input_precision="ieee"` when the additions are at most 32 bits wide and `M, N ≥ 16`
    (`CombineBroadcastMulReducePattern`, `Combine.cpp:112-200`). Covered by the new
    `tl.sum_bcast_mul`.

  The other patterns are exact for what the library models:
  - masked-load `select` folding;
  - `addptr` merging when it cannot overflow;
  - a 1-D `reshape` feeding only reductions or histograms may reorder elements. Reductions take
    their leaves in any order (`Tree.Covers`), and a histogram does not depend on order. The
    model writes such a reshape as `tl.reshape_any`. The pattern does not look at how many
    operands the reduction has, so an index reduction (`tl.argmax`, `max(…,
    return_indices=True)`) pairs reordered values with a fresh, unreordered `arange`
    (`Combine.cpp:203-221`, `core.py:3093-3107`). The model must keep that pairing (round 2).
- **S8. Reverse scans** (formerly Q12). `reverse=True` is lowered as `flip(scan(flip(x)))`
  (`triton-3.8.0-ScanOpToLLVM.cpp:491-501, 559-562`). Every combination is
  `combine_fn(acc, cur)` with the earlier segment first, and an empty accumulator gives `cur`
  with no identity element (`triton-3.8.0-ReduceScanCommon.h:41-63`). So a reverse scan's
  position `j` combines `n-1, …, j` in that order. `ScanLeaves` now allows only that order.
- **S9. `dot_scaled` scale formats** (formerly Q14). Scale tensors may be `i8` (`e8m0`) or a
  float type (`triton-3.8.0-TritonOps.td:692-693`), and the group (scale factor) is `16` or
  `32`, deduced from the shapes (`triton-3.8.0-Ops.cpp:368-425`). So NVFP4-style
  `float8e4nv` scales with group `16` do reach `tl.dot_scaled`. The library now has
  `ScaleFmt` (`e8m0` or `float d`) and requires a group of `16` or `32`.

  The emulation decodes an `e8m0` scale by a bit shift, so `0x00` becomes `0`, not `2^-127`.
  With an `fp16` compute type it rounds scales to `fp16`. It converts float scales with their
  sign (`triton-3.8.0-DecomposeScaledBlocked.cpp:58-96, 141-225`). The platform relation
  covers all of this; a `DotScaledAcc` instance must be judged for the path the target takes.
- **S10. `/` on floats** (formerly Q6). Settled from the 3.8.0 and pinned LLVM sources,
  2026-10-26. The library keeps `/` as the platform's `P.div` and states its accuracy as rules:
  - `Rules.Num.DivFull`: `float32` `/` is `div.full.f32`, within `2^-22·|x/y| + 2^-148` for
    `|x/y| ≤ 2^126`. The PTX ISA gives 2 ulp; TVA's own measurement is `evidence/num-div-f32`.
  - `Rules.Num.DivRn`: `/` is IEEE; this holds for `float64` (`div.rn.f64`).
  - `DivAcc` now takes the quotient bound `qmax` and is stated over `ℚ`.

  The lowering chain and its citations are in [rules/num.md](rules/num.md) "Division".
  `DivFull.of_divRn` shows that `DivFull` is satisfiable. `fp16`/`bf16` `/` runs in `fp32`, so
  `DivFull` covers it after the model's conversions.

- **S11. Synchronisation between programs** (formerly Q4, on release sequences). Settled by
  the second library review, 2026-10-26: there is no synchronises-with edge at all. A program
  is many threads. A scalar atomic is issued by one thread under a predicate, with no barrier
  before it (`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:756-787, 914-916`); the only barriers
  come after a scalar atomic whose result is used (`:612, 807, 943`). So a release does not
  publish the program's other threads' writes. Tutorial 05 inserts `tl.debug_barrier()` before
  releasing its lock for this reason, and models drop that barrier. Happens-before is now
  program order only, so plain-memory hand-offs between programs race and are excluded. It
  also removed a partial-overlap hole: the old `SW` needed only overlapping bytes, not equal
  ones. The question of release sequences no longer arises. A future progress rule must give
  `tl.debug_barrier` a meaning before it allows publication.
- **S12. Second library review, 2026-10-26: findings fixed.** Six reviewers covered core ops,
  memory, execution semantics, tensors/scans/`dot_scaled`, numerics, and the examples (a folder since
  removed).
  - **Claims.**
    - `Compiler.Faithful` needs `Launch.Valid`: grid limits, `num_ctas = 1`, and block shapes
      that are powers of two with at most `2^20` elements.
    - Bundles prove `Claim.Consistent` (`exists_total_faithful`, `Platform.ref`).
    - VecAdd's `Inst.ok` now requires a power-of-two `BLOCK`.
  - **Memory model.** No synchronises-with (S11).
  - **Conversions.**
    - `tl.cvt` covers only the standard formats and takes `fp_downcast_rounding`.
    - `tl.cvtFrom8` makes fp8 upcasts exact only on finite values: `float8e5 → bfloat16`
      maps inf/NaN to finite numbers.
    - `tl.cvt8` and `P.cvt` take the rounding mode.
    - `tl.itof` allows double rounding through a wider format: before sm_90, and after the
      MLIR fold `truncf(sitofp x) → sitofp x`.
  - **Integers and loops.**
    - Shifts take integer amounts, and `>>` is explicit `tl.ashr`/`tl.lshr`, chosen by the
      left operand's source type.
    - `tl.forRange` requires `tl.RangeOk`, a signed-range check, since `scf.for` compares
      signed.
  - **`tl.dot`.** `InputPrecision` has `bf16x3`/`bf16x6`, and `DotCfg.maxImpreciseAcc` is the
    resolved value.
  - **Descriptors.** A descriptor has its fixed `block`, and `int32` shape entries.
  - **`dot_scaled`.**
    - Storage dtypes are checked, and only `e2m1` may be packed off `K`.
    - The two sides' scale formats are independent.
    - `DotScaledAcc` is per output element, with domains `dom`/`sdom` and `K`-dependent
      `rel`/`abs`. It was false on the sm_90 emulation path, where a scaled element can
      overflow in `bf16`.
  - **Numeric rules.**
    - `MathAcc` needs the natural domain (`MathFn.natDom`) and room in the format, because of
      Mathlib's junk values and `exp 126 > maxVal`.
    - New rule `MathAt` for special values.
  - **New operations.**
    - `tl.maxAll`, `tl.max0`;
    - `tl.randint`/`tl.rand` (Philox, exact; Random123 known answers).
  - **Documentation.**
    - Hints apply at the value's definition.
    - `multiple_of`'s `contig` must divide Triton's contiguity (withdrawn: what a `contig` above 1
      promises is open, library issue L2).
    - `-0.0` literals are `+0.0`.
    - Scan promotions are per operation.
    - Name shadowing and instances of library classes are rejected.
    - The fusion check covers `sub`, `cumsum` and scan relations.
    - `scalar atomic_add(p, 0)` is a load.
    - New exclusions: plain-memory hand-off, work-handing counters, `num_ctas > 1`, poison
      lanes, unsigned `range` bounds, 64-bit Philox.
  - **Verification pass** (a fresh reviewer, after the fixes).
    - Conversions Triton does not compile are now undefined behaviour in the model: `rtz`
      from `float16` into fp8, fp8 → `float64`, `tl.itof`/`x.to(int)` with fp8.
    - The `truncf(sitofp)` fold also reaches the implicit conversions of stores, atomics and
      `desc.store`.
    - `tl.rand` is a product (fusion).
    - `>>` with a Python-scalar left operand dispatches on the right tensor.
    - `Platform.ref` gained special values, IEEE conversions and IEEE float atomics, so it
      witnesses `MathAt`, `CvtAcc` and `AtomicAddAcc` too.
    - `Launch.Valid` checks only the listed `blocks`; the reviewer checks the rest.
  - **Witnesses** (follow-up). `Platform.ref` is now the most accurate platform. It witnesses
    every numeric rule for bounds of at least one rounding, including:
    - `MathAcc` without an absolute term where results are normal;
    - the new `tl.dot` rule `DotAcc`, for float and integer accumulators;
    - `DotScaledAcc`.

    The conditions are exactly one correct rounding: `rel ≥ u` (`FloatV.exists_nearest`),
    `abs ≥ η` outside the normal range, and only nonnegativity at `K = 0`. An unsigned integer
    accumulator cannot arise, since Triton 3.8.0's only integer `tl.dot` is
    `int8 × int8 → int32`.

## Open

- **Q9. `tl.multiple_of` contiguity.** The hint's meaning depends on the contiguity Triton
  assigns to the tensor, from its axis analysis. `contig` in the model is a claim about that
  analysis. Should the library pin the analysis rules for common patterns (`arange`, `+ scalar`,
  `* scalar`) so that `contig` becomes a computed fact?
  Tracked as library issue L2: until it is settled, the library's reading is `contig = 1`.
- **Q10. Relaxed atomics** (now also descriptor reductions, which are relaxed TMA
  reductions). Settled for pointer atomics by library issue L157 (user decision 2026-11-08): a
  `tl.atomic_*` of any `sem` whose returned value the source never uses, at `gpu` or `sys` scope,
  is modelled as the `acq_rel` atomic (`MemSem.unused`, `tl.atomic_add_unused` and the others), so
  two of them on the same bytes from different programs do not race; cells they share can be
  reached by no plain load or store of the launch (that would race). The reasons (PTX ISA 9.5
  §8.7, §8.10.3: relaxed `atom` operations at a common scope, on the same bytes, are morally strong
  and atomic; the semantics has no synchronises-with, so an atomic's order is observable only
  through a result the program uses) are in `MemSem.unused`'s docstring, and the device evidence in
  `evidence/relaxed-atomics`. Its premise, that the semantics already orders a program's atomics
  only by their dependences, is false across threads (library issue L161, Q17), and the launches
  where that shows are excluded. Open: relaxed atomics whose result is used, on shared cells, and
  descriptor reductions into elements other programs also reduce into (TMA reductions run in the
  async proxy; their atomicity with each other is not established here).
- **Q11. Proxy ordering for descriptors.** TMA copies use the async proxy. In PTX, a pointer
  (generic-proxy) access followed by a TMA access to the same global bytes needs
  `fence.proxy.async`. Since library issue L15 (round 3), two accesses of one program to the same
  bytes, one writing, are a conflict within the program whatever proxy they use, so the question
  remains for two reads (a pointer load and a descriptor load of the same bytes), which no fence
  orders anyway, and for two atomics, a pointer atomic and a descriptor reduction on the same bytes,
  which do not conflict but must be atomic with respect to each other across the two proxies.
  Until a reviewer confirms both, models in which the same bytes are accessed both through pointers
  and through descriptors stay out of scope (`docs/exclusions.md`).
- **Q13. Undocumented out-of-bounds behaviour.** `tl.histogram` with a value outside
  `[0, bins)` and `desc.scatter` with an out-of-bounds row are undefined behaviour in the
  library, and `desc.gather` gives unspecified values there. These are conservative. Pin the
  lowerings (`HistogramOpToLLVM`, the TMA gather4/scatter4 path) to relax them if a kernel
  needs it.
- **Q7. Granularity.** A tensor load or store is one step of the interleaving. This is claimed
  harmless for launches without a race between programs or a conflict within a program (L15),
  except for the order of a program's atomics: a tensor step that uses one lane's result holds back
  every later atomic of the program (Q17, library issues L161, L169: those launches are excluded). A reviewer may ask for the argument in Lean (a
  finer-interleaving refinement). The width of the device's atomics can also differ from the
  model's events. On sm_90, the target (library issue L172, round 38;
  [evidence/half-atomics](../evidence/half-atomics/README.md), H200):
  - a lane of a `float16` or `bfloat16` `atomic_add` is one `atom.add.noftz.f16` (`.bf16`), and a
    lane of a 16-bit `atomic_cas` one `atom.cas.b16`; `ptxas` compiles each to a 32-bit
    compare-and-swap loop (`ATOMG.E.CAS`) on the aligned word that holds the half, which writes the
    neighbouring half back with the value it read;
  - contiguous unmasked lanes that one thread holds become one vector atomic
    (`atom.add.noftz.v8.f16`, `.v8.bf16`; `ATOMG.E.ADD.F16`, `.BF16`): Triton 3.8.0 vectorises a
    `float16`, `bfloat16` or `float32` atomic add where the target supports vector atomics (sm_90),
    and packs two `float16` lanes into one `f16x2` atomic only where it does not
    (`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:636-646, 729-740`).

  The model's events for these atomics are 2 bytes and `Event.Strong`; the device's accesses are
  not those events (a 4-byte compare-and-swap, a 16-byte vector atomic). In
  evidence/half-atomics no value was wrong: in each form, with its result used and unused, no
  lane's new value differed from one IEEE addition (0 of 1024), and with 32 programs adding into the even halves of a
  buffer while 32 others stored plain values into the odd halves, 0 of 40960 cells were wrong over
  20 launches per format. That is a test, not a proof: this is part of the compiler trust for now.
  Confirm it before relying on fp16/bf16 atomics shared at 2-byte granularity.
- **Q15. Poison as undefined behaviour.** `x.to(int)` out of range and shifts by the width or
  more are LLVM poison, not immediate undefined behaviour, but the library reads them as
  undefined. That is sound but over-restrictive: `tl.load(p, mask=m).to(tl.int8)` without
  `other` cannot be proved, because masked-off lanes may hold NaN. A per-lane "any value"
  reading would need a reviewer check that such lanes reach no control flow, address or
  store.
- **Q16. The 16-byte offset rule for descriptors.** The documentation states it for `load` and
  `store`. The library also requires it of `desc.atomic_*`, `gather` and `scatter`, which is a
  conservative reading. Pin the TMA lowering to relax it.
- **Q17. A per-lane model of atomics whose result is used** (library issue L161, round 35). `Step.defer`
  lets a pending atomic be passed only by a later step that is the same for every old value, and a
  tensor `tl.load` or `tl.store` is one step, so a step that uses any lane's result holds back every
  later atomic of the program; on the device each thread orders only its own accesses (library
  review lr-34a; `evidence/atomic-result-order`: a two-program launch with `acq_rel` and a
  one-program launch with `acq_rel` and with `relaxed` whose device outcomes the semantics lacks, in
  200 of 200 runs each). By the user's decision (2026-11-08, "let's
  exclude with Fix B") the model is kept and every launch in which a program uses an atomic's
  result on a cell another program, or another atomic operation of the same program, also accesses
  is excluded (`docs/exclusions.md`; `TVA.UsesSharedAtomicResult`; lr-34a's launch is under it,
  `TVA.AtomicResult.crosswarp_excluded`). To do when a kernel needs atomic results across programs
  or calls (work queues, tickets, locks, "the last program reduces"): per-lane load and store steps
  that may be deferred while they depend on a pending atomic, with the thread mapping of each
  operation's layout, or an argument that a given kernel's layouts give each lane one thread
  throughout. A `tl.debug_barrier()` between the use of a result and the program's next atomic on
  the cell gave only outcomes the semantics has (`evidence/atomic-result-order`, `variants.py`: 0 of
  100 outside, against 100 of 100 without it; library issue L165), so such a model may give the
  barrier the meaning of ordering a program's accesses; until then it has none ("No barrier
  semantics" in SEMANTICS.md).
