# Library review, round 9, by lr-9a

## Verdicts

- **L4** ACCEPT — Every fact of `TVA.Rules.Num.Sm90` is true of Triton 3.8.0 on the H200, on TVA's
  own evidence, and can be cited by name. The lowerings agree with the pinned sources: `tl.exp` is
  `fmul` by `log2 e` and `llvm.nvvm.ex2.approx.f32` (`ExpOpConversionApprox`,
  `primitives/triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:634-658`); `/` on `float32` is
  `llvm.nvvm.div.full` (`:553-580`); the fp8 conversions are `cvt.rn.satfinite.{e4m3,e5m2}x2`, with
  `bfloat16` widened first (`:186-246`); `enable_reflect_ftz` defaults to true and
  `max_num_imprecise_acc_default` is `2^30` on sm_90 (`triton-3.8.0-nvidia-compiler.py:122, 214`);
  `semantic.py` gives every non-fp8 dot `max_num_imprecise_acc = 0`; `dot_scaled` becomes a
  `bfloat16` `tl.dot` after an exact scaling (`triton-3.8.0-DecomposeScaledBlocked.cpp`). I re-ran
  all six evidence folders in full, and each output is identical to its `out.txt`. I then tested
  what the evidence does not settle. With `enable_fp_fusion=False`, a launch option `Faithful`
  admits, the six math functions give the same bits as the default on all `2^32` inputs, although
  the PTX changes to `mul.rn`/`add.rn`; the ieee dot stays `fma.rn.f32`. The PTX of
  `tl.log(x) + y` ends `__nv_logf` in a `selp`, so no later addition can fuse into it. In
  `tl.dot`, subnormal `float16`, `bfloat16` and fp8 operands paired with large partners (products
  up to about `2`) are not flushed: no element is outside its fact. The worst case of the truncation
  window (accumulator `2^(w+1)`, every product just below the step) reaches `0.47` of `relDot8` on
  `wgmma` and `0.12` of `relDot16`. A register-operand `wgmma` (A converted in the kernel) gives the
  same bits as the shared-memory form for all four 16-bit and fp8 operand types. `CvtRn` holds
  against the Lean definition itself: the device equals `Format.cvt … .rn` bit for bit on every
  `float16`/`bfloat16` input and on 428k `float32` inputs around every fp8 tie, evaluated by
  `lake env lean --run`. `AtomicAddF32` holds on every pointer form the evidence skips: the scalar
  `atom.add.f32`, sem `acq_rel`/`relaxed`/`acquire`/`release`, scope `cta`/`gpu`/`sys`, with the old
  value used. It also holds on the 2-D TMA reduction, from a host descriptor and from
  `tl.make_tensor_descriptor`. The facts hold together: `Platform.sm90ref` satisfies each of them,
  and those proofs use only `propext`, `Classical.choice` and `Quot.sound` (work/axioms.out).

  **The two bounds that are met.** For exp2 the account is right. A flushed result's error is the
  reference itself, which is below `2^-126 = abs`. A result that is flushed while its true value
  lies just above `2^-126` stays inside the bound too, because the relative error is at most
  `2^-22.71 < 2^-22`. The ratio tends to `1/(1 + 2^-22)` from below and cannot reach 1; "reached
  exactly" is loose wording. For the ieee dot the account is wrong in its detail, but the fact
  holds. The `0.499` comes from the relative term on the `allones` input, where the error is
  `(K−1)·2^-24·mag` against `(K+1)·2^-23·mag`, a ratio of `(K−1)/(2(K+1))` (work/probes.out part
  1). It does not come from the absolute term on tiny inputs, where the ratio is 0. That is a factor
  2 of margin over the classical bound, as the other facts have (wording, below). "Constants whose
  meaning changed: none" is correct.

  **Tutorial needs.** No in-scope tutorial needs any item of "What the library does not state"
  except the fp8 saturation (NEW gap, below). Tutorial 10's mxfp4 and nvfp4 launches, with the
  tutorial's own configuration, are rejected on the host on the H200 (out of shared memory;
  work/probes.out part 3), and its mxfp8 and mixed launches run.
- **L33** ACCEPT — `sumc_err`, `sumc_err_covers` and `sumc_err_list` are stated for exactly the
  relation the semantics gives (`sumWith`: some tree with `t.Covers n` and `t.Eval (AddRel
  fusion)`; `tl.cumsumc` through `ScanLeaves.perm`). Their per-node premise is
  `FloatV.AddRel.err` (`3u·(|a|+|b|) + 4η`), which holds for every fusion `AddRel` allows.
  `Tree.eval_err_abs` keeps `errBound`'s shape. `sum_bcast_mul_err` takes both disjuncts of
  `tl.sum_bcast_mul`'s definition (`Lang/Ops.lean:673-680`) for `float32`, and its ieee side cites
  `DotF32Ieee`, which holds on the device (above), also with `enable_fp_fusion=False`. Every result
  uses standard axioms only. The bounds are as strong as `AddRel` allows.
- **L34** ACCEPT — `SafeA.atomic` follows `tl.atomicWith`: one step per enabled lane, for every
  permutation of the enabled lanes, with arbitrary old values on the disabled lanes. The
  integer-addition, max/min and float-accumulation lemmas are proved for any order, with standard
  axioms. On the device a tensor atomic is one read-modify-write per enabled lane, even when the
  layout holds an element in several threads. I checked 116 configurations (1-D tensors of 1 to 512
  elements with and without a mask, 2-D tensors, a scalar atomic, `num_warps` 1 to 8; work/atomic_lanes.out):
  every cell was incremented once per program. `AtomicAddF32`, which `stepRel_atomicAdd`
  consumes, holds on every form tested (L4 above; work/atomic_scalar.out, work/atomic_tma2d.out).

## New issues

- **NEW** soundness — A `[rule]` field can state a device fact with constants the kernel chose, and
  nothing mechanical stops it. The gate's `kinds` check (`tools/tvr.py`, `kinds_problems`) refuses
  only constants in the kernel's own namespace, so a field such as `exp_acc : Rules.Num.MathAcc P
  .exp .binary32 Set.univ (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ))` passes it. `Platform.ref`
  (`mathAcc_of_eta`) satisfies the field, so `consistent` passes as well. The field is false of the
  device: `tl.exp` reaches `2^-17.99` relative error (evidence/num-math-f32). The guidance leads to
  it. `docs/PROVER_GUIDE.md`'s table gives "accuracy of `tl.exp` …: `Rules.Num.MathAcc`" and
  "accuracy of `tl.dot`: `Rules.Num.DotAcc`" (the shapes, not the named facts). It says "`rel` and
  `abs` of `Platform.ref` are the bundle's `MathAcc` constants" (`:199`) and never mentions
  `Platform.sm90ref`. `docs/REVIEWER_GUIDE.md` §4 (`:121`) tells the kernel reviewer that "the gate
  ensures the statement adds no constant of the kernel's own". Only the checklist in
  `docs/rules/num.md` ("No constants of the kernel's own", `RULE_MISCITED`) still catches it.
  **Fix:** have `kinds` accept, in a `[rule]` field, only a stated fact applied to `P` and
  `RULE_ARGS`: a `TVA.Rules.Num.Sm90` constant, `DivFull`, `DivRn` with `.binary64`, a `…Total` rule
  with enumeration arguments, or `Compiler.Faithful`. It should never accept `MathAcc`, `MathAt`,
  `DivAcc`, `CvtAcc`, `CvtRn`, `DotAcc`, `DotScaledAcc` or `AtomicAddAcc` themselves. Correct the two
  guides to match: cite the named fact, and prove `consistent` with `Platform.sm90ref`.
- **NEW** gap — The fp8 conversions are stated only for `|x| ≤ maxVal` (`CvtRn`), so a claim can say
  nothing about larger values. In-scope tutorials convert unbounded `float32` accumulators into
  fp8: tutorial 08's `grouped_matmul_tma_kernel` (`accumulator.to(dtype)` with `FP8`), tutorial 09
  (`accumulator.to(tl.float8e4nv)` in four kernels), tutorial 06's `acc.to(float8e5)` (an average of
  `v` that can round just above 57344), and tutorial 10's `output_type == 2`. evidence/num-cvt8 already
  shows on every input that finite values beyond `maxVal` and `±∞` give `±maxVal`, and NaN gives
  `0x7f`. That is `Format.cvt f g .rn true .canonical` (sat on) on non-NaN inputs. **Fix:** state it
  as a named fact for the six pairs (for example `CvtSatF32E4m3`: for non-NaN `x`, `r =
  cvt … .rn true .canonical x`), citing num-cvt8, and add it to the table in `rules/num.md`.
- **NEW** gap — Device statements in `lean/TVA/Math/Float` rest on validation that TVA does not
  have. `F32.lean`'s module docstring cites a "Validation record (ported from v1): … H200 NVL
  (driver 610.57.04, nvcc 13.2, 2026-10-20)". `Convert.lean` and `Format.lean` (`NaNRule.canonical`)
  cite "the conformance records `docs/evidence/conformance/cvt.md` (… nvcc 13.2, 2026-10-25)" and
  "findings F1, F2". TVA has no such folder or records: they are nvcc runs, not Triton 3.8.0. Protocol
  §2 (Independence) allows such evidence "not even as a pointer", yet `tvr audit` prints
  `independent`. The statements themselves hold through Triton 3.8.0. work/cvt_std.py checks every
  `float32` input into `float16` and `bfloat16` with `"rtne"` and `"rtz"`, every `float16` and
  `bfloat16` widening, `float16`↔`bfloat16`, and `float32` through `float64`: 0 differences from the
  exact rounding. work/exact_ops.py runs `+`, `-`, `*` on all three formats, and `div_rn`, `sqrt_rn`
  on `float32`, with 2·10^8 adversarial pairs each: 0 differences. work/cvt_lean.py finds the Lean
  `Format.cvt` (`.rn`, `.rz`) equal to the device. **Fix:** put TVA's own evidence for the
  standard conversions and the exact operations in `evidence/` (these scripts can serve), and
  replace the citations with it. Have the audit refuse a cited `docs/…` or `evidence/…` path that
  does not exist.
- **NEW** wording — The evidence accounts of the facts are wrong in four places.
  evidence/num-dot README (conclusion 4, and the "(tiny inputs)" label in its table) attributes
  `DotF32Ieee`'s `0.499` to the absolute term on tiny inputs, and the round's notes say "by
  construction". The ratio is the relative term on the `allones` input, `(K−1)/(2(K+1))`. Part 2's
  "worst input kind" column names the kind with the largest `err/u`, not the largest ratio.
  `docs/rules/num.md` ("`tl.dot`") calls `(K+1)·2^-23` "the classical bound for `K` roundings";
  it is about twice `γ_K ≈ K·2^-24`. evidence/num-math-f32 conclusion 3 and `rules/num.md` (after the
  facts table) say exp2's bound is "reached exactly"; it is approached from below and cannot be
  reached. In "What the library does not state", "any approximate function on `float16`,
  `bfloat16`" names functions Triton rejects (`triton-3.8.0-math.py`: `fp32`/`fp64` only). The
  list's `dot_scaled` entry does not name `e2m1` by `e2m1`, which compiles and runs on sm_90 at a
  small tile (work/probes.out part 2). Tutorial 10's fp4 launches are rejected only because its
  tiles do not fit in shared memory. **Fix:** correct the four texts.

## Re-ran

- All six evidence folders, in full, each output identical to its `out.txt`: num-math-f32 (every
  `float32` input), num-div-f32 (3.08·10^10 pairs), num-cvt8 (every input), num-dot (Parts 1–4),
  num-dot-scaled, num-atomic-add. Run with Triton 3.8.0 on the H200 NVL, GPU 0, 2026-10-28, from
  TVA's `.venv`.
- work/math_nofusion.py: the six math functions on every `float32` input, the default against
  `enable_fp_fusion=False`, bit for bit, with the PTX of both.
- work/dot_adv.py: A, subnormal operands against large partners; B, the truncation's worst case;
  C, register-A `wgmma` against shared-memory A, bit for bit, for `float16`, `bfloat16`,
  `float8e4nv` and `float8e5`; D, the ieee dot with `enable_fp_fusion=False`.
- work/atomic_scalar.py: scalar and strided pointer atomics with every `sem` and `scope`, and with
  the old value used. work/atomic_tma2d.py: 2-D host and in-kernel descriptors. work/atomic_lanes.py:
  one read-modify-write per enabled lane, 116 configurations.
- work/cvt_lean.lean with work/cvt_lean.py: the Lean `Format.cvt` against the device. For fp8:
  every `float16`/`bfloat16` input and 428k `float32` inputs. For `float32` into
  `float16`/`bfloat16`, `.rn` and `.rz`: 392k inputs.
- work/cvt_std.py: the standard conversions, every input. work/exact_ops.py: `+`, `-`, `*`, `div_rn`
  and `sqrt_rn`, sampled.
- work/probes.py: which input gives the ieee dot its ratio; `e2m1` by `e2m1` `dot_scaled` on sm_90;
  tutorial 10's kernel in each format with its own configuration.
- work/axioms.lean: `#print axioms` for the round's new results.
- The release check, from the coordinator's run of the merged tree (not re-run): build and
  `leanchecker` pass, the audit prints `independent`, and only `vec_add_f32`'s gate and pages
  fail, as expected.

## Not checked

- Division beyond the evidence: `2^64` pairs cannot be run exhaustively, and I added no division
  inputs of my own.
- `tl.dot` beyond the largest `K` each block shape compiles. The facts' extension to every `K`
  rests on the per-chunk derivation, which my worst-case inputs (B) approach within the stated
  factors.
- The proofs of `SumFused` and `SharedAcc` line by line: Lean checks them. I read their statements
  against the semantics and their uses.
- Kernel claims: none cites the new facts yet.
