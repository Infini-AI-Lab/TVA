# Library batch, round 9: author's notes

## L4: Device facts with constants, owned by the library, on TVA's own evidence

- **What changed.** New module `TVA.Rules.Num.Sm90` (`lean/TVA/Rules/Num/Sm90.lean`): one named
  constant per fact, with its constants inside the definition and, in its docstring, Triton's
  lowering and the evidence folder. The facts are:
  - math: `ExpF32`, `Exp2F32`, `LogF32`, `Log2F32`, `SqrtF32`, `RsqrtF32`, `ExpF32NegInf`,
    `Exp2F32NegInf`;
  - conversions: `CvtF32E4m3`, `CvtF32E5m2`, `CvtF16E4m3`, `CvtF16E5m2`, `CvtBF16E4m3`,
    `CvtBF16E5m2`;
  - `tl.dot`: `DotF16`, `DotBF16`, `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3`,
    `DotF32Ieee`;
  - `tl.dot_scaled`: `DotScaledE4m3`, `DotScaledE4m3E2m1`;
  - `AtomicAddF32` (the L34 fact).

  `TVA.Rules.Num` gains `CvtRn` (exact rounding to nearest even into fp8), `CvtRn.cvtAcc`, and
  the weakening lemmas `MathAcc.mono`, `DotAcc.mono`, `DotScaledAcc.mono`. Its `DivFull`
  docstring now cites TVA's own measurement. New module `TVA.Lemmas.Sm90`: `Platform.sm90ref`
  satisfies every fact (`Platform.sm90ref.expF32`, …, `.atomicAddF32`), so one platform proves
  `Claim.Consistent` for any bundle that cites them. `docs/rules/num.md` is rewritten. It now
  has: an example citation by name; a table from each operation to its facts, for kernel
  reviewers; the facts table (constants, lowering, evidence); one section per operation; what is
  not stated; satisfiability; and division with TVA's measurement.

  Files outside round 9: `lean/TVA.lean` (four imports), `docs/LIBRARY.md` (rows for the new
  modules and names), and `docs/review-questions.md` (S10: one line now cites
  `evidence/num-div-f32`).
- **Why.** Before this round, kernels chose their own constants, and the facts they relied on
  had been measured by another project. Facts are now the library's and named, so a kernel cites
  a name and never writes a constant (as L32 requires). Each fact rests on TVA's own run on the
  H200, with margin.
- **Evidence.** Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28. Each folder has a README, `run.py`
  and `out.txt`.
  - `evidence/num-math-f32`: every `float32` input through six functions. Nothing is outside a
    bound. The largest ratio to the bound is 0.60. `exp2` reaches 1.000, but only on flushed
    results, which is the bound by construction. `exp`/`exp2` of `-inf` give `+0`.
  - `evidence/num-div-f32`: 3.08·10^10 adversarial pairs. Nothing is outside the bound; the
    largest ratio is 0.651 (2.11 ulp).
  - `evidence/num-cvt8`: every input of every source format. Each conversion equals round to
    nearest even; values above the range saturate.
  - `evidence/num-dot`: the accumulation window on both paths, then seven operand pairs, seven
    block shapes, and every `K` that compiles (up to 8192). Nothing is outside a bound. The
    largest ratios are 0.124 (16-bit), 0.388 (fp8 on `wgmma`, a 2.5% relative error) and 0.499
    (`ieee`, by construction).
  - `evidence/num-dot-scaled`: 6.3·10^6 elements, both `fast_math` values. Nothing is outside the
    bound; the largest ratio is 0.108. The scale `0x00` is read as `0` and `0xFF` as NaN.
- **Constants whose meaning changed.** None. `DivFull`'s `2^-22`, `2^-148` and `qmax = 2^126`
  are unchanged. The new constants are each fact's own; `rules/num.md` lists them.

## L33: No error bound for reductions with fusable additions

- **What changed.** New module `TVA.Lemmas.SumFused`:
  - `Tree.eval_err_abs`: a tree whose nodes are each within `δ·(|a| + |b|) + η` of `a + b`
    (when `|a| + |b| ≤ M`) is within `Tree.errBound δ η depth nodes A` of the exact sum. This is
    the same bound as `Tree.eval_err`'s.
  - `addRel_node`: a node `AddRel fusion` is within `3u·(|a| + |b|) + 4η`
    (`FloatV.AddRel.err`).
  - `sumc_err`, `sumc_err_covers`, `sumc_err_list` cover `tl.sumc`/`sum1c`/`sum0c` and
    `tl.cumsumc` (through `ScanLeaves.perm`), for any `fusion`, so for `E.fusion`.
  - `sum_bcast_mul_tree_err` covers the tree form of `tl.sum_bcast_mul` on `float32`.
    `sum_bcast_mul_err` covers both forms; it takes `DotF32Ieee` for the `ieee` dot.
- **Why.** With fusion, `1 + (-1)` can give about `2^-24`, so no node is within `δ·|a + b| + η`.
  `Tree.eval_err`'s premise cannot hold, but a bound relative to `|a| + |b|` does, and it keeps
  `errBound`'s shape.
- **Evidence.** None needed: the results are proved, with standard axioms only.
- **Constants whose meaning changed.** None. The per-node constants are `3u` and `4η`, with room
  `A + errBound ≤ maxVal/2 − 2η`.

## L34: The shared-atomics kit is narrow

- **What changed.** New module `TVA.Lemmas.SharedAcc`:
  - `SafeA.atomic` (with `SafeA.rmwSeq_shared`): a tensor atomic on shared cells, optionally
    masked. Each enabled lane adds one logged step, for every order of the enabled lanes; the
    order is `acq_rel` at `gpu`/`sys` scope.
  - `MapsCell`, `replay_maps`: replaying the steps on a cell.
  - Integer addition of any width: `replay_add_any`, `mapsCell_other_add`, `sum_proj_any`.
  - Integer `max`/`min` of any width, in any order: `IntV.toInt_max`/`toInt_min`,
    `foldl_max_toInt`, `foldl_max_spec`, `foldl_min_spec`.
  - Float accumulation in any order: `StepRel`, `AddsApprox`, `replay_err` (the bound is
    `DotLoop.bound ρ α n (|x₀| + Σ|w|)`), and `stepRel_atomicAdd` from `AtomicAddAcc`.

  The device fact `AtomicAddF32` (`AtomicAddAcc` with `2^-23`, `2^-124`) is in
  `TVA.Rules.Num.Sm90`, with the section "Float `atomic_add`" in `rules/num.md`.
- **Why.** Before this round, the only rules were an unmasked scalar atomic and `int32`
  addition.
- **Evidence.** `evidence/num-atomic-add`: about 2.2·10^8 pairs, on pointer atomics
  (`acq_rel`, `relaxed`; `atom.global.….add.v4.f32`) and on the TMA reduction. Nothing is
  outside the fact, and the largest ratio is 0.500. The tight bound `2^-24·|o+v| + 2^-125`
  (round to nearest, subnormals flushed on the pointer path) is reached by construction, so the
  fact doubles both constants. `atomic_add(p, 0.0, relaxed)` is a plain load.
- **Constants whose meaning changed.** None. `AtomicAddF32`'s constants are new.
