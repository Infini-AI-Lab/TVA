# Library review, round 14, by lr-14a

## Verdicts

- **L34** ACCEPT — lr-9b's and lr-12b's points are both met. A step now travels with its own
  operand (`Paired`), and `Paired.merge` builds the whole log's operands as `∑ i, ops i` along
  the interleaving that `shared` returns. `replay_err` counts only the positions that carry an
  operand (`n = (L.filterMap Prod.snd).length`); a step on another cell is `none`
  (`MapsCell d a id`, `accStep_other`), so split-K's `(1+2^-23)^(M·N·S)` is gone.
  `replay_err_shared`, `replay_max` and `replay_min` bound or give the cell against `x₀` and
  every program's operands, in any order. `accStep_atomicAdd` pairs a logged step with the lane's
  operand exactly as `tl.atomic_add`'s model logs it (`addUpd` on `float32` is
  `P.atomicAdd .binary32 o v n`, `lean/TVA/Lang/Ops.lean:246-260`). `Paired` with a fixed
  multiset is order-free (`Paired.perm`), so a postcondition built from it is `PermInv`. On the
  device side the kit cites `AtomicAddF32` "for any `sem`". I ran the pointer `atomic_add` on
  `float32` at all 4 `sem` × 3 `scope` values, `2^22` lanes each (random, subnormal and
  cancelling operands). The 12 runs give bit-identical cells (`atom.global.{gpu,cta,sys}.{acq_rel,
  acquire,release,relaxed}.add.v4.f32`; `work/atomic_sem.out`). So the fact holds for every `sem`
  and `scope`, as it must, since the model's `P.atomicAdd` takes neither; its evidence ran only
  two `sem` values at `gpu`.
- **L64** REOPEN — The six `CvtSat…` facts are true of `x.to(fp8)`. I re-ran float-layer
  part A's six `sat1` tests on every `float16`, `bfloat16` and `float32` input: all agree, and the
  `sat0` control still differs. Every NaN pattern of `float16` and `bfloat16`, and 139,262
  `float32` NaN patterns of both signs, give `0x7f`, and `±∞` gives `±maxVal`
  (`work/fp8_nan.out`). The new `Platform.ref` (`.rn` saturating, `.rz` unchanged) matches the
  device, and so does `sm90ref.cvtRn` with `g.Regular`. **What is wrong (soundness):** the
  batch also says that the facts cover fp8 constants. `docs/rules/num.md`, "Conversions into
  fp8", says: "A constant converted into fp8 is folded by the compiler only when the conversion
  is exact (`TVA.Lang.Scalar`…), so the facts cover it too". The author's notes say the same,
  and SEMANTICS "Constants" models a Python number built in an fp8 dtype as "`FloatV.litF32`,
  then `tl.cvt8`" (also `Operand.constIn?` in `TVA.Lang.Promote` and `TVA.Lang.Scalar`'s
  docstring). That covers `tl.full(shape, c, fp8)` and a value of `tl.where` (an operator on an
  fp8 tensor is rejected on the host). With `CvtSatF32E5m2`, that model now gives `±57344` (`0x7b`/`0xfb`) for
  `c = ±inf` and for any `c` whose `float32` is `±∞` (`1e39`). The device gives `±∞`
  (`0x7c`/`0xfc`) for `tl.full([16], float('-inf'), tl.float8e5)` and for
  `tl.where(m, x8, float('-inf'))` (`work/fp8_const.out`, `work/fp8_store.out`;
  `work/Fp8Const.lean` evaluates both library values). The cause: `scalar_constant` builds an
  fp8 constant as `arith.truncf` of the `float32` constant (`triton-3.8.0-semantic.py:601-613`).
  MLIR folds a `truncf` whenever APFloat converts it without loss
  (`llvm-ArithOps.cpp:1591-1611, 1788-1825`), and `±∞` into the IEEE-like `f8E5M2` is
  lossless, so the device's `.satfinite` never runs. Before this round `CvtRn` left `P.cvt` at
  `±∞` open, so the model was merely vacuous there. With `CvtSat` it is false: a claim "the
  stored byte is `0xfb`" is provable and wrong. Two other cases go through the device's
  conversion, as the model says: a stored Python number and `other=` (both a `float32` tensor,
  then `tt.fp_to_fp`, which folds only zeros, `triton-3.8.0-Ops.cpp:996-1026`) saturate to
  `0xfb`. An inexact fp8 constant (`0.1`, `460.0`, `1000.0`, `1e6`), and `±∞` or NaN into
  `float8e4nv`, does not compile (`llvm.fptrunc` to `i8`). So a constant the compiler
  cannot fold is rejected on the host, never converted on the device. **Fix:** in SEMANTICS "Constants" (and `constIn?`,
  `TVA.Lang.Scalar`), state that a Python number built directly in an fp8 dtype is the exact
  conversion `FloatV.cvt (FloatV.litF32 c)`. That conversion keeps `±∞` in `float8e5`. It
  compiles only when it is exact, and it is not `tl.cvt8`. Then remove "so the facts cover it
  too" from `docs/rules/num.md`, or restrict it to the paths that convert on the device. The
  six facts can stay as they are.
- **L66** ACCEPT — Each item is fixed and right.
  - **`ieee_ratio.py`:** re-run outside the tree, its output is identical byte for byte. The
    account is right: `acc = 2^25 − 2` with products just below its ulp, the first `fma` rounds
    to `2^25`, and the `K − 1` later products are lost. That gives error ≈ `(K−1)·2^-24·mag`
    and ratio `(K−1)/(2(K+1))`.
  - **`γ_K`:** `(K+1)·2^-23/γ_K = 2(K+1)(1−K·2^-24)/K`, which is `2.25` at `K = 8` and falls
    toward `2`.
  - **exp2's ratio:** `y/(2^-22·y + 2^-126)` increases in `y`, so it stays below
    `1/(1+2^-22)` for `y < 2^-126`. "Not reached" is right.
  - **`_check_dtype`:** it rejects `float16` and `bfloat16` on the host (`math.py:8-30`).
    `work/math_f16.out`: `tl.exp`, `exp2`, `log`, `sqrt`, `rsqrt`, `sin` and `erf` all raise
    `CompilationError` on both formats.
  - **"times their magnitudes' sum":** fixed in both docstrings, in `num.md` and in the README.
  - **The strong load:** it matches `isPromotableToNVPTXLD` (`acquire`/`relaxed`, a scalar, a
    constant zero, `−0.0` included) and the `LoadAcquireOp` at the op's scope
    (`LoadStoreOpToLLVM.cpp:648-687, 762-806`).
  - **`negInfF32_eq_negInf`:** proved by `decide`.
- **L67** ACCEPT — `DivRn P .binary64` now stands on TVA's own evidence, and it is true.
  - **Lowering:** `arith.divf` on `f64` becomes `llvm.nvvm.div.rn.d` (`FDivOpConversion`,
    `triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:553-580`), and the PTX shows `div.rn.f64`.
    MLIR's folds (`x / 1.0`, two constants, `(-x)/(-y)`) are exact or IEEE.
  - **Re-run:** I re-ran `evidence/num-div-f64` in full outside the tree: all 24 families,
    452,986,128 pairs. Its `out.txt` is identical byte for byte.
  - **Independent reference:** the evidence's reference is TVA's own `FloatV.divRn`, so I also
    checked against one that is not. 218,103,808 pairs from five families of my own (uniform
    patterns, operands spread over every exponent with subnormal and overflowing quotients,
    divisors within `2^22` ulps of `±1`, quotients next to midpoints, short significands with
    ties in the subnormal range) went through both `/` and `tl.fdiv`. Against the host CPU's
    IEEE `float64` division, with subnormals kept and checked against exact fractions, there were
    0 differences (`work/div_f64.out`).
  - **Citations:** the families cover what lr-9b asked for. The fact's row in `num.md`, the
    "Division" bullet and the `DivRn` docstring all cite the folder.

## New issues

- **NEW** wording — `evidence/num-cvt8/README.md`, conclusion 2 ("infinities saturate … NaN
  gives the NaN `0x7f`") and "This folder shows the results beyond the range on every input".
  `run.py` converts `±∞` and NaN only from `float32`, and only the canonical NaN; for `float16`
  and `bfloat16` infinities and every other NaN pattern the statement rests on float-layer. The
  statement is true: `work/fp8_nan.out` runs every `float16`/`bfloat16` NaN of both signs,
  139,262 `float32` NaN patterns, and `±∞` of each format. Fix: say which inputs this folder
  ran, or add those inputs to `run.py`.
- **NEW** wording — `evidence/num-dot/README.md` ("The ratio of `DotF32Ieee`") says the largest
  ratio "equals `(K−1)/(2(K+1))` to four digits". At `K = 512` the measured `0.498035` prints as
  `0.4980` against `0.4981` (`ieee_ratio.out`). Fix: "within `2·10^-5` of, and below,
  `(K−1)/(2(K+1))`".
- **NEW** wording — `AtomicAddF32` (`lean/TVA/Rules/Num/Sm90.lean`, its row in
  `docs/rules/num.md`) covers every `sem` and `scope`, since `P.atomicAdd` has neither, but
  `evidence/num-atomic-add` runs only `acq_rel` and `relaxed` at scope `gpu`. Fix: add the other
  combinations to the evidence, as in `work/atomic_sem.out` (all 12 bit-identical), and say in
  its README that the fact covers every `sem` and `scope`.

## Re-ran

All runs used Triton 3.8.0 on the H200 NVL, GPU 0, on 2026-10-28. The evidence scripts ran from
unchanged copies outside the tree.
- **evidence/float-layer part A, in full.** I rebuilt the native driver from
  `lean/TVA/Math/Defs.lean` and `Driver.lean` as `build.sh` does. I ran `A.cvt.f16.e4m3.sat1`,
  `A.cvt.f16.e5m2.sat1`, `A.cvt.bf16.e4m3.sat1`, `A.cvt.bf16.e5m2.sat1`, `A.cvt.f32.e4m3.sat1`
  and `A.cvt.f32.e5m2.sat1` on every input (`2^16`, `2^16`, `2^16`, `2^16`, `2^32`, `2^32`), and
  all agree. The control `A.cvt.f16.e4m3.sat0` differs on 14,720 inputs, as recorded
  (`work/float_layer_rerun.out`).
- **evidence/num-dot/ieee_ratio.py, in full.** Its output is identical to `ieee_ratio.out`.
- **evidence/num-div-f64, in full:** all 452,986,128 pairs, with the same driver. Its `out.txt`
  is identical byte for byte (`work/num_div_f64_rerun.out`).
- **Tests of my own:**
  - `work/fp8_const.py`: fp8 constants built from Python numbers through `tl.full`, `tl.where`,
    `other=`, a `float32` constant `.to` and a loaded `float32` `.to`, for 17 values × 2 formats.
  - `work/fp8_store.py`: stored Python numbers and tensors through a `float8e5` pointer.
  - `work/Fp8Const.lean`: the library's `cvtSat` against `cvt` of `±∞`, checked with
    `lake env lean`.
  - `work/fp8_nan.py`: NaN and `±∞` patterns into both fp8 formats.
  - `work/div_f64.py`: 218,103,808 `float64` pairs through `/` and `tl.fdiv` against the CPU's
    IEEE division. This reference is independent of the Lean definition.
  - `work/math_f16.py`: approximate functions on `float16` and `bfloat16`.
  - `work/atomic_sem.py`: float `atomic_add` at every `sem` and `scope`.
- **The release check:** I read the coordinator's output on the merged tree (`release check
  PASS`, 11 of 11 steps) and did not re-run it, as the coordinator asked.

## Not checked

- I did not re-run the other evidence folders whose text L66 changed (num-math-f32, num-atomic-add
  `run.py`, num-dot `run.py`). I checked their new accounts by reading the sources and by the
  arithmetic above.
- I did not re-run float-layer's other 144 tests.
- For L34 I read the statements and proofs' hypotheses. I did not write a kernel-style Lean
  application of `Paired`, `replay_err_shared` or `replay_max`; the author's scratch file is not
  shipped.
- Outside the batch, I surveyed `DivFull`, `ExpF32`…`RsqrtF32`, the `Dot…` and `DotScaled…`
  facts only by reading. I ran nothing new on them.
