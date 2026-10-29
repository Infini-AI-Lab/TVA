# Library review, round 14, by lr-14b

## Verdicts

- **L34** ACCEPT — The statements say what lr-9b and lr-12b asked for. `Paired` pairs each
  step of a log with an optional operand, and `Paired.merge` turns the programs' own operand lists
  into the whole interleaved log's (`∑ i, ops i`). `replay_err` counts only the steps paired with
  an operand, so a step on another cell (`accStep_other`) adds nothing to `n` or to the absolute
  term. `replay_err_shared`, `replay_max` and `replay_min` bound or give the cell against every
  program's operands, in any order. Every result uses the standard axioms only. I applied them as
  a kernel proof would (work/Usability.lean, which checks without errors):
  - U1 proves one program's masked `float32` `tl.atomic_add` with `SafeA.atomic`. Its log is
    paired by `Paired.lanes`, with `accStep_atomicAdd` (from `AtomicAddF32`; `tl.addUpd`'s
    `P.atomicAdd` unifies) and `accStep_other`, and the operand multiset does not depend on the
    lane order.
  - U2 gets `PermInv` from `Paired.perm`.
  - U3 feeds `shared`'s `∃ m, Q i (proj G i) m` into `replay_err_shared`.
  - U4 is lr-9b's case: the operands `1` and `1/2` added to `2^24` leave the cell within about
    `4` of `2^24 + 3/2`, and the device's `2^24` is inside that.
  - U5 runs `replay_max` at `int32` from `tl.maxUpd`, and `replay_min` at `uint8`.
  - U9 composes `shared` with `meets_of_faithful` into `Claim.Meets`.

  One usability note: passing the room hypothesis to `replay_err_shared` as a bare term makes the
  elaborator run out of heartbeats on `Format.binary32.maxVal`, and `(by exact h)` avoids it. The
  kit does not cover float `atomic_max`/`atomic_min` (`fBound` is a relation). No in-scope
  tutorial needs them: none uses an atomic outside tutorial 05's excluded function.
- **L64** REOPEN — The six `CvtSat…` facts hold for conversions at run time:
  - I re-ran evidence/float-layer's six `sat1` tests in full (every `float16`, `bfloat16` and
    `float32` input), and all agree. From every source format, NaN gives `0x7f` and `±inf` gives
    `±maxVal` (work/cvt8_nan.out).
  - `Platform.ref`'s saturating `.rn` keeps the facts satisfiable together. U8 shows that
    `sm90ref` satisfies the six `CvtSat` facts, the `CvtRn` facts through `CvtSat.cvtRn`,
    `CvtTotal` at `.rz` and the other facts.
  - `cvtSat_rn_err`, `cvtSat_of_gt` and `cvtSat_of_isInf` apply as a kernel would use them (U6:
    a finite accumulator, `1000`, `+inf`, NaN).

  But the batch's sentence about constants is false, and with the new facts it lets a wrong claim
  through. `docs/rules/num.md` ("Conversions into fp8") says a constant converted into fp8 "is
  folded by the compiler only when the conversion is exact …, so the facts cover it too".
  SEMANTICS ("Constants") builds a Python number in an fp8 dtype as `FloatV.litF32`, then
  `tl.cvt8`. For `±inf` into `float8e5` the fold is exact, and it keeps `±inf`.
  `tl.full([B], ±inf, tl.float8e5)`, and `tl.where(m, ±inf, x)` with a `float8e5` `x`, store
  `0x7c`/`0xfc` (`±inf`), with no conversion in the PTX (work/cvt8_inf.out, cvt8_const.out). The
  model with `CvtSatF32E5m2` gives `±57344` (U10 proves it). A conversion of the same `float32`
  value at run time gives `0x7b`/`0xfb`. In those two places every inexact constant is rejected on
  the host: `0.1`, `449`, `1000`, and `±inf` or NaN into `float8e4nv`. `other=`, a stored value
  and `.to` convert at run time and agree with the facts.

  **Fix.** In SEMANTICS ("Constants", the fp8 bullet) and in the docstrings of `TVA.Lang.Scalar`
  and `Operand.constIn?`, state what Triton does: it converts the `float32` constant exactly at
  compile time, not by `tl.cvt8` (`FloatV.cvt`, which keeps `±inf` in `float8e5`; NaN gives a
  NaN, a zero gives `+0`), and it rejects the kernel on the host when that conversion is inexact.
  Correct num.md's sentence: the `CvtSat…` facts are about conversions at run time. Back both
  with TVA's evidence (these probes can serve).
- **L66** ACCEPT — Every item is fixed as asked, and no fact changed:
  - I re-ran evidence/num-dot/ieee_ratio.py (work/ieee_rerun), and its output is identical to
    `ieee_ratio.out` byte for byte. Its account checks by hand at `K = 8`: `acc = 2^25 − 2`,
    products `2 − 2^-22`; the first `fma` rounds up to `2^25`, and the next seven products are
    lost. The error is about `14 = 7·2^-24·mag` against a bound of `36`, a ratio of
    `7/18 = (K−1)/(2(K+1))`.
  - num.md and the README state `γ_K` and "about twice".
  - num.md and evidence/num-math-f32 (conclusion 3 and its table cell) say exp2's ratio
    "approaches `1` from below … and is not reached".
  - The not-stated list matches `primitives/triton-3.8.0-math.py`: `_check_dtype` takes
    `fp32`/`fp64` on every approximate function, and `fp32` only on `sqrt_rn` and `div_rn`.
  - "times their magnitudes' sum" is in the `DotF16` and `DotE4m3` docstrings, num.md and the
    README.
  - The strong load cites `isPromotableToNVPTXLD` (`LoadStoreOpToLLVM.cpp:648-687`) and the PTX's
    `ld.global.gpu.relaxed.b32` in evidence/num-atomic-add/out.txt.
  - `negInfF32_eq_negInf` (`decide`, no axioms) rewrites `ExpF32NegInf` into a statement at
    `FloatV.negInf` (U7). The library has no other name for `float32` `−∞`.
- **L67** ACCEPT — evidence/num-div-f64 is TVA's own (Triton 3.8.0, H200 NVL, GPU 0,
  2026-10-28), and it compares with the library's own definition: `Format.div .rn` compiled from
  `lean/TVA/Math/Defs.lean`, which is `FloatV.divRn`. It is adversarial where that matters:
  subnormal dividends, quotients and divisors, divisors near `±1`, overflow, quotients next to
  midpoints, special values, constant divisors and `tl.fdiv`. The hash's odd multipliers make any
  single difference visible.
  - I re-ran it sampled (`--quick`: the first `2^20`-record chunk of every family, with the
    shipped seeds). All 24,118,544 pairs agree, and the PTX is `div.rn.f64` everywhere except the
    folded `x / 1.0`.
  - I ran a test of my own (work/div64_own.py): 671,088,640 pairs in ten families, through two
    kernels (`x / y` with 2 warps, `(-x) / (-y)` with 8), against the host's IEEE division. None
    differs.

  The table row, "Division" and the `DivRn` docstring cite the evidence.

## New issues

- **NEW** wording — Two texts promise more than the code:
  - `docs/LIBRARY.md`'s `Encoding` row says the module has "the constants of each format". For
    `float8e5` and `float8e4nv` it has only `maxVal`: there is no `e4m3_u`, `e4m3_eta`, `e5m2_u`
    or `e5m2_eta`, which a kernel needs to evaluate `FloatV.cvtSat_rn_err`'s `|c|·u + η`. (U6
    derives them from `Format.u_eq` and `eta_eq` by `rfl`.)
  - `replay_err`'s docstring (`lean/TVA/Lemmas/SharedAcc.lean`) names "the operands `W`". The
    statement has no such name: the operands are `L.filterMap Prod.snd`.

  Fix: add the four constants (or say that fp8 has only `maxVal`), and name the operands as the
  statement does.

## Re-ran

- evidence/num-dot/ieee_ratio.py, in full, from a copy in work/ieee_rerun (the script writes its
  output beside itself): the output is identical to `ieee_ratio.out`, byte for byte.
- evidence/num-div-f64, sampled: `run.py --quick`, with the float-layer driver built from the
  current `Defs.lean`. That is 24,118,544 pairs, the first chunk of each of the 24 families with
  the shipped seeds. Every family agrees, and the PTX is as recorded.
- evidence/float-layer, in full: the six tests the `CvtSat…` facts cite (work/results.out). The
  tests are `A.cvt.{f16,bf16}.{e4m3,e5m2}.sat1` (every input) and `A.cvt.f32.{e4m3,e5m2}.sat1`
  (all `2^32` inputs), and all agree.
- Added:
  - work/div64_own.py (with its `.out`);
  - work/cvt8_const.py, cvt8_inf.py and cvt8_nan.py (with their `.out`): fp8 constants and
    special values;
  - work/Usability.lean (with its `.out`): U1–U10 and `#print axioms`, standard axioms only.

## Not checked

- evidence/num-div-f64 in full (452,986,128 pairs); sampled as above.
- evidence/num-cvt8, num-math-f32 and num-atomic-add, not re-run: L66 changed only their texts,
  which I read against their outputs.
- The release check: I read the coordinator's run on the merged tree (11 of 11 steps pass) and did
  not repeat it.
- The survey of `lean/TVA/Lemmas`: I read every module's statements and docstrings against
  `docs/LIBRARY.md` and the `tl.*` definitions they are about, but not the proofs (Lean checks
  them). I did not file L99's and L100's items (round 19) again.
- fp8 constants in places other than those in work/cvt8_inf.py: `tl.full`, a value of `tl.where`,
  an operator, `other=`, a stored value, `.to`.
