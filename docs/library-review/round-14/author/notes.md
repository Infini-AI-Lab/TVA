# Library batch, round 14: author's notes

Issues: L34, L64, L66, L67 ([ISSUES.md](../../ISSUES.md)), 2026-10-28: the numeric follow-ups on
round 12's float layer (L65). Built in a separate tree after rounds 11, 12, 13, 15 and 16.

**Build.** `lake build TVA` is green. Every new or restated theorem of the round uses only
`propext`, `Classical.choice` and `Quot.sound` (`#print axioms` on each); `tvr audit` prints
`independent`. A scratch Lean file (not shipped) applied the new lemmas and facts as a kernel
proof would: a tensor atomic's lanes paired with `accStep_atomicAdd` and `accStep_other`, the
`PermInv` of a `Paired` postcondition, `replay_err_shared`, `replay_max` at `int32` and
`replay_min` at `uint8` (`IntV.max`/`min` with `IntV.toInt` unify at `d.Val`), lr-9b's two equal
steps with operands `1` and `1/2`, `CvtSat.cvtRn`, `FloatV.cvtSat_rn_err` from a `CvtSat…` fact,
and `Platform.sm90ref` satisfying the new facts with the old ones: it checks without errors.

**Kernel.** `kernels/vec_add_f32` does not build, as before this round (its errors are in
`launch_valid` and later, unrelated to this batch); I did not edit it or run `tvr check`.

**Release check** (`tvr release`, once, on this tree with these notes): 7 of 11 steps pass
(shipped, packages, sandbox, build, library, audit, and the lint of `vec_add_f32`'s reviewer
record); `tvr lint library 14 author` prints `regular`. The failures are the kernel's and those of
the rounds still under review: `gate vec_add_f32` (it does not build, and its shipped gate is
stale), `pages` (regenerated from that gate), `records` (round 12's reviewer folders, in progress,
are not recorded) and `lint library 16 lr-16a` (placeholders, in progress).

**Files outside my list.** `docs/LIBRARY.md` (the rows of the changed modules),
`docs/PROVER_GUIDE.md` (the shared-atomics paragraph, and the `tl.cvt8` facts in the table of
rules), `lean/TVA/Lemmas/Consistency.lean` (`Platform.ref`'s conversions, below) and
`lean/TVA/Lemmas/Sm90.lean` (the new facts of `Platform.sm90ref`).

## L34: the shared-atomics kit had no accumulation per operand (float sums, `max`/`min`)

- **What changed.** `lean/TVA/Lemmas/SharedAcc.lean`, after `SafeA.atomic` and the integer
  addition kit (both unchanged):
  - **Steps paired with their operands, per position.** `Paired R log ops`: the steps of `log`,
    in order, each with an optional operand (`none` for a step that leaves the cell alone), every
    pair satisfying `R`, and `ops` the multiset of the operands present. `Paired.nil`,
    `Paired.append`, `Paired.lanes` (a tensor atomic's log extension `order.map step` with each
    lane's operand), `Paired.perm` (a reordering of the log keeps it, so a postcondition built
    from it is `PermInv`; from `exists_perm_of_map_perm`), `Paired.mono`, and **`Paired.merge`**:
    if each program's part `proj G i` of the interleaved log that `shared` returns is paired with
    `ops i`, the whole log `G.map Prod.snd` is paired with `∑ i, ops i`. `replay_fold`: the cell
    after a replay is the left fold over the operands, in the log's order.
  - **Float accumulation.** `AccStep d a val good ρ α B s o` (`some w`: the step adds `w` up to
    `ρ·|x + w| + α`, as `AddsApprox`; `none`: `MapsCell d a id`), `accStep_other`,
    `accStep_atomicAdd` (from `Rules.Num.AtomicAddAcc`). `replay_err` is restated per position:
    over a list of step–operand pairs whose first components the replay runs, the cell ends
    within `DotLoop.bound ρ α n (|x₀| + Σ|w|)` of `x₀ + Σ w`, the sums over the operands and `n`
    their number (the steps on the cell only; the old statement counted every step of the log).
    **`replay_err_shared`**: with `∀ i, Paired (AccStep …) (proj G i) (ops i)` and the replay of
    `G`, the cell is within `DotLoop.bound ρ α (∑ card) (|x₀| + ∑ Σ|w|)` of `x₀ + ∑ᵢ Σ (ops i)`:
    the sum of every program's operands, in any interleaving.
  - **Integer `max`/`min`.** `foldl_max_spec` and `foldl_min_spec` are restated over a list of
    operands with an order key (`key (op x v) = max (key x) (key v)`: `IntV.max` with
    `IntV.toInt_max`); **`replay_max`**, **`replay_min`**: with every program's log paired with its
    operands (`MapsCell d a (o.elim y (op y))`), the cell ends at a value whose key is at least
    (at most) the initial value's and every operand's, and equals one of them. `IntV.lt_toInt`,
    `toInt_max`, `toInt_min` are unchanged; `foldl_max_toInt` is removed (the key replaces it).
  - Docs: `docs/LIBRARY.md` (`SharedAcc` row), `docs/rules/num.md` ("Float `atomic_add`"),
    `docs/PROVER_GUIDE.md` (shared atomics).
- **Why.** lr-9b: a logged step records the cell and its old and new values, not the operand, so
  two programs adding `1.0` and `0.5` to a `float32` cell holding `2^24` log the same step, and no
  weight `w : AtomicStep → ℚ` gives each its operand. The operand now travels with each position
  of the log, and `Paired.merge` turns the programs' own operand lists into the whole log's
  (lr-9b's first fix; recording the operand in the step would change `Prog.rmw`, whose update is
  a relation with no operand). For integer addition a step determines its increment
  (`new − old`), so `replay_add_any` and `sum_proj_any` stay as they are, which lr-9b found right.
- **Evidence.** None needed (lemmas); the device behaviour they use, one read-modify-write per
  enabled lane, is round 9's (lr-9a, 116 configurations).
- **Constants whose meaning changed.** `TVA.replay_err`, `TVA.foldl_max_spec`,
  `TVA.foldl_min_spec` (new statements); removed: `TVA.foldl_max_toInt`; new:
  `TVA.exists_perm_of_map_perm`, `TVA.Paired` (`nil`, `append`, `lanes`, `perm`, `mono`, `merge`),
  `TVA.replay_fold`, `TVA.replay_max`, `TVA.replay_min`, `TVA.AccStep`, `TVA.accStep_other`,
  `TVA.accStep_atomicAdd`, `TVA.replay_err_shared`. The other constants of
  `TVA.Lemmas.SharedAcc` keep their statements (the file changed, so every item that reaches it
  has a new digest).

## L64: fp8 conversion of values beyond `maxVal` (saturation) was not stated

- **What changed.**
  - `lean/TVA/Rules/Num.lean`: the shape **`CvtSat P f g`**: every result of `P.cvt f g .rn x`
    is `FloatV.cvtSat .rn x` up to NaN payload, on every input (in range the rounding, beyond the
    range and for `±∞` the value `±maxVal`, NaN a NaN); **`CvtSat.cvtRn`** (`g.Regular`): it
    implies `CvtRn P f g`. `CvtRn`'s docstring points to it.
  - `lean/TVA/Rules/Num/Sm90.lean`: six facts, **`CvtSatF32E4m3`**, **`CvtSatF32E5m2`**,
    **`CvtSatF16E4m3`**, **`CvtSatF16E5m2`**, **`CvtSatBF16E4m3`**, **`CvtSatBF16E5m2`**, each
    `CvtSat P f g` for its pair, with the instructions and the evidence. The `CvtRn` facts keep
    their names and statements.
  - `docs/rules/num.md`: the six facts in "Facts the library states" (one row, each name a code
    span in the first column; `stated_facts` reads 38 facts), in the kernel reviewer's table for
    `tl.cvt8`, in the definitions table (`CvtSat`), in "Conversions into fp8" (what a kernel
    converting an accumulator cites) and in "Satisfiability"; "the fp8 conversions … and their
    saturation beyond `maxVal`" is gone from "What the library does not state".
  - Lemmas for kernels: `Format.roundMag_sat_of_gt` (`lean/TVA/Math/Rounding.lean`: a magnitude
    above `maxVal`, rounded with saturation in either mode, is `maxMag`); in
    `lean/TVA/Lemmas/FloatErr.lean`, `FloatV.cvtSat_of_gt` (finite `|x| > maxVal`: finite, value
    `±maxVal` of `x`'s sign), `cvtSat_of_isInf` (`±∞`: `±maxVal`), **`cvtSat_rn_err`** (finite
    `x`: finite and within `|c|·u + η` of the clamped `c = max (−maxVal) (min x maxVal)`), and
    `isNaN_of_isInf`.
  - Satisfiability: `Platform.ref`'s conversion to nearest is now saturating
    (`NaNEq (if rnd = .rn then cvtSat .rn x else cvtWith rnd x) r`; toward zero unchanged, as the
    device keeps `±∞` with `"rtz"` into `float8e5`); `Platform.ref.cvtAcc` is proved again on it
    (same statement), `Platform.ref.cvtSat` is new. `Platform.sm90ref.cvtSat` and the six
    `Platform.sm90ref.cvtSat…` are new; `Platform.sm90ref.cvtRn` now takes `g.Regular` and
    derives `CvtRn` from `CvtSat` (the six `cvtF32E4m3` … keep their statements).
  - Evidence README: `evidence/num-cvt8` conclusion 2 now names the facts.
- **Why.** lr-9a: in-scope tutorials (08, 09, 06, 10: 8 rows of the coverage matrix) convert
  unbounded `float32` accumulators into fp8, and `CvtRn` says nothing beyond `maxVal`. The
  statement is on every input up to NaN payload, which the evidence covers (lr-9a proposed
  non-NaN inputs; NaN gives a NaN on every input, so the stronger statement holds). A constant
  converted into fp8 is folded only when exact (`TVA.Lang.Scalar`), so it is covered too.
- **Evidence.** `evidence/float-layer` part A (round 12): the library's own `Format.cvt … .rn true`
  against the device on every `float16`, `bfloat16` and `float32` input, all six pairs, bit for bit
  up to NaN payload (`A.cvt.*.e4m3.sat1`, `A.cvt.*.e5m2.sat1`); I re-ran those six tests outside
  the tree with the same driver (2026-10-28, GPU 0): each agrees again on every input.
  `evidence/num-cvt8` (library issue L4): every input, results above `maxVal` and `±∞` give
  `±maxVal`, NaN gives `0x7f`.
- **Constants whose meaning changed.** New: `TVA.Rules.Num.CvtSat`, `TVA.Rules.Num.CvtSat.cvtRn`,
  `TVA.Rules.Num.CvtSatF32E4m3`, `CvtSatF32E5m2`, `CvtSatF16E4m3`, `CvtSatF16E5m2`,
  `CvtSatBF16E4m3`, `CvtSatBF16E5m2`, `TVA.Math.Float.Format.roundMag_sat_of_gt`,
  `TVA.FloatV.cvtSat_of_gt`, `cvtSat_of_isInf`, `cvtSat_rn_err`, `isNaN_of_isInf`,
  `TVA.Platform.ref.cvtSat`, `TVA.Platform.sm90ref.cvtSat`, `cvtSatF32E4m3` … `cvtSatBF16E5m2`.
  Changed: `TVA.Platform.ref` (its `cvt` field, for `.rn`) and so `TVA.Platform.sm90ref`;
  `TVA.Platform.sm90ref.cvtRn` (its arguments); `TVA.Rules.Num.CvtRn` (docstring only). Through
  the files hashed whole, every item whose cone reaches `TVA.Rules.Num`, `TVA.Rules.Num.Sm90`,
  `TVA.Math.Rounding`, `TVA.Lemmas.FloatErr`, `Consistency` or `Sm90`, and through
  `docs/rules/num.md` every item that cites a fact, has a new digest.

## L66: wording of the evidence accounts and the fact page

- **What changed.**
  - **`DotF32Ieee`'s ratio `0.499`, re-derived.** New `evidence/num-dot/ieee_ratio.py`
    (`ieee_ratio.out`): run.py's kernel and input kinds on `16 × 16`, one warp, `K = 8 … 1024`,
    the largest ratio per kind, and the worst element recomputed exactly with fractions. The
    largest ratio is always on `allones` and equals `(K−1)/(2(K+1))` to four digits; `tiny` is at
    `0.0000`. The worst element has an accumulator with all significand bits set
    (`2^25 − 2` at `K = 8`) and `K` products just below its ulp: the first `fma` rounds up to a
    power of two, where the ulp doubles, and the `K − 1` later products are lost, an error just
    below `(K−1)·2^-24·mag` (`6.999996 … 1022.94` units) against the relative term
    `(K+1)·2^-23·mag`. So lr-9a's account is right, and lr-9b's `K·2^-150` against
    `(K+1)·2^-149` (the absolute term) is not what reaches `0.499`. One account now in
    `evidence/num-dot/README.md` (the table's `(tiny inputs)` replaced, conclusion 4 rewritten, a
    section "The ratio of `DotF32Ieee`") and in `docs/rules/num.md` ("`tl.dot`"). The README also
    says that `out.txt`'s Part 2 column "err/u (worst input kind)" names the kind with the
    largest relative error, not the largest ratio. Round 9's notes keep their "by construction";
    they are a recorded record, and this round's account replaces it.
  - **"the classical bound for `K` roundings"**: `docs/rules/num.md` and evidence/num-dot
    conclusion 4 now give `γ_K = K·2^-24/(1 − K·2^-24) ≈ K·2^-24` and say that `(K+1)·2^-23` is
    about twice it at every `K` that compiles.
  - **exp2 "reached exactly"**: evidence/num-math-f32 conclusion 3 (and its table cell) and
    `docs/rules/num.md` (after the facts table) say the ratio approaches `1` from below
    (`2^x/(2^-22·2^x + 2^-126) < 1/(1 + 2^-22)` for a flushed result) and is not reached.
  - **"What the library does not state"**: approximate functions on `float16`/`bfloat16` are
    now described as rejected by Triton on the host (`_check_dtype` with `fp32`/`fp64`,
    `primitives/triton-3.8.0-math.py:8-30, 95-232`), leaving `float64` as not stated; the
    `tl.dot_scaled` entry names `"e2m1"` by `"e2m1"` (which compiles and runs on sm_90) and
    `"e2m1"` by `"e4m3"`.
  - **"within `j·2^-23` of their magnitudes' sum"** → "within `j·2^-23` times their magnitudes'
    sum" (and `33·2^-13` likewise): the `DotF16` and `DotE4m3` docstrings, `docs/rules/num.md`,
    evidence/num-dot conclusions 2 and 3.
  - **"plain load"** → a strong load at the atomic's scope and order with no store
    (`ld.global.gpu.relaxed.b32`, Triton's `LoadAcquireOp`,
    `triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:648-687, 762-806`): `docs/rules/num.md` ("Float
    `atomic_add`" and "Device questions") and `evidence/num-atomic-add/README.md`.
  - **Two names for `float32` `−∞`**: round 12 removed `Softmax.negInf32`; the remaining pair,
    `Rules.Num.negInfF32` (the facts) and the generic `FloatV.negInf` (`TVA.Lemmas.FloatMax`,
    which the softmax lemmas use), is now equated by `Rules.Num.negInfF32_eq_negInf`
    (`lean/TVA/Rules/Num/Sm90.lean`, which imports `TVA.Lemmas.FloatMax` for it; `decide`).
- **Why.** lr-9a and lr-9b found these accounts wrong or loose; none changes a fact.
- **Evidence.** `evidence/num-dot/ieee_ratio.py` with `ieee_ratio.out` (Triton 3.8.0, H200 NVL,
  GPU 0, 2026-10-28; a second run gave the same output byte for byte); the PTX in
  `evidence/num-atomic-add/out.txt`.
- **Constants whose meaning changed.** New: `TVA.Rules.Num.negInfF32_eq_negInf`. Docstrings
  only: `TVA.Rules.Num.DotF16`, `TVA.Rules.Num.DotE4m3`. The statements of the facts are
  unchanged; `docs/rules/num.md` and `lean/TVA/Rules/Num/Sm90.lean` changed, so every item citing
  a fact has a new digest.

## L67: `DivRn P .binary64` had no device evidence of TVA's own

- **What changed.** New `evidence/num-div-f64` (README, `run.py`, `out.txt`). The fact's row in
  `docs/rules/num.md` cites it, "Division" has a `float64` bullet, and the `DivRn` docstring in
  `lean/TVA/Rules/Num.lean` cites it. The statement of `DivRn` is unchanged.
- **Why.** lr-9b: the table gave "the lowering" as the evidence; a device fact needs TVA's own
  run.
- **Evidence.** `evidence/num-div-f64`: `x / y` on `float64` (and `tl.fdiv(x, y)`, and `x / c`
  for 11 constant divisors) on the device against the library's own `FloatV.divRn` (round 12's
  native driver of `TVA.Math.Defs`, spec `div:f64`), compared per chunk of `2^20` pairs by a
  64-bit hash with an odd multiplier per index, NaN up to payload. 452,986,128 pairs in 24
  families: uniform patterns and evidence/float-layer's structured records, divisors within `2^20`
  ulps of `±1` and in `[1/2, 2)`, subnormal dividends, quotients and divisors, powers of two,
  quotients near overflow, next to the exact value (`x = RN(q·y)`) and next to midpoints
  (`x = RN(mid·y)` in exact arithmetic), every pair of 36 special values, and the constants
  `1`, `−1`, `2`, `0.5`, `3`, `0.1`, `7`, `2^-1074`, `2^1023`, `1e308`, `1/3`. Every pair agrees.
  The PTX divides with `div.rn.f64` everywhere except `x / 1.0`, which the compiler folds to `x`.
  Triton 3.8.0, H200 NVL, GPU 0, 2026-10-28; I ran it twice (the second run after removing the
  timing column from `out.txt`): the same verdicts.
- **Constants whose meaning changed.** None (`TVA.Rules.Num.DivRn`'s docstring only; the file
  and `docs/rules/num.md` changed, so the items citing `DivRn P .binary64` have new digests).
