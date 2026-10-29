# Library batch, round 12: author's notes

Issue: L65 ([ISSUES.md](../../ISSUES.md)), 2026-10-28. TVA's float layer, written for TVA from
IEEE 754-2019, the OCP 8-bit and 4-bit formats and Mathlib by an author who has not seen the
project the deleted layer came from, nor the deleted files. Built in a separate tree beside rounds
11 (semantics) and 13 (tools).

**Build.** `lake build TVA` is green. Every constant of `TVA.Math.*`, `TVA.Lang.Scalar`,
`TVA.Rules.Num`, `TVA.Rules.Num.Sm90` and `TVA.Lemmas.FloatErr`, `FusedErr`, `FloatMax`,
`Consistency`, `Sm90`, `Softmax`, `Sum`, `SumFused`, `DotLoop`, `Scan` (1230 constants) uses no
axiom beyond `propext`, `Classical.choice` and `Quot.sound`; `tvr library-check` passes (hygiene,
and `leanchecker TVA` re-checks every declaration); `tvr audit` prints `independent`.

**Kernel that does not build or pass the gate:** `kernels/vec_add_f32`, as before this round
(expected). Every kernel item that reaches a float constant has a new digest. I did not edit the
kernel folder or run `tvr check`.

**Release check** (`tvr release`, once, on this tree): 9 of 11 steps pass (shipped, packages,
sandbox, build, library, records, lint, audit). The two failures are the kernel's: `gate
vec_add_f32` (it does not build) and `pages` (`kernels/README.md` and `vec_add_f32/REVIEW.md`
differ from the pages regenerated from its stale gate).

**Files of other rounds.** Kept small:
- `lean/TVA.lean`: the imports of the new modules instead of the deleted ones;
- `lean/TVA/Lang/Scalar.lean`, outside its float parts: the module docstring's float line (it
  cites `TVA.Math.Arith` and `evidence/float-layer`);
- `lean/TVA/Lemmas/Scan.lean`: `FloatV.sortKey` is `Format.orderKey`, and the `SortKey.Lawful`
  proof uses `Format.cmp_lt_iff`, `cmp_eq_iff` (the instance's statement is unchanged);
- `lean/TVA/Lemmas/Consistency.lean`: the proofs of `FloatV.exists_nearest` and its neighbours,
  on the new lemma names (statements unchanged);
- `lean/TVA/Lemmas/Sum.lean`, `DotLoop.lean`, `SumFused.lean`, `Softmax.lean`, `Sm90.lean`:
  imports, and proofs on the new names (`Softmax.lean` imports
  `Mathlib.Analysis.Complex.ExponentialBounds` itself, which a deleted module brought before);
- `docs/plan/examples.md`: the module names `FusedErr`, `FloatMax`.

Reviewers' work files of earlier rounds that import deleted modules
(`round-9/lr-9a/work/cvt_lean.lean` and others) no longer build; I left them as they are.

**Observation for round 11** (not changed): `TVA.Lang.Promote`'s `okPy` tests a Python float's
sign with `v.toRat < 0`; `toRat` of a NaN with the sign bit set is a negative formula value, so
such a NaN counts as negative there, where Python's `v < 0` is false for every NaN (for `−∞` both
say negative). `FloatV.lt v 0` (ordered) would match Python, if the front end tests it that way.

## L65: TVA's float layer, written independently

- **What changed.**
  - **Deleted**: `lean/TVA/Math/Float/` and the bytes module of `lean/TVA/Math` (the copied
    layer), and `lean/TVA/Lemmas/SoftmaxMax.lean`, `Fused.lean`, `Num.lean`.
  - **New layer, `lean/TVA/Math/`** (namespaces `TVA.Math.Float` and `TVA.Math.Bytes`, as
    before, so `Format`, `RoundMode` and the cards keep their names):
    - `Defs`: every definition, in core Lean, so that `evidence/float-layer` compiles exactly
      these constants to native code. A format is `⟨e, m, specials⟩`, `Specials` being `.ieee`
      (`binary16`, `bfloat16`, `binary32`, `binary64`, `e5m2`), `.nanOnly` (`e4m3`: no infinity,
      NaN only `0x7f`/`0xff`) or `.finite` (`e2m1`). A pattern is a natural number; `toRat` is its
      exact value (`magVal` of the magnitude, negated by the sign bit); `isNaN`, `isInf`,
      `isFinite`, `isZero`, `sign`, `mag`, `setSign`, `maxVal`, `maxMag`, `nanMag`,
      `u = 2^-(m+1)`, `eta = 2^(emin−m−1)`, `Regular` (`e ≥ 2`, `m ≥ 1`), `orderKey`.
      `roundRat rnd sat zs q` rounds an exact rational: the quantum `2^Q`,
      `Q = max(⌊log₂|q|⌋, emin) − m` (`qexp`, with `qlog2` from the bit lengths of numerator and
      denominator), the significand rounded by `rint` (to nearest even, or toward zero), packed
      into the fields (`pack`); an overflow (`encode`) gives `±∞` (`.ieee`), NaN (`.nanOnly`) or
      `±maxVal` (`.finite`, saturation `sat`, or `.rz`). `sqrtMag` rounds `√a` from the integer
      square root. The operations `add`, `sub` (`x + (−y)`), `mul`, `fma`, `div`, `sqrt`, `neg`,
      `absPat`, `maxNum`, `minNum`, `maxNaN`, `minNaN`, `cmp` and `cvt f g rnd sat` follow IEEE
      754-2019 §5.4, §6, §9.6: the special operands first, every NaN result the canonical NaN, the
      sign of an exact zero by §6.3. `Wider f g`. `powFast`, `intPowFast`: what compiled code runs
      for `^` (`2^e` as a shift), installed by `@[csimp]` with proofs that they are `Nat.pow` and
      `Int.pow` (proofs are unaffected; without them the evidence's `float16` addition ran about 3 times
      slower).
    - `LittleEndian` (`TVA.Math.Bytes`): `Byte`, `leBytes`, `leValue`, `leValue_leBytes`,
      `leBytes_get`.
    - `Encoding`, `Rounding`, `Arith`: the proofs, among them `pow2_eq`, `u_eq`, `eta_eq`,
      `qlog2_spec` (`qlog2` is `⌊log₂⌋`), `qexp_eq`, `toRat_lt_iff_orderKey`,
      `abs_toRat_le_maxVal`, `maxMag_eq`, `roundRat_spec`, `roundRat_rn_err` (finite, within
      `|q|·u + η`, for `|q| ≤ maxVal`), `roundRat_rn_err_normal` (`|q|·u` from `2^emin`),
      `roundRat_rz_err`, `roundRat_rn_err_of_finite` (relative to a finite result, also beyond
      `maxVal`), `add_of_finite` … `cvt_of_finite`, `cmp_lt_iff`, `cmp_eq_iff`, `cvt_of_wider`
      with 13 `wider_*` pairs, and each format's constants (`binary32_u`, `e4m3_maxVal`, …).
  - **What the deleted lemma modules provided, proved afresh**: `TVA.Lemmas.FloatErr` (for
    `Num`): `FloatV.add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err`, `cvt_err`,
    `ofRat_err`, `ofPat_roundRat_err(_normal)`, `cvtWith_rz_err`, `toRat_cvt_of_wider` and the
    encoding lemmas; `TVA.Lemmas.FusedErr` (for `Fused`): `RoundsTo.err`, `RoundsTo.abs_le`,
    `FloatV.AddRel.err`, `FloatV.mulAdd_err`; `TVA.Lemmas.FloatMax` (for `SoftmaxMax`):
    comparisons by value, `max_mem`, `MaxRel.mem`, `MaxRel.le`, `MaxRel.ge`, `Tree.eval_maxRel`,
    `FloatV.negInf`, `posInf`, `max_negInf_left`, `MaxRel.negInf_left`, `sub_negInf_left`,
    `sub_negInf_right`, `sub_negInf_negInf`.
  - **For round 14**: `FloatV.cvtSat rnd` (saturating: `±maxVal` beyond the range and for `±∞`)
    with `FloatV.cvtSat_of_le` (it is `cvtWith` within range), for L64: the evidence finds
    `x.to(fp8)` is `cvtSat .rn` on every input. `FloatV.negInf` is generic; L66's second name for
    `float32` `−∞` was in the deleted `SoftmaxMax`, so only `Rules.Num.negInfF32` remains. The
    evidence's `B.div.f64` is device evidence for `DivRn .binary64` (L67 asks for a folder of its
    own, with divisors near 1).
  - **Renames** (names the rest of the library used): the modules `TVA.Lemmas.Num` →
    `TVA.Lemmas.FloatErr`, `TVA.Lemmas.Fused` → `TVA.Lemmas.FusedErr`, `TVA.Lemmas.SoftmaxMax` →
    `TVA.Lemmas.FloatMax`, `TVA.Math.Float.*` → `TVA.Math.Defs`, `Encoding`, `Rounding`, `Arith`,
    the bytes module → `TVA.Math.LittleEndian`; `Format.special` → `Format.specials`,
    `Format.ofRat` → `Format.roundRat .rn false false`, `Format.negate` → `Format.neg`,
    `ieeeOrderKey` → `Format.orderKey`, `Format.compare` → `Format.cmp`. Not provided:
    `Format.magRat_le_maxVal` (`Format.abs_toRat_le_maxVal` serves) and `Softmax.negInf32`
    (`FloatV.negInf`). Kept: every `Rules.Num` card and `Rules.Num.Sm90` fact, `DivFull`,
    `DivRn`, `DivFull.of_divRn`, `Platform.ref`, `Platform.sm90ref` and their fields, and the
    names `docs/SEMANTICS.md` cites (`FloatV.litF64`, `litF32`, `litF16`, `litBF16`,
    `FloatV.ofPat`, `FloatV.NaNEq`, `FloatV.MaxRel`, `FloatV.cvt`, `FloatV.cvtWith`,
    `RoundMode.rz`).
  - **Docs**: `docs/LIBRARY.md` (the Math table, the `FloatErr`, `FusedErr`, `FloatMax` rows),
    `docs/rules/num.md` ("Exact operations need no rule": the layer and its evidence).
- **Why.** Protocol §2 (Independence): the layer and three lemma modules were another project's
  code, and their docstrings cited validation TVA does not have.
- **Evidence.** [evidence/float-layer](../../../../evidence/float-layer/README.md), TVA's own: the
  definitions of `TVA.Math.Defs`, compiled to native code, against Triton 3.8.0 on the H200 NVL,
  compared per chunk of `2^20` inputs through a 64-bit hash with an odd multiplier per index (NaN
  up to payload; a differing chunk is dumped on both sides). 150 tests, 1.26·10^11 inputs:
  - every input or pair of 29 operations over `2^32`: `float32` decoded into `float64`, into
    `float16`/`bfloat16` (`"rtne"`, `"rtz"`), into both fp8 formats; `+ - *` on every `float16`
    and `bfloat16` pair; the `float16` comparisons and `tl.maximum`/`tl.minimum` (both
    `propagate_nan`) on every pair; `tl.sqrt_rn`, `-x`, `tl.abs` on every `float32`; every
    `int32`/`uint32` into `float32` and every `float32` into `int32`/`uint32`;
  - every input of the 8- and 16-bit formats and integers (conversions, sign operations, `ftoi`),
    the 16 `e2m1` values through `tl.dot_scaled`, and the memory encoding of 15 dtypes (part E);
  - structured samples (`2^22` to `2^26`) elsewhere: `float32` and `float64` `+ - *`,
    `tl.div_rn`/`/`, `tl.fma`, comparisons; `tl.fma` on 16-bit formats; 64-bit integers; `float64`
    conversions.

  143 agree on every input. The 7 that differ are the non-saturating fp8 conversions the device
  does not perform: `x.to(fp8)` gives `±maxVal` where the library's `sat := false` overflows to
  `±∞`/NaN (every difference is of that kind); the `sat := true` variant agrees on every input,
  and `"rtz"` into `float8e5` is `sat := false` (the device keeps `±∞`). The docstrings of the
  layer cite only this folder. `evidence/literals` (round 11's, for `FloatV.litF64` …
  `litBF16`, which now round with this layer), re-run on the new layer, printed exactly its
  recorded `out.txt`.
- **Constants whose meaning changed.**
  - `TVA.Math.*`: all (new definitions; the deleted modules' constants are gone). Its
    conventions, for a reviewer to compare with the device and with the cards: `toRat` of `±∞`
    and NaN is the formula's value beyond `maxVal` (no statement uses it); every NaN result is the
    canonical NaN `nanMag` (positive, all ones); a nonzero `q` that rounds to zero keeps its sign,
    an exact zero takes `zs` (`x + y`: `−0` only for two `−0`; products and quotients: the
    exclusive or; `fma`: the product's sign and `z`'s); `sqrt(−0) = −0`; `maxNum(−0, +0) = +0`,
    `minNum(−0, +0) = −0`; a conversion of `±∞` gives `±∞`, NaN into `e4m3`, `±maxVal` when
    saturating.
  - `TVA.Lang.Scalar` (float parts): `FloatV.add`, `sub`, `mul`, `fma`, `divRn`, `sqrtRn`, `neg`,
    `abs`, `max`, `min`, `maxNaN`, `minNaN`, `lt`, `le`, `eq`, `ne`, `ofRat`, `cvt`, `cvtWith`,
    `nan`, `isNaN`, `isInf`, `isFinite`, `isZero`, `sign`, `toRat`, `toReal`, and through them
    `litF64`, `litF32`, `litF16`, `litBF16` (the infinity now written
    `Format.bfloat16.setSign … infMag`) and `DType.toBits`/`ofBits` on floats: the same text over
    the new layer. New: `FloatV.cvtSat`.
  - Through those: every `tl.*` float operation and float card (`TVA.Lang.Ops`, `Platform`,
    `Rules.Num`, `Rules.Num.Sm90`, `Lang.Promote`'s float constants), and `FloatV.sortKey`
    (`TVA.Lemmas.Scan`). Their text is unchanged; what they mean is what the new layer defines,
    which the evidence compares with the device. Where the deleted layer's values differed I
    cannot say, having not seen it.
