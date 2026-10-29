# Library review, round 12, by lr-12a

## Verdicts

- **L65** ACCEPT — The new layer means what IEEE 754-2019 and the OCP formats say, and what Triton
  3.8.0 runs on the H200 NVL, wherever the library uses it. Read against the standard: `roundRat`
  (quantum `2^(max(⌊log₂|q|⌋, emin) − m)`, ties to even, carry into the next binade, overflow to
  `±∞`, to NaN in `e4m3`, to `±maxVal` under `rz` or `sat`), the special cases and §6.3 zero signs
  of `add`/`mul`/`fma`/`div`/`sqrt`, `maxNum`/`minNum`, `cmp` (infinities order through their
  formula value `2^(bias+1)`, above every finite value), the `.nanOnly` `e4m3` (NaN `0x7f`/`0xff`
  only, `maxVal` 448) and `.finite` `e2m1` (values `0 … 6`). On the device: I rebuilt the driver
  from the merged `Defs.lean` and re-ran 142 of evidence/float-layer's tests (Re-ran), each with
  its recorded verdict and difference count (and the 15 samples results.out lacks agree). My own
  adversarial inputs (work/adv.py, 37 tests of up to 200000 inputs) agree bit for bit up to NaN
  payload. They include ties at every
  boundary, gradual underflow, ties at half the smallest subnormal, `maxVal` plus half an ulp,
  products that overflow or underflow before an `fma` adds them, the signs of exact zeros, and
  double-rounding traps (work/adv_stats.out counts the cases; a double rounding through `float32`
  would differ on 47934 of the `float64 → float16` inputs, and the device agrees with one
  rounding). The operations use the variant the device runs: `.rn` arithmetic without flushing,
  non-saturating conversions between the standard formats, `tl.cvt8` left to `P.cvt`, whose facts
  (`CvtRn`) say nothing beyond `maxVal`, where the saturation lies. The facts kept their meaning:
  their text is unchanged, and the constants they rest on evaluate as the page says
  (work/litnan.out: `e4m3`/`e5m2`/`e2m1` `maxVal` 448/57344/6, `maxMag` `0x7e`/`0x7b`, `negInfF32`
  infinite and negative, `zeroF32` `+0`). evidence/num-cvt8, evidence/literals and evidence/promotion's
  Lean comparison re-ran identical to their recorded output on the new layer. The error lemmas
  depend only on `propext`, `Classical.choice`, `Quot.sound` (work/axioms.out). The layer's
  docstrings cite only evidence/float-layer, and no file refers to a deleted module. Reach: the
  notes list `TVA.Math.*` and everything through it, and the digests re-open every item whose cone
  reaches a changed file. The defects below are wording, or (the NaN constants) a defect of the
  constants' exactness, which runs through the layer's canonical NaN but is not the layer's: its
  operations are up to NaN payload, as the library uses them.

## New issues

- **NEW** soundness — A Python NaN constant does not have the pattern the library states.
  `FloatV.litF32`, `litF16` and `litBF16` (`lean/TVA/Lang/Scalar.lean`, used by
  `Operand.constIn?` and SEMANTICS "Constants", which says the constant "takes the value Triton
  builds for it") give the canonical NaN `0x7fffffff`, `0x7fff`, `0x7fff` for `float('nan')` and
  for `-float('nan')` alike (work/litnan.out). Triton 3.8.0 builds `0x7fc00000`, `0x7e00`,
  `0x7fc0`, and `0xffc00000`, `0xfe00`, `0xffc0` for the negated NaN (work/nan_probe.out,
  `tl.full`): C's double-to-float conversion and APFloat keep the sign and the quiet payload.
  `litF64` is right (the double itself). These constants are exact functions, not up to NaN
  payload, so a model that stores `tl.full(…, float('nan'), tl.float32)` claims bits the device
  does not write. evidence/literals compares NaN as "any NaN equals any NaN", so it does not support
  the patterns either. Fix: give the three functions the device's NaN patterns (sign kept; the
  quiet NaN with the double's top payload bits for `float32` and `float16`, `±0x7fc0` for the
  `bfloat16` text `nan`/`-nan`), or state in SEMANTICS and `constIn?` that a NaN constant is some
  NaN; and compare NaN bit for bit, with both signs, in evidence/literals.
- **NEW** wording — `FloatV.nan` (`lean/TVA/Lang/Scalar.lean`) is "The canonical NaN NVIDIA
  arithmetic produces: positive, all-ones exponent and fraction". That holds for `float32`,
  `float16` and `bfloat16` arithmetic, not for `float64`: `∞ − ∞`, `∞ · 0` and `fma` give
  `0xfff8000000000000` (negative), and a NaN operand keeps its quieted payload; `float32 → float64`
  and `bfloat16 → float32` keep the payload too (work/nan_probe.out). The operations are up to NaN
  payload, so no meaning changes. Say "the library's canonical NaN", true of the device's
  `float32`/`float16`/`bfloat16` arithmetic and fp8 conversions only.
- **NEW** wording — evidence/float-layer/README.md (and round 12's notes) say "The 7 that differ
  are the non-saturating variants". One of the 7, `A.cvt.f32.e5m2.rz.sat1.sample`, is a saturating
  variant (`"rtz"` into `float8e5`). Its "`sat := true` differs exactly on `±∞` (listed)" is true,
  since it follows from the exhaustive `A.cvt.f32.e5m2.rz.sat0` and the definitions (under `.rz`,
  `sat` changes only the result for `±∞`). But "listed" covers only the 4 of its 64 differing chunks
  that run.py dumps (`bad[:4]`). Say six non-saturating `"rtne"` tests and one saturating `"rtz"`
  test, and give that argument for "exactly".
- **NEW** wording — evidence/float-layer's re-run instructions fail as written. build.sh calls
  `lean` and `leanc` from `PATH` and never uses the `BIN` it computes. With the user's `PATH` there is
  no `lean`, and with elan's proxies the step run in `evidence/float-layer` finds no toolchain
  (no `lean-toolchain` above it, no elan default): "no default toolchain configured". It builds once
  the v4.35.0 toolchain's `bin` is on `PATH`, as I ran it. Run both steps through the toolchain
  that `lean/lean-toolchain` names (for example `lake env` from `lean/`), or say so in the README.

## Re-ran

- evidence/float-layer: `build.sh` on the merged tree (with the v4.35.0 toolchain's `bin` on
  `PATH`), then `run.py` from my work folder (work/float-layer-rerun/results.out): every test
  except the 30 over `2^32` inputs, 135 tests (every `2^8`/`2^16` exhaustive test, every sampled
  test, and the 15 `.sample` tests results.out lacks). Each verdict and difference count is the
  recorded one; the 15 extra agree.
- evidence/float-layer, exhaustive over `2^32` (the same results.out): `B.max.f16`,
  `B.minnan.f16`, `B.add.f16`, `B.mul.bf16`, `B.sqrt.f32`, `A.cvt.f32.e4m3.sat1`,
  `A.cvt.f32.e5m2.sat1`, all agree on every input, as recorded.
- evidence/float-layer `e2m1.py` and `bytes.py`: output identical to `e2m1.out`, `bytes.out`.
- evidence/literals: `out.txt` identical (work/literals-rerun.out).
- evidence/promotion `lean_compare.py`: identical to `lean-compare-output.txt` (0 of 37896
  differ).
- evidence/num-cvt8: identical to `out.txt` (work/num-cvt8-rerun.out).
- Added: work/adv.py (adversarial `+ - * fma max min /` on `float32`/`float64`, `+ - * fma` and
  `max`/`min` on `float16`, `+ - * fma` on `bfloat16`, conversions `float64 → float32/float16/
  bfloat16` and `float32 → float16/bfloat16`, `int64`/`uint64 → float16/bfloat16/float32/float64`;
  37 tests, up to 200000 inputs each (fewer for the integers, whose out-of-range draws are
  dropped), all agree, with double-rounding controls; work/adv.out), work/adv_stats.py
  (what those inputs exercise), work/nan_probe.py (the device's NaN patterns), work/litnan.lean
  (the library's NaN constants, the facts' constants, `tl.rand`'s `scale32 = 0x2fffffff`),
  work/axioms.lean.

## Not checked

- 22 of evidence/float-layer's 29 recorded exhaustive tests over `2^32` not re-run in full (the
  machine was shared): I re-ran their `2^26`-input samples, which agree.
- evidence/num-math-f32, num-div-f32, num-dot, num-dot-scaled and num-atomic-add not re-run: the
  facts' text is unchanged and their references are independent of the layer; I checked only
  that the layer's meaning they rest on (`toRat`, `isFinite`, `maxVal`, `u`, `eta`) is the
  device's.
- The lemma modules (lr-12b's area) beyond the statements the device facts use.
