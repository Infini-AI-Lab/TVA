# Library review, round 35, by lr-35b

## Verdicts

- **L160** ACCEPT — `cmd_release` (`tools/tvr.py`) now runs `build` (`lake build TVA`), `library`, `records`, `lint` on unrecorded rounds and `audit`, and no kernel gate, certificate or page step; the comment says what of `kernels/` is still read (records, lint, audit). Checked by reading the code only: the release check was not run (coordinator's instruction).
- **L161** ACCEPT — Re-ran evidence/atomic-result-order on the H200: launch A and launch B are outside the semantics in 200 of 200 runs each, with both `acq_rel` and `relaxed` (`work/rerun_atomic/out.txt`), as the README says. `TVA.AtomicResult.crosswarp_excluded` uses only `propext`, `Classical.choice` and `Quot.sound` (`work/L161Axioms.txt`). The exclusion's second case (another atomic operation of the same program) is the one launch B needs. A plain access by the same program to a cell another lane's atomic reaches is already `IntraRace` (`Sem/Launch.lean:219`, any two conflicting events of one program), so that case is covered too.
- **L162** ACCEPT — SEMANTICS.md "Atomics" and the `Ops.lean` docstring name both readings, give the cases where they differ, explain why both are sound, and say what to do when a reviewer cannot tell. The L161 exclusion reads "uses" the same way. Read only.
- **L163** ACCEPT — evidence/div-fusion re-ran bit for bit (`work/rerun_divfusion/out.txt` matches `out.txt`: 0 of 430080 outside `AddRel`). My own spellings are also all inside the model (`work/extra_forms.py`): `(a/b)*d + c`, `d*(a/b) + c`, `tl.where(m, a/b, b/a) + c`, `a/b + a/b`, `int32 i / j + c`, `tl.maximum(a/b, 0) + c` and `tl.where(True, a/b, 0) + c`. Constant divisors (`work/constdiv.py`; 13 `tl.constexpr` values from `2^-149` to the largest float, ±3, 7, 0.1): ptxas folds them into one or two `FMUL`s by constants, and the last of these fuses. Every quotient is within `DivFull` (worst 0.428 of the card) and every sum within `AddRel`. A sweep of 34 `op(x) + c` forms (`work/fusion_sweep.py`) checks exact Sterbenz sums against the stored value and diffs the SASS. Only `*`, `/` and `tl.sigmoid` become `FFMA`. No other operation fuses: `exp`, `exp2`, `log`, `log2`, `sqrt`, `sqrt_rn`, `rsqrt`, `sin`, `cos`, `erf`, `tl.fdiv(ieee_rounding=True)`, `tl.div_rn`, `tl.softmax(ieee_rounding=True)`, `tl.fma(a,b,±0)`, `a + a`, `maximum`, `floor`, `ceil`, `clamp`, `abs`, conversions, `cumsum` and `float64` `/`, `sqrt` and `fma` all give 0 inexact sums of 8192 and add exactly one `FADD`/`DADD`.
- **L164** REOPEN — The new wording is still wrong (`work/atom_maxmin.txt`, H200). Triton chooses the atom by the operand's sign bit (`semantic.py:1326`, `_signbit`), so exactly one atom runs per enabled lane, NaN operands included. A `+NaN` operand of `atomic_max` replaced every non-NaN cell. 7 of the 16 NaN-operand lanes wrote the NaN, in each op. For `atomic_min` the pair is `min.s32` and `max.u32`. `fBound` holds: 0 of 128 new values are neither the old value nor the operand. Fix SEMANTICS.md "Atomics" and the Part P note in evidence/relaxed-atomics/README.md to read: "exactly one per enabled lane, chosen by the operand's sign bit: `max.s32`/`min.u32` (max), `min.s32`/`max.u32` (min)". Drop "neither with a NaN operand" and "or none".

## New issues

- **NEW** wording — `tl.fdiv(x, y, ieee_rounding=True)` is `tt.precise_divf` (`semantic.py:337-340`), that is `div.rn.f32` like `tl.div_rn`, but the library gives it no meaning. SEMANTICS.md, `Ops.lean` and num.md "Division" state only `ieee_rounding=False`. `tl.softmax(..., ieee_rounding=True)` (`standard.py:56-64`) also ends in that IEEE division, yet SEMANTICS.md calls `tl.softmax` a product without qualification. That is sound but loose. On the device both are IEEE and not fused (`work/fusion_sweep.txt`). Fix: state that `ieee_rounding=True` is `tl.div_rn` (IEEE, not a product), and that `tl.softmax`'s division follows its `ieee_rounding`.

## Re-ran

- evidence/div-fusion (`work/rerun_divfusion/`, H200 NVL, Triton 3.8.0, 2026-11-08): output identical to `out.txt`.
- evidence/atomic-result-order (`work/rerun_atomic/`): 200 of 200 outside the semantics in every row, as the README says.
- `#print axioms TVA.AtomicResult.crosswarp_excluded` (`work/L161Axioms.lean`): the standard axioms only.
- My own probes in `work/`: `fusion_sweep.py`, `extra_forms.py`, `constdiv.py` and `atom_maxmin.py`, each with its `.txt` (H200 NVL, Triton 3.8.0, 2026-11-08).

## Not checked

- The release check (`tvr release`) was not run (coordinator's instruction); L160 was judged from the code.
- L163's list of affected claims (`layer_norm_fwd_f16`, `layer_norm_fwd_bf16`, and the other 39 said to be unaffected) was spot-checked only, not audited claim by claim.
- The rest of `Lemmas/AtomicResult.lean` beyond its axioms and statement.
