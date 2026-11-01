# Review of grouped_matmul_tma_f16, round 3, by r-gmm2

## Verdict

accept
Library round 25 restated `DotF16` as `DotLiftAcc`. The prover re-proved the claim on it, and changed nothing in the model, conditions, assumptions or map.
- `assumption:dot` still cites `Rules.Num.DotF16 P`, the fact num.md lists for the model's `tl.dot`.
- `OutSpec`'s tolerance now takes `dotLift`, the products' lifted magnitudes, which is the measure the restated fact bounds the error by. It is still explicit and computable. It equals the old one when no operand is subnormal, so the README's figures (0.204% and 0.253% at K = 2^13, thresholds 13633 and 15617) still hold.
- On subnormal-heavy inputs the tolerance is looser, as the README says. Post still rejects a kernel that flushes subnormal inputs.

L126 settles my F6. I re-accept every item.

## Findings

- **release F6** — L126 records the gap faithfully and round 25 fixed it: `DotF16` is now `DotLiftAcc`, and the claim is re-proved on it with subnormal inputs still covered

## Accepted

All items.

## Checked

- The prover's round-2 response and notes, REVIEW.md's changed items and coordinator notes, L126 in ISSUES.md, and the diffs of `Defs.lean`, `Basic.lean`, `README.md`, `lean/TVA/Rules/Num.lean`, `lean/TVA/Rules/Num/Sm90.lean` and `docs/rules/num.md`.
- `assumption:dot`: the field is `Rules.Num.DotF16 P`, now `∀ p, DotLiftAcc P ⟨.f16, .f16, .f32, p, 0⟩ relDot16 absDot16`. The constants are unchanged. num.md's operation table still lists `DotF16 P` for `float16` operands into `float32` with `maxImpreciseAcc = 0` and any precision, and its "Facts the library states" row is restated to match. The docstring states the lifted shape.
- `assumption:dot_total`: `DotTotal` and the model's `DotCfg` are unchanged. It re-opened only because `docs/rules/num.md` and `lean/TVA/Rules/Num.lean` are in its digest.
- `spec`, by `#print` with full names and `#eval` (work/Lift.lean): `dotLift` sums `liftMag .f16 (va k) · liftMag .f16 (vb k)`, and `liftMag .f16` is `|q|` on zero and normal values and `2^-14` on every nonzero subnormal. `DotLoop.stepRel`, `stepAbs` and `bound` are unchanged, so `rho`, `alpha`, `tol` and the README's figures for normal operands stand.
- How meaningful the tolerance is, in exact rationals:
  - On inputs made only of subnormal-times-large products it can exceed `|S|`: all `a = 2^-24`, `b = 65504` at `K = 2^13` give `S = 31.98` and `tol = 66.8`. That is the shape of the library's fact, and the README states it.
  - It still rejects a flush-to-zero kernel: `a = 1023·2^-24`, `b = 1` give `S = 0.4995` and a bound of `0.00126`.
  - For the host's `torch.rand` inputs, each subnormal operand adds less than `2^-14` to `A`.
- The README's new sentences: the specification's `A`, the bounds `Σ|a·b| ≤ A ≤ Σ(|a| + 2^-14)(|b| + 2^-14)`, the factor `2^10`, the assumption's wording, the proof outline (`DotLift.dotAdd_step`, `liftV`, `partS_end`) and the "What changed (library round 25)" section, all against the Lean.
- The axioms of `main`, `consistent` and `satisfiable`: `propext`, `Classical.choice`, `Quot.sound`.
- `model`, `map`, `conditions` and `assumption:faithful` kept their digests, and the diff leaves the model, `Inst.ok`, `Pre` and the map as I accepted them.

## Not checked

- Whether the restated `DotF16` is true of the device (the library's matter; no library review followed round 25).
- The proofs step by step: Lean and the gate check them.
