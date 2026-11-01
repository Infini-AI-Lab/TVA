# Prover's notes on matmul_f16, round 2

## Changes

- Re-proved on the restated `DotF16` (library round 25, L126: `DotLiftAcc`, magnitudes `|acc| + Σ liftMag a·liftMag b`).
  - Defs.lean: the new `mag16` (the lifted float16 magnitude, `Rules.Num.liftMag .f16` by definition) and `dotMag` (`Σ mag16·mag16`) replace `dotAbs` as the tolerance's magnitude `A` in `OutSpec`. The `dot` docstring states the restated fact.
  - Basic.lean: `partA` sums the lifted products `liftV`. `acc_step` cites the new fact. New lemmas: `liftMag_f16`, `mag16_nonneg`, `mag16_zero`, `abs_mul_le_mag16`, `mag16_le`, `liftV_nonneg`, `liftV_le`, `abs_prodV_le`. `prodV_le` is removed.
- README: Specification, Assumptions and Proof updated, and a new section, "What changed (library round 25)". The tolerance figures are unchanged where no operand is subnormal.

## Responses

- **F7** REFERRED — library issue L126, fixed in round 25; the claim is re-proved on the restated `DotF16`, with subnormal operands counted at `2^-14` in the tolerance's magnitude.

## Library requests

None.

## Gate

PASS, digest `565503552a0c46d3`
