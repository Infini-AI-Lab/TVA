# Prover's notes on grouped_matmul_tma_f16, round 2

## Changes

- Re-proved on the restated `Rules.Num.DotF16` (library round 25, L126: `DotLiftAcc`, the error relative to `|acc| + Σ liftMag a·liftMag b`). `Defs.lean`: `OutSpec`'s tolerance is `tol (K g) (dotLift …)`, the sum of the products' lifted magnitudes, in place of `dotAbs` (equal when no operand is subnormal); docstrings of `K_le`, `tol`, `OutSpec` and the `dot` assumption follow. `Basic.lean`: `partA` sums the lifted products (`liftV`), `acc_step` uses `DotLift.dotAdd_step`, `partS_end` ends at `dotLift`. README: Specification, Assumptions, Proof and Narrowings follow, and a section "What changed (library round 25)"; figures for normal operands unchanged. Model, conditions, assumptions and map unchanged.

## Responses

- **F5** REFERRED — fixed by L126 (round 25); the claim is re-proved on `DotLiftAcc`, with the tolerance over lifted magnitudes.
- **F6** REFERRED — the same gap, fixed by L126 (round 25); re-proved as for F5.

## Library requests

None.

## Gate

PASS, digest `902c06ecf3fe60ae`
