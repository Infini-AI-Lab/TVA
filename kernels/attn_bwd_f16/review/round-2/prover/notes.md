# Prover's notes on attn_bwd_f16, round 2

## Changes

- I re-proved the claim on library round 25's restated `Rules.Num.DotF16` (L126). It is now a
  `DotLiftAcc` fact, relative to `|acc| + Σ liftMag a·liftMag b`, with a subnormal `float16`
  operand counted at `2^-14`. Subnormal operands stay covered.
- `Spec.lean`:
  - new `lm` (the lifted magnitude of a `float16` value) and `lift16`;
  - `aQK` and `aVD` are `Σ lm·lm`;
  - new `p16L` and `ds16L`;
  - the accumulation term of `dVTol`, `dKAcc` and `dQAcc` is over `p16L·lm dO`, `ds16L·lm Q` and
    `ds16L·lm K`;
  - the `dot` assumption's docstring describes the lifted fact.
- Proofs:
  - `dot0` and `accI_dotAdd` use `DotLift.dotAdd_step`;
  - new lemmas `lm_le_lift16`, `f16_small_zero`, `lm_le`, `abs_mul_le_lm`, `sum_bd_le_lm`,
    `p16L_bd`, `ds16L_bd`, `vanishL`;
  - `Basic`, `Step`, `Out` and `Program` were adjusted to match.
- README:
  - a new section, "What changed (library round 25)";
  - the tolerance text and the inline script follow the new formulas;
  - the table's figures moved by at most `0.07` percentage points.

## Responses

- **F5** REFERRED — L126, fixed in library round 25: `DotF16` is restated with lifted magnitudes,
  and I re-proved the claim on it. The specification counts each `tl.dot` operand by `lm`, and
  bounds the fp16 `P` and `dS` by `p16L` and `ds16L`. Subnormal operands stay covered.
- **F6** REFERRED — L126, fixed in library round 25: `DotF16` and `DotBF16` are restated in the
  lifted form the finding asked for, with new evidence. I re-proved the claim: the steps on lines
  303, 310 and 358 bound each product's lifted magnitude (`lm_le_lift16`), and the per-pair dots use
  the lifted `aQK` and `aVD`.

## Library requests

None.

## Gate

PASS, digest `b28f50727b67c8a8`
