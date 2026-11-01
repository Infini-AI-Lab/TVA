# Prover's notes on attn_fwd_f16, round 2

## Changes

- Re-proved on the restated `DotF16` (library round 25, L126: `DotLiftAcc`, relative to lifted magnitudes).
- `Model`: `scoreAbs` is now `scoreMag`, `Σ_e liftMag q·liftMag k` (equal to `Σ_e |q·k|` for zero and normal values), in `HeadOk` and `Post`'s premise on `X`; `omega0` gains `66·2^-36` and `absA` goes from `2^-85` to `2^-22`, with docstrings saying why; `HeadOk`'s docstring bounds the lift on the host's inputs (below `0.031`).
- `Inner`: `qk_lane` and `pv_lane` cite the lifted fact; `Step.astep` bounds the `p·v` `tl.dot`'s lifted magnitudes (`DotLift.liftMag_mul_le`): `2^-14·|v|` per attended key and `5·2^-9` per block; `RowAcc`, `Row`, `Final` follow the new constants.
- Tolerances: `tolO`, `tolM` unchanged; `tolS` about `3%` larger (`3.0·10^-8` to `3.1·10^-8`); `tolF` from `3.0·10^-8` to about `N/64·2^-22` (`1.5·10^-5` at `N = 4096`, `6.4·10^-5` at `N = 2^14`, `3.3·10^-4` at `N = 2^16`).
- `README.md`: a section "What changed (library round 25)", the table and the figures updated.

## Responses

- **F6** REFERRED — L126, fixed in round 25; the claim is re-proved on the lifted `DotF16`

## Library requests

None.

## Gate

PASS, digest `df22d153664804d2`
