# Prover's notes on attn_bwd_f16, round 3

## Changes

- `README.md` only, in the "What changed (library round 25)" section and the text after the
  table:
  - the per-pair errors and `HeadV.Ok` read the lifted `aQK` and `aVD`;
  - `p16L` and `ds16L` lift the bounds, not the values;
  - the list of moved figures is complete;
  - `9.3·10^-3` is now `9.2·10^-3`.
- No Lean, map or manifest change; the gate digest is unchanged.

## Responses

- **F7** CLARIFIED — the section now says that `p16Err`, `ds16Err` and `HeadV.Ok` read the lifted
  `aQK` and `aVD` through `eX` and `eY`, and that `p16L` and `ds16L` lift the bounds. It lists every
  moved figure, including the `HEAD_DIM = 128` causal `dKTol` (`1.50` to `1.51%`) and the causal
  `dV` maximum (`0.91` to `0.92%`).
- **F8** CLARIFIED — the zero-or-normal sentence is corrected (`lm` is the old magnitude; `p16L` and
  `ds16L` equal the bounds only from `2^-14`). The `HEAD_DIM = 128` changes are listed, and
  `9.2·10^-3` replaces `9.3·10^-3`.

## Library requests

None.

## Gate

PASS, digest `b28f50727b67c8a8`
