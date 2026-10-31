# Prover's notes on matmul_f16, round 1

## Changes

- `Inst.ok.K_le` is now `K ≤ 2^13` (was `2^16`), so the tolerance is meaningful at the largest covered size (F4, F3).
  - The new lemma `tol_rel_lt` (Defs.lean) proves that the relative part of `tol` is below 0.21% for every `K ≤ 2^13`, about half a bfloat16 rounding.
  - `safe_program` derives the `K ≤ 2^16` that `acc_step` needs from the new bound.
  - The README's Specification, Conditions and Narrowings give the new figures, the bfloat16 reason, and the zero-mean (randn) caveat.
- The docstring of `tol` quotes `tol`'s own values, and states the bound `2T·(ρ·A + α)` separately (F5).
- The docstring of `model` and the map note of line 294 cover `K = 1` as the bound of `range(0, 1)` (F1, F6).
- The map notes of lines 269, 270, 273 and 302 say that `stride_ak = stride_bn = stride_cn = 1` on every host launch (F6).
- The docstring of `Inst.ok.c_rows` and the README say that `c_rows` also excludes column-major and transposed layouts of `c`, which do not overlap (F2).

## Responses

- **F1** REVISED — `model`'s docstring (and the map note of line 294) now covers `K = 1` as the bound of `range(0, 1)`.
- **F2** REVISED — `c_rows`'s docstring and the README now say it excludes column-major and transposed layouts of `c` too, though they do not overlap.
- **F3** REVISED — with F4, `K ≤ 2^13`; the README states that the bound is relative to `Σ|a·b|`, about 12% of a typical randn `|S|` at `2^13`.
- **F4** REVISED — option (a): `K_le` is `K ≤ 2^13`, with `tol_rel_lt` (below 0.21%, about half a bfloat16 rounding) and the reason in the write-up.
- **F5** REVISED — `tol`'s docstring quotes `tol`'s own values (1.0e-3·A at 4096, 2.0e-3·A at 2^13), and gives the `2T·(ρ·A + α)` bound separately.
- **F6** REVISED — as F1, plus map notes on the host's `stride_ak = stride_bn = stride_cn = 1` (`tl.assume(True)`, `64 * 1`).

## Library requests

None.

## Gate

PASS, digest `27825a0f1b176612`
