# Prover's notes on streamk_matmul_f16, round 1

## Changes

- README, Specification: the test's data is no longer said to have no subnormal operand. `A` is the
  test's normaliser `(|A| @ |B|)[i, j]` up to the excess of the few subnormal operands of float16
  `torch.randn` (at most `1.9·10^-7` relative on the reviewers' sampled shapes), so the per-shape
  bounds hold to the digits given (F1, F2).
- README, Specification: one sentence on what `Post` does not reject. From `K = 1921` (`T = 31`) on,
  the tolerance exceeds `4.88·10^-4·A`, the largest relative error of one rounding to `float16`
  (`5.07·10^-4·A` at `K = 2048`, `1.015·10^-3·A` at `K = 4096`), so `Post` cannot tell a result
  rounded to `float16` from a `float32` one there. This follows from the library's `DotF16` constant
  (r-sk2's filing note 1).
- `Inst.ok.K_le`'s docstring: `0.21%` of `A` (`dotMag`; `Σ|a·b|` when no operand is subnormal), in
  place of `0.21%` of `Σ|a·b|` (F3).
- `testInst`'s docstring: the first test instance is line 100, not 99 (r-sk2's filing note 3).
- Nothing else changed. No item digest changed, and the gate digest is the same.

## Responses

- **F1** CLARIFIED — README: `A` is the test's normaliser up to a subnormal excess of at most `1.9·10^-7` relative; the figures stand.
- **F2** CLARIFIED — the same README change as F1.
- **F3** CLARIFIED — `K_le`'s docstring now says `0.21%` of `A` (`dotMag`), `Σ|a·b|` without subnormal operands.

## Library requests

None.

## Gate

PASS, digest `16c9926eb17e72c9`
