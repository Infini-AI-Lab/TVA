# Prover's notes on block_scaled_matmul_mxfp8, round 1

## Changes

- README.md only (no Lean, no condition, no map change). The benchmark's launches as the script
  makes them are now stated as not covered: they run after `proton.start(…, hook="triton")`
  (line 756) has installed Triton's hooks, which stay installed until `proton.finalize()`
  (line 764), outside `Compiler.Faithful`'s scope point 3. The claim covers the test's launch
  (line 749) and launches of the benchmark's sizes made with no hook installed. This is said in the
  introduction, in Conditions ("Which of the host's launches are covered", the part (A) preamble,
  the point-4 and point-9 bullets, the part (B) paragraph) and in a new Narrowings bullet; the
  environment bullet is updated.
- The `K ≤ 2^14` Narrowings bullet now says the proof does not need the bound (only the `int32`
  bounds and the room condition). It is kept so that the tolerance stays meaningful at every
  covered `K`.

## Responses

- **F1** CLARIFIED — write-up only: the benchmark's hooked launches are out of scope (point 3);
  the test's launch and hook-free launches of the benchmark's sizes are covered.
- **F2** CLARIFIED — the same point, write-up only; `Inst.ok` still covers the benchmark's sizes,
  so no condition changes. I keep `K ≤ 2^14` (non-blocking remark) so the tolerance stays
  meaningful.

## Library requests

None.

## Gate

PASS, digest `2fac55fa1cb0c3c5`
