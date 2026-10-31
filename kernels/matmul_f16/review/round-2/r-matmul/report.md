# Review of matmul_f16, round 2, by r-matmul

## Verdict

accept
The prover settled all three of my findings, and I find nothing new. `K_le` is now `K ≤ 2^13`, which still covers the host's largest `K = 4096`. There the tolerance, below 0.21% of `Σ|a·b|` (`tol_rel_lt`), still rejects a `bfloat16`-rounded result, and the write-up states the reason. The docstring and map wording are fixed, and the model's definitions are unchanged. I accept every item at its current digest.

## Findings

- **release F4** — `K_le` is `K ≤ 2^13`, `tol_rel_lt` bounds the tolerance's relative part below 0.21%, and Post rejects a `bfloat16`-rounded result at every covered size; the reason is in the README and the docstring
- **release F5** — `tol`'s docstring quotes `tol`'s own values and states `2T·(ρ·A + α)` separately; the figures check
- **release F6** — `model`'s docstring and the map note of line 294 cover `K = 1`, and the map notes of lines 269, 270, 273 and 302 cover the host's strides equal to `1`

## Accepted

All items.

## Checked

- What changed, from REVIEW.md, the prover's notes and the gate snapshots: between `71b09e4a7d289560` and `27825a0f1b176612`, only `Inst.ok.K_le` and `Inst.ok.mk` changed among the cone constants. The model and map digests changed through docstrings and map notes. `matmul_kernel`, `model` and `launch` hash as before, and the reprint in `work/Recheck.lean` matches round 1.
- `Defs.lean` read in full again: `K_le : d.K ≤ 2 ^ 13` with its docstring, the new `tol_rel_lt` (`(1 + relDot16 64)^(steps K) − 1 < 21/10000` for `K ≤ 2^13`), the `tol` and `model` docstrings, and the `c_rows` docstring. `Program.lean` derives `K ≤ 2^16` from `K_le` for `acc_step`, and `satisfiable` (`K = 512`) still meets `K_le`.
- `work/Recheck.lean`, checked with Lean. At `K = 2^13`, the relative part of `tol` plus a `float16` half-ulp (`2^-11`) stays below `2^-8/(1+2^-8)`. So a result rounded to `bfloat16` on a one-sign input with a near-largest rounding error has no `float32` `v` within `tol` that rounds to it, and Post rejects it. `#print axioms` for `tol_rel_lt`, `main`, `consistent` and `satisfiable` shows only `propext`, `Classical.choice` and `Quot.sound`.
- The README: the Specification (2.02e-3 at `2^13`, 0.21%, the `bfloat16` reason, 4% and 12% of a typical `randn` `|S|` at 4096 and `2^13`, the crossovers 15745 and 34368), the `K_le` and `c_rows` rows of the Conditions table, the coverage paragraph (`K ≤ 4096 ≤ 2^13`), the Narrowings (`K ≤ 2^13`, rows of `c` within `stride_cm`) and the Proof outline. All match the Lean and my numbers.
- The revised map notes of lines 269, 270, 273, 294 and 302 against the source and SEMANTICS.md (Loops, Scalar arguments, Constants).

## Not checked

- What I listed in round 1 still holds and is unchanged by the revision: the proofs beyond Lean's check, part (B) of `Compiler.Faithful`'s scope, whether Triton accepts `tl.assume(True)` (the model is a no-op either way), and the internals of `triton.testing.do_bench`.
