# Review of softmax_f32, round 2, by r-softmax2

## Verdict

accept
The new `Post` settles both of my round-1 specification findings. Each row has one common factor `sc` with `|sc − 1| ≤ sumTol B = (B + 86)·2^-24`. That is the only part that grows with `BLOCK_SIZE`, and it is `87·2^-24` above the library's tree term. Each output must be within `(elemTol + gapTol·(x_max − x_j))·σ_j + floorTol·σ_max` of `sc·σ_j`. That bound is relative, does not depend on `B`, and has an absolute part of `2^-133·σ_max`. So a bfloat16 or float16 result fails `Post` at every covered `BLOCK_SIZE ≥ 2`, and small outputs can no longer be returned as `0`. The limit the prover flags is acceptable: at `BLOCK_SIZE = 1024`, `sumTol` (`6.6·10^-5`) is above the test's `rtol` (`10^-5`), but the library's own tree term (`6.1·10^-5`) already is, and the README states it. The README's F6 sentence is fixed. One wording slip remains in its example, filed as a new DOC. Model, map, conditions and assumptions are unchanged since round 1. I accept every item.

## Findings

- **DOC** `doc:kernels/softmax_f32/README.md` — README 'Covered in particular': 16384-column rows are covered up to 131072 rows, not 131073, and the limit is (n_rows - 1)·stride < 2^31, not '2^31 elements or more'
- **release F4** — relTol is gone. Only the common factor grows with B, within 87·2^-24 of the tree term, and half precision fails Post at every covered B.
- **release F5** — absTol is gone. The absolute part is 2^-133·σ_max, small outputs are held relatively, and 0 no longer meets Post.
- **release F6** — the sentence now has in_off/out_off and 'at most'. Its remaining off-by-one is the new DOC.

## Accepted

All items.

## Checked

- The new `Post`, `softmaxMax`, `softmaxGap`, `sumTol`, `elemTol`, `gapTol`, `floorTol` and `RowDone`, printed with full names (`work/spec.lean`). Every name is `TVA.*`, Mathlib's (`Real.log`, `Real.exp`, `iSup`, `Finset.sum`) or the kernel's. `RowDone` is `Post`'s row clause.
- `#print axioms` of `main`, `consistent` and `satisfiable`: `propext`, `Classical.choice`, `Quot.sound`.
- The edges, Lean-checked:
  - `softmaxMax` of an empty row is `0`;
  - a column with `σ = 0` (an input `−∞`) has gap `0`, so its output is held to `floorTol·σ_max`.
- The tolerances at the host's sizes, Lean-checked: `sumTol` at `256`, `1024` and `2^14`, and `sumTol B − (B − 1)·2^-24 = 87·2^-24` for every `B`.
- The README's half-precision argument, recomputed: row `(0, −1)`, float16 `|δ_0 − δ_1| = 5.16·10^-4`, bfloat16 `3.00·10^-3`, against the bound `1.14·10^-6`.
- The README's `[20, 0, …, 0]` figures: `σ ≈ 2.06·10^-9`, held to `4.07·10^-6` relative.
- Plausible wrong kernels against the new `Post`:
  - half-precision outputs fail;
  - zeroing small lanes fails;
  - dropping the max subtraction overflows on covered inputs;
  - `other=0.0` padding fails on short rows.
  - Only a uniform rescaling of a row within `sumTol` passes, which the library's tree bound forces.
- The statements of `exp_lane`, `abs_mul_exp_le16`, `sum_common`, `final_common` and `row_bound`, read to see how they give `Post`'s form.
- The README against the Lean: Specification (table, why a common factor, meaning at covered sizes, the test's limit), Conditions ('Covered in particular', part B paragraph), Proof outline and Narrowings.

## Not checked

- The proofs of the new lemmas beyond Lean's check. I read their statements, not every step.
- The README's part (B) paragraph on the benchmark's stream (r-softmax's F2), beyond reading it: it is not my finding.
