# Prover's notes on softmax_f32, round 1

## Changes

- `Post` (Model.lean) is restated with one factor `sc` per row, `|sc − 1| ≤ sumTol B = (B + 86)·2^-24`, and per-element bounds `|c_j − sc·σ_j| ≤ (elemTol + gapTol·softmaxGap_j)·σ_j + floorTol·softmaxMax`. The new definitions are `softmaxMax`, `softmaxGap`, `sumTol`, `elemTol`, `gapTol` and `floorTol`; `relTol` and `absTol` are removed. Model, conditions, map and assumptions are unchanged, and their digests are the same.
- Err.lean has a new error analysis. `exp_lane` keeps each lane's error relative, with a part that grows with `|s|`. New lemmas: `exp_neg_nat_le`, `exp_neg16_le`, `abs_mul_exp_le16`, `sum_common` (the common factor) and `final_common` (each output). `lane_facts` and `row_bound` are restated. `sum_rel` and `final_bound` are removed.
- Proof.lean: `RowDone` follows `Post`'s new row clause, and `safe_row` handles an empty row (`n_cols = 0`) separately.
- README: Specification is rewritten, with the tolerance table, why a common factor, the meaning at covered sizes and the half-precision rejection argument. In Conditions, the 'Covered in particular' sentence is fixed and there is a paragraph on the benchmark's stream (scope point 1). The Proof outline and the `BLOCK_SIZE` narrowing are updated.

## Responses

- **F1** REVISED — Post has a common factor per row (within (B + 86)·2^-24) and B-independent relative per-element bounds; float16 and bfloat16 results now fail it at every BLOCK_SIZE ≥ 2.
- **F2** CLARIFIED — README says the benchmark's fresh stream relies on scope point 1 (part B): only launches that start after randn has written x are covered.
- **F3** CLARIFIED — README 'Covered in particular' now includes in_off/out_off and the buffer lengths, for example 131073 rows of 16384 columns.
- **F4** REVISED — relTol is replaced by sumTol(B) = (B + 86)·2^-24 for the common factor only, 87·2^-24 above the library's tree bound and below bfloat16's rounding at every covered B.
- **F5** REVISED — absTol is replaced by floorTol·softmaxMax = 2^-133·σ_max; per-element bounds are relative, growing with x_max − x_j by 3·2^-24 per unit.
- **F6** CLARIFIED — same README fix as F3, with 'at most' for rows_grid and the int32 wrap as the kernel's own limit.

## Library requests

None.

## Gate

PASS, digest `b822ec44212cc840`
