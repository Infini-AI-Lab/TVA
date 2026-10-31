# Review of softmax_f32, round 2, by r-softmax

## Verdict

accept
The new `Post` says what an fp32 softmax is for. One factor per row, within `(B + 86)·2^-24` of
`1`, takes the row sum's error, the only error that grows with `BLOCK_SIZE`; each output is then
held to `(elemTol + gapTol·(x_max − x_j))·σ_j + 2^-133·σ_max`, relative and independent of `B`.
Because the factor cancels between two outputs of a row, `float16` and `bfloat16` results, a sum
that leaves out a large term, and `other=0.0` padding all fail it on covered inputs, at every
covered `BLOCK_SIZE ≥ 2`. The limit the prover names, `sumTol(1024) = 6.6e-5` against the test's
`rtol = 1e-5`, is the library's bound for a sum along any tree (`sumTol` is within `87·2^-24` of
it). The kernel cannot sharpen that bound, and the README states the limit. F1–F3 are settled; one
non-blocking wording finding is new.

## Findings

- **DOC** `doc:kernels/softmax_f32/README.md` — Two wording slips: README says 'the last condition alone' limits n_rows (it is in_off, not the buffer lengths); Err.lean's lane_facts docstring still gives the old 2^-22·a + 2^-23 bound
- **release F1** — Post has a common factor per row and B-independent relative per-element bounds; half precision, a dropped large term and `other=0.0` are rejected; the remaining limit is the library's any-tree sum bound, stated in the README.
- **release F2** — README states the benchmark's fresh-stream launches rely on scope point 1.
- **release F3** — README's coverage sentence now includes `(n_rows − 1)·stride < 2^31` and the buffer lengths.

## Accepted

All items.

## Checked

- `Post`, `softmaxMax`, `softmaxGap`, `sumTol`, `elemTol`, `gapTol`, `floorTol` in Model.lean, printed with `pp.fullNames` (work/Check2.lean): every name is `TVA.*` or a kernel definition. `softmaxMax` is the row's largest `σ` (at least each `σ_k`, at most any bound of them, `0` for an empty row: Lean-checked). `softmaxGap = log(σ_max/σ_j)` is `x_max − x_j` for a finite input and `0` for `−∞`.
- The new statements in Err.lean (`exp_lane`, `abs_mul_exp_le16`, `sum_common`, `final_common`, `lane_facts`, `row_bound`) and `RowDone`/`safe_row` in Proof.lean: `RowDone` is `Post`'s row clause, the empty row is handled, and `row_bound`'s `sc = A/S` is the row's common factor; no new instance, notation, `open` or option.
- `#print axioms` of `main`, `consistent`, `satisfiable`: `propext`, `Classical.choice`, `Quot.sound`.
- That the factor cancels between two outputs (Lean-checked), and, with the claim's constants, that the covered row `(0, −1)` allows a relative-error difference of `1.1e-6` against `5.2e-4` (`float16`) and `3.0e-3` (`bfloat16`); a sum without its largest term and `other=0.0` padding are rejected too (work/rejection.py). The small outputs of `[20, 0, …, 0]` are held to about `4e-6`.
- `sumTol` against the library's tree bound `(B − 1)·2^-24` at `B = 256, 1024, 2^14`, and the README's statement of the test-size limit.
- The README's Specification (tolerance table, the common factor, the meaning at covered sizes, the half-precision argument, its numbers), the Conditions paragraphs for F2 and F3, and the Proof outline, against the Lean.
- That model, map, conditions and the six assumptions keep their round-1 digests (REVIEW.md), so my round-1 judgment of them stands.

## Not checked

- The proofs' steps beyond their statements (Lean and the gate's replay check them).
- Part (B) of `Compiler.Faithful`'s scope (assumed).
