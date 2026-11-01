# Review of matmul_f16, round 3, by r-matmul

## Verdict

accept
Library round 26 added the fp8 loss facts and fixed some wording. Nothing that `assumption:dot`, `assumption:dot_total` or `spec` relies on changed in meaning: `DotF16` (`DotLiftAcc`), its listing and constants, `DotTotal`, and the specification's `tol`, `dotMag` and `mag16` are as before. I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- The coordinator's brief, the summaries of L124 and L127–L133 in `docs/library-review/ISSUES.md`, `REVIEW.md`'s per-item changes, and `git diff HEAD` of `docs/rules/num.md` and `lean/TVA`.
- The Lean diff only adds `DotChunkLoss`, the four `…ChunkLoss` facts and their lemmas in the two fp8 chunk files, none of which my items reach.
- In `num.md`, these are unchanged:
  - the `DotF16`/`DotBF16` row of "Facts the library states";
  - the reviewer's table row for `float16`/`bfloat16` operands;
  - the `DotLiftAcc` and `…Total` rows;
  - the 16-bit bullet of the chunk model.
  The L130 rewording of the chunk model's opening keeps 16 products per 16-bit instruction on both paths.
- The gate snapshots `565503552a0c46d3` and `471e3872cf6cb659`: only `num.md`'s file digest changed in the three re-opened items. The kernel's files are unchanged.

## Not checked

- Nothing new. What round 1 lists as not checked still holds.
