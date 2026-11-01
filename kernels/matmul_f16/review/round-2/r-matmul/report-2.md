# Review of matmul_f16, round 2, by r-matmul

## Verdict

accept
Library round 24 withdrew the fp8 `tl.dot` facts (L125) and added the fp8 chunk facts (L124). Nothing that `assumption:dot`, `assumption:dot_total` or `spec` relies on changed: `DotF16`, its listing, `DotTotal`, `relDot16`, `absDot16` and the page's statements about the `float16` dot are as before. I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- L124 and L125 in `docs/library-review/ISSUES.md`, and the coordinator's note in `REVIEW.md`.
- The diff of `lean/TVA/Rules/Num/Sm90.lean`. It only removes `DotF8`, `DotE4m3`, `DotE5m2`, `DotE4m3E5m2` and `DotE5m2E4m3`, and moves `chunks8`, `relDot8` and `absDot8` to `Dot8Chunk.lean`. `DotF16`, `DotBF16`, `relDot16` and `absDot16` are untouched.
- The diff of `docs/rules/num.md`. The `DotF16` row of "Facts the library states" (`relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126`) is unchanged. So are the reviewer's table row that lists `DotF16` for `float16` operands into `float32` with `maxImpreciseAcc = 0` and any precision, the `…Total` row, and the chunk-model bullet on 16-bit operands. The changes are all fp8.
- The gate snapshots `27825a0f1b176612` and `00f23d84e4f9dd74`: only the whole-file digests of the two files changed in the three re-opened items. Every constant in their cones hashes as before, and the kernel's own files are unchanged.

## Not checked

- Nothing new. What round 1 lists as not checked still holds.
