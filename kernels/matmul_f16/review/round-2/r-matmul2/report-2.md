# Review of matmul_f16, round 2, by r-matmul2

## Verdict

accept
Library round 24 re-opened `assumption:dot`, `assumption:dot_total` and `spec`. The diff withdraws only the fp8 `tl.dot` facts, adds the fp8 chunk facts, and moves `chunks8`, `relDot8` and `absDot8`. `DotF16`, `relDot16`, `absDot16`, `DotTotal`, their listings and the page's 16-bit statements are unchanged, and the kernel's files did not change. So the re-opened items rest on what I accepted before, and I re-accept every item. The change does raise one library doubt, filed as a non-blocking `LIBRARY_REPORT`: L125's subnormal-operand window may affect `DotF16` too.

## Findings

- **LIBRARY_REPORT** `decl:TVA.Rules.Num.DotF16` — L125's mechanism (a subnormal operand's exponent field setting the `wgmma` window) may also break `DotF16`; num-dot's fp16 Part 2 has the same input gap

## Accepted

All items.

## Checked

- `git diff` of `docs/rules/num.md` and `lean/TVA/Rules/Num/Sm90.lean`. In Sm90.lean only one hunk changed, after `DotF16`/`DotBF16`, removing the fp8 constants and facts. `DotF16`, `relDot16` and `absDot16` are as before.
- `num.md`: `DotF16`'s row in "Facts the library states", the `DotTotal` row, and the fp16/bf16 row of "What a kernel reviewer checks" are unchanged. The 16-bit bullet of the chunk-model section is unchanged; only the section's list of facts changed.
- L124 and L125 in `docs/library-review/ISSUES.md`, the coordinator's note in `REVIEW.md`, and `REVIEW.md`'s table: only the three re-opened items are stale, by these two files.
- Whether L125's mechanism carries over to 16-bit operands. evidence/num-dot Part 2 and evidence/num-dot-chunk show no fp16 test where a subnormal-operand product sets the window (the `LIBRARY_REPORT`).

## Not checked

- Whether fp16 `wgmma` aligns by exponent fields with subnormal operands. That is a device question for the library, and I ran nothing on a GPU.
