# Review of matmul_f16, round 3, by r-matmul2

## Verdict

accept
Library round 26 re-opened `assumption:dot`, `assumption:dot_total` and `spec`, through `docs/rules/num.md` only. Its changes concern the fp8 `tl.dot` (the new loss facts, L124) and wording (L127, L130). The `DotF16` and `DotTotal` facts, their listings and constants, the page's 16-bit statements, and the lemmas the specification and proof use are unchanged, as are the kernel's files. I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- `git diff HEAD` of `docs/rules/num.md`: the `DotF16` and `DotTotal` rows and the fp16/bf16 row of "What a kernel reviewer checks" are unchanged. The 16-bit bullet is unchanged; the chunk-model opening is reworded with the same meaning for 16-bit operands (16 products per instruction on both paths). All other changes are fp8.
- `git diff HEAD` of `lean/TVA`: only additions to the two `Dot8Chunk` files, apart from removed module-docstring lines. `Rules/Num.lean`, `Rules/Num/Sm90.lean`, `DotLoop` and `DotLift` are unchanged, and so are the `Dot8Chunk` lemmas the proof uses.
- L124 and L127–L133 in `docs/library-review/ISSUES.md`, the round-26 author's notes on 16-bit operands, and `REVIEW.md`'s list of what changed per item.

## Not checked

- The new fp8 facts and their evidence, which this kernel does not cite. The library is trusted in kernel review.
