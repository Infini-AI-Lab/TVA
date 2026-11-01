# Review of attn_fwd_f16, round 4, by r-fwd

## Verdict

accept
Library round 26 re-opened nine assumption items and `spec` only because `docs/rules/num.md` is hashed whole. The page's changes add the fp8 loss facts and reword two fp8 passages. Nothing this kernel cites changed in meaning: its facts, their listings, their constants and the 16-bit statements are as before. I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of `docs/rules/num.md` (against `HEAD`). It adds rows and a section for `DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`, `DotE4m3E5m2ChunkLoss` and `DotE5m2E4m3ChunkLoss`, rewords the fp8 accumulator sentence (L127), and rewrites the opening of "`tl.dot` (the chunk model)" (L130), which still says 16 products per instruction for 16-bit operands. The rows of `DotF16`, `Exp2F32`, `Exp2F32NegInf`, `Log2F32`, `DivFull` and the totality rules are unchanged, and so is the mapping row for 16-bit `tl.dot`.
- The diffs of `lean/TVA`: only `Rules/Num/Dot8Chunk.lean` and `Lemmas/Dot8Chunk.lean` changed, with additions. `REVIEW.md` lists only `file:docs/rules/num.md` as changed for every re-opened item.
- L124 and L127–L133 in `docs/library-review/ISSUES.md`, and the round-26 author's notes: none changes a 16-bit fact, a constant this claim uses, or a lemma its specification uses.

## Not checked

- The new fp8 loss facts and their evidence, which this kernel does not cite.
