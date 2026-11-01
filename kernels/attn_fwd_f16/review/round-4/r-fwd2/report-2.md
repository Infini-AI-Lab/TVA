# Review of attn_fwd_f16, round 4, by r-fwd2

## Verdict

accept
Library round 26 re-opened `spec` and nine assumption items, but only through `docs/rules/num.md`. None of the round-26 changes touch what this kernel cites or what its specification uses: they add new fp8 loss facts and fix wording. I re-accept every item and raise nothing.

## Findings

None.

## Accepted

All items.

## Checked

- `REVIEW.md`'s list of what changed per item: `file:docs/rules/num.md` only. The kernel's files are unchanged, at gate `11e67701bc53c248`.
- The diff of `docs/rules/num.md` and `lean/TVA`:
  - It adds `DotChunkLoss` and the four `DotE*ChunkLoss` facts, with their rows, paragraphs and satisfiability lines, in `Rules/Num/Dot8Chunk.lean` and `Lemmas/Dot8Chunk.lean`.
  - In those files it only adds definitions and lemmas; the one deleted line is a docstring.
  - In `num.md` it fixes wording under L127 and L130. L130's new opening still says 16 products per instruction for 16-bit operands on both paths.
- Unchanged in Lean, in the listings and in the constants:
  - `DotF16` (`DotLiftAcc`, `liftMag`, `relDot16`, `absDot16`), `DotTotal`, `Exp2F32`, `Exp2F32NegInf`, `MathTotal`, `Log2F32`, `DivFull` and `DivTotal`;
  - the reviewer-table rows for these operations;
  - the 16-bit chunk-model bullet.
- L124 and L127–L133 in `ISSUES.md`, and the round-26 author's notes: no kernel was edited, and every existing fact and lemma is unchanged.

## Not checked

- The new fp8 facts themselves: this kernel does not cite them.
