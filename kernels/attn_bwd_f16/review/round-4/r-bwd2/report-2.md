# Review of attn_bwd_f16, round 4, by r-bwd2

## Verdict

accept
Library round 26 re-opened `spec` and the four `Num` assumptions, only through `docs/rules/num.md`.
Nothing they rely on changed. `DotF16`, `DotTotal`, `Exp2F32` and `MathTotal`, the constants
`relDot16` and `absDot16`, the lifted-magnitude definitions, and the lemmas the specification uses
are unchanged, and so are their rows in num.md. The round adds fp8 loss facts, which this kernel
does not cite, and fixes wording. I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- L124 and L127–L133 in `docs/library-review/ISSUES.md`, the round-26 author notes, and the stale
  reasons in `REVIEW.md`. Every re-opened item names only `file:docs/rules/num.md`.
- The Lean diff against the round-25 commit. It touches only `Rules/Num/Dot8Chunk.lean` and
  `Lemmas/Dot8Chunk.lean`, adding the fp8 loss facts and their lemmas. These are unchanged:
  - `DotF16`, `relDot16` and `absDot16`;
  - `DotLiftAcc`, `liftMag` and `minNormal8`;
  - `Exp2F32`, `DotTotal` and `MathTotal`;
  - `DotLift.dotAdd_step`, `stepRel` and `stepAbs`.
- The diff of `docs/rules/num.md`.
  - Unchanged: the rows in "Facts the library states" for `DotF16`/`DotBF16`, `Exp2F32` and
    `MathTotal`/`DotTotal`, and the reviewer table's row for `float16` dots.
  - New or changed: the `DotChunkLoss` row and the fp8 loss rows and sections; the listing date;
    the fp8 reviewer row; the fp8 chunk wording (L127); the satisfiability list.
  - The chunk model's opening is reworded (L130). Its 16-bit statement, 16 products per
    instruction on both paths, means what it meant.

## Not checked

- The new fp8 facts and their evidence. The kernel does not cite them.
