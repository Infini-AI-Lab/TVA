# Review of attn_fwd_f16, round 2, by r-fwd

## Verdict

accept
Library round 24 (L124, L125) re-opened the assumption items and `spec` only because `docs/rules/num.md` and `lean/TVA/Rules/Num/Sm90.lean` are hashed whole. The changes withdraw the fp8 `tl.dot` facts and add the fp8 chunk facts, which this kernel does not cite. Every fact, constant and statement the kernel's items rely on is unchanged, so my earlier judgment stands, and I re-accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- The diff of `lean/TVA/Rules/Num/Sm90.lean`: it removes only `chunks8`, `relDot8`, `absDot8`, `DotF8` and `DotE4m3` … `DotE5m2E4m3`. `relDot16`, `absDot16`, `DotF16`, `Exp2F32`, `Exp2F32NegInf` and `Log2F32` are as before, and so are `DivFull`, `DotTotal`, `MathTotal` and `DivTotal` in `lean/TVA/Rules/Num.lean`, which did not change.
- The diff of `docs/rules/num.md`: the rows of `DotF16`, `Exp2F32`, `Exp2F32NegInf`, `Log2F32`, `DivFull` and the totality rules in the first table of "Facts the library states" are unchanged. So are the kernel-reviewer mapping row for 16-bit `tl.dot` and the 16-bit bullet of the chunk-model section. Only fp8 rows, sections and text changed.
- The diff of `lean/TVA/Lemmas/Sm90.lean`: it removes only the fp8 lemmas, not the `Platform.sm90ref` facts that `consistent` uses.
- L124 and L125 in `docs/library-review/ISSUES.md`, and the coordinator's note and STALE marks in `REVIEW.md`. No kernel file changed.

## Not checked

- The new fp8 chunk facts and their evidence, which this kernel does not cite.
