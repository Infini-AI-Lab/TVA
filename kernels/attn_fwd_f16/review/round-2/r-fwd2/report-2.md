# Review of attn_fwd_f16, round 2, by r-fwd2

## Verdict

accept
Library round 24 changed `docs/rules/num.md` and `Sm90.lean` only for the fp8 `tl.dot` facts (withdrawn) and the new fp8 chunk facts. Nothing this kernel cites changed: the facts, their listings, the constants and the page's statements about them are as before. So I re-accept the re-opened items. I add one non-blocking library report: L125's subnormal-window mechanism may also affect `DotF16`.

## Findings

- **LIBRARY_REPORT** `decl:TVA.Rules.Num.DotF16` — L125's subnormal-window mechanism may also break `DotF16` on `wgmma`, and `evidence/num-dot` has no such input; this kernel's `p.to(float16)` produces subnormals.

## Accepted

All items.

## Checked

- The diff of `docs/rules/num.md` and `lean/TVA/Rules/Num/Sm90.lean`, and L124 and L125 in `ISSUES.md`. I also read the coordinator's note in `REVIEW.md`.
- `DotF16`, `relDot16` and `absDot16` are unchanged in Lean and in "Facts the library states". So are `DotTotal`, `Exp2F32`, `Exp2F32NegInf`, `MathTotal`, `Log2F32`, `DivFull` and `DivTotal`.
- The page's statements on these facts are unchanged: the 16-bit chunk bullet, the approximate functions, Division, and the reviewer's table rows.
- The spec's tolerances depend on `relDot16` only.

## Not checked

- Whether `DotF16` holds with a subnormal operand setting the window: that is the library's matter (the report above).
