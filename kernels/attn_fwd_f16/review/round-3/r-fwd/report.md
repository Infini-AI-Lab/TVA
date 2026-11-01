# Review of attn_fwd_f16, round 3, by r-fwd

## Verdict

accept
The re-proof on the restated `DotF16` (library round 25, L126) keeps the claim meaningful. `assumption:dot` cites the library's fact as listed. The other re-opened assumptions rest on facts and constants that did not change. The score magnitude in `HeadOk` and in `Post`'s premise is now the lifted `scoreMag`, which the host's inputs still meet with a wide margin (below 23 and 240 against `2^10`). `tolO` and `tolM` are unchanged. `tolF` grows to `1.5·10^-5` at the test's `N`, `6.4·10^-5` at the benchmark's and `3.3·10^-4` at `N = 2^16`. On the host's unit-scale data this is far below the relative term, so the tolerance stays meaningful at every covered size. I accept every item. My one finding is a non-blocking question: `tolF`'s floor comes from a loose step of the analysis.

## Findings

- **QUESTION** `item:spec` — `tolF`'s `N/64·2^-22` floor bounds each block's `p·v` lift per key with `p ≤ 2.3`, not through `Σ p = l_i`; normalised it is about `5·10^-9`. Tighten it, or say in the README that the floor comes from the analysis.

## Accepted

All items.

## Checked

- L126, the coordinator's notes, the prover's response and notes, and the README's section "What changed (library round 25)".
- `lean/TVA/Rules/Num.lean` (`DotLiftAcc`, `liftMag`, `minNormal8`) and the restated `DotF16` in `Sm90.lean`: `relDot16` and `absDot16` are unchanged, and the fact is still `∀ p`, `maxImpreciseAcc = 0`, as `docs/rules/num.md` lists it. Its mapping row and its row in "Facts the library states" name it.
- The diffs of the library files that re-opened the other assumption items. `Exp2F32`, `Exp2F32NegInf`, `Log2F32`, `DivFull`, `DotTotal`, `MathTotal` and `DivTotal` are unchanged, and so are their rows in `num.md`.
- The changed `Model.lean` (`scoreMag`, `HeadOk`, `absA`, `omega0`, `Post`'s premise) with full names (`work/names.lean`). `liftMag` is `TVA.Rules.Num.liftMag .f16`. The model, the launch and the conditions' digests are unchanged. No new instance, notation or option appears in the kernel's Lean. `main`'s axioms are the standard three.
- `HeadOk`'s coverage of the host's data. The lift adds at most `σ·(2^-14·(Σ|q| + Σ|k|) + 128·2^-28)`, which is `0.006` (test) and `0.031` (benchmark) with `Σ|q| ≤ √128·|q|` and the χ² tail. The result stays below 23 and 240, except with probability below `4·10^-6`.
- The tolerances evaluated exactly with the new constants, at the test's, the benchmark's and the largest covered sizes, and the README's table (`work/tol.py`, `work/tol.out`): every figure agrees.
- How the `p·v` lift enters `absA`, `omega0` and `tolF` (the finding).

## Not checked

- The changed proofs line by line (Lean checks them).
- The library's new evidence for `DotF16` (`evidence/num-dot-sub16-round25`), which kernel review trusts.
