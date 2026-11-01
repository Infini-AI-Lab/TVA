# Review of attn_bwd_f16, round 3, by r-bwd

## Verdict

accept

The claim is re-proved on library round 25's restated `DotF16` (`DotLiftAcc`, relative to lifted magnitudes), and that settles my F6, which I release. The re-opened items are right at their new digests:
- `assumption:dot` cites the fact as listed;
- the spec counts each `tl.dot` operand by `lm`, and the accumulated `P` and `dS` by `p16L` and `ds16L`;
- the README's inline script follows the new formulas, and running it as printed reproduces every figure.

One non-blocking DOC on the wording of the README's "What changed" section.

## Findings

- **DOC** `doc:kernels/attn_bwd_f16/README.md` — non-blocking: "What changed (library round 25)" overstates what stayed the same.
  - The per-pair errors and `HeadV.Ok` read the lifted `aQK` and `aVD`.
  - `p16L` and `ds16L` lift the bounds, not the values.
  - The list of moved figures leaves out the benchmark's `HEAD_DIM = 128` causal `dKTol` 90th percentile (1.50 to 1.51%) and the causal `dV` maximum (0.91 to 0.92%).
- **release F6** — settled by L126 (library round 25). The library restated `DotF16` and `DotBF16` as `DotLiftAcc` on evidence that the old form failed, and the claim is re-proved on it.

## Accepted

All items.

## Checked

- **`assumption:dot`.**
  - It is `Rules.Num.DotF16 P`, as num.md's row in "Facts the library states" lists it: `DotLiftAcc` with `relDot16` and `absDot16`, a subnormal `float16` operand at `2^-14`.
  - `work/spec3.out` prints it, `DotLiftAcc`, `liftMag` and `minNormal8` with full names.
  - The cone changes match the restatement.
- **`assumption:dot_total`, `exp2`, `exp2_total`.** They re-opened only through whole-file digests:
  - no changed line of the Num rule files or num.md touches `DotTotal`, `MathTotal` or `Exp2F32`;
  - `DotLoop.stepRel` and `stepAbs` (`rhoA`, `alphaA`) are unchanged.
- **`spec`.**
  - `lm = liftMag .f16`. `lift16` is a valid bound, since a `float16` below `2^-24` is `0` (`f16_small_zero`).
  - `aQK` and `aVD` are sums of `lm·lm`.
  - `dVTol`, `dKAcc` and `dQAcc` accumulate over `p16L·lm dO`, `ds16L·lm Q` and `ds16L·lm K`.
  - `pBr`, `pE`, `p16Err` and `dsErr` changed digest with no text change: their literal `2`'s auxiliary proof is now `lift16._proof_1`.
  - The tolerance is still explicit, computable and meaningful.
- **The README's inline script.**
  - It matches Spec.lean line by line.
  - Run as printed (numpy 2.3.5, `work/readme_script.out`), it reproduces the table, the counted maxima and the `dQ` worst rows.
  - Every move from round 2 is at most 0.07 percentage points.
- **Axioms.** `main`, `consistent` and `satisfiable` use only `propext`, `Classical.choice` and `Quot.sound`.
- **The prover's responses.** The notes, `REVIEW.md` and L126.

## Not checked

- The library's evidence for the restated fact (`evidence/num-dot-sub16-round25`). Kernel reviewers trust the library.
- The other changed proof files line by line. The gate checks them, and `main`'s statement is what the items fix.
