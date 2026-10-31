# Review of attn_bwd_f16, round 2, by r-bwd2

## Verdict

accept
The README now says how each figure of the tolerance table was computed and includes the script.
I ran the script, and every figure matches its output. The script evaluates `Spec.lean`'s own
formulas. The layout bullet now also covers empty launches, and says so correctly. Only the README
changed, and the gate and the item digests are unchanged, so I accept every item again. I raise
nothing.

## Findings

None.

## Accepted

All items.

## Checked

- The README's inline script, run as written (numpy 2.3.5; `work/readme_script.out`):
  - every table entry matches the script's output: `A` (7, 15, 82, 166–168) and the median / 90th percentile of `dVTol`, `dKTol` and `dQTol` for the test and the benchmark, non-causal and causal;
  - the largest counted ratios (`1.04%` / `1.19%` non-causal, `0.91%` / `5.2%` causal) match;
  - the `dQ` worst rows' absolute tolerances (`2.6·10^-4` to `9.2·10^-3`) match.
- The script against `Spec.lean`, term by term:
  - the per-pair quantities: `eSub`, `xE`, `eX`, `yE`, `eY`, `yB`, `pBr`, `pE`, `dsE`;
  - the error bounds: `p16Err`, `p16B`, `dsErr`, `dsB`, `ds16Err`, `ds16B`;
  - the tolerances: `stepsKV`, `stepsQ`, `accB`, `dVTol`, `dKS`/`dKAcc`/`dKTol`, `dQS`/`dQAcc`/`dQTol`;
  - `Allowed`, `smF` and `ln2F`.
- The prose against the script: the draws, `arg_k` as line 597 computes it, the stand-ins for `M`, `o` and `delta`, the sampled rows, the attention-mass filter, and the percentiles.
- The layout bullet's new sentence: an empty tensor is contiguous for any strides, and such a launch has a grid of `0` programs.

## Not checked

- The forward's and the preprocess's own `float32` roundings of `M` and `delta`. The script replaces them with an exact computation, as the README says.
