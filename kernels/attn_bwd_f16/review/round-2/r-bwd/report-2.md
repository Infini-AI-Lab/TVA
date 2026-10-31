# Review of attn_bwd_f16, round 2, by r-bwd

## Verdict

accept
The prover's two README-only fixes are right, and the gate and every item's digest are unchanged. The layout bullet now handles empty launches correctly. The tolerance section now says how each figure was computed and gives the script inline. The script follows `Spec.lean`'s formulas, and running it as printed reproduces the table. I raise nothing and accept all items again.

## Findings

None.

## Accepted

All items.

## Checked

- The layout bullet. A non-empty launch has `N_CTX ≥ 128` and `HEAD_DIM ≥ 16`, so only `stride(0)` with `BATCH = 1` or `stride(1)` with `N_HEAD = 1` can be non-canonical. An empty launch (`N_CTX = 0`, which `n_mod` allows, or `BATCH = 0` or `N_HEAD = 0`) is covered only with the canonical strides, since `is_contiguous` holds for any strides of an empty tensor, and its grid has no program.
- The inline script against `Spec.lean`, line by line:
  - `eSub`, `xE`/`eX`, `yE`/`eY`, `yB`, `pBr`, `pE`, `dsE`, `p16Err`, `p16B`, `dsErr`, `dsB`, `ds16Err` and `ds16B`;
  - `stepsKV` and `stepsQ`, `accB`, `dVTol`, `dKTol` and `dQTol`;
  - `Allowed` as `n ≤ m`, and the forward's causal mask.
- The script's inputs against the description: `arg_k` as line 597 builds it, the draws and their order, the sampled rows, the attention-mass filter and the weighting of `A`.
- The figures: I ran the script as printed (`work/readme_script.out`). Every median and 90th percentile of the table, `A`, the largest counted ratios (1.04 / 1.19%; 0.91 / 5.2%), and the absolute range of the worst `dQ` rows (`2.6·10^-4` to about `9.2·10^-3`) come out as the README states.

## Not checked

- Anything outside the two changed README sections; the rest is unchanged since my round-1 review and first round-2 filing.
