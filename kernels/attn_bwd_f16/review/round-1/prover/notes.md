# Prover's notes on attn_bwd_f16, round 1

## Changes

- `README.md` only; no Lean, map or manifest change.
- Specification: the tolerance is restated with its input dependence:
  `r ≈ ln 2·(HEAD_DIM + 2)·2^-22·A + 2^-11 + ((1 + ρ)^T − 1)`, plus the floor `≈ N_CTX·2^-25·mean|dO|`,
  plus the `eY` term of `dK` and `dQ`. The section also has a table of figures I computed by evaluating
  `Spec.lean`'s formulas on one head of the test's and the benchmark's inputs.
- Narrowings:
  - the `N_CTX` bullet bounds only the accumulation part (`≤ 0.43%`);
  - the layout bullet cites `do.is_contiguous()` and the equal strides, and names the launches left out:
    a non-canonical `stride(0)` with `BATCH = 1`, or `stride(1)` with `N_HEAD = 1`;
  - a new bullet: only a Python `float` `sm_scale` is covered.

## Responses

- **F1** CLARIFIED — README, Specification: the summary sentence is replaced by the tolerance with its input dependence. dVTol is about r*sum_m P|dO| plus an absolute floor, with r = ln2*(HEAD_DIM+2)*2^-22*A + 2^-11 + ((1+rho)^T - 1). Here A is the P|dO|-weighted mean of sum_c|K*Q| over the row; the accumulation part is at most 0.43% at T <= 516. The floor is about N_CTX*2^-25*mean|dO|. dK and dQ also carry eY, the error of the computed dP - Delta, which does not shrink when dS cancels. I computed the figures myself by evaluating Spec.lean's formulas on one head of the host's inputs (seed 20, sampled rows, attention mass >= 1/8 for dV/dK), as median / 90th percentile. Test, N_CTX = 4096: 0.13-0.23% for all three outputs. Benchmark, N_CTX = 2^14, HEAD_DIM = 128 (A about 167): dV 0.79-0.90 / 0.87-0.95%, dK 0.85-0.96 / 1.12-1.50%, dQ 0.95-1.0 / 4.8-16%. The dQ tail comes from dS cancelling, where the absolute tolerances are 2.6e-4 to 9.3e-3. The Narrowings bullet on N_CTX now says only the accumulation part is bounded by 0.43%. No Lean change.
- **F2** CLARIFIED — README, Narrowings. (1) The layout bullet now cites both do.is_contiguous() (line 585) and the equal strides (line 586). With both, every dimension of size greater than 1 has its canonical stride. Since N_CTX >= 128 and HEAD_DIM >= 16, the claim leaves out exactly the launches with a non-canonical stride(0) when BATCH = 1 or stride(1) when N_HEAD = 1, because the model passes the canonical strides. (2) A new bullet: only a Python float sm_scale is covered (Inst.sm_scale is the double the host passes; the test passes 0.5 and the benchmark 1.3). An int sm_scale gives another parameter type (the constant 1, or int32) and is not covered. No Lean change.
- **F3** CLARIFIED — Same revision as F1. The README no longer gives a fixed percentage. It gives r as a function of A, the P|dO|-weighted sum_c|K*Q|, which HeadV.Ok bounds only through xE + eX <= 15, plus the floor and the eY term that dominate on rows with little attention or a cancelling dS. My figures are close to yours: at the benchmark's N_CTX = 2^14, HEAD_DIM = 128, the median is about 0.8-1.0% for dV and dK and the 90th percentile for dQ is 4.8-16%; the formula reproduces about 0.84%. The Narrowings sentence 'the tolerance stays below 0.5%' now reads 'the accumulation part ... stays at most 0.43%'. No Lean change.
- **F4** CLARIFIED — Same revision as F2 (1). The README now says that do.is_contiguous() and the equal strides fix the canonical strides on every dimension of size > 1. The only launches the claim leaves out are those where a size-1 BATCH or N_HEAD dimension has a non-canonical stride. The kernel would multiply such a stride by 0, but those launches are not instances because the model passes the canonical strides. No Lean change.

## Library requests

None.

## Gate

PASS, digest `e2e71723960bcd92`
