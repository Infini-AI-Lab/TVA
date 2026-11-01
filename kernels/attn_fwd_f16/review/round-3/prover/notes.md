# Prover's notes on attn_fwd_f16, round 3

## Changes

- `item:spec` (Model): `absA` is now `1/2^37` (was `1/2^22`); there is a new `kapA = 1/2^29`;
  `absQ0` gains `facc·kapA/denQ`. `tolF` falls to `3.2·10^-8` at the test's sizes, `3.4·10^-8` at
  the benchmark's and at most `4.2·10^-8` at `N = 2^16`, `X = 2^10` (was `1.5·10^-5`,
  `6.4·10^-5`, `3.3·10^-4`). `tolO`, `tolS` and `tolM` are unchanged.
- Proof: `AInv` has a term `kapAr·Σ wt`. `astep` bounds a subnormal `v`'s lift through
  `|p| ≤ (1 + psi16r)·wt + omR`. The new `acc_step'` keeps the old absolute part scaled by
  `alpha`'s `λ`. `ratio_err` takes `κ·W`. `final_o` and `tol_facts` follow.
- README: the round-25 section (with a round-3 note), the tolerance paragraph, the table, the
  `tolF` prose with its threshold, the proof outline and the narrowing on the tolerance's form.

## Responses

- **F7** REVISED: the `p·v` lift of a subnormal `v` is now bounded through `Σp = l_i`, as
  suggested (`item:spec`).
- **F8** CLARIFIED: the README sentence is qualified. The remaining threshold, `5.5·10^-8`, lies
  below `float16`'s smallest subnormal, where the bound still fixes the output to within one
  subnormal step.
- **F9** CLARIFIED: `absA`'s docstring gives its new value, about `0.52·2^-37`.

## Library requests

None.

## Gate

PASS, digest `7c52bcaa1a441cb2`
