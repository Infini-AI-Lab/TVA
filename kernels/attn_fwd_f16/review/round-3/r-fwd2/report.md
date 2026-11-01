# Review of attn_fwd_f16, round 3, by r-fwd2

## Verdict

accept
The claim was re-proved on the restated `DotF16` (L126), and it holds up.
- `assumption:dot` is the fact the library lists.
- The host's inputs still meet `HeadOk` and Post's premise with `scoreMag`.
- The README's tolerance figures are right.
- The tolerance stays meaningful at the largest covered sizes on the kernel's real data.

The new `tolF` (about `N/64·2^-22`) is an absolute floor. For heads with small `v` it now outweighs the relative term, and the README's "always fixes the output within its own magnitude" no longer holds there. That is a wording finding only. L126 settles my F6.

## Findings

- **DOC** `doc:kernels/attn_fwd_f16/README.md` — The Specification's "the bound always fixes the output to well within its own magnitude" reads only `tolO`. With `tolF` about `N/64·2^-22`, the bound is below `attnAbs` only when `attnAbs` exceeds about `4.4·10^-4` at `N = 2^16` (`X = 2^10`). `HeadOk` allows normal-`float16` `v` near `2^-12`, where the bound exceeds the output. On the host's data, `tolF` is at most about 3% of `tolO·attnAbs`. Qualify the sentence. Non-blocking.
- **DOC** `doc:kernels/attn_fwd_f16/lean/AttnFwd/Model.lean` — `absA`'s docstring says `relDot16 64·5·2^-9` is "about `1.5·2^-23`". It is `330·2^-31 ≈ 1.29·2^-23`. The constant `2^-22` is sound. Non-blocking.
- **release F6** — settled by L126 (FIXED, round 25): `DotF16` is now `DotLiftAcc`, with a subnormal `float16` operand counted at `2^-14`. The claim cites it as listed, and `qk_lane` and `pv_lane`/`astep` use the lifted magnitudes.

## Accepted

All items.

## Checked

- The prover's response and notes (`review/round-2/prover/`), `REVIEW.md`, the new `claim.md` (§4 and §5), and the README's "What changed (library round 25)" and Specification sections.
- L126 in `ISSUES.md`, plus the diffs of `docs/rules/num.md`, `Rules/Num.lean` and `Rules/Num/Sm90.lean`.
  - Only `minNormal8`, `liftMag`, `DotLiftAcc`, its weakening lemmas, `DotF16` and `DotBF16` changed.
  - The facts behind the other eight assumption items, and their reviewer-table and listing rows, are unchanged.
- `assumption:dot`:
  - It is `Rules.Num.DotF16 P`, the listed fact for `float16` `tl.dot` into `float32`. `DotF16` covers every precision, so it applies to `cfg16`, which uses `tf32` with `maxImpreciseAcc 0`.
  - `work/check_lift.lean` prints `Assumptions`, `DotF16`, `liftMag`, `scoreMag`, `HeadOk`, `absA` and `omega0`, and the axioms of `main` (the standard three).
  - It also checks `liftMag .f16` on a subnormal and on a normal value.
- The changed Lean, in full:
  - `Model.lean`: `scoreMag`, `HeadOk`, `absA`, `omega0` and `Post`.
  - `Inner.lean`: `qk_lane`, `pv_lane` and `StInv`.
  - `Row.lean`: `rowCtx.bv` is now `scoreMag`; `score_le` uses `abs_mul_le_liftMag`.
  - `RowAcc.lean`: `omR`, `om0` and `absAr`.
  - `Step.lean`: `astep`'s `hlift1`, `hL`, `htau` and `hadd`, which use `liftMag p·liftMag v ≤ |pv| + 2^-14·|v| + 5·2^-15` with `p ≤ 2.3`, and a masked key's `p = 0`.
  - `Final.lean`: `TolFacts` and `final_o`.
  - `Program.lean`: `OutO`, `OutM` and `row_out`.
- Host coverage (`work/tol.py`):
  - By `sum_liftMag_mul_le`, Cauchy-Schwarz and a `χ²_128` Chernoff tail with a union bound over `2^22` rows (probability `2.6·10^-6`), the lift adds at most `0.0059` (test) and `0.0305` (benchmark) to `σ·Σ|q·k|`.
  - My round-1 bounds of `22.6` and `239.8` therefore stay below `23` and `240`, far under `2^10`.
  - The conditions item's digest is unchanged: `HeadOk` sits in `spec`.
- The tolerances, evaluated exactly with the new constants (`work/tol.out`):
  - Every README figure holds: `tolS` grows 3.2%; `tolF` is `5.1·10^-7`, `1.5·10^-5`, `6.4·10^-5`, `2.5·10^-4` and `3.3·10^-4` at the table's rows; `tolS·N` is at most `2.8·10^-3`.
  - `tolF` is below `float16`'s half-ulp at 1.
  - It is at most about 3% of `tolO·attnAbs` for the host's `v`.
- r-fwd's round-3 filing:
  - I reached F7's estimate independently: charged relative to `p`, the `v` lift would add about `66·2^-36 ≈ 9.6·10^-10`.
  - I agree it is non-blocking.

## Not checked

- Whether `DotLiftAcc` and its constants are true on the device: that is the library's matter (L126, evidence `num-dot-sub16-round25`). I trusted the library lemmas (`DotLift.liftMag_mul_le`, `sum_liftMag_mul_le`) as Lean checks them.
- The unchanged items (`model`, `map`, `conditions`, `assumption:faithful`) beyond confirming that their digests did not change.
