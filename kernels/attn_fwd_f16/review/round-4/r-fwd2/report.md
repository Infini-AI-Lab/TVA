# Review of attn_fwd_f16, round 4, by r-fwd2

## Verdict

accept
The prover's F7 revision bounds a subnormal `v`'s lift in the `p·v` `tl.dot` through the block's weights (`kapA = 2^-29` per unit of weight) instead of `p ≤ 2.3` on every key. `absA` drops to `2^-37`, and `tolF` falls to at most `4.2·10^-8` at every covered size. `tolO`, `tolS` and `tolM` are unchanged. The README's figures and the docstrings' constants check. My F8 and F9 are fixed, and I raise nothing new.

## Findings

- **release F8** — the README now says "up to the two absolute terms". It gives the threshold `tolF/(1 − tolO) ≤ 5.54·10^-8`, below `2^-24`, where the bound holds the output within one subnormal step. Both check.
- **release F9** — `absA` is now `2^-37`. Its docstring's "about `0.52·2^-37`" is `66·2^-44 ≈ 0.516·2^-37`, and "at most `2^-22 + 2^-32`" per block holds.

## Accepted

All items.

## Checked

- The prover's round-3 response and notes, `REVIEW.md` (only `spec` stale: `absA`, `kapA`, `absQ0`), and the README's "What changed" (with the round-3 note), its tolerance paragraph, table and `tolF` prose, and the narrowing on the tolerance's form.
- `Model.lean`'s new `absA`, `kapA`, `absQ0` and `tolF`, and their docstrings. `work/check_spec.lean` prints them with full names, along with `kapAr`, `absAr` and `AInv`. It checks the `absA` and `kapA` docstring figures. It shows that `main` depends on the standard three axioms only.
- The proof's new pieces, read for meaning:
  - In `astep`, `AInv` gains `kapAr·psum wt`, and `hk` bounds `66·2^-22·(1 + psi16r)/2^14` by `kapAr`.
  - In `absQ0`, the `kapA` term is divided by `denQ` only, because `Σwt` cancels against `l_i ≥ (1 − betaL)·Σwt`.
  - Lean checks the proofs; the gate passes.
- I evaluated the tolerances exactly (`work/tol.py`, `work/tol.out`):
  - `tolF` is `3.17·10^-8`, `3.21·10^-8`, `3.37·10^-8`, `3.93·10^-8` and `4.23·10^-8` at the README's rows (round 3: up to `3.3·10^-4`).
  - The largest threshold where `tolF` outweighs `tolO·attnAbs` is `5.54·10^-8 < 2^-24`.
  - `tolO`, `tolS` and `tolM` are unchanged from round 3.
- r-fwd's F7 (REVISED): the change does what F7 asked.

## Not checked

- The items whose digests did not change (`model`, `map`, `conditions`, every assumption) beyond confirming that.
