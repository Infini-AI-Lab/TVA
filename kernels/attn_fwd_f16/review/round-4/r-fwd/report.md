# Review of attn_fwd_f16, round 4, by r-fwd

## Verdict

accept
The prover tightened `tolF` as F7 suggested. The `p·v` lift of a subnormal `v` is now bounded through the block's weights (`kapA = 2^-29` per unit of weight), and `absA` keeps only the per-key remainder (`2^-37`). `tolF` is now `3.2·10^-8` at the test's sizes, `3.4·10^-8` at the benchmark's and at most `4.2·10^-8` at the largest covered sizes. It no longer grows with `N` and is mostly the output's `float16` rounding near zero. Only `spec` re-opened; `Post`'s form, `tolO`, `tolS` and `tolM` are unchanged. I release F7 and accept every item.

## Findings

- **release F7** — the `p·v` lift is now bounded through `Σp = l_i` (`kapA`), and `tolF` falls to at most `4.2·10^-8`, which my exact re-evaluation confirms.

## Accepted

All items.

## Checked

- The prover's response and notes (F7, and r-fwd2's F8 and F9), and the README's round-3 note, `tolF` paragraph and table.
- The changed definitions with full names (`work/names.lean`): `absA = 2^-37`, `kapA = 2^-29`, `absQ0` with its new `facc·kapA/denQ` term, and `tolF`. No new instance, notation or option appears in the kernel's Lean. `main`'s axioms are the standard three.
- `kapA`'s derivation: `66·2^-36·8/7 ≈ 0.59·2^-29`, given `psi16 ≤ 1/7` at the largest covered `X` and `D`. `absA`'s docstring figure (F9): `66·2^-22·64·(2^-14·(2^-25 + 2^-125) + 2^-28) ≈ 0.516·2^-37`, below `2^-37`.
- The tolerances evaluated exactly with the new constants at the test's, the benchmark's and the largest covered sizes, and the README's table (`work/tol.py`, `work/tol.out`). Every figure agrees, and F8's threshold `tolF/(1 − tolO) ≈ 5.5·10^-8` holds.

## Not checked

- The changed proofs line by line (`AInv`, `acc_step'`, `ratio_err`, `tol_facts`): Lean checks them.
