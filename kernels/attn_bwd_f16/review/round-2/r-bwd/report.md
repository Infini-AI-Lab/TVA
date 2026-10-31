# Review of attn_bwd_f16, round 2, by r-bwd

## Verdict

accept
The prover changed only `README.md`, and the gate and every item's digest are the ones I accepted in round 1. The rewrite settles both of my findings. The tolerance is now stated as a function of the inputs, with figures that match `Spec.lean` and my own simulation of the host's data. The layout narrowing now cites both host assertions and names the strides it leaves out. The new `sm_scale` bullet is correct. I release F3 and F4 and accept all items again.

## Findings

- **release F3** — the README now gives the tolerance as `r(A)` plus the floor and the `eY` term, with no fixed percentage and about 0.8 to 1% at the benchmark's `N_CTX = 2^14`
- **release F4** — the README now cites `do.is_contiguous()` and the equal strides and leaves out only a non-canonical `stride(0)` with `BATCH = 1` or `stride(1)` with `N_HEAD = 1`

## Accepted

All items.

## Checked

- The prover's response and notes for F1 to F4, and r-bwd2's round-1 filing (the same two points as mine).
- README, Specification, "How large the tolerances are", against `Spec.lean`. `r` is the linearisation of `p16Err` (`2^eX − 1 ≈ ln 2·eX`, `eX ≈ rH·aQK`), `u16` and `accB`. The floor is `eta16` per allowed pair, `16384·2^-25·0.8 ≈ 4·10^-4`. The `α` term is below `10^-33`. `eY` is the `rH·aVD` part of `dsErr`. The figure `0.36% + 0.05% + 0.43% ≈ 0.84%` is right for `A = 167`. The table's `A` and percentages agree with my round-1 runs (`round-1/r-bwd/work/premises.out`): `dV` about 0.8 to 1.06% and `dK` 0.7 to 1.1% at the benchmark's largest size, and large `dQ` ratios only where `dS` cancels.
- README, Narrowings: the `N_CTX` bullet (the accumulation part `≤ 0.43%`); the layout bullet, against lines 585–586 and 608–621 and `model`'s strides; and the new `sm_scale` bullet, against SEMANTICS.md, "Scalar arguments".
- That the gate (`e2e71723960bcd92`) and the item digests are unchanged since round 1, so my round-1 checks of the model, map, conditions, spec and assumptions still apply.

## Not checked

- The prover's table bit for bit: I did not rerun their sampling (every `N_CTX/32`-th row, attention mass `≥ 1/8`, sums over `j`); I compared it with my own runs and with the formula.
