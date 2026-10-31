# Review of attn_bwd_f16, round 2, by r-bwd2

## Verdict

accept
The prover answered both of my wording findings by rewriting only the README, and the rewrite now
matches the Lean. The tolerance is given as a function of the inputs, with the accumulation part
bounded by 0.43% at the largest covered size. The layout narrowing cites both host asserts and
names exactly the launches left out, and the Python-float `sm_scale` narrowing is listed. The gate
and the item digests are unchanged since I accepted every item in round 1, and I raise nothing new.

## Findings

- **release F1** — the Specification now states the tolerance's input dependence (`r` as a function of `A`, the per-pair floor, the `eY` term of `dK`, `dQ`) as `Spec.lean` gives it, and no longer claims a fixed percentage
- **release F2** — the Narrowings cite `do.is_contiguous()` and the equal strides, name the two non-canonical strides on a size-1 dimension that are left out, and list the Python-float `sm_scale`

## Accepted

All items.

## Checked

- The prover's response and notes for round 1. Only `README.md` changed; the gate digest `e2e71723960bcd92` and every item digest are the ones I accepted.
- The rewritten "How large the tolerances are" against `Spec.lean`:
  - `r ≈ ln 2·(HEAD_DIM + 2)·2^-22·A + 2^-11 + ((1 + ρ)^T − 1)` follows from `p16Err`, `pBr` and `accB`.
  - The floor: `η16` per allowed pair, `N_CTX·2^-25·mean|dO| ≈ 3.9·10^-4` at `N_CTX = 2^14` (python).
  - `dKTol` and `dQTol` carry `2^x·eY` with `eY ≈ (HEAD_DIM + 2)·2^-22·Σ|V·dO|` (`dsErr`).
  - The sum `0.36% + 0.05% + 0.43%` at `A = 167` (python), against my round-1 estimate (`work/tol.py`, about 0.81% at `A ≈ 153`) and r-bwd's round-1 simulated rows.
  - The cancelling-`dS` case of query row 0 with `CAUSAL`.
- The Narrowings against the source:
  - Lines 585–586 fix the canonical stride of every dimension of size > 1, leaving only `stride(0)` with `BATCH = 1` and `stride(1)` with `N_HEAD = 1`. There the kernel multiplies the stride by `bhid // H = 0` or `bhid % H = 0`.
  - An `int` `sm_scale` is a compile-time `1` or an `int32` (SEMANTICS.md, "Scalar arguments").
  - The `N_CTX` bullet bounds only the accumulation part.
- r-bwd's round-1 filing, report and work, after round 1 closed.

## Not checked

- The README table's exact figures: the prover's evaluation script is not in the kernel folder. I checked them only for size, against my formula estimate and r-bwd's simulation.
