# Prover's notes on attn_fwd_e5m2, round 1

## Changes

- Specification: `Post` adds a second bound for `o`, `|c − attnRef| ≤ tolV d init bh n e K`, for every `K` bounding `|vEff_j − attnRef|` over the row's keys (Model: `magCp`, `magNext`, `magA`, `magD`, `tolV`). It is a magnitude bound over the computed `p`, with no `pE` in it. `Post`'s docstring says that neither tolerance need be below `|attnRef|`.
- Proof: a new module `Mag` (`lminB`, `lmin_real`, `p_big`, `lane_pq`, `lane_E`, `estep_real`, `final_v`, `ref_le`, `magA_nonneg`, `magD_nonneg`). `mstep_plain` and `mstep_masked` also give that the maximum is kept or some lane's exponent is within `2^-11` of `0`, and `astep` also returns the dot's error. `RowSt` carries `l_i ≥ lminB t` and `|acc − r·l_i| ≤ magA·l_i + magD` (`mag_tail`), and `OutO` and `row_out` prove the new clause. The model, the map, the conditions and the assumptions are unchanged.
- Write-up: four places now say first that only the first `⌈Z·H/HEAD_DIM⌉` heads read `v` and every other head outputs `±0`: `kernel.json`'s title, the README's opening and its kernel paragraph, and the root docstring. The README's Specification describes `tolV`. Its table gives `tolO`, `tolV` and `tolM`, with the benchmark at `D = 128`, and a new section, "Where `Post` does not tell `o` from `0`", cites L124. The Narrowings are rewritten without "meaningful on the test's shapes and data". The Proof section and the module table are updated.

## Responses

- **F1** REVISED — `Post` adds `tolV` (benchmark `D = 128`: about 7, was about `10^4`), and the README states plainly where `Post` does not tell `o` from `0`; the library's part is L124.
- **F2** REFERRED — library issue L124.
- **F3** CLARIFIED — the title, the README's opening and the root docstring say first that only `⌈Z·H/HEAD_DIM⌉` heads read `v`, the others output `±0`.
- **F4** REVISED — the magnitude bound is proved and part of `Post` (`tolV`: 6.5 and 6.7 at the benchmark, `0.31·|attnRef|` on a one-key row); the rest is stated as a limitation, citing L124.
- **F5** CLARIFIED — as F3; "meaningful on the test's data" is gone; the benchmark's `D = 128` row is in the table.

## Library requests

- As L124 (F2): a form of `Num.DotE5m2` that does not charge `relDot8` on `|acc|` at every call, for example one relative to each chunk's largest product. Two things would follow: the `(1 + relDot8 64)^nblk` growth of both tolerances would stop, and the scores' factor `pE` would shrink, which is what would let `Post` tell `o` from `0` on the benchmark's data.

## Gate

PASS, digest `53b679fa0085b99d`
