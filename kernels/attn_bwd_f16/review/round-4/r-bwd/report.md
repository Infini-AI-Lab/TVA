# Review of attn_bwd_f16, round 4, by r-bwd

## Verdict

accept

Only the README changed, and its section "What changed (library round 25)" now answers F7 in full. I release F7 and accept all items again at the unchanged gate.

## Findings

- **release F7** — the section now says that the per-pair errors and `HeadV.Ok` read the lifted `aQK` and `aVD`, and that `p16L` and `ds16L` lift the bounds, not the values. It lists every moved figure.

## Accepted

All items.

## Checked

- The revised section against F7's three points and F8's three, and against my round-2 and round-3 outputs of the README script.
  - Moved table cells: the four listed are the only ones.
  - Causal `dV` maximum: `0.91` to `0.92%`.
  - `dQTol` 90th percentiles, which moved below the table's rounding: `2.06` to `2.13`, `3.27` to `3.30`, `4.75` to `4.76` and `15.92` to `15.93%`.
- The worst `dQ` rows' tolerance now reads `9.2·10^-3`. My round-3 four-digit run gives `0.00925`, and r-bwd2's gives `0.0092498`.
- The gate and the item digests are unchanged.

## Not checked

- No new samples (as asked). The figures were reproduced in round 3.
