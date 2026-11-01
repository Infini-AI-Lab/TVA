# Review of attn_bwd_f16, round 4, by r-bwd2

## Verdict

accept
Only the README changed, and the gate is still `b28f50727b67c8a8`. The section "What changed
(library round 25)" now says correctly that `lm` is the old magnitude on zero and normal values,
and that `p16L` and `ds16L` lift the bounds, not the values. Its Figures bullet lists every moved
figure, and the worst `dQ` absolute tolerance reads `9.2·10^-3`. I release F8 and accept all
items.

## Findings

- **release F8** — the change log now describes `lm`, `p16L` and `ds16L` correctly, lists every moved figure (HEAD_DIM 128 included), and gives `9.2·10^-3`.

## Accepted

All items.

## Checked

- The section "What changed (library round 25)" against `Spec.lean`:
  - `lm` against `liftMag`;
  - `p16L` and `ds16L` against `lift16` (`0` below `2^-24`, else `max B 2^-14`);
  - the reading of the lifted `aQK` and `aVD` by `p16Err`, `ds16Err` and `HeadV.Ok`, through `eX`
    and `eY`.
- The bound "each subnormal operand adds at most `2^-14` times its partner's `lm`". It holds:
  `lm x·lm y − |x·y| ≤ 2^-14·lm y + 2^-14·lm x`.
- The moved figures it lists (four table cells, the causal `dV` maximum, four `dQ` 90th
  percentiles). I checked them against my round-2 and round-3 runs of the README's script.
- The table and the `9.2·10^-3` in the text after it.
- The prover's CLARIFIED responses to F7 and F8.

## Not checked

- New samples: none were taken, as the coordinator asked.
