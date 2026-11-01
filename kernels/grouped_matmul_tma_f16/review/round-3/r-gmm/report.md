# Review of grouped_matmul_tma_f16, round 3, by r-gmm

## Verdict

accept
The prover re-proved the claim on `DotF16` as library round 25 restates it (L126). The tolerance stays explicit and computable: the sum of the products' lifted magnitudes (`dotLift`) replaces the sum of their magnitudes. This equals the old value on zero and normal `float16` operands, so every figure the README gives for them still holds. With a subnormal operand it is larger, by no more than the device's error under the restated fact allows. The cited fact is the one num.md lists, and the other items are unchanged. L126 settles my F5.

## Findings

- **release F5** — L126 restates DotF16 as DotLiftAcc (a subnormal float16 operand counted at 2^-14), and the claim is re-proved on it.

## Accepted

All items.

## Checked

- The prover's response and notes for round 2, `REVIEW.md`'s list of what changed, L126 in `docs/library-review/ISSUES.md`, and the diff of `Defs.lean`, `Basic.lean` and the README, with its section "What changed (library round 25)".
- `assumption:dot`: the field is still `Rules.Num.DotF16 P`. That is now `DotLiftAcc P ⟨f16, f16, f32, p, 0⟩ relDot16 absDot16` for every precision (work/lift.lean), the fact num.md's kind table and "Facts the library states" list for an fp16 `tl.dot` into `float32`. `relDot16` and `absDot16` are unchanged. The docstring states the lifted bound.
- `spec`:
  - `OutSpec`'s tolerance is `tol (K g) (dotLift …)`, where `dotLift` sums `TVA.Rules.Num.liftMag .f16` of both operands, the magnitude the fact measures by;
  - on zero and normal values `liftMag` is `|q|` (checked in Lean), so for the host's normal inputs the tolerance and the README's figures (0.204% and 0.253% at K = 2^13, thresholds 13633 and 15617; my round-1 work/tol.py) hold, and a `bfloat16`-rounded result is still rejected;
  - a subnormal operand counts at `2^-14`, at most `2^10` times its magnitude (the smallest subnormal `2^-24`, checked);
  - the padded lanes' `+0.0` lifts to `0`.
- `assumption:dot_total`: `DotTotal` and `DotAcc` are unchanged in `Rules/Num.lean`; the item re-opened only because the files changed.
- The model, conditions, map and `faithful` item digests are unchanged. The axioms are `propext`, `Classical.choice` and `Quot.sound`.

## Not checked

- Whether the restated `DotF16` is true of the device (`evidence/num-dot-sub16-round25`): the library's matter, trusted.
