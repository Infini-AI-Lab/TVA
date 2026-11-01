# Review of attn_bwd_f16, round 3, by r-bwd2

## Verdict

accept
Library round 25 restated `Rules.Num.DotF16` as a `DotLiftAcc` fact (L126), and the claim is
re-proved on it. Every re-opened item is right at its new digest, and I re-accept every item. I
release F5, which L126 settles. I raise one new DOC finding, on the README's change log. It does
not block.

The DOC finding:

- In "What changed (library round 25)", the sentence "For inputs and values that are zero or
  normal these are the old magnitudes" is right for `lm`, but not for `p16L` and `ds16L`. These
  are `lift16` of the bounds `p16B` and `ds16B`, so they differ from the old magnitudes whenever
  the bound is below `2^-14`, whatever the value is. They are `2^-14` when the bound is in
  `[2^-24, 2^-14)`, which includes a `float16` `P` or `dS` of `0`. They are `0` when the bound is
  below `2^-24`.
- The Figures bullet leaves out that the benchmark's `HEAD_DIM = 128` row also changed: causal
  `dKTol`, 90th percentile, went from `1.50` to `1.51%`. So did the largest causal `dV` ratio,
  from `0.91` to `0.92%`.
- The largest `dQ` absolute tolerance is `0.0092498`, so `9.2·10^-3`, not `9.3·10^-3`.

The evidence is in `work/readme_script.out` and `work/floor.out`.

## Findings

- **DOC** `doc:kernels/attn_bwd_f16/README.md` — "What changed (library round 25)": "for inputs and values that are zero or normal these are the old magnitudes" is wrong for `p16L` and `ds16L`, and the Figures bullet omits the `HEAD_DIM = 128` changes.
- **release F5** — settled by L126. `DotF16` is now `∀ p, DotLiftAcc P ⟨.f16, .f16, .f32, p, 0⟩ relDot16 absDot16`, with a subnormal operand counted at `2^-14`, the mechanism I asked about. num.md lists it, and the claim is re-proved with every `tl.dot` operand counted by its lifted magnitude.

## Accepted

All items.

## Checked

- The changed items, from the gate's snapshots. The re-opened items are `assumption:dot`,
  `dot_total`, `exp2`, `exp2_total` and `spec`. `model`, `map`, `conditions` and `faithful` are
  unchanged.
- `assumption:dot`:
  - The field is `Rules.Num.DotF16 P`, the fact as listed in num.md's "Facts the library states"
    (restated 2026-11-01).
  - I read `minNormal8`, `liftMag` and `DotLiftAcc` in `Rules/Num.lean`, and `DotF16` in
    `Num/Sm90.lean`.
  - The field's docstring describes the lifted bound correctly.
- `dot_total`, `exp2`, `exp2_total`: I diffed `Num.lean`, `Sm90.lean` and `num.md` against the
  round-24 commit. `DotTotal`, `Exp2F32`, `MathTotal` and their num.md rows are unchanged.
- `spec`, the definitions:
  - I read the new `lm` and `lift16`, the lifted `aQK` and `aVD`, `p16L` and `ds16L`, and the
    accumulations' `B` in `dVTol`, `dKAcc` and `dQAcc`. They are explicit and computable.
  - `lift16 B` bounds `lm x` for `|x| ≤ B`: a `float16` value below `2^-24` is `0`.
  - Masked pairs give `p16B = 0`, so `lift16` gives `0`.
  - `rH` and `aH` are still `relDot16` and `absDot16`. `stepRel` and `stepAbs` are unchanged.
  - `pBr`, `pE`, `p16Err` and `dsErr` changed digest only through `aQK` and `aVD`.
- `spec`, the README's figures:
  - The README's inline script matches the formulas, and its output matches every table figure.
  - Each change from my round-2 figures is at most 0.07 percentage points.
  - The floor statement still holds. On the benchmark's low-mass rows the lift adds at most about
    `5·10^-5` to the `2^-25`-per-pair floor of `3.9·10^-4`; about 250 lifted pairs per row at
    most.
- The prover's responses to F5 and F6 (REFERRED to L126), and L126 in ISSUES.md.

## Not checked

- The changed proof files (`Acc`, `Basic`, `Lane`, `Out`, `Program`, `Step`). The gate checks
  them.
- The evidence `num-dot-sub16-round25`. I trust the library.
