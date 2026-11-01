# Review of matmul_f16, round 3, by r-matmul

## Verdict

accept
The claim is re-proved on `DotF16` as library round 25 restated it (L126: `DotLiftAcc`, magnitudes `|acc| + Σ liftMag a·liftMag b`, the same `relDot16`/`absDot16`). `assumption:dot` cites the fact as the library lists it. The specification's tolerance keeps its formula, with `A` now the lifted magnitude `dotMag`, which counts magnitudes exactly as the fact does. The tolerance is still explicit and computable. With no subnormal operand it is the round-2 tolerance, so every README figure holds. It still rejects a `bfloat16`-rounded result on normal inputs, and it rejects flushing subnormal operands to zero. The `REFERRED` answer to F7 settles that library report. I accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- L126 in `docs/library-review/ISSUES.md`, the prover's response and notes, `REVIEW.md`'s coordinator notes and per-item changes, and the diffs:
  - library: `Rules/Num.lean` (`minNormal8`, `liftMag`, `DotLiftAcc`, `DotLiftAcc.mono`, `DotAcc.liftAcc`), `Rules/Num/Sm90.lean` (`DotF16`, `DotBF16`), `Rules/Num/Dot8Chunk.lean`, `Lemmas/DotLoop.lean` and `docs/rules/num.md`;
  - kernel: `Defs.lean`, `Basic.lean`, `Program.lean` and the README.
- `assumption:dot`:
  - `DotF16` is `∀ p, DotLiftAcc P ⟨f16, f16, f32, p, 0⟩ relDot16 absDot16` (printed with full names).
  - It is listed in "Facts the library states" and in the reviewer's table row for `float16` operands into `float32` with `maxImpreciseAcc = 0` and any precision.
  - `relDot16 K = (K+2)·2^-22` and `absDot16 K = (K+2)·2^-126` are unchanged.
  - The field's docstring and the README's Assumptions say what `DotLiftAcc` says.
- `assumption:dot_total`: re-opened through two files only; `DotTotal` and its listing are unchanged.
- `spec`, with `work/Lift.lean` (checked with Lean):
  - `mag16` is `Rules.Num.liftMag .f16` by `rfl`, so `dotMag` is the fact's lifted sum over the row and the column.
  - `mag16 q = |q|` for zero and normal `q`, so with no subnormal operand `A = Σ|a·b|` and the round-2 figures (1.26e-4, 1.01e-3, below 0.21% up to `2^13`) and the `bfloat16` rejection stand.
  - The README's per-term bound `|x·y| + 2^-14·(|x|+|y|) + 2^-28` is proved.
  - `tol(2^13, 2^-14) < 2^-15`, so flushing a subnormal operand (`2^-15·1`) to zero breaks `Post`.
  - On the host's `randn` data, subnormal `float16` values are rare and add at most about `2^-14·|partner|` to `A`.
- The kernel's Lean changes:
  - `mag16` and `dotMag` replace `dotAbs`; `partA` sums `liftV`, and `acc_step` cites the restated fact.
  - There is no instance, notation, attribute or export.
  - Model, launch, conditions and map are untouched: the gate snapshots `00f23d84e4f9dd74` and `565503552a0c46d3` differ only in the `assumption:dot`, `assumption:dot_total` and `spec` cones.
  - `#print axioms` for `main`, `consistent` and `satisfiable` shows only `propext`, `Classical.choice` and `Quot.sound`.
- The README's "What changed (library round 25)" section, and its revised Specification, Assumptions and Proof, against the Lean.

## Not checked

- What round 1 lists as not checked still holds. Whether the restated `DotF16` holds on the device is the library's matter (no library review followed round 25), and I trust it as the protocol asks.
