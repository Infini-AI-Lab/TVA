# Review of matmul_f16, round 3, by r-matmul2

## Verdict

accept
Library round 25 (L126) restated `DotF16` as `DotLiftAcc`, relative to lifted magnitudes; that settles my round-2 `LIBRARY_REPORT` (F7, `REFERRED`). The prover's re-proof changes only the tolerance's magnitude: `A` is now `dotMag`, the sum of the products' lifted magnitudes, where `mag16` is the library's `liftMag .f16` by definition. On zero and normal operands this is `Σ|a·b|`, so the tolerance, its figures and the `bfloat16` rejection are as in round 2. A subnormal operand adds a tiny, bounded term. `assumption:dot` is the right listed fact, and `dot_total` did not change. The model, conditions and map keep their accepted digests. I re-accept every item.

## Findings

- **release F7** — L126 restates `DotF16` with a subnormal `float16` operand counted at `2^-14`, and the claim is re-proved on it.

## Accepted

All items.

## Checked

- L126 in `docs/library-review/ISSUES.md`, the coordinator's notes and per-item changes in `REVIEW.md`, and the prover's response and notes (round 2).
- `assumption:dot`. `DotF16 P` is `∀ p, DotLiftAcc P ⟨f16, f16, f32, p, 0⟩ relDot16 absDot16`, still listed in num.md for this operation and configuration, with `relDot16` and `absDot16` unchanged. I printed `DotLiftAcc`, `liftMag` and `minNormal8` (work/Names.lean), and the field's docstring describes them correctly. `dot_total` is unchanged.
- `spec`. I read `Defs.lean`'s new `mag16`, `dotMag`, `OutSpec` and `tol` docstring, and `Basic.lean`'s `liftV`, `partA` and the re-proved `acc_step`. In work/Cover.lean: `mag16 = liftMag .f16` by `rfl`; `mag16` of a subnormal is `2^-14`; a normal value and zero keep their own magnitude; a subnormal `2^-24` times `2^14` counts `1`.
- The README's figures. They are unchanged without subnormal operands. The bound on a subnormal's added term (`2^-14·|y| + 2^-28`, and `DotLift.liftMag_mul_le`) holds. The tolerance at `K = 512`, `4096` and `2^13`, the coverage of the host's sizes, and the `bfloat16` example from round 2 all re-ran in work/Cover.lean.
- The model printed with full names and explicit instances: identical to round 2. `#print axioms` on `main`, `consistent` and `satisfiable`: `propext`, `Classical.choice`, `Quot.sound`. The kernel's Lean has no instance, notation, attribute, export, set_option or renaming.

## Not checked

- The library's restated fact and its evidence (`evidence/num-dot-sub16-round25`), which kernel review trusts. No library review followed round 25.
- Part (B) of `Compiler.Faithful`'s scope (assumed), and the proof steps beyond what Lean and the gate check.
