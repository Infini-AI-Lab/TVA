# Review of matmul_f16, round 2, by r-matmul2

## Verdict

accept
The prover's revision settles my three round-1 findings (F1, F2, F3), and I find nothing new. The model is unchanged: it shows the same constants, with the same counts, as in round 1. The `model` and `map` digests moved only through the docstring of `model` (now covering `K = 1`) and the map notes. `Inst.ok.K_le` is now `K ≤ 2^13`. That still covers every launch the script makes (largest `K` 4096). At the new limit the tolerance is about 0.2% of `Σ|a·b|`, so `Post` still rejects a `bfloat16`-rounded result, which my work file checks on an input. The `c_rows` wording and the tolerance figures now match the Lean. The spec, the assumptions and the other conditions are unchanged, so I accept every item.

## Findings

- **release F1** — `model`'s docstring and the map note of line 294 now cover `K = 1` (`range(0, 1)`: one trip, an `int32` induction variable).
- **release F2** — the `c_rows` docstring and the README now say it excludes column-major and transposed layouts of `c` too, though they do not overlap.
- **release F3** — with `K ≤ 2^13`, `tol` is about 0.2% of `Σ|a·b|` at the largest covered size, and the README explains how that compares with `|S|`; checked in my work file.

## Accepted

All items.

## Checked

- REVIEW.md, the prover's response and notes, and every change they name: `Defs.lean`, read in full again (`K_le`, the `c_rows` and `tol` docstrings, the `model` docstring, the new `tol_rel_lt`), the map notes of lines 229–244, 269, 270, 273, 294 and 302, and the README's Specification, Conditions and Narrowings.
- The other kernel Lean files. They declare the same constants as in round 1, and the only proof change is that `safe_program` derives `K ≤ 2^16` from `K_le`. There is no instance, notation, attribute, export, set_option or renaming. `#print axioms` on `main`, `consistent` and `satisfiable`: `propext`, `Classical.choice`, `Quot.sound`.
- The model printed with full names and explicit instances (work/Explicit.lean, work/Names.lean): the same constants and counts as in round 1, so the transcription I judged in round 1 is the one claimed now.
- Coverage (work/Cover.lean): `Inst.ok` still holds for every square row-major launch up to `n = 4096` (the test and the benchmark).
- The tolerance at `K = 2^13` (work/Cover.lean, norm_num): between `2.01e-3·A` and `2.03e-3·A`, with an absolute part below `10^-34`. A `bfloat16`-rounded result breaks `OutSpec` there: for `S = 1 + 2^-8 - 2^-10` with `A = S`, the stored `1` is farther from `S` than the finite-case bound allows.
- The figures in the `tol` docstring and the README (4% and 12% of a typical randn `|S|`; crossovers near `K = 15745` and `34368`).

## Not checked

- The library itself (trusted), and part (B) of `Compiler.Faithful`'s scope (assumed).
- The proof steps beyond what Lean and the gate's build and replay check.
