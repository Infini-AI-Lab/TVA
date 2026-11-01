# Review of matmul_e5m2, round 2, by r-e5m2b

## Verdict

accept

The claim now cites `DotE5m2ChunkLoss`, and the tolerance composes the minimum of its two arms per chunk; the gate passes at 752f03b36e81309e. `assumption:dot` cites the listed fact for the model's `tl.dot`. The tolerance is explicit and computable, and it is exactly the library's `Dot8Chunk.errM` on the loop's chunks. It is never above the round-24 tolerance, so the rejection lemmas still hold. The README's figures hold on exact evaluation. I accept all items, with one non-blocking wording finding on `OutSpec`'s docstring.

## Findings

- **DOC** `decl:Kernels.MatmulE5m2.OutSpec` — `OutSpec`'s docstring still describes `E = tol va vb K` as computed "from the exact partial sums and each chunk's largest lifted product". It now also uses each chunk's exact magnitude (`chunkAbs`, the second arm), which is what makes the bound small after a large early product. Adding "and each chunk's exact magnitude" would resolve it.

## Accepted

All items.

## Checked

- `assumption:dot`: `DotE5m2ChunkLoss P` is `DotF8ChunkLoss P .f8e5 .f8e5`. num.md's fp8 `tl.dot` row lists it, for any `input_precision` with `maxImpreciseAcc = 2^30`, which matches the model's line 300. The docstring states the fact as the library does, including both arms and the unlifted `A`.
- `spec`, Lean (work file `ChunkLoss`, run with `lake env lean`):
  - `errAt_eq_errM`: `errAt` equals `Dot8Chunk.errM ρ σ α κ ρ' 0` on the loop's chunk sums, `chunkAbs` and `chunkMax`. So the tolerance is the fact's composed bound, with the fact's constants.
  - `errAt_le_errOld`: `errAt` is never above the round-24 recursion, so `Post` only got stronger.
  - The second arm is the smaller exactly when `X + E > (8192/31)·A ≈ 264·A`, the README's "about `2^8·A_j`".
  - `main`, `consistent`, `satisfiable`, `tol_spike`, `zero_rejected`, `drop_rejected` and `tol_le_tolA` use only the standard axioms.
- `chunk_step` and `acc_step`: `M = max(|S| + E, m) ≤ X + E`, and both arms are monotone in `M`. Each step's `chunkAbs8` is the loop's `chunkAbs`, with disabled lanes contributing zero.
- The figures, by exact rational evaluation (work file `chunkloss_figs.py`), all hold:
  - one product `1` at `K − 1`: `0.81%` at both sizes;
  - `tol_spike`'s worst over every covered `K`: `71713·2^-23 ≈ 0.855%`, attained;
  - `drop_rejected` with `s = 0`: `0.0268 → 0.0165` and `0.0658 → 0.0175`;
  - one product at `0`: `13.7% → 1.55%` and `179.3% → 7.25%`;
  - all-ones: unchanged at `6.3%` and `73.9%`, never on the second arm (`X + E ≤ 7035 < 8456`);
  - first chunk `1`s, then `1/16`s: with both operands `1/16`, `12.5% → 11.8%` and `142.7% → 73.7%`;
  - `x = y = 1.25`: `0.0141 < 0.015` at the worst covered `K` and `0.0134 < 0.014` at `512`.
- The `torch.randn` data: I used the script's seed `0` and `float16` randn cast to `float8e5` on GPU 0, with 512 rows at `K = 512` and 120 at `K = 4096`. No chunk takes the second arm, and new/old is exactly `1`, so "unchanged" holds. The README's medians (`1.5` against `20.5`, `52` against `52`) lie within the spread of my 20-draw medians.

## Not checked

- I did not check whether `DotE5m2ChunkLoss` is true on the device; kernel review trusts the library, and no library review followed round 26.
- I did not re-check the other items (model, map, conditions, the other assumptions), which did not change.
