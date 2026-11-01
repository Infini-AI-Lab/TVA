# Review of matmul_e5m2, round 2, by r-e5m2

## Verdict

accept
The claim now cites `DotE5m2ChunkLoss`, the fact the library lists for this `tl.dot`. The new tolerance is what that fact gives chunk by chunk (the library's `errM`), and it is explicit and computable. A zero result, a skipped step, `S/2`, `-S` and a float8e5 rounding are still rejected at every covered `K`. Every README figure I recomputed holds. I accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- **`assumption:dot`.** It is `DotF8ChunkLoss P .f8e5 .f8e5`: for every precision, `DotChunkLoss` on the model's configuration, with constants `2^-7`, `2^-12`, `2^-126`, `κ = 2` and `ρ' = 2^-12`. num.md lists it for an fp8 × fp8 `tl.dot` into `float32` at `maxImpreciseAcc = 2^30`, and its docstring says what the fact says.
- **`spec`.** `errAt`'s step is the fact's per-chunk bound:
  - take `M = max(|S_j| + E_j, m_j) ≤ X_j + E_j`; both arms of the minimum grow with `M`, and `|s + c| ≤ |S'| + E` (`chunk_step`);
  - the fact's `chunkAbs8` for chunk `c` of step `t` is the loop's `chunkAbs` at `2t + c`, with disabled lanes `0`;
  - this is the library's `errM`.
- **The README's figures**, recomputed exactly in rationals:

  | Input | `K = 512` | `K = 4096` |
  |---|---|---|
  | one product at `K − 1` | `0.806%` | `0.806%` |
  | that input, worst covered `K` (`K = 4033`, `K mod 64 = 1`) | `0.855%` (`71713·2^-23`) | |
  | pair input, `s = 0` | `0.0268 → 0.0165` | `0.0658 → 0.0175` |
  | one product at index `0` | `13.70% → 1.547%` | `179.30% → 7.253%` |
  | all ones, relative to `S` | `6.30%`, unchanged | `73.91%`, unchanged |
  | first chunk `1`s, then `1/16`s (row and column) | `12.46% → 11.83%` | `142.68% → 73.73%` |

  On all ones the second arm is never the smaller. The `x = y = 1.25` bounds are below `0.015` (worst covered `K`) and `0.014` (`K = 512`).
- **`torch.randn` data cast to float8e5**, on my own draws (twice 20, then 200): the second arm is never the smaller, so the old and new tolerances agree exactly. Over the 200 draws the median tolerance is `1.51` at `K = 512`, and `48` against a median `|S|` of `42` at `K = 4096`. This is the same order as the README's `1.5` and `52`.
- **Axioms.** `main`, `consistent`, `satisfiable`, `tol_spike`, `zero_rejected` and `drop_rejected` use only the three standard axioms (`work/names6.lean`).
- **Non-blocking:**
  - `OutSpec`'s docstring still says `E` comes from the exact partial sums and each chunk's largest lifted product. It now also depends on each chunk's exact magnitude, as `tol`'s docstring says.
  - The README says the tolerance at `K = 4096` is below `|S|` on half the draws. That holds for its own 20 draws; over my 200 draws it held on about 40%.

## Not checked

Nothing.
