# Review of matmul_e5m2, round 2, by r-e5m2b

## Verdict

accept
After library round 24 the claim cites `DotE5m2Chunk`, the fact the library now lists for the model's fp8 `tl.dot`, and its tolerance is that fact applied chunk by chunk against the exact partial sums and each chunk's largest lifted product. The tolerance is explicit and computable, and it is still meaningful at the largest covered `K`: a zero result, a skipped loop, any skipped step, `S/2` and `−S` break `Post` at every `1 ≤ K ≤ 4096` (`zero_rejected`, `drop_rejected`, `tol_spike`, re-proved). The README's figures hold by my own evaluation. `cvt`, `cvt_total` and `dot_total` rest on facts that round 24 left unchanged. I have no finding and accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- `assumption:dot`: `Rules.Num.DotE5m2Chunk P` is `DotF8Chunk P .f8e5 .f8e5`, `DotChunkAcc` for `float8e5` by `float8e5` into `float32` with `maxImpreciseAcc = 2^30` and every precision, with `rhoChunk8`, `sigmaChunk8`, `alphaChunk8`; num.md lists it, first column, for the model's operation; the docstring says what the fact says (`work/Rechunk.lean`, full names).
- `spec`: `liftV`, `partSum` and `chunkMax` over chunks of 32, `errAt`'s recursion, `tol` over `2·cdiv(K, 64)` chunks, `tolA` and `dotLift`; that the recursion is what the fact gives with `M = max(|S_j| + E_j, m_j)` (`chunk_step`), that it is the library's own `errB`, and that chunk `c` of step `t` is chunk `2t + c` of the loop (`sum_chunk8`, `acc_step` read in full); disabled lanes count `0` in both products and lifted products.
- `Reject.lean` read in full: `tol_spike` (now `(1 + ρ + σ)^2 − 1 ≈ 1.62%` of the lifted product), `zero_rejected` and `drop_rejected` keep their statements; a check that `S/2` is rejected at `K = 4096` (`work/Rechunk.lean`).
- The README's figures, evaluated exactly with `python3 -I`: `0.81%` and `1.62%` (worst at `K = 4064`), `0.027` and `0.066`, `13.7%` and `179.3%` (crossing `|X|` at `K = 2753`), `6.3%` and `73.9%`, `tolA`'s `0.137` and `1.793`, the `0.027` bound at `1.5625`; randn medians of `1.5` at `K = 512` and about `|S|` at `K = 4096` on my own 20 draws.
- `cvt`, `cvt_total`, `dot_total`: definitions unchanged in Lean; num.md's rows for them and the round-24 author's notes show no change to them.
- `#print axioms` of `main`, `consistent`, `satisfiable`, `zero_rejected`, `drop_rejected`: only the three standard axioms; Basic, Defs, Program, Reject and the root module have no name-changing command.

## Not checked

- Whether `DotE5m2Chunk` is true of the device (trusted; round 24 had no library review).
- The unchanged items (`model`, `map`, `conditions`, `faithful`) beyond their unchanged digests.
- I did not re-run the gate or `lake build`.
