# Review of matmul_e5m2, round 2, by r-e5m2

## Verdict

accept
This is the re-review after library round 24, at gate `3b92ad18abf5649a`. `assumption:dot` now cites `DotE5m2Chunk P`, which is the listed fact for the model's fp8 `tl.dot` configuration. The spec's tolerance is the chunk recursion that this fact gives, computed from the exact partial sums and each chunk's largest lifted product. It is explicit and computable, it covers subnormal operands, and at every covered `K` it still rejects a zero result, a skipped step, `S/2`, `−S` and a float8e5 rounding. At `K = 4096`, on the script's randn data, it is about `|S|`. That is the device's error as the library's fact allows it, and the README says so. The README's figures hold. `cvt`, `cvt_total` and `dot_total` re-opened only because their rule files changed; their statements are unchanged and still listed. I accept every item.

## Findings

None.

## Accepted

All items.

## Checked

- `assumption:dot`: `DotE5m2Chunk P` is `DotF8Chunk P .f8e5 .f8e5`, which for every precision is `DotChunkAcc` on the model's `DotCfg` (`f8e5`, `f8e5`, `f32`, `2^30`) with `2^-7`, `2^-12` and `2^-126`. It is the fact listed in docs/rules/num.md for this operation, and the field's docstring says what the fact says.
- The spec:
  - `errAt` is the per-chunk consequence of the fact: `chunk_step` with `M = max(|S_j| + E_j, m_j)`.
  - Chunk `c` of step `t` is chunk `2t + c` of the loop (`sum_chunk8`).
  - A disabled lane has lifted product `0`, so `chunkMax` of `liftV` covers the fact's `Mj` condition.
  - `liftV` counts a subnormal `float8e5` operand as `2^-14`.
- The new and changed Lean in full (`Defs`, `Basic` with `chunk_step`, `acc_step` and `errAt_le_room`, `Program`'s `AccInv`, `Reject`, the root module's `consistent`):
  - No construct changes the meaning of a library name.
  - The names and the rejection lemmas printed with full names (`work/names3.lean`).
  - `main`, `consistent`, `satisfiable`, `zero_rejected` and `drop_rejected` use only `propext`, `Classical.choice` and `Quot.sound`.
- The README's figures, recomputed exactly:
  - `0.806%`, and `1.618%` when `K mod 64` is in `1..32`;
  - `0.0268` and `0.0658` for the pair input;
  - `13.70%` and `179.30%` for one product at index 0, crossing `|X|` at `K = 2753`;
  - `6.30%` and `73.91%` for all-ones inputs;
  - `tolA`'s relative part, `0.1370` and `1.7930`;
  - the float8e5-rounding bound, `0.02605`.

  The randn medians agree in order of magnitude with my own simulation, which includes the fp8 cast.
- `cvt`, `cvt_total` and `dot_total`: `CvtF32E5m2`, `CvtTotal` and `DotTotal` are unchanged in the Lean and still listed in "Facts the library states".

## Not checked

- Whether `DotE5m2Chunk` is true of the device, and its evidence: that is the library's.
- Part (B) of `Compiler.Faithful`'s scope (the environment).
- The proofs step by step, beyond what Lean checks and what the names mean.
