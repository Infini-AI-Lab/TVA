# Review of grouped_matmul_tma_f16, round 2, by r-gmm2

## Verdict

accept
Library round 24 re-opened `assumption:dot`, `assumption:dot_total` and `spec`. It withdrew and added only fp8 `tl.dot` facts, and moved `chunks8`, `relDot8` and `absDot8` without change. Nothing these items rely on changed: `DotF16`, its listing, `relDot16`, `absDot16`, `DotTotal` and the page's float16 statements are as I accepted them. I re-accept every item. Reading L125 raised one doubt, filed as a non-blocking LIBRARY_REPORT: does the same subnormal-window mechanism break `DotF16`? num-dot Part 2 never tested that input.

## Findings

- **LIBRARY_REPORT** `decl:TVA.Rules.Num.DotF16` — L125's subnormal-window failure is untested for float16 on wgmma; num-dot Part 2, on which DotF16 rests, had no such input

## Accepted

All items.

## Checked

- The diff of `lean/TVA/Rules/Num/Sm90.lean`: only the fp8 definitions (`chunks8`, `relDot8`, `absDot8`, `DotF8`, `DotE4m3` … `DotE5m2E4m3`) are removed. `relDot16`, `absDot16` and `DotF16` (`∀ p, DotAcc P ⟨.f16, .f16, .f32, p, 0⟩ (fun _ => True) relDot16 absDot16`) are unchanged.
- The diff of `docs/rules/num.md`: the operation table still lists `DotF16 P` for `float16` operands into `float32` with `maxImpreciseAcc = 0` and any precision. The "Facts the library states" row for `DotF16` keeps the same constants and evidence. The 16-bit paragraph of the chunk model is unchanged. The other changes concern fp8 only (rows, the new chunk section, `DotChunkAcc`).
- `DotTotal` and `DotCfg` are not in the diff, so `dot_total`'s configuration is still the model's.
- `docs/library-review/ISSUES.md` L124 and L125, the coordinator's note in REVIEW.md, and the note added to `evidence/num-dot`'s README. This kernel cites no fp8 fact.

## Not checked

- Whether `DotF16` holds when a product with a subnormal `float16` operand sets the `wgmma` window (the library's matter; the LIBRARY_REPORT).
