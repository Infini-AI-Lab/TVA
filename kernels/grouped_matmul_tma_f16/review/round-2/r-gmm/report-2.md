# Review of grouped_matmul_tma_f16, round 2, by r-gmm

## Verdict

accept
Library round 24 (L124, L125) re-opened `assumption:dot`, `assumption:dot_total` and `spec` through `docs/rules/num.md` and `lean/TVA/Rules/Num/Sm90.lean`. Nothing those items rely on changed. `DotF16`, `relDot16`, `absDot16` and `DotTotal` are the same in the Lean, `DotF16`'s row in "Facts the library states" is the same, and so is the page's statement about the 16-bit dot. The change withdraws, adds or moves only fp8 facts and constants, and this kernel cites none of them. The kernel's own files are unchanged, so I re-accept every item. L125's mechanism raises a new doubt about `DotF16` itself, which I file as a non-blocking `LIBRARY_REPORT`.

## Findings

- **LIBRARY_REPORT** `decl:TVA.Rules.Num.DotF16` — Does DotF16 survive L125's mechanism? A subnormal float16 operand may set the wgmma window as a subnormal fp8 one does, and num-dot Part 2 had no such input

## Accepted

All items.

## Checked

- The library diff (`docs/rules/num.md`, `lean/TVA/Rules/Num/Sm90.lean`):
  - `DotF16`, `DotBF16`, `relDot16 = (K+2)·2^-22` and `absDot16 = (K+2)·2^-126` are untouched;
  - in num.md, the float16 row of the kind table and the `DotF16`/`DotBF16` row of "Facts the library states" are unchanged; the `…Total` row still lists `DotTotal`;
  - the 16-bit bullet of "`tl.dot` (the chunk model)" is unchanged;
  - the only changes are the fp8 facts withdrawn (`DotE4m3` …, `DotF8`), the chunk facts and their section added, and `chunks8`, `relDot8` and `absDot8` moved.
- L124 and L125 in `docs/library-review/ISSUES.md`, and the coordinator's note in `REVIEW.md`.
- The gate digest is now `f0e0fd43db6ed2b8`; the kernel's own files are unchanged.

## Not checked

- Whether `DotF16` holds when a subnormal `float16` operand sets the window: a library matter (the LIBRARY_REPORT).
