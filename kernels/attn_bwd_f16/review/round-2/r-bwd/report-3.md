# Review of attn_bwd_f16, round 2, by r-bwd

## Verdict

accept
Library round 24 re-opened five of my items (`assumption:dot`, `assumption:dot_total`, `assumption:exp2`, `assumption:exp2_total` and `spec`) only because their digests hold `docs/rules/num.md` and `lean/TVA/Rules/Num/Sm90.lean` whole. Nothing the kernel cites changed: `DotF16`, `DotTotal`, `Exp2F32`, `MathTotal`, the constants `relDot16` and `absDot16`, and their listing and text in num.md are as they were. I accept all items again. The round's own finding (L125) raises a doubt about `DotF16` on subnormal `float16` operands, which this kernel's `P` and `dS` often are. I file it as a non-blocking `LIBRARY_REPORT` and carry on as if the library were right.

## Findings

- **LIBRARY_REPORT** `decl:TVA.Rules.Num.DotF16` — L125's mechanism (a subnormal operand's exponent field, read as the smallest normal's, sets the wgmma window) is untested for float16, and this kernel's fp16 P and dS are often subnormal

## Accepted

All items.

## Checked

- L124 and L125 in `docs/library-review/ISSUES.md`, and the coordinator's note in `REVIEW.md`.
- The diff of `lean/TVA/Rules/Num/Sm90.lean`. It removes the fp8 facts `DotF8`, `DotE4m3`, `DotE5m2`, `DotE4m3E5m2` and `DotE5m2E4m3`, and moves `chunks8`, `relDot8` and `absDot8`. `DotF16`, `relDot16`, `absDot16` and `Exp2F32` are outside the changed lines.
- The diff of `docs/rules/num.md`. Its changes are confined to the fp8 parts:
  - the fp8 row of the reviewer's table;
  - the fp8 rows of "Facts the library states", replaced by the chunk facts;
  - the new chunk section;
  - the fp8 part of the chunk-model text;
  - the alignment question and the satisfiability list.

  The rows for `MathTotal`/`DotTotal`, `Exp2F32` and `DotF16`, and the 16-bit paragraph of "`tl.dot` (the chunk model)", are unchanged.
- That the kernel's own files and the other items' digests (`faithful`, `conditions`, `map`, `model`) are unchanged.
- `evidence/num-dot`'s new note and `evidence/num-dot-chunk` Parts A and F, for the `LIBRARY_REPORT`.

## Not checked

- Whether `DotF16` holds on inputs where a subnormal `float16` operand sets the window: a library matter, raised as the `LIBRARY_REPORT`.
