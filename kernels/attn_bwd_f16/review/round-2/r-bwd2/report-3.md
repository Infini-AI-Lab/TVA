# Review of attn_bwd_f16, round 2, by r-bwd2

## Verdict

accept
Library round 24 re-opened `spec` and the four `Num` assumptions. Nothing they rely on changed:
`DotF16`, `Exp2F32`, `DotTotal`, `MathTotal`, the constants `relDot16` and `absDot16`, their rows
in "Facts the library states", and the page's statements about them are all as before. The round
withdrew only the fp8 whole-dot facts and added the fp8 chunk facts, which this kernel does not
cite. I re-accept every item. I also file one non-blocking `LIBRARY_REPORT`: round 24's finding
about subnormal operands on `wgmma` raises the same question for `DotF16`, and this kernel feeds
it subnormal `float16` operands.

## Findings

- **LIBRARY_REPORT** `decl:TVA.Rules.Num.DotF16` — does `DotF16` hold when a subnormal `float16` operand sets the `wgmma` window (L125's mechanism)? This kernel's `ppT` and `dsT` are often subnormal

## Accepted

All items.

## Checked

- L124 and L125 in `docs/library-review/ISSUES.md`, and the coordinator's note in `REVIEW.md`.
- The diff of `lean/TVA/Rules/Num/Sm90.lean`:
  - it removes only `DotF8` and the four fp8 whole-dot facts, and moves `chunks8`, `relDot8` and `absDot8`;
  - `DotF16`, `relDot16`, `absDot16` and `Exp2F32` are unchanged.
- The diff of `docs/rules/num.md`:
  - unchanged: the rows for `MathTotal`/`DotTotal`, `Exp2F32` and `DotF16` in "Facts the library states", the reviewer table's row for `float16` dots, and the 16-bit bullet of the chunk model;
  - new or changed: the fp8 rows, the fp8 chunk section, the `DotChunkAcc` definition, and the alignment point under "Device questions".
- Why each item re-opened. Every item's cone reaches the changed files: `spec` through `rH` and `aH`, the four `Num` assumptions through the rule pages. No constant those cones hold changed.

## Not checked

- Whether `DotF16` holds on subnormal `float16` operands. That is a device fact, which is the library's matter (the `LIBRARY_REPORT`).
