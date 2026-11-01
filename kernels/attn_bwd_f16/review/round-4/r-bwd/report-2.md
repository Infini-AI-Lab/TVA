# Review of attn_bwd_f16, round 4, by r-bwd

## Verdict

accept

Library round 26 re-opened `assumption:dot`, `assumption:dot_total`, `assumption:exp2`, `assumption:exp2_total` and `spec` only because their digests hash `docs/rules/num.md` whole. Nothing they rely on changed, so I accept all items again.

## Findings

None.

## Accepted

All items.

## Checked

- **Cones.** I compared the item cones between the round-3 and round-26 snapshots. The only change in each re-opened item is `file:docs/rules/num.md`; no declaration was added, removed or changed.
- **Lean diff.** It only adds fp8 material in the two `Dot8Chunk` files: the `ChunkLoss` facts, `chunkAbs8`, `kappaChunk8`, `rhoLossChunk8`, `errM` and its lemmas. Its only removed lines are a docstring bullet. None of it is in this kernel's cones.
- **num.md diff.** These are unchanged:
  - the rows of "Facts the library states" for `DotF16`/`DotBF16`, `Exp2F32` and `MathTotal`/`DotTotal`;
  - the reviewer's row for 16-bit dots;
  - the `DotLiftAcc` definition row;
  - the 16-bit bullet of the chunk model.

  L130's reworded opening still says 16 products per instruction for 16-bit operands on both paths.
- **What I read.** L124 and L127 to L133, the round-26 author's notes, and the coordinator's note in `REVIEW.md`.

## Not checked

- The new fp8 facts and their evidence. This kernel does not cite them, and kernel reviewers trust the library.
