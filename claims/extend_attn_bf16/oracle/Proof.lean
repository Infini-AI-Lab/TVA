import Proof.Main

/-!
# The proof of `extend_attn_bf16`

`main : claim.Holds` is proved by `correct_of_safe`: one program per
`(request, head, block of BLOCK_M new tokens)`.

1. `Proof.Core*` are the real analysis of the online softmax: per row, the running maximum, sum
   `deno` and accumulator within a running bound of the exact sums at the computed scores
   (`RowInv`, `ColInv`), giving the claim's `tolRel` (`final_bound`).
2. `Proof.Float*`, `Row*`, `Col*`, `Score*`, `Dot*`, `Acc*` turn each float operation of a trip into an
   inequality over `ℝ` from the device facts, and assemble one trip (`RowFl.trip`, `row_output`).
3. `Proof.Pieces` cuts the kernel body into named pieces (`body_eq` is `rfl`); `Proof.Setup`,
   `Scalars`, `Inv`, `Spec` fix the footprints, the scalars, the loop invariants and the facts the
   loads hand over; `Proof.Qi`, `Lanes`, `Load` are the integer and load lemmas.
4. `Proof.Preamble`, `TileQ`, `PrefStep`, `ExtStep`, `Epilogue` walk the scalar steps, the load of
   `q`, one trip of each loop, and the epilogue; `PrefGlue`, `ExtGlue` link a trip's float facts to
   the invariant.
5. `Proof.Program` runs the loops and composes the walk into `safe_program`; `Proof.Main` shows the
   programs' tiles of `o_extend` are disjoint and read no output, and that together they give `Post`.
-/
