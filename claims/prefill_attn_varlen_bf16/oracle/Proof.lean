import Proof.Main

/-!
# The proof of `prefill_attn_varlen_bf16`

`main : claim.Holds` is proved by `correct_of_safe`: one program per `(sequence, head, query block)`.

1. `Proof.NumBase`, `NumStep`, `NumEnd` are the real analysis of the online softmax. The loop keeps,
   for each query row, the running maximum `m`, sum `l` and the accumulator within a running bound
   (`Close`) of the exact sums at the computed scores (`RowInv`, `prop_step`, `rowstep`, `colstep`);
   after the last trip the growth bounds (`GLmax`, `GAmax`) give the claim's `tolRel`
   (`final_ineq`, `final_bound`).
2. `Proof.Floats` turns each float operation of a trip (the score `tl.dot`, `tl.exp`, the row
   max and sum trees, `/`, `tl.addc`, the `bfloat16` conversions, the second `tl.dot`) into an
   inequality over `ℝ` from the device facts, and assembles one trip (`Trip.trip`).
3. `Proof.Setup` fixes the footprints and the pieces of the kernel body (`restProg`, `loopBody`,
   `postProg`), the loop invariant `LoopInv` and the exact data of a row.
4. `Proof.Access` proves the pointers, masks and footprints of the four tile accesses.
5. `Proof.LoopStep` walks one trip, `Proof.Program` the epilogue, the first loads, the loop and
   the scalar steps: each program is `Safe`.
6. `Proof.Main` shows the programs' tiles of `o` are disjoint and read no output, and that the
   programs together give `Post`.
-/
