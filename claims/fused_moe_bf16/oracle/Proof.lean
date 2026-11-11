import Proof.Walk

/-!
# `fused_moe_bf16`

Proof of `main : claim.Holds`. Program `i` of the grid owns the tile `(pidM i, pidN i)`: the lanes
of block `pidM i` of the expert assignment, whose flat index is routed, and the columns
`pidN i * BN + j < N`. It reads the input buffers (`Rfp`), writes its tile of `C` (`Wfp`) and
leaves `outElem` in every element of it (`Qfp`).

1. `Proof.Pid`, `Proof.Foot`: the grouped tile map is a bijection onto the tile grid; the tiles
   are disjoint, `C` shares no byte with the inputs, and the tiles' results assemble into `Post`
   (`post_of_tiles`).
2. `Proof.Tol`: the arithmetic of the `K` loop, chunk by chunk (`trip_step`): after `t` trips each
   accumulator lane is within `accErr` of the exact partial sum, and `accErr` at the last trip is
   below the claim's tolerance `tol`.
3. `Proof.Defs`, `Proof.Store`, `Proof.Inv`, `Proof.Trip`, `Proof.Tail`: the model's body cut into
   pieces; the store of a tile (`safe_storeC`); the loop invariant (`InInv`), one trip
   (`trip_safe`) and the epilogue (`tail_safe`).
4. `Proof.Walk`: `safe_program` walks one program, line by line.
5. Here, `main`, by `correct_of_safe`.
-/

namespace Claims.FusedMoe

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun _ => Rfp d) (Wfp d init) (Qfp d init) (launch_valid hd)
    (fun i => safe_program hd hP hpre i) ?_ ?_ ?_
  · intro i j hij
    exact Disjoint.union_right (R_W_disjoint hd hpre i).symm (W_disjoint_W hd hpre hij)
  · exact fun i => R_W_disjoint hd hpre i
  · exact fun fin hQ hfr => post_of_tiles hd hpre fin hQ hfr

end Claims.FusedMoe
