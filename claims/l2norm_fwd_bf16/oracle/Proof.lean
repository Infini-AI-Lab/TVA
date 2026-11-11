import Claim
import TVA.Proof
import Proof.Defs
import Proof.Walk

/-!
# `l2norm_fwd_bf16`: the proof

The model (`l2norm_fwd_kernel`), the conditions (`Pre`) and the specification (`Post`) are in
`Claim.lean`. The route is `correct_of_safe`: one program of the grid at a time.

1. Footprints (`Proof.Lanes`, `Proof.Defs`): program `p` owns the rows `p·BT + [0, BT)` below `T`.
   It reads the `[BT, D]` tile of `x` (`R`), writes its tile of `y` and its `BT` elements of
   `rstd` (`W`), and leaves the `rstd` and `y` values `Post` asks for on its rows (`Q`). Tiles of
   distinct programs are disjoint (`hdisj`, `hRW`), and the programs' results give `Post` (`hpost`).
2. Numeric bounds, per row (`Proof.NumSum`, `Proof.RowSum`, `Proof.NumRstd`, `Proof.NumY`): the
   products and the fusable row sum over `BD ≤ 512` lanes (`sum_err`), then `+ eps`, `tl.sqrt` and
   `1 / ·` (`rstd_err`, within `2^-12` relative), then `x · rstd` and its conversion
   (`y_err_bf16`, and the `float32` case).
3. The leaf (`Proof.Leaf16`, `Proof.Leaf32`): from the values the walk computed, the final memory
   satisfies `Q`, for a `bfloat16` and for a `float32` `y` (`qn16`, `qn32`).
4. The walk (`Proof.Walk`, `safe_program`): each program is `Safe`, one `tl.*` operation at a
   time, with the index arithmetic of the `int64` rows and `int32` columns.
5. `main`: the launch is valid (`launch_valid`), the footprints are disjoint, and
   `correct_of_safe` composes the programs.
-/

namespace Claims.L2NormFwd

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (R d init) (W d init) (Q d init) (launch_valid hd) (fun i => ?_)
    (fun i j hij => hdisj hd init i j hij) (fun i => hRW hd init i) (fun fin hloc hfr => hpost hd hloc hfr)
  exact safe_program hd hpre P (hP.mathTotal _ _) hP.sqrtF32 (hP.divTotal _) hP.divFull i

end Claims.L2NormFwd
