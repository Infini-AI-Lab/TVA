import Claim
import TVA.Proof
import Proof.Defs
import Proof.Walk

/-!
# `l2norm_bwd_bf16`: the proof

The model (`l2norm_bwd_kernel`), the conditions (`Pre`) and the specification (`Post`) are in
`Claim.lean`. The route is `correct_of_safe`: one program of the grid at a time.

1. Footprints (`Proof.Lanes`, `Proof.Defs`): program `p` owns the rows `p·BT + [0, BT)` below `T`.
   It reads its `[BT, D]` tiles of `y` and `dy` and its `BT` elements of `rstd` (`R`), writes its
   tile of `dx` (`W`), and leaves the `dx` values `Post` asks for on its rows (`Q`). Tiles of
   distinct programs are disjoint (`hdisj`, `hRW`), and the programs' results give `Post`
   (`hpost`).
2. Numeric bounds (`Proof.Num`, `Proof.RowSum`): the row sum of the products `dy·y` over
   `BD ≤ 512` lanes along a tree of depth at most `9` (`rowsum_err`), then per element the three
   products and the fusable subtraction (`dx_err32`), and the `bfloat16` rounding (`dx_err16`).
3. The leaf (`Proof.Leaf`, `leaf`): from the values the walk computed, the final memory satisfies
   `Q`.
4. The walk (`Proof.Walk`, `safe_program`): each program is `Safe`, one `tl.*` operation at a
   time, for each dtype of `y`, `dy` and `dx`, with the index arithmetic of the `int64` rows and
   `int32` columns.
5. `main`: the launch is valid (`launch_valid`), the footprints are disjoint, and
   `correct_of_safe` composes the programs.
-/

namespace Claims.L2NormBwd

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (R d init) (W d init) (Q d init) (launch_valid hd) (fun i => ?_)
    (fun i j hij => hdisj hd init i j hij) (fun i => hRW hd init i)
    (fun fin hloc hfr => hpost hd hloc hfr)
  exact safe_program hd hpre P hP.sumDepth i

end Claims.L2NormBwd
