import Claim
import TVA.Proof
import Proof.Basic
import Proof.Foot
import Proof.Tol
import Proof.Walk

/-!
# The claim, from one program's safety
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- **The claim.** Each program is safe on its own `128 × 256` tile of `c` (`prog_safe`); the tiles'
footprints are disjoint, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · exact prog_safe P hd hpre i hP.dotScaledE4m3 hP.dotScaledE4m3E2m1 (hP.dotScaledTotal _)
      (hP.dotScaledTotal _)
  · intro i j hij
    exact prog_disj hd i j hij
  · intro i
    exact prog_rw hd i
  · intro fin hloc hfr
    exact prog_post hd hpre fin hloc hfr

end Claims.BlockScaledMxfp8
