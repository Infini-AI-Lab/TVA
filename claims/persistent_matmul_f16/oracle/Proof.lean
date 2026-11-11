import Claim
import TVA.Proof
import Proof.Geometry
import Proof.Safe
import Proof.Post

/-!
# `persistent_matmul_f16`: the claim

Program `i` reads the matrices `a` and `b` (`Rd`) and writes the `c` tiles of its outer trips
(`Wr`); what it leaves is `Qd`: every `c` element of its tiles as `OutSpec` says.

1. `Proof/Geometry.lean`: the tile of each trip, and the footprints `Rd`, `Wr`: distinct programs
   write disjoint tiles, none of which `a` or `b` meets, and the tiles cover `c`;
2. `Proof/Numeric.lean`: the error bound `errB` of `acc += tl.dot(...)` over the `K` loop and
   its comparison with `tol`;
3. `Proof/Compute.lean`, `Proof/Lanes.lean`: the model's scalar operations, offsets, pointers and
   masks, evaluated at the instance;
4. `Proof/Invariant.lean`: the loop invariants and the `OutSpec` of a finished tile;
5. `Proof/InnerSafe.lean`, `Proof/TripSafe.lean`, `Proof/Safe.lean`: one inner step, one trip,
   one program are safe and leave their tiles as `Qd` says;
6. `Proof/Post.lean`: the programs' results give `Post`;
7. `main` joins them with `correct_of_safe`.
-/

namespace Claims.PersistentMatmul

open TVA

/-- **The claim**: every covered launch ends normally with `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => Rd d init i) (fun i => Wr d init i) (fun i m => Qd d init i m)
    (launch_valid hd) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro i
    exact safe_body hd hpre ((launch d).env P) (hP.dotF16 .tf32) (hP.dotTotal _)
  · -- no program writes another's footprint
    intro i j hij
    exact Disjoint.union_right (Wr_disjoint_Rd' hd) (Wr_disjoint hd hij)
  · -- no program reads what it writes
    intro i
    exact (Wr_disjoint_Rd hd).symm
  · -- the programs' results give `Post`
    intro fin hloc hfr
    exact post_of_tiles (d := d) (init := init) (fin := fin) hd hloc hfr

end Claims.PersistentMatmul
