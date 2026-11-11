import Proof.Program
import Proof.Trip
import Proof.TripVf
import Proof.Foot

/-!
# `chunk_delta_h_bf16`: the claim

`correct_of_safe` composes the programs: each is `Safe` (`safe_program`), the footprints are disjoint
(`disjoint_W`, `disjoint_RW`) and the programs' results give `Post` (`post_of_locals`).
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- A program of the launch is below the grid size. -/
theorem lt_grid {d : Inst} (i : Fin (launch d).size) : i.1 < gridN d := by
  exact (launch_size d) ▸ i.2

/-- **`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is correct.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  refine correct_of_safe (fun _ => Rset d) (fun i => W d i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · refine safe_program hd hP hpre (fun hvf => trip_nonvf hd hP hpre hvf) (fun hvf => trip_vf hd hP hpre hvf)
      (lt_grid i) _ ?_
    simp only [Launch.ctx, claim, launch]
    exact Nat.mod_eq_of_lt (lt_grid i)
  · exact fun i j hij => disjoint_W hd (lt_grid i) (lt_grid j) (fun h => hij (Fin.ext h))
  · exact fun i => disjoint_RW hd (lt_grid i)
  · intro fin hloc hfr
    exact post_of_locals hd hloc hfr

end Claims.ChunkDeltaH
