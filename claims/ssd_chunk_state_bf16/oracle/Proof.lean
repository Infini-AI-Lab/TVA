import Claim
import Proof.Defs
import Proof.SafeMain
import Proof.SafeProg

/-!
# `ssd_chunk_state_bf16`: the proof of the claim

1. `Proof.Defs`: the footprints of program `i` (`R`: everything but `states`; `W`: its tile of
   `states`), its post-condition `Q`, and the arithmetic of the grid;
2. `Proof.Num`, `Proof.Loop`, `Proof.BClause`: the numeric core, the error of the scaled `B` and of
   the `tl.dot` accumulation loop over the chunk's tokens;
3. `Proof.Ptr`, `Proof.SafeMain`: one program walked line by line (`safe_program`);
4. `Proof.SafeProg`: the programs' results give `Post` (`post_of_locals`);
5. `main`: the launch is valid, the tiles are disjoint, and `correct_of_safe` composes the programs.
-/

set_option maxHeartbeats 800000

namespace Claims.SsdChunkState

open TVA

/-- **The claim.** Each program is safe on its own tile (`safe_program`); the tiles' footprints
are disjoint, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (R d init) (W d init) (Q d init) (launch_valid hd) (fun i => ?_)
    (fun i j hij => disjoint_W hd hpre hij) (fun i => disjoint_RW hd hpre i)
    (fun fin hloc hfr => post_of_locals hd hpre hloc hfr)
  exact safe_program hd hpre _ i (hP.mathTotal _ _) (hP.dotTotal _) hP.expF32 hP.dotBF16
