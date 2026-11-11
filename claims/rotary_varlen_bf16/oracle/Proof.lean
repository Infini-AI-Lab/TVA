import Claim
import Proof.Basic
import Proof.Launch
import Proof.Safe
import Proof.Tol
import Proof.Foot
import Proof.FootDisj

/-!
# `rotary_varlen_bf16`

The kernel is `rotary_embedding_kernel`. Its launch is the 3-D grid `(g0 d, B, H)`; program `i`
splits into `(ax0, ax1, ax2)`, and each program rotates the channels of the head `ax2` of the
`BT`-token chunk `chunkI` of the sequence `seqI`. Programs read shared metadata but write disjoint
elements of `y`, so `correct_of_safe` composes them:

1. `Basic`: the per-program metadata, the read footprint `R`, the write footprint `W` and the
   local post-condition `Q`;
2. `Launch`: the launch is valid;
3. `Safe` (with `Body`, `State`, `Ocs`, `TailNE`, `TailI`, `Tol`, `Common`): one program is `Safe`,
   walked line by line: the prologue reads the metadata, the offset line computes each token's
   table row, and the non-interleaved or interleaved tail loads, rotates (each output within the
   claim's tolerance of the exact rotation, `Tol`) and stores;
4. `FootDisj`: distinct programs write disjoint bytes of `y`, which no program reads;
5. `Foot`: the footprints are as stated, and local results with unwritten bytes unchanged give
   `Post`.
-/

namespace Claims.RotaryVarlen

open TVA

/-- **The claim**: every covered launch of `rotary_embedding_kernel` ends normally with `Post`. -/
theorem main : claim.Holds := by
  intro P hP (d : Inst) hd init hpre
  refine correct_of_safe (fun i => R d init i) (fun i => W d init i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_)
    ?_ ?_ ?_
  · exact safe_program P hd hpre i.isLt
  · intro i j hij
    exact foot_disj hd hpre i.isLt j.isLt fun h => hij (Fin.ext h)
  · intro i
    exact foot_rw hd hpre i.isLt
  · intro fin hloc hfr
    exact post_of_local hd hpre hloc hfr

end Claims.RotaryVarlen
