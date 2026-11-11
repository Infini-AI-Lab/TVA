import Claim
import TVA.Proof
import Proof.Safe

/-!
# The proof of `ssd_bmm_chunk_bf16`

Program `(pid, 0, pid_ch)` of the grid owns one `64 × 128` tile of `out`, of the chunk
`pid_ch / ngroups` and the group `pid_ch % ngroups`. It only reads `a`, `b` and `seq_idx`, so the
proof is the independent-programs route, `correct_of_safe`:

1. `Base`: the launch is valid; the program's chunk, group and tile (`cOf`, `hOf`, `mOf`, `nOf`);
   its footprints `R` (the inputs) and `W` (its tile of `out`) and its post-condition `Q`.
2. `Post`, `Foot`: the tiles of distinct programs are disjoint and none reads `out`; the
   programs' results, assembled, give `Post` (`post_of_Q`).
3. `Numeric`, `Inv`: the tolerance `tol` absorbs the dot loop's error bound; the loop invariant
   `inv` says that after `t` turns the accumulator is within `DotLoop.bound` of the exact dot
   product of the first `t · 32` terms, and the pointer tiles sit `t` slabs on.
4. `Peel`, `Lane`, `LaneGet`, `LaneMask`: the kernel's integer arithmetic, line by line: no
   `int32` or `int64` product wraps, the lanes of the pointer tiles and of the masks.
5. `InvZero`, `Load`, `Step`: the invariant holds before the loop; one turn of the loop keeps it
   (the loads' footprints, and `Device.DotBF16` for `acc += tl.dot(a, b)`).
6. `PostSafe`, `Safe`: the `seq_idx` loads, `tl.where` and the store; the whole body is `Safe`
   (`body_safe`).
-/

namespace Claims.SsdBmm

open TVA

/-- **The claim.** Each program is safe on its own tile (`body_safe`); the tiles are disjoint and
no program reads `out`, and the programs' results give `Post` (`post_of_Q`). -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  exact correct_of_safe (R d init) (W d init) (Q d init) (launch_valid hd)
    (fun i => body_safe hd hpre hP i) (fun i j hij => w_disjoint hd hij)
    (fun i => rw_disjoint hd) (fun fin hloc hfr => post_of_Q hd hloc hfr)

end Claims.SsdBmm
