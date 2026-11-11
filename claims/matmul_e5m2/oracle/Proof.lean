import TVA.Proof
import Claim
import Proof.Defs
import Proof.Disj
import Proof.Post
import Proof.Tile.Body

/-!
# The proof of `matmul_e5m2`

Program `p` of the grid computes the `128 × 256` tile `(pid_m p, pid_n p)` of `c`. It reads the
matrices `a` and `b` (`R`), writes its tile of `c` (`W`), and leaves `OutSpec` in it (`Q`).

1. `tile_safe` (`Proof/Tile/Body.lean`): one program, walked line by line with the rules of
   `TVA.Proof.Rules`. The grouped program-id arithmetic (`Proof/Pid.lean`) gives the tile; the loop
   over `K` carries an invariant on the partial values of each accumulator lane, one trip at a time
   from the device's `float8e5` chunked dot fact (`Proof/Tile/Acc1.lean`, `Proof/Tile/Chunk.lean`);
   the final `float32 → float16` rounding gives `OutSpec` (`Proof/Tile/Acc2.lean`).
2. `W_disjoint`, `W_R_disjoint`, `R_W_disjoint_of_ok` (`Proof/Disj.lean`): distinct programs own
   distinct tiles of `c`, and `c` shares no byte with `a` or `b`.
3. `post_of_tiles` (`Proof/Post.lean`): the tiles cover the matrix `c`, so the programs' results
   give `Post`.
4. `main`: `correct_of_safe` composes the programs.
-/

namespace Claims.MatmulE5m2

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d init i) (fun i => W d init i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · exact tile_safe d hd i init hpre hP.dotE5m2ChunkLoss (hP.dotTotal _) (hP.cvtTotal _ _ _)
  · intro i j hne
    have hne' : (i : Nat) ≠ (j : Nat) := fun h => hne (Fin.ext h)
    have hi : (i : Nat) < grid d := Nat.lt_of_lt_of_eq i.isLt (launch_size d)
    have hj : (j : Nat) < grid d := Nat.lt_of_lt_of_eq j.isLt (launch_size d)
    rw [Set.disjoint_union_right]
    exact ⟨W_R_disjoint (init := init) hd, W_disjoint (init := init) hd hi hj hne'⟩
  · intro i
    exact R_W_disjoint_of_ok d init (i : Nat) hd
  · intro fin hloc hfr
    refine post_of_tiles (d := d) (fun p hp => hloc ⟨p, ?_⟩) (fun z hz => hfr z (fun i hzi => ?_))
    · exact Nat.lt_of_lt_of_eq hp (launch_size d).symm
    · exact hz (Buf.elems_mono d.c (cBlk_subset_cIdx d (i : Nat)) hzi)

end Claims.MatmulE5m2
