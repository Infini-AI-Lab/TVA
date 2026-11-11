import Claim
import TVA.Proof
import Proof.Safe
import Proof.Post

/-!
# The proof of `kv_cache_write_bf16`

Program `i` of the grid handles one tile of one token's row: it reads the token's slot, then its
tile of `key` and `value`, and writes the matching elements of the two caches. The programs write
disjoint cache elements and no program reads what any writes, so `correct_of_safe` composes them:

1. `Proof.Basic`, `Proof.Lanes`, `Proof.Idx`, `Proof.Tgt`: the model's body, each lane's
   position, source element and cache element, and the `int32`/`int64` offsets not wrapping;
2. `Proof.BodyBf16`, `Proof.BodyFp8`, `Proof.BodyFp8in`: one program is `Safe` in each mode
   (`safe_body`, in `Proof.Safe`, dispatches on the mode);
3. `Proof.Disj`: the footprints: `R_disjoint_W` and `W_disjoint` (distinct slots and distinct
   lanes address distinct cache elements);
4. `Proof.Post`: the programs' results give `Post`;
5. `main`: the launch is valid, and 2-4 are the premises of `correct_of_safe`.
-/

namespace Claims.KvCacheWrite

open TVA

/-- The claim holds: every valid launch of the kernel satisfies the postcondition, via the safety-based correctness criterion. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun _ => R d) (fun i => W d init (i : Nat)) (fun i m => Q d init (i : Nat) m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · exact safe_body hd hpre ((launch d).env P) (hP.divTotal _) hP.divFull (hP.cvtTotal _ _ _) hP.cvtSatF32E4m3 i
  · intro i j hij
    exact Set.disjoint_union_right.mpr
      ⟨(R_disjoint_W hd hpre i).symm, W_disjoint hd hpre hij⟩
  · intro i
    exact R_disjoint_W hd hpre i
  · intro fin hQ hframe
    exact post_of hd hQ hframe

end Claims.KvCacheWrite
