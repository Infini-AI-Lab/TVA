import Proof.Basic
import Proof.Trip
import Proof.Copy
import Proof.Program
import Proof.Disj
import Proof.Post

/-!
# `write_req_to_token_pool_i32`: `write_req_to_token_pool_triton`

Program `b` serves request `b`: it reads its entries of the index buffers and of the prefix
pointers, `extend_lens[0 … b - 1]`, its prefix tensor and its window of `out_cache_loc`, and
writes its row's first `seq_lens[b]` elements of the table. The proof:

1. `Proof.Trip`: one trip of a copy loop (`safe_trip`), for any source buffer and table window;
2. `Proof.Copy`: a whole copy loop, `tl.cdiv` and the `range` (`safe_copy`);
3. `Proof.Program`: the scalar loads, the prefix loop, the running sum of `extend_lens` and the
   loop of the new locations (`safe_program`);
4. `main`: by `correct_of_safe`, with `Proof.Disj` (read footprints miss the table, distinct
   programs write distinct rows) and `Proof.Post` (the programs' results give `Post`).
-/

namespace Claims.WriteReqToTokenPool

open TVA

/-- **The claim.** Each program is safe on its own row (`safe_program`), the footprints are
disjoint, and the programs' results give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hb : ∀ i : Fin (launch d).size, (i : Nat) < d.bs :=
    fun i => lt_of_lt_of_eq i.isLt (launch_size d)
  refine correct_of_safe (fun i => R d init (i : Nat)) (fun i => W d init (i : Nat))
    (fun i m => Q d init (i : Nat) m) (launch_valid hd) (fun i => safe_program hd hpre _ i)
    (fun i j hij => ?_) (fun i => R_disj_W hd hpre (hb i))
    (fun fin hloc hfr => post_of hpre fin hloc hfr)
  refine Disjoint.union_right ?_ (W_disj_W hpre (hb i) (hb j) (fun h => hij (Fin.ext h)))
  exact (R_disj_table hd hpre (hb j)).mono_left (W_sub_bytes hpre (hb i))

end Claims.WriteReqToTokenPool
