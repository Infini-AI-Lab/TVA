import Proof.Basic
import Proof.Draft
import Proof.Tail
import Proof.Program
import Proof.Post

/-!
# `combine_draft_tokens_i32`: `_combine_sampled_and_draft_tokens_kernel`

Program `b` serves request `b`: it reads its entries of the inputs, writes its logits indices
and, past its prefill, its last sampled token and its drafts into its query slots of
`input_ids`. The proof:

1. `Proof.Program`: one program is safe (`safe_program`): the loads and the logits indices, then
   `Proof.Tail` (the prefill test and the last sampled token) and `Proof.Draft` (the drafts);
2. `Proof.Post`: no program writes an input, distinct programs write disjoint elements, and the
   programs' results give `Post`;
3. `main`: by `correct_of_safe`.
-/

namespace Claims.CombineDraftTokens

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hlt : ∀ i : Fin (launch d).size, (i : Nat) < d.B := fun i =>
    lt_of_lt_of_eq i.isLt (launch_size d)
  refine correct_of_safe (fun _ => R d) (fun i => W d init (i : Nat))
    (fun i m => Q d init (i : Nat) m) (launch_valid hd) (fun i => safe_program hd hpre _ i)
    (fun i j hij => ?_) (fun i => R_disj_W hd hpre (hlt i))
    (fun fin hloc hfr => post_of hpre fin hloc hfr)
  refine Disjoint.union_right (R_disj_W hd hpre (hlt i)).symm ?_
  exact W_disj hd hpre (hlt i) (hlt j) (fun h => hij (Fin.ext h))

end Claims.CombineDraftTokens
