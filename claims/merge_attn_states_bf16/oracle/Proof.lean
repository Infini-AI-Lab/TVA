import Claim
import TVA.Proof
import Proof.Defs
import Proof.Prog
import Proof.Safe
import Proof.Disj
import Proof.Post

/-!
# `merge_attn_states_bf16`

The grid is `T × H` single-block programs, one per `(token, head)`. Program `(t, h)` reads the two
lse scalars and the two `bfloat16` rows of its `(t, h)`, writes its row of `output` (and, when the
merged lse is passed, its element of `output_lse`), and leaves either the merged or the copied
values there. The proof:

1. `Proof.Defs`, `Proof.Prog`: the grid, the footprints `Rf`/`Wf`, the two branches of the kernel as
   programs (`mergeProg`, `copyProg`);
2. `Proof.Addr`: the kernel's `int32` pointer arithmetic gives the buffers' elements (no wrap);
3. `Proof.Copy`, `Proof.Merge`: the copy branch and the merge branch are safe, with their results,
   using the numerical bounds of `Proof.Num*` (the device facts for `exp`, `log`, `/`);
4. `Proof.Safe`: one program is safe, by the branch on `token < prefill_tokens_with_context`;
5. `Proof.Disj`: the footprints of distinct programs are disjoint;
6. `Proof.Post`: the local results give `Post`; `main` assembles them with `correct_of_safe`.
-/

namespace Claims.MergeAttnStates

open TVA

/-- **The claim**: the launch is valid, every program is safe on its own footprints, the footprints
are disjoint, and the programs' local results give `Post`. -/
theorem main : claim.Holds := by
  intro P hP (d : Inst) hd init hpre
  refine correct_of_safe (fun i => Rq d init i) (fun i => Wq d init i) (fun i m => Qq d init i m)
    (launch_valid hd) (fun i => safe_program hP hd hpre i)
    (fun i j hij => disj_write hd init i j hij) (fun i => disj_read_write hd init i)
    (fun fin hloc hfr => post_of_local hd hpre hloc hfr)

end Claims.MergeAttnStates
