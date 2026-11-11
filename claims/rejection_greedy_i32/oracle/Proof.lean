import Proof.Run

/-!
# The proof of `rejection_greedy_i32`

Program `r` of the grid is request `r`. It reads only the input buffers (`Rf`), writes only its
own row of the output (`Wf d r`), and leaves there the row of `expect` (`Qf`), so different rows
share no byte and `correct_of_safe` composes the programs:

1. `Proof.Basic`: the body split into the parts the proof works on, and the footprints;
2. `Proof.Diff`, `Proof.Arith`: the first mismatch, the bounds on `cu` and the addresses;
3. `Proof.Inv`: the loop invariant, one trip at a time;
4. `Proof.Run`: `safe_program`, one program walked line by line;
5. `main`: the launch is valid, the rows are disjoint from each other and from the inputs, and
   the rows give `Post`.
-/

namespace Claims.RejectionGreedy

open TVA

/-- **The claim.** Each program is safe on its own row (`safe_program`); the footprints are
disjoint, and together the rows give `Post`. -/
theorem main : claim.Holds := by
  intro P _ (d : Inst) hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun _ => Rf d) (fun i => Wf d i) (fun i m => Qf d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · -- program `i`
    rw [body_eq, launch_pid]
    exact safe_program hd hpre _ (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- a program's row shares no byte with what another program reads or writes
    intro i j hij
    have hiB : (i : Nat) < d.B := lt_of_lt_of_eq i.isLt (launch_size d)
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (Wf_disjoint_Rf hd hiB) (Wf_disjoint hij')
  · -- a program's reads share no byte with its writes
    intro i
    have hiB : (i : Nat) < d.B := lt_of_lt_of_eq i.isLt (launch_size d)
    exact (Wf_disjoint_Rf hd hiB).symm
  · -- what the programs leave is `Post`
    intro fin hloc hfr
    refine ⟨fun r hr j hj => ?_, fun z hz => ?_⟩
    · have hi : r < (launch d).size := by rw [launch_size]; exact hr
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨r, hi⟩
      have hread : fin.read .i32 (d.out.ptr (oidx d r j)).addr =
          m.read .i32 (d.out.ptr (oidx d r j)).addr :=
        Mem.read_congr fun x hx => hagree x (Or.inr (Wt_sub_Wf hj hx))
      rw [hread]
      exact hQ j hj
    · refine hfr z fun i hzi => hz (d.out.elems_mono (fun j hj => rowSet_sub ?_ hj) hzi)
      exact lt_of_lt_of_eq i.isLt (launch_size d)

end Claims.RejectionGreedy
