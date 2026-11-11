import Proof.Walk

/-!
# The launch

Footprints of the programs, their disjointness, and `main`: `correct_of_safe` applied to
`safe_prog`.
-/

namespace Claims.MoeSumReduce

open TVA

/-- The bytes program `i` reads. -/
def R (d : Inst) (i : Nat) : Set Int :=
  Rin d (((launch d).ctx i).pid 0) (((launch d).ctx i).pid 1)

/-- The bytes program `i` writes. -/
def W (d : Inst) (i : Nat) : Set Int :=
  Wout d (((launch d).ctx i).pid 0) (((launch d).ctx i).pid 1)

/-- What program `i` leaves in memory. -/
def Q (d : Inst) (i : Nat) (m : Mem) : Prop :=
  Qout d (((launch d).ctx i).pid 0) (((launch d).ctx i).pid 1) m

/-- Every element `rset` names is an element of the input buffer. -/
theorem rset_lt {d : Inst} (hd : d.ok) {p0 p1 : Nat} {j : Nat} (hj : j ∈ rset d p0 p1) :
    j < d.input.len := by
  obtain ⟨t, ht, i, hi, hie⟩ := hj
  rw [← hie]
  exact inIdx_lt hd i hi ht

/-- Every element `wtile` names is an element of the output buffer. -/
theorem wtile_lt {d : Inst} (hd : d.ok) {p0 p1 : Nat} (hp0 : p0 < gT d) (hp1 : p1 < gD d)
    {j : Nat} (hj : j ∈ wtile d p0 p1) : j < d.output.len := by
  have h : j < d.T * d.D := wtile_subset hd hp0 hp1 hj
  have h2 := hd.output_len
  omega

/-- **The claim.** Each program is safe (`safe_prog`), their footprints are pairwise disjoint
(`wtile_disjoint`, and `hd.disj` for the input), and their results give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => R d i) (fun i => W d i) (fun i m => Q d i m)
    (launch_valid hd) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro i
    refine (body_eq d _).symm ▸ ?_
    exact safe_prog d hd hpre ((launch d).ctx (i : Nat))
      (launch_pid0_lt hd (i : Nat)) (launch_pid1_lt hd (i : Nat))
  · -- no program writes another program's footprint
    intro i j hij
    have hne : ((launch d).ctx (i : Nat)).pid 0 ≠ ((launch d).ctx (j : Nat)).pid 0 ∨
        ((launch d).ctx (i : Nat)).pid 1 ≠ ((launch d).ctx (j : Nat)).pid 1 := by
      by_contra h
      push Not at h
      exact hij (Fin.ext (pid_inj hd i.isLt j.isLt h.1 h.2))
    refine Disjoint.union_right ?_ ?_
    · exact Buf.elems_disjoint_of d.output d.input hd.disj
        (fun e he => wtile_lt hd (launch_pid0_lt hd (i : Nat)) (launch_pid1_lt hd (i : Nat)) he)
        (fun e he => rset_lt hd he)
    · exact Buf.elems_disjoint d.output
        (wtile_disjoint hd (launch_pid0_lt hd (i : Nat)) (launch_pid1_lt hd (i : Nat))
          (launch_pid0_lt hd (j : Nat)) (launch_pid1_lt hd (j : Nat)) hne)
  · -- no program reads what it writes
    intro i
    exact Buf.elems_disjoint_of d.input d.output hd.disj.symm
      (fun e he => rset_lt hd he)
      (fun e he => wtile_lt hd (launch_pid0_lt hd (i : Nat)) (launch_pid1_lt hd (i : Nat)) he)
  · -- the programs' results give `Post`
    intro fin hloc hfr
    refine ⟨?_, ?_⟩
    · -- every output element holds what its program left there
      intro t j ht hj
      have hM := block_m_pos hd
      have hD := block_d_pos hd
      obtain ⟨m, hQm, hagree⟩ := hloc ⟨t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM), prog_lt hd ht hj⟩
      have hpid0 := prog_pid0 (d := d) hd ht hj
      have hpid1 := prog_pid1 (d := d) hd ht hj
      unfold Q at hQm
      rw [hpid0, hpid1] at hQm
      obtain ⟨c, hc, hcin⟩ := hQm t j ht hj rfl rfl
      refine ⟨c, ?_, hcin⟩
      have hin : (t * d.D + j) ∈ wtile d (t / d.BLOCK_M) (j / d.BLOCK_DIM) :=
        ⟨ownerLane d t j hM hD, owner_en hd ht hj, owner_out hd ht hj⟩
      have hW : Mem.span .bf16 (d.output.ptr ((t * d.D + j : Nat) : Int)).addr ⊆
          W d (t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM)) := by
        intro x hx
        show x ∈ Wout d (((launch d).ctx (t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM))).pid 0)
          (((launch d).ctx (t / d.BLOCK_M + gT d * (j / d.BLOCK_DIM))).pid 1)
        rw [hpid0, hpid1]
        exact Buf.span_sub_elems d.output hin hx
      rw [← hc]
      exact Mem.read_congr (d := .bf16)
        (a := (d.output.ptr ((t * d.D + j : Nat) : Int)).addr)
        (fun x hx => hagree x (Or.inr (hW hx)))
    · -- every other byte is unchanged
      intro z hz
      refine hfr z fun i hzi => hz ?_
      show z ∈ d.output.elems {e | e < d.T * d.D}
      exact d.output.elems_mono
        (wtile_subset hd (launch_pid0_lt hd (i : Nat)) (launch_pid1_lt hd (i : Nat))) hzi

end Claims.MoeSumReduce
