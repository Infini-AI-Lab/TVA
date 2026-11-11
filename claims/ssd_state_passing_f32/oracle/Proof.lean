import Claim
import TVA.Proof
import Proof.Post

/-!
# The proof of `ssd_state_passing_f32`

Program `(pm, h)` of the grid `(cdiv(dim, 1024), 1, nheads)` carries the state of the `1024` lanes
`pm·1024 + i` of head `h` across the chunks, in slots: slot `c` is the state entering chunk `c`,
stored in `out`'s chunk `c`, and slot `nchunks` the final state. The proof:

1. `Proof.Meta`: the host's metadata (`seq_idx`, `chunk_offsets`) from `query_start_loc`: the
   request ending each chunk, the logical chunk index the kernel computes, and the offset it reads;
2. `Proof.Num`: the numerics of one chunk, from the device fact for `tl.exp` and the IEEE
   operations, against the claim's tolerance `tol`;
3. `Proof.Lanes`, `Proof.Inv`: the lanes and slots a program owns, their footprints, and the
   invariant of the loop (`InvI` / `InvN`);
4. `Proof.Tail`, `Proof.TripI`, `Proof.TripN`: one trip walked line by line with the rules of
   `TVA.Proof.Rules`, with and without initial states, restoring the invariant;
5. `Proof.Loop`, `Proof.Walk`: the loop by `safe_forRange`, and the loads and stores before it:
   each program is `Safe`, reads only inputs, writes only its slots, and ends with every slot done;
6. `Proof.Grid`, `Proof.Post`, and `main`: distinct programs write disjoint slots, and the slots
   give `Post`.
-/
namespace Claims.SsdStatePassing

open TVA

/-- **The claim.** Each program is safe on its own slots (`safe_program_I` / `safe_program_N`);
distinct programs write disjoint bytes, none writes an input, and the slots give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hg := gx_pos hd
  have hne : ∀ i j : Fin (launch d).size, i ≠ j →
      ((i : Nat) / gx d, (i : Nat) % gx d) ≠ ((j : Nat) / gx d, (j : Nat) % gx d) := by
    intro i j hij hc
    obtain ⟨h1, h2⟩ := Prod.mk.inj hc
    apply hij
    apply Fin.ext
    have e1 := Nat.div_add_mod (i : Nat) (gx d)
    have e2 := Nat.div_add_mod (j : Nat) (gx d)
    rw [h1, h2] at e1
    omega
  refine correct_of_safe (fun _ => Rin d) (fun i => Wp d ((i : Nat) / gx d) ((i : Nat) % gx d))
    (fun i m => Done d init ((i : Nat) % gx d) ((i : Nat) / gx d) d.nchunks m) (launch_valid hd)
    (fun i => ?_) (fun i j hij => ?_) (fun i => ?_) (fun fin hloc hfr => ?_)
  · obtain ⟨p0, p1, p2⟩ := pid_eq d i
    have hpm : (i : Nat) % gx d < gx d := Nat.mod_lt _ hg
    cases hi : d.hasInit
    · exact safe_program_N hd hP hpre hi hpm (grid_h_lt i) p0 p1 p2
    · exact safe_program_I hd hP hpre hi hpm (grid_h_lt i) p0 p1 p2
  · exact Disjoint.union_right (Wp_disjoint_Rin hd (grid_h_lt i)).symm.symm
      (Wp_disjoint hd (grid_h_lt i) (grid_h_lt j) (hne i j hij))
  · exact (Wp_disjoint_Rin hd (grid_h_lt i)).symm
  · refine post_of_done hd (fun i => ?_) fun x hx => hfr x (fun i => hx i)
    obtain ⟨mem, hq, hag⟩ := hloc i
    exact ⟨mem, hq, hag⟩

end Claims.SsdStatePassing
