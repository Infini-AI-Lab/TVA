import Claim
import TVA.Proof
import Proof.Consts
import Proof.Lane
import Proof.Exp
import Proof.Block
import Proof.Final
import Proof.Defs
import Proof.Addr
import Proof.Walk
import Proof.Program

/-!
# `logsumexp_fwd_f32`: the proof

The route is `correct_of_safe`, one program of the grid `(N, ND)` at a time.

1. `Proof.Defs`: program `p` handles row `pn p` and block `pi p`; it reads the block's elements of
   `x` (`Rf`), writes one element of `z` (`Wf`) and leaves the value `Post` asks for (`Qf`).
2. `Proof.Addr`: the `int64` index arithmetic (no wrap) and the pointers.
3. The numerics, per block: `Proof.Lane` (the scale's product, the subtraction of a valid and of
   a masked lane), `Proof.Exp` (`tl.exp` of a lane, the block's max), `Proof.Block` (the sum along
   a tree of depth at most `16` and its `tl.log`: within `100u` of `log Σ e^(s_l)`),
   `Proof.Final` (the claim's tolerance, without and with a scale).
4. `Proof.Walk`, `Proof.Program`: each program is `Safe`, one `tl.*` operation at a time.
5. `main`: the launch is valid, the footprints are disjoint, and the programs' results give
   `Post`.
-/

namespace Claims.LogsumexpFwd

open TVA

/-- The program of row `n`, block `i`, as an element of the grid. -/
def progOf (d : Inst) {n i : Nat} (hn : n < d.N) (hi : i < ND d) : Fin (launch d).size :=
  ⟨n + d.N * i, by rw [launch_size]; exact (pid_of hn hi).1⟩

/-- A program's write footprint lies among `z`'s written elements. -/
theorem Wf_sub {d : Inst} (p : Fin (launch d).size) :
    Wf d (pn d p) (pi d p) ⊆ d.z.elems (zSet d) := by
  have hp : (p : Nat) < d.N * ND d := lt_of_lt_of_eq p.isLt (launch_size d)
  exact Buf.elems_mono _ (zBlk_sub (pn_lt hp) (pi_lt hp))

/-- A program's read footprint lies among `x`'s read elements. -/
theorem Rf_sub {d : Inst} (p : Fin (launch d).size) :
    Rf d (pn d p) (pi d p) ⊆ d.x.elems (xSet d) := by
  have hp : (p : Nat) < d.N * ND d := lt_of_lt_of_eq p.isLt (launch_size d)
  exact Buf.elems_mono _ (xBlk_sub (pn_lt hp))

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun p => Rf d (pn d p) (pi d p)) (fun p => Wf d (pn d p) (pi d p))
    (fun p m => Qf d init (pn d p) (pi d p) m) (launch_valid hd)
    (fun p => safe_program hP hd hpre p) ?_ ?_ ?_
  · -- distinct programs: disjoint writes, and no write meets another's reads
    intro p q hpq
    have hp : (p : Nat) < d.N * ND d := lt_of_lt_of_eq p.isLt (launch_size d)
    have hq : (q : Nat) < d.N * ND d := lt_of_lt_of_eq q.isLt (launch_size d)
    refine Disjoint.union_right ?_ ?_
    · exact (hd.z_x.mono (Wf_sub p) (Rf_sub q))
    · refine Buf.elems_disjoint _ ?_
      rw [Set.disjoint_singleton]
      intro he
      obtain ⟨h1, h2⟩ := zIdx_inj (pi_lt hp) (pi_lt hq) he
      exact hpq (Fin.ext (pid_inj hp hq h1 h2))
  · intro p
    exact (hd.z_x.mono (Wf_sub p) (Rf_sub p)).symm
  · intro fin hloc hfr
    refine ⟨fun n hn i hi => ?_, fun b hb => hfr b fun p hbp => hb (Wf_sub p hbp)⟩
    obtain ⟨hpl, hpn, hpi⟩ := pid_of hn hi
    obtain ⟨m, hQ, hag⟩ := hloc (progOf d hn hi)
    have hpn' : pn d (progOf d hn hi) = n := hpn
    have hpi' : pi d (progOf d hn hi) = i := hpi
    rw [hpn', hpi'] at hQ
    rw [hpn', hpi'] at hag
    refine ⟨xsOf d init n, fun j hj => ?_, ?_⟩
    · have hlt : xIdx d n j < d.x.len := lt_of_lt_of_le (xIdx_lt hn hj) hd.x_len
      obtain ⟨v, hv⟩ := Buf.read_some d.x hpre.1 hlt
      simp only [xsOf]
      rw [hv]
      rfl
    · obtain ⟨c, hc, hok⟩ := hQ
      refine ⟨c, ?_, hok⟩
      rw [← hc]
      exact Mem.read_congr fun x hx => hag x (Or.inr ⟨_, rfl, hx⟩)

end Claims.LogsumexpFwd
