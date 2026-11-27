import Proof.Basic
import Proof.Loop
import Proof.Store
import Proof.Pad
import Proof.Lane
import Proof.Trip
import Proof.Req
import Proof.Post

/-!
# `slot_mappings_i64`: `_compute_slot_mappings_kernel`

Program `i` of the grid is `(g, b) = (i % G, i / G)`. The proof:

1. `Proof.Pad`: the padding program `(g, R)` stores `-1` over `[qsl R, N)` (`safe_padProg`);
2. `Proof.Lane`, `Proof.Trip`, `Proof.Req`: a request's program `(g, b)` stores each token's slot
   (`safe_reqProg`): the local positions (`safe_cpSec`), one trip of the loop (`safe_tripProg`),
   the loop (`Proof.Loop`, `safe_tokLoop`) and the scalars before it;
3. `main`: by `correct_of_safe`, every program reading only input bytes (`Rall`), the write
   footprints of distinct programs disjoint (`Proof.Post`), and the programs' stores giving `Post`.
-/

namespace Claims.SlotMappings

open TVA

/-- Program `i`'s body is the padding program or a request's program. -/
theorem body_eq (d : Inst) (i : Nat) :
    (model d).body ((launch d).ctx i) =
      if IntV.eq (IntV.ofNat (i / d.G % (d.R + 1)) : I32)
          ((IntV.ofNat (d.R + 1) : I32) - (IntV.ofInt 1 : I32)) then
        padProg d (i % d.G) (i / d.G % (d.R + 1))
      else reqProg d (i % d.G) (i / d.G % (d.R + 1)) := by
  simp only [model, _compute_slot_mappings_kernel, tl.program_id, tl.num_programs, launch_pid0,
    launch_pid1, launch_grid1]
  rfl

/-- **One program** of the grid is safe from `init`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) :
    Safe E (Rall d) (Wp d init ((i : Nat) % d.G) ((i : Nat) / d.G))
      (fun _ m => Qp d init ((i : Nat) % d.G) ((i : Nat) / d.G) m) init
      ((model d).body ((launch d).ctx i)) := by
  obtain ⟨hg, hb⟩ := prog_lt hd i
  have hR := hd.R_le
  rw [body_eq, Nat.mod_eq_of_lt hb]
  have hcond : IntV.eq (IntV.ofNat ((i : Nat) / d.G) : I32)
      ((IntV.ofNat (d.R + 1) : I32) - (IntV.ofInt 1 : I32)) =
      decide ((i : Nat) / d.G = d.R) := by
    have h1 : (i : Nat) / d.G < 2 ^ 31 := by omega
    have h2 : d.R + 1 < 2 ^ 31 := by omega
    have h3 : (IntV.ofInt 1 : I32).toInt = ((1 : Nat) : Int) := IntV.I32.toInt_ofInt (by norm_num)
    rw [IntV.eq_eq, toInt_ofNat32 h1,
      IntV.I32.sub_nat (toInt_ofNat32 h2) h3 (by omega) h2]
    simp only [Nat.add_sub_cancel, Nat.cast_inj]
  rw [hcond]
  by_cases hbR : (i : Nat) / d.G = d.R
  · simp only [hbR, decide_true, ite_true]
    exact safe_padProg hd hpre E hg
  · simp only [hbR, decide_false, Bool.false_eq_true, ite_false]
    exact safe_reqProg hd hpre E hg (by omega)

/-- Program `b * G + g` is `(g, b)`. -/
theorem prog_of {d : Inst} (hd : d.ok) {g b : Nat} (hg : g < d.G) (hb : b ≤ d.R) :
    b * d.G + g < (launch d).size ∧ (b * d.G + g) % d.G = g ∧ (b * d.G + g) / d.G = b := by
  have hG := hd.G_pos
  refine ⟨?_, ?_, ?_⟩
  · rw [launch_size]
    have : b * d.G + g < (b + 1) * d.G := by rw [Nat.add_mul]; omega
    have : (b + 1) * d.G ≤ (d.R + 1) * d.G := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm d.G]; omega
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hg]
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hg, zero_add]

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hRD := Rall_disjoint hd
  refine correct_of_safe (fun _ => Rall d)
    (fun i => Wp d init ((i : Nat) % d.G) ((i : Nat) / d.G))
    (fun i m => Qp d init ((i : Nat) % d.G) ((i : Nat) / d.G) m) (launch_valid hd)
    (fun i => safe_program hd hpre _ i) ?_ ?_ ?_
  · intro i j hij
    obtain ⟨hgi, hbi⟩ := prog_lt hd i
    obtain ⟨hgj, hbj⟩ := prog_lt hd j
    refine Disjoint.union_right ?_ ?_
    · exact Set.disjoint_of_subset_left (Wp_sub hd hpre hgi (by omega)) hRD
    · refine Wp_disjoint hd hpre (by omega) (by omega) ?_
      by_contra h
      push Not at h
      apply hij
      apply Fin.ext
      rw [← Nat.mod_add_div (i : Nat) d.G, ← Nat.mod_add_div (j : Nat) d.G, h.1, h.2]
  · intro i
    obtain ⟨hgi, hbi⟩ := prog_lt hd i
    exact (Set.disjoint_of_subset_left (Wp_sub hd hpre hgi (by omega)) hRD).symm
  · intro fin hloc hfr
    -- the element `(g, t)` of program `(g, b)` reads what that program left
    have hread : ∀ g b (t : Nat), g < d.G → b ≤ d.R → qsl d init b ≤ (t : Int) → (t : Int) < hiB d init b →
        fin.read .i64 (d.slot_mappings.ptr ((g * d.S + t : Nat) : Int)).addr =
          some (val d init g b t) := by
      intro g b t hg hb h1 h2
      obtain ⟨hlt, hmod, hdiv⟩ := prog_of hd hg hb
      obtain ⟨m, hQ, hag⟩ := hloc ⟨b * d.G + g, hlt⟩
      simp only [hmod, hdiv] at hQ hag
      rw [Mem.read_congr fun x hx => hag x (Or.inr ⟨g * d.S + t, ⟨t, h1, h2, rfl⟩, hx⟩)]
      exact hQ t h1 h2
    refine ⟨fun g hg b hb t h1 h2 => ?_, fun g hg t h1 h2 => ?_, fun z hz => hfr z fun i hi => ?_⟩
    · have := hread g b t hg hb.le h1 (by simp only [hiB, hb, ite_true]; exact h2)
      simpa only [val, hb, ite_true] using this
    · have := hread g d.R t hg (le_refl _) h1 (by simp only [hiB, lt_irrefl, ite_false]; exact_mod_cast h2)
      simpa only [val, lt_irrefl, ite_false] using this
    · obtain ⟨hgi, hbi⟩ := prog_lt hd i
      exact hz (Wp_sub_outW hpre hgi (by omega) hi)

end Claims.SlotMappings
