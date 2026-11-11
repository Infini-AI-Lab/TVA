import Claim
import TVA.Proof
import Proof.Basic
import Proof.Index
import Proof.Lanes
import Proof.Value
import Proof.Safe

/-!
# `attn_bwd_pre_f16`: the claim

The launch is valid (`launch_valid`), each program is safe with the read set `R` and the write set
`W` of its rows (`prog_safe`: lanes are pointers into `O`, `DO`, `Delta`, with no `int32` wrap; the
products and the sum are bounded by `row_value`), the programs' write sets are pairwise disjoint
(`writeSet_disjoint`), and every row below `Z · H · N_CTX` is written by some program
(`write_cover`). `correct_of_safe` then gives `Post`.
-/

namespace Claims.AttnBwdPre

open TVA


/-- The read footprint of program `i`, as `meets_independent` wants it. -/
def Rf (d : Inst) (init : Mem) (i : Fin (launch d).size) : Set Int := R d init i

/-- The write footprint of program `i`, as `meets_independent` wants it. -/
def Wf (d : Inst) (init : Mem) (i : Fin (launch d).size) : Set Int := W d init i

/-- What program `i` leaves, as `meets_independent` wants it. -/
def Qf (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop := Q d init i m

/-- **The frame and the rows give `Post`.** If every program leaves the rows it writes as
`OutSpec` says, and every byte no program writes is unchanged, then the launch's result satisfies
`Post`. -/
theorem post_of_rows (d : Inst) (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i, ∃ m, Qf d init i m ∧ ∀ x ∈ Rf d init i ∪ Wf d init i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ Wf d init i) → fin x = init x) : Post d init fin := by
  constructor
  · intro bh m hb hm
    have hr : bh * d.N_CTX + m < d.Z * d.H * d.N_CTX := by
      have h1 : bh + 1 ≤ d.Z * d.H := by omega
      calc bh * d.N_CTX + m < bh * d.N_CTX + d.N_CTX := Nat.add_lt_add_left hm _
        _ = (bh + 1) * d.N_CTX := by ring
        _ ≤ d.Z * d.H * d.N_CTX := Nat.mul_le_mul_right _ h1
    obtain ⟨i, l, hrow⟩ := write_cover d hd hr
    obtain ⟨m', hQ, hagree⟩ := hloc i
    obtain ⟨c, hc, hspec⟩ := hQ l
    refine ⟨c, ?_, ?_⟩
    · rw [← hrow, ← hc]
      refine Mem.read_congr fun x hx => hagree x ?_
      exact Or.inr (Buf.span_sub_elems d.Delta (S := writeSet d i) (j := writeRow d i l)
        ⟨l, rfl⟩ hx)
    · intro vo vdo hvo hvdo
      rw [← hrow] at hvo hvdo
      exact hspec vo vdo hvo hvdo
  · intro z hz
    exact hfr z fun i hzi => hz (d.Delta.elems_mono (writeSet_sub d hd i) hzi)

/-- **The claim.** The launch is valid, each program is safe (`prog_safe`), the footprints are
disjoint, and the programs' rows give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (Rf d init) (Wf d init) (Qf d init) (launch_valid d hd)
    (fun i => prog_safe hd hpre ((launch d).env P) hP.sumDepth i) ?_ ?_ ?_
  · -- no program writes another's bytes
    intro i j hij
    simp only [R, W, Rf, Wf]
    refine Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_
    · exact (hd.Delta_O).mono (d.Delta.elems_mono (writeSet_sub d hd i))
        (d.O.elems_mono (readSet_sub d hd j))
    · exact (hd.Delta_DO).mono (d.Delta.elems_mono (writeSet_sub d hd i))
        (d.DO.elems_mono (readSet_sub d hd j))
    · exact d.Delta.elems_disjoint (writeSet_disjoint d hd hij)
  · -- no program reads what it writes
    intro i
    simp only [R, W, Rf, Wf]
    refine Disjoint.union_left ?_ ?_
    · exact (hd.Delta_O.symm).mono (d.O.elems_mono (readSet_sub d hd i))
        (d.Delta.elems_mono (writeSet_sub d hd i))
    · exact (hd.Delta_DO.symm).mono (d.DO.elems_mono (readSet_sub d hd i))
        (d.Delta.elems_mono (writeSet_sub d hd i))
  · -- the programs' results give `Post`
    intro fin hloc hfr
    exact post_of_rows d hd hloc hfr

end Claims.AttnBwdPre
