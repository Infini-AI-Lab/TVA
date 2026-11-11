import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.GeoBound
import Proof.GeoCover
import Proof.GeoDisj

/-!
# The programs' results give the postcondition

Every written element is written by an enabled lane of some program (`cover_exists`), and outside
the written elements nothing changes.
-/

namespace Claims.ChunkCumsum

open TVA

/-- **The programs' results give `Post`**: each lane's program left its value, every other byte is unchanged, and the coverage lemmas say every position is some lane. -/
theorem main_post {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m : Mem,
      Q d (i : Nat) m ∧ ∀ x ∈ R d (i : Nat) ∪ W d (i : Nat), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d (i : Nat)) → fin x = init x) :
    Post d init fin := by
  unfold Post
  constructor
  · intro n p h hn hp1 hp2 hh
    obtain ⟨i, l, hig, hl, he, hn', hp', hh'⟩ := cover_exists hd hT hn hp1 hp2 hh
    have hfin : i < (launch d).size := by rw [launch_size]; exact hig
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨i, hfin⟩
    simp only [] at hQ hagree
    obtain ⟨v, hread, hprop⟩ := hQ l hl he
    -- the lane's element is the element of the postcondition
    have helem : elem d i l = p * d.H + h := by
      unfold elem
      rw [hp', hh']
    -- the lane's element is one that the program writes
    have hmem : elem d i l ∈ Buf.img (elemF d i) (enF d i) := by
      refine ⟨(⟨l, hl⟩, ()), ?_, rfl⟩
      simp only [enF]
      exact he
    refine ⟨v, ?_, ?_⟩
    · -- `fin` and `m` agree on the bytes read
      have helemI : (↑(elem d i l) : Int) = ↑p * ↑d.H + ↑h := by
        rw [helem]
        push_cast
        ring
      rw [← helemI]
      refine (Mem.read_congr (m₁ := fin) (m₂ := m) ?_).trans hread
      intro x hx
      exact hagree x (Or.inr (Buf.span_sub_elems d.o hmem hx))
    · -- the value is the scan's
      intro hIR
      rw [← hn', ← hp', ← hh'] at hIR
      have := hprop hIR
      rwa [hn', hp', hh'] at this
  · intro z hz
    exact hfr z fun i hzi =>
      hz (W_subset_written hd hT (lt_of_lt_of_eq i.isLt (launch_size d)) hzi)

end Claims.ChunkCumsum
