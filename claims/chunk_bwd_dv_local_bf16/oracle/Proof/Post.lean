import Claim
import TVA.Proof
import Proof.Basic
import Proof.GeoCover
import Proof.GeoDisj

/-!
# The programs' results give the post-condition

Every token of every sequence and every value head is an enabled lane of some program
(`cover_exists`), and outside the written elements nothing changes.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- Program `r + nChunks · bh` is program `(r, bh)`. -/
theorem prog_index {d : Inst} {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d) :
    (r + nChunks d * bh) % nChunks d = r ∧ (r + nChunks d * bh) / nChunks d % nRows d = bh := by
  have hn : 0 < nChunks d := by omega
  refine ⟨?_, ?_⟩
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]
  · rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt hr, Nat.zero_add, Nat.mod_eq_of_lt hbh]

/-- A program's written elements are elements `Post` lists as written. -/
theorem wSet_subset_written {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d) : wSet d r bh ⊆ written d := by
  rintro x ⟨l, e, hl, he, hev, rfl⟩
  exact ⟨nOf d r bh, pos d r bh l, hvOf d bh, e, nOf_lt hd hT hr hbh, lo_le_pos d r bh l,
    pos_lt_hi hT (nOf_lt hd hT hr hbh) he, hvOf_lt hd bh, hev, rfl⟩

/-- **The programs' results give `Post`**: each lane's program left its value, every other byte
is unchanged, and the coverage lemma says every token is some lane. -/
theorem main_post {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m : Mem,
      Q d init (i % nChunks d) (i / nChunks d % nRows d) m ∧
        ∀ x ∈ R d (i % nChunks d) (i / nChunks d % nRows d) ∪
          W d (i % nChunks d) (i / nChunks d % nRows d), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size,
        x ∉ W d (i % nChunks d) (i / nChunks d % nRows d)) → fin x = init x) :
    Post d init fin := by
  refine ⟨?_, ?_⟩
  · intro n p hv e hn hp1 hp2 hhv he
    obtain ⟨r, bh, l, hr, hbh, hl, hen, hn', hp', hh'⟩ := cover_exists hd hT hn hp1 hp2 hhv
    have hlt : r + nChunks d * bh < (launch d).size := by
      rw [launch_size]
      have : r + nChunks d * bh < nChunks d * nRows d := by
        calc r + nChunks d * bh < nChunks d + nChunks d * bh := by omega
          _ = nChunks d * (bh + 1) := by ring
          _ ≤ nChunks d * nRows d := Nat.mul_le_mul_left _ hbh
      exact this
    obtain ⟨hi1, hi2⟩ := prog_index hr hbh
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨r + nChunks d * bh, hlt⟩
    simp only [hi1, hi2] at hQ hagree
    obtain ⟨c, hread, hfin, herr⟩ := hQ l hl hen e he
    have hel : wEl d r bh l e = oIdx d p hv e := by
      unfold wEl; rw [hp', hh']
    rw [hel] at hread
    rw [hn', hp', hh'] at herr
    refine ⟨c, ?_, hfin, herr⟩
    have hmem : oIdx d p hv e ∈ wSet d r bh := ⟨l, e, hl, hen, he, hel.symm⟩
    unfold rdB at hread ⊢
    refine (Mem.read_congr (m₁ := fin) (m₂ := m) ?_).trans hread
    intro x hx
    exact hagree x (Or.inr (Buf.span_sub_elems d.dv hmem hx))
  · intro z hz
    refine hfr z fun i hzi => hz ?_
    have hs : (i : Nat) < nChunks d * nRows d := lt_of_lt_of_eq i.isLt (launch_size d)
    have hi : (i : Nat) % nChunks d < nChunks d :=
      Nat.mod_lt _ (Nat.pos_of_ne_zero fun h => by simp [h] at hs)
    have hj : (i : Nat) / nChunks d % nRows d < nRows d :=
      Nat.mod_lt _ (Nat.pos_of_ne_zero fun h => by simp [h] at hs)
    exact Buf.elems_mono d.dv (wSet_subset_written hd hT hi hj) hzi

end Claims.ChunkBwdDvLocal
