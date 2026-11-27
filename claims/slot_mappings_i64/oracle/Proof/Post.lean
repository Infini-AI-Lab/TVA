import Proof.Req

/-!
# `slot_mappings_i64`: the footprints of distinct programs, and `Post`

Programs `(g, b)` and `(g', b')` write the elements `(g, t)` for `t` in `[qsl b, hiB b)`:
distinct groups write distinct rows (`t < N ≤ S`), and the ranges of distinct `b` are disjoint
because `query_start_loc` is nondecreasing. Every program reads input bytes only, which the output
does not share.
-/

namespace Claims.SlotMappings

open TVA

/-- The ranges of `b < b'` are ordered. -/
theorem hiB_le_qsl {d : Inst} {init : Mem} (hpre : Pre d init) {b b' : Nat} (hbb : b < b')
    (hb' : b' ≤ d.R) : hiB d init b ≤ qsl d init b' := by
  have hb : b < d.R := by omega
  simp only [hiB, hb, ite_true]
  exact qsl_mono hpre (by omega) hb'

/-- The write footprints of programs `(g, b) ≠ (g', b')` are disjoint. -/
theorem Wp_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {g b g' b' : Nat}
    (hb : b ≤ d.R) (hb' : b' ≤ d.R) (hne : g ≠ g' ∨ b ≠ b') :
    Disjoint (Wp d init g b) (Wp d init g' b') := by
  refine Buf.elems_disjoint _ (Set.disjoint_left.mpr ?_)
  rintro j ⟨t, ht1, ht2, rfl⟩ ⟨t', ht1', ht2', hj⟩
  have htN : (t : Int) < d.N := lt_of_lt_of_le ht2 (range_ok hpre hb).2
  have htN' : (t' : Int) < d.N := lt_of_lt_of_le ht2' (range_ok hpre hb').2
  have hS := hd.N_S
  have htS : t < d.S := by omega
  have htS' : t' < d.S := by omega
  have hgg : g = g' := by
    have h1 : (g * d.S + t) / d.S = g := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt htS, zero_add]
    have h2 : (g' * d.S + t') / d.S = g' := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt htS', zero_add]
    rw [← h1, ← h2, hj]
  subst hgg
  have htt : t = t' := by omega
  subst htt
  have hbb : b ≠ b' := by rcases hne with h | h; exact absurd rfl h; exact h
  rcases Nat.lt_or_gt_of_ne hbb with h | h
  · have := hiB_le_qsl hpre h hb'; omega
  · have := hiB_le_qsl hpre h hb; omega

/-- A program's write footprint lies in the output. -/
theorem Wp_sub {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {g b : Nat}
    (hg : g < d.G) (hb : b ≤ d.R) : Wp d init g b ⊆ d.slot_mappings.bytes :=
  Buf.elems_sub_bytes _ (tok_lt hd hg (range_ok hpre hb).2)

/-- The output shares no byte with an input. -/
theorem Rall_disjoint {d : Inst} (hd : d.ok) : Disjoint d.slot_mappings.bytes (Rall d) := by
  simp only [Rall, Set.disjoint_union_right]
  refine ⟨⟨⟨⟨⟨⟨⟨⟨hd.disj_idx, hd.disj_qsl⟩, hd.disj_pos⟩, hd.disj_ptrs⟩, hd.disj_strides⟩,
    hd.disj_sizes⟩, hd.disj_en⟩, hd.disj_sh⟩, ?_⟩
  refine Set.disjoint_left.mpr fun z hz ⟨g, hg, hzg⟩ => ?_
  exact Set.disjoint_left.mp (hd.disj_bt g hg) hz hzg

/-- A program's write footprint lies in `outW`. -/
theorem Wp_sub_outW {d : Inst} {init : Mem} (hpre : Pre d init) {g b : Nat} (hg : g < d.G)
    (hb : b ≤ d.R) : Wp d init g b ⊆ d.slot_mappings.elems (outW d init) := by
  refine Buf.elems_mono _ ?_
  rintro j ⟨t, h1, h2, rfl⟩
  have h0 := qsl_mono hpre (Nat.zero_le b) hb
  have := (range_ok hpre hb).2
  exact ⟨g, hg, t, by omega, by omega, rfl⟩

end Claims.SlotMappings
