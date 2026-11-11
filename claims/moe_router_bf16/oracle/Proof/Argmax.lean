import Claim
import TVA.Proof

/-!
# What the two argmax reductions return

`tl.argmaxAll true true` picks the largest value and, among equals, the smallest index.
These lemmas extract that from `tl.ArgRes` for the two calls of the block: the first over the
soft-capped logits, the second over the same row with the first winner replaced by `-inf`.
-/

namespace Claims.MoeRouter

open TVA
open TVA

private theorem f32_lt_sortKey {u v : F32} (hu : u.isNaN = false) (hv : v.isNaN = false) :
    CmpKey.lt u v = true ↔ FloatV.sortKey u < FloatV.sortKey v := by
  show FloatV.lt u v = true ↔ FloatV.sortKey u < FloatV.sortKey v
  unfold FloatV.lt FloatV.sortKey
  rw [← Format.cmp_lt_iff hu hv]
  simp

private theorem f32_eq_sortKey {u v : F32} (hu : u.isNaN = false) (hv : v.isNaN = false) :
    CmpKey.eq u v = true ↔ FloatV.sortKey u = FloatV.sortKey v := by
  show FloatV.eq u v = true ↔ FloatV.sortKey u = FloatV.sortKey v
  unfold FloatV.eq FloatV.sortKey
  rw [beq_iff_eq, Format.cmp_eq_iff hu hv, Format.toRat_eq_iff_orderKey]

private theorem isNaN_of_isFinite {x : F32} (h : x.isFinite = true) : x.isNaN = false :=
  Format.not_isNaN_of_isFinite h

private theorem toRat_le_iff_sortKey_le (u v : F32) :
    u.toRat ≤ v.toRat ↔ FloatV.sortKey u ≤ FloatV.sortKey v := by
  unfold FloatV.sortKey FloatV.toRat
  rw [← not_lt, ← not_lt]
  exact not_congr (Format.toRat_lt_iff_orderKey (f := Format.binary32) v.pat u.pat)

private theorem toRat_lt_iff_sortKey_lt (u v : F32) :
    u.toRat < v.toRat ↔ FloatV.sortKey u < FloatV.sortKey v := by
  unfold FloatV.sortKey FloatV.toRat
  exact Format.toRat_lt_iff_orderKey (f := Format.binary32) u.pat v.pat

private theorem toReal_le_iff (u v : F32) : u.toReal ≤ v.toReal ↔ u.toRat ≤ v.toRat :=
  Rat.cast_le

private theorem toReal_lt_iff (u v : F32) : u.toReal < v.toReal ↔ u.toRat < v.toRat :=
  Rat.cast_lt

private theorem argCombine_tieLeft_f32 (isMax : Bool) (a b : F32 × Nat)
    (ha : a.1.isNaN = false) (hb : b.1.isNaN = false) :
    tl.argCombine isMax true a b =
      if (if isMax then FloatV.sortKey b.1 < FloatV.sortKey a.1
          else FloatV.sortKey a.1 < FloatV.sortKey b.1) ∨
         (FloatV.sortKey a.1 = FloatV.sortKey b.1 ∧ a.2 < b.2) then a else b := by
  have hw : (((if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) ||
      (true && CmpKey.eq a.1 b.1 && decide (a.2 < b.2))) = true ↔
      ((if isMax then FloatV.sortKey b.1 < FloatV.sortKey a.1
        else FloatV.sortKey a.1 < FloatV.sortKey b.1) ∨
        (FloatV.sortKey a.1 = FloatV.sortKey b.1 ∧ a.2 < b.2))) := by
    cases isMax <;>
      simp [f32_lt_sortKey ha hb, f32_lt_sortKey hb ha, f32_eq_sortKey ha hb]
  simp only [tl.argCombine]
  by_cases h : (if isMax then FloatV.sortKey b.1 < FloatV.sortKey a.1
      else FloatV.sortKey a.1 < FloatV.sortKey b.1) ∨
      (FloatV.sortKey a.1 = FloatV.sortKey b.1 ∧ a.2 < b.2)
  · rw [ite_eq_left (hw.mpr h), ite_eq_left h]
  · have h' : ¬ (((if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) ||
        (true && CmpKey.eq a.1 b.1 && decide (a.2 < b.2))) = true) := fun h' => h (hw.mp h')
    rw [ite_eq_right h', ite_eq_right h]

private theorem eval_argCombine_first_f32 (isMax : Bool) (g : Nat → F32) :
    ∀ {t : Tree} {p : F32 × Nat}, t.leaves.Nodup →
      (∀ j ∈ t.leaves, (g j).isNaN = false) →
      t.Eval (fun a b c => c = tl.argCombine isMax true a b) (fun j => (g j, j)) p →
      p.2 ∈ t.leaves ∧ p.1 = g p.2 ∧ ∀ j ∈ t.leaves,
        (if isMax then FloatV.sortKey (g j) ≤ FloatV.sortKey (g p.2)
         else FloatV.sortKey (g p.2) ≤ FloatV.sortKey (g j)) ∧
        (FloatV.sortKey (g j) = FloatV.sortKey (g p.2) → p.2 ≤ j)
  | .leaf i, _, _, _, .leaf _ => by
    refine ⟨by simp [Tree.leaves], rfl, fun j hj => ?_⟩
    simp only [Tree.leaves, List.mem_singleton] at hj
    subst hj; cases isMax <;> simp
  | .node l r, _, hnd, hx, .node (a := ⟨va, ia⟩) (b := ⟨vb, ib⟩) ha hb hc => by
    simp only [Tree.leaves, List.nodup_append] at hnd
    obtain ⟨hndl, hndr, hdisj⟩ := hnd
    have hxl : ∀ j ∈ l.leaves, (g j).isNaN = false := fun j hj => hx j (by simp [Tree.leaves, hj])
    have hxr : ∀ j ∈ r.leaves, (g j).isNaN = false := fun j hj => hx j (by simp [Tree.leaves, hj])
    obtain ⟨hal, haf, hab⟩ := eval_argCombine_first_f32 isMax g hndl hxl ha
    obtain ⟨hbl, hbf, hbb⟩ := eval_argCombine_first_f32 isMax g hndr hxr hb
    simp only at hal haf hab hbl hbf hbb
    subst haf hbf hc
    rw [argCombine_tieLeft_f32 isMax (g ia, ia) (g ib, ib) (hxl ia hal) (hxr ib hbl)]
    simp only [Tree.leaves, List.mem_append]
    by_cases hw : (if isMax then FloatV.sortKey (g ib) < FloatV.sortKey (g ia)
        else FloatV.sortKey (g ia) < FloatV.sortKey (g ib)) ∨
        (FloatV.sortKey (g ia) = FloatV.sortKey (g ib) ∧ ia < ib)
    · rw [ite_eq_left hw]
      refine ⟨Or.inl hal, rfl, fun j hj => ?_⟩
      rcases hj with hj | hj
      · exact hab j hj
      · obtain ⟨h1, h2⟩ := hbb j hj
        have hjne : ia ≠ j := hdisj _ hal _ hj
        cases isMax <;> simp only [Bool.false_eq_true, ite_false, ite_true] at hw h1 ⊢ <;>
          exact ⟨by omega, fun e => by omega⟩
    · rw [ite_eq_right hw]
      refine ⟨Or.inr hbl, rfl, fun j hj => ?_⟩
      rcases hj with hj | hj
      · obtain ⟨h1, h2⟩ := hab j hj
        have hne : ia ≠ ib := hdisj _ hal _ hbl
        cases isMax <;> simp only [Bool.false_eq_true, ite_false, ite_true] at hw h1 ⊢ <;>
          exact ⟨by omega, fun e => by omega⟩
      · exact hbb j hj


/-- The first argmax, on a row of finite values: a real index, an index equal to it, and the
two order properties of a first maximum. -/
theorem argRes_finite {n : Nat} (_hn : 0 < n) {g : Nat → F32}
    (hfin : ∀ j, j < n → (g j).isFinite = true)
    {v : F32} {k : I32} (h : tl.ArgRes true true n g (v, k)) :
    ∃ i, i < n ∧ k = IntV.ofNat i ∧
      (∀ j, j < n → (g j).toReal ≤ (g i).toReal) ∧
      (∀ j, j < i → (g j).toReal < (g i).toReal) := by
  have hno : ∀ j, j < n → CmpKey.nan (g j) = false := fun j hj => isNaN_of_isFinite (hfin j hj)
  obtain ⟨p, t, hcov, hev, hvk⟩ := tl.ArgRes.tree_of_noNaN hno h
  simp only [Prod.mk.injEq] at hvk
  obtain ⟨hv, hk⟩ := hvk
  have hmem : ∀ a, a ∈ t.leaves ↔ a < n := fun a => by rw [hcov.mem_iff, List.mem_range]
  have hnd : t.leaves.Nodup := hcov.nodup_iff.mpr List.nodup_range
  have hxl : ∀ j ∈ t.leaves, (g j).isNaN = false := fun j hj => hno j ((hmem j).mp hj)
  obtain ⟨hpl, hpf, hmax⟩ := eval_argCombine_first_f32 true g hnd hxl hev
  have hpn : p.2 < n := (hmem p.2).mp hpl
  refine ⟨p.2, hpn, hk, ?_, ?_⟩
  · intro j hj
    have h1 : FloatV.sortKey (g j) ≤ FloatV.sortKey (g p.2) := by
      have h := (hmax j ((hmem j).mpr hj)).1
      simpa using h
    rw [toReal_le_iff, toRat_le_iff_sortKey_le]
    exact h1
  · intro j hj
    have hmem' : j ∈ t.leaves := (hmem j).mpr (by omega)
    have h1 : FloatV.sortKey (g j) ≤ FloatV.sortKey (g p.2) := by
      have h := (hmax j hmem').1
      simpa using h
    have h2 : FloatV.sortKey (g j) = FloatV.sortKey (g p.2) → p.2 ≤ j := (hmax j hmem').2
    have hlt : FloatV.sortKey (g j) < FloatV.sortKey (g p.2) := by
      rcases lt_or_eq_of_le h1 with h | h
      · exact h
      · exact absurd (h2 h) (by omega)
    rw [toReal_lt_iff, toRat_lt_iff_sortKey_lt]
    exact hlt

/-- The maximum of a row of finite values is the value at any index that dominates the row. -/
theorem maxAll_eq {n : Nat} (_hn : 0 < n) {g : Nat → F32}
    (hfin : ∀ j, j < n → (g j).isFinite = true)
    {m : F32} (hm : ∃ t, t.Covers n ∧ Tree.Eval (FloatV.MaxRel true) g t m)
    {i : Nat} (hi : i < n) (hle : ∀ j, j < n → (g j).toReal ≤ (g i).toReal) :
    m.toReal = (g i).toReal := by
  obtain ⟨t, hcov, hev⟩ := hm
  have hmem : ∀ a, a ∈ t.leaves ↔ a < n := fun a => by rw [hcov.mem_iff, List.mem_range]
  have hno : ∀ j ∈ t.leaves, (g j).isNaN = false := fun j hj =>
    isNaN_of_isFinite (hfin j ((hmem j).mp hj))
  obtain ⟨⟨i', hi', hm'⟩, hall⟩ := Tree.eval_maxRel hev hno
  have hi'lt : i' < n := (hmem i').mp hi'
  have hni : (g i').isNaN = false := hno i' hi'
  have hnm : m.isNaN = false := by rw [hm']; exact hni
  have hi_mem : i ∈ t.leaves := (hmem i).mpr hi
  have h1 : (g i).toReal ≤ m.toReal := by
    rw [toReal_le_iff]
    exact (FloatV.le_iff (hno i hi_mem) hnm).mp (by simpa using hall i hi_mem)
  have h2 : m.toReal ≤ (g i).toReal := by
    rw [hm']
    exact hle i' hi'lt
  linarith

/-- The second argmax, on a row whose entry at `i0` has been replaced by a value below every
other: a real index different from `i0`, and the two order properties of a first maximum of
the remaining entries. -/
theorem argRes_masked {n : Nat} (hn : 1 < n) {g : Nat → F32} {lo : F32}
    (hfin : ∀ j, j < n → (g j).isFinite = true)
    (hlo : lo.isNaN = false)
    (hlo_lt : ∀ j, j < n → FloatV.sortKey lo < FloatV.sortKey (g j))
    {i0 : Nat} (_hi0 : i0 < n) {v : F32} {k : I32}
    (h : tl.ArgRes true true n (fun j => if j = i0 then lo else g j) (v, k)) :
    ∃ i, i < n ∧ i ≠ i0 ∧ k = IntV.ofNat i ∧
      (∀ j, j < n → j ≠ i0 → (g j).toReal ≤ (g i).toReal) ∧
      (∀ j, j < i → j ≠ i0 → (g j).toReal < (g i).toReal) := by
  set G : Nat → F32 := fun j => if j = i0 then lo else g j with hG
  have hh : tl.ArgRes true true n G (v, k) := by rw [hG]; exact h
  have hno : ∀ j, j < n → CmpKey.nan (G j) = false := by
    intro j hj
    by_cases hj0 : j = i0
    · rw [show G j = lo by simp [G, hj0]]
      exact hlo
    · rw [show G j = g j by simp [G, hj0]]
      exact isNaN_of_isFinite (hfin j hj)
  obtain ⟨p, t, hcov, hev, hvk⟩ := tl.ArgRes.tree_of_noNaN hno hh
  simp only [Prod.mk.injEq] at hvk
  obtain ⟨hv, hk⟩ := hvk
  have hmem : ∀ a, a ∈ t.leaves ↔ a < n := fun a => by rw [hcov.mem_iff, List.mem_range]
  have hnd : t.leaves.Nodup := hcov.nodup_iff.mpr List.nodup_range
  have hxl : ∀ j ∈ t.leaves, (G j).isNaN = false := fun j hj => hno j ((hmem j).mp hj)
  obtain ⟨hpl, hpf, hmax⟩ := eval_argCombine_first_f32 true G hnd hxl hev
  have hpn : p.2 < n := (hmem p.2).mp hpl
  have huntie : p.2 ≠ i0 := by
    intro he
    obtain ⟨j, hj, hj0⟩ : ∃ j, j < n ∧ j ≠ i0 := by
      rcases Nat.eq_zero_or_pos i0 with h0 | h0
      · exact ⟨1, by omega, by omega⟩
      · exact ⟨0, by omega, by omega⟩
    have hjmem : j ∈ t.leaves := (hmem j).mpr hj
    have h1 : FloatV.sortKey (G j) ≤ FloatV.sortKey (G p.2) := by
      have h := (hmax j hjmem).1
      simpa using h
    have hGj : G j = g j := by simp [G, hj0]
    have hGp : G p.2 = lo := by simp [G, he]
    rw [hGj, hGp] at h1
    exact absurd h1 (not_le.mpr (hlo_lt j hj))
  refine ⟨p.2, hpn, huntie, hk, ?_, ?_⟩
  · intro j hj hj0
    have hjmem : j ∈ t.leaves := (hmem j).mpr hj
    have hGj : G j = g j := by simp [G, hj0]
    have hGp : G p.2 = g p.2 := by simp [G, huntie]
    have h1 : FloatV.sortKey (G j) ≤ FloatV.sortKey (G p.2) := by
      have h := (hmax j hjmem).1
      simpa using h
    rw [hGj, hGp] at h1
    rw [toReal_le_iff, toRat_le_iff_sortKey_le]
    exact h1
  · intro j hj hj0
    have hjmem : j ∈ t.leaves := (hmem j).mpr (by omega)
    have hGj : G j = g j := by simp [G, hj0]
    have hGp : G p.2 = g p.2 := by simp [G, huntie]
    have h1 : FloatV.sortKey (G j) ≤ FloatV.sortKey (G p.2) := by
      have h := (hmax j hjmem).1
      simpa using h
    have h2 : FloatV.sortKey (G j) = FloatV.sortKey (G p.2) → p.2 ≤ j := by
      have h := (hmax j hjmem).2
      simpa using h
    rw [hGj, hGp] at h1 h2
    have hlt : FloatV.sortKey (g j) < FloatV.sortKey (g p.2) := by
      rcases lt_or_eq_of_le h1 with h | h
      · exact h
      · exact absurd (h2 h) (by omega)
    rw [toReal_lt_iff, toRat_lt_iff_sortKey_lt]
    exact hlt

end Claims.MoeRouter
