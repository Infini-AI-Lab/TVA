import Proof.Defs
import Proof.Pid

/-!
# `lora_shrink_bf16`: the shared cells and the footprints

The output elements are the shared cells: distinct elements have disjoint bytes, the cells' bytes
are exactly the output's footprint `outIdx`, and they are disjoint from everything the programs
read. The log of a program is order-insensitive (`Q_permInv`).
-/

namespace Claims.LoraShrink

open TVA

/-- Two in-range indices `(u, a)`, `(u', b)` with `u * n + a = u' * n + b` coincide. -/
theorem div_mod_inj {n a b u u' : Nat} (ha : a < n) (hb : b < n) (h : u * n + a = u' * n + b) :
    u = u' ∧ a = b := by
  have h1 : (u * n + a) % n = a := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ha]
  have h2 : (u' * n + b) % n = b := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  have hab : a = b := by rw [← h1, ← h2, h]
  subst hab
  exact ⟨Nat.eq_of_mul_eq_mul_right (lt_of_le_of_lt (Nat.zero_le _) ha) (Nat.add_right_cancel h), rfl⟩

/-- The element index of the output is injective on in-range elements. -/
theorem outAt_inj (d : Inst) {s t j s' t' j' : Nat} (ht : t < d.M) (hj : j < d.R)
    (ht' : t' < d.M) (hj' : j' < d.R) (h : outAt d s t j = outAt d s' t' j') :
    s = s' ∧ t = t' ∧ j = j' := by
  unfold outAt at h
  have e1 : s * (d.M * d.R) + t * d.R + j = (s * d.M + t) * d.R + j := by ring
  have e2 : s' * (d.M * d.R) + t' * d.R + j' = (s' * d.M + t') * d.R + j' := by ring
  rw [e1, e2] at h
  obtain ⟨h1, h2⟩ := div_mod_inj hj hj' h
  obtain ⟨h3, h4⟩ := div_mod_inj ht ht' h1
  exact ⟨h3, h4, h2⟩

/-- Distinct output elements have disjoint bytes. -/
theorem cell_disjoint (d : Inst) {s t j s' t' j' : Nat} (ht : t < d.M) (hj : j < d.R)
    (ht' : t' < d.M) (hj' : j' < d.R) (hne : ¬ (s' = s ∧ t' = t ∧ j' = j)) :
    Disjoint (Mem.span .f32 (cellAddr d s' t' j')) (Mem.span .f32 (cellAddr d s t j)) := by
  unfold cellAddr
  refine d.out.span_disjoint ?_
  intro h
  have h' : outAt d s' t' j' = outAt d s t j := by exact_mod_cast h
  exact hne (outAt_inj d ht' hj' ht hj h')

/-- The shared cells' bytes are the output's footprint. -/
theorem cellBytes_eq (d : Inst) : cellBytes (Cells d) = d.out.elems (outIdx d) := by
  ext x
  constructor
  · rintro ⟨c, ⟨s, t, j, hs, ht, hj, rfl⟩, hx⟩
    exact ⟨_, ⟨s, t, j, hs, ht, hj, rfl⟩, hx⟩
  · rintro ⟨e, ⟨s, t, j, hs, ht, hj, rfl⟩, hx⟩
    exact ⟨_, ⟨s, t, j, hs, ht, hj, rfl⟩, hx⟩

/-- A cell is an element of the output buffer. -/
theorem cell_lt (d : Inst) (hd : d.ok) {s t j : Nat} (hs : s < d.S) (ht : t < d.M) (hj : j < d.R) :
    outAt d s t j < d.out.len := hd.out_len s t j hs ht hj

/-- The shared cells' bytes lie inside the output buffer. -/
theorem cellBytes_sub (d : Inst) (hd : d.ok) : cellBytes (Cells d) ⊆ d.out.bytes := by
  rw [cellBytes_eq]
  exact d.out.elems_sub_bytes (by
    rintro e ⟨s, t, j, hs, ht, hj, rfl⟩
    exact hd.out_len s t j hs ht hj)

/-- The shared cells are disjoint from everything the programs read. -/
theorem cells_disjoint_Rall (d : Inst) (hd : d.ok) : Disjoint (cellBytes (Cells d)) (Rall d) := by
  rw [Set.disjoint_left]
  intro x hx hr
  have hxo := cellBytes_sub d hd hx
  unfold Rall at hr
  simp only [Set.mem_union, Set.mem_iUnion, Set.mem_ofPred_eq] at hr
  rcases hr with ((((( h | ⟨s, hs, h⟩) | ⟨hS, h⟩) | h) | h) | h) | h
  · exact Set.disjoint_left.mp hd.out_inp hxo h
  · exact Set.disjoint_left.mp (hd.out_lora s hs) hxo h
  · exact Set.disjoint_left.mp (hd.out_tab hS) hxo h
  · exact Set.disjoint_left.mp hd.out_tid hxo h
  · exact Set.disjoint_left.mp hd.out_cnt hxo h
  · exact Set.disjoint_left.mp hd.out_start hxo h
  · exact Set.disjoint_left.mp hd.out_id hxo h

end Claims.LoraShrink
