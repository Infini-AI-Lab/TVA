import Proof.Basic

/-!
# `kv_indices_i32`: Reads and writes of one program do not meet.
-/

namespace Claims.KvIndices

open TVA

/-- A program reads nothing it writes: its output window is disjoint from its inputs. -/
theorem R_disj_W_aux {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) : Disjoint (R d init (i : Nat)) (W d init (i : Nat)) := by
  obtain ⟨-, -, -, -, -, -, hreq, -, hsum, hind⟩ := hpre
  have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  -- the written elements are elements of `kv_indices`
  have hWsub : ∀ j ∈ outElems d init (i : Nat), j < d.kv_indices.len := by
    intro j hj
    obtain ⟨p, hp, hjeq⟩ := hj
    have hs : ind d init ((i : Nat) + 1) = ind d init (i : Nat) + slen d init (i : Nat) :=
      hsum (i : Nat) hi
    have hle : ind d init ((i : Nat) + 1) ≤ (d.kv_indices.len : Int) :=
      (hind ((i : Nat) + 1) (by omega)).2
    have hlt : (j : Int) < (d.kv_indices.len : Int) := by omega
    exact_mod_cast hlt
  have h1 : Disjoint (d.req_pool_indices.elems {(i : Nat)})
      (d.kv_indices.elems (outElems d init (i : Nat))) :=
    Buf.elems_disjoint_of _ _ hd.disj.2.1.symm
      (fun j hj => by
        simp only [Set.mem_singleton_iff] at hj
        rw [hj]
        exact lt_of_lt_of_le hi hd.rplen)
      (fun j hj => hWsub j hj)
  have h2 : Disjoint (d.kv_indptr.elems {(i : Nat)})
      (d.kv_indices.elems (outElems d init (i : Nat))) :=
    Buf.elems_disjoint_of _ _ hd.disj.2.2.2.1.symm
      (fun j hj => by
        simp only [Set.mem_singleton_iff] at hj
        rw [hj]
        exact lt_of_lt_of_le (Nat.lt_succ_of_lt hi) hd.plen)
      (fun j hj => hWsub j hj)
  have h3 : Disjoint (d.seq_lens.elems {(i : Nat)})
      (d.kv_indices.elems (outElems d init (i : Nat))) :=
    Buf.elems_disjoint_of _ _ hd.disj.2.2.1.symm
      (fun j hj => by
        simp only [Set.mem_singleton_iff] at hj
        rw [hj]
        exact lt_of_lt_of_le hi hd.llen)
      (fun j hj => hWsub j hj)
  have h4 : Disjoint (if d.hasStart = true then d.kv_start_idx.elems {(i : Nat)} else ∅)
      (d.kv_indices.elems (outElems d init (i : Nat))) := by
    by_cases hh : d.hasStart = true
    · rw [ite_eq_left hh]
      exact Buf.elems_disjoint_of _ _ (hd.disj.2.2.2.2 hh).symm
        (fun j hj => by
          simp only [Set.mem_singleton_iff] at hj
          rw [hj]
          exact lt_of_lt_of_le hi (hd.stlen hh))
        (fun j hj => hWsub j hj)
    · rw [ite_eq_right hh]
      exact Set.empty_disjoint _
  have h5 : Disjoint (d.req_to_token.elems (tabElems d init (i : Nat)))
      (d.kv_indices.elems (outElems d init (i : Nat))) :=
    Buf.elems_disjoint_of _ _ hd.disj.1.symm
      (fun j hj => by
        obtain ⟨p, hp, hjeq⟩ := hj
        have hb := (hreq (i : Nat) hi).2.2.2.2 p hp
        have hlt : (j : Int) < (d.req_to_token.len : Int) := by omega
        exact_mod_cast hlt)
      (fun j hj => hWsub j hj)
  unfold R W
  simp only [Set.disjoint_union_left]
  exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩

end Claims.KvIndices
