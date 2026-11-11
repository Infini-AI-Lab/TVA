import Proof.Basic

/-!
# `kv_indices_i32`: Output windows of distinct programs are disjoint from the others' footprints.
-/

namespace Claims.KvIndices

open TVA

/-- A program's output window is disjoint from every other program's reads and writes. -/
theorem W_disj_aux {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (W d init (i : Nat)) (R d init (j : Nat) ∪ W d init (j : Nat)) := by
  rcases hpre with ⟨hrt, hrp, hsl, hki, hko, hst, hreq, hctx, hsum, hind⟩
  have hi : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hj : (j : Nat) < d.bs := lt_of_lt_of_eq j.isLt (launch_size d)
  have hne : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
  have hstep : ∀ b, b < d.bs → ind d init b ≤ ind d init (b + 1) := by
    intro b hb
    have h1 := hsum b hb
    have h2 := (hreq b hb).1
    omega
  have hmono : ∀ a b, a ≤ b → b ≤ d.bs → ind d init a ≤ ind d init b := by
    intro a b hab hb
    refine Nat.le_induction (m := a) (P := fun n _ => n ≤ d.bs → ind d init a ≤ ind d init n)
      ?_ ?_ b hab hb
    · intro _
      exact le_refl _
    · intro n hn ih hbn
      have h1 : ind d init a ≤ ind d init n := ih (by omega)
      have h2 : ind d init n ≤ ind d init (n + 1) := hstep n (by omega)
      omega
  have hout : Disjoint (outElems d init (i : Nat)) (outElems d init (j : Nat)) := by
    rw [Set.disjoint_left]
    rintro k ⟨p, hp, hkp⟩ ⟨q, hq, hkq⟩
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · have hm := hmono ((i : Nat) + 1) (j : Nat) (by omega) (by omega)
      have hs := hsum (i : Nat) hi
      have hsl := (hreq (i : Nat) hi).1
      omega
    · have hm := hmono ((j : Nat) + 1) (i : Nat) (by omega) (by omega)
      have hs := hsum (j : Nat) hj
      have hsl := (hreq (j : Nat) hj).1
      omega
  have hsub : ∀ k ∈ outElems d init (i : Nat), k < d.kv_indices.len := by
    rintro k ⟨p, hp, hk⟩
    have hs := hsum (i : Nat) hi
    have hb := (hind ((i : Nat) + 1) (by omega)).2
    have hsl := (hreq (i : Nat) hi).1
    have hk' : (k : Int) < (d.kv_indices.len : Int) := by omega
    exact_mod_cast hk'
  have hA : Disjoint (d.kv_indices.elems (outElems d init (i : Nat)))
      (d.req_pool_indices.elems {(j : Nat)}) :=
    Buf.elems_disjoint_of d.kv_indices d.req_pool_indices hd.disj.2.1 hsub (fun k hk => by
      rw [Set.mem_singleton_iff] at hk
      subst hk
      exact lt_of_lt_of_le hj hd.rplen)
  have hB : Disjoint (d.kv_indices.elems (outElems d init (i : Nat)))
      (d.kv_indptr.elems {(j : Nat)}) :=
    Buf.elems_disjoint_of d.kv_indices d.kv_indptr hd.disj.2.2.2.1 hsub (fun k hk => by
      rw [Set.mem_singleton_iff] at hk
      subst hk
      have hpl := hd.plen
      omega)
  have hC : Disjoint (d.kv_indices.elems (outElems d init (i : Nat)))
      (d.seq_lens.elems {(j : Nat)}) :=
    Buf.elems_disjoint_of d.kv_indices d.seq_lens hd.disj.2.2.1 hsub (fun k hk => by
      rw [Set.mem_singleton_iff] at hk
      subst hk
      exact lt_of_lt_of_le hj hd.llen)
  have hD : Disjoint (d.kv_indices.elems (outElems d init (i : Nat)))
      (if d.hasStart = true then d.kv_start_idx.elems {(j : Nat)} else ∅) := by
    split_ifs with hs
    · exact Buf.elems_disjoint_of d.kv_indices d.kv_start_idx (hd.disj.2.2.2.2 hs) hsub (by
        intro k hk
        rw [Set.mem_singleton_iff] at hk
        subst hk
        exact lt_of_lt_of_le hj (hd.stlen hs))
    · exact disjoint_bot_right
  have hE : Disjoint (d.kv_indices.elems (outElems d init (i : Nat)))
      (d.req_to_token.elems (tabElems d init (j : Nat))) :=
    Buf.elems_disjoint_of d.kv_indices d.req_to_token hd.disj.1 hsub (by
      rintro k ⟨p, hp, hkp⟩
      have h := ((hreq (j : Nat) hj).2.2.2.2 p hp).2
      have hk' : (k : Int) < (d.req_to_token.len : Int) := by omega
      exact_mod_cast hk')
  have hW : Disjoint (d.kv_indices.elems (outElems d init (i : Nat)))
      (d.kv_indices.elems (outElems d init (j : Nat))) :=
    Buf.elems_disjoint d.kv_indices hout
  rw [Set.disjoint_left]
  intro x hx hr
  simp only [R, Set.mem_union] at hr
  rcases hr with ((((h1 | h2) | h3) | h4) | h5) | h6
  · exact Set.disjoint_left.mp hA hx h1
  · exact Set.disjoint_left.mp hB hx h2
  · exact Set.disjoint_left.mp hC hx h3
  · exact Set.disjoint_left.mp hD hx h4
  · exact Set.disjoint_left.mp hE hx h5
  · exact Set.disjoint_left.mp hW hx h6

end Claims.KvIndices
