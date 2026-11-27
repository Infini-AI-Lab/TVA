import Proof.Basic

/-!
# `write_req_to_token_pool_i32`: the footprints do not meet

Every read footprint lies in buffers that share no byte with the table, every write footprint in
the table, and two programs that write use distinct rows, each written only within its first
`seq_lens[b] ≤ max_context_len` elements.
-/

namespace Claims.WriteReqToTokenPool

open TVA

/-- A window of no elements is empty. -/
theorem win_empty {a L : Int} (h : L ≤ 0) : win a L = ∅ := by
  ext e
  simp only [win, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
  rintro ⟨o, ho, -⟩
  have : (0 : Int) ≤ o := by positivity
  linarith

/-- A program's write footprint lies in the table. -/
theorem W_sub_bytes {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.bs) :
    W d init b ⊆ d.req_to_token.bytes := by
  obtain ⟨hp0, hps, hsm, hr0, hrl⟩ := hpre.req b hb
  have : 0 ≤ rpi d init b * (d.max_context_len : Int) := by positivity
  exact d.req_to_token.elems_sub_bytes (win_lt fun o ho => ⟨by linarith, by linarith⟩)

/-- A program's read footprint shares no byte with the table. -/
theorem R_disj_table {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b : Nat}
    (hb : b < d.bs) : Disjoint d.req_to_token.bytes (R d init b) := by
  obtain ⟨d1, d2, d3, d4, d5, d6⟩ := hd.disj
  have hsingle : ∀ n : Nat, b < n → ∀ j ∈ ({b} : Set Nat), j < n := fun n h j hj => by
    rw [Set.mem_singleton_iff.mp hj]; exact h
  have hcb : 0 ≤ cum d init b := by
    unfold cum
    exact Finset.sum_nonneg fun i hi => (hpre.ext i (by simp at hi; omega)).1
  simp only [R]
  refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right (Disjoint.union_right
    (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_) ?_) ?_) ?_
  · exact d1.mono_right (d.req_pool_indices.elems_sub_bytes (hsingle _ (by have := hd.rplen; omega)))
  · exact d2.mono_right (d.prefix_pointers.elems_sub_bytes (hsingle _ (by have := hd.pplen; omega)))
  · exact d3.mono_right (d.prefix_lens.elems_sub_bytes (hsingle _ (by have := hd.prelen; omega)))
  · exact d4.mono_right (d.seq_lens.elems_sub_bytes (hsingle _ (by have := hd.slen; omega)))
  · exact d5.mono_right (d.extend_lens.elems_sub_bytes fun j (hj : j < b) => by
      have := hd.elen; omega)
  · refine (hd.prefix_disj b hb).mono_right ((d.prefixT b).elems_sub_bytes (win_lt fun o ho => ?_))
    have hpos : 0 < pre d init b := lt_of_le_of_lt (by positivity) ho
    obtain ⟨-, hlen, -⟩ := hpre.prefixes b hb hpos
    constructor <;> linarith
  · refine d6.mono_right (d.out_cache_loc.elems_sub_bytes (win_lt fun o ho => ?_))
    have := hpre.window b hb
    constructor <;> linarith

/-- Programs read nothing they write. -/
theorem R_disj_W {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b : Nat}
    (hb : b < d.bs) : Disjoint (R d init b) (W d init b) :=
  ((R_disj_table hd hpre hb).mono_left (W_sub_bytes hpre hb)).symm

/-- Two distinct programs write disjoint elements of the table. -/
theorem W_disj_W {d : Inst} {init : Mem} (hpre : Pre d init) {b c : Nat} (hb : b < d.bs)
    (hc : c < d.bs) (hbc : b ≠ c) : Disjoint (W d init b) (W d init c) := by
  by_cases hsb : seq d init b ≤ 0
  · simp only [W, rowElems, win_empty hsb, Buf.elems]; simp
  by_cases hsc : seq d init c ≤ 0
  · simp only [W, rowElems, win_empty hsc, Buf.elems]; simp
  push Not at hsb hsc
  have hne := hpre.rows b c hb hc hbc hsb hsc
  obtain ⟨-, -, hsmb, hrb0, -⟩ := hpre.req b hb
  obtain ⟨-, -, hsmc, hrc0, -⟩ := hpre.req c hc
  refine d.req_to_token.elems_disjoint (Set.disjoint_left.mpr ?_)
  rintro e ⟨o, ho, he⟩ ⟨o', ho', he'⟩
  have h0 : (0 : Int) ≤ o := by positivity
  have h0' : (0 : Int) ≤ o' := by positivity
  rcases lt_or_gt_of_ne hne with h | h
  · have : rpi d init b + 1 ≤ rpi d init c := h
    nlinarith
  · have : rpi d init c + 1 ≤ rpi d init b := h
    nlinarith

end Claims.WriteReqToTokenPool
