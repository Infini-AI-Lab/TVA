import Proof.FootAux1
import Proof.FootAux2

/-!
# `chunk_delta_h_bf16`: the launch, the footprints and `Post`

* the launch is valid; the program index is decoded as `progOf` encodes;
* the element maps `hIdx`, `uIdx`, `sIdx` are injective on the cells of a sequence and
  distinct programs write disjoint cells; no program reads what any program writes;
* `ColPre` and `Col` read the output buffers only at their cells (`ColPre_congr`);
* the programs' results give `Post` (`post_of_locals`).
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-! ## The launch -/

theorem launch_size (d : Inst) : (launch d).size = gridN d := by
  simp [Launch.size, launch]

/-- The block dimensions of the launch are powers of two. -/
theorem pow2_of_cases {k : Nat} (h : k = 1 ∨ k = 32 ∨ k = 64) : ∃ e, k = 2 ^ e := by
  rcases h with rfl | rfl | rfl
  · exact ⟨0, rfl⟩
  · exact ⟨5, rfl⟩
  · exact ⟨6, rfl⟩

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg := hd.grid_lt
  refine ⟨hg, by simp [launch], by simp [launch], by rw [launch_size]; exact hg, rfl, rfl, rfl, ?_⟩
  intro s hs
  simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
  refine ⟨?_, ?_⟩
  · intro k hk
    apply pow2_of_cases
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [BV, BT] at hk <;> omega
  · rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> norm_num [BV, BT]

/-- Program indices below the grid size decode to a sequence, head and block in range, and
`progOf` encodes them back. -/
theorem decode {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gridN d) :
    nOf d i < d.N ∧ hvOf d i < d.HV ∧ jvOf d i < NV d ∧
      progOf d (nOf d i) (hvOf d i) (jvOf d i) = i := by
  have hNV : 0 < NV d := by
    have := hd.V_pos; unfold NV BV; omega
  have hHV := hd.HV_pos
  have hg : gridN d = NV d * d.N * d.HV := rfl
  have h1 : i / NV d / d.HV * d.HV + i / NV d % d.HV = i / NV d := by
    have := Nat.div_add_mod (i / NV d) d.HV; rw [Nat.mul_comm] at this; exact this
  have h2 : i / NV d * NV d + i % NV d = i := by
    have := Nat.div_add_mod i (NV d); rw [Nat.mul_comm] at this; exact this
  refine ⟨?_, Nat.mod_lt _ hHV, Nat.mod_lt _ hNV, ?_⟩
  · unfold nOf
    rw [Nat.div_lt_iff_lt_mul hHV, Nat.div_lt_iff_lt_mul hNV]
    have : d.N * d.HV * NV d = NV d * d.N * d.HV := by ring
    omega
  · unfold progOf nOf hvOf jvOf
    rw [h1, h2]

/-! ## Cells -/

/-- Cells of `h` with different `(chunk, head, row, column)` are different, when the row is below
`K` and the column below `V`. -/
theorem hIdx_inj {d : Inst} {c₀ c₀' hv hv' c c' j j' : Nat} (hhv : hv < d.HV) (hhv' : hv' < d.HV)
    (hc : c < d.K) (hc' : c' < d.K) (hj : j < d.V) (hj' : j' < d.V)
    (h : hIdx d c₀ hv c j = hIdx d c₀' hv' c' j') :
    c₀ = c₀' ∧ hv = hv' ∧ c = c' ∧ j = j' := by
  unfold hIdx at h
  split_ifs at h
  · obtain ⟨h1, h2, h3, h4⟩ := idx4_inj hhv hhv' hj hj' hc hc' h
    exact ⟨h1, h2, h4, h3⟩
  · exact idx4_inj hhv hhv' hc hc' hj hj' h

/-- Cells of `v_new`/`u` with different `(token, head, column)` are different. -/
theorem uIdx_inj {d : Inst} {b b' t t' hv hv' j j' : Nat} (ht : t < d.T) (ht' : t' < d.T)
    (hhv : hv < d.HV) (hhv' : hv' < d.HV) (hj : j < d.V) (hj' : j' < d.V)
    (h : uIdx d b t hv j = uIdx d b' t' hv' j') :
    b = b' ∧ t = t' ∧ hv = hv' ∧ j = j' := by
  unfold uIdx at h
  exact idx4_inj ht ht' hhv hhv' hj hj' h

/-- Cells of `ht`/`h0` with different `(sequence, head, row, column)` are different. -/
theorem sIdx_inj {d : Inst} {n n' hv hv' c c' j j' : Nat} (hhv : hv < d.HV) (hhv' : hv' < d.HV)
    (hc : c < d.K) (hc' : c' < d.K) (hj : j < d.V) (hj' : j' < d.V)
    (h : sIdx d n hv c j = sIdx d n' hv' c' j') :
    n = n' ∧ hv = hv' ∧ c = c' ∧ j = j' := by
  unfold sIdx at h
  split_ifs at h
  · obtain ⟨h1, h2, h3, h4⟩ := idx4_inj hhv hhv' hj hj' hc hc' h
    exact ⟨h1, h2, h4, h3⟩
  · exact idx4_inj hhv hhv' hc hc' hj hj' h

/-- The first token of sequence `n` plus its length is at most `T`, and different sequences of a
variable-length batch occupy different tokens; in a fixed-length batch they are different rows. -/
theorem tokens_disjoint {d : Inst} (hd : d.ok) {n n' s s' : Nat} (hn : n < d.N) (hn' : n' < d.N)
    (hs : s < lenOf d n) (hs' : s' < lenOf d n') (hne : n ≠ n') :
    rowOf d n * d.T + (startOf d n + s) ≠ rowOf d n' * d.T + (startOf d n' + s') := by
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · have := tok_lt_next hd h hn' hs; omega
  · have := tok_lt_next hd h hn hs'; omega

/-- Chunks of different sequences are different cells of `h`'s chunk axis: `cbase n + t`,
`t < nch n`, are below `NTt` and different for different sequences. -/
theorem chunks_disjoint {d : Inst} (_hd : d.ok) {n n' t t' : Nat} (_hn : n < d.N) (_hn' : n' < d.N)
    (ht : t < nch d n) (ht' : t' < nch d n') (hne : n ≠ n') :
    cbase d n + t ≠ cbase d n' + t' := by
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · have := cbase_add_nch_le (d := d) h; omega
  · have := cbase_add_nch_le (d := d) h; omega

/-! ## Bounds of a program's cells -/

/-- The cells of chunk `t` of `h` are inside its tensor. -/
theorem hCells_lt {d : Inst} (hd : d.ok) {n hv jv t : Nat} (hn : n < d.N) (hhv : hv < d.HV)
    (ht : t < nch d n) : ∀ y ∈ hCells d n hv jv t, y < d.B * NTt d * d.HV * d.K * d.V := by
  rintro y ⟨c, j, hc, hj, rfl⟩
  exact hIdx_lt (chunk_lt hd hn ht) hhv hc hj.2.2

/-- The cells of chunk `t` of `v_new` are inside its tensor. -/
theorem vCells_lt {d : Inst} (hd : d.ok) {n hv jv t : Nat} (hn : n < d.N) (hhv : hv < d.HV) :
    ∀ y ∈ vCells d n hv jv t, y < d.B * d.T * d.HV * d.V := by
  rintro y ⟨i, j, hi, hj, rfl⟩
  have h1 := tok_lt hd hn (cl_lt hi)
  exact uIdx_lt (row_lt hd hn) (by omega) hhv hj.2.2

/-- The cells of `ht` that a program writes are inside its tensor. -/
theorem tCells_lt {d : Inst} {n hv jv : Nat} (hn : n < d.N) (hhv : hv < d.HV) :
    ∀ y ∈ tCells d n hv jv, y < d.N * d.HV * d.K * d.V := by
  rintro y ⟨c, j, hc, hj, rfl⟩
  exact sIdx_lt hn hhv hc hj.2.2

/-- The elements of `h` that a program writes, over all its chunks. -/
def hAll (d : Inst) (n hv jv : Nat) : Set Nat :=
  {x | ∃ t, t < nch d n ∧ x ∈ hCells d n hv jv t}

/-- The elements of `v_new` that a program writes, over all its chunks. -/
def vAll (d : Inst) (n hv jv : Nat) : Set Nat :=
  {x | ∃ t, t < nch d n ∧ x ∈ vCells d n hv jv t}

/-- A program's writes lie in the three outputs, at the cells of `hAll`, `vAll`, `tCells`. -/
theorem W_sub_parts (d : Inst) (i : Nat) :
    W d i ⊆ d.h.elems (hAll d (nOf d i) (hvOf d i) (jvOf d i)) ∪
      d.vn.elems (vAll d (nOf d i) (hvOf d i) (jvOf d i)) ∪
      Wend d (nOf d i) (hvOf d i) (jvOf d i) := by
  rintro x (⟨t, ht, hx | hx⟩ | hx)
  · have hsub : hCells d (nOf d i) (hvOf d i) (jvOf d i) t ⊆
        hAll d (nOf d i) (hvOf d i) (jvOf d i) := fun y hy => ⟨t, ht, hy⟩
    exact Or.inl (Or.inl (Buf.elems_mono d.h hsub hx))
  · have hsub : vCells d (nOf d i) (hvOf d i) (jvOf d i) t ⊆
        vAll d (nOf d i) (hvOf d i) (jvOf d i) := fun y hy => ⟨t, ht, hy⟩
    exact Or.inl (Or.inr (Buf.elems_mono d.vn hsub hx))
  · exact Or.inr hx

/-- The cells of `hAll` are inside the tensor of `h`. -/
theorem hAll_lt {d : Inst} (hd : d.ok) {n hv jv : Nat} (hn : n < d.N) (hhv : hv < d.HV) :
    ∀ y ∈ hAll d n hv jv, y < d.B * NTt d * d.HV * d.K * d.V := by
  rintro y ⟨t, ht, hy⟩
  exact hCells_lt hd hn hhv ht y hy

/-- The cells of `vAll` are inside the tensor of `v_new`. -/
theorem vAll_lt {d : Inst} (hd : d.ok) {n hv jv : Nat} (hn : n < d.N) (hhv : hv < d.HV) :
    ∀ y ∈ vAll d n hv jv, y < d.B * d.T * d.HV * d.V := by
  rintro y ⟨t, -, hy⟩
  exact vCells_lt hd hn hhv y hy

/-- The final state's store happens only with `STORE_FINAL_STATE`, at the cells of `tCells`. -/
theorem Wend_mem {d : Inst} {n hv jv : Nat} {x : Int} (hx : x ∈ Wend d n hv jv) :
    d.storeHt = true ∧ x ∈ d.ht.elems (tCells d n hv jv) := by
  unfold Wend at hx
  split_ifs at hx with hst
  · exact ⟨hst, hx⟩
  · exact absurd hx (Set.notMem_empty x)

/-- Every cell program `i` writes is inside its buffer. -/
theorem W_sub_outputs {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gridN d) :
    W d i ⊆ d.h.elems {x | x < d.B * NTt d * d.HV * d.K * d.V} ∪
      d.vn.elems {x | x < d.B * d.T * d.HV * d.V} ∪
      (if d.storeHt then d.ht.elems {x | x < d.N * d.HV * d.K * d.V} else ∅) := by
  obtain ⟨hn, hhv, -, -⟩ := decode hd hi
  intro x hx
  rcases W_sub_parts d i hx with (hx | hx) | hx
  · exact Or.inl (Or.inl (Buf.elems_mono d.h (hAll_lt hd hn hhv) hx))
  · exact Or.inl (Or.inr (Buf.elems_mono d.vn (vAll_lt hd hn hhv) hx))
  · obtain ⟨hst, hx⟩ := Wend_mem hx
    refine Or.inr ?_
    simp only [hst, ↓reduceIte]
    exact Buf.elems_mono d.ht (tCells_lt hn hhv) hx

/-- A trip's bytes are inside the program's footprint `W`. -/
theorem Wt_sub_W {d : Inst} {i t : Nat} (ht : t < nch d (nOf d i)) :
    Wt d (nOf d i) (hvOf d i) (jvOf d i) t ⊆ W d i :=
  fun _ hx => Or.inl ⟨t, ht, hx⟩

/-- A program's writes lie inside the three output buffers. -/
theorem W_sub_bytes {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gridN d) :
    W d i ⊆ d.h.bytes ∪ d.vn.bytes ∪ d.ht.bytes := by
  obtain ⟨hn, hhv, -, -⟩ := decode hd hi
  intro x hx
  rcases W_sub_parts d i hx with (hx | hx) | hx
  · exact Or.inl (Or.inl (Buf.elems_sub_bytes d.h
      (fun y hy => lt_of_lt_of_le (hAll_lt hd hn hhv y hy) hd.h_len) hx))
  · exact Or.inl (Or.inr (Buf.elems_sub_bytes d.vn
      (fun y hy => lt_of_lt_of_le (vAll_lt hd hn hhv y hy) hd.vn_len) hx))
  · obtain ⟨hst, hx⟩ := Wend_mem hx
    exact Or.inr (Buf.elems_sub_bytes d.ht
      (fun y hy => lt_of_lt_of_le (tCells_lt hn hhv y hy) (hd.ht_len hst)) hx)

/-! ## Disjointness -/

/-- A program is determined by its sequence, head and block, and two blocks with a common column
are one block. -/
theorem prog_eq {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gridN d) (hj : j < gridN d)
    (hn : nOf d i = nOf d j) (hv : hvOf d i = hvOf d j) {y : Nat}
    (h1 : inBlk d (jvOf d i) y) (h2 : inBlk d (jvOf d j) y) : i = j := by
  have hjv : jvOf d i = jvOf d j := by
    unfold inBlk BV at h1 h2; omega
  rw [← (decode hd hi).2.2.2, ← (decode hd hj).2.2.2, hn, hv, hjv]

/-- Different programs write different cells of `h`. -/
theorem hAll_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gridN d) (hj : j < gridN d)
    (hij : i ≠ j) : Disjoint (hAll d (nOf d i) (hvOf d i) (jvOf d i))
      (hAll d (nOf d j) (hvOf d j) (jvOf d j)) := by
  obtain ⟨hn, hhv, -, -⟩ := decode hd hi
  obtain ⟨hn', hhv', -, -⟩ := decode hd hj
  rw [Set.disjoint_left]
  rintro x ⟨t, ht, c, y, hc, hy, rfl⟩ ⟨t', ht', c', y', hc', hy', heq⟩
  obtain ⟨h1, h2, -, h4⟩ := hIdx_inj hhv hhv' hc hc' hy.2.2 hy'.2.2 heq
  have hne : nOf d i = nOf d j := by
    by_contra hne
    exact chunks_disjoint hd hn hn' ht ht' hne h1
  subst h4
  exact hij (prog_eq hd hi hj hne h2 hy hy')

/-- Different programs write different cells of `v_new`. -/
theorem vAll_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gridN d) (hj : j < gridN d)
    (hij : i ≠ j) : Disjoint (vAll d (nOf d i) (hvOf d i) (jvOf d i))
      (vAll d (nOf d j) (hvOf d j) (jvOf d j)) := by
  obtain ⟨hn, hhv, -, -⟩ := decode hd hi
  obtain ⟨hn', hhv', -, -⟩ := decode hd hj
  rw [Set.disjoint_left]
  rintro x ⟨t, ht, s, y, hs, hy, rfl⟩ ⟨t', ht', s', y', hs', hy', heq⟩
  have k1 := tok_lt hd hn (cl_lt hs)
  have k2 := tok_lt hd hn' (cl_lt hs')
  obtain ⟨h1, h2, h3, h4⟩ := uIdx_inj (by omega) (by omega) hhv hhv' hy.2.2 hy'.2.2 heq
  have hne : nOf d i = nOf d j := by
    by_contra hne
    refine tokens_disjoint hd hn hn' (cl_lt hs) (cl_lt hs') hne ?_
    rw [h1]; omega
  subst h4
  exact hij (prog_eq hd hi hj hne h3 hy hy')

/-- Different programs write different cells of `ht`. -/
theorem tCells_disjoint {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gridN d) (hj : j < gridN d)
    (hij : i ≠ j) : Disjoint (tCells d (nOf d i) (hvOf d i) (jvOf d i))
      (tCells d (nOf d j) (hvOf d j) (jvOf d j)) := by
  obtain ⟨hn, hhv, -, -⟩ := decode hd hi
  obtain ⟨hn', hhv', -, -⟩ := decode hd hj
  rw [Set.disjoint_left]
  rintro x ⟨c, y, hc, hy, rfl⟩ ⟨c', y', hc', hy', heq⟩
  obtain ⟨h1, h2, -, h4⟩ := sIdx_inj hhv hhv' hc hc' hy.2.2 hy'.2.2 heq
  subst h4
  exact hij (prog_eq hd hi hj h1 h2 hy hy')

/-- Cells inside the tensors of `h` and of `v_new` occupy different bytes. -/
theorem hvn_disjoint {d : Inst} (hd : d.ok) {S S' : Set Nat}
    (hS : ∀ y ∈ S, y < d.B * NTt d * d.HV * d.K * d.V) (hS' : ∀ y ∈ S', y < d.B * d.T * d.HV * d.V) :
    Disjoint (d.h.elems S) (d.vn.elems S') :=
  Buf.elems_disjoint_of d.h d.vn hd.disj_h.2.2.2.2.1
    (fun y hy => lt_of_lt_of_le (hS y hy) hd.h_len)
    (fun y hy => lt_of_lt_of_le (hS' y hy) hd.vn_len)

/-- Cells inside the tensors of `h` and of `ht` occupy different bytes. -/
theorem hht_disjoint {d : Inst} (hd : d.ok) (hst : d.storeHt = true) {S S' : Set Nat}
    (hS : ∀ y ∈ S, y < d.B * NTt d * d.HV * d.K * d.V) (hS' : ∀ y ∈ S', y < d.N * d.HV * d.K * d.V) :
    Disjoint (d.h.elems S) (d.ht.elems S') :=
  Buf.elems_disjoint_of d.h d.ht hd.disj_h.2.2.2.2.2.2.1
    (fun y hy => lt_of_lt_of_le (hS y hy) hd.h_len)
    (fun y hy => lt_of_lt_of_le (hS' y hy) (hd.ht_len hst))

/-- Cells inside the tensors of `v_new` and of `ht` occupy different bytes. -/
theorem vnht_disjoint {d : Inst} (hd : d.ok) (hst : d.storeHt = true) {S S' : Set Nat}
    (hS : ∀ y ∈ S, y < d.B * d.T * d.HV * d.V) (hS' : ∀ y ∈ S', y < d.N * d.HV * d.K * d.V) :
    Disjoint (d.vn.elems S) (d.ht.elems S') :=
  Buf.elems_disjoint_of d.vn d.ht hd.disj_vn.2.2.2.2.2.1
    (fun y hy => lt_of_lt_of_le (hS y hy) hd.vn_len)
    (fun y hy => lt_of_lt_of_le (hS' y hy) (hd.ht_len hst))

/-- **No program reads what it writes.** -/
theorem disjoint_RW {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gridN d) :
    Disjoint (Rset d) (W d i) := by
  rw [Set.disjoint_left]
  intro x hxR hxW
  rcases W_sub_bytes hd hi hxW with (h | h) | h
  · exact hxR.1 h
  · exact hxR.2.1 h
  · exact hxR.2.2 h

/-- **No program writes what another program reads or writes.** -/
theorem disjoint_W {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < gridN d) (hj : j < gridN d)
    (hij : i ≠ j) : Disjoint (W d i) (Rset d ∪ W d j) := by
  obtain ⟨hn, hhv, -, -⟩ := decode hd hi
  obtain ⟨hn', hhv', -, -⟩ := decode hd hj
  refine Set.disjoint_union_right.mpr ⟨(disjoint_RW hd hi).symm, ?_⟩
  rw [Set.disjoint_left]
  intro x hxi hxj
  rcases W_sub_parts d i hxi with (hi1 | hi1) | hi1 <;>
    rcases W_sub_parts d j hxj with (hj1 | hj1) | hj1
  · exact Set.disjoint_left.mp (Buf.elems_disjoint d.h (hAll_disjoint hd hi hj hij)) hi1 hj1
  · exact Set.disjoint_left.mp (hvn_disjoint hd (hAll_lt hd hn hhv) (vAll_lt hd hn' hhv')) hi1 hj1
  · obtain ⟨hst, hj1⟩ := Wend_mem hj1
    exact Set.disjoint_left.mp
      (hht_disjoint hd hst (hAll_lt hd hn hhv) (tCells_lt hn' hhv')) hi1 hj1
  · exact Set.disjoint_left.mp
      (hvn_disjoint hd (hAll_lt hd hn' hhv') (vAll_lt hd hn hhv)).symm hi1 hj1
  · exact Set.disjoint_left.mp (Buf.elems_disjoint d.vn (vAll_disjoint hd hi hj hij)) hi1 hj1
  · obtain ⟨hst, hj1⟩ := Wend_mem hj1
    exact Set.disjoint_left.mp
      (vnht_disjoint hd hst (vAll_lt hd hn hhv) (tCells_lt hn' hhv')) hi1 hj1
  · obtain ⟨hst, hi1⟩ := Wend_mem hi1
    exact Set.disjoint_left.mp
      (hht_disjoint hd hst (hAll_lt hd hn' hhv') (tCells_lt hn hhv)).symm hi1 hj1
  · obtain ⟨hst, hi1⟩ := Wend_mem hi1
    exact Set.disjoint_left.mp
      (vnht_disjoint hd hst (vAll_lt hd hn' hhv') (tCells_lt hn hhv)).symm hi1 hj1
  · obtain ⟨-, hi1⟩ := Wend_mem hi1
    obtain ⟨-, hj1⟩ := Wend_mem hj1
    exact Set.disjoint_left.mp (Buf.elems_disjoint d.ht (tCells_disjoint hd hi hj hij)) hi1 hj1

/-- The inputs lie in `Rset`: a buffer disjoint from the three outputs is read-only. -/
theorem bytes_sub_Rset {d : Inst} {dd : DType} (b : Buf dd) (h1 : Disjoint b.bytes d.h.bytes)
    (h2 : Disjoint b.bytes d.vn.bytes) (h3 : Disjoint b.bytes d.ht.bytes) :
    b.bytes ⊆ Rset d :=
  fun _ hx => ⟨fun h => Set.disjoint_left.mp h1 hx h, fun h => Set.disjoint_left.mp h2 hx h,
    fun h => Set.disjoint_left.mp h3 hx h⟩

/-- Within one program, the cells of different chunks do not meet: the cells `colCells ... t` of
column `j` are outside chunk `t`'s writes. -/
theorem colCells_disjoint_Wt {d : Inst} (hd : d.ok) {n hv jv j t : Nat} (hn : n < d.N)
    (hhv : hv < d.HV) (hj : inBlk d jv j) (ht : t < nch d n) :
    Disjoint (colCells d n hv j t) (Wt d n hv jv t) := by
  have hA : ∀ y ∈ {x | ∃ t' c, t' < t ∧ c < d.K ∧ x = hIdx d (cbase d n + t') hv c j},
      y < d.B * NTt d * d.HV * d.K * d.V := by
    rintro y ⟨t', c, ht', hc, rfl⟩
    exact hIdx_lt (chunk_lt hd hn (lt_trans ht' ht)) hhv hc hj.2.2
  have hB : ∀ y ∈ {x | ∃ s, s < min (lenOf d n) (BT * t) ∧
      x = uIdx d (rowOf d n) (startOf d n + s) hv j}, y < d.B * d.T * d.HV * d.V := by
    rintro y ⟨s, hs, rfl⟩
    exact uIdx_lt (row_lt hd hn) (tok_lt hd hn (lt_of_lt_of_le hs (min_le_left _ _))) hhv hj.2.2
  unfold colCells Wt
  refine Set.disjoint_union_left.mpr ⟨Set.disjoint_union_right.mpr ⟨?_, ?_⟩,
    Set.disjoint_union_right.mpr ⟨?_, ?_⟩⟩
  · refine Buf.elems_disjoint d.h (Set.disjoint_left.mpr ?_)
    rintro x ⟨t', c, ht', hc, rfl⟩ ⟨c', y, hc', hy, heq⟩
    obtain ⟨h1, -⟩ := hIdx_inj hhv hhv hc hc' hj.2.2 hy.2.2 heq
    omega
  · exact hvn_disjoint hd hA (vCells_lt hd hn hhv)
  · exact (hvn_disjoint hd (hCells_lt hd hn hhv ht) hB).symm
  · refine Buf.elems_disjoint d.vn (Set.disjoint_left.mpr ?_)
    rintro x ⟨s, hs, rfl⟩ ⟨i, y, hi, hy, heq⟩
    have k1 := tok_lt hd hn (lt_of_lt_of_le hs (min_le_left _ _))
    have k2 := tok_lt hd hn (cl_lt hi)
    obtain ⟨-, h2, -⟩ := uIdx_inj (by omega) (by omega) hhv hhv hj.2.2 hy.2.2 heq
    have := lt_of_lt_of_le hs (min_le_right _ _)
    omega

/-- Chunk `t`'s and chunk `u`'s writes of one program are disjoint for `t ≠ u`. -/
theorem Wt_disjoint {d : Inst} (hd : d.ok) {n hv jv t u : Nat} (hn : n < d.N)
    (hhv : hv < d.HV) (htu : t ≠ u) (ht : t < nch d n) (hu : u < nch d n) :
    Disjoint (Wt d n hv jv t) (Wt d n hv jv u) := by
  unfold Wt
  refine Set.disjoint_union_left.mpr ⟨Set.disjoint_union_right.mpr ⟨?_, ?_⟩,
    Set.disjoint_union_right.mpr ⟨?_, ?_⟩⟩
  · refine Buf.elems_disjoint d.h (Set.disjoint_left.mpr ?_)
    rintro x ⟨c, y, hc, hy, rfl⟩ ⟨c', y', hc', hy', heq⟩
    obtain ⟨h1, -⟩ := hIdx_inj hhv hhv hc hc' hy.2.2 hy'.2.2 heq
    omega
  · exact hvn_disjoint hd (hCells_lt hd hn hhv ht) (vCells_lt hd hn hhv)
  · exact (hvn_disjoint hd (hCells_lt hd hn hhv hu) (vCells_lt hd hn hhv)).symm
  · refine Buf.elems_disjoint d.vn (Set.disjoint_left.mpr ?_)
    rintro x ⟨i, y, hi, hy, rfl⟩ ⟨i', y', hi', hy', heq⟩
    have k1 := tok_lt hd hn (cl_lt hi)
    have k2 := tok_lt hd hn (cl_lt hi')
    obtain ⟨-, h2, -⟩ := uIdx_inj (by omega) (by omega) hhv hhv hy.2.2 hy'.2.2 heq
    have c1 : i < BT := lt_of_lt_of_le hi (min_le_left _ _)
    have c2 : i' < BT := lt_of_lt_of_le hi' (min_le_left _ _)
    apply htu
    unfold BT at c1 c2 h2
    omega

/-- The final state's cells are disjoint from every chunk's writes. -/
theorem Wend_disjoint_iterW {d : Inst} (hd : d.ok) {n hv jv : Nat} (hn : n < d.N)
    (hhv : hv < d.HV) :
    Disjoint (iterW (Wt d n hv jv) (nch d n)) (Wend d n hv jv) := by
  rw [Set.disjoint_left]
  rintro x ⟨t, ht, hx⟩ hxe
  obtain ⟨hst, hxe⟩ := Wend_mem hxe
  rcases hx with hx | hx
  · exact Set.disjoint_left.mp (hht_disjoint hd hst (hCells_lt hd hn hhv ht) (tCells_lt hn hhv)) hx hxe
  · exact Set.disjoint_left.mp (vnht_disjoint hd hst (vCells_lt hd hn hhv) (tCells_lt hn hhv)) hx hxe

/-! ## `ColPre` and `Col` read the outputs at their cells only -/

/-- `ColPre` of `m` gives `ColPre` of any memory that agrees with `m` on the cells `colCells`. -/
theorem ColPre_of_agree {d : Inst} {init m m' : Mem} {n hv j t : Nat} {Y : Nat → Nat → F32}
    (h : ∀ x ∈ colCells d n hv j t, m' x = m x) (hP : ColPre d init m n hv j t Y) :
    ColPre d init m' n hv j t Y := by
  have hh : ∀ t', t' < t → SameH d m m' n hv j t' := by
    intro t' ht' c hc
    refine rdB_congr fun x hx => h x (Or.inl ?_)
    exact ⟨hIdx d (cbase d n + t') hv c j, ⟨t', c, ht', hc, rfl⟩, hx⟩
  have hvv : ∀ s, s < min (lenOf d n) (BT * t) →
      rdB m' d.vn (uIdx d (rowOf d n) (startOf d n + s) hv j) =
        rdB m d.vn (uIdx d (rowOf d n) (startOf d n + s) hv j) := by
    intro s hs
    refine rdB_congr fun x hx => h x (Or.inr ?_)
    exact ⟨uIdx d (rowOf d n) (startOf d n + s) hv j, ⟨s, hs, rfl⟩, hx⟩
  obtain ⟨p1, p2, p3, p4, p5⟩ := hP
  refine ⟨?_, ?_, p3, ?_, ?_⟩
  · intro s hs
    obtain ⟨x, hx, hrest⟩ := p1 s hs
    have hs' : s / BT < t := by
      have := lt_of_lt_of_le hs (min_le_right _ _)
      unfold BT at *; omega
    refine ⟨x, by rw [hvv s hs]; exact hx, fun hV hSt => ?_⟩
    obtain ⟨hfin, hb⟩ := hrest hV fun t' ht' =>
      (StOk_congr (hh t' (by omega))).mp (hSt t' ht')
    refine ⟨hfin, ?_⟩
    rw [vrefS_congr (hh _ hs'), tolV_congr (hh _ hs')]
    exact hb
  · intro t' ht' c hc
    rw [hh t' ht' c hc]
    exact p2 t' ht' c hc
  · intro t' ht' c hc hfin
    rw [hh t' ht' c hc]
    exact p4 t' ht' c hc hfin
  · intro t' ht' hV hSt
    obtain ⟨hf, hb⟩ := p5 t' ht' hV fun t'' ht'' =>
      (StOk_congr (hh t'' (by omega))).mp (hSt t'' ht'')
    refine ⟨hf, ?_⟩
    simpa only [Step_congr (hh t' ht'), errStep_congr (hh t' ht')] using hb

/-- `ColPre` depends on the memory only through the cells `colCells`. -/
theorem ColPre_congr {d : Inst} {init m m' : Mem} {n hv j t : Nat} {Y : Nat → Nat → F32}
    (h : ∀ x ∈ colCells d n hv j t, m' x = m x) :
    ColPre d init m n hv j t Y ↔ ColPre d init m' n hv j t Y :=
  ⟨ColPre_of_agree h, ColPre_of_agree fun x hx => (h x hx).symm⟩

/-- `Col` is `ColPre` after all the chunks, with the final state stored. -/
theorem pre_of_col {d : Inst} {init m : Mem} {n hv j : Nat} (h : Col d init m n hv j) :
    ∃ Y, ColPre d init m n hv j (nch d n) Y ∧
      (d.storeHt = true → ∀ c, c < d.K → rdF m d.ht (sIdx d n hv c j) = some (Y (nch d n) c)) := by
  obtain ⟨c1, c2, Y, y1, y2, y3, y4⟩ := h
  exact ⟨Y, ⟨fun s hs => c1 s (lt_of_lt_of_le hs (min_le_left _ _)), c2, y1, y2, y3⟩, y4⟩

/-- `Col` of `m` gives `Col` of any memory that agrees with `m` on the cells of the column. -/
theorem Col_of_agree {d : Inst} {init m m' : Mem} {n hv j : Nat}
    (h : ∀ x ∈ colCells d n hv j (nch d n) ∪
        (if d.storeHt then d.ht.elems {y | ∃ c, c < d.K ∧ y = sIdx d n hv c j} else ∅),
      m' x = m x)
    (hC : Col d init m n hv j) : Col d init m' n hv j := by
  obtain ⟨Y, hP, hT⟩ := pre_of_col hC
  refine col_of_pre (ColPre_of_agree (fun x hx => h x (Or.inl hx)) hP) fun hst c hc => ?_
  rw [rdF_congr fun x hx => h x (Or.inr ?_)]
  · exact hT hst c hc
  · simp only [hst, ↓reduceIte]
    exact ⟨sIdx d n hv c j, ⟨c, hc, rfl⟩, hx⟩

/-- `Col` depends on the memory only through the cells of column `j` of its sequence and head. -/
theorem Col_congr {d : Inst} {init m m' : Mem} {n hv j : Nat}
    (h : ∀ x ∈ colCells d n hv j (nch d n) ∪
        (if d.storeHt then d.ht.elems {y | ∃ c, c < d.K ∧ y = sIdx d n hv c j} else ∅),
      m' x = m x) :
    Col d init m n hv j ↔ Col d init m' n hv j :=
  ⟨Col_of_agree h, Col_of_agree fun x hx => (h x hx).symm⟩

/-- The cells of column `j` of block `jv` are written by the program that owns the block. -/
theorem colCells_sub_W {d : Inst} (_hd : d.ok) {i : Nat} (_hi : i < gridN d) {jj : Nat}
    (_hjj : jj < BV) (hj : inBlk d (jvOf d i) (BV * jvOf d i + jj)) :
    colCells d (nOf d i) (hvOf d i) (BV * jvOf d i + jj) (nch d (nOf d i)) ∪
        (if d.storeHt then d.ht.elems {y | ∃ c, c < d.K ∧
          y = sIdx d (nOf d i) (hvOf d i) c (BV * jvOf d i + jj)} else ∅) ⊆ W d i := by
  rintro x ((⟨y, ⟨t', c, ht', hc, rfl⟩, hx⟩ | ⟨y, ⟨s, hs, rfl⟩, hx⟩) | hx)
  · exact Wt_sub_W ht'
      (Or.inl ⟨_, ⟨c, BV * jvOf d i + jj, hc, hj, rfl⟩, hx⟩)
  · have hs1 := lt_of_lt_of_le hs (min_le_left _ _)
    have hs2 := lt_of_lt_of_le hs (min_le_right _ _)
    have e : startOf d (nOf d i) + s =
        startOf d (nOf d i) + BT * (s / BT) + s % BT := by unfold BT; omega
    have ht' : s / BT < nch d (nOf d i) := by unfold nch BT at *; omega
    have hcl : s % BT < cl d (nOf d i) (s / BT) := by unfold cl nch BT at *; omega
    exact Wt_sub_W ht'
      (Or.inr ⟨_, ⟨s % BT, BV * jvOf d i + jj, hcl, hj, by rw [e]⟩, hx⟩)
  · split_ifs at hx with hst
    obtain ⟨y, ⟨c, hc, rfl⟩, hx⟩ := hx
    refine Or.inr ?_
    unfold Wend
    simp only [hst, ↓reduceIte]
    exact ⟨_, ⟨c, BV * jvOf d i + jj, hc, hj, rfl⟩, hx⟩
    exact absurd hx (Set.notMem_empty x)

/-! ## `Post` -/

/-- **The programs' results give `Post`.** -/
theorem post_of_locals {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init i m ∧ ∀ x ∈ Rset d ∪ W d i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d i) → fin x = init x) : Post d init fin := by
  rw [post_iff]
  refine ⟨?_, ?_⟩
  · intro n hv j hn hhv hj
    have hjv : j / BV < NV d := by unfold NV BV at *; omega
    obtain ⟨i, hi⟩ : ∃ i, i = progOf d n hv (j / BV) := ⟨_, rfl⟩
    have hlt : i < gridN d := by
      have h1 : n * d.HV + hv < d.N * d.HV := pair_lt hn hhv
      have h2 := pair_lt h1 hjv
      have : gridN d = d.N * d.HV * NV d := by unfold gridN; change NV d * d.N * d.HV = _; ring
      rw [hi, this]; exact h2
    obtain ⟨hn1, hhv1, hjv1, hdec⟩ := decode hd hlt
    have hdec' := hdec.trans hi
    unfold progOf at hdec'
    obtain ⟨h1, h2⟩ := pair_inj hjv1 hjv hdec'
    obtain ⟨h3, h4⟩ := pair_inj hhv1 hhv h1
    have hblk : inBlk d (jvOf d i) (BV * jvOf d i + j % BV) := by
      rw [h2]; unfold inBlk BV at *; omega
    have hjj : j % BV < BV := Nat.mod_lt _ (by unfold BV; omega)
    have ej : BV * jvOf d i + j % BV = j := by rw [h2]; unfold BV; omega
    obtain ⟨m, hQ, hframe⟩ := hloc ⟨i, by rw [launch_size]; exact hlt⟩
    have hC := hQ (j % BV) hjj hblk
    have hsub := colCells_sub_W hd hlt hjj hblk
    rw [h3, h4, ej] at hC hsub
    refine (Col_congr (m := m) (m' := fin) fun x hx => ?_).mp hC
    exact hframe x (Or.inr (hsub hx))
  · intro z hz
    refine hfr z fun i hzi => hz ?_
    exact W_sub_outputs hd (lt_of_lt_of_eq i.isLt (launch_size d)) hzi

end Claims.ChunkDeltaH
